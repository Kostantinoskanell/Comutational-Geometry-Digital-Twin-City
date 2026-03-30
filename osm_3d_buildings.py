"""Create 3D building meshes from OSM footprints and return a directed street graph."""

from __future__ import annotations

import re
from typing import Iterator, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pyvista as pv
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate

LEVEL_HEIGHT_M = 3.0
PARAPET_INSET_M = 0.35
PARAPET_HEIGHT_M = 1.0


def _iter_polygon_parts(geometry: BaseGeometry) -> Iterator[Polygon]:
    """Yield polygon parts from Polygon or MultiPolygon geometries."""
    if geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty:
                yield part


def _polygon_to_footprint(poly: Polygon) -> pv.PolyData | None:
    """Convert a shapely polygon exterior ring to a flat PyVista polygon at z=0."""
    if poly.is_empty or poly.exterior is None:
        return None

    exterior_xy = np.asarray(poly.exterior.coords, dtype=float)
    if exterior_xy.shape[0] < 4:
        return None

    # Drop the repeated closing vertex expected in shapely rings.
    if np.allclose(exterior_xy[0], exterior_xy[-1]):
        exterior_xy = exterior_xy[:-1]

    if exterior_xy.shape[0] < 3:
        return None

    z = np.zeros((exterior_xy.shape[0], 1), dtype=float)
    points = np.hstack((exterior_xy[:, :2], z))

    face = np.hstack(([points.shape[0]], np.arange(points.shape[0], dtype=np.int64)))
    return pv.PolyData(points, faces=face)


def _parse_height_value(value: object) -> float | None:
    """Parse OSM height-like values to meters (supports metric and feet)."""
    if value is None:
        return None

    if isinstance(value, (int, float)) and np.isfinite(value):
        h = float(value)
        return h if h > 0.0 else None

    text = str(value).strip().lower()
    if not text:
        return None

    # OSM tags can contain multiple candidates like "12;14".
    token = text.split(";", 1)[0].strip().replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+", token)
    if match is None:
        return None

    try:
        number = float(match.group(0))
    except ValueError:
        return None

    if number <= 0.0:
        return None

    if "ft" in token or "foot" in token or "feet" in token or "'" in token:
        number *= 0.3048
    elif "cm" in token:
        number *= 0.01
    elif "mm" in token:
        number *= 0.001

    return number if number > 0.0 else None


def _roof_shape_from_row(row: gpd.GeoSeries) -> str:
    raw = row.get("roof:shape")
    if raw is None:
        return ""
    return str(raw).strip().lower()


def _make_pitched_roof_mesh(poly: Polygon, base_height: float, roof_raise: float) -> pv.PolyData | None:
    """Create triangular roof faces from footprint perimeter to elevated centroid."""
    if poly.is_empty or poly.exterior is None:
        return None

    ring = np.asarray(poly.exterior.coords, dtype=float)
    if ring.shape[0] < 4:
        return None
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        return None

    roof_z = float(base_height)
    base_pts = np.column_stack((ring[:, :2], np.full((ring.shape[0],), roof_z, dtype=float)))

    centroid = poly.centroid
    apex = np.array([[float(centroid.x), float(centroid.y), roof_z + float(roof_raise)]], dtype=float)
    points = np.vstack((base_pts, apex))
    apex_idx = points.shape[0] - 1

    faces: list[int] = []
    n = base_pts.shape[0]
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, i, j, apex_idx])

    return pv.PolyData(points, faces=np.asarray(faces, dtype=np.int64))


def _make_parapet_mesh(poly: Polygon, base_height: float) -> pv.PolyData | None:
    """Create a roof parapet ring by subtracting an inward buffered polygon."""
    if poly.is_empty:
        return None

    inner = poly.buffer(-PARAPET_INSET_M)
    if inner.is_empty:
        return None

    ring = poly.difference(inner)
    if ring.is_empty:
        return None

    parapet = _geometry_to_extruded_mesh(ring, PARAPET_HEIGHT_M)
    if parapet.n_points == 0:
        return None

    return parapet.translate((0.0, 0.0, float(base_height)), inplace=False)


def _resolve_building_height_and_roof(row: gpd.GeoSeries, default_height: float) -> tuple[float, str]:
    """Resolve dynamic building height from levels and capture roof shape."""
    roof_shape = _roof_shape_from_row(row)

    levels = _parse_height_value(row.get("building:levels"))
    if levels is not None:
        return max(2.5, levels * LEVEL_HEIGHT_M), roof_shape

    # Fallbacks for sparse OSM tags.
    for key in ("levels", "height", "building:height"):
        value = _parse_height_value(row.get(key))
        if value is not None:
            if key in ("levels",):
                return max(2.5, value * LEVEL_HEIGHT_M), roof_shape
            return max(2.5, value), roof_shape

    return max(2.5, float(default_height)), roof_shape


def _geometry_to_extruded_mesh(geometry: BaseGeometry, height: float) -> pv.PolyData:
    """Triangulate a polygonal geometry and extrude it by the given height."""
    if geometry is None or geometry.is_empty:
        return pv.PolyData()

    polys: list[Polygon] = []
    if isinstance(geometry, Polygon):
        polys = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polys = [p for p in geometry.geoms if not p.is_empty]
    else:
        return pv.PolyData()

    out: pv.PolyData | None = None
    for poly in polys:
        # Triangulate and keep only triangles whose representative point is inside
        # the polygon (handles holes from sidewalk difference geometries).
        tri_parts = triangulate(poly)
        for tri in tri_parts:
            if tri.is_empty:
                continue
            if not poly.contains(tri.representative_point()):
                continue

            footprint = _polygon_to_footprint(tri)
            if footprint is None or footprint.n_points < 3:
                continue

            extruded = footprint.extrude((0.0, 0.0, float(height)), capping=True)
            if extruded.n_points == 0:
                continue
            out = extruded if out is None else out.merge(extruded)

    return pv.PolyData() if out is None else out


def build_road_and_sidewalk_meshes_from_graph(
    projected_graph: nx.MultiDiGraph,
    road_buffer_m: float = 3.0,
    sidewalk_buffer_m: float = 5.0,
    road_extrude_z: float = 0.1,
    sidewalk_extrude_z: float = 0.2,
) -> tuple[pv.PolyData, pv.PolyData]:
    """Build road/sidewalk meshes from street centerlines using geometric buffers."""
    if road_buffer_m <= 0.0:
        raise ValueError("road_buffer_m must be > 0.")
    if sidewalk_buffer_m <= road_buffer_m:
        raise ValueError("sidewalk_buffer_m must be > road_buffer_m.")

    edges_gdf = ox.graph_to_gdfs(
        projected_graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )
    if edges_gdf.empty:
        return pv.PolyData(), pv.PolyData()

    centerlines = edges_gdf.geometry
    road_polygon = centerlines.buffer(float(road_buffer_m)).unary_union
    sidewalk_outer = centerlines.buffer(float(sidewalk_buffer_m)).unary_union
    sidewalk_polygon = sidewalk_outer.difference(road_polygon)

    road_mesh = _geometry_to_extruded_mesh(road_polygon, float(road_extrude_z))
    sidewalk_mesh = _geometry_to_extruded_mesh(sidewalk_polygon, float(sidewalk_extrude_z))

    return road_mesh, sidewalk_mesh


def build_3d_city_with_street_surfaces(
    address: str,
    radius: float,
    extrusion_height: float = 10.0,
    road_buffer_m: float = 3.0,
    sidewalk_buffer_m: float = 5.0,
    road_extrude_z: float = 0.1,
    sidewalk_extrude_z: float = 0.2,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData, nx.MultiDiGraph]:
    """Fetch OSM data and return building, road, sidewalk meshes + projected graph."""
    buildings_mesh, projected_graph = build_3d_buildings_and_street_graph(
        address=address,
        radius=radius,
        extrusion_height=extrusion_height,
    )
    road_mesh, sidewalk_mesh = build_road_and_sidewalk_meshes_from_graph(
        projected_graph=projected_graph,
        road_buffer_m=road_buffer_m,
        sidewalk_buffer_m=sidewalk_buffer_m,
        road_extrude_z=road_extrude_z,
        sidewalk_extrude_z=sidewalk_extrude_z,
    )
    return buildings_mesh, road_mesh, sidewalk_mesh, projected_graph


def build_3d_buildings_and_street_graph(
    address: str,
    radius: float,
    extrusion_height: float = 10.0,
) -> Tuple[pv.PolyData, nx.MultiDiGraph]:
    """Fetch OSM buildings and streets around an address and build 3D buildings.

    Parameters
    ----------
    address:
        Human-readable address string used for geocoding.
    radius:
        Search radius in meters.
    extrusion_height:
        Fallback height in meters used only when OSM has no usable height metadata.

    Returns
    -------
    tuple[pv.PolyData, networkx.MultiDiGraph]
        Combined 3D building mesh and projected directed street graph.
    """
    center = ox.geocode(address)

    building_tags = {"building": True}
    buildings: gpd.GeoDataFrame = ox.features_from_point(
        center,
        tags=building_tags,
        dist=radius,
    )

    street_graph = ox.graph_from_point(
        center,
        dist=radius,
        network_type="all",
        simplify=True,
    )

    projected_buildings = ox.projection.project_gdf(buildings)
    projected_graph = ox.projection.project_graph(street_graph)

    extruded_meshes: list[pv.PolyData] = []
    for _, row in projected_buildings.iterrows():
        geometry = row.geometry
        if geometry is None:
            continue

        building_height, roof_shape = _resolve_building_height_and_roof(row, float(extrusion_height))

        for poly in _iter_polygon_parts(geometry):
            footprint = _polygon_to_footprint(poly)
            if footprint is None or footprint.n_points < 3:
                continue

            body = footprint.extrude((0.0, 0.0, float(building_height)), capping=True)
            if body.n_points == 0:
                continue

            refined = body
            if roof_shape in {"pyramidal", "gabled", "gable"}:
                # Add a simple pitched roof represented by perimeter-to-apex triangles.
                roof_raise = max(1.5, 0.25 * float(building_height))
                roof_mesh = _make_pitched_roof_mesh(poly, base_height=float(building_height), roof_raise=roof_raise)
                if roof_mesh is not None and roof_mesh.n_points > 0:
                    refined = refined.merge(roof_mesh)
            else:
                # Flat or missing roof shape: add a thin parapet ring on top.
                parapet_mesh = _make_parapet_mesh(poly, base_height=float(building_height))
                if parapet_mesh is not None and parapet_mesh.n_points > 0:
                    refined = refined.merge(parapet_mesh)

            extruded_meshes.append(refined)

    if not extruded_meshes:
        return pv.PolyData(), projected_graph

    combined_mesh = extruded_meshes[0].copy()
    for mesh in extruded_meshes[1:]:
        combined_mesh = combined_mesh.merge(mesh)

    return combined_mesh, projected_graph
