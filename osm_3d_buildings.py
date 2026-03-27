"""Create 3D building meshes from OSM footprints and return a directed street graph."""

from __future__ import annotations

from typing import Iterator, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pyvista as pv
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


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
        Height in meters used to extrude each building footprint (default: 10).

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
    for geometry in projected_buildings.geometry:
        if geometry is None:
            continue

        for poly in _iter_polygon_parts(geometry):
            footprint = _polygon_to_footprint(poly)
            if footprint is None or footprint.n_points < 3:
                continue

            extruded = footprint.extrude((0.0, 0.0, float(extrusion_height)), capping=True)
            if extruded.n_points > 0:
                extruded_meshes.append(extruded)

    if not extruded_meshes:
        return pv.PolyData(), projected_graph

    combined_mesh = extruded_meshes[0].copy()
    for mesh in extruded_meshes[1:]:
        combined_mesh = combined_mesh.merge(mesh)

    return combined_mesh, projected_graph
