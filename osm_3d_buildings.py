"""Create 3D building meshes from OSM footprints and return a directed street graph."""

from __future__ import annotations

import re
from typing import Iterator, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pyvista as pv
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, triangulate, unary_union

from overture_source import _radius_to_bbox
from turn_restrictions import load_osm_direction_graph_cached, resolve_osm_road_direction

LEVEL_HEIGHT_M = 3.0
PARAPET_INSET_M = 0.35
PARAPET_HEIGHT_M = 1.0
CAR_HIGHWAY_TAGS = {"primary", "secondary", "residential", "unclassified",
                    "tertiary", "tertiary_link", "primary_link", "secondary_link",
                    "trunk", "trunk_link", "motorway_link", "living_street", "road"}

# Half-width buffer (metres) per highway type used for the road surface polygon.
# Each type is buffered separately and then union-ed so wide primary roads don't
# artificially widen adjacent narrow residential streets.
HIGHWAY_ROAD_WIDTHS: dict[str, float] = {
    "motorway":       7.0,
    "motorway_link":  5.5,
    "trunk":          6.5,
    "trunk_link":     5.0,
    "primary":        5.5,
    "primary_link":   4.5,
    "secondary":      5.0,
    "secondary_link": 4.0,
    "tertiary":       4.5,
    "tertiary_link":  3.5,
    "residential":    4.0,
    "living_street":  3.0,
    "unclassified":   4.0,
    "road":           3.5,
    "service":        2.5,
}
PEDESTRIAN_HIGHWAY_TAGS = {"footway", "pedestrian", "path", "cycleway"}
MAX_SEGMENT_BUFFER_AREA_M2 = 3000.0
SEGMENT_AREA_EXPANSION_FACTOR = 8.0
CAR_SURFACE_RGB = np.array([0x33, 0x33, 0x33], dtype=np.uint8)
PED_SURFACE_RGB = np.array([0xA0, 0xA0, 0xA0], dtype=np.uint8)
PARK_SURFACE_RGB = np.array([0x27, 0xae, 0x60], dtype=np.uint8)  # Green


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


def _highway_values(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = [raw]
    out: set[str] = set()
    for value in values:
        tag = str(value).strip().lower()
        if tag:
            out.add(tag)
    return out


def _iter_linestring_segments(geometry: BaseGeometry | None) -> Iterator[LineString]:
    """Yield 2-point LineString segments from (Multi)LineString-like geometry."""
    if geometry is None or geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_linestring_segments(part)
        return
    if not hasattr(geometry, "coords"):
        return

    coords = np.asarray(geometry.coords, dtype=float)
    if coords.shape[0] < 2:
        return

    xy = coords[:, :2]
    for i in range(xy.shape[0] - 1):
        a = xy[i]
        b = xy[i + 1]
        if np.allclose(a, b):
            continue
        yield LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])


def _is_sane_buffer_polygon(poly: BaseGeometry) -> bool:
    if poly is None or poly.is_empty:
        return False
    if not poly.is_valid:
        return False
    area = float(poly.area)
    if not np.isfinite(area) or area <= 0.0:
        return False
    return True


def _is_sane_segment_buffer_polygon(
    poly: BaseGeometry,
    segment_length_m: float,
    buffer_m: float,
    max_area_m2: float,
) -> bool:
    if not _is_sane_buffer_polygon(poly):
        return False

    # Expected area for buffered segment:
    # rectangle (2r * L) + end caps (pi r^2).
    expected = max(1e-6, (2.0 * float(buffer_m) * float(segment_length_m)) + (np.pi * float(buffer_m) ** 2))
    dynamic_cap = max(float(max_area_m2), expected * float(SEGMENT_AREA_EXPANSION_FACTOR))
    if float(poly.area) > dynamic_cap:
        return False
    return True


def _buffer_edges_by_highway_class(
    edges_gdf: gpd.GeoDataFrame,
    allowed_tags: set[str],
    buffer_m: float,
    max_segment_area_m2: float,
) -> BaseGeometry:
    line_geometries: list[BaseGeometry] = []
    buffered_parts: list[BaseGeometry] = []
    dropped_large = 0
    dropped_invalid = 0

    for _, row in edges_gdf.iterrows():
        tags = _highway_values(row.get("highway"))
        if not (tags & allowed_tags):
            continue

        geometry = row.geometry
        if geometry is not None and not geometry.is_empty:
            line_geometries.append(geometry)

    if line_geometries:
        try:
            merged_lines = linemerge(unary_union(line_geometries))
        except Exception:
            merged_lines = unary_union(line_geometries)

        try:
            merged_poly = merged_lines.buffer(
                float(buffer_m),
                join_style=1,
                cap_style=1,
            )
        except TypeError:
            merged_poly = merged_lines.buffer(float(buffer_m))
        except Exception:
            merged_poly = None

        # Light morphological close/open heals tiny numeric cracks at turns/junctions.
        if _is_sane_buffer_polygon(merged_poly):
            eps = max(0.05, float(buffer_m) * 0.15)
            try:
                merged_poly = merged_poly.buffer(eps).buffer(-eps)
            except Exception:
                pass
            if _is_sane_buffer_polygon(merged_poly):
                return merged_poly

        if merged_poly is not None and not merged_poly.is_empty:
            if not merged_poly.is_valid:
                dropped_invalid += 1

    for _, row in edges_gdf.iterrows():
        tags = _highway_values(row.get("highway"))
        if not (tags & allowed_tags):
            continue

        geometry = row.geometry
        for seg in _iter_linestring_segments(geometry):
            seg_len = float(seg.length)
            seg_poly = seg.buffer(float(buffer_m), join_style=1, cap_style=1)
            if not _is_sane_segment_buffer_polygon(
                seg_poly,
                segment_length_m=seg_len,
                buffer_m=float(buffer_m),
                max_area_m2=float(max_segment_area_m2),
            ):
                if not seg_poly.is_empty:
                    if not seg_poly.is_valid:
                        dropped_invalid += 1
                    else:
                        dropped_large += 1
                continue
            buffered_parts.append(seg_poly)

    if dropped_large > 0 or dropped_invalid > 0:
        print(
            "Discarded buffered segments "
            f"(large={dropped_large}, invalid={dropped_invalid}) for tags={sorted(allowed_tags)}"
        )

    if not buffered_parts:
        return Polygon()
    return unary_union(buffered_parts)


def _apply_surface_color(mesh: pv.PolyData, rgb: np.ndarray) -> pv.PolyData:
    if mesh.n_cells <= 0:
        return mesh
    colors = np.tile(np.asarray(rgb, dtype=np.uint8), (mesh.n_cells, 1))
    mesh.cell_data["surface_rgb"] = colors
    try:
        mesh.set_active_scalars("surface_rgb", preference="cell")
    except Exception:
        pass
    return mesh


def build_road_and_sidewalk_meshes_from_graph(
    projected_graph: nx.MultiDiGraph,
    road_buffer_m: float = 4.5,
    sidewalk_buffer_m: float = 2.0,
    road_extrude_z: float = 0.1,
    sidewalk_extrude_z: float = 0.15,
    max_segment_area_m2: float = MAX_SEGMENT_BUFFER_AREA_M2,
) -> tuple[pv.PolyData, pv.PolyData]:
    """Build car/pedestrian surface meshes from street centerlines using buffered segments.

    Each highway type is buffered with its own width from HIGHWAY_ROAD_WIDTHS so
    primary arteries don't widen adjacent residential streets.  road_buffer_m is
    used as the fallback for any type not listed in that dict.
    """
    if road_buffer_m <= 0.0:
        raise ValueError("road_buffer_m must be > 0.")
    if sidewalk_buffer_m <= 0.0:
        raise ValueError("sidewalk_buffer_m must be > 0.")
    if max_segment_area_m2 <= 0.0:
        raise ValueError("max_segment_area_m2 must be > 0.")

    edges_gdf = ox.graph_to_gdfs(
        projected_graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )
    if edges_gdf.empty:
        return pv.PolyData(), pv.PolyData()

    # Per-type road buffering: buffer each highway class separately then union.
    road_parts: list = []
    for hw_tag in CAR_HIGHWAY_TAGS:
        w = HIGHWAY_ROAD_WIDTHS.get(hw_tag, road_buffer_m)
        part = _buffer_edges_by_highway_class(
            edges_gdf=edges_gdf,
            allowed_tags={hw_tag},
            buffer_m=float(w),
            max_segment_area_m2=float(max_segment_area_m2),
        )
        if part is not None and not part.is_empty:
            road_parts.append(part)

    if road_parts:
        car_roads_polygon = unary_union(road_parts)
    else:
        car_roads_polygon = Polygon()

    pedestrian_roads_polygon = _buffer_edges_by_highway_class(
        edges_gdf=edges_gdf,
        allowed_tags=PEDESTRIAN_HIGHWAY_TAGS,
        buffer_m=float(sidewalk_buffer_m),
        max_segment_area_m2=float(max_segment_area_m2),
    )

    road_mesh = _geometry_to_extruded_mesh(car_roads_polygon, float(road_extrude_z))
    sidewalk_mesh = _geometry_to_extruded_mesh(pedestrian_roads_polygon, float(sidewalk_extrude_z))
    road_mesh = _apply_surface_color(road_mesh, CAR_SURFACE_RGB)
    sidewalk_mesh = _apply_surface_color(sidewalk_mesh, PED_SURFACE_RGB)

    return road_mesh, sidewalk_mesh


def _fetch_ms_building_heights(center_lat: float, center_lon: float, radius_m: float, proj_str: str):
    """Fetch Microsoft Building Footprints heights from Planetary Computer.

    Returns (centroids_xy, heights, kdtree) in local proj_str CRS, or (None, None, None) on failure.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        return None, None, None

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None, None

    try:
        # Build WGS84 bounding box
        deg_per_m = radius_m / 111_320.0
        lon_pad = deg_per_m / max(0.01, abs(np.cos(np.radians(center_lat))))
        lat_pad = deg_per_m
        _bbox = [center_lon - lon_pad, center_lat - lat_pad,
                 center_lon + lon_pad, center_lat + lat_pad]

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["ms-buildings"],
            bbox=_bbox,
            max_items=4,
        )
        items = list(search.items())
        if not items:
            return None, None, None

        frames = []
        for item in items:
            asset = item.assets.get("data")
            if asset is None:
                continue
            signed_href = planetary_computer.sign(asset).href
            try:
                gdf = gpd.read_parquet(signed_href, columns=["geometry", "height"])
                frames.append(gdf)
            except Exception:
                continue

        if not frames:
            return None, None, None

        import pandas as _pd
        combined = _pd.concat(frames, ignore_index=True)
        combined = combined[combined["height"].notna() & (combined["height"] > 0.0)]
        if combined.empty:
            return None, None, None

        # Project to local CRS
        combined = combined.set_crs("EPSG:4326", allow_override=True).to_crs(proj_str)
        centroids = np.array([[g.centroid.x, g.centroid.y] for g in combined.geometry])
        heights = combined["height"].to_numpy(dtype=float)

        # Filter non-finite
        valid = np.isfinite(centroids).all(axis=1) & np.isfinite(heights)
        centroids, heights = centroids[valid], heights[valid]
        if len(centroids) == 0:
            return None, None, None

        tree = cKDTree(centroids)
        print(f"[ms-buildings] {len(centroids)} footprints with height data loaded")
        return centroids, heights, tree

    except Exception as _e:
        print(f"[ms-buildings] skipped: {_e}")
        return None, None, None


def _fetch_dem_sampler(center_lat: float, center_lon: float, radius_m: float, proj_str: str):
    """Fetch Copernicus DEM GLO-30 and return a terrain elevation sampler.

    Returns callable sampler(xy: np.ndarray (N,2)) → elevations (N,) metres
    in the local proj_str CRS, or None on failure / missing dependencies.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        return None
    try:
        import rasterio
        from rasterio.merge import merge as _rio_merge
    except ImportError:
        return None
    try:
        from pyproj import Transformer
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        return None

    # ── Disk cache: the merged elevation grid is ~50 MB of remote GeoTIFF
    # reads via STAC — cache it as .npz so later runs skip the network.
    from pathlib import Path as _Path
    _dem_cache_dir = _Path(__file__).resolve().parent / "cache"
    _dem_cache = _dem_cache_dir / (
        f"dem_{center_lat:.5f}_{center_lon:.5f}_{radius_m:.0f}.npz")

    def _sampler_from_grid(ys, xs, elev):
        from scipy.interpolate import RegularGridInterpolator as _RGI
        from pyproj import Transformer as _Tr
        interp = _RGI((ys, xs), elev, method="linear",
                      bounds_error=False, fill_value=0.0)
        _tr = _Tr.from_crs(proj_str, "EPSG:4326", always_xy=True)

        def _sampler(xy_local: np.ndarray) -> np.ndarray:
            lons, lats = _tr.transform(xy_local[:, 0], xy_local[:, 1])
            return interp(np.column_stack([lats, lons]))
        return _sampler

    if _dem_cache.exists():
        try:
            _z = np.load(_dem_cache)
            print(f"[dem] loaded cache {_dem_cache.name} "
                  f"({_z['elev'].shape[0]}×{_z['elev'].shape[1]} cells)")
            return _sampler_from_grid(_z["ys"], _z["xs"], _z["elev"])
        except Exception as _ce:
            print(f"[dem] cache read failed ({_ce}) — re-fetching")

    try:
        deg_per_m = radius_m / 111_320.0
        lon_pad = deg_per_m / max(0.01, abs(np.cos(np.radians(center_lat))))
        lat_pad = deg_per_m
        bbox_wgs84 = [
            center_lon - lon_pad, center_lat - lat_pad,
            center_lon + lon_pad, center_lat + lat_pad,
        ]

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox_wgs84,
            max_items=4,
        )
        items = list(search.items())
        if not items:
            return None

        datasets = []
        for item in items:
            asset = item.assets.get("data")
            if asset is None:
                continue
            signed_href = planetary_computer.sign(asset).href
            try:
                datasets.append(rasterio.open(signed_href))
            except Exception:
                continue

        if not datasets:
            return None

        if len(datasets) == 1:
            elev = datasets[0].read(1).astype(float)
            transform = datasets[0].transform
        else:
            merged, transform = _rio_merge(datasets)
            elev = merged[0].astype(float)

        for ds in datasets:
            ds.close()

        nrows, ncols = elev.shape
        # Cell-centre coordinates in WGS84 (lon = x-axis, lat = y-axis)
        xs = transform.c + (np.arange(ncols) + 0.5) * transform.a   # longitudes, E-increasing
        ys = transform.f + (np.arange(nrows) + 0.5) * transform.e   # latitudes, N-to-S (decreasing)

        elev[elev < -500] = 0.0  # nodata / ocean → sea level

        # RegularGridInterpolator requires monotonically increasing axes — flip north-up rasters
        if ys[0] > ys[-1]:
            ys = ys[::-1]
            elev = elev[::-1, :]

        try:
            _dem_cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_dem_cache, ys=ys, xs=xs, elev=elev)
            print(f"[dem] saved cache {_dem_cache.name}")
        except Exception as _se:
            print(f"[dem] cache save failed: {_se}")

        print(f"[dem] Copernicus DEM GLO-30: {nrows}×{ncols} cells, {len(items)} tile(s)")
        return _sampler_from_grid(ys, xs, elev)

    except Exception as _e:
        print(f"[dem] skipped: {_e}")
        return None


def _fetch_land_fill_mesh(
    center: tuple,
    radius: float,
    proj_str: str,
) -> "pv.PolyData | None":
    """Fetch OSM land-use / natural polygons and build a flat coloured fill mesh.

    The mesh lives at Z = 0.02 m — above the base ground plane (Z = -0.10) but
    below road surfaces (Z = 0.10).  Per-cell 'RGB' scalars are attached so one
    add_mesh(scalars='RGB', rgb=True) call renders all categories at once.
    Returns None on any failure; never raises.
    """
    FILL_Z = 0.02

    def _hex(h: str) -> np.ndarray:
        h = h.lstrip("#")
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.uint8)

    _COLOUR: dict[str, np.ndarray] = {
        # ── water bodies ─────────────────────────────────────────────
        "water":             _hex("#2e6ea6"),   # deep sea blue
        "bay":               _hex("#2e6ea6"),
        "strait":            _hex("#2e6ea6"),
        "river":             _hex("#3578b0"),
        "reservoir":         _hex("#3578b0"),
        "lake":              _hex("#3578b0"),
        "pond":              _hex("#3578b0"),
        "riverbank":         _hex("#3578b0"),
        "dock":              _hex("#2a5a8a"),
        "wetland":           _hex("#4a7858"),   # blue-green tint
        # ── natural / rural ───────────────────────────────────────────
        "farmland":          _hex("#8a8e6a"),   # muted olive
        "farmyard":          _hex("#8a8e6a"),
        "sand":              _hex("#c8c090"),   # pale dune
        "beach":             _hex("#c8c090"),
        # ── urban built areas — cool slate palette ────────────────────
        "residential":       _hex("#474d55"),   # dark cool concrete
        "commercial":        _hex("#4e545c"),   # slightly lighter
        "retail":            _hex("#4e545c"),
        "industrial":        _hex("#373c42"),   # heavier, darker
        "construction":      _hex("#3e4349"),   # raw concrete
        "brownfield":        _hex("#3a3f45"),
        "parking":           _hex("#565b62"),   # light asphalt
        # ── soft green cover ──────────────────────────────────────────
        "scrub":             _hex("#5a7850"),   # muted scrub
        "heath":             _hex("#5a7850"),
        "grassland":         _hex("#527848"),
        "grass":             _hex("#3e7838"),
        "meadow":            _hex("#3e7838"),
        "recreation_ground": _hex("#3a8040"),
        "village_green":     _hex("#3a8040"),
        "allotments":        _hex("#4a7840"),   # slightly darker
        "cemetery":          _hex("#4a7048"),   # quieter green
        "garden":            _hex("#3e8038"),
        "park":              _hex("#367832"),   # vivid park green
        "pitch":             _hex("#2a7028"),   # sports pitch
        "playground":        _hex("#3a8040"),
        "wood":              _hex("#255828"),   # deep forest
        "forest":            _hex("#255828"),
    }

    try:
        gdf = ox.features_from_point(center, tags={
            "landuse": list({"farmland", "farmyard", "residential", "commercial", "retail",
                              "industrial", "construction", "brownfield", "grass", "meadow",
                              "recreation_ground", "village_green", "allotments", "cemetery",
                              "garden", "forest", "reservoir"}),
            "natural": list({"grassland", "wetland", "scrub", "heath", "sand", "beach", "wood",
                              "water", "bay", "strait"}),
            "waterway": list({"riverbank", "dock"}),
            "water": list({"river", "lake", "pond", "reservoir"}),
            "leisure": list({"park", "garden", "recreation_ground", "pitch", "playground"}),
            "amenity": ["parking"],
        }, dist=float(radius))
    except Exception as exc:
        print(f"[fill] OSM land-use fetch failed: {exc}")
        return None

    if gdf.empty:
        print("[fill] no land-use polygons found")
        return None

    try:
        proj_gdf = gdf.to_crs(proj_str)
    except Exception as exc:
        print(f"[fill] CRS projection failed: {exc}")
        return None

    from shapely.geometry import Point as _FPt
    clip_circle = _FPt(0.0, 0.0).buffer(float(radius) * 1.05)

    all_pts:   list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_rgb:   list[np.ndarray] = []
    total_verts = 0

    for _, row in proj_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lu  = str(row.get("landuse",   "") or "").lower()
        nat = str(row.get("natural",   "") or "").lower()
        lei = str(row.get("leisure",   "") or "").lower()
        ame = str(row.get("amenity",   "") or "").lower()
        wwy = str(row.get("waterway",  "") or "").lower()
        wat = str(row.get("water",     "") or "").lower()
        cat = lu or nat or lei or ame or wwy or wat
        rgb = _COLOUR.get(cat)
        if rgb is None:
            continue

        try:
            clipped = geom.intersection(clip_circle)
        except Exception:
            clipped = geom
        if clipped.is_empty:
            continue

        for poly in _iter_polygon_parts(clipped):
            if poly.is_empty or poly.exterior is None:
                continue
            coords = np.asarray(poly.exterior.coords, dtype=float)
            if len(coords) < 4:
                continue
            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            n = len(coords)
            pts = np.zeros((n, 3), dtype=float)
            pts[:, 0] = coords[:, 0]
            pts[:, 1] = coords[:, 1]
            pts[:, 2] = FILL_Z
            face = np.empty(n + 1, dtype=np.int64)
            face[0] = n
            face[1:] = np.arange(total_verts, total_verts + n, dtype=np.int64)
            all_pts.append(pts)
            all_faces.append(face)
            all_rgb.append(rgb)
            total_verts += n

    if not all_pts:
        print("[fill] no valid polygons after clipping")
        return None

    try:
        pts_arr   = np.concatenate(all_pts,   axis=0).astype(float)
        faces_arr = np.concatenate(all_faces,  axis=0).astype(np.int64)
        rgb_arr   = np.array(all_rgb, dtype=np.uint8)
        fill_mesh = pv.PolyData(pts_arr, faces_arr)
        fill_mesh.cell_data["RGB"] = rgb_arr
        print(f"[fill] land-use mesh: {fill_mesh.n_cells} polygons, {fill_mesh.n_points} verts")
        return fill_mesh
    except Exception as exc:
        print(f"[fill] mesh construction failed: {exc}")
        return None


def build_3d_city_with_street_surfaces(
    address: str,
    radius: float,
    extrusion_height: float = 10.0,
    road_buffer_m: float = 3.0,
    sidewalk_buffer_m: float = 1.5,
    road_extrude_z: float = 0.1,
    sidewalk_extrude_z: float = 0.15,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData, nx.MultiDiGraph]:
    """Fetch OSM data and return building, road, sidewalk meshes + projected graph."""
    buildings_mesh, projected_graph, _places, _park_mesh, _water_mesh = build_3d_buildings_and_street_graph(
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
    _dem = projected_graph.graph.get("terrain_sampler")
    if _dem is not None:
        for _tm in (road_mesh, sidewalk_mesh, _park_mesh):
            if _tm is not None and hasattr(_tm, "points") and _tm.n_points > 0:
                _pts = _tm.points.copy()
                _pts[:, 2] = _dem(_pts[:, :2]) + _pts[:, 2]
                _tm.points = _pts
    return buildings_mesh, road_mesh, sidewalk_mesh, projected_graph, _park_mesh


def build_3d_buildings_and_street_graph(
    address: str,
    radius: float,
    extrusion_height: float = 10.0,
    cache_dir=None,
    use_cache: bool = False,
    cache_context_key: str = "",
    use_dem: bool = True,
    use_ms_buildings: bool = True,
) -> Tuple[pv.PolyData, nx.MultiDiGraph, list]:
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
    proj_str = f"+proj=tmerc +lat_0={center[0]} +lon_0={center[1]} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    building_tags = {"building": True}
    try:
        buildings: gpd.GeoDataFrame = ox.features_from_point(
            center,
            tags=building_tags,
            dist=radius,
        )
    except Exception:
        buildings = gpd.GeoDataFrame()

    try:
        street_graph = ox.graph_from_point(
            center,
            dist=radius,
            network_type="all",
            simplify=True,
        )
    except Exception:
        street_graph = nx.MultiDiGraph()

    if not buildings.empty:
        projected_buildings = buildings.to_crs(proj_str)
    else:
        projected_buildings = gpd.GeoDataFrame()
        
    if len(street_graph) > 0:
        projected_graph = ox.projection.project_graph(street_graph, to_crs=proj_str)
    else:
        projected_graph = nx.MultiDiGraph()
        projected_graph.graph["crs"] = proj_str

    projected_graph.graph["scene_lat"] = float(center[0])
    projected_graph.graph["scene_lon"] = float(center[1])

    # Fetch Parks and Parking
    park_tags = {"leisure": "park", "landuse": ["grass", "meadow", "recreation_ground", "village_green"]}
    try:
        parks: gpd.GeoDataFrame = ox.features_from_point(center, tags=park_tags, dist=radius)
        if not parks.empty:
            projected_parks = parks.to_crs(proj_str)
        else:
            projected_parks = gpd.GeoDataFrame()
    except Exception:
        projected_parks = gpd.GeoDataFrame()

    parking_tags = {"amenity": "parking"}
    try:
        parking: gpd.GeoDataFrame = ox.features_from_point(center, tags=parking_tags, dist=radius)
        if not parking.empty:
            projected_parking = parking.to_crs(proj_str)
        else:
            projected_parking = gpd.GeoDataFrame()
    except Exception:
        projected_parking = gpd.GeoDataFrame()

    # Fetch individual tree nodes and forest areas
    try:
        _trees_gdf = ox.features_from_point(center, tags={"natural": "tree"}, dist=radius)
        proj_trees = _trees_gdf.to_crs(proj_str) if not _trees_gdf.empty else gpd.GeoDataFrame()
    except Exception:
        proj_trees = gpd.GeoDataFrame()
    try:
        _forests_gdf = ox.features_from_point(center, tags={"landuse": "forest"}, dist=radius)
        proj_forests = _forests_gdf.to_crs(proj_str) if not _forests_gdf.empty else gpd.GeoDataFrame()
    except Exception:
        proj_forests = gpd.GeoDataFrame()
    try:
        _cross_gdf = ox.features_from_point(center, tags={"highway": "crossing"}, dist=radius)
        proj_crossings = _cross_gdf.to_crs(proj_str) if not _cross_gdf.empty else gpd.GeoDataFrame()
    except Exception:
        proj_crossings = gpd.GeoDataFrame()

    bbox = _radius_to_bbox(float(center[0]), float(center[1]), float(radius))
    osm_direction_graph = load_osm_direction_graph_cached(
        bbox_wsen=bbox,
        target_crs=projected_graph.graph.get("crs"),
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_context_key=cache_context_key,
    )
    projected_graph.graph["source_bbox_wsen"] = bbox
    
    from shapely.geometry import Point as _Pt
    from shapely.geometry import MultiLineString as _MLS
    clip_circle = _Pt(0.0, 0.0).buffer(float(radius))

    for u, v, data in projected_graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            nu = projected_graph.nodes[u]
            nv = projected_graph.nodes[v]
            if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
                geom = LineString([(float(nu["x"]), float(nu["y"])), (float(nv["x"]), float(nv["y"]))])
        
        if geom is not None:
            clipped = geom.intersection(clip_circle)
            if clipped.is_empty:
                continue
            if isinstance(clipped, _MLS):
                parts = list(clipped.geoms)
                clipped = max(parts, key=lambda g: g.length)
            if isinstance(clipped, LineString) and not clipped.is_empty:
                data["geometry"] = clipped
                geom = clipped
                
        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=float)
            if coords.shape[0] >= 2:
                oneway, legal_forward = resolve_osm_road_direction(
                    (float(coords[0][0]), float(coords[0][1])),
                    (float(coords[-1][0]), float(coords[-1][1])),
                    osm_direction_graph,
                )
                data["oneway"] = bool(data.get("oneway", False)) if oneway is None else bool(oneway)
                data["oneway_legal_forward"] = bool(legal_forward) if legal_forward is not None else True


    _ms_centroids, _ms_heights, _ms_tree = (
        _fetch_ms_building_heights(float(center[0]), float(center[1]), float(radius), proj_str)
        if use_ms_buildings else (None, None, None)
    )

    _dem_sampler = (
        _fetch_dem_sampler(float(center[0]), float(center[1]), float(radius), proj_str)
        if use_dem else None
    )
    if _dem_sampler is not None:
        projected_graph.graph["terrain_sampler"] = _dem_sampler

    extruded_meshes: list[pv.PolyData] = []
    _n_buildings = 0
    res_points = []
    com_points = []

    for _, row in projected_buildings.iterrows():
        geometry = row.geometry
        if geometry is None:
            continue

        _ms_default = float(extrusion_height)
        if _ms_tree is not None:
            try:
                _c = geometry.centroid
                _ms_dist, _ms_idx = _ms_tree.query([float(_c.x), float(_c.y)])
                if _ms_dist < 30.0 and float(_ms_heights[_ms_idx]) > 0.0:
                    _ms_default = float(_ms_heights[_ms_idx])
            except Exception:
                pass
        building_height, roof_shape = _resolve_building_height_and_roof(row, _ms_default)

        # NOTE: no DEM lift here — the scene renders flat by default and
        # TerrainMixin._drape_buildings lifts building actors dynamically when
        # the Terrain toggle is on.  Baking the lift made buildings float
        # above the flat roads (and double-lifted them with terrain active).

        for poly in _iter_polygon_parts(geometry):
            footprint = _polygon_to_footprint(poly)
            if footprint is None or footprint.n_points < 3:
                continue

            body = footprint.extrude((0.0, 0.0, float(building_height)), capping=True)
            if body.n_points == 0:
                continue

            refined = body
            if roof_shape in {"pyramidal", "gabled", "gable"}:
                # Pitched roof: connect wall-top perimeter to an elevated centroid.
                roof_raise = max(1.5, 0.25 * float(building_height))
                roof_mesh = _make_pitched_roof_mesh(poly, base_height=float(building_height), roof_raise=roof_raise)
                if roof_mesh is not None and roof_mesh.n_points > 0:
                    refined = refined.merge(roof_mesh)
            elif roof_shape in {"", "flat"}:
                # Flat/missing roof shape: add a thin parapet ring on top.
                parapet_mesh = _make_parapet_mesh(poly, base_height=float(building_height))
                if parapet_mesh is not None and parapet_mesh.n_points > 0:
                    refined = refined.merge(parapet_mesh)
            bclass = 0 # concrete
            b_tag = str(row.get("building", "")).lower()
            if b_tag in {"apartments", "house", "residential", "detached", "terrace"}:
                bclass = 4 # residential
                res_points.append((poly.centroid.x, poly.centroid.y))
            elif b_tag in {"commercial", "retail", "office", "supermarket", "shop"}:
                bclass = 3 # commercial
                com_points.append((poly.centroid.x, poly.centroid.y))
                
            refined.cell_data["building_class"] = np.full(refined.n_cells, bclass, dtype=np.uint8)
            extruded_meshes.append(refined)
            _n_buildings += 1
            if _n_buildings % 50 == 0:
                print(f"[build] extruded {_n_buildings} buildings...")

    if not extruded_meshes:
        combined_mesh = pv.PolyData()
    else:
        print(f"[build] merging {len(extruded_meshes)} building meshes...")
        combined_mesh = pv.MultiBlock(extruded_meshes).combine(merge_points=False)
        
    park_meshes = []
    if not projected_parks.empty:
        for _, row in projected_parks.iterrows():
            geometry = row.geometry
            if geometry is None: continue
            for poly in _iter_polygon_parts(geometry):
                footprint = _polygon_to_footprint(poly)
                if footprint is None or footprint.n_points < 3: continue
                park_extruded = footprint.extrude((0.0, 0.0, 0.02), capping=True)
                if park_extruded.n_points > 0:
                    if _dem_sampler is not None:
                        try:
                            _pc = poly.centroid
                            _pz = float(_dem_sampler(np.array([[float(_pc.x), float(_pc.y)]]))[0])
                            if _pz != 0.0:
                                park_extruded = park_extruded.translate([0.0, 0.0, _pz])
                        except Exception:
                            pass
                    park_meshes.append(park_extruded)
    
    if park_meshes:
        park_combined = pv.MultiBlock(park_meshes).combine(merge_points=False)
        park_combined = _apply_surface_color(park_combined, PARK_SURFACE_RGB)
    else:
        park_combined = None

    parking_meshes = []
    if not projected_parking.empty:
        for _, row in projected_parking.iterrows():
            geometry = row.geometry
            if geometry is None: continue
            for poly in _iter_polygon_parts(geometry):
                footprint = _polygon_to_footprint(poly)
                if footprint is None or footprint.n_points < 3: continue
                com_points.append((poly.centroid.x, poly.centroid.y))
                parking_extruded = footprint.extrude((0.0, 0.0, 0.015), capping=True)
                if parking_extruded.n_points > 0:
                    parking_meshes.append(parking_extruded)
                    
    if parking_meshes:
        parking_combined = pv.MultiBlock(parking_meshes).combine(merge_points=False)
        # Dark grey for parking
        parking_combined = _apply_surface_color(parking_combined, np.array([0x1f, 0x1f, 0x1f], dtype=np.uint8))
        if park_combined is not None:
            park_combined = park_combined.merge(parking_combined)
        else:
            park_combined = parking_combined

    # Annotate nearest nodes
    if len(projected_graph) > 0:
        if res_points:
            rx, ry = zip(*res_points)
            r_nodes = ox.nearest_nodes(projected_graph, rx, ry)
            for n in r_nodes:
                projected_graph.nodes[n]["is_residential"] = True
        if com_points:
            cx, cy = zip(*com_points)
            c_nodes = ox.nearest_nodes(projected_graph, cx, cy)
            for n in c_nodes:
                projected_graph.nodes[n]["is_commercial"] = True

    # Collect tree positions (individual nodes + sampled points inside forest areas)
    _tree_data: list[dict] = []
    for _, _tr in proj_trees.iterrows():
        _tg = _tr.geometry
        if _tg is None or not hasattr(_tg, "x"):
            continue  # skip Ways/Relations — only Point nodes
        _sp = str(_tr.get("species", _tr.get("species:en", _tr.get("taxon", "")))).lower()
        _tree_data.append({"x": float(_tg.x), "y": float(_tg.y), "species": _sp})
    from shapely.geometry import Point as _ShPt
    for _, _fr in proj_forests.iterrows():
        _fg = _fr.geometry
        if _fg is None or _fg.is_empty:
            continue
        _fb = _fg.bounds
        _nft = 0
        for _fx in np.arange(_fb[0] + 5.0, _fb[2], 15.0):
            for _fy in np.arange(_fb[1] + 5.0, _fb[3], 15.0):
                if _nft >= 40:
                    break
                if _fg.contains(_ShPt(_fx, _fy)):
                    _tree_data.append({"x": float(_fx), "y": float(_fy), "species": ""})
                    _nft += 1
            if _nft >= 40:
                break
    if _tree_data:
        print(f"[trees] {len(_tree_data)} trees fetched ({sum(1 for t in _tree_data if t['species'])} with species tag)")
    projected_graph.graph["trees"] = _tree_data

    # Build crossing data: position + road direction for each highway=crossing node
    _cross_data: list[dict] = []
    if not proj_crossings.empty:
        _emids: list = []
        _edirs: list = []
        for _eu, _ev, _edata in projected_graph.edges(data=True):
            _eg = _edata.get("geometry")
            if _eg is not None and hasattr(_eg, "coords"):
                _ec = np.asarray(_eg.coords, dtype=float)
            else:
                _un = projected_graph.nodes.get(_eu, {})
                _vn = projected_graph.nodes.get(_ev, {})
                if "x" not in _un or "x" not in _vn:
                    continue
                _ec = np.array([[_un["x"], _un["y"]], [_vn["x"], _vn["y"]]])
            if len(_ec) < 2:
                continue
            _emids.append(_ec[len(_ec) // 2, :2])
            _ev2 = _ec[-1, :2] - _ec[0, :2]
            _en = np.linalg.norm(_ev2)
            _edirs.append(_ev2 / _en if _en > 0 else np.array([1.0, 0.0]))
        if _emids:
            from scipy.spatial import cKDTree as _CKDx
            _ek = _CKDx(np.array(_emids))
            for _, _cr in proj_crossings.iterrows():
                _cg = _cr.geometry
                if _cg is None or not hasattr(_cg, "x"):
                    continue
                _cx2, _cy2 = float(_cg.x), float(_cg.y)
                _, _ei = _ek.query([_cx2, _cy2])
                _dx2, _dy2 = float(_edirs[_ei][0]), float(_edirs[_ei][1])
                _cross_data.append({"x": _cx2, "y": _cy2, "dx": _dx2, "dy": _dy2})
            if len(_cross_data) > 200:
                _cross_data = _cross_data[:200]
    if _cross_data:
        print(f"[crossings] {len(_cross_data)} pedestrian crossings")
    projected_graph.graph["crossings"] = _cross_data

    return combined_mesh, projected_graph, [], park_combined, None
