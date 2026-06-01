"""Overture Maps data source — drop-in replacement for osm_3d_buildings.py.

Public API (identical signatures to osm_3d_buildings.py):
    build_3d_buildings_and_street_graph(address, radius, extrusion_height)
        -> (buildings_mesh: pv.PolyData, street_graph: nx.MultiDiGraph)

    build_road_and_sidewalk_meshes_from_graph(projected_graph, ...)
        -> (road_mesh: pv.PolyData, sidewalk_mesh: pv.PolyData)

Everything downstream (shadow_engine, streetlight_ga, spatial_trees, main.py,
app_core.py) stays completely unchanged.

Install dependencies:
    pip install overturemaps pyproj geopy geopandas shapely pyarrow pyvista networkx
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import networkx as nx
import numpy as np
import pyarrow as pa
import pyvista as pv
from geopy.geocoders import Nominatim
from pyproj import Geod, Transformer
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import unary_union

try:
    import overturemaps
    _OVERTURE_AVAILABLE = True
except ImportError:
    _OVERTURE_AVAILABLE = False
    warnings.warn(
        "overturemaps package not found. Install with: pip install overturemaps",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Overture building subclasses we treat as "pedestrian only" for road filtering
_PED_CLASSES = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway",
                "track", "service"}

# Default height when Overture has no height and no num_floors
DEFAULT_BUILDING_HEIGHT_M = 10.0
METERS_PER_FLOOR = 3.5


# ---------------------------------------------------------------------------
# Geocoding helpers
# ---------------------------------------------------------------------------

def _geocode_address(address: str) -> tuple[float, float]:
    """Return (lat, lon) for a free-text address via Nominatim."""
    geolocator = Nominatim(user_agent="city_digital_twin/1.0")
    location = geolocator.geocode(address, timeout=15)
    if location is None:
        raise ValueError(f"Could not geocode address: {address!r}")
    return float(location.latitude), float(location.longitude)


def _radius_to_bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Convert (lat, lon, radius_m) to (west, south, east, north) bounding box.

    Uses a WGS-84 geodesic (pyproj.Geod) for accurate degree offsets at any
    latitude — much more precise than the equatorial approximation that was
    used previously.
    """
    geod = Geod(ellps="WGS84")
    # fwd(lon, lat, azimuth_deg, distance_m) → (lon_out, lat_out, back_az)
    _, lat_n, _ = geod.fwd(lon, lat,   0, radius_m)   # north
    _, lat_s, _ = geod.fwd(lon, lat, 180, radius_m)   # south
    lon_e, _, _ = geod.fwd(lon, lat,  90, radius_m)   # east
    lon_w, _, _ = geod.fwd(lon, lat, 270, radius_m)   # west
    return (lon_w, lat_s, lon_e, lat_n)


# ---------------------------------------------------------------------------
# Projection helpers (WGS-84 → local metric CRS, like OSMnx does)
# ---------------------------------------------------------------------------

def _make_local_transformer(lat: float, lon: float) -> tuple[Transformer, Transformer]:
    """
    Build forward (WGS84→metric) and inverse (metric→WGS84) Transformers
    using an Azimuthal Equidistant projection centred on (lat, lon).
    This gives distances in metres from the scene origin — identical in spirit
    to what OSMnx does with its UTM / local CRS projection.
    """
    proj_str = (
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    to_wgs84 = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    return to_local, to_wgs84


def _project_coords_to_local(
    coords: list[tuple[float, float]],
    transformer: Transformer,
) -> list[tuple[float, float]]:
    """Project [(lon, lat), ...] → [(x_m, y_m), ...] in local metric CRS."""
    projected = []
    for lon, lat in coords:
        x, y = transformer.transform(lon, lat)
        projected.append((x, y))
    return projected


def _project_polygon(poly: Polygon, transformer: Transformer) -> Polygon:
    """Project a Shapely Polygon from WGS-84 to local metric CRS."""
    exterior = _project_coords_to_local(list(poly.exterior.coords), transformer)
    interiors = [
        _project_coords_to_local(list(ring.coords), transformer)
        for ring in poly.interiors
    ]
    return Polygon(exterior, interiors)


def _project_linestring(line: LineString, transformer: Transformer) -> LineString:
    pts = _project_coords_to_local(list(line.coords), transformer)
    return LineString(pts)


# ---------------------------------------------------------------------------
# Overture: fetch buildings
# ---------------------------------------------------------------------------

def _fetch_overture_buildings(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """
    Fetch building features from Overture Maps for the given bbox.

    Returns a list of dicts, each with:
        geometry  : Shapely Polygon/MultiPolygon (WGS-84)
        height    : float (metres) — may be None
        num_floors: int — may be None
        subtype   : str — may be None
    """
    if not _OVERTURE_AVAILABLE:
        raise RuntimeError("overturemaps package is required. pip install overturemaps")

    west, south, east, north = bbox
    reader = overturemaps.record_batch_reader("building", bbox=bbox)

    buildings = []
    import shapely.wkb as _shapely_wkb  # FIXED: import once per call, not once per row
    for batch in reader:
        batch_dict = batch.to_pydict()
        n = len(batch_dict["geometry"])

        for i in range(n):
            raw_geom = batch_dict["geometry"][i]
            if raw_geom is None:
                continue

            # geometry column is WKB bytes in recent Overture releases
            try:
                geom = _shapely_wkb.loads(bytes(raw_geom))
            except Exception:
                # Fallback: try as GeoJSON dict
                try:
                    geom = shape(raw_geom)
                except Exception:
                    continue

            if geom is None or geom.is_empty:
                continue

            height = batch_dict.get("height", [None] * n)[i]
            num_floors = batch_dict.get("num_floors", [None] * n)[i]
            subtype = batch_dict.get("subtype", [None] * n)[i]

            facade_material = batch_dict.get("facade_material", [None] * n)[i]
            roof_shape = batch_dict.get("roof_shape", [None] * n)[i]
            # Fallback: some releases nest these inside a properties struct
            if facade_material is None or roof_shape is None:
                try:
                    props = batch_dict.get("properties", [None] * n)[i]
                    if isinstance(props, dict):
                        facade_material = facade_material or props.get("facade_material")
                        roof_shape = roof_shape or props.get("roof_shape")
                except Exception:
                    pass

            buildings.append({
                "geometry": geom,
                "height": float(height) if height is not None else None,
                "num_floors": int(num_floors) if num_floors is not None else None,
                "subtype": str(subtype) if subtype is not None else None,
                "facade_material": str(facade_material) if facade_material is not None else None,
                "roof_shape": str(roof_shape) if roof_shape is not None else None,
            })

    return buildings


def _resolve_height(b: dict, default: float) -> float:
    """Pick the best available height for a building feature."""
    if b["height"] is not None and b["height"] > 0:
        return b["height"]
    if b["num_floors"] is not None and b["num_floors"] > 0:
        return b["num_floors"] * METERS_PER_FLOOR
    return default


# ---------------------------------------------------------------------------
# Segment field parsing helpers
# ---------------------------------------------------------------------------

def _seg_road_col(d: dict, i: int, n: int) -> dict | None:
    """Extract the road property struct for segment row i (defensive)."""
    try:
        col = d.get("road", [None] * n)[i]
        return col if isinstance(col, dict) else None
    except Exception:
        return None


def _parse_speed_limits(road: dict | None) -> list[dict]:
    """Parse speed_limits list from road struct → list of {value_kmh, when}."""
    if not isinstance(road, dict):
        return []
    raw = road.get("speed_limits") or []
    if not isinstance(raw, (list, tuple)):
        return []
    result = []
    for sl in raw:
        if not isinstance(sl, dict):
            continue
        ms = sl.get("max_speed") or {}
        if not isinstance(ms, dict):
            try:
                result.append({"value_kmh": float(ms), "when": None})
            except Exception:
                pass
            continue
        val = ms.get("value")
        if val is None:
            continue
        unit = str(ms.get("unit") or "km/h").lower()
        val_kmh = float(val) * (1.60934 if "mph" in unit else 1.0)
        result.append({"value_kmh": round(val_kmh, 1), "when": sl.get("when")})
    return result


def _parse_access_restrictions(road: dict | None) -> list[dict]:
    """Parse access_restrictions list from road struct."""
    if not isinstance(road, dict):
        return []
    raw = road.get("access_restrictions") or []
    if not isinstance(raw, (list, tuple)):
        return []
    result = []
    for ar in raw:
        if not isinstance(ar, dict):
            continue
        access_type = ar.get("access_type")
        vehicles = (
            ar.get("vehicle_types")
            or ar.get("vehicle")
            or ar.get("vehicle_categories")
        )
        result.append({
            "access_type": str(access_type) if access_type is not None else None,
            "vehicles": vehicles,
        })
    return result


def _parse_road_surface_str(road: dict | None) -> str | None:
    """Extract road surface string from road struct."""
    if not isinstance(road, dict):
        return None
    surfaces = road.get("road_surface") or []
    if isinstance(surfaces, (list, tuple)) and surfaces:
        first = surfaces[0]
        if isinstance(first, dict):
            v = first.get("value")
            return str(v) if v is not None else None
        return str(first) if first is not None else None
    if isinstance(surfaces, str):
        return surfaces
    return None


def _parse_width_m(road: dict | None) -> float | None:
    """Extract first width rule value in metres from road struct."""
    if not isinstance(road, dict):
        return None
    rules = road.get("width_rules") or []
    if isinstance(rules, (list, tuple)) and rules:
        first = rules[0]
        if isinstance(first, dict):
            v = first.get("value")
            try:
                return float(v) if v is not None else None
            except Exception:
                return None
        try:
            return float(first)
        except Exception:
            return None
    return None


def _parse_lanes_count(road: dict | None) -> int | None:
    """Extract lane count from road.lanes (list length or explicit count field)."""
    if not isinstance(road, dict):
        return None
    lanes = road.get("lanes")
    if lanes is None:
        return None
    if isinstance(lanes, int):
        return lanes if lanes > 0 else None
    if isinstance(lanes, (list, tuple)):
        return len(lanes) if lanes else None
    if isinstance(lanes, dict):
        for key in ("count", "value", "total"):
            v = lanes.get(key)
            if v is not None:
                try:
                    c = int(v)
                    return c if c > 0 else None
                except Exception:
                    pass
    return None


def _parse_connector_ids(d: dict, i: int, n: int) -> list[str]:
    """Extract connector IDs from the segment connectors column."""
    result = []
    try:
        conn_col = d.get("connectors", [None] * n)[i]
        if isinstance(conn_col, (list, tuple)):
            for entry in conn_col:
                if isinstance(entry, dict):
                    cid = entry.get("connector_id") or entry.get("id")
                    if cid is not None:
                        result.append(str(cid))
                elif entry is not None:
                    result.append(str(entry))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Overture: fetch segments (road network)
# ---------------------------------------------------------------------------

def _fetch_overture_segments(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """
    Fetch transportation segment features from Overture Maps.

    Returns list of dicts with keys:
        id                 : str | None
        geometry           : Shapely LineString (WGS-84)
        subtype            : str
        class_             : str
        is_oneway          : bool
        road_surface       : str | None
        speed_limits       : list[dict]   — [{value_kmh, when}, ...]
        access_restrictions: list[dict]   — [{access_type, vehicles}, ...]
        width_m            : float | None
        lanes              : int | None
        connector_ids      : list[str]
    """
    if not _OVERTURE_AVAILABLE:
        raise RuntimeError("overturemaps package is required. pip install overturemaps")

    reader = overturemaps.record_batch_reader("segment", bbox=bbox)

    segments = []
    import shapely.wkb as _shapely_wkb  # FIXED: import once per call, not once per row
    for batch in reader:
        d = batch.to_pydict()
        n = len(d["geometry"])
        seg_ids = d.get("id", [None] * n)

        for i in range(n):
            raw_geom = d["geometry"][i]
            if raw_geom is None:
                continue
            try:
                geom = _shapely_wkb.loads(bytes(raw_geom))
            except Exception:
                try:
                    geom = shape(raw_geom)
                except Exception:
                    continue

            if geom is None or geom.is_empty:
                continue
            if not isinstance(geom, LineString):
                continue

            subtype = d.get("subtype", [None] * n)[i]
            class_ = d.get("class", [None] * n)[i]
            road = _seg_road_col(d, i, n)

            # Try multiple Overture schema locations for oneway flag
            is_oneway = False
            try:
                if isinstance(road, dict):
                    # Schema v1: road.restrictions.use_as_through_traffic
                    restrictions = road.get("restrictions", {})
                    if isinstance(restrictions, dict):
                        if restrictions.get("use_as_through_traffic") == "no":
                            is_oneway = True
                    # Schema v2: road.flags list (older releases)
                    flags = road.get("flags", [])
                    if isinstance(flags, list):
                        if any("one_way" in str(f).lower() or "oneway" in str(f).lower() for f in flags):
                            is_oneway = True
                    # Schema v3: road.lanes list — if all lanes have same direction
                    lanes = road.get("lanes", [])
                    if isinstance(lanes, list) and len(lanes) > 0:
                        directions = set()
                        for lane in lanes:
                            if isinstance(lane, dict):
                                d_val = lane.get("direction", "")
                                if d_val:
                                    directions.add(str(d_val).lower())
                        if directions == {"forward"} or directions == {"backward"}:
                            is_oneway = True
            except Exception:
                pass

            # road_surface: check road struct first, then top-level column
            road_surface = _parse_road_surface_str(road)
            if road_surface is None:
                try:
                    rs_col = d.get("road_surface", [None] * n)[i]
                    if isinstance(rs_col, dict):
                        road_surface = rs_col.get("value")
                    elif isinstance(rs_col, str):
                        road_surface = rs_col
                except Exception:
                    pass

            seg_dict = {
                "id": str(seg_ids[i]) if seg_ids[i] is not None else None,
                "geometry": geom,
                "subtype": str(subtype) if subtype else "road",
                "class_": str(class_) if class_ else "unclassified",
                "is_oneway": is_oneway,
                "road_surface": road_surface,
                "speed_limits": _parse_speed_limits(road),
                "access_restrictions": _parse_access_restrictions(road),
                "width_m": _parse_width_m(road),
                "lanes": _parse_lanes_count(road),
                "connector_ids": _parse_connector_ids(d, i, n),
            }
            seg_dict["_raw_road"] = road  # For inspection
            segments.append(seg_dict)

    oneway_count = sum(1 for s in segments if s.get("is_oneway", False))
    print(f"[Overture] One-way segments: {oneway_count} / {len(segments)}")
    if oneway_count == 0 and segments:
        print(f"[Overture] Sample road column: {segments[0].get('_raw_road')}")
    return segments


# ---------------------------------------------------------------------------
# Build 3D PyVista mesh from buildings
# ---------------------------------------------------------------------------

def _roof_apex_height(polygon: Polygon) -> float:
    """Estimate a sensible roof apex height above the wall top for a given footprint."""
    # Use the inscribed-circle approximation: roof height ≈ 30% of inradius, clamped.
    inradius = (2.0 * polygon.area / max(1e-6, polygon.length))
    return float(np.clip(inradius * 0.30, 1.5, 8.0))


def _polygon_to_pyvista(
    polygon: Polygon,
    height: float,
    roof_shape: str = "flat",
) -> Optional[pv.PolyData]:
    """Extrude a 2D polygon (local metric coords) into a 3D PyVista PolyData.

    Supported roof_shape values
    ---------------------------
    flat                     : horizontal cap at `height`  (default)
    pyramidal / hipped /
      dome / onion           : single apex above centroid
    gabled / saltbox /
      gambrel / mansard      : ridge along the long axis of the footprint
    """
    if polygon.is_empty or not polygon.is_valid:
        return None

    try:
        coords = np.asarray(polygon.exterior.coords, dtype=float)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) < 3:
            return None

        n = len(coords)
        xy = coords[:, :2]
        bottom = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(n)])
        top    = np.column_stack([xy[:, 0], xy[:, 1], np.full(n, height)])

        faces: list[int] = []

        # ── Side walls (shared by all roof types) ──────────────────────────
        for j in range(n):
            jn = (j + 1) % n
            faces.extend([3, j,      jn,      j  + n])
            faces.extend([3, jn,     jn + n,  j  + n])

        # ── Bottom cap ─────────────────────────────────────────────────────
        bc = np.mean(bottom, axis=0)
        bc_idx = 2 * n
        pts = np.vstack([bottom, top, bc.reshape(1, 3)])
        for j in range(n):
            jn = (j + 1) % n
            faces.extend([3, jn, j, bc_idx])  # reversed winding → outward normal

        # ── Roof ───────────────────────────────────────────────────────────
        rs = str(roof_shape).lower() if roof_shape else "flat"

        if rs in {"pyramidal", "hipped", "dome", "onion"}:
            # Single apex above centroid
            roof_h  = _roof_apex_height(polygon)
            cx, cy  = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
            apex    = np.array([[cx, cy, height + roof_h]])
            apex_idx = pts.shape[0]
            pts = np.vstack([pts, apex])
            for j in range(n):
                jn = (j + 1) % n
                faces.extend([3, j + n, jn + n, apex_idx])

        elif rs in {"gabled", "saltbox", "gambrel", "mansard"}:
            # Ridge along the long axis of the minimum rotated bounding rectangle.
            roof_h = _roof_apex_height(polygon)
            mrr    = polygon.minimum_rotated_rectangle
            mc     = np.asarray(mrr.exterior.coords[:-1], dtype=float)  # 4 corners

            d01 = np.linalg.norm(mc[1] - mc[0])
            d12 = np.linalg.norm(mc[2] - mc[1])
            if d01 >= d12:          # long axis is 0→1 (and 3→2)
                rp1 = (mc[0] + mc[3]) * 0.5
                rp2 = (mc[1] + mc[2]) * 0.5
            else:                   # long axis is 1→2 (and 0→3)
                rp1 = (mc[0] + mc[1]) * 0.5
                rp2 = (mc[3] + mc[2]) * 0.5

            r1 = np.array([[rp1[0], rp1[1], height + roof_h]])
            r2 = np.array([[rp2[0], rp2[1], height + roof_h]])
            r1_idx = pts.shape[0]
            r2_idx = pts.shape[0] + 1
            pts = np.vstack([pts, r1, r2])

            # Assign each top-ring vertex to its nearest ridge endpoint.
            def _near(j: int) -> int:
                d1 = (xy[j, 0] - rp1[0])**2 + (xy[j, 1] - rp1[1])**2
                d2 = (xy[j, 0] - rp2[0])**2 + (xy[j, 1] - rp2[1])**2
                return r1_idx if d1 <= d2 else r2_idx

            assign = [_near(j) for j in range(n)]
            for j in range(n):
                jn   = (j + 1) % n
                ri_j = assign[j]
                ri_n = assign[jn]
                if ri_j == ri_n:
                    faces.extend([3, j + n, jn + n, ri_j])
                else:
                    # Quad spanning the ridge: split into two triangles.
                    faces.extend([3, j  + n, jn + n, ri_j])
                    faces.extend([3, jn + n, ri_n,   ri_j])

        else:  # flat (default) — horizontal cap
            tc      = np.mean(top, axis=0)
            tc_idx  = pts.shape[0]
            pts = np.vstack([pts, tc.reshape(1, 3)])
            for j in range(n):
                jn = (j + 1) % n
                faces.extend([3, j + n, jn + n, tc_idx])

        mesh = pv.PolyData(pts, np.asarray(faces, dtype=np.int32))
        return mesh.clean()
    except Exception:
        return None


def _build_buildings_mesh(
    buildings: list[dict],
    transformer: Transformer,
    extrusion_height: float,
) -> pv.PolyData:
    """Convert Overture building features → single merged PyVista PolyData."""
    meshes = []
    n_flat = n_pyramidal = n_gabled = n_other = 0
    class_map = {"concrete": 0, "brick": 1, "glass": 2, "wood": 0, "metal": 2, "stone": 1}
    for b in buildings:
        geom = b["geometry"]
        height = _resolve_height(b, extrusion_height)
        # Use facade_material for building_class, fallback to height heuristic
        material = b.get("facade_material") or "concrete"
        bclass = class_map.get(str(material).lower(), 0)
        roof_shape = b.get("roof_shape") or "flat"

        polys = (
            list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        )
        for poly in polys:
            local_poly = _project_polygon(poly, transformer)
            mesh = _polygon_to_pyvista(local_poly, height, roof_shape=roof_shape)
            if mesh is not None and mesh.n_points > 0:
                mesh.cell_data["building_class"] = np.full(mesh.n_cells, bclass, dtype=np.uint8)
                meshes.append(mesh)

        rs = str(roof_shape).lower()
        if rs == "flat" or rs == "none":
            n_flat += 1
        elif rs in {"pyramidal", "hipped", "dome", "onion"}:
            n_pyramidal += 1
        elif rs in {"gabled", "saltbox", "gambrel", "mansard"}:
            n_gabled += 1
        else:
            n_other += 1

    print(f"[overture] roofs — flat:{n_flat}  pyramidal/hipped:{n_pyramidal}  "
          f"gabled:{n_gabled}  other:{n_other}")

    if not meshes:
        return pv.PolyData()
    if len(meshes) == 1:
        return meshes[0]
    combined = meshes[0].merge(meshes[1:], merge_points=False)
    return combined


# ---------------------------------------------------------------------------
# Build NetworkX street graph from Overture segments
# ---------------------------------------------------------------------------

def _build_street_graph(
    segments: list[dict],
    transformer: Transformer,
    clip_radius_m: float | None = None,
) -> nx.MultiDiGraph:
    """
    Build a NetworkX MultiDiGraph from Overture segment features.

    Each node carries:
        x, y   : float  (local metric CRS, metres from scene origin)
        osmid  : int    (positive hash of rounded (x, y) — stable per session)

    Each edge carries:
        highway            : str   (= class_, for downstream OSMnx compatibility)
        geometry           : Shapely LineString in local metric CRS
        length             : float (metres)
        oneway             : bool
        maxspeed           : float | None  (km/h, first unconditional speed limit)
        lanes              : int | None
        surface            : str | None
        speed_limits       : list[dict]
        access_restrictions: list[dict]
        width_m            : float | None
        segment_id         : str | None
    """
    G = nx.MultiDiGraph()
    node_map: dict[tuple[float, float], int] = {}
    node_counter = 0

    # segment_id → list[connector_id] for future turn-restriction use
    segment_to_connectors: dict[str, list[str]] = {}

    def _get_or_create_node(xy: tuple[float, float]) -> int:
        nonlocal node_counter
        key = (round(xy[0], 2), round(xy[1], 2))
        if key not in node_map:
            node_id = node_counter  # FIXED: use counter as ID, not hash() which can collide
            node_map[key] = node_id
            G.add_node(node_id, x=key[0], y=key[1], osmid=node_id)
            node_counter += 1
        return node_map[key]

    for seg in segments:
        geom_wgs84: LineString = seg["geometry"]
        local_line = _project_linestring(geom_wgs84, transformer)

        # Clip the projected geometry to the scene radius.
        # Overture returns the *full* geometry of any road that intersects the
        # search bbox — a highway that starts 90 km away and passes through the
        # scene is returned in full.  Clipping here keeps only the in-scene
        # portion and prevents outlier coordinates from bloating every downstream
        # mesh and the camera frustum.
        if clip_radius_m is not None and clip_radius_m > 0:
            from shapely.geometry import Point as _Pt
            _clip_circle = _Pt(0.0, 0.0).buffer(float(clip_radius_m))
            _clipped = local_line.intersection(_clip_circle)
            if _clipped.is_empty:
                continue
            # intersection may return MultiLineString for roads that re-enter
            from shapely.geometry import MultiLineString as _MLS
            if isinstance(_clipped, _MLS):
                # keep longest segment
                parts = list(_clipped.geoms)
                _clipped = max(parts, key=lambda g: g.length)
            if not isinstance(_clipped, LineString) or _clipped.is_empty:
                continue
            local_line = _clipped

        coords = list(local_line.coords)
        if len(coords) < 2:
            continue

        start_xy = coords[0]
        end_xy = coords[-1]
        u = _get_or_create_node(start_xy)
        v = _get_or_create_node(end_xy)
        length = local_line.length

        # Derive maxspeed: prefer first unconditional (when=None) speed limit
        speed_limits = seg.get("speed_limits", [])
        maxspeed: float | None = None
        for sl in speed_limits:
            if isinstance(sl, dict) and sl.get("when") is None:
                maxspeed = sl.get("value_kmh")
                break
        if maxspeed is None and speed_limits:
            first = speed_limits[0]
            if isinstance(first, dict):
                maxspeed = first.get("value_kmh")

        seg_id = seg.get("id")
        conn_ids = seg.get("connector_ids", [])
        if seg_id and conn_ids:
            segment_to_connectors[seg_id] = conn_ids

        edge_data: dict = {
            "highway": seg["class_"],
            "geometry": local_line,
            "length": length,
            "oneway": seg["is_oneway"],
            "maxspeed": maxspeed,
            "lanes": seg.get("lanes"),
            "surface": seg.get("road_surface"),
            "speed_limits": speed_limits,
            "access_restrictions": seg.get("access_restrictions", []),
            "width_m": seg.get("width_m"),
            "segment_id": seg_id,
        }

        G.add_edge(u, v, **edge_data)
        if not seg["is_oneway"]:
            G.add_edge(v, u, **{**edge_data, "geometry": LineString(coords[::-1])})

    # Store the segment→connector mapping as a graph-level attribute
    G.graph["segment_to_connectors"] = segment_to_connectors

    n_oneway = sum(1 for _, _, d in G.edges(data=True) if d.get("oneway"))
    n_total  = G.number_of_edges()
    print(f"[overture] street graph: {G.number_of_nodes()} nodes, {n_total} edges, "
          f"{n_oneway} one-way ({100*n_oneway//max(1,n_total)}%)")

    return G


# ---------------------------------------------------------------------------
# Overture: fetch connectors (topological intersection nodes)
# ---------------------------------------------------------------------------

def _fetch_overture_connectors(
    bbox: tuple[float, float, float, float],
    transformer: Transformer,
) -> tuple[dict[str, tuple[float, float]], set[tuple[str, str]]]:
    """
    Fetch connector features from Overture Maps and project to local metric CRS.

    Connectors are the topological endpoints where segments join and carry
    ``prohibited_transitions`` lists that encode turn restrictions.

    Returns
    -------
    connectors : dict[str, (x_m, y_m)]
        connector_id → projected position in local metric CRS.
    prohibited_pairs : set[tuple[str, str]]
        Set of (from_segment_id, to_segment_id) pairs that are forbidden
        at any connector in the bounding box.  Empty if Overture has no data.
    """
    if not _OVERTURE_AVAILABLE:
        return {}, set()

    connectors: dict[str, tuple[float, float]] = {}
    prohibited_pairs: set[tuple[str, str]] = set()

    import shapely.wkb as _shapely_wkb
    try:
        reader = overturemaps.record_batch_reader("connector", bbox=bbox)
        for batch in reader:
            d = batch.to_pydict()
            n = len(d.get("geometry", []))
            ids = d.get("id", [None] * n)

            # prohibited_transitions column: list-of-dicts per connector
            # Schema: [{from_segment_id: str, to_segment_id: str}, ...]
            pt_col = d.get("prohibited_transitions", [None] * n)

            for i in range(n):
                raw_geom = d["geometry"][i]
                if raw_geom is None:
                    continue
                try:
                    geom = _shapely_wkb.loads(bytes(raw_geom))
                except Exception:
                    try:
                        geom = shape(raw_geom)
                    except Exception:
                        continue

                if not isinstance(geom, Point):
                    continue

                x_m, y_m = transformer.transform(geom.x, geom.y)
                connector_id = str(ids[i]) if ids[i] is not None else None
                if connector_id:
                    connectors[connector_id] = (float(x_m), float(y_m))

                # Parse prohibited_transitions for this connector
                pt_entry = pt_col[i] if pt_col and i < len(pt_col) else None
                if pt_entry and isinstance(pt_entry, (list, tuple)):
                    for pt in pt_entry:
                        if not isinstance(pt, dict):
                            continue
                        from_seg = pt.get("from_segment_id") or pt.get("from")
                        to_seg   = pt.get("to_segment_id")   or pt.get("to")
                        if from_seg and to_seg:
                            prohibited_pairs.add((str(from_seg), str(to_seg)))

    except Exception as exc:
        warnings.warn(
            f"[Overture] Connector fetch failed (non-fatal): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    if prohibited_pairs:
        print(f"[Overture] {len(prohibited_pairs)} prohibited turn pair(s) from connectors")

    return connectors, prohibited_pairs


# ---------------------------------------------------------------------------
# Road & sidewalk surface meshes (same logic as osm_3d_buildings.py)
# ---------------------------------------------------------------------------

def _buffer_linestring_mesh(
    line: LineString,
    buffer_m: float,
    extrude_z: float,
) -> Optional[pv.PolyData]:
    """Buffer a LineString and extrude it to a thin flat slab."""
    try:
        poly = line.buffer(buffer_m, cap_style=2, join_style=2)
        if poly.is_empty or not poly.is_valid:
            return None
        return _polygon_to_pyvista(poly, extrude_z)
    except Exception:
        return None


def build_road_and_sidewalk_meshes_from_graph(
    projected_graph: nx.MultiDiGraph,
    road_buffer_m: float = 3.0,
    sidewalk_buffer_m: float = 1.5,
    road_extrude_z: float = 0.10,
    sidewalk_extrude_z: float = 0.15,
) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Build road and sidewalk surface meshes from the street graph.
    Identical signature to osm_3d_buildings.build_road_and_sidewalk_meshes_from_graph.
    """
    road_meshes = []
    sidewalk_meshes = []

    for u, v, data in projected_graph.edges(data=True):
        geom = data.get("geometry")
        hw = data.get("highway", "")

        if geom is None or not isinstance(geom, LineString):
            # Reconstruct from node coords
            nu = projected_graph.nodes.get(u, {})
            nv = projected_graph.nodes.get(v, {})
            if "x" not in nu or "x" not in nv:
                continue
            geom = LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])

        is_ped = hw in _PED_CLASSES
        if is_ped:
            m = _buffer_linestring_mesh(geom, sidewalk_buffer_m, sidewalk_extrude_z)
            if m is not None and m.n_points > 0:
                sidewalk_meshes.append(m)
        else:
            m = _buffer_linestring_mesh(geom, road_buffer_m, road_extrude_z)
            if m is not None and m.n_points > 0:
                road_meshes.append(m)

    def _merge(meshes: list[pv.PolyData]) -> pv.PolyData:
        if not meshes:
            return pv.PolyData()
        if len(meshes) == 1:
            return meshes[0]
        return meshes[0].merge(meshes[1:], merge_points=False)

    return _merge(road_meshes), _merge(sidewalk_meshes)


# ---------------------------------------------------------------------------
# POI (place) fetch & icon helpers
# ---------------------------------------------------------------------------

CATEGORY_ICON: dict[str, str] = {
    # Food & drink
    "restaurant":        "🍽",
    "cafe":              "☕",
    "bar":               "🍺",
    "fast_food":         "🍔",
    "bakery":            "🥐",
    # Retail
    "supermarket":       "🛒",
    "convenience_store": "🏪",
    "pharmacy":          "💊",
    "clothing_store":    "👔",
    # Transport
    "parking":           "🅿",
    "gas_station":       "⛽",
    "bus_stop":          "🚌",
    # Civic / services
    "bank":              "🏦",
    "atm":               "💳",
    "hospital":          "🏥",
    "school":            "🏫",
    "hotel":             "🏨",
    "post_office":       "📮",
    # Recreation
    "park":              "🌳",
    "gym":               "🏋",
    "museum":            "🏛",
    "church":            "⛪",
    # Default
    "__default__":       "📍",
}

CATEGORY_COLOR: dict[str, str] = {
    "restaurant": "#e07b54",
    "cafe":       "#a0785a",
    "bar":        "#f0c040",
    "fast_food":  "#e8a030",
    "supermarket":"#50c878",
    "pharmacy":   "#60b0e0",
    "parking":    "#8888cc",
    "hospital":   "#e04040",
    "park":       "#40b060",
    "bank":       "#c0a030",
    "__default__":"#aaaaaa",
}


def _poi_icon(categories: list[str]) -> str:
    for cat in categories:
        if cat in CATEGORY_ICON:
            return CATEGORY_ICON[cat]
    return CATEGORY_ICON["__default__"]


def _poi_color(categories: list[str]) -> str:
    for cat in categories:
        if cat in CATEGORY_COLOR:
            return CATEGORY_COLOR[cat]
    return CATEGORY_COLOR["__default__"]


def _fetch_overture_places(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """
    Fetch place features from Overture Maps for the given bbox.

    Returns a list of dicts with:
        geometry   : Shapely Point (WGS-84)
        name       : str | None
        categories : list[str]   — primary category first
        confidence : float | None
        brand      : str | None
        address    : str | None
    """
    if not _OVERTURE_AVAILABLE:
        return []

    import shapely.wkb as _wkb

    places: list[dict] = []
    try:
        reader = overturemaps.record_batch_reader("place", bbox=bbox)
    except Exception as exc:
        warnings.warn(f"[Overture] Place fetch failed: {exc}", RuntimeWarning, stacklevel=2)
        return []

    for batch in reader:
        d = batch.to_pydict()
        n = len(d["geometry"])

        for i in range(n):
            raw_geom = d["geometry"][i]
            if raw_geom is None:
                continue
            try:
                geom = _wkb.loads(bytes(raw_geom))
            except Exception:
                try:
                    geom = shape(raw_geom)
                except Exception:
                    continue

            if not isinstance(geom, Point) or geom.is_empty:
                continue

            # ── name ──────────────────────────────────────────────────────
            name = None
            try:
                names_col = d.get("names", [None] * n)[i]
                if isinstance(names_col, dict):
                    primary = names_col.get("primary")
                    if isinstance(primary, str):
                        name = primary
                    else:
                        common = names_col.get("common") or []
                        if common and isinstance(common[0], dict):
                            name = common[0].get("value")
                elif isinstance(names_col, str):
                    name = names_col
            except Exception:
                pass

            # ── categories ────────────────────────────────────────────────
            categories: list[str] = []
            try:
                cats_col = d.get("categories", [None] * n)[i]
                if isinstance(cats_col, dict):
                    primary_cat = cats_col.get("primary")
                    if isinstance(primary_cat, str):
                        categories.append(primary_cat)
                    alt = cats_col.get("alternate") or []
                    categories.extend(str(c) for c in alt if isinstance(c, str))
                elif isinstance(cats_col, (list, tuple)):
                    categories = [str(c) for c in cats_col if c]
            except Exception:
                pass

            # ── confidence ────────────────────────────────────────────────
            confidence = None
            try:
                conf_raw = d.get("confidence", [None] * n)[i]
                if conf_raw is not None:
                    confidence = float(conf_raw)
            except Exception:
                pass

            # ── brand ─────────────────────────────────────────────────────
            brand = None
            try:
                brand_col = d.get("brand", [None] * n)[i]
                if isinstance(brand_col, dict):
                    brand_names = brand_col.get("names") or {}
                    if isinstance(brand_names, dict):
                        brand = brand_names.get("primary") or brand_names.get("common")
                elif isinstance(brand_col, str):
                    brand = brand_col
            except Exception:
                pass

            # ── address ───────────────────────────────────────────────────
            address = None
            try:
                addr_col = d.get("addresses", [None] * n)[i]
                if isinstance(addr_col, (list, tuple)) and addr_col:
                    first = addr_col[0]
                    if isinstance(first, dict):
                        parts = [first.get("freeform"), first.get("locality")]
                        address = ", ".join(p for p in parts if p)
            except Exception:
                pass

            places.append({
                "geometry": geom,
                "name": name,
                "categories": categories,
                "confidence": confidence,
                "brand": brand,
                "address": address,
            })

    return places


def fetch_and_project_places(
    bbox: tuple[float, float, float, float],
    transformer: Transformer,
    min_confidence: float = 0.7,
    clip_radius_m: float | None = None,
) -> list[dict]:
    """
    Fetch Overture places, project to local metric CRS, filter by confidence.

    Returns list of dicts adding:
        x, y : float  (local metric CRS)
    """
    raw = _fetch_overture_places(bbox)
    out: list[dict] = []
    for p in raw:
        if p["confidence"] is not None and p["confidence"] < min_confidence:
            continue
        x, y = transformer.transform(p["geometry"].x, p["geometry"].y)
        if clip_radius_m is not None:
            if x * x + y * y > clip_radius_m ** 2:
                continue
        out.append({**p, "x": float(x), "y": float(y)})
    print(f"[Overture] {len(out)} places after confidence filter (≥{min_confidence})")
    return out


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def build_3d_buildings_and_street_graph(
    address: str,
    radius: float = 150.0,
    extrusion_height: float = 10.0,
) -> tuple[pv.PolyData, nx.MultiDiGraph, list[dict]]:
    """
    Fetch Overture Maps data and build a 3D city model.

    Drop-in replacement for osm_3d_buildings.build_3d_buildings_and_street_graph().
    Returns the same types:
        buildings_mesh : pv.PolyData  (3D extruded buildings in local metric CRS)
        street_graph   : nx.MultiDiGraph (same schema as OSMnx output)

    Parameters
    ----------
    address : str
        Free-text address to geocode (e.g. "Patras, Greece").
    radius : float
        Search radius in metres.
    extrusion_height : float
        Fallback extrusion height when Overture has no height data.
    """
    if not _OVERTURE_AVAILABLE:
        raise RuntimeError(
            "overturemaps is not installed.\n"
            "Run: pip install overturemaps\n"
            "Then retry."
        )

    # 1. Geocode address
    lat, lon = _geocode_address(address)
    print(f"[Overture] Geocoded '{address}' → lat={lat:.5f}, lon={lon:.5f}")

    # 2. Compute bounding box (buildings use exact radius; roads use 20% padding
    # so segments that start outside but enter the area are also captured)
    bbox = _radius_to_bbox(lat, lon, radius)
    bbox_roads = _radius_to_bbox(lat, lon, radius * 1.2)
    west, south, east, north = bbox
    print(f"[Overture] Bbox: W={west:.5f} S={south:.5f} E={east:.5f} N={north:.5f}")

    # 3. Build local metric projection (origin = scene centre)
    to_local, _ = _make_local_transformer(lat, lon)

    # 4. Fetch and build buildings
    print("[Overture] Fetching buildings...")
    buildings = _fetch_overture_buildings(bbox)
    print(f"[Overture] {len(buildings)} building features received")

    buildings_mesh = _build_buildings_mesh(buildings, to_local, extrusion_height)
    print(
        f"[Overture] Buildings mesh: "
        f"{buildings_mesh.n_points} points, {buildings_mesh.n_cells} cells"
    )

    if buildings_mesh.n_points == 0:
        warnings.warn(
            "No buildings found in this area. "
            "Try a larger radius or check the address.",
            RuntimeWarning,
            stacklevel=2,
        )

    # 5. Fetch and build street graph (padded bbox for better road coverage)
    print("[Overture] Fetching road segments...")
    segments = _fetch_overture_segments(bbox_roads)
    print(f"[Overture] {len(segments)} segment features received")
    from collections import Counter
    class_counts = Counter(s.get("class_", "unknown") for s in segments)
    print(f"[Overture] Segment classes: {dict(class_counts)}")

    street_graph = _build_street_graph(segments, to_local, clip_radius_m=radius * 1.3)
    street_graph.graph["scene_lat"] = float(lat)
    street_graph.graph["scene_lon"] = float(lon)
    print(
        f"[Overture] Street graph: "
        f"{street_graph.number_of_nodes()} nodes, "
        f"{street_graph.number_of_edges()} edges"
    )

    # 6. Fetch connectors — position map + turn-restriction pairs
    print("[Overture] Fetching connectors...")
    connectors, prohibited_pairs = _fetch_overture_connectors(bbox_roads, to_local)
    print(f"[Overture] {len(connectors)} connector(s), {len(prohibited_pairs)} prohibited transition(s)")
    street_graph.graph["connectors"] = connectors
    street_graph.graph["prohibited_turn_pairs"] = prohibited_pairs  # used by turn_restrictions.py

    # 7. Fetch places
    print("[Overture] Fetching places...")
    places = fetch_and_project_places(
        bbox, to_local,
        min_confidence=0.75,
        clip_radius_m=radius,
    )

    return buildings_mesh, street_graph, places