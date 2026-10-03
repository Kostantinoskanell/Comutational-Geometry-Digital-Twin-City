from __future__ import annotations

import hashlib
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import warnings
try:
    warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)
except AttributeError:
    warnings.filterwarnings("ignore", message=".*extract_surface.*")

from shadow_engine import build_coverage_matrix
from spatial_trees import OctreeNode
from streetlight_ga import build_candidate_grid_points
from overture_source import build_road_and_sidewalk_meshes_from_graph


# ── Robust cache I/O ───────────────────────────────────────────────────────
_UNPICKLABLE_GRAPH_KEYS = ("terrain_sampler", "terrain_sampler_base")


def _atomic_pickle(obj, path: Path) -> None:
    """Write-then-rename so a failed or interrupted dump can never leave a
    truncated cache file behind (which crashed the next startup)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _save_graph_cache(street_graph, path: Path) -> None:
    """Pickle a street graph without its (closure) terrain samplers."""
    held = {k: street_graph.graph.pop(k) for k in _UNPICKLABLE_GRAPH_KEYS if k in street_graph.graph}
    try:
        _atomic_pickle(street_graph, path)
    finally:
        street_graph.graph.update(held)


def _load_pickle_or_none(path: Path, label: str):
    """Load a cache pickle; a corrupt/partial file counts as a cache miss."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[cache] {label} cache {Path(path).name} unreadable ({type(exc).__name__}) — rebuilding")
        try:
            Path(path).unlink()
        except Exception:
            pass
        return None

ROAD_BUFFER_M = 3.0
SIDEWALK_BUFFER_M = 1.5
ROAD_EXTRUDE_Z = 0.1
SIDEWALK_EXTRUDE_Z = 0.15


def _apply_atmosphere(plotter, hour: float, is_night: bool, preset: str) -> None:
    """Apply renderer atmospheric effects (fog + AA settings) for current time-of-day."""
    try:
        plotter.renderer.SetUseDepthPeeling(False)
    except Exception:
        pass

    try:
        # Touch active camera to ensure renderer internals are initialized.
        _ = plotter.renderer.GetActiveCamera()
    except Exception:
        pass

    try:
        vtk_renderer = plotter.renderer.GetVTKObject()
    except Exception:
        vtk_renderer = plotter.renderer

    try:
        vtk_renderer.SetUseFXAA(True)
    except Exception:
        pass

    h = float(hour)
    _preset = str(preset).strip().lower()  # reserved for future preset-specific tuning
    _ = _preset

    if bool(is_night):
        fog_color = (0.03, 0.04, 0.08)
        fog_start, fog_end = 40.0, 200.0
    elif (6.0 <= h <= 8.0) or (16.0 <= h <= 18.0):
        fog_color = (0.9, 0.75, 0.5)
        fog_start, fog_end = 80.0, 350.0
    elif 10.0 <= h <= 14.0:
        fog_color = (0.85, 0.9, 1.0)
        fog_start, fog_end = 150.0, 500.0
    else:
        # Neutral transition band outside explicit windows.
        fog_color = (0.82, 0.86, 0.92)
        fog_start, fog_end = 120.0, 420.0

    try:
        vtk_renderer.SetFog(True)
        vtk_renderer.SetFogColor(*fog_color)
        vtk_renderer.SetFogStart(float(fog_start))
        vtk_renderer.SetFogEnd(float(fog_end))
    except Exception:
        pass


def _segments_to_polydata(segments: list[tuple[np.ndarray, np.ndarray]]) -> pv.PolyData | None:
    if not segments:
        return None

    pts: list[np.ndarray] = []
    lines: list[int] = []
    for a, b in segments:
        i = len(pts)
        pts.append(np.asarray(a, dtype=float))
        pts.append(np.asarray(b, dtype=float))
        lines.extend([2, i, i + 1])

    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.asarray(lines, dtype=np.int32)
    return poly


def _edge_bridge_height(data) -> float:
    """Elevation (m) for a bridge/overpass edge, 0.0 for ground-level edges.

    Uses the OSM `bridge` tag (any value except "no") and the `layer` tag
    (stacked crossings: layer=2 rides above layer=1).
    """
    bridge = data.get("bridge")
    if isinstance(bridge, (list, tuple)):
        bridge = bridge[0] if bridge else None
    is_bridge = bridge not in (None, False, "no", "")
    if not is_bridge:
        return 0.0
    try:
        layer = data.get("layer")
        if isinstance(layer, (list, tuple)):
            layer = layer[0] if layer else 1
        layer_n = max(1, int(float(layer)))
    except (TypeError, ValueError):
        layer_n = 1
    return 5.0 * layer_n


def _bridge_z_offsets(xy: np.ndarray, height: float, ramp_len: float = 18.0) -> np.ndarray:
    """Per-vertex z offsets giving a ramped bridge profile along a polyline.

    Rises linearly from 0 at each end to `height` over `ramp_len` metres, so
    vehicles/lines climb onto the deck instead of teleporting up.  Bridges
    shorter than 2×ramp_len peak at their midpoint proportionally.
    """
    seg = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-6 or height <= 0.0:
        return np.zeros(xy.shape[0], dtype=float)
    ramp = min(float(ramp_len), total / 2.0)
    from_start = cum
    from_end = total - cum
    return height * np.clip(np.minimum(from_start, from_end) / max(ramp, 1e-6), 0.0, 1.0)


def _street_line_layers(street_graph) -> tuple[pv.PolyData | None, pv.PolyData | None]:
    """Split graph edges into vehicle and pedestrian polyline layers.

    Bridge/overpass edges (OSM `bridge` tag) are elevated with a ramped
    profile so they render above the crossing road instead of overlapping it.
    """
    ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}
    vehicle_segments: list[tuple[np.ndarray, np.ndarray]] = []
    ped_segments: list[tuple[np.ndarray, np.ndarray]] = []
    _BASE_Z = 0.35   # above road (0.1) and sidewalk (0.15) surfaces — always visible

    for u, v, data in street_graph.edges(data=True):
        hw = data.get("highway")
        if isinstance(hw, (list, tuple)):
            hw_vals = {str(x) for x in hw}
        elif hw is None:
            hw_vals = set()
        else:
            hw_vals = {str(hw)}

        is_ped = len(hw_vals & ped_tags) > 0
        bridge_h = _edge_bridge_height(data)

        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=float)
            if coords.shape[0] >= 2:
                xy = coords[:, :2]
                bz = _bridge_z_offsets(xy, bridge_h)
                for i in range(xy.shape[0] - 1):
                    a = np.array([xy[i, 0], xy[i, 1], _BASE_Z + bz[i]], dtype=float)
                    b = np.array([xy[i + 1, 0], xy[i + 1, 1], _BASE_Z + bz[i + 1]], dtype=float)
                    (ped_segments if is_ped else vehicle_segments).append((a, b))
                continue

        nu = street_graph.nodes.get(u, {})
        nv = street_graph.nodes.get(v, {})
        if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
            _xy2 = np.array([[float(nu["x"]), float(nu["y"])],
                             [float(nv["x"]), float(nv["y"])]], dtype=float)
            bz = _bridge_z_offsets(_xy2, bridge_h)
            a = np.array([_xy2[0, 0], _xy2[0, 1], _BASE_Z + bz[0]], dtype=float)
            b = np.array([_xy2[1, 0], _xy2[1, 1], _BASE_Z + bz[1]], dtype=float)
            (ped_segments if is_ped else vehicle_segments).append((a, b))

    return _segments_to_polydata(vehicle_segments), _segments_to_polydata(ped_segments)


def _street_direction_arrows(street_graph, max_arrows: int = 180) -> pv.PolyData | None:
    """Build small arrow glyphs that show the direction of directed street edges."""
    arrow_z = 0.45
    ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}

    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for u, v, data in street_graph.edges(data=True):
        hw = data.get("highway")
        if isinstance(hw, (list, tuple)):
            hw_vals = {str(x) for x in hw}
        elif hw is None:
            hw_vals = set()
        else:
            hw_vals = {str(hw)}

        if len(hw_vals & ped_tags) > 0:
            continue

        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=float)
            if coords.shape[0] >= 2:
                xy = coords[:, :2].copy()
                nu2 = street_graph.nodes.get(u, {})
                nv2 = street_graph.nodes.get(v, {})
                if "x" in nu2 and "y" in nu2 and "x" in nv2 and "y" in nv2:
                    u_xy = np.array([float(nu2["x"]), float(nu2["y"])], dtype=float)
                    v_xy = np.array([float(nv2["x"]), float(nv2["y"])], dtype=float)
                    fwd_err = float(
                        np.linalg.norm(xy[0] - u_xy) + np.linalg.norm(xy[-1] - v_xy)
                    )
                    rev_err = float(
                        np.linalg.norm(xy[0] - v_xy) + np.linalg.norm(xy[-1] - u_xy)
                    )
                    if rev_err + 1e-6 < fwd_err:
                        xy = xy[::-1]
                seg_vec2 = np.asarray(xy[-1] - xy[0], dtype=float)
                seg_len2 = float(np.linalg.norm(seg_vec2))
                if seg_len2 < 1e-6:
                    continue
                seg_dir = np.array([seg_vec2[0] / seg_len2, seg_vec2[1] / seg_len2, 0.0], dtype=float)
                for i in range(xy.shape[0] - 1):
                    a = np.array([xy[i, 0], xy[i, 1], arrow_z], dtype=float)
                    b = np.array([xy[i + 1, 0], xy[i + 1, 1], arrow_z], dtype=float)
                    segments.append((a, b, seg_dir))
            continue

        nu = street_graph.nodes.get(u, {})
        nv = street_graph.nodes.get(v, {})
        if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
            a = np.array([float(nu["x"]), float(nu["y"]), arrow_z], dtype=float)
            b = np.array([float(nv["x"]), float(nv["y"]), arrow_z], dtype=float)
            seg_vec = np.asarray([b[0] - a[0], b[1] - a[1]], dtype=float)
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-6:
                continue
            seg_dir = np.array([seg_vec[0] / seg_len, seg_vec[1] / seg_len, 0.0], dtype=float)
            segments.append((a, b, seg_dir))

    if not segments:
        return None

    step = max(1, math.ceil(len(segments) / max(1, int(max_arrows))))
    points: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    scales: list[float] = []
    for idx, (a, b, seg_dir) in enumerate(segments):
        if idx % step != 0:
            continue

        vec = np.asarray(b - a, dtype=float)
        length = float(np.linalg.norm(vec[:2]))
        if length < 1e-6:
            continue

        midpoint = np.asarray((a + b) * 0.5, dtype=float)
        midpoint[2] = arrow_z + 0.07
        points.append(midpoint)
        vectors.append(seg_dir)
        scales.append(min(max(length * 0.22, 1.6), 8.0))

    if not points:
        return None

    poly = pv.PolyData(np.asarray(points, dtype=float))
    poly["direction"] = np.asarray(vectors, dtype=float)
    poly["scale"] = np.asarray(scales, dtype=float)
    return poly.glyph(
        orient="direction",
        scale="scale",
        factor=1.25,
        geom=pv.Arrow(tip_length=0.42, tip_radius=0.13, shaft_radius=0.05),
    )


def _street_node_markers(street_graph, min_degree: int = 3) -> np.ndarray | None:
    """Return a sparse set of graph nodes to render as square markers."""
    coords: list[list[float]] = []
    undirected = street_graph.to_undirected(as_view=True)
    for node, degree in undirected.degree():
        if degree < int(min_degree):
            continue

        data = street_graph.nodes.get(node, {})
        if "x" in data and "y" in data:
            coords.append([float(data["x"]), float(data["y"]), 0.14])

    if not coords:
        return None
    return np.asarray(coords, dtype=float)


def _mesh_triangles(mesh: pv.PolyData) -> np.ndarray:
    tri = mesh.triangulate()
    if tri.n_cells == 0:
        return np.empty((0, 3, 3), dtype=float)

    faces = tri.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangle-only faces after triangulation.")

    pts = np.asarray(tri.points, dtype=float)
    return pts[faces[:, 1:4]]


def _build_octree_from_buildings(buildings_mesh: pv.PolyData) -> OctreeNode:
    triangles = _mesh_triangles(buildings_mesh)
    if triangles.shape[0] == 0:
        raise ValueError("Building mesh has no triangles to insert in octree.")

    tri_min = np.min(triangles.reshape(-1, 3), axis=0)
    tri_max = np.max(triangles.reshape(-1, 3), axis=0)

    # Expand bounds slightly to avoid numeric edge cases at box boundaries.
    pad = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    root = OctreeNode(aabb_min=tri_min - pad, aabb_max=tri_max + pad)
    # Fast path: current LOS kernels consume a flattened triangle array and do not
    # traverse octree children, so avoid expensive recursive insertion here.
    tri_arr = np.ascontiguousarray(triangles, dtype=np.float64)
    root.triangles = [tri for tri in tri_arr]
    root.children = None
    setattr(root, "_triangle_array_cache", tri_arr)
    return root


def _build_ground_mesh_for_tests(buildings_mesh: pv.PolyData, resolution: int = 80) -> pv.PolyData:
    if buildings_mesh.n_points == 0:
        raise ValueError("Cannot build ground test mesh without building points.")

    xmin, xmax, ymin, ymax, _, _ = buildings_mesh.bounds
    dx = xmax - xmin
    dy = ymax - ymin
    margin = max(5.0, 0.15 * max(dx, dy, 1.0))

    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 0.0)
    res = max(8, int(resolution))
    plane = pv.Plane(
        center=center,
        i_size=(xmax - xmin) + 2.0 * margin,
        j_size=(ymax - ymin) + 2.0 * margin,
        i_resolution=res,
        j_resolution=res,
    )
    return plane.triangulate()


def _build_grid_points_from_ground_mesh(
    ground_mesh: pv.PolyData,
    grid_step: float,
    sidewalk_polygon=None,
) -> np.ndarray:
    if sidewalk_polygon is not None and not sidewalk_polygon.is_empty:
        min_x, min_y, max_x, max_y = sidewalk_polygon.bounds
        return build_candidate_grid_points(
            bounds_min_xy=np.array([min_x, min_y], dtype=float),
            bounds_max_xy=np.array([max_x, max_y], dtype=float),
            grid_step=float(grid_step),
            sidewalk_polygon=sidewalk_polygon,
        )

    pts = np.asarray(ground_mesh.points, dtype=float)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    x0, y0 = np.min(pts[:, :2], axis=0)
    x1, y1 = np.max(pts[:, :2], axis=0)
    return build_candidate_grid_points(
        bounds_min_xy=np.array([x0, y0], dtype=float),
        bounds_max_xy=np.array([x1, y1], dtype=float),
        grid_step=float(grid_step),
        sidewalk_polygon=sidewalk_polygon,
    )


def _combine_ground_surfaces(
    road_mesh: pv.PolyData | None,
    sidewalk_mesh: pv.PolyData | None,
) -> pv.PolyData | None:
    parts: list[pv.PolyData] = []

    if road_mesh is not None and road_mesh.n_cells > 0:
        road = road_mesh.copy()
        road.cell_data["surface_kind"] = np.zeros((road.n_cells,), dtype=np.uint8)
        parts.append(road)

    if sidewalk_mesh is not None and sidewalk_mesh.n_cells > 0:
        sidewalk = sidewalk_mesh.copy()
        sidewalk.cell_data["surface_kind"] = np.ones((sidewalk.n_cells,), dtype=np.uint8)
        parts.append(sidewalk)

    if not parts:
        return None

    merged = parts[0]
    for mesh in parts[1:]:
        merged = merged.merge(mesh)
    tri = merged.triangulate()
    kinds = tri.cell_data.get("surface_kind")
    if kinds is None or np.asarray(kinds).reshape(-1).shape[0] != tri.n_cells:
        tri.cell_data["surface_kind"] = np.zeros((tri.n_cells,), dtype=np.uint8)
    else:
        tri.cell_data["surface_kind"] = np.asarray(kinds, dtype=np.uint8).reshape(-1)
    return tri


def _normalize_ground_mesh(mesh: pv.PolyData) -> pv.PolyData:
    tri = mesh.triangulate()
    if not isinstance(tri, pv.PolyData):
        tri = tri.extract_surface()
    kinds = tri.cell_data.get("surface_kind")
    if kinds is None or np.asarray(kinds).reshape(-1).shape[0] != tri.n_cells:
        tri.cell_data["surface_kind"] = np.zeros((tri.n_cells,), dtype=np.uint8)
    else:
        tri.cell_data["surface_kind"] = np.asarray(kinds, dtype=np.uint8).reshape(-1)
    return tri


def _cache_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _load_or_fetch_osm_cached(
    address: str,
    radius: float,
    extrusion_height: float,
    cache_dir: Path,
    use_cache: bool,
    data_source: str = "overture",
    use_dem: bool = True,
    use_ms_buildings: bool = True,
) -> tuple[pv.PolyData, object, pv.PolyData, pv.PolyData, list]:
    key = _cache_key(
        "osm",
        "v12_flat_buildings",   # v12: buildings no longer baked at DEM height
        data_source,
        address,
        radius,
        extrusion_height,
        ROAD_BUFFER_M,
        SIDEWALK_BUFFER_M,
        ROAD_EXTRUDE_Z,
        SIDEWALK_EXTRUDE_Z,
    )
    mesh_path = cache_dir / f"osm_{key}_buildings.vtp"
    graph_path = cache_dir / f"osm_{key}_graph.pkl"
    road_path = cache_dir / f"osm_{key}_road.vtp"
    sidewalk_path = cache_dir / f"osm_{key}_sidewalk.vtp"

    water_path = cache_dir / f"osm_{key}_water.vtp"

    _cached_graph = (_load_pickle_or_none(graph_path, "street-graph")
                     if use_cache and mesh_path.exists() and graph_path.exists() else None)
    if _cached_graph is not None:
        buildings_mesh = pv.read(mesh_path)
        street_graph = _cached_graph

        if road_path.exists() and sidewalk_path.exists():
            road_mesh = pv.read(road_path)
            sidewalk_mesh = pv.read(sidewalk_path)
            water_mesh = pv.read(water_path) if water_path.exists() else None
        else:
            road_mesh, sidewalk_mesh = build_road_and_sidewalk_meshes_from_graph(
                projected_graph=street_graph,
                road_buffer_m=ROAD_BUFFER_M,
                sidewalk_buffer_m=SIDEWALK_BUFFER_M,
                road_extrude_z=ROAD_EXTRUDE_Z,
                sidewalk_extrude_z=SIDEWALK_EXTRUDE_Z,
            )
            if use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if not isinstance(road_mesh, pv.PolyData):
                    road_mesh = road_mesh.extract_surface()
                if not isinstance(sidewalk_mesh, pv.PolyData):
                    sidewalk_mesh = sidewalk_mesh.extract_surface()
                road_mesh.save(road_path)
                sidewalk_mesh.save(sidewalk_path)

        from overture_source import _geocode_address, _radius_to_bbox, _make_local_transformer, fetch_and_project_places
        _lat, _lon = _geocode_address(address)   # disk-cached geocode
        street_graph.graph["scene_lat"] = float(_lat)
        street_graph.graph["scene_lon"] = float(_lon)

        # Places: cached to disk — the Overture POI fetch is hundreds of rows
        # over the network and identical for the same scene key.
        places_path = cache_dir / f"osm_{key}_places.pkl"
        places = None
        if use_cache and places_path.exists():
            try:
                with open(places_path, "rb") as _pf:
                    places = pickle.load(_pf)
                print(f"[places] loaded {len(places)} POIs from cache")
            except Exception:
                places = None
        if places is None:
            _bbox = _radius_to_bbox(_lat, _lon, radius)
            _to_local, _ = _make_local_transformer(_lat, _lon)
            places = fetch_and_project_places(_bbox, _to_local, min_confidence=0.75, clip_radius_m=radius)
            if use_cache:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    _atomic_pickle(places, places_path)
                except Exception as _pe:
                    print(f"[places] cache save failed: {_pe}")

        if use_dem:
            from osm_3d_buildings import _fetch_dem_sampler
            _crs = street_graph.graph.get("crs")
            if _crs is not None:
                street_graph.graph["terrain_sampler"] = _fetch_dem_sampler(float(_lat), float(_lon), float(radius), str(_crs))

        # Fetch land-use fill mesh if not already in cached graph, or palette changed.
        _FILL_VER = 3   # bump when _COLOUR palette changes to force re-fetch
        if "fill_pts" not in street_graph.graph or street_graph.graph.get("fill_ver", 0) < _FILL_VER:
            try:
                from osm_3d_buildings import _fetch_land_fill_mesh as _fill_fn
                _fl_crs = str(street_graph.graph.get("crs", ""))
                if _fl_crs:
                    _fill = _fill_fn((float(_lat), float(_lon)), float(radius), _fl_crs)
                    if _fill is not None and _fill.n_cells > 0:
                        street_graph.graph["fill_pts"]   = np.asarray(_fill.points,          dtype=float)
                        street_graph.graph["fill_faces"]  = np.asarray(_fill.faces,           dtype=np.int64)
                        street_graph.graph["fill_rgb"]    = np.asarray(_fill.cell_data["RGB"], dtype=np.uint8)
                        street_graph.graph["fill_ver"]   = _FILL_VER
                        _save_graph_cache(street_graph, graph_path)
                        print("[fill] updated graph cache with land-use fill data (palette v2)")
            except Exception as _fe:
                print(f"[fill] land-use fetch skipped (cached graph): {_fe}")

        print(f"Loaded OSM cache: {mesh_path.name}")
        return buildings_mesh, street_graph, road_mesh, sidewalk_mesh, places, water_mesh

    if data_source == "osm":
        import osm_3d_buildings as src_module
        print("[source] Using OpenStreetMap via OSMnx")
    else:
        import overture_source as src_module
        print("[source] Using Overture Maps")

    buildings_mesh, street_graph, places, park_mesh, water_mesh = src_module.build_3d_buildings_and_street_graph(
        address=address,
        radius=radius,
        extrusion_height=extrusion_height,
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_context_key=key,
        use_dem=use_dem,
        use_ms_buildings=use_ms_buildings,
    )
    road_mesh, sidewalk_mesh = src_module.build_road_and_sidewalk_meshes_from_graph(
        projected_graph=street_graph,
        road_buffer_m=ROAD_BUFFER_M,
        sidewalk_buffer_m=SIDEWALK_BUFFER_M,
        road_extrude_z=ROAD_EXTRUDE_Z,
        sidewalk_extrude_z=SIDEWALK_EXTRUDE_Z,
    )
    # Park mesh intentionally not merged into sidewalk — fill mesh handles park
    # coloring as flat solid polygons; the extruded park_mesh creates triangulated
    # walls visible as crosshatch from above.

    # Build and store land-use fill mesh in the graph so it's pickled with cache
    _FILL_VER = 3
    if "fill_pts" not in street_graph.graph or street_graph.graph.get("fill_ver", 0) < _FILL_VER:
        try:
            from osm_3d_buildings import _fetch_land_fill_mesh as _fill_fn
            _fl_lat = float(street_graph.graph.get("scene_lat", 0.0))
            _fl_lon = float(street_graph.graph.get("scene_lon", 0.0))
            _fl_crs = str(street_graph.graph.get("crs", ""))
            if _fl_crs:
                _fill = _fill_fn((_fl_lat, _fl_lon), float(radius), _fl_crs)
                if _fill is not None and _fill.n_cells > 0:
                    street_graph.graph["fill_pts"]   = np.asarray(_fill.points,          dtype=float)
                    street_graph.graph["fill_faces"]  = np.asarray(_fill.faces,           dtype=np.int64)
                    street_graph.graph["fill_rgb"]    = np.asarray(_fill.cell_data["RGB"], dtype=np.uint8)
                    street_graph.graph["fill_ver"]   = _FILL_VER
        except Exception as _fe:
            print(f"[fill] land-use fetch skipped: {_fe}")

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(buildings_mesh, pv.PolyData):
            buildings_mesh = buildings_mesh.extract_surface()
        if not isinstance(road_mesh, pv.PolyData):
            road_mesh = road_mesh.extract_surface()
        if not isinstance(sidewalk_mesh, pv.PolyData):
            sidewalk_mesh = sidewalk_mesh.extract_surface()
        buildings_mesh.save(mesh_path)
        road_mesh.save(road_path)
        sidewalk_mesh.save(sidewalk_path)
        
        _save_graph_cache(street_graph, graph_path)
            
        if water_mesh is not None:
            if not isinstance(water_mesh, pv.PolyData):
                water_mesh = water_mesh.extract_surface()
            water_mesh.save(water_path)

        # Places → disk so cached runs skip the Overture POI fetch entirely
        try:
            _atomic_pickle(places, cache_dir / f"osm_{key}_places.pkl")
        except Exception as _pe:
            print(f"[places] cache save failed: {_pe}")
        print(f"Saved OSM cache: {mesh_path.name}")

    return buildings_mesh, street_graph, road_mesh, sidewalk_mesh, places, water_mesh


def _load_or_build_coverage_matrix_cached(
    cache_dir: Path,
    use_cache: bool,
    cache_context_key: str,
    grid_points: np.ndarray,
    ground_mesh: pv.PolyData,
    octree_root: OctreeNode,
    street_graph,
    radius: float,
    pole_height: float,
    n_jobs: int | None,
) -> np.ndarray:
    grid = np.ascontiguousarray(grid_points, dtype=np.float64)
    grid_digest = hashlib.sha1(grid.tobytes()).hexdigest()[:20] if grid.size else "empty"
    key = _cache_key(
        "coverage",
        cache_context_key,
        radius,
        pole_height,
        grid.shape,
        grid_digest,
        ground_mesh.bounds,
        ground_mesh.n_cells,
        ground_mesh.n_points,
    )
    cov_path = cache_dir / f"cov_{key}.npy"

    if use_cache and cov_path.exists():
        cov = np.load(cov_path, allow_pickle=False)
        print(f"Loaded coverage cache: {cov_path.name}")
        return cov.astype(bool, copy=False)

    t0 = time.perf_counter()
    cov = build_coverage_matrix(
        grid_points=grid_points,
        ground_mesh=ground_mesh,
        octree_root=octree_root,
        radius=radius,
        pole_height=pole_height,
        street_graph=street_graph,
        n_jobs=n_jobs,
    )
    dt = time.perf_counter() - t0
    print(f"Coverage matrix built in {dt:.2f}s (shape={cov.shape})")

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cov_path, cov, allow_pickle=False)
        print(f"Saved coverage cache: {cov_path.name}")

    return cov


def _load_or_build_spatial_cache(
    cache_dir: Path,
    use_cache: bool,
    cache_context_key: str,
    buildings_mesh: pv.PolyData,
    ground_resolution: int,
    preferred_ground_mesh: pv.PolyData | None = None,
) -> tuple[pv.PolyData, OctreeNode]:
    """Load or build cached ground mesh + octree used in shadow/GA workflows."""
    ground_path = cache_dir / f"spatial_{cache_context_key}_ground.vtp"
    octree_path = cache_dir / f"spatial_{cache_context_key}_octree.pkl"

    _oct = (_load_pickle_or_none(octree_path, "octree")
            if use_cache and ground_path.exists() and octree_path.exists() else None)
    if _oct is not None:
        ground_mesh = _normalize_ground_mesh(pv.read(ground_path))
        print(f"Loaded spatial cache: {ground_path.name}")
        return ground_mesh, _oct

    if preferred_ground_mesh is not None and preferred_ground_mesh.n_cells > 0:
        ground_mesh = _normalize_ground_mesh(preferred_ground_mesh.copy())
    else:
        ground_mesh = _normalize_ground_mesh(
            _build_ground_mesh_for_tests(buildings_mesh, resolution=ground_resolution)
        )
    octree_root = _build_octree_from_buildings(buildings_mesh)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(ground_mesh, pv.PolyData):
            ground_mesh = ground_mesh.extract_surface()
        ground_mesh.save(ground_path)
        _atomic_pickle(octree_root, octree_path)
        print(f"Saved spatial cache: {ground_path.name}")

    return ground_mesh, octree_root


def _sun_dir_from_hour(hour: float) -> np.ndarray:
    """Map local hour [0, 24] to a sun direction vector."""
    hour = float(np.clip(hour, 0.0, 24.0))

    # Elevation is positive between roughly 06:00 and 18:00.
    elevation = math.sin(math.pi * (hour - 6.0) / 12.0)
    if elevation <= 0.0:
        return np.array([0.0, 0.0, -1.0], dtype=float)

    azimuth = 2.0 * math.pi * (hour - 6.0) / 24.0
    xy_mag = math.sqrt(max(0.0, 1.0 - elevation * elevation))

    return np.array(
        [
            math.cos(azimuth) * xy_mag,
            math.sin(azimuth) * xy_mag,
            elevation,
        ],
        dtype=float,
    )


def _load_hdri(hour: float, preset: str) -> pv.Texture:
    """Load a time-of-day HDRI texture with local-asset preference and safe fallback."""
    h = float(hour)
    p = str(preset).strip().lower()
    hdri_dir = Path(__file__).parent / "assets" / "hdri"

    is_night = h < 6.0 or h >= 18.0
    is_golden = (6.0 <= h <= 8.0) or (16.0 <= h <= 18.0) or ("sunset" in p and 14.0 <= h <= 19.0)

    if is_night:
        name = "night.hdr"
    elif is_golden:
        name = "golden_hour.hdr"
    else:
        name = "day.hdr"

    local_path = hdri_dir / name
    if local_path.exists():
        try:
            local_data = pv.read(local_path)
            if isinstance(local_data, pv.Texture):
                return local_data
            return pv.Texture(local_data)
        except Exception as _exc:
            print(f"[hdri] failed to load local HDRI {local_path.name}: {_exc}")

    # Fallback: disabled online download to prevent network hangs on main Cocoa thread.
    raise FileNotFoundError("Dynamic HDRI assets not found, offline fallback to gradient backgrounds.")


def _build_spotlight_discs(positions_xy: np.ndarray, radius: float, z: float = 0.08) -> pv.PolyData | None:
    if positions_xy.size == 0:
        return None

    discs = [
        pv.Disc(center=(float(xy[0]), float(xy[1]), z), inner=0.0, outer=float(radius), c_res=48)
        for xy in positions_xy
    ]
    return pv.MultiBlock(discs).combine(merge_points=False)


def _initial_light_positions(ground_mesh: pv.PolyData, n_lights: int, seed: int) -> np.ndarray:
    pts = np.asarray(ground_mesh.points, dtype=float)
    xy = np.unique(pts[:, :2], axis=0)
    if xy.shape[0] == 0:
        raise ValueError("Ground mesh has no points for initial light placement.")

    rng = np.random.default_rng(seed)
    count = max(1, int(n_lights))
    if xy.shape[0] >= count:
        idx = rng.choice(xy.shape[0], size=count, replace=False)
    else:
        idx = rng.choice(xy.shape[0], size=count, replace=True)

    return np.asarray(xy[idx], dtype=float)
