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


def _street_line_layers(street_graph) -> tuple[pv.PolyData | None, pv.PolyData | None]:
    """Split graph edges into vehicle and pedestrian polyline layers."""
    ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}
    vehicle_segments: list[tuple[np.ndarray, np.ndarray]] = []
    ped_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for u, v, data in street_graph.edges(data=True):
        hw = data.get("highway")
        if isinstance(hw, (list, tuple)):
            hw_vals = {str(x) for x in hw}
        elif hw is None:
            hw_vals = set()
        else:
            hw_vals = {str(hw)}

        is_ped = len(hw_vals & ped_tags) > 0

        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=float)
            if coords.shape[0] >= 2:
                xy = coords[:, :2]
                for i in range(xy.shape[0] - 1):
                    a = np.array([xy[i, 0], xy[i, 1], 0.05], dtype=float)
                    b = np.array([xy[i + 1, 0], xy[i + 1, 1], 0.05], dtype=float)
                    (ped_segments if is_ped else vehicle_segments).append((a, b))
                continue

        nu = street_graph.nodes.get(u, {})
        nv = street_graph.nodes.get(v, {})
        if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
            a = np.array([float(nu["x"]), float(nu["y"]), 0.05], dtype=float)
            b = np.array([float(nv["x"]), float(nv["y"]), 0.05], dtype=float)
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
) -> tuple[pv.PolyData, object, pv.PolyData, pv.PolyData, list]:
    key = _cache_key(
        "osm",
        "v10_parking_bclass",
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

    if use_cache and mesh_path.exists() and graph_path.exists():
        buildings_mesh = pv.read(mesh_path)
        with open(graph_path, "rb") as f:
            street_graph = pickle.load(f)

        if road_path.exists() and sidewalk_path.exists():
            road_mesh = pv.read(road_path)
            sidewalk_mesh = pv.read(sidewalk_path)
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

        # Places are not cached to disk — re-fetch each run (fast network call)
        from overture_source import _geocode_address, _radius_to_bbox, _make_local_transformer, fetch_and_project_places
        _lat, _lon = _geocode_address(address)
        street_graph.graph["scene_lat"] = float(_lat)
        street_graph.graph["scene_lon"] = float(_lon)
        _bbox = _radius_to_bbox(_lat, _lon, radius)
        _to_local, _ = _make_local_transformer(_lat, _lon)
        places = fetch_and_project_places(_bbox, _to_local, min_confidence=0.75, clip_radius_m=radius)
        print(f"Loaded OSM cache: {mesh_path.name}")
        return buildings_mesh, street_graph, road_mesh, sidewalk_mesh, places

    if data_source == "osm":
        import osm_3d_buildings as src_module
        print("[source] Using OpenStreetMap via OSMnx")
    else:
        import overture_source as src_module
        print("[source] Using Overture Maps")

    buildings_mesh, street_graph, places, park_mesh = src_module.build_3d_buildings_and_street_graph(
        address=address,
        radius=radius,
        extrusion_height=extrusion_height,
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_context_key=key,
    )
    road_mesh, sidewalk_mesh = src_module.build_road_and_sidewalk_meshes_from_graph(
        projected_graph=street_graph,
        road_buffer_m=ROAD_BUFFER_M,
        sidewalk_buffer_m=SIDEWALK_BUFFER_M,
        road_extrude_z=ROAD_EXTRUDE_Z,
        sidewalk_extrude_z=SIDEWALK_EXTRUDE_Z,
    )
    if park_mesh is not None:
        sidewalk_mesh = sidewalk_mesh.merge(park_mesh)

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
        with open(graph_path, "wb") as f:
            pickle.dump(street_graph, f)
        print(f"Saved OSM cache: {mesh_path.name}")

    return buildings_mesh, street_graph, road_mesh, sidewalk_mesh, places


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

    if use_cache and ground_path.exists() and octree_path.exists():
        ground_mesh = _normalize_ground_mesh(pv.read(ground_path))
        with open(octree_path, "rb") as f:
            octree_root = pickle.load(f)
        print(f"Loaded spatial cache: {ground_path.name}")
        return ground_mesh, octree_root

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
        with open(octree_path, "wb") as f:
            pickle.dump(octree_root, f)
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
