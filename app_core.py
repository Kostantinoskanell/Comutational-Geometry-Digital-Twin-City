from __future__ import annotations

import hashlib
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from osm_3d_buildings import build_3d_buildings_and_street_graph
from shadow_engine import build_coverage_matrix
from spatial_trees import OctreeNode
from streetlight_ga import build_candidate_grid_points


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
    ped_tags = {"footway", "pedestrian", "path", "steps", "bridleway"}
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


def _cache_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _load_or_fetch_osm_cached(
    address: str,
    radius: float,
    extrusion_height: float,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[pv.PolyData, object]:
    key = _cache_key("osm", "v3_refined_roofs", address, radius, extrusion_height)
    mesh_path = cache_dir / f"osm_{key}_buildings.vtp"
    graph_path = cache_dir / f"osm_{key}_graph.pkl"

    if use_cache and mesh_path.exists() and graph_path.exists():
        buildings_mesh = pv.read(mesh_path)
        with open(graph_path, "rb") as f:
            street_graph = pickle.load(f)
        print(f"Loaded OSM cache: {mesh_path.name}")
        return buildings_mesh, street_graph

    buildings_mesh, street_graph = build_3d_buildings_and_street_graph(
        address=address,
        radius=radius,
        extrusion_height=extrusion_height,
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        buildings_mesh.save(mesh_path)
        with open(graph_path, "wb") as f:
            pickle.dump(street_graph, f)
        print(f"Saved OSM cache: {mesh_path.name}")

    return buildings_mesh, street_graph


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
) -> tuple[pv.PolyData, OctreeNode]:
    """Load or build cached ground mesh + octree used in shadow/GA workflows."""
    ground_path = cache_dir / f"spatial_{cache_context_key}_ground.vtp"
    octree_path = cache_dir / f"spatial_{cache_context_key}_octree.pkl"

    if use_cache and ground_path.exists() and octree_path.exists():
        ground_mesh = pv.read(ground_path)
        with open(octree_path, "rb") as f:
            octree_root = pickle.load(f)
        print(f"Loaded spatial cache: {ground_path.name}")
        return ground_mesh, octree_root

    ground_mesh = _build_ground_mesh_for_tests(buildings_mesh, resolution=ground_resolution)
    octree_root = _build_octree_from_buildings(buildings_mesh)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
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
