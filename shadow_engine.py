"""Shadow computation utilities using ray casting with Numba JIT optimization."""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
from typing import Tuple

import networkx as nx
import numpy as np
import pyvista as pv
from numba import njit, prange, set_num_threads
from scipy.spatial import cKDTree
from tqdm import tqdm

EPS = 1e-9

# Global variables for worker-local access to cKDTree
_COV_GRID: np.ndarray | None = None
_COV_CENTROIDS: np.ndarray | None = None
_COV_RADIUS: float = 0.0
_COV_POLE_HEIGHT: float = 0.0
_COV_SEG_STARTS: np.ndarray | None = None
_COV_SEG_ENDS: np.ndarray | None = None
_COV_CONE_COSINE: float = 0.0
_COV_CENTROID_TREE: cKDTree | None = None
_COV_BUILDING_TRIANGLES: np.ndarray | None = None
_COV_BUILDING_AABB_MINS: np.ndarray | None = None
_COV_BUILDING_AABB_MAXS: np.ndarray | None = None
_GROUND_TRI_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_NUMBA_KERNELS_READY = False
import threading as _threading
_NUMBA_WARMUP_LOCK = _threading.Lock()


# ============================================================================
# NUMBA-COMPILED CORE FUNCTIONS (High Performance Path)
# ============================================================================

@njit
def _ray_aabb_intersection_numba(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> bool:
    """Ray-AABB intersection using slab method (Numba-compiled).
    
    All parameters must be float64 arrays for optimal Numba performance.
    """
    tmin = -np.inf
    tmax = np.inf

    for i in range(3):
        if abs(ray_dir[i]) < EPS:
            if ray_origin[i] < aabb_min[i] or ray_origin[i] > aabb_max[i]:
                return False
            continue

        inv_d = 1.0 / ray_dir[i]
        t1 = (aabb_min[i] - ray_origin[i]) * inv_d
        t2 = (aabb_max[i] - ray_origin[i]) * inv_d

        t_near = min(t1, t2)
        t_far = max(t1, t2)

        tmin = max(tmin, t_near)
        tmax = min(tmax, t_far)

        if tmax < tmin:
            return False

    return tmax >= 0.0


@njit
def _ray_triangle_intersection_t_numba(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    t_min: float = 1e-6,
) -> float:
    """Ray-triangle intersection using Möller-Trumbore (Numba-compiled).
    
    Returns intersection distance t, or -1.0 if no intersection.
    All arrays must be float64 for optimal Numba performance.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(ray_dir, edge2)
    det = np.dot(edge1, pvec)

    if abs(det) < EPS:
        return -1.0

    inv_det = 1.0 / det
    tvec = ray_origin - v0

    u = np.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return -1.0

    qvec = np.cross(tvec, edge1)
    v = np.dot(ray_dir, qvec) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return -1.0

    t = np.dot(edge2, qvec) * inv_det
    if t <= t_min:
        return -1.0

    return t


@njit(parallel=True)
def _compute_shadows_numba(
    centroids: np.ndarray,
    areas: np.ndarray,
    ray_dir: np.ndarray,
    building_triangles: np.ndarray,
    building_aabb_mins: np.ndarray,
    building_aabb_maxs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Parallel shadow computation using Numba (compiled to machine code).
    
    Parameters
    ----------
    centroids : np.ndarray
        Ground triangle centroids, shape (K, 3)
    areas : np.ndarray
        Ground triangle areas, shape (K,)
    ray_dir : np.ndarray
        Sun direction (normalized), shape (3,)
    building_triangles : np.ndarray
        Building triangle vertices, shape (M, 3, 3)
    building_aabb_mins : np.ndarray
        Building-triangle AABB mins, shape (M, 3)
    building_aabb_maxs : np.ndarray
        Building-triangle AABB maxs, shape (M, 3)
    
    Returns
    -------
    tuple[np.ndarray, float]
        Shadowed mask (K,) and lit area ratio
    """
    n_tris = centroids.shape[0]
    n_building = building_triangles.shape[0]
    shadowed = np.zeros(n_tris, dtype=np.bool_)

    # Parallel loop over ground triangles
    for i in prange(n_tris):
        centroid = centroids[i]
        is_shadowed = False

        # Check each building triangle for occlusion
        for j in range(n_building):
            if not _ray_aabb_intersection_numba(
                centroid,
                ray_dir,
                building_aabb_mins[j],
                building_aabb_maxs[j],
            ):
                continue

            tri = building_triangles[j]
            v0, v1, v2 = tri[0], tri[1], tri[2]
            
            t_hit = _ray_triangle_intersection_t_numba(centroid, ray_dir, v0, v1, v2, t_min=1e-5)
            if t_hit > 0.0:
                is_shadowed = True
                break

        shadowed[i] = is_shadowed

    total_area = np.sum(areas)
    if total_area <= EPS:
        return shadowed, 0.0

    lit_area = np.sum(areas[~shadowed])
    ratio = lit_area / total_area

    return shadowed, ratio


@njit(parallel=True)
def _compute_spotlight_coverage_numba(
    light_pos: np.ndarray,
    centroids: np.ndarray,
    orientation: np.ndarray,
    cone_cosine: float,
    building_triangles: np.ndarray,
    building_aabb_mins: np.ndarray,
    building_aabb_maxs: np.ndarray,
    candidate_indices: np.ndarray,
    illuminated: np.ndarray,
) -> None:
    """Parallel spotlight coverage computation (Numba-compiled).
    
    All heavy lifting via Numba to avoid Python GIL.
    """
    n_building = building_triangles.shape[0]
    
    for idx in prange(candidate_indices.shape[0]):
        i = candidate_indices[idx]
        target = centroids[i]
        ray_vec = target - light_pos
        ray_len = np.sqrt(np.sum(ray_vec ** 2))

        if ray_len <= EPS:
            illuminated[i] = True
            continue

        ray_dir = ray_vec / ray_len

        # Omnidirectional: no cone restriction (removed spotlight_half_angle check)
        blocked = False
        for j in range(n_building):
            if not _ray_aabb_intersection_numba(
                light_pos,
                ray_dir,
                building_aabb_mins[j],
                building_aabb_maxs[j],
            ):
                continue

            tri = building_triangles[j]
            v0, v1, v2 = tri[0], tri[1], tri[2]
            
            t_hit = _ray_triangle_intersection_t_numba(light_pos, ray_dir, v0, v1, v2, t_min=1e-6)
            if t_hit > 0.0 and t_hit < (ray_len - 1e-5):
                blocked = True
                break

        illuminated[i] = not blocked


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _warmup_numba_kernels() -> None:
    global _NUMBA_KERNELS_READY
    if _NUMBA_KERNELS_READY:
        return
    with _NUMBA_WARMUP_LOCK:
        if _NUMBA_KERNELS_READY:  # double-checked locking
            return

        centroids = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        areas = np.array([1.0], dtype=np.float64)
        ray_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        building_triangles = np.array([[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]]], dtype=np.float64)
        tri_mins = np.min(building_triangles, axis=1)
        tri_maxs = np.max(building_triangles, axis=1)

        _compute_shadows_numba(
            centroids,
            areas,
            ray_dir,
            building_triangles,
            tri_mins,
            tri_maxs,
        )

        illuminated = np.zeros((1,), dtype=np.bool_)
        _compute_spotlight_coverage_numba(
            np.array([0.0, 0.0, 3.0], dtype=np.float64),
            centroids,
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            -1.0,
            building_triangles,
            tri_mins,
            tri_maxs,
            np.array([0], dtype=np.int64),
            illuminated,
        )
        _NUMBA_KERNELS_READY = True


def _auto_chunk_size(n_rows: int, n_jobs: int) -> int:
    """Choose a chunk size based on workload and worker count."""
    if n_rows <= 0:
        return 1
    if n_jobs <= 1:
        return max(1, n_rows)

    if n_rows <= n_jobs * 8:
        # Keep tiny workloads highly divisible to saturate workers.
        return 1

    target_tasks_per_worker = 8
    tasks = max(1, n_jobs * target_tasks_per_worker)
    chunk = int(np.ceil(n_rows / tasks))
    return int(max(4, min(512, chunk)))


def _extract_ground_triangles_and_areas(ground_mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Return triangle vertices and areas from a mesh, triangulating only when needed."""
    if ground_mesh.n_cells == 0:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    face_arr = np.asarray(ground_mesh.faces)
    tri_mesh = ground_mesh
    if face_arr.size != ground_mesh.n_cells * 4:
        tri_mesh = ground_mesh.triangulate()
        face_arr = np.asarray(tri_mesh.faces)

    faces = face_arr.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        tri_mesh = ground_mesh.triangulate()
        faces = np.asarray(tri_mesh.faces).reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise ValueError("Ground mesh triangulation failed: non-triangle faces present.")

    pts = np.asarray(tri_mesh.points, dtype=np.float64)
    triangles = pts[faces[:, 1:4]].astype(np.float64)

    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))

    return triangles, areas


def _extract_ground_triangles_areas_centroids_cached(
    ground_mesh: pv.PolyData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (id(ground_mesh), int(ground_mesh.n_points), int(ground_mesh.n_cells))
    cached = _GROUND_TRI_CACHE.get(key)
    if cached is not None:
        return cached

    triangles, areas = _extract_ground_triangles_and_areas(ground_mesh)
    centroids = np.mean(triangles, axis=1).astype(np.float64) if triangles.shape[0] > 0 else np.empty((0, 3), dtype=np.float64)
    out = (triangles, areas, centroids)
    _GROUND_TRI_CACHE[key] = out
    return out


def _extract_building_triangles(octree_root) -> np.ndarray:
    """Flatten all building triangles from octree into a single array."""
    cached = getattr(octree_root, "_triangle_array_cache", None)
    if isinstance(cached, np.ndarray):
        return np.asarray(cached, dtype=np.float64)

    triangles_list = []
    
    def _collect_triangles(node):
        if node.triangles is not None:
            for tri in node.triangles:
                triangles_list.append(np.asarray(tri, dtype=np.float64))
        if node.children is not None:
            for child in node.children:
                _collect_triangles(child)
    
    _collect_triangles(octree_root)
    
    if not triangles_list:
        out = np.empty((0, 3, 3), dtype=np.float64)
    else:
        out = np.ascontiguousarray(np.array(triangles_list, dtype=np.float64))

    try:
        setattr(octree_root, "_triangle_array_cache", out)
        setattr(octree_root, "_triangle_aabb_mins_cache", np.min(out, axis=1) if out.shape[0] > 0 else np.empty((0, 3), dtype=np.float64))
        setattr(octree_root, "_triangle_aabb_maxs_cache", np.max(out, axis=1) if out.shape[0] > 0 else np.empty((0, 3), dtype=np.float64))
    except Exception:
        pass
    return out


def _extract_building_triangle_aabbs(octree_root, building_triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri_mins = getattr(octree_root, "_triangle_aabb_mins_cache", None)
    tri_maxs = getattr(octree_root, "_triangle_aabb_maxs_cache", None)
    if isinstance(tri_mins, np.ndarray) and isinstance(tri_maxs, np.ndarray):
        return np.asarray(tri_mins, dtype=np.float64), np.asarray(tri_maxs, dtype=np.float64)

    if building_triangles.shape[0] == 0:
        mins = np.empty((0, 3), dtype=np.float64)
        maxs = np.empty((0, 3), dtype=np.float64)
    else:
        mins = np.min(building_triangles, axis=1).astype(np.float64, copy=False)
        maxs = np.max(building_triangles, axis=1).astype(np.float64, copy=False)

    try:
        setattr(octree_root, "_triangle_aabb_mins_cache", mins)
        setattr(octree_root, "_triangle_aabb_maxs_cache", maxs)
    except Exception:
        pass
    return mins, maxs


def _extract_street_segments_xy(street_graph: nx.MultiDiGraph | None) -> tuple[np.ndarray, np.ndarray]:
    """Extract XY street segments from graph edge geometry or endpoint nodes."""
    if street_graph is None:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []

    for u, v, data in street_graph.edges(data=True):
        geom = data.get("geometry")

        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=np.float64)
            if coords.shape[0] < 2:
                continue
            xy = coords[:, :2]
            for i in range(xy.shape[0] - 1):
                starts.append((float(xy[i, 0]), float(xy[i, 1])))
                ends.append((float(xy[i + 1, 0]), float(xy[i + 1, 1])))
            continue

        nu = street_graph.nodes.get(u, {})
        nv = street_graph.nodes.get(v, {})
        if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
            starts.append((float(nu["x"]), float(nu["y"])))
            ends.append((float(nv["x"]), float(nv["y"])))

    if not starts:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64)


def _nearest_point_on_segments(
    point_xy: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray | None:
    """Return nearest XY point on a set of 2D segments, or None if no segments."""
    if seg_starts.shape[0] == 0:
        return None

    p = np.asarray(point_xy, dtype=np.float64)
    if p.shape != (2,):
        raise ValueError("point_xy must have shape (2,).")

    v = seg_ends - seg_starts
    w = p - seg_starts

    vv = np.einsum("ij,ij->i", v, v)
    t = np.zeros_like(vv)

    valid = vv > EPS
    t[valid] = np.einsum("ij,ij->i", w[valid], v[valid]) / vv[valid]
    t = np.clip(t, 0.0, 1.0)

    proj = seg_starts + t[:, None] * v
    if np.any(~valid):
        proj[~valid] = seg_starts[~valid]

    delta = p - proj
    d2 = np.einsum("ij,ij->i", delta, delta)
    idx = int(np.argmin(d2))
    return np.asarray(proj[idx], dtype=np.float64)


def _spotlight_orientation_vector(
    light_xy: np.ndarray,
    pole_height: float,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray:
    """Compute unit orientation vector from light top to nearest street point."""
    nearest_xy = _nearest_point_on_segments(light_xy, seg_starts, seg_ends)
    if nearest_xy is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    light_pos = np.array([float(light_xy[0]), float(light_xy[1]), float(pole_height)], dtype=np.float64)
    target = np.array([float(nearest_xy[0]), float(nearest_xy[1]), 0.0], dtype=np.float64)
    vec = target - light_pos
    nrm = float(np.linalg.norm(vec))
    if nrm <= EPS:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return vec / nrm


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def compute_shadows(
    ground_mesh: pv.PolyData,
    octree_root,
    sun_dir: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute shadow mask for ground triangles (Numba-accelerated).
    
    Parameters
    ----------
    ground_mesh : pv.PolyData
        PyVista ground mesh. Will be triangulated internally.
    octree_root : OctreeNode
        Root node of octree storing building triangles in 3D.
    sun_dir : np.ndarray
        Ray direction for shadow testing, shape (3,).
    
    Returns
    -------
    tuple[np.ndarray, float]
        Boolean shadow mask per ground triangle and lit_area / total_area ratio.
    """
    _warmup_numba_kernels()

    triangles, areas, centroids = _extract_ground_triangles_areas_centroids_cached(ground_mesh)
    n_tris = triangles.shape[0]

    if n_tris == 0:
        return np.zeros((0,), dtype=bool), 0.0

    direction = np.asarray(sun_dir, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise ValueError("sun_dir must be a non-zero vector.")
    direction = direction / norm

    building_triangles = _extract_building_triangles(octree_root)
    building_aabb_mins, building_aabb_maxs = _extract_building_triangle_aabbs(octree_root, building_triangles)
    
    print(f"Computing shadows for {n_tris} ground triangles with {building_triangles.shape[0]} building triangles...")
    _t0 = time.perf_counter()
    # Call Numba-compiled function
    shadowed, ratio = _compute_shadows_numba(
        centroids.astype(np.float64),
        areas.astype(np.float64),
        direction.astype(np.float64),
        building_triangles,
        building_aabb_mins,
        building_aabb_maxs,
    )

    print(f"[shadow] compute_shadows: {time.perf_counter() - _t0:.3f}s for {n_tris} cells")
    return shadowed, ratio


def compute_spotlight_coverage(
    light_xy: np.ndarray,
    radius: float,
    pole_height: float,
    ground_mesh: pv.PolyData,
    octree_root,
    street_graph: nx.MultiDiGraph | None = None,
) -> np.ndarray:
    """Compute ground-triangle illumination for a streetlight (Numba-accelerated).
    
    The light is modeled as a point source at (light_xy[0], light_xy[1], pole_height).
    A ground triangle is illuminated when:
    1) its centroid is within the light radius in XY, and
    2) no building triangle blocks LOS from light to centroid.
    
    Parameters
    ----------
    light_xy : np.ndarray
        Light XY coordinate, shape (2,).
    radius : float
        Light radius in XY plane.
    pole_height : float
        Z coordinate of the point light.
    ground_mesh : pv.PolyData
        Ground mesh to evaluate.
    octree_root : OctreeNode
        Octree storing building triangles.
    street_graph : nx.MultiDiGraph, optional
        Street network (not used for omnidirectional spotlight).
    
    Returns
    -------
    np.ndarray
        Boolean array per ground triangle: True means illuminated.
    """
    _warmup_numba_kernels()

    if radius <= 0.0:
        raise ValueError("radius must be > 0.")

    light_xy_arr = np.asarray(light_xy, dtype=np.float64)
    if light_xy_arr.shape != (2,):
        raise ValueError("light_xy must have shape (2,).")

    triangles, _, centroids = _extract_ground_triangles_areas_centroids_cached(ground_mesh)
    n_tris = triangles.shape[0]
    if n_tris == 0:
        return np.zeros((0,), dtype=bool)
    centroid_xy = centroids[:, :2]

    # Vectorized distance check using cKDTree
    tree = cKDTree(centroid_xy)
    within_radius_indices = tree.query_ball_point(light_xy_arr, r=float(radius))
    within_radius_mask = np.zeros(n_tris, dtype=bool)
    within_radius_mask[within_radius_indices] = True

    illuminated = np.zeros(n_tris, dtype=bool)

    # Extract building triangles for Numba function
    building_triangles = _extract_building_triangles(octree_root)
    building_aabb_mins, building_aabb_maxs = _extract_building_triangle_aabbs(octree_root, building_triangles)
    
    # Get candidate indices (those within radius)
    candidate_indices = np.where(within_radius_mask)[0].astype(np.int64)

    if candidate_indices.shape[0] > 0 and building_triangles.shape[0] > 0:
        light_pos = np.array([light_xy_arr[0], light_xy_arr[1], float(pole_height)], dtype=np.float64)
        
        # Omnidirectional: pass dummy orientation and cone_cosine (no cone check in Numba function)
        dummy_orientation = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dummy_cone_cosine = -1.0  # Always passes cone check since -1.0 < any dot product
        
        # Call Numba-compiled parallel function
        _compute_spotlight_coverage_numba(
            light_pos,
            centroids,
            dummy_orientation,
            dummy_cone_cosine,
            building_triangles,
            building_aabb_mins,
            building_aabb_maxs,
            candidate_indices,
            illuminated,
        )

    return illuminated


def _init_coverage_worker(
    grid: np.ndarray,
    centroids: np.ndarray,
    radius: float,
    pole_height: float,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    centroids_xy: np.ndarray,
    building_triangles: np.ndarray,
    building_aabb_mins: np.ndarray,
    building_aabb_maxs: np.ndarray,
) -> None:
    """Initializer for process workers computing coverage rows."""
    global _COV_GRID, _COV_CENTROIDS, _COV_RADIUS, _COV_POLE_HEIGHT
    global _COV_SEG_STARTS, _COV_SEG_ENDS, _COV_CONE_COSINE, _COV_CENTROID_TREE
    global _COV_BUILDING_TRIANGLES, _COV_BUILDING_AABB_MINS, _COV_BUILDING_AABB_MAXS
    try:
        # Process-level parallelism is used here; keep one Numba thread per worker
        # to avoid severe CPU oversubscription.
        set_num_threads(1)
    except Exception:
        pass

    _COV_GRID = grid
    _COV_CENTROIDS = centroids
    _COV_RADIUS = float(radius)
    _COV_POLE_HEIGHT = float(pole_height)
    _COV_SEG_STARTS = seg_starts
    _COV_SEG_ENDS = seg_ends
    _COV_CONE_COSINE = -1.0  # Omnidirectional: dummy value (always passes cone check)
    _COV_CENTROID_TREE = cKDTree(np.asarray(centroids_xy, dtype=np.float64))
    _COV_BUILDING_TRIANGLES = building_triangles
    _COV_BUILDING_AABB_MINS = building_aabb_mins
    _COV_BUILDING_AABB_MAXS = building_aabb_maxs


def _coverage_worker_chunk(row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a chunk of coverage rows in a worker process (Numba acceleration)."""
    if _COV_GRID is None or _COV_CENTROIDS is None:
        raise RuntimeError("Coverage worker was not initialized.")
    if _COV_BUILDING_TRIANGLES is None or _COV_BUILDING_TRIANGLES.shape[0] == 0:
        # Empty building data - all False
        idx = np.asarray(row_indices, dtype=np.int64)
        sub = np.zeros((idx.shape[0], _COV_CENTROIDS.shape[0]), dtype=bool)
        return idx, sub

    idx = np.asarray(row_indices, dtype=np.int64)
    sub = np.zeros((idx.shape[0], _COV_CENTROIDS.shape[0]), dtype=bool)

    for j, ridx in enumerate(idx):
        gp = _COV_GRID[int(ridx)].astype(np.float64)
        cand = np.asarray(_COV_CENTROID_TREE.query_ball_point(gp, r=_COV_RADIUS), dtype=np.int64)
        if cand.size == 0:
            continue

        light_pos = np.array([gp[0], gp[1], _COV_POLE_HEIGHT], dtype=np.float64)
        dummy_orientation = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Omnidirectional

        # Call Numba-compiled parallel function per grid point
        _compute_spotlight_coverage_numba(
            light_pos,
            _COV_CENTROIDS,
            dummy_orientation,
            _COV_CONE_COSINE,
            _COV_BUILDING_TRIANGLES,
            _COV_BUILDING_AABB_MINS,
            _COV_BUILDING_AABB_MAXS,
            cand,
            sub[j],
        )

    return idx, sub


def build_coverage_matrix(
    grid_points: np.ndarray,
    ground_mesh: pv.PolyData,
    octree_root,
    radius: float,
    pole_height: float,
    street_graph: nx.MultiDiGraph | None = None,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Precompute spotlight LOS coverage matrix (Numba-accelerated with progress bar).
    
    Parameters
    ----------
    grid_points : np.ndarray
        Candidate XY light coordinates, shape (M, 2).
    ground_mesh : pv.PolyData
        Triangulated ground mesh with K triangles.
    octree_root : OctreeNode
        Octree with building triangles.
    radius : float
        Spotlight influence radius in XY.
    pole_height : float
        Z height of each point light.
    n_jobs : int, optional
        Number of worker processes. Default uses all CPU cores.
    chunk_size : int, optional
        Grid points per chunk. If omitted, chosen adaptively.
    
    Returns
    -------
    np.ndarray
        Boolean matrix shape (M, K). Row i = triangles illuminated by grid_points[i].
    """
    _warmup_numba_kernels()

    grid = np.asarray(grid_points, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("grid_points must have shape (M, 2).")

    triangles, _, centroids = _extract_ground_triangles_areas_centroids_cached(ground_mesh)
    k = int(triangles.shape[0])
    m = int(grid.shape[0])
    matrix = np.zeros((m, k), dtype=bool)

    if m == 0 or k == 0:
        return matrix

    print(f"Building coverage matrix: {m} grid points × {k} ground triangles")

    centroids = np.asarray(centroids, dtype=np.float64)
    centroids_xy = np.asarray(centroids[:, :2], dtype=np.float64)
    seg_starts, seg_ends = _extract_street_segments_xy(street_graph)
    building_triangles = _extract_building_triangles(octree_root)
    building_aabb_mins, building_aabb_maxs = _extract_building_triangle_aabbs(octree_root, building_triangles)

    print(f"Extracted {building_triangles.shape[0]} building triangles for LOS checks")

    def _run_serial(into: np.ndarray) -> np.ndarray:
        tree = cKDTree(centroids_xy)
        for i in tqdm(range(m), desc="Coverage", unit="gridpoint"):
            gp = grid[i]
            cand = np.asarray(tree.query_ball_point(gp, r=radius), dtype=np.int64)
            if cand.size == 0:
                continue

            light_pos = np.array([gp[0], gp[1], pole_height], dtype=np.float64)
            dummy_orientation = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Omnidirectional

            _compute_spotlight_coverage_numba(
                light_pos,
                centroids,
                dummy_orientation,
                -1.0,  # Omnidirectional: dummy cone_cosine
                building_triangles,
                building_aabb_mins,
                building_aabb_maxs,
                cand,
                into[i],
            )
        return into

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))
    if n_jobs > 1:
        # Keep only very small workloads on the serial path.
        # Larger "small-M" workloads (e.g., ~40 lights) can still benefit from
        # process-level row parallelism.
        min_rows_for_parallel = max(16, n_jobs * 2)
        if m < min_rows_for_parallel:
            n_jobs = 1
    if chunk_size is None:
        chunk_size = _auto_chunk_size(m, n_jobs)
    else:
        chunk_size = max(1, int(chunk_size))

    if n_jobs == 1:
        print("Running single-worker (serial) coverage computation...")
        return _run_serial(matrix)

    # Parallel path with progress bar
    print(f"Running parallel coverage computation with {n_jobs} workers...")
    row_ids = np.arange(m, dtype=np.int64)
    chunks = [row_ids[i : i + chunk_size] for i in range(0, m, chunk_size)]

    try:
        with cf.ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_coverage_worker,
            initargs=(
                grid,
                centroids,
                float(radius),
                float(pole_height),
                seg_starts,
                seg_ends,
                centroids_xy,
                building_triangles,
                building_aabb_mins,
                building_aabb_maxs,
            ),
        ) as ex:
            futures = [ex.submit(_coverage_worker_chunk, chunk) for chunk in chunks]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Coverage chunks"):
                idx_chunk, sub = future.result()
                matrix[idx_chunk] = sub
    except Exception as exc:
        print(f"Parallel coverage failed ({exc}); falling back to serial computation...")
        return _run_serial(matrix)

    return matrix
