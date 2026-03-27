"""Shadow computation utilities using ray casting and octree acceleration."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pyvista as pv

EPS = 1e-9


def ray_aabb_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> bool:
    """Return True if a ray intersects an axis-aligned bounding box (slab method)."""
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_dir, dtype=float)
    bmin = np.asarray(aabb_min, dtype=float)
    bmax = np.asarray(aabb_max, dtype=float)

    tmin = -np.inf
    tmax = np.inf

    for i in range(3):
        if abs(direction[i]) < EPS:
            # Ray is parallel to this slab; it must be inside the slab to intersect.
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return False
            continue

        inv_d = 1.0 / direction[i]
        t1 = (bmin[i] - origin[i]) * inv_d
        t2 = (bmax[i] - origin[i]) * inv_d

        t_near = min(t1, t2)
        t_far = max(t1, t2)

        tmin = max(tmin, t_near)
        tmax = min(tmax, t_far)

        if tmax < tmin:
            return False

    # If max intersection is behind origin, no forward hit for ray casting.
    return tmax >= 0.0


def _ray_triangle_intersection_t(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    triangle: np.ndarray,
    t_min: float = 1e-6,
) -> float | None:
    """Return intersection distance t using Moller-Trumbore, or None if no hit."""
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_dir, dtype=float)
    tri = np.asarray(triangle, dtype=float)

    if tri.shape != (3, 3):
        raise ValueError("Triangle must have shape (3, 3).")

    v0, v1, v2 = tri
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))

    if abs(det) < EPS:
        return None

    inv_det = 1.0 / det
    tvec = origin - v0

    u = float(np.dot(tvec, pvec) * inv_det)
    if u < 0.0 or u > 1.0:
        return None

    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec) * inv_det)
    if v < 0.0 or (u + v) > 1.0:
        return None

    t = float(np.dot(edge2, qvec) * inv_det)
    if t <= t_min:
        return None

    return t


def ray_triangle_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    """Return True if a ray intersects a triangle (Moller-Trumbore)."""
    return _ray_triangle_intersection_t(ray_origin, ray_dir, triangle) is not None


def _extract_ground_triangles_and_areas(ground_mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate the mesh and return triangle vertices and corresponding areas."""
    tri_mesh = ground_mesh.triangulate()

    if tri_mesh.n_cells == 0:
        return np.empty((0, 3, 3), dtype=float), np.empty((0,), dtype=float)

    faces = tri_mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Ground mesh triangulation failed: non-triangle faces present.")

    pts = np.asarray(tri_mesh.points, dtype=float)
    triangles = pts[faces[:, 1:4]]

    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    return triangles, areas


def _iter_octree_candidate_triangles(octree_root, ray_origin: np.ndarray, ray_dir: np.ndarray):
    """Yield triangles from octree nodes whose AABBs are intersected by the ray."""
    stack = [octree_root]

    while stack:
        node = stack.pop()
        if not ray_aabb_intersection(ray_origin, ray_dir, node.aabb_min, node.aabb_max):
            continue

        for tri in node.triangles:
            yield tri

        if node.children is not None:
            stack.extend(node.children)


def compute_shadows(
    ground_mesh: pv.PolyData,
    octree_root,
    sun_dir: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute shadow mask for ground triangles and return lit/total area ratio.

    Parameters
    ----------
    ground_mesh:
        PyVista ground mesh. It will be triangulated internally.
    octree_root:
        Root node of an octree storing building triangles in 3D.
    sun_dir:
        Ray direction used for shadow testing.

    Returns
    -------
    tuple[np.ndarray, float]
        Boolean shadow mask per ground triangle and lit_area / total_area ratio.
    """
    triangles, areas = _extract_ground_triangles_and_areas(ground_mesh)
    n_tris = triangles.shape[0]

    if n_tris == 0:
        return np.zeros((0,), dtype=bool), 0.0

    direction = np.asarray(sun_dir, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        raise ValueError("sun_dir must be a non-zero vector.")
    direction = direction / norm

    centroids = np.mean(triangles, axis=1)
    shadowed = np.zeros((n_tris,), dtype=bool)

    for i, centroid in enumerate(centroids):
        is_shadowed = False

        for tri in _iter_octree_candidate_triangles(octree_root, centroid, direction):
            if _ray_triangle_intersection_t(centroid, direction, tri, t_min=1e-5) is not None:
                is_shadowed = True
                break

        shadowed[i] = is_shadowed

    total_area = float(np.sum(areas))
    if total_area <= EPS:
        return shadowed, 0.0

    lit_area = float(np.sum(areas[~shadowed]))
    ratio = lit_area / total_area

    return shadowed, ratio
