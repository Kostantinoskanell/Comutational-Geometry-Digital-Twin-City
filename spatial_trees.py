"""Spatial partitioning trees for triangle geometry using pure Python and NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

EPS = 1e-12


ArrayF = np.ndarray


def _as_triangle_3d(triangle: ArrayF) -> ArrayF:
    tri = np.asarray(triangle, dtype=float)
    if tri.shape != (3, 3):
        raise ValueError("Triangle must have shape (3, 3).")
    return tri


def _as_triangle_2d(triangle: ArrayF) -> ArrayF:
    tri = np.asarray(triangle, dtype=float)
    if tri.shape == (3, 3):
        return tri[:, :2]
    if tri.shape == (3, 2):
        return tri
    raise ValueError("Triangle must have shape (3, 3) or (3, 2).")


def _projected_interval(points: ArrayF, axis: ArrayF) -> tuple[float, float]:
    proj = points @ axis
    return float(np.min(proj)), float(np.max(proj))


def _separated_1d(min_a: float, max_a: float, min_b: float, max_b: float) -> bool:
    return max_a < min_b or max_b < min_a


def _triangle_aabb_intersects_3d(triangle: ArrayF, aabb_min: ArrayF, aabb_max: ArrayF) -> bool:
    """Triangle vs AABB overlap test via SAT in 3D."""
    tri = _as_triangle_3d(triangle)
    aabb_min = np.asarray(aabb_min, dtype=float)
    aabb_max = np.asarray(aabb_max, dtype=float)

    tri_min = np.min(tri, axis=0)
    tri_max = np.max(tri, axis=0)
    if np.any(tri_max < aabb_min) or np.any(tri_min > aabb_max):
        return False

    center = 0.5 * (aabb_min + aabb_max)
    half = 0.5 * (aabb_max - aabb_min)
    v = tri - center

    edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]]

    axes: list[ArrayF] = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    tri_normal = np.cross(edges[0], edges[1])
    axes.append(tri_normal)

    box_axes = axes[:3]
    for edge in edges:
        for box_axis in box_axes:
            axes.append(np.cross(edge, box_axis))

    for axis in axes:
        axis_len = np.linalg.norm(axis)
        if axis_len <= EPS:
            continue
        axis = axis / axis_len

        tri_lo, tri_hi = _projected_interval(v, axis)
        radius = float(np.dot(half, np.abs(axis)))
        box_lo, box_hi = -radius, radius

        if _separated_1d(tri_lo, tri_hi, box_lo, box_hi):
            return False

    return True


def _triangle_aabb_intersects_2d(triangle: ArrayF, aabb_min_xy: ArrayF, aabb_max_xy: ArrayF) -> bool:
    """Triangle vs AABB overlap test in XY only (quadtree variant)."""
    tri = _as_triangle_2d(triangle)
    aabb_min = np.asarray(aabb_min_xy, dtype=float)
    aabb_max = np.asarray(aabb_max_xy, dtype=float)

    tri_min = np.min(tri, axis=0)
    tri_max = np.max(tri, axis=0)
    if np.any(tri_max < aabb_min) or np.any(tri_min > aabb_max):
        return False

    center = 0.5 * (aabb_min + aabb_max)
    half = 0.5 * (aabb_max - aabb_min)
    v = tri - center

    edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]]
    axes: list[ArrayF] = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    for edge in edges:
        perp = np.array([-edge[1], edge[0]])
        axes.append(perp)

    for axis in axes:
        axis_len = np.linalg.norm(axis)
        if axis_len <= EPS:
            continue
        axis = axis / axis_len

        tri_lo, tri_hi = _projected_interval(v, axis)
        radius = float(np.dot(half, np.abs(axis)))
        box_lo, box_hi = -radius, radius

        if _separated_1d(tri_lo, tri_hi, box_lo, box_hi):
            return False

    return True


@dataclass
class OctreeNode:
    """Octree node storing triangles intersecting a 3D AABB."""

    aabb_min: ArrayF
    aabb_max: ArrayF
    depth: int = 0
    max_depth: int = 6
    max_triangles: int = 50
    triangles: List[ArrayF] = field(default_factory=list)
    children: list["OctreeNode"] | None = None

    def __post_init__(self) -> None:
        self.aabb_min = np.asarray(self.aabb_min, dtype=float)
        self.aabb_max = np.asarray(self.aabb_max, dtype=float)
        if self.aabb_min.shape != (3,) or self.aabb_max.shape != (3,):
            raise ValueError("Octree AABB bounds must be 3D vectors.")

    def subdivide(self) -> None:
        """Split this node into 8 equal octants."""
        if self.children is not None or self.depth >= self.max_depth:
            return

        mid = 0.5 * (self.aabb_min + self.aabb_max)
        children: list[OctreeNode] = []

        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    child_min = np.array(
                        [
                            self.aabb_min[0] if ix == 0 else mid[0],
                            self.aabb_min[1] if iy == 0 else mid[1],
                            self.aabb_min[2] if iz == 0 else mid[2],
                        ],
                        dtype=float,
                    )
                    child_max = np.array(
                        [
                            mid[0] if ix == 0 else self.aabb_max[0],
                            mid[1] if iy == 0 else self.aabb_max[1],
                            mid[2] if iz == 0 else self.aabb_max[2],
                        ],
                        dtype=float,
                    )
                    children.append(
                        OctreeNode(
                            aabb_min=child_min,
                            aabb_max=child_max,
                            depth=self.depth + 1,
                            max_depth=self.max_depth,
                            max_triangles=self.max_triangles,
                        )
                    )

        self.children = children

    def insert(self, triangle: ArrayF) -> bool:
        """Insert a triangle into this node or its descendants if intersecting."""
        tri = _as_triangle_3d(triangle)
        if not _triangle_aabb_intersects_3d(tri, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            inserted = False
            for child in self.children:
                if child.insert(tri):
                    inserted = True
            if not inserted:
                self.triangles.append(tri)
            return True

        self.triangles.append(tri)

        if len(self.triangles) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()

        return True

    def _redistribute(self) -> None:
        if self.children is None:
            return

        existing = self.triangles
        self.triangles = []

        for tri in existing:
            inserted = False
            for child in self.children:
                if child.insert(tri):
                    inserted = True
            if not inserted:
                self.triangles.append(tri)

    def insert_many(self, triangles: Iterable[ArrayF]) -> None:
        for tri in triangles:
            self.insert(tri)


@dataclass
class QuadtreeNode:
    """Quadtree node storing triangles by XY overlap, ignoring the Z axis."""

    aabb_min: ArrayF
    aabb_max: ArrayF
    depth: int = 0
    max_depth: int = 6
    max_triangles: int = 50
    triangles: List[ArrayF] = field(default_factory=list)
    children: list["QuadtreeNode"] | None = None

    def __post_init__(self) -> None:
        self.aabb_min = np.asarray(self.aabb_min, dtype=float)
        self.aabb_max = np.asarray(self.aabb_max, dtype=float)
        if self.aabb_min.shape != (2,) or self.aabb_max.shape != (2,):
            raise ValueError("Quadtree AABB bounds must be 2D vectors (x, y).")

    def subdivide(self) -> None:
        """Split this node into 4 equal quadrants in XY."""
        if self.children is not None or self.depth >= self.max_depth:
            return

        mid = 0.5 * (self.aabb_min + self.aabb_max)
        children: list[QuadtreeNode] = []

        for ix in (0, 1):
            for iy in (0, 1):
                child_min = np.array(
                    [
                        self.aabb_min[0] if ix == 0 else mid[0],
                        self.aabb_min[1] if iy == 0 else mid[1],
                    ],
                    dtype=float,
                )
                child_max = np.array(
                    [
                        mid[0] if ix == 0 else self.aabb_max[0],
                        mid[1] if iy == 0 else self.aabb_max[1],
                    ],
                    dtype=float,
                )
                children.append(
                    QuadtreeNode(
                        aabb_min=child_min,
                        aabb_max=child_max,
                        depth=self.depth + 1,
                        max_depth=self.max_depth,
                        max_triangles=self.max_triangles,
                    )
                )

        self.children = children

    def insert(self, triangle: ArrayF) -> bool:
        """Insert a triangle by XY overlap into this node or descendants."""
        tri2 = _as_triangle_2d(triangle)
        if not _triangle_aabb_intersects_2d(tri2, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            inserted = False
            for child in self.children:
                if child.insert(tri2):
                    inserted = True
            if not inserted:
                self.triangles.append(tri2)
            return True

        self.triangles.append(tri2)

        if len(self.triangles) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()

        return True

    def _redistribute(self) -> None:
        if self.children is None:
            return

        existing = self.triangles
        self.triangles = []

        for tri in existing:
            inserted = False
            for child in self.children:
                if child.insert(tri):
                    inserted = True
            if not inserted:
                self.triangles.append(tri)

    def insert_many(self, triangles: Iterable[ArrayF]) -> None:
        for tri in triangles:
            self.insert(tri)
