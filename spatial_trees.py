"""Spatial partitioning trees for triangle geometry using pure Python and NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

EPS = 1e-12


ArrayF = np.ndarray


def _aabb_overlaps(a_min: ArrayF, a_max: ArrayF, b_min: ArrayF, b_max: ArrayF) -> bool:
    return bool(np.all(a_max >= b_min - EPS) and np.all(a_min <= b_max + EPS))


def _aabb_contains(outer_min: ArrayF, outer_max: ArrayF, inner_min: ArrayF, inner_max: ArrayF) -> bool:
    return bool(np.all(inner_min >= outer_min - EPS) and np.all(inner_max <= outer_max + EPS))


def _point_in_aabb_3d(point: ArrayF, aabb_min: ArrayF, aabb_max: ArrayF) -> bool:
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        raise ValueError("Point must have shape (3,).")
    return bool(np.all(p >= aabb_min - EPS) and np.all(p <= aabb_max + EPS))


def _point_in_aabb_2d(point: ArrayF, aabb_min: ArrayF, aabb_max: ArrayF) -> bool:
    p = np.asarray(point, dtype=float)
    if p.shape != (2,):
        raise ValueError("Point must have shape (2,).")
    return bool(np.all(p >= aabb_min - EPS) and np.all(p <= aabb_max + EPS))


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


def _triangle_bbox_3d(triangle: ArrayF) -> tuple[ArrayF, ArrayF]:
    tri = _as_triangle_3d(triangle)
    return np.min(tri, axis=0), np.max(tri, axis=0)


def _triangle_bbox_2d(triangle: ArrayF) -> tuple[ArrayF, ArrayF]:
    tri2 = _as_triangle_2d(triangle)
    return np.min(tri2, axis=0), np.max(tri2, axis=0)


@dataclass
class OctreeNode:
    """Octree node storing triangles intersecting a 3D AABB."""

    aabb_min: ArrayF
    aabb_max: ArrayF
    depth: int = 0
    max_depth: int = 6
    max_triangles: int = 50
    triangles: List[ArrayF] = field(default_factory=list)
    points: List[ArrayF] = field(default_factory=list)
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
        """Insert triangle by AABB overlap and push to one fully-containing child when possible."""
        tri = _as_triangle_3d(triangle)
        tri_min, tri_max = _triangle_bbox_3d(tri)
        if not _aabb_overlaps(tri_min, tri_max, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            if not self._insert_into_single_containing_child(tri, tri_min, tri_max):
                self.triangles.append(tri)
            return True

        self.triangles.append(tri)

        if len(self.triangles) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()

        return True

    def _insert_into_single_containing_child(self, tri: ArrayF, tri_min: ArrayF, tri_max: ArrayF) -> bool:
        if self.children is None:
            return False

        containing: list[OctreeNode] = []
        for child in self.children:
            if _aabb_contains(child.aabb_min, child.aabb_max, tri_min, tri_max):
                containing.append(child)
                if len(containing) > 1:
                    return False

        if len(containing) == 1:
            containing[0].insert(tri)
            return True

        return False

    def _redistribute(self) -> None:
        if self.children is None:
            return

        existing = self.triangles
        self.triangles = []

        for tri in existing:
            tri_min, tri_max = _triangle_bbox_3d(tri)
            if not self._insert_into_single_containing_child(tri, tri_min, tri_max):
                self.triangles.append(tri)

    def _redistribute_points(self) -> None:
        if self.children is None:
            return

        existing = self.points
        self.points = []

        for point in existing:
            inserted = False
            for child in self.children:
                if child.insert_point(point):
                    inserted = True
                    break
            if not inserted:
                self.points.append(point)

    def insert_point(self, point: ArrayF) -> bool:
        """Insert a 3D point (e.g., triangle centroid/vertex) into the octree."""
        p = np.asarray(point, dtype=float)
        if p.shape != (3,):
            raise ValueError("Point must have shape (3,).")

        if not _point_in_aabb_3d(p, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            for child in self.children:
                if child.insert_point(p):
                    return True
            self.points.append(p)
            return True

        self.points.append(p)

        if len(self.points) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()
            self._redistribute_points()

        return True

    def get_bounding_boxes(self) -> list[tuple[float, float, float, float, float, float]]:
        """Return all node AABBs recursively as (min_x, max_x, min_y, max_y, min_z, max_z)."""
        out = [
            (
                float(self.aabb_min[0]),
                float(self.aabb_max[0]),
                float(self.aabb_min[1]),
                float(self.aabb_max[1]),
                float(self.aabb_min[2]),
                float(self.aabb_max[2]),
            )
        ]
        if self.children is not None:
            for child in self.children:
                out.extend(child.get_bounding_boxes())
        return out

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
    points: List[ArrayF] = field(default_factory=list)
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
        """Insert triangle by XY AABB overlap and push to one containing child when possible."""
        tri2 = _as_triangle_2d(triangle)
        tri_min, tri_max = _triangle_bbox_2d(tri2)
        if not _aabb_overlaps(tri_min, tri_max, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            if not self._insert_into_single_containing_child(tri2, tri_min, tri_max):
                self.triangles.append(tri2)
            return True

        self.triangles.append(tri2)

        if len(self.triangles) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()

        return True

    def _insert_into_single_containing_child(self, tri2: ArrayF, tri_min: ArrayF, tri_max: ArrayF) -> bool:
        if self.children is None:
            return False

        containing: list[QuadtreeNode] = []
        for child in self.children:
            if _aabb_contains(child.aabb_min, child.aabb_max, tri_min, tri_max):
                containing.append(child)
                if len(containing) > 1:
                    return False

        if len(containing) == 1:
            containing[0].insert(tri2)
            return True

        return False

    def _redistribute(self) -> None:
        if self.children is None:
            return

        existing = self.triangles
        self.triangles = []

        for tri in existing:
            tri_min, tri_max = _triangle_bbox_2d(tri)
            if not self._insert_into_single_containing_child(tri, tri_min, tri_max):
                self.triangles.append(tri)

    def _redistribute_points(self) -> None:
        if self.children is None:
            return

        existing = self.points
        self.points = []

        for point in existing:
            inserted = False
            for child in self.children:
                if child.insert_point(point):
                    inserted = True
                    break
            if not inserted:
                self.points.append(point)

    def insert_point(self, point: ArrayF) -> bool:
        """Insert a 2D point (x, y) or 3D point (x, y, z) into the quadtree."""
        p = np.asarray(point, dtype=float)
        if p.shape == (3,):
            p2 = p[:2]
        elif p.shape == (2,):
            p2 = p
        else:
            raise ValueError("Point must have shape (2,) or (3,).")

        if not _point_in_aabb_2d(p2, self.aabb_min, self.aabb_max):
            return False

        if self.children is not None:
            for child in self.children:
                if child.insert_point(p2):
                    return True
            self.points.append(p2)
            return True

        self.points.append(p2)

        if len(self.points) > self.max_triangles and self.depth < self.max_depth:
            self.subdivide()
            self._redistribute()
            self._redistribute_points()

        return True

    def get_bounding_boxes(self) -> list[tuple[float, float, float, float, float, float]]:
        """Return all node bounds with a flattened Z range for wireframe rendering."""
        out = [
            (
                float(self.aabb_min[0]),
                float(self.aabb_max[0]),
                float(self.aabb_min[1]),
                float(self.aabb_max[1]),
                0.0,
                0.0,
            )
        ]
        if self.children is not None:
            for child in self.children:
                out.extend(child.get_bounding_boxes())
        return out

    def insert_many(self, triangles: Iterable[ArrayF]) -> None:
        for tri in triangles:
            self.insert(tri)
