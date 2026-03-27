"""Genetic optimization of streetlight placement over a triangulated ground mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv

from spatial_trees import QuadtreeNode

EPS = 1e-9


@dataclass
class GAConfig:
    n_lights: int
    light_radius: float
    w1_dark: float
    w2_double_lit: float
    grid_step: float
    population_size: int = 64
    generations: int = 80
    mutation_rate: float = 0.12
    elitism: int = 1
    tournament_k: int = 3
    max_depth: int = 6
    max_triangles: int = 50
    seed: int | None = None


class StreetlightGA:
    """Genetic optimizer for streetlight coordinates on a 2D grid."""

    def __init__(self, ground_mesh: pv.PolyData, config: GAConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        (
            self.triangles_xy,
            self.centroids_xy,
            self.areas,
            self.bounds_min,
            self.bounds_max,
        ) = self._extract_ground_triangles(ground_mesh)

        self.total_area = float(np.sum(self.areas))
        self.quadtree_root = self._build_quadtree()
        self.triangle_key_to_indices = self._build_triangle_lookup()

        self.grid_points = self._build_grid_points()
        if self.grid_points.shape[0] == 0:
            raise ValueError("No valid grid points were generated from bounds/grid_step.")

        if self.config.n_lights <= 0:
            raise ValueError("n_lights must be > 0.")

        if self.config.light_radius <= 0:
            raise ValueError("light_radius must be > 0.")

    @staticmethod
    def _extract_ground_triangles(
        ground_mesh: pv.PolyData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tri_mesh = ground_mesh.triangulate()
        if tri_mesh.n_cells == 0:
            raise ValueError("ground_mesh contains no cells after triangulation.")

        faces = tri_mesh.faces.reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise ValueError("ground_mesh triangulation produced non-triangle faces.")

        points = np.asarray(tri_mesh.points, dtype=float)
        triangles_3d = points[faces[:, 1:4]]

        triangles_xy = triangles_3d[:, :, :2]
        centroids_xy = np.mean(triangles_xy, axis=1)

        e1 = triangles_3d[:, 1] - triangles_3d[:, 0]
        e2 = triangles_3d[:, 2] - triangles_3d[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

        bounds_min = np.min(points[:, :2], axis=0)
        bounds_max = np.max(points[:, :2], axis=0)

        return triangles_xy, centroids_xy, areas, bounds_min, bounds_max

    def _build_quadtree(self) -> QuadtreeNode:
        root = QuadtreeNode(
            aabb_min=self.bounds_min,
            aabb_max=self.bounds_max,
            depth=0,
            max_depth=self.config.max_depth,
            max_triangles=self.config.max_triangles,
        )

        for tri in self.triangles_xy:
            root.insert(tri)

        return root

    def _build_triangle_lookup(self) -> Dict[bytes, List[int]]:
        lookup: Dict[bytes, List[int]] = {}
        for idx, tri in enumerate(self.triangles_xy):
            key = np.ascontiguousarray(tri).tobytes()
            lookup.setdefault(key, []).append(idx)
        return lookup

    def _build_grid_points(self) -> np.ndarray:
        step = float(self.config.grid_step)
        if step <= 0:
            raise ValueError("grid_step must be > 0.")

        x0, y0 = self.bounds_min
        x1, y1 = self.bounds_max

        xs = np.arange(x0, x1 + step * 0.5, step, dtype=float)
        ys = np.arange(y0, y1 + step * 0.5, step, dtype=float)

        if xs.size == 0 or ys.size == 0:
            return np.empty((0, 2), dtype=float)

        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack((gx.ravel(), gy.ravel()))

    @staticmethod
    def _circle_aabb_intersects(center: np.ndarray, radius: float, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        closest = np.clip(center, aabb_min, aabb_max)
        return float(np.dot(closest - center, closest - center)) <= radius * radius + EPS

    def _query_triangle_indices_in_radius(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        stack = [self.quadtree_root]
        out_indices: set[int] = set()

        while stack:
            node = stack.pop()
            if not self._circle_aabb_intersects(center_xy, radius, node.aabb_min, node.aabb_max):
                continue

            for tri in node.triangles:
                key = np.ascontiguousarray(tri).tobytes()
                tri_indices = self.triangle_key_to_indices.get(key, [])
                for idx in tri_indices:
                    out_indices.add(idx)

            if node.children is not None:
                stack.extend(node.children)

        if not out_indices:
            return np.empty((0,), dtype=np.int64)

        return np.fromiter(out_indices, dtype=np.int64)

    def evaluate_cost(self, light_positions: np.ndarray) -> float:
        """Compute J = w1 * dark_area + w2 * double_lit_area."""
        positions = np.asarray(light_positions, dtype=float)
        if positions.shape != (self.config.n_lights, 2):
            raise ValueError("light_positions must have shape (n_lights, 2).")

        lit_counts = np.zeros(self.triangles_xy.shape[0], dtype=np.int32)
        r2 = self.config.light_radius * self.config.light_radius

        for light_xy in positions:
            candidate_indices = self._query_triangle_indices_in_radius(light_xy, self.config.light_radius)
            if candidate_indices.size == 0:
                continue

            delta = self.centroids_xy[candidate_indices] - light_xy
            inside = np.einsum("ij,ij->i", delta, delta) <= r2 + EPS
            lit_counts[candidate_indices[inside]] += 1

        dark_area = float(np.sum(self.areas[lit_counts == 0]))
        double_lit_area = float(np.sum(self.areas[lit_counts >= 2]))

        return self.config.w1_dark * dark_area + self.config.w2_double_lit * double_lit_area

    def _random_individual(self) -> np.ndarray:
        n_grid = self.grid_points.shape[0]
        n_lights = self.config.n_lights

        if n_grid >= n_lights:
            sel = self.rng.choice(n_grid, size=n_lights, replace=False)
        else:
            sel = self.rng.choice(n_grid, size=n_lights, replace=True)

        return np.array(self.grid_points[sel], dtype=float)

    def _initialize_population(self) -> np.ndarray:
        return np.array([self._random_individual() for _ in range(self.config.population_size)], dtype=float)

    def _tournament_select(self, population: np.ndarray, costs: np.ndarray) -> np.ndarray:
        k = max(2, int(self.config.tournament_k))
        idxs = self.rng.choice(population.shape[0], size=k, replace=False)
        best = idxs[np.argmin(costs[idxs])]
        return np.array(population[best], copy=True)

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.config.n_lights) < 0.5
        child = np.array(parent_a, copy=True)
        child[mask] = parent_b[mask]
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        out = np.array(individual, copy=True)
        for i in range(self.config.n_lights):
            if self.rng.random() < self.config.mutation_rate:
                gidx = self.rng.integers(0, self.grid_points.shape[0])
                out[i] = self.grid_points[gidx]
        return out

    def optimize(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run GA and return (best_positions, best_cost, history)."""
        population = self._initialize_population()
        history = np.empty((self.config.generations,), dtype=float)

        best_positions = np.array(population[0], copy=True)
        best_cost = np.inf

        for gen in range(self.config.generations):
            costs = np.array([self.evaluate_cost(ind) for ind in population], dtype=float)

            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            history[gen] = gen_best_cost

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_positions = np.array(population[gen_best_idx], copy=True)

            elite_count = max(0, min(self.config.elitism, self.config.population_size))
            elites = np.argsort(costs)[:elite_count]
            next_pop: list[np.ndarray] = [np.array(population[i], copy=True) for i in elites]

            while len(next_pop) < self.config.population_size:
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)

            population = np.array(next_pop, dtype=float)

        return best_positions, best_cost, history


def optimize_streetlights(
    ground_mesh: pv.PolyData,
    n_lights: int,
    light_radius: float,
    w1: float,
    w2: float,
    grid_step: float,
    population_size: int = 64,
    generations: int = 80,
    mutation_rate: float = 0.12,
    seed: int | None = None,
) -> Dict[str, np.ndarray | float]:
    """Convenience wrapper for running the streetlight GA optimization."""
    config = GAConfig(
        n_lights=n_lights,
        light_radius=light_radius,
        w1_dark=w1,
        w2_double_lit=w2,
        grid_step=grid_step,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        seed=seed,
    )

    solver = StreetlightGA(ground_mesh=ground_mesh, config=config)
    best_positions, best_cost, history = solver.optimize()

    lit_ratio = 1.0
    if solver.total_area > EPS:
        # Recover dark and double-lit effects indirectly through final score decomposition.
        # We recompute counts once for the best layout and derive lit ratio directly.
        lit_counts = np.zeros(solver.triangles_xy.shape[0], dtype=np.int32)
        r2 = config.light_radius * config.light_radius

        for light_xy in best_positions:
            cand = solver._query_triangle_indices_in_radius(light_xy, config.light_radius)
            if cand.size == 0:
                continue
            delta = solver.centroids_xy[cand] - light_xy
            inside = np.einsum("ij,ij->i", delta, delta) <= r2 + EPS
            lit_counts[cand[inside]] += 1

        lit_area = float(np.sum(solver.areas[lit_counts > 0]))
        lit_ratio = lit_area / solver.total_area

    return {
        "best_positions": best_positions,
        "best_cost": float(best_cost),
        "history": history,
        "lit_ratio": float(lit_ratio),
    }
