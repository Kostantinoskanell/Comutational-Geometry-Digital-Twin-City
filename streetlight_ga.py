"""Genetic optimization of streetlight placement over a triangulated ground mesh."""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pyvista as pv
from shapely import contains_xy, intersects_xy
from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry

from shadow_engine import build_coverage_matrix
from spatial_trees import QuadtreeNode

EPS = 1e-9


def build_candidate_grid_points(
    bounds_min_xy: np.ndarray,
    bounds_max_xy: np.ndarray,
    grid_step: float,
    sidewalk_polygon: BaseGeometry | None = None,
) -> np.ndarray:
    """Build 2D grid candidates and optionally keep only sidewalk-intersecting points."""
    step = float(grid_step)
    if step <= 0:
        raise ValueError("grid_step must be > 0.")

    bmin = np.asarray(bounds_min_xy, dtype=float)
    bmax = np.asarray(bounds_max_xy, dtype=float)
    if bmin.shape != (2,) or bmax.shape != (2,):
        raise ValueError("bounds_min_xy and bounds_max_xy must have shape (2,).")

    x0, y0 = bmin
    x1, y1 = bmax
    xs = np.arange(x0, x1 + step * 0.5, step, dtype=float)
    ys = np.arange(y0, y1 + step * 0.5, step, dtype=float)

    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=float)

    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack((gx.ravel(), gy.ravel()))

    if sidewalk_polygon is None or sidewalk_polygon.is_empty:
        return grid

    # Shapely 2.x vectorized predicates are much faster than per-point Python calls.
    x = np.asarray(grid[:, 0], dtype=float)
    y = np.asarray(grid[:, 1], dtype=float)
    keep = np.asarray(contains_xy(sidewalk_polygon, x, y) | intersects_xy(sidewalk_polygon, x, y), dtype=bool)
    return grid[keep]


def build_sidewalk_polygon_from_street_graph(
    street_graph: nx.MultiDiGraph,
    road_buffer_m: float = 4.0,
    sidewalk_buffer_m: float = 6.0,
) -> BaseGeometry | None:
    """Construct sidewalk polygon from centerline buffers (sidewalk ring minus road)."""
    if road_buffer_m <= 0.0:
        raise ValueError("road_buffer_m must be > 0.")
    if sidewalk_buffer_m <= road_buffer_m:
        raise ValueError("sidewalk_buffer_m must be > road_buffer_m.")

    seg_starts, seg_ends = _extract_street_segments_xy(street_graph)
    if seg_starts.shape[0] == 0:
        return None

    lines = [
        ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        for a, b in zip(seg_starts, seg_ends)
    ]
    centerlines = MultiLineString(lines)

    road_polygon = centerlines.buffer(float(road_buffer_m))
    sidewalk_outer = centerlines.buffer(float(sidewalk_buffer_m))
    sidewalk_polygon = sidewalk_outer.difference(road_polygon)

    if sidewalk_polygon.is_empty:
        return None
    return sidewalk_polygon


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
    fitness_workers: int = 1
    progress_every: int = 5
    verbose: bool = False


class StreetlightGA:
    """Genetic optimizer for streetlight coordinates on a 2D grid."""

    def __init__(
        self,
        ground_mesh: pv.PolyData,
        config: GAConfig,
        sidewalk_polygon: BaseGeometry | None = None,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.sidewalk_polygon = sidewalk_polygon

        (
            self.triangles_xy,
            self.centroids_xy,
            self.areas,
            self.bounds_min,
            self.bounds_max,
        ) = self._extract_ground_triangles(ground_mesh)

        self.total_area = float(np.sum(self.areas))
        self.quadtree_root: QuadtreeNode | None = None
        self.triangle_key_to_indices: Dict[bytes, List[int]] | None = None

        self.grid_points = self._build_grid_points()
        if self.grid_points.shape[0] == 0:
            raise ValueError("No valid grid points were generated from bounds/grid_step.")

        if self.config.n_lights <= 0:
            raise ValueError("n_lights must be > 0.")

        if self.config.light_radius <= 0:
            raise ValueError("light_radius must be > 0.")

        if self.grid_points.shape[0] < self.config.n_lights:
            raise ValueError(
                "n_lights exceeds available sidewalk candidate points "
                f"({self.config.n_lights} > {self.grid_points.shape[0]}). "
                "Increase search area or reduce n_lights."
            )

        self.precomputed_coverage: np.ndarray | None = None
        self._grid_index_lookup: Dict[bytes, int] = {
            np.ascontiguousarray(p).tobytes(): int(i) for i, p in enumerate(self.grid_points)
        }
        self.weights = np.ones(self.triangles_xy.shape[0], dtype=float)

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
        # Fallback to full ground mesh bounds if sidewalk polygon is unavailable
        if self.sidewalk_polygon is None or self.sidewalk_polygon.is_empty:
            # Use full ground mesh bounds without sidewalk filtering
            min_x, min_y = self.bounds_min
            max_x, max_y = self.bounds_max
            return build_candidate_grid_points(
                bounds_min_xy=np.array([min_x, min_y], dtype=float),
                bounds_max_xy=np.array([max_x, max_y], dtype=float),
                grid_step=self.config.grid_step,
                sidewalk_polygon=None,
            )

        min_x, min_y, max_x, max_y = self.sidewalk_polygon.bounds
        return build_candidate_grid_points(
            bounds_min_xy=np.array([min_x, min_y], dtype=float),
            bounds_max_xy=np.array([max_x, max_y], dtype=float),
            grid_step=self.config.grid_step,
            sidewalk_polygon=self.sidewalk_polygon,
        )

    @staticmethod
    def _circle_aabb_intersects(center: np.ndarray, radius: float, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        closest = np.clip(center, aabb_min, aabb_max)
        return float(np.dot(closest - center, closest - center)) <= radius * radius + EPS

    def _ensure_spatial_index(self) -> None:
        """Build quadtree + triangle lookup on demand for radius neighborhood queries."""
        if self.quadtree_root is None:
            self.quadtree_root = self._build_quadtree()
        if self.triangle_key_to_indices is None:
            self.triangle_key_to_indices = self._build_triangle_lookup()

    def _query_triangle_indices_in_radius(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        self._ensure_spatial_index()
        if self.quadtree_root is None or self.triangle_key_to_indices is None:
            return np.empty((0,), dtype=np.int64)

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

    def evaluate_cost(self, individual_indices: np.ndarray) -> float:
        """Night-mode objective using coverage-matrix row indices.

        The chromosome is an integer array of length N where each value is a row
        index into the (M, K) coverage matrix.
        """
        if self.precomputed_coverage is None:
            raise ValueError("Precomputed coverage matrix is required for evaluate_cost.")

        idx = np.asarray(individual_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.shape[0] != self.config.n_lights:
            raise ValueError("individual_indices must have shape (n_lights,).")

        if np.any(idx < 0) or np.any(idx >= self.precomputed_coverage.shape[0]):
            raise ValueError("individual_indices contains out-of-range grid row indices.")

        counts = np.sum(self.precomputed_coverage[idx], axis=0, dtype=np.int32)
        dark_mask = counts == 0
        double_mask = counts >= 2

        area_dark = float(np.sum(self.areas[dark_mask] * self.weights[dark_mask]))
        area_double_lit = float(np.sum(self.areas[double_mask]))

        return self.config.w1_dark * area_dark + self.config.w2_double_lit * area_double_lit

    def set_precomputed_coverage(self, coverage_matrix: np.ndarray) -> None:
        """Attach precomputed (M, K) boolean matrix where M is grid points and K is triangles."""
        cov = np.asarray(coverage_matrix)
        if cov.ndim != 2:
            raise ValueError("coverage_matrix must be 2D with shape (M, K).")
        if cov.shape[0] != self.grid_points.shape[0]:
            raise ValueError("coverage_matrix row count must match number of grid points.")
        if cov.shape[1] != self.triangles_xy.shape[0]:
            raise ValueError("coverage_matrix column count must match number of ground triangles.")

        self.precomputed_coverage = cov.astype(bool, copy=False)

    def set_triangle_weights(self, weights: np.ndarray) -> None:
        """Set per-triangle priority weights used in weighted dark-area penalty."""
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != self.triangles_xy.shape[0]:
            raise ValueError("weights must have shape (K,) matching number of ground triangles.")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative.")
        self.weights = w

    def build_radius_coverage_matrix(self) -> np.ndarray:
        """Build fast geometric (radius-only) coverage matrix with shape (M, K)."""
        m = self.grid_points.shape[0]
        k = self.centroids_xy.shape[0]
        cov = np.zeros((m, k), dtype=bool)
        r2 = self.config.light_radius * self.config.light_radius

        for i, gp in enumerate(self.grid_points):
            delta = self.centroids_xy - gp
            cov[i] = np.einsum("ij,ij->i", delta, delta) <= r2 + EPS

        return cov

    def counts_from_indices(self, individual_indices: np.ndarray) -> np.ndarray:
        """Return per-triangle illumination counts from chromosome row indices."""
        if self.precomputed_coverage is None:
            raise ValueError("Precomputed coverage matrix is required.")
        idx = np.asarray(individual_indices, dtype=np.int64)
        return np.sum(self.precomputed_coverage[idx], axis=0, dtype=np.int32)

    def _random_individual(self) -> np.ndarray:
        n_grid = self.grid_points.shape[0]
        n_lights = self.config.n_lights

        sel = self.rng.choice(n_grid, size=n_lights, replace=False)
        return np.asarray(sel, dtype=np.int64)

    def _initialize_population(self) -> np.ndarray:
        return np.array([self._random_individual() for _ in range(self.config.population_size)], dtype=np.int64)

    def _tournament_select(self, population: np.ndarray, costs: np.ndarray) -> np.ndarray:
        k = max(2, int(self.config.tournament_k))
        replace = population.shape[0] < k
        idxs = self.rng.choice(population.shape[0], size=k, replace=replace)
        best = idxs[np.argmin(costs[idxs])]
        return np.array(population[best], dtype=np.int64, copy=True)

    def _repair_unique_indices(self, individual: np.ndarray) -> np.ndarray:
        """Ensure each chromosome uses distinct candidate rows (no duplicate lights)."""
        out = np.asarray(individual, dtype=np.int64).copy()
        if out.ndim != 1 or out.shape[0] != self.config.n_lights:
            raise ValueError("individual must have shape (n_lights,).")

        n_grid = int(self.grid_points.shape[0])
        if n_grid <= 0:
            raise ValueError("No grid points available.")

        np.clip(out, 0, n_grid - 1, out=out)
        unique_vals, first_idx = np.unique(out, return_index=True)
        if unique_vals.shape[0] == out.shape[0]:
            return out

        keep = np.zeros((out.shape[0],), dtype=bool)
        keep[first_idx] = True

        missing = np.setdiff1d(np.arange(n_grid, dtype=np.int64), unique_vals, assume_unique=False)
        self.rng.shuffle(missing)
        mptr = 0

        for i in range(out.shape[0]):
            if keep[i]:
                continue
            if mptr < missing.shape[0]:
                out[i] = missing[mptr]
                mptr += 1
            else:
                out[i] = int(self.rng.integers(0, n_grid))

        return out

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.config.n_lights) < 0.5
        child = np.array(parent_a, dtype=np.int64, copy=True)
        child[mask] = parent_b[mask]
        return self._repair_unique_indices(child)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        out = np.array(individual, dtype=np.int64, copy=True)
        for i in range(self.config.n_lights):
            if self.rng.random() < self.config.mutation_rate:
                gidx = self.rng.integers(0, self.grid_points.shape[0])
                out[i] = int(gidx)
        return self._repair_unique_indices(out)

    def _evaluate_population_costs_chunk(self, pop_chunk: np.ndarray, weighted_areas: np.ndarray) -> np.ndarray:
        counts = np.sum(self.precomputed_coverage[pop_chunk], axis=1, dtype=np.int32)  # FIXED: int16→int32 prevents overflow for K>32767
        dark_mask = counts == 0
        double_mask = counts >= 2
        area_dark = dark_mask @ weighted_areas
        area_double_lit = double_mask @ self.areas
        return self.config.w1_dark * area_dark + self.config.w2_double_lit * area_double_lit

    def _evaluate_population_costs(
        self,
        population: np.ndarray,
        executor: cf.Executor | None = None,
    ) -> np.ndarray:
        if self.precomputed_coverage is None:
            raise ValueError("Precomputed coverage matrix is required for cost evaluation.")

        pop = np.asarray(population, dtype=np.int64)
        if pop.ndim != 2 or pop.shape[1] != self.config.n_lights:
            raise ValueError("population must have shape (P, n_lights).")

        if np.any(pop < 0) or np.any(pop >= self.precomputed_coverage.shape[0]):
            raise ValueError("population contains out-of-range grid row indices.")

        weighted_areas = self.areas * self.weights
        workers = max(1, int(self.config.fitness_workers))

        if workers <= 1 or pop.shape[0] < max(8, workers * 2):
            return self._evaluate_population_costs_chunk(pop, weighted_areas)

        chunks = [chunk for chunk in np.array_split(pop, workers) if chunk.shape[0] > 0]
        if not chunks:
            return np.empty((0,), dtype=float)

        if executor is None:
            with cf.ThreadPoolExecutor(max_workers=workers) as local_executor:
                futures = [
                    local_executor.submit(self._evaluate_population_costs_chunk, chunk, weighted_areas)
                    for chunk in chunks
                ]
                parts = [f.result() for f in futures]
        else:
            futures = [executor.submit(self._evaluate_population_costs_chunk, chunk, weighted_areas) for chunk in chunks]
            parts = [f.result() for f in futures]

        return np.concatenate(parts, axis=0)

    def optimize(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run GA and return (best_positions, best_cost, history)."""
        population = self._initialize_population()
        history = np.empty((self.config.generations,), dtype=float)

        best_indices = np.array(population[0], dtype=np.int64, copy=True)
        best_cost = np.inf
        t0 = time.perf_counter()
        progress_every = max(1, int(self.config.progress_every))
        workers = max(1, int(self.config.fitness_workers))
        use_parallel_fitness = workers > 1 and self.config.population_size >= max(8, workers * 2)
        fitness_executor: cf.ThreadPoolExecutor | None = None
        if use_parallel_fitness:
            fitness_executor = cf.ThreadPoolExecutor(max_workers=workers)

        try:
            for gen in range(self.config.generations):
                costs = self._evaluate_population_costs(population, executor=fitness_executor)

                gen_best_idx = int(np.argmin(costs))
                gen_best_cost = float(costs[gen_best_idx])
                history[gen] = gen_best_cost

                if gen_best_cost < best_cost:
                    best_cost = gen_best_cost
                    best_indices = np.array(population[gen_best_idx], dtype=np.int64, copy=True)

                elite_count = max(0, min(self.config.elitism, self.config.population_size))
                elites = np.argsort(costs)[:elite_count]
                next_pop: list[np.ndarray] = [
                    np.array(population[i], dtype=np.int64, copy=True) for i in elites
                ]

                while len(next_pop) < self.config.population_size:
                    p1 = self._tournament_select(population, costs)
                    p2 = self._tournament_select(population, costs)
                    child = self._crossover(p1, p2)
                    child = self._mutate(child)
                    next_pop.append(child)

                population = np.array(next_pop, dtype=np.int64)

                if self.config.verbose and (
                    gen == 0
                    or ((gen + 1) % progress_every == 0)
                    or (gen + 1 == self.config.generations)
                ):
                    elapsed = time.perf_counter() - t0
                    worker_msg = f", fitness_workers={workers}" if use_parallel_fitness else ""
                    print(
                        f"GA progress {gen + 1:>3}/{self.config.generations}: "
                        f"gen_best={gen_best_cost:.6f}, global_best={best_cost:.6f}, "
                        f"elapsed={elapsed:.1f}s{worker_msg}"
                    )
        finally:
            if fitness_executor is not None:
                fitness_executor.shutdown(wait=True)

        return best_indices, best_cost, history


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
    octree_root=None,
    pole_height: float = 3.0,
    use_precomputed_coverage: bool = False,
    ground_weights: np.ndarray | None = None,
    precomputed_coverage_matrix: np.ndarray | None = None,
    street_graph: nx.MultiDiGraph | None = None,
    sidewalk_polygon: BaseGeometry | None = None,
    ga_jobs: int = 1,
    ga_progress_every: int = 5,
    ga_verbose: bool = False,
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
        fitness_workers=max(1, int(ga_jobs)),
        progress_every=max(1, int(ga_progress_every)),
        verbose=bool(ga_verbose),
    )

    resolved_sidewalk_polygon = sidewalk_polygon
    if resolved_sidewalk_polygon is None and street_graph is not None:
        resolved_sidewalk_polygon = build_sidewalk_polygon_from_street_graph(street_graph)
    # Fallback allowed: if sidewalk_polygon is still None or empty, 
    # StreetlightGA will use the full ground mesh bounds with street-proximity weighting.

    solver = StreetlightGA(
        ground_mesh=ground_mesh,
        config=config,
        sidewalk_polygon=resolved_sidewalk_polygon,
    )

    if precomputed_coverage_matrix is not None:
        solver.set_precomputed_coverage(precomputed_coverage_matrix)
    else:
        # Always use coverage matrix with LOS checks (no pure-radius fallback)
        if octree_root is None:
            raise ValueError("octree_root is required for coverage matrix computation.")
        coverage = build_coverage_matrix(
            grid_points=solver.grid_points,
            ground_mesh=ground_mesh,
            octree_root=octree_root,
            radius=light_radius,
            pole_height=pole_height,
            street_graph=street_graph,
        )
        solver.set_precomputed_coverage(coverage)

    if ground_weights is not None:
        solver.set_triangle_weights(ground_weights)
    elif street_graph is not None:
        # Auto-compute street-proximity weights when not provided
        auto_weights = compute_ground_weights(ground_mesh, street_graph, max_dist=15.0)
        if auto_weights.shape[0] == solver.weights.shape[0]:
            solver.set_triangle_weights(auto_weights)

    best_indices, best_cost, history = solver.optimize()
    best_positions = solver.grid_points[best_indices]

    lit_ratio = 1.0
    if solver.total_area > EPS:
        lit_counts = solver.counts_from_indices(best_indices)
        lit_area = float(np.sum(solver.areas[lit_counts > 0]))
        lit_ratio = lit_area / solver.total_area

    return {
        "best_positions": best_positions,
        "best_indices": best_indices,
        "final_counts": solver.counts_from_indices(best_indices),
        "best_cost": float(best_cost),
        "history": history,
        "lit_ratio": float(lit_ratio),
    }


def _extract_street_segments_xy(street_graph: nx.MultiDiGraph) -> tuple[np.ndarray, np.ndarray]:
    """Extract XY street segments from graph edge geometry or endpoint nodes."""
    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []

    for u, v, data in street_graph.edges(data=True):
        geom = data.get("geometry")

        if geom is not None and hasattr(geom, "coords"):
            coords = np.asarray(geom.coords, dtype=float)
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
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)

    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _min_distance_point_to_segments(point_xy: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray) -> float:
    """Compute minimum Euclidean distance from a point to a set of 2D segments."""
    p = np.asarray(point_xy, dtype=float)
    v = seg_ends - seg_starts
    w = p - seg_starts

    vv = np.einsum("ij,ij->i", v, v)
    near_zero = vv <= EPS

    t = np.zeros_like(vv)
    valid = ~near_zero
    t[valid] = np.einsum("ij,ij->i", w[valid], v[valid]) / vv[valid]
    t = np.clip(t, 0.0, 1.0)

    proj = seg_starts + t[:, None] * v
    delta = p - proj
    d2 = np.einsum("ij,ij->i", delta, delta)

    if np.any(near_zero):
        dd = p - seg_starts[near_zero]
        d2[near_zero] = np.einsum("ij,ij->i", dd, dd)

    return float(np.sqrt(np.min(d2)))


def compute_ground_weights(
    ground_mesh: pv.PolyData,
    street_graph: nx.MultiDiGraph,
    max_dist: float = 15.0,
) -> np.ndarray:
    """Compute triangle priority weights based on centroid distance to streets.

    Triangles whose centroid lies within ``max_dist`` of any street segment receive
    weight ``10.0``; all others receive weight ``1.0``.
    """
    if max_dist < 0.0:
        raise ValueError("max_dist must be >= 0.")

    tri_mesh = ground_mesh.triangulate()
    if tri_mesh.n_cells == 0:
        return np.empty((0,), dtype=float)

    faces = tri_mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("ground_mesh triangulation produced non-triangle faces.")

    points = np.asarray(tri_mesh.points, dtype=float)
    tri_pts = points[faces[:, 1:4]]
    centroids_xy = np.mean(tri_pts[:, :, :2], axis=1)

    seg_starts, seg_ends = _extract_street_segments_xy(street_graph)
    if seg_starts.shape[0] == 0:
        return np.ones((centroids_xy.shape[0],), dtype=float)

    weights = np.ones((centroids_xy.shape[0],), dtype=float)
    # FIXED: replace O(N×M) Python loop with cKDTree pre-filter + exact check only on candidates
    from scipy.spatial import cKDTree
    seg_mids = (seg_starts + seg_ends) * 0.5
    tree = cKDTree(seg_mids)
    # 2× max_dist conservative bound: catches segments whose midpoint is far but endpoint is close
    candidate_mask = np.zeros(centroids_xy.shape[0], dtype=bool)
    idxs = tree.query_ball_point(centroids_xy, r=max_dist * 2.0 + EPS)
    rough_candidates = np.where([len(x) > 0 for x in idxs])[0]
    for i in rough_candidates:
        if _min_distance_point_to_segments(centroids_xy[i], seg_starts, seg_ends) <= max_dist + EPS:
            candidate_mask[i] = True
    weights[candidate_mask] = 10.0
    return weights


def _smart_light_positions(
    ground_mesh: "pv.PolyData",
    street_graph: "nx.MultiDiGraph",
    n_lights: int,
    grid_step: float,
    seed: int,
) -> np.ndarray:
    """Greedy streetlight layout: prefer sidewalk/road candidates with even spacing."""
    from app_core import _build_grid_points_from_ground_mesh

    sidewalk_polygon = None
    try:
        sidewalk_polygon = build_sidewalk_polygon_from_street_graph(street_graph)
    except Exception:
        sidewalk_polygon = None

    step = max(4.0, float(grid_step))
    candidates = _build_grid_points_from_ground_mesh(
        ground_mesh, step, sidewalk_polygon=sidewalk_polygon,
    )
    if candidates.shape[0] == 0 and sidewalk_polygon is not None:
        candidates = _build_grid_points_from_ground_mesh(ground_mesh, step, sidewalk_polygon=None)
    if candidates.shape[0] == 0:
        raise ValueError("Ground mesh has no candidate points for smart light placement.")

    candidates = np.asarray(candidates, dtype=float)
    count = max(1, int(n_lights))
    if candidates.shape[0] <= count:
        return candidates.copy()

    node_xy: list[tuple[float, float]] = []
    for _, data in street_graph.nodes(data=True):
        if "x" in data and "y" in data:
            node_xy.append((float(data["x"]), float(data["y"])))
    nodes = np.asarray(node_xy, dtype=float) if node_xy else candidates

    center = np.mean(nodes, axis=0)
    scene_span = float(max(np.ptp(candidates[:, 0]), np.ptp(candidates[:, 1]), 1.0))
    density_radius = max(18.0, min(55.0, scene_span * 0.12))
    density = np.zeros(candidates.shape[0], dtype=float)
    for start in range(0, candidates.shape[0], 2048):
        chunk = candidates[start:start + 2048]
        d2 = np.sum((chunk[:, None, :] - nodes[None, :, :]) ** 2, axis=2)
        density[start:start + chunk.shape[0]] = np.sum(d2 <= density_radius * density_radius, axis=1)
    density = density / max(float(np.max(density)), 1.0)

    center_dist = np.linalg.norm(candidates - center, axis=1)
    center_score = 1.0 - center_dist / max(float(np.max(center_dist)), 1e-9)
    first_idx = int(np.argmax(0.65 * density + 0.35 * center_score))

    selected = [first_idx]
    min_dist = np.linalg.norm(candidates - candidates[first_idx], axis=1)
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(0.0, 1e-6, size=candidates.shape[0])
    target_spacing = max(12.0, scene_span / max(np.sqrt(float(count)) * 1.6, 1.0))

    while len(selected) < count:
        spacing_score = np.clip(min_dist / target_spacing, 0.0, 1.0)
        score = 0.72 * spacing_score + 0.28 * density + jitter
        score[np.asarray(selected, dtype=np.int64)] = -np.inf
        next_idx = int(np.argmax(score))
        if not np.isfinite(score[next_idx]):
            break
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(candidates - candidates[next_idx], axis=1))

    return candidates[np.asarray(selected, dtype=np.int64)]
