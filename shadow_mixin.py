from __future__ import annotations
import hashlib
import time
import numpy as np
import concurrent.futures as cf
from pathlib import Path
from shadow_engine import (
    _compute_shadows_numba,
    _extract_building_triangle_aabbs,
    _extract_building_triangles,
    build_coverage_matrix,
    compute_shadows,
)
from spatial_trees import OctreeNode


class ShadowMixin:
    def _surface_kind(self, mesh) -> np.ndarray:
        raw = mesh.cell_data.get("surface_kind")
        if raw is None:
            return np.zeros((mesh.n_cells,), dtype=np.uint8)
        kinds = np.asarray(raw, dtype=np.uint8).reshape(-1)
        if kinds.shape[0] != mesh.n_cells:
            return np.zeros((mesh.n_cells,), dtype=np.uint8)
        return kinds

    def _compose_day_surface_classes(self, shadow_mask_local: np.ndarray) -> np.ndarray:
        mask = np.asarray(shadow_mask_local, dtype=bool).reshape(-1)
        kinds = self._surface_kind(self.ground_mesh)
        if mask.shape[0] != kinds.shape[0]:
            mask = np.zeros((kinds.shape[0],), dtype=bool)

        road = kinds == 0
        sidewalk = ~road
        classes = np.zeros((kinds.shape[0],), dtype=np.uint8)
        classes[road & ~mask] = 1
        classes[sidewalk & mask] = 2
        classes[sidewalk & ~mask] = 3
        return classes

    def _compose_night_surface_classes(self, illum: np.ndarray) -> np.ndarray:
        coverage = np.asarray(illum, dtype=np.uint8).reshape(-1)
        kinds = self._surface_kind(self.ground_mesh)
        if coverage.shape[0] != kinds.shape[0]:
            coverage = np.zeros((kinds.shape[0],), dtype=np.uint8)

        road = kinds == 0
        sidewalk = ~road
        classes = np.zeros((kinds.shape[0],), dtype=np.uint8)

        single = coverage == 1
        double = coverage >= 2

        classes[road & single] = 1
        classes[road & double] = 2
        classes[sidewalk & (coverage == 0)] = 3
        classes[sidewalk & single] = 4
        classes[sidewalk & double] = 5
        return classes

    def _ensure_octree(self) -> OctreeNode:
        from app_core import _load_or_build_spatial_cache, _build_octree_from_buildings
        if self.octree_root is None:
            t0 = time.perf_counter()
            if self.args.fast_startup:
                try:
                    gm_cached, oc_cached = _load_or_build_spatial_cache(
                        cache_dir=self.cache_dir,
                        use_cache=self.use_cache,
                        cache_context_key=self.cache_context_key,
                        buildings_mesh=self.buildings_mesh,
                        ground_resolution=self.effective_ground_res,
                        preferred_ground_mesh=self.ground_mesh,
                    )
                    if self.ground_mesh is None:
                        self.ground_mesh = gm_cached
                    self.octree_root = oc_cached
                    print("Loaded lighting geometry cache.")
                except Exception as exc:
                    print(f"Spatial cache unavailable ({exc}). Rebuilding lighting geometry cache...")
            if self.octree_root is None:
                print("Building lighting geometry cache...")
                self.octree_root = _build_octree_from_buildings(self.buildings_mesh)
                dt = time.perf_counter() - t0
                print(f"Lighting geometry cache ready in {dt:.2f}s")
        return self.octree_root

    def _build_edge_shadow_cache(self, hour: float) -> np.ndarray:
        """Return per-edge shadow fraction at current hour (shape=(len(car_paths),))."""
        from app_core import _sun_dir_from_hour
        n_paths = len(self.car_paths)
        if n_paths == 0:
            return np.empty((0,), dtype=float)

        hour_key = round(float(hour), 1)
        cache = self.scene_state.get("edge_shadow_cache")
        if not isinstance(cache, dict):
            cache = {}
            self.scene_state["edge_shadow_cache"] = cache
        cached = cache.get(hour_key)
        if cached is not None:
            return np.asarray(cached, dtype=float)

        sun_dir = np.asarray(_sun_dir_from_hour(float(hour)), dtype=np.float64)
        if float(sun_dir[2]) <= 0.0:
            out_night = np.ones((n_paths,), dtype=float)
            cache[hour_key] = out_night
            return out_night

        direction = sun_dir / max(1e-12, float(np.linalg.norm(sun_dir)))
        root = self._ensure_octree()
        building_triangles = _extract_building_triangles(root)
        tri_mins, tri_maxs = _extract_building_triangle_aabbs(root, building_triangles)

        n_samples = 5
        centroids = np.empty((n_paths * n_samples, 3), dtype=np.float64)
        for pi, path in enumerate(self.car_paths):
            pts = np.asarray(path["points"], dtype=np.float64)
            cum = np.asarray(path["cum_len"], dtype=np.float64)
            total = float(path["length"])
            s = np.linspace(0.0, max(0.0, total), n_samples, dtype=np.float64)
            if pts.shape[0] < 2:
                base = pi * n_samples
                centroids[base:base + n_samples, 0] = 0.0
                centroids[base:base + n_samples, 1] = 0.0
                centroids[base:base + n_samples, 2] = 4.0
                continue

            seg_idx = np.searchsorted(cum, s, side="right") - 1
            seg_idx = np.clip(seg_idx, 0, pts.shape[0] - 2)
            l0 = cum[seg_idx]
            l1 = cum[seg_idx + 1]
            denom = np.maximum(1e-9, l1 - l0)
            t = (s - l0) / denom
            p0 = pts[seg_idx]
            p1 = pts[seg_idx + 1]
            sample_pts = p0 + (p1 - p0) * t[:, None]

            base = pi * n_samples
            centroids[base:base + n_samples, 0:2] = sample_pts[:, 0:2]
            centroids[base:base + n_samples, 2] = 4.0

        areas = np.ones((centroids.shape[0],), dtype=np.float64)
        # NON-blocking lock: this runs on the MAIN thread (editor clicks,
        # time-of-day slider).  If the background shadow worker holds the
        # kernel right now, return a stale/neutral estimate instead of either
        # freezing the UI for seconds or crashing numba's workqueue layer.
        from shadow_engine import NUMBA_KERNEL_LOCK
        if not NUMBA_KERNEL_LOCK.acquire(blocking=False):
            for _prev_key in sorted(cache.keys(), key=lambda k: abs(k - hour_key)):
                return np.asarray(cache[_prev_key], dtype=float)   # nearest hour
            return np.zeros((n_paths,), dtype=float)               # no data yet
        try:
            shadowed, _ = _compute_shadows_numba(
                centroids,
                areas,
                direction.astype(np.float64),
                building_triangles,
                tri_mins,
                tri_maxs,
            )
        finally:
            NUMBA_KERNEL_LOCK.release()
        out = np.mean(shadowed.reshape(n_paths, n_samples), axis=1, dtype=np.float64)
        cache[hour_key] = out
        return out

    def _apply_shadow_result_to_ground(
        self, hour: float, spot_radius: float, is_night: bool,
        mask_or_coverage,        # bool ndarray (day) or int16 ndarray (night)
        lit_ratio_or_pct: float,
    ) -> None:
            """Apply pre-computed shadow/coverage data to the ground mesh (main thread)."""
            style = self._style()
            if is_night:
                coverage_count = np.asarray(mask_or_coverage, dtype=np.int16)
                night_copy = self.scene_state["night_ground_copy"]
                if coverage_count.shape[0] != night_copy.n_cells:
                    print("[shadow-apply] coverage shape mismatch, skipping night ground update.")
                    return
                illum = np.zeros(night_copy.n_cells, dtype=np.uint8)
                illum[coverage_count >= 1] = 1
                illum[coverage_count >= 2] = 2
                night_copy.cell_data["night_surface_class"] = self._compose_night_surface_classes(illum)
                lit_pct = lit_ratio_or_pct
                self.scene_state["ground_actor"] = self.plotter.add_mesh(
                    night_copy,
                    scalars="night_surface_class",
                    clim=[0, 5],
                    cmap=self.night_surface_cmap,
                    show_edges=False,
                    show_scalar_bar=False,
                    opacity=0.9,
                    name="ground_mesh",
                    # Semantic colors (lit / unlit / double-lit) — must not be
                    # dimmed by the (now working) 0.1-intensity night lights,
                    # or the streetlight pools go invisible
                    lighting=False,
                )
                self.plotter.add_text(
                    f"Hour {hour:04.1f}  |  Night Mode  |  Lit {lit_pct:05.1f}%",
                    position=(0.18, 0.02),
                    name="status",
                    font_size=9,
                    viewport=True,
                    color="white",
                )
            else:
                mask = np.asarray(mask_or_coverage, dtype=bool)
                lit_ratio = float(lit_ratio_or_pct)
                day_copy = self.scene_state["day_ground_copy"]
                day_copy.cell_data["day_surface_class"] = self._compose_day_surface_classes(mask)
                self.scene_state["ground_actor"] = self.plotter.add_mesh(
                    day_copy,
                    scalars="day_surface_class",
                    clim=[0, 3],
                    cmap=self.day_surface_cmap,
                    show_edges=False,
                    show_scalar_bar=False,
                    opacity=0.9,
                    name="ground_mesh",
                    lighting=False,   # semantic shadow classes, not lit geometry
                )
                self.plotter.add_text(
                    f"Hour {hour:04.1f}  |  Lit Ratio {lit_ratio:.3f}",
                    position=(0.18, 0.02),
                    name="status",
                    font_size=9,
                    viewport=True,
                    color="black",
                )
            self.plotter.render()

    def _shadow_bg_worker(
        self, _hour: float, _spot_radius: float, _is_night: bool, _sun_dir,
        _n_cells: int,
        _triangles: np.ndarray,       # (N, 3, 3) float64 — pre-extracted on main thread
        _areas: np.ndarray,            # (N,)      float64
        _centroids: np.ndarray,        # (N, 3)    float64
        _octree_snap,
        _best_positions_snap: np.ndarray,
        _street_graph_snap,
    ):
            """Runs in background thread; returns (mask_or_coverage, lit_ratio_or_pct).
                NOTE: NO VTK calls allowed here — all geometry is pre-extracted as numpy.
                """
            if _is_night:
                night_key = (
                    float(_spot_radius),
                    int(_best_positions_snap.shape[0]),
                    hashlib.sha1(
                        np.ascontiguousarray(_best_positions_snap, dtype=np.float64).tobytes()
                    ).hexdigest()[:16],
                )
                # Use cached coverage if still valid
                if (
                    "cached_night_coverage" in self.scene_state
                    and self.scene_state.get("cached_night_key") == night_key
                ):
                    coverage_count = np.asarray(self.scene_state["cached_night_coverage"], dtype=np.int16)
                else:
                    print(
                        f"[bg-shadow] Building night coverage "
                        f"(lights={len(_best_positions_snap)}, "
                        f"cells={_n_cells}, radius={_spot_radius:.1f}m)..."
                    )
                    t0 = time.perf_counter()
                    # Build coverage using pre-extracted numpy arrays (no VTK calls)
                    from shadow_engine import (
                        _extract_street_segments_xy,
                        _extract_building_triangles,
                        _extract_building_triangle_aabbs,
                        _compute_spotlight_coverage_numba,
                    )
                    from scipy.spatial import cKDTree
                    from tqdm import tqdm as _tqdm
                    grid = np.asarray(_best_positions_snap, dtype=np.float64)
                    centroids_xy = _centroids[:, :2].astype(np.float64)
                    _seg_starts, _seg_ends = _extract_street_segments_xy(_street_graph_snap)
                    building_triangles = _extract_building_triangles(_octree_snap)
                    building_aabb_mins, building_aabb_maxs = _extract_building_triangle_aabbs(_octree_snap, building_triangles)
                    m = int(grid.shape[0])
                    k = int(_centroids.shape[0])
                    cov = np.zeros((m, k), dtype=bool)
                    pole_h = float(self.args.pole_height)
                    radius_f = float(_spot_radius)
                    tree = cKDTree(centroids_xy)
                    dummy_orientation = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                    for i in _tqdm(range(m), desc="Coverage", unit="gridpoint"):
                        gp = grid[i]
                        cand = np.asarray(tree.query_ball_point(gp, r=radius_f), dtype=np.int64)
                        if cand.size == 0:
                            continue
                        light_pos = np.array([gp[0], gp[1], pole_h], dtype=np.float64)
                        # Correct call: cand=indices array, cov[i]=in-place bool output row
                        from shadow_engine import NUMBA_KERNEL_LOCK
                        with NUMBA_KERNEL_LOCK:
                            _compute_spotlight_coverage_numba(
                                light_pos, _centroids, dummy_orientation, -1.0,
                                building_triangles, building_aabb_mins, building_aabb_maxs,
                                cand,
                                cov[i],
                            )
                    coverage_count = np.sum(cov, axis=0, dtype=np.int16)
                    self.scene_state["cached_night_coverage"] = coverage_count
                    self.scene_state["cached_night_key"] = night_key
                    dt = time.perf_counter() - t0
                    print(f"[bg-shadow] Night coverage done in {dt:.2f}s")
                illum_count = float(np.count_nonzero(coverage_count >= 1))
                lit_pct = 100.0 * illum_count / float(max(1, coverage_count.size))
                return coverage_count, lit_pct
            else:
                day_cache = self.scene_state.get("cached_day_shadows")
                if not isinstance(day_cache, dict):
                    day_cache = {}
                    self.scene_state["cached_day_shadows"] = day_cache
                hour_key = round(float(_hour), 2)
                cached_day = day_cache.get(hour_key)
                if cached_day is not None:
                    return cached_day  # (mask, lit_ratio) already cached
                print(
                    f"[bg-shadow] Computing day shadows "
                    f"(hour={_hour:.1f}, cells={_n_cells})..."
                )
                t0 = time.perf_counter()
                # Use pre-extracted numpy arrays (no VTK calls)
                from shadow_engine import (
                    _extract_building_triangles,
                    _extract_building_triangle_aabbs,
                    _compute_shadows_numba,
                )
                direction = np.asarray(_sun_dir, dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                direction = direction / max(norm, 1e-9)
                building_triangles = _extract_building_triangles(_octree_snap)
                building_aabb_mins, building_aabb_maxs = _extract_building_triangle_aabbs(_octree_snap, building_triangles)
                from shadow_engine import NUMBA_KERNEL_LOCK
                with NUMBA_KERNEL_LOCK:
                    shadowed, ratio = _compute_shadows_numba(
                        _centroids.astype(np.float64),
                        _areas.astype(np.float64),
                        direction,
                        building_triangles,
                        building_aabb_mins,
                        building_aabb_maxs,
                    )
                result = (shadowed.astype(bool, copy=False), float(ratio))
                day_cache[hour_key] = result
                if len(day_cache) > 12:
                    oldest = next(iter(day_cache))
                    day_cache.pop(oldest, None)
                dt = time.perf_counter() - t0
                print(f"[bg-shadow] Day shadows done for hour={hour_key:.2f} in {dt:.2f}s")
                return result

    def _render_ground(self, hour: float, spot_radius: float, _use_update: bool = False) -> None:
            """Compatibility shim — delegates to the async render path."""
            from app_core import _sun_dir_from_hour
            sun_dir = _sun_dir_from_hour(hour)
            is_night = bool(sun_dir[2] <= 0.0)
            self.scene_state["is_night"] = is_night
            self.edge_shadow_frac = self._build_edge_shadow_cache(float(hour))
            style = self._style()
            # Apply fast visual updates immediately so the scene looks correct
            self._apply_visual_updates(hour, spot_radius, is_night, sun_dir, style)
            self.plotter.render()
            # Queue the heavy shadow computation in the background
            self._schedule_shadow_job(float(hour), float(spot_radius), is_night, sun_dir)

    def _night_cache_key_for(self, _spot_radius: float) -> tuple[float, int, str]:
            return (
                float(_spot_radius),
                int(self.best_positions.shape[0]),
                hashlib.sha1(np.ascontiguousarray(self.best_positions, dtype=np.float64).tobytes()).hexdigest()[:16],
            )

    def _schedule_shadow_job(self, _hour: float, _spot_radius: float, _is_night: bool, _sun_dir) -> None:
            """Submit shadow computation to background executor.
                All VTK geometry is extracted HERE on the main thread (thread-safe).
                The background thread receives only pure numpy arrays.
                """
            running = self.scene_state.get("_shadow_future")
            if running is not None and not running.done():
                # Store latest request; _poll will re-schedule when current job finishes
                self.scene_state["_shadow_pending"] = {
                    "hour": _hour, "spot_radius": _spot_radius,
                    "is_night": _is_night, "sun_dir": _sun_dir,
                }
                return

            self.scene_state["_shadow_pending"] = None
            self.plotter.add_text(
                "Computing shadows…",
                position=(0.18, 0.02), name="status",
                font_size=9, viewport=True, color="#ffca28",
            )
            try:
                self.plotter.render()
            except Exception:
                pass

            # --- Pre-extract VTK geometry on the MAIN THREAD (thread-safe) ---
            from shadow_engine import _extract_ground_triangles_areas_centroids_cached
            try:
                _tris, _areas, _cents = _extract_ground_triangles_areas_centroids_cached(self.ground_mesh)
                _tris = np.ascontiguousarray(_tris, dtype=np.float64)
                _areas = np.ascontiguousarray(_areas, dtype=np.float64)
                _cents = np.ascontiguousarray(_cents, dtype=np.float64)
                _n_cells = int(self.ground_mesh.n_cells)
            except Exception as _geo_exc:
                print(f"[schedule-shadow] geometry extraction failed: {_geo_exc}")
                return
            # -----------------------------------------------------------------

            _oct = self._ensure_octree()
            fut = self._shadow_executor.submit(
                self._shadow_bg_worker,
                _hour, _spot_radius, _is_night, _sun_dir,
                _n_cells,
                _tris,    # pure numpy — safe for background thread
                _areas,
                _cents,
                _oct,
                self.best_positions.copy(),
                self.street_graph,
            )
            # Attach metadata so _poll can apply results
            fut._render_args = (_hour, _spot_radius, _is_night)
            self.scene_state["_shadow_future"] = fut

    def _request_shadow_render(self, _hour: float, _spot_radius: float) -> None:
            from app_core import _sun_dir_from_hour
            self.scene_state["hour"] = float(_hour)
            self.scene_state["spot_radius"] = float(_spot_radius)
            sun_dir = _sun_dir_from_hour(_hour)
            is_night = bool(sun_dir[2] <= 0.0)
            self.scene_state["is_night"] = is_night
            self.edge_shadow_frac = self._build_edge_shadow_cache(float(_hour))
            style = self._style()
            try:
                # Fast visual update — returns immediately
                self._apply_visual_updates(_hour, _spot_radius, is_night, sun_dir, style)
                self.plotter.render()
            except Exception as _exc:
                print(f"[render] fast visual update failed: {_exc}")
            # Queue heavy computation in background
            self._schedule_shadow_job(float(_hour), float(_spot_radius), is_night, sun_dir)

    def _poll_shadow_job(self, _: int) -> None:
            """Called every 250 ms. Applies finished shadow results to the main thread."""
            fut = self.scene_state.get("_shadow_future")
            if fut is None or not fut.done():
                return
            # Consume the future
            self.scene_state["_shadow_future"] = None
            try:
                result = fut.result()
                hour, spot_radius, is_night = fut._render_args
                mask_or_cov, ratio_or_pct = result
                self._apply_shadow_result_to_ground(hour, spot_radius, is_night, mask_or_cov, ratio_or_pct)
            except Exception as _exc:
                import traceback as _tb
                print(f"[poll-shadow] applying result failed: {_exc}")
                _tb.print_exc()
            # If a newer request was queued while we were computing, run it now
            pending = self.scene_state.get("_shadow_pending")
            if pending is not None:
                self.scene_state["_shadow_pending"] = None
                self._schedule_shadow_job(
                    pending["hour"], pending["spot_radius"],
                    pending["is_night"], pending["sun_dir"],
                )
