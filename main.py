from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from app_cli import apply_gui_inputs, parse_args
from app_core import (
    _build_grid_points_from_ground_mesh,
    _build_ground_mesh_for_tests,
    _build_octree_from_buildings,
    _build_spotlight_discs,
    _cache_key,
    _initial_light_positions,
    _load_or_build_coverage_matrix_cached,
    _load_or_build_spatial_cache,
    _load_or_fetch_osm_cached,
    _street_line_layers,
    _sun_dir_from_hour,
)
from shadow_engine import build_coverage_matrix, compute_shadows
from spatial_trees import OctreeNode
from streetlight_ga import (
    build_sidewalk_polygon_from_street_graph,
    optimize_streetlights,
)


def main() -> None:
    def _stage(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    args = parse_args()
    if args.gui:
        args = apply_gui_inputs(args)
    args.coverage_jobs = max(0, int(args.coverage_jobs))
    args.ga_jobs = max(1, int(args.ga_jobs))
    args.ga_progress_every = max(1, int(args.ga_progress_every))
    try:
        import numba

        requested_threads = max(args.ga_jobs, args.coverage_jobs if args.coverage_jobs > 0 else 0)
        if requested_threads > 0:
            numba.set_num_threads(min(requested_threads, os.cpu_count() or requested_threads))
    except Exception:
        pass

    cache_dir = Path(args.cache_dir)
    use_cache = bool(args.fast_startup)
    effective_ground_res = int(args.ground_resolution)
    if args.fast_startup:
        # Lower default test-plane resolution in fast profile unless user already requested lower.
        effective_ground_res = min(effective_ground_res, 48)
    elif args.radius >= 300.0 and int(args.ground_resolution) == 80:
        # Keep interactive viewer responsive for large radii when using default resolution.
        effective_ground_res = 48
        print(
            "Large radius detected with default ground resolution; "
            f"using resolution={effective_ground_res} for interactive rendering."
        )
    cache_context_key = _cache_key("v3_refined_roofs", args.address, args.radius, args.height, effective_ground_res)

    ground_mesh: pv.PolyData | None = None
    shadow_mask: np.ndarray | None = None
    best_positions: np.ndarray | None = None
    octree_root: OctreeNode | None = None
    vehicle_roads: pv.PolyData | None = None
    pedestrian_roads: pv.PolyData | None = None

    numba_threads = "n/a"
    try:
        import numba

        numba_threads = str(numba.get_num_threads())
    except Exception:
        pass
    _stage(
        "Run config: "
        f"mode={args.mode}, fast_startup={args.fast_startup}, "
        f"coverage_jobs={args.coverage_jobs}, ga_jobs={args.ga_jobs}, "
        f"ga_progress_every={args.ga_progress_every}, "
        f"radius={args.radius}, lights={args.n_lights}, "
        f"logical_cpu={os.cpu_count() or 1}, numba_threads={numba_threads}"
    )
    _stage(f"Loading OSM data for: {args.address}")
    t_osm = time.perf_counter()
    try:
        buildings_mesh, street_graph = _load_or_fetch_osm_cached(
            address=args.address,
            radius=args.radius,
            extrusion_height=args.height,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    except Exception as exc:
        print(f"Failed to fetch/build geometry: {exc}")
        return
    _stage(f"OSM + geometry ready in {time.perf_counter() - t_osm:.2f}s")

    print(
        "Street graph loaded: "
        f"{street_graph.number_of_nodes()} nodes, {street_graph.number_of_edges()} edges"
    )

    if not args.hide_roads:
        t_roads = time.perf_counter()
        try:
            vehicle_roads, pedestrian_roads = _street_line_layers(street_graph)
        except Exception as exc:
            print(f"Road layer extraction failed: {exc}")
        else:
            _stage(f"Road layers extracted in {time.perf_counter() - t_roads:.2f}s")

    if buildings_mesh.n_points == 0:
        print("No buildings found in this area. Try a larger radius or a denser location.")
        return

    run_shadow = args.mode in {"shadow", "all"} and (args.no_view or args.optimize_on_open or args.mode == "shadow")
    run_ga = args.mode in {"ga", "all"} and (args.no_view or args.optimize_on_open or args.mode == "ga")

    if run_shadow:
        _stage("Starting shadow stage...")
        t_shadow = time.perf_counter()
        try:
            if args.fast_startup:
                _stage("Preparing spatial cache for shadows...")
                ground_mesh, octree_root = _load_or_build_spatial_cache(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_context_key=cache_context_key,
                    buildings_mesh=buildings_mesh,
                    ground_resolution=effective_ground_res,
                )
            else:
                ground_mesh = _build_ground_mesh_for_tests(buildings_mesh, resolution=effective_ground_res)
                octree_root = _build_octree_from_buildings(buildings_mesh)
            shadow_mask, lit_ratio = compute_shadows(
                ground_mesh=ground_mesh,
                octree_root=octree_root,
                sun_dir=np.asarray(args.sun_dir, dtype=float),
            )
            shadowed = int(np.count_nonzero(shadow_mask))
            total = int(shadow_mask.size)
            print(f"Shadow test complete: {shadowed}/{total} ground triangles shadowed")
            print(f"Lit area ratio: {lit_ratio:.6f}")
            _stage(f"Shadow stage done in {time.perf_counter() - t_shadow:.2f}s")
        except Exception as exc:
            print(f"Shadow test failed: {exc}")

    if run_ga:
        _stage("Starting GA stage...")
        t_ga = time.perf_counter()
        try:
            if ground_mesh is None:
                ground_mesh = _build_ground_mesh_for_tests(buildings_mesh, resolution=effective_ground_res)
            if octree_root is None:
                octree_root = _build_octree_from_buildings(buildings_mesh)

            if args.fast_startup:
                ground_mesh, octree_root = _load_or_build_spatial_cache(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_context_key=cache_context_key,
                    buildings_mesh=buildings_mesh,
                    ground_resolution=effective_ground_res,
                )

            coverage_matrix = None
            if args.fast_startup:
                _stage("Building sidewalk polygon + candidate grid...")
                sidewalk_polygon = build_sidewalk_polygon_from_street_graph(street_graph)
                grid_points = _build_grid_points_from_ground_mesh(
                    ground_mesh,
                    args.grid_step,
                    sidewalk_polygon=sidewalk_polygon,
                )
                _stage(f"Candidate points: {grid_points.shape[0]}")
                _stage("Loading/building coverage matrix...")
                coverage_matrix = _load_or_build_coverage_matrix_cached(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_context_key=cache_context_key,
                    grid_points=grid_points,
                    ground_mesh=ground_mesh,
                    octree_root=octree_root,
                    street_graph=street_graph,
                    radius=args.light_radius,
                    pole_height=args.pole_height,
                    n_jobs=(None if args.coverage_jobs == 0 else args.coverage_jobs),
                )

            _stage("Running GA optimization...")
            ga_result = optimize_streetlights(
                ground_mesh=ground_mesh,
                n_lights=args.n_lights,
                light_radius=args.light_radius,
                w1=args.w1,
                w2=args.w2,
                grid_step=args.grid_step,
                population_size=args.population,
                generations=args.generations,
                mutation_rate=args.mutation,
                seed=args.seed,
                octree_root=octree_root,
                pole_height=args.pole_height,
                use_precomputed_coverage=bool(args.fast_startup),
                precomputed_coverage_matrix=coverage_matrix,
                street_graph=street_graph,
                ga_jobs=max(1, int(args.ga_jobs)),
                ga_progress_every=max(1, int(args.ga_progress_every)),
                ga_verbose=True,
            )
            best_positions = np.asarray(ga_result["best_positions"], dtype=float)
            print(f"GA test complete: best_cost={ga_result['best_cost']:.6f}")
            print(f"GA lit ratio: {ga_result['lit_ratio']:.6f}")
            print("Best light coordinates (x, y):")
            for row in best_positions:
                print(f"  {row[0]:.3f}, {row[1]:.3f}")
            _stage(f"GA stage done in {time.perf_counter() - t_ga:.2f}s")
        except Exception as exc:
            print(f"GA test failed: {exc}")

    if args.mode != "view" and args.mode != "all":
        return

    if args.no_view:
        print(
            "Mesh generated successfully: "
            f"{buildings_mesh.n_points} points, {buildings_mesh.n_cells} cells"
        )
        return

    _stage("Opening 3D viewer...")
    try:
        if ground_mesh is None:
            ground_mesh = _build_ground_mesh_for_tests(buildings_mesh, resolution=effective_ground_res)
        if best_positions is None:
            best_positions = _initial_light_positions(ground_mesh, args.n_lights, args.seed)

        plotter = pv.Plotter()

        style_presets: dict[str, dict[str, object]] = {
            "mini": {
                "day_bg": "#f6f1df",
                "night_bg": "#0f1728",
                "building": "#ece5cf",
                "building_edge": "#5f5a4f",
                "vehicle": "#5d6673",
                "ped": "#6aa06f",
                "day_ground": ["#efe8cf", "#8b98aa"],
                "night_ground": ["#0f1728", "#f4b942", "#fff2b2"],
                "light_day": "#7f8ea3",
                "light_night": "#ffd166",
                "disc": "#ffc75f",
            },
            "coastal": {
                "day_bg": "#eaf7f7",
                "night_bg": "#081b2c",
                "building": "#f7efe5",
                "building_edge": "#5d5c5c",
                "vehicle": "#4c6a8a",
                "ped": "#56b48a",
                "day_ground": ["#f2f7f5", "#7996b0"],
                "night_ground": ["#10223a", "#ffd16a", "#fff1c8"],
                "light_day": "#5d7fa3",
                "light_night": "#ffd57a",
                "disc": "#ffcf70",
            },
            "sunset": {
                "day_bg": "#fff4e8",
                "night_bg": "#1c1330",
                "building": "#f6e5d6",
                "building_edge": "#634f4f",
                "vehicle": "#7a5a5a",
                "ped": "#6b9a59",
                "day_ground": ["#fdebd5", "#8b7399"],
                "night_ground": ["#24153f", "#ffb86b", "#ffe6ba"],
                "light_day": "#9a7d84",
                "light_night": "#ffcd73",
                "disc": "#ffb86b",
            },
        }

        scene_state: dict[str, object] = {
            "hour": 12.0,
            "spot_radius": float(args.light_radius),
            "ground_actor": None,
            "spotlight_actor": None,
            "lights_actor": None,
            "is_night": False,
            "show_roads": not args.hide_roads,
            "preset": "mini",
            "building_actor": None,
            "vehicle_actor": None,
            "ped_actor": None,
        }

        def _style() -> dict[str, object]:
            return style_presets[str(scene_state["preset"])]

        def _set_actor_visibility(actor: object, visible: bool) -> None:
            if actor is None:
                return
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass

        def _ensure_octree() -> OctreeNode:
            nonlocal ground_mesh, octree_root
            if octree_root is None:
                t0 = time.perf_counter()
                if args.fast_startup:
                    try:
                        gm_cached, oc_cached = _load_or_build_spatial_cache(
                            cache_dir=cache_dir,
                            use_cache=use_cache,
                            cache_context_key=cache_context_key,
                            buildings_mesh=buildings_mesh,
                            ground_resolution=effective_ground_res,
                        )
                        if ground_mesh is None:
                            ground_mesh = gm_cached
                        octree_root = oc_cached
                        print("Loaded lighting geometry cache.")
                    except Exception as exc:
                        print(f"Spatial cache unavailable ({exc}). Rebuilding lighting geometry cache...")
                if octree_root is None:
                    print("Building lighting geometry cache...")
                    octree_root = _build_octree_from_buildings(buildings_mesh)
                    dt = time.perf_counter() - t0
                    print(f"Lighting geometry cache ready in {dt:.2f}s")
            return octree_root

        st = _style()
        plotter.set_background(str(st["day_bg"]))
        scene_state["building_actor"] = plotter.add_mesh(
            buildings_mesh,
            color=str(st["building"]),
            show_edges=True,
            edge_color=str(st["building_edge"]),
            opacity=0.92,
            label="Buildings",
        )

        if vehicle_roads is not None:
            scene_state["vehicle_actor"] = plotter.add_mesh(
                vehicle_roads,
                color=str(st["vehicle"]),
                line_width=3,
                opacity=0.9,
                render_lines_as_tubes=True,
            )
        if pedestrian_roads is not None:
            scene_state["ped_actor"] = plotter.add_mesh(
                pedestrian_roads,
                color=str(st["ped"]),
                line_width=2,
                opacity=0.85,
                render_lines_as_tubes=True,
            )
        _set_actor_visibility(scene_state["vehicle_actor"], bool(scene_state["show_roads"]))
        _set_actor_visibility(scene_state["ped_actor"], bool(scene_state["show_roads"]))

        scene_state["lights_actor"] = None

        # Fast initial render so the window appears immediately.
        scene_state["ground_actor"] = plotter.add_mesh(
            ground_mesh,
            color=str(st["day_ground"][0]),
            show_edges=False,
            opacity=0.72,
        )
        scene_state["spotlight_actor"] = None
        plotter.add_text("Live panel: sliders + toggles + presets", name="status", font_size=10)

        def _render_ground(hour: float, spot_radius: float) -> None:
            prev_ground = scene_state.get("ground_actor")
            prev_spot = scene_state.get("spotlight_actor")
            prev_lights = scene_state.get("lights_actor")
            if prev_ground is not None:
                plotter.remove_actor(prev_ground, reset_camera=False)
            if prev_spot is not None:
                plotter.remove_actor(prev_spot, reset_camera=False)
            if prev_lights is not None:
                plotter.remove_actor(prev_lights, reset_camera=False)

            sun_dir = _sun_dir_from_hour(hour)
            is_night = bool(sun_dir[2] <= 0.0)
            scene_state["is_night"] = is_night
            style = _style()

            b_actor = scene_state.get("building_actor")
            if b_actor is not None:
                try:
                    b_actor.GetProperty().SetColor(pv.Color(str(style["building"])).float_rgb)
                    b_actor.GetProperty().SetEdgeColor(pv.Color(str(style["building_edge"])).float_rgb)
                except Exception:
                    pass
            v_actor = scene_state.get("vehicle_actor")
            if v_actor is not None:
                try:
                    v_actor.GetProperty().SetColor(pv.Color(str(style["vehicle"])).float_rgb)
                except Exception:
                    pass
            p_actor = scene_state.get("ped_actor")
            if p_actor is not None:
                try:
                    p_actor.GetProperty().SetColor(pv.Color(str(style["ped"])).float_rgb)
                except Exception:
                    pass

            if is_night:
                night_ground = ground_mesh.copy()

                night_key = (
                    float(spot_radius),
                    int(best_positions.shape[0]),
                    hashlib.sha1(np.ascontiguousarray(best_positions, dtype=np.float64).tobytes()).hexdigest()[:16],
                )
                if "cached_night_coverage" not in scene_state or scene_state.get("cached_night_key") != night_key:
                    print(f"Pre-computing night coverage for {len(best_positions)} lights (batched)...")
                    t0 = time.perf_counter()
                    cov = build_coverage_matrix(
                        grid_points=np.asarray(best_positions, dtype=np.float64),
                        ground_mesh=ground_mesh,
                        octree_root=_ensure_octree(),
                        radius=float(spot_radius),
                        pole_height=float(args.pole_height),
                        street_graph=street_graph,
                        n_jobs=(None if args.coverage_jobs == 0 else args.coverage_jobs),
                    )
                    coverage_count = np.sum(cov, axis=0, dtype=np.int16)
                    scene_state["cached_night_coverage"] = coverage_count
                    scene_state["cached_night_key"] = night_key
                    dt = time.perf_counter() - t0
                    print(f"Night coverage cache populated in {dt:.2f}s")
                else:
                    coverage_count = np.asarray(scene_state["cached_night_coverage"], dtype=np.int16)

                illum = np.zeros(night_ground.n_cells, dtype=np.uint8)
                illum[coverage_count >= 1] = 1
                illum[coverage_count >= 2] = 2
                night_ground.cell_data["night_illum"] = illum

                lit_pct = 100.0 * float(np.count_nonzero(illum >= 1)) / float(max(1, illum.size))
                scene_state["ground_actor"] = plotter.add_mesh(
                    night_ground,
                    scalars="night_illum",
                    clim=[0, 2],
                    cmap=list(style["night_ground"]),
                    show_edges=False,
                    opacity=0.9,
                )
                plotter.set_background(str(style["night_bg"]))
                plotter.add_text(
                    f"Hour {hour:04.1f}  |  Night Mode  |  Lit {lit_pct:05.1f}%",
                    name="status",
                    font_size=10,
                )
            else:
                day_cache = scene_state.get("cached_day_shadows")
                if not isinstance(day_cache, dict):
                    day_cache = {}
                    scene_state["cached_day_shadows"] = day_cache

                hour_key = round(float(hour), 2)
                cached_day = day_cache.get(hour_key)
                if cached_day is None:
                    t0 = time.perf_counter()
                    mask, lit_ratio = compute_shadows(
                        ground_mesh=ground_mesh,
                        octree_root=_ensure_octree(),
                        sun_dir=sun_dir,
                    )
                    day_cache[hour_key] = (mask.astype(bool, copy=False), float(lit_ratio))
                    if len(day_cache) > 12:
                        oldest = next(iter(day_cache))
                        day_cache.pop(oldest, None)
                    dt = time.perf_counter() - t0
                    print(f"Computed day shadows for hour={hour_key:.2f} in {dt:.2f}s")
                else:
                    mask, lit_ratio = cached_day
                shaded = ground_mesh.copy()
                shaded.cell_data["shadow"] = mask.astype(np.uint8)
                scene_state["ground_actor"] = plotter.add_mesh(
                    shaded,
                    scalars="shadow",
                    clim=[0, 1],
                    cmap=list(style["day_ground"]),
                    show_edges=False,
                    opacity=0.72,
                )
                plotter.set_background(str(style["day_bg"]))
                plotter.add_text(f"Hour {hour:04.1f}  |  Lit Ratio {lit_ratio:.3f}", name="status", font_size=10)

            if is_night:
                light_points_local = np.column_stack(
                    (best_positions[:, 0], best_positions[:, 1], np.full(best_positions.shape[0], 0.5))
                )
                scene_state["lights_actor"] = plotter.add_points(
                    light_points_local,
                    color=str(style["light_night"]),
                    point_size=18,
                    render_points_as_spheres=True,
                )

                discs = _build_spotlight_discs(best_positions, spot_radius)
                if discs is not None:
                    scene_state["spotlight_actor"] = plotter.add_mesh(
                        discs,
                        color=str(style["disc"]),
                        opacity=0.38,
                        show_edges=False,
                    )

            _set_actor_visibility(scene_state.get("vehicle_actor"), bool(scene_state["show_roads"]))
            _set_actor_visibility(scene_state.get("ped_actor"), bool(scene_state["show_roads"]))

            plotter.render()

        def _on_time_change(value: float) -> None:
            scene_state["hour"] = float(value)
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _on_radius_change(value: float) -> None:
            scene_state["spot_radius"] = float(value)
            # Invalidate cached coverage since spotlight radius changed
            scene_state.pop("cached_night_coverage", None)
            scene_state.pop("cached_night_key", None)
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        plotter.add_slider_widget(
            _on_time_change,
            rng=[0.0, 24.0],
            value=12.0,
            title="Hour",
            pointa=(0.025, 0.10),
            pointb=(0.31, 0.10),
            style="modern",
            interaction_event="end",
        )
        plotter.add_slider_widget(
            _on_radius_change,
            rng=[max(5.0, args.light_radius * 0.4), args.light_radius * 3.0],
            value=float(args.light_radius),
            title="Spotlight Radius",
            pointa=(0.36, 0.10),
            pointb=(0.68, 0.10),
            style="modern",
            interaction_event="end",
        )

        def _toggle_roads(value: bool) -> None:
            scene_state["show_roads"] = bool(value)
            _set_actor_visibility(scene_state.get("vehicle_actor"), bool(value))
            _set_actor_visibility(scene_state.get("ped_actor"), bool(value))
            plotter.render()

        def _preset_mini(_: bool) -> None:
            scene_state["preset"] = "mini"
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_coastal(_: bool) -> None:
            scene_state["preset"] = "coastal"
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_sunset(_: bool) -> None:
            scene_state["preset"] = "sunset"
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        plotter.add_text("Controls", position=(0.025, 0.22), name="panel_title", font_size=11)
        plotter.add_text("Roads", position=(0.025, 0.18), name="panel_roads", font_size=9)
        plotter.add_checkbox_button_widget(
            _toggle_roads,
            value=bool(scene_state["show_roads"]),
            position=(20.0, 165.0),
            size=20,
            color_on="#6aa06f",
            color_off="#5d6673",
        )

        plotter.add_text("Mini", position=(0.025, 0.145), name="preset_mini_t", font_size=8)
        plotter.add_checkbox_button_widget(
            _preset_mini,
            value=True,
            position=(20.0, 135.0),
            size=16,
            color_on="#ffd166",
            color_off="#8d99ae",
        )
        plotter.add_text("Coastal", position=(0.10, 0.145), name="preset_coastal_t", font_size=8)
        plotter.add_checkbox_button_widget(
            _preset_coastal,
            value=False,
            position=(92.0, 135.0),
            size=16,
            color_on="#56b48a",
            color_off="#8d99ae",
        )
        plotter.add_text("Sunset", position=(0.20, 0.145), name="preset_sunset_t", font_size=8)
        plotter.add_checkbox_button_widget(
            _preset_sunset,
            value=False,
            position=(182.0, 135.0),
            size=16,
            color_on="#ffb86b",
            color_off="#8d99ae",
        )

        if args.optimize_on_open and best_positions is None:
            _stage("Optimize-on-open enabled: running GA before first interactive render...")
            t_open_ga = time.perf_counter()
            coverage_matrix = None
            if args.fast_startup:
                _stage("Optimize-on-open: building sidewalk polygon + candidate grid...")
                sidewalk_polygon = build_sidewalk_polygon_from_street_graph(street_graph)
                grid_points = _build_grid_points_from_ground_mesh(
                    ground_mesh,
                    args.grid_step,
                    sidewalk_polygon=sidewalk_polygon,
                )
                _stage(f"Optimize-on-open candidate points: {grid_points.shape[0]}")
                _stage("Optimize-on-open: loading/building coverage matrix...")
                coverage_matrix = _load_or_build_coverage_matrix_cached(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_context_key=cache_context_key,
                    grid_points=grid_points,
                    ground_mesh=ground_mesh,
                    octree_root=octree_root,
                    street_graph=street_graph,
                    radius=args.light_radius,
                    pole_height=args.pole_height,
                    n_jobs=(None if args.coverage_jobs == 0 else args.coverage_jobs),
                )

            ga_result = optimize_streetlights(
                ground_mesh=ground_mesh,
                n_lights=args.n_lights,
                light_radius=args.light_radius,
                w1=args.w1,
                w2=args.w2,
                grid_step=args.grid_step,
                population_size=args.population,
                generations=args.generations,
                mutation_rate=args.mutation,
                seed=args.seed,
                octree_root=octree_root,
                pole_height=args.pole_height,
                use_precomputed_coverage=bool(args.fast_startup),
                precomputed_coverage_matrix=coverage_matrix,
                street_graph=street_graph,
                ga_jobs=max(1, int(args.ga_jobs)),
                ga_progress_every=max(1, int(args.ga_progress_every)),
                ga_verbose=True,
            )
            best_positions = np.asarray(ga_result["best_positions"], dtype=float)
            scene_state.pop("cached_night_coverage", None)
            scene_state.pop("cached_night_key", None)
            _stage(f"Optimize-on-open GA finished in {time.perf_counter() - t_open_ga:.2f}s")
            # Note: Don't call _render_ground() here; let it happen after viewer opens
            # to avoid blocking on expensive coverage calculations before the window is responsive

        plotter.camera.ParallelProjectionOn()
        plotter.view_isometric()

        # Defer initial ground rendering to after the window is responsive.
        auto_render_on_open = buildings_mesh.n_cells <= 50_000
        if best_positions is not None and auto_render_on_open:
            done = {"value": False}

            def _deferred_initial_render(_: int) -> None:
                if done["value"]:
                    return
                done["value"] = True
                try:
                    _render_ground(hour=12.0, spot_radius=float(args.light_radius))
                except Exception as exc:
                    print(f"Deferred render failed: {exc}")

            plotter.add_timer_event(max_steps=1, duration=200, callback=_deferred_initial_render)
        elif not auto_render_on_open:
            plotter.add_text(
                "Large scene detected: initial lighting deferred. Move sliders to compute on demand.",
                name="status",
                font_size=10,
            )

        plotter.show()
    except Exception as exc:
        print(f"Visualization failed: {exc}")

if __name__ == "__main__":
    main()
