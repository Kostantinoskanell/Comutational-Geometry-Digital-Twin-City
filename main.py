from __future__ import annotations

import os

# macOS + conda: tell VTK/Qt to use CALayer rendering so the Cocoa event loop
# can be acquired correctly when launched from a terminal (non-GUI Python).
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

import ctypes


def _pick_numba_layer() -> str:
    """Return the Numba threading layer to use.

    The original crash ("workqueue: Concurrent access detected") was caused by
    calling Numba from a background thread while the main thread was also using
    it.  That race is now gone — warmup is synchronous and nothing else touches
    Numba before warmup completes.  ``workqueue`` is therefore safe and is
    always available without extra packages.

    TBB/OMP would give better parallel performance but require exact ABI
    alignment between the ``numba`` build and the TBB/OMP libraries in the
    active conda env, which is fragile.  Override via the environment variable
    ``NUMBA_THREADING_LAYER`` if you want to force a different layer.
    """
    return "workqueue"


# Set BEFORE numba is imported so the JIT compiler sees the chosen layer.
os.environ.setdefault("NUMBA_THREADING_LAYER", _pick_numba_layer())
print(f"[numba] threading layer: {os.environ['NUMBA_THREADING_LAYER']}")

import hashlib
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pyvista as pv

from app_cli import apply_gui_inputs, parse_args

# app_core is imported inside main() AFTER the data-source env-var is set
# so the conditional import in app_core picks up the GUI / CLI choice.
from shadow_engine import (
    _compute_shadows_numba,
    _extract_building_triangle_aabbs,
    _extract_building_triangles,
    build_coverage_matrix,
    compute_shadows,
)
from spatial_trees import OctreeNode
from streetlight_ga import (
    build_sidewalk_polygon_from_street_graph,
    optimize_streetlights,
)
from idm import IDMParams, idm_tick
from solar_physics import SolarParams
from solar_routing import (
    build_edge_costs,
    build_graph_with_costs,
    find_energy_optimal_route,
    find_joint_optimal_route,
    find_pareto_routes,
    nearest_graph_node,
)
from turn_restrictions import build_next_edges as _build_next_edges


def main() -> None:
    def _stage(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    args = parse_args()
    if args.gui:
        # Run Tkinter GUI in a subprocess to avoid NSApp conflict with VTK on macOS.
        # Tkinter acquires NSApplication then destroys it; if that happens in-process,
        # PyVista can no longer acquire the Cocoa event loop and VTK deadlocks.
        # A child process exits cleanly before PyVista ever touches Cocoa.
        import subprocess as _sp
        import json as _json
        import sys as _sys
        # Forward parent CLI args (minus --gui/--no-gui) so the GUI opens
        # pre-populated with whatever the user passed on the command line.
        _fwd = [a for a in _sys.argv[1:] if a not in ("--gui", "--no-gui")]
        _argv_setup = "import sys; sys.argv = ['gui'] + " + repr(_fwd) + "; "
        # os._exit(0) bypasses Python's normal interpreter teardown, which can
        # hang on macOS when Tkinter has touched NSApplication.
        _gui_proc = _sp.run(
            [_sys.executable, "-c",
             _argv_setup +
             "from app_cli import apply_gui_inputs, parse_args; import json, os, sys; "
             "a = apply_gui_inputs(parse_args()); "
             "sys.stdout.write(json.dumps(vars(a)) + '\\n'); sys.stdout.flush(); os._exit(0)"],
            capture_output=True, text=True,
        )
        if _gui_proc.stderr.strip():
            print(f"[gui] subprocess stderr:\n{_gui_proc.stderr[:1000]}")
        if _gui_proc.returncode != 0 and not _gui_proc.stdout.strip():
            print(f"[gui] subprocess exited with code {_gui_proc.returncode}, "
                  "continuing with CLI defaults")
        elif _gui_proc.stdout.strip():
            try:
                for _k, _v in _json.loads(_gui_proc.stdout).items():
                    setattr(args, _k, _v)
            except Exception as _e:
                print(f"[gui] could not parse subprocess output: {_e}")
    args.coverage_jobs = max(0, int(args.coverage_jobs))
    args.ga_jobs = max(1, int(args.ga_jobs))
    args.ga_progress_every = max(1, int(args.ga_progress_every))
    args.n_cars = max(0, int(args.n_cars))
    args.car_detail = str(getattr(args, "car_detail", "ultra")).strip().lower()
    if args.car_detail not in {"ultra", "low"}:
        args.car_detail = "ultra"
    args.traffic_speed = max(0.0, float(args.traffic_speed))
    args.debug_cars = bool(getattr(args, "debug_cars", False))

    # Wire data source choice to app_core's conditional import.
    # Must happen before app_core is first imported so its module-level
    # conditional runs with the correct env var.
    os.environ["CITY_DATA_SOURCE"] = getattr(args, "data_source", "overture")
    from app_core import (
        _combine_ground_surfaces,
        _build_grid_points_from_ground_mesh,
        _build_ground_mesh_for_tests,
        _build_octree_from_buildings,
        _build_spotlight_discs,
        _cache_key,
        _initial_light_positions,
        _load_or_build_coverage_matrix_cached,
        _load_or_build_spatial_cache,
        _load_or_fetch_osm_cached,
        _normalize_ground_mesh,
        _street_direction_arrows,
        _street_node_markers,
        _street_line_layers,
        _load_hdri,
        _apply_atmosphere,
        _sun_dir_from_hour,
    )

    try:
        import numba

        requested_threads = max(args.ga_jobs, args.coverage_jobs if args.coverage_jobs > 0 else 0)
        if requested_threads > 0:
            numba.set_num_threads(min(requested_threads, os.cpu_count() or requested_threads))
        # FIXED: when coverage workers will run in parallel, cap main process at 1 Numba thread
        # to prevent N_cores² oversubscription from forked workers inheriting full thread pool
        if args.coverage_jobs == 0 or args.coverage_jobs > 1:
            numba.set_num_threads(1)
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
    cache_context_key = _cache_key(
        "v9_pbr_class",  # bump: buildings now carry building_class cell data for PBR split
        getattr(args, "data_source", "overture"),
        args.address,
        args.radius,
        args.height,
        effective_ground_res,
    )

    ground_mesh: pv.PolyData | None = None
    shadow_mask: np.ndarray | None = None
    best_positions: np.ndarray | None = None
    octree_root: OctreeNode | None = None
    road_surface_mesh: pv.PolyData | None = None
    sidewalk_surface_mesh: pv.PolyData | None = None
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
        f"radius={args.radius}, lights={args.n_lights}, cars={args.n_cars}, "
        f"car_detail={args.car_detail}, traffic_speed={args.traffic_speed:.2f}, "
        f"debug_cars={args.debug_cars}, "
        f"logical_cpu={os.cpu_count() or 1}, numba_threads={numba_threads}"
    )
    _stage(f"Data source: {getattr(args, 'data_source', 'overture').upper()}")
    _stage(f"Flags: no_view={args.no_view}, optimize_on_open={args.optimize_on_open}, mode={args.mode}")
    _stage(f"Loading data for: {args.address}")
    t_osm = time.perf_counter()
    try:
        buildings_mesh, street_graph, road_surface_mesh, sidewalk_surface_mesh, places = _load_or_fetch_osm_cached(
            address=args.address,
            radius=args.radius,
            extrusion_height=args.height,
            cache_dir=cache_dir,
            use_cache=use_cache,
            data_source=getattr(args, "data_source", "overture"),
        )
    except Exception as exc:
        print(f"Failed to fetch/build geometry: {exc}")
        return
    _stage(f"OSM + geometry ready in {time.perf_counter() - t_osm:.2f}s")

    # Warm up Numba JIT synchronously so concurrent access is impossible.
    from shadow_engine import _warmup_numba_kernels
    _stage("Warming up Numba JIT (first run only)...")
    _warmup_numba_kernels()
    _stage("Numba ready")

    print(
        "Street graph loaded: "
        f"{street_graph.number_of_nodes()} nodes, {street_graph.number_of_edges()} edges"
    )


    # ── Sanitize road/sidewalk meshes: clip to scene bounding box ────────────
    # Overture road features are fetched by *intersecting* bbox, so a feature
    # whose geometry starts 90 km away and passes through the search area is
    # returned in full.  That one feature bloats the road mesh to 99 km and
    # poisons ground_mesh, the camera frustum, and shadow computation.
    # Fix: clip both meshes to buildings_mesh bounds + generous margin.
    if buildings_mesh.n_points > 0:
        _bx0, _bx1, _by0, _by1 = buildings_mesh.bounds[:4]
        _margin = max(80.0, 0.5 * max(_bx1 - _bx0, _by1 - _by0))
        _clip = [_bx0 - _margin, _bx1 + _margin,
                 _by0 - _margin, _by1 + _margin,
                 -5.0, 200.0]
        print(
            f"[clip] sanitising road meshes to "
            f"x[{_clip[0]:.0f}→{_clip[1]:.0f}] y[{_clip[2]:.0f}→{_clip[3]:.0f}]"
        )
        if road_surface_mesh is not None and road_surface_mesh.n_cells > 0:
            _before = road_surface_mesh.n_cells
            _r = road_surface_mesh.clip_box(_clip, invert=False)
            road_surface_mesh = _r if _r.n_cells > 0 else None
            _after = road_surface_mesh.n_cells if road_surface_mesh else 0
            if _before != _after:
                print(f"[clip] road cells {_before} → {_after}  (removed {_before - _after} outlier cells)")
        if sidewalk_surface_mesh is not None and sidewalk_surface_mesh.n_cells > 0:
            _before = sidewalk_surface_mesh.n_cells
            _s = sidewalk_surface_mesh.clip_box(_clip, invert=False)
            sidewalk_surface_mesh = _s if _s.n_cells > 0 else None
            _after = sidewalk_surface_mesh.n_cells if sidewalk_surface_mesh else 0
            if _before != _after:
                print(f"[clip] sidewalk cells {_before} → {_after}")
    # ─────────────────────────────────────────────────────────────────────────

    combined_ground = _combine_ground_surfaces(road_surface_mesh, sidewalk_surface_mesh)
    if combined_ground is not None and combined_ground.n_cells > 0:
        ground_mesh = _normalize_ground_mesh(combined_ground)

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
                    preferred_ground_mesh=ground_mesh,
                )
            else:
                if ground_mesh is None:
                    ground_mesh = _normalize_ground_mesh(
                        _build_ground_mesh_for_tests(buildings_mesh, resolution=effective_ground_res)
                    )
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
                ground_mesh = _normalize_ground_mesh(
                    _build_ground_mesh_for_tests(buildings_mesh, resolution=effective_ground_res)
                )
            if octree_root is None:
                octree_root = _build_octree_from_buildings(buildings_mesh)

            if args.fast_startup:
                ground_mesh, octree_root = _load_or_build_spatial_cache(
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_context_key=cache_context_key,
                    buildings_mesh=buildings_mesh,
                    ground_resolution=effective_ground_res,
                    preferred_ground_mesh=ground_mesh,
                )

            coverage_matrix = None
            sidewalk_polygon = None
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
                sidewalk_polygon=sidewalk_polygon,
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
        ground_mesh = _normalize_ground_mesh(ground_mesh)
        if best_positions is None:
            best_positions = _initial_light_positions(ground_mesh, args.n_lights, args.seed)

        plotter = pv.Plotter(title="City Digital Twin", window_size=[1400, 900])
        _stage("Plotter created")

        style_presets: dict[str, dict[str, object]] = {
            "mini": {
                "day_bg": "#0f1117",
                "night_bg": "#070a12",
                "building": "#1e2d45",
                "building_edge": "#2a4a6b",
                "vehicle": "#00bcd4",
                "ped": "#69f0ae",
                "car": "#ff6b6b",
                "day_ground": ["#0d1117", "#1a2744"],
                "night_ground": ["#070a12", "#f4b942", "#fff2b2"],
                "light_day": "#37474f",
                "light_night": "#ffd54f",
                "disc": "#ffca28",
            },
            "coastal": {
                "day_bg": "#0a1628",
                "night_bg": "#040d1a",
                "building": "#1a3a5c",
                "building_edge": "#2e5f8a",
                "vehicle": "#4dd0e1",
                "ped": "#80cbc4",
                "car": "#ff8a65",
                "day_ground": ["#0d1f33", "#1a3a5c"],
                "night_ground": ["#050e1a", "#ffd16a", "#fff1c8"],
                "light_day": "#2a4a6b",
                "light_night": "#ffe57f",
                "disc": "#ffd740",
            },
            "sunset": {
                "day_bg": "#1a0f1e",
                "night_bg": "#0d0710",
                "building": "#3d1c5a",
                "building_edge": "#6a2d8a",
                "vehicle": "#ce93d8",
                "ped": "#a5d6a7",
                "car": "#ffd54f",
                "day_ground": ["#1a0f1e", "#3d1c5a"],
                "night_ground": ["#0d0710", "#ffb86b", "#ffe6ba"],
                "light_day": "#6a2d8a",
                "light_night": "#ffd740",
                "disc": "#ffab40",
            },
        }

        scene_state: dict[str, object] = {
            "hour": 12.0,
            "spot_radius": float(args.light_radius),
            "ground_actor": None,
            "spotlight_actor": None,
            "lights_actor": None,
            "street_arrows_actor": None,
            "is_night": False,
            "show_roads": not args.hide_roads,
            "show_cars": True,
            "show_arrows": True,
            "preset": "mini",
            "building_actor": None,
            "vehicle_actor": None,
            "ped_actor": None,
            "car_actors": {},
            "interactive_ready": False,
            "poi_actor": None,
            "poi_labels_actor": None,
            "show_pois": True,
            "show_poi_names": True,
            "scene_lat": float(street_graph.graph.get("scene_lat", np.nan)),
            "scene_lon": float(street_graph.graph.get("scene_lon", np.nan)),
            "route_alpha": 0.5,
            "route_hour": 12.0,
            "solar_params": SolarParams(),
            "edge_costs_cache": {},
            "pareto_chart": None,
            "_ultra_car_actors": None,
            "tl_actor": None,
        }
        scene_state["route_hour"] = float(scene_state["hour"])

        def _style() -> dict[str, object]:
            return style_presets[str(scene_state["preset"])]

        def _set_actor_visibility(actor: object, visible: bool) -> None:
            if actor is None:
                return
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass

        def _parse_maxspeed(raw) -> float:
            """Convert a raw maxspeed edge attribute (str, int, float, or None) to m/s.
            Handles: None → 13.9 m/s (50 km/h default), numeric → km/h÷3.6,
            "50" → 50 km/h, "30 mph" → mph×1.60934÷3.6, "50 km/h" → km/h÷3.6,
            lists/mixed → use first numeric value found."""
            if raw is None:
                return 13.9
            if isinstance(raw, (int, float)):
                return float(raw) / 3.6
            if isinstance(raw, list):
                raw = raw[0] if raw else None
                if raw is None:
                    return 13.9
            s = str(raw).strip().lower()
            try:
                return float(s) / 3.6
            except ValueError:
                pass
            import re as _re
            m = _re.search(r"(\d+(?:\.\d+)?)\s*(mph|km/h)?", s)
            if m:
                val = float(m.group(1))
                unit = m.group(2) or "km/h"
                if unit == "mph":
                    return val * 1.60934 / 3.6
                return val / 3.6
            return 13.9

        def _extract_drivable_paths(graph, z_level: float = 0.38) -> tuple[list[dict[str, object]], dict[object, list[int]]]:
            """Extract directed drivable polylines from graph edges."""
            ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}
            paths: list[dict[str, object]] = []
            outgoing: dict[object, list[int]] = {}

            for u, v, data in graph.edges(data=True):
                hw = data.get("highway")
                if isinstance(hw, (list, tuple, set)):
                    hw_vals = {str(x).strip().lower() for x in hw if str(x).strip()}
                elif hw is None:
                    hw_vals = set()
                else:
                    tag = str(hw).strip().lower()
                    hw_vals = {tag} if tag else set()

                if hw_vals & ped_tags:
                    continue

                geom = data.get("geometry")
                if geom is not None and hasattr(geom, "coords"):
                    coords = np.asarray(geom.coords, dtype=float)
                    if coords.shape[0] < 2:
                        continue
                    xy = np.asarray(coords[:, :2], dtype=float)
                else:
                    nu = graph.nodes.get(u, {})
                    nv = graph.nodes.get(v, {})
                    if "x" not in nu or "y" not in nu or "x" not in nv or "y" not in nv:
                        continue
                    xy = np.array(
                        [
                            [float(nu["x"]), float(nu["y"])],
                            [float(nv["x"]), float(nv["y"])],
                        ],
                        dtype=float,
                    )

                nu = graph.nodes.get(u, {})
                nv = graph.nodes.get(v, {})
                if "x" in nu and "y" in nu and "x" in nv and "y" in nv:
                    u_xy = np.array([float(nu["x"]), float(nu["y"])], dtype=float)
                    v_xy = np.array([float(nv["x"]), float(nv["y"])], dtype=float)
                    forward_err = float(np.linalg.norm(xy[0] - u_xy) + np.linalg.norm(xy[-1] - v_xy))
                    reverse_err = float(np.linalg.norm(xy[0] - v_xy) + np.linalg.norm(xy[-1] - u_xy))
                    if reverse_err + 1e-6 < forward_err:
                        xy = xy[::-1]

                if xy.shape[0] < 2:
                    continue

                keep = np.ones((xy.shape[0],), dtype=bool)
                keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > 1e-6
                xy = xy[keep]
                if xy.shape[0] < 2:
                    continue

                points = np.column_stack((xy, np.full((xy.shape[0],), float(z_level), dtype=float)))
                seg_lengths = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
                valid = seg_lengths > 1e-6
                if not np.any(valid):
                    continue

                if not np.all(valid):
                    idx_keep = [0]
                    idx_keep.extend(int(i + 1) for i, ok in enumerate(valid) if ok)
                    points = points[np.unique(np.asarray(idx_keep, dtype=np.int64))]
                    if points.shape[0] < 2:
                        continue
                    seg_lengths = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)

                cum_len = np.concatenate(([0.0], np.cumsum(seg_lengths, dtype=float)))
                total_len = float(cum_len[-1])
                if total_len <= 1e-6:
                    continue

                idx = len(paths)
                paths.append(
                    {
                        "u": u,
                        "v": v,
                        "points": points,
                        "cum_len": cum_len,
                        "length": total_len,
                        "maxspeed_ms": float(np.clip(
                            _parse_maxspeed(data.get("maxspeed")),
                            2.8, 41.7,
                        )),
                        "segment_id": data.get("segment_id"),  # ← ADD THIS LINE ONLY
                    }
                )
                outgoing.setdefault(u, []).append(idx)

            return paths, outgoing

        def _car_pose_on_path(path: dict[str, object], dist_m: float) -> np.ndarray:
            points = np.asarray(path["points"], dtype=float)
            cum = np.asarray(path["cum_len"], dtype=float)
            total = float(path["length"])
            d = float(np.clip(dist_m, 0.0, max(total - 1e-9, 0.0)))

            seg_idx = int(np.searchsorted(cum, d, side="right") - 1)
            seg_idx = int(np.clip(seg_idx, 0, points.shape[0] - 2))

            l0 = float(cum[seg_idx])
            l1 = float(cum[seg_idx + 1])
            denom = max(1e-9, l1 - l0)
            t = (d - l0) / denom
            p0 = points[seg_idx]
            p1 = points[seg_idx + 1]
            return p0 + (p1 - p0) * t

        def _car_heading_deg_on_path(path: dict[str, object], dist_m: float) -> float:
            """Return heading angle in degrees (around Z) for the car's direction of travel."""
            points = np.asarray(path["points"], dtype=float)
            cum = np.asarray(path["cum_len"], dtype=float)
            total = float(path["length"])
            d = float(np.clip(dist_m, 0.0, max(total - 1e-9, 0.0)))
            seg_idx = int(np.searchsorted(cum, d, side="right") - 1)
            seg_idx = int(np.clip(seg_idx, 0, points.shape[0] - 2))
            p0, p1 = points[seg_idx], points[seg_idx + 1]
            heading = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
            return (heading + 180.0) % 360.0  # ← FIX: Add 180° for OBJ model orientation

        car_paths, car_outgoing = _extract_drivable_paths(street_graph, z_level=0.5)
        _stage(f"Car paths extracted: {len(car_paths)} drivable edges")
        car_path_lengths = np.asarray([float(p["length"]) for p in car_paths], dtype=float)
        edge_shadow_frac = np.zeros((len(car_paths),), dtype=float)
        car_shape_low = pv.Cube(center=(0.0, 0.0, 0.0), x_length=2.2, y_length=1.1, z_length=0.5)
        # Build car_next_edges with turn restrictions (Overture prohibited_transitions
        # + OSM 'restriction' edge attrs) applied on top of the non-reverse filter.
        car_next_edges = _build_next_edges(car_paths, car_outgoing, street_graph)
        _stage("Car routing pre-computed (turn restrictions applied)")

        # ── Load OBJ car models for ultra detail mode ────────────────────────
        _CAR_MODELS_DIR = Path(__file__).parent / "assets" / "models"
        _CAR_OBJ_FILES = [
            "sedan-sports.obj", "hatchback-sports.obj", "suv.obj",
            "van.obj", "delivery.obj", "truck-flat.obj",
        ]
        # Per-car body colors (varied fleet look)
        _CAR_BODY_COLORS = [
            "#c0392b", "#2980b9", "#27ae60", "#f39c12",
            "#8e44ad", "#e74c3c", "#3498db", "#16a085",
            "#d35400", "#2c3e50", "#1abc9c", "#e67e22",
        ]
        car_obj_templates: list[pv.PolyData] = []
        car_obj_lengths: list[float] = []  # bumper-to-bumper length per template (metres)
        if args.car_detail == "ultra":
            for _fname in _CAR_OBJ_FILES:
                _fpath = _CAR_MODELS_DIR / _fname
                if not _fpath.exists():
                    continue
                try:
                    _raw = pv.read(str(_fpath))
                    # Kenney OBJs may load as MultiBlock (one block per named group)
                    if not isinstance(_raw, pv.PolyData):
                        _raw = _raw.extract_geometry()
                    # Kenney models: Y-up, car nose faces -Z.
                    # Target: Z-up (VTK), car nose faces +X.
                    # rotate_x(90°): Y→Z (height up)
                    # rotate_z(-90°): former-Y (length) → +X
                    _raw = _raw.rotate_x(90.0)
                    _raw = _raw.rotate_z(-90.0)
                    # Scale to realistic metric size (~2 m wide, ~4 m long)
                    _raw.points = _raw.points * 1.6
                    # Length = X-extent after transform (car nose faces +X)
                    _car_len = float(_raw.bounds[1] - _raw.bounds[0])
                    car_obj_templates.append(_raw)
                    car_obj_lengths.append(_car_len)
                    print(f"[cars] loaded OBJ model: {_fname} ({_raw.n_points} pts, length={_car_len:.2f} m)")
                except Exception as _exc:
                    print(f"[cars] failed to load {_fname}: {_exc}")
            if not car_obj_templates:
                print("[cars] no OBJ models found in assets/models; falling back to sphere mode")
        # ────────────────────────────────────────────────────────────────────

        # ── Build traffic lights ─────────────────────────────────────────────
        from traffic_lights import build_traffic_lights, tick_all, build_light_mesh, update_light_mesh, build_light_glyphs
        traffic_lights_dict = build_traffic_lights(
            street_graph,
            car_paths,
            min_degree=3,
            traffic_speed=float(args.traffic_speed),
        )
        scene_state["traffic_lights"] = traffic_lights_dict
        # ────────────────────────────────────────────────────────────────────

        # ── IDM (Intelligent Driver Model) parameters ────────────────────────
        # Acceleration model: a = A*(1 - (v/v0)^DELTA - (s*/s)^2)
        # s* = S0 + max(0, v*T + v*dv / (2*sqrt(A*B)))
        _idm_params = IDMParams(
            a_max = 1.5,   # m/s²  max acceleration
            b     = 2.5,   # m/s²  comfortable deceleration
            T     = 1.5,   # s     desired time gap
            s0    = 2.0,   # m     minimum jam gap
            delta = 4,     # acceleration exponent
        )
        # ────────────────────────────────────────────────────────────────────

        if bool(args.debug_cars):
            total_len = float(np.sum(car_path_lengths)) if car_path_lengths.size > 0 else 0.0
            mean_len = total_len / float(max(1, len(car_paths)))
            print(
                "[cars-debug] graph summary: "
                f"drivable_paths={len(car_paths)}, total_path_len={total_len:.1f}m, mean_path_len={mean_len:.1f}m"
            )
            print(
                "[cars-debug] settings: "
                f"requested_cars={args.n_cars}, detail={args.car_detail}, traffic_speed={args.traffic_speed:.2f}"
            )

        car_rng = np.random.default_rng(int(args.seed) + 2027)
        car_anim: dict[str, object] = {
            "enabled": bool(args.n_cars > 0 and len(car_paths) > 0),
            "edge_idx": np.empty((0,), dtype=np.int64),
            "dist": np.empty((0,), dtype=float),
            "speed": np.empty((0,), dtype=float),
            "last_t": time.perf_counter(),
        }
        car_debug_state: dict[str, float | int] = {
            "tick": 0,
            "next_log_t": time.perf_counter() + 1.5,
        }
        if bool(car_anim["enabled"]):
            n_cars = max(0, int(args.n_cars))
            if np.sum(car_path_lengths) > 1e-9:
                probs = car_path_lengths / np.sum(car_path_lengths)
            else:
                probs = None
            edge_idx = car_rng.choice(len(car_paths), size=n_cars, replace=True, p=probs)
            dist = np.array(
                [car_rng.uniform(0.0, float(car_path_lengths[int(i)])) for i in edge_idx],
                dtype=float,
            )
            desired_speed_base = np.array(
                [
                    car_paths[int(i)]["maxspeed_ms"] * car_rng.uniform(0.7, 1.0)
                    for i in edge_idx
                ],
                dtype=float,
            )
            speed = desired_speed_base * float(args.traffic_speed)
            car_anim["edge_idx"] = np.asarray(edge_idx, dtype=np.int64)
            car_anim["dist"] = dist
            car_anim["speed"] = speed.copy()           # actual instantaneous velocity (m/s)
            car_anim["desired_speed"] = speed.copy()   # free-flow target v_0 per car (m/s)
            car_anim["desired_speed_base"] = desired_speed_base.copy()  # per-car base v_0 (m/s)
            car_anim["accel"] = np.zeros(n_cars, dtype=float)  # current IDM acceleration
            # Assign a fixed OBJ model index per car (for ultra mode)
            if car_obj_templates:
                car_anim["model_idx"] = car_rng.integers(0, len(car_obj_templates), size=n_cars)
                # Per-car bumper length from the assigned OBJ template
                car_anim["car_len"] = np.array(
                    [car_obj_lengths[int(m) % len(car_obj_lengths)]
                     for m in car_anim["model_idx"]],
                    dtype=float,
                )
            else:
                car_anim["model_idx"] = np.zeros(n_cars, dtype=np.int64)
                car_anim["car_len"] = np.full(n_cars, 4.0, dtype=float)  # default 4 m
            if bool(args.debug_cars):
                print(
                    "[cars-debug] initialized: "
                    f"active_cars={n_cars}, speed_mean={float(np.mean(speed)):.2f}m/s, speed_max={float(np.max(speed)):.2f}m/s"
                )
        elif bool(args.debug_cars):
            if int(args.n_cars) <= 0:
                print("[cars-debug] disabled: n-cars <= 0")
            elif len(car_paths) == 0:
                print("[cars-debug] disabled: no drivable road paths found")

        def _log_car_debug(tag: str) -> None:
            if not bool(args.debug_cars):
                return
            if not bool(car_anim["enabled"]):
                print(f"[cars-debug] {tag}: cars are disabled")
                return
            edge_idx = np.asarray(car_anim["edge_idx"], dtype=np.int64)
            dist = np.asarray(car_anim["dist"], dtype=float)
            speed = np.asarray(car_anim["speed"], dtype=float)
            n = int(edge_idx.shape[0])
            if n == 0:
                print(f"[cars-debug] {tag}: no active cars")
                return
            safe_len = np.maximum(car_path_lengths[edge_idx], 1e-6)
            mean_progress = float(np.mean(dist / safe_len))
            unique_edges = int(np.unique(edge_idx).size)
            first_path = int(edge_idx[0])
            first_pos = _car_pose_on_path(car_paths[first_path], float(dist[0]))
            print(
                "[cars-debug] "
                f"{tag}: n={n}, unique_edges={unique_edges}, "
                f"mean_speed={float(np.mean(speed)):.2f}m/s, mean_progress={mean_progress:.2f}, "
                f"sample_xy=({first_pos[0]:.2f},{first_pos[1]:.2f})"
            )

        def _advance_cars(dt: float) -> None:
            """Delegate to the standalone IDM module."""
            idm_tick(
                car_anim       = car_anim,
                car_paths      = car_paths,
                car_next_edges = car_next_edges,
                traffic_lights = traffic_lights_dict,
                dt             = dt,
                params         = _idm_params,
                rng            = car_rng,
                traffic_speed  = float(args.traffic_speed),
            )

        def _log_idm_diagnostics(tag: str) -> None:
            """Periodic IDM health check — active only when --debug-cars is set."""
            if not bool(args.debug_cars):
                return
            if not bool(car_anim["enabled"]):
                return

            spd   = np.asarray(car_anim["speed"],         dtype=float)
            des   = np.asarray(car_anim["desired_speed"],  dtype=float)
            acc   = np.asarray(car_anim["accel"],          dtype=float)
            eidx  = np.asarray(car_anim["edge_idx"],       dtype=np.int64)
            dist  = np.asarray(car_anim["dist"],           dtype=float)
            clen  = np.asarray(car_anim["car_len"],        dtype=float)
            n     = int(spd.shape[0])

            # ── (1) Negative speed — symplectic clip should prevent this ──────
            neg_mask = spd < -0.001
            if np.any(neg_mask):
                print(f"[IDM-WARN  ] {tag}: NEGATIVE SPEED in {int(np.sum(neg_mask))} cars "
                      f"| min={float(np.min(spd)):.4f} m/s  → clip failure")

            # ── (2) Overspeed — should never exceed desired ───────────────────
            over_mask = spd > des + 0.05
            if np.any(over_mask):
                print(f"[IDM-WARN  ] {tag}: OVERSPEED in {int(np.sum(over_mask))} cars "
                      f"| max_excess={float(np.max(spd - des)):.3f} m/s")

            # ── (3) Inter-car gap < 0 (phasing through) ──────────────────────
            from idm import build_edge_car_map as _ecm_fn
            ecm = _ecm_fn(eidx, dist)
            min_gap = float("inf")
            collision_pairs: list[str] = []
            for e, bucket in ecm.items():
                for k in range(len(bucket) - 1):
                    d_follower, fi = bucket[k]
                    d_leader,   li = bucket[k + 1]
                    g = d_leader - d_follower - float(clen[fi])
                    if g < min_gap:
                        min_gap = g
                    if g < 0.0:
                        collision_pairs.append(
                            f"edge={e} cars=({fi},{li}) gap={g:.2f}m"
                        )
            if collision_pairs:
                print(f"[IDM-WARN  ] {tag}: PHASING THROUGH — {len(collision_pairs)} pair(s): "
                      + "; ".join(collision_pairs[:3]))
            elif min_gap < _idm_params.s0:
                print(f"[IDM-WARN  ] {tag}: min gap={min_gap:.2f} m < s0={_idm_params.s0} m "
                      "(tight but no collision)")

            # ── (4) Gridlock — all cars stopped ──────────────────────────────
            n_stopped = int(np.sum(spd < 0.15))
            pct_stopped = 100.0 * n_stopped / max(n, 1)
            if pct_stopped >= 90.0:
                print(f"[IDM-WARN  ] {tag}: GRIDLOCK — {n_stopped}/{n} cars stopped "
                      f"({pct_stopped:.0f}%)")

            # ── (5) Oscillation — high accel variance on a single edge ────────
            worst_var = 0.0
            for e, bucket in ecm.items():
                if len(bucket) < 3:
                    continue
                idxs = [ci for _, ci in bucket]
                var = float(np.var(acc[idxs]))
                if var > worst_var:
                    worst_var = var
            if worst_var > (_idm_params.a_max * 2.5) ** 2:
                print(f"[IDM-WARN  ] {tag}: OSCILLATION — accel variance={worst_var:.2f} "
                      f"on busiest edge (threshold={((_idm_params.a_max * 2.5)**2):.2f})")

            # ── (6) Summary stats (always printed when debug_cars active) ─────
            n_free   = int(np.sum(spd > des * 0.85))   # essentially free-flow
            n_queued = int(np.sum(spd < des * 0.30))   # queued / near-stopped
            print(
                f"[IDM-diag  ] {tag}: n={n} | "
                f"spd mean={float(np.mean(spd)):.2f} max={float(np.max(spd)):.2f} "
                f"min={float(np.min(spd)):.2f} m/s | "
                f"stopped={n_stopped} queued={n_queued} free={n_free} | "
                f"acc mean={float(np.mean(acc)):+.3f} "
                f"max={float(np.max(acc)):+.3f} "
                f"min={float(np.min(acc)):+.3f} m/s² | "
                f"min_gap={min_gap:.1f} m"
            )

        def _sample_car_positions() -> np.ndarray:
            edge_idx = np.asarray(car_anim["edge_idx"], dtype=np.int64)
            dist = np.asarray(car_anim["dist"], dtype=float)
            n = edge_idx.shape[0]
            positions = np.zeros((n, 3), dtype=float)
            for i in range(n):
                positions[i] = _car_pose_on_path(car_paths[int(edge_idx[i])], float(dist[i]))
            return positions

        def _sample_car_headings() -> np.ndarray:
            edge_idx = np.asarray(car_anim["edge_idx"], dtype=np.int64)
            dist = np.asarray(car_anim["dist"], dtype=float)
            n = edge_idx.shape[0]
            headings = np.zeros(n, dtype=float)
            for i in range(n):
                headings[i] = _car_heading_deg_on_path(car_paths[int(edge_idx[i])], float(dist[i]))
            return headings

        def _render_cars() -> None:
            actors = scene_state.get("car_actors")
            if not isinstance(actors, dict):
                actors = {}
                scene_state["car_actors"] = actors

            is_first = actors.get("cars") is None and not scene_state.get("_ultra_car_actors")
            if is_first:
                print("[cars] _render_cars: FIRST CALL — creating actor")

            if not bool(car_anim["enabled"]):
                for name, actor in list(actors.items()):
                    if actor is not None:
                        plotter.remove_actor(actor, reset_camera=False)
                    actors.pop(name, None)
                scene_state.pop("_car_mesh", None)
                for _ua in scene_state.get("_ultra_car_actors") or []:
                    try:
                        plotter.remove_actor(_ua, reset_camera=False)
                    except Exception:
                        pass
                scene_state["_ultra_car_actors"] = None
                return

            positions = _sample_car_positions()
            if positions.shape[0] == 0:
                return

            # ── Ultra mode: per-car OBJ mesh actors ─────────────────────────
            if args.car_detail == "ultra" and car_obj_templates:
                headings = _sample_car_headings()
                ultra_actors = scene_state.get("_ultra_car_actors")

                if not ultra_actors:
                    # First call — create one actor per car from its assigned template
                    n = positions.shape[0]
                    model_idx = np.asarray(car_anim.get("model_idx", np.zeros(n, dtype=np.int64)), dtype=np.int64)
                    ultra_actors = []
                    for i in range(n):
                        tmpl = car_obj_templates[int(model_idx[i]) % len(car_obj_templates)]
                        color = _CAR_BODY_COLORS[i % len(_CAR_BODY_COLORS)]
                        try:
                            _actor = plotter.add_mesh(
                                tmpl.copy(),
                                color=color,
                                smooth_shading=True,
                                lighting=True,
                                reset_camera=False,
                            )
                            _actor.SetPosition(
                                float(positions[i, 0]),
                                float(positions[i, 1]),
                                float(positions[i, 2]),
                            )
                            _actor.SetOrientation(0.0, 0.0, float(headings[i]))
                            ultra_actors.append(_actor)
                        except Exception as _exc:
                            print(f"[cars] OBJ actor {i} creation failed: {_exc}")
                    scene_state["_ultra_car_actors"] = ultra_actors
                    print(f"[cars] created {len(ultra_actors)} OBJ car actors")
                else:
                    # Subsequent calls — update transform only (no add_mesh)
                    for i, _actor in enumerate(ultra_actors):
                        if i >= positions.shape[0]:
                            break
                        _actor.SetPosition(
                            float(positions[i, 0]),
                            float(positions[i, 1]),
                            float(positions[i, 2]),
                        )
                        _actor.SetOrientation(0.0, 0.0, float(headings[i]))

                show = bool(scene_state["show_cars"])
                for _actor in ultra_actors:
                    _set_actor_visibility(_actor, show)
                return
            # ── Fallback: sphere point-cloud actor ───────────────────────────

            point_size = 28 if args.car_detail == "low" else 22

            if actors.get("cars") is None:
                # First call — create the mesh and actor once.
                car_mesh = pv.PolyData(np.asarray(positions, dtype=float))
                scene_state["_car_mesh"] = car_mesh
                try:
                    car_actor = plotter.add_mesh(
                        car_mesh,
                        name="cars",
                        style="points",
                        point_size=point_size,
                        color=str(_style()["car"]),
                        render_points_as_spheres=True,
                        lighting=True,
                        reset_camera=False,
                    )
                    print("[cars] _render_cars: plotter.add_mesh() succeeded")
                except Exception as _exc:
                    import traceback as _tb
                    print(f"[cars] _render_cars: plotter.add_mesh() FAILED: {_exc}")
                    _tb.print_exc()
                    return
                actors["cars"] = car_actor
            else:
                # Subsequent calls — update points IN PLACE, never call add_mesh again.
                car_mesh = scene_state.get("_car_mesh")
                if car_mesh is not None:
                    car_mesh.points = np.asarray(positions, dtype=float)

            car_actor = actors.get("cars")
            if car_actor is not None:
                try:
                    car_actor.GetProperty().SetColor(
                        pv.Color(str(_style()["car"])).float_rgb
                    )
                except Exception:
                    pass
                _set_actor_visibility(car_actor, bool(scene_state["show_cars"]))

        day_surface_cmap = [
            "#2f343b",  # road shadow
            "#4a5058",  # road lit
            "#5d636b",  # sidewalk shadow
            "#929aa4",  # sidewalk lit
        ]
        night_surface_cmap = [
            "#22272e",  # road dark
            "#5a616b",  # road single-lit
            "#ffd166",  # road double-lit
            "#3c434c",  # sidewalk dark
            "#8f97a2",  # sidewalk single-lit
            "#ffe4a3",  # sidewalk double-lit
        ]

        def _surface_kind(mesh: pv.PolyData) -> np.ndarray:
            raw = mesh.cell_data.get("surface_kind")
            if raw is None:
                return np.zeros((mesh.n_cells,), dtype=np.uint8)
            kinds = np.asarray(raw, dtype=np.uint8).reshape(-1)
            if kinds.shape[0] != mesh.n_cells:
                return np.zeros((mesh.n_cells,), dtype=np.uint8)
            return kinds

        def _compose_day_surface_classes(shadow_mask_local: np.ndarray) -> np.ndarray:
            mask = np.asarray(shadow_mask_local, dtype=bool).reshape(-1)
            kinds = _surface_kind(ground_mesh)
            if mask.shape[0] != kinds.shape[0]:
                mask = np.zeros((kinds.shape[0],), dtype=bool)

            road = kinds == 0
            sidewalk = ~road
            classes = np.zeros((kinds.shape[0],), dtype=np.uint8)
            classes[road & ~mask] = 1
            classes[sidewalk & mask] = 2
            classes[sidewalk & ~mask] = 3
            return classes

        def _compose_night_surface_classes(illum: np.ndarray) -> np.ndarray:
            coverage = np.asarray(illum, dtype=np.uint8).reshape(-1)
            kinds = _surface_kind(ground_mesh)
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
                            preferred_ground_mesh=ground_mesh,
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

        def _build_edge_shadow_cache(hour: float) -> np.ndarray:
            """Return per-edge shadow fraction at current hour (shape=(len(car_paths),))."""
            n_paths = len(car_paths)
            if n_paths == 0:
                return np.empty((0,), dtype=float)

            hour_key = round(float(hour), 1)
            cache = scene_state.get("edge_shadow_cache")
            if not isinstance(cache, dict):
                cache = {}
                scene_state["edge_shadow_cache"] = cache
            cached = cache.get(hour_key)
            if cached is not None:
                return np.asarray(cached, dtype=float)

            sun_dir = np.asarray(_sun_dir_from_hour(float(hour)), dtype=np.float64)
            if float(sun_dir[2]) <= 0.0:
                out_night = np.ones((n_paths,), dtype=float)
                cache[hour_key] = out_night
                return out_night

            direction = sun_dir / max(1e-12, float(np.linalg.norm(sun_dir)))
            root = _ensure_octree()
            building_triangles = _extract_building_triangles(root)
            tri_mins, tri_maxs = _extract_building_triangle_aabbs(root, building_triangles)

            n_samples = 5
            centroids = np.empty((n_paths * n_samples, 3), dtype=np.float64)
            for pi, path in enumerate(car_paths):
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
            shadowed, _ = _compute_shadows_numba(
                centroids,
                areas,
                direction.astype(np.float64),
                building_triangles,
                tri_mins,
                tri_maxs,
            )
            out = np.mean(shadowed.reshape(n_paths, n_samples), axis=1, dtype=np.float64)
            cache[hour_key] = out
            return out

        st = _style()
        plotter.set_background(str(st["day_bg"]))
        _stage(f"Adding buildings mesh ({buildings_mesh.n_cells} cells)...")

        # PBR environment is refreshed in _render_ground() for time-of-day HDRI.
        renderer = plotter.renderer
        try:
            renderer.SetUseSSAO(True)
            # Radius 4.0 is calibrated for this scene's metric scale (~150 m extent).
            renderer.SetSSAORadius(4.0)
            renderer.SetSSAOBias(0.025)
            renderer.SetSSAOKernelSize(128)
            renderer.SetSSAOBlur(True)
            print("[ssao] SSAO enabled")
        except Exception:
            print("[ssao] SSAO unavailable")

        _PBR_CLASSES = {
            0: dict(name="concrete", color="#c8b89a", metallic=0.0, roughness=0.85),
            1: dict(name="brick",    color="#b5724a", metallic=0.0, roughness=0.90),
            2: dict(name="glass",    color="#8cb8d8", metallic=0.0, roughness=0.08),
        }
        _pbr_actors: dict[str, object] = {}

        if "building_class" in (buildings_mesh.cell_data.keys() if buildings_mesh.n_cells > 0 else []):
            _bclass_arr = np.asarray(buildings_mesh.cell_data["building_class"])
            for _cid, _cp in _PBR_CLASSES.items():
                _idx = np.where(_bclass_arr == _cid)[0]
                if _idx.size == 0:
                    continue
                _cmesh = buildings_mesh.extract_cells(_idx).extract_surface()
                try:
                    _cmesh = _cmesh.compute_normals(
                        cell_normals=False, point_normals=True,
                        split_vertices=True, auto_orient_normals=True,
                    )
                except Exception:
                    pass
                _actor = plotter.add_mesh(
                    _cmesh,
                    pbr=True,
                    metallic=_cp["metallic"],
                    roughness=_cp["roughness"],
                    color=_cp["color"],
                    smooth_shading=True,
                    show_edges=True,
                    edge_color=str(st["building_edge"]),
                    line_width=0.4,
                    opacity=1.0,
                )
                _pbr_actors[_cp["name"]] = _actor
                print(f"[pbr] {_cp['name']}: {_idx.size} cells")
            # First actor is the backward-compat handle used by _render_ground
            scene_state["building_actor"] = next(iter(_pbr_actors.values()), None)
        else:
            # Fallback: old cache without building_class — single flat-shaded actor
            print("[pbr] building_class not found in mesh; using flat shading (re-run to rebuild cache)")
            scene_state["building_actor"] = plotter.add_mesh(
                buildings_mesh,
                color=str(st["building"]),
                show_edges=True,
                edge_color=str(st["building_edge"]),
                line_width=0.5,
                opacity=0.95,
                smooth_shading=True,
            )

        scene_state["_building_actors_pbr"] = _pbr_actors

        if vehicle_roads is not None:
            scene_state["vehicle_actor"] = plotter.add_mesh(
                vehicle_roads,
                color=str(st["vehicle"]),
                line_width=4,
                opacity=1.0,
                render_lines_as_tubes=True,
            )
        if pedestrian_roads is not None:
            scene_state["ped_actor"] = plotter.add_mesh(
                pedestrian_roads,
                color=str(st["ped"]),
                line_width=2.5,
                opacity=0.9,
                render_lines_as_tubes=True,
            )
        _set_actor_visibility(scene_state["vehicle_actor"], bool(scene_state["show_roads"]))
        _set_actor_visibility(scene_state["ped_actor"], bool(scene_state["show_roads"]))

        _stage("Building + road meshes added to viewer")
        street_arrows = _street_direction_arrows(street_graph)
        if street_arrows is not None:
            scene_state["street_arrows_actor"] = plotter.add_mesh(
                street_arrows,
                color="#00e5ff",
                opacity=0.7,
                smooth_shading=False,
                lighting=False,
            )

        if scene_state["street_arrows_actor"] is not None:
            _set_actor_visibility(scene_state["street_arrows_actor"], True)
            print(f"[arrows] visibility set to True (actor={scene_state['street_arrows_actor']})")
        else:
            print("[arrows] ERROR: street_arrows_actor is None!")
        _stage("Street arrows added")
        _render_cars()
        _stage("Cars rendered")
        if bool(args.debug_cars):
            _log_car_debug("initial-render")

        # ── Traffic light glyphs ─────────────────────────────────────────────
        _tl_mesh = build_light_mesh(traffic_lights_dict)
        scene_state["_tl_mesh"] = _tl_mesh
        if _tl_mesh.n_points > 0:
            _tl_sphere = pv.Sphere(radius=1.4, theta_resolution=10, phi_resolution=10)
            _tl_glyphs = build_light_glyphs(_tl_mesh, _tl_sphere)
            scene_state["_tl_glyphs"] = _tl_glyphs
            scene_state["tl_actor"] = plotter.add_mesh(
                _tl_glyphs,
                scalars="colors",
                rgb=True,
                smooth_shading=True,
                pbr=True,
                metallic=0.1,
                roughness=0.4,
                lighting=True,
            )
            _stage(f"Traffic lights: {_tl_mesh.n_points} intersections")

            # Patch: Only update color array, not geometry, every tick
            def _update_traffic_light_colors():
                from traffic_lights import update_light_mesh
                update_light_mesh(_tl_mesh, traffic_lights_dict)
            scene_state["_update_traffic_light_colors"] = _update_traffic_light_colors
        else:
            scene_state["tl_actor"] = None

        def _timer_callback():
            """Animation timer callback: updates traffic light colors each frame."""
            update_colors = scene_state.get("_update_traffic_light_colors")
            if update_colors is not None:
                update_colors()
            # Insert additional animation logic here if needed.

        # ── Road info picker (Cmd/Ctrl + left-click) ─────────────────────────
        from shapely.geometry import Point as _SPoint, LineString as _LS
        _road_edge_data: list[dict] = []
        for _u, _v, _d in street_graph.edges(data=True):
            _geom = _d.get("geometry")
            if _geom is None:
                _nu = street_graph.nodes.get(_u, {})
                _nv = street_graph.nodes.get(_v, {})
                if "x" in _nu and "x" in _nv:
                    _geom = _LS([(_nu["x"], _nu["y"]), (_nv["x"], _nv["y"])])
            if _geom is not None:
                _road_edge_data.append({
                    "geom": _geom,
                    "highway": _d.get("highway", "unclassified"),
                    "maxspeed": _d.get("maxspeed"),
                    "lanes": _d.get("lanes"),
                    "surface": _d.get("surface"),
                    "oneway": _d.get("oneway", False),
                    "width_m": _d.get("width_m"),
                })

        _pick_modifier_held = [False]

        def _on_modifier_press() -> None:
            _pick_modifier_held[0] = True

        def _on_modifier_release() -> None:
            _pick_modifier_held[0] = False

        plotter.add_key_event("Meta",    _on_modifier_press)
        plotter.add_key_event("Control", _on_modifier_press)

        route_state = {
            "stage": 0,
            "source_node": None,
            "target_node": None,
            "route_actors": [],
        }

        def _remove_pareto_chart() -> None:
            _chart = scene_state.get("pareto_chart")
            if _chart is None:
                return
            try:
                plotter.remove_chart(_chart)
            except Exception:
                pass
            scene_state["pareto_chart"] = None

        def _clear_route_actors(keep_markers: bool = False) -> None:
            actors = route_state.get("route_actors", [])
            start_idx = 2 if keep_markers else 0
            to_remove = list(actors)[start_idx:]
            for _a in list(actors):
                if _a not in to_remove:
                    continue
                try:
                    plotter.remove_actor(_a, reset_camera=False)
                except Exception:
                    pass
            if keep_markers:
                route_state["route_actors"] = list(actors)[:2]
            else:
                route_state["route_actors"] = []

        def _node_xy(_node_id: object) -> tuple[float, float] | None:
            _d = street_graph.nodes.get(_node_id, {})
            if "x" in _d and "y" in _d:
                return float(_d["x"]), float(_d["y"])
            return None

        def _edge_best(_g: nx.MultiDiGraph, _u: object, _v: object, _weight: str) -> dict | None:
            _ed = _g.get_edge_data(_u, _v)
            if not _ed:
                return None
            if isinstance(_ed, dict) and all(isinstance(v, dict) for v in _ed.values()):
                return min(_ed.values(), key=lambda x: float(x.get(_weight, np.inf)))
            return _ed

        def _route_polyline(_g: nx.MultiDiGraph, _nodes: list[object], _weight: str) -> np.ndarray | None:
            if len(_nodes) < 2:
                return None
            _pts: list[list[float]] = []
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _ed = _edge_best(_g, _a, _b, _weight)
                if _ed is None:
                    continue
                _geom = _ed.get("geometry") if isinstance(_ed, dict) else None
                if _geom is not None and hasattr(_geom, "coords"):
                    _xy = np.asarray(_geom.coords, dtype=float)[:, :2]
                else:
                    _pa = _node_xy(_a)
                    _pb = _node_xy(_b)
                    if _pa is None or _pb is None:
                        continue
                    _xy = np.asarray([_pa, _pb], dtype=float)
                if _xy.shape[0] == 0:
                    continue
                for _i in range(_xy.shape[0]):
                    _p = [_xy[_i, 0], _xy[_i, 1], 1.5]
                    if _pts and _i == 0 and np.allclose(_pts[-1][:2], _p[:2]):
                        continue
                    _pts.append(_p)
            if len(_pts) < 2:
                return None
            return np.asarray(_pts, dtype=float)

        def _route_stats(_nodes: list[object], _uv_metrics: dict[tuple[object, object], dict[str, float]]) -> dict[str, float]:
            _dist = _time = _mech = _solar = _net = 0.0
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _m = _uv_metrics.get((_a, _b))
                if _m is None:
                    _edge_data = _edge_best(street_graph, _a, _b, "length")
                    if _edge_data is not None:
                        _dist += float(_edge_data.get("length", 0.0))
                    continue
                _dist += float(_m.get("length_m", 0.0))
                _time += float(_m.get("travel_time_s", 0.0))
                _mech += float(_m.get("mechanical_J", 0.0))
                _solar += float(_m.get("solar_J", 0.0))
                _net += float(_m.get("net_energy_J", 0.0))
            return {
                "distance_m": _dist,
                "travel_time_s": _time,
                "mechanical_J": _mech,
                "solar_J": _solar,
                "net_energy_J": _net,
            }

        def _path_time_energy(_g: nx.MultiDiGraph, _nodes: list[object]) -> tuple[float, float]:
            _time = 0.0
            _energy = 0.0
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _tt = _edge_best(_g, _a, _b, "travel_time_s")
                _ne = _edge_best(_g, _a, _b, "net_energy_J")
                if _tt is not None:
                    _time += float(_tt.get("travel_time_s", 0.0))
                if _ne is not None:
                    _energy += float(_ne.get("net_energy_J", 0.0))
            return _time, _energy

        def _update_pareto_chart(_g: nx.MultiDiGraph, _src: object, _tgt: object) -> None:
            _remove_pareto_chart()
            _pareto = find_pareto_routes(_g, _src, _tgt, k=8)
            if len(_pareto) < 2:
                return

            _candidates: list[dict[str, object]] = []
            try:
                _gen = nx.shortest_simple_paths(_g, _src, _tgt, weight="travel_time_s")
                for _idx, _nodes in enumerate(_gen):
                    if _idx >= 8:
                        break
                    _t_s, _e_j = _path_time_energy(_g, list(_nodes))
                    _candidates.append({
                        "idx": int(_idx),
                        "nodes": list(_nodes),
                        "time_s": float(_t_s),
                        "energy_J": float(_e_j),
                    })
            except nx.NetworkXNoPath:
                return

            if not _candidates:
                return

            _px = np.asarray([c["time_s"] / 60.0 for c in _candidates], dtype=float)
            _py = np.asarray([c["energy_J"] / 3600.0 for c in _candidates], dtype=float)

            _pareto_nodes = {tuple(r.get("nodes", [])) for r in _pareto}
            _p_idx = [int(c["idx"]) for c in _candidates if tuple(c["nodes"]) in _pareto_nodes]
            _p_x = np.asarray([_px[i] for i in _p_idx], dtype=float)
            _p_y = np.asarray([_py[i] for i in _p_idx], dtype=float)
            if _p_x.size < 2:
                return

            _chart = pv.Chart2D()
            _chart.title = "Route Pareto Front — time vs net energy"
            _chart.x_label = "Travel time (min)"
            _chart.y_label = "Net energy (Wh)"
            _chart.size = (260, 180)
            _chart.loc = (0.73, 0.02)

            _chart.scatter(_px, _py, color="#888888", size=6)
            _chart.scatter(_p_x, _p_y, color="#ffd54f", size=10)
            for _idx in _p_idx:
                try:
                    _chart.text(float(_px[_idx]), float(_py[_idx]), str(_idx), color="#ffd54f")
                except Exception:
                    pass

            try:
                plotter.add_chart(_chart)
                scene_state["pareto_chart"] = _chart
            except Exception as _exc:
                print(f"[route] pareto chart unavailable: {_exc}")

        def _update_route_stats_overlay(_summary: dict[str, dict[str, float]]) -> None:
            if not _summary:
                plotter.add_text(
                    "",
                    position=(0.68, 0.15),
                    name="route_stats_overlay",
                    font_size=8,
                    color="#f4f4f4",
                    viewport=True,
                )
                return

            def _line(_name: str, _s: dict[str, float]) -> str:
                _net = float(_s.get("net_energy_J", 0.0))
                _sign = "▼" if _net <= 0.0 else "▲"
                return (
                    f"{_name:<8}  d={_s.get('distance_m', 0.0):7.1f} m  "
                    f"t={_s.get('travel_time_s', 0.0) / 60.0:6.2f} min  "
                    f"mech={_s.get('mechanical_J', 0.0) / 3600.0:7.2f} Wh  "
                    f"solar={_s.get('solar_J', 0.0) / 3600.0:7.2f} Wh  "
                    f"net={_net / 3600.0:7.2f} Wh {_sign}"
                )

            txt = "\n".join([
                _line("Energy", _summary.get("energy", {})),
                _line("Joint", _summary.get("joint", {})),
                _line("Shortest", _summary.get("shortest", {})),
            ])
            plotter.add_text(
                txt,
                position=(0.68, 0.15),
                name="route_stats_overlay",
                font_size=8,
                color="#f4f4f4",
                viewport=True,
            )

        def _cost_cache_key(_hour: float, _alpha: float, _params: SolarParams) -> tuple[float, ...]:
            return (
                round(float(_hour), 2),
                round(float(_alpha), 3),
                round(float(_params.roof_area_m2), 3),
                round(float(_params.panel_efficiency), 4),
                round(float(_params.temperature_derating), 4),
                round(float(_params.vehicle_mass_kg), 3),
                round(float(_params.rolling_coeff), 5),
                round(float(_params.drag_coeff), 5),
                round(float(_params.frontal_area_m2), 4),
            )

        def _compute_and_render_routes(_src: object, _tgt: object) -> None:
            _lat = float(scene_state.get("scene_lat", np.nan))
            _lon = float(scene_state.get("scene_lon", np.nan))
            if not np.isfinite(_lat) or not np.isfinite(_lon):
                print("[route] scene_lat/scene_lon unavailable; cannot build solar costs")
                return

            _alpha = float(scene_state.get("route_alpha", 0.5))
            _route_hour = float(scene_state.get("route_hour", scene_state.get("hour", 12.0)))
            _params_obj = scene_state.get("solar_params")
            _params = _params_obj if isinstance(_params_obj, SolarParams) else SolarParams()

            _shadow = _build_edge_shadow_cache(_route_hour)
            _cost_cache = scene_state.get("edge_costs_cache")
            if not isinstance(_cost_cache, dict):
                _cost_cache = {}
                scene_state["edge_costs_cache"] = _cost_cache

            _ckey = _cost_cache_key(_route_hour, _alpha, _params)
            _cached_costs = _cost_cache.get(_ckey)
            if _cached_costs is None:
                _cached_costs = build_edge_costs(
                    car_paths=car_paths,
                    edge_shadow_frac=np.asarray(_shadow, dtype=float),
                    lat_deg=_lat,
                    lon_deg=_lon,
                    hour_local=_route_hour,
                    params=_params,
                    alpha=_alpha,
                )
                _cost_cache[_ckey] = _cached_costs

            _g_cost = build_graph_with_costs(street_graph, car_paths, _cached_costs)
            _energy_nodes = find_energy_optimal_route(_g_cost, _src, _tgt)
            _joint_nodes = find_joint_optimal_route(_g_cost, _src, _tgt, alpha=_alpha)
            try:
                _short_nodes = list(nx.shortest_path(_g_cost, _src, _tgt, weight="length"))
            except nx.NetworkXNoPath:
                _short_nodes = []

            _clear_route_actors(keep_markers=True)
            _route_specs = [
                ("energy", _energy_nodes, "#32cd32", "net_energy_J"),
                ("joint", _joint_nodes, "#ff9f1a", "combined_score"),
                ("shortest", _short_nodes, "#ffffff", "length"),
            ]
            for _name, _nodes, _color, _w in _route_specs:
                _pl = _route_polyline(_g_cost, _nodes, _w)
                if _pl is None:
                    continue
                _actor = plotter.add_lines(_pl, color=_color, width=5, connected=True)
                route_state["route_actors"].append(_actor)

            _uv_metrics: dict[tuple[object, object], dict[str, float]] = {}
            for _i, _p in enumerate(car_paths):
                _k = (_p.get("u"), _p.get("v"))
                _row = {
                    "length_m": float(_p.get("length", 0.0)),
                    "travel_time_s": float(_cached_costs["travel_time_s"][_i]),
                    "mechanical_J": float(_cached_costs["mechanical_J"][_i]),
                    "solar_J": float(_cached_costs["solar_J"][_i]),
                    "net_energy_J": float(_cached_costs["net_energy_J"][_i]),
                }
                _old = _uv_metrics.get(_k)
                if _old is None or _row["net_energy_J"] < _old["net_energy_J"]:
                    _uv_metrics[_k] = _row

            _summary: dict[str, dict[str, float]] = {}
            for _name, _nodes, _color, _w in _route_specs:
                if len(_nodes) < 2:
                    print(f"[route] {_name}: no path")
                    _summary[_name] = {
                        "distance_m": 0.0,
                        "travel_time_s": 0.0,
                        "mechanical_J": 0.0,
                        "solar_J": 0.0,
                        "net_energy_J": 0.0,
                    }
                    continue
                _s = _route_stats(_nodes, _uv_metrics)
                _summary[_name] = _s
                print(
                    f"[route] {_name}: distance={_s['distance_m']:.1f}m, "
                    f"time={_s['travel_time_s']:.1f}s, mech={_s['mechanical_J']:.1f}J, "
                    f"solar={_s['solar_J']:.1f}J, net={_s['net_energy_J']:.1f}J"
                )
            _update_route_stats_overlay(_summary)
            _update_pareto_chart(_g_cost, _src, _tgt)

        def _road_pick_callback(point) -> None:
            if not _pick_modifier_held[0]:
                return
            if point is None or len(point) < 2:
                return
            px, py = float(point[0]), float(point[1])
            pt = _SPoint(px, py)
            best = min(_road_edge_data, key=lambda e: e["geom"].distance(pt), default=None)
            if best is None or best["geom"].distance(pt) > 20.0:
                return
            hw    = best["highway"] or "unclassified"
            speed = best["maxspeed"]
            lanes = best["lanes"]
            surf  = best["surface"] or "unknown"
            ow    = "one-way" if best["oneway"] else "two-way"
            wid   = f"{best['width_m']:.1f} m" if best["width_m"] else "unknown"
            spd_s = f"{speed:.0f} km/h" if speed else "no data"
            info = (
                f"Road type:   {hw}\n"
                f"Direction:   {ow}\n"
                f"Speed limit: {spd_s}\n"
                f"Lanes:       {lanes if lanes else 'unknown'}\n"
                f"Width:       {wid}\n"
                f"Surface:     {surf}"
            )
            plotter.add_text(
                info,
                position=(0.68, 0.90),
                name="road_info_overlay",
                font_size=9,
                color="white",
                font="courier",
                shadow=True,
            )

            # Ctrl/Cmd-click route workflow state machine.
            try:
                _snap_node = nearest_graph_node(street_graph, px, py)
            except Exception as _exc:
                print(f"[route] Node snap failed: {_exc}")
                _pick_modifier_held[0] = False
                return

            if route_state["stage"] == 2:
                _clear_route_actors()
                _remove_pareto_chart()
                route_state["stage"] = 0
                route_state["source_node"] = None
                route_state["target_node"] = None
                _update_route_stats_overlay({})
                print("[route] Route selection reset")
                _pick_modifier_held[0] = False
                return

            _xy = _node_xy(_snap_node)
            if _xy is None:
                print(f"[route] Node {_snap_node} has no local coordinates")
                _pick_modifier_held[0] = False
                return

            if route_state["stage"] == 0:
                _clear_route_actors()
                route_state["source_node"] = _snap_node
                _src_actor = plotter.add_mesh(
                    pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0), theta_resolution=18, phi_resolution=18),
                    color="#33cc66",
                    render=False,
                )
                route_state["route_actors"] = [_src_actor]
                route_state["stage"] = 1
                print(f"[route] Source set: node {_snap_node}")
                _pick_modifier_held[0] = False
                return

            route_state["target_node"] = _snap_node
            _tgt_actor = plotter.add_mesh(
                pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0), theta_resolution=18, phi_resolution=18),
                color="#ff4d4d",
                render=False,
            )
            route_state.setdefault("route_actors", []).append(_tgt_actor)
            print(f"[route] Target set: node {_snap_node}")
            _compute_and_render_routes(route_state["source_node"], route_state["target_node"])

            route_state["stage"] = 2
            _pick_modifier_held[0] = False  # consume modifier

        plotter.add_key_event(
            "Escape",
            lambda: plotter.add_text("", position=(0.68, 0.90), name="road_info_overlay"),
        )
        try:
            plotter.enable_surface_point_picking(
                callback=_road_pick_callback,
                show_message=False,
                show_point=False,
                tolerance=0.025,
            )
        except Exception as _exc:
            print(f"[picker] enable_surface_point_picking not available: {_exc}")
        # ─────────────────────────────────────────────────────────────────────

        # ── POI labels ───────────────────────────────────────────────────────
        from overture_source import _poi_color

        # Short ASCII category tags — VTK text renderer cannot handle Unicode
        _CAT_ABBREV: dict[str, str] = {
            "restaurant": "Rest", "cafe": "Cafe", "bar": "Bar",
            "fast_food": "Food", "bakery": "Bakery",
            "supermarket": "Mkt", "convenience_store": "Conv",
            "pharmacy": "Rx", "clothing_store": "Shop",
            "parking": "P", "gas_station": "Gas", "bus_stop": "Bus",
            "bank": "Bank", "atm": "ATM", "hospital": "Hosp",
            "school": "School", "hotel": "Hotel", "post_office": "Post",
            "park": "Park", "gym": "Gym", "museum": "Mus", "church": "Ch",
        }

        def _hex_to_rgb(h: str) -> list[int]:
            h = h.lstrip("#")
            return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

        def _poi_label(p: dict) -> str:
            cats = p["categories"]
            tag = next((f"[{_CAT_ABBREV[c]}] " for c in cats if c in _CAT_ABBREV), "")
            name = p["name"] or (cats[0] if cats else "place")
            # Strip non-ASCII: VTK's text renderer only handles ASCII
            name = name.encode("ascii", "ignore").decode("ascii").strip() or "place"
            max_name = 28 - len(tag)
            if len(name) > max_name:
                name = name[:max(max_name - 3, 4)] + "..."
            return tag + name

        if places:
            _poi_z = 2.5
            _poi_pts = np.array([[p["x"], p["y"], _poi_z] for p in places], dtype=float)
            _poi_mesh = pv.PolyData(_poi_pts)
            _poi_mesh["labels"] = [_poi_label(p) for p in places]
            _poi_mesh.point_data["colors"] = np.array(
                [_hex_to_rgb(_poi_color(p["categories"])) for p in places],
                dtype=np.uint8,
            )

            # Dots actor — colored spheres, controlled by "POIs" toggle
            _poi_dots_actor = plotter.add_mesh(
                _poi_mesh,
                scalars="colors",
                rgb=True,
                style="points",
                point_size=14,
                render_points_as_spheres=True,
                lighting=False,
                reset_camera=False,
            )

            # Labels actor — text only (tiny 1-px dot hidden), controlled by "Names" toggle
            _poi_labels_actor = plotter.add_point_labels(
                _poi_mesh,
                "labels",
                point_size=1,
                font_size=14,
                text_color="white",
                render_points_as_spheres=False,
                always_visible=False,
                shadow=True,
                shape_opacity=0.55,
                shape_color="#111111",
                tolerance=0.01,
            )

            scene_state["poi_actor"] = _poi_dots_actor
            scene_state["poi_labels_actor"] = _poi_labels_actor
            scene_state["show_pois"] = True
            scene_state["show_poi_names"] = True
            print(f"[poi] {len(places)} POIs: {len(places)} dots + {len(places)} labels")
        else:
            scene_state["poi_actor"] = None
            scene_state["poi_labels_actor"] = None
            scene_state["show_pois"] = False
            scene_state["show_poi_names"] = True
            print("[poi] No places found in this area")
        # ─────────────────────────────────────────────────────────────────────

        scene_state["lights_actor"] = None

        # Opt4: pre-allocate ground copies once so _render_ground avoids repeated .copy().
        scene_state["day_ground_copy"]   = ground_mesh.copy()
        scene_state["night_ground_copy"] = ground_mesh.copy()

        # Opt2: threading.Event to prevent duplicate shadow warmup runs.
        import threading as _threading
        _shadow_warmup_done = _threading.Event()
        scene_state["_shadow_warmup_done"] = _shadow_warmup_done

        # Fast initial render so the window appears immediately.
        initial_ground = ground_mesh.copy()
        initial_ground.cell_data["surface_class"] = _compose_day_surface_classes(
            np.zeros((initial_ground.n_cells,), dtype=bool)
        )
        scene_state["ground_actor"] = plotter.add_mesh(
            initial_ground,
            scalars="surface_class",
            clim=[0, 3],
            cmap=day_surface_cmap,
            show_edges=False,
            opacity=0.9,
        )
        scene_state["spotlight_actor"] = None
        plotter.add_text("City Digital Twin", position=(0.18, 0.02), name="status", font_size=9)
        _stage(f"Initial ground actor added ({initial_ground.n_cells} cells)")

        def _render_ground(hour: float, spot_radius: float, _use_update: bool = False) -> None:
            nonlocal ground_mesh, edge_shadow_frac
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
            edge_shadow_frac = _build_edge_shadow_cache(float(hour))
            style = _style()

            try:
                hdri_tex = _load_hdri(float(hour), str(scene_state["preset"]))
                plotter.set_environment_texture(hdri_tex)
                plotter.renderer.UseImageBasedLightingOn()
            except Exception as _exc:
                print(f"[pbr] dynamic HDRI unavailable ({_exc}); using standard shading")

            try:
                plotter.remove_all_lights()
                _sun_pos = np.asarray(sun_dir, dtype=float) * 500.0
                _sun_intensity = max(0.0, float(sun_dir[2])) * 0.9 + 0.1
                _sun_light = pv.Light(
                    light_type="scene light",
                    position=(float(_sun_pos[0]), float(_sun_pos[1]), float(_sun_pos[2])),
                    focal_point=(0.0, 0.0, 0.0),
                    intensity=float(_sun_intensity),
                )
                plotter.add_light(_sun_light)
            except Exception as _exc:
                print(f"[light] dynamic scene light unavailable ({_exc})")

            b_actor = scene_state.get("building_actor")
            _pbr_actors = scene_state.get("_building_actors_pbr", {})
            # PBR buildings: only update edge color (SetColor would override the material albedo).
            # Non-PBR fallback: update both color and edge color as before.
            _edge_rgb = pv.Color(str(style["building_edge"])).float_rgb
            if _pbr_actors:
                for _ba in _pbr_actors.values():
                    try:
                        _ba.GetProperty().SetEdgeColor(_edge_rgb)
                    except Exception:
                        pass
            elif b_actor is not None:
                try:
                    b_actor.GetProperty().SetColor(pv.Color(str(style["building"])).float_rgb)
                    b_actor.GetProperty().SetEdgeColor(_edge_rgb)
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
            car_actors = scene_state.get("car_actors")
            if isinstance(car_actors, dict):
                car_actor = car_actors.get("cars")
                if car_actor is not None:
                    try:
                        car_actor.GetProperty().SetColor(pv.Color(str(style["car"])).float_rgb)
                    except Exception:
                        pass

            if is_night:
                night_copy = scene_state["night_ground_copy"]

                night_key = (
                    float(spot_radius),
                    int(best_positions.shape[0]),
                    hashlib.sha1(np.ascontiguousarray(best_positions, dtype=np.float64).tobytes()).hexdigest()[:16],
                )
                if "cached_night_coverage" not in scene_state or scene_state.get("cached_night_key") != night_key:
                    print(f"[{time.strftime('%H:%M:%S')}] Building night coverage (lights={len(best_positions)}, cells={ground_mesh.n_cells}, radius={spot_radius:.1f}m)...")
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

                if coverage_count.shape[0] != night_copy.n_cells:
                    print(
                        "Night coverage cache mismatch with ground mesh "
                        f"({coverage_count.shape[0]} vs {night_copy.n_cells}); rebuilding once."
                    )
                    ground_mesh = _normalize_ground_mesh(ground_mesh)
                    scene_state["night_ground_copy"] = ground_mesh.copy()
                    scene_state["day_ground_copy"]   = ground_mesh.copy()
                    night_copy = scene_state["night_ground_copy"]
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

                illum = np.zeros(night_copy.n_cells, dtype=np.uint8)
                illum[coverage_count >= 1] = 1
                illum[coverage_count >= 2] = 2
                night_copy.cell_data["night_surface_class"] = _compose_night_surface_classes(illum)

                lit_pct = 100.0 * float(np.count_nonzero(illum >= 1)) / float(max(1, illum.size))
                scene_state["ground_actor"] = plotter.add_mesh(
                    night_copy,
                    scalars="night_surface_class",
                    clim=[0, 5],
                    cmap=night_surface_cmap,
                    show_edges=False,
                    opacity=0.9,
                )
                plotter.set_background(str(style["night_bg"]))
                plotter.add_text(
                    f"Hour {hour:04.1f}  |  Night Mode  |  Lit {lit_pct:05.1f}%",
                    position=(0.18, 0.02),
                    name="status",
                    font_size=9,
                )
            else:
                day_cache = scene_state.get("cached_day_shadows")
                if not isinstance(day_cache, dict):
                    day_cache = {}
                    scene_state["cached_day_shadows"] = day_cache

                hour_key = round(float(hour), 2)
                cached_day = day_cache.get(hour_key)
                if cached_day is None:
                    print(f"[{time.strftime('%H:%M:%S')}] Computing shadows (hour={hour:.1f}, cells={ground_mesh.n_cells}, octree={'ready' if octree_root else 'building'})...")
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

                    # Opt2: pre-warm 3 neighbor hours in background after first miss.
                    _warmup_done_ev = scene_state.get("_shadow_warmup_done")
                    if _warmup_done_ev is not None and not _warmup_done_ev.is_set():
                        _warmup_done_ev.set()
                        _neighbor_hours = [
                            float(np.clip(hour_key + offset, 6.0, 18.0))
                            for offset in (-1.0, 1.0, 2.0)
                            if round(float(np.clip(hour_key + offset, 6.0, 18.0)), 2) not in day_cache
                        ]
                        _warmup_octree = _ensure_octree()
                        _warmup_mesh = ground_mesh

                        def _warm_neighbors(
                            _hours: list = _neighbor_hours,
                            _dc: dict = day_cache,
                            _om: object = _warmup_octree,
                            _gm: object = _warmup_mesh,
                        ) -> None:
                            import threading as _t
                            for _h in _hours:
                                _hk = round(_h, 2)
                                if _hk in _dc:
                                    continue
                                try:
                                    _sd = _sun_dir_from_hour(_h)
                                    _m, _lr = compute_shadows(ground_mesh=_gm, octree_root=_om, sun_dir=_sd)
                                    _dc[_hk] = (_m.astype(bool, copy=False), float(_lr))
                                    print(f"[perf] pre-warmed shadow cache for hour={_hk:.2f}")
                                except Exception:
                                    pass

                        # Background pre-warming disabled: concurrent Numba threads
                        # on the Cocoa run loop thread cause deadlocks on macOS.
                else:
                    mask, lit_ratio = cached_day
                day_copy = scene_state["day_ground_copy"]
                day_copy.cell_data["day_surface_class"] = _compose_day_surface_classes(mask)
                scene_state["ground_actor"] = plotter.add_mesh(
                    day_copy,
                    scalars="day_surface_class",
                    clim=[0, 3],
                    cmap=day_surface_cmap,
                    show_edges=False,
                    opacity=0.9,
                )
                plotter.set_background(str(style["day_bg"]))
                plotter.add_text(f"Hour {hour:04.1f}  |  Lit Ratio {lit_ratio:.3f}", position=(0.18, 0.02), name="status", font_size=9)

            if is_night:
                light_points_local = np.column_stack(
                    (best_positions[:, 0], best_positions[:, 1], np.full(best_positions.shape[0], 0.5))
                )
                scene_state["lights_actor"] = plotter.add_points(
                    light_points_local,
                    color=str(style["light_night"]),
                    point_size=24,
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
            _render_cars()
            _apply_atmosphere(
                plotter,
                hour=float(hour),
                is_night=bool(is_night),
                preset=str(scene_state["preset"]),
            )

            if _use_update:
                plotter.update(stime=1)
            else:
                plotter.render()

        def _on_time_change(value: float) -> None:
            scene_state["hour"] = float(value)
            if not bool(scene_state.get("interactive_ready", False)):
                return
            try:
                _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))
            except Exception as _exc:
                print(f"[render] shadow render error: {_exc}")

        def _on_radius_change(value: float) -> None:
            scene_state["spot_radius"] = float(value)
            # Invalidate cached coverage since spotlight radius changed
            scene_state.pop("cached_night_coverage", None)
            scene_state.pop("cached_night_key", None)
            if not bool(scene_state.get("interactive_ready", False)):
                return
            try:
                _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))
            except Exception as _exc:
                print(f"[render] spotlight render error: {_exc}")

        plotter.add_slider_widget(
            _on_time_change,
            rng=[0.0, 24.0],
            value=12.0,
            title="Hour",
            pointa=(0.02, 0.14),
            pointb=(0.30, 0.14),
            style="modern",
            interaction_event="always",
        )
        plotter.add_slider_widget(
            _on_radius_change,
            rng=[max(5.0, args.light_radius * 0.4), args.light_radius * 3.0],
            value=float(args.light_radius),
            title="Spotlight Radius",
            pointa=(0.02, 0.08),
            pointb=(0.30, 0.08),
            style="modern",
            interaction_event="end",
        )

        def _toggle_roads(value: bool) -> None:
            scene_state["show_roads"] = bool(value)
            _set_actor_visibility(scene_state.get("vehicle_actor"), bool(value))
            _set_actor_visibility(scene_state.get("ped_actor"), bool(value))
            plotter.render()

        def _toggle_cars(value: bool) -> None:
            scene_state["show_cars"] = bool(value)
            car_actors = scene_state.get("car_actors")
            if isinstance(car_actors, dict):
                for actor in car_actors.values():
                    _set_actor_visibility(actor, bool(value))
            for _ua in scene_state.get("_ultra_car_actors") or []:
                _set_actor_visibility(_ua, bool(value))
            plotter.render()

        def _preset_mini(_: bool) -> None:
            scene_state["preset"] = "mini"
            if not bool(scene_state.get("interactive_ready", False)):
                return
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_coastal(_: bool) -> None:
            scene_state["preset"] = "coastal"
            if not bool(scene_state.get("interactive_ready", False)):
                return
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_sunset(_: bool) -> None:
            scene_state["preset"] = "sunset"
            if not bool(scene_state.get("interactive_ready", False)):
                return
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _toggle_arrows(value: bool) -> None:
            scene_state["show_arrows"] = bool(value)
            _set_actor_visibility(scene_state.get("street_arrows_actor"), bool(value))
            plotter.render()

        def _on_optimize(_: bool) -> None:
            if not bool(scene_state.get("interactive_ready", False)):
                return
            if bool(scene_state.get("ga_running", False)):
                # Already running — ignore double-click
                return
            scene_state["ga_running"] = True
            plotter.add_text("Running GA optimization...", position=(0.18, 0.02), name="status", font_size=9)
            plotter.render()

            def _do_ga_thread() -> None:
                nonlocal best_positions
                try:
                    cov_matrix_local = None
                    sw_poly_local = None
                    if args.fast_startup:
                        sw_poly_local = build_sidewalk_polygon_from_street_graph(street_graph)
                        gp = _build_grid_points_from_ground_mesh(
                            ground_mesh, args.grid_step, sidewalk_polygon=sw_poly_local
                        )
                        cov_matrix_local = _load_or_build_coverage_matrix_cached(
                            cache_dir=cache_dir,
                            use_cache=use_cache,
                            cache_context_key=cache_context_key,
                            grid_points=gp,
                            ground_mesh=ground_mesh,
                            octree_root=_ensure_octree(),
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
                        octree_root=_ensure_octree(),
                        pole_height=args.pole_height,
                        use_precomputed_coverage=bool(args.fast_startup),
                        precomputed_coverage_matrix=cov_matrix_local,
                        street_graph=street_graph,
                        sidewalk_polygon=sw_poly_local,
                        ga_jobs=max(1, int(args.ga_jobs)),
                        ga_progress_every=max(1, int(args.ga_progress_every)),
                        ga_verbose=True,
                    )
                    best_positions = np.asarray(ga_result["best_positions"], dtype=float)
                    scene_state.pop("cached_night_coverage", None)
                    scene_state.pop("cached_night_key", None)
                    scene_state["ga_done"] = True
                    print(f"[ga] Optimization complete: cost={ga_result['best_cost']:.4f}, lit={ga_result['lit_ratio']:.3f}")
                except Exception as _exc:
                    print(f"[ga] Background GA failed: {_exc}")
                    scene_state["ga_done"] = True
                finally:
                    scene_state["ga_running"] = False

            import threading as _threading_ga
            _threading_ga.Thread(target=_do_ga_thread, daemon=True).start()

        def _poll_ga_done(_: int) -> None:
            """Timer callback: fires every 500ms to flush the GA result into the viewer."""
            try:
                if not bool(scene_state.pop("ga_done", False)):
                    return
                try:
                    _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]), _use_update=True)
                except Exception as _exc:
                    print(f"[ga] Post-GA render failed: {_exc}")
            except Exception as _exc:
                print(f"[ga-poll ERROR] {_exc}")

        # ── Clean upper-left control panel ───────────────────────────────────
        # Layout: checkbox at x=8px, label text at x_norm=0.042 (~59px).
        # Rows count down from PANEL_TOP in 30px steps (window height=900).
        # Sliders remain at the bottom (y_norm 0.08 and 0.14) — no overlap.

        _CX = 8       # checkbox pixel x
        _CS = 22      # checkbox size (px)
        _TX = 0.042   # text normalized x (label)
        _DX = 0.105   # text normalized x (one-word description, right of label)

        def _panel_text_color():
            # Choose text color for panel for readability
            bg = str(_style()["day_bg"] if not scene_state.get("is_night") else _style()["night_bg"])
            # Simple luminance check
            bg = bg.lstrip("#")
            r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "#f4f4f4" if luminance < 128 else "#222"

        def _panel_desc_color():
            # Muted but still readable
            base = _panel_text_color()
            if base == "#f4f4f4":
                return "#e0e0e0"
            return "#444"

        _DC = _panel_desc_color()
        _TC = _panel_text_color()

        def _cy(row: int) -> tuple[float, float]:
            """Return (checkbox_y_px, text_y_norm) for given row (0=top)."""
            y_px = 836 - row * 30
            return float(y_px), float(y_px + 2) / 900.0

        # ── Section: Visibility ───────────────────────────────────────────
        plotter.add_text("Visibility", position=(_TX, _cy(0)[1] + 0.006),
             name="panel_vis_hdr", font_size=10, color=_TC, viewport=True)

        _r1 = _cy(1)
        plotter.add_text("Roads",   position=(_TX, _r1[1]), name="panel_roads",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("streets", position=(_DX, _r1[1]), name="panel_roads_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _toggle_roads, value=bool(scene_state["show_roads"]),
            position=(_CX, _r1[0]), size=_CS, color_on="#6aa06f", color_off="#5d6673",
        )

        _r2 = _cy(2)
        plotter.add_text("Cars",    position=(_TX, _r2[1]), name="panel_cars",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("traffic", position=(_DX, _r2[1]), name="panel_cars_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _toggle_cars, value=bool(scene_state["show_cars"]),
            position=(_CX, _r2[0]), size=_CS, color_on="#ff6b6b", color_off="#5d6673",
        )

        _r3 = _cy(3)
        plotter.add_text("Arrows",    position=(_TX, _r3[1]), name="panel_arrows",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("direction", position=(_DX, _r3[1]), name="panel_arrows_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _toggle_arrows, value=True,
            position=(_CX, _r3[0]), size=_CS, color_on="#ff3b30", color_off="#5d6673",
        )

        _r4 = _cy(4)
        plotter.add_text("Lights",  position=(_TX, _r4[1]), name="panel_lights",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("signals", position=(_DX, _r4[1]), name="panel_lights_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            lambda val: _set_actor_visibility(scene_state.get("tl_actor"), val),
            value=True,
            position=(_CX, _r4[0]), size=_CS, color_on="#22cc55", color_off="#555555",
        )

        _r5 = _cy(5)
        plotter.add_text("POIs",   position=(_TX, _r5[1]), name="panel_pois",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("places", position=(_DX, _r5[1]), name="panel_pois_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            lambda val: _set_actor_visibility(scene_state.get("poi_actor"), val),
            value=True,
            position=(_CX, _r5[0]), size=_CS, color_on="#e07b54", color_off="#555555",
        )

        _r6 = _cy(6)
        plotter.add_text("Names",  position=(_TX, _r6[1]), name="panel_names",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("labels", position=(_DX, _r6[1]), name="panel_names_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            lambda val: _set_actor_visibility(scene_state.get("poi_labels_actor"), val),
            value=True,
            position=(_CX, _r6[0]), size=_CS, color_on="#f5a623", color_off="#555555",
        )

        _r7 = _cy(7)
        plotter.add_text("Optimize", position=(_TX, _r7[1]), name="panel_optimize",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("compute",  position=(_DX, _r7[1]), name="panel_optimize_d", font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _on_optimize, value=False,
            position=(_CX, _r7[0]), size=_CS, color_on="#ffd166", color_off="#4a5568",
        )

        _r_ssao = _cy(8.5)
        plotter.add_text("SSAO",      position=(_TX, _r_ssao[1]), name="panel_ssao",   font_size=9, color=_TC, viewport=True)
        plotter.add_text("occlusion", position=(_DX, _r_ssao[1]), name="panel_ssao_d", font_size=8, color=_DC, viewport=True)

        def _toggle_ssao(val: bool) -> None:
            try:
                renderer.SetUseSSAO(bool(val))
                plotter.render()
            except Exception:
                print("[ssao] SSAO unavailable")

        _ssao_enabled = False
        try:
            _ssao_enabled = bool(renderer.GetUseSSAO())
        except Exception:
            _ssao_enabled = False
        plotter.add_checkbox_button_widget(
            _toggle_ssao, value=_ssao_enabled,
            position=(_CX, _r_ssao[0]), size=_CS, color_on="#6cb6ff", color_off="#5d6673",
        )

        # ── Section: Style presets ────────────────────────────────────────
        _rs = _cy(9)
        plotter.add_text("Style", position=(_TX, _rs[1] + 0.006),
             name="panel_style_hdr", font_size=10, color=_TC, viewport=True)

        _r9 = _cy(10)
        plotter.add_text("Mini",    position=(_TX, _r9[1]), name="preset_mini_t",    font_size=9, viewport=True)
        plotter.add_text("compact", position=(_DX, _r9[1]), name="preset_mini_d",    font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _preset_mini, value=True,
            position=(_CX, _r9[0]), size=_CS, color_on="#ffd166", color_off="#8d99ae",
        )

        _r10 = _cy(11)
        plotter.add_text("Coastal", position=(_TX, _r10[1]), name="preset_coastal_t",   font_size=9, viewport=True)
        plotter.add_text("ocean",   position=(_DX, _r10[1]), name="preset_coastal_d",   font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _preset_coastal, value=False,
            position=(_CX, _r10[0]), size=_CS, color_on="#56b48a", color_off="#8d99ae",
        )

        _r11 = _cy(12)
        plotter.add_text("Sunset", position=(_TX, _r11[1]), name="preset_sunset_t",   font_size=9, viewport=True)
        plotter.add_text("warm",   position=(_DX, _r11[1]), name="preset_sunset_d",   font_size=8, color=_DC, viewport=True)
        plotter.add_checkbox_button_widget(
            _preset_sunset, value=False,
            position=(_CX, _r11[0]), size=_CS, color_on="#ffb86b", color_off="#8d99ae",
        )

        # ── Section: Solar routing ────────────────────────────────────────
        _r_sol = _cy(13)
        plotter.add_text("Solar", position=(_TX, _r_sol[1] + 0.006),
             name="panel_solar_hdr", font_size=10, color=_TC, viewport=True)

        _r_alpha = _cy(14)
        _r_hour = _cy(15)
        _r_area = _cy(16)

        def _on_route_alpha_change(value: float) -> None:
            scene_state["route_alpha"] = float(np.clip(value, 0.0, 1.0))
            if (
                route_state.get("stage") == 2
                and route_state.get("source_node") is not None
                and route_state.get("target_node") is not None
            ):
                try:
                    _compute_and_render_routes(route_state["source_node"], route_state["target_node"])
                    plotter.render()
                except Exception as _exc:
                    print(f"[route] alpha update failed: {_exc}")

        def _on_route_hour_change(value: float) -> None:
            scene_state["route_hour"] = float(np.clip(value, 6.0, 18.0))
            scene_state["edge_shadow_cache"] = {}
            scene_state["edge_costs_cache"] = {}

        def _on_panel_area_change(value: float) -> None:
            _old = scene_state.get("solar_params")
            if isinstance(_old, SolarParams):
                scene_state["solar_params"] = SolarParams(
                    roof_area_m2=float(np.clip(value, 0.5, 3.0)),
                    panel_efficiency=float(_old.panel_efficiency),
                    temperature_derating=float(_old.temperature_derating),
                    vehicle_mass_kg=float(_old.vehicle_mass_kg),
                    rolling_coeff=float(_old.rolling_coeff),
                    drag_coeff=float(_old.drag_coeff),
                    frontal_area_m2=float(_old.frontal_area_m2),
                )
            else:
                scene_state["solar_params"] = SolarParams(roof_area_m2=float(np.clip(value, 0.5, 3.0)))

        plotter.add_slider_widget(
            _on_route_alpha_change,
            rng=[0.0, 1.0],
            value=float(scene_state.get("route_alpha", 0.5)),
            title="α time↔energy",
            pointa=(0.73, _r_alpha[1]),
            pointb=(0.97, _r_alpha[1]),
            style="modern",
            interaction_event="always",
        )
        plotter.add_slider_widget(
            _on_route_hour_change,
            rng=[6.0, 18.0],
            value=float(scene_state.get("route_hour", scene_state.get("hour", 12.0))),
            title="Route hour",
            pointa=(0.73, _r_hour[1]),
            pointb=(0.97, _r_hour[1]),
            style="modern",
            interaction_event="always",
        )
        plotter.add_slider_widget(
            _on_panel_area_change,
            rng=[0.5, 3.0],
            value=float(getattr(scene_state.get("solar_params"), "roof_area_m2", 1.6)),
            title="Panel m²",
            pointa=(0.73, _r_area[1]),
            pointb=(0.97, _r_area[1]),
            style="modern",
            interaction_event="always",
        )

        plotter.add_text(
            "",
            position=(0.68, 0.15),
            name="route_stats_overlay",
            font_size=8,
            color="#f4f4f4",
            viewport=True,
        )
        # ─────────────────────────────────────────────────────────────────────

        if args.optimize_on_open and best_positions is None:
            _stage("Optimize-on-open enabled: running GA before first interactive render...")
            t_open_ga = time.perf_counter()
            coverage_matrix = None
            sidewalk_polygon = None
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
                sidewalk_polygon=sidewalk_polygon,
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

        # Keyboard navigation for camera pan/rotate/zoom.
        x0, x1, y0, y1, _, _ = ground_mesh.bounds
        scene_span = max(1.0, float(max(x1 - x0, y1 - y0)))
        pan_step = 0.04 * scene_span
        rotate_step = 6.0

        def _pan_camera(dx: float, dy: float) -> None:
            cam = plotter.camera
            pos = np.asarray(cam.position, dtype=float)
            focal = np.asarray(cam.focal_point, dtype=float)
            up = np.asarray(cam.up, dtype=float)
            view = focal - pos
            nv = float(np.linalg.norm(view))
            nu = float(np.linalg.norm(up))
            if nv <= 1e-9 or nu <= 1e-9:
                return
            view = view / nv
            up = up / nu
            right = np.cross(view, up)
            nr = float(np.linalg.norm(right))
            if nr <= 1e-9:
                return
            right = right / nr
            shift = (right * float(dx)) + (up * float(dy))
            cam.position = tuple(pos + shift)
            cam.focal_point = tuple(focal + shift)
            plotter.render()

        def _rotate_camera(azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> None:
            cam = plotter.camera
            if abs(float(azimuth_deg)) > 0.0:
                cam.Azimuth(float(azimuth_deg))
            if abs(float(elevation_deg)) > 0.0:
                cam.Elevation(float(elevation_deg))
            cam.OrthogonalizeViewUp()
            plotter.render()

        def _zoom_camera(factor: float) -> None:
            cam = plotter.camera
            cam.Dolly(float(factor))
            plotter.reset_camera_clipping_range()
            plotter.render()

        def _reset_camera_view() -> None:
            plotter.view_isometric()
            plotter.render()

        plotter.add_key_event("Left", lambda: _pan_camera(-pan_step, 0.0))
        plotter.add_key_event("Right", lambda: _pan_camera(+pan_step, 0.0))
        plotter.add_key_event("Up", lambda: _pan_camera(0.0, +pan_step))
        plotter.add_key_event("Down", lambda: _pan_camera(0.0, -pan_step))
        plotter.add_key_event("a", lambda: _pan_camera(-pan_step, 0.0))
        plotter.add_key_event("d", lambda: _pan_camera(+pan_step, 0.0))
        plotter.add_key_event("w", lambda: _pan_camera(0.0, +pan_step))
        plotter.add_key_event("s", lambda: _pan_camera(0.0, -pan_step))
        plotter.add_key_event("q", lambda: _rotate_camera(-rotate_step, 0.0))
        plotter.add_key_event("e", lambda: _rotate_camera(+rotate_step, 0.0))
        plotter.add_key_event("r", lambda: _rotate_camera(0.0, +rotate_step))
        plotter.add_key_event("f", lambda: _rotate_camera(0.0, -rotate_step))
        plotter.add_key_event("z", lambda: _zoom_camera(1.12))
        plotter.add_key_event("x", lambda: _zoom_camera(1.0 / 1.12))
        plotter.add_key_event("c", _reset_camera_view)
        if bool(args.debug_cars):
            plotter.add_key_event("p", lambda: _log_car_debug("manual"))
            print("[cars-debug] press 'p' in the viewer to print car status")

        if bool(car_anim["enabled"]) and float(args.traffic_speed) > 0.0:
            car_anim["last_t"] = time.perf_counter()
            car_timer_ms = 80
            if args.car_detail == "low":
                car_timer_ms = max(car_timer_ms, 120)
            if ground_mesh.n_cells >= 120_000:
                car_timer_ms = 90
                print(
                    "Large ground mesh detected "
                    f"({ground_mesh.n_cells} triangles): reducing car animation rate to ~{int(round(1000.0 / car_timer_ms))} FPS."
                )

            def _animate_cars(_: int) -> None:
                _animate_cars._tick = getattr(_animate_cars, "_tick", 0) + 1
                tick = _animate_cars._tick
                if tick == 1:
                    print("[cars] tick 1 — first animation frame")
                elif tick == 2:
                    print("[cars] tick 2 — if you see this, the freeze is fixed")
                elif tick == 10:
                    print("[cars] tick 10 — animation running stably")
                elif tick % 100 == 0:
                    print(f"[cars] tick {tick} — still running")
                try:
                    if not hasattr(_animate_cars, "_first_tick_logged"):
                        _animate_cars._first_tick_logged = True
                        print("[timer] _animate_cars first tick fired")
                    now = time.perf_counter()
                    last_t = float(car_anim.get("last_t", now))
                    dt = float(np.clip(now - last_t, 0.0, 0.10))
                    car_anim["last_t"] = now
                    if dt <= 0.0:
                        return
                    _advance_cars(dt)
                    _render_cars()

                    # ── Tick and redraw traffic lights every 10 frames ────────
                    _tl = scene_state.get("traffic_lights")
                    if _tl:
                        tick_all(_tl, dt=dt, traffic_speed=float(args.traffic_speed))
                        _tl_tick = getattr(_animate_cars, "_tl_tick", 0) + 1
                        _animate_cars._tl_tick = _tl_tick
                        if _tl_tick % 10 == 0:
                            _tl_m = scene_state.get("_tl_mesh")
                            _tl_glyphs = scene_state.get("_tl_glyphs")
                            _tl_actor = scene_state.get("tl_actor")
                            if (_tl_m is not None and _tl_m.n_points > 0
                                    and _tl_glyphs is not None and _tl_actor is not None):
                                # Update source mesh colors in-place
                                update_light_mesh(_tl_m, _tl)
                                # Broadcast updated colors from source points to glyph points.
                                # Each input point maps to n_per output points (sphere resolution).
                                try:
                                    n_lights = _tl_m.n_points
                                    n_glyph_pts = _tl_glyphs.n_points
                                    if n_glyph_pts > 0 and n_glyph_pts % n_lights == 0:
                                        n_per = n_glyph_pts // n_lights
                                        src_colors = np.asarray(_tl_m["colors"], dtype=np.uint8)
                                        _tl_glyphs["colors"] = np.repeat(src_colors, n_per, axis=0)
                                        # Mark the mapper's input as modified so VTK re-uploads colors to GPU
                                        _tl_glyphs.Modified()
                                except Exception as _tl_exc:
                                    print(f"[tl] color update failed: {_tl_exc}")
                    # ─────────────────────────────────────────────────────────
                    if bool(args.debug_cars):
                        car_debug_state["tick"] = int(car_debug_state.get("tick", 0)) + 1
                        next_log_t = float(car_debug_state.get("next_log_t", now + 1.5))
                        if now >= next_log_t:
                            _log_car_debug(f"tick={int(car_debug_state['tick'])}")
                            _log_idm_diagnostics(f"tick={int(car_debug_state['tick'])}")
                            car_debug_state["next_log_t"] = now + 2.5
                    plotter.update(stime=1)
                except Exception as _exc:
                    import traceback as _tb
                    print(f"[cars-timer ERROR] {_exc}")
                    _tb.print_exc()

            plotter.add_timer_event(
                max_steps=10_000_000,
                duration=car_timer_ms,
                callback=_animate_cars,
            )
        elif bool(args.debug_cars):
            if not bool(car_anim["enabled"]):
                print("[cars-debug] animation timer not started: no active cars")
            elif float(args.traffic_speed) <= 0.0:
                print("[cars-debug] animation timer not started: traffic_speed <= 0")

        def _mark_interactive_ready(_: int) -> None:
            try:
                print("[timer] _mark_interactive_ready fired")
                scene_state["interactive_ready"] = True
                if bool(args.debug_cars):
                    print("[viewer-debug] interactive callbacks enabled")
            except Exception as _exc:
                print(f"[ready-timer ERROR] {_exc}")

        plotter.add_timer_event(max_steps=1, duration=300, callback=_mark_interactive_ready)
        # Poll every 500 ms: re-renders the scene once the background GA thread finishes.
        plotter.add_timer_event(max_steps=10_000_000, duration=500, callback=_poll_ga_done)
        _stage("Controls and widgets wired")

        _stage(f"Ground mesh: {ground_mesh.n_cells} cells — shadows will render at midday on startup")

        # Defer the initial shadow render until after plotter.show() has opened the
        # window.  Calling _render_ground (which calls plotter.render()) before
        # plotter.show() on macOS tries to realize a VTK render pass without a live
        # Cocoa window, which deadlocks.
        def _deferred_initial_render(_: int) -> None:
            try:
                print("[timer] _deferred_initial_render fired")
                # Render ground with flat color — no shadow computation yet.
                # This is instant and does NOT block the Cocoa event loop.
                plain_copy = scene_state["day_ground_copy"]
                plain_copy.cell_data["day_surface_class"] = np.zeros(
                    plain_copy.n_cells, dtype=np.uint8
                )
                prev = scene_state.get("ground_actor")
                if prev is not None:
                    plotter.remove_actor(prev, reset_camera=False)
                scene_state["ground_actor"] = plotter.add_mesh(
                    plain_copy,
                    scalars="day_surface_class",
                    clim=[0, 5],
                    cmap=day_surface_cmap,
                    show_edges=False,
                    opacity=0.9,
                )
                plotter.add_text(
                    "Move the Hour slider to compute shadows",
                    position=(0.18, 0.02),
                    name="status",
                    font_size=9,
                )
                plotter.update(stime=1)
            except Exception as _exc:
                import traceback as _tb
                print(f"[init-render ERROR] {_exc}")
                _tb.print_exc()

        plotter.add_timer_event(max_steps=1, duration=500, callback=_deferred_initial_render)
        _stage("Calling plotter.show() — window should appear now")
        plotter.show(auto_close=False, interactive_update=False)
        print("[viewer] plotter.show() returned — window closed by user")
    except Exception as exc:
        print(f"Visualization failed: {exc}")

if __name__ == "__main__":
    main()