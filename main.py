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
import concurrent.futures as cf
from pathlib import Path

import networkx as nx
import numpy as np
import pyvista as pv
import warnings
try:
    warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)
except AttributeError:
    warnings.filterwarnings("ignore", message=".*extract_surface.*")
import vtk

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


def _load_obj_without_render_window(
    _plotter: "pv.Plotter", obj_path: str, _mtl_path: str, _texture_dir: str
) -> list[tuple["pv.PolyData", "vtk.vtkProperty", "vtk.vtkTexture | None"]]:
    import vtk
    import pyvista as pv

    def _property(rgb: tuple[float, float, float]) -> "vtk.vtkProperty":
        prop = vtk.vtkProperty()
        prop.SetColor(*rgb)
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(24.0)
        try:
            prop.SetInterpolationToPhong()
        except Exception:
            pass
        return prop

    def _fix_normals(mesh: "pv.PolyData") -> None:
        if mesh.point_data.active_normals_name == "None" or (
            mesh.point_data.active_normals_name
            and mesh.point_data.active_normals_name not in mesh.point_data
        ):
            try:
                mesh.point_data.active_normals_name = None
            except Exception:
                pass

    def _transform(mesh: "pv.PolyData") -> "pv.PolyData":
        _fix_normals(mesh)
        out = mesh.rotate_x(90.0, inplace=False)
        out = out.rotate_z(-90.0, inplace=False)
        out.points = out.points * 1.6
        return out

    material_colors = {
        "colormap": (0.12, 0.78, 0.36),
        "Mat_1": (0.03, 0.08, 0.10),
    }

    # Do not use vtkOBJImporter here. Even with off-screen rendering it creates a
    # vtkRenderWindow before the main plotter.show(), which can take Cocoa/NSApp
    # down a dead-end on macOS. Parse OBJ material groups directly instead.
    try:
        vertices: list[tuple[float, float, float]] = []
        faces_by_material: dict[str, list[list[int]]] = {}
        current_material = "default"
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as obj_file:
            for line in obj_file:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("usemtl "):
                    current_material = line.split(None, 1)[1].strip() or "default"
                elif line.startswith("f "):
                    ids: list[int] = []
                    for token in line.split()[1:]:
                        raw_idx = token.split("/", 1)[0]
                        if not raw_idx:
                            continue
                        idx = int(raw_idx)
                        if idx < 0:
                            idx = len(vertices) + idx + 1
                        ids.append(idx - 1)
                    if len(ids) >= 3:
                        faces_by_material.setdefault(current_material, []).append(ids)

        if vertices and faces_by_material:
            all_points = np.asarray(vertices, dtype=float)
            parts: list[tuple["pv.PolyData", "vtk.vtkProperty", "vtk.vtkTexture | None"]] = []
            for material_name, material_faces in faces_by_material.items():
                used = sorted({idx for face in material_faces for idx in face})
                remap = {old: new for new, old in enumerate(used)}
                points = all_points[np.asarray(used, dtype=np.int64)]
                faces: list[int] = []
                for face in material_faces:
                    faces.append(len(face))
                    faces.extend(remap[idx] for idx in face)
                mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
                mesh = _transform(mesh)
                color = material_colors.get(material_name, (0.18, 0.80, 0.38))
                parts.append((mesh, _property(color), None))
            return parts
    except Exception as exc:
        print(f"[cars] OBJ material parse failed for {obj_path}: {exc}; falling back to pv.read")

    raw = pv.read(obj_path)
    if not isinstance(raw, pv.PolyData):
        raw = raw.extract_geometry()
    return [(_transform(raw), _property((0.18, 0.80, 0.38)), None)]


def _smart_light_positions(
    ground_mesh: "pv.PolyData",
    street_graph: "nx.MultiDiGraph",
    n_lights: int,
    grid_step: float,
    seed: int,
) -> np.ndarray:
    """Greedy streetlight layout: prefer sidewalk/road candidates with even spacing.

    This is intentionally simple and deterministic.  It starts near dense road
    geometry, then repeatedly picks candidates that are far from already chosen
    lights while still favoring road-node density.
    """
    from app_core import _build_grid_points_from_ground_mesh

    sidewalk_polygon = None
    try:
        sidewalk_polygon = build_sidewalk_polygon_from_street_graph(street_graph)
    except Exception:
        sidewalk_polygon = None

    step = max(4.0, float(grid_step))
    candidates = _build_grid_points_from_ground_mesh(
        ground_mesh,
        step,
        sidewalk_polygon=sidewalk_polygon,
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
    args.n_cars = int(args.n_cars)
    args.car_detail = str(getattr(args, "car_detail", "ultra")).strip().lower()
    if args.car_detail not in {"ultra", "low"}:
        args.car_detail = "ultra"
    args.traffic_speed = max(0.0, float(args.traffic_speed))
    args.debug_cars = bool(getattr(args, "debug_cars", False))
    args.solar_fleet = bool(getattr(args, "solar_fleet", False))
    args.solo = bool(getattr(args, "solo", False))
    args.light_strategy = str(getattr(args, "light_strategy", "smart")).strip().lower()
    if args.light_strategy not in {"smart", "ga"}:
        args.light_strategy = "smart"
    if args.solo:
        args.n_cars = 1
        args.car_detail = "ultra"


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
        f"light_strategy={args.light_strategy}, "
        f"car_detail={args.car_detail}, traffic_speed={args.traffic_speed:.2f}, "
        f"solar_fleet={args.solar_fleet}, debug_cars={args.debug_cars}, "
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

    def _compute_smart_lights() -> np.ndarray:
        _stage("Running smart streetlight placement...")
        return _smart_light_positions(
            ground_mesh=ground_mesh,
            street_graph=street_graph,
            n_lights=args.n_lights,
            grid_step=args.grid_step,
            seed=args.seed,
        )

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

            if args.light_strategy == "smart":
                best_positions = np.asarray(_compute_smart_lights(), dtype=float)
                print(f"Smart light placement complete: {best_positions.shape[0]} lights")
            else:
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
            if args.light_strategy == "smart":
                best_positions = _compute_smart_lights()
            else:
                best_positions = _initial_light_positions(ground_mesh, args.n_lights, args.seed)

        plotter = pv.Plotter(title="City Digital Twin", window_size=[1400, 900])

        def _pump_window_events() -> None:
            """Process pending window-manager events (minimize, resize, close)."""
            try:
                iren = plotter.iren
                if iren is not None:
                    iren.ProcessEvents()
                    return
            except Exception:
                pass
            try:
                rw = plotter.render_window
                if rw is not None and hasattr(rw, "ProcessEvents"):
                    rw.ProcessEvents()
            except Exception:
                pass
        _stage("Plotter created")

        style_presets: dict[str, dict[str, object]] = {
            "mini": {
                "day_bg": ["#cce6ff", "#1e3d59"],
                "night_bg": ["#02060c", "#0d1b2a"],
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
                "day_bg": ["#b2ebf2", "#006064"],
                "night_bg": ["#000a12", "#002030"],
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
                "day_bg": ["#ffcc80", "#4a148c"],
                "night_bg": ["#05000a", "#1a002c"],
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
            "show_streetlights": True,
            "show_traffic_signals": True,
            "preset": "mini",
            "building_actor": None,
            "vehicle_actor": None,
            "ped_actor": None,
            "car_actors": {},
            "interactive_ready": False,
            "poi_actor": None,
            "poi_labels_actor": None,
            "show_pois": False,
            "show_poi_names": False,
            "scene_lat": float(street_graph.graph.get("scene_lat", np.nan)),
            "scene_lon": float(street_graph.graph.get("scene_lon", np.nan)),
            "route_alpha": 0.5,
            "route_hour": 12.0,
            "solar_params": SolarParams(),
            "solar_fleet": bool(args.solar_fleet),
            "solar_model_idx": None,
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

                # If the edge is explicitly marked one-way, only accept the canonical
                # direction (u → v as stored in the graph).  The graph was built so that
                # one-way edges only have u→v; this guard handles any leakage.
                if data.get("oneway") is True:
                    # Canonical direction is u→v; the graph should not have v→u, but if
                    # a bidirectional duplicate snuck in check the geometry orientation.
                    geom_check = data.get("geometry")
                    if geom_check is not None and hasattr(geom_check, "coords"):
                        _coords = list(geom_check.coords)
                        if len(_coords) >= 2:
                            nu_c = graph.nodes.get(u, {})
                            # If the geometry runs end→start relative to (u,v) node positions,
                            # this is the forbidden reverse copy — skip it.
                            _gx0, _gy0 = float(_coords[0][0]), float(_coords[0][1])
                            _nux = float(nu_c.get("x", _gx0))
                            _nuy = float(nu_c.get("y", _gy0))
                            _fwd_err = abs(_gx0 - _nux) + abs(_gy0 - _nuy)
                            _nv_c = graph.nodes.get(v, {})
                            _rev_err = (
                                abs(_gx0 - float(_nv_c.get("x", _gx0)))
                                + abs(_gy0 - float(_nv_c.get("y", _gy0)))
                            )
                            if _rev_err + 1e-3 < _fwd_err:
                                continue  # geometry runs v→u; this is the illegal reverse edge

                if bool(data.get("oneway", False)) and not bool(data.get("oneway_legal_forward", True)):
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

                from shapely.geometry import LineString
                raw_lanes = data.get("lanes", 1)
                if isinstance(raw_lanes, list):
                    raw_lanes = raw_lanes[0]
                try:
                    num_lanes = max(1, int(float(raw_lanes)))
                except Exception:
                    num_lanes = 1

                lane_width = 3.0
                maxspeed = float(np.clip(_parse_maxspeed(data.get("maxspeed")), 2.8, 41.7))
                seg_id = data.get("segment_id")

                base_line = LineString(points[:, :2])
                lane_indices = []

                for lane_i in range(num_lanes):
                    offset_dist = (lane_i - (num_lanes - 1) / 2.0) * lane_width
                    if abs(offset_dist) < 1e-3:
                        lane_points = points
                    else:
                        try:
                            # Positive is left, negative is right
                            offset_line = base_line.offset_curve(offset_dist)
                            if offset_line.is_empty:
                                lane_points = points
                            else:
                                if offset_line.geom_type == 'MultiLineString':
                                    offset_line = list(offset_line.geoms)[0]
                                off_xy = np.asarray(offset_line.coords)
                                lane_points = np.column_stack((off_xy, np.full((off_xy.shape[0],), float(z_level), dtype=float)))
                        except Exception:
                            lane_points = points

                    lane_seg_lengths = np.linalg.norm(lane_points[1:, :2] - lane_points[:-1, :2], axis=1)
                    lane_cum_len = np.concatenate(([0.0], np.cumsum(lane_seg_lengths, dtype=float)))
                    lane_total_len = float(lane_cum_len[-1])
                    if lane_total_len <= 1e-6:
                        continue

                    nv_data = graph.nodes.get(v, {})
                    idx = len(paths)
                    paths.append({
                        "u": u,
                        "v": v,
                        "points": lane_points,
                        "cum_len": lane_cum_len,
                        "length": lane_total_len,
                        "maxspeed_ms": maxspeed,
                        "segment_id": seg_id,
                        "lane_index": lane_i,
                        "num_lanes": num_lanes,
                        "adjacent_left_path": -1,
                        "adjacent_right_path": -1,
                        "junction": data.get("junction"),
                        "control": nv_data.get("highway")
                    })
                    outgoing.setdefault(u, []).append(idx)
                    lane_indices.append(idx)

                # Cross-link adjacent lanes
                for i, p_idx in enumerate(lane_indices):
                    if i > 0:
                        paths[p_idx]["adjacent_right_path"] = lane_indices[i - 1]
                    if i < len(lane_indices) - 1:
                        paths[p_idx]["adjacent_left_path"] = lane_indices[i + 1]

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
        # Build car_next_edges from the directed OSM-resolved graph. Overture
        # connector restrictions are ignored here because the car simulation
        # should follow OSM direction data only.
        car_next_edges = _build_next_edges(
            car_paths,
            car_outgoing,
            street_graph,
            include_overture=False,
        )
        
        roundabout_yield_map: dict[int, np.ndarray] = {}
        for p_idx, path in enumerate(car_paths):
            if path.get("junction") != "roundabout":
                for next_p_idx_raw in car_next_edges[p_idx]:
                    next_p = int(next_p_idx_raw)
                    if car_paths[next_p].get("junction") == "roundabout":
                        ring_paths = []
                        for rp_idx, rp in enumerate(car_paths):
                            if rp["v"] == path["v"] and rp.get("junction") == "roundabout":
                                ring_paths.append(rp_idx)
                        if ring_paths:
                            roundabout_yield_map[p_idx] = np.array(ring_paths, dtype=np.int64)
                            
        _stage("Car routing pre-computed (OSM directions only)")

        # ── Load OBJ car models for ultra detail mode ────────────────────────
        _CAR_MODELS_DIR = Path(__file__).parent / "assets" / "models"
        _SOLAR_CAR_OBJ = "solarcar.obj"
        _CAR_OBJ_FILES = [
            "sedan-sports.obj", "hatchback-sports.obj", "suv.obj",
            "van.obj", "delivery.obj", "truck-flat.obj",
            _SOLAR_CAR_OBJ,
        ]
        _SOLAR_CAR_COLOR = "#2ecc71"
        # Per-car body colors (varied fleet look)
        _CAR_BODY_COLORS = [
            "#c0392b", "#2980b9", "#27ae60", "#f39c12",
            "#8e44ad", "#e74c3c", "#3498db", "#16a085",
            "#d35400", "#2c3e50", "#1abc9c", "#e67e22",
        ]
        car_obj_templates: list = []
        car_obj_lengths: list[float] = []  # bumper-to-bumper length per template (metres)
        car_solar_model_idx: int | None = None
        if args.car_detail == "ultra":
            for _fname in _CAR_OBJ_FILES:
                _fpath = _CAR_MODELS_DIR / _fname
                if not _fpath.exists():
                    continue
                try:
                    if _fname == _SOLAR_CAR_OBJ:
                        _mtl_path = _CAR_MODELS_DIR / _fname.replace(".obj", ".mtl")
                        _parts = _load_obj_without_render_window(
                            plotter, str(_fpath), str(_mtl_path), str(_CAR_MODELS_DIR)
                        )
                        _n_pts = sum(p[0].n_points for p in _parts)
                        _min_x = min(p[0].bounds[0] for p in _parts)
                        _max_x = max(p[0].bounds[1] for p in _parts)
                        _car_len = float(_max_x - _min_x)
                        car_solar_model_idx = len(car_obj_templates)
                        car_obj_templates.append(_parts)
                        car_obj_lengths.append(_car_len)
                        print(f"[cars] loaded solarcar with {len(_parts)} materials ({_n_pts} pts, length={_car_len:.2f} m)")
                        continue

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
        scene_state["solar_model_idx"] = car_solar_model_idx
        if bool(args.solar_fleet):
            if car_solar_model_idx is None:
                print("[cars] --solar-fleet requested but solarcar.obj missing; disabling solar fleet")
                args.solar_fleet = False
                scene_state["solar_fleet"] = False
            else:
                print(f"[cars] solar fleet: all cars use {_SOLAR_CAR_OBJ} (template index {car_solar_model_idx})")
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

        n_cars = int(args.n_cars)
        n_parked_arg = int(getattr(args, "n_parked_cars", -1))
        
        # Automatic density scaling
        if n_cars == -1 or n_parked_arg == -1:
            total_buildings = buildings_mesh.n_cells if buildings_mesh is not None else 0
            
            if n_cars == -1:
                n_cars = max(1, int(total_buildings * 0.05))
                n_cars = min(60, n_cars) # Cap active cars to protect CPU
                
            if n_parked_arg == -1:
                args.n_parked_cars = max(1, int(total_buildings * 0.25)) # Pass this safely for parked cars logic
                
        n_cars = max(0, n_cars)

        car_rng = np.random.default_rng(int(args.seed) + 2027)
        car_anim: dict[str, object] = {
            "enabled": bool(n_cars > 0 and len(car_paths) > 0),
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
            res_nodes = [n for n, d in street_graph.nodes(data=True) if d.get("is_residential")]
            com_nodes = [n for n, d in street_graph.nodes(data=True) if d.get("is_commercial")]
            
            if not com_nodes and places:
                try:
                    from scipy.spatial import cKDTree
                    n_ids = list(street_graph.nodes)
                    n_pts = np.array([[float(street_graph.nodes[n]["x"]), float(street_graph.nodes[n]["y"])] for n in n_ids])
                    tree = cKDTree(n_pts)
                    p_pts = np.array([[float(p["x"]), float(p["y"])] for p in places])
                    _, indices = tree.query(p_pts)
                    com_nodes = [n_ids[i] for i in indices]
                except Exception:
                    pass
            if not res_nodes:
                res_nodes = list(street_graph.nodes)
                
            planned_edges = [None for _ in range(n_cars)]
            planned_cursor = np.zeros(n_cars, dtype=np.int64)
            edge_map = {(p["u"], p["v"]): idx for idx, p in enumerate(car_paths)}
            
            if res_nodes and com_nodes:
                for c in range(n_cars):
                    src = car_rng.choice(res_nodes)
                    dst = car_rng.choice(com_nodes)
                    try:
                        path_nodes = nx.shortest_path(street_graph, src, dst, weight="length")
                        plan = []
                        for i in range(len(path_nodes)-1):
                            uv = (path_nodes[i], path_nodes[i+1])
                            if uv in edge_map: plan.append(edge_map[uv])
                            else: break
                        if plan:
                            planned_edges[c] = plan
                            edge_idx[c] = plan[0]
                            dist[c] = 0.0
                    except nx.NetworkXNoPath:
                        pass
                        
            car_anim["planned_edges"] = planned_edges
            car_anim["planned_cursor"] = planned_cursor
            car_anim["edge_idx"] = np.asarray(edge_idx, dtype=np.int64)
            car_anim["dist"] = dist
            
            # Assign a fixed OBJ model index per car (for ultra mode)
            if car_obj_templates:
                if bool(args.solar_fleet) and car_solar_model_idx is not None:
                    car_anim["model_idx"] = np.full(n_cars, car_solar_model_idx, dtype=np.int64)
                else:
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
            
            car_anim["edge_idx"] = np.zeros(n_cars, dtype=np.int64)
            car_anim["dist"] = np.zeros(n_cars, dtype=float)
            car_anim["speed"] = np.zeros(n_cars, dtype=float)
            car_anim["stop_wait"] = np.zeros(n_cars, dtype=float)
            car_anim["desired_speed"] = np.zeros(n_cars, dtype=float)
            
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
                roundabout_yield_map = roundabout_yield_map,
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
            n_paths = len(car_paths)
            positions = np.zeros((n, 3), dtype=float)
            if n_paths == 0:
                return positions
            safe_e = np.clip(edge_idx, 0, n_paths - 1)
            for i in range(n):
                positions[i] = _car_pose_on_path(car_paths[int(safe_e[i])], float(dist[i]))
            return positions

        def _sample_car_headings() -> np.ndarray:
            edge_idx = np.asarray(car_anim["edge_idx"], dtype=np.int64)
            dist = np.asarray(car_anim["dist"], dtype=float)
            n = edge_idx.shape[0]
            n_paths = len(car_paths)
            headings = np.zeros(n, dtype=float)
            if n_paths == 0:
                return headings
            safe_e = np.clip(edge_idx, 0, n_paths - 1)
            for i in range(n):
                headings[i] = _car_heading_deg_on_path(car_paths[int(safe_e[i])], float(dist[i]))
            return headings

        def _render_cars() -> None:
            actors = scene_state.get("car_actors")
            if not isinstance(actors, dict):
                actors = {}
                scene_state["car_actors"] = actors

            is_first = actors.get("cars") is None and not scene_state.get("_ultra_car_actors")
            if is_first:
                print("[cars] _render_cars: FIRST CALL — creating actor")
                if args.car_detail == "ultra" and car_obj_templates and places and len(car_paths) > 0:
                    parked_actors = []
                    import random
                    res_places = [p for p in places if "residential" in p.get("categories", [])]
                    pool = res_places if res_places else places
                    
                    target_parked = int(getattr(args, "n_parked_cars", len(pool)))
                    if target_parked == -1: target_parked = len(pool)
                    
                    n_parked = min(len(pool), target_parked, 400) # hard cap at 400 actors to prevent VTK stutter
                    parked_places = random.sample(pool, n_parked)
                    
                    try:
                        from scipy.spatial import cKDTree
                        all_points = []
                        point_to_path = []
                        for p_idx, path in enumerate(car_paths):
                            pts = path["points"]
                            all_points.append(pts[:, :2])
                            point_to_path.extend([(p_idx, i) for i in range(len(pts))])
                        
                        if all_points:
                            all_points = np.vstack(all_points)
                            tree = cKDTree(all_points)
                            
                            for p in parked_places:
                                px, py = float(p["x"]), float(p["y"])
                                _, idx = tree.query([px, py])
                                path_idx, pt_idx = point_to_path[idx]
                                pts = car_paths[path_idx]["points"]
                                
                                if pt_idx < len(pts) - 1:
                                    p1, p2 = pts[pt_idx], pts[pt_idx+1]
                                else:
                                    p1, p2 = pts[pt_idx-1], pts[pt_idx]
                                    
                                rx, ry, rz = float(p1[0]), float(p1[1]), float(p1[2])
                                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                                heading = np.degrees(np.arctan2(dy, dx))
                                
                                norm = np.linalg.norm([dx, dy])
                                if norm > 1e-6:
                                    nx, ny = dy / norm, -dx / norm
                                else:
                                    nx, ny = 1.0, 0.0
                                    
                                if (px - rx) * nx + (py - ry) * ny < 0:
                                    nx, ny = -nx, -ny
                                
                                px_off = rx + nx * 2.5
                                py_off = ry + ny * 2.5
                                
                                mid = random.randint(0, len(car_obj_templates)-1)
                                tmpl = car_obj_templates[mid]
                                color = _CAR_BODY_COLORS[mid % len(_CAR_BODY_COLORS)]
                                
                                if isinstance(tmpl, list):
                                    import vtk
                                    _assembly = vtk.vtkAssembly()
                                    for _poly, _prop, _tex in tmpl:
                                        _mapper = vtk.vtkPolyDataMapper()
                                        _mapper.SetInputData(_poly)
                                        _part_act = vtk.vtkActor()
                                        _part_act.SetMapper(_mapper)
                                        _part_act.SetProperty(_prop)
                                        if _tex: _part_act.SetTexture(_tex)
                                        _assembly.AddPart(_part_act)
                                    plotter.add_actor(_assembly)
                                    _actor = _assembly
                                else:
                                    _actor = plotter.add_mesh(tmpl.copy(), color=color, smooth_shading=True, lighting=True, reset_camera=False)
                                _actor.SetPosition(px_off, py_off, rz)
                                _actor.SetOrientation(0.0, 0.0, heading)
                                parked_actors.append(_actor)
                        scene_state["_parked_car_actors"] = parked_actors
                        print(f"[cars] created {len(parked_actors)} parked cars")
                    except Exception as e:
                        print(f"[cars] Failed to create parked cars: {e}")

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
                
                for _pa in scene_state.get("_parked_car_actors") or []:
                    try:
                        plotter.remove_actor(_pa, reset_camera=False)
                    except Exception:
                        pass
                scene_state["_parked_car_actors"] = None
                
                return

            positions = _sample_car_positions()
            if positions.shape[0] == 0:
                return
            car_anim["pos"] = positions
            try:
                _update_selected_car_marker()
            except NameError:
                pass

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
                        mid = int(model_idx[i]) % len(car_obj_templates)
                        tmpl = car_obj_templates[mid]
                        if car_solar_model_idx is not None and mid == car_solar_model_idx:
                            color = _SOLAR_CAR_COLOR
                        else:
                            color = _CAR_BODY_COLORS[i % len(_CAR_BODY_COLORS)]
                        try:
                            if isinstance(tmpl, list):
                                import vtk
                                _assembly = vtk.vtkAssembly()
                                for _poly, _prop, _tex in tmpl:
                                    _mapper = vtk.vtkPolyDataMapper()
                                    _mapper.SetInputData(_poly)
                                    _part_act = vtk.vtkActor()
                                    _part_act.SetMapper(_mapper)
                                    _part_act.SetProperty(_prop)
                                    if _tex:
                                        _part_act.SetTexture(_tex)
                                    _assembly.AddPart(_part_act)
                                plotter.add_actor(_assembly)
                                _actor = _assembly
                            else:
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
                    _last_h = scene_state.get("_ultra_car_headings")
                    if not isinstance(_last_h, np.ndarray) or _last_h.shape[0] != positions.shape[0]:
                        _last_h = np.full(positions.shape[0], np.nan, dtype=float)
                    for i, _actor in enumerate(ultra_actors):
                        if i >= positions.shape[0]:
                            break
                        try:
                            _actor.SetPosition(
                                float(positions[i, 0]),
                                float(positions[i, 1]),
                                float(positions[i, 2]),
                            )
                            h = float(headings[i])
                            if not np.isfinite(_last_h[i]) or abs(h - float(_last_h[i])) > 2.0:
                                _actor.SetOrientation(0.0, 0.0, h)
                                _last_h[i] = h
                        except Exception as _exc:
                            if not getattr(_render_cars, "_vtk_warned", False):
                                print(f"[cars] VTK transform update failed: {_exc}")
                                _render_cars._vtk_warned = True
                    scene_state["_ultra_car_headings"] = _last_h

                show = bool(scene_state["show_cars"])
                for _actor in ultra_actors:
                    _set_actor_visibility(_actor, show)
            else:
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

            def _car_pick_callback(mesh, cell_id) -> None:
                if mesh is None or cell_id < 0:
                    return
                try:
                    cx = float(mesh.cell_centers().points[cell_id, 0])
                    cy = float(mesh.cell_centers().points[cell_id, 1])
                    pos_array = np.asarray(car_anim.get("pos", []))
                    if pos_array.shape[0] == 0:
                        return
                    pos_array = pos_array[:, :2]
                    dist_sq = np.sum((pos_array - np.array([cx, cy]))**2, axis=1)
                    car_idx = int(np.argmin(dist_sq))
                    _e = int(car_anim["edge_idx"][car_idx])
                    _d = float(car_anim["dist"][car_idx])
                    _p3d = _car_pose_on_path(car_paths[_e], _d)
                    _snap_node = nearest_graph_node(street_graph, _p3d[0], _p3d[1])
                    
                    _clear_route_actors()
                    route_state["source_node"] = _snap_node
                    _xy = _node_xy(_snap_node)
                    if _xy is not None:
                        _src_actor = plotter.add_mesh(
                            pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0), theta_resolution=18, phi_resolution=18),
                            color="#33cc66",
                            render=False,
                        )
                        route_state["route_actors"] = [_src_actor]
                    _select_car_for_route(car_idx, _snap_node)
                    route_state["stage"] = 1
                    print(f"[route] Car {car_idx} selected as source — click target node.")
                    
                    if getattr(args, "solo", False):
                        scene_state["solo_solar_Wh"] = 0.0
                        scene_state["solo_mech_Wh"] = 0.0
                except Exception as _exc:
                    print(f"[cars] Pick failed: {_exc}")



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
        bg = st.get("day_bg")
        solid_bg = str(bg[0] if isinstance(bg, list) else bg)
        plotter.set_background(solid_bg)
        _stage(f"Adding buildings mesh ({buildings_mesh.n_cells} cells)...")

        # PBR environment is refreshed in _render_ground() for time-of-day HDRI.
        renderer = plotter.renderer
        # Configure SSAO parameters but do NOT enable yet.
        # SSAO + hundreds of text-label actors causes VTK's first render to hang
        # on macOS Cocoa. We defer SetUseSSAO(True) to a timer after show().
        try:
            renderer.SetSSAORadius(4.0)
            renderer.SetSSAOBias(0.025)
            renderer.SetSSAOKernelSize(128)
            renderer.SetSSAOBlur(True)
            print("[ssao] SSAO configured (deferred activation)")
        except Exception:
            print("[ssao] SSAO unavailable")

        _PBR_CLASSES = {
            0: dict(name="concrete", color="#c8b89a", metallic=0.0, roughness=0.85),
            1: dict(name="brick",    color="#b5724a", metallic=0.0, roughness=0.90),
            2: dict(name="glass",       color="#8cb8d8", metallic=0.0, roughness=0.08),
            3: dict(name="commercial",  color="#3498db", metallic=0.0, roughness=0.6),
            4: dict(name="residential", color="#e74c3c", metallic=0.0, roughness=0.7),
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
                _set_actor_visibility(scene_state["tl_actor"], bool(scene_state["show_traffic_signals"]))
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

        # ── Map click: road info + route planning (source → target) ──────────
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

        route_state = {
                "stage": 0,
                "source_node": None,
                "target_node": None,
                "route_actors": [],
                "selected_car_idx": None,
                "selected_marker_actor": None,
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

        def _clear_selected_car(clear_plan: bool = False) -> None:
                idx = route_state.get("selected_car_idx")
                if clear_plan and idx is not None:
                    try:
                        plans = car_anim.get("planned_edges")
                        if isinstance(plans, list) and 0 <= int(idx) < len(plans):
                            plans[int(idx)] = None
                        cursors = car_anim.get("planned_cursor")
                        if isinstance(cursors, np.ndarray) and 0 <= int(idx) < cursors.shape[0]:
                            cursors[int(idx)] = 0
                    except Exception:
                        pass
                route_state["selected_car_idx"] = None
                marker = route_state.get("selected_marker_actor")
                if marker is not None:
                    try:
                        plotter.remove_actor(marker, reset_camera=False)
                    except Exception:
                        pass
                route_state["selected_marker_actor"] = None
                plotter.add_text("", position=(0.36, 0.93), name="selected_car_overlay", viewport=True)

        def _update_selected_car_marker() -> None:
                idx = route_state.get("selected_car_idx")
                if idx is None:
                    return
                try:
                    i = int(idx)
                    positions = np.asarray(car_anim.get("pos", []), dtype=float)
                    if positions.ndim != 2 or i < 0 or i >= positions.shape[0]:
                        return
                    pos = positions[i]
                    marker = route_state.get("selected_marker_actor")
                    if marker is None:
                        marker = plotter.add_mesh(
                            pv.Sphere(radius=3.2, center=(0.0, 0.0, 0.0), theta_resolution=24, phi_resolution=12),
                            color="#ffd400",
                            opacity=0.35,
                            render=False,
                            reset_camera=False,
                        )
                        route_state["selected_marker_actor"] = marker
                        marker.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]) + 1.8)
                    else:
                        marker.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]) + 1.8)
                except Exception:
                    pass

        def _select_car_for_route(car_idx: int, source_node: object) -> None:
                _clear_selected_car(clear_plan=False)
                route_state["selected_car_idx"] = int(car_idx)
                plotter.add_text(
                    f"Selected car #{int(car_idx)} - click target road node",
                    position=(0.36, 0.93),
                    name="selected_car_overlay",
                    font_size=10,
                    color="#ffd400",
                    viewport=True,
                )
                _update_selected_car_marker()

        def _route_nodes_to_car_edges(_nodes: list[object]) -> list[int]:
                if len(_nodes) < 2:
                    return []
                by_uv: dict[tuple[object, object], list[int]] = {}
                for _i, _path in enumerate(car_paths):
                    by_uv.setdefault((_path.get("u"), _path.get("v")), []).append(_i)
                edges: list[int] = []
                for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                    matches = by_uv.get((_a, _b), [])
                    if not matches:
                        return []
                    edges.append(min(matches, key=lambda _idx: float(car_paths[int(_idx)].get("length", np.inf))))
                return edges

        def _assign_selected_car_route(_nodes: list[object]) -> None:
                idx = route_state.get("selected_car_idx")
                if idx is None or len(_nodes) < 2:
                    return
                try:
                    car_idx = int(idx)
                    edges = _route_nodes_to_car_edges(list(_nodes))
                    if not edges:
                        print("[route] Selected car route could not be mapped to drivable edges")
                        return
                    plans = car_anim.get("planned_edges")
                    cursors = car_anim.get("planned_cursor")
                    if not isinstance(plans, list) or car_idx < 0 or car_idx >= len(plans):
                        return
                    if not isinstance(cursors, np.ndarray) or car_idx >= cursors.shape[0]:
                        return
                    current_edge = int(edges[0])
                    car_anim["edge_idx"][car_idx] = current_edge
                    car_anim["dist"][car_idx] = min(
                        float(car_anim["dist"][car_idx]),
                        max(0.0, float(car_paths[current_edge]["length"]) - 1e-6),
                    )
                    base_speed = float(car_paths[current_edge]["maxspeed_ms"]) * 0.85
                    if "desired_speed_base" in car_anim:
                        car_anim["desired_speed_base"][car_idx] = base_speed
                    if "desired_speed" in car_anim:
                        car_anim["desired_speed"][car_idx] = base_speed * float(args.traffic_speed)
                    if "speed" in car_anim:
                        car_anim["speed"][car_idx] = min(
                            float(car_anim["speed"][car_idx]),
                            base_speed * float(args.traffic_speed),
                        )
                    plans[car_idx] = np.asarray(edges, dtype=np.int64)
                    cursors[car_idx] = 0
                    route_state["selected_route_nodes"] = list(_nodes)
                    plotter.add_text(
                        f"Selected car #{car_idx} following planned route ({len(edges)} road segments)",
                        position=(0.36, 0.93),
                        name="selected_car_overlay",
                        font_size=10,
                        color="#ffd400",
                        viewport=True,
                    )
                    print(f"[route] Car {car_idx} assigned to planned route with {len(edges)} edges")
                except Exception as _exc:
                    print(f"[route] Could not assign selected car route: {_exc}")

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
                if not bool(scene_state.get("solar_fleet", False)):
                    return
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
                        position=(0.55, 0.04),
                        name="route_stats_overlay",
                        font_size=8,
                        color="#f4f4f4",
                        viewport=True,
                    )
                    return

                _use_solar = bool(scene_state.get("solar_fleet", False))

                def _line(_name: str, _s: dict[str, float]) -> str:
                    _net = float(_s.get("net_energy_J", 0.0))
                    _sign = "▼" if _net <= 0.0 else "▲"
                    base = (
                        f"{_name:<8}  d={_s.get('distance_m', 0.0):7.1f} m  "
                        f"t={_s.get('travel_time_s', 0.0) / 60.0:6.2f} min  "
                        f"mech={_s.get('mechanical_J', 0.0) / 3600.0:7.2f} Wh  "
                    )
                    if _use_solar:
                        return (
                            f"{base}"
                            f"solar={_s.get('solar_J', 0.0) / 3600.0:7.2f} Wh  "
                            f"net={_net / 3600.0:7.2f} Wh {_sign}"
                        )
                    return f"{base}net={_net / 3600.0:7.2f} Wh {_sign}  (no solar)"

                txt = "\n".join([
                    _line("Energy", _summary.get("energy", {})),
                    _line("Joint", _summary.get("joint", {})),
                    _line("Shortest", _summary.get("shortest", {})),
                ])
                plotter.add_text(
                    txt,
                    position=(0.55, 0.04),
                    name="route_stats_overlay",
                    font_size=8,
                    color="#f4f4f4",
                    viewport=True,
                )

        def _cost_cache_key(
                _hour: float,
                _alpha: float,
                _params: SolarParams,
                _use_solar: bool,
        ) -> tuple[float, ...]:
                return (
                    bool(_use_solar),
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
                _use_solar = bool(scene_state.get("solar_fleet", False))

                if _use_solar:
                    _shadow = _build_edge_shadow_cache(_route_hour)
                else:
                    _shadow = np.ones((len(car_paths),), dtype=float)

                _cost_cache = scene_state.get("edge_costs_cache")
                if not isinstance(_cost_cache, dict):
                    _cost_cache = {}
                    scene_state["edge_costs_cache"] = _cost_cache

                edge_shadow_frac = np.asarray(_shadow, dtype=float)
                # Include a shadow fingerprint so the cache is invalidated whenever
                # edge_shadow_frac changes (e.g. after the time-of-day slider moves).
                _shadow_fp = int(np.sum(edge_shadow_frac * 1000).round())  # cheap hash
                _costs_key = (
                    round(float(scene_state.get("route_hour", _route_hour)), 2),
                    round(float(scene_state.get("route_alpha", _alpha)), 3),
                    _shadow_fp,
                )
                _cached_costs = _cost_cache.get(_costs_key)
                if _cached_costs is None:
                    _cached_costs = build_edge_costs(
                        car_paths=car_paths,
                        edge_shadow_frac=edge_shadow_frac,
                        lat_deg=_lat,
                        lon_deg=_lon,
                        hour_local=_route_hour,
                        params=_params,
                        alpha=_alpha,
                        use_solar=_use_solar,
                    )
                    _cost_cache[_costs_key] = _cached_costs

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
                    if route_state.get("selected_car_idx") is not None and len(_joint_nodes) >= 2:
                        _assign_selected_car_route(_joint_nodes)
                    _update_pareto_chart(_g_cost, _src, _tgt)

        def _road_pick_callback(point) -> None:
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
                )

                # Click route workflow: 1st click = source, 2nd = target, 3rd = reset.
                try:
                    nearest_geom = best["geom"].interpolate(best["geom"].project(pt))
                    _snap_node = nearest_graph_node(street_graph, nearest_geom.x, nearest_geom.y)
                    if route_state["stage"] == 1 and _snap_node == route_state["source_node"]:
                        _nodes_sorted = sorted(
                            street_graph.nodes(data=True),
                            key=lambda n: (float(n[1]["x"]) - nearest_geom.x)**2 + (float(n[1]["y"]) - nearest_geom.y)**2
                        )
                        for n_id, _ in _nodes_sorted:
                            if n_id != route_state["source_node"]:
                                _snap_node = n_id
                                print(f"[route] snapped to second-nearest node {_snap_node} (source node was the closest)")
                                break
                except Exception as _exc:
                    print(f"[route] Node snap failed: {_exc}")
                    return

                    if route_state["stage"] == 2:
                        _clear_route_actors()
                        _remove_pareto_chart()
                        _clear_selected_car(clear_plan=True)
                        route_state["stage"] = 0
                    route_state["source_node"] = None
                    route_state["target_node"] = None
                    _update_route_stats_overlay({})
                    print("[route] Route selection reset — click again for a new source")
                    return

                _xy = _node_xy(_snap_node)
                if _xy is None:
                    print(f"[route] Node {_snap_node} has no local coordinates")
                    return

                    if route_state["stage"] == 0:
                        _clear_route_actors()
                        _clear_selected_car(clear_plan=True)
                        route_state["source_node"] = _snap_node
                    _src_actor = plotter.add_mesh(
                        pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0), theta_resolution=18, phi_resolution=18),
                        color="#33cc66",
                        render=False,
                    )
                    route_state["route_actors"] = [_src_actor]
                    route_state["stage"] = 1
                    print(f"[route] Source set: node {_snap_node} — click target")
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

        plotter.add_key_event(
                "Escape",
                lambda: plotter.add_text("", position=(0.68, 0.90), name="road_info_overlay"),
        )
        def _rebuild_traffic_and_arrows() -> None:
            """Re-extracts paths, redraws arrows, and resets IDM state without replacing ground mesh."""
            nonlocal car_paths, car_outgoing, car_next_edges, car_path_lengths
            
            _stage("Rebuilding traffic and arrows...")
            car_paths, car_outgoing = _extract_drivable_paths(street_graph, z_level=0.5)
            car_path_lengths = np.asarray([float(p["length"]) for p in car_paths], dtype=float)
            
            car_next_edges = _build_next_edges(
                car_paths,
                car_outgoing,
                street_graph,
                include_overture=False,
            )
            
            nonlocal roundabout_yield_map
            roundabout_yield_map = {}
            for p_idx, path in enumerate(car_paths):
                if path.get("junction") != "roundabout":
                    for next_p_idx_raw in car_next_edges[p_idx]:
                        next_p = int(next_p_idx_raw)
                        if car_paths[next_p].get("junction") == "roundabout":
                            ring_paths = []
                            for rp_idx, rp in enumerate(car_paths):
                                if rp["v"] == path["v"] and rp.get("junction") == "roundabout":
                                    ring_paths.append(rp_idx)
                            if ring_paths:
                                roundabout_yield_map[p_idx] = np.array(ring_paths, dtype=np.int64)
                                
            if scene_state.get("street_arrows_actor") is not None:
                plotter.remove_actor(scene_state["street_arrows_actor"])
            arrows = _street_direction_arrows(street_graph)
            if arrows is not None:
                scene_state["street_arrows_actor"] = plotter.add_mesh(
                    arrows, color="#00e5ff", opacity=0.7, smooth_shading=False, lighting=False, render=False
                )
                _set_actor_visibility(scene_state["street_arrows_actor"], scene_state.get("show_arrows", True))
                
            # Render Stop Signs
            if scene_state.get("stop_signs_actor") is not None:
                plotter.remove_actor(scene_state["stop_signs_actor"])
            stop_nodes = [n for n, d in street_graph.nodes(data=True) if d.get("highway") == "stop"]
            if stop_nodes:
                stop_meshes = []
                for n in stop_nodes:
                    ndata = street_graph.nodes[n]
                    nx, ny = float(ndata.get("x", 0)), float(ndata.get("y", 0))
                    # Red octagon (8-sided cylinder)
                    cyl = pv.Cylinder(center=(nx, ny, 1.5), direction=(0, 0, 1), radius=1.2, height=0.3, resolution=8)
                    stop_meshes.append(cyl)
                
                merged_stops = stop_meshes[0]
                for i in range(1, len(stop_meshes)):
                    merged_stops = merged_stops.merge(stop_meshes[i])
                scene_state["stop_signs_actor"] = plotter.add_mesh(
                    merged_stops, color="#cc0000", smooth_shading=False, lighting=True, render=False
                )
            else:
                scene_state["stop_signs_actor"] = None
            
            n = len(car_anim["edge_idx"])
            car_anim["edge_idx"] = np.zeros(n, dtype=np.int64)
            car_anim["dist"] = np.zeros(n, dtype=float)
            car_anim["speed"] = np.zeros(n, dtype=float)
            car_anim["stop_wait"] = np.zeros(n, dtype=float)
            car_anim["planned_edges"] = [None for _ in range(n)]
            car_anim["planned_cursor"] = np.zeros(n, dtype=np.int64)
            
            plotter.render()
            print("[editor] Traffic logic and arrows rebuilt successfully")

        # Editor Mode State
        scene_state["editor_mode"] = "view"
        
        def _set_editor_mode(mode: str) -> None:
            scene_state["editor_mode"] = mode
            modes = ["view", "roads", "roundabouts", "lights", "stops", "streetlights"]
            text = "  |  ".join(f"[{i+1}] {m.capitalize()}" for i, m in enumerate(modes))
            plotter.add_text(f"Mode: {mode.upper()}   |   {text}", position=(10, 10), name="editor_mode_overlay", font_size=12, color="white")
            print(f"[editor] Switched to mode: {mode}")

        _set_editor_mode("view")
        plotter.add_key_event("1", lambda: _set_editor_mode("view"))
        plotter.add_key_event("2", lambda: _set_editor_mode("roads"))
        plotter.add_key_event("3", lambda: _set_editor_mode("roundabouts"))
        plotter.add_key_event("4", lambda: _set_editor_mode("lights"))
        plotter.add_key_event("5", lambda: _set_editor_mode("stops"))
        plotter.add_key_event("6", lambda: _set_editor_mode("streetlights"))

        def _unified_pick_callback(point, picker=None):
            """Single surface picker — handles road info, car-click routing, and editor mode."""
            mode = scene_state.get("editor_mode", "view")
            
            if mode == "roads":
                try:
                    import osmnx as ox
                    from shapely.geometry import LineString
                    px, py = float(point[0]), float(point[1])
                    u, v, key = ox.nearest_edges(street_graph, px, py)
                    data = street_graph.get_edge_data(u, v, key).copy()
                    
                    if "geometry" in data:
                        data["geometry"] = LineString(list(data["geometry"].coords)[::-1])
                    
                    # Flip direction tags if present
                    if "oneway" in data:
                        data["oneway"] = True
                    data["oneway_legal_forward"] = not data.get("oneway_legal_forward", True)
                        
                    street_graph.remove_edge(u, v, key=key)
                    street_graph.add_edge(v, u, key=key, **data)
                    
                    print(f"[editor] Reversed road edge ({u} -> {v}) to ({v} -> {u})")
                    _rebuild_traffic_and_arrows()
                except Exception as exc:
                    print(f"[editor] Road reversal failed: {exc}")
                return
            elif mode == "roundabouts":
                try:
                    import osmnx as ox
                    from shapely.geometry import LineString
                    import uuid
                    px, py = float(point[0]), float(point[1])
                    _snap_node = nearest_graph_node(street_graph, px, py)
                    
                    cx, cy = float(street_graph.nodes[_snap_node]["x"]), float(street_graph.nodes[_snap_node]["y"])
                    radius = 14.0
                    n_pts = 8
                    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
                    ring_nodes = []
                    for i, a in enumerate(angles):
                        nx_ = cx + radius * np.cos(a)
                        ny_ = cy + radius * np.sin(a)
                        n_id = f"ra_{_snap_node}_{i}"
                        street_graph.add_node(n_id, x=nx_, y=ny_, osmid=n_id)
                        ring_nodes.append(n_id)
                        
                    for i in range(n_pts):
                        u = ring_nodes[i]
                        v = ring_nodes[(i+1)%n_pts]
                        geom = LineString([(street_graph.nodes[u]["x"], street_graph.nodes[u]["y"]), 
                                           (street_graph.nodes[v]["x"], street_graph.nodes[v]["y"])])
                        street_graph.add_edge(u, v, key=0, length=geom.length, geometry=geom, oneway=True, junction="roundabout")
                        
                    in_edges = list(street_graph.in_edges(_snap_node, data=True, keys=True))
                    out_edges = list(street_graph.out_edges(_snap_node, data=True, keys=True))
                    
                    def _nearest_ring_node(x, y):
                        dists = [( (street_graph.nodes[rn]["x"]-x)**2 + (street_graph.nodes[rn]["y"]-y)**2, rn ) for rn in ring_nodes]
                        return min(dists)[1]
                        
                    for u, _, k, d in in_edges:
                        if u == _snap_node: continue
                        ux, uy = float(street_graph.nodes[u]["x"]), float(street_graph.nodes[u]["y"])
                        rn = _nearest_ring_node(ux, uy)
                        if "geometry" in d:
                            coords = list(d["geometry"].coords)
                            coords[-1] = (float(street_graph.nodes[rn]["x"]), float(street_graph.nodes[rn]["y"]))
                            d["geometry"] = LineString(coords)
                            d["length"] = d["geometry"].length
                        street_graph.remove_edge(u, _snap_node, key=k)
                        street_graph.add_edge(u, rn, key=k, **d)
                        
                    for _, v, k, d in out_edges:
                        if v == _snap_node: continue
                        vx, vy = float(street_graph.nodes[v]["x"]), float(street_graph.nodes[v]["y"])
                        rn = _nearest_ring_node(vx, vy)
                        if "geometry" in d:
                            coords = list(d["geometry"].coords)
                            coords[0] = (float(street_graph.nodes[rn]["x"]), float(street_graph.nodes[rn]["y"]))
                            d["geometry"] = LineString(coords)
                            d["length"] = d["geometry"].length
                        street_graph.remove_edge(_snap_node, v, key=k)
                        street_graph.add_edge(rn, v, key=k, **d)
                        
                    street_graph.remove_node(_snap_node)
                    print(f"[editor] Generated roundabout at node {_snap_node}")
                    
                    # Also redraw the visual indicator
                    plotter.add_mesh(
                        pv.Cylinder(center=(cx, cy, 0.5), direction=(0, 0, 1), radius=radius-2, height=0.2),
                        color="#808080",
                        pbr=False,
                        lighting=False,
                        name=f"roundabout_{_snap_node}"
                    )
                    _rebuild_traffic_and_arrows()
                except Exception as exc:
                    print(f"[editor] Roundabout generation failed: {exc}")
                return
            elif mode == "lights":
                try:
                    px, py = float(point[0]), float(point[1])
                    _snap_node = nearest_graph_node(street_graph, px, py)
                    
                    tlights = scene_state.get("traffic_lights", {})
                    if _snap_node in tlights:
                        del tlights[_snap_node]
                        print(f"[editor] Removed traffic light at node {_snap_node}")
                    else:
                        from traffic_lights import TrafficLight, _group_edges_by_axis, _edge_bearing
                        in_edges = []
                        for idx, path in enumerate(car_paths):
                            if path["v"] == _snap_node:
                                bearing = _edge_bearing(street_graph, path["u"], path["v"])
                                in_edges.append((idx, path["u"], path["v"], bearing))
                        if in_edges:
                            n_phases = 2 if len(in_edges) >= 3 else 1
                            groups = _group_edges_by_axis(in_edges, n_phases=n_phases)
                            ndata = street_graph.nodes[_snap_node]
                            
                            light = TrafficLight(
                                node_id=_snap_node,
                                x=float(ndata.get("x", 0.0)),
                                y=float(ndata.get("y", 0.0)),
                                green_groups=groups,
                                n_phases=n_phases,
                                offset=0.0
                            )
                            tlights[_snap_node] = light
                            print(f"[editor] Added traffic light at node {_snap_node}")
                        else:
                            print(f"[editor] Cannot add light: no incoming paths at node {_snap_node}")
                            return
                            
                    scene_state["traffic_lights"] = tlights
                    
                    if scene_state.get("tl_actor"):
                        plotter.remove_actor(scene_state["tl_actor"])
                        
                    from traffic_lights import build_light_mesh, build_light_glyphs
                    _tl_mesh = build_light_mesh(tlights)
                    scene_state["_tl_mesh"] = _tl_mesh
                    if _tl_mesh.n_points > 0:
                        _tl_sphere = pv.Sphere(radius=1.4, theta_resolution=10, phi_resolution=10)
                        _tl_glyphs = build_light_glyphs(_tl_mesh, _tl_sphere)
                        scene_state["tl_actor"] = plotter.add_mesh(
                            _tl_glyphs, scalars="colors", rgb=True,
                            smooth_shading=True, pbr=True, metallic=0.1, roughness=0.4, lighting=True
                        )
                        _set_actor_visibility(scene_state["tl_actor"], bool(scene_state.get("show_traffic_signals", True)))
                    else:
                        scene_state["tl_actor"] = None
                        
                    plotter.render()
                except Exception as exc:
                    print(f"[editor] Traffic light toggle failed: {exc}")
                return
            elif mode == "stops":
                try:
                    px, py = float(point[0]), float(point[1])
                    _snap_node = nearest_graph_node(street_graph, px, py)
                    
                    current_tag = street_graph.nodes[_snap_node].get("highway")
                    if current_tag == "stop":
                        street_graph.nodes[_snap_node]["highway"] = None
                        print(f"[editor] Removed stop sign at node {_snap_node}")
                    else:
                        street_graph.nodes[_snap_node]["highway"] = "stop"
                        print(f"[editor] Added stop sign at node {_snap_node}")
                        
                    _rebuild_traffic_and_arrows()
                except Exception as exc:
                    print(f"[editor] Stop sign toggle failed: {exc}")
                return
            elif mode == "streetlights":
                try:
                    nonlocal best_positions
                    px, py = float(point[0]), float(point[1])
                    if best_positions is not None and best_positions.shape[0] > 0:
                        _dists = np.linalg.norm(best_positions[:, :2] - np.array([px, py]), axis=1)
                        _nearest_idx = int(np.argmin(_dists))
                        if float(_dists[_nearest_idx]) < 5.0:
                            best_positions = np.delete(best_positions, _nearest_idx, axis=0)
                            print(f"[editor] Removed streetlight near ({px:.1f}, {py:.1f})")
                        else:
                            best_positions = np.vstack([best_positions, [px, py]])
                            print(f"[editor] Added streetlight at ({px:.1f}, {py:.1f})")
                    else:
                        best_positions = np.array([[px, py]])
                        print(f"[editor] Added first streetlight at ({px:.1f}, {py:.1f})")
                    
                    scene_state["show_streetlights"] = True
                    scene_state.pop("cached_night_coverage", None)
                    scene_state.pop("cached_night_key", None)
                    _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))
                except Exception as exc:
                    print(f"[editor] Streetlight toggle failed: {exc}")
                return

            # ── Car proximity check: if a car is within 8 m of the click, treat as car click ──
            _pos_arr = car_anim.get("pos")
            if _pos_arr is not None:
                _pos_arr = np.asarray(_pos_arr, dtype=float)
                if _pos_arr.shape[0] > 0 and _pos_arr.ndim == 2 and _pos_arr.shape[1] >= 2:
                    _click_xy = np.array([float(point[0]), float(point[1])], dtype=float)
                    _dists = np.linalg.norm(_pos_arr[:, :2] - _click_xy, axis=1)
                    _nearest_car = int(np.argmin(_dists))
                    if float(_dists[_nearest_car]) < 8.0:
                        # Delegate to the existing car-pick logic
                        try:
                            _e = int(car_anim["edge_idx"][_nearest_car])
                            _d = float(car_anim["dist"][_nearest_car])
                            _p3d = _car_pose_on_path(car_paths[_e], _d)
                            _snap_node = nearest_graph_node(street_graph, _p3d[0], _p3d[1])
                            _clear_route_actors()
                            route_state["source_node"] = _snap_node
                            _xy = _node_xy(_snap_node)
                            if _xy is not None:
                                _src_actor = plotter.add_mesh(
                                    pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0),
                                              theta_resolution=18, phi_resolution=18),
                                    color="#33cc66", render=False,
                                )
                                route_state["route_actors"] = [_src_actor]
                            _select_car_for_route(_nearest_car, _snap_node)
                            route_state["stage"] = 1
                            print(f"[route] Car {_nearest_car} selected as source — click target.")
                            if getattr(args, "solo", False):
                                scene_state["solo_solar_Wh"] = 0.0
                                scene_state["solo_mech_Wh"] = 0.0
                        except Exception as _exc:
                            print(f"[cars] Car pick failed: {_exc}")
                        return   # do NOT fall through to road-info logic

            # ── Road info + route planning (original _road_pick_callback logic) ──
            _road_pick_callback(point)

        def _register_unified_picker() -> None:
            try:
                plotter.disable_picking()
            except Exception:
                pass
            try:
                plotter.enable_surface_point_picking(
                    callback=_unified_pick_callback,
                    show_message=False,
                    show_point=False,
                    tolerance=0.025,
                )
                print("[picker] unified surface picker registered")
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
                display_places = [p for p in places if "residential" not in p.get("categories", [])]
                if display_places:
                    _poi_z = 2.5
                    _poi_pts = np.array([[p["x"], p["y"], _poi_z] for p in display_places], dtype=float)
                    _poi_mesh = pv.PolyData(_poi_pts)
                    _poi_mesh["labels"] = [_poi_label(p) for p in display_places]
                    _poi_mesh.point_data["colors"] = np.array(
                        [_hex_to_rgb(_poi_color(p["categories"])) for p in display_places],
                        dtype=np.uint8,
                    )
                    scene_state["_poi_mesh"] = _poi_mesh
                else:
                    scene_state["_poi_mesh"] = None
                scene_state["poi_actor"] = None
                scene_state["poi_labels_actor"] = None
                scene_state["show_pois"] = False
                scene_state["show_poi_names"] = False
                print(f"[poi] {len(places)} POIs prepared for lazy loading")
        else:
                scene_state["_poi_mesh"] = None
                scene_state["poi_actor"] = None
                scene_state["poi_labels_actor"] = None
                scene_state["show_pois"] = False
                scene_state["show_poi_names"] = False
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
                name="ground_mesh",
        )
        scene_state["spotlight_actor"] = None
        plotter.add_text("Move the Hour slider to compute shadows", position=(0.18, 0.02), name="status", font_size=9, viewport=True)
        _stage(f"Initial ground actor added ({initial_ground.n_cells} cells)")

        # --- Async shadow rendering infrastructure ---
        # A single-worker executor ensures only one shadow job runs at a time.
        # (concurrent.futures is already imported as `cf` at the top of the file)
        _shadow_executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-bg")
        # scene_state keys used by the async system:
        #   "_shadow_future"   : Future | None — the running background job
        #   "_shadow_pending"  : dict | None   — args queued while a job is running

        def _apply_visual_updates(hour: float, spot_radius: float, is_night: bool, sun_dir, style: dict) -> None:
                """Apply all non-blocking visual updates (background, lights, sun sphere, actor colors).
                This runs on the main thread and returns immediately."""
                # Background / skybox
                has_skybox = False
                try:
                    hdri_tex = _load_hdri(float(hour), str(scene_state["preset"]))
                    plotter.set_environment_texture(hdri_tex)
                    plotter.renderer.UseImageBasedLightingOn()
                    try:
                        plotter.add_background_cube_map(hdri_tex)
                        has_skybox = True
                    except Exception as _bg_exc:
                        print(f"[pbr] failed to set background cubemap: {_bg_exc}")
                except Exception as _exc:
                    print(f"[pbr] dynamic HDRI unavailable ({_exc}); using gradient background")

                if not has_skybox:
                    bg = style.get("day_bg" if not is_night else "night_bg")
                    if isinstance(bg, list) and len(bg) == 2:
                        plotter.set_background(str(bg[0]), top=str(bg[1]))
                    else:
                        plotter.set_background(str(bg))

                # Sun sphere
                try:
                    plotter.remove_actor("sun_sphere", reset_camera=False)
                except Exception:
                    pass
                if not is_night:
                    try:
                        _sun_pos = np.asarray(sun_dir, dtype=float) * 500.0
                        _sun_sphere = pv.Sphere(
                            radius=15.0,
                            center=(float(_sun_pos[0]), float(_sun_pos[1]), float(_sun_pos[2])),
                        )
                        plotter.add_mesh(
                            _sun_sphere, color="#ffeb3b", emissive=True,
                            name="sun_sphere", lighting=False,
                        )
                    except Exception as _exc:
                        print(f"[sun-sphere] failed: {_exc}")

                # Dynamic scene lights
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
                    _sun_light.ambient = (0.25, 0.25, 0.25)
                    _sun_light.diffuse = (0.85, 0.85, 0.85)
                    _sun_light.specular = (0.1, 0.1, 0.1)
                    plotter.add_light(_sun_light)
                    if is_night:
                        _moon_light = pv.Light(
                            light_type="scene light",
                            position=(0.0, 0.0, 500.0),
                            intensity=0.15,
                        )
                        _moon_light.ambient = (0.1, 0.1, 0.12)
                        _moon_light.diffuse = (0.15, 0.15, 0.18)
                        plotter.add_light(_moon_light)
                except Exception as _exc:
                    print(f"[light] dynamic scene light setup failed: {_exc}")

                # Actor colors
                b_actor = scene_state.get("building_actor")
                _pbr_actors = scene_state.get("_building_actors_pbr", {})
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

                # Street-light poles and bulbs
                try:
                    light_color = str(style["light_night"]) if is_night else str(style["light_day"])
                    
                    if best_positions.shape[0] > 0:
                        lp = np.column_stack((best_positions[:, 0], best_positions[:, 1], np.full(best_positions.shape[0], float(args.pole_height))))
                        poles_lines = []
                        for pt in lp:
                            poles_lines.append(pv.Line((pt[0], pt[1], 0.0), (pt[0], pt[1], float(args.pole_height))))
                        poles_mesh = pv.MultiBlock(poles_lines).combine()
                        
                        bulb_mesh = pv.PolyData(lp)
                        bulb_glyph = bulb_mesh.glyph(geom=pv.Sphere(radius=0.4 if is_night else 0.25))
                        
                        scene_state["lights_actor"] = plotter.add_mesh(
                            bulb_glyph,
                            color=light_color,
                            name="streetlight_points",
                        )
                        scene_state["poles_actor"] = plotter.add_mesh(
                            poles_mesh,
                            color="#555555",
                            line_width=3,
                            name="streetlight_poles",
                        )
                        _set_actor_visibility(scene_state.get("lights_actor"), bool(scene_state["show_streetlights"]))
                        _set_actor_visibility(scene_state.get("poles_actor"), bool(scene_state["show_streetlights"]))
                except Exception as _exc:
                    print(f"[lights] failed to add lights actor: {_exc}")

                # Spotlight discs (night only)
                if is_night:
                    discs = _build_spotlight_discs(best_positions, spot_radius)
                    if discs is not None:
                        scene_state["spotlight_actor"] = plotter.add_mesh(
                            discs,
                            color=str(style["disc"]),
                            opacity=0.38,
                            show_edges=False,
                            name="spotlight_discs",
                        )
                    _set_actor_visibility(scene_state.get("spotlight_actor"), bool(scene_state["show_streetlights"]))
                else:
                    try:
                        plotter.remove_actor("spotlight_discs", reset_camera=False)
                        scene_state["spotlight_actor"] = None
                    except Exception:
                        pass

                _set_actor_visibility(scene_state.get("vehicle_actor"), bool(scene_state["show_roads"]))
                _set_actor_visibility(scene_state.get("ped_actor"), bool(scene_state["show_roads"]))
                _render_cars()
                _apply_atmosphere(
                    plotter,
                    hour=float(hour),
                    is_night=bool(is_night),
                    preset=str(scene_state["preset"]),
                )

        def _apply_shadow_result_to_ground(
            hour: float, spot_radius: float, is_night: bool,
            mask_or_coverage,        # bool ndarray (day) or int16 ndarray (night)
            lit_ratio_or_pct: float,
        ) -> None:
                """Apply pre-computed shadow/coverage data to the ground mesh (main thread)."""
                nonlocal ground_mesh
                style = _style()
                if is_night:
                    coverage_count = np.asarray(mask_or_coverage, dtype=np.int16)
                    night_copy = scene_state["night_ground_copy"]
                    if coverage_count.shape[0] != night_copy.n_cells:
                        print("[shadow-apply] coverage shape mismatch, skipping night ground update.")
                        return
                    illum = np.zeros(night_copy.n_cells, dtype=np.uint8)
                    illum[coverage_count >= 1] = 1
                    illum[coverage_count >= 2] = 2
                    night_copy.cell_data["night_surface_class"] = _compose_night_surface_classes(illum)
                    lit_pct = lit_ratio_or_pct
                    scene_state["ground_actor"] = plotter.add_mesh(
                        night_copy,
                        scalars="night_surface_class",
                        clim=[0, 5],
                        cmap=night_surface_cmap,
                        show_edges=False,
                        opacity=0.9,
                        name="ground_mesh",
                    )
                    plotter.add_text(
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
                    day_copy = scene_state["day_ground_copy"]
                    day_copy.cell_data["day_surface_class"] = _compose_day_surface_classes(mask)
                    scene_state["ground_actor"] = plotter.add_mesh(
                        day_copy,
                        scalars="day_surface_class",
                        clim=[0, 3],
                        cmap=day_surface_cmap,
                        show_edges=False,
                        opacity=0.9,
                        name="ground_mesh",
                    )
                    plotter.add_text(
                        f"Hour {hour:04.1f}  |  Lit Ratio {lit_ratio:.3f}",
                        position=(0.18, 0.02),
                        name="status",
                        font_size=9,
                        viewport=True,
                        color="black",
                    )
                plotter.render()

        def _shadow_bg_worker(
            _hour: float, _spot_radius: float, _is_night: bool, _sun_dir,
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
                        "cached_night_coverage" in scene_state
                        and scene_state.get("cached_night_key") == night_key
                    ):
                        coverage_count = np.asarray(scene_state["cached_night_coverage"], dtype=np.int16)
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
                        pole_h = float(args.pole_height)
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
                            _compute_spotlight_coverage_numba(
                                light_pos, _centroids, dummy_orientation, -1.0,
                                building_triangles, building_aabb_mins, building_aabb_maxs,
                                cand,
                                cov[i],
                            )
                        coverage_count = np.sum(cov, axis=0, dtype=np.int16)
                        scene_state["cached_night_coverage"] = coverage_count
                        scene_state["cached_night_key"] = night_key
                        dt = time.perf_counter() - t0
                        print(f"[bg-shadow] Night coverage done in {dt:.2f}s")
                    illum_count = float(np.count_nonzero(coverage_count >= 1))
                    lit_pct = 100.0 * illum_count / float(max(1, coverage_count.size))
                    return coverage_count, lit_pct
                else:
                    day_cache = scene_state.get("cached_day_shadows")
                    if not isinstance(day_cache, dict):
                        day_cache = {}
                        scene_state["cached_day_shadows"] = day_cache
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

        def _render_ground(hour: float, spot_radius: float, _use_update: bool = False) -> None:
                """Compatibility shim — delegates to the async render path."""
                nonlocal ground_mesh, edge_shadow_frac
                sun_dir = _sun_dir_from_hour(hour)
                is_night = bool(sun_dir[2] <= 0.0)
                scene_state["is_night"] = is_night
                edge_shadow_frac = _build_edge_shadow_cache(float(hour))
                style = _style()
                # Apply fast visual updates immediately so the scene looks correct
                _apply_visual_updates(hour, spot_radius, is_night, sun_dir, style)
                plotter.render()
                # Queue the heavy shadow computation in the background
                _schedule_shadow_job(float(hour), float(spot_radius), is_night, sun_dir)

        def _night_cache_key_for(_spot_radius: float) -> tuple[float, int, str]:
                return (
                    float(_spot_radius),
                    int(best_positions.shape[0]),
                    hashlib.sha1(np.ascontiguousarray(best_positions, dtype=np.float64).tobytes()).hexdigest()[:16],
                )

        def _schedule_shadow_job(_hour: float, _spot_radius: float, _is_night: bool, _sun_dir) -> None:
                """Submit shadow computation to background executor.
                All VTK geometry is extracted HERE on the main thread (thread-safe).
                The background thread receives only pure numpy arrays.
                """
                running = scene_state.get("_shadow_future")
                if running is not None and not running.done():
                    # Store latest request; _poll will re-schedule when current job finishes
                    scene_state["_shadow_pending"] = {
                        "hour": _hour, "spot_radius": _spot_radius,
                        "is_night": _is_night, "sun_dir": _sun_dir,
                    }
                    return

                scene_state["_shadow_pending"] = None
                plotter.add_text(
                    "Computing shadows…",
                    position=(0.18, 0.02), name="status",
                    font_size=9, viewport=True, color="#ffca28",
                )
                try:
                    plotter.render()
                except Exception:
                    pass

                # --- Pre-extract VTK geometry on the MAIN THREAD (thread-safe) ---
                from shadow_engine import _extract_ground_triangles_areas_centroids_cached
                try:
                    _tris, _areas, _cents = _extract_ground_triangles_areas_centroids_cached(ground_mesh)
                    _tris = np.ascontiguousarray(_tris, dtype=np.float64)
                    _areas = np.ascontiguousarray(_areas, dtype=np.float64)
                    _cents = np.ascontiguousarray(_cents, dtype=np.float64)
                    _n_cells = int(ground_mesh.n_cells)
                except Exception as _geo_exc:
                    print(f"[schedule-shadow] geometry extraction failed: {_geo_exc}")
                    return
                # -----------------------------------------------------------------

                _oct = _ensure_octree()
                fut = _shadow_executor.submit(
                    _shadow_bg_worker,
                    _hour, _spot_radius, _is_night, _sun_dir,
                    _n_cells,
                    _tris,    # pure numpy — safe for background thread
                    _areas,
                    _cents,
                    _oct,
                    best_positions.copy(),
                    street_graph,
                )
                # Attach metadata so _poll can apply results
                fut._render_args = (_hour, _spot_radius, _is_night)
                scene_state["_shadow_future"] = fut

        def _request_shadow_render(_hour: float, _spot_radius: float) -> None:
                scene_state["hour"] = float(_hour)
                scene_state["spot_radius"] = float(_spot_radius)
                sun_dir = _sun_dir_from_hour(_hour)
                is_night = bool(sun_dir[2] <= 0.0)
                scene_state["is_night"] = is_night
                nonlocal edge_shadow_frac
                edge_shadow_frac = _build_edge_shadow_cache(float(_hour))
                style = _style()
                try:
                    # Fast visual update — returns immediately
                    _apply_visual_updates(_hour, _spot_radius, is_night, sun_dir, style)
                    plotter.render()
                except Exception as _exc:
                    print(f"[render] fast visual update failed: {_exc}")
                # Queue heavy computation in background
                _schedule_shadow_job(float(_hour), float(_spot_radius), is_night, sun_dir)

        def _poll_shadow_job(_: int) -> None:
                """Called every 250 ms. Applies finished shadow results to the main thread."""
                fut = scene_state.get("_shadow_future")
                if fut is None or not fut.done():
                    return
                # Consume the future
                scene_state["_shadow_future"] = None
                try:
                    result = fut.result()
                    hour, spot_radius, is_night = fut._render_args
                    mask_or_cov, ratio_or_pct = result
                    _apply_shadow_result_to_ground(hour, spot_radius, is_night, mask_or_cov, ratio_or_pct)
                except Exception as _exc:
                    import traceback as _tb
                    print(f"[poll-shadow] applying result failed: {_exc}")
                    _tb.print_exc()
                # If a newer request was queued while we were computing, run it now
                pending = scene_state.get("_shadow_pending")
                if pending is not None:
                    scene_state["_shadow_pending"] = None
                    _schedule_shadow_job(
                        pending["hour"], pending["spot_radius"],
                        pending["is_night"], pending["sun_dir"],
                    )

        def _on_time_change(value: float) -> None:
                scene_state["hour"] = float(value)
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _on_radius_change(value: float) -> None:
                scene_state["spot_radius"] = float(value)
                # Invalidate cached coverage since spotlight radius changed
                scene_state.pop("cached_night_coverage", None)
                scene_state.pop("cached_night_key", None)
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        plotter.add_slider_widget(
                _on_time_change,
                rng=[0.0, 24.0],
                value=12.0,
                title="Hour",
                pointa=(0.02, 0.14),
                pointb=(0.24, 0.14),
                style="modern",
                interaction_event="end",
                title_height=0.018,
                slider_width=0.025,
                tube_width=0.008,
        )
        plotter.add_slider_widget(
                _on_radius_change,
                rng=[max(5.0, args.light_radius * 0.4), args.light_radius * 3.0],
                value=float(args.light_radius),
                title="Light radius",
                pointa=(0.02, 0.08),
                pointb=(0.24, 0.08),
                style="modern",
                interaction_event="end",
                title_height=0.018,
                slider_width=0.025,
                tube_width=0.008,
        )

        def _toggle_roads(value: bool) -> None:
                scene_state["show_roads"] = bool(value)
                _set_actor_visibility(scene_state.get("vehicle_actor"), bool(value))
                _set_actor_visibility(scene_state.get("ped_actor"), bool(value))
                plotter.update()

        def _toggle_cars(value: bool) -> None:
                scene_state["show_cars"] = bool(value)
                car_actors = scene_state.get("car_actors")
                if isinstance(car_actors, dict):
                    for actor in car_actors.values():
                        _set_actor_visibility(actor, bool(value))
                for _ua in scene_state.get("_ultra_car_actors") or []:
                    _set_actor_visibility(_ua, bool(value))
                plotter.update()

        def _preset_mini(_: bool) -> None:
                scene_state["preset"] = "mini"
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_coastal(_: bool) -> None:
                scene_state["preset"] = "coastal"
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _preset_sunset(_: bool) -> None:
                scene_state["preset"] = "sunset"
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                _request_shadow_render(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _toggle_arrows(value: bool) -> None:
                scene_state["show_arrows"] = bool(value)
                _set_actor_visibility(scene_state.get("street_arrows_actor"), bool(value))
                plotter.update()

        def _toggle_streetlights(value: bool) -> None:
                scene_state["show_streetlights"] = bool(value)
                _set_actor_visibility(scene_state.get("lights_actor"), bool(value))
                _set_actor_visibility(scene_state.get("spotlight_actor"), bool(value))
                plotter.update()

        def _toggle_traffic_signals(value: bool) -> None:
                scene_state["show_traffic_signals"] = bool(value)
                _set_actor_visibility(scene_state.get("tl_actor"), bool(value))
                plotter.update()

        def _on_optimize(_: bool) -> None:
                if not bool(scene_state.get("interactive_ready", False)):
                    return
                if bool(scene_state.get("ga_running", False)):
                    # Already running — ignore double-click
                    return
                scene_state["ga_running"] = True
                plotter.add_text("Running streetlight placement...", position=(0.18, 0.02), name="status", font_size=9, viewport=True)
                plotter.update()

                def _do_ga_thread() -> None:
                    nonlocal best_positions
                    try:
                        if args.light_strategy == "smart":
                            best_positions = np.asarray(_compute_smart_lights(), dtype=float)
                            print(f"[lights] Smart placement complete: {best_positions.shape[0]} lights")
                        else:
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
                            print(f"[ga] Optimization complete: cost={ga_result['best_cost']:.4f}, lit={ga_result['lit_ratio']:.3f}")
                        scene_state.pop("cached_night_coverage", None)
                        scene_state.pop("cached_night_key", None)
                        scene_state["ga_done"] = True
                    except Exception as _exc:
                        print(f"[lights] Background placement failed: {_exc}")
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
        _TX = 35      # text pixel x (label)

        def _panel_text_color():
                # Choose text color for panel for readability
                bg_val = _style()["day_bg"] if not scene_state.get("is_night") else _style()["night_bg"]
                bg = str(bg_val[0] if isinstance(bg_val, list) else bg_val)
                # Simple luminance check
                bg = bg.lstrip("#")
                r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                return "#f4f4f4" if luminance < 128 else "#222222"

        def _panel_desc_color():
                # Muted but still readable
                base = _panel_text_color()
                if base == "#f4f4f4":
                    return "#e0e0e0"
                return "#444444"

        _TC = _panel_text_color()
        _DC = _panel_desc_color()

        def _cy(row: int) -> float:
                """Return checkbox_y_px for given row (0=top)."""
                y_px = 836 - row * 30
                return float(y_px)

        # ── Section: Visibility ───────────────────────────────────────────
        plotter.add_text("Controls", position=(_TX, _cy(0) + 2),
             name="panel_vis_hdr", font_size=11, color=_TC, viewport=False)

        _r1 = _cy(1)
        plotter.add_text("Roads",   position=(_TX, _r1), name="panel_roads",   font_size=9, color=_TC, viewport=False)
        plotter.add_checkbox_button_widget(
                _toggle_roads, value=bool(scene_state["show_roads"]),
            position=(_CX, _r1), size=_CS, color_on="#6aa06f", color_off="#3a3f4b",
        )

        _r2 = _cy(2)
        plotter.add_text("Cars",    position=(_TX, _r2), name="panel_cars",   font_size=9, color=_TC, viewport=False)
        plotter.add_checkbox_button_widget(
                _toggle_cars, value=bool(scene_state["show_cars"]),
            position=(_CX, _r2), size=_CS, color_on="#ff6b6b", color_off="#3a3f4b",
        )

        _r3 = _cy(3)
        plotter.add_text("Arrows",    position=(_TX, _r3), name="panel_arrows",   font_size=9, color=_TC, viewport=False)
        plotter.add_checkbox_button_widget(
                _toggle_arrows, value=True,
            position=(_CX, _r3), size=_CS, color_on="#ff3b30", color_off="#3a3f4b",
        )

        _r4 = _cy(4)
        plotter.add_text("Streetlights",  position=(_TX, _r4), name="panel_lights",   font_size=9, color=_TC, viewport=False)
        plotter.add_checkbox_button_widget(
            _toggle_streetlights, value=bool(scene_state["show_streetlights"]),
            position=(_CX, _r4), size=_CS, color_on="#ffd740", color_off="#3a3f4b",
        )

        _r5 = _cy(5)
        plotter.add_text("Signals",  position=(_TX, _r5), name="panel_signals",   font_size=9, color=_TC, viewport=False)
        plotter.add_checkbox_button_widget(
            _toggle_traffic_signals, value=bool(scene_state["show_traffic_signals"]),
            position=(_CX, _r5), size=_CS, color_on="#22cc55", color_off="#3a3f4b",
        )

        def _toggle_pois_lazy(val: bool) -> None:
            scene_state["show_pois"] = bool(val)
            actor = scene_state.get("poi_actor")
            mesh = scene_state.get("_poi_mesh")
            if val and actor is None and mesh is not None:
                scene_state["poi_actor"] = plotter.add_mesh(
                    mesh,
                    scalars="colors",
                    rgb=True,
                    style="points",
                    point_size=14,
                    render_points_as_spheres=True,
                    lighting=False,
                    reset_camera=False,
                )
                print("[poi] POI dots created")
            elif actor is not None:
                _set_actor_visibility(actor, val)

        def _toggle_poi_names_lazy(val: bool) -> None:
            scene_state["show_poi_names"] = bool(val)
            actor = scene_state.get("poi_labels_actor")
            mesh = scene_state.get("_poi_mesh")
            if val and actor is None and mesh is not None:
                scene_state["poi_labels_actor"] = plotter.add_point_labels(
                    mesh,
                    "labels",
                    point_size=1,
                    font_size=14,
                    text_color="white",
                    render_points_as_spheres=False,
                    always_visible=False,
                    shape_opacity=0.55,
                    shape_color="#111111",
                    tolerance=0.01,
                )
                print("[poi] POI labels created")
            elif actor is not None:
                _set_actor_visibility(actor, val)

        _r6 = _cy(6)
        plotter.add_text("POIs",   position=(_TX, _r6[1]), name="panel_pois",   font_size=9, color=_TC, viewport=True)
        plotter.add_checkbox_button_widget(
            _toggle_pois_lazy,
            value=False,
            position=(_CX, _r6[0]), size=_CS, color_on="#e07b54", color_off="#3a3f4b",
        )

        _r7 = _cy(7)
        plotter.add_text("Names",  position=(_TX, _r7[1]), name="panel_names",   font_size=9, color=_TC, viewport=True)
        plotter.add_checkbox_button_widget(
            _toggle_poi_names_lazy,
            value=False,
            position=(_CX, _r7[0]), size=_CS, color_on="#f5a623", color_off="#3a3f4b",
        )

        _r8 = _cy(8)
        plotter.add_text("Place lights", position=(_TX, _r8[1]), name="panel_optimize",   font_size=9, color=_TC, viewport=True)
        plotter.add_checkbox_button_widget(
            _on_optimize, value=False,
            position=(_CX, _r8[0]), size=_CS, color_on="#ffd166", color_off="#3a3f4b",
        )

        _r_ssao = _cy(9.2)
        plotter.add_text("SSAO",      position=(_TX, _r_ssao), name="panel_ssao",   font_size=9, color=_TC, viewport=False)

        def _toggle_ssao(val: bool) -> None:
                try:
                    renderer.SetUseSSAO(bool(val))
                    plotter.update()
                except Exception:
                    print("[ssao] SSAO unavailable")

        _ssao_enabled = False
        try:
                _ssao_enabled = bool(renderer.GetUseSSAO())
        except Exception:
                _ssao_enabled = False
        plotter.add_checkbox_button_widget(
            _toggle_ssao, value=_ssao_enabled,
            position=(_CX, _r_ssao), size=_CS, color_on="#6cb6ff", color_off="#3a3f4b",
        )

        # Separator removed as it was rendered in 3D space instead of UI space

        # ── Section: Style presets ────────────────────────────────────────
        _rs = _cy(10)
        plotter.add_text("Style", position=(_TX, _rs + 2),
                 name="panel_style_hdr", font_size=11, color=_TC, viewport=False)

        _r9 = _cy(11)
        plotter.add_text("Mini",    position=(_TX, _r9), name="preset_mini_t",    font_size=9, viewport=False)
        plotter.add_checkbox_button_widget(
            _preset_mini, value=True,
            position=(_CX, _r9), size=_CS, color_on="#ffd166", color_off="#3a3f4b",
        )

        _r10 = _cy(12)
        plotter.add_text("Coastal", position=(_TX, _r10), name="preset_coastal_t",   font_size=9, viewport=False)
        plotter.add_checkbox_button_widget(
            _preset_coastal, value=False,
            position=(_CX, _r10), size=_CS, color_on="#56b48a", color_off="#3a3f4b",
        )

        _r11 = _cy(13)
        plotter.add_text("Sunset", position=(_TX, _r11), name="preset_sunset_t",   font_size=9, viewport=False)
        plotter.add_checkbox_button_widget(
            _preset_sunset, value=False,
            position=(_CX, _r11), size=_CS, color_on="#ffb86b", color_off="#3a3f4b",
        )

        # ── Section: Solar routing (solarcar fleet only) ──────────────────
        if bool(scene_state.get("solar_fleet", False)):
            _r_sol = _cy(14)
            plotter.add_text("Solar car", position=(_TX, _r_sol + 2),
                             name="panel_solar_hdr", font_size=10, color=_TC, viewport=False)

            _r_alpha = _cy(15)
            _r_hour = _cy(16)
            _r_area = _cy(17)

            def _on_route_alpha_change(value: float) -> None:
                scene_state["route_alpha"] = float(np.clip(value, 0.0, 1.0))
                if (
                    route_state.get("stage") == 2
                    and route_state.get("source_node") is not None
                    and route_state.get("target_node") is not None
                ):
                    try:
                        _compute_and_render_routes(route_state["source_node"], route_state["target_node"])
                        plotter.update()
                    except Exception as _exc:
                        print(f"[route] alpha update failed: {_exc}")

            def _on_route_hour_change(value: float) -> None:
                scene_state["route_hour"] = float(np.clip(value, 6.0, 18.0))
                scene_state["edge_shadow_cache"] = {}
                scene_state["edge_costs_cache"] = {}
                if (
                    route_state.get("stage") == 2
                    and route_state.get("source_node") is not None
                    and route_state.get("target_node") is not None
                ):
                    try:
                        _compute_and_render_routes(route_state["source_node"], route_state["target_node"])
                        plotter.update()
                    except Exception as _exc:
                        print(f"[route] route hour update failed: {_exc}")

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
                scene_state["edge_costs_cache"] = {}
                if (
                    route_state.get("stage") == 2
                    and route_state.get("source_node") is not None
                    and route_state.get("target_node") is not None
                ):
                    try:
                        _compute_and_render_routes(route_state["source_node"], route_state["target_node"])
                        plotter.update()
                    except Exception as _exc:
                        print(f"[route] panel area update failed: {_exc}")

            plotter.add_slider_widget(
                _on_route_alpha_change,
                rng=[0.0, 1.0],
                value=float(scene_state.get("route_alpha", 0.5)),
                title="Route mix",
                pointa=(0.76, _r_alpha[1]),
                pointb=(0.94, _r_alpha[1]),
                style="modern",
                interaction_event="always",
                title_height=0.018,
                slider_width=0.025,
                tube_width=0.008,
            )
            plotter.add_slider_widget(
                _on_route_hour_change,
                rng=[6.0, 18.0],
                value=float(scene_state.get("route_hour", scene_state.get("hour", 12.0))),
                title="Route hr",
                pointa=(0.76, _r_hour[1]),
                pointb=(0.94, _r_hour[1]),
                style="modern",
                interaction_event="always",
                title_height=0.018,
                slider_width=0.025,
                tube_width=0.008,
            )
            plotter.add_slider_widget(
                _on_panel_area_change,
                rng=[0.5, 3.0],
                value=float(getattr(scene_state.get("solar_params"), "roof_area_m2", 1.6)),
                title="Panel m2",
                pointa=(0.76, _r_area[1]),
                pointb=(0.94, _r_area[1]),
                style="modern",
                interaction_event="always",
                title_height=0.018,
                slider_width=0.025,
                tube_width=0.008,
            )
        else:
            plotter.add_text(
                "Solar routing: enable\n'All solar cars' at startup",
                position=(_TX, _cy(13)),
                name="panel_solar_off",
                font_size=8,
                color=_DC,
                viewport=False,
            )

        plotter.add_text(
                "",
                position=(0.55, 0.04),
                name="route_stats_overlay",
                font_size=8,
                color="#f4f4f4",
                viewport=True,
        )
        # ─────────────────────────────────────────────────────────────────────

        if args.optimize_on_open and best_positions is None:
                _stage(f"Optimize-on-open enabled: running {args.light_strategy} streetlight placement...")
                t_open_ga = time.perf_counter()
                if args.light_strategy == "smart":
                    best_positions = np.asarray(_compute_smart_lights(), dtype=float)
                else:
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
                _stage(f"Optimize-on-open placement finished in {time.perf_counter() - t_open_ga:.2f}s")
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
                plotter.update()

        def _rotate_camera(azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> None:
                cam = plotter.camera
                if abs(float(azimuth_deg)) > 0.0:
                    cam.Azimuth(float(azimuth_deg))
                if abs(float(elevation_deg)) > 0.0:
                    cam.Elevation(float(elevation_deg))
                cam.OrthogonalizeViewUp()
                plotter.update()

        def _zoom_camera(factor: float) -> None:
                cam = plotter.camera
                cam.Dolly(float(factor))
                plotter.reset_camera_clipping_range()
                plotter.update()

        def _reset_camera_view() -> None:
                plotter.view_isometric()
                plotter.update()

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

        _runtime_timer_starters = []

        if bool(car_anim["enabled"]) and float(args.traffic_speed) > 0.0:
                car_anim["last_t"] = time.perf_counter()
                car_timer_ms = 80
                _n_car_actors = max(0, int(args.n_cars))
                if args.car_detail == "low":
                    car_timer_ms = max(car_timer_ms, 120)
                elif args.car_detail == "ultra" and _n_car_actors >= 8:
                    car_timer_ms = 100
                if args.car_detail == "ultra" and _n_car_actors >= 12:
                    car_timer_ms = 120
                # Full plotter.render() every N sim ticks (macOS VTK is fragile at 12+ FPS).
                _car_render_stride = 4 if args.car_detail == "ultra" else 2
                if ground_mesh.n_cells >= 120_000:
                    car_timer_ms = max(car_timer_ms, 90)
                    _car_render_stride = max(_car_render_stride, 3)
                    print(
                        "Large ground mesh detected "
                        f"({ground_mesh.n_cells} triangles): reducing car animation rate to ~{int(round(1000.0 / car_timer_ms))} FPS."
                    )
                if args.car_detail == "ultra" and _n_car_actors >= 8:
                    print(
                        f"[cars] ultra OBJ mode: {_n_car_actors} actors, "
                        f"timer={car_timer_ms}ms, render every {_car_render_stride} tick(s)"
                    )

                def _animate_cars(_: int) -> None:
                    if not bool(scene_state.get("interactive_ready", False)):
                        return
                    _animate_cars._tick = getattr(_animate_cars, "_tick", 0) + 1
                    tick = _animate_cars._tick
                    try:
                        scene_state["_car_tick_active"] = True
                        now = time.perf_counter()
                        last_t = float(car_anim.get("last_t", now))
                        dt = float(np.clip(now - last_t, 0.0, 0.10))
                        car_anim["last_t"] = now
                        if dt <= 0.0:
                            scene_state["_car_tick_active"] = False
                            return
                        _advance_cars(dt)
                        
                        # Keep pos array fresh for the unified picker proximity test
                        car_anim["pos"] = _sample_car_positions()

                        if args.solo and len(np.asarray(car_anim["edge_idx"])) > 0:
                            _speed    = float(car_anim["speed"][0])
                            _e_idx    = int(car_anim["edge_idx"][0])
                            _shadow   = float(edge_shadow_frac[_e_idx])
                            _hour     = float(scene_state.get("hour", 12.0))
                            _lat      = float(street_graph.graph.get("scene_lat", 51.5))
                            _lon      = float(street_graph.graph.get("scene_lon", 0.0))

                            import solar_physics as _sp_mod
                            _sp = _sp_mod.SolarParams()

                            # Step 1: sun geometry
                            _el_rad, _ = _sp_mod.sun_angles(
                                lat_deg=_lat, lon_deg=_lon, hour_local=_hour
                            )
                            # Step 2: clear-sky irradiance decomposition
                            _ghi, _dni, _dhi = _sp_mod.clear_sky_ghi(_el_rad)
                            # Step 3: irradiance on flat horizontal panel (W/m²)
                            _g_panel = _sp_mod.panel_irradiance(
                                ghi=_ghi, dhi=_dhi, dni=_dni, sun_elevation_rad=_el_rad
                            )

                            # Power (W) — consistent units throughout
                            _w_solar = (
                                _g_panel
                                * (1.0 - _shadow)
                                * _sp.roof_area_m2
                                * _sp.panel_efficiency
                                * _sp.temperature_derating
                            )
                            # mechanical_energy_joules returns J for one second at constant speed;
                            # divide by 3600 to get W (= J/s expressed as Wh/s for accumulation)
                            _w_mech = _sp_mod.mechanical_energy_joules(
                                length_m=_speed * 1.0,   # distance covered in 1 s at current speed
                                speed_ms=_speed,
                                vehicle_mass_kg=_sp.vehicle_mass_kg,
                                rolling_coeff=_sp.rolling_coeff,
                                drag_coeff=_sp.drag_coeff,
                                frontal_area_m2=_sp.frontal_area_m2,
                            ) / 3600.0                    # J → Wh consumed per second

                            # Accumulate Wh: multiply W by dt, then convert seconds → hours
                            scene_state["solo_solar_Wh"] = (
                                float(scene_state.get("solo_solar_Wh", 0.0))
                                + _w_solar * dt / 3600.0
                            )
                            scene_state["solo_mech_Wh"] = (
                                float(scene_state.get("solo_mech_Wh", 0.0))
                                + _w_mech * dt          # already Wh/s × s = Wh
                            )
                            
                            if tick % 30 == 0:
                                _spd_kmh = _speed * 3.6
                                _sol_wh = scene_state["solo_solar_Wh"]
                                _mech_wh = scene_state["solo_mech_Wh"]
                                _net_wh = _sol_wh - _mech_wh
                                _sign = "▲" if _net_wh >= 0 else "▼"
                                _shade_icon = "▨ shaded" if _shadow > 0.5 else "☀ open"
                                
                                _txt = (
                                    f"Speed:      {_spd_kmh:4.1f} km/h\n"
                                    f"Solar Pwr:  {_w_solar:4.1f} W\n"
                                    f"Mech Nrg:   {_mech_wh:4.1f} Wh\n"
                                    f"Solar Hrv:  {_sol_wh:4.1f} Wh\n"
                                    f"Net Nrg:    {abs(_net_wh):4.1f} Wh {_sign}\n"
                                    f"Shadow:     {_shade_icon}"
                                )
                                plotter.add_text(
                                    _txt,
                                    position=(0.35, 0.04),
                                    name="solo_telemetry",
                                    viewport=True,
                                    font_size=10,
                                    color="#f4f4f4"
                                )

                        _render_cars()

                        # ── Tick and redraw traffic lights every 10 frames ────────
                        _tl = scene_state.get("traffic_lights")
                        if _tl:
                            tick_all(_tl, dt=dt, traffic_speed=float(args.traffic_speed))
                            _tl_tick = getattr(_animate_cars, "_tl_tick", 0) + 1
                            _animate_cars._tl_tick = _tl_tick
                            if _tl_tick % 15 == 0:
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
                        if tick % _car_render_stride == 0:
                            try:
                                plotter.update()
                            except Exception as _render_exc:
                                print(f"[cars] plotter.render failed: {_render_exc}")
                        scene_state["_car_tick_active"] = False
                    except Exception as _exc:
                        scene_state["_car_tick_active"] = False
                        import traceback as _tb
                        print(f"[cars-timer ERROR] {_exc}")
                        _tb.print_exc()

        # Register unified picker and callbacks
        _register_unified_picker()

        import sys

        # Define callbacks first
        def _mark_interactive_ready(_: int) -> None:
            try:
                print("[viewer] interactive callbacks enabled")
                scene_state["interactive_ready"] = True
                if bool(args.debug_cars):
                    print("[viewer-debug] interactive callbacks enabled")
            except Exception as _exc:
                print(f"[ready-timer ERROR] {_exc}")

        def _deferred_ssao_enable(_: int) -> None:
            try:
                renderer.SetUseSSAO(True)
                plotter.update()
                print("[ssao] SSAO activated (deferred)")
            except Exception as _exc:
                print(f"[ssao] deferred SSAO failed: {_exc}")

        def _deferred_initial_render(_: int) -> None:
            try:
                print("[timer] _deferred_initial_render: triggering initial shadow computation")
                _request_shadow_render(12.0, float(args.light_radius))
            except Exception as _exc:
                print(f"[init-render ERROR] {_exc}")

        if sys.platform != "darwin":
            # Add shadow polling timer
            plotter.add_timer_event(max_steps=10_000_000, duration=250, callback=_poll_shadow_job)

            # Add GA polling timer
            plotter.add_timer_event(max_steps=10_000_000, duration=500, callback=_poll_ga_done)

            # Add car animation timer
            if bool(car_anim["enabled"]) and float(args.traffic_speed) > 0.0:
                plotter.add_timer_event(max_steps=10_000_000, duration=car_timer_ms, callback=_animate_cars)
                print(f"[cars] animation timer registered ({car_timer_ms} ms)")
            elif bool(args.debug_cars):
                if not bool(car_anim["enabled"]):
                    print("[cars-debug] animation timer not started: no active cars")
                elif float(args.traffic_speed) <= 0.0:
                    print("[cars-debug] animation timer not started: traffic_speed <= 0")

            # Ready callback
            plotter.add_timer_event(max_steps=1, duration=300, callback=_mark_interactive_ready)

            # Deferred SSAO activation
            plotter.add_timer_event(max_steps=1, duration=800, callback=_deferred_ssao_enable)

            # Deferred initial shadow render
            plotter.add_timer_event(max_steps=1, duration=500, callback=_deferred_initial_render)

        import sys

        _stage("Calling plotter.show() — window should appear now")
        if sys.platform == "darwin":
            # macOS: VTK's native iren.Start() deadlocks under conda/Cocoa, so we
            # run a non-blocking show + manual event pump to keep Cocoa responsive.
            plotter.show(auto_close=False, interactive_update=True)
            print("[viewer] Entering manual event loop (macOS)...")
            iren = getattr(plotter, "iren", None)

            # macOS: VTK add_timer_event callbacks do NOT fire under manual ProcessEvents()
            # (NSTimers need the native NSRunLoop that iren.Start() would run — which
            # deadlocks under conda). So we drive every timer callback ourselves, by time.
            t0 = time.perf_counter()
            fired = {"ready": False, "init": False, "ssao": False}
            last = {"shadow": 0.0, "ga": 0.0, "cars": 0.0}
            cars_on = bool(car_anim.get("enabled", False)) and float(args.traffic_speed) > 0.0

            try:
                while hasattr(plotter, "render_window") and getattr(plotter, "render_window") is not None and not getattr(plotter, "_closed", False):
                    # Pump the Cocoa interactor explicitly
                    if iren is not None:
                        vtk_iren = getattr(iren, "interactor", None)
                        if vtk_iren is not None and hasattr(vtk_iren, "ProcessEvents"):
                            vtk_iren.ProcessEvents()

                    ms = (time.perf_counter() - t0) * 1000.0

                    # one-shot deferred callbacks
                    if not fired["ready"] and ms >= 300:
                        fired["ready"] = True; _mark_interactive_ready(0)
                    if not fired["init"] and ms >= 500:
                        fired["init"] = True; _deferred_initial_render(0)
                    if not fired["ssao"] and ms >= 800:
                        fired["ssao"] = True; _deferred_ssao_enable(0)

                    # periodic callbacks
                    if ms - last["shadow"] >= 250:
                        last["shadow"] = ms; _poll_shadow_job(0)
                    if ms - last["ga"] >= 500:
                        last["ga"] = ms; _poll_ga_done(0)
                    if cars_on and ms - last["cars"] >= car_timer_ms:
                        last["cars"] = ms; _animate_cars(0)

                    plotter.update()
                    time.sleep(0.005)
            except KeyboardInterrupt:
                print("\n[viewer] user interrupted – closing window")
            finally:
                plotter.close()
                print("[viewer] plotter.close() returned — viewer window closed")
        else:
            # Windows/Linux: use VTK's native blocking interactor loop. It keeps the
            # OS window message pump alive (responsive) and fires the add_timer_event
            # callbacks that drive the simulation. The manual pump above is a
            # macOS-only workaround and leaves the Win32 window "Not Responding".
            print("[viewer] Entering native VTK event loop (non-macOS)...")
            try:
                plotter.show(auto_close=False)
            except KeyboardInterrupt:
                print("\n[viewer] user interrupted – closing window")
            finally:
                plotter.close()
                print("[viewer] plotter.close() returned — viewer window closed")

        # --- Extract and Print Interesting Simulation Statistics ---
        try:
            print("\n" + "="*50)
            print("🚗 SIMULATION STATISTICS SUMMARY 🏙️")
            print("="*50)
            
            # Buildings & POIs
            _bm = locals().get('buildings_mesh')
            total_cells = _bm.n_cells if _bm is not None else 0
            _places = locals().get('places', [])
            poi_count = len(_places) if _places is not None else 0
            res_places = sum(1 for p in _places if "residential" in p.get("categories", [])) if _places is not None else 0
            print(f"Buildings geometry: {total_cells} cells")
            print(f"Points of Interest: {poi_count} (of which {res_places} are residential)")
            
            # Road network
            _sg = locals().get('street_graph')
            if _sg is not None:
                n_nodes = _sg.number_of_nodes()
                n_edges = _sg.number_of_edges()
                print(f"Road Network:       {n_nodes} nodes, {n_edges} edges")
                
                n_stops = sum(1 for _, d in _sg.nodes(data=True) if d.get("highway") == "stop")
                print(f"Stop Signs:         {n_stops}")
                
            # Traffic Lights
            _ss = locals().get('scene_state', {})
            n_tlights = len(_ss.get("traffic_lights", {}))
            print(f"Traffic Lights:     {n_tlights}")
            
            # Streetlights
            _bp = locals().get('best_positions')
            if _bp is not None:
                print(f"City Streetlights:  {_bp.shape[0]}")
            
            # Cars
            _ca = locals().get('car_anim')
            if _ca is not None and "dist" in _ca:
                n_active = len(_ca["dist"])
                print(f"Active Cars:        {n_active}")
                
            # Parked cars
            _args = locals().get('args')
            if _args is not None:
                n_parked = getattr(_args, 'n_parked_cars', 0)
                print(f"Parked Cars:        {n_parked}")
                
            print("="*50 + "\n")
        except Exception as _stat_exc:
            print(f"[stats] failed to generate summary statistics: {_stat_exc}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Visualization failed: {exc}")

if __name__ == "__main__":
    main()
