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

# Conda environments often fail to export PROJ_DATA/PROJ_LIB automatically,
# causing silent or loud projection errors ("Cannot find proj.db"). 
# Fix it here by asking pyproj where its data directory is.
try:
    import pyproj
    if "PROJ_DATA" not in os.environ and "PROJ_LIB" not in os.environ:
        _proj_dir = pyproj.datadir.get_data_dir()
        if _proj_dir:
            os.environ["PROJ_DATA"] = _proj_dir
except Exception:
    pass

import networkx as nx
import numpy as np
import pyvista as pv

# Enable faulthandler so segfaults print a Python traceback instead of silently dying
import faulthandler
faulthandler.enable()

_combine_ground_surfaces = None
_build_grid_points_from_ground_mesh = None
_build_ground_mesh_for_tests = None
_build_octree_from_buildings = None
_build_spotlight_discs = None
_cache_key = None
_initial_light_positions = None
_load_or_build_coverage_matrix_cached = None
_load_or_build_spatial_cache = None
_load_or_fetch_osm_cached = None
_normalize_ground_mesh = None
_street_direction_arrows = None
_street_node_markers = None
_street_line_layers = None
_load_hdri = None
_apply_atmosphere = None
_sun_dir_from_hour = None
from shapely.geometry import Point as _SPoint, LineString as _LS

import warnings
try:
    warnings.filterwarnings("ignore", category=pv.PyVistaFutureWarning)
except AttributeError:
    warnings.filterwarnings("ignore", message=".*extract_surface.*")
import vtk

# ── Break PyVista's VTK-error → logging feedback loop ────────────────────────
# On import, pyvista calls send_errors_to_logging(), which makes a
# vtkStringOutputWindow the global VTK output window AND attaches an Observer
# whose __call__ does logging.warning(...). On macOS that forms a loop:
#   VTK message → Observer → logging.warning → output ("WARNING:root:…") →
#   re-enters VTK → Observer → …  (each cycle prepends another "WARNING:root:")
# The recursive storm crashes the process during plotter.show().
#
# Fix: replace the global output window with an *unobserved* vtkFileOutputWindow
# (PyVista's own public API). No Observer ⇒ no logging ⇒ no loop. The genuine
# VTK warnings are written to vtk_errors.log so the real cause is still visible.
# This is the same SetInstance() pattern pyvista already uses at import, so it
# is safe on Apple Silicon (unlike attaching a brand-new observed window).
try:
    from pyvista.core.utilities.observers import set_error_output_file
    set_error_output_file(str(Path(__file__).parent / "vtk_errors.log"))
except Exception as _vtk_log_exc:  # pragma: no cover
    print(f"[vtk] could not redirect VTK error output: {_vtk_log_exc}")

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
from traffic_lights import (
    build_traffic_lights as _build_traffic_lights,
    tick_all,
    build_light_mesh,
    update_light_mesh,
    build_light_glyphs,
)


import contextlib

@contextlib.contextmanager
def _suppress_vtk_warnings():
    """Temporarily silence VTK's WARN|/ERR| output window messages.

    VTK writes diagnostics through ``vtkOutputWindow`` (not Python's
    ``warnings`` module), so ``warnings.filterwarnings`` has no effect.
    We swap in a null output window for the duration of the block and
    restore the original on exit.  This is the standard VTK pattern for
    suppressing noisy-but-harmless loader warnings (e.g. vtkOBJReader's
    "unexpected data at end of line").
    """
    import vtk
    orig = vtk.vtkOutputWindow.GetInstance()
    null_win = vtk.vtkOutputWindow()
    null_win.SetGlobalWarningDisplay(0)
    vtk.vtkOutputWindow.SetInstance(null_win)
    try:
        yield
    finally:
        vtk.vtkOutputWindow.SetInstance(orig)


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


from car_mixin import CarMixin
from ped_mixin import PedMixin
from cyclist_mixin import CyclistMixin
from bus_mixin import BusMixin
from emergency_mixin import EmergencyMixin
from shadow_mixin import ShadowMixin
from route_mixin import RouteMixin
from ui_mixin import UIMixin
from analysis_mixin import AnalysisMixin
from weather_mixin import WeatherMixin
from tod_mixin import TODMixin
from parking_mixin import ParkingMixin
from terrain_mixin import TerrainMixin
from heatmap_mixin import HeatmapMixin
from sumo_mixin import SumoMixin
from demand_mixin import DemandMixin
from walk_mixin import WalkMixin
from scenario_mixin import ScenarioMixin
from flood_mixin import FloodMixin
from vtk_timers import add_timer
from render.survey_mixin import SurveyMixin
from render.lighting_mixin import LightingMixin
from flood_lab_mixin import FloodLabMixin

class DigitalTwinApp(CarMixin, PedMixin, CyclistMixin, BusMixin, EmergencyMixin,
                     WeatherMixin, TODMixin, ParkingMixin, TerrainMixin, HeatmapMixin,
                     SumoMixin, DemandMixin, WalkMixin,
                     ShadowMixin, RouteMixin, UIMixin, AnalysisMixin, ScenarioMixin,
                     FloodMixin, FloodLabMixin, SurveyMixin, LightingMixin):
    def __init__(self):
        # Guards run() against a half-built object: __init__ has a couple of
        # early `return`s (no buildings found, geometry fetch failed) that
        # only exit the constructor — Python still hands back a "successful"
        # object either way, so without this flag __main__'s app.run() would
        # proceed and crash with a confusing AttributeError on some
        # never-assigned attribute (e.g. self.buildings_mesh) instead of the
        # actual, already-printed root cause.
        self._init_ok = False

        self.args = parse_args()
        if self.args.gui:
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
                        setattr(self.args, _k, _v)
                except Exception as _e:
                    print(f"[gui] could not parse subprocess output: {_e}")
        self.args.coverage_jobs = max(0, int(self.args.coverage_jobs))
        self.args.ga_jobs = max(1, int(self.args.ga_jobs))
        self.args.ga_progress_every = max(1, int(self.args.ga_progress_every))
        self.args.n_cars = int(self.args.n_cars)
        self.args.car_detail = str(getattr(self.args, "car_detail", "ultra")).strip().lower()
        if self.args.car_detail not in {"ultra", "low"}:
            self.args.car_detail = "ultra"
        self.args.traffic_speed = max(0.0, float(self.args.traffic_speed))
        self.args.debug_cars = bool(getattr(self.args, "debug_cars", False))
        self.args.solar_fleet = bool(getattr(self.args, "solar_fleet", False))
        self.args.flood_analysis = bool(getattr(self.args, "flood_analysis", False))
        self.args.flood_storm = str(getattr(self.args, "flood_storm", "v1_nov2025")).strip().lower()
        if self.args.flood_storm not in {"t2", "t10", "t10cc", "t50", "flat30", "v1_nov2025"}:
            self.args.flood_storm = "v1_nov2025"
        self.args.flood_phase = str(getattr(self.args, "flood_phase", "after")).strip().lower()
        if self.args.flood_phase not in {"before", "after"}:
            self.args.flood_phase = "after"
        self.args.solo = bool(getattr(self.args, "solo", False))
        self.args.light_strategy = str(getattr(self.args, "light_strategy", "smart")).strip().lower()
        if self.args.light_strategy not in {"smart", "ga"}:
            self.args.light_strategy = "smart"
        if self.args.solo:
            self.args.n_cars = 1
            self.args.car_detail = "ultra"


        # Wire data source choice to app_core's conditional import.
        # Must happen before app_core is first imported so its module-level
        # conditional runs with the correct env var.
        os.environ["CITY_DATA_SOURCE"] = getattr(self.args, "data_source", "overture")
        global _combine_ground_surfaces, _build_grid_points_from_ground_mesh, _build_ground_mesh_for_tests, _build_octree_from_buildings, _build_spotlight_discs, _cache_key, _initial_light_positions, _load_or_build_coverage_matrix_cached, _load_or_build_spatial_cache, _load_or_fetch_osm_cached, _normalize_ground_mesh, _street_direction_arrows, _street_node_markers, _street_line_layers, _load_hdri, _apply_atmosphere, _sun_dir_from_hour
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

            requested_threads = max(self.args.ga_jobs, self.args.coverage_jobs if self.args.coverage_jobs > 0 else 0)
            if requested_threads > 0:
                numba.set_num_threads(min(requested_threads, os.cpu_count() or requested_threads))
            # FIXED: when coverage workers will run in parallel, cap main process at 1 Numba thread
            # to prevent N_cores² oversubscription from forked workers inheriting full thread pool
            if self.args.coverage_jobs == 0 or self.args.coverage_jobs > 1:
                numba.set_num_threads(1)
        except Exception:
            pass

        self.cache_dir = Path(self.args.cache_dir)
        self.use_cache = bool(self.args.fast_startup)
        self.effective_ground_res = int(self.args.ground_resolution)
        if self.args.fast_startup:
            # Lower default test-plane resolution in fast profile unless user already requested lower.
            self.effective_ground_res = min(self.effective_ground_res, 48)
        elif self.args.radius >= 300.0 and int(self.args.ground_resolution) == 80:
            # Keep interactive viewer responsive for large radii when using default resolution.
            self.effective_ground_res = 48
            print(
                "Large radius detected with default ground resolution; "
                f"using resolution={self.effective_ground_res} for interactive rendering."
            )
        self.cache_context_key = _cache_key(
            "v9_pbr_class",  # bump: buildings now carry building_class cell data for PBR split
            getattr(self.args, "data_source", "overture"),
            self.args.address,
            self.args.radius,
            self.args.height,
            self.effective_ground_res,
        )

        self.ground_mesh: pv.PolyData | None = None
        shadow_mask: np.ndarray | None = None
        self.best_positions: np.ndarray | None = None
        self.octree_root: OctreeNode | None = None
        road_surface_mesh: pv.PolyData | None = None
        sidewalk_surface_mesh: pv.PolyData | None = None
        self.vehicle_roads: pv.PolyData | None = None
        self.pedestrian_roads: pv.PolyData | None = None

        numba_threads = "n/a"
        try:
            import numba

            numba_threads = str(numba.get_num_threads())
        except Exception:
            pass
        self._stage(
            "Run config: "
            f"mode={self.args.mode}, fast_startup={self.args.fast_startup}, "
            f"coverage_jobs={self.args.coverage_jobs}, ga_jobs={self.args.ga_jobs}, "
            f"ga_progress_every={self.args.ga_progress_every}, "
            f"radius={self.args.radius}, lights={self.args.n_lights}, cars={self.args.n_cars}, "
            f"light_strategy={self.args.light_strategy}, "
            f"car_detail={self.args.car_detail}, traffic_speed={self.args.traffic_speed:.2f}, "
            f"solar_fleet={self.args.solar_fleet}, debug_cars={self.args.debug_cars}, "
            f"logical_cpu={os.cpu_count() or 1}, numba_threads={numba_threads}"
        )
        self._stage(f"Data source: {getattr(self.args, 'data_source', 'overture').upper()}")
        self._stage(f"Flags: no_view={self.args.no_view}, optimize_on_open={self.args.optimize_on_open}, mode={self.args.mode}")
        self._stage(f"Loading data for: {self.args.address}")
        t_osm = time.perf_counter()
        try:
            self.buildings_mesh, self.street_graph, road_surface_mesh, sidewalk_surface_mesh, self.places, self.water_mesh = _load_or_fetch_osm_cached(
                address=self.args.address,
                radius=self.args.radius,
                extrusion_height=self.args.height,
                cache_dir=self.cache_dir,
                use_cache=self.use_cache,
                data_source=getattr(self.args, "data_source", "overture"),
                use_dem=getattr(self.args, "use_dem", True),
                use_ms_buildings=getattr(self.args, "use_ms_buildings", True),
            )
        except Exception as exc:
            print(f"Failed to fetch/build geometry: {exc}")
            return
        self._stage(f"OSM + geometry ready in {time.perf_counter() - t_osm:.2f}s")

        # Drone-survey DTM becomes THE terrain sampler before anything samples it.
        self._init_survey_terrain()

        # Warm up Numba JIT synchronously so concurrent access is impossible.
        from shadow_engine import _warmup_numba_kernels
        self._stage("Warming up Numba JIT (first run only)...")
        _warmup_numba_kernels()
        self._stage("Numba ready")

        print(
            "Street graph loaded: "
            f"{self.street_graph.number_of_nodes()} nodes, {self.street_graph.number_of_edges()} edges"
        )


        # ── Sanitize road/sidewalk meshes: clip to scene bounding box ────────────
        # Overture road features are fetched by *intersecting* bbox, so a feature
        # whose geometry starts 90 km away and passes through the search area is
        # returned in full.  That one feature bloats the road mesh to 99 km and
        # poisons ground_mesh, the camera frustum, and shadow computation.
        # Fix: clip both meshes to buildings_mesh bounds + generous margin.
        if self.buildings_mesh.n_points > 0:
            _bx0, _bx1, _by0, _by1 = self.buildings_mesh.bounds[:4]
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
            self.ground_mesh = _normalize_ground_mesh(combined_ground)

        if not self.args.hide_roads:
            t_roads = time.perf_counter()
            try:
                self.vehicle_roads, self.pedestrian_roads = _street_line_layers(self.street_graph)
            except Exception as exc:
                print(f"Road layer extraction failed: {exc}")
            else:
                self._stage(f"Road layers extracted in {time.perf_counter() - t_roads:.2f}s")

        if self.buildings_mesh.n_points == 0:
            print("No buildings found in this area. Try a larger radius or a denser location.")
            return

        run_shadow = self.args.mode in {"shadow", "all"} and (self.args.no_view or self.args.optimize_on_open or self.args.mode == "shadow")
        run_ga = self.args.mode in {"ga", "all"} and (self.args.no_view or self.args.optimize_on_open or self.args.mode == "ga")

        if run_shadow:
            self._stage("Starting shadow stage...")
            t_shadow = time.perf_counter()
            try:
                if self.args.fast_startup:
                    self._stage("Preparing spatial cache for shadows...")
                    self.ground_mesh, self.octree_root = _load_or_build_spatial_cache(
                        cache_dir=self.cache_dir,
                        use_cache=self.use_cache,
                        cache_context_key=self.cache_context_key,
                        buildings_mesh=self.buildings_mesh,
                        ground_resolution=self.effective_ground_res,
                        preferred_ground_mesh=self.ground_mesh,
                    )
                else:
                    if self.ground_mesh is None:
                        self.ground_mesh = _normalize_ground_mesh(
                            _build_ground_mesh_for_tests(self.buildings_mesh, resolution=self.effective_ground_res)
                        )
                    self.octree_root = _build_octree_from_buildings(self.buildings_mesh)
                shadow_mask, lit_ratio = compute_shadows(
                    ground_mesh=self.ground_mesh,
                    octree_root=self.octree_root,
                    sun_dir=np.asarray(self.args.sun_dir, dtype=float),
                )
                shadowed = int(np.count_nonzero(shadow_mask))
                total = int(shadow_mask.size)
                print(f"Shadow test complete: {shadowed}/{total} ground triangles shadowed")
                print(f"Lit area ratio: {lit_ratio:.6f}")
                self._stage(f"Shadow stage done in {time.perf_counter() - t_shadow:.2f}s")
            except Exception as exc:
                print(f"Shadow test failed: {exc}")

        if run_ga:
            self._stage("Starting GA stage...")
            t_ga = time.perf_counter()
            try:
                if self.ground_mesh is None:
                    self.ground_mesh = _normalize_ground_mesh(
                        _build_ground_mesh_for_tests(self.buildings_mesh, resolution=self.effective_ground_res)
                    )
                if self.octree_root is None:
                    self.octree_root = _build_octree_from_buildings(self.buildings_mesh)

                if self.args.fast_startup:
                    self.ground_mesh, self.octree_root = _load_or_build_spatial_cache(
                        cache_dir=self.cache_dir,
                        use_cache=self.use_cache,
                        cache_context_key=self.cache_context_key,
                        buildings_mesh=self.buildings_mesh,
                        ground_resolution=self.effective_ground_res,
                        preferred_ground_mesh=self.ground_mesh,
                    )

                if self.args.light_strategy == "smart":
                    self.best_positions = np.asarray(self._compute_smart_lights(), dtype=float)
                    print(f"Smart light placement complete: {self.best_positions.shape[0]} lights")
                else:
                    coverage_matrix = None
                    sidewalk_polygon = None
                    if self.args.fast_startup:
                        self._stage("Building sidewalk polygon + candidate grid...")
                        sidewalk_polygon = build_sidewalk_polygon_from_street_graph(self.street_graph)
                        grid_points = _build_grid_points_from_ground_mesh(
                            self.ground_mesh,
                            self.args.grid_step,
                            sidewalk_polygon=sidewalk_polygon,
                        )
                        self._stage(f"Candidate points: {grid_points.shape[0]}")
                        self._stage("Loading/building coverage matrix...")
                        coverage_matrix = _load_or_build_coverage_matrix_cached(
                            cache_dir=self.cache_dir,
                            use_cache=self.use_cache,
                            cache_context_key=self.cache_context_key,
                            grid_points=grid_points,
                            ground_mesh=self.ground_mesh,
                            octree_root=self.octree_root,
                            street_graph=self.street_graph,
                            radius=self.args.light_radius,
                            pole_height=self.args.pole_height,
                            n_jobs=(None if self.args.coverage_jobs == 0 else self.args.coverage_jobs),
                        )

                    self._stage("Running GA optimization...")
                    ga_result = optimize_streetlights(
                        ground_mesh=self.ground_mesh,
                        n_lights=self.args.n_lights,
                        light_radius=self.args.light_radius,
                        w1=self.args.w1,
                        w2=self.args.w2,
                        grid_step=self.args.grid_step,
                        population_size=self.args.population,
                        generations=self.args.generations,
                        mutation_rate=self.args.mutation,
                        seed=self.args.seed,
                        octree_root=self.octree_root,
                        pole_height=self.args.pole_height,
                        use_precomputed_coverage=bool(self.args.fast_startup),
                        precomputed_coverage_matrix=coverage_matrix,
                        street_graph=self.street_graph,
                        sidewalk_polygon=sidewalk_polygon,
                        ga_jobs=max(1, int(self.args.ga_jobs)),
                        ga_progress_every=max(1, int(self.args.ga_progress_every)),
                        ga_verbose=True,
                    )
                    self.best_positions = np.asarray(ga_result["best_positions"], dtype=float)
                    print(f"GA test complete: best_cost={ga_result['best_cost']:.6f}")
                    print(f"GA lit ratio: {ga_result['lit_ratio']:.6f}")
                print("Best light coordinates (x, y):")
                for row in self.best_positions:
                    print(f"  {row[0]:.3f}, {row[1]:.3f}")
                self._stage(f"GA stage done in {time.perf_counter() - t_ga:.2f}s")
            except Exception as exc:
                print(f"GA test failed: {exc}")

        self._init_ok = True

    def run(self) -> None:
        """Open the 3D viewer. Must be called from the top-level __main__ block
        (not from __init__) so that macOS Cocoa has a valid NSRunLoop at the
        outermost Python frame when plotter.show() is called."""
        if not self._init_ok:
            print("[app] initialization did not complete (see the error above) — nothing to run.")
            return

        # Travel-time validation runs without the viewer (and even with --no-view).
        if int(getattr(self.args, "validate_od", 0)) > 0:
            self._run_travel_time_validation()

        # Congested validation
        if int(getattr(self.args, "validate_congested", 0)) > 0:
            try:
                from validation import validate_congested
                n_cong = int(self.args.validate_congested)
                out = f"validation_congested_{n_cong}pairs.json"
                validate_congested(
                    self.street_graph, self.car_anim, self.car_paths,
                    n_pairs=n_cong, seed=int(self.args.seed), out_path=out
                )
            except Exception as exc:
                print(f"[validate-cong] failed: {exc}")

        # Engine comparison
        if int(getattr(self.args, "validate_engines", 0)) > 0:
            try:
                from validation import validate_engine_comparison
                n_eng = int(self.args.validate_engines)
                _sumo_conn = None
                if hasattr(self, "sumo") and self.sumo.get("enabled"):
                    _sumo_conn = self.sumo.get("conn")
                validate_engine_comparison(
                    self.street_graph, self.car_paths, self.car_anim,
                    sumo_conn=_sumo_conn, n_pairs=n_eng, seed=int(self.args.seed)
                )
            except Exception as exc:
                print(f"[validate-engines] failed: {exc}")

        if self.args.mode != "view" and self.args.mode != "all":
            return
        if self.args.no_view:
            print(
                "Mesh generated successfully: "
                f"{self.buildings_mesh.n_points} points, {self.buildings_mesh.n_cells} cells"
            )
            return
        self._stage("Opening 3D viewer...")
        try:
            if self.ground_mesh is None:
                self.ground_mesh = _build_ground_mesh_for_tests(self.buildings_mesh, resolution=self.effective_ground_res)
            self.ground_mesh = _normalize_ground_mesh(self.ground_mesh)
            if self.best_positions is None:
                if self.args.light_strategy == "smart":
                    self.best_positions = self._compute_smart_lights()
                else:
                    self.best_positions = _initial_light_positions(self.ground_mesh, self.args.n_lights, self.args.seed)

            self.plotter = pv.Plotter(title="City Digital Twin", window_size=[1400, 900])
            self._stage("Plotter created")

            self.style_presets: dict[str, dict[str, object]] = {
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

            self.scene_state: dict[str, object] = {
                "hour": 12.0,
                "spot_radius": float(self.args.light_radius),
                "ground_actor": None,
                "spotlight_actor": None,
                "lights_actor": None,
                "street_arrows_actor": None,
                "is_night": False,
                "show_roads": not self.args.hide_roads,
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
                "scene_lat": float(self.street_graph.graph.get("scene_lat", np.nan)),
                "scene_lon": float(self.street_graph.graph.get("scene_lon", np.nan)),
                "route_alpha": 0.5,
                "route_hour": 12.0,
                "solar_params": SolarParams(),
                "solar_fleet": bool(self.args.solar_fleet),
                "solar_model_idx": None,
                "edge_costs_cache": {},
                "pareto_chart": None,
                "_ultra_car_actors": None,
                "tl_actor": None,
            }
            self.scene_state["route_hour"] = float(self.scene_state["hour"])

            self.car_paths, self.car_outgoing = self._extract_drivable_paths(self.street_graph, z_level=0.5)
            self._stage(f"Car paths extracted: {len(self.car_paths)} drivable edges")
            self.car_path_lengths = np.asarray([float(p["length"]) for p in self.car_paths], dtype=float)
            self.adj_left_paths  = np.array([p.get("adjacent_left_path",  -1) for p in self.car_paths], dtype=np.int64)
            self.adj_right_paths = np.array([p.get("adjacent_right_path", -1) for p in self.car_paths], dtype=np.int64)
            self.edge_shadow_frac = np.zeros((len(self.car_paths),), dtype=float)
            car_shape_low = pv.Cube(center=(0.0, 0.0, 0.0), x_length=2.2, y_length=1.1, z_length=0.5)
            # Build car_next_edges from the directed OSM-resolved graph. Overture
            # connector restrictions are ignored here because the car simulation
            # should follow OSM direction data only.
            self.car_next_edges = _build_next_edges(
                self.car_paths,
                self.car_outgoing,
                self.street_graph,
                include_overture=False,
            )
            
            self.roundabout_yield_map: dict[int, np.ndarray] = {}
            for p_idx, path in enumerate(self.car_paths):
                if path.get("junction") != "roundabout":
                    for next_p_idx_raw in self.car_next_edges[p_idx]:
                        next_p = int(next_p_idx_raw)
                        if self.car_paths[next_p].get("junction") == "roundabout":
                            ring_paths = []
                            for rp_idx, rp in enumerate(self.car_paths):
                                if rp["v"] == path["v"] and rp.get("junction") == "roundabout":
                                    ring_paths.append(rp_idx)
                            if ring_paths:
                                self.roundabout_yield_map[p_idx] = np.array(ring_paths, dtype=np.int64)
                                
            self._stage("Car routing pre-computed (OSM directions only)")

            # ── Load OBJ car models for ultra detail mode ────────────────────────
            _CAR_MODELS_DIR = Path(__file__).parent / "assets" / "models"
            _SOLAR_CAR_OBJ = "solarcar.obj"
            _CAR_OBJ_FILES = [
                "sedan-sports.obj", "hatchback-sports.obj", "suv.obj",
                "van.obj", "delivery.obj", "truck-flat.obj",
                _SOLAR_CAR_OBJ,
            ]
            self._SOLAR_CAR_COLOR = "#2ecc71"
            # Per-car body colors (varied fleet look)
            self._CAR_BODY_COLORS = [
                "#c0392b", "#2980b9", "#27ae60", "#f39c12",
                "#8e44ad", "#e74c3c", "#3498db", "#16a085",
                "#d35400", "#2c3e50", "#1abc9c", "#e67e22",
            ]
            self.car_obj_templates: list = []
            self.car_obj_lengths: list[float] = []  # bumper-to-bumper length per template (metres)
            self.car_solar_model_idx: int | None = None
            if self.args.car_detail == "ultra":
                for _fname in _CAR_OBJ_FILES:
                    _fpath = _CAR_MODELS_DIR / _fname
                    if not _fpath.exists():
                        continue
                    try:
                        if _fname == _SOLAR_CAR_OBJ:
                            _mtl_path = _CAR_MODELS_DIR / _fname.replace(".obj", ".mtl")
                            with _suppress_vtk_warnings():
                                _parts = _load_obj_without_render_window(
                                    self.plotter, str(_fpath), str(_mtl_path), str(_CAR_MODELS_DIR)
                                )
                            _n_pts = sum(p[0].n_points for p in _parts)
                            _min_x = min(p[0].bounds[0] for p in _parts)
                            _max_x = max(p[0].bounds[1] for p in _parts)
                            _car_len = float(_max_x - _min_x)
                            self.car_solar_model_idx = len(self.car_obj_templates)
                            self.car_obj_templates.append(_parts)
                            self.car_obj_lengths.append(_car_len)
                            print(f"[cars] loaded solarcar with {len(_parts)} materials ({_n_pts} pts, length={_car_len:.2f} m)")
                            continue

                        with _suppress_vtk_warnings():
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
                        self.car_obj_templates.append(_raw)
                        self.car_obj_lengths.append(_car_len)
                        print(f"[cars] loaded OBJ model: {_fname} ({_raw.n_points} pts, length={_car_len:.2f} m)")
                    except Exception as _exc:
                        print(f"[cars] failed to load {_fname}: {_exc}")
                if not self.car_obj_templates:
                    print("[cars] no OBJ models found in assets/models; falling back to sphere mode")
            self.scene_state["solar_model_idx"] = self.car_solar_model_idx
            if bool(self.args.solar_fleet):
                if self.car_solar_model_idx is None:
                    print("[cars] --solar-fleet requested but solarcar.obj missing; disabling solar fleet")
                    self.args.solar_fleet = False
                    self.scene_state["solar_fleet"] = False
                else:
                    print(f"[cars] solar fleet: all cars use {_SOLAR_CAR_OBJ} (template index {self.car_solar_model_idx})")
            # ────────────────────────────────────────────────────────────────────

            # ── Build traffic lights ─────────────────────────────────────────────
            from traffic_lights import build_traffic_lights, tick_all, build_light_mesh, update_light_mesh, build_light_glyphs
            self.traffic_lights_dict = build_traffic_lights(
                self.street_graph,
                self.car_paths,
                min_degree=3,
                traffic_speed=float(self.args.traffic_speed),
            )
            self.scene_state["traffic_lights"] = self.traffic_lights_dict
            # ────────────────────────────────────────────────────────────────────

            # ── IDM (Intelligent Driver Model) parameters ────────────────────────
            # Acceleration model: a = A*(1 - (v/v0)^DELTA - (s*/s)^2)
            # s* = S0 + max(0, v*T + v*dv / (2*sqrt(A*B)))
            self._idm_params = IDMParams(
                a_max = 1.5,   # m/s²  max acceleration
                b     = 2.5,   # m/s²  comfortable deceleration
                T     = 1.5,   # s     desired time gap
                s0    = 2.0,   # m     minimum jam gap
                delta = 4,     # acceleration exponent
            )
            # ────────────────────────────────────────────────────────────────────

            if bool(self.args.debug_cars):
                total_len = float(np.sum(self.car_path_lengths)) if self.car_path_lengths.size > 0 else 0.0
                mean_len = total_len / float(max(1, len(self.car_paths)))
                print(
                    "[cars-debug] graph summary: "
                    f"drivable_paths={len(self.car_paths)}, total_path_len={total_len:.1f}m, mean_path_len={mean_len:.1f}m"
                )
                print(
                    "[cars-debug] settings: "
                    f"requested_cars={self.args.n_cars}, detail={self.args.car_detail}, traffic_speed={self.args.traffic_speed:.2f}"
                )

            # --profile: default to 200 cars when n_cars wasn't explicitly set
            if bool(getattr(self.args, "profile", False)) and int(self.args.n_cars) == -1:
                self.args.n_cars = 200
                print("[profile] --n-cars not set; defaulting to 200 for profiling run")

            n_cars = int(self.args.n_cars)
            n_parked_arg = int(getattr(self.args, "n_parked_cars", -1))

            # Automatic density scaling
            if n_cars == -1 or n_parked_arg == -1:
                total_buildings = self.buildings_mesh.n_cells if self.buildings_mesh is not None else 0

                if n_cars == -1:
                    # Scale with the road network, not the building count: one
                    # active car per ~35 m of drivable lane gives believable
                    # urban density on any city size.  IDM is fully vectorised
                    # (profiled at 200+ cars), so the CPU cap can sit at 300.
                    _road_m = float(np.sum(self.car_path_lengths)) if len(self.car_paths) else 0.0
                    n_cars = int(np.clip(_road_m / 35.0, 20, 300))
                    print(f"[cars] auto-density: {_road_m/1000.0:.1f} km of lane "
                          f"→ {n_cars} active cars")
                    
                if n_parked_arg == -1:
                    # total_buildings counts mesh TRIANGLES, not buildings —
                    # 25% of it gave five-digit parked fleets (13k+ in SF).
                    # Scale with kerb space instead: ~1 parked car per 20 m of
                    # lane, capped to keep actor count sane.
                    _road_m2 = float(np.sum(self.car_path_lengths)) if len(self.car_paths) else 0.0
                    self.args.n_parked_cars = int(np.clip(_road_m2 / 20.0, 20, 500))
                    print(f"[cars] auto parked: {self.args.n_parked_cars}")
                    
            n_cars = max(0, n_cars)

            self.car_rng = np.random.default_rng(int(self.args.seed) + 2027)
            self.car_anim: dict[str, object] = {
                "enabled": bool(n_cars > 0 and len(self.car_paths) > 0),
                "edge_idx": np.empty((0,), dtype=np.int64),
                "dist": np.empty((0,), dtype=float),
                "speed": np.empty((0,), dtype=float),
                "last_t": time.perf_counter(),
            }
            self.car_debug_state: dict[str, float | int] = {
                "tick": 0,
                "next_log_t": time.perf_counter() + 1.5,
            }
            if bool(self.car_anim["enabled"]):
                if np.sum(self.car_path_lengths) > 1e-9:
                    probs = self.car_path_lengths / np.sum(self.car_path_lengths)
                else:
                    probs = None
                edge_idx = self.car_rng.choice(len(self.car_paths), size=n_cars, replace=True, p=probs)
                dist = np.array(
                    [self.car_rng.uniform(0.0, float(self.car_path_lengths[int(i)])) for i in edge_idx],
                    dtype=float,
                )
                desired_speed_base = np.array(
                    [
                        self.car_paths[int(i)]["maxspeed_ms"] * self.car_rng.uniform(0.7, 1.0)
                        for i in edge_idx
                    ],
                    dtype=float,
                )
                speed = desired_speed_base * float(self.args.traffic_speed)
                self.car_anim["edge_idx"] = np.asarray(edge_idx, dtype=np.int64)
                self.car_anim["dist"] = dist
                self.car_anim["speed"] = speed.copy()           # actual instantaneous velocity (m/s)
                self.car_anim["desired_speed"] = speed.copy()   # free-flow target v_0 per car (m/s)
                self.car_anim["desired_speed_base"] = desired_speed_base.copy()  # per-car base v_0 (m/s)
                self.car_anim["accel"] = np.zeros(n_cars, dtype=float)  # current IDM acceleration
                res_nodes = [n for n, d in self.street_graph.nodes(data=True) if d.get("is_residential")]
                com_nodes = [n for n, d in self.street_graph.nodes(data=True) if d.get("is_commercial")]
                
                if not com_nodes and self.places:
                    try:
                        from scipy.spatial import cKDTree
                        n_ids = list(self.street_graph.nodes)
                        n_pts = np.array([[float(self.street_graph.nodes[n]["x"]), float(self.street_graph.nodes[n]["y"])] for n in n_ids])
                        tree = cKDTree(n_pts)
                        p_pts = np.array([[float(p["x"]), float(p["y"])] for p in self.places])
                        _, indices = tree.query(p_pts)
                        com_nodes = [n_ids[i] for i in indices]
                    except Exception:
                        pass
                if not res_nodes:
                    res_nodes = list(self.street_graph.nodes)
                    
                planned_edges = [None for _ in range(n_cars)]
                planned_cursor = np.zeros(n_cars, dtype=np.int64)
                edge_map = {(p["u"], p["v"]): idx for idx, p in enumerate(self.car_paths)}
                
                if res_nodes and com_nodes:
                    for c in range(n_cars):
                        src = self.car_rng.choice(res_nodes)
                        dst = self.car_rng.choice(com_nodes)
                        try:
                            path_nodes = nx.shortest_path(self.street_graph, src, dst, weight="length")
                            plan = []
                            for i in range(len(path_nodes)-1):
                                uv = (path_nodes[i], path_nodes[i+1])
                                if uv in edge_map: plan.append(edge_map[uv])
                                else: break
                            if plan:
                                planned_edges[c] = plan
                                edge_idx[c] = plan[0]
                                _path_len = float(self.car_path_lengths[int(plan[0])])
                                dist[c] = float(self.car_rng.uniform(0.0, max(_path_len * 0.3, 0.0)))
                        except nx.NetworkXNoPath:
                            pass
                            
                self.car_anim["planned_edges"] = planned_edges
                self.car_anim["planned_cursor"] = planned_cursor
                self.car_anim["edge_idx"] = np.asarray(edge_idx, dtype=np.int64)
                self.car_anim["dist"] = dist
                
                # Assign a fixed OBJ model index per car (for ultra mode)
                if self.car_obj_templates:
                    if bool(self.args.solar_fleet) and self.car_solar_model_idx is not None:
                        self.car_anim["model_idx"] = np.full(n_cars, self.car_solar_model_idx, dtype=np.int64)
                    else:
                        self.car_anim["model_idx"] = self.car_rng.integers(0, len(self.car_obj_templates), size=n_cars)
                    # Per-car bumper length from the assigned OBJ template
                    self.car_anim["car_len"] = np.array(
                        [self.car_obj_lengths[int(m) % len(self.car_obj_lengths)]
                         for m in self.car_anim["model_idx"]],
                        dtype=float,
                    )
                else:
                    self.car_anim["model_idx"] = np.zeros(n_cars, dtype=np.int64)
                    self.car_anim["car_len"] = np.full(n_cars, 4.0, dtype=float)  # default 4 m
                
                # stop_wait must be zeroed (IDM reads it on tick 1)
                self.car_anim["stop_wait"] = np.zeros(n_cars, dtype=float)

                # Per-car IDM parameter heterogeneity (seeded for reproducibility)
                _idm_rng = np.random.default_rng(int(getattr(self.args, "seed", 42)))
                _n_cars = int(len(self.car_anim.get("edge_idx", [])))
                self.car_anim["idm_T_arr"] = np.clip(_idm_rng.normal(1.4, 0.3, _n_cars), 0.8, 2.5).astype(float)
                self.car_anim["idm_a_arr"] = np.clip(_idm_rng.normal(1.6, 0.4, _n_cars), 0.9, 2.8).astype(float)
                self.car_anim["idm_b_arr"] = np.clip(_idm_rng.normal(2.0, 0.4, _n_cars), 1.0, 3.5).astype(float)
                self.car_anim["idm_speed_factor"] = np.clip(_idm_rng.normal(1.0, 0.12, _n_cars), 0.7, 1.3).astype(float)
                # Apply speed factor: scale each car's desired free-flow target by its individual multiplier
                self.car_anim["desired_speed_base"] *= self.car_anim["idm_speed_factor"]
                self.car_anim["desired_speed"] = (self.car_anim["desired_speed_base"] * float(self.args.traffic_speed)).copy()
                self.car_anim["speed"] = self.car_anim["desired_speed"].copy()

                if bool(self.args.debug_cars):
                    print(
                        "[cars-debug] initialized: "
                        f"active_cars={n_cars}, speed_mean={float(np.mean(speed)):.2f}m/s, speed_max={float(np.max(speed)):.2f}m/s"
                    )
            elif bool(self.args.debug_cars):
                if int(self.args.n_cars) <= 0:
                    print("[cars-debug] disabled: n-cars <= 0")
                elif len(self.car_paths) == 0:
                    print("[cars-debug] disabled: no drivable road paths found")

            # ── Gravity O-D demand model (replaces random turns) ──────────────
            self._init_demand()

            # ── Pedestrian agents ─────────────────────────────────────────────
            _n_peds = int(getattr(self.args, "n_peds", -1))
            self._init_peds(n_peds=_n_peds)

            # ── Cyclist agents ────────────────────────────────────────────────
            _n_cyclists = int(getattr(self.args, "n_cyclists", -1))
            self._init_cyclists(n_cyclists=_n_cyclists)

            # ── Bus agents (GTFS) ─────────────────────────────────────────────
            _gtfs_path = str(getattr(self.args, "gtfs", "") or "")
            _n_buses   = int(getattr(self.args, "n_buses", -1))
            self._init_buses(gtfs_path=_gtfs_path if _gtfs_path else None,
                             n_buses=_n_buses)

            # ── GTFS-Realtime live vehicle layer (optional) ───────────────────
            self._init_gtfs_rt(
                feed_url=str(getattr(self.args, "gtfs_rt_url", "") or ""),
                api_key=str(getattr(self.args, "gtfs_rt_key", "") or ""),
                api_key_param=str(getattr(self.args, "gtfs_rt_key_param", "") or ""),
                interval=float(getattr(self.args, "gtfs_rt_interval", 15.0)),
            )

            # ── SUMO engine mode: resolve --engine and auto-build scene if needed ─
            _engine = str(getattr(self.args, "engine", "idm") or "idm")
            _sumo_cfg_path = str(getattr(self.args, "sumo_cfg", "") or "")

            if _engine == "sumo" and not _sumo_cfg_path:
                # Auto-build the SUMO scene from the geocoded location + radius.
                # lat/lon are stored in the street graph after OSM load.
                _lat = float(self.street_graph.graph.get("scene_lat", 0.0) or 0.0)
                _lon = float(self.street_graph.graph.get("scene_lon", 0.0) or 0.0)
                _radius  = float(getattr(self.args, "radius", 300.0))
                _cdir    = str(getattr(self.args, "cache_dir", ".cache") or ".cache")
                try:
                    from sumo_engine import ensure_sumo_scene as _ensure_sumo
                    _sumo_cfg_path = _ensure_sumo(_lat, _lon, _radius, _cdir)
                except Exception as _se_exc:
                    print(f"[engine] sumo_engine import failed: {_se_exc}")
                    _sumo_cfg_path = ""
                if _sumo_cfg_path:
                    self.args.sumo_cfg = _sumo_cfg_path
                else:
                    print("[engine] WARNING: SUMO scene build failed — "
                          "falling back to IDM engine")
                    _engine = "idm"

            # Persist the resolved engine into scene_state so overlays can check it.
            self.scene_state["engine"] = _engine

            # ── SUMO co-simulation (optional) ─────────────────────────────────
            self._init_sumo(
                cfg=_sumo_cfg_path,
                binary=str(getattr(self.args, "sumo_binary", "sumo") or "sumo"),
                step_length=float(getattr(self.args, "sumo_step", 0.1)),
                use_gui=bool(getattr(self.args, "sumo_gui", False)),
                net_file=str(getattr(self.args, "sumo_net", "") or ""),
                port=(int(self.args.sumo_port) if getattr(self.args, "sumo_port", 0) else None),
            )

            # ── Emergency vehicles ────────────────────────────────────────────
            self._init_emergency()

            # ── Weather + Time-Of-Day ─────────────────────────────────────────
            self._init_weather()
            self._init_tod()

            # ── Parking simulation ────────────────────────────────────────────
            self._init_parking()

            # ── Walk / follow camera ──────────────────────────────────────────
            self._init_walk_cam()

            self.day_surface_cmap = [
                "#3a4048",  # road shadow  (slightly lighter — avoids near-black)
                "#5a6270",  # road lit     (medium asphalt — readable from above)
                "#6e7880",  # sidewalk shadow
                "#a8b2bc",  # sidewalk lit (light concrete)
            ]
            self.night_surface_cmap = [
                "#22272e",  # road dark
                "#5a616b",  # road single-lit
                "#ffd166",  # road double-lit
                "#3c434c",  # sidewalk dark
                "#8f97a2",  # sidewalk single-lit
                "#ffe4a3",  # sidewalk double-lit
            ]

            st = self._style()
            bg = st.get("day_bg")
            solid_bg = str(bg[0] if isinstance(bg, list) else bg)
            self.plotter.set_background(solid_bg)

            # ── Ground fill: base earth plane + OSM land-use colour patches ──────
            # Scene has water when either the extruded water mesh is non-empty or
            # the land-use fill contains water-coloured cells (#2e6ea6).
            _has_water = (self.water_mesh is not None and self.water_mesh.n_cells > 0)
            if not _has_water:
                _frgb = self.street_graph.graph.get("fill_rgb")
                if _frgb is not None and len(_frgb) > 0:
                    _frgb_arr = np.asarray(_frgb, dtype=np.int64)
                    _has_water = bool(np.any(np.all(_frgb_arr == [46, 110, 166], axis=1)))

            if self.buildings_mesh.n_points > 0:
                _bx0, _bx1, _by0, _by1 = self.buildings_mesh.bounds[:4]
                _pad = 100.0
                _base_pts = np.array([
                    [_bx0 - _pad, _by0 - _pad, -0.10],
                    [_bx1 + _pad, _by0 - _pad, -0.10],
                    [_bx1 + _pad, _by1 + _pad, -0.10],
                    [_bx0 - _pad, _by1 + _pad, -0.10],
                ], dtype=float)
                _base_mesh = pv.PolyData(_base_pts, faces=np.array([4, 0, 1, 2, 3]))
                self.plotter.add_mesh(
                    _base_mesh, color="#3a3f46",   # dark slate — matches land style
                    smooth_shading=False, lighting=False,
                )

                # Sea-background plane: closes the gap where the OSM water polygon
                # stops short of the viewport in open-sea areas. Sits above the
                # base ground (-0.10) but below the fill mesh (0.02), and extends
                # 3x beyond the scene (floored at 4000m) so the horizon reads as
                # water at any zoom level — for small --radius scenes, 3x the tiny
                # city bbox alone (e.g. ~1.5km pad for radius=250m) reads as a
                # small disconnected patch once the camera zooms out to an
                # establishing shot, rather than a proper horizon. Only added
                # when the scene actually contains water — inland scenes keep the
                # dark slate ground in untagged gaps.
                if _has_water:
                    _sea_pad = max(max(_bx1 - _bx0, _by1 - _by0) * 3.0, 4000.0)
                    _sea_pts = np.array([
                        [_bx0 - _sea_pad, _by0 - _sea_pad, -0.05],
                        [_bx1 + _sea_pad, _by0 - _sea_pad, -0.05],
                        [_bx1 + _sea_pad, _by1 + _sea_pad, -0.05],
                        [_bx0 - _sea_pad, _by1 + _sea_pad, -0.05],
                    ], dtype=float)
                    # lighting=True (VTK computes specular per-pixel, not just
                    # per-vertex, so a single flat quad already gets a correct
                    # sun-glint highlight across its whole surface — no extra
                    # geometry needed) + matching water_mesh's color/shading so
                    # this padding plane reads as a continuation of the real
                    # (small) OSM water polygon instead of a flat dead patch
                    # visibly seaming against it.
                    _sea_mesh = pv.PolyData(_sea_pts, faces=np.array([4, 0, 1, 2, 3]))
                    self.scene_state["sea_actor"] = self.plotter.add_mesh(
                        _sea_mesh, color="#3d7ab5",
                        smooth_shading=True, lighting=True, specular=0.6, specular_power=20,
                        name="sea_background",
                    )
                    self._stage("Sea-background plane added")

            _fill_pts   = self.street_graph.graph.get("fill_pts")
            _fill_faces = self.street_graph.graph.get("fill_faces")
            _fill_rgb   = self.street_graph.graph.get("fill_rgb")
            if (_fill_pts is not None and _fill_faces is not None and _fill_rgb is not None
                    and len(_fill_pts) > 0):
                try:
                    _fill_mesh = pv.PolyData(
                        np.asarray(_fill_pts, dtype=float),
                        np.asarray(_fill_faces, dtype=np.int64),
                    )
                    _fill_mesh.cell_data["RGB"] = np.asarray(_fill_rgb, dtype=np.uint8)
                    _fill_actor = self.plotter.add_mesh(
                        _fill_mesh, scalars="RGB", rgb=True,
                        smooth_shading=False, lighting=False,
                        show_scalar_bar=False,
                    )
                    self.scene_state["fill_actor"] = _fill_actor
                    self._stage(f"Land-use fill: {_fill_mesh.n_cells} patches")
                except Exception as _fe:
                    print(f"[fill] render failed: {_fe}")

            if self.water_mesh is not None and self.water_mesh.n_cells > 0:
                try:
                    self.scene_state["water_actor"] = self.plotter.add_mesh(
                        self.water_mesh, color="#3d7ab5",
                        smooth_shading=True, lighting=True, opacity=0.85,
                    )
                    self._stage(f"Water mesh: {self.water_mesh.n_cells} cells")
                except Exception as _we:
                    print(f"[water] render failed: {_we}")
            # ─────────────────────────────────────────────────────────────────────

            # Ortho-textured ground over the drone-survey footprint (WG)
            try:
                self._init_photoreal_ground()
            except Exception as _pg:
                print(f"[survey] photoreal ground skipped: {_pg}")

            # Terrain: must run AFTER fill mesh is rendered so it can replace it
            self._init_terrain()

            # Air-quality / noise heatmap (hidden until 'q' pressed)
            self._init_aq_overlay()

            # Beirut flood analysis (opt-in, --flood-analysis): import precomputed
            # flood-depth + green-corridor design and build the two static overlays.
            if bool(self.args.flood_analysis):
                try:
                    self._import_and_render_flood()
                except Exception as _fe:
                    print(f"[flood] flood analysis dispatch failed: {_fe}")

            self._stage(f"Adding buildings mesh ({self.buildings_mesh.n_cells} cells)...")

            # PBR environment is refreshed in _render_ground() for time-of-day HDRI.
            self.renderer = self.plotter.renderer
            # Configure SSAO parameters but do NOT enable yet.
            # SSAO + hundreds of text-label actors causes VTK's first render to hang
            # on macOS Cocoa. We defer SetUseSSAO(True) to a timer after show().
            try:
                self.renderer.SetSSAORadius(4.0)
                self.renderer.SetSSAOBias(0.025)
                self.renderer.SetSSAOKernelSize(128)
                self.renderer.SetSSAOBlur(True)
                print("[ssao] SSAO configured (deferred activation)")
            except Exception:
                print("[ssao] SSAO unavailable")
            try:
                self._init_postfx()
            except Exception as _pfx:
                self.postfx = None
                print(f"[postfx] post-processing unavailable ({_pfx}); legacy renderer path")

            import colorsys as _cs
            _pbr_rng = np.random.default_rng(int(self.args.seed) ^ 0x4321)
            _PBR_CLASSES = {
                0: dict(name="concrete", color="#c8b89a", metallic=0.0, roughness=0.85),
                1: dict(name="brick",    color="#b5724a", metallic=0.0, roughness=0.90),
                2: dict(name="glass",       color="#8cb8d8", metallic=0.0, roughness=0.08),
                3: dict(name="commercial",  color="#3498db", metallic=0.0, roughness=0.6),
                4: dict(name="residential", color="#e74c3c", metallic=0.0, roughness=0.7),
            }
            _pbr_actors: dict[str, object] = {}

            if "building_class" in (self.buildings_mesh.cell_data.keys() if self.buildings_mesh.n_cells > 0 else []):
                _bclass_arr = np.asarray(self.buildings_mesh.cell_data["building_class"])
                for _cid, _cp in _PBR_CLASSES.items():
                    _idx = np.where(_bclass_arr == _cid)[0]
                    if _idx.size == 0:
                        continue
                    _cmesh = self.buildings_mesh.extract_cells(_idx).extract_surface()
                    try:
                        _cmesh = _cmesh.compute_normals(
                            cell_normals=False, point_normals=True,
                            split_vertices=True, auto_orient_normals=True,
                        )
                    except Exception:
                        pass
                    _ch, _cl, _cs_ = _cs.rgb_to_hls(*pv.Color(_cp["color"]).float_rgb)
                    _ch = (_ch + _pbr_rng.uniform(-8.0 / 360.0, 8.0 / 360.0)) % 1.0
                    _cr, _cg, _cb = _cs.hls_to_rgb(_ch, _cl, _cs_)
                    _varied = "#{:02x}{:02x}{:02x}".format(int(_cr * 255), int(_cg * 255), int(_cb * 255))
                    _actor = self.plotter.add_mesh(
                        _cmesh,
                        pbr=True,
                        metallic=_cp["metallic"],
                        roughness=_cp["roughness"],
                        color=_varied,
                        smooth_shading=True,
                        # Triangle wireframe only in the legacy look; otherwise true
                        # feature edges are drawn once below (no wall diagonals).
                        show_edges=str(getattr(self.args, "render_quality", "quality")) == "legacy",
                        edge_color=str(st["building_edge"]),
                        line_width=0.4,
                        opacity=1.0,
                    )
                    _pbr_actors[_cp["name"]] = _actor
                    print(f"[pbr] {_cp['name']}: {_idx.size} cells (hue {_ch * 360:.0f}°)")
                # First actor is the backward-compat handle used by _render_ground
                self.scene_state["building_actor"] = next(iter(_pbr_actors.values()), None)
            else:
                    # Fallback: old cache without building_class — single flat-shaded actor
                    print("[pbr] building_class not found in mesh; using flat shading (re-run to rebuild cache)")
                    self.scene_state["building_actor"] = self.plotter.add_mesh(
                        self.buildings_mesh,
                        color=str(st["building"]),
                        show_edges=True,
                        edge_color=str(st["building_edge"]),
                        line_width=0.5,
                        opacity=0.95,
                        smooth_shading=True,
                    )

            self.scene_state["_building_actors_pbr"] = _pbr_actors
            if str(getattr(self.args, "render_quality", "quality")) != "legacy" and self.buildings_mesh.n_cells:
                try:
                    # Creases/silhouette-ready outlines: feature edges of the
                    # point-merged mesh, so coplanar triangulation diagonals vanish.
                    _outline = self.buildings_mesh.extract_surface(algorithm="dataset_surface").clean(
                        tolerance=1e-6).extract_feature_edges(
                        feature_angle=30.0, boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
                    if _outline.n_cells:
                        self.scene_state["building_outline_actor"] = self.plotter.add_mesh(
                            _outline, color=str(st["building_edge"]), line_width=1.0, opacity=0.55,
                            name="building_outline", reset_camera=False)
                        self.scene_state["building_outline_actor"].PickableOff()   # never steal clicks
                except Exception as _oe:
                    print(f"[buildings] outline edges skipped: {_oe}")
            try:
                self._init_photoreal_roofs()
            except Exception as _pr:
                print(f"[survey] photoreal rooftops skipped: {_pr}")
            try:
                self._init_survey_structures()
            except Exception as _ps:
                print(f"[survey] survey structures skipped: {_ps}")
            try:
                self._init_facades()
            except Exception as _pf:
                print(f"[facades] skipped: {_pf}")

            try:
                self._init_flood_lab()
            except Exception as _fl:
                self.flood_lab = None
                print(f"[flood-lab] disabled: {_fl}")

            # ── Trees from OSM (natural=tree + landuse=forest) ────────────────────
            _tree_data = self.street_graph.graph.get("trees", [])
            if _tree_data:
                if len(_tree_data) > 600:
                    _tree_data = [_tree_data[i] for i in np.random.default_rng(42).permutation(len(_tree_data))[:600]]
                # Procedural Mediterranean species (render.trees), grouped by
                # (species, variant); flat seeds + templates feed the drape.
                from render.trees import add_tree_groups
                _txy_arr = np.array([[t["x"], t["y"]] for t in _tree_data], dtype=float)
                add_tree_groups(self.plotter, self.scene_state, "osm", _txy_arr,
                                tags=[str(t.get("species", "")) for t in _tree_data])
                self._stage(f"Trees: {len(_tree_data)} rendered (OSM)")

            if not _tree_data:
                # No OSM tree data — scatter street trees perpendicular to
                # residential/tertiary road edges (realistic urban canopy).
                try:
                    _rng_t  = np.random.default_rng(77)
                    _syn_xy = []
                    _NO_TREE = {"motorway", "trunk", "primary", "secondary",
                                "motorway_link", "trunk_link", "primary_link"}
                    for _u, _v, _ed in self.street_graph.edges(data=True):
                        if str(_ed.get("highway", "")) in _NO_TREE:
                            continue
                        _ux = float(self.street_graph.nodes[_u].get("x", 0))
                        _uy = float(self.street_graph.nodes[_u].get("y", 0))
                        _vx = float(self.street_graph.nodes[_v].get("x", 0))
                        _vy = float(self.street_graph.nodes[_v].get("y", 0))
                        _dx, _dy = _vx - _ux, _vy - _uy
                        _l = float(np.hypot(_dx, _dy))
                        if _l < 8.0:
                            continue
                        # Perpendicular unit vector
                        _px, _py = -_dy / _l, _dx / _l
                        # Place 1–2 trees per edge, 4 m off road, random side
                        for _ in range(1 + int(_l > 40)):
                            _t = _rng_t.uniform(0.2, 0.8)
                            _mx = _ux + _dx * _t
                            _my = _uy + _dy * _t
                            _side = _rng_t.choice([-1, 1])
                            if _rng_t.random() < 0.45:
                                _syn_xy.append([_mx + _px * 4.0 * _side,
                                                _my + _py * 4.0 * _side])
                    _syn_xy = _syn_xy[:500]
                    if _syn_xy:
                        from render.trees import add_tree_groups
                        add_tree_groups(self.plotter, self.scene_state, "syn", np.array(_syn_xy, dtype=float))
                        self._stage(f"Trees: {len(_syn_xy)} street trees (synthetic fallback)")
                except Exception as _te:
                    print(f"[trees] synthetic fallback failed: {_te}")

            _cross_data = self.street_graph.graph.get("crossings", [])
            if _cross_data:
                _STRIPE_W = 0.45    # stripe width along road direction (m)
                _STRIPE_N = 6       # white stripes per crossing
                _GAP = 0.45         # gap between stripes (m)
                _ROAD_HALF = 3.0    # half-width of crossing span (m)
                _Z_OFF = 0.025      # above road surface
                # Built FLAT — the terrain drape toggle lifts/restores the
                # stored actor like every other layer (baking DEM in here made
                # the stripes float when terrain was off and never re-drape).
                _total_cr = _STRIPE_N * _STRIPE_W + (_STRIPE_N - 1) * _GAP
                _s0_cr = -_total_cr / 2.0
                _cross_parts = []
                for _cd in _cross_data:
                    _cx, _cy = _cd["x"], _cd["y"]
                    _rdx, _rdy = _cd["dx"], _cd["dy"]
                    _pdx, _pdy = -_rdy, _rdx          # perpendicular = crossing direction
                    _cz = _Z_OFF
                    for _si in range(_STRIPE_N):
                        _sc = _s0_cr + _si * (_STRIPE_W + _GAP) + _STRIPE_W / 2.0
                        _sh = _STRIPE_W / 2.0
                        _spts = np.array([
                            [_cx + _pdx * (-_ROAD_HALF) + _rdx * (_sc - _sh),
                             _cy + _pdy * (-_ROAD_HALF) + _rdy * (_sc - _sh), _cz],
                            [_cx + _pdx * _ROAD_HALF + _rdx * (_sc - _sh),
                             _cy + _pdy * _ROAD_HALF + _rdy * (_sc - _sh), _cz],
                            [_cx + _pdx * _ROAD_HALF + _rdx * (_sc + _sh),
                             _cy + _pdy * _ROAD_HALF + _rdy * (_sc + _sh), _cz],
                            [_cx + _pdx * (-_ROAD_HALF) + _rdx * (_sc + _sh),
                             _cy + _pdy * (-_ROAD_HALF) + _rdy * (_sc + _sh), _cz],
                        ])
                        _cross_parts.append(pv.PolyData(_spts, faces=np.array([4, 0, 1, 2, 3])))
                if _cross_parts:
                    _cross_mesh = pv.MultiBlock(_cross_parts).combine(merge_points=False)
                    self.scene_state["crosswalks_actor"] = self.plotter.add_mesh(
                        _cross_mesh, color="#f5f3e8",
                        smooth_shading=False, pbr=True, roughness=0.95, metallic=0.0,
                    )
                    self._stage(f"Crossings: {len(_cross_data)} rendered")

            # ── Lane markings ─────────────────────────────────────────────────────
            # • Yellow solid centre line on every bidirectional road
            # • White dashed dividers between lanes on multi-lane (lanes_total ≥ 4)
            _PED_HW = {"footway", "pedestrian", "path", "cycleway", "steps", "track"}
            _MARK_Z   = 0.038          # just above road surface
            _CL_HW    = 0.12           # centre-line half-width (m)
            _DIV_HW   = 0.07           # lane-divider half-width (m)
            _DASH_LEN = 3.0            # dashed divider dash length (m)
            _DASH_GAP = 3.0            # gap between dashes (m)
            # Built FLAT — terrain drape lifts the stored actors (see crossings)

            _cl_parts  = []   # centre-line quads (yellow)
            _div_parts = []   # lane-divider quads (white dashed)
            _seen_cl   = set()

            for _mu, _mv, _mdata in self.street_graph.edges(data=True):
                _hw_str = str(_mdata.get("highway", "")).lower()
                if _hw_str in _PED_HW:
                    continue

                _geom = _mdata.get("geometry")
                if _geom is not None and hasattr(_geom, "coords"):
                    _mcoords = np.asarray(_geom.coords, dtype=float)[:, :2]
                else:
                    _mun = self.street_graph.nodes.get(_mu, {})
                    _mvn = self.street_graph.nodes.get(_mv, {})
                    if "x" not in _mun or "x" not in _mvn:
                        continue
                    _mcoords = np.array([[_mun["x"], _mun["y"]], [_mvn["x"], _mvn["y"]]])
                if len(_mcoords) < 2:
                    continue

                _is_oneway = bool(_mdata.get("oneway", False))
                _raw_ln = _mdata.get("lanes", 1)
                if isinstance(_raw_ln, list): _raw_ln = _raw_ln[0]
                try: _total_lanes = max(1, int(float(_raw_ln)))
                except Exception: _total_lanes = 1

                # Centre line: bidirectional roads only, deduplicated by edge pair
                if not _is_oneway:
                    _ck = frozenset([_mu, _mv])
                    if _ck not in _seen_cl:
                        _seen_cl.add(_ck)
                        for _si in range(len(_mcoords) - 1):
                            _sp0, _sp1 = _mcoords[_si], _mcoords[_si + 1]
                            _sv = _sp1 - _sp0
                            _sl = float(np.linalg.norm(_sv))
                            if _sl < 0.3:
                                continue
                            _sd = _sv / _sl
                            _sp = np.array([-_sd[1], _sd[0]])
                            _zc = _MARK_Z
                            _cl_parts.append(pv.PolyData(
                                np.array([
                                    [_sp0[0]-_sp[0]*_CL_HW, _sp0[1]-_sp[1]*_CL_HW, _zc],
                                    [_sp0[0]+_sp[0]*_CL_HW, _sp0[1]+_sp[1]*_CL_HW, _zc],
                                    [_sp1[0]+_sp[0]*_CL_HW, _sp1[1]+_sp[1]*_CL_HW, _zc],
                                    [_sp1[0]-_sp[0]*_CL_HW, _sp1[1]-_sp[1]*_CL_HW, _zc],
                                ]),
                                faces=np.array([4, 0, 1, 2, 3])
                            ))

                # Dashed lane dividers: only on roads with ≥ 4 total lanes (2 each way)
                if _total_lanes >= 4:
                    _lw = 3.0
                    _divs_per_dir = max(1, _total_lanes // 2) - 1  # internal dividers
                    for _di in range(_divs_per_dir):
                        _div_off = -(_di + 1.0) * _lw  # right-side offsets
                        try:
                            from shapely.geometry import LineString as _LS2
                            _bl2 = _LS2(_mcoords)
                            _dl = _bl2.offset_curve(_div_off)
                            if _dl.is_empty: continue
                            if _dl.geom_type == "MultiLineString": _dl = list(_dl.geoms)[0]
                            _dc = np.asarray(_dl.coords, dtype=float)
                        except Exception:
                            continue
                        if len(_dc) < 2: continue
                        # Accumulate arc-length along divider, place dashes
                        _dvecs = _dc[1:] - _dc[:-1]
                        _dlens = np.linalg.norm(_dvecs, axis=1)
                        _dcum  = np.concatenate(([0.0], np.cumsum(_dlens)))
                        _total_d = float(_dcum[-1])
                        _t = 0.0
                        while _t + _DASH_LEN < _total_d:
                            _t0, _t1 = _t, _t + _DASH_LEN
                            # Interpolate start/end of dash along the divider
                            def _interp(arc):
                                _ii = int(np.searchsorted(_dcum, arc, side="right")) - 1
                                _ii = min(_ii, len(_dc) - 2)
                                _frac = (arc - _dcum[_ii]) / max(_dlens[_ii], 1e-9)
                                return _dc[_ii] + _frac * (_dc[_ii+1] - _dc[_ii])
                            _pa, _pb = _interp(_t0), _interp(_t1)
                            _dv2 = _pb - _pa; _dlen2 = float(np.linalg.norm(_dv2))
                            if _dlen2 < 0.1: _t += _DASH_LEN + _DASH_GAP; continue
                            _dd = _dv2 / _dlen2
                            _dp = np.array([-_dd[1], _dd[0]])
                            _zd = _MARK_Z
                            _div_parts.append(pv.PolyData(
                                np.array([
                                    [_pa[0]-_dp[0]*_DIV_HW, _pa[1]-_dp[1]*_DIV_HW, _zd],
                                    [_pa[0]+_dp[0]*_DIV_HW, _pa[1]+_dp[1]*_DIV_HW, _zd],
                                    [_pb[0]+_dp[0]*_DIV_HW, _pb[1]+_dp[1]*_DIV_HW, _zd],
                                    [_pb[0]-_dp[0]*_DIV_HW, _pb[1]-_dp[1]*_DIV_HW, _zd],
                                ]),
                                faces=np.array([4, 0, 1, 2, 3])
                            ))
                            _t += _DASH_LEN + _DASH_GAP

            if _cl_parts:
                _cl_mesh = pv.MultiBlock(_cl_parts).combine(merge_points=False)
                self.scene_state["lane_marks_cl_actor"] = self.plotter.add_mesh(
                    _cl_mesh, color="#e8c438",
                    smooth_shading=False, pbr=False, roughness=0.98, metallic=0.0)
            if _div_parts:
                _div_mesh = pv.MultiBlock(_div_parts).combine(merge_points=False)
                self.scene_state["lane_marks_div_actor"] = self.plotter.add_mesh(
                    _div_mesh, color="#e8e8d8",
                    smooth_shading=False, pbr=False, roughness=0.98, metallic=0.0)
            _n_markings = len(_cl_parts) + len(_div_parts)
            if _n_markings:
                self._stage(f"Lane markings: {len(_cl_parts)} centre-line + {len(_div_parts)} divider segments")
            # ─────────────────────────────────────────────────────────────────────

            if self.vehicle_roads is not None:
                    self.scene_state["vehicle_actor"] = self.plotter.add_mesh(
                        self.vehicle_roads,
                        color=str(st["vehicle"]),
                        line_width=4,
                        opacity=1.0,
                        render_lines_as_tubes=True,
                    )
            if self.pedestrian_roads is not None:
                    self.scene_state["ped_actor"] = self.plotter.add_mesh(
                        self.pedestrian_roads,
                        color=str(st["ped"]),
                        line_width=2.5,
                        opacity=0.9,
                        render_lines_as_tubes=True,
                    )
            self._set_actor_visibility(self.scene_state["vehicle_actor"], bool(self.scene_state["show_roads"]))
            self._set_actor_visibility(self.scene_state["ped_actor"], bool(self.scene_state["show_roads"]))

            self._stage("Building + road meshes added to viewer")
            street_arrows = _street_direction_arrows(self.street_graph)
            if street_arrows is not None:
                    self.scene_state["street_arrows_actor"] = self.plotter.add_mesh(
                        street_arrows,
                        color="#00e5ff",
                        opacity=0.7,
                        smooth_shading=False,
                        lighting=False,
                    )

            if self.scene_state["street_arrows_actor"] is not None:
                    self._set_actor_visibility(self.scene_state["street_arrows_actor"], True)
                    print("[arrows] visibility set to True")
            else:
                    print("[arrows] ERROR: street_arrows_actor is None!")
            self._stage("Street arrows added")
            self.route_state = {
                    "stage": 0,
                    "source_node": None,
                    "target_node": None,
                    "route_actors": [],
                    "selected_car_idx": None,
                    "selected_marker_actor": None,
            }
            self._render_cars()
            self._stage("Cars rendered")
            if bool(self.args.debug_cars):
                    self._log_car_debug("initial-render")

            # ── Traffic light glyphs ─────────────────────────────────────────────
            self._tl_mesh = build_light_mesh(self.traffic_lights_dict)
            self.scene_state["_tl_mesh"] = self._tl_mesh
            if self._tl_mesh.n_points > 0:
                    _tl_sphere = pv.Sphere(radius=1.4, theta_resolution=10, phi_resolution=10)
                    _tl_glyphs = build_light_glyphs(self._tl_mesh, _tl_sphere)
                    self.scene_state["_tl_glyphs"] = _tl_glyphs
                    self.scene_state["tl_actor"] = self.plotter.add_mesh(
                        _tl_glyphs,
                        scalars="colors",
                        rgb=True,
                        smooth_shading=True,
                        pbr=True,
                        metallic=0.1,
                        roughness=0.4,
                        lighting=True,
                    )
                    self._set_actor_visibility(self.scene_state["tl_actor"], bool(self.scene_state["show_traffic_signals"]))
                    self._stage(f"Traffic lights: {self._tl_mesh.n_points} intersections")
                    self.scene_state["_update_traffic_light_colors"] = self._update_traffic_light_colors
            else:
                    self.scene_state["tl_actor"] = None

            # ── Terrain draping is OPT-IN via the Terrain checkbox ────────────────
            # The scene always starts flat; _toggle_terrain shows the surface and
            # applies the drape together.  Auto-draping at startup forced hilly
            # cities into a broken half-lifted view before the user asked for it.

            # ── Map click: road info + route planning (source → target) ──────────
            # Shapely imports moved to top of file
            self._road_edge_data: list[dict] = []
            for _u, _v, _d in self.street_graph.edges(data=True):
                    _geom = _d.get("geometry")
                    if _geom is None:
                        _nu = self.street_graph.nodes.get(_u, {})
                        _nv = self.street_graph.nodes.get(_v, {})
                        if "x" in _nu and "x" in _nv:
                            _geom = _LS([(_nu["x"], _nu["y"]), (_nv["x"], _nv["y"])])
                    if _geom is not None:
                        self._road_edge_data.append({
                            "geom": _geom,
                            "highway": _d.get("highway", "unclassified"),
                            "maxspeed": _d.get("maxspeed"),
                            "lanes": _d.get("lanes"),
                            "surface": _d.get("surface"),
                            "oneway": _d.get("oneway", False),
                            "width_m": _d.get("width_m"),
                        })

            # route_state initialization moved before _render_cars()

            def _on_escape():
                # Exit walk/follow mode first; always clear road info overlay
                if self.scene_state.get("walk_mode") is not None:
                    self._walk_exit()
                self.plotter.add_text("", position=(0.68, 0.90), name="road_info_overlay")
            self.plotter.add_key_event("Escape", _on_escape)

            # pyvista registers its own default key bindings at plotter creation:
            #   'v' → isometric_view_interactive()  (camera reset — clobbers centrality)
            #   'b' → emulated left-button press    (clobbers bus-route toggle)
            #   'C' → enable_cell_picking()         (hijacks the click pipeline)
            #   'Up'/'Down' → camera zoom           (fights our arrow-key pan/walk)
            # Clear them so only our bindings fire.
            for _k in ("v", "b", "C", "Up", "Down"):
                try:
                    self.plotter.clear_events_for_key(_k)
                except Exception:
                    try:
                        self.plotter.iren.clear_events_for_key(_k)
                    except Exception:
                        pass

            # Editor Mode State
            self.scene_state["editor_mode"] = "view"

            self._set_editor_mode("view")
            self.plotter.add_key_event("1", lambda: self._set_editor_mode("view"))
            self.plotter.add_key_event("2", lambda: self._set_editor_mode("roads"))
            def _key3_roundabouts():
                self._set_editor_mode("roundabouts")
                # VTK's default '3' key handler toggles anaglyph stereo rendering,
                # turning the whole screen magenta. Cancel it immediately.
                try:
                    self.plotter.render_window.StereoRenderOff()
                except Exception:
                    pass
            self.plotter.add_key_event("3", _key3_roundabouts)
            self.plotter.add_key_event("4", lambda: self._set_editor_mode("lights"))
            self.plotter.add_key_event("5", lambda: self._set_editor_mode("stops"))
            self.plotter.add_key_event("6", lambda: self._set_editor_mode("streetlights"))
            # 'g' = buildings editor ('7' belongs to scenario save A)
            self.plotter.add_key_event("g", lambda: self._set_editor_mode("buildings"))
            # 'y' = highway/bridge editor
            self.plotter.add_key_event("y", lambda: self._set_editor_mode("highway"))
            # ── Green-corridor authoring tools (Part 4) ─────────────────────────
            # 'l'/'j'/'k' chosen after checking every add_key_event call in the
            # whole repo, including ones registered lazily at runtime (not just
            # in this file): 't' looked free here but tod_mixin.py's
            # _animate_tod() registers 't' -> _toggle_tod() on its first tick,
            # which fires AFTER this setup block and would silently steal the
            # binding (pyvista's add_key_event overwrites same-key callbacks),
            # making the trees tool unreachable. 'r' = camera elevation and
            # 'u' = SUMO congestion toggle (below) were already taken too.
            # Only 'l' and 'q' were free repo-wide; 'l' used here, 'q' left free.
            self.plotter.add_key_event("l", lambda: self._set_editor_mode("trees"))
            self.plotter.add_key_event("j", lambda: self._set_editor_mode("greenspace"))
            self.plotter.add_key_event("k", lambda: self._set_editor_mode("stairs"))
            self.plotter.add_key_event("S", lambda: self._set_editor_mode("strip"))
            self.plotter.add_key_event("D", lambda: self._set_editor_mode("drains"))
            self.plotter.add_key_event("F", lambda: self.flood_lab_run(design=False) if getattr(self, "flood_lab", None) else None)
            self.plotter.add_key_event("H", lambda: self.flood_lab_fly_hotspot() if getattr(self, "flood_lab", None) else None)
            self.plotter.add_key_event("E", lambda: self.flood_lab_run(design=True) if getattr(self, "flood_lab", None) else None)
            # Finalize an in-progress greenspace polygon (only acts in that mode)
            self.plotter.add_key_event("Return", lambda: self._finalize_greenspace())
            # ─────────────────────────────────────────────────────────────────────

            # ── Scenario comparison keys ──────────────────────────────────────────
            self._init_scenarios()
            self.plotter.add_key_event("7", lambda: self._scenario_save("A"))
            self.plotter.add_key_event("8", lambda: self._scenario_save("B"))
            self.plotter.add_key_event("9", lambda: self._scenario_run_comparison())
            self.plotter.add_key_event("0", lambda: self._scenario_toggle_diff())
            # ─────────────────────────────────────────────────────────────────────

            # ── Analysis overlay keys ─────────────────────────────────────────────
            self.plotter.add_key_event("h", lambda: self._toggle_heatmap())
            self.plotter.add_key_event("n", lambda: self._toggle_noise_map())
            self.plotter.add_key_event("v", lambda: self._toggle_centrality())  # 'c' reserved for camera reset

            # ── SUMO / transit keys — bound up-front so they never fail silently.
            # (Previously 'u'/'i' only registered inside the SUMO tick and 'b'
            # inside the bus tick, so without --engine sumo / --gtfs the keys
            # did nothing with no explanation.)
            def _key_hint(msg: str) -> None:
                print(f"[keys] {msg}")
                try:
                    self.plotter.add_text(msg, position=(0.30, 0.08),
                                          name="key_hint", font_size=10,
                                          viewport=True, color="#ffcc66")
                except Exception:
                    pass

            def _sumo_key_state() -> str:
                """'ok' | 'starting' | 'off' — for accurate key hints."""
                rt = getattr(self, "sumo", None)
                if rt and rt.get("enabled"):
                    return "ok"
                if self.scene_state.get("engine") == "sumo" or (
                        rt is not None and str(getattr(self.args, "engine", "")) == "sumo"):
                    return "starting"
                return "off"

            def _key_u():
                st = _sumo_key_state()
                if st == "ok":
                    self._sumo_toggle_congestion()
                elif st == "starting":
                    _key_hint("'u': SUMO not connected yet — check terminal for [sumo] messages")
                else:
                    _key_hint("'u' congestion overlay needs SUMO — restart with --engine sumo")

            def _key_i():
                st = _sumo_key_state()
                if st == "ok":
                    self._sumo_trigger_incident()
                elif st == "starting":
                    _key_hint("'i': SUMO not connected yet — check terminal for [sumo] messages")
                else:
                    _key_hint("'i' breakdown incident needs SUMO — restart with --engine sumo")

            def _key_b():
                has_buses = bool(getattr(self, "buses", None))
                has_live  = bool((getattr(self, "gtfs_rt", None) or {}).get("enabled"))
                if has_buses or has_live:
                    self._toggle_transit_overlay()
                else:
                    _key_hint("'b' bus overlay needs GTFS — restart with --gtfs <feed.zip>")

            self.plotter.add_key_event("u", _key_u)
            self.plotter.add_key_event("i", _key_i)
            self.plotter.add_key_event("b", _key_b)
            # ─────────────────────────────────────────────────────────────────────

            # ── Init analysis grid (after ground_mesh is guaranteed to be set) ───
            self._init_analysis()
            # ─────────────────────────────────────────────────────────────────────

            # ── POI labels ───────────────────────────────────────────────────────
            from overture_source import _poi_color

            # Short ASCII category tags — VTK text renderer cannot handle Unicode
            self._CAT_ABBREV: dict[str, str] = {
                    "restaurant": "Rest", "cafe": "Cafe", "bar": "Bar",
                    "fast_food": "Food", "bakery": "Bakery",
                    "supermarket": "Mkt", "convenience_store": "Conv",
                    "pharmacy": "Rx", "clothing_store": "Shop",
                    "parking": "P", "gas_station": "Gas", "bus_stop": "Bus",
                    "bank": "Bank", "atm": "ATM", "hospital": "Hosp",
                    "school": "School", "hotel": "Hotel", "post_office": "Post",
                    "park": "Park", "gym": "Gym", "museum": "Mus", "church": "Ch",
            }

            if self.places:
                    display_places = [p for p in self.places if "residential" not in p.get("categories", [])]
                    if display_places:
                        _poi_z = 2.5
                        _poi_pts = np.array([[p["x"], p["y"], _poi_z] for p in display_places], dtype=float)
                        _poi_mesh = pv.PolyData(_poi_pts)
                        _poi_mesh["labels"] = [self._poi_label(p) for p in display_places]
                        _poi_mesh.point_data["colors"] = np.array(
                            [self._hex_to_rgb(_poi_color(p["categories"])) for p in display_places],
                            dtype=np.uint8,
                        )
                        self.scene_state["_poi_mesh"] = _poi_mesh
                    else:
                        self.scene_state["_poi_mesh"] = None
                    self.scene_state["poi_actor"] = None
                    self.scene_state["poi_labels_actor"] = None
                    self.scene_state["show_pois"] = False
                    self.scene_state["show_poi_names"] = False
                    print(f"[poi] {len(self.places)} POIs prepared for lazy loading")
            else:
                    self.scene_state["_poi_mesh"] = None
                    self.scene_state["poi_actor"] = None
                    self.scene_state["poi_labels_actor"] = None
                    self.scene_state["show_pois"] = False
                    self.scene_state["show_poi_names"] = False
                    print("[poi] No places found in this area")
            # ─────────────────────────────────────────────────────────────────────

            self.scene_state["lights_actor"] = None

            # Opt4: pre-allocate ground copies once so _render_ground avoids repeated .copy().
            self.scene_state["day_ground_copy"]   = self.ground_mesh.copy()
            self.scene_state["night_ground_copy"] = self.ground_mesh.copy()

            # Opt2: threading.Event to prevent duplicate shadow warmup runs.
            import threading as _threading
            _shadow_warmup_done = _threading.Event()
            self.scene_state["_shadow_warmup_done"] = _shadow_warmup_done

            # Fast initial render so the window appears immediately.
            initial_ground = self.ground_mesh.copy()
            initial_ground.cell_data["surface_class"] = self._compose_day_surface_classes(
                    np.zeros((initial_ground.n_cells,), dtype=bool)
            )
            self.scene_state["ground_actor"] = self.plotter.add_mesh(
                    initial_ground,
                    scalars="surface_class",
                    clim=[0, 3],
                    cmap=self.day_surface_cmap,
                    show_edges=False,
                    show_scalar_bar=False,
                    opacity=0.9,
                    name="ground_mesh",
            )
            self.scene_state["spotlight_actor"] = None
            self.plotter.add_text("Move the Hour slider to compute shadows", position=(0.18, 0.02), name="status", font_size=9, viewport=True)
            self._stage(f"Initial ground actor added ({initial_ground.n_cells} cells)")

            # --- Async shadow rendering infrastructure ---
            # A single-worker executor ensures only one shadow job runs at a time.
            # (concurrent.futures is already imported as `cf` at the top of the file)
            self._shadow_executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-bg")

            self.plotter.add_slider_widget(
                    self._on_time_change,
                    rng=[0.0, 24.0],
                    value=12.0,
                    title="Hour",
                    pointa=(0.19, 0.14),
                    pointb=(0.41, 0.14),
                    style="modern",
                    interaction_event="end",
                    title_height=0.018,
                    slider_width=0.025,
                    tube_width=0.008,
            )
            self.plotter.add_slider_widget(
                    self._on_radius_change,
                    rng=[max(5.0, self.args.light_radius * 0.4), self.args.light_radius * 3.0],
                    value=float(self.args.light_radius),
                    title="Light radius",
                    pointa=(0.19, 0.08),
                    pointb=(0.41, 0.08),
                    style="modern",
                    interaction_event="end",
                    title_height=0.018,
                    slider_width=0.025,
                    tube_width=0.008,
            )

            # ── Clean upper-left control panel ───────────────────────────────────
            # Layout: checkbox at x=8px, label text at x_norm=0.042 (~59px).
            # Rows count down from PANEL_TOP in 30px steps (window height=900).
            # Sliders sit at the bottom (y_norm 0.08 / 0.14) RIGHT of the panel column
            # (x_norm 0.19-0.41): at x 0.02 they overlapped the flood rows 22-25.

            _CX = 8       # checkbox pixel x
            _CS = 22      # checkbox size (px)
            _TX = 35      # text pixel x (label)

            _TC = self._panel_text_color()
            _DC = self._panel_desc_color()

            # Backdrop first so it is drawn behind the labels and checkboxes
            try:
                self._add_panel_backdrop(2, self._cy(25 if (bool(getattr(self.args, 'flood_analysis', False)) or getattr(self, 'flood_lab', None) is None) else 22) - 22, 232, self._cy(0) + 30)
            except Exception as _bd:
                print(f"[ui] panel backdrop skipped: {_bd}")

            # ── Section: Visibility ───────────────────────────────────────────
            self.plotter.add_text("Controls", position=(_TX, self._cy(0) + 2),
                 name="panel_vis_hdr", font_size=11, color=_TC, viewport=False)

            _r1 = self._cy(1)
            self.plotter.add_text("Roads",   position=(_TX, _r1), name="panel_roads",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                    self._toggle_roads, value=bool(self.scene_state["show_roads"]),
                position=(_CX, _r1), size=_CS, color_on="#6aa06f", color_off="#3a3f4b",
            )

            _r2 = self._cy(2)
            self.plotter.add_text("Cars",    position=(_TX, _r2), name="panel_cars",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                    self._toggle_cars, value=bool(self.scene_state["show_cars"]),
                position=(_CX, _r2), size=_CS, color_on="#ff6b6b", color_off="#3a3f4b",
            )

            _r3 = self._cy(3)
            self.plotter.add_text("Arrows",    position=(_TX, _r3), name="panel_arrows",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                    self._toggle_arrows, value=True,
                position=(_CX, _r3), size=_CS, color_on="#ff3b30", color_off="#3a3f4b",
            )

            _r4 = self._cy(4)
            self.plotter.add_text("Streetlights",  position=(_TX, _r4), name="panel_lights",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._toggle_streetlights, value=bool(self.scene_state["show_streetlights"]),
                position=(_CX, _r4), size=_CS, color_on="#ffd740", color_off="#3a3f4b",
            )

            _r5 = self._cy(5)
            self.plotter.add_text("Signals",  position=(_TX, _r5), name="panel_signals",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._toggle_traffic_signals, value=bool(self.scene_state["show_traffic_signals"]),
                position=(_CX, _r5), size=_CS, color_on="#22cc55", color_off="#3a3f4b",
            )

            _r6 = self._cy(6)
            self.plotter.add_text("POIs",   position=(_TX, _r6), name="panel_pois",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._toggle_pois_lazy,
                value=False,
                position=(_CX, _r6), size=_CS, color_on="#e07b54", color_off="#3a3f4b",
            )

            _r7 = self._cy(7)
            self.plotter.add_text("Names",  position=(_TX, _r7), name="panel_names",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._toggle_poi_names_lazy,
                value=False,
                position=(_CX, _r7), size=_CS, color_on="#f5a623", color_off="#3a3f4b",
            )

            _r8 = self._cy(8)
            self.plotter.add_text("Place lights", position=(_TX, _r8), name="panel_optimize",   font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._on_optimize, value=False,
                position=(_CX, _r8), size=_CS, color_on="#ffd166", color_off="#3a3f4b",
            )

            _r_ssao = self._cy(9.2)
            self.plotter.add_text("SSAO",      position=(_TX, _r_ssao), name="panel_ssao",   font_size=9, color=_TC, viewport=False)

            # Post-FX chain: show the state the deferred activation will set.
            _ssao_enabled = (bool(getattr(self, "_postfx_ssao_wanted", False))
                             if getattr(self, "postfx", None) is not None else self._postfx_ssao_on())
            self.plotter.add_checkbox_button_widget(
                self._toggle_ssao, value=_ssao_enabled,
                position=(_CX, _r_ssao), size=_CS, color_on="#6cb6ff", color_off="#3a3f4b",
            )

            # Separator removed as it was rendered in 3D space instead of UI space

            # ── Section: Style presets ────────────────────────────────────────
            _rs = self._cy(10)
            self.plotter.add_text("Style", position=(_TX, _rs + 2),
                     name="panel_style_hdr", font_size=11, color=_TC, viewport=False)

            _r9 = self._cy(11)
            self.plotter.add_text("Mini",    position=(_TX, _r9), name="preset_mini_t",    font_size=9, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._preset_mini, value=True,
                position=(_CX, _r9), size=_CS, color_on="#ffd166", color_off="#3a3f4b",
            )

            _r10 = self._cy(12)
            self.plotter.add_text("Coastal", position=(_TX, _r10), name="preset_coastal_t",   font_size=9, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._preset_coastal, value=False,
                position=(_CX, _r10), size=_CS, color_on="#56b48a", color_off="#3a3f4b",
            )

            _r11 = self._cy(13)
            self.plotter.add_text("Sunset", position=(_TX, _r11), name="preset_sunset_t",   font_size=9, viewport=False)
            self.plotter.add_checkbox_button_widget(
                self._preset_sunset, value=False,
                position=(_CX, _r11), size=_CS, color_on="#ffb86b", color_off="#3a3f4b",
            )

            # ── Section: Environment toggles ──────────────────────────────────
            _re = self._cy(14)
            self.plotter.add_text("Environment", position=(_TX, _re + 2),
                     name="panel_env_hdr", font_size=11, color=_TC, viewport=False)

            _rw = self._cy(15)
            self.plotter.add_text("Weather", position=(_TX, _rw), name="panel_weather_t", font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                lambda _v: self._toggle_weather(), value=False,
                position=(_CX, _rw), size=_CS, color_on="#5bc4f0", color_off="#3a3f4b",
            )

            _rg = self._cy(16)
            self.plotter.add_text("Terrain", position=(_TX, _rg), name="panel_terrain_t", font_size=9, color=_TC, viewport=False)
            # Terrain starts OFF (flat scene) — checkbox turns surface+drape on
            _terrain_on = bool(self.scene_state.get("_terrain_visible", False))
            self.plotter.add_checkbox_button_widget(
                lambda _v: self._toggle_terrain() if hasattr(self, "_toggle_terrain") else None,
                value=_terrain_on,
                position=(_CX, _rg), size=_CS, color_on="#7ec87e", color_off="#3a3f4b",
            )

            _rq = self._cy(17)
            self.plotter.add_text("AQ overlay", position=(_TX, _rq), name="panel_aq_t", font_size=9, color=_TC, viewport=False)
            self.plotter.add_checkbox_button_widget(
                lambda _v: self._toggle_aq_overlay() if hasattr(self, "_toggle_aq_overlay") else None,
                value=False,
                position=(_CX, _rq), size=_CS, color_on="#e05c5c", color_off="#3a3f4b",
            )

            # Beirut flood puddles + green-corridor material overlays — only
            # meaningful (actors exist) when --flood-analysis built them at startup.
            # Rows 22/23 (not 18/20) to avoid colliding with the "Solar routing"
            # section below, which occupies rows 15/18/19/21.
            # (The Flood Lab replaces these precomputed-result toggles; they remain for --flood-analysis.)
            if bool(getattr(self.args, "flood_analysis", False)) or getattr(self, "flood_lab", None) is None:
                _rfl = self._cy(22)
                self.plotter.add_text("Flood puddles", position=(_TX, _rfl), name="panel_flood_t", font_size=9, color=_TC, viewport=False)
                self.plotter.add_checkbox_button_widget(
                    lambda _v: self._toggle_flood_overlay() if hasattr(self, "_toggle_flood_overlay") else None,
                    value=False,
                    position=(_CX, _rfl), size=_CS, color_on="#4aa3e0", color_off="#3a3f4b",
                )

                _rco = self._cy(23)
                self.plotter.add_text("Corridor GI", position=(_TX, _rco), name="panel_corridor_t", font_size=9, color=_TC, viewport=False)
                self.plotter.add_checkbox_button_widget(
                    lambda _v: self._toggle_corridor_overlay() if hasattr(self, "_toggle_corridor_overlay") else None,
                    value=False,
                    position=(_CX, _rco), size=_CS, color_on="#4ecb71", color_off="#3a3f4b",
                )

                # Puddle time-lapse play/pause — only does anything if the
                # selected storm/phase had depth_*.npy frame snapshots saved
                # (scripts/flood_gpu.py --save-every); otherwise _toggle_flood_animation
                # prints a clear "no time-series" message instead of silently no-op'ing.
                _rfa = self._cy(24)
                self.plotter.add_text("Flood time-lapse ▶", position=(_TX, _rfa), name="panel_flood_anim_t", font_size=9, color=_TC, viewport=False)
                self.plotter.add_checkbox_button_widget(
                    lambda _v: self._toggle_flood_animation() if hasattr(self, "_toggle_flood_animation") else None,
                    value=False,
                    position=(_CX, _rfa), size=_CS, color_on="#e0a84a", color_off="#3a3f4b",
                )

            # Photoreal survey ground vs stylized analysis ground (shadow and
            # night-lighting classes live on the stylized layer underneath).
            if self.scene_state.get("survey_ground_actor") is not None:
                _rpg = self._cy(25 if (bool(getattr(self.args, 'flood_analysis', False)) or getattr(self, 'flood_lab', None) is None) else 22)
                self.plotter.add_text("Photoreal ground", position=(_TX, _rpg), name="panel_photoreal_t", font_size=9, color=_TC, viewport=False)
                self.plotter.add_checkbox_button_widget(
                    lambda _v: (self._set_photoreal_ground(bool(_v)), self.plotter.render()),
                    value=bool(self.scene_state.get("photoreal_ground", False)),
                    position=(_CX, _rpg), size=_CS, color_on="#c9a86a", color_off="#3a3f4b",
                )

            # ── Flood Lab panel (right column) ────────────────────────────────
            if getattr(self, "flood_lab", None) is not None:
                try:
                    self._build_flood_lab_panel()
                except Exception as _flp:
                    import traceback
                    traceback.print_exc()
                    print(f"[flood-lab] panel failed: {_flp}")

            # ── Section: Solar routing (solarcar fleet only) ──────────────────
            if bool(self.scene_state.get("solar_fleet", False)):
                _r_sol = self._cy(19)
                self.plotter.add_text("Solar car", position=(_TX, _r_sol + 2),
                                 name="panel_solar_hdr", font_size=10, color=_TC, viewport=False)

                _r_alpha = self._cy(15)
                _r_hour  = self._cy(18)
                _r_area  = self._cy(21)

                self.plotter.add_slider_widget(
                    self._on_route_alpha_change,
                    rng=[0.0, 1.0],
                    value=float(self.scene_state.get("route_alpha", 0.5)),
                    title="Route mix",
                    pointa=(0.76, _r_alpha / 900.0),
                    pointb=(0.94, _r_alpha / 900.0),
                    style="modern",
                    interaction_event="always",
                    title_height=0.018,
                    slider_width=0.025,
                    tube_width=0.008,
                )
                self.plotter.add_slider_widget(
                    self._on_route_hour_change,
                    rng=[6.0, 18.0],
                    value=float(self.scene_state.get("route_hour", self.scene_state.get("hour", 12.0))),
                    title="Route hr",
                    pointa=(0.76, _r_hour / 900.0),
                    pointb=(0.94, _r_hour / 900.0),
                    style="modern",
                    interaction_event="always",
                    title_height=0.018,
                    slider_width=0.025,
                    tube_width=0.008,
                )
                self.plotter.add_slider_widget(
                    self._on_panel_area_change,
                    rng=[0.5, 3.0],
                    value=float(getattr(self.scene_state.get("solar_params"), "roof_area_m2", 1.6)),
                    title="Panel m2",
                    pointa=(0.76, _r_area / 900.0),
                    pointb=(0.94, _r_area / 900.0),
                    style="modern",
                    interaction_event="always",
                    title_height=0.018,
                    slider_width=0.025,
                    tube_width=0.008,
                )
            else:
                self.plotter.add_text(
                    "Solar routing: enable\n'All solar cars' at startup",
                    position=(_TX, self._cy(19)),
                    name="panel_solar_off",
                    font_size=8,
                    color=_DC,
                    viewport=False,
                )

            self.plotter.add_text(
                    "",
                    position=(0.55, 0.04),
                    name="route_stats_overlay",
                    font_size=8,
                    color="#f4f4f4",
                    viewport=True,
            )
            # ─────────────────────────────────────────────────────────────────────

            if self.args.optimize_on_open and self.best_positions is None:
                    self._stage(f"Optimize-on-open enabled: running {self.args.light_strategy} streetlight placement...")
                    t_open_ga = time.perf_counter()
                    if self.args.light_strategy == "smart":
                        self.best_positions = np.asarray(self._compute_smart_lights(), dtype=float)
                    else:
                        coverage_matrix = None
                        sidewalk_polygon = None
                        if self.args.fast_startup:
                            self._stage("Optimize-on-open: building sidewalk polygon + candidate grid...")
                            sidewalk_polygon = build_sidewalk_polygon_from_street_graph(self.street_graph)
                            grid_points = _build_grid_points_from_ground_mesh(
                                self.ground_mesh,
                                self.args.grid_step,
                                sidewalk_polygon=sidewalk_polygon,
                            )
                            self._stage(f"Optimize-on-open candidate points: {grid_points.shape[0]}")
                            self._stage("Optimize-on-open: loading/building coverage matrix...")
                            coverage_matrix = _load_or_build_coverage_matrix_cached(
                                cache_dir=self.cache_dir,
                                use_cache=self.use_cache,
                                cache_context_key=self.cache_context_key,
                                grid_points=grid_points,
                                ground_mesh=self.ground_mesh,
                                octree_root=self.octree_root,
                                street_graph=self.street_graph,
                                radius=self.args.light_radius,
                                pole_height=self.args.pole_height,
                                n_jobs=(None if self.args.coverage_jobs == 0 else self.args.coverage_jobs),
                            )

                        ga_result = optimize_streetlights(
                            ground_mesh=self.ground_mesh,
                            n_lights=self.args.n_lights,
                            light_radius=self.args.light_radius,
                            w1=self.args.w1,
                            w2=self.args.w2,
                            grid_step=self.args.grid_step,
                            population_size=self.args.population,
                            generations=self.args.generations,
                            mutation_rate=self.args.mutation,
                            seed=self.args.seed,
                            octree_root=self.octree_root,
                            pole_height=self.args.pole_height,
                            use_precomputed_coverage=bool(self.args.fast_startup),
                            precomputed_coverage_matrix=coverage_matrix,
                            street_graph=self.street_graph,
                            sidewalk_polygon=sidewalk_polygon,
                            ga_jobs=max(1, int(self.args.ga_jobs)),
                            ga_progress_every=max(1, int(self.args.ga_progress_every)),
                            ga_verbose=True,
                        )
                        self.best_positions = np.asarray(ga_result["best_positions"], dtype=float)
                    self.scene_state.pop("cached_night_coverage", None)
                    self.scene_state.pop("cached_night_key", None)
                    self._stage(f"Optimize-on-open placement finished in {time.perf_counter() - t_open_ga:.2f}s")
                    # Note: Don't call _render_ground() here; let it happen after viewer opens
                    # to avoid blocking on expensive coverage calculations before the window is responsive

            self.plotter.camera.ParallelProjectionOn()
            self.plotter.view_isometric()

            # Keyboard navigation for camera pan/rotate/zoom.
            x0, x1, y0, y1, _, _ = self.ground_mesh.bounds
            scene_span = max(1.0, float(max(x1 - x0, y1 - y0)))
            pan_step = 0.04 * scene_span
            rotate_step = 6.0

            # Store pan_step so WalkMixin arrow handlers can fall back to it
            self._walk_pan_step = pan_step

            # Arrow keys: walk-aware (WalkMixin routes to movement in free-walk mode)
            self.plotter.add_key_event("Left",  lambda: self._walk_arrow_left())
            self.plotter.add_key_event("Right", lambda: self._walk_arrow_right())
            self.plotter.add_key_event("Up",    lambda: self._walk_arrow_up())
            self.plotter.add_key_event("Down",  lambda: self._walk_arrow_down())
            # WASD: in free-walk mode these MOVE (WalkMixin._walk_key consumes
            # the press); otherwise a/s/d keep their camera-pan actions.
            # 'w' is walk-only — weather is toggled via its panel checkbox.
            self.plotter.add_key_event("a", lambda: self._walk_key("a") or self._pan_camera(-pan_step, 0.0))
            self.plotter.add_key_event("d", lambda: self._walk_key("d") or self._pan_camera(+pan_step, 0.0))
            self.plotter.add_key_event("w", lambda: self._walk_key("w"))
            self.plotter.add_key_event("s", lambda: self._walk_key("s") or self._pan_camera(0.0, -pan_step))
            # 'q' intentionally NOT bound: pyvista hard-registers q→close-window
            # ("Add no matter what"); the AQ overlay is toggled via its panel
            # checkbox instead, and the close binding is cleared at
            # interactive-ready (car_mixin._mark_interactive_ready).
            self.plotter.add_key_event("e", lambda: self._rotate_camera(+rotate_step, 0.0))
            self.plotter.add_key_event("r", lambda: self._rotate_camera(0.0, +rotate_step))
            # 'f' = follow / first-person / free-walk camera (WalkMixin)
            self.plotter.add_key_event("f", lambda: self._on_f_key())
            self.plotter.add_key_event("z", lambda: self._zoom_camera(1.12))
            self.plotter.add_key_event("x", lambda: self._zoom_camera(1.0 / 1.12))
            self.plotter.add_key_event("c", self._reset_camera_view)
            # 't' day/night is registered by tod_mixin (lazy, flag-guarded) —
            # registering here too made every press toggle TWICE (= no-op).
            # 'o' = undo last editor operation (roads/roundabouts/lights/stops/streetlights)
            self.plotter.add_key_event("o", lambda: self._editor_undo())
            if bool(self.args.debug_cars):
                    self.plotter.add_key_event("p", lambda: self._log_car_debug("manual"))
                    print("[cars-debug] press 'p' in the viewer to print car status")

            _runtime_timer_starters = []

            car_timer_ms = 80   # default; may be overridden below based on detail/car-count
            if bool(self.car_anim["enabled"]) and float(self.args.traffic_speed) > 0.0:
                    self.car_anim["last_t"] = time.perf_counter()
                    car_timer_ms = 80
                    _n_car_actors = max(0, int(self.args.n_cars))
                    if self.args.car_detail == "low":
                        car_timer_ms = max(car_timer_ms, 120)
                    elif self.args.car_detail == "ultra" and _n_car_actors >= 8:
                        car_timer_ms = 100
                    if self.args.car_detail == "ultra" and _n_car_actors >= 12:
                        car_timer_ms = 120
                    # Full plotter.render() every N sim ticks (macOS VTK is fragile at 12+ FPS).
                    self._car_render_stride = 4 if self.args.car_detail == "ultra" else 2
                    if self.ground_mesh.n_cells >= 120_000:
                        car_timer_ms = max(car_timer_ms, 90)
                        self._car_render_stride = max(self._car_render_stride, 3)
                        print(
                            "Large ground mesh detected "
                            f"({self.ground_mesh.n_cells} triangles): reducing car animation rate to ~{int(round(1000.0 / car_timer_ms))} FPS."
                        )
                    if self.args.car_detail == "ultra" and _n_car_actors >= 8:
                        print(
                            f"[cars] ultra OBJ mode: {_n_car_actors} actors, "
                            f"timer={car_timer_ms}ms, render every {self._car_render_stride} tick(s)"
                        )

            # Register unified picker and callbacks
            self._register_unified_picker()

            import sys

            if sys.platform != "darwin":
                # Windows/Linux: VTK's native interactor loop pumps the OS message
                # queue AND fires these NSTimer/Win-timer-backed callbacks.
                add_timer(self.plotter, 250, self._poll_shadow_job)
                if hasattr(self, "_scenario_poll"):
                    add_timer(self.plotter, 250, lambda *_: self._scenario_poll())
                add_timer(self.plotter, 500, self._poll_ga_done)
                if bool(self.car_anim["enabled"]) and float(self.args.traffic_speed) > 0.0:
                    add_timer(self.plotter, car_timer_ms, self._animate_cars)
                    print(f"[cars] animation timer registered ({car_timer_ms} ms)")
                if bool(self.ped_anim.get("enabled", False)):
                    add_timer(self.plotter, car_timer_ms, self._animate_peds)
                    print(f"[peds] animation timer registered ({car_timer_ms} ms)")
                if bool(self.cyclist_anim.get("enabled", False)):
                    add_timer(self.plotter, car_timer_ms, self._animate_cyclists)
                    print(f"[cyclists] animation timer registered ({car_timer_ms} ms)")
                if getattr(self, "buses", []) or getattr(self, "gtfs_rt", {}).get("enabled"):
                    add_timer(self.plotter, car_timer_ms, self._animate_buses)
                    print(f"[buses] animation timer registered ({car_timer_ms} ms)")
                add_timer(self.plotter, car_timer_ms, self._animate_emergency)
                print(f"[emergency] animation timer registered ({car_timer_ms} ms)")
                add_timer(self.plotter, car_timer_ms, self._animate_weather)
                print(f"[weather] animation timer registered ({car_timer_ms} ms)")
                add_timer(self.plotter, car_timer_ms, self._animate_tod)
                print(f"[tod] animation timer registered ({car_timer_ms} ms)")
                add_timer(self.plotter, 500, self._animate_parking)
                print("[parking] animation timer registered (500 ms)")
                add_timer(self.plotter, car_timer_ms, self._animate_aq_overlay)
                print(f"[heatmap] animation timer registered ({car_timer_ms} ms)")
                if hasattr(self, "_animate_flood_puddles"):
                    add_timer(self.plotter, car_timer_ms, self._animate_flood_puddles)
                    print(f"[flood] time-lapse animation timer registered ({car_timer_ms} ms)")
                if getattr(self, "sumo", {}).get("enabled"):
                    add_timer(self.plotter, car_timer_ms, self._animate_sumo)
                    print(f"[sumo] animation timer registered ({car_timer_ms} ms)")
                if bool(self.args.debug_cars):
                    if not bool(self.car_anim["enabled"]):
                        print("[cars-debug] animation timer not started: no active cars")
                    elif float(self.args.traffic_speed) <= 0.0:
                        print("[cars-debug] animation timer not started: traffic_speed <= 0")
                add_timer(self.plotter, 300, self._mark_interactive_ready, repeating=False)
                add_timer(self.plotter, 800, self._deferred_ssao_enable, repeating=False)
                add_timer(self.plotter, 500, self._deferred_initial_render, repeating=False)

            self._show_key_legend()
            self._print_startup_banner()

            self._stage("Calling plotter.show() — window should appear now")
            if sys.platform == "darwin":
                # macOS: VTK's native iren.Start() deadlocks under conda/Cocoa, so we
                # run a non-blocking show + manual event pump to keep Cocoa responsive.
                # This matches last_useful_main.py, which runs cleanly on this machine.
                self.plotter.show(auto_close=False, interactive_update=True)
                print("[viewer] Entering manual event loop (macOS)...")
                iren = getattr(self.plotter, "iren", None)

                t0 = time.perf_counter()
                fired = {"ready": False, "init": False, "ssao": False}
                last = {"shadow": 0.0, "ga": 0.0, "weather": 0.0, "tod": 0.0,
                        "parking": 0.0, "heatmap": 0.0, "tl": 0.0, "analysis": 0.0,
                        "flood": 0.0}
                # ── Fixed-rate physics (20 Hz) + per-frame interpolated render ──
                _PHYS_DT      = 0.050        # physics step: 50 ms = 20 Hz
                _TL_DTMS      = 100.0        # traffic-light color refresh: 10 Hz
                _OV_DTMS      = 333.0        # analysis overlays: 3 Hz
                _phys_accum   = 0.0          # wall-clock seconds since last physics step
                _frame_last_t = t0           # for per-frame delta
                cars_on      = bool(self.car_anim.get("enabled", False)) and float(self.args.traffic_speed) > 0.0
                peds_on      = bool(self.ped_anim.get("enabled", False))
                cyclists_on  = bool(self.cyclist_anim.get("enabled", False))
                buses_on     = bool(getattr(self, "buses", [])) or bool(getattr(self, "gtfs_rt", {}).get("enabled"))
                sumo_on      = bool(getattr(self, "sumo", {}).get("enabled"))
                emergency_on = True

                # ── Profiling setup ───────────────────────────────────────────────
                _prof = None
                _prof_end_ms = float("inf")
                if bool(getattr(self.args, "profile", False)):
                    from profiler import FrameProfiler
                    _prof = FrameProfiler()
                    _n_active = int(len(self.car_anim.get("edge_idx", [])))
                    _prof.start(n_cars=_n_active)
                    _prof_dur  = float(getattr(self.args, "profile_duration", 60.0))
                    _prof_end_ms = _prof_dur * 1000.0
                    print(f"[profile] active — {_prof_dur:.0f}s run, {_n_active} cars, "
                          f"output → {getattr(self.args, 'profile_output', 'profile_report.json')}")

                try:
                    while hasattr(self.plotter, "render_window") and getattr(self.plotter, "render_window") is not None and not getattr(self.plotter, "_closed", False):
                        # ── event_pump ────────────────────────────────────────────
                        if _prof: _pt = time.perf_counter()
                        if iren is not None:
                            vtk_iren = getattr(iren, "interactor", None)
                            if vtk_iren is not None and hasattr(vtk_iren, "ProcessEvents"):
                                vtk_iren.ProcessEvents()
                        if _prof: _prof.record("event_pump", (time.perf_counter() - _pt) * 1e3)

                        _now_t = time.perf_counter()
                        ms = (_now_t - t0) * 1000.0
                        _frame_dt = min(_now_t - _frame_last_t, 0.10)
                        _frame_last_t = _now_t

                        # one-shot deferred callbacks (not profiled — fire once only)
                        if not fired["ready"] and ms >= 300:
                            fired["ready"] = True; self._mark_interactive_ready(0)
                        if not fired["init"] and ms >= 500:
                            fired["init"] = True; self._deferred_initial_render(0)
                        if not fired["ssao"] and ms >= 800:
                            fired["ssao"] = True; self._deferred_ssao_enable(0)

                        # ── idm_sim: fixed-rate physics (20 Hz accumulator) ───────
                        if _prof: _pt = time.perf_counter()
                        _phys_accum += _frame_dt
                        _phys_ready  = bool(self.scene_state.get("interactive_ready", False))

                        while _phys_accum >= _PHYS_DT:
                            _phys_accum -= _PHYS_DT

                            if cars_on and _phys_ready:
                                _cp = self.car_anim.get("_curr_pos")
                                if _cp is not None:
                                    self.car_anim["_prev_pos"] = _cp
                                self._advance_cars(_PHYS_DT)
                                _new_pos = self._sample_car_positions()
                                self.car_anim["_curr_pos"] = _new_pos
                                self.car_anim["pos"]       = _new_pos
                                if self.car_anim.get("_prev_pos") is None:
                                    self.car_anim["_prev_pos"] = _new_pos.copy()
                                self._physics_tl_fsm(_PHYS_DT)

                            if peds_on and _phys_ready:
                                _pp = self.ped_anim.get("_curr_pos")
                                if _pp is not None:
                                    self.ped_anim["_prev_pos"] = _pp
                                self._advance_peds(_PHYS_DT)
                                _pnew = self._sample_ped_positions()
                                self.ped_anim["_curr_pos"] = _pnew
                                if self.ped_anim.get("_prev_pos") is None:
                                    self.ped_anim["_prev_pos"] = _pnew.copy()

                            if cyclists_on and _phys_ready:
                                _cyp = self.cyclist_anim.get("_curr_pos")
                                if _cyp is not None:
                                    self.cyclist_anim["_prev_pos"] = _cyp
                                self._advance_cyclists(_PHYS_DT)
                                _cynew = self._sample_cyclist_positions()
                                self.cyclist_anim["_curr_pos"] = _cynew
                                if self.cyclist_anim.get("_prev_pos") is None:
                                    self.cyclist_anim["_prev_pos"] = _cynew.copy()

                        if _prof: _prof.record("idm_sim", (time.perf_counter() - _pt) * 1e3)

                        # ── vtk_actors: interpolated render (every frame) ─────────
                        if _prof: _pt = time.perf_counter()
                        _alpha = min(_phys_accum / _PHYS_DT, 1.0)

                        if cars_on:
                            _pprev = self.car_anim.get("_prev_pos")
                            _pcurr = self.car_anim.get("_curr_pos")
                            if (_pprev is not None and _pcurr is not None
                                    and _pprev.shape == _pcurr.shape):
                                _ipos = _pprev + _alpha * (_pcurr - _pprev)
                                # Relocated cars (deadlock teleport, parking-exit
                                # recycle, reroute scatter) would otherwise streak
                                # across the map for one physics step: snap any car
                                # that moved further than physically possible.
                                _jump = np.linalg.norm(_pcurr[:, :2] - _pprev[:, :2], axis=1) > 15.0
                                if _jump.any():
                                    _ipos[_jump] = _pcurr[_jump]
                            else:
                                _ipos = _pcurr
                            self._render_cars(_ipos)

                        if peds_on:
                            _pprev = self.ped_anim.get("_prev_pos")
                            _pcurr = self.ped_anim.get("_curr_pos")
                            if (_pprev is not None and _pcurr is not None
                                    and _pprev.shape == _pcurr.shape):
                                _ipos = _pprev + _alpha * (_pcurr - _pprev)
                            else:
                                _ipos = _pcurr
                            self._render_peds(_ipos)

                        if cyclists_on:
                            _pprev = self.cyclist_anim.get("_prev_pos")
                            _pcurr = self.cyclist_anim.get("_curr_pos")
                            if (_pprev is not None and _pcurr is not None
                                    and _pprev.shape == _pcurr.shape):
                                _ipos = _pprev + _alpha * (_pcurr - _pprev)
                            else:
                                _ipos = _pcurr
                            self._render_cyclists(_ipos)

                        if buses_on:      self._animate_buses(0)
                        if emergency_on:  self._animate_emergency(0)
                        if sumo_on:       self._animate_sumo(_frame_dt)
                        self._animate_walk(0)

                        if _prof: _prof.record("vtk_actors", (time.perf_counter() - _pt) * 1e3)

                        # ── overlay: rate-limited UI updates ──────────────────────
                        if _prof: _pt = time.perf_counter()

                        if ms - last["tl"] >= _TL_DTMS:
                            last["tl"] = ms
                            self._render_tl_colors()

                        if ms - last["analysis"] >= _OV_DTMS:
                            last["analysis"] = ms
                            if hasattr(self, "_update_analysis"):
                                try:
                                    self._update_analysis()
                                except Exception:
                                    pass

                        if ms - last["shadow"] >= 250:
                            last["shadow"] = ms; self._poll_shadow_job(0)
                            if hasattr(self, "_scenario_poll"):
                                self._scenario_poll()
                        if ms - last["ga"] >= 500:
                            last["ga"] = ms; self._poll_ga_done(0)
                        if ms - last["weather"] >= car_timer_ms:
                            last["weather"] = ms; self._animate_weather(0)
                        if ms - last["tod"] >= car_timer_ms:
                            last["tod"] = ms; self._animate_tod(0)
                        if ms - last["parking"] >= 500:
                            last["parking"] = ms; self._animate_parking(0)
                        if ms - last["heatmap"] >= _OV_DTMS:
                            last["heatmap"] = ms; self._animate_aq_overlay(0)
                        if ms - last["flood"] >= _OV_DTMS and hasattr(self, "_animate_flood_puddles"):
                            last["flood"] = ms; self._animate_flood_puddles(0)

                        if _prof: _prof.record("overlay", (time.perf_counter() - _pt) * 1e3)

                        # ── render: VTK GPU flush ─────────────────────────────────
                        if _prof: _pt = time.perf_counter()
                        self.plotter.update()
                        if _prof: _prof.record("render", (time.perf_counter() - _pt) * 1e3)

                        time.sleep(0.005)

                        # ── end of frame ──────────────────────────────────────────
                        if _prof:
                            _prof.end_frame()
                            if ms >= _prof_end_ms:
                                print(f"\n[profile] {_prof_dur:.0f}s elapsed — stopping.")
                                break

                except KeyboardInterrupt:
                    print("\n[viewer] user interrupted – closing window")
                finally:
                    # ── profiling output ──────────────────────────────────────────
                    if _prof:
                        _prof.print_table()
                        _prof.save_json(getattr(self.args, "profile_output", "profile_report.json"))
                    self._close_sumo()
                    self.plotter.close()
                    print("[viewer] plotter.close() returned — viewer window closed")
            else:
                print("[viewer] entering native VTK event loop (non-macOS)...", flush=True)
                try:
                    self.plotter.show(auto_close=False)
                except KeyboardInterrupt:
                    print("\n[viewer] user interrupted – closing window")
                finally:
                    self._close_sumo()
                    self.plotter.close()
                    print("[viewer] plotter.close() returned — viewer window closed")

            # --- Extract and Print Interesting Simulation Statistics ---
            try:
                print("\n" + "="*50)
                print("SIMULATION STATISTICS SUMMARY")
                print("="*50)

                total_cells = self.buildings_mesh.n_cells if self.buildings_mesh is not None else 0
                _places = self.places if self.places is not None else []
                poi_count = len(_places)
                res_places = sum(1 for p in _places if "residential" in p.get("categories", []))
                print(f"Buildings geometry: {total_cells} cells")
                print(f"Points of Interest: {poi_count} (of which {res_places} are residential)")

                if self.street_graph is not None:
                    print(f"Road Network:       {self.street_graph.number_of_nodes()} nodes, {self.street_graph.number_of_edges()} edges")
                    n_stops = sum(1 for _, d in self.street_graph.nodes(data=True) if d.get("highway") == "stop")
                    print(f"Stop Signs:         {n_stops}")

                n_tlights = len(self.scene_state.get("traffic_lights", {}))
                print(f"Traffic Lights:     {n_tlights}")

                if self.best_positions is not None:
                    print(f"City Streetlights:  {self.best_positions.shape[0]}")

                if self.car_anim is not None and "dist" in self.car_anim:
                    print(f"Active Cars:        {len(self.car_anim['dist'])}")

                n_parked = getattr(self.args, 'n_parked_cars', 0)
                print(f"Parked Cars:        {n_parked}")

                print("="*50 + "\n")
            except Exception as _stat_exc:
                print(f"[stats] failed to generate summary statistics: {_stat_exc}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Visualization failed: {exc}")
    def _active_layers(self) -> list[str]:
        """Human-readable list of optional layers that actually loaded."""
        layers = []
        if bool(self.car_anim.get("enabled")) and float(self.args.traffic_speed) > 0.0:
            _cars = f"IDM cars (x{int(self.args.n_cars)})" if int(self.args.n_cars) >= 0 else "IDM cars"
            if getattr(self, "demand", None) is not None:
                _cars += " + gravity O-D demand"
            layers.append(_cars)
        if getattr(self, "sumo", {}).get("enabled"):
            layers.append("SUMO co-sim")
        if getattr(self, "gtfs_rt", {}).get("enabled"):
            layers.append("GTFS-Realtime buses")
        if getattr(self, "buses", []):
            layers.append(f"GTFS buses ({len(self.buses)})")
        if bool(self.ped_anim.get("enabled")):
            layers.append("pedestrians")
        if bool(self.cyclist_anim.get("enabled")):
            layers.append("cyclists")
        if getattr(self, "_parking_slots", None) is not None and len(getattr(self, "_parking_slots", [])) > 0:
            layers.append("parking sim")
        if self.scene_state.get("terrain_actor") is not None:
            layers.append("3D terrain")
        return layers

    def _print_startup_banner(self) -> None:
        """Console key-bindings + active-layers banner (discoverability)."""
        layers = self._active_layers()
        print("\n" + "=" * 64)
        print("  CITY DIGITAL TWIN — viewer ready")
        print("=" * 64)
        if layers:
            print("  Active layers: " + ", ".join(layers))
        print("  ── Keyboard ──────────────────────────────────────────────")
        print("    w  weather  (clear → rain → snow)     t  time-lapse day/night")
        print("    q  air-quality: NOx → CO₂ → PM → off  g  terrain on/off")
        print("    h  traffic heatmap   n  noise map     v  centrality overlay")
        print("    f  follow/1st-person/free-walk (cycle) — Esc to exit")
        print("    m  spawn emergency vehicle (max 3; again to clear)")
        print("    c  reset camera    a/d  pan   z/x  zoom   e/r  rotate")
        print("    arrows: pan camera  (in free-walk: ↑↓ move, ←→ turn)")
        print("    mouse-drag  orbit / look around")
        print("    7/8 save scenario A/B   9 run comparison   0 toggle diff")
        print("  ── On-screen ─────────────────────────────────────────────")
        print("    Left panel: Controls · Style · Environment (Weather/Terrain/AQ)")
        print("    Bottom-right: key legend   Bottom: mode bar [1-6]")
        print("=" * 64 + "\n")

    def _show_key_legend(self) -> None:
        """Compact, unobtrusive key legend in the bottom-right corner."""
        try:
            legend = (
                "t day/night   v centrality   h heatmap   n noise\n"
                "f follow/walk (click ped = 1st person)   c reset\n"
                "b bus routes   u congestion   i incident   m emergency\n"
                "g add building   o undo edit   weather/terrain/AQ → checkboxes"
            )
            self.plotter.add_text(
                legend,
                position=(0.74, 0.015),
                name="key_legend",
                font_size=8,
                viewport=True,
                color="#9fb3c8",
            )
        except Exception:
            pass

    def _run_travel_time_validation(self) -> None:
        """Compare model free-flow route times vs a routing engine (Tier-1 accuracy check)."""
        try:
            from validation import run_validation
        except Exception as exc:
            print(f"[validate] module unavailable: {exc}")
            return
        n = int(getattr(self.args, "validate_od", 0))
        engine = str(getattr(self.args, "validate_engine", "osrm"))
        host = str(getattr(self.args, "validate_osrm_host", "https://router.project-osrm.org"))
        out = f"validation_{engine}_{n}pairs.json"
        try:
            run_validation(
                self.street_graph, n_pairs=n, engine=engine,
                osrm_host=host, seed=int(getattr(self.args, "seed", 42)),
                out_path=out,
            )
        except Exception as exc:
            print(f"[validate] failed: {exc}")

    def _stage(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _pump_window_events(self) -> None:
        """Process pending window-manager events (minimize, resize, close)."""
        try:
            iren = self.plotter.iren
            if iren is not None:
                iren.ProcessEvents()
                return
        except Exception:
            pass
        try:
            rw = self.plotter.render_window
            if rw is not None and hasattr(rw, "ProcessEvents"):
                rw.ProcessEvents()
        except Exception:
            pass

    def _style(self) -> dict[str, object]:
        return self.style_presets[str(self.scene_state["preset"])]

    def _set_actor_visibility(self, actor: object, visible: bool) -> None:
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
        except Exception:
            pass

    def _parse_maxspeed(self, raw) -> float:
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

    def _extract_drivable_paths(self, graph, z_level: float = 0.38) -> tuple[list[dict[str, object]], dict[object, list[int]]]:
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

            # Bridge/overpass edges ride on an elevated ramped profile
            from app_core import _edge_bridge_height, _bridge_z_offsets
            _bridge_h = _edge_bridge_height(data)
            _z_col = np.full((xy.shape[0],), float(z_level), dtype=float)
            if _bridge_h > 0.0:
                _z_col += _bridge_z_offsets(xy, _bridge_h)

            points = np.column_stack((xy, _z_col))
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
            _raw_ms = data.get("maxspeed")
            if _raw_ms is None:
                # Infer from highway type when OSM maxspeed tag is absent
                _hw_str = str(data.get("highway", "")).lower()
                if _hw_str in ("motorway", "motorway_link"):
                    maxspeed = 33.3
                elif _hw_str in ("trunk", "trunk_link"):
                    maxspeed = 22.2
                elif _hw_str in ("primary", "primary_link"):
                    maxspeed = 13.9
                elif _hw_str in ("secondary", "secondary_link"):
                    maxspeed = 11.1
                elif _hw_str in ("service", "living_street"):
                    maxspeed = 5.6
                elif _hw_str in ("tertiary", "tertiary_link", "residential", "unclassified"):
                    maxspeed = 8.3
                else:
                    maxspeed = 13.9
            else:
                maxspeed = float(np.clip(self._parse_maxspeed(_raw_ms), 2.8, 41.7))
            seg_id = data.get("segment_id")

            base_line = LineString(points[:, :2])
            lane_indices = []

            # For right-hand (keep-right) traffic:
            # • oneway roads: spread all lanes symmetrically around the centreline
            # • bidirectional roads: each direction only owns half the lanes,
            #   all offset to the RIGHT side so opposing traffic stays separated.
            _oneway = bool(data.get("oneway", False))
            lanes_this_dir = num_lanes if _oneway else max(1, num_lanes // 2)

            for lane_i in range(lanes_this_dir):
                if _oneway:
                    # Symmetric spread: lane 0 is rightmost
                    offset_dist = (lane_i - (lanes_this_dir - 1) / 2.0) * lane_width
                else:
                    # Keep-right: all lanes on right side.
                    # Negative = right of direction-of-travel in Shapely convention.
                    offset_dist = -((lane_i + 0.5) * lane_width)

                if abs(offset_dist) < 1e-3:
                    lane_points = points
                else:
                    try:
                        offset_line = base_line.offset_curve(offset_dist)
                        if offset_line.is_empty:
                            lane_points = points
                        else:
                            if offset_line.geom_type == 'MultiLineString':
                                offset_line = list(offset_line.geoms)[0]
                            off_xy = np.asarray(offset_line.coords)
                            _lane_z = np.full((off_xy.shape[0],), float(z_level), dtype=float)
                            if _bridge_h > 0.0:
                                _lane_z += _bridge_z_offsets(off_xy[:, :2], _bridge_h)
                            lane_points = np.column_stack((off_xy, _lane_z))
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
                    "num_lanes": lanes_this_dir,
                    "adjacent_left_path": -1,
                    "adjacent_right_path": -1,
                    "junction": data.get("junction"),
                    "control": nv_data.get("highway"),
                    "highway": next(iter(hw_vals), "unclassified"),
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

if __name__ == '__main__':
    app = DigitalTwinApp()
    if not app._init_ok:
        raise SystemExit(1)
    app.run()
