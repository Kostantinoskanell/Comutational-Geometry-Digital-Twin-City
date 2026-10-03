"""FloodLabMixin — run a flood study inside the twin, design a corridor, compare.

The flow the app is built around:

  1. the city as it is today (baseline);
  2. [Run flood simulation]: the live numba solver (floodsim) rains the storm on
     the terrain on a worker thread while the viewer streams its depth field —
     the sky goes overcast, rain falls at the hyetograph's intensity, streets
     wet, puddles form where the water really collects;
  3. the architect edits the city with the existing tools (trees, areas,
     strips of bioswale / permeable paving / bike lane, drains, buildings) —
     every edit lands in a DesignModel;
  4. [Run with my design] repeats the same storm on the edited terrain and the
     lab reports the change (flooded area, depth, infiltration, outflow,
     hazard) and paints where it got better / worse.

State lives in `self.flood_lab` (None when the lab is unavailable: no
terrain package, or the scene does not overlap the flood-study domain).
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv

from floodsim import design as FD
from floodsim.metrics import FLOOD_THRESHOLD_M, summarize
from floodsim.model import DesignModel, DesignObject
from floodsim.runner import FloodRun
from floodsim.terrain import (OFFICIAL_MATERIAL, available_storms, load_inputs, load_storm,
                              terrain_available)
OFFICIAL_SPINE = OFFICIAL_MATERIAL.parent / "spine.json"

RES_MODES = (("Preview 2 m", 2.0), ("Fine 1 m", 1.0))
VIEW_MODES = ("Water", "Depth map", "Hazard", "Change")
# material cycle for the area / strip tools: (class, display colour)
MATERIAL_CYCLE = ((5, "#5aa84a"), (6, "#2f9a78"), (3, "#8fae4e"), (8, "#aab4bd"),
                  (4, "#d8c9a4"), (2, "#c8644a"), (7, "#a88c5c"))
STRIP_WIDTHS_M = (2.0, 3.0, 4.5, 6.0)
STORM_ORDER = ("v1_nov2025", "t2", "t10", "t50", "t10cc", "flat30")
HUD_X = 245           # HUD text x (px): clear of the left control panel
LAB_X = 1150          # panel x (px) — same coordinate system as the left control panel
Z_DEPTH = 0.20
DIFF_EPS_M = 0.02


_WIN = {"w": 1400, "h": 900}      # refreshed from the real window when the panel is built


def _lab_depth_cmap():
    """Depth legend for the lab layer: opaque from the first centimetres (thin street flows are
    1-3 px wide at overview scale and a ramp-in alpha made them vanish): blue -> indigo ->
    magenta -> red over 0 - 1 m (kept off cyan: the street-graph overlay is cyan)."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    base = LinearSegmentedColormap.from_list(
        "lab_depth", [(0.0, "#6f9bff"), (0.10, "#3358f0"), (0.30, "#5b2fd0"), (0.65, "#c0158f"), (1.0, "#ff3b30")])(np.linspace(0, 1, 256))
    base[:, 3] = 0.92
    base[0, 3] = 0.0
    base[1:4, 3] = 0.0                      # < ~1.5 cm: film, not flooding
    return ListedColormap(base, name="lab_depth")


def _panel_y(row: int) -> float:
    return (_WIN["h"] - 64.0) - 31.0 * row          # 836 at the app's 900 px window (matches the left panel)


class FloodLabMixin:

    # ------------------------------------------------------------------ init
    def _init_flood_lab(self) -> None:
        self.flood_lab = None
        if bool(getattr(self.args, "no_flood_lab", False)) or not terrain_available():
            return
        try:
            to_local = self._get_local_transformer()
        except Exception as exc:
            print(f"[flood-lab] unavailable (no local transformer: {exc})")
            return
        res_name = str(getattr(self.args, "flood_lab_res", "preview")).lower()
        res_idx = 1 if res_name.startswith("fine") else 0
        try:
            base = load_inputs(res=RES_MODES[res_idx][1])
            from floodsim.georef import SolverGeoref
            georef = SolverGeoref(base.meta["transform"], to_local)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[flood-lab] unavailable: {exc}")
            return
        sb = self.buildings_mesh.bounds if getattr(self, "buildings_mesh", None) is not None else None
        if sb is not None and not georef.overlaps(tuple(sb[:4])):
            print("[flood-lab] the scene does not overlap the Al-Masar flood-study domain — "
                  "use --address 'beirut corridor' (or --preset beirut-corridor) to enable it")
            return
        storms = [s for s in STORM_ORDER if s in available_storms()]
        first = str(getattr(self.args, "flood_storm", "v1_nov2025"))
        self.flood_lab = {
            "res_idx": res_idx, "base": {RES_MODES[res_idx][1]: base}, "georef": {RES_MODES[res_idx][1]: georef},
            "to_local": to_local, "storms": storms or ["v1_nov2025"],
            "storm_idx": max(0, (storms or [first]).index(first)) if first in storms else 0,
            "view_idx": 0, "material_idx": 0, "strip_idx": 1,
            "design": DesignModel(), "run": None, "run_tag": None, "runs": {}, "last_seq": -1,
            "layers": {}, "replay": {"playing": False, "t": 0.0, "speed": 120.0, "tag": None, "last": None},
            "buttons": {}, "status": "", "official_mat": None, "design_dirty": False,
            "storm_def": None, "overcast": 0.0, "env_t": 0.0,
        }
        try:                                     # JIT-warm the kernels now, not on the first click
            import copy
            from floodsim.engine import FloodEngine
            w = copy.copy(base)
            w.drains = None
            eng = FloodEngine(w)
            for _ in range(2):
                eng.step(1e-5, 10.0)
        except Exception as exc:
            print(f"[flood-lab] solver warm-up skipped: {exc}")
        self.scene_state["flood_lab_active"] = True
        print(f"[flood-lab] ready: {base.dem.shape[1]}x{base.dem.shape[0]} cells @ {base.res:.0f} m, "
              f"storms {self.flood_lab['storms']}")

    # --------------------------------------------------------------- helpers
    def _lab_res(self) -> float:
        return RES_MODES[self.flood_lab["res_idx"]][1]

    def _lab_inputs(self):
        """Base inputs + georef at the selected resolution (cached)."""
        lab = self.flood_lab
        res = self._lab_res()
        if res not in lab["base"]:
            from floodsim.georef import SolverGeoref
            lab["base"][res] = load_inputs(res=res)
            lab["georef"][res] = SolverGeoref(lab["base"][res].meta["transform"], lab["to_local"])
        return lab["base"][res], lab["georef"][res]

    def _lab_storm(self) -> dict:
        lab = self.flood_lab
        name = lab["storms"][lab["storm_idx"]]
        if lab["storm_def"] is None or lab["storm_def"].get("_key") != name:
            lab["storm_def"] = dict(load_storm(name), _key=name)
        return lab["storm_def"]

    def _lab_say(self, msg: str) -> None:
        self.flood_lab["status"] = msg
        self._lab_draw_text()

    # ----------------------------------------------------------------- panel
    def _build_flood_lab_panel(self) -> None:
        lab = self.flood_lab
        if lab is None or self.plotter is None:
            return
        TC = self._panel_text_color()
        try:
            _WIN["w"], _WIN["h"] = (int(v) for v in self.plotter.window_size)
        except Exception:
            pass
        global LAB_X
        LAB_X = _WIN["w"] - 292
        try:
            self._add_panel_backdrop(LAB_X - 8, _panel_y(20) - 8, LAB_X + 284, _panel_y(0) + 30)
        except Exception:
            pass

        def label(name, text, row, size=9, bold=False):
            self.plotter.add_text(text, position=(LAB_X + 30, _panel_y(row)), name=name, font_size=size,
                                  color=TC, viewport=False)

        label("lab_title", "FLOOD LAB", 0, 11)

        def button(key, text, row, cb, color="#4aa3e0"):
            label(f"lab_lbl_{key}", text or " ", row)
            holder = {}

            def _cb(v, cb=cb):
                try:
                    cb()
                finally:
                    try:                      # push-button: spring back
                        holder["w"].GetRepresentation().SetState(0)
                    except Exception:
                        pass
            w = self.plotter.add_checkbox_button_widget(_cb, value=False, position=(LAB_X, _panel_y(row)),
                                                        size=22, color_on=color, color_off="#3a3f4b")
            holder["w"] = w
            lab["buttons"][key] = (w, row)

        button("run_base", "Run flood: existing city", 1, lambda: self.flood_lab_run(design=False), "#3fb27f")
        button("run_design", "Run flood: with my design", 2, lambda: self.flood_lab_run(design=True), "#e0a84a")
        button("stop", "Stop / clear results", 3, self.flood_lab_clear_results, "#e05c5c")
        button("storm", "", 4, self._lab_cycle_storm)
        button("res", "", 5, self._lab_cycle_res)
        button("view", "", 6, self._lab_cycle_view)
        button("replay", "Replay last run", 7, self.flood_lab_replay, "#9b7fe0")
        button("fly", "Fly to worst flooding", 8, self.flood_lab_fly_hotspot, "#4aa3e0")
        button("export", "Export report (PNG + MD)", 9, self.flood_lab_export, "#c9a86a")
        label("lab_hdr_design", "DESIGN TOOLS", 11, 10)
        button("material", "", 12, self._lab_cycle_material, "#5aa84a")
        button("tool_area", "Tool: area (click, Enter)", 13, lambda: self._lab_tool("greenspace"), "#5aa84a")
        button("tool_strip", "Tool: strip (click, Enter)", 14, lambda: self._lab_tool("strip"), "#8fae4e")
        button("tool_tree", "Tool: street trees", 15, lambda: self._lab_tool("trees"), "#3d8a3a")
        button("tool_drain", "Tool: storm drain", 16, lambda: self._lab_tool("drains"), "#4aa3e0")
        button("tool_bldg", "Tool: building", 17, lambda: self._lab_tool("buildings"), "#b0a090")
        button("official", "Official Masar corridor", 18, self._lab_toggle_official, "#c9a86a")
        button("clear_design", "Clear my design", 19, self.flood_lab_clear_design, "#e05c5c")
        self._lab_refresh_labels()
        self._lab_draw_text()

    def _lab_refresh_labels(self) -> None:
        lab = self.flood_lab
        TC = self._panel_text_color()
        storm = self._lab_storm()
        mat = FD.LABELS[MATERIAL_CYCLE[lab["material_idx"]][0]]
        rows = {
            "storm": (4, f"Storm: {self._storm_title(storm)}"),
            "res": (5, f"Grid: {RES_MODES[lab['res_idx']][0]}"),
            "view": (6, f"View: {VIEW_MODES[lab['view_idx']]}"),
            "material": (12, f"Material: {mat[:17]}"),
            "official": (18, "Official Masar corridor: " + ("ON" if lab["design"].official else "off")),
        }
        for key, (row, text) in rows.items():
            self.plotter.add_text(text, position=(LAB_X + 30, _panel_y(row)), name=f"lab_lbl_{key}",
                                  font_size=9, color=TC, viewport=False)

    @staticmethod
    def _storm_title(storm: dict) -> str:
        nm = str(storm.get("name", storm.get("_key", "")))
        return f"{nm} ({storm.get('total_mm', 0):.0f} mm)"

    def _lab_cycle_storm(self) -> None:
        lab = self.flood_lab
        lab["storm_idx"] = (lab["storm_idx"] + 1) % len(lab["storms"])
        lab["storm_def"] = None
        lab["runs"].clear()                                  # results belong to the storm they ran
        lab["replay"]["playing"] = False
        ly = lab["layers"].get(self._lab_res())
        if ly is not None:
            self._lab_set_depth(np.zeros((ly["h"], ly["w"]), np.float32))
            for k in ("depth_actor", "diff_actor", "haz_actor"):
                ly[k].SetVisibility(False)
        self._lab_refresh_labels()
        self._lab_say(f"Storm: {self._storm_title(self._lab_storm())} — press Run")

    def _lab_cycle_res(self) -> None:
        lab = self.flood_lab
        if lab.get("run") is not None and lab["run"].running():
            return
        lab["res_idx"] = (lab["res_idx"] + 1) % len(RES_MODES)
        self._lab_clear_layers()
        lab["runs"].clear()
        self._lab_refresh_labels()
        self._lab_say(f"Grid: {RES_MODES[lab['res_idx']][0]} (previous results cleared)")

    def _lab_cycle_view(self) -> None:
        lab = self.flood_lab
        lab["view_idx"] = (lab["view_idx"] + 1) % len(VIEW_MODES)
        self._lab_refresh_labels()
        self._lab_show_current()
        self.plotter.render()

    def _lab_cycle_material(self) -> None:
        lab = self.flood_lab
        lab["material_idx"] = (lab["material_idx"] + 1) % len(MATERIAL_CYCLE)
        self._lab_refresh_labels()
        c = MATERIAL_CYCLE[lab["material_idx"]][0]
        i, n, d, _ = FD.MATERIALS[c]
        self._lab_say(f"Material: {FD.LABELS[c]} — infiltration {i:.0f} mm/h, Manning n {n:.3f}, detention {d * 100:.0f} cm")

    def _lab_tool(self, mode: str) -> None:
        self._set_editor_mode(mode)
        hints = {"greenspace": "click polygon corners, Enter to finish", "strip": "click along the strip, Enter to finish",
                 "trees": "click to plant street trees", "drains": "click to place storm drains",
                 "buildings": "click to place a building"}
        self._lab_say(f"Tool: {mode} — {hints.get(mode, '')}  (key 1 = back to view)")

    def _lab_corridor_xy(self):
        """Official corridor centreline in the local frame (n, 2), or None."""
        lab = self.flood_lab
        if "corridor_xy" not in lab:
            lab["corridor_xy"] = None
            try:
                import json
                sp = json.loads(OFFICIAL_SPINE.read_text())
                pts = np.asarray(sp["spine"], dtype=float)
                _, georef = self._lab_inputs()
                lab["corridor_xy"] = georef.utm_to_local(pts[:, 0], pts[:, 1])
            except Exception as exc:
                print(f"[flood-lab] corridor spine unavailable: {exc}")
        return lab["corridor_xy"]

    def flood_lab_fly_corridor(self) -> None:
        """Camera to the official Al-Masar corridor (centre of its centreline), looking down it."""
        xy = self._lab_corridor_xy()
        if xy is None or len(xy) < 2:
            return
        mid = xy[len(xy) // 2]
        a, b = xy[max(len(xy) // 2 - 20, 0)], xy[min(len(xy) // 2 + 20, len(xy) - 1)]
        along = np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1]))
        extent = float(np.ptp(xy, axis=0).max())
        dem = self.street_graph.graph.get("terrain_sampler")
        z = float(np.asarray(dem(np.array([mid])))[0]) if (dem is not None and self.scene_state.get("_terrain_drape_active")) else 0.0
        cam = self.plotter.camera
        cam.ParallelProjectionOff()
        d = max(260.0, 0.55 * extent)
        el, az = np.radians(52.0), np.radians(along + 90.0)          # look across the corridor
        pos = (mid[0] + d * np.cos(el) * np.sin(az), mid[1] - d * np.cos(el) * np.cos(az), z + d * np.sin(el))
        self.plotter.camera_position = [pos, (mid[0], mid[1], z), (0, 0, 1)]
        cam.view_angle = 40
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _lab_toggle_official(self) -> None:
        lab = self.flood_lab
        d = lab["design"]
        if not d.official and not OFFICIAL_MATERIAL.exists():
            self._lab_say("Official corridor data not found (Beirut_Project-main/output/corridor_gi_cut)")
            return
        d.official = not d.official
        d.version += 1
        if d.official and lab["official_mat"] is None:
            lab["official_mat"] = np.load(OFFICIAL_MATERIAL)
        self._lab_official_decal(d.official)
        n_trees = self._lab_official_trees(d.official)
        if d.official:
            self.flood_lab_fly_corridor()
        self._lab_refresh_labels()
        self._lab_say("Official Masar corridor design " + (f"loaded ({n_trees} trees) — run flood with design" if d.official else "removed"))

    # ------------------------------------------------------- design registry
    def _design_add(self, obj: DesignObject) -> None:
        lab = getattr(self, "flood_lab", None)
        if lab is None:
            return
        lab["design"].add(obj)
        self._lab_say(f"Design: {self._design_summary_text()}")

    def _design_sync_after_undo(self) -> None:
        """Editor undo removes actors by name; drop design objects whose actor is gone."""
        lab = getattr(self, "flood_lab", None)
        if lab is None:
            return
        alive = {o.key for o in lab["design"].objects if o.key in self.scene_state}
        if lab["design"].remove_missing(alive):
            self._lab_say(f"Design: {self._design_summary_text()}")

    def _design_summary_text(self) -> str:
        s = self.flood_lab["design"].summary()
        return ", ".join(f"{v} {k}" for k, v in s.items()) if s else "empty"

    def flood_lab_clear_design(self) -> None:
        lab = self.flood_lab
        kept = 0
        for o in list(lab["design"].objects):
            if o.kind == "building":
                kept += 1                                    # merged into the city mesh: undo with 'o'
                continue
            for key in (o.key, *o.aux):
                self.scene_state.pop(key, None)
                try:
                    self.plotter.remove_actor(key, reset_camera=False)
                except Exception:
                    pass
                if key.startswith("_tree_"):                 # also drop the drape registration (no ghost trees)
                    for dct in ("_tree_seeds_flat", "_tree_templates", "_tree_actor_kwargs"):
                        (self.scene_state.get(dct) or {}).pop(key[len("_tree_"):], None)
        lab["design"].objects = [o for o in lab["design"].objects if o.kind == "building"]
        lab["design"].official = False
        lab["design"].version += 1
        self._lab_official_decal(False)
        self._lab_official_trees(False)
        self._lab_refresh_labels()
        self._lab_say("Design cleared" + (f" ({kept} placed building(s) stay in the city — press 'o' to undo them)" if kept else ""))

    # --------------------------------------------------------------- layers
    def _lab_layers(self) -> dict:
        """(Re)build the render layers on the current solver grid."""
        lab = self.flood_lab
        res = self._lab_res()
        if res in lab["layers"]:
            return lab["layers"][res]
        base, georef = self._lab_inputs()
        h, w = base.dem.shape
        xy = georef.xy
        pts = np.column_stack([xy, np.full(len(xy), Z_DEPTH)])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = (w, h, 1)
        grid.point_data["depth"] = np.zeros(h * w, np.float32)
        grid.point_data["diff"] = np.zeros(h * w, np.float32)
        grid.point_data["haz"] = np.zeros(h * w, np.float32)
        depth_actor = self.plotter.add_mesh(grid, scalars="depth", cmap=_lab_depth_cmap(), clim=[0.0, 1.0],
                                            show_scalar_bar=False, lighting=False, reset_camera=False,
                                            name=f"lab_depth_{int(res * 10)}")
        depth_actor.PickableOff()
        depth_actor.SetVisibility(False)
        layers = {"grid": grid, "depth_actor": depth_actor, "w": w, "h": h, "water": None, "diff_actor": None, "haz_actor": None}
        # physical water (photoreal view) when PBR materials exist
        if getattr(self, "materials", None) is not None and not bool(getattr(self.args, "no_physical_sky", False)):
            try:
                from render.water import FloodWaterSurface
                sw = FloodWaterSurface(self.plotter, w, h, 0, 0, 1, 1, Z_DEPTH, np.zeros(h * w, np.float32),
                                       name=f"lab_water_{int(res * 10)}", xy=xy)
                sw.set_visible(False)
                layers["water"] = sw
            except Exception as exc:
                print(f"[flood-lab] physical water unavailable: {exc}")
        # change map
        import matplotlib
        from matplotlib.colors import ListedColormap
        rd = matplotlib.colormaps["RdBu"](np.linspace(0, 1, 256))     # red = deeper, blue = shallower (reversed below)
        cm = rd[::-1].copy()
        a = np.abs(np.linspace(-1, 1, 256))
        cm[:, 3] = np.clip((a - 0.03) / 0.25, 0, 1) * 0.85
        diff_actor = self.plotter.add_mesh(grid, scalars="diff", cmap=ListedColormap(cm), clim=[-0.25, 0.25],
                                           show_scalar_bar=False, lighting=False, reset_camera=False,
                                           name=f"lab_diff_{int(res * 10)}")
        diff_actor.PickableOff()
        diff_actor.SetVisibility(False)
        layers["diff_actor"] = diff_actor
        hz = matplotlib.colormaps["YlOrRd"](np.linspace(0, 1, 256))
        hz[:, 3] = np.clip(np.linspace(-0.05, 1, 256) * 1.4, 0, 0.9)
        haz_actor = self.plotter.add_mesh(grid, scalars="haz", cmap=ListedColormap(hz), clim=[0.0, 1.5],
                                          show_scalar_bar=False, lighting=False, reset_camera=False,
                                          name=f"lab_haz_{int(res * 10)}")
        haz_actor.PickableOff()
        haz_actor.SetVisibility(False)
        layers["haz_actor"] = haz_actor
        # One grid, three layers: VTK colours by the dataset's ACTIVE scalars unless the
        # mapper is told which array to use, so each mapper selects its own.
        for k, arr in (("depth_actor", "depth"), ("diff_actor", "diff"), ("haz_actor", "haz")):
            m = layers[k].GetMapper()
            m.SetScalarModeToUsePointFieldData()
            m.SelectColorArray(arr)
            m.SetScalarVisibility(True)
            m.ColorByArrayComponent(arr, 0)
        from vtk import vtkMapper
        for k in ("depth_actor", "diff_actor", "haz_actor"):
            m = layers[k].GetMapper()
            m.SetResolveCoincidentTopologyToPolygonOffset()
            m.SetRelativeCoincidentTopologyPolygonOffsetParameters(-3.0, -3.0)
            self.scene_state[f"lab_{k}"] = layers[k]
        lab["layers"][res] = layers
        if self.scene_state.get("_terrain_drape_active"):
            self._lab_drape(True)
        return layers

    def _lab_clear_layers(self) -> None:
        lab = self.flood_lab
        for res, ly in list(lab["layers"].items()):
            for k in ("depth_actor", "diff_actor", "haz_actor"):
                try:
                    self.plotter.remove_actor(ly[k], reset_camera=False)
                except Exception:
                    pass
                self.scene_state.pop(f"lab_{k}", None)
            if ly.get("water") is not None:
                try:
                    self.plotter.remove_actor(ly["water"].actor, reset_camera=False)
                except Exception:
                    pass
        lab["layers"].clear()
        for k in ("depth_actor", "diff_actor", "haz_actor"):
            self.scene_state.pop(f"_orig_lab_{k}_z", None)

    def _lab_drape(self, on: bool) -> None:
        """Terrain drape / restore for the lab layers (called by terrain_mixin)."""
        lab = getattr(self, "flood_lab", None)
        if lab is None:
            return
        dem = self.street_graph.graph.get("terrain_sampler")
        for res, ly in lab["layers"].items():
            for k in ("depth_actor", "diff_actor", "haz_actor"):
                key = f"lab_{k}"
                if on and dem is not None:
                    # all three share one grid: lift the points once, via the first actor
                    if k == "depth_actor":
                        self._drape_over_ground(key, dem)
                else:
                    if k == "depth_actor":
                        self._restore_actor_points(key)
            if ly.get("water") is not None:
                if on and dem is not None:
                    base = np.asarray(dem(ly["water"].xy), dtype=float) + 0.3
                    photo = self.scene_state.get("survey_ground_actor")
                    if photo is not None:
                        from render.survey_mixin import _PHOTO_Z, _surface_height_at
                        top = _surface_height_at(photo.GetMapper().GetInputDataObject(0, 0), ly["water"].xy)
                        if top is not None:
                            ok = np.isfinite(top)
                            base[ok] = np.maximum(base[ok], top[ok] - _PHOTO_Z)
                    ly["water"].set_base(base)
                else:
                    ly["water"].set_base(None)

    # ------------------------------------------------------ showing results
    def _lab_set_depth(self, depth: np.ndarray, extra: dict | None = None) -> None:
        ly = self._lab_layers()
        ly["grid"].point_data["depth"][:] = np.asarray(depth, np.float32).ravel()
        ly["grid"].GetPointData().GetArray("depth").Modified()
        if ly["water"] is not None:
            ly["water"].set_depth(depth)
        ly["grid"].Modified()

    def _lab_apply_view(self) -> None:
        """Visibility of the layers for the current view mode."""
        lab = self.flood_lab
        ly = lab["layers"].get(self._lab_res())
        if ly is None:
            return
        view = VIEW_MODES[lab["view_idx"]]
        photoreal = bool(self.scene_state.get("photoreal_ground", False)) and ly["water"] is not None
        show_water = (view == "Water") and photoreal
        far = self._lab_camera_far()
        lab["far_state"] = far
        # Overview scale: 2 m street flows are 1-3 px wide and the photoreal water reads as dark
        # asphalt, so the legend-coloured depth map is blended in; close up, only the water shows.
        ly["depth_actor"].SetVisibility((view == "Depth map") or (view == "Water" and (not photoreal or far)))
        if ly["water"] is not None:
            ly["water"].set_visible(show_water)
        ly["haz_actor"].SetVisibility(view == "Hazard")
        ly["diff_actor"].SetVisibility(view == "Change")

    def _lab_camera_far(self) -> bool:
        cam = self.plotter.camera
        try:
            if cam.GetParallelProjection():
                return float(cam.GetParallelScale()) > 220.0
            return float(cam.GetDistance()) > 330.0
        except Exception:
            return False

    def _lab_show_current(self) -> None:
        """Paint the view for the active record (or the last one)."""
        lab = self.flood_lab
        rec = self._lab_best_record()
        if rec is None:
            return
        ly = self._lab_layers()
        res = rec["result"]
        view = VIEW_MODES[lab["view_idx"]]
        if view in ("Water", "Depth map"):
            self._lab_set_depth(res.max_depth if rec.get("showing_max", True) else res.final_depth)
        elif view == "Hazard":
            ly["grid"].point_data["haz"][:] = res.max_hazard.ravel()
            ly["grid"].GetPointData().GetArray("haz").Modified()
        elif view == "Change":
            base, dsg = lab["runs"].get("baseline"), lab["runs"].get("design")
            if base is not None and dsg is not None:
                diff = (dsg["result"].max_depth - base["result"].max_depth).astype(np.float32)
                ly["grid"].point_data["diff"][:] = diff.ravel()
                ly["grid"].GetPointData().GetArray("diff").Modified()
            else:
                self._lab_say("Change view needs both runs: existing city and with design")
        ly["grid"].Modified()
        self._lab_apply_view()

    def _lab_best_record(self):
        lab = self.flood_lab
        return lab["runs"].get("design") or lab["runs"].get("baseline")

    # ---------------------------------------------------------------- running
    def flood_lab_run(self, design: bool = False, block: bool = False, on_tick=None) -> bool:
        lab = getattr(self, "flood_lab", None)
        if lab is None:
            print("[flood-lab] not available in this scene")
            return False
        if lab.get("run") is not None and lab["run"].running():
            self._lab_say("A run is already in progress")
            return False
        base, georef = self._lab_inputs()
        storm = self._lab_storm()
        if design:
            if lab["design"].is_empty():
                self._lab_say("No design yet — place trees/areas/strips/drains or load the official corridor")
                return False
            if lab["design"].official and lab["official_mat"] is None and OFFICIAL_MATERIAL.exists():
                lab["official_mat"] = np.load(OFFICIAL_MATERIAL)
            inp = lab["design"].to_inputs(base, georef, lab["official_mat"])
        else:
            import copy
            inp = copy.copy(base)
            inp.drains = None                                 # study baseline: blocked inlets
        tag = "design" if design else "baseline"
        save_every = 60.0 if self._lab_res() >= 2.0 else 120.0
        run = FloodRun(inp, storm, label=tag, save_every=save_every, chunk_s=15.0 if self._lab_res() >= 2.0 else 30.0)
        lab.update(run=run, run_tag=tag, last_seq=-1, run_inp=inp, run_t0=time.time())
        lab["replay"]["playing"] = False
        # stale visuals of the other record stay hidden while a new run streams in
        self._lab_set_depth(np.zeros_like(inp.dem))
        ly = self._lab_layers()
        lab["view_idx_before_run"] = lab["view_idx"]
        if VIEW_MODES[lab["view_idx"]] in ("Hazard", "Change"):
            lab["view_idx"] = 0
            self._lab_refresh_labels()
        self._lab_apply_view()
        self._lab_begin_environment()
        self._lab_frame_domain()
        run.start()
        self._lab_say(f"Running {tag} flood simulation…")
        if block:
            while not run.done.is_set():
                self._flood_lab_tick()
                if on_tick is not None:
                    on_tick(run.snapshot())
                time.sleep(0.03)
            self._flood_lab_tick()
        return True

    def flood_lab_clear_results(self) -> None:
        lab = self.flood_lab
        run = lab.get("run")
        if run is not None and run.running():
            run.cancel()
            run.join(5.0)
        lab["run"] = None
        lab["runs"].clear()
        lab["replay"]["playing"] = False
        lab["status"] = "Results cleared"
        ly = lab["layers"].get(self._lab_res())
        if ly is not None:
            self._lab_set_depth(np.zeros((ly["h"], ly["w"]), np.float32))
            for k in ("depth_actor", "diff_actor", "haz_actor"):
                ly[k].SetVisibility(False)
        self._lab_end_environment()
        self._lab_draw_text()

    def _lab_frame_domain(self, force: bool = False) -> bool:
        """If the camera is the app's default whole-city overview, move in on the flood-study
        domain so the water is legible (parallel scale 1300 m shows ~1/20 of the screen to it).
        Keeps the view direction; never touches a camera the user has already set closer."""
        cam = self.plotter.camera
        try:
            parallel = bool(cam.GetParallelProjection())
            scale = float(cam.GetParallelScale())
        except Exception:
            return False
        if not parallel or (scale < 900.0 and not force):
            return False
        _, georef = self._lab_inputs()
        x0, x1, y0, y1 = georef.bounds()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dem = self.street_graph.graph.get("terrain_sampler")
        cz = float(np.asarray(dem(np.array([[cx, cy]])))[0]) if (dem is not None and self.scene_state.get("_terrain_drape_active")) else 0.0
        pos = np.asarray(cam.position, dtype=float)
        fp = np.asarray(cam.focal_point, dtype=float)
        direction = (pos - fp)
        direction /= max(np.linalg.norm(direction), 1e-9)
        # Streets are the point and sit between 20-40 m towers: look down at ~68 deg elevation
        # (the 45 deg isometric default hides most of the ground behind the buildings),
        # keeping the user's compass direction.
        az = np.arctan2(direction[1], direction[0]) if np.hypot(direction[0], direction[1]) > 1e-6 else 0.8
        el = np.radians(68.0)
        if np.degrees(np.arcsin(np.clip(direction[2], -1, 1))) < 68.0:
            direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        new_fp = np.array([cx, cy, cz])
        cam.focal_point = tuple(new_fp)
        cam.position = tuple(new_fp + direction * float(np.linalg.norm(pos - fp)))
        cam.SetParallelScale(max((y1 - y0) * 0.36, (x1 - x0) * 0.62))
        self.plotter.renderer.ResetCameraClippingRange()
        return True

    # ------------------------------------------------------------ environment
    def _lab_begin_environment(self) -> None:
        """Storm look: remember user weather, start the overcast ramp."""
        lab = self.flood_lab
        lab["env_t"] = 0.0

    def _lab_end_environment(self) -> None:
        lab = self.flood_lab
        lab["overcast"] = 0.0
        try:
            self._release_rain_from_storm()
        except Exception:
            pass
        self._lab_apply_overcast(0.0, force=True)

    def _lab_drive_environment(self, sim_t: float, running: bool) -> str:
        """Rain streaks, wet streets and overcast sky from the hyetograph at sim time."""
        lab = self.flood_lab
        suffix = ""
        try:
            suffix = self._drive_rain_from_storm(sim_t, lab["storms"][lab["storm_idx"]])
        except Exception as exc:
            print(f"[flood-lab] rain drive failed: {exc}")
        storm = self._lab_storm()
        steps = storm["steps"]
        raining = any(s[0] <= sim_t < s[1] and s[2] > 0 for s in steps)
        target = 1.0 if (raining or sim_t < steps[-1][1] + 120.0) else 0.0
        cur = lab["overcast"]
        cur += float(np.clip(target - cur, -0.02, 0.02))
        self._lab_apply_overcast(cur)
        return suffix

    def _lab_apply_overcast(self, c: float, force: bool = False) -> None:
        lab = self.flood_lab
        c = float(np.clip(c, 0.0, 1.0))
        applied = lab.get("overcast_applied", 0.0)
        settled = abs(c - applied) < 0.1 and not (c in (0.0, 1.0) and c != applied)
        if not force and settled:
            lab["overcast"] = c
            return
        lab["overcast"] = c
        lab["overcast_applied"] = c
        self.scene_state["overcast"] = c
        try:
            hour = float(self.scene_state.get("hour", 12.0))
            from app_core import _sun_dir_from_hour
            sun_dir = _sun_dir_from_hour(hour)
            style = self._style()
            self._apply_visual_updates(hour, float(self.scene_state.get("spot_radius", 40.0)),
                                       bool(self.scene_state.get("is_night", False)), sun_dir, style)
        except Exception as exc:
            print(f"[flood-lab] overcast update failed: {exc}")

    # ------------------------------------------------------------------- tick
    def _flood_lab_tick(self) -> None:
        lab = getattr(self, "flood_lab", None)
        if lab is None:
            return
        if lab["layers"] and VIEW_MODES[lab["view_idx"]] == "Water" and self._lab_camera_far() != lab.get("far_state"):
            self._lab_apply_view()                       # zoomed in/out: swap depth-map blend <-> water only
        run = lab.get("run")
        if run is not None:
            snap = run.snapshot()
            if snap["seq"] != lab["last_seq"]:
                lab["last_seq"] = snap["seq"]
                self._lab_set_depth(snap["depth"])
            storm = self._lab_storm()
            suffix = self._lab_drive_environment(snap["t"], not snap["finished"])
            frac = snap["fraction"]
            m, sec = int(snap["t"] // 60), int(snap["t"] % 60)
            lab["status"] = (f"{'EXISTING CITY' if lab['run_tag'] == 'baseline' else 'WITH DESIGN'}  {m:02d}:{sec:02d} / {int(storm['duration'] // 60)} min"
                             f"  |  stored {snap['stored']:,.0f} m3  |  max depth {snap['hmax']:.1f} m{suffix}")
            if time.perf_counter() - lab.get("hud_t", 0.0) > 0.25 or run.done.is_set():
                lab["hud_t"] = time.perf_counter()
                self._lab_draw_text()
            if run.done.is_set():
                self._lab_finish_run()
            return
        rp = lab["replay"]
        if rp["playing"]:
            self._lab_replay_tick()

    def _lab_finish_run(self) -> None:
        lab = self.flood_lab
        run, tag = lab["run"], lab["run_tag"]
        lab["run"] = None
        if run.error:
            self._lab_say(f"Run failed: {run.error}")
            self._lab_end_environment()
            return
        res = run.result
        inp = lab["run_inp"]
        summ = summarize(res, inp)
        try:
            summ.update(self._lab_building_exposure(res, inp))
        except Exception as exc:
            print(f"[flood-lab] building exposure skipped: {exc}")
        rec = {"result": res, "inp": inp, "summary": summ, "storm": run.storm, "res": inp.res,
               "storm_key": lab["storms"][lab["storm_idx"]], "design_version": lab["design"].version,
               "design": None if tag == "baseline" else lab["design"].summary(), "showing_max": True}
        for other in [k for k, r in lab["runs"].items() if k != tag and r.get("storm_key") != rec["storm_key"]]:
            del lab["runs"][other]                           # never compare different storms
        lab["runs"][tag] = rec
        if tag == "design":
            lab["design_version_run"] = lab["design"].version
        lab["view_idx"] = 0
        self._lab_refresh_labels()
        self._lab_show_current()
        self._lab_end_environment()
        lab["replay"].update(tag=tag, t=0.0, playing=False)
        self._lab_draw_text(final=True)
        print(f"[flood-lab] {tag} finished in {summ['wall_s']:.0f}s: flooded {summ['flooded_area_ha']:.2f} ha, "
              f"max {summ['max_depth_m']:.2f} m, closure {summ['closure_rel']:.1e}")
        try:
            self.plotter.render()
        except Exception:
            pass

    def _lab_building_rings(self) -> list:
        """Per building: (row slice, col slice, mask) of the solver cells in a 3 m ring around its
        footprint (the facade zone), cached per solver grid + building set."""
        lab = self.flood_lab
        bm = getattr(self, "buildings_mesh", None)
        base, georef = self._lab_inputs()
        key = (id(bm), bm.n_cells if bm is not None else 0, base.res)
        if lab.get("ring_key") == key:
            return lab["rings"]
        from render.building_reconstruct import building_footprints
        rings = []
        for fp in building_footprints(bm):
            if fp is None or fp.is_empty or fp.area < 8.0:
                continue
            band = fp.buffer(3.0).difference(fp)
            if band.is_empty:
                continue
            polys = list(band.geoms) if hasattr(band, "geoms") else [band]
            acc = None
            for poly in polys:
                if poly.geom_type != "Polygon" or poly.is_empty:
                    continue
                res = georef.polygon_fraction(np.asarray(poly.exterior.coords)[:-1], sub=2)
                if res is None:
                    continue
                (rs, cs), blk = res
                if acc is None:
                    acc = [rs, cs, blk > 0.15]
                else:                                        # merge parts of a multipart ring
                    r0, r1 = min(acc[0].start, rs.start), max(acc[0].stop, rs.stop)
                    c0, c1 = min(acc[1].start, cs.start), max(acc[1].stop, cs.stop)
                    m = np.zeros((r1 - r0, c1 - c0), bool)
                    m[acc[0].start - r0:acc[0].stop - r0, acc[1].start - c0:acc[1].stop - c0] |= acc[2]
                    m[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] |= blk > 0.15
                    acc = [slice(r0, r1), slice(c0, c1), m]
            if acc is not None and acc[2].any():
                rings.append(tuple(acc))
        lab["rings"], lab["ring_key"] = rings, key
        return rings

    def _lab_building_exposure(self, res, inp) -> dict:
        """Buildings with flood water at the facade. A building counts as affected when at least
        a fifth of the open cells in its 3 m facade ring are under water deeper than the
        threshold (any single wet cell would flag ~70 % of a dense district: 10 % of the open
        ground floods in this storm)."""
        rings = self._lab_building_rings()
        if not rings:
            return {"buildings_total": 0, "buildings_flooded_10cm": 0, "buildings_flooded_30cm": 0}
        h, w = res.max_depth.shape
        open_mask = ~(inp.building if inp.building is not None else np.zeros((h, w), bool))
        md = res.max_depth
        f10 = np.zeros(len(rings))
        f30 = np.zeros(len(rings))
        for k, (rs, cs, m) in enumerate(rings):
            sel = m & open_mask[rs, cs]
            if sel.any():
                d = md[rs, cs][sel]
                f10[k], f30[k] = (d > 0.10).mean(), (d > 0.30).mean()
        return {"buildings_total": len(rings), "buildings_flooded_10cm": int((f10 >= 0.20).sum()),
                "buildings_flooded_30cm": int((f30 >= 0.20).sum())}

    # ----------------------------------------------------------------- replay
    def flood_lab_replay(self) -> None:
        lab = self.flood_lab
        rec = self._lab_best_record()
        if rec is None:
            self._lab_say("Nothing to replay yet — run a flood first")
            return
        rp = lab["replay"]
        rp.update(playing=True, t=0.0, tag="design" if lab["runs"].get("design") else "baseline", last=None)
        rec["showing_max"] = False
        lab["view_idx"] = 0
        self._lab_refresh_labels()
        self._lab_apply_view()
        self._lab_begin_environment()

    def _lab_replay_tick(self) -> None:
        lab = self.flood_lab
        rp = lab["replay"]
        rec = lab["runs"].get(rp["tag"])
        if rec is None:
            rp["playing"] = False
            return
        frames = rec["result"].frames
        storm = rec["storm"]
        now = time.perf_counter()
        last = rp["last"]
        rp["last"] = now
        if last is None:
            return
        rp["t"] += min(now - last, 0.2) * rp["speed"]
        dur = float(storm["duration"])
        if rp["t"] >= dur:
            rp["playing"] = False
            rec["showing_max"] = True
            self._lab_show_current()
            self._lab_end_environment()
            self._lab_draw_text(final=True)
            return
        ts = np.array([f[0] for f in frames])
        i = int(np.clip(np.searchsorted(ts, rp["t"], side="right") - 1, 0, len(frames) - 2))
        a = float(np.clip((rp["t"] - ts[i]) / max(ts[i + 1] - ts[i], 1e-9), 0.0, 1.0))
        depth = frames[i][1] * (1 - a) + frames[i + 1][1] * a
        self._lab_set_depth(depth)
        suffix = self._lab_drive_environment(rp["t"], True)
        lab["status"] = (f"REPLAY {'existing city' if rp['tag'] == 'baseline' else 'with design'}  "
                         f"{int(rp['t'] // 60):02d}:{int(rp['t'] % 60):02d} / {int(dur // 60)} min{suffix}")
        self._lab_draw_text()

    # --------------------------------------------------------------------- HUD
    def _lab_hud_actor(self, k: int):
        lab = self.flood_lab
        actors = lab.setdefault("hud_actors", {})
        if k not in actors:
            a = self.plotter.add_text(" ", position=(HUD_X, _WIN["h"] - 28 - 26 * k), name=f"lab_hud_{k}",
                                      font_size=10 if k == 0 else 9, color="#ffffff", viewport=False, shadow=True)
            actors[k] = a
        return actors[k]

    def _lab_draw_text(self, final: bool = False) -> None:
        lab = getattr(self, "flood_lab", None)
        if lab is None or self.plotter is None:
            return
        lines = []
        if lab["status"]:
            lines.append(lab["status"])
        runs = lab["runs"]
        if runs:
            def fmt(tag, rec):
                s = rec["summary"]
                return (f"{tag:<9} {s['flooded_area_ha']:5.2f} ha flooded | {s['flooded_area_deep_ha']:4.2f} ha > 30 cm | "
                        f"max {s['max_depth_m']:.2f} m | infiltrated {s['infiltrated_pct']:4.1f} % | "
                        f"{s.get('buildings_flooded_10cm', 0)} buildings flooded")
            for tag in ("baseline", "design"):
                if tag in runs:
                    lines.append(fmt("existing" if tag == "baseline" else "design", runs[tag]))
            if "baseline" in runs and "design" in runs:
                b, d = runs["baseline"]["summary"], runs["design"]["summary"]
                dfa = 100 * (d["flooded_area_ha"] - b["flooded_area_ha"]) / max(b["flooded_area_ha"], 1e-9)
                dmax = d["max_depth_m"] - b["max_depth_m"]
                dh = 100 * (d["hazardous_area_ha"] - b["hazardous_area_ha"]) / max(b["hazardous_area_ha"], 1e-9)
                stale = runs["design"].get("design_version") != lab["design"].version
                lines.append(("CHANGE*   " if stale else "CHANGE    ") + f"flooded area {dfa:+.1f} %  |  max depth {dmax * 100:+.0f} cm  |  "
                             f"hazard area {dh:+.1f} %  |  infiltration {d['infiltrated_pct'] - b['infiltrated_pct']:+.1f} pts  |  "
                             f"buildings affected {b.get('buildings_flooded_10cm', 0)} -> {d.get('buildings_flooded_10cm', 0)}")
                roi = self._lab_roi_change()
                if roi is not None:
                    lines.append(f"          within 60 m of the design: flooded area {roi:+.0f} %")
                if stale:
                    lines.append("          * the design was edited after this run — run flood with design again")
        try:                                         # keep each line clear of the right-hand panel
            wpx = int(self.plotter.window_size[0])
        except Exception:
            wpx = 1400
        max_chars = max(40, int((wpx - HUD_X - 320) / 8.2))
        lines = [ln if len(ln) <= max_chars else ln[:max_chars - 1].rstrip() + "…" for ln in lines]
        key = tuple(lines)
        if key == lab.get("hud_key"):
            return
        lab["hud_key"] = key
        try:
            _WIN["w"], _WIN["h"] = (int(v) for v in self.plotter.window_size)
        except Exception:
            pass
        for k in range(6):
            a = self._lab_hud_actor(k)
            a.SetPosition(HUD_X, _WIN["h"] - 28 - 26 * k)
            a.SetInput(lines[k] if k < len(lines) else " ")

    def _lab_roi_change(self) -> float | None:
        """Flooded-area change (%) within 60 m of any design cover."""
        lab = self.flood_lab
        b, d = lab["runs"].get("baseline"), lab["runs"].get("design")
        if b is None or d is None:
            return None
        if "roi" in d and d.get("roi_vs") == id(b):
            return d["roi"]
        d["roi"], d["roi_vs"] = self._lab_roi_compute(b, d), id(b)
        return d["roi"]

    def _lab_roi_compute(self, b, d) -> float | None:
        base, georef = self._lab_inputs()
        h, w = base.dem.shape
        cover = np.zeros((h, w), bool)
        try:
            inp_d = d["inp"]
            diff = np.abs(inp_d.infil_mmh - b["inp"].infil_mmh) + np.abs(inp_d.dem - b["inp"].dem) * 100
            cover = diff > 1e-6
        except Exception:
            return None
        if not cover.any():
            return None
        from scipy import ndimage as ndi
        near = ndi.distance_transform_edt(~cover) * base.res <= 60.0
        og = base.valid & ~base.water & (~base.building if base.building is not None else True) & near
        fb = (b["result"].max_depth > FLOOD_THRESHOLD_M) & og
        fd = (d["result"].max_depth > FLOOD_THRESHOLD_M) & og
        return 100.0 * (fd.sum() - fb.sum()) / max(int(fb.sum()), 1)

    # ------------------------------------------------------------- export
    def flood_lab_export(self, out_dir=None) -> str | None:
        """Write the study to disk: summary.json, depth rasters (.npy + georeference),
        a before / after / change figure (PNG) and a one-page report (Markdown)."""
        import json
        from datetime import datetime
        from pathlib import Path
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.figure import Figure

        lab = self.flood_lab
        runs = lab["runs"]
        if not runs:
            self._lab_say("Nothing to export — run a flood first")
            return None
        base, georef = self._lab_inputs()
        root = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "reports"
        folder = root / datetime.now().strftime("flood_%Y%m%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        storm = self._lab_storm()
        summary = {"storm": storm.get("name"), "storm_total_mm": storm.get("total_mm"), "grid_m": base.res,
                   "runs": {}, "design": lab["design"].summary()}
        for tag, rec in runs.items():
            summary["runs"][tag] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                    for k, v in rec["summary"].items()}
            np.save(folder / f"max_depth_{tag}.npy", rec["result"].max_depth)
            np.save(folder / f"max_hazard_{tag}.npy", rec["result"].max_hazard)
        summary["georeference"] = base.meta["transform"]
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))

        h, w = base.dem.shape
        res = base.res
        extent = [0, w * res, 0, h * res]
        og = base.valid & ~base.water
        bg = np.where(base.building if base.building is not None else False, 0.25, np.where(og, 0.85, 1.0))
        tags = [t for t in ("baseline", "design") if t in runs]
        ncol = len(tags) + (1 if len(tags) == 2 else 0)
        fig = Figure(figsize=(5.2 * ncol, 9.2))
        FigureCanvasAgg(fig)                                 # private canvas: no global backend change
        axes = fig.subplots(1, ncol, squeeze=False)
        for ax, tag in zip(axes[0], tags):
            md = np.where(og, runs[tag]["result"].max_depth, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=extent)
            im = ax.imshow(np.ma.masked_less(md, 0.03), cmap="turbo", vmin=0, vmax=1.0, extent=extent, alpha=0.9)
            s_ = runs[tag]["summary"]
            ax.set_title(f"{'Existing city' if tag == 'baseline' else 'With design'}\n"
                         f"{s_['flooded_area_ha']:.2f} ha flooded, max {s_['max_depth_m']:.1f} m", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        if len(tags) == 2:
            ax = axes[0][-1]
            diff = np.where(og, runs["design"]["result"].max_depth - runs["baseline"]["result"].max_depth, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, extent=extent)
            imd = ax.imshow(np.ma.masked_where(np.abs(diff) < 0.02, diff), cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0, vmin=-0.4, vmax=0.4), extent=extent)
            ax.set_title("Change in maximum depth\n(blue = shallower, red = deeper)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(imd, ax=ax, fraction=0.04, label="m")
        fig.colorbar(im, ax=axes[0][:len(tags)].tolist(), fraction=0.02, label="maximum flood depth (m)")
        fig.suptitle(f"Pluvial flood study — {storm.get('name')} ({storm.get('total_mm', 0):.0f} mm), {res:.0f} m grid", fontsize=12)
        fig.savefig(folder / "flood_maps.png", dpi=150, bbox_inches="tight")

        lines = [f"# Flood study report", "", f"Storm: **{storm.get('name')}** ({storm.get('total_mm', 0):.1f} mm). Solver: local-inertial shallow-water, {res:.0f} m grid.", ""]
        keys = [("flooded_area_ha", "Flooded area (> 10 cm), ha"), ("flooded_area_deep_ha", "Deep flooding (> 30 cm), ha"),
                ("max_depth_m", "Maximum depth, m"), ("infiltrated_pct", "Rain infiltrated, %"),
                ("volume_outflow_m3", "Outflow to port/sea, m3"), ("hazardous_area_ha", "Hazardous area (h(v+0.5) > 0.75), ha"),
                ("buildings_flooded_10cm", "Buildings with >= 20 % of the facade zone under > 10 cm")]
        lines += ["| metric | " + " | ".join(tags) + (" | change |" if len(tags) == 2 else " |"),
                  "|---|" + "---|" * (len(tags) + (1 if len(tags) == 2 else 0))]
        for k, label in keys:
            vals = [runs[t]["summary"].get(k) for t in tags]
            row = f"| {label} | " + " | ".join("-" if v is None else f"{v:,.2f}" for v in vals)
            if len(tags) == 2 and None not in vals:
                row += f" | {vals[1] - vals[0]:+,.2f} ({100 * (vals[1] - vals[0]) / max(abs(vals[0]), 1e-9):+.1f} %)"
            lines.append(row + " |")
        lines += ["", f"Design: {self._design_summary_text()}", "", "![maps](flood_maps.png)", "",
                  "Method note: live local-inertial solver validated against the published 0.5 m GPU runs (flooded-area IoU 0.86, depth correlation 0.95); "
                  "the study's 600 drain inlets are treated as clogged (25 Nov 2025 failure mode); results at this grid size are for design comparison, not for sizing."]
        (folder / "report.md").write_text("\n".join(lines))
        self._lab_say(f"Exported flood study to {folder}")
        print(f"[flood-lab] exported to {folder}")
        return str(folder)

    # ------------------------------------------------------------ hotspots
    def _lab_hotspots(self, depth: np.ndarray | None = None, k: int = 6, sep_m: float = 70.0) -> list[tuple]:
        """Places where people would actually meet the flood: open street/ground cells
        wet 8 cm - 1.2 m (not 3 m courtyard pits), ranked by the wet area in a 30 m
        window. Returns [(x, y, wet_area_m2)] in the local frame."""
        from scipy import ndimage as ndi
        lab = self.flood_lab
        rec = self._lab_best_record()
        if depth is None:
            if rec is None:
                return []
            depth = rec["result"].max_depth
        base, georef = self._lab_inputs()
        og = base.valid & ~base.water & ~(base.building if base.building is not None else False)
        wet = og & (depth > 0.08) & (depth < 1.2)
        win = max(3, int(round(30.0 / base.res)))
        score = ndi.uniform_filter(wet.astype(float), win) * win * win * base.res ** 2
        h, w = score.shape
        out, taken = [], []
        for j in np.argsort(score.ravel())[::-1][:20000]:
            if len(out) >= k or score.ravel()[j] < 150.0:
                break
            x, y = georef.xy[j]
            if any((x - a) ** 2 + (y - b) ** 2 < sep_m ** 2 for a, b in taken):
                continue
            taken.append((x, y))
            out.append((float(x), float(y), float(score.ravel()[j])))
        return out

    def _lab_clear_camera(self, x: float, y: float, dist: float = 80.0, prefer_az: float = 25.0) -> list:
        """Camera [pos, focal, up] looking at (x, y) with an unobstructed line of sight:
        candidate azimuths x elevations x distances are ray-traced against the building
        mesh (in the flat frame, where it lives) and the first clear one nearest the
        preferred azimuth / a street-level elevation wins."""
        dem = self.street_graph.graph.get("terrain_sampler")
        draped = bool(self.scene_state.get("_terrain_drape_active")) and dem is not None

        def gz(px, py):
            return float(np.asarray(dem(np.array([[px, py]])))[0]) if draped else 0.0

        gt = gz(x, y)
        cur = getattr(self, "buildings_mesh", None)
        sig = (id(cur), cur.n_cells if cur is not None else 0)
        bm = getattr(self, "_lab_bsurf", None)
        if cur is not None and (bm is None or getattr(self, "_lab_bsurf_sig", None) != sig):
            bm = self._lab_bsurf = cur.extract_surface(algorithm="dataset_surface").triangulate()
            self._lab_bsurf_sig = sig
        target_flat = np.array([x, y, 1.0])
        best = None
        for d in (dist, dist * 1.35, dist * 1.8):
            for el in (22, 30, 40, 52, 64):
                for daz in (0, 25, -25, 50, -50, 90, -90, 130, -130, 180):
                    az = np.radians(prefer_az + daz)
                    cx, cy = x + d * np.cos(np.radians(el)) * np.sin(az), y - d * np.cos(np.radians(el)) * np.cos(az)
                    zc_flat = d * np.sin(np.radians(el)) + 1.0
                    clear = True
                    if bm is not None:
                        pts, _ = bm.ray_trace(np.array([cx, cy, zc_flat]), target_flat, first_point=False)
                        clear = len(pts) == 0
                        if clear and zc_flat < 60.0:      # camera itself inside a footprint?
                            up, _ = bm.ray_trace(np.array([cx, cy, 0.0]), np.array([cx, cy, zc_flat]), first_point=False)
                            clear = len(up) == 0
                    if clear:
                        best = (cx, cy, gz(cx, cy) + zc_flat)
                        break
                if best:
                    break
            if best:
                break
        if best is None:                                        # fall back: look straight down
            best = (x + 1.0, y - 1.0, gt + 2.2 * dist)
        return [best, (x, y, gt + 0.5), (0, 0, 1)]

    def flood_lab_fly_hotspot(self, index: int | None = None) -> tuple | None:
        """Camera to the next worst-flooded street (button / key)."""
        lab = self.flood_lab
        spots = self._lab_hotspots()
        if not spots:
            self._lab_say("No flooding to fly to — run a flood first")
            return None
        i = (lab.get("hot_idx", -1) + 1) % len(spots) if index is None else int(index) % len(spots)
        lab["hot_idx"] = i
        x, y, a = spots[i]
        cam = self.plotter.camera
        try:
            cam.ParallelProjectionOff()
        except Exception:
            pass
        pos, fp, up = self._lab_clear_camera(x, y, 75.0, 25 + 40 * i)
        z = fp[2] - 0.5
        self.plotter.camera_position = [pos, fp, up]
        self.plotter.camera.view_angle = 42
        self.plotter.renderer.ResetCameraClippingRange()
        self._lab_say(f"Flood hotspot {i + 1}/{len(spots)}: {a:,.0f} m2 wet within 30 m")
        self.plotter.render()
        return (x, y, z)

    # ------------------------------------------------------ official trees
    def _lab_official_trees(self, on: bool) -> int:
        """Visual street/garden trees in the official corridor's garden cells (hydrology is
        already in the garden material; these make the built corridor read as green)."""
        lab = self.flood_lab
        # remove previous
        for key in list(lab.get("official_tree_keys", [])):
            actor = self.scene_state.pop(f"_tree_{key}", None)
            try:
                self.plotter.remove_actor(f"_tree_{key}", reset_camera=False)
            except Exception:
                pass
            for dct in ("_tree_seeds_flat", "_tree_templates", "_tree_actor_kwargs"):
                (self.scene_state.get(dct) or {}).pop(key, None)
        lab["official_tree_keys"] = []
        if not on or lab["official_mat"] is None:
            return 0
        import json
        from floodsim.terrain import TERRAIN_DIR
        from render.trees import add_tree_groups
        tr = json.loads((TERRAIN_DIR / "dem_transform.json").read_text())
        mat = lab["official_mat"]
        step = 20                                            # 10 m at 0.5 m cells
        rr, cc = np.mgrid[step // 2:mat.shape[0]:step, step // 2:mat.shape[1]:step]
        rr, cc = rr.ravel(), cc.ravel()
        k = 3                                                # the 1.5 m around the tree must be garden
        inside = np.array([(mat[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1] == 5).all() for r, c in zip(rr, cc)])
        rr, cc = rr[inside], cc[inside]
        if rr.size == 0:
            return 0
        _, georef = self._lab_inputs()
        e = float(tr["minx"]) + (cc + 0.5) * float(tr["res"])
        n = float(tr["maxy"]) - (rr + 0.5) * float(tr["res"])
        xy = georef.utm_to_local(e, n)
        before = set((self.scene_state.get("_tree_seeds_flat") or {}).keys())
        add_tree_groups(self.plotter, self.scene_state, "offc", xy)
        keys = [k_ for k_ in (self.scene_state.get("_tree_seeds_flat") or {}) if k_.startswith("offc_") and k_ not in before]
        lab["official_tree_keys"] = keys
        if self.scene_state.get("_terrain_drape_active"):
            dem = self.street_graph.graph.get("terrain_sampler")
            if dem is not None:
                self._drape_trees(dem)
        return len(xy)

    # ------------------------------------------------------- official decal
    def _lab_official_decal(self, on: bool) -> None:
        """Paint the official corridor's material classes as ground decals."""
        lab = self.flood_lab
        name = "lab_official_decal"
        try:
            self.plotter.remove_actor(name, reset_camera=False)
        except Exception:
            pass
        self.scene_state.pop("lab_official_actor", None)
        if not on or lab["official_mat"] is None:
            return
        base, georef = self._lab_inputs()
        fr = FD.class_fractions_from_raster(lab["official_mat"], base.meta["factor"], base.dem.shape)
        h, w = base.dem.shape
        total = sum(fr.values())
        dom = np.zeros((h, w), np.int32)
        best = np.zeros((h, w))
        for c, a in fr.items():
            m = a > best
            dom[m] = c
            best = np.where(m, a, best)
        rgba = np.zeros((h * w, 4), np.float32)
        cols = {c: pv.Color(col).float_rgb for c, col in MATERIAL_CYCLE}
        cols.update({1: (0.3, 0.3, 0.32), 9: (0.25, 0.55, 0.25)})
        flat = dom.ravel()
        for c in np.unique(flat[flat > 0]):
            m = flat == c
            rgba[m, :3] = cols.get(int(c), (0.4, 0.8, 0.4))
        rgba[:, 3] = np.where(total.ravel() > 0.25, 0.85, 0.0)
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([georef.xy, np.full(len(georef.xy), 0.194)])
        grid.dimensions = (w, h, 1)
        # per-CELL colour (class at the cell's own point): per-point colours blend across class edges
        cell = (rgba.reshape(h, w, 4)[:-1, :-1] * 255).astype(np.uint8).reshape(-1, 4)
        grid.cell_data["rgba"] = cell
        actor = self.plotter.add_mesh(grid, scalars="rgba", rgba=True, lighting=False, show_scalar_bar=False,
                                      reset_camera=False, name=name)
        actor.PickableOff()
        m = actor.GetMapper()
        m.SetResolveCoincidentTopologyToPolygonOffset()
        m.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)     # over the photo, under the flood layers (-3)
        self.scene_state["lab_official_actor"] = actor
        if self.scene_state.get("_terrain_drape_active"):
            dem = self.street_graph.graph.get("terrain_sampler")
            if dem is not None:
                self._drape_over_ground("lab_official_actor", dem)
