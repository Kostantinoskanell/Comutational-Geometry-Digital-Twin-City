"""SumoMixin — SUMO co-simulation layer.

SUMO drives the vehicles (validated microsimulation); this app renders them.
Vehicles are stepped on SUMO's own cadence and eased toward their latest
position each frame so motion stays smooth between steps.

Coordinate path:  SUMO net XY → (lon, lat) via sumolib  →  app local metres via
the same pyproj transformer used by the GTFS-Realtime layer.

CLI:  --sumo-cfg  (.sumocfg)   enables this layer
      --sumo-binary / --sumo-gui / --sumo-step / --sumo-net / --sumo-port

Run with --n-cars 0 for a pure SUMO scene, or alongside the IDM cars to compare.
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv


# Vehicle-class → colour.  Matches SUMO vClass strings.
_VCLASS_COLOR = {
    "passenger":   "#d64545",
    "private":     "#d64545",
    "hov":         "#d64545",
    "taxi":        "#f2c14e",
    "bus":         "#00E5A0",
    "coach":       "#00E5A0",
    "minibus":     "#00c78c",
    "tram":        "#00b5cc",
    "rail":        "#00b5cc",
    "rail_urban":  "#00b5cc",
    "truck":       "#e08a3c",
    "trailer":     "#e08a3c",
    "delivery":    "#e0a83c",
    "emergency":   "#ffffff",
    "police":      "#4a90d9",
    "authority":   "#4a90d9",
    "motorcycle":  "#c45ad0",
    "moped":       "#c45ad0",
    "bicycle":     "#4aa3df",
    "pedestrian":  "#a0d080",
}
_DEFAULT_COLOR   = "#aaaaaa"
_PERSON_COLOR    = "#a0d080"   # SUMO pedestrians

# Map each SUMO vClass string to a glyph pool key (one pool = one mesh template)
_VCLASS_POOL_KEY: dict[str, str] = {
    "passenger": "car",   "private": "car",   "hov": "car",
    "taxi":      "taxi",
    "bus":       "bus",   "coach":  "bus",    "minibus":    "bus",
    "truck":     "truck", "trailer":"truck",
    "tram":      "tram",  "rail":   "tram",   "rail_urban": "tram",
    "delivery":  "van",
    "motorcycle":"moto",  "moped":  "moto",
    "bicycle":   "bicycle",
    "emergency": "emerg", "police": "emerg",  "authority":  "emerg",
}
_DEFAULT_POOL_KEY = "car"

# Colour for each pool (used when creating its GlyphInstances)
_POOL_COLOR: dict[str, str] = {
    "car":     "#d64545",
    "taxi":    "#f2c14e",
    "bus":     "#00E5A0",
    "truck":   "#e08a3c",
    "tram":    "#00b5cc",
    "van":     "#e0a83c",
    "moto":    "#c45ad0",
    "bicycle": "#4aa3df",
    "emerg":   "#ffffff",
}


class SumoMixin:

    _SUMO_CONV         = 0.88     # fraction converged per SUMO step (frame-rate independent)
    _SUMO_SCENE_MARGIN = 250.0    # metres beyond scene bounds before culling

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_sumo(self, cfg: str = "", binary: str = "sumo",
                   step_length: float = 0.1, use_gui: bool = False,
                   net_file: str = "", port: int | None = None) -> None:
        self.sumo: dict = {
            "enabled":      False,
            "conn":         None,
            "transformer":  None,
            "bounds":       None,
            "actors":       {},    # vid  -> actor
            "current":      {},    # vid  -> np.array([x,y,z,heading])
            "p_actors":     {},    # pid  -> actor  (persons)
            "p_current":    {},    # pid  -> np.array([x,y,z])
            "meshes":       {},    # cache: class-key -> base mesh
            "last_step_t":  0.0,
            "step_length":  max(0.01, float(step_length)),
        }
        if not cfg:
            return

        try:
            from sumo_bridge import SumoConnection
        except Exception as exc:
            print(f"[sumo] bridge unavailable: {exc}")
            return

        # WGS84 → local projected metres (same derivation as GTFS-RT)
        proj_str = self.street_graph.graph.get("proj_str", "")
        if not proj_str:
            lat = float(getattr(self.args, "lat", 0.0))
            lon = float(getattr(self.args, "lon", 0.0))
            proj_str = (f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 "
                        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
            # Store it: graph_to_sumo_plainxml (editor network rebuilds)
            # needs it, otherwise every edit fails with "no proj_str in graph"
            self.street_graph.graph["proj_str"] = proj_str
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
        except Exception as exc:
            print(f"[sumo] CRS transform unavailable: {exc}")
            return

        bounds = None
        try:
            if self.buildings_mesh is not None and self.buildings_mesh.n_points > 0:
                bx0, bx1, by0, by1 = self.buildings_mesh.bounds[:4]
                m = self._SUMO_SCENE_MARGIN
                bounds = (bx0 - m, bx1 + m, by0 - m, by1 + m)
        except Exception:
            pass

        conn = SumoConnection(
            sumo_cfg=cfg, binary=binary, step_length=step_length,
            use_gui=use_gui, net_file=net_file, port=port,
        )
        if not conn.start():
            # In sumo engine mode this is fatal — degrade to IDM with a clear warning
            if getattr(self.args, "engine", "idm") == "sumo":
                print(
                    "[sumo] WARNING: --engine sumo requested but SUMO connection failed. "
                    "Degrading to IDM engine. Check your SUMO installation, $SUMO_HOME, "
                    "and the --sumo-cfg path."
                )
                if hasattr(self, "scene_state"):
                    self.scene_state["engine"] = "idm"
            return   # message already printed by the bridge

        self.sumo.update({
            "enabled":     True,
            "conn":        conn,
            "transformer": transformer,
            "bounds":      bounds,
        })
        print("[sumo] co-simulation layer enabled "
              "(run with --n-cars 0 for a pure SUMO scene)")

    # ------------------------------------------------------------------
    # Mesh per vehicle class (cached, realistic proportions)
    # ------------------------------------------------------------------

    def _sumo_mesh_for(self, vclass: str, length: float) -> pv.PolyData:
        meshes = self.sumo["meshes"]

        # Bus / coach — use the GLB-based bus mesh if available
        if vclass in ("bus", "coach", "minibus") and getattr(self, "bus_mesh", None) is not None:
            return self.bus_mesh

        # Choose a cache key based on class, not just size
        if vclass in ("motorcycle", "moped"):
            key = "moto"
        elif vclass in ("bicycle",):
            key = "bicycle"
        elif vclass in ("truck", "trailer"):
            key = "truck"
        elif vclass in ("tram", "rail", "rail_urban"):
            key = "rail"
        elif vclass in ("bus", "coach", "minibus"):
            key = "bus_box"        # fallback if no bus_mesh
        elif vclass in ("delivery",):
            key = "van"
        elif length >= 7.0:
            key = "big"
        else:
            key = "car"

        m = meshes.get(key)
        if m is None:
            if key == "moto":
                # narrow, upright, short
                m = pv.Box(bounds=(-1.1, 1.1, -0.18, 0.18, 0.0, 1.15))
            elif key == "bicycle":
                # very narrow, low
                m = pv.Box(bounds=(-0.85, 0.85, -0.12, 0.12, 0.0, 0.90))
            elif key == "truck":
                # length from SUMO (typically 7-16 m), taller cab
                l = max(7.0, length)
                m = pv.Box(bounds=(-l / 2, l / 2, -1.3, 1.3, 0.0, 3.8))
            elif key == "rail":
                l = max(15.0, length)
                m = pv.Box(bounds=(-l / 2, l / 2, -1.5, 1.5, 0.0, 3.5))
            elif key == "bus_box":
                l = max(10.0, length)
                m = pv.Box(bounds=(-l / 2, l / 2, -1.25, 1.25, 0.0, 3.0))
            elif key == "van":
                m = pv.Box(bounds=(-2.8, 2.8, -1.0, 1.0, 0.0, 2.1))
            elif key == "big":
                l = max(7.0, length)
                m = pv.Box(bounds=(-l / 2, l / 2, -1.3, 1.3, 0.0, 3.0))
            else:   # car
                m = pv.Box(bounds=(-2.25, 2.25, -0.9, 0.9, 0.0, 1.45))
            meshes[key] = m
        return m

    def _sumo_person_mesh(self) -> pv.PolyData:
        """Tiny cylinder for SUMO pedestrians."""
        meshes = self.sumo["meshes"]
        if "person" not in meshes:
            meshes["person"] = pv.Cylinder(
                center=(0.0, 0.0, 0.85), direction=(0, 0, 1),
                radius=0.22, height=1.7, resolution=6,
            )
        return meshes["person"]

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------

    def _animate_sumo(self, frame_dt: float) -> None:
        self._sumo_check_rebuild_done()   # always poll, even while rebuild is running
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        conn = rt["conn"]
        if conn is None or not conn.ready:
            return

        try:
            now = time.perf_counter()
            if now - float(rt["last_step_t"]) >= rt["step_length"]:
                rt["last_step_t"] = now
                conn.step()
                rt["_snapshot"]   = conn.vehicles()
                rt["_p_snapshot"] = conn.persons()

            _step = float(rt["step_length"])
            self._render_sumo_vehicles(rt.get("_snapshot", []), frame_dt, _step)
            self._render_sumo_persons(rt.get("_p_snapshot", []), frame_dt, _step)

            # Analytics keys ('i' incident, 'u' congestion) are bound centrally
            # at startup in main_ast6 — registering them here again would make
            # each press fire twice (toggle on+off = visible no-op).
            if not rt.get("_analytics_keys"):
                rt["_analytics_keys"] = True
                print("[sumo] 'i' = breakdown incident, 'u' = live congestion overlay")

            # 1 Hz analytics: KPI snapshot, congestion overlay, incident releases
            if now - float(rt.get("_analytics_t", 0.0)) >= 1.0:
                rt["_analytics_t"] = now
                try:
                    rt["_kpi"] = conn.kpi_snapshot()
                except Exception:
                    pass
                if rt.get("_cong_visible"):
                    self._sumo_update_congestion()
                self._sumo_render_tl_lights()
                # Auto-release expired breakdowns
                _inc = rt.get("_incidents", [])
                still = []
                for vid, t_rel in _inc:
                    if now >= t_rel:
                        conn.incident_release(vid)
                        try:
                            self.plotter.remove_actor(f"incident_{vid}")
                        except Exception:
                            pass
                    else:
                        still.append((vid, t_rel))
                rt["_incidents"] = still
        except Exception as exc:
            print(f"[sumo] animate error: {exc}")

    # ------------------------------------------------------------------
    # Analytics — incidents + live congestion overlay
    # ------------------------------------------------------------------

    def _sumo_trigger_incident(self, duration_s: float = 60.0) -> None:
        """'i' — break down a random vehicle for duration_s (jam generator)."""
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled") or rt.get("conn") is None:
            return
        vid = rt["conn"].incident_break_random_vehicle(duration_s)
        if vid is None:
            print("[sumo] no vehicles to break down")
            return
        rt.setdefault("_incidents", []).append(
            (vid, time.perf_counter() + float(duration_s)))
        # Red warning marker over the broken vehicle
        try:
            pos = rt["current"].get(vid)
            if pos is not None:
                self.plotter.add_mesh(
                    pv.Cone(center=(float(pos[0]), float(pos[1]), 7.0),
                            direction=(0, 0, -1), radius=2.0, height=4.0),
                    color="#ff2222", lighting=False, name=f"incident_{vid}",
                    reset_camera=False,
                )
        except Exception:
            pass

    def _sumo_render_tl_lights(self) -> None:
        """Render SUMO's ACTUAL signal states as bulbs (1 Hz).

        In SUMO mode the FSM traffic-light bulbs show colors SUMO's cars do
        not obey — misleading.  This draws one bulb per controlled link at the
        end of its incoming lane, colored from getRedYellowGreenState, and
        hides the FSM bulb actor while active.
        """
        rt = self.sumo
        conn = rt.get("conn")
        transformer = rt.get("transformer")
        if conn is None or transformer is None:
            return
        try:
            links = conn.tl_link_states()
        except Exception:
            return
        if not links:
            return

        # Hide the FSM bulbs once — SUMO's controllers are authoritative here
        if not rt.get("_fsm_tl_hidden"):
            rt["_fsm_tl_hidden"] = True
            try:
                _fsm = self.scene_state.get("tl_actor")
                if _fsm is not None:
                    _fsm.VisibilityOff()
                    print("[sumo] FSM traffic-light bulbs hidden — "
                          "showing SUMO's actual signal states")
            except Exception:
                pass

        _COL = {"G": [0, 200, 80], "g": [0, 200, 80],
                "y": [255, 210, 0], "Y": [255, 210, 0]}
        xy = np.array([transformer.transform(lo, la) for lo, la, _ in links], dtype=float)
        z = np.full(xy.shape[0], 4.8)
        dem = self.street_graph.graph.get("terrain_sampler")
        if bool(self.scene_state.get("_terrain_drape_active")) and dem is not None:
            try:
                z += np.asarray(dem(xy), dtype=float)
            except Exception:
                pass
        pts = np.column_stack((xy, z))
        rgb = np.array([_COL.get(ch, [220, 40, 40]) for _, _, ch in links], dtype=np.uint8)

        cloud = pv.PolyData(pts)
        cloud["colors"] = rgb
        try:
            self.plotter.remove_actor("sumo_tl_lights")
        except Exception:
            pass
        try:
            self.plotter.add_mesh(
                cloud, scalars="colors", rgb=True,
                point_size=14, render_points_as_spheres=True,
                lighting=False, reset_camera=False, name="sumo_tl_lights",
            )
        except Exception as exc:
            print(f"[sumo] TL light render failed: {exc}")

    def _sumo_toggle_congestion(self) -> None:
        """'u' — toggle the live per-edge congestion overlay."""
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        vis = not bool(rt.get("_cong_visible"))
        rt["_cong_visible"] = vis
        if vis:
            print("[sumo] congestion overlay ON (green=free, red=jammed)")
            self._sumo_update_congestion()
        else:
            print("[sumo] congestion overlay OFF")
            try:
                self.plotter.remove_actor("sumo_congestion")
            except Exception:
                pass

    def _sumo_update_congestion(self) -> None:
        """Rebuild the congestion polyline overlay (1 Hz while visible).

        Uses SUMO's own edge geometry so it works with any net (initial
        auto-built nets have different edge ids than the editor exporter).
        """
        rt = self.sumo
        conn = rt.get("conn")
        transformer = rt.get("transformer")
        if conn is None or transformer is None:
            return
        stats = conn.edge_congestion()
        if not stats:
            try:
                self.plotter.remove_actor("sumo_congestion")
            except Exception:
                pass
            return

        dem = self.street_graph.graph.get("terrain_sampler")
        drape_on = bool(self.scene_state.get("_terrain_drape_active")) and dem is not None

        pts_all: list[np.ndarray] = []
        lines: list[int] = []
        colors: list[list[int]] = []
        for eid, (mean_v, ff_v, n_veh) in stats.items():
            shape = conn.edge_shape_lonlat(eid)
            if not shape:
                continue
            xy = np.array([transformer.transform(lo, la) for lo, la in shape], dtype=float)
            z = np.full(xy.shape[0], 1.2)
            if drape_on:
                try:
                    z += np.asarray(dem(xy), dtype=float)
                except Exception:
                    pass
            pts = np.column_stack((xy, z))
            ratio = mean_v / max(ff_v, 0.1)
            if ratio > 0.8:
                col = [60, 200, 80]      # free-flowing
            elif ratio > 0.5:
                col = [255, 190, 0]      # slowing
            else:
                col = [230, 40, 40]      # jammed
            base = sum(p.shape[0] for p in pts_all)
            pts_all.append(pts)
            lines.extend([pts.shape[0], *range(base, base + pts.shape[0])])
            colors.extend([col] * 1)     # one colour per line cell

        if not pts_all:
            return
        mesh = pv.PolyData(np.vstack(pts_all))
        mesh.lines = np.asarray(lines, dtype=np.int64)
        mesh.cell_data["RGB"] = np.asarray(colors, dtype=np.uint8)
        try:
            self.plotter.remove_actor("sumo_congestion")
        except Exception:
            pass
        try:
            self.plotter.add_mesh(
                mesh, scalars="RGB", rgb=True, line_width=7,
                render_lines_as_tubes=True, lighting=False,
                reset_camera=False, name="sumo_congestion",
            )
        except Exception as exc:
            print(f"[sumo] congestion render failed: {exc}")

    # ------------------------------------------------------------------
    # Vehicle rendering  (pooled glyph instances — no actor creation per tick)
    # ------------------------------------------------------------------

    def _render_sumo_vehicles(self, vehicles: list[dict], frame_dt: float = 0.016, step_length: float = 0.1) -> None:
        rt          = self.sumo
        transformer = rt["transformer"]
        bounds      = rt["bounds"]
        current     = rt["current"]   # lerp state — physics, unchanged
        dem         = self.street_graph.graph.get("terrain_sampler")
        drape_on    = bool(self.scene_state.get("_terrain_drape_active")) and dem is not None

        # Lazy-init glyph pool registry
        if "glyph_pools" not in rt:
            rt["glyph_pools"]     = {}   # pool_key → GlyphInstances
            rt["glyph_positions"] = {}   # pool_key → list[(x,y,z,heading)]

        glyph_pools     = rt["glyph_pools"]
        glyph_positions = rt["glyph_positions"]

        _lerp_k = float(min(1.0, 1.0 - (1.0 - self._SUMO_CONV) ** (frame_dt / max(step_length, 1e-6))))

        # Reset per-class accumulator
        for k in glyph_positions:
            glyph_positions[k] = []

        counts: dict[str, int] = {}
        seen:   set[str]       = set()

        # First pass: transform + bounds check → collect valid vehicles
        valid: list[tuple] = []  # (vid, x, y, heading, vclass, length)
        for v in vehicles:
            vid = v["id"]
            try:
                x, y = transformer.transform(v["lon"], v["lat"])
            except Exception:
                continue
            if bounds is not None and not (
                bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]
            ):
                continue
            heading = (90.0 - float(v.get("angle", 0.0))) % 360.0
            vclass  = str(v.get("vclass", "passenger"))
            length  = float(v.get("length", 4.5))
            valid.append((vid, x, y, heading, vclass, length))

        # Vectorized DEM lookup — one scipy call for all vehicles instead of N
        if drape_on and valid:
            _xy = np.array([[x, y] for _, x, y, _, _, _ in valid], dtype=float)
            try:
                _h_arr = np.asarray(dem(_xy), dtype=float)
            except Exception:
                _h_arr = np.zeros(len(valid))
        else:
            _h_arr = np.zeros(len(valid))

        # Second pass: lerp + accumulate into glyph pools
        for _i, (vid, x, y, heading, vclass, length) in enumerate(valid):
            z      = float(_h_arr[_i])
            target = np.array([x, y, z, heading], dtype=float)

            seen.add(vid)
            counts[vclass] = counts.get(vclass, 0) + 1

            # Smooth lerp (physics layer — keep as-is)
            if vid not in current:
                current[vid] = target.copy()
            else:
                cur = current[vid]
                cur[:3] += (target[:3] - cur[:3]) * _lerp_k
                da = (target[3] - cur[3] + 180.0) % 360.0 - 180.0
                cur[3] = (cur[3] + da * _lerp_k) % 360.0

            pos      = current[vid]
            pool_key = _VCLASS_POOL_KEY.get(vclass, _DEFAULT_POOL_KEY)

            # Create pool on first encounter of this class key
            if pool_key not in glyph_pools:
                from glyph_instance import GlyphInstances
                mesh  = self._sumo_mesh_for(vclass, length)
                color = _POOL_COLOR.get(pool_key, _DEFAULT_COLOR)
                glyph_pools[pool_key]     = GlyphInstances(mesh, 32, color, self.plotter)
                glyph_positions[pool_key] = []
                print(f"[sumo] glyph pool created for class '{pool_key}'")

            glyph_positions.setdefault(pool_key, []).append(
                (float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3]))
            )

        # Remove lerp state for departed vehicles (no actor to delete)
        for vid in list(current.keys()):
            if vid not in seen:
                current.pop(vid, None)

        # Flush all pools — one numpy write per class, zero actor creation
        for pool_key, pool in glyph_pools.items():
            items = glyph_positions.get(pool_key) or []
            if items:
                positions = np.array([[x, y, z] for x, y, z, _ in items])
                headings  = np.array([h          for _, _, _, h in items])
                pool.update(positions, headings)
            else:
                pool.update(np.zeros((0, 3)))

        self._sumo_hud(counts, len(seen))

    # ------------------------------------------------------------------
    # Person (pedestrian) rendering  (single glyph pool — no actor churn)
    # ------------------------------------------------------------------

    def _render_sumo_persons(self, persons: list[dict], frame_dt: float = 0.016, step_length: float = 0.1) -> None:
        if not persons:
            return
        rt          = self.sumo
        transformer = rt["transformer"]
        bounds      = rt["bounds"]
        p_current   = rt["p_current"]
        dem         = self.street_graph.graph.get("terrain_sampler")
        drape_on    = bool(self.scene_state.get("_terrain_drape_active")) and dem is not None

        # Lazy-init person glyph pool
        if "p_glyph" not in rt:
            from glyph_instance import GlyphInstances
            rt["p_glyph"] = GlyphInstances(
                self._sumo_person_mesh(), 64, _PERSON_COLOR, self.plotter
            )
            print("[sumo] person glyph pool created")

        _lerp_k = float(min(1.0, 1.0 - (1.0 - self._SUMO_CONV) ** (frame_dt / max(step_length, 1e-6))))

        p_glyph          = rt["p_glyph"]
        active_positions = []
        seen: set[str]   = set()

        # First pass: transform + bounds check → collect valid persons
        valid_p: list[tuple] = []  # (pid, x, y)
        for p in persons:
            pid = p["id"]
            try:
                x, y = transformer.transform(p["lon"], p["lat"])
            except Exception:
                continue
            if bounds is not None and not (
                bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]
            ):
                continue
            valid_p.append((pid, x, y))

        # Vectorized DEM lookup — one call for all persons instead of N
        if drape_on and valid_p:
            _pxy = np.array([[x, y] for _, x, y in valid_p], dtype=float)
            try:
                _ph_arr = np.asarray(dem(_pxy), dtype=float)
            except Exception:
                _ph_arr = np.zeros(len(valid_p))
        else:
            _ph_arr = np.zeros(len(valid_p))

        # Second pass: lerp + accumulate
        for _i, (pid, x, y) in enumerate(valid_p):
            z      = float(_ph_arr[_i])
            target = np.array([x, y, z], dtype=float)
            seen.add(pid)

            if pid not in p_current:
                p_current[pid] = target.copy()
            else:
                p_current[pid] += (target - p_current[pid]) * _lerp_k

            active_positions.append(p_current[pid].copy())

        # Remove lerp state for departed persons (no actor to delete)
        for pid in list(p_current.keys()):
            if pid not in seen:
                p_current.pop(pid, None)

        if active_positions:
            p_glyph.update(np.array(active_positions), None)  # cylinders: no heading
        else:
            p_glyph.update(np.zeros((0, 3)))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _sumo_hud(self, counts: dict[str, int], total: int) -> None:
        try:
            # Build a compact class summary (omit classes with 0)
            order = ["passenger", "private", "taxi", "bus", "coach", "minibus",
                     "truck", "trailer", "delivery", "motorcycle", "moped",
                     "bicycle", "tram", "rail", "emergency"]
            parts = []
            for cls in order:
                n = counts.get(cls, 0)
                if n:
                    label = {
                        "passenger": "cars", "private": "cars", "taxi": "taxi",
                        "bus": "buses", "coach": "buses", "minibus": "buses",
                        "truck": "trucks", "trailer": "trucks",
                        "delivery": "vans", "motorcycle": "motos",
                        "moped": "mopeds", "bicycle": "bikes",
                        "tram": "trams", "rail": "rail",
                        "emergency": "emerg",
                    }.get(cls, cls)
                    parts.append(f"{n} {label}")
            # merge duplicated labels (e.g. passenger + private both → "cars")
            merged: dict[str, int] = {}
            for cls, n in counts.items():
                lbl = {
                    "passenger": "cars", "private": "cars", "hov": "cars",
                    "taxi": "taxi", "bus": "buses", "coach": "buses",
                    "minibus": "buses", "truck": "trucks", "trailer": "trucks",
                    "delivery": "vans", "motorcycle": "motos", "moped": "mopeds",
                    "bicycle": "bikes", "tram": "trams", "rail": "rail",
                    "rail_urban": "rail", "emergency": "emerg", "police": "emerg",
                }.get(cls, cls)
                merged[lbl] = merged.get(lbl, 0) + n
            label_order = ["cars","buses","trucks","vans","taxi","motos","mopeds",
                           "bikes","trams","rail","emerg"]
            summary = "  ".join(
                f"{merged[l]} {l}" for l in label_order if l in merged
            )
            # append any unknown classes
            known_lbls = set(label_order)
            extra = "  ".join(
                f"{n} {l}" for l, n in merged.items() if l not in known_lbls and n
            )
            if extra:
                summary = (summary + "  " + extra).strip()

            t = self.sumo["conn"].sim_time()
            line1 = f"⚙ SUMO co-sim  t={t:.0f}s"
            line2 = summary if summary else f"{total} vehicles"
            # City-health KPIs (refreshed at 1 Hz in _animate_sumo)
            _kpi = self.sumo.get("_kpi") or {}
            line3 = ""
            if _kpi:
                line3 = (f"delay {_kpi.get('mean_timeloss_s', 0.0):.0f}s"
                         f"  wait {_kpi.get('mean_waiting_s', 0.0):.0f}s"
                         f"  arrived {_kpi.get('arrived_total', 0)}")
                _col = int(_kpi.get("collisions", 0))
                if _col:
                    line3 += f"  ⚠ {_col} collision{'s' if _col > 1 else ''}"
                # Pedestrians currently waiting (at stops / for rides)
                _n_wait = sum(1 for p in (self.sumo.get("_p_snapshot") or [])
                              if int(p.get("stage", 2)) == 1)
                if _n_wait:
                    line3 += f"  {_n_wait} ped waiting"
            hud = f"{line1}\n{line2}" + (f"\n{line3}" if line3 else "")
            self.plotter.add_text(
                hud,
                position=(0.02, 0.89), name="sumo_hud",
                font_size=9, viewport=True, color="#ffcf6b",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Click-to-route — called from ui_mixin's unified picker (main thread)
    # ------------------------------------------------------------------

    def _sumo_nearest_vehicle(self, px: float, py: float, max_dist: float = 8.0) -> str | None:
        """ID of the rendered SUMO vehicle nearest to a local-metres click, or None."""
        best_vid: str | None = None
        best_dist = float(max_dist)
        for vid, pos in self.sumo.get("current", {}).items():
            try:
                d = float(np.hypot(float(pos[0]) - px, float(pos[1]) - py))
            except Exception:
                continue
            if d < best_dist:
                best_dist = d
                best_vid = vid
        return best_vid

    def _sumo_route_click(self, px: float, py: float) -> bool:
        """Two-stage click routing for SUMO vehicles.

        Stage 0: click near a vehicle → select it (green marker follows selection print).
        Stage 1: click anywhere → nearest net edge becomes the vehicle's new target;
                 the remaining route is drawn as a cyan polyline.
        Returns True when the click was consumed by this workflow.
        """
        rt = self.sumo
        vid = rt.get("_route_sel_vid")

        # ── Stage 0: no selection yet — try to pick a vehicle ──
        if vid is None:
            vid = self._sumo_nearest_vehicle(px, py)
            if vid is None:
                return False   # click not consumed
            rt["_route_sel_vid"] = vid
            print(f"[sumo-route] vehicle {vid} selected — click a destination")
            try:
                pos = rt["current"][vid]
                self.plotter.add_mesh(
                    pv.Sphere(radius=2.0, center=(float(pos[0]), float(pos[1]), 2.0),
                              theta_resolution=18, phi_resolution=18),
                    color="#33cc66", name="sumo_route_src",
                )
            except Exception:
                pass
            return True

        # ── Stage 1: have a selection — retarget to the clicked location ──
        def _clear_selection() -> None:
            rt["_route_sel_vid"] = None
            try:
                self.plotter.remove_actor("sumo_route_src")
            except Exception:
                pass

        conn = rt["conn"]
        if conn is None or not conn.ready:
            print("[sumo-route] SUMO connection not ready — selection cleared")
            _clear_selection()
            return True

        # Local metres → WGS84 (inverse of the render transformer), cached
        to_wgs = rt.get("_to_wgs")
        if to_wgs is None:
            try:
                from pyproj import Transformer
                proj_str = self.street_graph.graph.get("proj_str", "")
                to_wgs = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
                rt["_to_wgs"] = to_wgs
            except Exception as exc:
                print(f"[sumo-route] inverse CRS transform unavailable: {exc}")
                _clear_selection()
                return True

        try:
            lon, lat = to_wgs.transform(px, py)
        except Exception as exc:
            print(f"[sumo-route] click → lon/lat failed: {exc}")
            _clear_selection()
            return True

        edge_id = conn.nearest_edge_id(lon, lat)
        if edge_id is None:
            print("[sumo-route] no drivable edge near click — try again")
            return True   # keep selection

        # Vehicle may have left the simulation while we were choosing
        try:
            if vid not in conn._traci.vehicle.getIDList():
                print(f"[sumo-route] vehicle {vid} left the simulation")
                _clear_selection()
                return True
        except Exception:
            pass

        ok = conn.vehicle_set_target(vid, edge_id)
        if ok:
            shape = conn.vehicle_route_shape_lonlat(vid)
            if shape and len(shape) >= 2:
                try:
                    transformer = rt["transformer"]
                    pts = np.array(
                        [[*transformer.transform(lo, la), 1.5] for lo, la in shape],
                        dtype=float,
                    )
                    prev = rt.get("_route_actor")
                    if prev is not None:
                        try:
                            self.plotter.remove_actor(prev)
                        except Exception:
                            pass
                    rt["_route_actor"] = self.plotter.add_lines(
                        pts, color="#00e5ff", width=5, connected=True,
                    )
                    # ETA via SUMO's router (no vehicle spawn needed)
                    _eta = ""
                    try:
                        _cur_edge = str(conn._traci.vehicle.getRoadID(vid))
                        if _cur_edge and not _cur_edge.startswith(":"):
                            _tt = conn.find_route_time(_cur_edge, edge_id)
                            if _tt:
                                _eta = f", ETA ~{_tt:.0f}s"
                    except Exception:
                        pass
                    print(f"[sumo-route] vehicle {vid} → edge {edge_id} "
                          f"({len(shape)} route points{_eta})")
                except Exception as exc:
                    print(f"[sumo-route] route overlay failed: {exc}")

        _clear_selection()
        return True

    # ------------------------------------------------------------------
    # Editor sync — called from ui_mixin after in-app road edits
    # ------------------------------------------------------------------

    def _sumo_sync_tl(self, node_id, enabled: bool) -> None:
        """Toggle a traffic light via TraCI without restarting SUMO.

        node_id  : the graph node ID (typically an OSM int) at the edited junction
        enabled  : True = activate TL, False = disable (switch to 'off' program)
        """
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        conn = rt.get("conn")
        if conn is None or not conn.ready:
            return

        tl_id = str(node_id)
        available = conn.tl_ids()

        if tl_id not in available:
            # SUMO sometimes uses cluster IDs for joined junctions.
            matches = [t for t in available if tl_id in t]
            if matches:
                tl_id = matches[0]
            else:
                print(f"[sumo] TL sync: {node_id} is not a signalised junction in "
                      "SUMO — skipped (try rebuilding the network)")
                return

        if not enabled:
            ok = conn.tl_set_program(tl_id, "off")
            print(f"[sumo] TL {tl_id}: {'disabled (→ off)' if ok else 'disable failed'}")
        else:
            # Try restoring program '0' (default netconvert output).
            ok = conn.tl_set_program(tl_id, "0")
            if not ok:
                # Fall back to whichever program was defined first.
                try:
                    defs = conn._traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
                    if defs:
                        ok = conn.tl_set_program(tl_id, defs[0].programID)
                except Exception:
                    pass
            print(f"[sumo] TL {tl_id}: {'enabled' if ok else 'enable failed (junction may lack a TL program)'}")

    def _sumo_geometry_rebuild(self) -> None:
        """Export current street_graph → netconvert → new net.xml → restart SUMO.

        Runs netconvert + randomTrips in a background thread so the UI stays
        responsive.  Camera and UI state are preserved across the restart.
        Displays an overlay while the rebuild is in progress.
        """
        import threading

        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        conn = rt.get("conn")
        if conn is None:
            return

        # Snapshot camera so we can restore it after restart
        try:
            rt["_rebuild_cam_pos"]   = tuple(self.plotter.camera.position)
            rt["_rebuild_cam_focus"] = tuple(self.plotter.camera.focal_point)
            rt["_rebuild_cam_up"]    = tuple(self.plotter.camera.up)
        except Exception:
            pass

        # Pause SUMO animation and close the current connection
        rt["enabled"] = False
        conn.close()

        # Show status overlay
        try:
            self.plotter.add_text(
                "SUMO: rebuilding network…",
                position=(0.30, 0.50),
                name="_sumo_rebuild_overlay",
                font_size=14,
                color="#ffd54f",
                viewport=True,
            )
            self.plotter.render()
        except Exception:
            pass

        def _worker() -> None:
            from sumo_network_patch import (
                graph_to_sumo_plainxml,
                rebuild_sumo_net,
                generate_routes,
                update_sumocfg,
            )
            from pathlib import Path

            cfg_path = conn.sumo_cfg
            out_dir  = str(Path(cfg_path).parent)
            prefix   = "patch"
            net_out  = str(Path(out_dir) / "patched.net.xml")
            rou_out  = str(Path(out_dir) / "patched.rou.xml")
            trips_out = str(Path(out_dir) / "patched.trips.xml")

            try:
                print("[sumo] Exporting graph as SUMO plain-XML…")
                if not graph_to_sumo_plainxml(self.street_graph, out_dir, prefix):
                    print("[sumo] Graph export failed — SUMO stays disconnected")
                    return

                print("[sumo] Running netconvert on patched graph…")
                if not rebuild_sumo_net(prefix, out_dir, net_out):
                    print("[sumo] netconvert failed — SUMO stays disconnected")
                    return

                print("[sumo] Generating demand (randomTrips)…")
                _seed = int(getattr(self.args, "seed", 42))
                if not generate_routes(net_out, trips_out, rou_out, seed=_seed):
                    print("[sumo] Route generation failed — SUMO stays disconnected")
                    return

                print("[sumo] Updating .sumocfg…")
                update_sumocfg(cfg_path, net_out, rou_out)

                rt["_pending_restart"] = True
            except Exception as exc:
                import traceback
                print(f"[sumo] rebuild worker error: {exc}")
                traceback.print_exc()

        t = threading.Thread(target=_worker, daemon=True)
        rt["_rebuild_thread"] = t
        t.start()

    def _sumo_check_rebuild_done(self) -> None:
        """Poll for background SUMO rebuild completion (called from _animate_sumo)."""
        rt = getattr(self, "sumo", None)
        if not rt:
            return

        t = rt.get("_rebuild_thread")
        if t is None or t.is_alive():
            return

        rt["_rebuild_thread"] = None  # thread is done

        # Clear overlay
        try:
            self.plotter.add_text(
                "",
                position=(0.30, 0.50),
                name="_sumo_rebuild_overlay",
                font_size=14,
                color="#ffd54f",
                viewport=True,
            )
        except Exception:
            pass

        if not rt.pop("_pending_restart", False):
            print("[sumo] Network rebuild failed — falling back to old SUMO network")
            conn = rt.get("conn")
            if conn is not None:
                if conn.restart():
                    rt["enabled"] = True
                    print("[sumo] SUMO reconnected with old network")
                else:
                    print("[sumo] SUMO failed to reconnect with old network")
            return

        conn = rt.get("conn")
        if conn is None:
            return

        print("[sumo] Restarting SUMO connection after network patch…")
        ok = conn.restart()
        if ok:
            rt["enabled"] = True
            print("[sumo] SUMO reconnected with patched network")
        else:
            print("[sumo] SUMO reconnect failed after network patch")

        # Restore camera
        try:
            if "_rebuild_cam_pos" in rt:
                self.plotter.camera.position    = rt.pop("_rebuild_cam_pos")
                self.plotter.camera.focal_point = rt.pop("_rebuild_cam_focus")
                self.plotter.camera.up          = rt.pop("_rebuild_cam_up")
                self.plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _close_sumo(self) -> None:
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        conn = rt.get("conn")
        if conn is not None:
            conn.close()
        # Remove glyph actors from the renderer
        for pool in (rt.get("glyph_pools") or {}).values():
            try:
                pool.remove(self.plotter)
            except Exception:
                pass
        p_glyph = rt.get("p_glyph")
        if p_glyph is not None:
            try:
                p_glyph.remove(self.plotter)
            except Exception:
                pass
        # Remove the click-routing overlay, if any
        route_actor = rt.pop("_route_actor", None)
        if route_actor is not None:
            try:
                self.plotter.remove_actor(route_actor)
            except Exception:
                pass
        # Remove analytics overlays; restore FSM traffic-light bulbs
        for _name in ("sumo_tl_lights", "sumo_congestion"):
            try:
                self.plotter.remove_actor(_name)
            except Exception:
                pass
        if rt.pop("_fsm_tl_hidden", None):
            try:
                _fsm = self.scene_state.get("tl_actor")
                if _fsm is not None:
                    _fsm.VisibilityOn()
            except Exception:
                pass
        print("[sumo] connection closed")
