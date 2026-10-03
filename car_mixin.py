from __future__ import annotations
import math
import time
import numpy as np
import pyvista as pv
from idm import idm_tick
from traffic_lights import tick_all, update_light_mesh
from solar_routing import nearest_graph_node

try:
    from terrain_drape import car_heights_from_profiles as _car_heights_from_profiles
except ImportError:
    _car_heights_from_profiles = None


def _rush_hour_multiplier(hour: float) -> float:
    """Return a traffic intensity multiplier (0.3–1.4) for the given hour (0–24).

    Peaks at 08:00 (morning commute) and 18:00 (evening commute).
    Night trough around 03:00.
    """
    morning = math.exp(-0.5 * ((hour - 8.0) / 1.2) ** 2)
    evening = math.exp(-0.5 * ((hour - 18.0) / 1.2) ** 2)
    peak = max(morning, evening)
    night_dip = 0.4 * math.exp(-0.5 * ((hour - 3.0) / 2.0) ** 2) if hour < 10.0 else 0.0
    return float(min(1.4, max(0.6, 0.7 + 0.7 * peak - night_dip)))


class CarMixin:
    def _car_pose_on_path(self, path: dict[str, object], dist_m: float) -> np.ndarray:
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

    def _car_heading_deg_on_path(self, path: dict[str, object], dist_m: float) -> float:
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

    def _log_car_debug(self, tag: str) -> None:
        if not bool(self.args.debug_cars):
            return
        if not bool(self.car_anim["enabled"]):
            print(f"[cars-debug] {tag}: cars are disabled")
            return
        edge_idx = np.asarray(self.car_anim["edge_idx"], dtype=np.int64)
        dist = np.asarray(self.car_anim["dist"], dtype=float)
        speed = np.asarray(self.car_anim["speed"], dtype=float)
        n = int(edge_idx.shape[0])
        if n == 0:
            print(f"[cars-debug] {tag}: no active cars")
            return
        safe_len = np.maximum(self.car_path_lengths[edge_idx], 1e-6)
        mean_progress = float(np.mean(dist / safe_len))
        unique_edges = int(np.unique(edge_idx).size)
        first_path = int(edge_idx[0])
        first_pos = self._car_pose_on_path(self.car_paths[first_path], float(dist[0]))
        print(
            "[cars-debug] "
            f"{tag}: n={n}, unique_edges={unique_edges}, "
            f"mean_speed={float(np.mean(speed)):.2f}m/s, mean_progress={mean_progress:.2f}, "
            f"sample_xy=({first_pos[0]:.2f},{first_pos[1]:.2f})"
        )

    def _advance_cars(self, dt: float) -> None:
        """Delegate to the standalone IDM module."""
        _hour = float(self.scene_state.get("hour", 12.0))
        _effective_speed = float(self.args.traffic_speed) * _rush_hour_multiplier(_hour)

        # Per-car speed ceiling for this tick, passed to idm_tick as speed_cap:
        # walking pace within 12 m of an active pedestrian crossing, near-stop
        # within 30 m of an emergency vehicle.
        _cap = None
        _pos = np.asarray(self.car_anim.get("pos", []), dtype=float)
        if bool(self.car_anim.get("enabled")) and _pos.ndim == 2 and _pos.shape[0] > 0:
            from scipy.spatial import cKDTree as _CKT
            _crossing_pts = getattr(self, "ped_anim", {}).get("active_crossings", [])
            if _crossing_pts:
                _d, _ = _CKT(np.asarray(_crossing_pts, dtype=float)[:, :2]).query(_pos[:, :2])
                if np.any(_d < 12.0):
                    _cap = np.full(_pos.shape[0], np.inf)
                    _cap[_d < 12.0] = 1.4
            _em_positions = self._em_active_positions() if hasattr(self, "_em_active_positions") else []
            if _em_positions:
                _em_pts = np.array([p[:2] for p in _em_positions], dtype=float)
                _d, _ = _CKT(_em_pts).query(_pos[:, :2])
                if np.any(_d < 30.0):
                    if _cap is None:
                        _cap = np.full(_pos.shape[0], np.inf)
                    _cap[_d < 30.0] = np.minimum(_cap[_d < 30.0], 0.5)

        idm_tick(
            car_anim       = self.car_anim,
            car_paths      = self.car_paths,
            car_next_edges = self.car_next_edges,
            traffic_lights = self.traffic_lights_dict,
            dt             = dt,
            params         = self._idm_params,
            rng            = self.car_rng,
            traffic_speed  = _effective_speed,
            roundabout_yield_map = self.roundabout_yield_map,
            adj_left  = getattr(self, "adj_left_paths",  None),
            adj_right = getattr(self, "adj_right_paths", None),
            speed_cap = _cap,
        )

        # Gravity O-D demand: re-trip cars that have arrived (throttled).
        if getattr(self, "demand", None) is not None:
            try:
                self._reassign_finished_trips(_hour)
            except Exception:
                pass

    def _log_idm_diagnostics(self, tag: str) -> None:
        """Periodic IDM health check — active only when --debug-cars is set."""
        if not bool(self.args.debug_cars):
            return
        if not bool(self.car_anim["enabled"]):
            return

        spd   = np.asarray(self.car_anim["speed"],         dtype=float)
        des   = np.asarray(self.car_anim["desired_speed"],  dtype=float)
        acc   = np.asarray(self.car_anim["accel"],          dtype=float)
        eidx  = np.asarray(self.car_anim["edge_idx"],       dtype=np.int64)
        dist  = np.asarray(self.car_anim["dist"],           dtype=float)
        clen  = np.asarray(self.car_anim["car_len"],        dtype=float)
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
        elif min_gap < self._idm_params.s0:
            print(f"[IDM-WARN  ] {tag}: min gap={min_gap:.2f} m < s0={self._idm_params.s0} m "
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
        if worst_var > (self._idm_params.a_max * 2.5) ** 2:
            print(f"[IDM-WARN  ] {tag}: OSCILLATION — accel variance={worst_var:.2f} "
                  f"on busiest edge (threshold={((self._idm_params.a_max * 2.5)**2):.2f})")

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

    def _sample_car_positions(self) -> np.ndarray:
        # One-time: ensure path arrays are numpy (eliminates repeated np.asarray
        # list→array conversion in _car_pose_on_path — major hot-path savings).
        if not self.scene_state.get("_paths_np_converted") and self.car_paths:
            for _p in self.car_paths:
                if not isinstance(_p.get("points"), np.ndarray):
                    _p["points"] = np.asarray(_p["points"], dtype=np.float64)
                if not isinstance(_p.get("cum_len"), np.ndarray):
                    _p["cum_len"] = np.asarray(_p["cum_len"], dtype=np.float64)
            self.scene_state["_paths_np_converted"] = True

        edge_idx = np.asarray(self.car_anim["edge_idx"], dtype=np.int64)
        dist = np.asarray(self.car_anim["dist"], dtype=float)
        n = edge_idx.shape[0]
        n_paths = len(self.car_paths)
        positions = np.zeros((n, 3), dtype=float)
        if n_paths == 0:
            return positions
        safe_e = np.clip(edge_idx, 0, n_paths - 1)
        for i in range(n):
            positions[i] = self._car_pose_on_path(self.car_paths[int(safe_e[i])], float(dist[i]))
        # Apply terrain height only when draping is active.
        # When terrain is OFF, keep the path z (0.5 m road level) so agents
        # remain consistent with the flat road mesh.
        if n > 0 and self.scene_state.get("_terrain_drape_active"):
            _profiles = self.scene_state.get("_car_h_profiles")
            _ok = (_profiles is not None and _profiles.shape[0] > 0
                   and _car_heights_from_profiles is not None
                   and _profiles.shape[0] >= n_paths)
            if _ok:
                try:
                    positions[:, 2] = _car_heights_from_profiles(
                        _profiles,
                        self.scene_state["_car_h_lens"],
                        self.scene_state["_car_h_counts"],
                        safe_e, dist,
                    )
                    return positions
                except Exception as _exc:
                    _ctr = self.scene_state.get("_dem_fallback_ctr", 0) + 1
                    self.scene_state["_dem_fallback_ctr"] = _ctr
                    if _ctr <= 3 or _ctr % 200 == 0:
                        print(f"[car] profile lookup failed (#{_ctr}): {_exc} — falling back to live DEM")
            else:
                # Profiles missing or shape mismatch — log once then every 200 frames
                _ctr = self.scene_state.get("_dem_fallback_ctr", 0) + 1
                self.scene_state["_dem_fallback_ctr"] = _ctr
                if _ctr <= 3 or _ctr % 200 == 0:
                    _why = ("profiles=None" if _profiles is None
                            else f"profiles.shape[0]={_profiles.shape[0]} < n_paths={n_paths}"
                            if _profiles.shape[0] < n_paths
                            else "car_heights_from_profiles not importable")
                    print(f"[car] WARNING terrain active but profiles not ready (#{_ctr}): {_why} — using live DEM (slow!)")
            # Live DEM fallback (slow — should not happen in steady state)
            _dem = self.street_graph.graph.get("terrain_sampler")
            if _dem is not None:
                try:
                    positions[:, 2] = _dem(positions[:, :2])
                except Exception:
                    pass
        return positions

    def _sample_car_headings(self) -> np.ndarray:
        edge_idx = np.asarray(self.car_anim["edge_idx"], dtype=np.int64)
        dist = np.asarray(self.car_anim["dist"], dtype=float)
        n = edge_idx.shape[0]
        n_paths = len(self.car_paths)
        headings = np.zeros(n, dtype=float)
        if n_paths == 0:
            return headings
        safe_e = np.clip(edge_idx, 0, n_paths - 1)
        for i in range(n):
            headings[i] = self._car_heading_deg_on_path(self.car_paths[int(safe_e[i])], float(dist[i]))
        return headings

    def _car_pick_callback(self, mesh, cell_id) -> None:
        if mesh is None or cell_id < 0:
            return
        try:
            cx = float(mesh.cell_centers().points[cell_id, 0])
            cy = float(mesh.cell_centers().points[cell_id, 1])
            pos_array = np.asarray(self.car_anim.get("pos", []))
            if pos_array.shape[0] == 0:
                return
            pos_array = pos_array[:, :2]
            dist_sq = np.sum((pos_array - np.array([cx, cy]))**2, axis=1)
            car_idx = int(np.argmin(dist_sq))
            _e = int(self.car_anim["edge_idx"][car_idx])
            _d = float(self.car_anim["dist"][car_idx])
            _p3d = self._car_pose_on_path(self.car_paths[_e], _d)
            _snap_node = nearest_graph_node(self.street_graph, _p3d[0], _p3d[1])

            self._clear_route_actors()
            self.route_state["source_node"] = _snap_node
            _xy = self._node_xy(_snap_node)
            if _xy is not None:
                _src_actor = self.plotter.add_mesh(
                    pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0), theta_resolution=18, phi_resolution=18),
                    color="#33cc66",
                    render=False,
                )
                self.route_state["route_actors"] = [_src_actor]
            self._select_car_for_route(car_idx, _snap_node)
            self.route_state["stage"] = 1
            print(f"[route] Car {car_idx} selected as source — click target node.")

            if getattr(self.args, "solo", False):
                self.scene_state["solo_solar_Wh"] = 0.0
                self.scene_state["solo_mech_Wh"] = 0.0
        except Exception as _exc:
            print(f"[cars] Pick failed: {_exc}")

    def _physics_tl_fsm(self, dt: float) -> None:
        """Advance traffic-light FSM states at physics rate (20 Hz)."""
        _tl = self.scene_state.get("traffic_lights")
        if _tl:
            tick_all(_tl, dt=dt, traffic_speed=float(self.args.traffic_speed))

    def _render_tl_colors(self) -> None:
        """Push current traffic-light colors to the VTK glyph mesh (≤10 Hz)."""
        _tl        = self.scene_state.get("traffic_lights")
        _tl_m      = self.scene_state.get("_tl_mesh")
        _tl_glyphs = self.scene_state.get("_tl_glyphs")
        _tl_actor  = self.scene_state.get("tl_actor")
        if (_tl is None or _tl_m is None or _tl_m.n_points == 0
                or _tl_glyphs is None or _tl_actor is None):
            return
        update_light_mesh(_tl_m, _tl)
        try:
            n_lights    = _tl_m.n_points
            n_glyph_pts = _tl_glyphs.n_points
            if n_glyph_pts > 0 and n_glyph_pts % n_lights == 0:
                n_per      = n_glyph_pts // n_lights
                src_colors = np.asarray(_tl_m["colors"], dtype=np.uint8)
                _tl_glyphs["colors"] = np.repeat(src_colors, n_per, axis=0)
                _tl_glyphs.Modified()
        except Exception as _tl_exc:
            print(f"[tl] color update failed: {_tl_exc}")

    def _update_solo_panel(self, dt: float, tick: int) -> None:
        """Update the solo-mode solar/energy telemetry overlay."""
        if not (self.args.solo and len(np.asarray(self.car_anim["edge_idx"])) > 0):
            return
        _speed  = float(self.car_anim["speed"][0])
        _e_idx  = int(self.car_anim["edge_idx"][0])
        _shadow = float(self.edge_shadow_frac[_e_idx])
        _hour   = float(self.scene_state.get("hour", 12.0))
        _lat    = float(self.street_graph.graph.get("scene_lat", 51.5))
        _lon    = float(self.street_graph.graph.get("scene_lon", 0.0))
        import solar_physics as _sp_mod
        _sp = _sp_mod.SolarParams()
        _el_rad, _ = _sp_mod.sun_angles(lat_deg=_lat, lon_deg=_lon, hour_local=_hour)
        _ghi, _dni, _dhi = _sp_mod.clear_sky_ghi(_el_rad)
        _g_panel = _sp_mod.panel_irradiance(ghi=_ghi, dhi=_dhi, dni=_dni, sun_elevation_rad=_el_rad)
        _w_solar = (
            _g_panel * (1.0 - _shadow)
            * _sp.roof_area_m2 * _sp.panel_efficiency * _sp.temperature_derating
        )
        _w_mech = _sp_mod.mechanical_energy_joules(
            length_m=_speed * 1.0, speed_ms=_speed,
            vehicle_mass_kg=_sp.vehicle_mass_kg, rolling_coeff=_sp.rolling_coeff,
            drag_coeff=_sp.drag_coeff, frontal_area_m2=_sp.frontal_area_m2,
        ) / 3600.0
        self.scene_state["solo_solar_Wh"] = (
            float(self.scene_state.get("solo_solar_Wh", 0.0)) + _w_solar * dt / 3600.0
        )
        self.scene_state["solo_mech_Wh"] = (
            float(self.scene_state.get("solo_mech_Wh", 0.0)) + _w_mech * dt
        )
        if tick % 30 == 0:
            _spd_kmh    = _speed * 3.6
            _sol_wh     = self.scene_state["solo_solar_Wh"]
            _mech_wh    = self.scene_state["solo_mech_Wh"]
            _net_wh     = _sol_wh - _mech_wh
            _sign       = "▲" if _net_wh >= 0 else "▼"
            _shade_icon = "▨ shaded" if _shadow > 0.5 else "☀ open"
            _txt = (
                f"Speed:      {_spd_kmh:4.1f} km/h\n"
                f"Solar Pwr:  {_w_solar:4.1f} W\n"
                f"Mech Nrg:   {_mech_wh:4.1f} Wh\n"
                f"Solar Hrv:  {_sol_wh:4.1f} Wh\n"
                f"Net Nrg:    {abs(_net_wh):4.1f} Wh {_sign}\n"
                f"Shadow:     {_shade_icon}"
            )
            self.plotter.add_text(
                _txt, position=(0.35, 0.04), name="solo_telemetry",
                viewport=True, font_size=10, color="#f4f4f4",
            )

    def _render_cars(self, positions=None, headings=None) -> None:
        actors = self.scene_state.get("car_actors")
        if not isinstance(actors, dict):
            actors = {}
            self.scene_state["car_actors"] = actors

        is_first = actors.get("cars") is None and not self.scene_state.get("_ultra_car_actors")
        if is_first:
            print("[cars] _render_cars: FIRST CALL — creating actor")
            if self.args.car_detail == "ultra" and self.car_obj_templates and len(self.car_paths) > 0:
                parked_actors = []
                target_parked = int(getattr(self.args, "n_parked_cars", 80))
                if target_parked <= 0:
                    target_parked = 80
                target_parked = min(target_parked, 400)

                # Walk along local road edges, sample parking spots every 10 m, 3.5 m to the side
                _PARKING_HW = {"residential", "unclassified", "tertiary", "tertiary_link",
                               "service", "living_street"}
                _SPACING = 10.0
                _sample_pts: list[tuple[float, float, float, float]] = []
                for path in self.car_paths:
                    if str(path.get("highway", "")).lower() not in _PARKING_HW:
                        continue
                    pts = np.asarray(path["points"], dtype=float)
                    cum = np.asarray(path["cum_len"], dtype=float)
                    total = float(path["length"])
                    if total < _SPACING:
                        continue
                    d = 4.0
                    while d < total - 4.0:
                        si = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
                        t = (d - float(cum[si])) / max(1e-9, float(cum[si + 1]) - float(cum[si]))
                        pos = pts[si] + (pts[si + 1] - pts[si]) * t
                        dx, dy = float(pts[si + 1][0] - pts[si][0]), float(pts[si + 1][1] - pts[si][1])
                        nd = float(np.sqrt(dx * dx + dy * dy))
                        if nd > 1e-6:
                            _nx, _ny = -dy / nd, dx / nd  # right-hand perpendicular
                            _sample_pts.append((float(pos[0]) + _nx * 3.5, float(pos[1]) + _ny * 3.5,
                                                float(pos[2]), float(np.degrees(np.arctan2(dy, dx)))))
                        d += _SPACING

                if _sample_pts:
                    try:
                        from scipy.spatial import cKDTree
                        _inter_pts = []
                        for _n, _nd in self.street_graph.nodes(data=True):
                            if self.street_graph.degree(_n) >= 3 and "x" in _nd and "y" in _nd:
                                _inter_pts.append([float(_nd["x"]), float(_nd["y"])])
                        _inter_tree = cKDTree(np.array(_inter_pts)) if _inter_pts else None
                        _INTER_CLEARANCE = 20.0

                        _placed = 0
                        _p_pos, _p_hdg, _p_mid, _p_col = [], [], [], []
                        for _pi in self.car_rng.permutation(len(_sample_pts)):
                            if _placed >= target_parked:
                                break
                            px_off, py_off, rz, heading = _sample_pts[_pi]
                            if _inter_tree is not None:
                                _d_inter, _ = _inter_tree.query([px_off, py_off])
                                if _d_inter < _INTER_CLEARANCE:
                                    continue
                            mid = int(self.car_rng.integers(0, len(self.car_obj_templates)))
                            _p_pos.append((px_off, py_off, rz))
                            _p_hdg.append(heading)
                            _p_mid.append(mid)
                            _p_col.append(self._CAR_BODY_COLORS[mid % len(self._CAR_BODY_COLORS)])
                            _placed += 1
                        if _placed:
                            # One instanced draw per car model (render.instanced_cars)
                            from render.instanced_cars import InstancedFleet
                            self.scene_state["_parked_fleet"] = InstancedFleet(
                                self.plotter, self.car_obj_templates, _p_mid,
                                [pv.Color(c).int_rgb for c in _p_col], _p_pos, _p_hdg)
                        self.scene_state["_parked_car_actors"] = parked_actors
                        print(f"[cars] created {_placed} parked cars (instanced)")
                    except Exception as e:
                        print(f"[cars] Failed to create parked cars: {e}")

        if not bool(self.car_anim["enabled"]):
            for name, actor in list(actors.items()):
                if actor is not None:
                    self.plotter.remove_actor(actor, reset_camera=False)
                actors.pop(name, None)
            self.scene_state.pop("_car_mesh", None)
            for _ua in self.scene_state.get("_ultra_car_actors") or []:
                try:
                    self.plotter.remove_actor(_ua, reset_camera=False)
                except Exception:
                    pass
            self.scene_state["_ultra_car_actors"] = None

            for _pa in self.scene_state.get("_parked_car_actors") or []:
                try:
                    self.plotter.remove_actor(_pa, reset_camera=False)
                except Exception:
                    pass
            self.scene_state["_parked_car_actors"] = None
            for _fk in ("_ultra_fleet", "_parked_fleet"):
                _fl = self.scene_state.pop(_fk, None)
                if _fl is not None:
                    _fl.remove()

            return

        if positions is None:
            positions = self._sample_car_positions()
        if positions.shape[0] == 0:
            return
        self.car_anim["pos"] = positions
        try:
            self._update_selected_car_marker()
        except NameError:
            pass

        # ── Ultra mode: per-car OBJ mesh actors ─────────────────────────
        if self.args.car_detail == "ultra" and self.car_obj_templates:
            headings = self._sample_car_headings()
            fleet = self.scene_state.get("_ultra_fleet")
            if fleet is None or fleet.n != positions.shape[0]:
                if fleet is not None:
                    fleet.remove()
                # One instanced draw per car model instead of one actor per car
                from render.instanced_cars import InstancedFleet
                n = positions.shape[0]
                model_idx = np.asarray(self.car_anim.get("model_idx", np.zeros(n, dtype=np.int64)), dtype=np.int64)
                model_idx = (model_idx[:n] if model_idx.shape[0] >= n
                             else np.resize(model_idx, n)) % len(self.car_obj_templates)
                colours = []
                for i in range(n):
                    if self.car_solar_model_idx is not None and int(model_idx[i]) == self.car_solar_model_idx:
                        colours.append(pv.Color(self._SOLAR_CAR_COLOR).int_rgb)
                    else:
                        colours.append(pv.Color(self._CAR_BODY_COLORS[i % len(self._CAR_BODY_COLORS)]).int_rgb)
                fleet = InstancedFleet(self.plotter, self.car_obj_templates, model_idx, colours,
                                       positions, headings)
                self.scene_state["_ultra_fleet"] = fleet
                self.scene_state["_ultra_car_actors"] = fleet.actors
                print(f"[cars] created {n} OBJ cars as {len(fleet.actors)} instanced draw(s)")
            else:
                fleet.update(positions, headings)
            fleet.set_visible(bool(self.scene_state["show_cars"]))
        else:
            # ── Standard/low-detail: GlyphInstances with oriented car-box ────────
            from glyph_instance import GlyphInstances as _GlyphInstances
            _gi = actors.get("_car_glyph")
            if _gi is None:
                _car_tmpl = pv.Box(bounds=(-2.25, 2.25, -0.9, 0.9, 0.0, 1.45))
                _gi = _GlyphInstances(_car_tmpl, max(1, positions.shape[0]),
                                      str(self._style()["car"]), self.plotter)
                actors["_car_glyph"] = _gi
                actors["cars"] = _gi          # sentinel: keeps is_first False
                self.scene_state.pop("_car_mesh", None)
                print("[cars] _render_cars: GlyphInstances created (standard-detail mode)")
            _hdgs = headings if headings is not None else self._sample_car_headings()
            _gi.update(positions, _hdgs)
            _gi.set_visible(bool(self.scene_state["show_cars"]))

    # Patch: Only update color array, not geometry, every tick
    def _update_traffic_light_colors(self):
        update_light_mesh(self._tl_mesh, self.traffic_lights_dict)

    def _timer_callback(self):
            """Animation timer callback: updates traffic light colors each frame."""
            update_colors = self.scene_state.get("_update_traffic_light_colors")
            if update_colors is not None:
                update_colors()
            # Insert additional animation logic here if needed.

    def _clear_selected_car(self, clear_plan: bool = False) -> None:
            idx = self.route_state.get("selected_car_idx")
            if clear_plan and idx is not None:
                try:
                    plans = self.car_anim.get("planned_edges")
                    if isinstance(plans, list) and 0 <= int(idx) < len(plans):
                        plans[int(idx)] = None
                    cursors = self.car_anim.get("planned_cursor")
                    if isinstance(cursors, np.ndarray) and 0 <= int(idx) < cursors.shape[0]:
                        cursors[int(idx)] = 0
                except Exception:
                    pass
            self.route_state["selected_car_idx"] = None
            marker = self.route_state.get("selected_marker_actor")
            if marker is not None:
                try:
                    self.plotter.remove_actor(marker, reset_camera=False)
                except Exception:
                    pass
            self.route_state["selected_marker_actor"] = None
            self.plotter.add_text("", position=(0.36, 0.93), name="selected_car_overlay", viewport=True)

    def _update_selected_car_marker(self) -> None:
            idx = self.route_state.get("selected_car_idx")
            if idx is None:
                return
            try:
                i = int(idx)
                positions = np.asarray(self.car_anim.get("pos", []), dtype=float)
                if positions.ndim != 2 or i < 0 or i >= positions.shape[0]:
                    return
                pos = positions[i]
                marker = self.route_state.get("selected_marker_actor")
                if marker is None:
                    marker = self.plotter.add_mesh(
                        pv.Sphere(radius=3.2, center=(0.0, 0.0, 0.0), theta_resolution=24, phi_resolution=12),
                        color="#ffd400",
                        opacity=0.35,
                        render=False,
                        reset_camera=False,
                    )
                    self.route_state["selected_marker_actor"] = marker
                    marker.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]) + 1.8)
                else:
                    marker.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]) + 1.8)
            except Exception:
                pass

    def _select_car_for_route(self, car_idx: int, source_node: object) -> None:
            self._clear_selected_car(clear_plan=False)
            self.route_state["selected_car_idx"] = int(car_idx)
            self.plotter.add_text(
                f"Selected car #{int(car_idx)} - click target road node",
                position=(0.36, 0.93),
                name="selected_car_overlay",
                font_size=10,
                color="#ffd400",
                viewport=True,
            )
            self._update_selected_car_marker()

    def _route_nodes_to_car_edges(self, _nodes: list[object]) -> list[int]:
            if len(_nodes) < 2:
                return []
            by_uv: dict[tuple[object, object], list[int]] = {}
            for _i, _path in enumerate(self.car_paths):
                by_uv.setdefault((_path.get("u"), _path.get("v")), []).append(_i)
            edges: list[int] = []
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                matches = by_uv.get((_a, _b), [])
                if not matches:
                    return []
                edges.append(min(matches, key=lambda _idx: float(self.car_paths[int(_idx)].get("length", np.inf))))
            return edges

    def _assign_selected_car_route(self, _nodes: list[object]) -> None:
            idx = self.route_state.get("selected_car_idx")
            if idx is None or len(_nodes) < 2:
                return
            try:
                car_idx = int(idx)
                edges = self._route_nodes_to_car_edges(list(_nodes))
                if not edges:
                    print("[route] Selected car route could not be mapped to drivable edges")
                    return
                plans = self.car_anim.get("planned_edges")
                cursors = self.car_anim.get("planned_cursor")
                if not isinstance(plans, list) or car_idx < 0 or car_idx >= len(plans):
                    return
                if not isinstance(cursors, np.ndarray) or car_idx >= cursors.shape[0]:
                    return
                current_edge = int(edges[0])
                self.car_anim["edge_idx"][car_idx] = current_edge
                self.car_anim["dist"][car_idx] = min(
                    float(self.car_anim["dist"][car_idx]),
                    max(0.0, float(self.car_paths[current_edge]["length"]) - 1e-6),
                )
                base_speed = float(self.car_paths[current_edge]["maxspeed_ms"]) * 0.85
                if "desired_speed_base" in self.car_anim:
                    self.car_anim["desired_speed_base"][car_idx] = base_speed
                if "desired_speed" in self.car_anim:
                    self.car_anim["desired_speed"][car_idx] = base_speed * float(self.args.traffic_speed)
                if "speed" in self.car_anim:
                    self.car_anim["speed"][car_idx] = min(
                        float(self.car_anim["speed"][car_idx]),
                        base_speed * float(self.args.traffic_speed),
                    )
                plans[car_idx] = np.asarray(edges, dtype=np.int64)
                cursors[car_idx] = 0
                self.route_state["selected_route_nodes"] = list(_nodes)
                self.plotter.add_text(
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

    def _animate_cars(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        self._car_tick = getattr(self, "_car_tick", 0) + 1
        tick = self._car_tick
        try:
            self.scene_state["_car_tick_active"] = True
            now = time.perf_counter()
            last_t = float(self.car_anim.get("last_t", now))
            dt = float(np.clip(now - last_t, 0.0, 0.10))
            self.car_anim["last_t"] = now
            if dt <= 0.0:
                self.scene_state["_car_tick_active"] = False
                return
            if tick == 1:
                _spd = np.asarray(self.car_anim.get("speed", []), dtype=float)
                _eidx = np.asarray(self.car_anim.get("edge_idx", []), dtype=np.int64)
                _unique_edges = len(np.unique(_eidx))
                print(f"[cars] tick-1: n={len(_spd)} cars, {_unique_edges} unique edges, "
                      f"speed mean={float(np.mean(_spd)):.2f} max={float(np.max(_spd) if len(_spd) else 0):.2f} m/s")
            self._advance_cars(dt)

            # Keep pos array fresh for the unified picker proximity test
            self.car_anim["pos"] = self._sample_car_positions()

            # Update analysis overlays every 8 ticks (~130 ms at 60 fps)
            if tick % 8 == 0 and hasattr(self, "_update_analysis"):
                try:
                    self._update_analysis()
                except Exception as _ana_exc:
                    pass

            self._update_solo_panel(dt, tick)
            self._render_cars()
            self._physics_tl_fsm(dt)
            self._tl_tick = getattr(self, "_tl_tick", 0) + 1
            if self._tl_tick % 15 == 0:
                self._render_tl_colors()
            # ─────────────────────────────────────────────────────────
            if bool(self.args.debug_cars):
                self.car_debug_state["tick"] = int(self.car_debug_state.get("tick", 0)) + 1
                next_log_t = float(self.car_debug_state.get("next_log_t", now + 1.5))
                if now >= next_log_t:
                    self._log_car_debug(f"tick={int(self.car_debug_state['tick'])}")
                    self._log_idm_diagnostics(f"tick={int(self.car_debug_state['tick'])}")
                    self.car_debug_state["next_log_t"] = now + 2.5
            if tick % self._car_render_stride == 0:
                try:
                    self.plotter.update()
                except Exception as _render_exc:
                    print(f"[cars] plotter.render failed: {_render_exc}")
            self.scene_state["_car_tick_active"] = False
        except Exception as _exc:
            self.scene_state["_car_tick_active"] = False
            import traceback as _tb
            print(f"[cars-timer ERROR] {_exc}")
            _tb.print_exc()

    # Define callbacks first
    def _mark_interactive_ready(self, _: int) -> None:
        try:
            print("[viewer] interactive callbacks enabled")
            self.scene_state["interactive_ready"] = True
            # Kill VTK's built-in CharEvent bindings.  They fire IN ADDITION to
            # the app's add_key_event handlers: '3' toggles anaglyph stereo
            # (the whole screen goes wobbly magenta), 'w' switches everything
            # to wireframe, 's' back to surface, 'f' flies the camera, 'e'
            # exits.  The app's own keys use KeyPressEvent and keep working.
            try:
                self.plotter.iren.interactor.RemoveObservers("CharEvent")
                print("[viewer] VTK default char-key bindings disabled "
                      "(stereo '3', wireframe 'w', ...)")
            except Exception as _char_exc:
                print(f"[viewer] could not disable VTK char bindings: {_char_exc}")
            # pyvista hard-registers 'q' → close-window ("Add no matter what").
            # Clear it so a stray q can't kill the session mid-demo.
            try:
                self.plotter.iren.clear_events_for_key("q")
                print("[viewer] pyvista 'q'=close binding cleared")
            except Exception:
                pass
            if bool(self.args.debug_cars):
                print("[viewer-debug] interactive callbacks enabled")
        except Exception as _exc:
            print(f"[ready-timer ERROR] {_exc}")
