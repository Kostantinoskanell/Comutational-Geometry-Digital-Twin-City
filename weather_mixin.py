"""WeatherMixin — rain / snow particle simulation + wet-road shader.

Key binding registered lazily on first live animation tick:
  'w'  →  cycle  clear → rain → snow → clear

Rain:  1800 falling line-segment particles, wet-road specular darkening,
       car desired_speed reduced to 75 % of base.
Snow:  1200 falling point particles, roof accumulation (static glyph layer),
       car desired_speed reduced to 55 % of base.
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv


class WeatherMixin:

    def _init_weather(self) -> None:
        self.weather: dict = {
            "mode": "clear",         # "clear" | "rain" | "snow"
            "rain_pd": None,         # pv.PolyData updated in-place
            "rain_actor": None,
            "snow_pd": None,
            "snow_actor": None,
            "roof_actor": None,
            "rain_pts": None,        # np.ndarray (N, 3)
            "snow_pts": None,
            "snow_depth": 0.0,
            "wet_factor": 0.0,
            "_last_t": 0.0,
            "_key_registered": False,
            "_car_base_speed": None,
            "rain_scale": 1.0,       # 0..1 share of drops shown (hyetograph-driven during the flood replay)
            "_rain_k": None,
        }

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _toggle_weather(self) -> None:
        MODES = ["clear", "rain", "snow"]
        cur = self.weather["mode"]
        nxt = MODES[(MODES.index(cur) + 1) % len(MODES)]
        print(f"[weather] {cur} → {nxt}")
        self._clear_weather_actors()
        if nxt == "clear":
            self._restore_car_speeds()
            self.weather["wet_factor"] = 0.0
            self.weather["snow_depth"] = 0.0
            self.weather["rain_pts"] = None
            self.weather["snow_pts"] = None
            self._reset_wet_road()
        self.weather["mode"] = nxt

    def _clear_weather_actors(self) -> None:
        for k in ("rain_actor", "snow_actor", "roof_actor"):
            a = self.weather.get(k)
            if a is not None:
                try:
                    a.VisibilityOff()
                except Exception:
                    pass
                self.weather[k] = None
        self.weather["rain_pd"] = None
        self.weather["snow_pd"] = None

    # ------------------------------------------------------------------
    # Car speed helpers
    # ------------------------------------------------------------------

    def _restore_car_speeds(self) -> None:
        # SUMO vehicles back to normal speed (bypass the 1 Hz throttle)
        self.weather["_sumo_factor_t"] = 0.0
        self._apply_weather_sumo_factor(1.0)
        saved = self.weather.get("_car_base_speed")
        if saved is None:
            return
        if not hasattr(self, "car_anim") or not self.car_anim.get("enabled"):
            return
        desired = self.car_anim.get("desired_speed")
        if desired is not None and len(desired) == len(saved):
            self.car_anim["desired_speed"] = saved.copy()
        self.weather["_car_base_speed"] = None

    def _apply_weather_car_factor(self, factor: float) -> None:
        if not hasattr(self, "car_anim") or not self.car_anim.get("enabled"):
            return
        desired = self.car_anim.get("desired_speed")
        if desired is None or len(desired) == 0:
            return
        base_in_anim = self.car_anim.get("desired_speed_base")
        reference = base_in_anim if base_in_anim is not None else desired
        saved = self.weather.get("_car_base_speed")
        if saved is None or len(saved) != len(desired):
            self.weather["_car_base_speed"] = np.asarray(reference, dtype=float).copy()
            saved = self.weather["_car_base_speed"]
        traffic_speed = float(getattr(self.args, "traffic_speed", 1.0))
        self.car_anim["desired_speed"] = np.asarray(saved, dtype=float) * factor * traffic_speed
        self._apply_weather_sumo_factor(factor)

    def _apply_weather_sumo_factor(self, factor: float) -> None:
        """Push the weather slowdown to SUMO vehicles too (they ignored
        weather entirely before — in --engine sumo mode nothing slowed down).

        setSpeedFactor scales each vehicle's speed relative to road limits.
        Re-applied at 1 Hz so newly departed vehicles get it as well.
        """
        rt = getattr(self, "sumo", None)
        if not rt or not rt.get("enabled"):
            return
        conn = rt.get("conn")
        if conn is None or not getattr(conn, "ready", False):
            return
        now = time.perf_counter()
        if now - float(self.weather.get("_sumo_factor_t", 0.0)) < 1.0:
            return
        self.weather["_sumo_factor_t"] = now
        try:
            tr = conn._traci
            f = max(0.1, float(factor))
            for vid in tr.vehicle.getIDList():
                tr.vehicle.setSpeedFactor(vid, f)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Road wetness
    # ------------------------------------------------------------------

    def _apply_wet_road(self, wf: float) -> None:
        road_actor = self.scene_state.get("vehicle_actor")
        if road_actor is None:
            return
        try:
            r = 0.20 - 0.06 * wf
            g = 0.20 - 0.05 * wf
            b = 0.20 + 0.01 * wf
            road_actor.GetProperty().SetColor(r, g, b)
            road_actor.GetProperty().SetSpecular(0.45 * wf)
            road_actor.GetProperty().SetSpecularPower(60)
            road_actor.GetProperty().SetRoughness(0.9 - 0.65 * wf)     # PBR: wet asphalt is glossy
        except Exception:
            pass
        if hasattr(self, "_set_wet_film"):
            self._set_wet_film(wf)

    def _reset_wet_road(self) -> None:
        road_actor = self.scene_state.get("vehicle_actor")
        if road_actor is None:
            return
        try:
            style = self._style()
            rgb = pv.Color(str(style["vehicle"])).float_rgb
            road_actor.GetProperty().SetColor(*rgb)
            road_actor.GetProperty().SetSpecular(0.0)
            road_actor.GetProperty().SetRoughness(0.9)
        except Exception:
            pass
        if hasattr(self, "_set_wet_film"):
            self._set_wet_film(0.0)

    # ------------------------------------------------------------------
    # Rain tick
    # ------------------------------------------------------------------

    def _rain_volume(self) -> tuple[np.ndarray, float, float, float]:
        """(centre xy, half-size, ground z, top z) of the box the rain falls in.

        Street-level / perspective cameras: a box in front of the camera sized to the view
        distance. Parallel (isometric overview) or far cameras: the camera sits kilometres
        away along the view axis, so the box is built around the FOCAL POINT instead (a column
        ~half the visible width wide and ~120 m tall), which is what the screen shows."""
        cam = self.plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        fp = np.asarray(cam.focal_point, dtype=float)
        parallel = bool(cam.GetParallelProjection())
        dist = float(np.linalg.norm(fp - pos))
        if parallel or dist > 600.0:
            scale = float(cam.GetParallelScale()) if parallel else dist * 0.5
            R = float(np.clip(1.15 * scale, 80.0, 800.0))
            ground = float(fp[2]) - 10.0
            return fp[:2].copy(), R, ground, ground + 130.0
        v = fp - pos
        n = float(np.linalg.norm(v[:2]))
        d_xy = v[:2] / n if n > 1e-6 else np.array([0.0, 1.0])
        R = float(np.clip(0.42 * dist, 35.0, 420.0))
        centre = pos[:2] + d_xy * min(0.6 * R, 0.5 * dist)
        ground = float(min(fp[2], pos[2])) - 5.0
        top = float(max(pos[2], fp[2])) + float(np.clip(0.5 * R, 30.0, 160.0))
        return centre, R, ground, top

    def _tick_rain(self, dt: float) -> None:
        N = 14000
        c, R, g0, top = self._rain_volume()
        if self.weather["rain_pts"] is None or len(self.weather["rain_pts"]) != N:
            rng = np.random.default_rng(7)
            self.weather["rain_pts"] = np.column_stack([
                rng.uniform(-R, R, N) + c[0], rng.uniform(-R, R, N) + c[1], rng.uniform(g0, top, N)])

        pts: np.ndarray = self.weather["rain_pts"]
        pts[:, 2] -= 11.0 * dt                        # m/s: visual terminal speed (keeps streaks legible at 12 fps)
        pts[:, 0] += 1.8 * dt                         # light wind slant
        # wrap inside the camera-centred box: drops keep falling as the view moves
        pts[:, 0] = c[0] + (pts[:, 0] - c[0] + R) % (2 * R) - R
        pts[:, 1] = c[1] + (pts[:, 1] - c[1] + R) % (2 * R) - R
        span = max(top - g0, 1.0)
        pts[:, 2] = g0 + (pts[:, 2] - g0) % span      # re-enter at the top of the box

        self.weather["rain_pts"] = pts
        if self.weather.get("wet_override") is not None:      # rain that has actually fallen
            self.weather["wet_factor"] = float(self.weather["wet_override"])
        else:
            self.weather["wet_factor"] = min(1.0, float(self.weather["wet_factor"]) + dt * 0.5)
        self._apply_wet_road(float(self.weather["wet_factor"]))
        self._apply_weather_car_factor(0.60)   # was 0.75 — too subtle to notice

        # Each drop is a streak; its length grows with the view size so it reads at any zoom.
        L = float(np.clip(R * 0.05, 2.0, 22.0))
        ends = pts.copy()
        ends[:, 2] -= L
        ends[:, 0] -= 0.18 * L
        line_pts = np.empty((N * 2, 3), dtype=float)
        line_pts[0::2] = pts
        line_pts[1::2] = ends

        # Streak density follows the rain intensity (flood replay drives
        # rain_scale from the storm hyetograph; 1.0 = the manual Weather mode).
        k = int(round(N * float(np.clip(self.weather.get("rain_scale", 1.0), 0.0, 1.0))))
        k = max(k, 1) if self.weather.get("rain_scale", 1.0) > 0.0 else 0

        def _conn(k):
            conn = np.empty(k * 3, dtype=np.int64)
            conn[0::3] = 2
            conn[1::3] = np.arange(0, k * 2, 2, dtype=np.int64)
            conn[2::3] = np.arange(1, k * 2, 2, dtype=np.int64)
            return conn

        if self.weather["rain_pd"] is not None and self.weather.get("_rain_k") != k:
            self.weather["rain_pd"].lines = _conn(k)
            self.weather["_rain_k"] = k
        if self.weather["rain_pd"] is None or self.weather["rain_actor"] is None:
            conn = _conn(k)
            self.weather["_rain_k"] = k
            rain_pd = pv.PolyData()
            rain_pd.points = line_pts
            rain_pd.lines  = conn
            self.weather["rain_pd"] = rain_pd
            self.weather["rain_actor"] = self.plotter.add_mesh(
                rain_pd,
                color="#b8d4f8",
                opacity=0.65,
                lighting=False,
                line_width=2.2,
                name="weather_rain",
                reset_camera=False,
            )
            self.weather["rain_actor"].PickableOff()
        else:
            self.weather["rain_pd"].points = line_pts

    # ------------------------------------------------------------------
    # Snow tick
    # ------------------------------------------------------------------

    def _tick_snow(self, dt: float) -> None:
        bx0, bx1, by0, by1, _, bz1 = self._weather_bounds
        N = 1200

        if self.weather["snow_pts"] is None:
            rng = np.random.default_rng(13)
            self.weather["snow_pts"] = np.column_stack([
                rng.uniform(bx0, bx1, N).astype(float),
                rng.uniform(by0, by1, N).astype(float),
                rng.uniform(3.0, bz1, N).astype(float),
            ])

        pts: np.ndarray = self.weather["snow_pts"]
        pts[:, 2] -= 2.0 * dt
        rng2 = np.random.default_rng(int(time.perf_counter() * 1e4) & 0xFFFF)
        drift = rng2.uniform(-0.4, 0.4, (N, 2)).astype(float) * dt
        pts[:, 0] = np.clip(pts[:, 0] + drift[:, 0], bx0, bx1)
        pts[:, 1] = np.clip(pts[:, 1] + drift[:, 1], by0, by1)

        below = pts[:, 2] < 0.0
        n_below = int(np.sum(below))
        if n_below:
            rng3 = np.random.default_rng(int(time.perf_counter() * 1e6) & 0xFFFFFF)
            pts[below, 0] = rng3.uniform(bx0, bx1, n_below)
            pts[below, 1] = rng3.uniform(by0, by1, n_below)
            pts[below, 2] = rng3.uniform(bz1 * 0.4, bz1, n_below)

        self.weather["snow_pts"] = pts
        self.weather["snow_depth"] = min(0.5, float(self.weather["snow_depth"]) + dt * 0.003)
        self._apply_weather_car_factor(0.30)   # was 0.55 — crawl speed, clearly visible

        if self.weather["snow_pd"] is None or self.weather["snow_actor"] is None:
            snow_pd = pv.PolyData(pts.copy())
            self.weather["snow_pd"] = snow_pd
            self.weather["snow_actor"] = self.plotter.add_mesh(
                snow_pd,
                style="points",
                point_size=5,
                render_points_as_spheres=True,
                color="#e8f4ff",
                opacity=0.88,
                lighting=False,
                name="weather_snow",
            )
        else:
            self.weather["snow_pd"].points = pts.copy()

        self._update_roof_snow(float(self.weather["snow_depth"]))

    # ------------------------------------------------------------------
    # Roof snow
    # ------------------------------------------------------------------

    def _update_roof_snow(self, depth: float) -> None:
        if depth < 0.04:
            return
        if self.weather.get("roof_actor") is not None:
            return  # static once placed
        if self.buildings_mesh.n_points == 0:
            return
        try:
            b_pts = self.buildings_mesh.points
            z_max = float(b_pts[:, 2].max())
            z_rng = z_max - float(b_pts[:, 2].min())
            if z_rng < 0.5:
                return
            thresh = z_max - max(1.5, z_rng * 0.08)
            top_pts = b_pts[b_pts[:, 2] >= thresh].copy()
            if len(top_pts) < 10:
                return
            if len(top_pts) > 2500:
                idx = np.random.default_rng(42).choice(len(top_pts), 2500, replace=False)
                top_pts = top_pts[idx]
            top_pts[:, 2] += depth * 0.5
            snow_pd  = pv.PolyData(top_pts)
            snow_disc = pv.Disc(inner=0.0, outer=2.0, r_res=1, c_res=6)
            glyphs   = snow_pd.glyph(geom=snow_disc, orient=False, scale=False)
            self.weather["roof_actor"] = self.plotter.add_mesh(
                glyphs,
                color="#e8f4ff",
                opacity=0.92,
                smooth_shading=False,
                lighting=True,
                name="weather_roof_snow",
            )
        except Exception as exc:
            print(f"[weather] roof snow failed: {exc}")

    # ------------------------------------------------------------------
    # Main animation callback
    # ------------------------------------------------------------------

    def _animate_weather(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return

        # Weather is toggled via its panel checkbox — no keyboard shortcut.
        # ('w' is reserved for WASD free-walk movement; a key registration
        #  here previously double-fired with main_ast6's, skipping a state.)

        mode = self.weather["mode"]
        if mode == "clear":
            return

        now = time.perf_counter()
        last = float(self.weather.get("_last_t", now))
        dt = min(float(now - last), 0.15)
        self.weather["_last_t"] = float(now)

        if not hasattr(self, "_weather_bounds"):
            if self.buildings_mesh.n_points > 0:
                b = self.buildings_mesh.bounds
                pad = 60.0
                self._weather_bounds = (
                    float(b[0]) - pad, float(b[1]) + pad,
                    float(b[2]) - pad, float(b[3]) + pad,
                    0.0, 80.0,
                )
            else:
                return

        try:
            if mode == "rain":
                self._tick_rain(dt)
            elif mode == "snow":
                self._tick_snow(dt)
        except Exception as exc:
            print(f"[weather] tick error: {exc}")

        # HUD label (the Flood Lab run drives rain itself and has its own HUD)
        labels = {"clear": "", "rain": "🌧 Rain  (slowing traffic)", "snow": "❄ Snow  (slowing traffic)"}
        if self.weather.get("wet_override") is not None:
            labels["rain"] = ""
        try:
            self.plotter.add_text(
                labels.get(mode, ""),
                position=(0.36, 0.965),
                name="weather_hud",
                font_size=11,
                viewport=True,
                color="#d0e8ff" if mode == "rain" else "#e8f4ff",
            )
        except Exception:
            pass
