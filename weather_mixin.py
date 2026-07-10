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
        except Exception:
            pass

    def _reset_wet_road(self) -> None:
        road_actor = self.scene_state.get("vehicle_actor")
        if road_actor is None:
            return
        try:
            style = self._style()
            rgb = pv.Color(str(style["vehicle"])).float_rgb
            road_actor.GetProperty().SetColor(*rgb)
            road_actor.GetProperty().SetSpecular(0.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rain tick
    # ------------------------------------------------------------------

    def _tick_rain(self, dt: float) -> None:
        bx0, bx1, by0, by1, _, bz1 = self._weather_bounds
        N = 1800

        if self.weather["rain_pts"] is None:
            rng = np.random.default_rng(7)
            self.weather["rain_pts"] = np.column_stack([
                rng.uniform(bx0, bx1, N).astype(float),
                rng.uniform(by0, by1, N).astype(float),
                rng.uniform(3.0, bz1, N).astype(float),
            ])

        pts: np.ndarray = self.weather["rain_pts"]
        pts[:, 2] -= 18.0 * dt

        below = pts[:, 2] < 0.0
        n_below = int(np.sum(below))
        if n_below:
            rng2 = np.random.default_rng(int(time.perf_counter() * 1e6) & 0xFFFFFF)
            pts[below, 0] = rng2.uniform(bx0, bx1, n_below)
            pts[below, 1] = rng2.uniform(by0, by1, n_below)
            pts[below, 2] = rng2.uniform(bz1 * 0.5, bz1, n_below)

        self.weather["rain_pts"] = pts
        self.weather["wet_factor"] = min(1.0, float(self.weather["wet_factor"]) + dt * 0.5)
        self._apply_wet_road(float(self.weather["wet_factor"]))
        self._apply_weather_car_factor(0.60)   # was 0.75 — too subtle to notice

        # Build/update line-segment mesh (each drop = 2m vertical streak)
        ends = pts.copy()
        ends[:, 2] -= 2.0
        line_pts = np.empty((N * 2, 3), dtype=float)
        line_pts[0::2] = pts
        line_pts[1::2] = ends

        if self.weather["rain_pd"] is None or self.weather["rain_actor"] is None:
            conn = np.empty(N * 3, dtype=np.int64)
            conn[0::3] = 2
            conn[1::3] = np.arange(0, N * 2, 2, dtype=np.int64)
            conn[2::3] = np.arange(1, N * 2, 2, dtype=np.int64)
            rain_pd = pv.PolyData()
            rain_pd.points = line_pts
            rain_pd.lines  = conn
            self.weather["rain_pd"] = rain_pd
            self.weather["rain_actor"] = self.plotter.add_mesh(
                rain_pd,
                color="#a0c8f8",
                opacity=0.55,
                lighting=False,
                line_width=1.5,
                name="weather_rain",
            )
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

        # HUD label
        labels = {"clear": "", "rain": "🌧 Rain  (slowing traffic)", "snow": "❄ Snow  (slowing traffic)"}
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
