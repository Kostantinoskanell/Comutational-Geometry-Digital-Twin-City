"""TODMixin — real-time day/night cycle time-lapse.

Key binding registered lazily on first live animation tick:
  't'  →  toggle play / pause

When playing, simulated time advances at 2 h / real-second (full 24-h cycle
in ~12 real seconds).  Each tick applies immediate visual changes (sky colour,
sun position, scene lights, building window-glow ambient) and queues a shadow
recomputation once per simulated hour.

A HUD clock is displayed at the bottom-centre of the viewport.
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv


class TODMixin:

    def _init_tod(self) -> None:
        self.tod: dict = {
            "playing": False,
            "hour": float(self.scene_state.get("hour", 12.0)),
            "speed": 2.0,           # simulated hours per real second
            "_last_t": 0.0,
            "_last_shadow_h": -999.0,
            "_key_registered": False,
        }

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def _toggle_tod(self) -> None:
        playing = not bool(self.tod.get("playing"))
        self.tod["playing"] = playing
        if playing:
            self.tod["hour"] = float(self.scene_state.get("hour", 12.0))
            self.tod["_last_t"] = 0.0
            self.tod["_last_shadow_h"] = float(self.tod["hour"]) - 999.0
            print(f"[tod] time-lapse started at {self.tod['hour']:.1f}h  (press 't' to pause)")
        else:
            print(f"[tod] paused at {self.tod['hour']:.1f}h")

    # ------------------------------------------------------------------
    # Main animation callback
    # ------------------------------------------------------------------

    def _animate_tod(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return

        if not self.tod.get("_key_registered"):
            try:
                self.plotter.add_key_event("t", lambda: self._toggle_tod())
                self.tod["_key_registered"] = True
                print("[tod] 't' = toggle time-lapse play/pause")
            except Exception:
                pass

        if not bool(self.tod.get("playing")):
            return

        now = time.perf_counter()
        last = float(self.tod.get("_last_t", 0.0))
        if last <= 0.0:
            self.tod["_last_t"] = float(now)
            return
        dt = min(float(now - last), 0.2)
        self.tod["_last_t"] = float(now)

        # Advance simulated time
        hour = (float(self.tod["hour"]) + dt * float(self.tod["speed"])) % 24.0
        self.tod["hour"] = float(hour)
        self.scene_state["hour"] = float(hour)

        # Fast visual update every tick
        try:
            from app_core import _sun_dir_from_hour, _apply_atmosphere
            sun_dir  = _sun_dir_from_hour(float(hour))
            is_night = bool(float(sun_dir[2]) <= 0.0)
            self.scene_state["is_night"] = is_night
            style    = self._style()
            self._apply_tod_visuals(float(hour), sun_dir, is_night, style, _apply_atmosphere)
        except Exception as exc:
            print(f"[tod] visual update error: {exc}")
            return

        # Shadow recomputation once per simulated hour
        last_sh = float(self.tod.get("_last_shadow_h", -999.0))
        delta = float(hour) - last_sh
        if delta < 0.0:
            delta += 24.0
        if delta >= 1.0:
            self.tod["_last_shadow_h"] = float(hour)
            try:
                self._schedule_shadow_job(
                    float(hour),
                    float(self.scene_state.get("spot_radius", self.args.light_radius)),
                    is_night,
                    sun_dir,
                )
            except Exception:
                pass

        # HUD clock
        h_int = int(hour)
        m_int = int((float(hour) - h_int) * 60)
        clk_color = "#ffdd88" if is_night else "#333333"
        try:
            self.plotter.add_text(
                f"⏱ {h_int:02d}:{m_int:02d}",
                position=(0.44, 0.02),
                name="tod_clock",
                font_size=13,
                viewport=True,
                color=clk_color,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fast visual update (no shadow, no car re-render)
    # ------------------------------------------------------------------

    def _apply_tod_visuals(
        self,
        hour: float,
        sun_dir,
        is_night: bool,
        style: dict,
        _apply_atmosphere_fn,
    ) -> None:
        """Apply sky/sun/lights/window updates immediately without heavy shadow computation."""

        # Sky
        try:
            from app_core import _load_hdri
            hdri_tex = _load_hdri(float(hour), str(self.scene_state.get("preset", "mini")))
            self.plotter.set_environment_texture(hdri_tex)
            self.plotter.renderer.UseImageBasedLightingOn()
        except Exception:
            if not self._apply_sky_environment(float(hour), sun_dir):
                bg = style.get("day_bg" if not is_night else "night_bg")
                if isinstance(bg, list) and len(bg) == 2:
                    self.plotter.set_background(str(bg[0]), top=str(bg[1]))
                else:
                    self.plotter.set_background(str(bg or "#87ceeb"))

        # Sun sphere
        try:
            self.plotter.remove_actor("sun_sphere", reset_camera=False)
        except Exception:
            pass
        if not is_night:
            sun_pos = np.asarray(sun_dir, dtype=float) * 500.0
            try:
                self.plotter.add_mesh(
                    pv.Sphere(
                        radius=15.0,
                        center=(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2])),
                    ),
                    color="#ffeb3b",
                    emissive=True,
                    name="sun_sphere",
                    lighting=False,
                )
            except Exception:
                pass

        # Scene lights
        try:
            self.plotter.remove_all_lights()
            sun_pos   = np.asarray(sun_dir, dtype=float) * 500.0
            intensity = max(0.0, float(sun_dir[2])) * 0.9 + 0.1
            sun_light = pv.Light(
                light_type="scene light",
                position=(float(sun_pos[0]), float(sun_pos[1]), float(sun_pos[2])),
                focal_point=(0.0, 0.0, 0.0),
                intensity=float(intensity),
            )
            self.plotter.add_light(sun_light)
            if is_night:
                moon = pv.Light(
                    light_type="scene light",
                    position=(0.0, 0.0, 500.0),
                    intensity=0.12,
                )
                self.plotter.add_light(moon)
        except Exception:
            pass

        # Building window ambient (warm glow at night / dawn / dusk)
        self._update_window_lights(float(hour), is_night)

        # Road/ped actor base colours follow style (don't override weather darkening)
        if self.weather.get("mode", "clear") == "clear":
            v_actor = self.scene_state.get("vehicle_actor")
            if v_actor is not None:
                try:
                    v_actor.GetProperty().SetColor(pv.Color(str(style["vehicle"])).float_rgb)
                except Exception:
                    pass
            p_actor = self.scene_state.get("ped_actor")
            if p_actor is not None:
                try:
                    p_actor.GetProperty().SetColor(pv.Color(str(style["ped"])).float_rgb)
                except Exception:
                    pass

        # Atmosphere fog
        try:
            _apply_atmosphere_fn(
                self.plotter,
                hour=float(hour),
                is_night=bool(is_night),
                preset=str(self.scene_state.get("preset", "mini")),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Building window glow
    # ------------------------------------------------------------------

    def _update_window_lights(self, hour: float, is_night: bool) -> None:
        """Simulate lit windows by boosting building ambient at dusk/night/dawn,
        and by placing glowing point sprites on building faces during night hours."""
        if is_night:
            w_factor = 1.0
        elif 6.0 <= hour < 8.0:
            w_factor = max(0.0, 1.0 - (hour - 6.0) / 2.0)    # dawn: fade out
        elif 17.0 <= hour < 20.0:
            w_factor = (hour - 17.0) / 3.0                     # dusk: fade in
        else:
            w_factor = 0.0

        ambient = 0.05 + 0.55 * w_factor

        _pbr_actors = self.scene_state.get("_building_actors_pbr", {})
        for ba in _pbr_actors.values():
            try:
                ba.GetProperty().SetAmbient(float(ambient))
                if w_factor > 0.01:
                    ba.GetProperty().SetAmbientColor(1.0, 0.84, 0.58)  # warm amber
                else:
                    ba.GetProperty().SetAmbientColor(1.0, 1.0, 1.0)
            except Exception:
                pass

        b_actor = self.scene_state.get("building_actor")
        if not _pbr_actors and b_actor is not None:
            try:
                b_actor.GetProperty().SetAmbient(float(ambient))
                if w_factor > 0.01:
                    b_actor.GetProperty().SetAmbientColor(1.0, 0.84, 0.58)
                else:
                    b_actor.GetProperty().SetAmbientColor(1.0, 1.0, 1.0)
            except Exception:
                pass

        # Procedural facades carry real lit windows (emissive map); the old
        # point-sprite "windows" would double them, so only use those without.
        if self._set_facade_emission(float(w_factor)):
            self._update_window_light_points(12.0, False)     # removes any sprites
            return
        # Geometric window lights: small glowing point sprites on building faces
        self._update_window_light_points(hour, is_night)

    def _update_window_light_points(self, hour: float, is_night: bool) -> None:
        """Add or remove glowing point-sprite windows on building faces.

        Points are seeded by hour (snapped to integer) so they don't flicker
        every frame, and are only rebuilt when the hour bucket changes.
        Capped at ~300 points for performance.
        """
        # Daytime: remove any existing window light actors and bail out
        if not is_night and not (17.0 <= hour < 20.0):
            try:
                self.plotter.remove_actor("_window_lights", reset_camera=False)
            except Exception:
                pass
            self.scene_state.pop("_window_lights_hour", None)
            return

        # Check if we already rendered at a similar hour (within 1 h) to avoid
        # per-frame rebuilds
        last_wl_hour = self.scene_state.get("_window_lights_hour")
        if last_wl_hour is not None:
            diff = abs(float(hour) - float(last_wl_hour))
            if diff < 1.0:
                return

        # Gather building geometry — try PBR actors first, fall back to single actor
        poly_data = None
        try:
            _pbr_actors = self.scene_state.get("_building_actors_pbr", {})
            for ba in _pbr_actors.values():
                pd = ba.GetMapper().GetInput()
                if pd is not None and pd.GetNumberOfPoints() > 0:
                    poly_data = pd
                    break
            if poly_data is None:
                b_actor = self.scene_state.get("building_actor")
                if b_actor is not None:
                    pd = b_actor.GetMapper().GetInput()
                    if pd is not None and pd.GetNumberOfPoints() > 0:
                        poly_data = pd
        except Exception:
            return

        if poly_data is None:
            return

        try:
            # Sample random face centroids seeded by integer hour for stability
            n_cells = poly_data.GetNumberOfCells()
            if n_cells < 1:
                return

            seed = int(hour) % 24
            rng = np.random.default_rng(seed)

            n_windows = min(300, max(50, n_cells // 4))
            cell_ids = rng.integers(0, n_cells, size=n_windows)

            pts_list = []
            for cid in cell_ids:
                cell = poly_data.GetCell(int(cid))
                bounds = cell.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
                cx = (bounds[0] + bounds[1]) * 0.5
                cy = (bounds[2] + bounds[3]) * 0.5
                cz = (bounds[4] + bounds[5]) * 0.5
                # Add slight random offset so windows don't all cluster at face centres
                cx += rng.uniform(-0.5, 0.5)
                cy += rng.uniform(-0.5, 0.5)
                cz += rng.uniform(0.5, 3.0)   # shift upward slightly (above ground floor)
                pts_list.append((cx, cy, cz))

            if not pts_list:
                return

            pts = np.array(pts_list, dtype=float)
            self.plotter.add_points(
                pts,
                point_size=4,
                color="#f5c842",
                render_points_as_spheres=True,
                name="_window_lights",
            )
            self.scene_state["_window_lights_hour"] = float(hour)
        except Exception:
            pass
