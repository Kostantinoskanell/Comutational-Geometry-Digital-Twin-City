"""WalkMixin — first-person / follow-camera for the Digital Twin.

Press 'f' to cycle through modes:
  off  →  3rd-person follow (nearest pedestrian)
       →  1st-person POV (through the pedestrian's eyes)
       →  free walk (detached from agent, move with arrow keys)
       →  off  (overview camera restored)

Arrow keys in free-walk mode:
  Up / Down   — walk forward / backward at 2 m/step
  Left / Right — turn 8° left / right

Arrow keys otherwise — normal camera pan (unchanged).
Escape always exits walk mode immediately.
"""
from __future__ import annotations

import math
import numpy as np


_EYE_HEIGHT  = 1.65   # metres above ground for 1st-person and free-walk camera
_FOV_3P_BACK = 7.0    # metres behind agent in 3rd-person
_FOV_3P_UP   = 4.5    # metres above agent in 3rd-person
_WALK_STEP   = 2.0    # metres per arrow-key press in free-walk
_TURN_STEP   = 8.0    # degrees per arrow-key press in free-walk


class WalkMixin:
    """Camera follow / first-person walk mixin.

    Requires:
      - self.plotter (PyVista plotter)
      - self.scene_state dict
      - self.ped_anim dict  (from PedMixin)
      - self._sample_ped_positions() method (from PedMixin)
      - self.street_graph with optional "terrain_sampler" in .graph
      - self._reset_camera_view() method (from UIMixin / CameraUtils)
      - self.sumo dict (from SumoMixin, optional — for SUMO engine follow mode)
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_walk_cam(self) -> None:
        self.scene_state.setdefault("walk_mode", None)   # None|"3p"|"1p"|"1p_ctrl"|"free"
        self.scene_state.setdefault("walk_ped_idx",  None)
        self.scene_state.setdefault("walk_controlled_ped_idx", None)  # IDM ped being player-driven
        # SUMO follow state: target agent id + kind ("vehicle" or "person")
        self.scene_state.setdefault("walk_sumo_id",  None)
        self.scene_state.setdefault("walk_sumo_kind", None)
        self.scene_state.setdefault("walk_pos", None)     # np.array free-walk / controlled XYZ
        self.scene_state.setdefault("walk_yaw", 0.0)      # heading degrees (CCW from +X)
        self.scene_state.setdefault("walk_pitch", 0.0)    # look up/down degrees
        self.scene_state.setdefault("_walk_prev_pos", None)
        self._walk_overview_cam = None   # saved camera state for restoration
        self._walk_mouse_obs = None      # MouseMoveEvent observer tag (free-walk look)
        self._walk_mouse_last = None     # last mouse (x, y) for look deltas

    # ------------------------------------------------------------------
    # Key handler — cycles modes
    # ------------------------------------------------------------------

    def _on_f_key(self) -> None:
        mode = self.scene_state.get("walk_mode")

        if mode is None:
            # Prefer IDM pedestrians; fall back to SUMO agents if in sumo engine mode
            # or if no IDM peds are available.
            idx = self._walk_find_nearest_ped()
            if idx is not None:
                # Standard IDM ped follow
                self._walk_save_camera()
                self.scene_state["walk_mode"]     = "3p"
                self.scene_state["walk_ped_idx"]  = idx
                self.scene_state["walk_sumo_id"]  = None
                self.scene_state["walk_sumo_kind"] = None
                self.scene_state["_walk_prev_pos"] = None
                self._walk_hud("3rd-person follow (ped)  [f]=1st-person  [Esc]=exit")
            else:
                # Try SUMO agent as follow target
                sumo_target = self._walk_find_nearest_sumo_agent()
                if sumo_target is None:
                    print("[walk] no pedestrians or SUMO agents to follow")
                    return
                sid, skind = sumo_target
                self._walk_save_camera()
                self.scene_state["walk_mode"]      = "3p"
                self.scene_state["walk_ped_idx"]   = None
                self.scene_state["walk_sumo_id"]   = sid
                self.scene_state["walk_sumo_kind"] = skind
                self.scene_state["_walk_prev_pos"] = None
                self._walk_hud(
                    f"3rd-person follow (SUMO {skind} {sid})  "
                    "[f]=1st-person  [Esc]=exit"
                )

        elif mode == "3p":
            # Switch to 1st-person POV
            self.scene_state["walk_mode"] = "1p"
            self._walk_hud("1st-person view  [f]=free walk  [Esc]=exit")

        elif mode == "1p":
            # Detach from agent, enter free-walk at current camera position
            try:
                cam_pos = np.array(self.plotter.camera.position, dtype=float)
            except Exception:
                cam_pos = np.zeros(3)
            self.scene_state["walk_mode"]  = "free"
            self.scene_state["walk_pos"]   = cam_pos.copy()
            self.scene_state["walk_pitch"] = 0.0
            self.scene_state["walk_ped_idx"] = None
            self.scene_state["walk_sumo_id"] = None
            self._walk_mouse_look_on()
            self._walk_hud("free walk  [wasd]=move  mouse=look  [←→]=turn  [f]=exit")

        elif mode in ("free", "1p_ctrl"):
            self._walk_exit()

    def _walk_exit(self) -> None:
        self.scene_state["walk_mode"]               = None
        self.scene_state["walk_ped_idx"]            = None
        self.scene_state["walk_sumo_id"]            = None
        self.scene_state["walk_sumo_kind"]          = None
        self.scene_state["walk_pos"]                = None
        self.scene_state["walk_controlled_ped_idx"] = None
        self._walk_mouse_look_off()
        self._walk_restore_camera()
        self._walk_hud("")
        print("[walk] overview camera restored")

    # ------------------------------------------------------------------
    # Mouse look (free-walk) — trackball is suspended, mouse deltas steer
    # ------------------------------------------------------------------

    def _walk_mouse_look_on(self) -> None:
        # Swap to a null interactor style so the trackball stops fighting the
        # walk camera.  (renderer.SetInteractive(0) also worked but broke
        # pyvista's poked-renderer lookup — every pick raised
        # "Poked renderer not found in Plotter".)
        try:
            import vtk
            import weakref
            self._walk_saved_style = self.plotter.iren.interactor.GetInteractorStyle()
            _null_style = vtk.vtkInteractorStyleUser()
            # pyvista's pick handler walks GetInteractorStyle()._parent()._plotter
            # — a raw VTK style has no _parent, so every click raised
            # AttributeError and broke picking while walking.  Graft the same
            # weakref pyvista's own styles carry.
            _null_style._parent = getattr(
                self._walk_saved_style, "_parent", None) or weakref.ref(self.plotter.iren)
            self.plotter.iren.interactor.SetInteractorStyle(_null_style)
        except Exception:
            self._walk_saved_style = None
        self._walk_mouse_last = None
        try:
            self._walk_mouse_obs = self.plotter.iren.interactor.AddObserver(
                "MouseMoveEvent", self._walk_on_mouse_move)
            print("[walk] mouse look enabled — move mouse to look around")
        except Exception as exc:
            self._walk_mouse_obs = None
            print(f"[walk] mouse look unavailable: {exc}")

    def _walk_mouse_look_off(self) -> None:
        if self._walk_mouse_obs is not None:
            try:
                self.plotter.iren.interactor.RemoveObserver(self._walk_mouse_obs)
            except Exception:
                pass
            self._walk_mouse_obs = None
        self._walk_mouse_last = None
        try:
            if getattr(self, "_walk_saved_style", None) is not None:
                self.plotter.iren.interactor.SetInteractorStyle(self._walk_saved_style)
                self._walk_saved_style = None
            else:
                self.plotter.enable_trackball_style()
        except Exception:
            try:
                self.plotter.enable_trackball_style()
            except Exception:
                pass
        # SetInteractorStyle re-registers the style's OnChar observer which brings
        # back VTK's built-in key bindings ('3'=stereo, 'w'=wireframe, ...).
        # Strip them again every time we restore the style.
        try:
            self.plotter.iren.interactor.RemoveObservers("CharEvent")
        except Exception:
            pass

    def _walk_on_mouse_move(self, obj, _event) -> None:
        if self.scene_state.get("walk_mode") not in ("free", "1p_ctrl"):
            return
        try:
            x, y = obj.GetEventPosition()
        except Exception:
            return
        last = self._walk_mouse_last
        self._walk_mouse_last = (x, y)
        if last is None:
            return
        dx, dy = x - last[0], y - last[1]
        self.scene_state["walk_yaw"] = (
            self.scene_state.get("walk_yaw", 0.0) - dx * 0.25) % 360.0
        self.scene_state["walk_pitch"] = float(np.clip(
            self.scene_state.get("walk_pitch", 0.0) + dy * 0.20, -70.0, 70.0))

    # ------------------------------------------------------------------
    # WASD movement (free-walk).  Returns True when the key was consumed so
    # the global bindings (weather / camera pan) can skip their action.
    # ------------------------------------------------------------------

    def _walk_key(self, key: str) -> bool:
        if self.scene_state.get("walk_mode") != "free":
            return False
        if key == "w":
            self._walk_step_forward(_WALK_STEP)
        elif key == "s":
            self._walk_step_forward(-_WALK_STEP)
        elif key in ("a", "d"):
            yaw = self.scene_state.get("walk_yaw", 0.0)
            side = math.radians(yaw + (90.0 if key == "a" else -90.0))
            walk_pos = self.scene_state.get("walk_pos")
            if walk_pos is not None:
                walk_pos[:2] += np.array([math.cos(side), math.sin(side)]) * _WALK_STEP
                self.scene_state["walk_pos"] = walk_pos
        else:
            return False
        return True

    # ------------------------------------------------------------------
    # Click-to-select: enter 1st-person on the pedestrian nearest the click
    # ------------------------------------------------------------------

    def _walk_select_ped_at(self, px: float, py: float, max_dist: float = 3.5) -> bool:
        """If a pedestrian is within max_dist of the click, take direct control of
        THAT pedestrian (detach from IDM, steer with arrows).  Returns True when
        the click was consumed."""
        # IDM pedestrians — enter player-controlled mode
        try:
            if self.ped_anim.get("enabled"):
                positions = self._sample_ped_positions()
                if len(positions):
                    d = np.linalg.norm(
                        np.asarray(positions)[:, :2] - np.array([px, py]), axis=1)
                    i = int(np.argmin(d))
                    if float(d[i]) <= max_dist:
                        init_pos = positions[i].copy()
                        # Estimate current heading for smooth entry
                        try:
                            prev = self.scene_state.get("_walk_prev_pos")
                            if prev is not None:
                                delta = init_pos[:2] - np.asarray(prev, dtype=float)
                                if np.linalg.norm(delta) > 0.02:
                                    self.scene_state["walk_yaw"] = float(
                                        math.degrees(math.atan2(float(delta[1]), float(delta[0]))))
                        except Exception:
                            pass
                        self._walk_save_camera()
                        self._walk_mouse_look_on()
                        self.scene_state.update(
                            walk_mode="1p_ctrl",
                            walk_ped_idx=i,
                            walk_controlled_ped_idx=i,
                            walk_pos=init_pos,
                            walk_sumo_id=None,
                            walk_sumo_kind=None,
                            _walk_prev_pos=None,
                            walk_pitch=0.0,
                        )
                        self._walk_hud(
                            f"controlling ped #{i}  [↑↓]=walk  [←→]=turn  mouse=look  [f]=exit")
                        print(f"[walk] controlling pedestrian #{i} — detached from IDM")
                        return True
        except Exception:
            pass
        # SUMO persons
        try:
            rt = getattr(self, "sumo", None)
            if rt and rt.get("enabled"):
                best_id, best_d = None, float(max_dist)
                for pid, pos in (rt.get("p_current") or {}).items():
                    dd = float(np.hypot(float(pos[0]) - px, float(pos[1]) - py))
                    if dd < best_d:
                        best_d, best_id = dd, pid
                if best_id is not None:
                    self._walk_save_camera()
                    self.scene_state.update(
                        walk_mode="1p", walk_ped_idx=None,
                        walk_sumo_id=best_id, walk_sumo_kind="person",
                        _walk_prev_pos=None, walk_pitch=0.0)
                    self._walk_hud(
                        f"1st-person (SUMO person {best_id})  [f]=free walk")
                    print(f"[walk] following SUMO person {best_id} in first person")
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Animation tick — called every frame
    # ------------------------------------------------------------------

    def _animate_walk(self, _: int) -> None:
        mode = self.scene_state.get("walk_mode")
        if mode is None:
            return
        if not bool(self.scene_state.get("interactive_ready", False)):
            return

        if mode in ("3p", "1p"):
            self._walk_tick_follow(mode)
        elif mode == "free":
            self._walk_tick_free()
        elif mode == "1p_ctrl":
            self._walk_tick_controlled()

    def _walk_tick_follow(self, mode: str) -> None:
        """Update camera for 3p/1p follow mode — works for both IDM peds and SUMO agents."""
        # ── SUMO agent follow ────────────────────────────────────────────────
        sumo_id   = self.scene_state.get("walk_sumo_id")
        sumo_kind = self.scene_state.get("walk_sumo_kind", "vehicle")
        if sumo_id is not None:
            sumo = getattr(self, "sumo", None)
            if sumo and sumo.get("enabled"):
                lerp_dict = (sumo.get("current")   if sumo_kind == "vehicle"
                             else sumo.get("p_current"))
                entry = (lerp_dict or {}).get(sumo_id)
                if entry is not None:
                    pos = np.array([float(entry[0]), float(entry[1]), float(entry[2])],
                                   dtype=float)
                    # Heading from SUMO angle (vehicles only)
                    if sumo_kind == "vehicle" and len(entry) >= 4:
                        # SUMO angle: compass deg, 0=N CW → convert to CCW from +X
                        _sumo_ang = float(entry[3])
                        yaw = (90.0 - _sumo_ang) % 360.0
                        self.scene_state["walk_yaw"] = yaw
                    else:
                        # Persons: estimate heading from movement delta
                        prev = self.scene_state.get("_walk_prev_pos")
                        if prev is not None:
                            delta = pos[:2] - prev
                            if np.linalg.norm(delta) > 0.02:
                                yaw = math.degrees(
                                    math.atan2(float(delta[1]), float(delta[0]))
                                )
                                self.scene_state["walk_yaw"] = yaw
                    self.scene_state["_walk_prev_pos"] = pos[:2].copy()
                    self._walk_apply_camera(mode, pos)
                    return
            # SUMO agent gone — exit walk mode cleanly
            self._walk_exit()
            return

        # ── IDM pedestrian follow ────────────────────────────────────────────
        idx = self.scene_state.get("walk_ped_idx")
        if idx is None:
            return
        try:
            positions = self._sample_ped_positions()
        except Exception:
            return
        if idx >= len(positions):
            return

        pos = positions[idx]   # (x, y, z)

        # Estimate heading from movement delta
        prev = self.scene_state.get("_walk_prev_pos")
        yaw  = self.scene_state.get("walk_yaw", 0.0)
        if prev is not None:
            delta = pos[:2] - prev
            if np.linalg.norm(delta) > 0.02:
                yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
                self.scene_state["walk_yaw"] = yaw
        self.scene_state["_walk_prev_pos"] = pos[:2].copy()
        self._walk_apply_camera(mode, pos)

    def _walk_apply_camera(self, mode: str, pos: np.ndarray) -> None:
        """Shared camera positioning for 3p and 1p modes given an agent world position."""
        if not np.all(np.isfinite(np.asarray(pos, dtype=float))):
            return   # a NaN camera position renders pure background
        yaw     = self.scene_state.get("walk_yaw", 0.0)
        rad     = math.radians(yaw)
        fwd_hat = np.array([math.cos(rad), math.sin(rad), 0.0])
        try:
            if mode == "3p":
                cam_pos   = pos - fwd_hat * _FOV_3P_BACK + np.array([0, 0, _FOV_3P_UP])
                focal_pt  = pos + np.array([0, 0, _EYE_HEIGHT])
                self.plotter.camera.position    = tuple(float(v) for v in cam_pos)
                self.plotter.camera.focal_point = tuple(float(v) for v in focal_pt)
                self.plotter.camera.up          = (0.0, 0.0, 1.0)
            else:   # 1p
                pitch = math.radians(self.scene_state.get("walk_pitch", 0.0))
                look  = np.array([
                    math.cos(rad) * math.cos(pitch),
                    math.sin(rad) * math.cos(pitch),
                    math.sin(pitch),
                ])
                cam_pos  = pos + np.array([0, 0, _EYE_HEIGHT])
                focal_pt = cam_pos + look * 6.0
                self.plotter.camera.position    = tuple(float(v) for v in cam_pos)
                self.plotter.camera.focal_point = tuple(float(v) for v in focal_pt)
                self.plotter.camera.up          = (0.0, 0.0, 1.0)
            self.plotter.camera.ParallelProjectionOff()
            # Do NOT auto-reset clipping here: the sea plane spans ~7× the
            # city, blowing far to ~5000 m, and VTK then forces
            # near ≥ 0.001·far ≈ 5 m — clipping away the ground and every
            # nearby wall (screen shows only sky at most view angles).
            self.plotter.camera.clipping_range = (0.4, 6000.0)
        except Exception:
            pass

    def _walk_tick_free(self) -> None:
        walk_pos = self.scene_state.get("walk_pos")
        if walk_pos is None:
            return
        yaw   = self.scene_state.get("walk_yaw", 0.0)
        pitch = self.scene_state.get("walk_pitch", 0.0)
        rad   = math.radians(yaw)
        prad  = math.radians(pitch)
        fwd_hat = np.array([
            math.cos(rad) * math.cos(prad),
            math.sin(rad) * math.cos(prad),
            math.sin(prad),
        ])

        # Snap Z to ground (DEM only while terrain draping is active)
        walk_pos[2] = self._walk_ground_z(walk_pos, default_z=0.0)

        cam_pos  = walk_pos + np.array([0, 0, _EYE_HEIGHT])
        focal_pt = cam_pos + fwd_hat * 6.0
        try:
            self.plotter.camera.position    = tuple(float(v) for v in cam_pos)
            self.plotter.camera.focal_point = tuple(float(v) for v in focal_pt)
            self.plotter.camera.up          = (0.0, 0.0, 1.0)
            self.plotter.camera.ParallelProjectionOff()
            # Fixed range — see _walk_apply_camera for why auto-reset breaks here.
            self.plotter.camera.clipping_range = (0.4, 6000.0)
        except Exception:
            pass

    def _walk_tick_controlled(self) -> None:
        """1st-person camera tick for player-controlled pedestrian mode."""
        walk_pos = self.scene_state.get("walk_pos")
        if walk_pos is None:
            return
        walk_pos[2] = self._walk_ground_z(walk_pos, default_z=0.3)
        self.scene_state["walk_pos"] = walk_pos
        self._walk_apply_camera("1p", walk_pos)

    def _walk_ground_z(self, walk_pos, default_z: float = 0.0) -> float:
        """Ground height under the walker.

        DEM heights only apply while terrain draping is ACTIVE — with draping
        off the scene is flat at z≈0, and snapping the camera to real-world
        elevation (or NaN outside the DEM grid) lifted it into empty sky.
        """
        if self.scene_state.get("_terrain_drape_active"):
            dem = self.street_graph.graph.get("terrain_sampler")
            if dem is not None:
                try:
                    z = float(dem(walk_pos[:2].reshape(1, 2))[0])
                    if math.isfinite(z):
                        return z + 0.3
                except Exception:
                    pass
        return float(default_z)

    # ------------------------------------------------------------------
    # Arrow-key movement (free walk and controlled-ped mode)
    # ------------------------------------------------------------------

    def _walk_arrow_up(self) -> None:
        if self.scene_state.get("walk_mode") in ("free", "1p_ctrl"):
            self._walk_step_forward(_WALK_STEP)
        else:
            self._pan_camera(0.0, getattr(self, "_walk_pan_step", 50.0))

    def _walk_arrow_down(self) -> None:
        if self.scene_state.get("walk_mode") in ("free", "1p_ctrl"):
            self._walk_step_forward(-_WALK_STEP)
        else:
            self._pan_camera(0.0, -getattr(self, "_walk_pan_step", 50.0))

    def _walk_arrow_left(self) -> None:
        if self.scene_state.get("walk_mode") in ("free", "1p_ctrl"):
            self._walk_turn(-_TURN_STEP)
        else:
            self._pan_camera(-getattr(self, "_walk_pan_step", 50.0), 0.0)

    def _walk_arrow_right(self) -> None:
        if self.scene_state.get("walk_mode") in ("free", "1p_ctrl"):
            self._walk_turn(+_TURN_STEP)
        else:
            self._pan_camera(+getattr(self, "_walk_pan_step", 50.0), 0.0)

    def _walk_step_forward(self, dist: float) -> None:
        walk_pos = self.scene_state.get("walk_pos")
        if walk_pos is None:
            return
        yaw = self.scene_state.get("walk_yaw", 0.0)
        rad = math.radians(yaw)
        walk_pos[:2] += np.array([math.cos(rad), math.sin(rad)]) * dist
        self.scene_state["walk_pos"] = walk_pos

    def _walk_turn(self, delta_deg: float) -> None:
        self.scene_state["walk_yaw"] = (
            self.scene_state.get("walk_yaw", 0.0) + delta_deg
        ) % 360.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walk_find_nearest_ped(self) -> int | None:
        if not self.ped_anim.get("enabled"):
            return None
        try:
            positions = self._sample_ped_positions()
        except Exception:
            return None
        if len(positions) == 0:
            return None
        try:
            focal = np.array(self.plotter.camera.focal_point, dtype=float)
        except Exception:
            focal = np.zeros(3)
        dists = np.linalg.norm(positions[:, :2] - focal[:2], axis=1)
        return int(np.argmin(dists))

    def _walk_find_nearest_sumo_agent(self) -> tuple[str, str] | None:
        """Return (agent_id, kind) of the SUMO vehicle or person nearest to the
        current camera focal point, or None if no SUMO agents are present.

        Checks vehicles first, then persons.
        """
        sumo = getattr(self, "sumo", None)
        if not sumo or not sumo.get("enabled"):
            return None
        try:
            focal = np.array(self.plotter.camera.focal_point, dtype=float)
        except Exception:
            focal = np.zeros(3)

        best_dist = float("inf")
        best_id: str | None = None
        best_kind = "vehicle"

        for kind, lerp_dict in (("vehicle", sumo.get("current") or {}),
                                 ("person",  sumo.get("p_current") or {})):
            for aid, entry in lerp_dict.items():
                try:
                    dx = float(entry[0]) - float(focal[0])
                    dy = float(entry[1]) - float(focal[1])
                    d  = dx * dx + dy * dy
                    if d < best_dist:
                        best_dist = d
                        best_id   = aid
                        best_kind = kind
                except Exception:
                    continue

        if best_id is None:
            return None
        return (best_id, best_kind)

    def _walk_save_camera(self) -> None:
        try:
            self._walk_overview_cam = {
                "position":    tuple(self.plotter.camera.position),
                "focal_point": tuple(self.plotter.camera.focal_point),
                "up":          tuple(self.plotter.camera.up),
                "parallel":    bool(self.plotter.camera.GetParallelProjection()),
            }
        except Exception:
            self._walk_overview_cam = None

    def _walk_restore_camera(self) -> None:
        saved = self._walk_overview_cam
        if saved is None:
            try:
                self._reset_camera_view()
            except Exception:
                pass
            return
        try:
            self.plotter.camera.position    = saved["position"]
            self.plotter.camera.focal_point = saved["focal_point"]
            self.plotter.camera.up          = saved["up"]
            if saved.get("parallel"):
                self.plotter.camera.ParallelProjectionOn()
            else:
                self.plotter.camera.ParallelProjectionOff()
            self.renderer.ResetCameraClippingRange()
        except Exception:
            try:
                self._reset_camera_view()
            except Exception:
                pass

    def _walk_hud(self, msg: str) -> None:
        try:
            self.plotter.add_text(
                f"[walk] {msg}" if msg else "",
                position=(0.02, 0.95), name="walk_hud",
                font_size=9, viewport=True, color="#a0e0ff",
            )
        except Exception:
            pass
