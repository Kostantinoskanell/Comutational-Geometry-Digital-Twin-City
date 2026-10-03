"""ParkingMixin — static parked cars in OSM parking lots with enter/leave activity.

Parking polygons are extracted from the land-use fill mesh stored in
street_graph.graph["fill_pts/faces/rgb"] (cells whose RGB matches the parking
colour #787868 = (120,120,104)).  Cars are placed in a bbox-aligned grid inside
every polygon, then instanced as a single glyph mesh for GPU efficiency.

Every ~30 real-seconds a random car vacates its slot (glyph mesh rebuilt) and
returns after 45–90 s, giving a gentle sense of activity.  Connecting to the IDM
car system: on each leave event the nearest drivable-path edge is found and an
extra car is inserted into car_anim at that location so it drives away from the
lot visually.
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv

_PARKING_RGB = (120, 120, 104)   # hex #787868 from _fetch_land_fill_mesh


class ParkingMixin:

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_parking(self) -> None:
        self._parking_slots: np.ndarray      = np.empty(0, dtype=bool)
        self._parking_rng                     = np.random.default_rng(
            int(getattr(self.args, "seed", 42)) + 88_881
        )
        self._parking_last_event_t: float    = 0.0
        self._parking_deferred: list         = []   # (return_wall_t, slot_idx)
        self._parked_car_template: pv.PolyData | None = None

        polys = self._extract_parking_polys()
        if not polys:
            print("[parking] no parking polygons in fill mesh — skipped")
            return
        self._place_parked_cars(polys)

    # ------------------------------------------------------------------
    # Extract polygon outlines from fill mesh
    # ------------------------------------------------------------------

    def _extract_parking_polys(self) -> list[np.ndarray]:
        """Return list of (N,2) XY arrays for fill-mesh cells with parking colour."""
        pts   = self.street_graph.graph.get("fill_pts")
        faces = self.street_graph.graph.get("fill_faces")
        rgb   = self.street_graph.graph.get("fill_rgb")
        if pts is None or faces is None or rgb is None:
            return []

        pts_arr  = np.asarray(pts,   dtype=float)
        face_arr = np.asarray(faces, dtype=np.int64).ravel()
        rgb_arr  = np.asarray(rgb,   dtype=np.uint8)

        polys: list[np.ndarray] = []
        cursor   = 0
        cell_idx = 0
        n_cells  = len(rgb_arr)
        n_face   = len(face_arr)

        while cursor < n_face and cell_idx < n_cells:
            n_verts = int(face_arr[cursor])
            if cursor + n_verts >= n_face:
                break
            verts = face_arr[cursor + 1: cursor + 1 + n_verts]
            r = int(rgb_arr[cell_idx, 0])
            g = int(rgb_arr[cell_idx, 1])
            b = int(rgb_arr[cell_idx, 2])
            if (r, g, b) == _PARKING_RGB:
                polys.append(pts_arr[verts, :2].copy())
            cursor   += n_verts + 1
            cell_idx += 1

        return polys

    # ------------------------------------------------------------------
    # Place cars
    # ------------------------------------------------------------------

    def _place_parked_cars(self, polys: list[np.ndarray]) -> None:
        dem = self.street_graph.graph.get("terrain_sampler")

        ROW_SPACING = 5.5    # m between rows
        COL_SPACING = 2.35   # m between slots
        CAR_Z       = 0.30   # m above local ground (before DEM)

        all_pos: list[np.ndarray] = []

        for poly_xy in polys:
            x0, y0 = poly_xy.min(axis=0)
            x1, y1 = poly_xy.max(axis=0)
            if (x1 - x0) < 6.0 or (y1 - y0) < 6.0:
                continue
            xs = np.arange(x0 + COL_SPACING * 0.5, x1, COL_SPACING)
            ys = np.arange(y0 + ROW_SPACING  * 0.5, y1, ROW_SPACING)
            for gy in ys:
                for gx in xs:
                    all_pos.append(np.array([gx, gy, CAR_Z], dtype=float))

        if not all_pos:
            print("[parking] lots found but no grid slots fit inside them")
            return

        positions = np.array(all_pos, dtype=float)

        # Only pre-lift to DEM when draping is already active; otherwise the
        # cars would float above the flat roads. Terrain toggle handles the
        # lift/restore later via _drape_actor_points("_parked_cars_actor").
        if dem is not None and self.scene_state.get("_terrain_drape_active"):
            try:
                positions[:, 2] += dem(positions[:, :2]).astype(float)
            except Exception:
                pass

        # Cap to protect GPU
        MAX = min(len(positions), 400)
        if len(positions) > MAX:
            idx = self._parking_rng.choice(len(positions), MAX, replace=False)
            positions = positions[idx]

        tmpl = pv.Box(bounds=(-1.10, 1.10, -0.55, 0.55, 0.0, 0.55))
        self._parked_car_template = tmpl

        pts_pd = pv.PolyData(positions)
        glyphs = pts_pd.glyph(geom=tmpl, orient=False, scale=False)

        try:
            actor = self.plotter.add_mesh(
                glyphs,
                color="#5c6880",
                smooth_shading=False,
                lighting=True,
                reset_camera=False,
                name="parked_cars",
            )
            self.scene_state["_parked_cars_actor"] = actor
        except Exception as exc:
            print(f"[parking] glyph render failed: {exc}")
            return

        self._parking_slots = np.ones(len(positions), dtype=bool)
        self.scene_state["_parked_car_positions"] = positions.copy()
        print(f"[parking] {len(positions)} parked cars placed in {len(polys)} lot(s)")

    # ------------------------------------------------------------------
    # Leave / enter schedule
    # ------------------------------------------------------------------

    def _animate_parking(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        if len(self._parking_slots) == 0:
            return

        now     = time.perf_counter()
        changed = False

        # Process deferred returns
        still_pending = []
        for ret_t, slot_idx in self._parking_deferred:
            if now >= ret_t:
                if slot_idx < len(self._parking_slots):
                    self._parking_slots[slot_idx] = True
                    changed = True
            else:
                still_pending.append((ret_t, slot_idx))
        self._parking_deferred = still_pending

        # Leave event every ~30 real-seconds
        if now - self._parking_last_event_t >= 30.0:
            self._parking_last_event_t = now
            occupied = np.where(self._parking_slots)[0]
            if len(occupied) > 0:
                slot_idx = int(self._parking_rng.choice(occupied))
                self._parking_slots[slot_idx] = False
                ret_delay = float(self._parking_rng.uniform(45.0, 90.0))
                self._parking_deferred.append((now + ret_delay, slot_idx))
                changed = True
                # Optionally spawn a moving car from this slot into the IDM system
                self._spawn_exiting_car(slot_idx)

        if changed:
            self._rebuild_parked_glyph()

    def _pick_car_to_recycle(self) -> int | None:
        """Choose which existing fleet car re-enters traffic from a parking exit.

        The fleet is a fixed-size pool: every per-car array (car_len, accel,
        idm_*_arr, planned_edges, stuck_time, …) and the renderer's actor pool
        are sized once at startup, so an exit re-uses a car instead of growing
        the fleet. Prefers cars with no active demand trip (so no journey is
        cancelled), never the user-selected car, and among those the one
        farthest from the camera focal point so the relocation is off-screen.
        """
        edge_idx = self.car_anim.get("edge_idx")
        if edge_idx is None or len(edge_idx) == 0:
            return None
        n = int(len(edge_idx))
        candidates = np.ones(n, dtype=bool)

        selected = getattr(self, "route_state", {}).get("selected_car_idx")
        if selected is not None and 0 <= int(selected) < n:
            candidates[int(selected)] = False

        plans = self.car_anim.get("planned_edges")
        if plans is not None and len(plans) == n:
            idle = np.array([p is None for p in plans], dtype=bool)
            if (candidates & idle).any():
                candidates &= idle
        if not candidates.any():
            return None

        idx = np.flatnonzero(candidates)
        pos = self.car_anim.get("pos")
        try:
            focal = np.asarray(self.plotter.camera.focal_point, dtype=float)[:2]
            pos_xy = np.asarray(pos, dtype=float)[idx, :2]
            return int(idx[int(np.argmax(np.linalg.norm(pos_xy - focal, axis=1)))])
        except Exception:
            return int(self._parking_rng.choice(idx))

    def _spawn_exiting_car(self, slot_idx: int) -> None:
        """A car leaves a vacated parking slot and joins traffic (best-effort)."""
        if not hasattr(self, "car_anim") or not bool(self.car_anim.get("enabled")):
            return
        all_pos = self.scene_state.get("_parked_car_positions")
        if all_pos is None or slot_idx >= len(all_pos):
            return

        slot_xy = all_pos[slot_idx, :2]

        try:
            path_starts = np.array(
                [np.asarray(p["points"], dtype=float)[0, :2] for p in self.car_paths],
                dtype=float,
            )
            dists = np.linalg.norm(path_starts - slot_xy, axis=1)
            p_idx = int(np.argmin(dists))
            if float(dists[p_idx]) > 150.0:
                return

            i = self._pick_car_to_recycle()
            if i is None:
                return

            base_speed = float(
                self.car_paths[p_idx]["maxspeed_ms"] * self._parking_rng.uniform(0.6, 0.9)
            )
            traffic_speed = float(getattr(self.args, "traffic_speed", 1.0))
            ca = self.car_anim
            ca["edge_idx"][i] = p_idx
            ca["dist"][i] = 0.0
            ca["speed"][i] = 1.0
            if ca.get("desired_speed_base") is not None:
                ca["desired_speed_base"][i] = base_speed
            ca["desired_speed"][i] = base_speed * traffic_speed
            for key in ("accel", "stop_wait", "stuck_time"):
                arr = ca.get(key)
                if arr is not None and len(arr) > i:
                    arr[i] = 0.0
            plans = ca.get("planned_edges")
            if plans is not None and len(plans) > i:
                plans[i] = None
            cursors = ca.get("planned_cursor")
            if cursors is not None and len(cursors) > i:
                cursors[i] = 0
        except Exception as exc:
            print(f"[parking] exit spawn skipped: {exc}")

    def _rebuild_parked_glyph(self) -> None:
        """Rebuild the instanced glyph mesh reflecting current slot occupancy."""
        all_pos = self.scene_state.get("_parked_car_positions")
        tmpl    = self._parked_car_template
        if all_pos is None or tmpl is None:
            return

        n       = min(len(self._parking_slots), len(all_pos))
        occ_pos = all_pos[:n][self._parking_slots[:n]]

        if len(occ_pos) == 0:
            try:
                self.plotter.remove_actor("parked_cars", reset_camera=False)
            except Exception:
                pass
            return

        pts_pd = pv.PolyData(occ_pos)
        glyphs = pts_pd.glyph(geom=tmpl, orient=False, scale=False)
        try:
            self.plotter.add_mesh(
                glyphs,
                color="#5c6880",
                smooth_shading=False,
                lighting=True,
                reset_camera=False,
                name="parked_cars",
            )
        except Exception:
            pass
