from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# GLB loader shared by both PedMixin and CyclistMixin
# ---------------------------------------------------------------------------

def _load_glb_as_polydata(glb_path: str | Path) -> pv.PolyData | None:
    """Read a GLB file and return a single flat PolyData (None on failure).

    GLB files load as nested MultiBlock from PyVista.  We flatten the tree,
    merge all leaf meshes, apply the Kenney-style axis correction that the
    car OBJ loader also uses (rotate_x 90°, rotate_z -90°), and return a
    clean PolyData ready for scaling and placement.
    """
    try:
        raw = pv.read(str(glb_path))
    except Exception as exc:
        print(f"[glb] failed to read {glb_path}: {exc}")
        return None

    def _collect_leaves(mb, out: list[pv.PolyData]) -> None:
        for i in range(mb.n_blocks):
            b = mb[i]
            if b is None:
                continue
            if hasattr(b, "n_blocks"):
                _collect_leaves(b, out)
            elif hasattr(b, "n_points") and b.n_points > 0:
                # Ensure PolyData (UnstructuredGrid etc. can appear)
                if not isinstance(b, pv.PolyData):
                    try:
                        b = b.extract_surface()
                    except Exception:
                        continue
                out.append(b)

    if hasattr(raw, "n_blocks"):
        leaves: list[pv.PolyData] = []
        _collect_leaves(raw, leaves)
        if not leaves:
            return None
        if len(leaves) == 1:
            geo = leaves[0]
        else:
            geo = leaves[0].merge(leaves[1:], merge_points=False)
    elif isinstance(raw, pv.PolyData):
        geo = raw
    else:
        try:
            geo = raw.extract_surface()
        except Exception:
            return None

    # Axis correction: GLB is Y-up, VTK is Z-up.
    # Rotate by directly rewriting point coordinates to avoid triggering
    # PyVista's transform() path which errors on None-keyed vector arrays.
    #   rotate_x(90°):  (x, y, z) → (x, -z,  y)
    #   rotate_z(-90°): (x, y, z) → (y, -x,  z)
    # Combined:         (x, y, z) → (x, -z, y) → (-z, -x, y)
    pts = geo.points.copy()
    x, y, z = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
    geo.points[:, 0] = -z   # after rx: y_new=-z; after rz: x_new=y_new=-z
    geo.points[:, 1] = -x   # after rx: x stays x; after rz: y_new=-x
    geo.points[:, 2] =  y   # after rx: z_new=y; after rz: z stays z_new=y
    return geo


def _fold_arms_inplace(pts: np.ndarray, height: float) -> None:
    """Rotate T-posed arms (horizontal along ±Y) ~82° downward to a natural hang.

    After _load_glb_as_polydata's axis correction and scaling to 1.75 m:
      • Z-axis is vertical   (feet=0, head=height)
      • Y-axis is arm span   (arms extend ±Y in T-pose)
      • X-axis is depth      (front-to-back)

    Strategy: identify arm vertices as those whose |Y| is clearly beyond the
    torso width, then rotate them around an estimated shoulder joint.
    Works well for low-poly meshes where arm and torso vertices don't overlap.
    """
    if height < 1e-3 or len(pts) < 4:
        return

    z_min = float(pts[:, 2].min())

    # --- Estimate torso half-width from mid-body vertices ---
    torso_lo = z_min + height * 0.42
    torso_hi = z_min + height * 0.66
    torso_mask = (pts[:, 2] >= torso_lo) & (pts[:, 2] <= torso_hi)
    if torso_mask.sum() >= 2:
        torso_half_w = float(np.abs(pts[torso_mask, 1]).max())
    else:
        torso_half_w = height * 0.135      # fallback ≈ 0.24 m on 1.75 m model

    # Shoulder joint: at body edge, ~76 % of standing height
    shoulder_z    = z_min + height * 0.760
    shoulder_y    = torso_half_w            # shoulder is at the body edge in Y
    arm_threshold = torso_half_w * 1.22    # a bit beyond torso = clearly arm geometry

    # How much of the arm span is actually arm (not shoulder)?
    arm_half_reach = float(np.abs(pts[:, 1]).max()) - shoulder_y
    if arm_half_reach < 0.05:
        # Model arms are already folded or very short — nothing to do
        return

    FOLD_ANGLE_DEG = 82.0

    for side in (+1, -1):           # +1 = right arm (positive Y), −1 = left
        arm_mask = (pts[:, 1] * side) > arm_threshold
        if arm_mask.sum() == 0:
            continue

        # Vectors from the shoulder joint to each arm vertex
        Vy = pts[arm_mask, 1] - side * shoulder_y
        Vz = pts[arm_mask, 2] - shoulder_z

        # Rotation angle: fold arms downward.
        # For right arm (side=+1) rotate −82° (arm sweeps from +Y toward −Z).
        # For left  arm (side=−1) rotate +82° (symmetric).
        theta = np.radians(-FOLD_ANGLE_DEG * side)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        # 2-D rotation around X-axis through the shoulder joint:
        #  [Vy']   [ cos  −sin ] [Vy]
        #  [Vz'] = [ sin   cos ] [Vz]
        new_Vy = cos_t * Vy - sin_t * Vz
        new_Vz = sin_t * Vy + cos_t * Vz

        pts[arm_mask, 1] = side * shoulder_y + new_Vy
        pts[arm_mask, 2] = shoulder_z        + new_Vz


class PedMixin:
    # ------------------------------------------------------------------
    # Internal GLB template loader
    # ------------------------------------------------------------------

    def _load_ped_template(self) -> pv.PolyData | None:
        """Load low_poly_human.obj, scaled to 1.75 m standing height."""
        _MODELS_DIR = Path(__file__).parent / "assets" / "models"
        geo = _load_glb_as_polydata(_MODELS_DIR / "low_poly_human.obj")
        if geo is None:
            return None

        # Z extent = standing height → scale to 1.75 m.
        z_height = float(geo.bounds[5] - geo.bounds[4])
        if z_height < 1e-6:
            return None
        scale = 1.75 / z_height
        geo.points *= scale

        # Centre horizontally; keep feet at Z = 0.
        b = geo.bounds
        cx = (b[0] + b[1]) / 2.0
        cy = (b[2] + b[3]) / 2.0
        cz = float(b[4])
        geo.points -= np.array([cx, cy, cz], dtype=float)

        # Fold T-posed arms to a natural hanging position
        arm_span_before = float(geo.bounds[3] - geo.bounds[2])
        _fold_arms_inplace(geo.points, 1.75)
        arm_span_after  = float(geo.bounds[3] - geo.bounds[2])

        b = geo.bounds
        print(
            f"[peds] human template: "
            f"height={b[5]-b[4]:.2f}m  "
            f"arm-span {arm_span_before:.2f}m → {arm_span_after:.2f}m  "
            f"depth={b[1]-b[0]:.2f}m"
        )
        return geo

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_footway_paths(
        self, graph, z_level: float = 0.30
    ) -> tuple[list[dict], dict]:
        """Return (paths, outgoing) for pedestrian-walkable edges.

        Prefers footway/pedestrian/path/cycleway edges; falls back to every
        edge in the graph when fewer than 10 dedicated footway edges exist.
        """
        _PED_HW = {
            "footway", "pedestrian", "path", "cycleway",
            "steps", "bridleway", "track",
        }

        # Spatial index over registered crosswalk positions.
        _cross_pts = [
            [float(c["x"]), float(c["y"])]
            for c in graph.graph.get("crossings", [])
        ]
        _cross_tree = None
        if _cross_pts:
            from scipy.spatial import cKDTree as _CK
            _cross_tree = _CK(np.array(_cross_pts, dtype=float))

        def _build(edge_iter):
            paths: list[dict] = []
            outgoing: dict = {}
            for u, v, data in edge_iter:
                geom = data.get("geometry")
                if geom is not None and hasattr(geom, "coords"):
                    xy = np.asarray(geom.coords, dtype=float)[:, :2]
                else:
                    nu = graph.nodes.get(u, {}); nv = graph.nodes.get(v, {})
                    if "x" not in nu or "x" not in nv:
                        continue
                    xy = np.array(
                        [[float(nu["x"]), float(nu["y"])],
                         [float(nv["x"]), float(nv["y"])]],
                        dtype=float,
                    )
                if xy.shape[0] < 2:
                    continue
                keep = np.ones(xy.shape[0], dtype=bool)
                keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > 1e-6
                xy = xy[keep]
                if xy.shape[0] < 2:
                    continue

                pts = np.column_stack(
                    (xy, np.full(xy.shape[0], z_level, dtype=float))
                )
                seg_len = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)
                total = float(np.sum(seg_len))
                if total < 0.5:
                    continue
                cum = np.concatenate([[0.0], np.cumsum(seg_len)])

                is_cross_end = False
                if _cross_tree is not None:
                    nv_data = graph.nodes.get(v, {})
                    if "x" in nv_data and "y" in nv_data:
                        _d, _ = _cross_tree.query(
                            [float(nv_data["x"]), float(nv_data["y"])]
                        )
                        is_cross_end = bool(_d < 5.0)

                idx = len(paths)
                paths.append({
                    "u": u, "v": v,
                    "points": pts,
                    "cum_len": cum,
                    "length": total,
                    "is_crossing_end": is_cross_end,
                })
                outgoing.setdefault(u, []).append(idx)
            return paths, outgoing

        def _is_ped(data) -> bool:
            hw = data.get("highway")
            if isinstance(hw, (list, tuple, set)):
                return bool({str(x).lower() for x in hw} & _PED_HW)
            return str(hw or "").lower() in _PED_HW

        ped_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if _is_ped(d)]
        paths, outgoing = _build(ped_edges)

        if len(paths) < 10:
            print(
                f"[peds] only {len(paths)} footway edges — "
                "falling back to all road edges"
            )
            paths, outgoing = _build(graph.edges(data=True))

        return paths, outgoing

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_peds(self, n_peds: int = -1) -> None:
        self.ped_rng = np.random.default_rng(
            int(getattr(self.args, "seed", 42)) + 3751
        )
        self.ped_paths, self.ped_outgoing = self._extract_footway_paths(
            self.street_graph
        )

        if not self.ped_paths:
            print("[peds] no walkable edges — pedestrians disabled")
            self.ped_anim: dict = {"enabled": False, "active_crossings": []}
            return

        # ── Street crossings: link footway nodes facing each other across a
        # drivable road so pedestrians naturally cross streets (and are
        # exposed to traffic — see injury tracking in _advance_peds).
        try:
            from safety import build_crossing_paths, road_segments_from_paths
            _segs = road_segments_from_paths(getattr(self, "car_paths", []) or [])
            _new, _extra = build_crossing_paths(
                self.street_graph, self.ped_paths, _segs)
            if _new:
                self.ped_paths = self.ped_paths + _new
                for _k, _v in _extra.items():
                    self.ped_outgoing.setdefault(_k, []).extend(_v)
                print(f"[peds] {len(_new)//2} street crossings generated")
        except Exception as _exc:
            print(f"[peds] crossing generation skipped: {_exc}")

        n_paths = len(self.ped_paths)
        if n_peds == -1:
            n_peds = int(np.clip(n_paths, 50, 200))
        n_peds = max(0, int(n_peds))

        _by_uv = {(p["u"], p["v"]): i for i, p in enumerate(self.ped_paths)}
        self.ped_reverse: dict[int, int | None] = {
            i: _by_uv.get((p["v"], p["u"]))
            for i, p in enumerate(self.ped_paths)
        }

        probs = np.array([p["length"] for p in self.ped_paths], dtype=float)
        probs /= probs.sum()
        edge_idx = self.ped_rng.choice(n_paths, size=n_peds, replace=True, p=probs)
        dist = np.array(
            [float(self.ped_rng.uniform(0.0, self.ped_paths[int(e)]["length"]))
             for e in edge_idx],
            dtype=float,
        )
        speed = np.clip(
            1.2 + self.ped_rng.standard_normal(n_peds).astype(float) * 0.2,
            0.7, 1.8,
        )

        # Load human GLB template (None → fall back to sphere points)
        self.ped_template: pv.PolyData | None = self._load_ped_template()

        self.ped_anim = {
            "enabled": True,
            "edge_idx": edge_idx.astype(np.int64),
            "dist": dist,
            "speed": speed,
            "yield_timer": np.zeros(n_peds, dtype=float),
            "active_crossings": [],
            "injured": np.zeros(n_peds, dtype=bool),
            "last_t": time.perf_counter(),
        }
        self.scene_state["ped_injuries"] = 0
        print(f"[peds] {n_peds} agents on {n_paths} footway edges "
              f"({'GLB' if self.ped_template is not None else 'sphere'} mode)")

    # ------------------------------------------------------------------
    # Physics tick
    # ------------------------------------------------------------------

    def _advance_peds(self, dt: float) -> None:
        if not self.ped_anim.get("enabled"):
            return

        from safety import advance_peds_core, check_ped_car_collisions, ped_xy

        # Player-controlled pedestrian is frozen in IDM — its position comes from walk_pos
        _ctrl_idx = self.scene_state.get("walk_controlled_ped_idx")

        advance_peds_core(
            self.ped_anim, self.ped_paths, self.ped_outgoing,
            self.ped_reverse, self.ped_rng, dt, ctrl_idx=_ctrl_idx,
        )

        # ── Injury detection (IDM engine; SUMO handles its own collisions) ──
        if (self.scene_state.get("engine", "idm") == "idm"
                and bool(getattr(self, "car_anim", {}).get("enabled"))):
            car_pos = np.asarray(self.car_anim.get("pos", []), dtype=float)
            if car_pos.ndim == 2 and car_pos.shape[0] > 0:
                injured = self.ped_anim.get("injured")
                if injured is None:
                    injured = np.zeros(len(self.ped_anim["edge_idx"]), dtype=bool)
                    self.ped_anim["injured"] = injured
                pxy = ped_xy(self.ped_anim, self.ped_paths)
                hits, impact = check_ped_car_collisions(
                    pxy, injured,
                    car_pos[:, :2],
                    np.asarray(self.car_anim.get("speed", []), dtype=float),
                )
                if hits.shape[0]:
                    injured[hits] = True
                    total = int(np.count_nonzero(injured))
                    self.scene_state["ped_injuries"] = total
                    for h, s in zip(hits, impact):
                        sev = "SEVERE" if s >= 8.3 else "minor"
                        print(f"[safety] pedestrian #{int(h)} hit at "
                              f"{s*3.6:.0f} km/h ({sev}) — total injured: {total}")
                    try:
                        self.plotter.add_text(
                            f"⚠ pedestrians injured: {total}",
                            position=(0.68, 0.04), name="injury_hud",
                            font_size=11, viewport=True, color="#ff5544",
                        )
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Position & heading helpers
    # ------------------------------------------------------------------

    def _sample_ped_positions(self) -> np.ndarray:
        edge_idx = self.ped_anim["edge_idx"]
        dist     = self.ped_anim["dist"]
        n        = len(edge_idx)
        n_paths  = len(self.ped_paths)
        positions = np.zeros((n, 3), dtype=float)
        for i in range(n):
            eid = int(np.clip(edge_idx[i], 0, n_paths - 1))
            p   = self.ped_paths[eid]
            pts = p["points"]; cum = p["cum_len"]
            d   = float(np.clip(dist[i], 0.0, max(p["length"] - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            t   = (d - float(cum[seg])) / max(1e-9, float(cum[seg + 1]) - float(cum[seg]))
            positions[i] = pts[seg] + (pts[seg + 1] - pts[seg]) * t

        if n > 0 and self.scene_state.get("_terrain_drape_active"):
            _dem = getattr(self, "street_graph", None)
            if _dem is not None:
                _dem = _dem.graph.get("terrain_sampler")
            if _dem is not None:
                try:
                    positions[:, 2] = _dem(positions[:, :2]) + 0.30
                except Exception:
                    pass

        # Player-controlled pedestrian: substitute its position from walk_pos
        _ctrl_idx = self.scene_state.get("walk_controlled_ped_idx")
        _ctrl_pos = self.scene_state.get("walk_pos")
        if _ctrl_idx is not None and _ctrl_pos is not None and _ctrl_idx < n:
            positions[_ctrl_idx] = np.array(_ctrl_pos, dtype=float)

        return positions

    def _ped_headings_deg(self) -> np.ndarray:
        """Return heading angle (°) for each pedestrian (around Z axis)."""
        edge_idx = self.ped_anim["edge_idx"]
        dist     = self.ped_anim["dist"]
        n        = len(edge_idx)
        n_paths  = len(self.ped_paths)
        headings = np.zeros(n, dtype=float)
        for i in range(n):
            eid = int(np.clip(edge_idx[i], 0, n_paths - 1))
            p   = self.ped_paths[eid]
            pts = p["points"]; cum = p["cum_len"]
            d   = float(np.clip(dist[i], 0.0, max(p["length"] - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            p0, p1 = pts[seg], pts[seg + 1]
            h = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
            # +180° so model's -X "front" aligns with direction of travel
            headings[i] = (h + 180.0) % 360.0

        # Player-controlled pedestrian: face the direction of travel (walk_yaw)
        _ctrl_idx = self.scene_state.get("walk_controlled_ped_idx")
        if _ctrl_idx is not None and _ctrl_idx < n:
            headings[_ctrl_idx] = (self.scene_state.get("walk_yaw", 0.0) + 180.0) % 360.0

        return headings

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_peds(self, positions=None) -> None:
        if not self.ped_anim.get("enabled"):
            return

        if positions is None:
            positions = self._sample_ped_positions()
        n = positions.shape[0]
        if n == 0:
            return

        headings = self._ped_headings_deg()

        # ── Single glyph mapper for all pedestrians (GLB or cylinder) ────────
        # Created once on first call; mutated in-place every tick via numpy.
        # No actors or mesh copies are created during animation.
        gi = self.scene_state.get("_ped_glyph")
        if gi is None:
            from glyph_instance import GlyphInstances
            if self.ped_template is not None:
                template = self.ped_template
                mode_str = "GLB"
            else:
                template = pv.Cylinder(
                    center=(0.0, 0.0, 0.85), direction=(0, 0, 1),
                    radius=0.22, height=1.7, resolution=6,
                )
                mode_str = "cylinder"
            gi = GlyphInstances(template, n, "#d4a574", self.plotter)
            self.scene_state["_ped_glyph"] = gi
            print(f"[peds] glyph pool ready ({mode_str}, capacity≥{n})")

        # Injured pedestrians lie flat (90° X-tilt) at ground level
        tilt = None
        injured = self.ped_anim.get("injured")
        if injured is not None and np.any(injured):
            inj = np.asarray(injured, dtype=bool)[:n]
            tilt = np.zeros(n, dtype=float)
            tilt[inj] = 90.0
            positions = positions.copy()
            positions[inj, 2] = np.maximum(positions[inj, 2] - 0.55, 0.05)

        gi.update(positions, headings, tilt_x_deg=tilt)

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animate_peds(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        if not self.ped_anim.get("enabled"):
            return
        try:
            now  = time.perf_counter()
            last = float(self.ped_anim.get("last_t", now))
            dt   = float(np.clip(now - last, 0.0, 0.12))
            self.ped_anim["last_t"] = now
            if dt <= 0.0:
                return
            self._advance_peds(dt)
            self._render_peds()
        except Exception as exc:
            import traceback as _tb
            print(f"[peds-timer ERROR] {exc}")
            _tb.print_exc()
