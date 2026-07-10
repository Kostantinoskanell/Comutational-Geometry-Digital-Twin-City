from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pyvista as pv

from ped_mixin import _load_glb_as_polydata


# ---------------------------------------------------------------------------
# Build the composite bicycle + rider template mesh
# ---------------------------------------------------------------------------

def _build_cyclist_template(models_dir: Path) -> pv.PolyData | None:
    """
    Return a single PolyData from cyclist.obj.

    Coordinate frame after this function (matches the VTK/car convention):
      +X  → forward (direction of travel)
      +Y  → left
      +Z  → up
    """
    cyclist_geo = _load_glb_as_polydata(models_dir / "cyclist.obj")
    if cyclist_geo is None:
        print("[cyclists] cyclist.obj not found")
        return None

    # After axis correction in _load_glb_as_polydata, check dimensions.
    # We want the length (X) to be around 1.75m.
    # The original bounds might be length Y, but _load_glb_as_polydata might have swapped axes.
    # Let's scale it so its largest dimension (usually length) is 1.75m.
    dx = float(cyclist_geo.bounds[1] - cyclist_geo.bounds[0])
    dy = float(cyclist_geo.bounds[3] - cyclist_geo.bounds[2])
    dz = float(cyclist_geo.bounds[5] - cyclist_geo.bounds[4])
    
    max_len = max(dx, dy)
    if max_len > 1e-6:
        scale = 1.75 / max_len
        cyclist_geo.points *= scale

    # Centre horizontally; keep wheels at Z = 0.
    cx = (cyclist_geo.bounds[0] + cyclist_geo.bounds[1]) / 2.0
    cy = (cyclist_geo.bounds[2] + cyclist_geo.bounds[3]) / 2.0
    cz = float(cyclist_geo.bounds[4])
    cyclist_geo.points -= np.array([cx, cy, cz], dtype=float)

    total_h  = float(cyclist_geo.bounds[5] - cyclist_geo.bounds[4])
    total_l  = float(cyclist_geo.bounds[1] - cyclist_geo.bounds[0])
    print(
        f"[cyclists] loaded cyclist.obj template: "
        f"length={total_l:.2f} m  "
        f"height={total_h:.2f} m"
    )
    return cyclist_geo


# ---------------------------------------------------------------------------
# CyclistMixin
# ---------------------------------------------------------------------------

class CyclistMixin:
    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_cycleway_paths(
        self, graph, z_level: float = 0.32
    ) -> tuple[list[dict], dict]:
        """Return (paths, outgoing) for bicycle-accessible edges.

        Primary: highway=cycleway.
        Fallback: highway=path/track (shared-use paths) — applied when fewer
        than 5 dedicated cycleways exist.
        If still empty, fall back to all road edges so cyclists always appear.
        """
        _CYCLE_HW   = {"cycleway"}
        _SHARED_HW  = {"path", "track", "bridleway"}

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
                total   = float(np.sum(seg_len))
                if total < 1.0:
                    continue
                cum = np.concatenate([[0.0], np.cumsum(seg_len)])

                # Approximate max speed on each edge (used for IDM desired speed).
                hw = str(data.get("highway", "")).lower()
                if hw in {"cycleway"}:
                    maxspeed = 5.0      # 18 km/h — fast cycleway
                elif hw in {"path", "track"}:
                    maxspeed = 3.5      # 12.6 km/h — shared path
                else:
                    maxspeed = 4.0      # 14.4 km/h — general fallback

                idx = len(paths)
                paths.append({
                    "u": u, "v": v,
                    "points": pts, "cum_len": cum, "length": total,
                    "maxspeed_ms": maxspeed,
                    "highway": hw,
                })
                outgoing.setdefault(u, []).append(idx)
            return paths, outgoing

        def _hw_vals(data):
            hw = data.get("highway")
            if isinstance(hw, (list, tuple, set)):
                return {str(x).lower() for x in hw}
            return {str(hw or "").lower()}

        # Pass 1: dedicated cycleways
        cyc_edges  = [(u, v, d) for u, v, d in graph.edges(data=True)
                      if _hw_vals(d) & _CYCLE_HW]
        paths, outgoing = _build(cyc_edges)

        # Pass 2: shared paths
        if len(paths) < 5:
            shared_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                            if _hw_vals(d) & (_CYCLE_HW | _SHARED_HW)]
            paths, outgoing = _build(shared_edges)

        # Pass 3: all edges (universal fallback)
        if len(paths) < 5:
            print(f"[cyclists] only {len(paths)} cycle-tagged edges — using all roads")
            paths, outgoing = _build(graph.edges(data=True))

        return paths, outgoing

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_cyclists(self, n_cyclists: int = -1) -> None:
        """Initialise cyclist agents.  Call after street_graph is ready."""
        from idm import IDMParams, idm_tick as _idm_tick_fn   # local import
        self._idm_tick_fn = _idm_tick_fn

        self.cyclist_rng = np.random.default_rng(
            int(getattr(self.args, "seed", 42)) + 7193
        )
        self.cyclist_paths, self.cyclist_outgoing = self._extract_cycleway_paths(
            self.street_graph
        )

        if not self.cyclist_paths:
            print("[cyclists] no cycle edges — cyclists disabled")
            self.cyclist_anim: dict = {"enabled": False}
            return

        n_paths = len(self.cyclist_paths)
        if n_cyclists == -1:
            n_cyclists = int(np.clip(n_paths // 3, 10, 60))
        n_cyclists = max(0, int(n_cyclists))

        # Reverse edge map for dead-end bounce-backs
        _by_uv = {(p["u"], p["v"]): i for i, p in enumerate(self.cyclist_paths)}
        self.cyclist_reverse: dict[int, int | None] = {
            i: _by_uv.get((p["v"], p["u"]))
            for i, p in enumerate(self.cyclist_paths)
        }

        # Build car_next_edges structure (array of successor path indices per path)
        self.cyclist_next_edges: list[np.ndarray] = []
        for p in self.cyclist_paths:
            nexts = [
                j for j in self.cyclist_outgoing.get(p["v"], [])
            ]
            self.cyclist_next_edges.append(
                np.array(nexts, dtype=np.int64) if nexts else np.array([], dtype=np.int64)
            )

        probs = np.array([p["length"] for p in self.cyclist_paths], dtype=float)
        probs /= probs.sum()
        edge_idx = self.cyclist_rng.choice(n_paths, size=n_cyclists, replace=True, p=probs)

        desired_base = np.array(
            [float(self.cyclist_paths[int(e)]["maxspeed_ms"])
             * self.cyclist_rng.uniform(0.75, 1.0)
             for e in edge_idx],
            dtype=float,
        )
        speed = desired_base * self.cyclist_rng.uniform(0.3, 0.8, size=n_cyclists)

        # IDM params tuned for cyclists
        self.cyclist_idm_params = IDMParams(
            a_max=1.0,    # gentle acceleration (m/s²)
            b=1.5,        # comfortable braking
            T=1.2,        # headway (s)
            s0=1.0,       # minimum gap (m)
            delta=4,
        )

        # Bicycle length ≈ 1.75 m
        BIKE_LEN = 1.75

        self.cyclist_anim = {
            "enabled": True,
            "edge_idx": edge_idx.astype(np.int64),
            "dist": np.array(
                [float(self.cyclist_rng.uniform(
                    0.0, self.cyclist_paths[int(e)]["length"]))
                 for e in edge_idx],
                dtype=float,
            ),
            "speed": speed,
            "desired_speed": desired_base.copy(),
            "desired_speed_base": desired_base.copy(),
            "accel": np.zeros(n_cyclists, dtype=float),
            "car_len": np.full(n_cyclists, BIKE_LEN, dtype=float),
            "stop_wait": np.zeros(n_cyclists, dtype=float),
            "planned_edges": None,
            "planned_cursor": None,
            "last_t": time.perf_counter(),
        }

        # Load composite bicycle + rider template
        _MODELS_DIR = Path(__file__).parent / "assets" / "models"
        self.cyclist_template: pv.PolyData | None = _build_cyclist_template(_MODELS_DIR)

        print(
            f"[cyclists] {n_cyclists} agents on {n_paths} cycle edges "
            f"({'GLB' if self.cyclist_template is not None else 'diamond'} mode)"
        )

    # ------------------------------------------------------------------
    # Physics tick  (IDM — same engine as cars, lower params)
    # ------------------------------------------------------------------

    def _advance_cyclists(self, dt: float) -> None:
        if not self.cyclist_anim.get("enabled"):
            return
        _hour = float(self.scene_state.get("hour", 12.0))
        self._idm_tick_fn(
            car_anim             = self.cyclist_anim,
            car_paths            = self.cyclist_paths,
            car_next_edges       = self.cyclist_next_edges,
            traffic_lights       = self.traffic_lights_dict,
            dt                   = dt,
            params               = self.cyclist_idm_params,
            rng                  = self.cyclist_rng,
            traffic_speed        = float(getattr(self.args, "traffic_speed", 1.0)),
            roundabout_yield_map = {},
        )

    # ------------------------------------------------------------------
    # Position & heading helpers
    # ------------------------------------------------------------------

    def _sample_cyclist_positions(self) -> np.ndarray:
        edge_idx = self.cyclist_anim["edge_idx"]
        dist     = self.cyclist_anim["dist"]
        n        = len(edge_idx)
        n_paths  = len(self.cyclist_paths)
        positions = np.zeros((n, 3), dtype=float)
        for i in range(n):
            eid = int(np.clip(edge_idx[i], 0, n_paths - 1))
            p   = self.cyclist_paths[eid]
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
                    positions[:, 2] = _dem(positions[:, :2]) + 0.32
                except Exception:
                    pass
        return positions

    def _cyclist_headings_deg(self) -> np.ndarray:
        edge_idx = self.cyclist_anim["edge_idx"]
        dist     = self.cyclist_anim["dist"]
        n        = len(edge_idx)
        n_paths  = len(self.cyclist_paths)
        headings = np.zeros(n, dtype=float)
        for i in range(n):
            eid = int(np.clip(edge_idx[i], 0, n_paths - 1))
            p   = self.cyclist_paths[eid]
            pts = p["points"]; cum = p["cum_len"]
            d   = float(np.clip(dist[i], 0.0, max(p["length"] - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            p0, p1 = pts[seg], pts[seg + 1]
            h = float(np.degrees(np.arctan2(
                float(p1[1] - p0[1]), float(p1[0] - p0[0])
            )))
            headings[i] = (h + 180.0) % 360.0
        return headings

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    # Bright-green diamond glyph used when the GLB template is unavailable.
    _CYCLIST_DIAMOND: pv.PolyData | None = None

    @classmethod
    def _diamond_glyph(cls) -> pv.PolyData:
        if cls._CYCLIST_DIAMOND is None:
            # Elongated octahedron (diamond shape) ≈ bicycle silhouette
            pts = np.array([
                [ 0.9,  0.0,  0.4],  # nose
                [-0.9,  0.0,  0.4],  # tail
                [ 0.0,  0.3,  0.4],  # left
                [ 0.0, -0.3,  0.4],  # right
                [ 0.0,  0.0,  1.2],  # top
                [ 0.0,  0.0,  0.0],  # bottom
            ], dtype=float)
            faces = np.array([
                3, 0, 2, 4,  3, 0, 4, 3,
                3, 1, 4, 2,  3, 1, 3, 4,
                3, 0, 2, 5,  3, 0, 5, 3,
                3, 1, 5, 2,  3, 1, 3, 5,
            ], dtype=np.int64)
            cls._CYCLIST_DIAMOND = pv.PolyData(pts, faces)
        return cls._CYCLIST_DIAMOND

    def _render_cyclists(self, positions=None) -> None:
        if not self.cyclist_anim.get("enabled"):
            return

        if positions is None:
            positions = self._sample_cyclist_positions()
        n = positions.shape[0]
        if n == 0:
            return

        headings = self._cyclist_headings_deg()

        # ── Single glyph mapper for all cyclists (GLB or diamond) ────────────
        # Created once; mutated in-place every tick — no actor creation during animation.
        gi = self.scene_state.get("_cyclist_glyph")
        if gi is None:
            from glyph_instance import GlyphInstances
            use_glb  = self.cyclist_template is not None
            template = self.cyclist_template if use_glb else self._diamond_glyph()
            color    = "#4db848" if use_glb else "#27ae60"
            gi = GlyphInstances(template, n, color, self.plotter)
            self.scene_state["_cyclist_glyph"] = gi
            print(f"[cyclists] glyph pool ready "
                  f"({'GLB' if use_glb else 'diamond'}, capacity≥{n})")

        gi.update(positions, headings)

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animate_cyclists(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        if not self.cyclist_anim.get("enabled"):
            return
        try:
            now  = time.perf_counter()
            last = float(self.cyclist_anim.get("last_t", now))
            dt   = float(np.clip(now - last, 0.0, 0.12))
            self.cyclist_anim["last_t"] = now
            if dt <= 0.0:
                return
            self._advance_cyclists(dt)
            self._render_cyclists()
        except Exception as exc:
            import traceback as _tb
            print(f"[cyclists-timer ERROR] {exc}")
            _tb.print_exc()
