"""analysis_mixin.py — Traffic analysis overlays for the city digital twin.

Three overlays, each toggled by a dedicated key:

  H  — Traffic density heatmap
         Accumulates car XY positions into a 2D rolling histogram over the
         last N ticks, applies a Gaussian blur, and maps the result to a
         deep-blue → red colour ramp written into a flat grid mesh above
         the ground.

  N  — Noise pollution map
         For every grid cell, sums the distance-attenuated dB contribution
         of every moving car using a simplified CNOSSOS-EU road-noise model
         (L_w = L_ref + 10·log₁₀(v/50 km·h⁻¹) per vehicle; spherical
         spreading −20·log₁₀(d)).  Cells breaching 65 dB(A) are counted
         and shown in the status bar.

  C  — Network centrality
         Runs networkx.edge_betweenness_centrality once on the street graph
         and colours each road segment from dark-purple (low) to bright-
         yellow (high backbone edge) using the plasma palette.  Computed
         once on first toggle; subsequent toggles show/hide the cached actor.
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    pass


# ── CNOSSOS-EU constants ──────────────────────────────────────────────────────
# 85 dB(A) at 1 m ≈ real passenger-car pass-by at 50 km/h (CNOSSOS source power
# ~95 dB minus hemispherical ground correction).  Gives 65 dB at 10 m and
# 45 dB at 100 m — matching measured urban road-noise footprints.
_L_REF       = 85.0   # dB(A) reference for a single car at 1 m, 50 km/h
_BREACH_DB   = 65.0   # WHO / EU day-time breach threshold

def _collapse_edge_centrality(ec: dict) -> dict:
    """Normalize edge_betweenness_centrality keys to (u, v) pairs.

    On a MultiDiGraph networkx returns (u, v, key) triples; a plain (u, v)
    lookup then misses every edge and the whole overlay renders as zero.
    Keeps the strongest parallel edge per (u, v) pair.
    """
    ec_uv: dict = {}
    for k, val in ec.items():
        uv = (k[0], k[1])
        ec_uv[uv] = max(float(val), ec_uv.get(uv, 0.0))
    return ec_uv


# ── Grid configuration ────────────────────────────────────────────────────────
_GRID_CELL   = 4.0    # metres per analysis-grid cell
_GRID_Z      = 0.35   # height above z=0 (avoids z-fighting with ground mesh)
_HEAT_TICKS  = 180    # rolling window length in animation ticks
_HEAT_SIGMA  = 2.5    # Gaussian kernel σ in grid cells


class AnalysisMixin:
    """Mixin that adds heatmap, noise, and centrality overlays to DigitalTwinApp."""

    # ─────────────────────────────────────────────────────────────────────────
    # One-time setup
    # ─────────────────────────────────────────────────────────────────────────

    def _init_analysis(self) -> None:
        """Build the analysis grid and zero the accumulators.  Call once after
        scene geometry is loaded."""
        self.scene_state.setdefault("heatmap_on", False)
        self.scene_state.setdefault("noise_on",   False)
        self.scene_state.setdefault("centrality_on", False)

        # ── Scene bounding box ──────────────────────────────────────────────
        gm = getattr(self, "ground_mesh", None)
        if gm is not None and gm.n_cells > 0:
            b = gm.bounds                  # (xmin,xmax, ymin,ymax, zmin,zmax)
            xmin, xmax = float(b[0]), float(b[1])
            ymin, ymax = float(b[2]), float(b[3])
        else:
            xs = [float(d["x"]) for _, d in self.street_graph.nodes(data=True) if "x" in d]
            ys = [float(d["y"]) for _, d in self.street_graph.nodes(data=True) if "y" in d]
            pad = 20.0
            xmin, xmax = min(xs) - pad, max(xs) + pad
            ymin, ymax = min(ys) - pad, max(ys) + pad

        # Small padding so cars near the edge are still captured
        margin = 8.0
        xmin -= margin; xmax += margin
        ymin -= margin; ymax += margin

        nx = max(4, int(math.ceil((xmax - xmin) / _GRID_CELL)))
        ny = max(4, int(math.ceil((ymax - ymin) / _GRID_CELL)))
        self._ana_xmin, self._ana_xmax = xmin, xmax
        self._ana_ymin, self._ana_ymax = ymin, ymax
        self._ana_nx, self._ana_ny = nx, ny

        # ── Flat ImageData grid ─────────────────────────────────────────────
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, 1)
        grid.origin     = (xmin, ymin, _GRID_Z)
        grid.spacing    = (_GRID_CELL, _GRID_CELL, 0.1)
        n_cells = nx * ny
        grid.cell_data["value"] = np.zeros(n_cells, dtype=float)
        self._ana_grid    = grid
        self._ana_n_cells = n_cells

        # ── Cell centres for vectorised distance calculations ───────────────
        ix = np.tile(np.arange(nx), ny)
        iy = np.repeat(np.arange(ny), nx)
        self._ana_cx = xmin + (ix + 0.5) * _GRID_CELL
        self._ana_cy = ymin + (iy + 0.5) * _GRID_CELL

        # ── Heatmap rolling history ─────────────────────────────────────────
        self._heat_history: deque = deque(maxlen=_HEAT_TICKS)

        # ── Centrality cache ────────────────────────────────────────────────
        self._centrality_mesh: pv.PolyData | None = None

        # ── Actor slots ────────────────────────────────────────────────────
        self.scene_state["heatmap_actor"]    = None
        self.scene_state["noise_actor"]      = None
        self.scene_state["centrality_actor"] = None

        print(f"[analysis] grid {nx}×{ny} = {n_cells} cells @ {_GRID_CELL} m/cell")

    # ─────────────────────────────────────────────────────────────────────────
    # Toggle callbacks (bound to keyboard keys)
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_heatmap(self) -> None:
        on = not bool(self.scene_state.get("heatmap_on", False))
        self.scene_state["heatmap_on"] = on
        if not on:
            self._remove_analysis_actor("heatmap_actor")
            self.plotter.add_text("", position=(0.18, 0.04), name="noise_status", viewport=True)
            self.plotter.render()
            print("[analysis] heatmap OFF")
        else:
            # Noise and heatmap share the grid — can't both be on at once
            if self.scene_state.get("noise_on"):
                self.scene_state["noise_on"] = False
                self._remove_analysis_actor("noise_actor")
                try:
                    self.plotter.remove_scalar_bar("dB(A)")
                except Exception:
                    pass
            self._heat_history.clear()
            print("[analysis] heatmap ON  (accumulating …)")

    def _toggle_noise_map(self) -> None:
        on = not bool(self.scene_state.get("noise_on", False))
        self.scene_state["noise_on"] = on
        if not on:
            self._remove_analysis_actor("noise_actor")
            try:
                self.plotter.remove_scalar_bar("dB(A)")
            except Exception:
                pass
            self.plotter.add_text("", position=(0.18, 0.04), name="noise_status", viewport=True)
            self.plotter.render()
            print("[analysis] noise map OFF")
        else:
            if self.scene_state.get("heatmap_on"):
                self.scene_state["heatmap_on"] = False
                self._remove_analysis_actor("heatmap_actor")
            print("[analysis] noise map ON")

    def _toggle_centrality(self) -> None:
        on = not bool(self.scene_state.get("centrality_on", False))
        self.scene_state["centrality_on"] = on
        if not on:
            self._remove_analysis_actor("centrality_actor")
            try:
                self.plotter.remove_scalar_bar("Betweenness")
            except Exception:
                pass
            self.plotter.render()
            print("[analysis] centrality OFF")
        else:
            # Mutually exclusive: turn off heatmap / noise before showing centrality
            if self.scene_state.get("heatmap_on"):
                self.scene_state["heatmap_on"] = False
                self._remove_analysis_actor("heatmap_actor")
                self.plotter.add_text("", position=(0.18, 0.04), name="noise_status", viewport=True)
            if self.scene_state.get("noise_on"):
                self.scene_state["noise_on"] = False
                self._remove_analysis_actor("noise_actor")
                try:
                    self.plotter.remove_scalar_bar("dB(A)")
                except Exception:
                    pass
                self.plotter.add_text("", position=(0.18, 0.04), name="noise_status", viewport=True)
            self._build_centrality_overlay()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-tick update (called from _animate_cars every N ticks)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_traffic_positions_and_speeds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos: Nx3 float, speed: N float) from the active traffic engine.

        In sumo engine mode, reads from the SUMO snapshot so all analysis overlays
        (heatmap, noise) reflect SUMO vehicle positions and speeds rather than the
        IDM car_anim arrays.
        """
        engine = self.scene_state.get("engine", "idm")
        if engine == "sumo":
            sumo = getattr(self, "sumo", None)
            if sumo and sumo.get("enabled"):
                snap = sumo.get("_snapshot") or []
                tf   = sumo.get("transformer")
                if snap and tf is not None:
                    pts:  list[list[float]] = []
                    spds: list[float]       = []
                    for v in snap:
                        try:
                            x, y = tf.transform(v["lon"], v["lat"])
                            pts.append([x, y, 0.0])
                            spds.append(float(v.get("speed", 0.0)))
                        except Exception:
                            pass
                    if pts:
                        return (np.array(pts,  dtype=float),
                                np.array(spds, dtype=float))
            # SUMO enabled but snapshot empty yet — return empty arrays
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

        # IDM engine (default)
        pos   = np.asarray(self.car_anim.get("pos",   []), dtype=float)
        speed = np.asarray(self.car_anim.get("speed", []), dtype=float)
        if pos.ndim != 2:
            pos = np.zeros((0, 3), dtype=float)
        if speed.shape[0] != pos.shape[0]:
            speed = np.zeros(pos.shape[0], dtype=float)
        return pos, speed

    def _update_analysis(self) -> None:
        """Update whichever overlay is currently active.  Safe to call every tick."""
        if not hasattr(self, "_ana_grid"):
            return

        pos, speed = self._get_traffic_positions_and_speeds()
        if pos.ndim != 2 or pos.shape[0] == 0:
            return

        if self.scene_state.get("heatmap_on"):
            self._tick_heatmap(pos)

        if self.scene_state.get("noise_on"):
            self._tick_noise(pos, speed)

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap internals
    # ─────────────────────────────────────────────────────────────────────────

    def _tick_heatmap(self, pos: np.ndarray) -> None:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return

        # ── Bin car positions into grid cells ──────────────────────────────
        xi = np.floor((pos[:, 0] - self._ana_xmin) / _GRID_CELL).astype(np.int64)
        yi = np.floor((pos[:, 1] - self._ana_ymin) / _GRID_CELL).astype(np.int64)
        valid = (xi >= 0) & (xi < self._ana_nx) & (yi >= 0) & (yi < self._ana_ny)
        cell_ids = yi[valid] * self._ana_nx + xi[valid]

        frame = np.zeros(self._ana_n_cells, dtype=np.float32)
        np.add.at(frame, cell_ids, 1.0)
        self._heat_history.append(frame)

        # ── Rolling sum → Gaussian blur → normalise ────────────────────────
        acc = np.sum(self._heat_history, axis=0).astype(float)
        blurred = gaussian_filter(acc.reshape(self._ana_ny, self._ana_nx),
                                  sigma=_HEAT_SIGMA).reshape(-1)
        mx = float(blurred.max())
        if mx > 0:
            blurred /= mx

        self._ana_grid.cell_data["value"] = blurred
        self._ana_grid.Modified()

        if self.scene_state.get("heatmap_actor") is None:
            act = self.plotter.add_mesh(
                self._ana_grid,
                scalars="value",
                clim=[0.0, 1.0],
                cmap="hot",
                opacity=0.55,
                show_scalar_bar=False,
                lighting=False,
                reset_camera=False,
                name="heatmap_overlay",
            )
            self.scene_state["heatmap_actor"] = act

    # ─────────────────────────────────────────────────────────────────────────
    # Noise internals
    # ─────────────────────────────────────────────────────────────────────────

    def _tick_noise(self, pos: np.ndarray, speed: np.ndarray) -> None:
        n_cars   = pos.shape[0]
        n_cells  = self._ana_n_cells

        # ── Distance matrix (n_cars × n_cells) — pure NumPy broadcasting ──
        dx = pos[:, 0:1] - self._ana_cx[np.newaxis, :]
        dy = pos[:, 1:2] - self._ana_cy[np.newaxis, :]
        d  = np.sqrt(dx * dx + dy * dy)
        np.maximum(d, 1.0, out=d)           # 1 m minimum distance

        # ── CNOSSOS-EU simplified A-weighted emission level ────────────────
        v_kmh = np.maximum(speed * 3.6, 1.0)                    # (n_cars,)
        lw    = _L_REF + 10.0 * np.log10(v_kmh / 50.0)         # (n_cars,)

        # ── Spherical spreading + incoherent summation ─────────────────────
        l_at = lw[:, np.newaxis] - 20.0 * np.log10(d)          # (n_cars, n_cells)
        total_lin = np.sum(np.power(10.0, l_at / 10.0), axis=0) # (n_cells,)
        noise_db  = 10.0 * np.log10(np.maximum(total_lin, 1e-15))

        # Only colour cells that are genuinely noisy — quiet background shows transparent.
        # 45 dB cutoff = single car at ~100 m; halo radius grows with speed and
        # with the number of overlapping cars (+3 dB per doubling).
        display = noise_db.copy()
        display[display < 45.0] = np.nan

        self._ana_grid.cell_data["value"] = display
        self._ana_grid.Modified()

        if self.scene_state.get("noise_actor") is None:
            act = self.plotter.add_mesh(
                self._ana_grid,
                scalars="value",
                clim=[45.0, 78.0],
                cmap="RdYlGn_r",
                nan_opacity=0.0,
                opacity=0.72,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "dB(A)", "vertical": True,
                    "position_x": 0.88, "position_y": 0.25,
                    "height": 0.40,    "width": 0.055,
                    "label_font_size": 9, "title_font_size": 10,
                    "color": "white",
                },
                lighting=False,
                reset_camera=False,
                name="noise_overlay",
            )
            self.scene_state["noise_actor"] = act

        n_breach = int(np.count_nonzero(noise_db >= _BREACH_DB))
        self.plotter.add_text(
            f"Noise  |  ≥{int(_BREACH_DB)} dB breach: {n_breach} cells",
            position=(0.18, 0.04),
            name="noise_status",
            font_size=9,
            viewport=True,
            color="#ff6644",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Centrality internals
    # ─────────────────────────────────────────────────────────────────────────

    def _build_centrality_overlay(self) -> None:
        """Compute edge betweenness centrality and render coloured road tubes."""
        import networkx as nx

        # Remove stale actor
        self._remove_analysis_actor("centrality_actor")

        # Re-use cached mesh if available (centrality is expensive)
        if self._centrality_mesh is not None:
            print("[analysis] centrality: using cached mesh")
        else:
            print("[analysis] centrality: computing edge betweenness …")
            try:
                ec = nx.edge_betweenness_centrality(
                    self.street_graph, normalized=True, weight="length"
                )
            except Exception as exc:
                print(f"[analysis] centrality failed: {exc}")
                self.scene_state["centrality_on"] = False
                return
            print(f"[analysis] centrality done  ({len(ec)} edges)")

            ec_uv = _collapse_edge_centrality(ec)

            ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}
            pts_list: list[np.ndarray] = []
            val_list: list[float]      = []

            for u, v, data in self.street_graph.edges(data=True):
                hw = data.get("highway", "")
                if isinstance(hw, (list, tuple)):
                    hw = hw[0] if hw else ""
                if str(hw).lower() in ped_tags:
                    continue

                # Look up centrality for either direction
                val = max(ec_uv.get((u, v), 0.0), ec_uv.get((v, u), 0.0))

                geom = data.get("geometry")
                if geom is not None and hasattr(geom, "coords"):
                    coords = np.asarray(geom.coords, dtype=float)
                    if coords.shape[0] >= 2:
                        xy = coords[:, :2]
                        for i in range(xy.shape[0] - 1):
                            pts_list.append(np.array([xy[i,0],   xy[i,1],   0.15]))
                            pts_list.append(np.array([xy[i+1,0], xy[i+1,1], 0.15]))
                            val_list.append(val)
                        continue

                nu = self.street_graph.nodes.get(u, {})
                nv = self.street_graph.nodes.get(v, {})
                if "x" in nu and "x" in nv:
                    pts_list.append(np.array([float(nu["x"]), float(nu["y"]), 0.15]))
                    pts_list.append(np.array([float(nv["x"]), float(nv["y"]), 0.15]))
                    val_list.append(val)

            if not pts_list:
                print("[analysis] centrality: no edges to display")
                self.scene_state["centrality_on"] = False
                return

            all_pts = np.vstack(pts_list)
            n_segs  = len(val_list)
            lines   = np.empty((n_segs, 3), dtype=np.int64)
            lines[:, 0] = 2
            lines[:, 1] = np.arange(0, 2 * n_segs, 2)
            lines[:, 2] = np.arange(1, 2 * n_segs, 2)

            mesh = pv.PolyData()
            mesh.points = all_pts
            mesh.lines  = lines.ravel()
            # One value per point (both endpoints of a segment get the same value)
            mesh.point_data["centrality"] = np.repeat(np.asarray(val_list, dtype=float), 2)

            self._centrality_mesh = mesh

        act = self.plotter.add_mesh(
            self._centrality_mesh,
            scalars="centrality",
            cmap="plasma",
            line_width=4,
            render_lines_as_tubes=True,
            opacity=0.88,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Betweenness", "vertical": True,
                "position_x": 0.01,    "position_y": 0.25,
                "height": 0.40,        "width": 0.055,
                "label_font_size": 9,  "title_font_size": 10,
                "color": "white",
            },
            lighting=False,
            reset_camera=False,
            name="centrality_overlay",
        )
        self.scene_state["centrality_actor"] = act
        self.plotter.render()
        print("[analysis] centrality overlay added")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _remove_analysis_actor(self, key: str) -> None:
        act = self.scene_state.get(key)
        if act is not None:
            try:
                self.plotter.remove_actor(act, reset_camera=False)
            except Exception:
                pass
            self.scene_state[key] = None
