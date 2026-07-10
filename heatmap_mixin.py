"""HeatmapMixin — live air-quality / noise heatmap overlay.

Samples car and bus positions once per second, accumulates them into a 60×60
grid using np.histogram2d, applies exponential smoothing, and renders an
ImageData overlay at Z=5 m.

Noise model:  contribution = n_cars * 1.0 + n_buses * 3.0  per grid cell.
  (buses are ~3× louder / more polluting than a single car)

The colour map runs green → yellow → red with alpha ramping from 0 (empty
cells vanish completely) to 0.65 (peak density = solid red).

Key 'q'  — toggle overlay on/off  (overrides camera rotate-left).
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv

import emissions as _em

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:
    _gaussian_filter = None

# Human-readable pollutant labels for the HUD.
_POLLUTANT_LABEL = {
    "nox": "NOx",
    "co2": "CO₂",
    "pm":  "PM2.5",
}

# Build custom RGBA cmap once at import time so we don't depend on matplotlib
# being available at the call site.  256-step lookup: RGBA float [0,1].
try:
    import matplotlib.cm as _mcm
    import matplotlib.colors as _mc

    _base_vals = _mcm.get_cmap("RdYlGn_r", 256)(np.linspace(0.0, 1.0, 256))
    # Alpha ramp: 0 for the lowest 8 % of the range, then 0 → 0.65
    _alphas = np.clip(np.linspace(-0.12, 0.70, 256), 0.0, 0.70)
    _base_vals[:, 3] = _alphas
    _HEATMAP_CMAP = _mc.ListedColormap(_base_vals, name="aq_rg")
except Exception:
    _HEATMAP_CMAP = "RdYlGn_r"   # fallback: no alpha ramp but still works


GRID_W = 60   # cells in X
GRID_H = 60   # cells in Y
_SMOOTH_ALPHA = 0.30   # EMA weight for new sample  (higher = faster response)


class HeatmapMixin:

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_aq_overlay(self) -> None:
        """Create the ImageData grid and add it to the plotter (invisible at first)."""
        self._hm_visible  = False
        self._hm_grid: pv.ImageData | None = None
        self._hm_actor    = None
        self._hm_smooth   = None       # np.ndarray (GRID_W*GRID_H,), EMA state
        self._hm_last_t   = 0.0
        self._hm_clim_max = 0.01       # auto-expands with observed emission mass
        # Pollutant cycling: 'q' steps None(off) → nox → co2 → pm → None
        self._hm_pollutant = None
        self._hm_pollutant_cycle = ["nox", "co2", "pm"]
        # Gaussian dispersion sigma in grid cells (set once bounds known)
        self._hm_disp_sigma = 1.3

        if self.buildings_mesh is None or self.buildings_mesh.n_points == 0:
            print("[heatmap] buildings mesh empty — heatmap skipped")
            return

        bx0, bx1, by0, by1 = self.buildings_mesh.bounds[:4]
        pad = 40.0
        x0, x1 = bx0 - pad, bx1 + pad
        y0, y1 = by0 - pad, by1 + pad

        dx = (x1 - x0) / GRID_W
        dy = (y1 - y0) / GRID_H

        self._hm_x0, self._hm_y0 = x0, y0
        self._hm_x1, self._hm_y1 = x1, y1
        self._hm_dx, self._hm_dy = dx, dy

        # ImageData: (GRID_W+1, GRID_H+1, 2) → n_cells = GRID_W * GRID_H * 1
        grid = pv.ImageData()
        grid.dimensions = (GRID_W + 1, GRID_H + 1, 2)
        grid.origin     = (x0, y0, 5.0)
        grid.spacing    = (dx, dy, 0.10)   # 10 cm thick slab at Z=5m

        n_cells = GRID_W * GRID_H
        init_data = np.full(n_cells, np.nan, dtype=np.float32)
        grid.cell_data["aq"] = init_data

        self._hm_grid   = grid
        self._hm_smooth = np.zeros(n_cells, dtype=float)

        try:
            actor = self.plotter.add_mesh(
                grid,
                scalars="aq",
                cmap=_HEATMAP_CMAP,
                clim=[0.0, self._hm_clim_max],
                show_scalar_bar=False,
                lighting=False,
                opacity=1.0,           # alpha baked into colormap
                reset_camera=False,
                name="air_heatmap",
            )
            actor.VisibilityOff()       # hidden until user presses 'q'
            self._hm_actor = actor
            print(
                f"[heatmap] grid {GRID_W}×{GRID_H}  cell {dx:.1f}×{dy:.1f} m"
                f"  — press 'q' to toggle"
            )
        except Exception as exc:
            print(f"[heatmap] init failed: {exc}")
            self._hm_grid = None

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def _toggle_aq_overlay(self) -> None:
        """Cycle: off → NOx → CO₂ → PM2.5 → off."""
        if self._hm_actor is None:
            return

        cyc = self._hm_pollutant_cycle
        if self._hm_pollutant is None:
            self._hm_pollutant = cyc[0]
        else:
            i = cyc.index(self._hm_pollutant)
            self._hm_pollutant = cyc[i + 1] if i + 1 < len(cyc) else None

        self._hm_visible = self._hm_pollutant is not None
        # Reset accumulation + scale when switching pollutant (different units)
        if self._hm_smooth is not None:
            self._hm_smooth[:] = 0.0
        self._hm_clim_max = 0.01

        try:
            if self._hm_visible:
                self._hm_actor.VisibilityOn()
            else:
                self._hm_actor.VisibilityOff()
                self.plotter.add_text("", position=(0.34, 0.95),
                                      name="aq_label", viewport=True)
            self.plotter.render()
        except Exception:
            pass

        if self._hm_visible:
            print(f"[heatmap] air-quality: {_POLLUTANT_LABEL[self._hm_pollutant]} "
                  f"emission rate (dispersed)")
        else:
            print("[heatmap] OFF")

    # ------------------------------------------------------------------
    # Density sampling
    # ------------------------------------------------------------------

    def _sample_heatmap_density(self) -> np.ndarray:
        """Accumulate per-cell emission rate (g/s) of the active pollutant, then
        apply Gaussian dispersion.  Returns (GRID_W*GRID_H,) row-major array."""
        pollutant = self._hm_pollutant or "nox"
        x_edges = np.linspace(self._hm_x0, self._hm_x1, GRID_W + 1)
        y_edges = np.linspace(self._hm_y0, self._hm_y1, GRID_H + 1)

        grid = np.zeros((GRID_H, GRID_W), dtype=float)

        # --- IDM cars (passenger class) — skip in sumo engine mode ---
        if (self.scene_state.get("engine", "idm") == "idm"
                and hasattr(self, "car_anim") and bool(self.car_anim.get("enabled"))):
            edge_idx = np.asarray(self.car_anim.get("edge_idx", []), dtype=np.int64)
            dist_arr = np.asarray(self.car_anim.get("dist",     []), dtype=float)
            speed_arr= np.asarray(self.car_anim.get("speed",    []), dtype=float)

            if len(edge_idx) > 0:
                car_xy = np.zeros((len(edge_idx), 2), dtype=float)
                for i, (ei, d) in enumerate(zip(edge_idx, dist_arr)):
                    try:
                        p = self._car_pose_on_path(self.car_paths[int(ei)], float(d))
                        car_xy[i, 0] = float(p[0]); car_xy[i, 1] = float(p[1])
                    except Exception:
                        pass
                rate = _em.emission_rate_g_per_s(pollutant, speed_arr, "passenger")
                h, _, _ = np.histogram2d(
                    car_xy[:, 1], car_xy[:, 0],
                    bins=[y_edges, x_edges], weights=np.asarray(rate, dtype=float),
                )
                grid += h

        # --- Buses (diesel bus class, real per-bus speed) ---
        if hasattr(self, "buses") and self.buses:
            try:
                bus_pos = self._sample_bus_positions()   # (N,3)
                bus_spd = np.array([float(b.get("speed", 0.0)) for b in self.buses], dtype=float)
                if len(bus_pos) > 0:
                    rate = _em.emission_rate_g_per_s(pollutant, bus_spd, "bus")
                    h_bus, _, _ = np.histogram2d(
                        bus_pos[:, 1], bus_pos[:, 0],
                        bins=[y_edges, x_edges], weights=np.asarray(rate, dtype=float),
                    )
                    grid += h_bus
            except Exception:
                pass

        # --- SUMO co-sim vehicles ---
        # Prefer SUMO's NATIVE HBEFA emission model (accounts for acceleration,
        # not just speed); fall back to the speed-based EEA approximation.
        sumo = getattr(self, "sumo", None)
        if sumo and sumo.get("enabled"):
            try:
                tf = sumo.get("transformer")
                conn = sumo.get("conn")
                native = []
                if conn is not None and hasattr(conn, "vehicle_emission_snapshot"):
                    native = conn.vehicle_emission_snapshot(pollutant)
                if native and tf is not None:
                    sx, sy, srate = [], [], []
                    for v in native:
                        x, y = tf.transform(v["lon"], v["lat"])
                        sx.append(y); sy.append(x)   # histogram2d(row=Y, col=X)
                        srate.append(float(v["rate"]))
                    h_s, _, _ = np.histogram2d(
                        np.array(sx), np.array(sy),
                        bins=[y_edges, x_edges], weights=np.array(srate),
                    )
                    grid += h_s
                else:
                    snap = sumo.get("_snapshot", []) or []
                    if snap and tf is not None:
                        sx, sy, srate = [], [], []
                        for v in snap:
                            x, y = tf.transform(v["lon"], v["lat"])
                            sx.append(y); sy.append(x)
                            srate.append(float(_em.emission_rate_g_per_s(
                                pollutant, float(v.get("speed", 0.0)), v.get("vclass", "passenger"))))
                        h_s, _, _ = np.histogram2d(
                            np.array(sx), np.array(sy),
                            bins=[y_edges, x_edges], weights=np.array(srate),
                        )
                        grid += h_s
            except Exception:
                pass

        # --- Gaussian dispersion (pollutant spreads from the road) ---
        if _gaussian_filter is not None and grid.any():
            grid = _gaussian_filter(grid, sigma=self._hm_disp_sigma, mode="constant")

        return grid.ravel().astype(float)   # row-major → matches ImageData cell order

    # ------------------------------------------------------------------
    # Animation callback (fires every car_timer_ms, throttled inside to 1 Hz)
    # ------------------------------------------------------------------

    def _animate_aq_overlay(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        if self._hm_grid is None:
            return

        # AQ overlay is cycled via its panel checkbox — no keyboard shortcut.
        # ('q' is dangerous anyway: pyvista hard-binds it to close-window.)

        if not self._hm_visible:
            return

        now = time.perf_counter()
        if now - self._hm_last_t < 1.0:
            return
        self._hm_last_t = now

        try:
            raw = self._sample_heatmap_density()

            # Exponential moving average for smooth visual
            self._hm_smooth = (
                _SMOOTH_ALPHA * raw
                + (1.0 - _SMOOTH_ALPHA) * self._hm_smooth
            )

            # Auto-scale colour range in BOTH directions: ratchet up instantly,
            # decay down slowly (5 %/update).  A fixed floor of 0.01 g/s made
            # the overlay invisible with few vehicles (e.g. SUMO mode with a
            # handful of cars emits ~0.001 g/s/cell — below the 4 % display
            # threshold forever, so the map looked "stuck at zero").
            peak = float(np.percentile(self._hm_smooth, 99)) if self._hm_smooth.any() else 0.0
            if peak > 0.0:
                _target = peak * 1.1
                if _target > self._hm_clim_max:
                    self._hm_clim_max = _target                      # ratchet up instantly
                elif _target < self._hm_clim_max * 0.5:
                    self._hm_clim_max = max(_target, 1e-6)          # regime change: jump down
                else:
                    self._hm_clim_max = max(_target, self._hm_clim_max * 0.9, 1e-6)
                try:
                    self._hm_actor.GetMapper().SetScalarRange(0.0, self._hm_clim_max)
                except Exception:
                    pass

            # Build display array: near-zero cells → NaN so alpha=0 hides them
            display = self._hm_smooth.copy().astype(np.float32)
            threshold = self._hm_clim_max * 0.04
            display[display < threshold] = np.nan

            # In-place update — modify the VTK buffer directly
            existing = self._hm_grid.cell_data["aq"]
            np.copyto(existing, display, casting="unsafe")
            self._hm_grid.GetCellData().Modified()

            # HUD label with pollutant + peak rate
            try:
                lbl = _POLLUTANT_LABEL.get(self._hm_pollutant, "AQ")
                self.plotter.add_text(
                    f"Air quality — {lbl} emission (dispersed)   peak≈{peak:.2g} g/s",
                    position=(0.34, 0.95), name="aq_label",
                    font_size=10, viewport=True, color="#ffd0d0",
                )
            except Exception:
                pass

        except Exception as exc:
            print(f"[heatmap] update error: {exc}")
