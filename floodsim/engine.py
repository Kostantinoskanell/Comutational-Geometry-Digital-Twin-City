"""Rain-on-grid shallow-water engine for the twin (numba, multi-core).

The same numerics as Beirut_Project-main/scripts/flood_gpu.py's default
scheme — the Bates, Horritt & Fewtrell (2010) local-inertial shallow-water
update with semi-implicit Manning friction and the mass-conserving
outflux-scaling limiter — with hyetograph rain, rain-weight raster
(downspout rerouting), spatially varying Manning n and infiltration,
gully inlets with capacity caps, an open boundary with outflow volume
accounted, and a closing mass balance.

It exists so the twin can run a flood study LIVE on the M1 (a 2 m grid over
the corridor domain runs 1 simulated hour in seconds), stream the depth
field while it runs, and be re-run after every design edit. The published
0.5 m GPU runs remain the reference; floodsim.validate compares the two.

Kernels are cell/face-parallel (prange over rows) and allocation-free per
step. State is float64 so the mass balance closes to rounding error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

import threading as _threading

try:                                  # share the app's lock for parallel numba kernels
    from shadow_engine import NUMBA_KERNEL_LOCK as KERNEL_LOCK
except Exception:                     # standalone use (no app): a private lock
    KERNEL_LOCK = _threading.RLock()

G = 9.81
H_MIN = 1e-4
POROSITY_FLOOR = 0.05
ALPHA = 0.7          # CFL factor of the inertial scheme (as in flood_gpu.py)
DT_MAX = 5.0


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _momentum(depth, dem, n_cell, qx, qy, act_x, act_y, hfx, hfy, dt, res):
    """Face unit discharges from the local-inertial update."""
    h, w = depth.shape
    for i in prange(h):
        for j in range(w - 1):
            zl = dem[i, j] + depth[i, j]
            zr = dem[i, j + 1] + depth[i, j + 1]
            hz = max(dem[i, j], dem[i, j + 1])
            hflow = max(max(zl, zr) - hz, 0.0)
            if hflow > H_MIN:
                n = max(n_cell[i, j], n_cell[i, j + 1])
                q = qx[i, j]
                qx[i, j] = (q + G * hflow * dt * (zl - zr) / res) / (
                    1.0 + G * dt * n * n * abs(q) / hflow ** (7.0 / 3.0))
                act_x[i, j] = True
                hfx[i, j] = hflow
            else:
                qx[i, j] = 0.0
                act_x[i, j] = False
                hfx[i, j] = 1.0
    for i in prange(h - 1):
        for j in range(w):
            zt = dem[i, j] + depth[i, j]
            zb = dem[i + 1, j] + depth[i + 1, j]
            hz = max(dem[i, j], dem[i + 1, j])
            hflow = max(max(zt, zb) - hz, 0.0)
            if hflow > H_MIN:
                n = max(n_cell[i, j], n_cell[i + 1, j])
                q = qy[i, j]
                qy[i, j] = (q + G * hflow * dt * (zt - zb) / res) / (
                    1.0 + G * dt * n * n * abs(q) / hflow ** (7.0 / 3.0))
                act_y[i, j] = True
                hfy[i, j] = hflow
            else:
                qy[i, j] = 0.0
                act_y[i, j] = False
                hfy[i, j] = 1.0


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _limiter_scale(depth, qx, qy, scale, phi, phx, phy, dt, res):
    """Per-cell factor so no cell exports more water than it holds
    (open-area volume: depth * phi * res^2; face flux q * open width * res)."""
    h, w = depth.shape
    for i in prange(h):
        for j in range(w):
            out = 0.0
            if j < w - 1 and qx[i, j] > 0.0:
                out += qx[i, j] * phx[i, j]
            if j > 0 and qx[i, j - 1] < 0.0:
                out -= qx[i, j - 1] * phx[i, j - 1]
            if i < h - 1 and qy[i, j] > 0.0:
                out += qy[i, j] * phy[i, j]
            if i > 0 and qy[i - 1, j] < 0.0:
                out -= qy[i - 1, j] * phy[i - 1, j]
            s = depth[i, j] * phi[i, j] * res / (out * dt + 1e-12)
            scale[i, j] = 1.0 if s > 1.0 else s


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _continuity(depth, qx, qy, scale, phi, phx, phy, dt, res):
    """Apply the scaled face fluxes (and persist them, as flood_gpu does)."""
    h, w = depth.shape
    for i in prange(h):
        for j in range(w - 1):
            q = qx[i, j]
            qx[i, j] = q * scale[i, j] if q > 0.0 else q * scale[i, j + 1]
    for i in prange(h - 1):
        for j in range(w):
            q = qy[i, j]
            qy[i, j] = q * scale[i, j] if q > 0.0 else q * scale[i + 1, j]
    k = dt / res
    for i in prange(h):
        for j in range(w):
            d = 0.0
            if j < w - 1:
                d -= qx[i, j] * phx[i, j]
            if j > 0:
                d += qx[i, j - 1] * phx[i, j - 1]
            if i < h - 1:
                d -= qy[i, j] * phy[i, j]
            if i > 0:
                d += qy[i - 1, j] * phy[i - 1, j]
            depth[i, j] += d * k / phi[i, j]


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _sources(depth, rain_w, infil, keep, rain_ms, dt, row_infil, row_out, row_rain, row_max, out_map, phi):
    """Rain, infiltration, open boundary, max-depth; per-row volume sums."""
    h, w = depth.shape
    for i in prange(h):
        vi = 0.0
        vo = 0.0
        vr = 0.0
        mx = 0.0
        for j in range(w):
            d = depth[i, j]
            if rain_ms > 0.0:
                add = rain_ms * dt * rain_w[i, j]            # volume per cell area
                d += add / phi[i, j]
                vr += add
            f = infil[i, j] * dt
            if f > 0.0:
                di = f if f < d else d
                d -= di
                vi += di * phi[i, j]
            if keep[i, j] == 0.0:
                vo += d * phi[i, j]
                out_map[i, j] += d * phi[i, j]
                d = 0.0
            if d < 0.0:
                d = 0.0
            depth[i, j] = d
            if d > mx:
                mx = d
        row_infil[i] = vi
        row_out[i] = vo
        row_rain[i] = vr
        row_max[i] = mx


@njit(nogil=True, cache=True)
def _drains(depth, phi, rr, cc, cap, dt, area):
    vol = 0.0
    for k in range(rr.shape[0]):
        d = depth[rr[k], cc[k]]
        take = cap[k] * dt / (area * phi[rr[k], cc[k]])
        if take > d:
            take = d
        depth[rr[k], cc[k]] = d - take
        vol += take * phi[rr[k], cc[k]]
    return vol


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def _accumulate(depth, max_depth, qx, qy, act_x, act_y, hfx, hfy, max_vel, max_haz):
    h, w = depth.shape
    for i in prange(h):
        for j in range(w):
            d = depth[i, j]
            if d > max_depth[i, j]:
                max_depth[i, j] = d
            if d > 0.01:
                vx = 0.0
                vy = 0.0
                if j < w - 1 and act_x[i, j]:
                    vx += 0.5 * qx[i, j] / hfx[i, j]
                if j > 0 and act_x[i, j - 1]:
                    vx += 0.5 * qx[i, j - 1] / hfx[i, j - 1]
                if i < h - 1 and act_y[i, j]:
                    vy += 0.5 * qy[i, j] / hfy[i, j]
                if i > 0 and act_y[i - 1, j]:
                    vy += 0.5 * qy[i - 1, j] / hfy[i - 1, j]
                v = math.sqrt(vx * vx + vy * vy)
                if v > max_vel[i, j]:
                    max_vel[i, j] = v
                hz = d * (v + 0.5)
                if hz > max_haz[i, j]:
                    max_haz[i, j] = hz


@dataclass
class FloodInputs:
    """Everything the engine needs, on a regular grid (row 0 = north)."""
    dem: np.ndarray                  # (h, w) bed elevation, m
    res: float                       # cell size, m
    manning: np.ndarray              # (h, w)
    infil_mmh: np.ndarray            # (h, w) saturated rate, mm/h
    rain_weight: np.ndarray          # (h, w) share of the rain landing in each cell (mean ~1)
    valid: np.ndarray                # (h, w) bool: inside the surveyed domain
    water: np.ndarray                # (h, w) bool: sea/harbour (open boundary)
    drains: tuple | None = None      # (rows, cols, cap_m3s)
    building: np.ndarray | None = None
    open_frac: np.ndarray | None = None   # (h, w) fraction of the cell open to water (1 = no sub-grid obstruction)
    meta: dict = field(default_factory=dict)


@dataclass
class FloodResult:
    max_depth: np.ndarray
    max_vel: np.ndarray
    max_hazard: np.ndarray
    final_depth: np.ndarray
    frames: list                     # [(t_s, depth float32)]
    meta: dict


class FloodEngine:
    def __init__(self, inp: FloodInputs):
        self.inp = inp
        h, w = inp.dem.shape
        self.shape = (h, w)
        self.res = float(inp.res)
        self.area = self.res * self.res
        valid = inp.valid
        dem = np.where(valid, inp.dem, np.nanmin(inp.dem[valid]) - 5.0)
        self.dem = np.ascontiguousarray(dem, dtype=np.float64)
        self.n_cell = np.ascontiguousarray(np.broadcast_to(inp.manning, (h, w)), dtype=np.float64)
        self.rain_w = np.ascontiguousarray(inp.rain_weight, dtype=np.float64)
        self.rw_sum = float(self.rain_w.sum())
        self.infil = np.ascontiguousarray(inp.infil_mmh / 3.6e6, dtype=np.float64)
        out = ~valid
        out[0, :] = out[-1, :] = True
        out[:, 0] = out[:, -1] = True
        out |= inp.water
        self.keep = np.ascontiguousarray((~out).astype(np.float64))
        # sub-grid porosity: cell open fraction and open width of each face
        phi = np.ones((h, w)) if inp.open_frac is None else np.clip(inp.open_frac, POROSITY_FLOOR, 1.0)
        self.phi = np.ascontiguousarray(phi, dtype=np.float64)
        self.phx = np.ascontiguousarray(np.minimum(phi[:, :-1], phi[:, 1:]))
        self.phy = np.ascontiguousarray(np.minimum(phi[:-1, :], phi[1:, :]))
        if inp.drains is not None and len(inp.drains[0]):
            self.dr_r = np.ascontiguousarray(inp.drains[0], dtype=np.int64)
            self.dr_c = np.ascontiguousarray(inp.drains[1], dtype=np.int64)
            self.dr_cap = np.ascontiguousarray(inp.drains[2], dtype=np.float64)
        else:
            self.dr_r = self.dr_c = np.zeros(0, np.int64)
            self.dr_cap = np.zeros(0, np.float64)
        self.reset()

    def reset(self):
        h, w = self.shape
        self.depth = np.zeros((h, w))
        self.qx = np.zeros((h, w - 1))
        self.qy = np.zeros((h - 1, w))
        self.scale = np.ones((h, w))
        self.act_x = np.zeros((h, w - 1), np.bool_)
        self.act_y = np.zeros((h - 1, w), np.bool_)
        self.hfx = np.ones((h, w - 1))
        self.hfy = np.ones((h - 1, w))
        self.max_depth = np.zeros((h, w))
        self.max_vel = np.zeros((h, w))
        self.max_haz = np.zeros((h, w))
        self.row_infil = np.zeros(h)
        self.row_out = np.zeros(h)
        self.row_rain = np.zeros(h)
        self.row_max = np.zeros(h)
        self.out_map = np.zeros((h, w))      # volume (m3/m2) that left through each boundary cell
        self.t = 0.0
        self.it = 0
        self.vol_in = self.vol_infil = self.vol_drain = self.vol_out = 0.0
        self._hmax = 0.0

    # ------------------------------------------------------------------
    def step(self, rain_ms: float, t_limit: float) -> float:
        """One time step; returns dt."""
        hmax = self._hmax
        dt = min(ALPHA * self.res / math.sqrt(G * max(hmax, 0.01)), DT_MAX)
        dt = max(math.floor(dt * 1000.0) / 1000.0, 1e-3)
        dt = min(dt, max(t_limit - self.t, 1e-6))
        # numba's workqueue threading layer aborts the process if two threads launch parallel
        # kernels at once (the app's background shadow worker does): hold the shared lock for
        # one whole step, release between steps so that worker interleaves.
        with KERNEL_LOCK:
            self._step_kernels(rain_ms, dt)
        self.t += dt
        self.it += 1
        return dt

    def _step_kernels(self, rain_ms: float, dt: float) -> None:
        _momentum(self.depth, self.dem, self.n_cell, self.qx, self.qy,
                  self.act_x, self.act_y, self.hfx, self.hfy, dt, self.res)
        _limiter_scale(self.depth, self.qx, self.qy, self.scale, self.phi, self.phx, self.phy, dt, self.res)
        _continuity(self.depth, self.qx, self.qy, self.scale, self.phi, self.phx, self.phy, dt, self.res)
        _sources(self.depth, self.rain_w, self.infil, self.keep, rain_ms, dt,
                 self.row_infil, self.row_out, self.row_rain, self.row_max, self.out_map, self.phi)
        self.vol_in += float(self.row_rain.sum()) * self.area
        self.vol_infil += float(self.row_infil.sum()) * self.area
        if self.dr_r.size:
            self.vol_drain += _drains(self.depth, self.phi, self.dr_r, self.dr_c, self.dr_cap, dt, self.area) * self.area
        self.vol_out += float(self.row_out.sum()) * self.area
        if self.it % 5 == 0:
            _accumulate(self.depth, self.max_depth, self.qx, self.qy, self.act_x, self.act_y,
                        self.hfx, self.hfy, self.max_vel, self.max_haz)
        else:
            np.maximum(self.max_depth, self.depth, out=self.max_depth)
        self._hmax = float(self.row_max.max())

    def stored_m3(self) -> float:
        return float((self.depth * self.phi).sum()) * self.area

    def run(self, steps, duration: float, *, save_every: float = 60.0, chunk_s: float = 15.0,
            on_chunk=None, should_stop=None) -> FloodResult:
        """Run the storm. steps: [(t0_s, t1_s, mm/h)]. on_chunk(engine, t, frames_so_far)
        is called every chunk_s simulated seconds (and at each saved frame time)
        so the caller can stream the live depth field; should_stop() cancels."""
        import time as _time
        wall0 = _time.time()
        frames = []
        next_save = 0.0
        next_chunk = 0.0
        # rain-rate breakpoints: never step across a hyetograph edge
        edges = sorted({float(b) for s in steps for b in s[:2] if 0 < b < duration} | {float(duration)})

        def rain_at(tt):
            for t0, t1, mmh in steps:
                if t0 <= tt < t1:
                    return mmh / 3.6e6
            return 0.0

        while self.t < duration - 1e-9:
            if should_stop is not None and should_stop():
                break
            nxt = min([e for e in edges if e > self.t + 1e-9] or [duration])
            rain_ms = rain_at(self.t + 1e-9)
            self.step(rain_ms, nxt)
            if self.t >= next_save - 1e-9:
                frames.append((float(self.t), self.depth.astype(np.float32)))
                next_save += save_every
            if on_chunk is not None and self.t >= next_chunk:
                on_chunk(self, self.t, frames)
                next_chunk = self.t + chunk_s
        frames.append((float(self.t), self.depth.astype(np.float32)))
        stored = self.stored_m3()
        closure = self.vol_in - (self.vol_infil + self.vol_drain + self.vol_out + stored)
        meta = {
            "res": self.res, "duration": float(self.t), "n_iterations": self.it,
            "wall_s": round(_time.time() - wall0, 2),
            "vol_rain_m3": self.vol_in, "vol_infiltrated_m3": self.vol_infil,
            "vol_drained_m3": self.vol_drain, "vol_outflow_m3": self.vol_out,
            "vol_stored_end_m3": stored, "closure_m3": closure,
            "closure_rel": closure / max(self.vol_in, 1e-9),
            "cancelled": bool(should_stop() if should_stop is not None else False),
        }
        return FloodResult(self.max_depth.astype(np.float32), self.max_vel.astype(np.float32),
                           self.max_haz.astype(np.float32), self.depth.astype(np.float32), frames, meta)
