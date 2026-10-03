"""Background flood run for the twin: the solver thread streams frames to the UI.

numba kernels release the GIL (nogil=True), so the run proceeds on a worker
thread while the VTK timer on the main thread polls `snapshot()` and paints
the current depth field — puddles form on screen as the solver steps.

Thread contract: the worker only writes `self._live` (a depth copy + scalars)
under a lock, at chunk boundaries; the UI thread only reads it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from floodsim.engine import FloodEngine, FloodInputs, FloodResult


class FloodRun:
    def __init__(self, inp: FloodInputs, storm: dict, *, label: str = "run", save_every: float = 60.0,
                 chunk_s: float = 10.0, wall_budget_s: float | None = None):
        self.inp = inp
        self.storm = storm
        self.label = label
        self.save_every = float(save_every)
        self.chunk_s = float(chunk_s)
        self.engine = FloodEngine(inp)
        self.result: FloodResult | None = None
        self.error: str | None = None
        self.done = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._live = {"t": 0.0, "depth": np.zeros(inp.dem.shape, np.float32), "stored": 0.0,
                      "out": 0.0, "infil": 0.0, "rain": 0.0, "hmax": 0.0, "wall": 0.0, "seq": 0}
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    # -- control ---------------------------------------------------------
    def start(self) -> "FloodRun":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._work, name=f"flood-{self.label}", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._stop.set()

    def running(self) -> bool:
        return self._thread is not None and not self.done.is_set()

    def join(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)

    # -- worker ----------------------------------------------------------
    def _publish(self, eng: FloodEngine) -> None:
        d = eng.depth.astype(np.float32)
        with self._lock:
            lv = self._live
            lv.update(t=float(eng.t), depth=d, stored=eng.stored_m3(), out=eng.vol_out, infil=eng.vol_infil,
                      rain=eng.vol_in, hmax=float(eng._hmax), wall=time.time() - self._t0, seq=lv["seq"] + 1)

    def _work(self) -> None:
        try:
            self.result = self.engine.run(
                self.storm["steps"], float(self.storm["duration"]), save_every=self.save_every,
                chunk_s=self.chunk_s, on_chunk=lambda e, t, f: self._publish(e),
                should_stop=self._stop.is_set)
            self._publish(self.engine)
        except Exception as exc:                 # surface to the UI instead of dying silently
            import traceback
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            self.done.set()

    # -- UI side ---------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            lv = dict(self._live)
        lv["fraction"] = min(lv["t"] / max(float(self.storm["duration"]), 1e-9), 1.0)
        lv["finished"] = self.done.is_set()
        return lv
