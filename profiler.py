"""profiler.py — lightweight per-frame profiler for the Digital Twin event loop.

Categories
----------
idm_sim     _animate_cars() — IDM physics tick + traffic-light FSM
vtk_actors  _animate_peds/cyclists/buses/emergency/sumo/walk() — actor SetPosition calls
render      plotter.update() — VTK GPU flush
overlay     _poll_shadow_job / _animate_weather / _tod / _parking / _aq_overlay
event_pump  vtk_iren.ProcessEvents() — Cocoa event dispatch

Usage (from run())
------------------
    _prof = FrameProfiler()
    _prof.start()
    while ...:
        if cond: t=pc(); fn(); _prof.record("idm_sim", (pc()-t)*1e3)
        ...
        _prof.end_frame()
        if _prof.elapsed_s() >= duration: break
    _prof.print_table()
    _prof.save_json("profile_report.json")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


CATEGORIES = ("idm_sim", "vtk_actors", "render", "overlay", "event_pump")


class FrameProfiler:
    """Records per-invocation timings; one end_frame() call per loop iteration."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        self._frame_totals: dict[str, list[float]] = {c: [] for c in CATEGORIES}
        self._frame_acc: dict[str, float] = {c: 0.0 for c in CATEGORIES}
        self._frame_count: int = 0
        self._start: float = 0.0
        self._n_cars: int = 0

    def start(self, n_cars: int = 0) -> None:
        self._start = time.perf_counter()
        self._n_cars = n_cars

    def record(self, category: str, elapsed_ms: float) -> None:
        """Record one invocation of *category* that took *elapsed_ms* ms."""
        if category not in self._samples:
            return
        self._samples[category].append(elapsed_ms)
        self._frame_acc[category] += elapsed_ms

    def end_frame(self) -> None:
        """Commit the current frame's per-category accumulator."""
        for c in CATEGORIES:
            self._frame_totals[c].append(self._frame_acc[c])
            self._frame_acc[c] = 0.0
        self._frame_count += 1

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict with per-category stats (per-invocation and per-frame)."""
        out: dict = {
            "meta": {
                "n_frames": self._frame_count,
                "wall_s":   round(self.elapsed_s(), 3),
                "n_cars":   self._n_cars,
            },
            "per_invocation": {},
            "per_frame": {},
        }
        for cat in CATEGORIES:
            inv = np.array(self._samples[cat], dtype=float)
            frm = np.array(self._frame_totals[cat], dtype=float)
            out["per_invocation"][cat] = _stats(inv)
            out["per_frame"][cat]      = _stats(frm)
        return out

    def print_table(self) -> None:
        s = self.summary()
        meta = s["meta"]
        print()
        print("=" * 72)
        print(f"  PROFILING REPORT  —  {meta['wall_s']:.1f} s  ·  "
              f"{meta['n_frames']} frames  ·  {meta['n_cars']} cars")
        print("=" * 72)

        # Per-invocation table (only rows where calls > 0)
        print()
        print("  Per-invocation (each time the callback was actually called)")
        _print_stats_table(s["per_invocation"])

        print()
        print("  Per-frame (amortised cost per loop iteration, 0 when not called)")
        _print_stats_table(s["per_frame"])

        print()
        # Frame-budget sanity: how many ms per frame does each category consume?
        n_frames = max(meta["n_frames"], 1)
        total_ms = sum(
            float(np.sum(self._frame_totals[c])) for c in CATEGORIES
        )
        frame_ms = total_ms / n_frames
        print(f"  Total measured work: {frame_ms:.2f} ms/frame on average")
        rend = np.array(self._frame_totals["render"], dtype=float)
        if rend.size:
            fps_est = 1000.0 / float(np.mean(rend)) if float(np.mean(rend)) > 0 else 0.0
            print(f"  Implied render FPS:  {fps_est:.1f}  (based on render mean)")
        print("=" * 72)
        print()

    def save_json(self, path: str | Path = "profile_report.json") -> None:
        data = self.summary()
        # Also store raw sample arrays for offline analysis
        data["raw"] = {
            "per_invocation": {c: self._samples[c]      for c in CATEGORIES},
            "per_frame":      {c: self._frame_totals[c] for c in CATEGORIES},
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[profile] report saved → {path}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0,
                "p95_ms": 0.0, "max_ms": 0.0, "sum_ms": 0.0}
    return {
        "count":   int(arr.size),
        "mean_ms": round(float(np.mean(arr)),          3),
        "p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "max_ms":  round(float(np.max(arr)),            3),
        "sum_ms":  round(float(np.sum(arr)),            3),
    }


def _print_stats_table(section: dict) -> None:
    hdr = f"  {'Category':<14}  {'N':>6}  {'mean ms':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'max ms':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cat in CATEGORIES:
        st = section.get(cat, {})
        n  = st.get("count", 0)
        print(
            f"  {cat:<14}  {n:>6}  "
            f"{st.get('mean_ms', 0.0):>8.2f}  "
            f"{st.get('p50_ms',  0.0):>8.2f}  "
            f"{st.get('p95_ms',  0.0):>8.2f}  "
            f"{st.get('max_ms',  0.0):>8.2f}"
        )
