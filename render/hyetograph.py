"""Storm hyetographs from the flood engine's storm library (read-only).

Beirut_Project-main/storms/<name>.json: {"steps": [[t0_s, t1_s, mm_per_h], ...],
"duration": s, "total_mm": mm}. Used to drive the rain visuals during the
flood time-lapse so what falls on screen is the storm the solver ran.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STORMS_DIR = Path(__file__).resolve().parent.parent / "Beirut_Project-main" / "storms"
CLOUDBURST_MM_H = 90.0      # visual saturation (rain streak density 100 %)


def load_storm(name: str, storms_dir: Path = STORMS_DIR) -> dict | None:
    path = Path(storms_dir) / f"{name}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    steps = np.asarray(d.get("steps", []), dtype=float).reshape(-1, 3)
    return {"name": d.get("name", name), "steps": steps,
            "duration": float(d.get("duration", steps[:, 1].max() if len(steps) else 0.0)),
            "total_mm": float(d.get("total_mm", 0.0))}


def intensity_mm_h(storm: dict | None, t_s: float) -> float:
    if not storm or not len(storm["steps"]):
        return 0.0
    s = storm["steps"]
    hit = (s[:, 0] <= t_s) & (t_s < s[:, 1])
    return float(s[hit, 2][0]) if hit.any() else 0.0


def cumulative_mm(storm: dict | None, t_s: float) -> float:
    if not storm or not len(storm["steps"]):
        return 0.0
    s = storm["steps"]
    dt = np.clip(np.minimum(s[:, 1], t_s) - s[:, 0], 0.0, None)
    return float((dt * s[:, 2] / 3600.0).sum())
