"""Flood metrics shared by the app HUD, the validation study and the tests."""
from __future__ import annotations

import numpy as np

FLOOD_THRESHOLD_M = 0.10        # "flooded" = deeper than 10 cm (pedestrian / kerb-overtopping scale)


def summarize(result, inp, threshold: float = FLOOD_THRESHOLD_M, mask: np.ndarray | None = None) -> dict:
    """Headline numbers of a run on its solver grid. `mask` restricts the area
    statistics (e.g. to the corridor right-of-way or its surroundings)."""
    area = inp.res ** 2
    open_ground = inp.valid & ~inp.water & (~inp.building if inp.building is not None else True)
    if mask is not None:
        open_ground = open_ground & mask
    md = result.max_depth
    fl = open_ground & (md > threshold)
    m = result.meta
    return {
        "flooded_area_ha": float(fl.sum() * area / 1e4),
        "flooded_area_deep_ha": float((open_ground & (md > 0.30)).sum() * area / 1e4),
        "max_depth_m": float(md[open_ground].max()) if open_ground.any() else 0.0,
        "mean_depth_flooded_m": float(md[fl].mean()) if fl.any() else 0.0,
        "volume_rain_m3": m["vol_rain_m3"],
        "volume_infiltrated_m3": m["vol_infiltrated_m3"],
        "volume_drained_m3": m["vol_drained_m3"],
        "volume_outflow_m3": m["vol_outflow_m3"],
        "volume_stored_end_m3": m["vol_stored_end_m3"],
        "infiltrated_pct": 100.0 * m["vol_infiltrated_m3"] / max(m["vol_rain_m3"], 1e-9),
        "peak_hazard": float(result.max_hazard[open_ground].max()) if open_ground.any() else 0.0,
        "hazardous_area_ha": float((open_ground & (result.max_hazard > 0.75)).sum() * area / 1e4),   # EA/Defra: danger for some
        "closure_rel": m["closure_rel"],
        "wall_s": m["wall_s"],
    }


def building_exposure(result, inp, to_cell, footprints_xy: np.ndarray, threshold: float = FLOOD_THRESHOLD_M, ring_m: float = 3.0):
    """Max flood depth within `ring_m` of each building centroid (n,) — `to_cell`
    maps (n,2) local xy -> (rows, cols) on the solver grid."""
    rows, cols = to_cell(footprints_xy)
    r = int(np.ceil(ring_m / inp.res))
    h, w = result.max_depth.shape
    out = np.zeros(len(rows))
    md = result.max_depth
    for k, (i, j) in enumerate(zip(rows, cols)):
        if 0 <= i < h and 0 <= j < w:
            out[k] = md[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].max()
    return out
