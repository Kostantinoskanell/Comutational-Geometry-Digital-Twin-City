"""Validation of the live engine against the published reference runs.

Reference: Beirut_Project-main/output/corridor_runs/{before,after}_<storm>
(0.5 m GPU solver, 4700-5000 s wall on the study server). The live engine
runs the same storm on the 2 m-coarsened terrain; the comparison is made on
that coarse grid (reference depths block-averaged).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floodsim import design as D
from floodsim.engine import FloodEngine
from floodsim.metrics import FLOOD_THRESHOLD_M, summarize
from floodsim.terrain import OFFICIAL_MATERIAL, ROOT, load_inputs, load_storm

RUNS = ROOT / "Beirut_Project-main" / "output" / "corridor_runs"


def run_case(res: float, storm_name: str, with_corridor: bool, drains: bool = False, save_every: float = 300.0):
    inp = load_inputs(res=res)
    if not drains:
        inp.drains = None
    if with_corridor:
        mat = np.load(OFFICIAL_MATERIAL)
        fr = D.class_fractions_from_raster(mat, inp.meta["factor"], inp.dem.shape)
        inp = D.apply_design(inp, fr)
    storm = load_storm(storm_name)
    eng = FloodEngine(inp)
    result = eng.run(storm["steps"], storm["duration"], save_every=save_every, chunk_s=1e9)
    return inp, result


def reference(phase: str, storm: str, f: int):
    d = RUNS / f"{phase}_{storm}"
    meta = json.loads((d / "run_meta.json").read_text())
    md = np.load(d / "max_depth.npy").astype(np.float64)
    fd = np.load(d / "final_depth.npy").astype(np.float64)
    h, w = (md.shape[0] // f) * f, (md.shape[1] // f) * f

    def coarse(a):
        return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    return meta, coarse(md), coarse(fd)


def compare(res: float = 2.0, storm: str = "v1_nov2025") -> dict:
    out = {"res": res, "storm": storm}
    for phase, corridor in (("before", False), ("after", True)):
        inp, r = run_case(res, storm, corridor)
        meta, ref_md, ref_fd = reference(phase, storm, inp.meta["factor"])
        area = inp.res ** 2
        og = inp.valid & ~inp.water & ~inp.building
        ref_og = og[:ref_md.shape[0], :ref_md.shape[1]]
        mine = r.max_depth[:ref_md.shape[0], :ref_md.shape[1]]
        ref_fl, my_fl = ref_og & (ref_md > FLOOD_THRESHOLD_M), ref_og & (mine > FLOOD_THRESHOLD_M)
        iou = float((ref_fl & my_fl).sum() / max((ref_fl | my_fl).sum(), 1))
        both = ref_og & ((ref_md > 0.02) | (mine > 0.02))
        corr = float(np.corrcoef(ref_md[both], mine[both])[0, 1]) if both.sum() > 10 else float("nan")
        rmse = float(np.sqrt(np.mean((ref_md[ref_og] - mine[ref_og]) ** 2)))
        s = summarize(r, inp)
        out[phase] = {
            "flooded_ha_live": s["flooded_area_ha"], "flooded_ha_ref": float(ref_fl.sum() * area / 1e4),
            "iou_flooded": iou, "depth_corr": corr, "depth_rmse_m": rmse,
            "infil_pct_live": s["infiltrated_pct"], "infil_pct_ref": 100 * meta["vol_infiltrated_m3"] / meta["vol_rain_m3"],
            "outflow_live_m3": s["volume_outflow_m3"], "outflow_ref_m3": meta["vol_outflow_m3"],
            "stored_live_m3": s["volume_stored_end_m3"], "stored_ref_m3": meta["vol_stored_end_m3"],
            "rain_live_m3": s["volume_rain_m3"], "rain_ref_m3": meta["vol_rain_m3"],
            "max_depth_live": s["max_depth_m"], "wall_s": s["wall_s"], "closure_rel": s["closure_rel"],
        }
    b, a = out["before"], out["after"]
    out["flooded_area_change_pct_live"] = 100 * (a["flooded_ha_live"] - b["flooded_ha_live"]) / max(b["flooded_ha_live"], 1e-9)
    out["flooded_area_change_pct_ref"] = 100 * (a["flooded_ha_ref"] - b["flooded_ha_ref"]) / max(b["flooded_ha_ref"], 1e-9)
    return out


if __name__ == "__main__":
    import sys
    res = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    rep = compare(res)
    print(json.dumps(rep, indent=2))
