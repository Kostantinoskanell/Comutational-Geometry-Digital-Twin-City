"""Solver inputs for the twin's flood engine, from the published terrain grids.

Beirut_Project-main/output/terrain_cut_0.5 holds the 0.5 m rasters the
reference GPU solver consumes (DEM with buildings raised, Manning n,
infiltration, rain-weight downspout map, masks). The live engine runs on a
coarser grid (default 2 m) so a storm takes seconds; coarsening here keeps
what matters for pluvial flow:

  * buildings stay WALLS: a block that is >= 50 % building takes the block
    MAX of the DEM; otherwise the DEM is the mean of its non-building
    (street/ground) cells, so narrow streets are not bridged by roofs;
  * rain volume is conserved exactly (block mean of the rain-weight raster);
  * Manning n and infiltration are averaged over the open (non-building)
    cells of the block;
  * the sea / outside-survey cells become the open boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from floodsim.engine import FloodInputs

ROOT = Path(__file__).resolve().parent.parent
TERRAIN_DIR = ROOT / "Beirut_Project-main" / "output" / "terrain_cut_0.5"
STORMS_DIR = ROOT / "Beirut_Project-main" / "storms"
OFFICIAL_MATERIAL = ROOT / "Beirut_Project-main" / "output" / "corridor_gi_cut" / "material.npy"
CACHE = ROOT / ".cache" / "floodsim"
POROSITY_BLOCKED = 0.05            # wall cells keep a sliver so divisions stay finite (they hold no water)


def terrain_available(terrain_dir: Path = TERRAIN_DIR) -> bool:
    return (Path(terrain_dir) / "dem.npy").exists() and (Path(terrain_dir) / "masks.npz").exists()


def _block(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    H, W = (h // f) * f, (w // f) * f
    return a[:H, :W].reshape(H // f, f, W // f, f)


def load_inputs(terrain_dir: Path = TERRAIN_DIR, res: float = 2.0, cache: bool = True) -> FloodInputs:
    terrain_dir = Path(terrain_dir)
    tr = json.loads((terrain_dir / "dem_transform.json").read_text())
    base_res = float(tr["res"])
    f = max(1, int(round(res / base_res)))
    key = hashlib.sha1(f"{terrain_dir.resolve()}|{f}|{(terrain_dir / 'dem.npy').stat().st_mtime}|v5".encode()).hexdigest()[:16]
    cpath = CACHE / f"inputs_{key}.npz"
    if cache and cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        dr = (z["dr_r"], z["dr_c"], z["dr_cap"]) if z["dr_r"].size else None
        return FloodInputs(z["dem"], float(z["res"]), z["manning"], z["infil"], z["rain_w"], z["valid"], z["water"],
                           dr, z["building"], z["open_frac"], meta=json.loads(str(z["meta"])))

    dem = np.load(terrain_dir / "dem.npy").astype(np.float32)
    masks = np.load(terrain_dir / "masks.npz")
    bld = masks["building"]
    valid = masks["valid"]
    water = masks["water"]
    man = np.load(terrain_dir / "manning.npy").astype(np.float32)
    inf = np.load(terrain_dir / "infil_mmh.npy").astype(np.float32)
    rw = np.load(terrain_dir / "rain_weight.npy").astype(np.float32)

    B = _block(bld, f)
    bfrac = B.mean(axis=(1, 3))
    vfrac = _block(valid, f).mean(axis=(1, 3))
    wfrac = _block(water, f).mean(axis=(1, 3))
    D = _block(dem, f)
    fin = np.isfinite(D)                       # DEM is NaN outside the survey
    gmask = ~B & fin
    n_g = np.maximum(gmask.sum(axis=(1, 3)), 1)
    ground = np.where(gmask, D, 0.0).sum(axis=(1, 3)) / n_g
    wall = np.where(B & fin, D, -1e9).max(axis=(1, 3))
    any_fin = fin.any(axis=(1, 3))
    open_n = np.maximum((~B).sum(axis=(1, 3)), 1)
    blocked = (bfrac >= 0.5) & (wall > -1e8)
    dem_c = np.where(blocked, wall, np.where(gmask.any(axis=(1, 3)), ground, np.where(wall > -1e8, wall, np.nan)))
    dem_c = np.where(any_fin, dem_c, np.nan)
    man_c = (np.where(B, 0.0, _block(man, f))).sum(axis=(1, 3)) / open_n
    man_c = np.where(blocked, 0.05, man_c)
    inf_c = (np.where(B, 0.0, _block(inf, f))).sum(axis=(1, 3)) / open_n
    inf_c = np.where(blocked, 0.0, inf_c)
    rw_c = _block(rw, f).mean(axis=(1, 3))
    # Rain-weight on a wall cell is a downspout/courtyard share whose real
    # location is a GROUND cell. Left on the (roof-height) wall cell it would
    # be released from an artificial head of metres and shoot sideways, so it
    # is moved, volume-conserving, to the nearest open cell.
    if blocked.any() and (~blocked).any():
        from scipy import ndimage as ndi
        _, (ri, ci) = ndi.distance_transform_edt(blocked, return_indices=True)
        moved = np.zeros_like(rw_c)
        np.add.at(moved, (ri[blocked], ci[blocked]), rw_c[blocked])
        rw_c = np.where(blocked, 0.0, rw_c) + moved
    valid_c = (vfrac >= 0.5) & np.isfinite(dem_c)
    water_c = (wfrac >= 0.5) & valid_c
    # invalid cells: neutral bed so they never act as walls/sinks before the boundary removes them
    dem_c = np.where(valid_c, dem_c, np.nanmin(dem_c[valid_c]) - 5.0)

    dr = None
    p = terrain_dir / "drains.npz"
    if p.exists():
        d = np.load(p)
        rr, cc = d["rows"] // f, d["cols"] // f
        h, w = dem_c.shape
        ok = (rr < h) & (cc < w)
        cap = np.zeros((h, w))
        np.add.at(cap, (rr[ok], cc[ok]), d["cap"][ok])
        r2, c2 = np.nonzero(cap)
        dr = (r2.astype(np.int64), c2.astype(np.int64), cap[r2, c2])

    meta = {"transform": {"crs": tr.get("crs", "EPSG:32636"), "minx": float(tr["minx"]),
                          "maxy": float(tr.get("maxy", tr["miny"] + tr["height"] * base_res)),
                          "res": base_res * f, "width": int(dem_c.shape[1]), "height": int(dem_c.shape[0])},
            "factor": f, "source": str(terrain_dir)}
    open_frac = np.where(blocked, POROSITY_BLOCKED, 1.0 - bfrac)
    inp = FloodInputs(dem_c.astype(np.float64), base_res * f, man_c.astype(np.float64), inf_c.astype(np.float64),
                      rw_c.astype(np.float64), valid_c, water_c, dr, blocked, open_frac, meta=meta)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cpath, dem=inp.dem, res=inp.res, manning=inp.manning, infil=inp.infil_mmh, rain_w=inp.rain_weight,
                 valid=inp.valid, water=inp.water, building=blocked,
                 open_frac=open_frac, dr_r=dr[0] if dr else np.zeros(0, np.int64), dr_c=dr[1] if dr else np.zeros(0, np.int64),
                 dr_cap=dr[2] if dr else np.zeros(0), meta=np.array(json.dumps(meta)))
    return inp


def load_storm(name: str) -> dict:
    d = json.loads((STORMS_DIR / f"{name}.json").read_text())
    d["steps"] = [tuple(map(float, s)) for s in d["steps"]]
    return d


def available_storms() -> list[str]:
    return sorted(p.stem for p in STORMS_DIR.glob("*.json"))
