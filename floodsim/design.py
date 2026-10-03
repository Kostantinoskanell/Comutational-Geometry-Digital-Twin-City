"""Design -> solver inputs: the corridor material model.

The same material classes and properties as Beirut_Project-main/scripts/
bake_corridor.py (so a design authored in the twin is physically identical
to what the reference pipeline would bake): per class an infiltration rate,
a Manning roughness and a detention depression (the cell is lowered to hold
water). A design is a set of per-class AREA FRACTIONS on the solver grid;
properties are blended by fraction, so a 2 m solver cell that is 30 %
bioswale behaves 30 % like one.
"""
from __future__ import annotations

import numpy as np

from floodsim.engine import FloodInputs

# class: (infil mm/h, Manning n, depression m, label)  — bake_corridor.PROPS
MATERIALS = {
    1: (5.0, 0.016, 0.00, "vehicular lane"),
    2: (150.0, 0.020, 0.00, "porous bikelane"),
    3: (200.0, 0.150, 0.15, "bioswale"),
    4: (150.0, 0.020, 0.00, "porous sidewalk"),
    5: (100.0, 0.100, 0.00, "garden"),
    6: (250.0, 0.200, 0.40, "bioretention pond / rain garden"),
    7: (100.0, 0.120, 0.10, "terrace"),
}
# Extra authoring classes the twin adds (same literature basis as above)
MATERIALS.update({
    8: (60.0, 0.025, 0.00, "permeable paving"),
    9: (120.0, 0.080, 0.00, "tree pit / planted strip"),
    10: (0.0, 0.050, 0.00, "stairs / steps"),
})
LABELS = {k: v[3] for k, v in MATERIALS.items()}


def class_fractions_from_raster(material: np.ndarray, f: int, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """Fine class raster (0 = none) -> per-class area fraction on the coarse grid."""
    h, w = shape
    H, W = h * f, w * f
    m = np.zeros((H, W), material.dtype)
    hh, ww = min(H, material.shape[0]), min(W, material.shape[1])
    m[:hh, :ww] = material[:hh, :ww]
    B = m.reshape(h, f, w, f)
    return {c: (B == c).mean(axis=(1, 3)) for c in MATERIALS if (m == c).any()}


def apply_design(base: FloodInputs, fractions: dict[int, np.ndarray], *, keep_buildings: bool = True) -> FloodInputs:
    """A copy of `base` with the material fractions applied (see module doc)."""
    frs = {}
    for c, fr in fractions.items():
        fr = np.clip(np.asarray(fr, dtype=np.float64), 0.0, 1.0)
        if keep_buildings and base.building is not None:
            fr = np.where(base.building, 0.0, fr)
        frs[c] = fr
    S = sum(frs.values()) if frs else np.zeros_like(base.dem)
    cover = np.minimum(S, 1.0)
    norm = np.maximum(S, 1.0)                      # overlapping classes share the cell
    inf_new = sum(frs[c] / norm * MATERIALS[c][0] for c in frs) if frs else 0.0
    man_new = sum(frs[c] / norm * MATERIALS[c][1] for c in frs) if frs else 0.0
    dep = sum(frs[c] / norm * MATERIALS[c][2] for c in frs) if frs else 0.0
    out = FloodInputs(base.dem - dep, base.res, (1 - cover) * base.manning + man_new,
                      (1 - cover) * base.infil_mmh + inf_new, base.rain_weight, base.valid, base.water,
                      base.drains, base.building, base.open_frac, meta=dict(base.meta))
    out.meta["design_cover_m2"] = float((cover * base.res ** 2).sum())
    return out
