"""The design model: what the architect has authored, as flood-solver inputs.

Objects are stored in the twin's LOCAL frame (that is what the editor tools
click on); `to_inputs` rasterizes them on the solver grid through the
georeference and applies them to the base terrain with the shared material
table (floodsim.design). The model is the single source of truth for what a
"with design" run contains; the editor keeps it in sync (add / undo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from floodsim import design as D
from floodsim.engine import FloodInputs

TREE_PIT_RADIUS_M = 1.2          # planted pit/strip footprint per street tree
BUILDING_WALL_RISE_M = 12.0      # a placed building blocks flow like the surveyed ones


@dataclass
class DesignObject:
    key: str                     # unique key (actor name) so undo can remove it
    kind: str                    # "area" | "tree" | "building" | "drain"
    cls: int | None = None       # material class (area/tree)
    geom: np.ndarray | None = None   # (n,2) local polygon, or (1,2) point
    label: str = ""
    aux: tuple = ()               # extra actor keys removed together with this object (e.g. a tree's canopy)


@dataclass
class DesignModel:
    objects: list[DesignObject] = field(default_factory=list)
    official: bool = False                     # official Masar corridor material raster loaded
    version: int = 0

    # -- editing -----------------------------------------------------------
    def add(self, obj: DesignObject) -> None:
        self.objects = [o for o in self.objects if o.key != obj.key] + [obj]
        self.version += 1

    def remove_missing(self, alive_keys: set[str]) -> int:
        n0 = len(self.objects)
        self.objects = [o for o in self.objects if o.key in alive_keys]
        if len(self.objects) != n0:
            self.version += 1
        return n0 - len(self.objects)

    def clear(self) -> None:
        self.objects = []
        self.official = False
        self.version += 1

    def is_empty(self) -> bool:
        return not self.objects and not self.official

    # -- summary -----------------------------------------------------------
    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.objects:
            name = {"tree": "trees", "building": "buildings", "drain": "drains"}.get(o.kind, D.LABELS.get(o.cls, "areas"))
            out[name] = out.get(name, 0) + 1
        if self.official:
            out["official Masar corridor"] = 1
        return out

    # -- rasterization -----------------------------------------------------
    def to_inputs(self, base: FloodInputs, georef, official_material: np.ndarray | None = None) -> FloodInputs:
        h, w = base.dem.shape
        frs: dict[int, np.ndarray] = {}
        if self.official and official_material is not None:
            for c, fr in D.class_fractions_from_raster(official_material, base.meta["factor"], (h, w)).items():
                frs[c] = frs.get(c, 0) + fr

        def paint(cls, res):
            if res is None:
                return
            (rs, cs), blk = res
            arr = frs.setdefault(cls, np.zeros((h, w)))
            arr[rs, cs] = np.minimum(arr[rs, cs] + blk, 1.0)

        building_blk = np.zeros((h, w))
        for o in self.objects:
            if o.kind == "area" and o.geom is not None and len(o.geom) >= 3:
                paint(o.cls, georef.polygon_fraction(o.geom))
            elif o.kind == "tree" and o.geom is not None:
                x, y = np.asarray(o.geom).reshape(-1, 2)[0]
                paint(9, georef.disk_fraction(float(x), float(y), TREE_PIT_RADIUS_M))
            elif o.kind == "building" and o.geom is not None and len(o.geom) >= 3:
                res = georef.polygon_fraction(o.geom)
                if res is not None:
                    (rs, cs), blk = res
                    building_blk[rs, cs] = np.maximum(building_blk[rs, cs], blk)
        out = D.apply_design(base, frs, keep_buildings=True)
        # The city's existing 600 study inlets are CLOGGED in the studied storm (the
        # 25 Nov 2025 failure mode) and the baseline runs without them; a design only
        # adds the drains the architect places.
        out.drains = None
        if building_blk.any():
            wall = building_blk >= 0.5
            out.dem = np.where(wall, out.dem + BUILDING_WALL_RISE_M, out.dem)
            out.building = (out.building | wall) if out.building is not None else wall
            of = out.open_frac if out.open_frac is not None else np.ones((h, w))
            out.open_frac = np.where(wall, 0.05, np.minimum(of, 1.0 - 0.9 * building_blk))
            out.rain_weight = np.where(wall, 0.0, out.rain_weight)
        drains = [o for o in self.objects if o.kind == "drain" and o.geom is not None]
        if drains:
            xy = np.vstack([np.asarray(o.geom).reshape(-1, 2)[0] for o in drains])
            rr, cc = georef.local_to_cell(xy)
            ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            cap = np.zeros((h, w))
            np.add.at(cap, (rr[ok], cc[ok]), 0.03)       # one gully ~30 l/s (same order as the study's 600 inlets / 18 m3/s)
            r2, c2 = np.nonzero(cap)
            out.drains = (r2.astype(np.int64), c2.astype(np.int64), cap[r2, c2])
        out.meta["design_summary"] = self.summary()
        return out
