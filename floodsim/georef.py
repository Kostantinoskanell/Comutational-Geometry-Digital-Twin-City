"""Registration between the solver grid (UTM 36N raster) and the twin's local frame.

The solver grid is axis-aligned in UTM; the twin's local frame is an azimuthal
equidistant projection about the scene origin, so the two differ by a small
rotation (grid convergence + projection). Cell centres are mapped EXACTLY with
pyproj (no resampling anywhere): overlays are structured grids whose points
are the mapped cell centres, and design polygons are mapped back into the
raster's (col, row) space to rasterize them with sub-cell accuracy.
"""
from __future__ import annotations

import numpy as np

FLOOD_UTM_CRS = "EPSG:32636"


class SolverGeoref:
    def __init__(self, transform: dict, to_local):
        from pyproj import Transformer
        self.t = dict(transform)
        self.res = float(transform["res"])
        self.w, self.h = int(transform["width"]), int(transform["height"])
        self.minx, self.maxy = float(transform["minx"]), float(transform["maxy"])
        self._to_local = to_local
        crs = transform.get("crs", FLOOD_UTM_CRS)
        self._utm_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self._ll_to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        cols, rows = np.meshgrid(np.arange(self.w) + 0.5, np.arange(self.h) + 0.5)
        e = self.minx + cols * self.res
        n = self.maxy - rows * self.res
        lon, lat = self._utm_to_ll.transform(e.ravel(), n.ravel())
        x, y = to_local.transform(np.asarray(lon), np.asarray(lat))
        self.xy = np.column_stack([np.asarray(x), np.asarray(y)])          # (h*w, 2) local metres

    def utm_to_local(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        lon, lat = self._utm_to_ll.transform(np.asarray(e, float), np.asarray(n, float))
        x, y = self._to_local.transform(np.asarray(lon), np.asarray(lat))
        return np.column_stack([np.asarray(x), np.asarray(y)])

    # -- queries -----------------------------------------------------------
    def bounds(self) -> tuple[float, float, float, float]:
        return (float(self.xy[:, 0].min()), float(self.xy[:, 0].max()),
                float(self.xy[:, 1].min()), float(self.xy[:, 1].max()))

    def local_to_cr(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local metres -> fractional (col, row) in the raster (cell centres at +0.5)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lon, lat = self._to_local.transform(xy[:, 0], xy[:, 1], direction="INVERSE")
        e, n = self._ll_to_utm.transform(np.asarray(lon), np.asarray(lat))
        return (np.asarray(e) - self.minx) / self.res, (self.maxy - np.asarray(n)) / self.res

    def local_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, r = self.local_to_cr(xy)
        return np.floor(r).astype(int), np.floor(c).astype(int)

    def overlaps(self, local_bounds: tuple[float, float, float, float], min_frac: float = 0.05) -> bool:
        """Does a scene with these local bounds overlap the solver domain meaningfully?"""
        x0, x1, y0, y1 = self.bounds()
        bx0, bx1, by0, by1 = local_bounds
        ix = max(0.0, min(x1, bx1) - max(x0, bx0))
        iy = max(0.0, min(y1, by1) - max(y0, by0))
        return ix * iy >= min_frac * min((x1 - x0) * (y1 - y0), (bx1 - bx0) * (by1 - by0))

    # -- rasterization -----------------------------------------------------
    def polygon_fraction(self, poly_xy: np.ndarray, sub: int = 4) -> tuple[tuple[slice, slice], np.ndarray] | None:
        """Area fraction of every solver cell covered by a local-frame polygon:
        returns ((row_slice, col_slice), fraction block) or None if outside."""
        import shapely
        from shapely.geometry import Polygon
        c, r = self.local_to_cr(np.asarray(poly_xy, dtype=float))
        poly = Polygon(np.column_stack([c, r]))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        minc, minr, maxc, maxr = poly.bounds
        c0, c1 = max(int(np.floor(minc)), 0), min(int(np.ceil(maxc)), self.w)
        r0, r1 = max(int(np.floor(minr)), 0), min(int(np.ceil(maxr)), self.h)
        if c1 <= c0 or r1 <= r0:
            return None
        nr, nc = r1 - r0, c1 - c0
        sc = (np.arange(nc * sub) + 0.5) / sub + c0
        sr = (np.arange(nr * sub) + 0.5) / sub + r0
        SC, SR = np.meshgrid(sc, sr)
        inside = shapely.contains_xy(poly, SC.ravel(), SR.ravel()).reshape(nr * sub, nc * sub)
        frac = inside.reshape(nr, sub, nc, sub).mean(axis=(1, 3))
        return (slice(r0, r1), slice(c0, c1)), frac

    def disk_fraction(self, x: float, y: float, radius_m: float) -> tuple[tuple[slice, slice], np.ndarray] | None:
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return self.polygon_fraction(np.column_stack([x + radius_m * np.cos(ang), y + radius_m * np.sin(ang)]))
