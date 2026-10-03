"""Scene-side loader for the preprocessed drone survey (see render.survey_prep).

For one twin scene (local AEQD frame) this builds:

  * a fused terrain sampler: the 0.25 m drone DTM converted from ellipsoidal
    to EGM2008 orthometric heights (the datum the twin's Copernicus DEM uses),
    feathered into Copernicus at the edge of survey coverage, and Copernicus
    alone outside it. Drop-in replacement for street_graph.graph["terrain_sampler"].
  * an orthophoto texture resampled into the local frame, plus its coverage
    mask, for the photoreal ground layer.

Local->survey coordinates are computed exactly with pyproj on a coarse
lattice and bilinearly interpolated in between: AEQD->UTM is smooth, so at
a 16 m lattice spacing the interpolation error is far below a millimetre
while avoiding millions of pyproj calls.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent / "cache" / "survey"
LATTICE_STEP_M = 16.0
FEATHER_M = 15.0
MIN_PLAUSIBLE_ORTHO_M = -2.0
# Survey-edge degradation: with few overlapping drone views, photogrammetry
# at the flight boundary smears both the ortho and the DTM/DSM (tent-like
# ground, melted volumes). Detected as low image detail connected to the
# no-data boundary; smooth-but-valid interior surfaces (asphalt, flat roofs)
# are not boundary-connected, and water is excluded by colour.
QUALITY_WINDOW_M = 10.0
QUALITY_FRAC = 0.35          # of the scene's median local detail
QUALITY_BLOCK = 4            # texture px per quality cell
QUALITY_MIN_ISLAND_M2 = 400.0


def _edge_degradation_mask(texture: np.ndarray, tmask: np.ndarray, tex_res: float) -> np.ndarray:
    """Bool (H, W) at texture resolution: True where the ortho (and the
    co-derived DTM/DSM) is degraded at the survey boundary."""
    from scipy import ndimage as ndi
    th, tw = tmask.shape
    if not tmask.any():
        return np.zeros_like(tmask)
    b = QUALITY_BLOCK
    H, W = -(-th // b) * b, -(-tw // b) * b
    gray = np.zeros((H, W), np.float32)
    msk = np.zeros((H, W), np.float32)
    gray[:th, :tw] = texture.astype(np.float32).mean(axis=2)
    msk[:th, :tw] = tmask
    # Measure detail only away from the no-data outline: its jagged
    # (stair-stepped) edge is itself high-frequency and would mask the smear.
    inner = ndi.binary_erosion(msk > 0, np.ones((3, 3), bool), iterations=max(1, int(1.0 / tex_res)))
    lap = np.abs(ndi.laplace(gray)) * inner
    lap = lap.reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    cov = inner.reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    k = max(3, int(round(QUALITY_WINDOW_M / (tex_res * b))))
    num, den = ndi.uniform_filter(lap, k), ndi.uniform_filter(cov, k)
    q = np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan)
    inside = msk.reshape(H // b, b, W // b, b).mean(axis=(1, 3)) > 0.5
    if inside.sum() < 16:
        return np.zeros_like(tmask)
    low = inside & (np.nan_to_num(q, nan=0.0) < QUALITY_FRAC * np.nanmedian(q[inside]))
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[:th, :tw] = texture
    rgb = rgb.reshape(H // b, b, W // b, b, 3).mean(axis=(1, 3))
    # Sea / basin water is saturated turquoise; the smear is a desaturated
    # blue-grey, so hue alone is not enough.
    sat = (rgb.max(axis=-1) - rgb.min(axis=-1)) / np.maximum(rgb.max(axis=-1), 1.0)
    water = (rgb[..., 2] > rgb[..., 0] + 10) & (rgb[..., 1] > rgb[..., 0]) & (sat > 0.3)
    low &= ~ndi.binary_opening(water, np.ones((3, 3), bool))
    boundary = ndi.binary_dilation(~inside, np.ones((3, 3), bool), iterations=max(1, int(QUALITY_WINDOW_M / 2.0 / (tex_res * b)))) & inside
    lab, n = ndi.label(low)
    if not n:
        return np.zeros_like(tmask)
    touching = np.unique(lab[boundary & low])
    bad = np.isin(lab, touching[touching > 0])
    bad = ndi.binary_dilation(bad, np.ones((3, 3), bool), iterations=2) & inside
    if bad.any():
        # Specks of valid ortho stranded inside the removed smear.
        keep, nk = ndi.label(inside & ~bad)
        if nk:
            sizes = np.bincount(keep.ravel()) * (tex_res * b) ** 2
            bad |= (keep > 0) & (sizes[keep] < QUALITY_MIN_ISLAND_M2)
    return np.repeat(np.repeat(bad, b, axis=0), b, axis=1)[:th, :tw]


def survey_available(survey_dir: Path = SURVEY_DIR) -> bool:
    meta = survey_dir / "survey_meta.json"
    if not meta.exists():
        return False
    outputs = json.loads(meta.read_text()).get("outputs", {})
    return all(Path(outputs.get(k, "")).exists() for k in ("dtm", "rgb"))


def _survey_meta(survey_dir: Path) -> dict:
    return json.loads((survey_dir / "survey_meta.json").read_text())


def geoid_undulation(lon: float, lat: float) -> float | None:
    """EGM2008 geoid height N (m) at a point: h_ellipsoidal = H_orthometric + N.

    Uses PROJ's EGM2008 grid (fetched from the PROJ CDN on first use and
    cached by PROJ locally). Returns None if the grid is unavailable, which
    PROJ otherwise signals by silently returning N = 0 ("ballpark").
    """
    try:
        import pyproj
        from pyproj import Transformer
        was = pyproj.network.is_network_enabled()
        pyproj.network.set_network_enabled(True)
        try:
            t = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
            _, _, H = t.transform(lon, lat, 100.0)
        finally:
            pyproj.network.set_network_enabled(was)
        n = 100.0 - float(H)
        return n if abs(n) > 1e-6 else None
    except Exception:
        return None


class _LatticeMap:
    """Exact local->target CRS mapping on a coarse lattice, bilinear in between."""

    def __init__(self, transformer, x0: float, x1: float, y0: float, y1: float, step: float = LATTICE_STEP_M):
        self.lx = np.arange(x0, x1 + step, step)
        self.ly = np.arange(y0, y1 + step, step)
        gx, gy = np.meshgrid(self.lx, self.ly)
        e, n = transformer.transform(gx.ravel(), gy.ravel())
        self.e = np.asarray(e).reshape(gx.shape)
        self.n = np.asarray(n).reshape(gx.shape)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fx = np.clip((x - self.lx[0]) / (self.lx[1] - self.lx[0]), 0, len(self.lx) - 1.000001)
        fy = np.clip((y - self.ly[0]) / (self.ly[1] - self.ly[0]), 0, len(self.ly) - 1.000001)
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - ix, fy - iy
        out = []
        for g in (self.e, self.n):
            v00, v01 = g[iy, ix], g[iy, ix + 1]
            v10, v11 = g[iy + 1, ix], g[iy + 1, ix + 1]
            out.append((v00 * (1 - tx) + v01 * tx) * (1 - ty) + (v10 * (1 - tx) + v11 * tx) * ty)
        return out[0], out[1]


def _read_window(ds, e: np.ndarray, n: np.ndarray, out_res: float, band_indexes, margin_px: int = 4):
    """Read the source window covering (e, n) at ~out_res, via overviews if coarser."""
    from rasterio.windows import Window, from_bounds
    left, right = float(np.nanmin(e)), float(np.nanmax(e))
    bottom, top = float(np.nanmin(n)), float(np.nanmax(n))
    pad = margin_px * max(ds.res)
    win = from_bounds(left - pad, bottom - pad, right + pad, top + pad, ds.transform)
    win = win.round_offsets().round_lengths().intersection(Window(0, 0, ds.width, ds.height))
    scale = max(1.0, out_res / max(ds.res))
    out_h = max(1, int(round(win.height / scale)))
    out_w = max(1, int(round(win.width / scale)))
    data = ds.read(band_indexes, window=win, out_shape=(len(band_indexes), out_h, out_w)
                   if isinstance(band_indexes, (list, tuple)) else (out_h, out_w))
    wt = ds.window_transform(win) * ds.window_transform(win).scale(win.width / out_w, win.height / out_h)
    return data, wt, win, (out_h, out_w)


def _sample(img: np.ndarray, transform, e: np.ndarray, n: np.ndarray, order: int = 1, cval=np.nan):
    """Sample a (H, W) raster at easting/northing with map_coordinates."""
    from scipy.ndimage import map_coordinates
    inv = ~transform
    col, row = inv * (e, n)
    return map_coordinates(img, [row - 0.5, col - 0.5], order=order, cval=cval, prefilter=False)


@dataclass
class SurveyScene:
    x0: float
    y0: float
    dx: float
    elev: np.ndarray            # (ny, nx) fused orthometric heights on the local grid
    coverage: np.ndarray        # (ny, nx) bool: survey DTM valid
    geoid_n: float
    datum_source: str
    base_sampler: object
    texture: np.ndarray | None = None      # (H, W, 3) uint8, row 0 = NORTH edge
    texture_mask: np.ndarray | None = None  # (H, W) bool
    tex_extent: tuple | None = None          # (x0, x1, y0, y1) in local metres
    stats: dict | None = None
    ndsm: np.ndarray | None = None           # (ny, nx) DSM - DTM on the elevation grid (datum-free), NaN = no data

    def rgb_at(self, xy: np.ndarray) -> np.ndarray:
        """Ortho colour (N, 3) uint8 at local xy; zeros outside the texture."""
        out = np.zeros((np.asarray(xy).shape[0], 3), dtype=np.uint8)
        if self.texture is None or self.tex_extent is None:
            return out
        xy = np.asarray(xy, dtype=float)
        tx0, tx1, ty0, ty1 = self.tex_extent
        th, tw = self.texture.shape[:2]
        col = np.floor((xy[:, 0] - tx0) / (tx1 - tx0) * tw).astype(np.int64)
        row = np.floor((ty1 - xy[:, 1]) / (ty1 - ty0) * th).astype(np.int64)
        ok = (col >= 0) & (col < tw) & (row >= 0) & (row < th)
        out[ok] = self.texture[row[ok], col[ok]]
        return out

    def ndsm_at(self, xy: np.ndarray) -> np.ndarray:
        """Nearest-cell normalized surface height at local xy (NaN outside / no data)."""
        if self.ndsm is None:
            return np.full(np.asarray(xy).shape[0], np.nan)
        xy = np.asarray(xy, dtype=float)
        ix = np.rint((xy[:, 0] - self.x0) / self.dx).astype(np.int64)
        iy = np.rint((xy[:, 1] - self.y0) / self.dx).astype(np.int64)
        ny, nx = self.ndsm.shape
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(xy.shape[0], np.nan)
        out[ok] = self.ndsm[iy[ok], ix[ok]]
        return out

    def sampler(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        fx = (xy[:, 0] - self.x0) / self.dx
        fy = (xy[:, 1] - self.y0) / self.dx
        ny, nx = self.elev.shape
        inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
        out = np.empty(xy.shape[0], dtype=float)
        if inside.any():
            ix = np.minimum(fx[inside].astype(np.int64), nx - 2)
            iy = np.minimum(fy[inside].astype(np.int64), ny - 2)
            tx, ty = fx[inside] - ix, fy[inside] - iy
            g = self.elev
            out[inside] = ((g[iy, ix] * (1 - tx) + g[iy, ix + 1] * tx) * (1 - ty)
                           + (g[iy + 1, ix] * (1 - tx) + g[iy + 1, ix + 1] * tx) * ty)
        if (~inside).any():
            out[~inside] = (np.asarray(self.base_sampler(xy[~inside]), dtype=float)
                            if self.base_sampler is not None else 0.0)
        return out


def _cache_key(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def load_survey_scene(
    local_crs: str,
    extent: tuple[float, float, float, float],
    base_sampler,
    anchor_xy: np.ndarray | None = None,
    cache_dir: Path | None = None,
    elev_res: float = 0.5,
    tex_max_px: int = 8192,
    survey_dir: Path = SURVEY_DIR,
    with_texture: bool = True,
) -> SurveyScene | None:
    """Build the fused terrain + ortho texture for the local-frame `extent`
    (x0, x1, y0, y1). Returns None when the survey cache is missing or does
    not overlap the scene. `anchor_xy` (street-node positions) is used only
    for the datum cross-check against the Copernicus base DEM."""
    if not survey_available(survey_dir):
        return None
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt

    meta = _survey_meta(survey_dir)
    x0, x1, y0, y1 = map(float, extent)
    key = _cache_key("survey_scene_v4", local_crs, round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2),
                     elev_res, tex_max_px, json.dumps(meta["sources"], sort_keys=True))
    cache_dir = Path(cache_dir) if cache_dir is not None else survey_dir
    npz_path = cache_dir / f"survey_scene_{key}.npz"

    to_survey = Transformer.from_crs(local_crs, meta["crs"], always_xy=True)
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    lmap = _LatticeMap(to_survey, x0, x1, y0, y1)

    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        scene = SurveyScene(
            x0=x0, y0=y0, dx=float(info["dx"]), elev=z["elev"], coverage=z["coverage"],
            geoid_n=float(info["geoid_n"]), datum_source=info["datum_source"], base_sampler=base_sampler,
            texture=z["texture"] if "texture" in z.files else None,
            texture_mask=z["texture_mask"] if "texture_mask" in z.files else None,
            tex_extent=tuple(info["tex_extent"]) if info.get("tex_extent") else None,
            stats=info.get("stats"),
            ndsm=z["ndsm"] if "ndsm" in z.files else None,
        )
        print(f"[survey] loaded cached scene {npz_path.name}")
        return scene

    # ── Elevation grid on the local frame ──────────────────────────────────
    nx = int(np.floor((x1 - x0) / elev_res)) + 1
    ny = int(np.floor((y1 - y0) / elev_res)) + 1
    gx = x0 + np.arange(nx) * elev_res
    gy = y0 + np.arange(ny) * elev_res
    GX, GY = np.meshgrid(gx, gy)
    E, N = lmap(GX, GY)

    with rasterio.open(meta["outputs"]["dtm"]) as ds:
        sb = ds.bounds
        if E.max() < sb.left or E.min() > sb.right or N.max() < sb.bottom or N.min() > sb.top:
            print("[survey] scene does not overlap the drone survey — keeping the base DEM")
            return None
        dtm, wt, _, _ = _read_window(ds, E, N, out_res=min(elev_res / 2.0, 0.25), band_indexes=1)
    dtm_ellip = _sample(dtm.astype(np.float32), wt, E, N).reshape(GX.shape)
    coverage = np.isfinite(dtm_ellip)

    ndsm = None
    dsm_path = meta["outputs"].get("dsm")
    if dsm_path and Path(dsm_path).exists():
        with rasterio.open(dsm_path) as ds:
            dsm, dwt, _, _ = _read_window(ds, E, N, out_res=min(elev_res / 2.0, 0.25), band_indexes=1)
        dsm_ellip = _sample(dsm.astype(np.float32), dwt, E, N).reshape(GX.shape)
        # Surface above bare earth. Both are ellipsoidal heights from the same
        # survey, so the difference needs no datum conversion at all.
        ndsm = (dsm_ellip - dtm_ellip).astype(np.float32)
    if coverage.mean() < 0.01:
        print("[survey] <1% of the scene is covered by the survey — keeping the base DEM")
        return None

    # ── Datum: ellipsoidal -> EGM2008 orthometric ─────────────────────────
    clon, clat = to_wgs84.transform((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    geoid_n = geoid_undulation(clon, clat)
    base_grid = (np.asarray(base_sampler(np.column_stack([GX.ravel(), GY.ravel()])), dtype=float).reshape(GX.shape)
                 if base_sampler is not None else np.zeros_like(GX))

    empirical = None
    if anchor_xy is not None and len(anchor_xy) and base_sampler is not None:
        a = np.asarray(anchor_xy, dtype=float)[:, :2]
        fa = SurveyScene(x0, y0, elev_res, np.where(coverage, dtm_ellip, np.nan), coverage, 0.0, "", None)
        inside = (a[:, 0] > x0) & (a[:, 0] < x1) & (a[:, 1] > y0) & (a[:, 1] < y1)
        if inside.sum() >= 20:
            d_ellip = fa.sampler(a[inside])
            d_base = np.asarray(base_sampler(a[inside]), dtype=float)
            ok = np.isfinite(d_ellip) & np.isfinite(d_base)
            if ok.sum() >= 20:
                diff = d_ellip[ok] - d_base[ok]
                empirical = {"median": float(np.median(diff)),
                             "mad": float(np.median(np.abs(diff - np.median(diff)))), "n": int(ok.sum())}

    if geoid_n is not None:
        datum_source = "EGM2008 geoid grid (PROJ)"
        offset = geoid_n
    elif empirical is not None:
        datum_source = f"empirical median vs base DEM at {empirical['n']} street nodes (geoid grid unavailable)"
        offset = empirical["median"]
    else:
        print("[survey] no geoid grid and no anchors for a datum estimate — keeping the base DEM")
        return None
    dtm_ortho = dtm_ellip - offset

    # Photogrammetric artifacts: the DTM has patches near 0 m ELLIPSOIDAL
    # (i.e. ~-23 m orthometric) at the survey edge / over water. No land in
    # the domain is metres below sea level, so treat those as no-coverage
    # and let the base DEM (sea = 0) fill in rather than render pits.
    artifacts = coverage & (dtm_ortho < MIN_PLAUSIBLE_ORTHO_M)
    if artifacts.any():
        print(f"[survey] masked {int(artifacts.sum())} DTM cells below "
              f"{MIN_PLAUSIBLE_ORTHO_M} m orthometric as artifacts")
        coverage = coverage & ~artifacts

    # ── Ortho texture on the local frame ──────────────────────────────────
    texture = tmask = tex_extent = None
    if with_texture:
        tex_res = max((x1 - x0) / tex_max_px, (y1 - y0) / tex_max_px, float(meta.get("ortho_res_m", 0.1)))
        tw = int(np.floor((x1 - x0) / tex_res))
        th = int(np.floor((y1 - y0) / tex_res))
        tx = x0 + (np.arange(tw) + 0.5) * tex_res
        ty = y1 - (np.arange(th) + 0.5) * tex_res          # row 0 = north edge
        texture = np.zeros((th, tw, 3), dtype=np.uint8)
        tmask = np.zeros((th, tw), dtype=bool)
        with rasterio.open(meta["outputs"]["rgb"]) as ds:
            ETX, NTY = lmap(*np.meshgrid(tx[[0, -1]], ty[[0, -1]]))
            rgb, rwt, win, oshape = _read_window(ds, ETX, NTY, out_res=tex_res, band_indexes=[1, 2, 3])
            msk = ds.read_masks(1, window=win, out_shape=oshape)
        chunk = 512
        for r0 in range(0, th, chunk):
            r1 = min(th, r0 + chunk)
            CX, CY = np.meshgrid(tx, ty[r0:r1])
            ce, cn = lmap(CX, CY)
            m = _sample(msk.astype(np.float32), rwt, ce, cn, order=1, cval=0.0).reshape(CX.shape)
            tmask[r0:r1] = m >= 127.5
            for ch in range(3):
                v = _sample(rgb[ch].astype(np.float32), rwt, ce, cn, order=1, cval=0.0)
                texture[r0:r1, :, ch] = np.clip(v, 0, 255).astype(np.uint8).reshape(CX.shape)
        tex_extent = (x0, x0 + tw * tex_res, y1 - th * tex_res, y1)

        # Degraded survey edge: drop it from the ortho AND the DTM coverage
        # (same photogrammetric block), so the base DEM + stylized ground
        # take over there instead of a smeared tent of terrain.
        degraded = _edge_degradation_mask(texture, tmask, tex_res)
        if degraded.any():
            tmask &= ~degraded
            col = np.clip(((GX - tex_extent[0]) / tex_res).astype(np.int64), 0, tw - 1)
            row = np.clip(((tex_extent[3] - GY) / tex_res).astype(np.int64), 0, th - 1)
            bad_grid = degraded[row, col]
            print(f"[survey] masked {degraded.mean():.1%} of the ortho as degraded survey edge "
                  f"({int((coverage & bad_grid).sum())} DTM cells)")
            coverage = coverage & ~bad_grid
        print(f"[survey] ortho texture {tw}x{th} @ {tex_res:.2f} m/px, coverage {tmask.mean():.0%}")

    # Feather into the base DEM across the coverage edge so there is no cliff
    # where survey data ends (the base DEM is 30 m and a surface model).
    dist = distance_transform_edt(coverage) * elev_res
    w = np.clip(dist / FEATHER_M, 0.0, 1.0)
    elev = np.where(coverage, w * dtm_ortho + (1.0 - w) * base_grid, base_grid).astype(np.float32)

    stats = {"coverage_frac": float(coverage.mean()), "geoid_n": geoid_n, "applied_offset": float(offset),
             "empirical_vs_base": empirical,
             "elev_range": [float(np.nanmin(elev)), float(np.nanmax(elev))]}
    print(f"[survey] DTM {nx}x{ny} @ {elev_res} m, coverage {coverage.mean():.0%}, "
          f"datum offset {offset:.3f} m ({datum_source})"
          + (f"; cross-check vs base DEM: median {empirical['median']:.2f} m, MAD {empirical['mad']:.2f} m"
             if empirical else ""))

    if ndsm is not None:
        ndsm = np.where(coverage, ndsm, np.nan).astype(np.float32)
    scene = SurveyScene(x0=x0, y0=y0, dx=elev_res, elev=elev, coverage=coverage, geoid_n=float(offset),
                        datum_source=datum_source, base_sampler=base_sampler, stats=stats, ndsm=ndsm,
                        texture=texture, texture_mask=tmask, tex_extent=tex_extent)

    info = {"dx": elev_res, "geoid_n": float(offset), "datum_source": datum_source,
            "tex_extent": list(scene.tex_extent) if scene.tex_extent else None, "stats": stats}
    arrays = {"elev": elev, "coverage": coverage, "info": np.array(json.dumps(info))}
    if scene.ndsm is not None:
        arrays["ndsm"] = scene.ndsm
    if scene.texture is not None:
        arrays["texture"], arrays["texture_mask"] = scene.texture, scene.texture_mask
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    return scene


def build_photoreal_ground(scene: SurveyScene, mesh_res: float = 2.0, z: float = 0.18):
    """Flat, textured ground mesh over the survey-covered part of the scene.

    Built flat like every other drapeable layer; terrain_mixin lifts it with
    the fused sampler when Terrain is enabled. Cells without ortho coverage
    are dropped so the stylized ground shows through outside the survey.
    Returns (pv.PolyData with texture coordinates, pv.Texture) or None.
    """
    import pyvista as pv
    if scene.texture is None or scene.tex_extent is None:
        return None
    tx0, tx1, ty0, ty1 = scene.tex_extent
    nx = max(2, int(round((tx1 - tx0) / mesh_res)) + 1)
    ny = max(2, int(round((ty1 - ty0) / mesh_res)) + 1)
    plane = pv.Plane(center=((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0, z), direction=(0, 0, 1),
                     i_size=tx1 - tx0, j_size=ty1 - ty0, i_resolution=nx - 1, j_resolution=ny - 1)
    pts = plane.points
    plane.active_texture_coordinates = np.column_stack(
        [(pts[:, 0] - tx0) / (tx1 - tx0), (pts[:, 1] - ty0) / (ty1 - ty0)]).astype(np.float32)

    th, tw = scene.texture_mask.shape
    cc = plane.cell_centers().points
    col = np.clip(((cc[:, 0] - tx0) / (tx1 - tx0) * tw).astype(int), 0, tw - 1)
    row = np.clip(((ty1 - cc[:, 1]) / (ty1 - ty0) * th).astype(int), 0, th - 1)
    keep = scene.texture_mask[row, col]
    if not keep.any():
        return None
    ground = plane.extract_cells(np.flatnonzero(keep)).extract_surface(algorithm="dataset_surface")
    gp = ground.points
    ground.active_texture_coordinates = np.column_stack(
        [(gp[:, 0] - tx0) / (tx1 - tx0), (gp[:, 1] - ty0) / (ty1 - ty0)]).astype(np.float32)
    # pv.Texture(ndarray) treats row 0 as the TOP of the image (it flips into
    # VTK's bottom-up layout itself), which matches our north-up rows, so no
    # manual flip — verified by a rendered-orientation regression test.
    texture = pv.Texture(np.ascontiguousarray(scene.texture))
    texture.mipmap = True
    texture.interpolate = True
    return ground, texture


def build_roof_overlay(buildings_mesh, scene: SurveyScene, lift: float = 0.03, min_up: float = 0.7):
    """Ortho-textured copy of the buildings' upward-facing roof faces.

    Valid because buildings are co-registered to the survey roofs
    (render.building_reconstruct), so each roof face projects onto its own
    rooftop in the orthophoto. Faces are lifted `lift` metres to avoid
    z-fighting with the PBR roof underneath; only faces inside the ortho
    coverage are kept. Returns (pv.PolyData with t-coords, pv.Texture) or None.
    """
    import pyvista as pv
    if scene.texture is None or scene.tex_extent is None or buildings_mesh is None or buildings_mesh.n_cells == 0:
        return None
    tri = buildings_mesh.triangulate().compute_normals(cell_normals=True, point_normals=False,
                                                      auto_orient_normals=True, consistent_normals=True)
    nz = np.asarray(tri.cell_data["Normals"])[:, 2]
    centers = tri.cell_centers().points
    tx0, tx1, ty0, ty1 = scene.tex_extent
    th, tw = scene.texture_mask.shape
    col = np.clip(((centers[:, 0] - tx0) / (tx1 - tx0) * tw).astype(int), 0, tw - 1)
    row = np.clip(((ty1 - centers[:, 1]) / (ty1 - ty0) * th).astype(int), 0, th - 1)
    in_tex = ((centers[:, 0] >= tx0) & (centers[:, 0] <= tx1) & (centers[:, 1] >= ty0) & (centers[:, 1] <= ty1))
    keep = (nz > min_up) & (centers[:, 2] > 1.0) & in_tex & scene.texture_mask[row, col]
    if not keep.any():
        return None
    roofs = tri.extract_cells(np.flatnonzero(keep)).extract_surface(algorithm="dataset_surface")
    for name in ("Normals", "vtkOriginalCellIds", "vtkOriginalPointIds"):
        for data in (roofs.cell_data, roofs.point_data):
            if name in data:
                del data[name]
    roofs.points[:, 2] += lift
    p = roofs.points
    roofs.active_texture_coordinates = np.column_stack(
        [(p[:, 0] - tx0) / (tx1 - tx0), (p[:, 1] - ty0) / (ty1 - ty0)]).astype(np.float32)
    return roofs
