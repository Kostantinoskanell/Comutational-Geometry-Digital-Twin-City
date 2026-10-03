"""One-time preprocessing of the 5 cm drone survey into twin-ready rasters.

The source GeoTIFFs (EPSG:32636, 59000 x 41719 px, 4.5-7 GB each) are
strip-organised (256-row blocks spanning the full width) with no overviews
on the DSM/RGB, so every windowed read decodes whole strips — far too slow
to do at scene-build time. This streams each source once, in strip-aligned
row bands, and writes tiled, overviewed, compressed rasters:

    cache/survey/dtm_0p25m.tif    float32, NaN nodata, DEFLATE+pred3  (bare earth)
    cache/survey/dsm_0p25m.tif    float32, NaN nodata, DEFLATE+pred3  (surface)
    cache/survey/ortho_0p10m.tif  RGB uint8, JPEG/YCbCr, internal mask from alpha
    cache/survey/survey_meta.json provenance + source fingerprints

Elevations are kept ELLIPSOIDAL, exactly as surveyed; the twin-side loader
(render.survey_ground) derives the offset to the twin's orthometric DEM.

    python -m render.survey_prep [--src ~/Downloads/beirut] [--out cache/survey]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

DEFAULT_SRC = Path(os.environ.get("BEIRUT_SURVEY_DIR", Path.home() / "Downloads" / "beirut"))
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "cache" / "survey"

SOURCES = {
    "dtm": "Beirut_drone_*DTM_5cm_epsg32636*.tif*",
    "dsm": "Beirut_drone_*DSM_5cm_epsg32636*.tif*",
    "rgb": "Beirut_drone_*RGB_5cm_epsg32636*.tif*",
}
ELEV_FACTOR = 5   # 0.05 m -> 0.25 m
RGB_FACTOR = 2    # 0.05 m -> 0.10 m


def find_sources(src_dir: Path) -> dict[str, Path]:
    found = {}
    for key, pattern in SOURCES.items():
        hits = sorted(p for p in glob.glob(str(src_dir / pattern)) if not p.endswith((".tfw", ".ovr", ".aux.xml")))
        if not hits:
            raise FileNotFoundError(f"no {key.upper()} raster matching {pattern} in {src_dir}")
        found[key] = Path(hits[0])
    return found


def _fingerprint(p: Path) -> dict:
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}


def _band_rows(block_rows: int, factor: int, target_rows: int = 1300) -> int:
    """Row-band height that is a multiple of both the source strip height
    (so each strip is decoded exactly once) and the decimation factor."""
    step = int(np.lcm(block_rows, factor))
    return max(step, (target_rows // step) * step)


def _overview_levels(width: int, height: int, min_px: int = 256) -> list[int]:
    """Power-of-two overview factors down to ~min_px on the short side."""
    levels, f = [], 2
    while min(width, height) // f >= min_px:
        levels.append(f)
        f *= 2
    return levels


def _block_nanmean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f x f blocks ignoring NaN; all-NaN blocks stay NaN."""
    h, w = a.shape
    a = a[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f)
    valid = np.isfinite(a)
    s = np.where(valid, a, 0.0).sum(axis=(1, 3), dtype=np.float64)
    n = valid.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s / n).astype(np.float32)
    out[n == 0] = np.nan
    return out


def _pad_rows(a: np.ndarray, f: int, fill) -> np.ndarray:
    r = a.shape[-2] % f
    if r == 0:
        return a
    pad = [(0, 0)] * (a.ndim - 2) + [(0, f - r), (0, 0)]
    return np.pad(a, pad, constant_values=fill)


def downsample_elevation(src_path: Path, dst_path: Path, factor: int = ELEV_FACTOR) -> None:
    with rasterio.open(src_path) as src:
        out_w, out_h = src.width // factor, -(-src.height // factor)
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1, "crs": src.crs,
            "width": out_w, "height": out_h, "nodata": np.nan,
            "transform": src.transform * src.transform.scale(factor, factor),
            "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "deflate", "predictor": 3, "zlevel": 6, "BIGTIFF": "IF_SAFER",
        }
        band = _band_rows(src.block_shapes[0][0], factor)
        tmp = dst_path.with_suffix(".tmp.tif")
        t0 = time.time()
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, src.height, band):
                h = min(band, src.height - row)
                a = src.read(1, window=Window(0, row, src.width, h)).astype(np.float32)
                if src.nodata is not None and np.isfinite(src.nodata):
                    a[a == src.nodata] = np.nan
                a = _block_nanmean(_pad_rows(a, factor, np.nan), factor)
                dst.write(a[None], window=Window(0, row // factor, a.shape[1], a.shape[0]))
                print(f"  [{dst_path.name}] rows {row + h}/{src.height}  {time.time() - t0:.0f}s", flush=True)
        with rasterio.open(tmp, "r+") as dst:
            levels = _overview_levels(dst.width, dst.height)
            if levels:
                dst.build_overviews(levels, Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
        os.replace(tmp, dst_path)


def downsample_ortho(src_path: Path, dst_path: Path, factor: int = RGB_FACTOR, quality: int = 88) -> None:
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(src_path) as src:
            out_w, out_h = src.width // factor, -(-src.height // factor)
            profile = {
                "driver": "GTiff", "dtype": "uint8", "count": 3, "crs": src.crs,
                "width": out_w, "height": out_h,
                "transform": src.transform * src.transform.scale(factor, factor),
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "compress": "jpeg", "photometric": "ycbcr", "jpeg_quality": quality,
                "BIGTIFF": "IF_SAFER",
            }
            band = _band_rows(src.block_shapes[0][0], factor)
            tmp = dst_path.with_suffix(".tmp.tif")
            t0 = time.time()
            with rasterio.open(tmp, "w", **profile) as dst:
                for row in range(0, src.height, band):
                    h = min(band, src.height - row)
                    a = _pad_rows(src.read(window=Window(0, row, src.width, h)), factor, 0)
                    c, hh, ww = a.shape
                    ww -= ww % factor
                    blk = a[:, :, :ww].reshape(c, hh // factor, factor, ww // factor, factor).astype(np.float32)
                    alpha = blk[3] / 255.0
                    wsum = alpha.sum(axis=(1, 3))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rgb = (blk[:3] * alpha[None]).sum(axis=(2, 4)) / wsum[None]
                    rgb = np.nan_to_num(rgb, nan=0.0).clip(0, 255).astype(np.uint8)
                    mask = np.where(wsum / (factor * factor) >= 0.5, 255, 0).astype(np.uint8)
                    win = Window(0, row // factor, rgb.shape[2], rgb.shape[1])
                    dst.write(rgb, window=win)
                    dst.write_mask(mask, window=win)
                    print(f"  [{dst_path.name}] rows {row + h}/{src.height}  {time.time() - t0:.0f}s", flush=True)
            with rasterio.open(tmp, "r+") as dst:
                levels = _overview_levels(dst.width, dst.height)
                if levels:
                    dst.build_overviews(levels, Resampling.average)
            os.replace(tmp, dst_path)


def prepare(src_dir: Path = DEFAULT_SRC, out_dir: Path = DEFAULT_OUT, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = find_sources(src_dir)
    fingerprints = {k: _fingerprint(p) for k, p in sources.items()}
    meta_path = out_dir / "survey_meta.json"
    outputs = {"dtm": out_dir / "dtm_0p25m.tif", "dsm": out_dir / "dsm_0p25m.tif", "rgb": out_dir / "ortho_0p10m.tif"}

    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key, dst in outputs.items():
        fresh = dst.exists() and old.get("sources", {}).get(key) == fingerprints[key]
        if fresh and not force:
            print(f"[survey] {dst.name} up to date")
            continue
        print(f"[survey] building {dst.name} from {sources[key].name}")
        if key == "rgb":
            downsample_ortho(sources[key], dst)
        else:
            downsample_elevation(sources[key], dst)

    with rasterio.open(outputs["dtm"]) as ds:
        bounds, crs = list(ds.bounds), str(ds.crs)
    meta = {
        "sources": fingerprints,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "crs": crs,
        "bounds": bounds,
        "vertical_datum": "ellipsoidal (as surveyed); offset to the twin DEM derived at load time",
        "elev_res_m": 0.05 * ELEV_FACTOR,
        "ortho_res_m": 0.05 * RGB_FACTOR,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    meta = prepare(args.src.expanduser(), args.out, force=args.force)
    print(json.dumps(meta["outputs"], indent=2))


if __name__ == "__main__":
    main()
