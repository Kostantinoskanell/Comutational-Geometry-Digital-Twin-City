"""FloodMixin — Beirut pluvial-flood puddle overlay + green-corridor material overlay.

Imports precomputed flood-depth rasters and the green-corridor material/spine
layout produced by the sibling `Beirut_Project-main/` GPU pluvial flood solver
(read-only; this app never writes into it), reprojects them from the solver's
UTM 36N grid (EPSG:32636) into this app's local AEQD scene frame (the same
`to_local` transformer `overture_source._make_local_transformer` builds for the
session's geocoded address), resamples onto a regular grid, and renders two
STATIC `pv.StructuredGrid` overlays:

  - "flood_puddles"        blue depth-alpha overlay              (Part 3)
  - "corridor_materials"   8-class green-corridor GI zone overlay (Part 4 seed layout)

Both overlays are Beirut-specific — they only land somewhere sensible when the
app's `--address` geocodes to the corridor domain the solver covers (roughly
central Beirut). They are built ONCE after import (no animation timer) and
toggled on/off with `VisibilityOn()/Off()`, mirroring `heatmap_mixin.py`'s
`_toggle_aq_overlay` pattern.

Z handling: we deliberately do NOT try to align absolute elevation between the
flood solver's DEM and the twin's own terrain DEM (they use different vertical
datums — the flood LiDAR survey stores ellipsoidal heights with local sea level
around +26 m, not orthometric 0). Instead we treat the flood/corridor rasters
as pure depth/extent fields: they're built FLAT at a small constant z offset
(avoids z-fighting with roads/ground), exactly like every other draped layer
(roads, crosswalks, user buildings, ...) is built flat first — terrain
elevation is then applied uniformly by `terrain_mixin._apply_terrain_drape`,
which includes "flood_puddles_actor"/"corridor_materials_actor" in its
lift/restore lists, so these overlays follow the SAME terrain on/off toggle as
everything else. We use an explicit-point `pv.StructuredGrid` (not
`pv.ImageData`) specifically so `_drape_actor_points`'s generic per-point Z
lift (used for every other actor) works on it — `pv.ImageData` only supports a
flat, axis-aligned Z and can't be lifted point-by-point.

Expected `Beirut_Project-main/` output layout (produced by
`scripts/flood_gpu.py` via `scripts/run_corridor_study.sh`; the large `*.npy`
rasters are gitignored in the tracked checkout, so run the solver first if a
storm/phase combination is missing):

    output/terrain_cut_0.5/dem_transform.json          "before" grid georeference
    output/terrain_cut_corridor/dem_transform.json     "after" grid georeference
                                                        (falls back to the
                                                        "before" transform if
                                                        absent — same 0.5 m grid,
                                                        confirmed via
                                                        corridor_gi_cut/material.npy's
                                                        shape matching terrain_cut_0.5)
    output/corridor_runs/<phase>_<storm>/max_depth.npy (or final_depth.npy)
    output/corridor_gi_cut/spine.json                  corridor centerline (UTM)
    output/corridor_gi_cut/material.npy                8-class GI material raster
    output/corridor_gi_cut/material_summary.json       class label reference

Key 'no keybinding' — puddle/corridor overlays are toggled from the on-screen
control panel checkboxes (see main_ast6.py), not a hotkey, matching how the AQ
overlay's panel checkbox works.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv

try:
    from pyproj import Transformer as _Transformer
except Exception:
    _Transformer = None


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

BEIRUT_PROJECT_DIR = Path(__file__).resolve().parent / "Beirut_Project-main"
FLOOD_OUTPUT_DIR = BEIRUT_PROJECT_DIR / "output"
FLOOD_UTM_CRS = "EPSG:32636"

# Depths below this are treated as dry — avoids rendering solver noise as puddles.
DEPTH_DRY_THRESHOLD_M = 0.01

# Terrain dirs used by scripts/run_corridor_study.sh's default BEFORE/AFTER env
# vars — this is the canonical before/after pair the solver output is keyed on.
_TERRAIN_DIR_BY_PHASE = {
    "before": "terrain_cut_0.5",
    "after": "terrain_cut_corridor",
}

# Output dir the corridor before/after storm runs live under (run_corridor_study.sh
# default $OUT). Sibling variants exist (corridor_runs_v3, corridor_runs_v3_best,
# corridor_runs_v3_hllc) from later solver iterations; this is the one the
# published run_corridor_study.sh writes to by default.
CORRIDOR_RUNS_DIRNAME = "corridor_runs"

VALID_STORMS = ("t2", "t10", "t10cc", "t50", "flat30", "v1_nov2025")
VALID_PHASES = ("before", "after")

# material.npy class labels (output/corridor_gi_cut/material_summary.json)
MATERIAL_LABELS = {
    0: "(none)",
    1: "vehicular lane",
    2: "porous bikelane",
    3: "bioswale",
    4: "porous sidewalk",
    5: "garden",
    6: "bioretention pond / rain garden",
    7: "terrace",
}

# RGBA (0-1 floats) per material class. Bioswale / garden / bioretention pond
# read as green/water "GI" colors; vehicular lane / porous bikelane / porous
# sidewalk / terrace are neutral tones. Class 0 ("none") is fully transparent.
_MATERIAL_COLORS = {
    1: (0.55, 0.55, 0.58, 0.55),  # vehicular lane      — neutral grey
    2: (0.72, 0.58, 0.32, 0.55),  # porous bikelane      — neutral tan
    3: (0.18, 0.62, 0.38, 0.70),  # bioswale             — GI green
    4: (0.75, 0.71, 0.62, 0.50),  # porous sidewalk      — neutral beige
    5: (0.24, 0.75, 0.28, 0.65),  # garden               — GI green (brighter)
    6: (0.13, 0.55, 0.82, 0.75),  # bioretention pond    — GI blue-green water
    7: (0.66, 0.48, 0.38, 0.55),  # terrace              — neutral brown
}

# Z offsets (meters) above the twin's own terrain surface, to avoid z-fighting
# with roads/sidewalks/ground fill.
# Stacking (flat z): roads 0.10 < sidewalks 0.15 < photoreal ground 0.18 <
# corridor 0.19 < puddles / water 0.20. Overlays used to sit at 0.04/0.06,
# BELOW the road surface, so puddles were hidden exactly where floodwater
# runs (streets) and under the photo ground in the photoreal view.
_PUDDLE_Z_OFFSET = 0.20
_WATER_Z_OFFSET = 0.20        # physical water surface base
_CORRIDOR_Z_OFFSET = 0.19


# ---------------------------------------------------------------------------
# Colormaps (built once at import time, same idiom as heatmap_mixin.py)
# ---------------------------------------------------------------------------

def _build_puddle_cmap():
    """Blue alpha-ramp: transparent when dry, increasingly opaque/saturated
    with depth — same technique as heatmap_mixin.py's AQ colormap."""
    try:
        import matplotlib.colors as _mc
        import matplotlib.pyplot as _mplt

        # matplotlib.cm.get_cmap was removed in matplotlib>=3.9 (this env runs
        # 3.11); plt.get_cmap / matplotlib.colormaps[name] are the surviving
        # APIs. The bare string fallback below has NO alpha ramp, so getting
        # this wrong doesn't crash — it silently paints an opaque sheet
        # instead of a puddle overlay, which is much easier to miss.
        base = _mplt.get_cmap("Blues", 256)(np.linspace(0.20, 1.0, 256))
        alphas = np.clip(np.linspace(0.0, 0.75, 256), 0.0, 0.75)
        base[:, 3] = alphas
        return _mc.ListedColormap(base, name="flood_blue")
    except Exception:
        return "Blues"


def _build_corridor_cmap():
    """8-band discrete colormap, one solid band per material class 0-7 (class 0
    fully transparent). clim is set to [0, 8] by the caller so each integer
    class value lands in the middle of its 32-sample band."""
    try:
        import matplotlib.colors as _mc

        n_classes = 8
        arr = np.zeros((256, 4), dtype=float)
        for i in range(256):
            cls = min(int(i / 256 * n_classes), n_classes - 1)
            arr[i] = _MATERIAL_COLORS.get(cls, (0.0, 0.0, 0.0, 0.0))
        return _mc.ListedColormap(arr, name="corridor_materials")
    except Exception:
        return None


_PUDDLE_CMAP = _build_puddle_cmap()
_CORRIDOR_CMAP = _build_corridor_cmap()


# ---------------------------------------------------------------------------
# Small standalone helpers (no `self` needed)
# ---------------------------------------------------------------------------

def _cache_key(*parts: object) -> str:
    """Same idiom as app_core.py's _cache_key: short stable sha1 of the parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _grid_cell_centers_utm(transform: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) UTM cell-center coordinate grids, shape (height, width),
    from a dem_transform.json affine (minx, miny/maxy, res, width, height).
    Row 0 = north edge (maxy); row increases southward, matching how the
    solver's numpy rasters are laid out (standard raster/np.load convention)."""
    minx = float(transform["minx"])
    res = float(transform["res"])
    width = int(transform["width"])
    height = int(transform["height"])
    maxy = transform.get("maxy")
    if maxy is None:
        maxy = float(transform["miny"]) + height * res
    maxy = float(maxy)
    xs = minx + (np.arange(width) + 0.5) * res
    ys = maxy - (np.arange(height) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xs, ys)  # shape (height, width)
    return grid_x, grid_y


def _radius_mask(local_x: np.ndarray, local_y: np.ndarray, radius_m: float, margin_m: float = 200.0) -> np.ndarray:
    """Disc filter (scene radius + margin) around the scene origin — used by the
    time-series builder, which needs to apply the SAME crop to every frame's
    depth array (all frames share the same source-grid coordinates, only the
    depth values differ), so it needs the mask/indices, not filtered arrays."""
    if radius_m <= 0:
        return np.ones(local_x.shape, dtype=bool)
    r = float(radius_m) + float(margin_m)
    return (local_x ** 2 + local_y ** 2) <= r * r


class _RegularResampler:
    """Solver raster (regular UTM grid) -> twin grid (regular local grid).

    Replaces the Delaunay/griddata path: target cell centres are mapped
    EXACTLY into the solver's UTM frame (local -> WGS84 -> UTM with pyproj),
    and the raster is sampled there. Depth is first block-averaged to the
    target spacing (area-preserving, so narrow flooded channels are not
    aliased away by point sampling a 0.5 m grid at ~2 m), then interpolated
    bilinearly; categorical rasters (corridor materials) use the nearest
    cell. Cells outside the solver domain are 0 (dry / no material) instead
    of the old convex-hull nearest-neighbour extrapolation.

    Target grid extent/size are computed exactly as before (bounds of the
    solver cell centres inside the scene disc + margin, capped cells), so
    every overlay keeps its registration with the rest of the scene.
    """

    def __init__(self, transform: dict, to_local, radius: float, target_res_m: float,
                 max_grid_cells: int, shape: tuple[int, int] | None = None):
        tr = dict(transform)
        if shape is not None:
            tr["height"], tr["width"] = int(shape[0]), int(shape[1])
        self.minx = float(tr["minx"])
        self.res = float(tr["res"])
        self.width, self.height = int(tr["width"]), int(tr["height"])
        maxy = tr.get("maxy")
        self.maxy = float(maxy) if maxy is not None else float(tr["miny"]) + self.height * self.res

        gx, gy = _grid_cell_centers_utm(tr)
        utm_to_wgs84 = _Transformer.from_crs(FLOOD_UTM_CRS, "EPSG:4326", always_xy=True)
        lon, lat = utm_to_wgs84.transform(gx.ravel(), gy.ravel())
        lx, ly = to_local.transform(np.asarray(lon), np.asarray(lat))
        lx, ly = np.asarray(lx, dtype=float), np.asarray(ly, dtype=float)
        m = _radius_mask(lx, ly, radius) & np.isfinite(lx) & np.isfinite(ly)
        if m.sum() < 3:
            m = np.isfinite(lx) & np.isfinite(ly)
        self.scene_mask = m.reshape(self.height, self.width)   # solver cells inside the scene disc
        if m.sum() < 3:
            raise ValueError("too few valid reprojected points to resample onto a grid")
        x0, x1 = float(lx[m].min()), float(lx[m].max())
        y0, y1 = float(ly[m].min()), float(ly[m].max())
        self.nx = max(2, min(max_grid_cells, int((x1 - x0) / target_res_m) + 1))
        self.ny = max(2, min(max_grid_cells, int((y1 - y0) / target_res_m) + 1))
        self.dx = (x1 - x0) / max(self.nx - 1, 1)
        self.dy = (y1 - y0) / max(self.ny - 1, 1)
        self.x0, self.y0 = x0, y0

        tx, ty = np.meshgrid(np.linspace(x0, x1, self.nx), np.linspace(y0, y1, self.ny))
        tlon, tlat = to_local.transform(tx.ravel(), ty.ravel(), direction="INVERSE")
        wgs84_to_utm = _Transformer.from_crs("EPSG:4326", FLOOD_UTM_CRS, always_xy=True)
        e, n = wgs84_to_utm.transform(np.asarray(tlon), np.asarray(tlat))
        # fractional source (row, col) of every target cell centre
        self._col = (np.asarray(e) - self.minx) / self.res - 0.5
        self._row = (self.maxy - np.asarray(n)) / self.res - 0.5
        self.factor = max(1, int(np.floor(min(self.dx, self.dy) / self.res)))

    def sample(self, raster: np.ndarray, mode: str = "mean") -> np.ndarray:
        from scipy.ndimage import map_coordinates
        a = np.asarray(raster)
        if a.shape != (self.height, self.width):
            raise ValueError(f"raster shape {a.shape} != resampler source grid {(self.height, self.width)}")
        if mode == "nearest":
            out = map_coordinates(a.astype(np.float64), [np.rint(self._row), np.rint(self._col)],
                                  order=0, mode="constant", cval=0.0)
            return out.reshape(self.ny, self.nx)
        f = self.factor
        src = np.nan_to_num(a.astype(np.float64), nan=0.0)
        if f > 1:
            H, W = (self.height // f) * f, (self.width // f) * f
            src = src[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
            r = (self._row + 0.5) / f - 0.5
            c = (self._col + 0.5) / f - 0.5
        else:
            r, c = self._row, self._col
        out = map_coordinates(src, [r, c], order=1, mode="constant", cval=0.0, prefilter=False)
        return out.reshape(self.ny, self.nx)

    def native_max(self, raster: np.ndarray) -> float:
        """Maximum at the solver's own resolution within the scene disc: the
        display grid block-averages, so its max understates the peak depth."""
        vals = np.asarray(raster)[self.scene_mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.max()) if vals.size else 0.0

    def meta(self) -> dict:
        return {"origin_x": self.x0, "origin_y": self.y0, "spacing_x": self.dx, "spacing_y": self.dy,
                "nx": self.nx, "ny": self.ny}


class FloodMixin:

    # ------------------------------------------------------------------
    # Part 1 — offline ETL: reproject + resample + cache
    # ------------------------------------------------------------------

    def _get_local_transformer(self):
        """Return (and memoize) the same to_local AEQD transformer the app's
        scene build uses, per overture_source._make_local_transformer(lat, lon)
        for the session's geocoded --address. Cheap even the first time this is
        called standalone: _geocode_address() is disk-cached."""
        cached = getattr(self, "_flood_to_local", None)
        if cached is not None:
            return cached
        from overture_source import _geocode_address, _make_local_transformer

        address = str(getattr(self.args, "address", ""))
        lat, lon = _geocode_address(address)
        to_local, _ = _make_local_transformer(lat, lon)
        self._flood_to_local = to_local
        self._flood_origin_latlon = (lat, lon)
        return to_local

    def _warn_if_domain_far_from_scene(self, to_local, radius: float) -> None:
        """Sanity check: the flood/corridor domain (terrain_cut_0.5's footprint,
        same for every storm/phase) is a small, FIXED real-world area (Mar
        Mikhael/Achrafieh/Rmeil). If the scene's --address geocodes far from it
        relative to --radius, the built city and the flood overlay simply won't
        overlap — the overlay renders correctly but entirely outside the built
        disc, over the app's open-water backdrop, which looks like a broken/
        misaligned overlay even though the reprojection math is correct. Warn
        loudly and specifically instead of leaving the user to guess."""
        try:
            transform_path = FLOOD_OUTPUT_DIR / _TERRAIN_DIR_BY_PHASE["before"] / "dem_transform.json"
            if not transform_path.exists():
                return
            transform = _load_json(transform_path)
            minx, miny = float(transform["minx"]), float(transform["miny"])
            res, w, h = float(transform["res"]), int(transform["width"]), int(transform["height"])
            maxx, maxy = minx + w * res, miny + h * res
            corners_utm = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
            u2w = _Transformer.from_crs(FLOOD_UTM_CRS, "EPSG:4326", always_xy=True)
            lon, lat = u2w.transform(corners_utm[:, 0], corners_utm[:, 1])
            lx, ly = to_local.transform(np.asarray(lon), np.asarray(lat))
            lx, ly = np.asarray(lx), np.asarray(ly)
            corner_dist = np.hypot(lx, ly)
            nearest_edge_dist = float(corner_dist.min())  # optimistic (domain is convex-ish)
            center_dist = float(np.hypot(lx.mean(), ly.mean()))
            if nearest_edge_dist > float(radius):
                suggested_radius = int(np.ceil((corner_dist.max() + 50) / 100.0) * 100)
                print(
                    f"[flood] WARNING: the flood/corridor domain (Mar Mikhael/Achrafieh/Rmeil) is "
                    f"~{center_dist:.0f}m from the scene origin (--address geocode), and even its "
                    f"NEAREST corner is ~{nearest_edge_dist:.0f}m away — but --radius={radius:.0f}m. "
                    f"The built city and the flood overlay will NOT overlap; puddles/corridor will "
                    f"render outside the city disc, over open water. Fix: use --radius "
                    f"{suggested_radius} or larger, or pick an address inside the corridor itself "
                    f"(e.g. 'Sassine Square, Beirut, Lebanon', where the v1_nov2025 storm's real "
                    f"flooding was observed)."
                )
        except Exception as exc:
            print(f"[flood] domain-overlap sanity check skipped: {exc}")

    def import_flood_scenario(
        self,
        storm: str,
        phase: str,
        cache_dir=None,
        target_res_m: float = 1.0,
        max_grid_cells: int = 400,
    ) -> dict:
        """Read Beirut_Project-main flood-depth + corridor-design output for
        (storm, phase), reproject EPSG:32636 -> EPSG:4326 -> this app's local
        AEQD frame, resample to a regular grid, and cache the result under
        `cache_dir` (defaults to self.cache_dir) keyed by
        (storm, phase, address, radius, target_res_m) so re-running the same
        selection reuses the cache — mirrors app_core.py's `_cache_key` +
        `.npy`/`.json` sidecar idiom (see `_load_or_build_coverage_matrix_cached`).

        Returns a dict with keys: depth_grid, depth_meta, corridor_spine,
        corridor_material_grid, corridor_meta, ts_frames, ts_meta (the last
        two are None when no depth_*.npy time-series snapshots exist for this
        storm/phase — see `_load_or_build_flood_timeseries_cached`).
        """
        if phase not in VALID_PHASES:
            raise ValueError(f"phase must be one of {VALID_PHASES}, got {phase!r}")
        if storm not in VALID_STORMS:
            print(f"[flood] warning: storm {storm!r} not in known set {VALID_STORMS} — trying anyway")
        if _Transformer is None:
            raise RuntimeError("pyproj is required for flood import (pip install pyproj)")

        cache_dir = Path(cache_dir) if cache_dir is not None else Path(getattr(self, "cache_dir", "cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        to_local = self._get_local_transformer()

        address = str(getattr(self.args, "address", ""))
        radius = float(getattr(self.args, "radius", 0.0))
        self._warn_if_domain_far_from_scene(to_local, radius)

        # These two legs are independent data sources (storm depth rasters vs.
        # the corridor_gi_cut/ design) with different availability in a given
        # checkout — a missing storm scenario shouldn't also hide the corridor
        # overlay (which is far more likely to actually be present), so each
        # is imported in its own try/except rather than one failing the other.
        depth_grid = depth_meta = None
        try:
            depth_grid, depth_meta = self._load_or_build_flood_depth_cached(
                storm, phase, to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
            )
        except Exception as exc:
            print(f"[flood] flood-depth import failed (corridor overlay unaffected): {exc}")

        # Optional: the animated time-lapse. Independent try/except again —
        # a storm/phase with no saved frame snapshots (or a resample failure
        # on the time-series specifically) should still leave the static
        # max-depth overlay above intact.
        ts_frames = ts_meta = None
        try:
            ts_result = self._load_or_build_flood_timeseries_cached(
                storm, phase, to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
            )
            if ts_result is not None:
                ts_frames, ts_meta = ts_result
        except Exception as exc:
            print(f"[flood] flood time-series import failed (static overlay unaffected): {exc}")

        corridor_spine = corridor_material_grid = corridor_meta = None
        try:
            corridor_spine, corridor_material_grid, corridor_meta = self._load_or_build_corridor_cached(
                to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
            )
        except Exception as exc:
            print(f"[flood] corridor design import failed (puddle overlay unaffected): {exc}")

        return {
            "depth_grid": depth_grid,
            "depth_meta": depth_meta,
            "corridor_spine": corridor_spine,
            "corridor_material_grid": corridor_material_grid,
            "corridor_meta": corridor_meta,
            "ts_frames": ts_frames,
            "ts_meta": ts_meta,
        }

    # -- flood depth ----------------------------------------------------

    def _load_or_build_flood_depth_cached(
        self, storm, phase, to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
    ) -> tuple[np.ndarray, dict]:
        key = _cache_key("flood_v2_regular", storm, phase, address, radius, target_res_m, max_grid_cells)
        depth_path = cache_dir / f"flood_{key}_depth.npy"
        meta_path = cache_dir / f"flood_{key}_meta.json"

        if depth_path.exists() and meta_path.exists():
            print(f"[flood] loaded cached depth grid: {depth_path.name}")
            return np.load(depth_path), _load_json(meta_path)

        depth_grid, meta = self._build_flood_depth_grid(
            storm, phase, to_local, radius, target_res_m, max_grid_cells,
        )
        np.save(depth_path, depth_grid)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[flood] cached depth grid: {depth_path.name}")
        return depth_grid, meta

    @staticmethod
    def _resolve_terrain_transform(phase: str) -> tuple[dict, Path]:
        """Locate + load the dem_transform.json georeference for a phase,
        with the "after" -> "before" fallback (see module docstring). Shared
        by the static single-frame builder and the time-series builder so
        both resolve the exact same transform for a given phase."""
        terrain_dir_name = _TERRAIN_DIR_BY_PHASE[phase]
        transform_path = FLOOD_OUTPUT_DIR / terrain_dir_name / "dem_transform.json"
        if not transform_path.exists() and phase == "after":
            fallback = FLOOD_OUTPUT_DIR / _TERRAIN_DIR_BY_PHASE["before"] / "dem_transform.json"
            print(f"[flood] {transform_path} missing; falling back to before-phase transform: {fallback}")
            transform_path = fallback
        if not transform_path.exists():
            raise FileNotFoundError(
                f"flood dem_transform.json not found for phase={phase!r}: {transform_path}"
            )
        transform = _load_json(transform_path)
        if str(transform.get("crs", FLOOD_UTM_CRS)) != FLOOD_UTM_CRS:
            print(f"[flood] warning: unexpected CRS {transform.get('crs')!r} in {transform_path}")
        return transform, transform_path

    def _build_flood_depth_grid(
        self, storm, phase, to_local, radius, target_res_m, max_grid_cells,
    ) -> tuple[np.ndarray, dict]:
        transform, transform_path = self._resolve_terrain_transform(phase)

        out_dir = FLOOD_OUTPUT_DIR / CORRIDOR_RUNS_DIRNAME / f"{phase}_{storm}"
        depth_file = out_dir / "max_depth.npy"
        if not depth_file.exists():
            depth_file = out_dir / "final_depth.npy"
        if not depth_file.exists():
            raise FileNotFoundError(
                f"No flood depth raster for storm={storm!r} phase={phase!r}. Expected "
                f"{out_dir / 'max_depth.npy'} (produced by scripts/flood_gpu.py via "
                f"scripts/run_corridor_study.sh; *.npy outputs are gitignored so this "
                f"needs to be generated locally by running the solver first)."
            )
        depth = np.load(depth_file).astype(np.float64)
        depth[depth < DEPTH_DRY_THRESHOLD_M] = 0.0

        if (int(transform["height"]), int(transform["width"])) != depth.shape:
            raise ValueError(
                f"depth raster shape {depth.shape} != transform grid shape "
                f"{(int(transform['height']), int(transform['width']))} ({transform_path} vs {depth_file})"
            )
        rs = _RegularResampler(transform, to_local, radius, target_res_m, max_grid_cells)
        resampled = rs.sample(depth, "mean")
        x0, y0, dx, dy, nx, ny = rs.x0, rs.y0, rs.dx, rs.dy, rs.nx, rs.ny
        resampled[resampled < DEPTH_DRY_THRESHOLD_M] = 0.0
        resampled = resampled.astype(np.float32)

        meta = {
            "storm": storm,
            "phase": phase,
            "origin_x": x0, "origin_y": y0,
            "spacing_x": dx, "spacing_y": dy,
            "nx": nx, "ny": ny,
            "max_depth_m": float(resampled.max()) if resampled.size else 0.0,     # display grid
            "max_depth_native_m": rs.native_max(depth),                           # solver resolution
            "p95_depth_m": float(np.percentile(resampled, 95)) if resampled.size else 0.0,
            "units": "meters",
            "source_transform": str(transform_path),
            "source_depth_file": str(depth_file),
        }
        return resampled, meta

    # -- flood depth TIME SERIES (animated "puddles filling up") --------

    def _load_or_build_flood_timeseries_cached(
        self, storm, phase, to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
    ) -> tuple[np.ndarray, dict] | None:
        """Like `_load_or_build_flood_depth_cached` but for the FULL sequence
        of `depth_*.npy` snapshots `scripts/flood_gpu.py --save-every` writes
        during a solver run (e.g. every 300 simulated seconds), not just the
        final/max-depth raster. Returns (frames array shape (T, ny, nx),
        meta dict with "timestamps_s") or None if no frame snapshots exist
        for this storm/phase (older/partial solver runs, or one launched
        with --no-frames) — callers fall back to the static single-frame
        overlay in that case, so this is a soft/optional capability."""
        out_dir = FLOOD_OUTPUT_DIR / CORRIDOR_RUNS_DIRNAME / f"{phase}_{storm}"
        frame_paths = sorted(out_dir.glob("depth_*.npy"))
        if not frame_paths:
            return None

        key = _cache_key("flood_ts_v2_regular", storm, phase, address, radius, target_res_m, max_grid_cells)
        frames_path = cache_dir / f"flood_ts_{key}_frames.npy"
        meta_path = cache_dir / f"flood_ts_{key}_meta.json"

        if frames_path.exists() and meta_path.exists():
            print(f"[flood] loaded cached depth time-series: {frames_path.name}")
            return np.load(frames_path), _load_json(meta_path)

        frames, meta = self._build_flood_timeseries(
            storm, phase, to_local, radius, target_res_m, max_grid_cells, frame_paths,
        )
        np.save(frames_path, frames)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[flood] cached depth time-series: {frames_path.name} ({frames.shape[0]} frames)")
        return frames, meta

    def _build_flood_timeseries(
        self, storm, phase, to_local, radius, target_res_m, max_grid_cells, frame_paths,
    ) -> tuple[np.ndarray, dict]:
        transform, transform_path = self._resolve_terrain_transform(phase)

        # depth_NNNNNN.npy -> NNNNNN is the simulated-seconds timestamp
        # (scripts/flood_gpu.py's save-frame naming, confirmed against
        # run_meta.json / the solver's own "t=NNNN.Ns" progress log).
        timestamps = [int(p.stem.split("_")[1]) for p in frame_paths]

        first = np.load(frame_paths[0])
        if (int(transform["height"]), int(transform["width"])) != first.shape:
            raise ValueError(
                f"depth frame shape {first.shape} != transform grid shape "
                f"{(int(transform['height']), int(transform['width']))} ({transform_path} vs {frame_paths[0]})"
            )
        # One regular-grid resampler reused for every frame (see _RegularResampler)
        rs = _RegularResampler(transform, to_local, radius, target_res_m, max_grid_cells)
        resampler = {"x0": rs.x0, "y0": rs.y0, "dx": rs.dx, "dy": rs.dy, "nx": rs.nx, "ny": rs.ny}
        frames = np.empty((len(frame_paths), rs.ny, rs.nx), dtype=np.float32)
        native_max = 0.0
        for i, p in enumerate(frame_paths):
            depth = np.load(p).astype(np.float64)
            depth[depth < DEPTH_DRY_THRESHOLD_M] = 0.0
            native_max = max(native_max, rs.native_max(depth))
            grid_vals = rs.sample(depth, "mean")
            grid_vals[grid_vals < DEPTH_DRY_THRESHOLD_M] = 0.0
            frames[i] = grid_vals.astype(np.float32)

        wet = frames[frames > DEPTH_DRY_THRESHOLD_M]
        meta = {
            "storm": storm,
            "phase": phase,
            "timestamps_s": timestamps,
            "duration_s": timestamps[-1],
            "origin_x": resampler["x0"], "origin_y": resampler["y0"],
            "spacing_x": resampler["dx"], "spacing_y": resampler["dy"],
            "nx": resampler["nx"], "ny": resampler["ny"],
            "max_depth_m": float(frames.max()) if frames.size else 0.0,          # display grid
            "max_depth_native_m": native_max,                                     # solver resolution
            "wet_p95_depth_m": float(np.percentile(wet, 95)) if wet.size else 0.0,
            "units": "meters",
            "source_transform": str(transform_path),
            "source_frame_dir": str(frame_paths[0].parent),
            "n_frames": len(frame_paths),
        }
        return frames, meta

    # -- corridor design (spine + material raster) ----------------------

    def _load_or_build_corridor_cached(
        self, to_local, cache_dir, address, radius, target_res_m, max_grid_cells,
    ) -> tuple[dict, np.ndarray, dict]:
        key = _cache_key("corridor_v2_regular", address, radius, target_res_m, max_grid_cells)
        spine_path = cache_dir / f"corridor_{key}_spine.json"
        material_path = cache_dir / f"corridor_{key}_material.npy"
        meta_path = cache_dir / f"corridor_{key}_meta.json"

        if spine_path.exists() and material_path.exists() and meta_path.exists():
            print(f"[flood] loaded cached corridor design: {material_path.name}")
            return _load_json(spine_path), np.load(material_path), _load_json(meta_path)

        spine_local, material_grid, meta = self._build_corridor_design(
            to_local, radius, target_res_m, max_grid_cells,
        )
        np.save(material_path, material_grid)
        with open(spine_path, "w") as f:
            json.dump(spine_local, f)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[flood] cached corridor design: {material_path.name}")
        return spine_local, material_grid, meta

    def _build_corridor_design(self, to_local, radius, target_res_m, max_grid_cells) -> tuple[dict, np.ndarray, dict]:
        src_dir = FLOOD_OUTPUT_DIR / "corridor_gi_cut"
        material_src_path = src_dir / "material.npy"
        spine_src_path = src_dir / "spine.json"
        if not material_src_path.exists() or not spine_src_path.exists():
            raise FileNotFoundError(f"corridor design files missing under {src_dir}")

        material = np.load(material_src_path)

        # corridor_gi_cut has no dem_transform.json of its own; it shares the
        # pre-corridor 0.5 m terrain grid it was cut from (verified: material.npy's
        # shape (2348, 1142) matches terrain_cut_0.5/dem_transform.json exactly).
        transform_path = FLOOD_OUTPUT_DIR / _TERRAIN_DIR_BY_PHASE["before"] / "dem_transform.json"
        if not transform_path.exists():
            raise FileNotFoundError(f"corridor georeference transform not found: {transform_path}")
        transform = _load_json(transform_path)
        if (int(transform["height"]), int(transform["width"])) != material.shape:
            # A future material.npy cropped relative to the full terrain grid:
            # assume the same top-left (minx/maxy) origin/res at its own shape.
            print(
                f"[flood] corridor material.npy shape {material.shape} != terrain grid "
                f"{(int(transform['height']), int(transform['width']))}; assuming top-left aligned crop"
            )
        rs = _RegularResampler(transform, to_local, radius, target_res_m, max_grid_cells, shape=material.shape)
        material_grid = rs.sample(material, "nearest")
        x0, y0, dx, dy, nx, ny = rs.x0, rs.y0, rs.dx, rs.dy, rs.nx, rs.ny
        material_grid = np.rint(material_grid).astype(np.uint8)

        spine_raw = _load_json(spine_src_path)
        spine_local = self._reproject_spine(spine_raw, to_local)

        meta = {
            "origin_x": x0, "origin_y": y0,
            "spacing_x": dx, "spacing_y": dy,
            "nx": nx, "ny": ny,
            "class_labels": MATERIAL_LABELS,
            "source_material_file": str(material_src_path),
            "source_spine_file": str(spine_src_path),
            "source_transform": str(transform_path),
        }
        return spine_local, material_grid, meta

    @staticmethod
    def _reproject_spine(spine_raw: dict, to_local) -> dict:
        """spine.json format: {"crs": "EPSG:32636", "spine": [[x,y], ...]}."""
        pts = np.asarray(spine_raw.get("spine", []), dtype=float)
        if pts.size == 0:
            return {"crs": "local_aeqd", "spine": []}
        utm_to_wgs84 = _Transformer.from_crs(FLOOD_UTM_CRS, "EPSG:4326", always_xy=True)
        lon, lat = utm_to_wgs84.transform(pts[:, 0], pts[:, 1])
        local_x, local_y = to_local.transform(np.asarray(lon), np.asarray(lat))
        local_pts = np.column_stack([np.asarray(local_x), np.asarray(local_y)]).tolist()
        return {"crs": "local_aeqd", "spine": local_pts}

    # ------------------------------------------------------------------
    # Terrain height lookup (reuses terrain_mixin.py's DEM sampler)
    # ------------------------------------------------------------------

    def _build_structured_overlay_points(
        self, nx: int, ny: int, x0: float, y0: float, dx: float, dy: float, z_offset: float,
    ) -> np.ndarray:
        """(nx*ny, 3) point array in the row-major order pv.StructuredGrid
        expects for dimensions=(nx, ny, 1) (x varies fastest).

        Built FLAT at a constant z_offset, matching how every other draped
        layer (roads, crosswalks, user buildings, ...) is built — terrain
        elevation is applied afterwards by `terrain_mixin._apply_terrain_drape`
        (which now includes "flood_puddles_actor"/"corridor_materials_actor" in
        its lift/restore lists), so this overlay tracks the SAME on/off terrain
        toggle as everything else instead of always sampling the DEM regardless
        of whether draping is currently active."""
        xs = x0 + np.arange(nx) * dx
        ys = y0 + np.arange(ny) * dy
        grid_x, grid_y = np.meshgrid(xs, ys)  # shape (ny, nx), x fastest per row
        z = np.full(nx * ny, float(z_offset), dtype=float)
        return np.column_stack([grid_x.ravel(), grid_y.ravel(), z])

    # ------------------------------------------------------------------
    # Part 3 — puddle overlay (static)
    # ------------------------------------------------------------------

    def _init_puddle_overlay(
        self, depth_grid: np.ndarray, origin: tuple[float, float], spacing: tuple[float, float],
        clim_max: float | None = None,
    ) -> None:
        """Build the flood-depth ImageData-style overlay (StructuredGrid, see
        module docstring for why) and add it to the plotter, hidden until the
        user toggles it via the control-panel checkbox.

        `clim_max` lets a caller pin the colour scale to a value other than
        this grid's own wet-cell p95 — used by the animated time-lapse path
        (`_init_puddle_timeseries_overlay`), which seeds this with an all-dry
        frame 0 (own p95 would be 0) but wants the FINAL colour range fixed
        up front to the whole time-series' wet p95, so the scale doesn't
        visibly rescale itself as puddles fill in during playback."""
        self._flood_visible = False
        self._flood_grid = None
        self._flood_actor = None

        if depth_grid is None or getattr(depth_grid, "size", 0) == 0:
            print("[flood] puddle overlay skipped — empty depth grid")
            return
        if self.plotter is None:
            return

        ny, nx = depth_grid.shape
        x0, y0 = origin
        dx, dy = spacing

        try:
            points = self._build_structured_overlay_points(nx, ny, x0, y0, dx, dy, _PUDDLE_Z_OFFSET)
            grid = pv.StructuredGrid()
            grid.points = points
            grid.dimensions = (nx, ny, 1)
            grid.point_data["depth"] = depth_grid.astype(np.float32).ravel()

            if clim_max is None:
                # p95 over WET cells only — the grid is overwhelmingly dry (0.0),
                # so a whole-grid percentile is ~0 and the colour scale saturates
                # at the 5cm floor, making all real puddle depths look identical.
                wet = depth_grid[depth_grid > DEPTH_DRY_THRESHOLD_M]
                p95 = float(np.percentile(wet, 95)) if wet.size else 0.0
                clim_max = max(p95, 0.05)
            else:
                clim_max = max(float(clim_max), 0.05)

            actor = self.plotter.add_mesh(
                grid,
                scalars="depth",
                cmap=_PUDDLE_CMAP,
                clim=[0.0, clim_max],
                show_scalar_bar=False,
                lighting=False,
                opacity=1.0,   # alpha baked into colormap
                reset_camera=False,
                name="flood_puddles",
            )
            actor.VisibilityOff()
            self._flood_grid = grid
            self._flood_actor = actor
            self.scene_state["flood_puddles_actor"] = actor
            # Photoreal counterpart: physical water surface (render.water) on
            # the same grid; shown instead of the legend colours in the
            # photoreal view (see _sync_flood_layers).
            self._flood_water = None
            # Keyed on the render quality, not on self.materials: the flood
            # import runs before the post-FX/material manager is created.
            if str(getattr(self.args, "render_quality", "quality")) != "legacy" and \
                    not bool(getattr(self.args, "no_physical_sky", False)):
                try:
                    from render.water import FloodWaterSurface
                    # Base just above the photoreal ground (0.18): the analysis
                    # overlay's 0.06 would hide any water shallower than 12 cm
                    # under the photo.
                    self._flood_water = FloodWaterSurface(self.plotter, nx, ny, x0, y0, dx, dy,
                                                          _WATER_Z_OFFSET, depth_grid.astype(np.float32))
                    self._flood_water.set_visible(False)
                    self.scene_state["flood_water_actor"] = self._flood_water.actor
                except Exception as _wexc:
                    print(f"[flood] physical water surface unavailable: {_wexc}")
                    self._flood_water = None
            print(
                f"[flood] puddle overlay {nx}x{ny} cell {dx:.2f}x{dy:.2f} m "
                f"max_depth={float(depth_grid.max()):.2f}m — toggle via control panel"
            )
        except Exception as exc:
            print(f"[flood] puddle overlay init failed: {exc}")
            self._flood_grid = None
            self._flood_actor = None

    def _toggle_flood_overlay(self) -> None:
        if getattr(self, "_flood_actor", None) is None:
            print("[flood] no puddle overlay built — run with --flood-analysis first")
            return
        self._flood_visible = not bool(getattr(self, "_flood_visible", False))
        try:
            self._sync_flood_layers()
            self.plotter.render()
        except Exception:
            pass
        print(f"[flood] puddle overlay {'ON' if self._flood_visible else 'OFF'}")

    def _sync_flood_layers(self) -> None:
        """Puddles ON: physical water in the photoreal view, depth legend
        colours in the analysis view (and without a survey)."""
        vis = bool(getattr(self, "_flood_visible", False))
        water = getattr(self, "_flood_water", None)
        photoreal = water is not None and bool(self.scene_state.get("photoreal_ground", False))
        if getattr(self, "_flood_actor", None) is not None:
            self._flood_actor.SetVisibility(vis and not photoreal)
        if water is not None:
            water.set_visible(vis and photoreal)

    # ------------------------------------------------------------------
    # Part 3b — puddle TIME-LAPSE animation (optional, needs depth_*.npy
    # frame snapshots — see `_load_or_build_flood_timeseries_cached`)
    # ------------------------------------------------------------------

    def _init_puddle_timeseries_overlay(self, frames: np.ndarray | None, meta: dict | None) -> None:
        """Wire up animated playback state on top of the puddle actor built by
        `_init_puddle_overlay`. Must be called AFTER `_init_puddle_overlay` has
        built `self._flood_grid`/`self._flood_actor` from `frames[0]` (the
        driest frame) — this only adds the frame stack + timing state that
        `_animate_flood_puddles` steps through; it does not build any new
        geometry itself, so `_flood_grid`'s point/cell layout (and therefore
        terrain draping, visibility toggling, etc.) is identical whether or
        not a time-series happens to be available for this storm/phase."""
        self._flood_ts_frames = None
        self._flood_ts_timestamps = None
        self._flood_ts_playing = False

        if frames is None or meta is None or getattr(self, "_flood_grid", None) is None:
            return
        timestamps = np.asarray(meta.get("timestamps_s", []), dtype=float)
        if timestamps.size != frames.shape[0] or timestamps.size < 2:
            print("[flood] time-series has too few frames to animate — using static overlay only")
            return

        self._flood_ts_frames = frames
        self._flood_ts_timestamps = timestamps
        self._flood_ts_duration = float(timestamps[-1])
        self._flood_ts_sim_time = 0.0
        self._flood_ts_last_t = None
        # Compress the whole simulated storm into ~40s of real playback time
        # by default — long enough to read as a filling puddle, short enough
        # that a viewer doesn't wait minutes for a demo. `_toggle_flood_animation`
        # doesn't expose a speed control yet; this is a single sane default.
        self._flood_ts_speed = max(self._flood_ts_duration / 40.0, 1.0)
        print(
            f"[flood] time-lapse ready: {frames.shape[0]} frames over {self._flood_ts_duration:.0f}s "
            f"simulated (~{self._flood_ts_duration / self._flood_ts_speed:.0f}s real playback, looping) "
            f"— toggle via control panel"
        )

    def _animate_flood_puddles(self, _: int) -> None:
        """Timer callback (registered like `_animate_aq_overlay`/`_animate_tod`):
        advances simulated flood time and blends the two bracketing frames
        into `self._flood_grid`'s "depth" array in place, so puddles visibly
        fill in (and drain) as the storm replays. No-ops instantly unless the
        user has actually pressed play, so it's cheap to poll every tick."""
        if getattr(self, "flood_lab", None) is not None:
            try:
                self._flood_lab_tick()
            except Exception as exc:
                if not getattr(self, "_lab_tick_warned", False):
                    import traceback
                    traceback.print_exc()
                    print(f"[flood-lab] tick error: {exc}")
                    self._lab_tick_warned = True
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        if not bool(getattr(self, "_flood_ts_playing", False)):
            return
        frames = getattr(self, "_flood_ts_frames", None)
        timestamps = getattr(self, "_flood_ts_timestamps", None)
        grid = getattr(self, "_flood_grid", None)
        if frames is None or timestamps is None or grid is None:
            return

        import time as _time
        now = _time.perf_counter()
        last = getattr(self, "_flood_ts_last_t", None)
        self._flood_ts_last_t = now
        if last is None:
            return  # first tick after play: just establish the dt reference

        dt_real = max(0.0, now - last)
        speed = float(getattr(self, "_flood_ts_speed", 60.0))
        duration = float(getattr(self, "_flood_ts_duration", timestamps[-1]))
        sim_time = float(getattr(self, "_flood_ts_sim_time", 0.0)) + dt_real * speed
        if duration > 0:
            sim_time = sim_time % duration  # loop the storm continuously
        self._flood_ts_sim_time = sim_time

        try:
            n = timestamps.shape[0]
            idx = int(np.searchsorted(timestamps, sim_time, side="right")) - 1
            idx = max(0, min(idx, n - 2))
            t0, t1 = float(timestamps[idx]), float(timestamps[idx + 1])
            alpha = 0.0 if t1 <= t0 else float(np.clip((sim_time - t0) / (t1 - t0), 0.0, 1.0))
            blended = frames[idx] * (1.0 - alpha) + frames[idx + 1] * alpha

            existing = grid.point_data["depth"]
            np.copyto(existing, blended.ravel().astype(np.float32), casting="unsafe")
            grid.GetPointData().Modified()
            _water = getattr(self, "_flood_water", None)
            if _water is not None:
                _water.set_depth(blended)
                _water.advance(now)
            _rain_txt = self._drive_rain_from_storm(sim_time)

            self.plotter.add_text(
                f"Flood time-lapse — t={sim_time:04.0f}s / {duration:.0f}s  "
                f"({self.scene_state.get('flood_storm', '')}, {self.scene_state.get('flood_phase', '')})"
                f"{_rain_txt}",
                position=(0.02, 0.95), name="flood_ts_label", font_size=10, viewport=True, color="#bfe3ff",
            )
        except Exception as exc:
            print(f"[flood] time-lapse animation error: {exc}")

    def _drive_rain_from_storm(self, sim_time: float, storm_name: str | None = None) -> str:
        """Rain streak density + road wetness from the storm hyetograph at the
        replay time (render.hyetograph); returns a HUD suffix."""
        w = getattr(self, "weather", None)
        if w is None:
            return ""
        from render.hyetograph import CLOUDBURST_MM_H, cumulative_mm, intensity_mm_h, load_storm
        name = str(storm_name or self.scene_state.get("flood_storm", ""))
        if getattr(self, "_flood_storm_name", None) != name:
            self._flood_storm_def = load_storm(name)
            self._flood_storm_name = name
        storm = self._flood_storm_def
        if storm is None:
            return ""
        if "_weather_before_flood" not in self.__dict__:
            self._weather_before_flood = w.get("mode", "clear")
        i = intensity_mm_h(storm, sim_time)
        w["rain_scale"] = min(1.0, i / CLOUDBURST_MM_H)
        w["wet_override"] = float(np.clip(cumulative_mm(storm, sim_time) / 2.0, 0.0, 1.0))   # wet after ~2 mm
        w["mode"] = "rain"
        return f"  |  rain {i:.0f} mm/h, {cumulative_mm(storm, sim_time):.0f} mm"

    def _release_rain_from_storm(self) -> None:
        w = getattr(self, "weather", None)
        prev = self.__dict__.pop("_weather_before_flood", None)
        if w is None or prev is None:
            return
        w["rain_scale"] = 1.0
        w["wet_override"] = None
        if prev != "rain":
            self._clear_weather_actors()
            w["mode"] = prev
            if prev == "clear":
                self._restore_car_speeds()
                w["wet_factor"] = 0.0
                w["rain_pts"] = None
                self._reset_wet_road()

    def _toggle_flood_animation(self) -> None:
        if getattr(self, "_flood_ts_frames", None) is None:
            print(
                "[flood] no time-series loaded for this storm/phase — animation needs "
                "depth_*.npy frame snapshots (scripts/flood_gpu.py --save-every); "
                "the static max-depth puddle overlay is still available."
            )
            return
        self._flood_ts_playing = not bool(getattr(self, "_flood_ts_playing", False))
        if self._flood_ts_playing:
            self._flood_ts_last_t = None  # reset dt reference so the first tick doesn't jump
            if not bool(getattr(self, "_flood_visible", False)):
                self._toggle_flood_overlay()  # auto-show puddles when starting playback
        try:
            self.plotter.remove_actor("flood_ts_label", reset_camera=False)
        except Exception:
            pass
        if not self._flood_ts_playing:
            self._release_rain_from_storm()
        print(f"[flood] time-lapse {'PLAYING' if self._flood_ts_playing else 'PAUSED'}")

    # ------------------------------------------------------------------
    # Part 4 (seed layout only) — corridor material overlay (static)
    # ------------------------------------------------------------------

    def _init_corridor_overlay(self, material_grid: np.ndarray, origin: tuple[float, float], spacing: tuple[float, float]) -> None:
        """Build the 7-color (8-class, class 0 transparent) green-corridor
        material overlay. Same static/toggle pattern as the puddle overlay."""
        self._corridor_visible = False
        self._corridor_grid = None
        self._corridor_actor = None

        if material_grid is None or getattr(material_grid, "size", 0) == 0:
            print("[flood] corridor overlay skipped — empty material grid")
            return
        if self.plotter is None:
            return

        ny, nx = material_grid.shape
        x0, y0 = origin
        dx, dy = spacing

        try:
            points = self._build_structured_overlay_points(nx, ny, x0, y0, dx, dy, _CORRIDOR_Z_OFFSET)
            grid = pv.StructuredGrid()
            grid.points = points
            grid.dimensions = (nx, ny, 1)
            material_f = material_grid.astype(np.float32).ravel()
            # Fully-transparent class 0 cells still need a real scalar value
            # (alpha=0 in the cmap hides them; NaN would look identical but
            # values are cheaper/simpler to keep as-is here).
            grid.point_data["material"] = material_f

            actor = self.plotter.add_mesh(
                grid,
                scalars="material",
                cmap=_CORRIDOR_CMAP,
                clim=[0.0, 8.0],
                show_scalar_bar=False,
                lighting=False,
                opacity=1.0,
                reset_camera=False,
                name="corridor_materials",
            )
            actor.VisibilityOff()
            self._corridor_grid = grid
            self._corridor_actor = actor
            self.scene_state["corridor_materials_actor"] = actor
            n_active = int(np.count_nonzero(material_grid))
            print(
                f"[flood] corridor overlay {nx}x{ny} cell {dx:.2f}x{dy:.2f} m "
                f"({n_active} GI cells) — toggle via control panel"
            )
        except Exception as exc:
            print(f"[flood] corridor overlay init failed: {exc}")
            self._corridor_grid = None
            self._corridor_actor = None

    def _toggle_corridor_overlay(self) -> None:
        if getattr(self, "_corridor_actor", None) is None:
            print("[flood] no corridor overlay built — run with --flood-analysis first")
            return
        self._corridor_visible = not bool(getattr(self, "_corridor_visible", False))
        try:
            if self._corridor_visible:
                self._corridor_actor.VisibilityOn()
            else:
                self._corridor_actor.VisibilityOff()
            self.plotter.render()
        except Exception:
            pass
        print(f"[flood] corridor overlay {'ON' if self._corridor_visible else 'OFF'}")

    # ------------------------------------------------------------------
    # Orchestration — called once at startup when --flood-analysis is set
    # ------------------------------------------------------------------

    def _import_and_render_flood(self) -> None:
        """Import the selected storm/phase scenario + the corridor seed design,
        then build both overlays. Best-effort: prints and returns on failure
        (e.g. missing solver output) rather than aborting scene startup."""
        storm = str(getattr(self.args, "flood_storm", "v1_nov2025"))
        phase = str(getattr(self.args, "flood_phase", "after"))
        self._stage(f"Importing Beirut flood scenario: storm={storm} phase={phase} ...") \
            if hasattr(self, "_stage") else print(f"[flood] importing storm={storm} phase={phase} ...")
        try:
            result = self.import_flood_scenario(storm, phase)
        except Exception as exc:
            print(f"[flood] import failed — flood/corridor overlays disabled: {exc}")
            return

        ts_frames, ts_meta = result["ts_frames"], result["ts_meta"]
        depth_meta = result["depth_meta"]
        if ts_frames is not None and ts_meta is not None:
            # Prefer the animated time-lapse when frame snapshots exist: seed
            # the puddle grid with frame 0 (driest state, ready to "fill up")
            # but pin its colour scale to the whole time-series' wet p95 up
            # front (see `_init_puddle_overlay`'s clim_max docstring) so the
            # scale doesn't visibly rescale itself as playback progresses.
            self._init_puddle_overlay(
                ts_frames[0],
                origin=(ts_meta["origin_x"], ts_meta["origin_y"]),
                spacing=(ts_meta["spacing_x"], ts_meta["spacing_y"]),
                clim_max=ts_meta.get("wet_p95_depth_m"),
            )
            self._init_puddle_timeseries_overlay(ts_frames, ts_meta)
        elif result["depth_grid"] is not None and depth_meta is not None:
            self._init_puddle_overlay(
                result["depth_grid"],
                origin=(depth_meta["origin_x"], depth_meta["origin_y"]),
                spacing=(depth_meta["spacing_x"], depth_meta["spacing_y"]),
            )
        else:
            print(f"[flood] no depth data for storm={storm} phase={phase} — puddle overlay skipped")

        corridor_meta = result["corridor_meta"]
        if result["corridor_material_grid"] is not None and corridor_meta is not None:
            self._init_corridor_overlay(
                result["corridor_material_grid"],
                origin=(corridor_meta["origin_x"], corridor_meta["origin_y"]),
                spacing=(corridor_meta["spacing_x"], corridor_meta["spacing_y"]),
            )
        else:
            print("[flood] no corridor design data — corridor overlay skipped")

        self.scene_state["flood_storm"] = storm
        self.scene_state["flood_phase"] = phase
        self.scene_state["corridor_spine_local"] = result["corridor_spine"]
        _m = ts_meta or depth_meta or {}
        max_depth = _m.get("max_depth_native_m", _m.get("max_depth_m", 0.0))   # true (0.5 m) peak
        animated = "yes — toggle time-lapse via control panel" if ts_frames is not None else "no (static overlay only)"
        print(
            f"[flood] Beirut flood analysis ready: storm={storm} phase={phase} "
            f"max_depth={max_depth:.2f}m  animated={animated}"
        )
