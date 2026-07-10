"""sumo_engine.py — SUMO scene auto-build and cache helper.

Called by main_ast6.py when --engine sumo is requested but no --sumo-cfg is given.
Derives a stable cache key from (lat, lon, radius) and either returns the cached
.sumocfg path or triggers tools/build_sumo_scene.py to build a new one.

Public API
----------
ensure_sumo_scene(lat, lon, radius, cache_dir, *, vehicles, end_time) -> str
    Returns the absolute path to a valid .sumocfg, or "" on failure.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _scene_dir(lat: float, lon: float, radius: float, cache_dir: str) -> Path:
    """Return the canonical cache directory for this (lat, lon, radius) triple."""
    # Round to 4 dp ≈ 11 m precision — identical scenes reuse the same cache.
    lat_r = round(float(lat), 4)
    lon_r = round(float(lon), 4)
    rad_r = int(round(float(radius)))
    key   = f"{lat_r:.4f}_{lon_r:.4f}_{rad_r}m"
    return Path(cache_dir) / "sumo" / key


def ensure_sumo_scene(
    lat: float,
    lon: float,
    radius: float,
    cache_dir: str = ".cache",
    *,
    vehicles: int  = 300,
    end_time: float = 3600.0,
) -> str:
    """Return path to a .sumocfg for the given location, building it if absent.

    Parameters
    ----------
    lat, lon    : scene centre (WGS84 decimal degrees)
    radius      : scene radius in metres (matches --radius)
    cache_dir   : root cache directory (same as app --cache-dir)
    vehicles    : approximate concurrent SUMO vehicles (for randomTrips demand)
    end_time    : SUMO simulation end time in seconds

    Returns
    -------
    str — absolute path to the .sumocfg, or "" if the build failed.
    """
    if not lat or not lon:
        print("[sumo-engine] WARNING: lat/lon not available — cannot auto-build SUMO scene")
        return ""

    out_dir  = _scene_dir(lat, lon, radius, cache_dir)
    cfg_path = out_dir / "scene.sumocfg"

    # ── Already cached — return immediately ──────────────────────────────────
    if cfg_path.exists() and (out_dir / "scene.net.xml").exists():
        print(f"[sumo-engine] using cached scene: {cfg_path}")
        return str(cfg_path.resolve())

    # ── First run — build the scene ───────────────────────────────────────────
    print(
        f"[sumo-engine] Building SUMO scene for lat={lat:.4f} lon={lon:.4f} "
        f"radius={radius:.0f}m — first run, this may take ~60 s …"
    )
    print("[sumo-engine] (Press Ctrl+C to cancel and fall back to IDM engine)")

    # Locate build_sumo_scene.py relative to this file
    _here        = Path(__file__).parent
    build_script = _here / "tools" / "build_sumo_scene.py"
    if not build_script.exists():
        print(f"[sumo-engine] ERROR: build script not found: {build_script}")
        return ""

    cmd = [
        sys.executable,
        str(build_script),
        "--lat",      str(lat),
        "--lon",      str(lon),
        "--radius",   str(radius),
        "--out-dir",  str(out_dir),
        "--name",     "scene",
        "--vehicles", str(vehicles),
        "--end",      str(end_time),
    ]

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[sumo-engine] build_sumo_scene.py exited with code {result.returncode}")
            return ""
    except KeyboardInterrupt:
        print("\n[sumo-engine] build cancelled by user — falling back to IDM engine")
        return ""
    except Exception as exc:
        print(f"[sumo-engine] build failed: {exc}")
        return ""

    if not cfg_path.exists():
        print(f"[sumo-engine] ERROR: build finished but {cfg_path} not found")
        return ""

    print(f"[sumo-engine] scene ready: {cfg_path}")
    return str(cfg_path.resolve())
