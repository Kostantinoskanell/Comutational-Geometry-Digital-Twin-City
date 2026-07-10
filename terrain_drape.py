"""terrain_drape.py — Vectorized terrain-height helpers for scene draping.

Two roles:
1. Precompute DEM height profiles along car/bus paths at startup so the
   per-frame cost is O(N) NumPy advanced indexing instead of scipy interpolation.
2. Provide a vectorized lookup that the event loop calls every render tick.

Usage (from terrain_mixin.py)
------------------------------
    from terrain_drape import precompute_path_heights, car_heights_from_profiles

    prof, lens, counts = precompute_path_heights(
        car_paths, self._car_pose_on_path, dem, spacing_m=5.0,
    )
    # store in scene_state and call per-frame:
    h = car_heights_from_profiles(prof, lens, counts, edge_idx, dist)  # O(N) numpy
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# One-time precompute (called on terrain-toggle-on or startup)
# ---------------------------------------------------------------------------

def precompute_path_heights(
    car_paths: list,
    car_pose_fn,           # callable(path, dist_m) -> (x, y, z) array
    h_fn,                  # callable(xy: (N,2)) -> (N,) elevations
    spacing_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample terrain height at regular intervals along every car/bus path.

    Returns
    -------
    profiles     : float32 (n_paths, max_n)  — DEM height at each sample point
    path_lengths : float32 (n_paths,)        — total path length in metres
    path_counts  : int32   (n_paths,)        — number of valid samples per path
    """
    n_paths = len(car_paths)
    if n_paths == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    counts = np.array(
        [max(2, int(p["length"] / spacing_m) + 2) for p in car_paths],
        dtype=np.int32,
    )
    lens  = np.array([float(p["length"]) for p in car_paths], dtype=np.float32)
    max_n = int(counts.max())

    # Collect ALL sample points in one list so the DEM is called once (vectorized).
    all_xy: list[np.ndarray] = []
    cum_idx = np.zeros(n_paths + 1, dtype=np.int64)
    for i, (path, n) in enumerate(zip(car_paths, counts)):
        L  = float(path["length"])
        ds = np.linspace(0.0, L, int(n))
        xy = np.array(
            [car_pose_fn(path, float(d))[:2] for d in ds], dtype=np.float64
        )
        all_xy.append(xy)
        cum_idx[i + 1] = cum_idx[i] + int(n)

    all_xy_arr = np.concatenate(all_xy, axis=0)      # (total_samples, 2)
    all_h      = np.asarray(h_fn(all_xy_arr), dtype=np.float32)

    profiles = np.zeros((n_paths, max_n), dtype=np.float32)
    for i in range(n_paths):
        s, e = int(cum_idx[i]), int(cum_idx[i + 1])
        profiles[i, : e - s] = all_h[s:e]

    return profiles, lens, counts


# ---------------------------------------------------------------------------
# Per-frame lookup — no Python loops, fully vectorized
# ---------------------------------------------------------------------------

def car_heights_from_profiles(
    profiles: np.ndarray,       # (n_paths, max_n) float32
    path_lengths: np.ndarray,   # (n_paths,)        float32
    path_counts: np.ndarray,    # (n_paths,)         int32
    edge_idx: np.ndarray,       # (n,)               int64
    dist: np.ndarray,           # (n,)               float64
) -> np.ndarray:                # (n,)               float32
    """Vectorized O(N) terrain-height lookup from precomputed profiles.

    Uses NumPy advanced indexing — zero Python loops per frame.
    """
    if profiles.shape[0] == 0 or edge_idx.shape[0] == 0:
        return np.zeros(edge_idx.shape[0], dtype=np.float32)

    safe_e = np.clip(edge_idx, 0, profiles.shape[0] - 1).astype(np.int64)
    L      = path_lengths[safe_e].astype(np.float64)
    n_smp  = path_counts[safe_e].astype(np.int64)

    frac   = np.clip(np.asarray(dist, dtype=np.float64) / np.maximum(L, 1e-9), 0.0, 1.0)
    fi     = frac * (n_smp - 1)          # float index into profile row
    lo     = fi.astype(np.int64)
    hi     = np.minimum(lo + 1, n_smp - 1)
    alpha  = (fi - lo).astype(np.float32)

    # Advanced indexing: profiles[safe_e, lo] picks one element per row — O(N).
    h_lo = profiles[safe_e, lo]
    h_hi = profiles[safe_e, hi]
    return h_lo + alpha * (h_hi - h_lo)
