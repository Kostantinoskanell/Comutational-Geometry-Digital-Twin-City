"""demand_model.py — gravity-based origin–destination trip demand.

Replaces uniform-random O-D selection with a gravity model so traffic patterns
*emerge* from land use instead of being scripted:

  • Production zones  = residential graph nodes  (where trips start in the AM)
  • Attraction zones  = commercial/retail nodes   (where trips head in the AM)
  • Time-of-day flips the dominant direction (home→work in the morning peak,
    work→home in the evening peak, mixed midday/night).
  • Destination choice is distance-decayed: P(dest) ∝ A_dest · exp(−β·d), so
    most trips go to *nearby* attractors — the core gravity behaviour.
  • Zone weights use local clustering density (a downtown cluster of shops
    attracts more than an isolated one).

Combined with trip *chaining* in DemandMixin (a car that arrives is given a
fresh trip from where it now sits), this removes the "random turns forever"
fallback that made the old simulation read as noise.
"""
from __future__ import annotations

import math

import numpy as np


def outbound_prob(hour: float) -> float:
    """P(trip is residential→commercial) as a function of local hour [0,24).

    Morning peak (~08:00) → mostly outbound; evening peak (~18:00) → mostly
    inbound; midday / night → balanced.
    """
    h = float(hour) % 24.0
    morning = math.exp(-0.5 * ((h - 8.0) / 2.5) ** 2)   # outbound surge
    evening = math.exp(-0.5 * ((h - 18.0) / 2.5) ** 2)  # inbound surge
    return float(np.clip(0.5 + 0.4 * morning - 0.4 * evening, 0.1, 0.9))


def _cluster_weights(xy: np.ndarray, radius: float) -> np.ndarray:
    """Per-node weight = number of same-type nodes within `radius` (clustering).

    Falls back to uniform weights if scipy is unavailable or input is tiny.
    """
    n = len(xy)
    if n <= 2:
        return np.ones(max(n, 0), dtype=float)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        counts = tree.query_ball_point(xy, r=radius, return_length=True)
        return np.maximum(np.asarray(counts, dtype=float), 1.0)
    except Exception:
        return np.ones(n, dtype=float)


class DemandModel:

    def __init__(
        self,
        res_nodes: list,
        res_xy: np.ndarray,
        com_nodes: list,
        com_xy: np.ndarray,
        decay_length_m: float = 900.0,
        cluster_radius_m: float = 200.0,
    ) -> None:
        self.res_nodes = list(res_nodes)
        self.com_nodes = list(com_nodes)
        self.res_xy = np.asarray(res_xy, dtype=float).reshape(-1, 2)
        self.com_xy = np.asarray(com_xy, dtype=float).reshape(-1, 2)
        self.beta = 1.0 / max(50.0, float(decay_length_m))

        self.res_w = _cluster_weights(self.res_xy, cluster_radius_m)
        self.com_w = _cluster_weights(self.com_xy, cluster_radius_m)

    @property
    def usable(self) -> bool:
        return len(self.res_nodes) > 0 and len(self.com_nodes) > 0

    # ------------------------------------------------------------------
    # Destination choice — gravity with distance decay
    # ------------------------------------------------------------------

    def _pick_weighted(self, nodes, weights, rng) -> object:
        w = np.asarray(weights, dtype=float)
        s = float(w.sum())
        if s <= 0.0:
            return nodes[int(rng.integers(0, len(nodes)))]
        idx = int(rng.choice(len(nodes), p=w / s))
        return nodes[idx]

    def _pick_destination(self, origin_xy, dest_nodes, dest_xy, dest_w, rng) -> object:
        """P(dest) ∝ attraction · exp(−β·distance_from_origin)."""
        if len(dest_nodes) == 0:
            return None
        d = np.linalg.norm(dest_xy - np.asarray(origin_xy, dtype=float), axis=1)
        w = dest_w * np.exp(-self.beta * d)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            return dest_nodes[int(rng.integers(0, len(dest_nodes)))]
        idx = int(rng.choice(len(dest_nodes), p=w / s))
        return dest_nodes[idx]

    # ------------------------------------------------------------------
    # Public sampling
    # ------------------------------------------------------------------

    def sample_trip(self, hour: float, rng) -> tuple[object, object] | None:
        """Sample a full (origin, destination) pair for the given hour."""
        if not self.usable:
            return None
        outbound = rng.random() < outbound_prob(hour)
        if outbound:
            o_nodes, o_xy, o_w = self.res_nodes, self.res_xy, self.res_w
            d_nodes, d_xy, d_w = self.com_nodes, self.com_xy, self.com_w
        else:
            o_nodes, o_xy, o_w = self.com_nodes, self.com_xy, self.com_w
            d_nodes, d_xy, d_w = self.res_nodes, self.res_xy, self.res_w

        oi = int(self._pick_weighted_index(o_w, rng))
        origin = o_nodes[oi]
        dest = self._pick_destination(o_xy[oi], d_nodes, d_xy, d_w, rng)
        if dest is None or dest == origin:
            return None
        return origin, dest

    def pick_destination_from(self, origin_xy, hour: float, rng) -> object | None:
        """Given a car's current position, choose a destination by time-of-day.

        Used for trip *chaining*: a car that just arrived gets a new onward trip
        from where it now sits, with the AM/PM direction bias applied.
        """
        if not self.usable:
            return None
        outbound = rng.random() < outbound_prob(hour)
        if outbound:
            d_nodes, d_xy, d_w = self.com_nodes, self.com_xy, self.com_w
        else:
            d_nodes, d_xy, d_w = self.res_nodes, self.res_xy, self.res_w
        return self._pick_destination(origin_xy, d_nodes, d_xy, d_w, rng)

    def _pick_weighted_index(self, weights, rng) -> int:
        w = np.asarray(weights, dtype=float)
        s = float(w.sum())
        if s <= 0.0:
            return int(rng.integers(0, len(w)))
        return int(rng.choice(len(w), p=w / s))
