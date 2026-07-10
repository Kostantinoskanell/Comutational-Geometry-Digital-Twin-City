"""idm.py — Intelligent Driver Model for city digital twin car simulation.

Public API
----------
IDMParams
    Dataclass holding all IDM tuning constants with urban defaults.

build_edge_car_map(edge_idx, dist) -> dict
    O(N log N): bucket cars by path, sort each bucket by position.

find_leaders(edge_idx, dist, speed, desired, car_len,
             car_path_lengths, car_next_edges, car_paths,
             traffic_lights, params) -> (gap, dv, at_stopline)
    O(N): per-car gap and approach-rate arrays.
    Ghost leaders injected here for red/yellow traffic lights.

idm_accelerations(speed, desired, gap, dv, params) -> accel
    O(N): fully vectorised NumPy IDM formula — no Python loop.

idm_tick(car_anim, car_paths, car_next_edges, traffic_lights,
         dt, params, rng, traffic_speed, roundabout_yield_map=None) -> None
    Top-level per-tick function.  Mutates car_anim in-place.
    Implements symplectic Euler integration order:
        snapshot → edge_car_map → (gap, dv, at_stopline) → accel → v_new → positions → write-back

Variable-name contract (must match car_anim keys in main.py)
-------------------------------------------------------------
car_anim["edge_idx"]       int64  (N,)  path index for each car
car_anim["dist"]           float  (N,)  metres along that path
car_anim["speed"]          float  (N,)  actual instantaneous speed (m/s)
car_anim["desired_speed"]  float  (N,)  free-flow target v₀ (m/s)
car_anim["desired_speed_base"] float (N,) per-car base target before traffic_speed
car_anim["accel"]          float  (N,)  last IDM acceleration (debug/logging)
car_anim["car_len"]        float  (N,)  bumper-to-bumper vehicle length (m)
car_anim["enabled"]        bool         whether the car layer is active

Optional per-car heterogeneity arrays (used when present):
car_anim["idm_T_arr"]      float  (N,)  per-car time headway T
car_anim["idm_a_arr"]      float  (N,)  per-car max acceleration a_max
car_anim["idm_b_arr"]      float  (N,)  per-car comfortable deceleration b

car_paths[i] keys: "u", "v", "length", "maxspeed_ms", "points", "cum_len"
car_next_edges[i]: np.ndarray of int64 — path indices that follow path i
traffic_lights: dict[node_id, TrafficLight-like]
    TrafficLight must expose: .can_enter(path_idx: int) -> bool
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Avoid circular import at runtime; used only by type checkers / IDEs.
    from traffic_lights import TrafficLight  # noqa: F401


# ── Sentinel: gap == _FREE_FLOW means no leader found (pure free-flow) ──────
_FREE_FLOW: float = 1e30

# ── Turn geometry helpers ─────────────────────────────────────────────────

def _turn_angle_deg(cur_points: np.ndarray, nxt_points: np.ndarray) -> float:
    """Angle (degrees) between the exit heading of cur_points and the entry
    heading of nxt_points.  Returns 0 if either path has fewer than 2 points."""
    if cur_points.shape[0] < 2 or nxt_points.shape[0] < 2:
        return 0.0
    v_exit  = cur_points[-1, :2] - cur_points[-2, :2]
    v_enter = nxt_points[1,  :2] - nxt_points[0,  :2]
    n1 = float(np.linalg.norm(v_exit))
    n2 = float(np.linalg.norm(v_enter))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip(np.dot(v_exit, v_enter) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _turn_speed_ms(angle_deg: float) -> float | None:
    """Maximum safe cornering speed (m/s) for a turn of the given heading
    change.  Returns None for bends < 20° (essentially straight)."""
    if angle_deg < 20.0:
        return None     # straight or slight curve — no restriction
    elif angle_deg < 45.0:
        return 8.3      # 30 km/h  — gentle turn
    elif angle_deg < 90.0:
        return 5.6      # 20 km/h  — moderate turn
    elif angle_deg < 135.0:
        return 3.9      # 14 km/h  — sharp turn
    else:
        return 2.8      # 10 km/h  — very sharp / near U-turn


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IDM PARAMETERS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class IDMParams:
    """IDM tuning constants with urban defaults.

    Parameters
    ----------
    a_max : float
        Maximum acceleration (m/s²).  Urban = 1.5 (gentle; 0→50 km/h ≈ 9 s).
    b : float
        Comfortable deceleration (m/s²).  Urban = 2.5 (firm, not harsh).
    T : float
        Desired time headway (s).  Urban = 1.5 (slightly cautious vs
        motorway 1.0–1.2 s).
    s0 : float
        Minimum jam gap (m).  Bumper clearance at standstill.
    delta : int
        Acceleration exponent.  4 = standard IDM; makes acceleration
        taper smoothly near v₀.
    """

    a_max: float = 1.5
    b:     float = 2.5
    T:     float = 1.5
    s0:    float = 2.0
    delta: int   = 4

    # Precomputed once so the tight loop never calls sqrt.
    _sqrt_ab: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_sqrt_ab", float(np.sqrt(self.a_max * self.b)))

    @property
    def sqrt_ab(self) -> float:
        """sqrt(a_max × b) — denominator in the IDM desired-gap formula."""
        return self._sqrt_ab  # type: ignore[return-value]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — BUILD EDGE → SORTED-CAR MAP                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_edge_car_map(
    edge_idx: np.ndarray,  # (N,) int64
    dist:     np.ndarray,  # (N,) float64
) -> dict[int, list[tuple[float, int]]]:
    """Return edge_car_map[path_idx] = [(dist_along_edge, car_id), ...].

    The list for each edge is sorted in ascending dist order so index 0 is the
    rearmost car and the last entry is the frontmost (leader side).

    Complexity:  O(N log N) via numpy lexsort — faster than the original
    O(N) bucket + O(Σ k_e log k_e) sort because the sort is done in C.
    """
    order       = np.lexsort((dist, edge_idx))   # sort by (edge, dist)
    s_edge      = edge_idx[order]
    s_dist      = dist[order]
    unique_e, first_i, counts = np.unique(s_edge, return_index=True, return_counts=True)
    ecm: dict[int, list[tuple[float, int]]] = {}
    for e, fi, c in zip(unique_e.tolist(), first_i.tolist(), counts.tolist()):
        ecm[e] = list(zip(s_dist[fi:fi+c].tolist(), order[fi:fi+c].tolist()))
    return ecm


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — LEADER FINDING + GHOST LEADER INJECTION                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_leaders(
    edge_idx:         np.ndarray,        # (N,) int64
    dist:             np.ndarray,        # (N,) float64
    speed:            np.ndarray,        # (N,) float64  current velocities
    car_len:          np.ndarray,        # (N,) float64  vehicle lengths
    car_path_lengths: np.ndarray,        # (P,) float64  one entry per path
    car_next_edges:   list[np.ndarray],  # car_next_edges[p] = next path indices
    car_paths:        list[dict],        # path dicts — need "v" for TL lookup
    traffic_lights:   dict,              # node_id → TrafficLight-like object
    params:           IDMParams,
    desired:          np.ndarray | None = None, # (N,) float64 desired speeds
    adj_left:         np.ndarray | None = None, # (P,) int64
    adj_right:        np.ndarray | None = None, # (P,) int64
    rng:              np.random.Generator | None = None,
    roundabout_yield_map: dict[int, np.ndarray] | None = None,
    stop_wait:        np.ndarray | None = None, # (N,) float64
    dt:               float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gap[] and dv[] for every car using a three-priority hierarchy.

    Priority (lowest gap wins):
        1. Same-edge leader   — first car strictly ahead on the same path
        2. Cross-edge leader  — frontmost car on the most likely next path
        3. Ghost leader       — phantom stationary car at the stop line when
                                the traffic light FSM reports red/yellow

    Returns
    -------
    gap : (N,) float64
        Bumper-to-bumper gap in metres.  _FREE_FLOW (1e30) = no leader found.
    dv  : (N,) float64
        Approach rate = v_self − v_leader (m/s).  0.0 when no leader.
    at_stopline : (N,) bool
        True where car is effectively at the stop line and the light is red/yellow.
        Caller should hard-clamp speed to zero for these cars.

    Complexity:  O(N log N)  dominated by build_edge_car_map sort.
    """
    n = int(edge_idx.shape[0])
    gap_arr = np.full(n, _FREE_FLOW, dtype=float)
    dv_arr  = np.zeros(n, dtype=float)
    at_stopline_arr = np.zeros(n, dtype=bool)

    n_paths = int(car_path_lengths.shape[0])
    n_next  = len(car_next_edges)

    # ── Junction conflict pre-pass (O(N_near_junction)) ──────────────────────
    # Cars from DIFFERENT paths converging on the same junction node must yield
    # to whichever car is already closest.
    _JUNC_ZONE = 20.0
    _junc_ghost: dict[int, float] = {}
    _node_front: dict = {}
    for _ji in range(n):
        _je = int(edge_idx[_ji])
        if _je < 0 or _je >= n_paths:
            continue
        _jdte = float(car_path_lengths[_je]) - float(dist[_ji])
        if _jdte >= _JUNC_ZONE:
            continue
        _jnode = car_paths[_je]["v"]
        _nf = _node_front.get(_jnode)
        if _nf is None:
            _node_front[_jnode] = {_je: (_jdte, _ji)}
        else:
            if _je not in _nf or _jdte < _nf[_je][0]:
                _nf[_je] = (_jdte, _ji)
    for _jnode, _fronts in _node_front.items():
        if len(_fronts) <= 1:
            continue
        _sorted_f = sorted(_fronts.values(), key=lambda x: x[0])
        for _jdte, _jcar in _sorted_f[1:]:
            _ghost = max(0.1, _jdte - params.s0)
            if _jcar not in _junc_ghost or _ghost < _junc_ghost[_jcar]:
                _junc_ghost[_jcar] = _ghost

    # ── Phase 0: Build sorted arrays + ECM dicts (for lane-changing) ─────────
    # ECM is built once via numpy sort — faster than the per-car Python bucket loop.
    _order0     = np.lexsort((dist, edge_idx))
    _se0        = edge_idx[_order0]
    _sd0        = dist[_order0]
    _ue0, _fi0, _ec0 = np.unique(_se0, return_index=True, return_counts=True)
    ecm_dists: dict[int, list[float]] = {}
    ecm_ids:   dict[int, list[int]]   = {}
    for _e, _fi, _c in zip(_ue0.tolist(), _fi0.tolist(), _ec0.tolist()):
        ecm_dists[_e] = list(_sd0[_fi:_fi+_c])
        ecm_ids[_e]   = list(_order0[_fi:_fi+_c].tolist())

    # ── Phase 0b: Lane changing (per-car, uses+mutates ECM) ──────────────────
    # Only runs for slow cars (v < 0.8*v0) with a tight gap ahead.
    if desired is not None and adj_left is not None and adj_right is not None and rng is not None:
        for i in range(n):
            e = int(edge_idx[i])
            if e < 0 or e >= n_paths:
                continue
            v_i  = float(speed[i])
            v0_i = float(desired[i])
            if v_i >= v0_i * 0.8:
                continue  # fast enough — skip lane-change evaluation
            d_i   = float(dist[i])
            len_i = float(car_len[i])
            bucket_dists = ecm_dists.get(e)
            bucket_ids   = ecm_ids.get(e)
            curr_gap = _FREE_FLOW
            if bucket_dists and bucket_ids:
                pos = bisect.bisect_right(bucket_dists, d_i + 0.01)
                if pos < len(bucket_dists):
                    curr_gap = max(0.1, float(bucket_dists[pos]) - d_i - len_i)
            if curr_gap >= 15.0:
                continue  # wide open — no benefit to lane-change
            cand_l = int(adj_left[e])
            cand_r = int(adj_right[e])
            cands = []
            if cand_l >= 0: cands.append(cand_l)
            if cand_r >= 0: cands.append(cand_r)
            if not cands:
                continue
            rng.shuffle(cands)
            for cand_e in cands:
                c_dists = ecm_dists.get(cand_e)
                c_ids   = ecm_ids.get(cand_e)
                safe = True
                cand_gap = _FREE_FLOW
                if c_dists and c_ids:
                    pos = bisect.bisect_right(c_dists, d_i + 0.01)
                    if pos < len(c_dists):
                        leader_gap = max(0.1, float(c_dists[pos]) - d_i - len_i)
                        cand_gap = leader_gap
                        if leader_gap < 15.0:
                            safe = False
                    if safe and pos > 0:
                        f_d   = float(c_dists[pos - 1])
                        f_idx = c_ids[pos - 1]
                        if d_i - f_d - float(car_len[f_idx]) < 5.0:
                            safe = False
                if safe and cand_gap > curr_gap:
                    edge_idx[i] = cand_e
                    if bucket_dists and bucket_ids:
                        old_p = bisect.bisect_right(bucket_dists, d_i) - 1
                        if old_p >= 0 and bucket_ids[old_p] == i:
                            bucket_dists.pop(old_p)
                            bucket_ids.pop(old_p)
                    if c_dists is None:
                        c_dists = []; ecm_dists[cand_e] = c_dists
                        c_ids = [];   ecm_ids[cand_e]   = c_ids
                    new_p = bisect.bisect_right(c_dists, d_i)
                    c_dists.insert(new_p, d_i)
                    c_ids.insert(new_p, i)
                    break

    # ── Phase 1: Vectorized same-edge leader ─────────────────────────────────
    # Resort after lane-changing (edge_idx may have changed for some cars).
    # O(N log N) numpy — replaces 600-car Python bisect loop.
    order        = np.lexsort((dist, edge_idx))
    sorted_edge  = edge_idx[order]
    sorted_dist  = dist[order]
    sorted_speed = speed[order]
    sorted_clen  = car_len[order]

    # has_se[k] = True when sorted car k has an immediate same-edge leader at k+1
    has_se = np.zeros(n, dtype=bool)
    if n > 1:
        has_se[:-1] = sorted_edge[:-1] == sorted_edge[1:]

    fol = np.where(has_se)[0]
    if fol.size > 0:
        led    = fol + 1
        se_gap = np.maximum(0.1, sorted_dist[led] - sorted_dist[fol] - sorted_clen[fol])
        se_dv  = sorted_speed[fol] - np.maximum(0.0, sorted_speed[led])
        gap_arr[order[fol]] = se_gap
        dv_arr[order[fol]]  = se_dv

    # ── Phase 1b: Per-edge frontmost car (O(N) numpy scatter) ────────────────
    # Used by cross-edge and roundabout phases — no ECM dict lookup needed.
    edge_front_dist  = np.full(n_paths, -np.inf, dtype=float)
    edge_front_speed = np.zeros(n_paths, dtype=float)
    # Last car in sorted order for each edge = frontmost car on that edge
    last_sorted = np.where(~has_se)[0]
    _valid      = (sorted_edge[last_sorted] >= 0) & (sorted_edge[last_sorted] < n_paths)
    _lo         = last_sorted[_valid]
    if _lo.size > 0:
        edge_front_dist[sorted_edge[_lo]]  = sorted_dist[_lo]
        edge_front_speed[sorted_edge[_lo]] = sorted_speed[_lo]

    # ── Phase 2: Cross-edge leader (only for cars still at FREE_FLOW) ─────────
    no_se = np.where(gap_arr >= _FREE_FLOW)[0]
    for i in no_se:
        e = int(edge_idx[i])
        if e < 0 or e >= n_paths or e >= n_next:
            continue
        dist_to_end = float(car_path_lengths[e]) - float(dist[i])
        v_i   = float(speed[i])
        len_i = float(car_len[i])
        for ne_raw in car_next_edges[e]:
            ne = int(ne_raw)
            if ne < 0 or ne >= n_paths or edge_front_dist[ne] <= -np.inf:
                continue
            g = max(0.1, dist_to_end + float(edge_front_dist[ne]) - len_i)
            if g < gap_arr[i]:
                gap_arr[i] = g
                dv_arr[i]  = v_i - max(0.0, float(edge_front_speed[ne]))

    # ── Phase 3: Vectorized TL ghost leader ───────────────────────────────────
    # O(n_paths) Python loop + O(N) numpy — replaces 600-car per-car TL lookup.
    can_enter_path = np.ones(n_paths, dtype=bool)
    for pidx in range(n_paths):
        _node_id = car_paths[pidx]["v"]
        _tl = traffic_lights.get(_node_id)
        if _tl is not None and pidx in _tl.controlled_paths and not _tl.can_enter(pidx):
            can_enter_path[pidx] = False

    e_arr       = np.clip(edge_idx, 0, n_paths - 1)
    car_blocked = ~can_enter_path[e_arr]
    stopline_d  = car_path_lengths[e_arr] - dist
    ghost_gap_v = np.maximum(0.1, stopline_d - params.s0)

    at_stop = car_blocked & (stopline_d <= 1e-3)
    at_stopline_arr |= at_stop
    # Hold cars already at the stop line
    gap_arr[at_stop] = _FREE_FLOW
    dv_arr[at_stop]  = 0.0
    # Inject ghost for cars approaching red
    ghost_better = car_blocked & ~at_stop & (ghost_gap_v < gap_arr)
    gap_arr = np.where(ghost_better, ghost_gap_v, gap_arr)
    dv_arr  = np.where(ghost_better, speed, dv_arr)

    # ── Phase 3b: Stop signs (per-car, only cars near stop-sign edges) ────────
    if stop_wait is not None:
        for i in range(n):
            e = int(edge_idx[i])
            if e < 0 or e >= n_paths or car_paths[e].get("control") != "stop":
                continue
            sl_dist = float(car_path_lengths[e]) - float(dist[i])
            if sl_dist >= 15.0:
                continue
            v_i = float(speed[i])
            if sl_dist < params.s0 + 1.0 and v_i < 0.1:
                stop_wait[i] += dt
            if stop_wait[i] < 2.0:
                g = max(0.1, sl_dist - params.s0)
                if g < gap_arr[i]:
                    gap_arr[i] = g
                    dv_arr[i]  = v_i
                    at_stopline_arr[i] = (sl_dist <= 1e-3)

    # ── Phase 4: Roundabout yield (per-car, only yield-map edges) ────────────
    if roundabout_yield_map is not None:
        for i in range(n):
            e = int(edge_idx[i])
            if e not in roundabout_yield_map or gap_arr[i] < _FREE_FLOW:
                continue
            ring_edges = roundabout_yield_map[e]
            e_len = float(car_path_lengths[e])
            d_i   = float(dist[i])
            v_i   = float(speed[i])
            for re in ring_edges:
                re = int(re)
                if re < 0 or re >= n_paths or edge_front_dist[re] <= -np.inf:
                    continue
                dist_to_conflict = float(car_path_lengths[re]) - float(edge_front_dist[re])
                if dist_to_conflict < 15.0:
                    g = max(0.1, e_len - d_i - 0.5)
                    if g < gap_arr[i]:
                        gap_arr[i] = g
                        dv_arr[i]  = v_i

    # ── Phase 5: Junction conflict ghost ─────────────────────────────────────
    for _jcar, _jg in _junc_ghost.items():
        _jg_f = float(_jg)
        if _jg_f < gap_arr[_jcar]:
            gap_arr[_jcar] = _jg_f
            dv_arr[_jcar]  = float(speed[_jcar])

    return gap_arr, dv_arr, at_stopline_arr


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — VECTORISED IDM ACCELERATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def idm_accelerations(
    speed:   np.ndarray,   # (N,) current velocity
    desired: np.ndarray,   # (N,) free-flow target v₀
    gap:     np.ndarray,   # (N,) bumper-to-bumper gap (_FREE_FLOW = none)
    dv:      np.ndarray,   # (N,) approach rate (0 for free-flow cars)
    params:  IDMParams,
    a_arr:   np.ndarray | None = None,   # (N,) per-car max accel (overrides params.a_max)
    b_arr:   np.ndarray | None = None,   # (N,) per-car decel (overrides params.b)
    T_arr:   np.ndarray | None = None,   # (N,) per-car time headway (overrides params.T)
) -> np.ndarray:
    """Fully vectorised IDM acceleration for all N cars.

    IDM formula
    -----------
        a = a_max * [ 1  −  (v/v₀)^δ  −  (s*(v,Δv) / gap)² ]

        s*(v,Δv)  =  s₀  +  max(0,  v·T  +  v·Δv / (2·√(a·b)) )

    Free-flow branch (gap == _FREE_FLOW):
        interaction term = 0  ⟹  a = a_max * [1 − (v/v₀)^δ]

    Output is clipped to [−3b, a_max] (braking no harder than 3× comfortable).

    Complexity:  O(N)  — pure NumPy broadcasting, no Python loop.
    """
    # Guard against division by zero.
    v0_safe = np.maximum(desired, 1e-3)
    g_safe  = np.maximum(gap, 0.1)

    has_leader = gap < _FREE_FLOW     # bool (N,)

    # Per-car parameter arrays (fall back to scalar params when not provided)
    _a_max = a_arr if a_arr is not None else params.a_max
    _b     = b_arr if b_arr is not None else params.b
    _T     = T_arr if T_arr is not None else params.T
    _sqrt_ab = np.sqrt(_a_max * _b) if a_arr is not None or b_arr is not None else params.sqrt_ab

    # ── Free-flow deceleration term (applies to every car) ───────────────────
    free_term = (speed / v0_safe) ** params.delta

    # ── Desired gap s*(v, Δv) ─────────────────────────────────────────────
    # Only numerically meaningful for cars with a leader, but safe to compute
    # for all because np.where zeroes out the interaction term below.
    s_star = params.s0 + np.maximum(
        0.0,
        speed * _T + speed * dv / (2.0 * _sqrt_ab),
    )

    # ── Interaction term — zero for free-flow cars ────────────────────────
    interaction_term = np.where(has_leader, (s_star / g_safe) ** 2, 0.0)

    a = _a_max * (1.0 - free_term - interaction_term)
    return np.clip(a, -_b * 3.0, _a_max)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — POSITION ADVANCE + EDGE CROSSINGS                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _advance_positions(
    edge_idx:         np.ndarray,        # (N,) int64  — mutated in-place
    dist:             np.ndarray,        # (N,) float  — mutated in-place
    speed:            np.ndarray,        # (N,) float  — mutated in-place
    desired:          np.ndarray,        # (N,) float  — mutated in-place
    desired_base:     np.ndarray,        # (N,) float  — mutated in-place
    car_path_lengths: np.ndarray,        # (P,) float
    car_next_edges:   list[np.ndarray],
    car_paths:        list[dict],
    traffic_lights:   dict,
    params:           IDMParams,
    rng:              np.random.Generator,
    dt:               float,
    traffic_speed:    float,
    planned_edges:    object = None,
    planned_cursor:   np.ndarray | None = None,
) -> None:
    """Advance every car's position by v·dt, handling edge crossings.

        When a car crosses an edge boundary:
            - A successor edge is chosen at random from car_next_edges[current].
                If the current edge has no legal successor, the car stops at the
                end instead of inventing a local continuation.
            - desired_speed_base[i] is resampled from new edge maxspeed_ms × U(0.7, 1.0).
            - desired_speed[i] = desired_speed_base[i] × traffic_speed.
      - speed[i] is clamped to the new desired_speed so cars never exceed the
        posted limit of the road they just entered.

    Topology prevents vectorisation here; the inner while-loop limit (8 hops)
    stops degenerate infinite loops on very short edges.
    """
    n = int(edge_idx.shape[0])
    n_paths = int(car_path_lengths.shape[0])
    n_next = len(car_next_edges)

    for i in range(n):
        remaining = float(speed[i]) * float(dt)
        if remaining <= 1e-9:
            continue

        hops = 0
        while remaining > 1e-9 and hops < 8:
            cidx = int(edge_idx[i])
            if cidx < 0 or cidx >= n_paths:
                break
            c_len  = float(car_path_lengths[cidx])
            to_end = max(0.0, c_len - float(dist[i]))
            node_id = car_paths[cidx].get("v")
            tl = traffic_lights.get(node_id)
            if tl is not None and cidx in tl.controlled_paths and not tl.can_enter(cidx):
                stop_dist = max(0.0, c_len - float(params.s0))
                current_dist = float(dist[i])
                if current_dist >= stop_dist or remaining >= max(0.0, stop_dist - current_dist):
                    dist[i] = current_dist if current_dist >= stop_dist else stop_dist
                    speed[i] = 0.0
                    remaining = 0.0
                    break

            if remaining < to_end:
                dist[i] += remaining
                remaining = 0.0
                break

            # ── Cross the edge boundary ───────────────────────────────────
            remaining -= to_end

            nexts = car_next_edges[cidx] if cidx < n_next else np.empty(0, dtype=np.int64)
            forced_next: int | None = None
            if planned_edges is not None and planned_cursor is not None and i < planned_cursor.shape[0]:
                try:
                    plan = planned_edges[i]
                    if plan is not None:
                        plan_arr = np.asarray(plan, dtype=np.int64)
                        if plan_arr.size > 0:
                            cur = int(np.clip(int(planned_cursor[i]), 0, plan_arr.size - 1))
                            if int(plan_arr[cur]) != cidx:
                                matches = np.flatnonzero(plan_arr == cidx)
                                if matches.size > 0:
                                    cur = int(matches[0])
                            if cur + 1 < plan_arr.size:
                                candidate = int(plan_arr[cur + 1])
                                if nexts.size == 0 or bool(np.any(nexts == candidate)):
                                    forced_next = candidate
                                    planned_cursor[i] = cur + 1
                            else:
                                planned_edges[i] = None
                except Exception:
                    forced_next = None

            if forced_next is not None:
                edge_idx[i] = int(forced_next)
            elif nexts.size > 0:
                edge_idx[i] = int(rng.choice(nexts))
            else:
                # No legal OSM successor from this node: stop at the end instead
                # of inventing a local graph continuation.
                dist[i] = max(0.0, c_len - 1e-6)
                speed[i] = 0.0
                remaining = 0.0
                break

            dist[i] = 0.0

            # Resample free-flow target for new edge; keep momentum continuous.
            new_base_v0 = (
                float(car_paths[int(edge_idx[i])]["maxspeed_ms"])
                * float(rng.uniform(0.7, 1.0))
            )
            new_v0 = new_base_v0 * float(traffic_speed)
            desired_base[i] = new_base_v0
            desired[i] = new_v0

            # Hard-clamp entry speed to turn cornering limit.
            # This covers cars whose plan wasn't known at the pre-braking pass and
            # ensures speed never exceeds safe cornering speed at the junction.
            if car_paths[cidx].get("junction") != "roundabout":
                _v_turn = _turn_speed_ms(
                    _turn_angle_deg(car_paths[cidx]["points"],
                                    car_paths[int(edge_idx[i])]["points"])
                )
                if _v_turn is not None:
                    speed[i] = min(float(speed[i]), _v_turn)

            speed[i] = min(float(speed[i]), new_v0)
            hops += 1

        # If we hit the hop cap, carry leftover distance onto current edge (clamped).
        if remaining > 1e-9:
            cidx = int(edge_idx[i])
            c_len = float(car_path_lengths[cidx])
            dist[i] = min(c_len, float(dist[i]) + remaining)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TOP-LEVEL TICK                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def idm_tick(
    car_anim:       dict,
    car_paths:      list[dict],
    car_next_edges: list[np.ndarray],
    traffic_lights: dict,
    dt:             float,
    params:         IDMParams,
    rng:            np.random.Generator,
    traffic_speed: float = 1.0,
    roundabout_yield_map: dict[int, np.ndarray] | None = None,
    adj_left:  np.ndarray | None = None,   # (P,) int64, adjacent lane left
    adj_right: np.ndarray | None = None,   # (P,) int64, adjacent lane right
) -> None:
    """Full IDM simulation tick. Mutates car_anim in-place.

    Parameters
    ----------
    car_anim       : shared animation state dict
    car_paths      : list of path dicts
    car_next_edges : car_next_edges[p] = ndarray of successor path indices.
    traffic_lights : node_id → TrafficLight FSM
    dt             : simulation timestep in seconds.
    params         : IDMParams instance.
    rng            : numpy Generator used for stochastic edge-hop choices.
    traffic_speed  : global multiplier applied to road speed limits.
    """
    if not bool(car_anim.get("enabled", False)):
        return

    # ── Step 1: Snapshot (read phase — all subsequent reads use these arrays) ─
    edge_idx = np.array(car_anim["edge_idx"],      dtype=np.int64)
    dist     = np.array(car_anim["dist"],          dtype=float)
    speed    = np.array(car_anim["speed"],         dtype=float)
    desired_base_raw = car_anim.get("desired_speed_base")
    if desired_base_raw is None:
        # Backward-compatible fallback for older saved state.
        ts = max(1e-9, float(traffic_speed))
        desired_base = np.array(car_anim["desired_speed"], dtype=float) / ts
    else:
        desired_base = np.array(desired_base_raw, dtype=float)
    desired  = desired_base * float(traffic_speed)
    car_len  = np.asarray(car_anim["car_len"],     dtype=float)   # read-only

    car_path_lengths = np.asarray(
        [float(p["length"]) for p in car_paths], dtype=float
    )
    n_paths = int(car_path_lengths.shape[0])
    if n_paths > 0:
        edge_idx = np.clip(edge_idx, 0, n_paths - 1)
        dist = np.clip(dist, 0.0, np.maximum(car_path_lengths[edge_idx] - 1e-6, 0.0))

    # ── Step 2: Build edge → sorted-car map ──────────────────────────────────
    # (Implicitly done inside find_leaders; exposed separately for reuse.)

    # ── Step 2b: Turn approach — lower desired speed before a sharp corner ───
    # When a car is within TURN_LOOKAHEAD metres of its edge end, look ahead to
    # the next edge, compute the heading change, and reduce desired[i] to the
    # cornering speed limit.  IDM then decelerates naturally before the junction;
    # the hard clamp at crossing time (in _advance_positions) acts as a safety net.
    # desired_base is NOT touched so the car re-accelerates normally on the new edge.
    _TURN_LOOKAHEAD = 18.0  # metres
    _planned_edges_snap  = car_anim.get("planned_edges")
    _planned_cursor_snap = car_anim.get("planned_cursor")
    _n_planned = len(_planned_edges_snap) if _planned_edges_snap is not None else 0
    n_cars = int(edge_idx.shape[0])
    n_next = len(car_next_edges)

    # Vectorized filter — only iterate cars within TURN_LOOKAHEAD of edge end.
    # Replaces O(N) loop where most cars skip via `_dist_to_end > _TURN_LOOKAHEAD`.
    # Build roundabout flag per PATH (O(n_paths) Python) then index per CAR (O(N) numpy).
    _is_ra_path = (np.array([p.get("junction") == "roundabout" for p in car_paths], dtype=bool)
                   if n_paths > 0 else np.zeros(0, dtype=bool))
    _e_clip  = np.clip(edge_idx, 0, max(n_paths - 1, 0))
    _dte_all = car_path_lengths[_e_clip] - dist    # (N,) dist to edge end
    _near    = (edge_idx >= 0) & (edge_idx < n_paths) & ~_is_ra_path[_e_clip] & (_dte_all <= _TURN_LOOKAHEAD)
    for _i in np.where(_near)[0]:
        _cidx = int(edge_idx[_i])
        _next_e: int | None = None
        if _planned_edges_snap is not None and _planned_cursor_snap is not None and _i < _n_planned:
            try:
                _plan = _planned_edges_snap[_i]
                if _plan is not None:
                    _pa  = np.asarray(_plan, dtype=np.int64)
                    _cur = int(np.clip(int(_planned_cursor_snap[_i]), 0, _pa.size - 1))
                    if _pa.size > 0 and int(_pa[_cur]) != _cidx:
                        _m = np.flatnonzero(_pa == _cidx)
                        if _m.size > 0:
                            _cur = int(_m[0])
                    if _cur + 1 < _pa.size:
                        _next_e = int(_pa[_cur + 1])
            except Exception:
                pass
        if _next_e is None and _cidx < n_next and car_next_edges[_cidx].size > 0:
            _next_e = int(car_next_edges[_cidx][0])
        if _next_e is None or _next_e < 0 or _next_e >= n_paths:
            continue
        _angle = _turn_angle_deg(car_paths[_cidx]["points"], car_paths[_next_e]["points"])
        _v_lim = _turn_speed_ms(_angle)
        if _v_lim is not None and _v_lim < desired[_i]:
            desired[_i] = _v_lim  # IDM sees reduced v₀ → brakes smoothly

    # ── Step 3: Leader lookup + ghost-leader injection ───────────────────────
    stop_wait = np.array(car_anim["stop_wait"], dtype=float)

    gap, dv, at_stopline = find_leaders(
        edge_idx, dist, speed, car_len,
        car_path_lengths, car_next_edges, car_paths,
        traffic_lights, params,
        desired=desired,
        adj_left=adj_left,
        adj_right=adj_right,
        rng=rng,
        roundabout_yield_map=roundabout_yield_map,
        stop_wait=stop_wait,
        dt=dt,
    )

    # ── Step 4: Vectorised IDM accelerations — no Python loop ────────────────
    _a_arr = np.asarray(car_anim["idm_a_arr"], dtype=float) if "idm_a_arr" in car_anim else None
    _b_arr = np.asarray(car_anim["idm_b_arr"], dtype=float) if "idm_b_arr" in car_anim else None
    _T_arr = np.asarray(car_anim["idm_T_arr"], dtype=float) if "idm_T_arr" in car_anim else None
    accel = idm_accelerations(speed, desired, gap, dv, params, a_arr=_a_arr, b_arr=_b_arr, T_arr=_T_arr)

    # ── Step 5: Velocity update (symplectic Euler: all v before any x) ───────
    #   v_new = clip( v + a·dt,  lower=0,  upper=v₀ )
    #   • lower=0  → no reversal
    #   • upper=v₀ → no overspeed past desired free-flow target
    speed = np.clip(speed + accel * float(dt), 0.0, desired)
    speed[at_stopline] = 0.0

    # ── Step 6: Position advance + edge crossings ─────────────────────────────
    _advance_positions(
        edge_idx, dist, speed, desired, desired_base,
        car_path_lengths, car_next_edges, car_paths,
        traffic_lights, params,
        rng, dt, float(traffic_speed),
        car_anim.get("planned_edges"),
        car_anim.get("planned_cursor"),
    )

    # ── Step 7: Write back all mutated state ──────────────────────────────────
    car_anim["edge_idx"]      = edge_idx
    car_anim["dist"]          = dist
    car_anim["speed"]         = speed
    car_anim["desired_speed"] = desired
    car_anim["desired_speed_base"] = desired_base
    car_anim["accel"]         = accel
    car_anim["stop_wait"]     = stop_wait

    # ── Step 8: Deadlock recovery — teleport cars stuck > 8 s ────────────────
    # Cars that have been at near-zero speed while near a leader (junction ghost
    # or traffic light) for more than 8 seconds are teleported to a random edge
    # start.  This breaks circular deadlocks that the junction-yield pre-pass
    # cannot resolve when two cars are exactly equidistant from the conflict node.
    _STUCK_THRESH_S  = 8.0    # seconds before teleport
    _STUCK_SPEED_MS  = 0.12   # m/s: below this counts as "stopped"
    _STUCK_GAP_M     = 25.0   # only count stuck if within this gap of a leader

    n_cars = int(edge_idx.shape[0])
    stuck_time = np.asarray(
        car_anim.setdefault("stuck_time", np.zeros(n_cars, dtype=float)),
        dtype=float,
    )
    if stuck_time.shape[0] != n_cars:
        stuck_time = np.zeros(n_cars, dtype=float)

    # Exempt cars legitimately waiting at a red light — they are stopped
    # intentionally and should never be treated as deadlocked.
    _can_enter = np.ones(n_paths, dtype=bool)
    for _pidx in range(n_paths):
        _nid = car_paths[_pidx]["v"]
        _tl = traffic_lights.get(_nid)
        if (_tl is not None
                and _pidx in _tl.controlled_paths
                and not _tl.can_enter(_pidx)):
            _can_enter[_pidx] = False
    _tl_blocked = ~_can_enter[np.clip(edge_idx, 0, n_paths - 1)]

    stopped_mask = (speed < _STUCK_SPEED_MS) & (gap < _STUCK_GAP_M) & ~_tl_blocked
    stuck_time[stopped_mask]  += float(dt)
    stuck_time[~stopped_mask]  = 0.0

    deadlocked = np.where(stuck_time > _STUCK_THRESH_S)[0]
    if deadlocked.size > 0 and n_paths > 0:
        new_edges = rng.integers(0, n_paths, size=deadlocked.size)
        edge_idx[deadlocked] = new_edges
        dist[deadlocked]     = 0.0
        # Resample speed from new edge limit
        for _di, _ne in zip(deadlocked, new_edges):
            new_v0 = (float(car_paths[int(_ne)]["maxspeed_ms"])
                      * float(rng.uniform(0.7, 1.0)))
            desired_base[_di] = new_v0
            desired[_di]      = new_v0 * float(traffic_speed)
            speed[_di]        = new_v0 * 0.3
        stuck_time[deadlocked] = 0.0
        # Write back position arrays immediately after teleport
        car_anim["edge_idx"]           = edge_idx
        car_anim["dist"]               = dist
        car_anim["speed"]              = speed
        car_anim["desired_speed"]      = desired
        car_anim["desired_speed_base"] = desired_base

    car_anim["stuck_time"] = stuck_time
