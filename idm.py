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

    Complexity:  O(N)  to bucket  +  O(Σ k_e log k_e) ≤ O(N log N)  to sort.
    In practice with many short edges each bucket is tiny (1–3 cars) so the
    sort cost is effectively O(N).
    """
    ecm: dict[int, list[tuple[float, int]]] = {}
    n = int(edge_idx.shape[0])
    for i in range(n):
        e = int(edge_idx[i])
        bucket = ecm.get(e)
        if bucket is None:
            ecm[e] = [(float(dist[i]), i)]
        else:
            bucket.append((float(dist[i]), i))
    for bucket in ecm.values():
        bucket.sort()  # ascending by dist: follower → ... → leader
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

    # Build the sorted-bucket map once for the whole tick (reuse in the loop).
    ecm = build_edge_car_map(edge_idx, dist)
    # Parallel arrays per edge for O(log k) leader lookup via bisect.
    ecm_dists: dict[int, list[float]] = {}
    ecm_ids: dict[int, list[int]] = {}
    for edge, bucket in ecm.items():
        if not bucket:
            continue
        ecm_dists[edge] = [float(d) for d, _ in bucket]
        ecm_ids[edge] = [int(j) for _, j in bucket]

    n_paths = int(car_path_lengths.shape[0])
    n_next = len(car_next_edges)
    for i in range(n):
        e      = int(edge_idx[i])
        if e < 0 or e >= n_paths:
            continue
        d_i    = float(dist[i])
        v_i    = float(speed[i])
        len_i  = float(car_len[i])
        e_len  = float(car_path_lengths[e])

        # ── Heuristic Lane Changing ──────────────────────────────────────────
        if desired is not None and adj_left is not None and adj_right is not None and rng is not None:
            v0_i = float(desired[i])
            if v_i < v0_i * 0.8:
                bucket_dists = ecm_dists.get(e)
                bucket_ids = ecm_ids.get(e)
                curr_gap = _FREE_FLOW
                if bucket_dists and bucket_ids:
                    pos = bisect.bisect_right(bucket_dists, d_i + 0.01)
                    if pos < len(bucket_dists):
                        curr_gap = max(0.1, float(bucket_dists[pos]) - d_i - len_i)
                
                if curr_gap < 15.0:
                    cand_l = int(adj_left[e])
                    cand_r = int(adj_right[e])
                    cands = []
                    if cand_l >= 0: cands.append(cand_l)
                    if cand_r >= 0: cands.append(cand_r)
                    if cands:
                        rng.shuffle(cands)
                        for cand_e in cands:
                            c_dists = ecm_dists.get(cand_e)
                            c_ids = ecm_ids.get(cand_e)
                            safe = True
                            cand_gap = _FREE_FLOW
                            if c_dists and c_ids:
                                pos = bisect.bisect_right(c_dists, d_i + 0.01)
                                if pos < len(c_dists):
                                    leader_gap = max(0.1, float(c_dists[pos]) - d_i - len_i)
                                    cand_gap = leader_gap
                                    if leader_gap < 15.0: safe = False
                                if safe and pos > 0:
                                    f_d = float(c_dists[pos - 1])
                                    f_idx = c_ids[pos - 1]
                                    if d_i - f_d - float(car_len[f_idx]) < 5.0: safe = False
                            if safe and cand_gap > curr_gap:
                                # Switch lane!
                                edge_idx[i] = cand_e
                                e = cand_e
                                e_len = float(car_path_lengths[e])
                                if bucket_dists and bucket_ids:
                                    old_pos = bisect.bisect_right(bucket_dists, d_i) - 1
                                    if old_pos >= 0 and bucket_ids[old_pos] == i:
                                        bucket_dists.pop(old_pos)
                                        bucket_ids.pop(old_pos)
                                if c_dists is None:
                                    c_dists = []
                                    ecm_dists[cand_e] = c_dists
                                    c_ids = []
                                    ecm_ids[cand_e] = c_ids
                                new_pos = bisect.bisect_right(c_dists, d_i)
                                c_dists.insert(new_pos, d_i)
                                c_ids.insert(new_pos, i)
                                break

        best_gap: float = _FREE_FLOW
        best_dv:  float = 0.0

        # ── Priority 1: same-edge leader ─────────────────────────────────────
        # Jump directly to first car strictly ahead using binary search.
        bucket_dists = ecm_dists.get(e)
        bucket_ids = ecm_ids.get(e)
        if bucket_dists is not None and bucket_ids is not None:
            leader_pos = bisect.bisect_right(bucket_dists, d_i + 0.01)
            if leader_pos < len(bucket_dists):
                d_j = float(bucket_dists[leader_pos])
                j = int(bucket_ids[leader_pos])
                g = max(0.1, d_j - d_i - len_i)
                if g < best_gap:
                    best_gap = g
                    best_dv = v_i - max(0.0, float(speed[j]))

        # ── Priority 2: cross-edge leader ────────────────────────────────────
        # If no same-edge leader, look at the frontmost car on each next edge.
        # Choose the candidate that gives the smallest gap.
        if best_gap == _FREE_FLOW and e < n_next:
            dist_to_end = e_len - d_i
            for ne_raw in car_next_edges[e]:
                ne = int(ne_raw)
                next_dists = ecm_dists.get(ne)
                next_ids = ecm_ids.get(ne)
                if not next_dists or not next_ids:
                    continue
                d_j = float(next_dists[-1])
                j = int(next_ids[-1])
                g = max(0.1, dist_to_end + float(d_j) - len_i)
                if g < best_gap:
                    best_gap = g
                    best_dv  = v_i - max(0.0, float(speed[j]))

        # ── Priority 3: ghost leader at red/yellow stop line ─────────────────
        # Phantom stationary "car" placed params.s0 before the stop line so
        # the IDM brakes the car to rest exactly at the line.
        # Overrides a real leader only when the ghost is closer.
        node_id = car_paths[e]["v"]
        tl = traffic_lights.get(node_id)
        if tl is not None and e in tl.controlled_paths and not tl.can_enter(e):
            stopline_dist = e_len - d_i
            if stopline_dist <= 1e-3:
                # Avoid ghost-gap singularity when already clamped at stop line.
                at_stopline_arr[i] = True
                best_gap = _FREE_FLOW
                best_dv = 0.0
                
        # ── Priority 3b: Stop signs ──────────────────────────────────────────
        # If the intersection is controlled by a stop sign, the car must
        # halt, wait 2 seconds, and then it can proceed.
        if car_paths[e].get("control") == "stop" and stop_wait is not None:
            stopline_dist = e_len - d_i
            # If within 15 meters of the stop line, start evaluating stop logic
            if stopline_dist < 15.0:
                # Are we physically stopped at the line?
                if stopline_dist < params.s0 + 1.0 and v_i < 0.1:
                    stop_wait[i] += dt
                
                if stop_wait[i] < 2.0:
                    # Still need to wait, inject ghost leader at stopline
                    ghost_gap = max(0.1, stopline_dist - params.s0)
                    if ghost_gap < best_gap:
                        best_gap = ghost_gap
                        best_dv = v_i
                        at_stopline_arr[i] = (stopline_dist <= 1e-3)
                else:
                    # Wait time fulfilled! Allow it to proceed (unless there is cross traffic).
                    # (For a complete yield, we'd check conflicts. For now, 2s wait suffices).
                    pass
                
        # ── Priority 4: Roundabout Yield Logic ───────────────────────────────
        # If this car is about to enter a roundabout, check for cars already on the ring
        # that are approaching the same entry node.
        if best_gap == _FREE_FLOW and roundabout_yield_map is not None:
            if e in roundabout_yield_map:
                ring_edges = roundabout_yield_map[e]
                # Look at cars on the ring edges
                for re in ring_edges:
                    r_dists = ecm_dists.get(re)
                    r_ids = ecm_ids.get(re)
                    if not r_dists or not r_ids:
                        continue
                    # A car is approaching the node if it's on this ring edge.
                    # We check the closest one to the entry node (which is the leader of the ring edge).
                    # Actually, the frontmost car is r_dists[-1]
                    d_r = float(r_dists[-1])
                    j = int(r_ids[-1])
                    r_len = float(car_path_lengths[re])
                    dist_to_conflict = r_len - d_r
                    
                    # If a car is within 15 meters of the conflict point, yield to it.
                    if dist_to_conflict < 15.0:
                        # Inject a ghost leader at the entry node.
                        stopline_dist = e_len - d_i
                        g = max(0.1, stopline_dist - 0.5)
                        if g < best_gap:
                            best_gap = g
                            best_dv = v_i  # Brake to stop
                best_dv = 0.0
            else:
                ghost_gap = max(0.1, stopline_dist - params.s0)
                if ghost_gap < best_gap:
                    best_gap = ghost_gap
                    best_dv  = v_i            # v_leader = 0  ⟹  Δv = v_self

        gap_arr[i] = best_gap
        dv_arr[i]  = best_dv

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

    # ── Free-flow deceleration term (applies to every car) ───────────────────
    free_term = (speed / v0_safe) ** params.delta

    # ── Desired gap s*(v, Δv) ─────────────────────────────────────────────
    # Only numerically meaningful for cars with a leader, but safe to compute
    # for all because np.where zeroes out the interaction term below.
    s_star = params.s0 + np.maximum(
        0.0,
        speed * params.T + speed * dv / (2.0 * params.sqrt_ab),
    )

    # ── Interaction term — zero for free-flow cars ────────────────────────
    interaction_term = np.where(has_leader, (s_star / g_safe) ** 2, 0.0)

    a = params.a_max * (1.0 - free_term - interaction_term)
    return np.clip(a, -params.b * 3.0, params.a_max)


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
            speed[i]   = min(float(speed[i]), new_v0)
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

    # ── Step 3: Leader lookup + ghost-leader injection ───────────────────────
    stop_wait = np.array(car_anim["stop_wait"], dtype=float)
    
    gap, dv, at_stopline = find_leaders(
        edge_idx, dist, speed, car_len,
        car_path_lengths, car_next_edges, car_paths,
        traffic_lights, params,
        desired=desired,
        rng=rng,
        roundabout_yield_map=roundabout_yield_map,
        stop_wait=stop_wait,
        dt=dt,
    )

    # ── Step 4: Vectorised IDM accelerations — no Python loop ────────────────
    accel = idm_accelerations(speed, desired, gap, dv, params)

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
