"""safety.py — pedestrian street crossings, injury tracking, headless studies.

Three responsibilities, all VTK-free so everything runs headlessly:

1. build_crossing_paths()
     Generates synthetic street-crossing links between footway nodes that face
     each other across a drivable road.  The links plug into the existing
     ped-path random walk (ped_mixin), so pedestrians naturally cross streets.
     Each crossing is flagged is_road_crossing=True and is_crossing_end=True
     (the walker pauses 1.5–3 s before stepping onto the road — existing
     yield behaviour).

2. advance_peds_core() + check_ped_car_collisions()
     The pedestrian physics tick, extracted from PedMixin._advance_peds so
     the app and headless studies share one implementation.  Injured
     pedestrians freeze in place.  A car within HIT_RADIUS moving faster
     than HIT_MIN_SPEED injures the pedestrian (severity from impact speed).
     Pedestrians actively ON a road crossing broadcast their live position
     into ped_anim["active_crossings"], which car_mixin already consumes:
     cars within 12 m slow to walking pace.

3. run_safety_study()
     Full headless co-simulation (IDM cars + walking/crossing peds) over a
     street graph for N seconds, returning injury statistics.  Presets (e.g.
     signals on/off) let you quantify: "how many people get hurt if we
     remove this traffic light?"
"""
from __future__ import annotations

import math

import numpy as np

HIT_RADIUS      = 1.4   # m — car centre to ped centre
HIT_MIN_SPEED   = 2.0   # m/s — below this an impact is a nudge, not an injury
SEVERE_SPEED_MS = 8.3   # 30 km/h — above this an injury counts as severe


# ══════════════════════════════════════════════════════════════════════════════
# Path extraction (standalone versions for headless studies)
# ══════════════════════════════════════════════════════════════════════════════

_PED_HW = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway", "track"}


def _hw_tags(data) -> set:
    hw = data.get("highway")
    if isinstance(hw, (list, tuple, set)):
        return {str(x).strip().lower() for x in hw if str(x).strip()}
    tag = str(hw or "").strip().lower()
    return {tag} if tag else set()


def _edge_polyline(graph, u, v, data, z: float) -> np.ndarray | None:
    geom = data.get("geometry")
    if geom is not None and hasattr(geom, "coords"):
        xy = np.asarray(geom.coords, dtype=float)[:, :2]
    else:
        nu, nv = graph.nodes.get(u, {}), graph.nodes.get(v, {})
        if "x" not in nu or "x" not in nv:
            return None
        xy = np.array([[float(nu["x"]), float(nu["y"])],
                       [float(nv["x"]), float(nv["y"])]], dtype=float)
    if xy.shape[0] < 2:
        return None
    keep = np.ones(xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > 1e-6
    xy = xy[keep]
    if xy.shape[0] < 2:
        return None
    return np.column_stack((xy, np.full(xy.shape[0], z, dtype=float)))


def _paths_from_edges(graph, edge_iter, z: float, default_speed_kmh: float) -> tuple[list[dict], dict]:
    paths: list[dict] = []
    outgoing: dict = {}
    for u, v, data in edge_iter:
        pts = _edge_polyline(graph, u, v, data, z)
        if pts is None:
            continue
        seg = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)
        total = float(seg.sum())
        if total < 0.5:
            continue
        ms_raw = data.get("maxspeed", default_speed_kmh)
        try:
            if isinstance(ms_raw, (list, tuple)):
                ms_raw = ms_raw[0]
            ms_kmh = float(str(ms_raw).split()[0])
        except Exception:
            ms_kmh = default_speed_kmh
        idx = len(paths)
        paths.append({
            "u": u, "v": v, "points": pts,
            "cum_len": np.concatenate([[0.0], np.cumsum(seg)]),
            "length": total,
            "maxspeed_ms": ms_kmh / 3.6,
        })
        outgoing.setdefault(u, []).append(idx)
    return paths, outgoing


def drivable_paths_from_graph(graph, z: float = 0.5) -> tuple[list[dict], dict]:
    edges = [(u, v, d) for u, v, d in graph.edges(data=True)
             if not (_hw_tags(d) & _PED_HW)]
    return _paths_from_edges(graph, edges, z, default_speed_kmh=50.0)


def ped_paths_from_graph(graph, z: float = 0.3) -> tuple[list[dict], dict]:
    edges = [(u, v, d) for u, v, d in graph.edges(data=True)
             if _hw_tags(d) & _PED_HW]
    paths, outgoing = _paths_from_edges(graph, edges, z, default_speed_kmh=5.0)
    if len(paths) < 10:   # same fallback rule as ped_mixin
        paths, outgoing = _paths_from_edges(
            graph, list(graph.edges(data=True)), z, default_speed_kmh=5.0)
    return paths, outgoing


# ══════════════════════════════════════════════════════════════════════════════
# Street-crossing generation
# ══════════════════════════════════════════════════════════════════════════════

def road_segments_from_paths(car_paths: list[dict]) -> np.ndarray:
    """Flatten drivable polylines to an (S, 4) array [x1, y1, x2, y2]."""
    segs = []
    for p in car_paths:
        pts = np.asarray(p.get("points"))
        if pts is None or pts.shape[0] < 2:
            continue
        segs.append(np.column_stack([pts[:-1, 0], pts[:-1, 1],
                                     pts[1:, 0],  pts[1:, 1]]))
    if not segs:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(segs).astype(float)


def _segment_crosses_any(x1, y1, x2, y2, segs: np.ndarray) -> bool:
    """True if segment (x1,y1)-(x2,y2) properly intersects any road segment."""
    if segs.shape[0] == 0:
        return False
    # Vectorised orientation test (counter-clockwise sign)
    ax, ay, bx, by = x1, y1, x2, y2
    cx, cy, dx, dy = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]

    def _ccw(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = _ccw(ax, ay, bx, by, cx, cy)
    d2 = _ccw(ax, ay, bx, by, dx, dy)
    d3 = _ccw(cx, cy, dx, dy, ax, ay)
    d4 = _ccw(cx, cy, dx, dy, bx, by)
    # d3·d4 < 0: crossing endpoints strictly straddle the road line.
    # d1·d2 ≤ 0: road endpoints may TOUCH the crossing line — crossings often
    # land exactly on a road-segment endpoint (shared intersection node).
    hit = (d1 * d2 <= 0) & (d3 * d4 < 0)
    return bool(np.any(hit))


def build_crossing_paths(
    graph,
    ped_paths: list[dict],
    road_segments: np.ndarray,
    max_len: float = 30.0,
    min_len: float = 3.0,
    max_per_node: int = 2,
    z: float = 0.30,
) -> tuple[list[dict], dict]:
    """Create synthetic street-crossing links between footway nodes.

    A candidate link connects two footway nodes within [min_len, max_len]
    metres whose straight segment crosses at least one drivable road segment
    and that are not already connected by an existing pedestrian path.

    Returns (new_paths, extra_outgoing) in the exact ped_paths format; the
    caller appends new_paths and merges extra_outgoing.
    """
    # Footway node coordinates from path endpoints
    node_xy: dict = {}
    for p in ped_paths:
        pts = np.asarray(p["points"])
        node_xy.setdefault(p["u"], (float(pts[0, 0]),  float(pts[0, 1])))
        node_xy.setdefault(p["v"], (float(pts[-1, 0]), float(pts[-1, 1])))
    if len(node_xy) < 2 or road_segments.shape[0] == 0:
        return [], {}

    linked = {(p["u"], p["v"]) for p in ped_paths}
    nodes = list(node_xy.keys())
    coords = np.array([node_xy[n] for n in nodes], dtype=float)

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)

    new_paths: list[dict] = []
    extra_outgoing: dict = {}
    base_idx = len(ped_paths)
    added_per_node: dict = {}
    seen_pairs: set = set()

    pairs = tree.query_pairs(r=max_len)
    # Deterministic order: nearest crossings first
    pairs = sorted(pairs, key=lambda ij: float(
        np.hypot(*(coords[ij[0]] - coords[ij[1]]))))

    for i, j in pairs:
        na, nb = nodes[i], nodes[j]
        if (na, nb) in linked or (nb, na) in linked:
            continue
        if (na, nb) in seen_pairs:
            continue
        if added_per_node.get(na, 0) >= max_per_node:
            continue
        if added_per_node.get(nb, 0) >= max_per_node:
            continue
        (xa, ya), (xb, yb) = node_xy[na], node_xy[nb]
        d = math.hypot(xb - xa, yb - ya)
        if d < min_len:
            continue
        if not _segment_crosses_any(xa, ya, xb, yb, road_segments):
            continue

        seen_pairs.add((na, nb)); seen_pairs.add((nb, na))
        added_per_node[na] = added_per_node.get(na, 0) + 1
        added_per_node[nb] = added_per_node.get(nb, 0) + 1

        for (u, v, ux, uy, vx, vy) in ((na, nb, xa, ya, xb, yb),
                                       (nb, na, xb, yb, xa, ya)):
            pts = np.array([[ux, uy, z], [vx, vy, z]], dtype=float)
            idx = base_idx + len(new_paths)
            new_paths.append({
                "u": u, "v": v, "points": pts,
                "cum_len": np.array([0.0, d]),
                "length": d,
                "is_crossing_end": True,     # pause before stepping onto road
                "is_road_crossing": True,
            })
            extra_outgoing.setdefault(u, []).append(idx)

    return new_paths, extra_outgoing


# ══════════════════════════════════════════════════════════════════════════════
# Pedestrian tick (shared by PedMixin and headless studies)
# ══════════════════════════════════════════════════════════════════════════════

def ped_xy(ped_anim: dict, ped_paths: list[dict]) -> np.ndarray:
    """Current (N, 2) pedestrian XY positions via arc-length interpolation."""
    edge_idx = np.asarray(ped_anim["edge_idx"], dtype=np.int64)
    dist     = np.asarray(ped_anim["dist"], dtype=float)
    out = np.zeros((len(edge_idx), 2), dtype=float)
    n_paths = len(ped_paths)
    for i, (e, d) in enumerate(zip(edge_idx, dist)):
        if not (0 <= e < n_paths):
            continue
        p = ped_paths[int(e)]
        pts, cum = p["points"], p["cum_len"]
        dd = min(max(float(d), 0.0), float(p["length"]))
        out[i, 0] = np.interp(dd, cum, pts[:, 0])
        out[i, 1] = np.interp(dd, cum, pts[:, 1])
    return out


def advance_peds_core(
    ped_anim: dict,
    ped_paths: list[dict],
    ped_outgoing: dict,
    ped_reverse: dict,
    rng: np.random.Generator,
    dt: float,
    ctrl_idx: int | None = None,
) -> None:
    """One pedestrian physics step.  Mutates ped_anim in-place.

    Semantics preserved from PedMixin._advance_peds, plus:
    - injured pedestrians (ped_anim["injured"]) never move;
    - pedestrians on an is_road_crossing path register their LIVE position in
      active_crossings (cars within 12 m slow to walking pace — car_mixin).
    """
    edge_idx    = ped_anim["edge_idx"]
    dist        = ped_anim["dist"]
    speed       = ped_anim["speed"]
    yield_timer = ped_anim["yield_timer"]
    injured     = ped_anim.get("injured")
    n           = len(edge_idx)
    n_paths     = len(ped_paths)
    active_crossings: list[list[float]] = []

    for i in range(n):
        eid = int(edge_idx[i])
        if not (0 <= eid < n_paths):
            edge_idx[i] = 0; dist[i] = 0.0
            continue
        if i == ctrl_idx:
            continue          # player-controlled pedestrian
        if injured is not None and injured[i]:
            continue          # injured: frozen where they were hit

        path = ped_paths[eid]

        if yield_timer[i] > 0.0:
            yield_timer[i] = max(0.0, float(yield_timer[i]) - dt)
            pts = path["points"]
            active_crossings.append([float(pts[0, 0]), float(pts[0, 1])])
            continue

        dist[i] = float(dist[i]) + float(speed[i]) * dt

        # Mid-crossing: broadcast live position so cars yield
        if path.get("is_road_crossing", False):
            pts, cum = path["points"], path["cum_len"]
            dd = min(float(dist[i]), float(path["length"]))
            active_crossings.append([
                float(np.interp(dd, cum, pts[:, 0])),
                float(np.interp(dd, cum, pts[:, 1])),
            ])

        if dist[i] >= path["length"]:
            v_node = path["v"]
            candidates = ped_outgoing.get(v_node, [])
            if candidates:
                next_eid = int(rng.choice(candidates))
            else:
                rev = ped_reverse.get(eid)
                next_eid = rev if rev is not None else int(rng.integers(0, n_paths))
            edge_idx[i] = next_eid
            dist[i] = 0.0
            if ped_paths[next_eid].get("is_crossing_end", False):
                yield_timer[i] = float(rng.uniform(1.5, 3.0))

    ped_anim["active_crossings"] = active_crossings


def check_ped_car_collisions(
    ped_positions_xy: np.ndarray,
    injured: np.ndarray,
    car_positions_xy: np.ndarray,
    car_speeds: np.ndarray,
    hit_radius: float = HIT_RADIUS,
    min_speed: float = HIT_MIN_SPEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect new pedestrian injuries.

    Returns (newly_injured_idx, impact_speeds_ms).  Only cars moving faster
    than min_speed can injure; already-injured pedestrians are skipped.
    """
    if ped_positions_xy.shape[0] == 0 or car_positions_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)

    moving = np.asarray(car_speeds, dtype=float) > float(min_speed)
    if not np.any(moving):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)

    cars = np.asarray(car_positions_xy, dtype=float)[moving]
    spd  = np.asarray(car_speeds, dtype=float)[moving]

    # (N_ped, N_car) distance matrix — a few hundred each way, cheap
    diff = ped_positions_xy[:, None, :2] - cars[None, :, :2]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    hit_any = (d < float(hit_radius))
    cand = np.where(np.any(hit_any, axis=1) & ~np.asarray(injured, dtype=bool))[0]
    if cand.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    impact = np.array(
        [float(spd[hit_any[i]].max()) for i in cand], dtype=float)
    return cand.astype(np.int64), impact


# ══════════════════════════════════════════════════════════════════════════════
# Headless safety study
# ══════════════════════════════════════════════════════════════════════════════

def run_safety_study(
    graph,
    duration_s: float = 90.0,
    dt: float = 0.1,
    n_cars: int = 30,
    n_peds: int = 50,
    signals: bool = True,
    crossings: bool = True,
    seed: int = 42,
    traffic_speed: float = 1.0,
) -> dict:
    """Headless car+pedestrian co-simulation returning injury statistics.

    Presets are expressed through the arguments:
      signals=False    → remove all traffic lights (what if we turned them off?)
      crossings=False  → pedestrians never cross roads (baseline)
      traffic_speed    → global speed-limit multiplier (e.g. 0.6 = calmed 30-zone)

    Returns dict: injuries, severe, impact_speeds, crossings_built,
    ped_crossing_events, duration_s.
    """
    from idm import IDMParams, idm_tick
    from traffic_lights import build_traffic_lights, tick_all

    rng = np.random.default_rng(int(seed))

    car_paths, car_outgoing = drivable_paths_from_graph(graph)
    if not car_paths:
        return {"error": "no drivable edges"}
    ped_paths, ped_outgoing = ped_paths_from_graph(graph)
    if not ped_paths:
        return {"error": "no walkable edges"}

    # Street crossings
    n_crossings = 0
    if crossings:
        segs = road_segments_from_paths(car_paths)
        new_paths, extra_out = build_crossing_paths(graph, ped_paths, segs)
        ped_paths = ped_paths + new_paths
        for k, v in extra_out.items():
            ped_outgoing.setdefault(k, []).extend(v)
        n_crossings = len(new_paths) // 2

    _by_uv = {(p["u"], p["v"]): i for i, p in enumerate(ped_paths)}
    ped_reverse = {i: _by_uv.get((p["v"], p["u"])) for i, p in enumerate(ped_paths)}

    # Cars
    n_cp = len(car_paths)
    car_lengths = np.array([p["length"] for p in car_paths])
    edge_idx = rng.integers(0, n_cp, size=n_cars)
    car_anim = {
        "enabled": True,
        "edge_idx": edge_idx.astype(np.int64),
        "dist": rng.uniform(0, car_lengths[edge_idx] * 0.8),
        "speed": np.full(n_cars, 6.0),
        "desired_speed": np.array([car_paths[e]["maxspeed_ms"] for e in edge_idx]),
        "desired_speed_base": np.array([car_paths[e]["maxspeed_ms"] for e in edge_idx]),
        "accel": np.zeros(n_cars),
        "car_len": np.full(n_cars, 4.5),
        "stop_wait": np.zeros(n_cars),
        "planned_edges": [None] * n_cars,
        "planned_cursor": np.zeros(n_cars, dtype=np.int64),
    }
    next_edges = [
        np.array(car_outgoing.get(p["v"], []), dtype=np.int64) for p in car_paths
    ]

    lights = build_traffic_lights(graph, car_paths) if signals else {}

    # Pedestrians
    probs = np.array([p["length"] for p in ped_paths], dtype=float)
    probs /= probs.sum()
    p_edge = rng.choice(len(ped_paths), size=n_peds, replace=True, p=probs)
    ped_anim = {
        "enabled": True,
        "edge_idx": p_edge.astype(np.int64),
        "dist": np.array([float(rng.uniform(0, ped_paths[int(e)]["length"]))
                          for e in p_edge]),
        "speed": np.clip(1.2 + rng.standard_normal(n_peds) * 0.2, 0.7, 1.8),
        "yield_timer": np.zeros(n_peds),
        "injured": np.zeros(n_peds, dtype=bool),
        "active_crossings": [],
    }

    params = IDMParams()
    impact_speeds: list[float] = []
    crossing_events = 0
    n_steps = int(round(duration_s / dt))

    def _car_xy() -> np.ndarray:
        out = np.zeros((n_cars, 2), dtype=float)
        for i in range(n_cars):
            e = int(car_anim["edge_idx"][i])
            if not (0 <= e < n_cp):
                continue
            p = car_paths[e]
            dd = min(max(float(car_anim["dist"][i]), 0.0), float(p["length"]))
            out[i, 0] = np.interp(dd, p["cum_len"], p["points"][:, 0])
            out[i, 1] = np.interp(dd, p["cum_len"], p["points"][:, 1])
        return out

    prev_on_crossing = np.zeros(n_peds, dtype=bool)

    for _ in range(n_steps):
        if lights:
            tick_all(lights, dt, traffic_speed)

        # Cars yield near active crossings (mirror of car_mixin behaviour)
        cap = None
        cps = ped_anim.get("active_crossings") or []
        cxy = _car_xy()
        if cps:
            cp = np.asarray(cps, dtype=float)[:, :2]
            dmin = np.min(np.linalg.norm(
                cxy[:, None, :2] - cp[None, :, :], axis=2), axis=1)
            near = dmin < 12.0
            if np.any(near):
                cap = np.where(near, 1.4, np.inf)

        idm_tick(car_anim, car_paths, next_edges, lights, dt=dt,
                 params=params, rng=rng, traffic_speed=traffic_speed,
                 speed_cap=cap)

        advance_peds_core(ped_anim, ped_paths, ped_outgoing, ped_reverse, rng, dt)

        # Count crossing entries (exposure metric)
        on_crossing = np.array(
            [bool(ped_paths[int(e)].get("is_road_crossing", False))
             for e in ped_anim["edge_idx"]], dtype=bool)
        crossing_events += int(np.count_nonzero(on_crossing & ~prev_on_crossing))
        prev_on_crossing = on_crossing

        # Collisions
        pxy = ped_xy(ped_anim, ped_paths)
        hits, speeds = check_ped_car_collisions(
            pxy, ped_anim["injured"], _car_xy(), car_anim["speed"])
        if hits.shape[0]:
            ped_anim["injured"][hits] = True
            impact_speeds.extend(float(s) for s in speeds)

    impact = np.asarray(impact_speeds, dtype=float)
    return {
        "injuries": int(np.count_nonzero(ped_anim["injured"])),
        "severe": int(np.count_nonzero(impact >= SEVERE_SPEED_MS)),
        "impact_speeds_ms": [round(float(s), 2) for s in impact_speeds],
        "crossings_built": n_crossings,
        "ped_crossing_events": crossing_events,
        "duration_s": float(duration_s),
        "signals": bool(signals),
        "n_lights": len(lights),
    }
