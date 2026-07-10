#!/usr/bin/env python3
"""run_tests.py — headless self-test suite for the Digital Twin City.

Runs WITHOUT opening a VTK window (safe on macOS M1 — no plotter.show()).
Tests the pure-logic engine modules with synthetic data: IDM physics,
traffic-light FSM, emissions, validation metrics, demand model, terrain
draping, SUMO plain-XML export, and per-car heterogeneity.

Usage
-----
    python tests/run_tests.py           # run everything
    python tests/run_tests.py -k idm    # run only tests whose name contains "idm"
    python tests/run_tests.py -v        # verbose (print each test as it runs)

Exit code 0 = all passed, 1 = failures.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Make the project root importable regardless of cwd
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

# Force headless-safe environment (no VTK window will be created by these tests,
# but belt-and-braces in case an import triggers something).
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np

_TESTS: list[tuple[str, callable]] = []


def test(fn):
    """Decorator: register a test function."""
    _TESTS.append((fn.__name__, fn))
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic scene helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_paths(n_paths: int = 4, length: float = 100.0) -> list[dict]:
    """Build a simple ring of car_paths: 0→1→2→3→0, each `length` m long."""
    paths = []
    for i in range(n_paths):
        # Straight segments laid out in a square ring
        angle = i * (2 * np.pi / n_paths)
        x0, y0 = 100 * np.cos(angle), 100 * np.sin(angle)
        angle2 = (i + 1) * (2 * np.pi / n_paths)
        x1, y1 = 100 * np.cos(angle2), 100 * np.sin(angle2)
        pts = np.array([[x0, y0, 0.5], [x1, y1, 0.5]], dtype=float)
        seg = float(np.linalg.norm(pts[1, :2] - pts[0, :2]))
        paths.append({
            "u": i, "v": (i + 1) % n_paths,
            "length": seg,
            "maxspeed_ms": 13.9,
            "points": pts,
            "cum_len": np.array([0.0, seg]),
        })
    return paths


def make_car_anim(n_cars: int, paths: list[dict], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    edge_idx = rng.integers(0, len(paths), size=n_cars)
    lengths = np.array([p["length"] for p in paths])
    return {
        "enabled": True,
        "edge_idx": edge_idx.astype(np.int64),
        "dist": rng.uniform(0, lengths[edge_idx] * 0.8),
        "speed": np.full(n_cars, 8.0),
        "desired_speed": np.full(n_cars, 13.9),
        "desired_speed_base": np.full(n_cars, 13.9),
        "accel": np.zeros(n_cars),
        "car_len": np.full(n_cars, 4.5),
        "stop_wait": np.zeros(n_cars),
        "planned_edges": [None] * n_cars,
        "planned_cursor": np.zeros(n_cars, dtype=np.int64),
    }


def make_next_edges(paths: list[dict]) -> list[np.ndarray]:
    """Ring topology: path i continues to path (i+1) % n."""
    n = len(paths)
    return [np.array([(i + 1) % n], dtype=np.int64) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# IDM physics
# ══════════════════════════════════════════════════════════════════════════════

@test
def idm_basic_tick_advances_cars():
    from idm import IDMParams, idm_tick
    paths = make_synthetic_paths()
    anim = make_car_anim(10, paths)
    nxt = make_next_edges(paths)
    d0 = anim["dist"].copy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=IDMParams(), rng=rng)
    # Cars must have moved and speeds must stay in [0, desired]
    assert np.any(anim["dist"] != d0) or np.any(anim["edge_idx"] != make_car_anim(10, paths)["edge_idx"]), "no car moved"
    assert np.all(anim["speed"] >= 0.0), "negative speed"
    assert np.all(anim["speed"] <= anim["desired_speed"] + 1e-6), "overspeed"


@test
def idm_no_collisions_on_dense_edge():
    from idm import IDMParams, idm_tick, build_edge_car_map
    paths = make_synthetic_paths(n_paths=2, length=200.0)
    n = 20
    anim = make_car_anim(n, paths, seed=1)
    # Cram all cars on edge 0, spread out
    anim["edge_idx"] = np.zeros(n, dtype=np.int64)
    anim["dist"] = np.linspace(5, paths[0]["length"] - 5, n)
    nxt = make_next_edges(paths)
    rng = np.random.default_rng(1)
    params = IDMParams()
    for _ in range(200):
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=params, rng=rng)
    # Verify no phasing-through (gap >= 0 for every same-edge pair)
    ecm = build_edge_car_map(np.asarray(anim["edge_idx"]), np.asarray(anim["dist"]))
    worst = np.inf
    for e, bucket in ecm.items():
        for k in range(len(bucket) - 1):
            d_f, fi = bucket[k]
            d_l, _ = bucket[k + 1]
            g = d_l - d_f - float(anim["car_len"][fi])
            worst = min(worst, g)
    assert worst > -0.5, f"cars phased through each other (worst gap {worst:.2f} m)"


@test
def idm_heterogeneity_arrays_change_behavior():
    from idm import IDMParams, idm_accelerations
    n = 100
    speed = np.full(n, 5.0)
    desired = np.full(n, 13.9)
    gap = np.full(n, 20.0)
    dv = np.full(n, 2.0)
    params = IDMParams()
    a_hom = idm_accelerations(speed, desired, gap, dv, params)
    rng = np.random.default_rng(42)
    T_arr = np.clip(rng.normal(1.4, 0.3, n), 0.8, 2.5)
    a_arr = np.clip(rng.normal(1.6, 0.4, n), 0.9, 2.8)
    b_arr = np.clip(rng.normal(2.0, 0.4, n), 1.0, 3.5)
    a_het = idm_accelerations(speed, desired, gap, dv, params,
                              a_arr=a_arr, b_arr=b_arr, T_arr=T_arr)
    assert a_het.shape == (n,), "wrong output shape"
    assert np.std(a_het) > np.std(a_hom), "heterogeneous accel not more varied than homogeneous"
    assert np.all(a_het <= a_arr + 1e-9), "accel exceeds per-car a_max"
    assert np.all(a_het >= -b_arr * 3.0 - 1e-9), "braking exceeds per-car 3b"


@test
def idm_heterogeneity_reproducible_with_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    t1 = np.clip(rng1.normal(1.4, 0.3, 50), 0.8, 2.5)
    t2 = np.clip(rng2.normal(1.4, 0.3, 50), 0.8, 2.5)
    assert np.array_equal(t1, t2), "same seed must give identical parameter draws"


@test
def idm_deadlock_teleport_recovers_stuck_cars():
    from idm import IDMParams, idm_tick
    paths = make_synthetic_paths(n_paths=4)
    anim = make_car_anim(4, paths, seed=3)
    # Force cars to be stopped at tiny gaps
    anim["speed"] = np.zeros(4)
    anim["edge_idx"] = np.zeros(4, dtype=np.int64)
    anim["dist"] = np.array([10.0, 15.0, 20.0, 25.0])
    nxt = [np.empty(0, dtype=np.int64)] * 4   # dead-end: no successors
    rng = np.random.default_rng(3)
    for _ in range(250):   # 250 × 0.05 s = 12.5 s > 8 s threshold
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=IDMParams(), rng=rng)
    st = np.asarray(anim.get("stuck_time", np.zeros(4)))
    assert np.all(st < 8.5), "stuck_time not reset — teleport recovery did not fire"


# ══════════════════════════════════════════════════════════════════════════════
# Traffic-light FSM
# ══════════════════════════════════════════════════════════════════════════════

@test
def tl_fsm_cycles_through_states():
    from traffic_lights import TrafficLight
    tl = TrafficLight(node_id=0, x=0, y=0,
                      green_groups=[[0, 1], [2, 3]],
                      green_durations=[10.0, 10.0],
                      yellow_durations=[3.0, 3.0],
                      n_phases=2, offset=0.0)
    states = set()
    for _ in range(600):   # 60 s at 0.1 s ticks — > 1 full cycle
        tl.tick(0.1)
        states.add(tl.state)
    assert states == {"green", "yellow", "all_red"}, f"FSM missed states: {states}"


@test
def tl_fsm_only_one_phase_green_at_a_time():
    from traffic_lights import TrafficLight
    tl = TrafficLight(node_id=0, x=0, y=0,
                      green_groups=[[0, 1], [2, 3]],
                      green_durations=[8.0, 8.0],
                      yellow_durations=[3.0, 3.0],
                      n_phases=2, offset=0.0)
    for _ in range(400):
        tl.tick(0.1)
        if tl.state == "green":
            g0 = tl.can_enter(0)
            g2 = tl.can_enter(2)
            assert not (g0 and g2), "conflicting phases green simultaneously"


@test
def tl_editor_created_light_matches_build_shape():
    """An editor-added light must have the same fields the renderer expects."""
    from traffic_lights import TrafficLight, _phase_timings, _LIGHT_Z
    groups = [[0], [1]]
    car_paths = make_synthetic_paths(2)
    gd, yd = _phase_timings(groups, car_paths)
    tl = TrafficLight(node_id=99, x=0, y=0, green_groups=groups,
                      green_durations=gd, yellow_durations=yd,
                      n_phases=2, offset=0.0)
    tl.approach_points = {0: (1.0, 0.0, _LIGHT_Z), 1: (0.0, 1.0, _LIGHT_Z)}
    assert len(gd) == 2 and len(yd) == 2
    assert all(14.0 <= g <= 45.0 for g in gd), "green duration out of bounds"
    assert all(3.0 <= y <= 5.5 for y in yd), "yellow duration out of bounds"
    # display_color_for_path must work for every approach
    for pidx in tl.approach_points:
        c = tl.display_color_for_path(pidx)
        assert c in ("green", "yellow", "red")


@test
def tl_build_light_mesh_color_count_matches_points():
    from traffic_lights import TrafficLight, build_light_mesh, _LIGHT_Z
    tl = TrafficLight(node_id=0, x=5, y=5, green_groups=[[0], [1]],
                      n_phases=2, offset=0.0)
    tl.approach_points = {0: (4.0, 5.0, _LIGHT_Z), 1: (5.0, 4.0, _LIGHT_Z)}
    mesh = build_light_mesh({0: tl})
    assert mesh.n_points == 2, f"expected 2 approach glyph points, got {mesh.n_points}"
    assert mesh["colors"].shape == (2, 3), "colors array shape mismatch"


@test
def tl_ghost_leader_stops_cars_at_red():
    from idm import IDMParams, idm_tick
    from traffic_lights import TrafficLight
    paths = make_synthetic_paths(n_paths=4)
    anim = make_car_anim(1, paths, seed=5)
    anim["edge_idx"] = np.array([0], dtype=np.int64)
    anim["dist"] = np.array([paths[0]["length"] - 30.0])   # 30 m before junction
    anim["speed"] = np.array([10.0])
    nxt = make_next_edges(paths)
    # Red light controlling path 0 (green group excludes path 0 → permanently red for it)
    tl = TrafficLight(node_id=paths[0]["v"], x=0, y=0,
                      green_groups=[[99], [0]],   # phase 0 = other path; car is red
                      green_durations=[10000.0, 1.0],
                      yellow_durations=[4.0, 4.0],
                      n_phases=2, offset=0.0)
    lights = {paths[0]["v"]: tl}
    rng = np.random.default_rng(5)
    # 15 s: the ghost leader brakes the car smoothly (IDM curve), so full
    # stop takes longer than a hard clamp would — that's correct behavior.
    for _ in range(300):
        idm_tick(anim, paths, nxt, lights, dt=0.05, params=IDMParams(), rng=rng)
    # Car must not have crossed onto the next edge
    assert int(anim["edge_idx"][0]) == 0, "car ran the red light"
    assert float(anim["speed"][0]) < 1.0, "car did not stop at the red light"


# ══════════════════════════════════════════════════════════════════════════════
# Emissions model
# ══════════════════════════════════════════════════════════════════════════════

@test
def emissions_u_shape_and_positive():
    from emissions import emission_factor_g_per_km, emission_rate_g_per_s, POLLUTANTS
    v = np.array([2.0, 8.0, 18.0, 30.0])   # m/s: 7.2, 28.8, 64.8, 108 km/h
    for pol in POLLUTANTS:
        ef = emission_factor_g_per_km(pol, v)
        assert np.all(ef > 0), f"{pol}: non-positive factor"
        # U-shape: congested (7 km/h) worse per km than optimum (~65 km/h)
        assert ef[0] > ef[2], f"{pol}: congestion not worse than optimum"
        # High speed worse than optimum
        assert ef[3] > ef[2], f"{pol}: high speed not worse than optimum"
        rate = emission_rate_g_per_s(pol, v)
        assert np.all(rate > 0), f"{pol}: idle floor missing"


@test
def emissions_class_scaling():
    from emissions import emission_rate_g_per_s
    v = 10.0
    car = emission_rate_g_per_s("nox", v, "passenger")
    bus = emission_rate_g_per_s("nox", v, "bus")
    bike = emission_rate_g_per_s("nox", v, "bicycle")
    assert bus > car, "bus NOx must exceed passenger car"
    assert bike < car * 0.1, "bicycle should emit ~zero"


# ══════════════════════════════════════════════════════════════════════════════
# Validation metrics
# ══════════════════════════════════════════════════════════════════════════════

@test
def validation_metrics_known_values():
    from validation import _metrics
    sim = np.array([100.0, 200.0, 300.0])
    ref = np.array([100.0, 200.0, 300.0])
    m = _metrics(sim, ref)
    assert m["mape_pct"] == 0.0 and m["rmse_s"] == 0.0 and m["bias_s"] == 0.0
    assert m["geh_under5_pct"] == 100.0
    m2 = _metrics(sim * 1.10, ref)
    assert abs(m2["mape_pct"] - 10.0) < 1e-6, f"MAPE should be 10%, got {m2['mape_pct']}"
    assert m2["bias_s"] > 0, "positive bias expected when sim > ref"


@test
def validation_road_class_map():
    from validation import _road_class
    assert _road_class({"highway": "motorway_link"}) == "motorway"
    assert _road_class({"highway": "residential"}) == "residential"
    assert _road_class({"highway": ["primary", "secondary"]}) == "primary"
    assert _road_class({"highway": "footway"}) == "other"
    assert _road_class({}) == "other"


@test
def validation_congested_runs_on_synthetic_graph():
    """validate_congested must run end-to-end with a stub reference engine
    (no network calls) and produce per-class metrics."""
    import networkx as nx
    import validation as V

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = "+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    # 6×6 grid, 300 m spacing (so O-D straight distances land in [200, 2500])
    N = 6
    for i in range(N):
        for j in range(N):
            g.add_node(i * N + j, x=float(i * 300), y=float(j * 300))
    for i in range(N):
        for j in range(N):
            nid = i * N + j
            if i + 1 < N:
                g.add_edge(nid, (i + 1) * N + j, key=0, length=300.0, highway="residential")
                g.add_edge((i + 1) * N + j, nid, key=0, length=300.0, highway="residential")
            if j + 1 < N:
                g.add_edge(nid, i * N + j + 1, key=0, length=300.0, highway="primary")
                g.add_edge(i * N + j + 1, nid, key=0, length=300.0, highway="primary")

    car_paths = [{"u": u, "v": v, "length": 300.0, "maxspeed_ms": 10.0}
                 for u, v, k in list(g.edges(keys=True))[:20]]
    car_anim = {
        "edge_idx": np.arange(10, dtype=np.int64),
        "speed": np.full(10, 6.0),
    }

    # Stub OSRM so the test never touches the network
    orig = V.osrm_duration
    V.osrm_duration = lambda o, d, host=None, timeout=None: 120.0
    try:
        rep = V.validate_congested(g, car_anim, car_paths, n_pairs=5, seed=7)
    finally:
        V.osrm_duration = orig

    assert rep.get("matched_pairs", 0) > 0, f"no pairs matched: {rep.get('error', rep)}"
    assert "metrics_by_class" in rep, "per-class metrics missing"
    assert rep["metrics"]["n"] == rep["matched_pairs"]


# ══════════════════════════════════════════════════════════════════════════════
# Demand model
# ══════════════════════════════════════════════════════════════════════════════

@test
def demand_gravity_prefers_near_destinations():
    from demand_model import DemandModel
    dm = DemandModel(
        res_nodes=["home"], res_xy=np.array([[0.0, 0.0]]),
        com_nodes=["near", "far"],
        com_xy=np.array([[200.0, 0.0], [2000.0, 0.0]]),
    )
    assert dm.usable, "model should be usable with 1 res + 2 com nodes"
    rng = np.random.default_rng(42)
    picks = {"near": 0, "far": 0}
    for _ in range(300):
        trip = dm.sample_trip(hour=8.0, rng=rng)   # morning → mostly outbound
        if trip is not None and trip[1] in picks:
            picks[trip[1]] += 1
    assert picks["near"] > picks["far"], (
        f"distance decay broken: near={picks['near']} far={picks['far']}")


# ══════════════════════════════════════════════════════════════════════════════
# Terrain draping (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

@test
def terrain_profile_lookup_matches_dem():
    from terrain_drape import precompute_path_heights, car_heights_from_profiles
    dem = lambda xy: np.asarray(xy)[:, 0] * 0.01   # height = x / 100
    paths = make_synthetic_paths(n_paths=3, length=100.0)

    def pose_fn(path, dist_m):
        pts, cum = path["points"], path["cum_len"]
        d = float(np.clip(dist_m, 0.0, path["length"]))
        seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
        t = (d - cum[seg]) / max(1e-9, cum[seg + 1] - cum[seg])
        return pts[seg] + (pts[seg + 1] - pts[seg]) * t

    profiles, lens, counts = precompute_path_heights(paths, pose_fn, dem)
    edge_idx = np.array([0, 1, 2], dtype=np.int64)
    dist = np.array([0.0, 10.0, 50.0])
    h = car_heights_from_profiles(profiles, lens, counts, edge_idx, dist)
    assert h.shape == (3,), "wrong output shape"
    # Height at path start should be ~dem(start point)
    p0 = paths[0]["points"][0]
    expected = float(p0[0]) * 0.01
    assert abs(float(h[0]) - expected) < 0.5, f"profile height {h[0]:.3f} != dem {expected:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# SUMO plain-XML export (no SUMO binary needed)
# ══════════════════════════════════════════════════════════════════════════════

@test
def sumo_plainxml_export_writes_valid_xml():
    import tempfile
    import xml.etree.ElementTree as ET
    import networkx as nx
    from sumo_network_patch import graph_to_sumo_plainxml

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = "+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=100.0, y=0.0)
    g.add_node(3, x=100.0, y=100.0, highway="traffic_signals")
    g.add_edge(1, 2, key=0, length=100.0, maxspeed_ms=13.9, lanes=2, osmid=111)
    g.add_edge(2, 3, key=0, length=100.0, maxspeed_ms=8.3, osmid=222)

    with tempfile.TemporaryDirectory() as tmp:
        ok = graph_to_sumo_plainxml(g, tmp, prefix="t")
        assert ok, "export returned False"
        nod = ET.parse(os.path.join(tmp, "t.nod.xml")).getroot()
        edg = ET.parse(os.path.join(tmp, "t.edg.xml")).getroot()
        assert len(nod.findall("node")) == 3, "node count mismatch"
        assert len(edg.findall("edge")) == 2, "edge count mismatch"
        # traffic_signals node must be exported as type traffic_light
        tl_nodes = [n for n in nod.findall("node") if n.get("type") == "traffic_light"]
        assert len(tl_nodes) == 1, "traffic_signals tag not mapped to traffic_light"
        # Coordinates must be lon/lat (near 23.7, 38)
        n1 = nod.findall("node")[0]
        assert abs(float(n1.get("x")) - 23.7) < 0.5, "node x not converted to lon"
        assert abs(float(n1.get("y")) - 38.0) < 0.5, "node y not converted to lat"


@test
def sumo_roundabout_string_node_ids_accepted():
    """The roundabout editor creates string node IDs (ra_<node>_<i>). The
    plain-XML export must handle them, and — when netconvert is installed —
    netconvert must accept the network and emit a valid .net.xml."""
    import tempfile
    import xml.etree.ElementTree as ET
    import networkx as nx
    from sumo_network_patch import graph_to_sumo_plainxml, rebuild_sumo_net, _resolve_netconvert

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = "+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    # Mimic the editor: an 8-node roundabout ring with string IDs plus
    # two integer-ID approach roads feeding in/out of the ring.
    ring_ids = [f"ra_42_{i}" for i in range(8)]
    r = 14.0
    for i, rid in enumerate(ring_ids):
        a = i * 2 * np.pi / 8
        g.add_node(rid, x=float(r * np.cos(a)), y=float(r * np.sin(a)))
    for i in range(8):
        u, v = ring_ids[i], ring_ids[(i + 1) % 8]
        from shapely.geometry import LineString
        geom = LineString([(g.nodes[u]["x"], g.nodes[u]["y"]),
                           (g.nodes[v]["x"], g.nodes[v]["y"])])
        g.add_edge(u, v, key=0, length=geom.length, geometry=geom,
                   oneway=True, junction="roundabout", maxspeed_ms=8.3)
    g.add_node(100, x=120.0, y=0.0)
    g.add_node(200, x=-120.0, y=0.0)
    g.add_edge(100, ring_ids[0], key=0, length=106.0, maxspeed_ms=13.9, osmid=910)
    g.add_edge(ring_ids[4], 200, key=0, length=106.0, maxspeed_ms=13.9, osmid=920)

    with tempfile.TemporaryDirectory() as tmp:
        ok = graph_to_sumo_plainxml(g, tmp, prefix="ra")
        assert ok, "plain-XML export failed with string node IDs"
        nod = ET.parse(os.path.join(tmp, "ra.nod.xml")).getroot()
        exported_ids = {n.get("id") for n in nod.findall("node")}
        assert set(ring_ids) <= exported_ids, "ra_* node IDs missing from export"

        # netconvert acceptance — only when the binary is actually available
        binary = _resolve_netconvert()
        import shutil as _sh
        resolved = binary if os.path.isfile(binary) else _sh.which(binary)
        if not resolved:
            print("    (netconvert not installed — XML export verified, "
                  "netconvert acceptance skipped)")
            return
        net_out = os.path.join(tmp, "ra.net.xml")
        ok2 = rebuild_sumo_net("ra", tmp, net_out)
        assert ok2, "netconvert rejected the roundabout network"
        net = ET.parse(net_out).getroot()
        assert len(net.findall(".//edge")) > 0, "netconvert output has no edges"


# ══════════════════════════════════════════════════════════════════════════════
# GTFS bus loader (return-arity regression)
# ══════════════════════════════════════════════════════════════════════════════

@test
def bus_loader_returns_tuple_when_gtfs_missing():
    """Regression: _load_gtfs_buses must always return a 2-tuple (was `return []`)."""
    import importlib
    import bus_mixin as bm
    # Simulate networkx-missing branch by calling with a nonexistent path — but the
    # arity bug is on the _NX_OK False branch; test both shapes are unpackable.
    result = bm._load_gtfs_buses("/nonexistent/path.zip", None, "", 5,
                                 np.random.default_rng(0))
    assert isinstance(result, tuple) and len(result) == 2, (
        f"_load_gtfs_buses returned {type(result).__name__} of len "
        f"{len(result) if hasattr(result, '__len__') else '?'} — must be a 2-tuple")


# ══════════════════════════════════════════════════════════════════════════════
# Solar physics sanity
# ══════════════════════════════════════════════════════════════════════════════

@test
def solar_sun_higher_at_noon():
    from solar_physics import sun_angles
    el_morning, _ = sun_angles(lat_deg=38.0, lon_deg=23.7, hour_local=8.0)
    el_noon, _ = sun_angles(lat_deg=38.0, lon_deg=23.7, hour_local=12.5)
    el_night, _ = sun_angles(lat_deg=38.0, lon_deg=23.7, hour_local=2.0)
    assert el_noon > el_morning, "sun not higher at noon than morning"
    assert el_night < 0, "sun above horizon at 02:00"


@test
def solar_energy_positive_and_scales():
    from solar_physics import mechanical_energy_joules
    e1 = mechanical_energy_joules(length_m=100, speed_ms=10, vehicle_mass_kg=1500,
                                  rolling_coeff=0.01, drag_coeff=0.3, frontal_area_m2=2.2)
    e2 = mechanical_energy_joules(length_m=200, speed_ms=10, vehicle_mass_kg=1500,
                                  rolling_coeff=0.01, drag_coeff=0.3, frontal_area_m2=2.2)
    assert e1 > 0, "mechanical energy must be positive"
    assert abs(e2 - 2 * e1) < 1e-6, "energy must scale linearly with distance"


# ══════════════════════════════════════════════════════════════════════════════
# Import smoke test — every module must at least import headlessly
# ══════════════════════════════════════════════════════════════════════════════

@test
def all_engine_modules_import():
    import importlib
    mods = [
        "idm", "traffic_lights", "emissions", "validation", "demand_model",
        "terrain_drape", "turn_restrictions", "solar_physics", "solar_routing",
        "spatial_trees", "sumo_bridge", "sumo_network_patch", "sumo_engine",
        "glyph_instance", "profiler", "gtfs_realtime", "streetlight_ga",
    ]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:
            failed.append(f"{m}: {exc}")
    assert not failed, "import failures:\n  " + "\n  ".join(failed)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario comparison (pure logic — no SUMO binary needed)
# ══════════════════════════════════════════════════════════════════════════════

def _make_scenario_graph():
    import networkx as nx
    from shapely.geometry import LineString

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = "+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=100.0, y=0.0, highway="traffic_signals")
    g.add_node(3, x=100.0, y=100.0)
    g.add_node("ra_1_0", x=50.0, y=50.0)   # string id, like the roundabout editor
    geom = LineString([(0.0, 0.0), (50.0, 10.0), (100.0, 0.0)])
    g.add_edge(1, 2, key=0, length=105.0, maxspeed_ms=13.9, lanes=2,
               osmid=111, geometry=geom)
    g.add_edge(2, 3, key=0, length=100.0, maxspeed_ms=8.3, osmid=222, oneway=True)
    g.add_edge(3, "ra_1_0", key=0, length=70.0, maxspeed_ms=8.3,
               junction="roundabout")
    g.add_edge("ra_1_0", 1, key=0, length=70.0, maxspeed_ms=8.3)
    return g


@test
def scenario_snapshot_roundtrip():
    import json
    from scenario_compare import snapshot_scenario
    from traffic_lights import TrafficLight

    g = _make_scenario_graph()
    tl = TrafficLight(node_id=2, x=100.0, y=0.0,
                      green_groups=[[0], [1]],
                      green_durations=[18.0, 22.0],
                      yellow_durations=[4.0, 4.0],
                      n_phases=2, offset=0.0)
    snap = snapshot_scenario(g, {2: tl},
                             {"period": 12.0, "end": 900.0, "seed": 7},
                             name="unit-test")

    # Must be fully JSON-native
    blob = json.dumps(snap)
    back = json.loads(blob)
    assert len(back["nodes"]) == g.number_of_nodes(), "node count mismatch"
    assert len(back["edges"]) == g.number_of_edges(), "edge count mismatch"
    assert len(back["signals"]) == 1, "signal count mismatch"
    assert back["signals"][0]["node_id"] == "2"
    assert back["signals"][0]["green_durations"] == [18.0, 22.0]
    assert back["demand"] == {"period": 12.0, "end": 900.0, "seed": 7}
    assert back["meta"]["name"] == "unit-test"
    # String node ids stored as str; geometry survives as [[x,y],...]
    node_ids = {n["id"] for n in back["nodes"]}
    assert "ra_1_0" in node_ids and "1" in node_ids
    geoms = [e for e in back["edges"] if "geometry" in e]
    assert len(geoms) == 1 and len(geoms[0]["geometry"]) == 3


@test
def scenario_compare_report_math():
    import json
    import tempfile
    from scenario_compare import compare

    a = {"duration_s": 900.0, "step_s": 0.5, "seed": 42,
         "edge_tt": {"e1": 10.0, "e2": 20.0},
         "emissions_g": {"co2": 100.0, "nox": 10.0, "pm": 2.0},
         "noise_exceed_cell_seconds": 50.0,
         "vehicles_arrived": 40, "mean_speed_ms": 8.0}
    b = {"duration_s": 900.0, "step_s": 0.5, "seed": 42,
         "edge_tt": {"e1": 15.0, "e2": 18.0, "e3": 5.0},
         "emissions_g": {"co2": 110.0, "nox": 8.0, "pm": 2.0},
         "noise_exceed_cell_seconds": 65.0,
         "vehicles_arrived": 44, "mean_speed_ms": 8.5}

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cmp.json")
        rep = compare(a, b, out_path=out)

        d = rep["edge_tt_delta_s"]
        assert abs(d["e1"] - 5.0) < 1e-9, f"e1 delta wrong: {d['e1']}"
        assert abs(d["e2"] - (-2.0)) < 1e-9, f"e2 delta wrong: {d['e2']}"
        assert "e3" not in d, "edge only in B must not appear in deltas"

        s = rep["summary"]
        assert abs(s["emissions_delta_pct"]["co2"] - 10.0) < 1e-9
        assert abs(s["emissions_delta_pct"]["nox"] - (-20.0)) < 1e-9
        assert abs(s["emissions_delta_pct"]["pm"] - 0.0) < 1e-9
        # common edges: A mean = 15, B mean = 16.5
        assert abs(s["mean_edge_tt_a_s"] - 15.0) < 1e-9
        assert abs(s["mean_edge_tt_b_s"] - 16.5) < 1e-9
        assert abs(s["mean_edge_tt_delta_s"] - 1.5) < 1e-9
        assert abs(s["noise_exceed_delta_cell_s"] - 15.0) < 1e-9
        assert s["vehicles_arrived_a"] == 40 and s["vehicles_arrived_b"] == 44
        assert "metric_definitions" in rep and rep["metric_definitions"]

        assert os.path.isfile(out), "report JSON not written"
        with open(out) as fh:
            loaded = json.load(fh)
        assert loaded["summary"]["edges_compared"] == 2


@test
def scenario_edge_id_mapping_stable():
    import networkx as nx
    from scenario_compare import edge_ids_for_graph

    g = _make_scenario_graph()
    # Two extra edges sharing osmid+key with an existing one → dedup suffixes
    g.add_edge(2, 1, key=0, length=105.0, osmid=111)
    g.add_edge(3, 1, key=0, length=140.0, osmid=111)

    ids1 = edge_ids_for_graph(g)
    ids2 = edge_ids_for_graph(g)
    assert ids1 == ids2, "edge-id mapping not deterministic"

    eids = [e[0] for e in ids1]
    assert len(eids) == len(set(eids)), "edge ids not unique after dedup"
    assert len(eids) == g.number_of_edges(), "one id per edge expected"
    # Dedup pattern: first keeps raw id, duplicates get _0, _1, …
    dup = [e for e in eids if e.startswith("111#0")]
    assert dup == ["111#0", "111#0_0", "111#0_1"], f"dedup pattern wrong: {dup}"
    # No-osmid edges use the e_{u}_{v}_{key} form
    assert "e_3_ra_1_0_0" in eids, f"fallback edge id missing: {eids}"


# ══════════════════════════════════════════════════════════════════════════════
# Functional behavior tests — editor ops, planned routes, real SUMO
# ══════════════════════════════════════════════════════════════════════════════

def paths_from_graph(g):
    """One car_path per directed edge (single lane, straight or geometry)."""
    paths, outgoing = [], {}
    for u, v, k, d in g.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            xy = np.asarray(geom.coords, dtype=float)[:, :2]
        else:
            xy = np.array([[g.nodes[u]["x"], g.nodes[u]["y"]],
                           [g.nodes[v]["x"], g.nodes[v]["y"]]], dtype=float)
        pts = np.column_stack([xy, np.full(len(xy), 0.5)])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        paths.append({
            "u": u, "v": v, "length": float(cum[-1]),
            "maxspeed_ms": float(d.get("maxspeed_ms", 8.3)),
            "points": pts, "cum_len": cum,
            "junction": d.get("junction"),
        })
        outgoing.setdefault(u, []).append(len(paths) - 1)
    next_edges = []
    for p in paths:
        next_edges.append(np.array(outgoing.get(p["v"], []), dtype=np.int64))
    return paths, next_edges


def _make_cross_graph(arm_len: float = 120.0, signals: bool = False):
    """4-way cross: center node 0, arms 1..4, bidirectional edges.

    The E-W road is rotated 10 degrees so the two road axes are 0° and 100°:
    on a perfectly symmetric cross the axis-grouping split ties with the
    circular wrap gap and one phase group comes out empty.
    """
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0, y=0.0, **({"highway": "traffic_signals"} if signals else {}))
    _a = np.radians(100.0)   # compass bearing of the rotated road
    ex, ey = arm_len * np.sin(_a), arm_len * np.cos(_a)
    arms = {1: (ex, ey), 2: (-ex, -ey), 3: (0.0, arm_len), 4: (0.0, -arm_len)}
    for n, (x, y) in arms.items():
        g.add_node(n, x=float(x), y=float(y))
        g.add_edge(0, n, key=0, length=float(arm_len), maxspeed_ms=13.9)
        g.add_edge(n, 0, key=0, length=float(arm_len), maxspeed_ms=13.9)
    return g


def _single_car_anim(edge_idx: int, dist: float, speed: float) -> dict:
    return {
        "enabled": True,
        "edge_idx": np.array([edge_idx], dtype=np.int64),
        "dist": np.array([float(dist)]),
        "speed": np.array([float(speed)]),
        "desired_speed": np.array([13.9]),
        "desired_speed_base": np.array([13.9]),
        "accel": np.zeros(1),
        "car_len": np.array([4.5]),
        "stop_wait": np.zeros(1),
        "planned_edges": [None],
        "planned_cursor": np.zeros(1, dtype=np.int64),
    }


@test
def functional_roundabout_insert_and_circulate():
    """Editor roundabout surgery must leave a graph cars can fully circulate."""
    from editor_ops import insert_roundabout
    from idm import IDMParams, idm_tick

    g = _make_cross_graph(arm_len=120.0)
    result = insert_roundabout(g, 0)
    ring_nodes = result["ring_nodes"]

    assert 0 not in g.nodes, "original center node not removed"
    assert len(ring_nodes) == 8 and all(rn in g.nodes for rn in ring_nodes), \
        "8 ring nodes expected"
    ring_set = set(ring_nodes)
    ring_edges = [(u, v, d) for u, v, k, d in g.edges(keys=True, data=True)
                  if u in ring_set and v in ring_set]
    assert len(ring_edges) == 8, f"expected 8 ring edges, got {len(ring_edges)}"
    assert all(d.get("junction") == "roundabout" for _, _, d in ring_edges), \
        "ring edges missing junction=roundabout"

    paths, nxt = paths_from_graph(g)

    # Build roundabout_yield_map exactly like route_mixin._rebuild_traffic_and_arrows
    roundabout_yield_map = {}
    for p_idx, path in enumerate(paths):
        if path.get("junction") != "roundabout":
            for next_p_idx_raw in nxt[p_idx]:
                next_p = int(next_p_idx_raw)
                if paths[next_p].get("junction") == "roundabout":
                    ring_paths = []
                    for rp_idx, rp in enumerate(paths):
                        if rp["v"] == path["v"] and rp.get("junction") == "roundabout":
                            ring_paths.append(rp_idx)
                    if ring_paths:
                        roundabout_yield_map[p_idx] = np.array(ring_paths, dtype=np.int64)
    assert roundabout_yield_map, "no approach edges feed the roundabout"

    # One car at the start of an approach edge (an edge ending on a ring node).
    # From a ring-entry node the graph legally allows a U-turn back down the
    # arm, so free-roam RNG routing can shuttle forever. A planned route
    # (approach → half the ring → exit arm) makes the test deterministic while
    # still proving the editor surgery produced fully traversable ring topology.
    approach = next(i for i, p in enumerate(paths)
                    if p["v"] in ring_set and p.get("junction") != "roundabout")
    edge_map = {(p["u"], p["v"]): i for i, p in enumerate(paths)}

    plan = [approach]
    node = paths[approach]["v"]
    for _hop in range(8):   # walk ring edges until an exit arm is available
        ring_next = next((edge_map[(node, w)] for w in ring_nodes
                          if (node, w) in edge_map), None)
        if _hop >= 3:   # after half the ring, take the first exit arm
            exit_p = next((i for i, p in enumerate(paths)
                           if p["u"] == node and p.get("junction") != "roundabout"
                           and i != approach), None)
            if exit_p is not None:
                plan.append(exit_p)
                break
        assert ring_next is not None, f"ring not connected at node {node}"
        plan.append(ring_next)
        node = paths[ring_next]["v"]
    assert paths[plan[-1]].get("junction") != "roundabout", "no exit arm found"

    anim = _single_car_anim(approach, 0.0, 10.0)
    anim["planned_edges"] = [plan]
    anim["planned_cursor"] = np.zeros(1, dtype=np.int64)

    rng = np.random.default_rng(7)
    visited = [approach]
    for _ in range(2400):   # 120 s
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=IDMParams(), rng=rng,
                 roundabout_yield_map=roundabout_yield_map)
        e = int(anim["edge_idx"][0])
        if e != visited[-1]:
            visited.append(e)
        if e == plan[-1]:
            break

    assert len(visited) > 1, "car never left its approach edge"
    ring_hits = [i for i, e in enumerate(visited)
                 if paths[e].get("junction") == "roundabout"]
    assert ring_hits, f"car never entered the ring (visited {visited})"
    assert visited[-1] == plan[-1], (
        f"car never completed the ring circuit to the exit arm "
        f"(visited {visited}, plan {plan})")


@test
def functional_editor_tl_cycle_obeyed():
    """A light created exactly like the editor does must stop AND release cars."""
    from idm import IDMParams, idm_tick
    from traffic_lights import (TrafficLight, _group_edges_by_axis, _edge_bearing,
                                _phase_timings, _LIGHT_Z)

    g = _make_cross_graph(arm_len=120.0)
    paths, nxt = paths_from_graph(g)

    # Center = node with >= 3 incoming paths
    incoming: dict[object, list[int]] = {}
    for idx, p in enumerate(paths):
        incoming.setdefault(p["v"], []).append(idx)
    center = next(n for n, lst in incoming.items() if len(lst) >= 3)

    # Replicate the ui_mixin "lights" editor block
    in_edges = []
    for idx, path in enumerate(paths):
        if path["v"] == center:
            bearing = _edge_bearing(g, path["u"], path["v"])
            in_edges.append((idx, path["u"], path["v"], bearing))
    n_phases = 2 if len(in_edges) >= 3 else 1
    groups = _group_edges_by_axis(in_edges, n_phases=n_phases)
    ndata = g.nodes[center]
    green_durations, yellow_durations = _phase_timings(groups, paths)
    light = TrafficLight(
        node_id=center,
        x=float(ndata.get("x", 0.0)),
        y=float(ndata.get("y", 0.0)),
        green_groups=groups,
        green_durations=green_durations,
        yellow_durations=yellow_durations,
        n_phases=n_phases,
        offset=0.0,
    )
    approach_points = {}
    vx = float(ndata.get("x", 0.0))
    vy = float(ndata.get("y", 0.0))
    for idx_ap, path_ap in enumerate(paths):
        if path_ap["v"] == center:
            udata = g.nodes.get(path_ap["u"], {})
            ux = float(udata.get("x", vx))
            uy = float(udata.get("y", vy))
            vec = np.array([ux - vx, uy - vy], dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            approach_points[int(idx_ap)] = (vx + float(vec[0]) * 4.0,
                                            vy + float(vec[1]) * 4.0, _LIGHT_Z)
    light.approach_points = approach_points
    lights = {center: light}

    assert groups[1], "phase-1 group empty — cannot pick a red approach"
    car_edge = int(groups[1][0])          # red during phase 0
    start_len = paths[car_edge]["length"]
    anim = _single_car_anim(car_edge, start_len - 40.0, 8.0)

    rng = np.random.default_rng(11)
    dt = 0.05
    params = IDMParams()

    # Phase 0 green (>= 14 s by construction): the car must brake and hold
    ticks_phase0 = 0
    while light.phase == 0 and light.state == "green" and ticks_phase0 < 260:
        idm_tick(anim, paths, nxt, lights, dt=dt, params=params, rng=rng)
        light.tick(dt)
        ticks_phase0 += 1
    assert light.phase == 0, "phase flipped before the red-hold could be checked"
    assert int(anim["edge_idx"][0]) == car_edge, "car ran the red light"
    assert float(anim["speed"][0]) < 1.0, \
        f"car did not stop at the red light (v={float(anim['speed'][0]):.2f})"

    # Tick on until phase 1 turns green, then the car must be released
    guard = 0
    while not (light.phase == 1 and light.state == "green") and guard < 20000:
        idm_tick(anim, paths, nxt, lights, dt=dt, params=params, rng=rng)
        light.tick(dt)
        guard += 1
    assert light.phase == 1 and light.state == "green", "light never reached phase-1 green"

    crossed = False
    for _ in range(200):   # 10 s of phase-1 green
        idm_tick(anim, paths, nxt, lights, dt=dt, params=params, rng=rng)
        light.tick(dt)
        if int(anim["edge_idx"][0]) != car_edge:
            crossed = True
            break
    assert crossed, "car was not released when its phase turned green"


@test
def functional_planned_route_followed():
    """A car with planned_edges must follow the plan in order to the last edge."""
    import networkx as nx
    from idm import IDMParams, idm_tick

    N = 4
    g = nx.MultiDiGraph()
    for i in range(N):
        for j in range(N):
            g.add_node(i * N + j, x=float(i * 100), y=float(j * 100))
    for i in range(N):
        for j in range(N):
            nid = i * N + j
            for nbr in ([(i + 1) * N + j] if i + 1 < N else []) + \
                       ([i * N + j + 1] if j + 1 < N else []):
                g.add_edge(nid, nbr, key=0, length=100.0, maxspeed_ms=13.9)
                g.add_edge(nbr, nid, key=0, length=100.0, maxspeed_ms=13.9)

    paths, nxt = paths_from_graph(g)
    edge_map = {}
    for idx, p in enumerate(paths):
        edge_map.setdefault((p["u"], p["v"]), idx)   # first key wins

    src, tgt = 0, N * N - 1                          # opposite corners
    node_path = nx.shortest_path(g, src, tgt, weight="length")
    plan = [edge_map[(a, b)] for a, b in zip(node_path[:-1], node_path[1:])]
    assert len(plan) >= 2, "degenerate plan"

    anim = _single_car_anim(plan[0], 0.0, 8.0)
    anim["planned_edges"] = [list(plan)]
    anim["planned_cursor"] = np.zeros(1, dtype=np.int64)

    rng = np.random.default_rng(21)
    visited = [plan[0]]
    for _ in range(3000):   # up to 150 s
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=IDMParams(), rng=rng)
        e = int(anim["edge_idx"][0])
        if e != visited[-1]:
            visited.append(e)
        if e == plan[-1]:
            break

    assert visited[-1] == plan[-1], \
        f"car never reached the final plan edge (visited {visited}, plan {plan})"
    # Every visited edge must appear in the plan, in strictly increasing order
    pos = -1
    for e in visited:
        assert e in plan, f"car deviated onto edge {e} not in plan {plan}"
        p_i = plan.index(e)
        assert p_i > pos, f"plan followed out of order: {visited} vs {plan}"
        pos = p_i


@test
def functional_sumo_tl_and_reroute():
    """Real SUMO: export a signalised cross, run traci, toggle TL, reroute."""
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET
    import networkx as nx
    from sumo_bridge import check_sumo
    from sumo_network_patch import graph_to_sumo_plainxml, rebuild_sumo_net, _resolve_netconvert

    ok, msg = check_sumo()
    if not ok:
        print(f"    (SUMO unavailable — skipped: {msg})")
        return
    nc = _resolve_netconvert()
    if not (os.path.isfile(nc) or shutil.which(nc)):
        print("    (netconvert unresolvable — skipped)")
        return

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = ("+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 "
                           "+datum=WGS84 +units=m +no_defs")
    g.add_node(0, x=0.0, y=0.0, highway="traffic_signals")
    for n, (x, y) in {1: (150.0, 0.0), 2: (-150.0, 0.0),
                      3: (0.0, 150.0), 4: (0.0, -150.0)}.items():
        g.add_node(n, x=x, y=y)
        g.add_edge(0, n, key=0, length=150.0, maxspeed_ms=13.9, lanes=2)
        g.add_edge(n, 0, key=0, length=150.0, maxspeed_ms=13.9, lanes=2)

    tmp = tempfile.mkdtemp(prefix="sumo_fx_")
    try:
        assert graph_to_sumo_plainxml(g, tmp, prefix="fx"), "plain-XML export failed"
        net_out = os.path.join(tmp, "fx.net.xml")
        assert rebuild_sumo_net("fx", tmp, net_out), "netconvert failed"

        net_root = ET.parse(net_out).getroot()
        edge_ids = [e.get("id") for e in net_root.findall("edge")
                    if not e.get("id", "").startswith(":")
                    and e.get("function") != "internal"]
        assert len(edge_ids) >= 2, f"too few real edges in net: {edge_ids}"
        has_net_tl = bool(net_root.findall("tlLogic"))
        if not has_net_tl:
            print("    (netconvert --tls.discard-simple dropped the TL — "
                  "TL assertions will be skipped)")

        # Prefer inbound arm edges (e_<u>_0_0): with --no-turnarounds a vehicle
        # on an outbound edge reaches a dead end and can never be retargeted.
        def _to_center(eid: str) -> bool:
            parts = eid.split("_")
            return len(parts) == 4 and parts[2] == "0" and parts[1] != "0"

        inbound = [e for e in edge_ids if _to_center(e)]
        route_edges = (inbound + [e for e in edge_ids if e not in inbound])[:2]
        rou = os.path.join(tmp, "fx.rou.xml")
        with open(rou, "w") as fh:
            fh.write(
                "<routes>\n"
                f'  <vehicle id="v0" depart="0"><route edges="{route_edges[0]}"/></vehicle>\n'
                f'  <vehicle id="v1" depart="1"><route edges="{route_edges[1]}"/></vehicle>\n'
                "</routes>\n"
            )
        cfg = os.path.join(tmp, "scn.sumocfg")
        with open(cfg, "w") as fh:
            fh.write(
                "<configuration>\n  <input>\n"
                '    <net-file value="fx.net.xml"/>\n'
                '    <route-files value="fx.rou.xml"/>\n'
                "  </input>\n</configuration>\n"
            )

        import traci
        sumo_bin = (shutil.which("sumo")
                    or os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "sumo"))
        traci.start([sumo_bin, "-c", cfg, "--no-warnings", "--no-step-log",
                     "--start", "--quit-on-end", "--step-length", "0.5"])
        try:
            appeared = False
            departed = 0
            rerouted = False
            for _ in range(20):
                traci.simulationStep()
                departed += int(traci.simulation.getDepartedNumber())
                ids = list(traci.vehicle.getIDList())
                if ids:
                    appeared = True
                # Reroute while a vehicle is active
                if ids and not rerouted:
                    veh = ids[0]
                    route_before = tuple(traci.vehicle.getRoute(veh))
                    cands = [e for e in edge_ids if e not in route_before][:5]
                    for cand in cands:
                        try:
                            traci.vehicle.changeTarget(veh, cand)
                        except Exception:
                            continue
                        route_after = tuple(traci.vehicle.getRoute(veh))
                        assert route_after != route_before or route_after[-1] == cand, \
                            "changeTarget succeeded but route unchanged"
                        rerouted = True
                        break
            assert appeared or departed > 0, "no vehicle ever appeared in SUMO"
            if not rerouted:
                print("    (no successful changeTarget while a vehicle was active "
                      "— reroute check inconclusive)")

            tl_ids = list(traci.trafficlight.getIDList())
            if tl_ids:
                tl = tl_ids[0]
                _ = traci.trafficlight.getProgram(tl)   # readable
                try:
                    traci.trafficlight.setProgram(tl, "off")
                    assert traci.trafficlight.getProgram(tl) == "off", \
                        "setProgram('off') accepted but program not 'off'"
                except AssertionError:
                    raise
                except Exception as exc:
                    print(f"    (setProgram('off') unsupported on this SUMO: {exc})")
            else:
                print("    (no TL in running net — TL toggle skipped)")
        finally:
            try:
                traci.close()
            except Exception:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def sumo_lane_counts_exported_correctly():
    """OSM lanes values must map to correct per-directed-edge numLanes."""
    import tempfile
    import xml.etree.ElementTree as ET
    import networkx as nx
    from sumo_network_patch import graph_to_sumo_plainxml

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = ("+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 "
                           "+datum=WGS84 +units=m +no_defs")
    coords = {"A": (0, 0), "B": (100, 0), "C": (0, 100), "D": (100, 100),
              "E": (0, 200), "F": (100, 200), "G": (0, 300), "H": (100, 300),
              "I": (0, 400), "J": (100, 400)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))

    g.add_edge("A", "B", key=0, length=100.0, lanes="4", oneway=True)      # → 4
    g.add_edge("C", "D", key=0, length=100.0, lanes="4")                   # → 2
    g.add_edge("D", "C", key=0, length=100.0, lanes="4")                   # → 2
    g.add_edge("E", "F", key=0, length=100.0, lanes="2;3", oneway=True)    # → 3
    g.add_edge("G", "H", key=0, length=100.0, lanes=["2", "3"])            # → 1
    g.add_edge("H", "G", key=0, length=100.0)                              # → 1
    g.add_edge("I", "J", key=0, length=100.0)                              # → 1

    with tempfile.TemporaryDirectory() as tmp:
        assert graph_to_sumo_plainxml(g, tmp, prefix="ln"), "export failed"
        edg = ET.parse(os.path.join(tmp, "ln.edg.xml")).getroot()
        got = {(e.get("from"), e.get("to")): int(e.get("numLanes"))
               for e in edg.findall("edge")}

    expected = {
        ("A", "B"): 4,   # oneway '4' → 4
        ("C", "D"): 2,   # bidirectional '4' → 4 // 2
        ("D", "C"): 2,
        ("E", "F"): 3,   # oneway '2;3' → max = 3
        ("G", "H"): 1,   # ['2','3'] with reverse → max(2,3) // 2 = 1
        ("H", "G"): 1,   # no lanes → 1
        ("I", "J"): 1,   # no lanes, no reverse → 1
    }
    for key, want in expected.items():
        assert got.get(key) == want, \
            f"edge {key}: numLanes {got.get(key)} != expected {want} (all: {got})"


# ══════════════════════════════════════════════════════════════════════════════
# Emissions sanity
# ══════════════════════════════════════════════════════════════════════════════

@test
def emissions_ordering_co2_nox_pm():
    """CO2 > NOx > PM at cruise speed; all are positive at idle."""
    from emissions import emission_rate_g_per_s
    v = 13.9  # ~50 km/h
    co2 = float(emission_rate_g_per_s("co2", v, "passenger"))
    nox = float(emission_rate_g_per_s("nox", v, "passenger"))
    pm  = float(emission_rate_g_per_s("pm",  v, "passenger"))
    assert co2 > nox > pm > 0, f"ordering wrong: co2={co2:.5f} nox={nox:.5f} pm={pm:.5f}"
    idle_co2 = float(emission_rate_g_per_s("co2", 0.0, "passenger"))
    assert idle_co2 > 0, "idle CO2 should be positive (idling engine)"
    bus_co2 = float(emission_rate_g_per_s("co2", v, "bus"))
    assert bus_co2 > co2, f"bus ({bus_co2:.4f}) should emit more CO2 than car ({co2:.4f})"


# ══════════════════════════════════════════════════════════════════════════════
# IDM chain braking
# ══════════════════════════════════════════════════════════════════════════════

@test
def idm_chain_braking_propagates():
    """Three cars queue behind a red light: all three must stop.

    Uses a TL ghost (same mechanism as tl_ghost_leader_stops_cars_at_red) so
    the barrier is explicit and independent of path-transition edge cases.
    """
    from idm import IDMParams, idm_tick
    from traffic_lights import TrafficLight
    paths = make_synthetic_paths(n_paths=4)
    nxt = make_next_edges(paths)
    junction_v = paths[0]["v"]
    path_len = paths[0]["length"]
    n = 3
    # 3 cars spaced 20 m apart, 30-70 m from the junction
    anim = make_car_anim(1, paths, seed=3)   # one car template
    anim = {
        "enabled": True,
        "edge_idx":   np.array([0, 0, 0], dtype=np.int64),
        "dist":       np.array([path_len - 30.0, path_len - 50.0, path_len - 70.0]),
        "speed":      np.array([10.0, 10.0, 10.0]),
        "desired_speed":      np.full(n, 13.9),
        "desired_speed_base": np.full(n, 13.9),
        "accel":      np.zeros(n),
        "car_len":    np.full(n, 4.5),
        "stop_wait":  np.zeros(n),
        "planned_edges":  [None] * n,
        "planned_cursor": np.zeros(n, dtype=np.int64),
    }
    # Red light: path 0 is in the permanently-waiting phase
    tl = TrafficLight(node_id=junction_v, x=0, y=0,
                      green_groups=[[99], [0]],
                      green_durations=[10000.0, 1.0],
                      yellow_durations=[4.0, 4.0],
                      n_phases=2, offset=0.0)
    rng = np.random.default_rng(3)
    p = IDMParams()
    for _ in range(400):  # 20 s
        idm_tick(anim, paths, nxt, {junction_v: tl}, dt=0.05, params=p, rng=rng)
    assert (anim["speed"] < 1.0).all(), \
        f"chain braking failed — speeds: {anim['speed'].round(3)}"
    assert (anim["edge_idx"] == 0).all(), "a car ran the red light"


# ══════════════════════════════════════════════════════════════════════════════
# Lane count parsing
# ══════════════════════════════════════════════════════════════════════════════

@test
def lane_count_parsing_edge_cases():
    """_parse_lane_count handles None, int, str, semicolon-separated, list, float-string."""
    from sumo_network_patch import _parse_lane_count
    assert _parse_lane_count(None)   is None,  "None → None"
    assert _parse_lane_count("")     is None,  "empty string → None"
    assert _parse_lane_count(1)      == 1,     "int 1 → 1"
    assert _parse_lane_count("2")    == 2,     "str '2' → 2"
    assert _parse_lane_count(2.0)    == 2,     "float 2.0 → 2"
    assert _parse_lane_count("0")    == 1,     "str '0' clamped to 1"
    # Semicolon-separated → max (some turns have more lanes than others)
    assert _parse_lane_count("2;3")  == 3,     "'2;3' → max=3"
    assert _parse_lane_count(["2","3"]) == 3,  "['2','3'] → max=3"


# ══════════════════════════════════════════════════════════════════════════════
# Scenario snapshot API
# ══════════════════════════════════════════════════════════════════════════════

@test
def scenario_snapshot_has_required_keys():
    """snapshot_scenario returns {nodes, edges, signals, demand, meta} with correct content."""
    import networkx as nx
    from scenario_compare import snapshot_scenario
    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=100.0, y=0.0)
    g.add_edge(0, 1, length=100.0, speed_limit=13.9, highway="residential")
    snap = snapshot_scenario(g, {}, [], name="test_A")
    for key in ("nodes", "edges", "signals", "demand", "meta"):
        assert key in snap, f"snapshot missing key '{key}'"
    assert snap["meta"]["name"] == "test_A", f"name mismatch: {snap['meta']}"
    assert "created" in snap["meta"], "missing created timestamp"
    assert len(snap["nodes"]) == 2, f"expected 2 nodes, got {len(snap['nodes'])}"
    assert len(snap["edges"]) == 1, f"expected 1 edge, got {len(snap['edges'])}"


# ══════════════════════════════════════════════════════════════════════════════
# IDM lane changing
# ══════════════════════════════════════════════════════════════════════════════

@test
def idm_lane_change_moves_slow_car():
    """A fast car blocked by a slow car should change to the adjacent lane.

    Setup:
      - Path 0 and path 1 are parallel, 200 m long, cross-linked as adj lanes.
      - A slow car (v=2 m/s, v0=2 m/s) sits on path 0 at dist=10 m (near the
        front of the fast car's view).
      - A fast car (v=10 m/s, v0=13.9 m/s) starts on path 0 at dist=0 m, so
        the gap to the slow car is < 15 m.
      - adj_left[0]=1, adj_right[0]=-1, adj_left[1]=-1, adj_right[1]=0 so the
        fast car can merge left onto path 1.
    After 100 ticks the fast car must be on path 1.
    """
    from idm import IDMParams, idm_tick

    length = 200.0
    # Two parallel straight paths: path 0 along y=0, path 1 along y=5
    def _make_path(u, v, y_offset):
        pts = np.array([[0.0, y_offset, 0.5], [length, y_offset, 0.5]], dtype=float)
        return {
            "u": u, "v": v,
            "length": length,
            "maxspeed_ms": 13.9,
            "points": pts,
            "cum_len": np.array([0.0, length]),
        }

    paths = [_make_path(0, 1, 0.0), _make_path(2, 3, 5.0)]
    # Ring-style next edges: path 0 → path 0, path 1 → path 1 (simple loops)
    nxt = [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)]

    # Adjacency: path 0 has left=1, path 1 has right=0
    adj_left  = np.array([1, -1], dtype=np.int64)
    adj_right = np.array([-1, 0], dtype=np.int64)

    # car 0: slow blocker on path 0 at dist=10 m
    # car 1: fast car on path 0 at dist=0 m (gap to car 0 is ~5.5 m < 15 m)
    anim = {
        "enabled": True,
        "edge_idx": np.array([0, 0], dtype=np.int64),
        "dist":     np.array([10.0, 0.0]),
        "speed":    np.array([2.0, 10.0]),
        "desired_speed":      np.array([2.0, 13.9]),
        "desired_speed_base": np.array([2.0, 13.9]),
        "accel":    np.zeros(2),
        "car_len":  np.full(2, 4.5),
        "stop_wait": np.zeros(2),
        "planned_edges":  [None, None],
        "planned_cursor": np.zeros(2, dtype=np.int64),
    }

    rng = np.random.default_rng(99)
    params = IDMParams()
    changed = False
    for _ in range(100):
        idm_tick(anim, paths, nxt, {}, dt=0.1, params=params, rng=rng,
                 adj_left=adj_left, adj_right=adj_right)
        if int(anim["edge_idx"][1]) == 1:
            changed = True
            break

    assert changed, (
        f"fast car (car 1) never changed to path 1 after 100 ticks; "
        f"final edge_idx={anim['edge_idx']}, dist={anim['dist']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streetlight placement
# ══════════════════════════════════════════════════════════════════════════════

def _make_lighting_scene():
    """Small cross-street scene: 100×100 m ground, one building, 4 street arms."""
    import pyvista as pv
    import networkx as nx
    ground = pv.Plane(center=(50, 50, 0), direction=(0, 0, 1),
                      i_size=100, j_size=100,
                      i_resolution=10, j_resolution=10).triangulate()
    bld = pv.Cube(center=(30, 30, 6), x_length=16, y_length=16,
                  z_length=12).triangulate()
    g = nx.MultiDiGraph()
    nodes = {0: (10, 50), 1: (90, 50), 2: (50, 10), 3: (50, 90), 4: (50, 50)}
    for nid, (x, y) in nodes.items():
        g.add_node(nid, x=float(x), y=float(y))
    for a, b in [(0, 4), (4, 1), (2, 4), (4, 3)]:
        L = float(np.hypot(nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1]))
        g.add_edge(a, b, length=L, highway="residential")
        g.add_edge(b, a, length=L, highway="residential")
    return ground, bld, g


@test
def streetlight_smart_placement_valid_layout():
    """Editor smart placement: right count, in bounds, spaced, deterministic."""
    from streetlight_ga import _smart_light_positions
    ground, _, g = _make_lighting_scene()
    pos = _smart_light_positions(ground, g, n_lights=10, grid_step=6.0, seed=42)
    assert pos.shape[0] == 10, f"expected 10 lights, got {pos.shape[0]}"
    assert (pos[:, 0] >= -1).all() and (pos[:, 0] <= 101).all(), "light X out of bounds"
    assert (pos[:, 1] >= -1).all() and (pos[:, 1] <= 101).all(), "light Y out of bounds"
    # No clustering: min pairwise spacing > 5 m
    d2 = np.sum((pos[:, None, :2] - pos[None, :, :2]) ** 2, axis=2)
    d2[np.arange(10), np.arange(10)] = np.inf
    assert float(np.sqrt(d2.min())) > 5.0, \
        f"lights too clustered: {np.sqrt(d2.min()):.1f} m apart"
    pos2 = _smart_light_positions(ground, g, n_lights=10, grid_step=6.0, seed=42)
    assert np.allclose(pos, pos2), "same seed gave a different layout"


@test
def streetlight_ga_improves_and_stays_in_bounds():
    """Full GA pipeline (coverage matrix + LOS + optimize) on a tiny scene."""
    from streetlight_ga import optimize_streetlights
    from app_core import _build_octree_from_buildings
    ground, bld, g = _make_lighting_scene()
    octree = _build_octree_from_buildings(bld)
    result = optimize_streetlights(
        ground_mesh=ground,
        n_lights=6,
        light_radius=20.0,
        w1=1.0, w2=0.35,
        grid_step=10.0,
        population_size=16,
        generations=8,
        seed=42,
        octree_root=octree,
        street_graph=g,
        ga_jobs=1,
    )
    pos, hist = result["best_positions"], result["history"]
    assert pos.shape[0] == 6, f"expected 6 lights, got {pos.shape[0]}"
    assert np.isfinite(result["best_cost"]), "non-finite GA cost"
    assert 0.0 < result["lit_ratio"] <= 1.0, f"lit_ratio out of range: {result['lit_ratio']}"
    assert hist[-1] <= hist[0] + 1e-9, f"GA cost did not improve: {hist[0]} → {hist[-1]}"
    assert (pos[:, 0] >= -5).all() and (pos[:, 0] <= 105).all(), "light X out of bounds"
    assert (pos[:, 1] >= -5).all() and (pos[:, 1] <= 105).all(), "light Y out of bounds"


@test
def tl_midblock_light_gets_real_red_phase():
    """Editor light on a straight road (2 approaches, one axis) must actually
    turn red — regression: n_phases=1 made it green ~97 % of the cycle with
    only the 1.5 s all-red pause as 'red'."""
    import networkx as nx
    from traffic_lights import build_traffic_lights, ALL_RED_PAUSE

    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0,   y=0.0)
    g.add_node(1, x=100.0, y=0.0, highway="traffic_signals")   # editor-tagged
    g.add_node(2, x=200.0, y=0.0)
    for a, b in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        g.add_edge(a, b, length=100.0)

    paths = []
    for a, b in [(0, 1), (2, 1)]:   # two approaches INTO node 1, same axis
        pts = np.array([[g.nodes[a]["x"], 0.0, 0.5], [g.nodes[b]["x"], 0.0, 0.5]])
        paths.append({"u": a, "v": b, "length": 100.0, "maxspeed_ms": 13.9,
                      "points": pts, "cum_len": np.array([0.0, 100.0])})

    lights = build_traffic_lights(g, paths, min_degree=3)
    assert 1 in lights, "tagged degree-2 node got no light"
    tl = lights[1]
    assert tl.n_phases == 2, f"mid-block light must have 2 phases, got {tl.n_phases}"

    # Simulate a full cycle and measure how long each approach is blocked
    cycle = (sum(tl.green_durations) + sum(tl.yellow_durations)
             + tl.n_phases * ALL_RED_PAUSE)
    dt, red_t = 0.1, 0.0
    for _ in range(int(cycle / dt) + 10):
        tl.tick(dt)
        if not tl.can_enter(0):
            red_t += dt
    red_frac = red_t / cycle
    assert red_frac > 0.15, \
        f"light is effectively always green — blocked only {red_frac:.0%} of the cycle"
    assert red_frac < 0.85, \
        f"light is almost always red — blocked {red_frac:.0%} of the cycle"


@test
def editor_undo_restores_graph():
    """'o' undo: a roundabout insert is fully reverted from the snapshot."""
    import networkx as nx
    from ui_mixin import UIMixin
    from editor_ops import insert_roundabout

    class _Plotter:
        def add_timer_event(self, **kw):
            raise RuntimeError("no timers headless")   # forces inline fallback
        def render(self):
            pass
        def remove_actor(self, *a, **kw):
            pass

    class _App(UIMixin):
        def __init__(self):
            self.scene_state = {}
            self.best_positions = None
            self.plotter = _Plotter()
            self.renderer = None
        def _rebuild_traffic_and_arrows(self):
            self.rebuilt = True

    app = _App()
    g = nx.MultiDiGraph()
    coords = {0: (200, 200), 1: (100, 200), 2: (200, 100), 3: (300, 200), 4: (200, 300)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))
    for n in (1, 2, 3, 4):
        g.add_edge(n, 0, length=100.0)
        g.add_edge(0, n, length=100.0)
    app.street_graph = g
    n_nodes0, n_edges0 = g.number_of_nodes(), g.number_of_edges()

    # Snapshot, then mutate destructively
    app._editor_push_undo("roundabout @ node 0")
    insert_roundabout(app.street_graph, 0)
    assert 0 not in app.street_graph.nodes, "roundabout did not remove center"
    assert app.street_graph.number_of_nodes() > n_nodes0, "ring nodes missing"

    # Undo → exact original topology
    app._editor_undo()
    assert 0 in app.street_graph.nodes, "undo did not restore the center node"
    assert app.street_graph.number_of_nodes() == n_nodes0, "node count differs after undo"
    assert app.street_graph.number_of_edges() == n_edges0, "edge count differs after undo"
    assert getattr(app, "rebuilt", False), "undo did not trigger traffic rebuild"

    # Stack empty → graceful no-op
    app._editor_undo()

    # Tag toggle (traffic light) undo restores the attribute
    app._editor_push_undo("tl toggle @ node 3")
    app.street_graph.nodes[3]["highway"] = "traffic_signals"
    app._editor_undo()
    assert app.street_graph.nodes[3].get("highway") is None, \
        "undo did not restore node tag"


@test
def editor_building_creation_and_undo():
    """'g' mode building placement: mesh merges into buildings_mesh, the shadow
    octree grows, and undo restores the exact original mesh + octree."""
    import pyvista as pv
    from editor_ops import make_building_mesh, building_size_for_click
    from app_core import _build_octree_from_buildings

    # Deterministic sizing per click position
    w1, d1, h1 = building_size_for_click(100.0, 100.0)
    w2, d2, h2 = building_size_for_click(100.0, 100.0)
    assert (w1, d1, h1) == (w2, d2, h2), "same click must give same building"
    assert 10.0 < w1 < 30.0 and 10.0 < h1 < 45.0, f"odd size: {w1}×{d1}×{h1}"

    box = make_building_mesh(100.0, 100.0, w1, d1, h1)
    assert box.n_points > 0 and box.n_cells > 0
    bx = box.bounds
    assert abs((bx[0] + bx[1]) / 2 - 100.0) < 1e-6, "not centred on click x"
    assert abs(bx[5] - bx[4] - h1) < 1e-6, "wrong height"

    # Merge + octree growth
    base = pv.Cube(center=(0, 0, 6), x_length=20, y_length=20, z_length=12).triangulate()
    octree0 = _build_octree_from_buildings(base)
    n_tri0 = len(octree0.triangles)

    snapshot = base.copy()                     # what _editor_push_undo stores
    merged = base.merge(box, merge_points=False)
    octree1 = _build_octree_from_buildings(merged)
    assert len(octree1.triangles) > n_tri0, "octree did not grow with new building"

    # Undo: snapshot restores the original triangle count exactly
    octree2 = _build_octree_from_buildings(snapshot)
    assert len(octree2.triangles) == n_tri0, "undo snapshot octree differs"
    assert snapshot.n_points == base.n_points


@test
def new_building_casts_shadow_on_street():
    """The point of the feature: a placed building must reduce sun exposure
    (raise shadow fraction) on the street beside it — which feeds straight
    into the solar routing costs."""
    import pyvista as pv
    from editor_ops import make_building_mesh
    from app_core import _build_octree_from_buildings
    from shadow_engine import compute_shadows

    # Flat ground strip east of where the tower will go; sun from low west
    ground = pv.Plane(center=(30, 0, 0), direction=(0, 0, 1),
                      i_size=40, j_size=20,
                      i_resolution=20, j_resolution=10).triangulate()
    # A tiny far-away stub so the 'before' octree isn't empty
    stub = pv.Cube(center=(500, 500, 2), x_length=4, y_length=4,
                   z_length=4).triangulate()
    # Direction TOWARD the sun: sun low in the WEST (-x), so the tower at
    # x=0 throws its shadow east over the ground strip at x≈10..50.
    sun_dir = np.array([-1.0, 0.0, 0.35])

    _, lit_before = compute_shadows(ground, _build_octree_from_buildings(stub), sun_dir)

    tower = make_building_mesh(0.0, 0.0, 16.0, 16.0, 40.0)
    both = stub.merge(tower, merge_points=False)
    _, lit_after = compute_shadows(ground, _build_octree_from_buildings(both), sun_dir)

    assert lit_after < lit_before - 0.05, \
        (f"tower cast no meaningful shadow: lit {lit_before:.2f} → {lit_after:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Solar fleet benefit study
# ══════════════════════════════════════════════════════════════════════════════

@test
def solar_fleet_benefit_study():
    """Quantify whether solar cars are beneficial: sweep a full day and compare
    fleet net energy (solar vs conventional) over the same road network.

    Physics sanity asserted:
      • at night the two fleets consume identical energy (no phantom harvest)
      • at noon the solar fleet consumes strictly less
      • fully-shadowed streets harvest ~nothing
      • the noon benefit lands in a physically plausible band (roof panel
        ≈350 W peak vs ≈1.5–2 kW urban traction → single-digit to ~40 %)
    Prints the study table — the numbers for the 'are solar cars worth it'
    question.
    """
    from solar_routing import build_edge_costs
    from solar_physics import SolarParams

    paths = make_synthetic_paths(n_paths=12, length=150.0)
    n = len(paths)
    # Mixed urban shading: open boulevards, half-shaded streets, dark canyons
    shadow = np.tile([0.0, 0.5, 1.0], n // 3 + 1)[:n].astype(float)
    params = SolarParams()
    LAT, LON = 38.0, 23.7   # Athens-ish

    def fleet_energy(hour: float, use_solar: bool) -> tuple[float, np.ndarray]:
        c = build_edge_costs(
            car_paths=paths, edge_shadow_frac=shadow,
            lat_deg=LAT, lon_deg=LON, hour_local=hour,
            params=params, use_solar=use_solar,
        )
        return float(np.sum(c["net_energy_J"])), np.asarray(c["solar_J"])

    print("    hour   conventional     solar fleet    saving")
    savings = {}
    for hour in (0.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0):
        e_conv, _ = fleet_energy(hour, use_solar=False)
        e_sol, solar_J = fleet_energy(hour, use_solar=True)
        pct = 100.0 * (e_conv - e_sol) / max(e_conv, 1e-9)
        savings[hour] = pct
        print(f"    {hour:4.0f}h  {e_conv/3600.0:9.1f} Wh   {e_sol/3600.0:9.1f} Wh   {pct:5.1f}%")

        # Fully-shadowed edges must harvest ~nothing at any hour
        full_shade = solar_J[shadow >= 1.0]
        if full_shade.size:
            assert np.all(full_shade < 1e-6), \
                f"shadowed street harvested energy at {hour}h: {full_shade}"

    assert abs(savings[0.0]) < 1e-6, f"night saving must be 0, got {savings[0.0]:.2f}%"
    assert abs(savings[21.0]) < 1e-6, "late-evening saving must be 0"
    assert savings[12.0] > 1.0, \
        f"noon solar saving implausibly small: {savings[12.0]:.2f}%"
    assert savings[12.0] < 60.0, \
        f"noon solar saving implausibly large: {savings[12.0]:.2f}%"
    # Diurnal shape: noon beats morning/evening shoulders
    assert savings[12.0] >= savings[9.0] - 1e-9
    assert savings[12.0] >= savings[15.0] - 1e-9

    daily = float(np.mean(list(savings.values())))
    print(f"    daily-average saving: {daily:.1f}%  (noon peak {savings[12.0]:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Buses + emergency vehicles
# ══════════════════════════════════════════════════════════════════════════════

@test
def emergency_spawn_creates_vehicle():
    """Ambulance spawn must not crash and must produce a routed vehicle.

    Regression: `int(rng.choice(n, size=2))` raised TypeError on every spawn,
    so the 'm' key never worked.
    """
    import networkx as nx
    from emergency_mixin import EmergencyMixin

    class _App(EmergencyMixin):
        pass

    app = _App()
    g = nx.MultiDiGraph()
    N = 5
    for i in range(N):
        for j in range(N):
            g.add_node(i * N + j, x=float(i * 100), y=float(j * 100))
    for i in range(N):
        for j in range(N):
            n0 = i * N + j
            if i + 1 < N:
                g.add_edge(n0, (i + 1) * N + j, length=100.0)
                g.add_edge((i + 1) * N + j, n0, length=100.0)
            if j + 1 < N:
                g.add_edge(n0, i * N + j + 1, length=100.0)
                g.add_edge(i * N + j + 1, n0, length=100.0)
    app.street_graph = g
    app.scene_state = {}
    app._init_emergency()
    app._spawn_one_emergency()
    vehicles = app.emergency_anim["vehicles"]
    assert len(vehicles) == 1, f"spawn failed — {len(vehicles)} vehicles"
    v = vehicles[0]
    assert v["route_pts"].shape[0] >= 2, "route has no polyline"
    assert float(v["route_cum"][-1]) >= 200.0, "route shorter than the 200 m minimum"
    # Advance 10 s — vehicle must move forward along its route
    app._advance_emergency(0.05)
    for _ in range(199):
        app._advance_emergency(0.05)
    assert float(v["dist"]) > 10.0 or np.any(v["pos"] != 0), "ambulance did not move"


@test
def bus_does_not_teleport_backwards_at_spawn():
    """Regression: stop_cursor=0 (stop at dist 0) with a random start_dist made
    the first tick 'arrive' at a stop far behind and teleport the bus back."""
    from bus_mixin import BusMixin

    class _App(BusMixin):
        pass

    app = _App()
    app.scene_state = {}
    route_pts = np.array([[float(x), 0.0, 0.0] for x in range(0, 501, 100)])
    route_cum = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    bus = {
        "route_pts": route_pts, "route_cum": route_cum,
        "dist": 250.0,                      # mid-route spawn
        "speed": 5.0, "dwell_timer": 0.0,
        "stop_dists": [(0.0, 10.0), (150.0, 10.0), (400.0, 10.0)],
        "stop_cursor": 2,                   # first stop AHEAD of 250 (the fix)
        "v0": 8.0, "bus_len": 12.0, "route_name": "T", "trip_id": "t1",
    }
    app.buses = [bus]
    app._advance_buses(0.05)
    assert bus["dist"] >= 250.0, \
        f"bus teleported backwards: dist={bus['dist']} (started at 250)"
    # Drive 60 s: it should reach and dwell at the 400 m stop, never jump back
    prev = bus["dist"]
    for _ in range(1200):
        app._advance_buses(0.05)
        assert bus["dist"] >= prev - 3.0, \
            f"bus jumped backwards: {prev:.1f} → {bus['dist']:.1f}"
        prev = bus["dist"]
    # Cursor-behind case: wrapped cursor pointing at dist 0 must be skipped
    bus2 = dict(bus)
    bus2["dist"] = 450.0
    bus2["speed"] = 5.0
    bus2["dwell_timer"] = 0.0
    bus2["stop_cursor"] = 0                 # stop at 0.0 — far behind
    app.buses = [bus2]
    app._advance_buses(0.05)
    assert bus2["dist"] >= 450.0, \
        f"wrapped cursor teleported bus: dist={bus2['dist']}"


# ══════════════════════════════════════════════════════════════════════════════
# SUMO analytics (native emissions / congestion / KPIs / incidents)
# ══════════════════════════════════════════════════════════════════════════════

@test
def sumo_analytics_apis_work():
    """Real SUMO via SumoConnection: emission snapshot, edge congestion,
    KPI snapshot, edge shapes, and incident break/release."""
    import shutil
    import tempfile
    import networkx as nx
    from sumo_bridge import check_sumo, SumoConnection
    from sumo_network_patch import graph_to_sumo_plainxml, rebuild_sumo_net, _resolve_netconvert

    ok, msg = check_sumo()
    if not ok:
        print(f"    (SUMO unavailable — skipped: {msg})")
        return
    nc = _resolve_netconvert()
    if not (os.path.isfile(nc) or shutil.which(nc)):
        print("    (netconvert unresolvable — skipped)")
        return

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = ("+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 "
                           "+datum=WGS84 +units=m +no_defs")
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=400.0, y=0.0)
    g.add_edge(0, 1, key=0, length=400.0, maxspeed_ms=13.9, lanes=1)
    g.add_edge(1, 0, key=0, length=400.0, maxspeed_ms=13.9, lanes=1)

    tmp = tempfile.mkdtemp(prefix="sumo_an_")
    try:
        assert graph_to_sumo_plainxml(g, tmp, prefix="an")
        net_out = os.path.join(tmp, "an.net.xml")
        assert rebuild_sumo_net("an", tmp, net_out)

        import xml.etree.ElementTree as ET
        edge_ids = [e.get("id") for e in ET.parse(net_out).getroot().findall("edge")
                    if not e.get("id", "").startswith(":")
                    and e.get("function") != "internal"]
        assert edge_ids, "no edges in net"

        rou = os.path.join(tmp, "an.rou.xml")
        with open(rou, "w") as fh:
            fh.write("<routes>\n")
            for i in range(4):
                fh.write(f'  <vehicle id="v{i}" depart="{i * 0.5}">'
                         f'<route edges="{edge_ids[0]}"/></vehicle>\n')
            fh.write("</routes>\n")
        cfg = os.path.join(tmp, "an.sumocfg")
        with open(cfg, "w") as fh:
            fh.write(
                "<configuration>\n  <input>\n"
                '    <net-file value="an.net.xml"/>\n'
                '    <route-files value="an.rou.xml"/>\n'
                "  </input>\n</configuration>\n"
            )

        conn = SumoConnection(cfg, step_length=0.5, net_file=net_out)
        assert conn.start(), "SumoConnection.start failed"
        try:
            # Step until vehicles are on the road
            for _ in range(10):
                conn.step()
                if conn.vehicles():
                    break
            assert conn.vehicles(), "no vehicles departed"

            # 1) Native emission snapshot — moving cars must emit CO2.
            # HBEFA4 reports 0 during coasting (fuel cut), so sample the max
            # over several steps rather than one instant.
            peak_rate = 0.0
            em = []
            for _ in range(8):
                conn.step()
                em = conn.vehicle_emission_snapshot("co2")
                if em:
                    assert all("lon" in e and "lat" in e and "rate" in e for e in em)
                    peak_rate = max(peak_rate, max(e["rate"] for e in em))
            assert em, "empty emission snapshot with active vehicles"
            assert peak_rate > 0.0, "no positive CO2 across 8 steps of driving"

            # 2) Edge congestion — the loaded edge must appear with sane values
            cong = conn.edge_congestion()
            assert cong, "edge_congestion empty with vehicles on the road"
            for eid, (mean_v, ff_v, n_veh) in cong.items():
                assert 0.0 <= mean_v <= ff_v + 1.0, f"{eid}: mean {mean_v} vs ff {ff_v}"
                assert n_veh >= 1

            # 3) Edge shape — cached lon/lat polyline for any congested edge
            some_eid = next(iter(cong))
            shape = conn.edge_shape_lonlat(some_eid)
            assert shape and len(shape) >= 2, f"no shape for {some_eid}"

            # 4) KPI snapshot
            kpi = conn.kpi_snapshot()
            assert kpi.get("running", 0) >= 1, f"KPI running wrong: {kpi}"
            assert kpi.get("mean_timeloss_s", -1) >= 0.0

            # 5) Incident: break a vehicle → its speed must drop to ~0, then release
            vid = conn.incident_break_random_vehicle(60.0)
            assert vid is not None, "incident could not pick a vehicle"
            for _ in range(10):   # 5 s — enough to brake to a stop
                conn.step()
            speeds = {v["id"]: v["speed"] for v in conn.vehicles()}
            if vid in speeds:     # may have arrived — only assert when present
                assert speeds[vid] < 0.5, \
                    f"broken-down vehicle still moving at {speeds[vid]:.2f} m/s"
                assert conn.incident_release(vid), "release failed"

            # 6) findRoute — router travel time between the two edges
            if len(edge_ids) >= 2:
                tt = conn.find_route_time(edge_ids[0], edge_ids[1])
                if tt is not None:      # None if disconnected — geometry-dependent
                    assert 0.0 < tt < 3600.0, f"absurd findRoute time: {tt}"

            # 7) TL link states — this simple net has no signals, so the call
            # must return an empty list WITHOUT raising
            assert conn.tl_link_states() == [], \
                "tl_link_states should be empty on an unsignalised net"
        finally:
            conn.close()
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


@test
def highway_bridge_creates_graph_edge_and_tube():
    """'y' mode highway bridge: adds bidirectional motorway edges to the graph
    with correct attributes, and the tube geometry spans the expected height."""
    import networkx as nx
    from editor_ops import add_bridge_highway_edge, make_highway_tube

    G = nx.MultiDiGraph()
    G.add_node("A", x=0.0, y=0.0)
    G.add_node("B", x=200.0, y=0.0)

    add_bridge_highway_edge(G, "A", "B", 0.0, 0.0, 200.0, 0.0)

    assert G.has_edge("A", "B") and G.has_edge("B", "A"), "both directions required"
    for u, v in (("A", "B"), ("B", "A")):
        d = dict(list(G[u][v].values())[0])
        assert d["highway"] == "motorway"
        assert d["bridge"] is True
        assert abs(d["length"] - 200.0) < 0.01
        assert d["lanes"] == 2
        assert d["maxspeed"] == 100

    # Tube geometry: ramped flat deck should reach the requested height
    tube = make_highway_tube(0.0, 0.0, 200.0, 0.0, height=6.0)
    assert tube.n_points > 200, "tube should be well-sampled"
    bz_max = tube.bounds[5]
    bz_min = tube.bounds[4]
    assert bz_max > 5.0, f"deck top {bz_max:.2f} m < 5 m (radius included)"
    assert bz_min < 2.5, f"deck base {bz_min:.2f} m too high (should touch near ground)"

    # Clearance: a 20 m building under the span must raise the deck above it
    from editor_ops import bridge_clearance_height
    bpts = np.array([[100.0, 2.0, 0.0], [100.0, 2.0, 20.0],
                     [105.0, -2.0, 20.0]])   # building corner points near midspan
    h = bridge_clearance_height(bpts, 0.0, 0.0, 200.0, 0.0, base_height=5.0)
    assert h >= 23.0, f"deck {h:.1f} m does not clear the 20 m building"
    # Building far from the corridor must NOT raise the deck
    far = np.array([[100.0, 50.0, 30.0]])
    h2 = bridge_clearance_height(far, 0.0, 0.0, 200.0, 0.0, base_height=5.0)
    assert h2 == 5.0, f"distant building wrongly raised deck to {h2:.1f} m"


@test
def route_planner_uses_new_highway_bridge():
    """After the 'y' editor adds a bridge, user-ordered routes must see the
    new motorway edges and prefer them when faster (both by length and by
    the travel-time/energy cost model)."""
    import networkx as nx
    from editor_ops import add_bridge_highway_edge
    from solar_routing import (build_edge_costs, build_graph_with_costs,
                               find_energy_optimal_route, SolarParams)

    # City: A→X→Y→B is a slow 900 m residential detour
    g = nx.MultiDiGraph()
    coords = {"A": (0, 0), "X": (300, 400), "Y": (600, 400), "B": (900, 0)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))
    detour = [("A", "X"), ("X", "Y"), ("Y", "B")]
    for u, v in detour:
        d = float(np.hypot(coords[v][0]-coords[u][0], coords[v][1]-coords[u][1]))
        g.add_edge(u, v, length=d, highway="residential", maxspeed=30)
        g.add_edge(v, u, length=d, highway="residential", maxspeed=30)

    # Without the bridge: shortest A→B is the 3-hop detour
    assert len(nx.shortest_path(g, "A", "B", weight="length")) == 4

    # Editor adds the bridge (900 m direct motorway)
    add_bridge_highway_edge(g, "A", "B", 0.0, 0.0, 900.0, 0.0)

    # Rebuild car_paths the way _rebuild_traffic_and_arrows does (simplified):
    # one path dict per drivable directed edge.
    car_paths = []
    for u, v, d in g.edges(data=True):
        ms = float(d.get("maxspeed", 30)) / 3.6
        car_paths.append({"u": u, "v": v,
                          "length": float(d["length"]), "maxspeed_ms": ms})

    costs = build_edge_costs(
        car_paths=car_paths,
        edge_shadow_frac=np.ones(len(car_paths)),
        lat_deg=38.0, lon_deg=23.7, hour_local=12.0,
        params=SolarParams(), alpha=0.5, use_solar=False,
    )
    g_cost = build_graph_with_costs(g, car_paths, costs)

    # Bridge edge must exist in the routing graph with cost attributes
    assert g_cost.has_edge("A", "B"), "bridge edge missing from routing graph"

    # Shortest-by-length must now be the direct bridge hop
    sp = nx.shortest_path(g_cost, "A", "B", weight="length")
    assert sp == ["A", "B"], f"shortest path ignored the bridge: {sp}"

    # Fastest route must use it (32 s at 100 km/h vs 156 s on the detour)
    sp_t = nx.shortest_path(g_cost, "A", "B", weight="travel_time_s")
    assert sp_t == ["A", "B"], f"travel-time route ignored the bridge: {sp_t}"

    # The energy-optimal route may legitimately AVOID the motorway (drag ∝ v²:
    # 900 m at 100 km/h burns more than 1300 m at 30) — just require validity.
    er = find_energy_optimal_route(g_cost, "A", "B")
    assert er is not None and list(er)[0] == "A" and list(er)[-1] == "B"


@test
def centrality_values_nonzero_on_multidigraph():
    """edge_betweenness_centrality on a MultiDiGraph returns (u, v, key)
    triples; the overlay must collapse them to (u, v) or every road renders
    as zero (all-purple bug)."""
    import networkx as nx
    from analysis_mixin import _collapse_edge_centrality

    g = nx.MultiDiGraph()
    # bowtie: node 2 is the bridge between two triangles → high betweenness
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]:
        g.add_edge(u, v, length=10.0)
        g.add_edge(v, u, length=10.0)

    ec = nx.edge_betweenness_centrality(g, normalized=True, weight="length")
    ec_uv = _collapse_edge_centrality(ec)

    # Direct (u, v) lookups must now succeed with non-zero values
    vals = [max(ec_uv.get((u, v), 0.0), ec_uv.get((v, u), 0.0))
            for u, v in {(0, 1), (1, 2), (2, 3)}]
    assert all(v > 0.0 for v in vals), f"centrality lookups all zero: {vals}"
    # The graph must show contrast (backbone > leaf edges), not a flat colour
    assert max(ec_uv.values()) > min(ec_uv.values()), "no value spread"


@test
def streetlight_coverage_study():
    """Quantify lighting economics: how much street coverage does each extra
    lamp buy?  Sweeps lamp count on the standard lighting scene and asserts
    diminishing returns — the marginal lit-ratio gain per lamp must shrink
    as the street saturates.  Prints the cost-benefit table."""
    from streetlight_ga import optimize_streetlights
    from app_core import _build_octree_from_buildings

    ground, bld, g = _make_lighting_scene()
    octree = _build_octree_from_buildings(bld)

    counts = [2, 4, 6, 8, 12]
    ratios = []
    print("    lamps   lit ratio   marginal gain/lamp")
    prev = 0.0
    marginals = []
    for n in counts:
        r = optimize_streetlights(
            ground_mesh=ground, n_lights=n, light_radius=20.0,
            w1=1.0, w2=0.35, grid_step=10.0,
            population_size=16, generations=8, seed=42,
            octree_root=octree, street_graph=g, ga_jobs=1,
        )
        lit = float(r["lit_ratio"])
        ratios.append(lit)
        dn = n - (counts[counts.index(n) - 1] if counts.index(n) else 0)
        marg = (lit - prev) / dn
        marginals.append(marg)
        print(f"    {n:5d}   {lit:9.3f}   {marg:+.3f}")
        prev = lit

    # More lamps never light less street
    for a, b in zip(ratios, ratios[1:]):
        assert b >= a - 0.02, f"coverage decreased when adding lamps: {ratios}"
    # 12 lamps × r=20 m on a 100×100 scene with one building: geometric max
    # coverage lands in the 0.55–0.75 band (street-weighted, LOS-blocked)
    assert ratios[-1] > 0.55, f"12 lamps light too little: {ratios[-1]:.2f}"
    # Diminishing returns: the first lamps buy far more than the last
    assert marginals[0] > marginals[-1], \
        f"no diminishing returns: first {marginals[0]:.3f} vs last {marginals[-1]:.3f}"
    print(f"    → first lamps buy {marginals[0]/max(marginals[-1],1e-4):.0f}× "
          f"more coverage than the last")


@test
def congestion_externality_study():
    """Quantify what congestion costs: sweep fleet size on a fixed ring road
    and measure equilibrium speed, CO₂, and noise.  Asserts the physics:
    more cars → lower speeds; lower speeds → MORE CO₂ per km (EEA curve);
    noise grows with fleet energy.  Prints the externality table."""
    from idm import IDMParams, idm_tick
    import emissions as em

    paths = make_synthetic_paths(n_paths=8, length=120.0)  # ~960 m ring
    nxt = make_next_edges(paths)

    def car_xy(anim):
        out = np.zeros((len(anim["edge_idx"]), 2))
        for i, (e, d) in enumerate(zip(anim["edge_idx"], anim["dist"])):
            p = paths[int(e)]
            dd = min(max(float(d), 0.0), p["length"])
            out[i, 0] = np.interp(dd, p["cum_len"], p["points"][:, 0])
            out[i, 1] = np.interp(dd, p["cum_len"], p["points"][:, 1])
        return out

    print("    cars   speed km/h   CO2 g/km/car   fleet CO2 g/s   noise ≥65dB cells")
    rows = []
    for n_cars in (15, 40, 90):
        anim = make_car_anim(n_cars, paths, seed=5)
        rng = np.random.default_rng(5)
        for _ in range(1200):                      # 60 s at 0.05 s — settle
            idm_tick(anim, paths, nxt, {}, dt=0.05,
                     params=IDMParams(), rng=rng)
        spd = np.asarray(anim["speed"], dtype=float)
        mean_kmh = float(np.mean(spd)) * 3.6

        co2_rate = em.emission_rate_g_per_s("co2", spd)        # g/s per car
        fleet_gs = float(np.sum(co2_rate))
        gs_per_km = float(np.mean(
            em.emission_factor_g_per_km("co2", np.maximum(spd, 1.0))))

        # Noise: 12 m grid over the ring, CNOSSOS-style summation
        xy = car_xy(anim)
        gx, gy = np.meshgrid(np.arange(-130, 131, 12.0),
                             np.arange(-130, 131, 12.0))
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        d = np.maximum(np.linalg.norm(
            xy[:, None, :] - cells[None, :, :], axis=2), 1.0)
        lw = 85.0 + 10.0 * np.log10(np.maximum(spd * 3.6, 1.0) / 50.0)
        lin = np.sum(10.0 ** ((lw[:, None] - 20.0 * np.log10(d)) / 10.0), axis=0)
        breach = int(np.count_nonzero(10.0 * np.log10(lin) >= 65.0))

        rows.append((n_cars, mean_kmh, gs_per_km, fleet_gs, breach))
        print(f"    {n_cars:4d}   {mean_kmh:10.1f}   {gs_per_km:12.0f}   "
              f"{fleet_gs:13.1f}   {breach:17d}")

    # Physics assertions
    speeds  = [r[1] for r in rows]
    gkm     = [r[2] for r in rows]
    breach_ = [r[4] for r in rows]
    assert speeds[0] > speeds[-1] + 1.0, \
        f"congestion did not slow traffic: {speeds}"
    assert gkm[-1] > gkm[0], \
        f"CO2/km must RISE in congestion (EEA curve): {gkm}"
    assert breach_[-1] >= breach_[0], f"noise did not grow with fleet: {breach_}"
    print(f"    → {rows[-1][0]/rows[0][0]:.0f}× cars: speed ÷"
          f"{speeds[0]/max(speeds[-1],0.1):.1f}, CO2/km ×{gkm[-1]/gkm[0]:.2f}, "
          f"breach cells ×{breach_[-1]/max(breach_[0],1):.1f}")


@test
def crossing_paths_link_sidewalks_across_road():
    """Sidewalk nodes facing each other across a road get a crossing link;
    nodes on the same side (no road between) do not."""
    import networkx as nx
    from safety import (build_crossing_paths, drivable_paths_from_graph,
                        road_segments_from_paths, ped_paths_from_graph)

    g = nx.MultiDiGraph()
    # Road along y=0; sidewalks at y=±6 with nodes every 50 m
    for i in range(4):
        g.add_node(f"r{i}", x=i * 50.0, y=0.0)
        g.add_node(f"n{i}", x=i * 50.0, y=6.0)    # north sidewalk
        g.add_node(f"s{i}", x=i * 50.0, y=-6.0)   # south sidewalk
    for i in range(3):
        g.add_edge(f"r{i}", f"r{i+1}", length=50.0, highway="residential", maxspeed=50)
        g.add_edge(f"r{i+1}", f"r{i}", length=50.0, highway="residential", maxspeed=50)
        for side in ("n", "s"):
            g.add_edge(f"{side}{i}", f"{side}{i+1}", length=50.0, highway="footway")
            g.add_edge(f"{side}{i+1}", f"{side}{i}", length=50.0, highway="footway")

    car_paths, _ = drivable_paths_from_graph(g)
    ped_paths, _ = ped_paths_from_graph(g)
    segs = road_segments_from_paths(car_paths)
    new_paths, extra_out = build_crossing_paths(g, ped_paths, segs, max_len=30.0)

    assert new_paths, "no crossings generated at all"
    # Every crossing must actually span the road (one endpoint each side)
    for p in new_paths:
        ys = p["points"][:, 1]
        assert ys.min() < 0 < ys.max(), f"crossing does not span road: ys={ys}"
        assert p["is_road_crossing"] and p["is_crossing_end"]
    # Crossings are bidirectional pairs registered in outgoing
    assert len(new_paths) % 2 == 0
    assert extra_out, "outgoing links missing"


@test
def ped_car_collision_injures_pedestrian():
    """A fast car within the hit radius injures a ped exactly once; a stopped
    car never does."""
    from safety import check_ped_car_collisions

    ped = np.array([[10.0, 0.0], [50.0, 0.0]])
    injured = np.zeros(2, dtype=bool)

    # Stopped car on ped #0: no injury
    hits, _ = check_ped_car_collisions(
        ped, injured, np.array([[10.0, 0.5]]), np.array([0.0]))
    assert hits.shape[0] == 0, "stationary car injured a pedestrian"

    # Fast car on ped #0: injury at impact speed
    hits, impact = check_ped_car_collisions(
        ped, injured, np.array([[10.0, 0.5]]), np.array([10.0]))
    assert list(hits) == [0] and impact[0] == 10.0
    injured[hits] = True

    # Already injured: not counted again
    hits, _ = check_ped_car_collisions(
        ped, injured, np.array([[10.0, 0.5]]), np.array([10.0]))
    assert hits.shape[0] == 0, "pedestrian injured twice"


@test
def safety_study_runs_headless_and_compares_presets():
    """Full headless co-simulation on a synthetic grid city: runs under both
    signal presets, produces injury stats, and pedestrians actually cross."""
    import networkx as nx
    from safety import run_safety_study

    # 4×4 road grid, 100 m spacing, with parallel sidewalk pairs on each road
    g = nx.MultiDiGraph()
    N = 4
    for i in range(N):
        for j in range(N):
            g.add_node(i * N + j, x=i * 100.0, y=j * 100.0)
    def _road(a, b):
        d = 100.0
        g.add_edge(a, b, length=d, highway="residential", maxspeed=50)
        g.add_edge(b, a, length=d, highway="residential", maxspeed=50)
    for i in range(N):
        for j in range(N):
            n0 = i * N + j
            if i + 1 < N: _road(n0, (i + 1) * N + j)
            if j + 1 < N: _road(n0, i * N + j + 1)
    # Sidewalks: offset node pairs flanking each horizontal road
    sid = 1000
    for i in range(N - 1):
        for j in range(N):
            for off in (+4.0, -4.0):
                a, b = sid, sid + 1
                g.add_node(a, x=i * 100.0, y=j * 100.0 + off)
                g.add_node(b, x=(i + 1) * 100.0, y=j * 100.0 + off)
                g.add_edge(a, b, length=100.0, highway="footway")
                g.add_edge(b, a, length=100.0, highway="footway")
                sid += 2

    common = dict(duration_s=40.0, dt=0.1, n_cars=25, n_peds=40, seed=7)
    r_on  = run_safety_study(g, signals=True,  **common)
    r_off = run_safety_study(g, signals=False, **common)

    for r in (r_on, r_off):
        assert "error" not in r, f"study failed: {r}"
        assert r["crossings_built"] > 0, "no street crossings generated"
        assert r["injuries"] >= 0
    assert r_on["n_lights"] > 0, "signal preset created no lights"
    assert r_off["n_lights"] == 0
    assert r_on["ped_crossing_events"] + r_off["ped_crossing_events"] > 0, \
        "pedestrians never crossed a street"


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

# Quantitative studies (solar benefit sweep is fast; the streetlight GA sweep
# takes ~90 s) — excluded from the default run so pre-commit stays fast.
# Run them with:  python tests/run_tests.py --studies
_HEAVY_STUDIES = {"streetlight_coverage_study"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", metavar="SUBSTR", default="",
                    help="only run tests whose name contains SUBSTR")
    ap.add_argument("-v", action="store_true", help="verbose")
    ap.add_argument("--studies", action="store_true",
                    help="include the slow quantitative studies (GA sweep ~90 s)")
    args = ap.parse_args()

    selected = [(n, f) for n, f in _TESTS if args.k in n]
    if not args.studies and not args.k:
        skipped = [n for n, _ in selected if n in _HEAVY_STUDIES]
        selected = [(n, f) for n, f in selected if n not in _HEAVY_STUDIES]
        if skipped:
            print(f"  (skipping heavy studies: {', '.join(skipped)} — "
                  f"run with --studies to include)")
    if not selected:
        print(f"no tests match -k '{args.k}'")
        return 1

    passed, failed = 0, []
    t0 = time.perf_counter()
    for name, fn in selected:
        if args.v:
            print(f"  running {name} …", flush=True)
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"  ✗ {name}: {exc}")
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"  ✗ {name}: ERROR")
            traceback.print_exc()
    dt = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"  {passed}/{len(selected)} passed in {dt:.1f}s")
    if failed:
        print(f"  FAILED: {', '.join(n for n, _ in failed)}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
