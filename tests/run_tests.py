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
import json
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
    # Regular n_paths-gon chord length = 2R sin(pi/n) -> solve for R so each
    # edge is exactly `length` long (n_paths=2 is a straight there-and-back
    # line, not a polygon; chord = 2R directly).
    radius = length / 2.0 if n_paths <= 2 else length / (2.0 * np.sin(np.pi / n_paths))
    for i in range(n_paths):
        # Straight segments laid out in a regular ring of the requested chord length
        angle = i * (2 * np.pi / n_paths)
        x0, y0 = radius * np.cos(angle), radius * np.sin(angle)
        angle2 = (i + 1) * (2 * np.pi / n_paths)
        x1, y1 = radius * np.cos(angle2), radius * np.sin(angle2)
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
    # Never faster than the road's free-flow speed. (Cars may briefly exceed a
    # just-lowered tick target — a turn approach — while braking toward it.)
    assert np.all(anim["speed"] <= anim["desired_speed_base"] + 1e-6), "overspeed"


@test
def idm_speed_cap_brakes_smoothly_and_releases():
    """speed_cap must actually slow cars (it used to be written to a key
    idm_tick ignored), must do so with bounded deceleration (no one-step
    snap from 13 m/s to 0.5 m/s), and must be released when the cap goes."""
    from idm import IDMParams, idm_tick
    paths = make_synthetic_paths(n_paths=2, length=400.0)
    anim = make_car_anim(1, paths)
    anim["edge_idx"][:] = 0
    anim["dist"][:] = 5.0
    anim["speed"][:] = 13.0
    p, dt = IDMParams(), 0.05
    rng = np.random.default_rng(0)
    cap = np.array([0.5])
    prev = 13.0
    for _ in range(80):
        idm_tick(anim, paths, make_next_edges(paths), {}, dt=dt, params=p, rng=rng, speed_cap=cap)
        v = float(anim["speed"][0])
        assert prev - v <= 3.0 * p.b * dt + 1e-9, f"decel {(prev - v) / dt:.1f} m/s² exceeds 3b"
        prev = v
    assert prev < 1.0, f"capped car did not slow down (v={prev:.2f})"
    for _ in range(200):
        idm_tick(anim, paths, make_next_edges(paths), {}, dt=dt, params=p, rng=rng)
    assert float(anim["speed"][0]) > 5.0, "car did not re-accelerate after the cap was lifted"


@test
def parking_exit_recycles_car_without_breaking_idm():
    """Regression: parking exits used to append to only 5 of the per-car
    arrays, so the next idm_tick indexed car_len/accel/... out of range and
    killed the whole car layer. Exits must recycle a fleet car in place."""
    import types
    from idm import IDMParams, idm_tick
    from parking_mixin import ParkingMixin

    paths = make_synthetic_paths()
    anim = make_car_anim(10, paths)
    anim["pos"] = np.zeros((10, 3))
    anim["pos"][3] = [900.0, 900.0, 0.0]  # farthest from the camera -> recycled
    anim["planned_edges"][5] = np.array([1, 2])  # active trip: must not be taken

    host = types.SimpleNamespace(
        car_anim=anim, car_paths=paths,
        args=types.SimpleNamespace(traffic_speed=1.0),
        route_state={"selected_car_idx": None},
        plotter=types.SimpleNamespace(camera=types.SimpleNamespace(focal_point=(0.0, 0.0, 0.0))),
        scene_state={"_parked_car_positions": np.array([[paths[2]["points"][0][0] + 5.0,
                                                         paths[2]["points"][0][1], 0.0]])},
        _parking_rng=np.random.default_rng(1),
    )
    for name in ("_pick_car_to_recycle", "_spawn_exiting_car"):
        setattr(host, name, types.MethodType(getattr(ParkingMixin, name), host))

    sizes_before = {k: len(v) for k, v in anim.items() if hasattr(v, "__len__")}
    host._spawn_exiting_car(0)
    sizes_after = {k: len(v) for k, v in anim.items() if hasattr(v, "__len__")}
    assert sizes_before == sizes_after, f"fleet arrays changed size: {sizes_before} -> {sizes_after}"
    assert int(anim["edge_idx"][3]) == 2 and anim["dist"][3] == 0.0, "farthest idle car was not recycled onto the exit path"
    assert anim["planned_edges"][5] is not None, "a car on an active demand trip was recycled"

    rng = np.random.default_rng(0)
    for _ in range(40):
        idm_tick(anim, paths, make_next_edges(paths), {}, dt=0.05, params=IDMParams(), rng=rng)
    assert np.all(np.isfinite(anim["speed"])) and np.all(anim["speed"] >= 0.0)


@test
def vtk_timers_fire_only_their_own_ticks_and_clean_up():
    """pyvista 0.47's add_timer_event observers cross-fire on every timer and
    never unregister; a finished one-shot then DestroyTimer()s a stale id
    that VTK may have reused for a new timer. vtk_timers.add_timer must not."""
    import types
    from vtk_timers import add_timer

    class FakeIren:
        def __init__(self):
            self.obs, self.alive, self._next_tag, self.cur = {}, set(), 0, None
            self.renders = 0
        def AddObserver(self, _evt, fn):
            self._next_tag += 1
            self.obs[self._next_tag] = fn
            return self._next_tag
        def RemoveObserver(self, tag):
            self.obs.pop(tag, None)
        def _create(self):
            tid = min(set(range(1, 100)) - self.alive)   # VTK-style id reuse
            self.alive.add(tid)
            return tid
        CreateRepeatingTimer = CreateOneShotTimer = lambda self, _ms: self._create()
        def DestroyTimer(self, tid):
            self.alive.discard(tid)
        def GetTimerEventId(self):
            return self.cur
        def GetRenderWindow(self):
            return types.SimpleNamespace(Render=lambda: setattr(self, "renders", self.renders + 1))
        def fire(self, tid):
            self.cur = tid
            for fn in list(self.obs.values()):
                fn(self, "TimerEvent")

    iren = FakeIren()
    plotter = types.SimpleNamespace(iren=types.SimpleNamespace(interactor=iren))
    calls = {"a": 0, "b": 0, "c": 0}
    add_timer(plotter, 50, lambda s: calls.__setitem__("a", calls["a"] + 1))
    add_timer(plotter, 50, lambda s: calls.__setitem__("b", calls["b"] + 1), repeating=False)
    a_id, b_id = 1, 2

    for _ in range(5):
        iren.fire(a_id)
    assert calls == {"a": 5, "b": 0, "c": 0}, f"timers cross-fired: {calls}"
    iren.fire(b_id)
    iren.fire(b_id)
    assert calls["b"] == 1, "one-shot fired more than once"
    assert len(iren.obs) == 1, "finished one-shot left its observer registered"
    assert b_id not in iren.alive

    stop_c = add_timer(plotter, 50, lambda s: calls.__setitem__("c", calls["c"] + 1))
    assert b_id in iren.alive, "test setup: VTK should reuse the freed id"
    iren.fire(b_id)                      # this is now timer C's id
    assert calls["c"] == 1 and b_id in iren.alive, "a stale one-shot destroyed a reused timer id"
    stop_c()
    assert b_id not in iren.alive and len(iren.obs) == 1


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
def idm_planned_route_cannot_jump_to_a_disconnected_edge():
    """Regression: `if nexts.size == 0 or ...` let a stale/bad planned route
    force a car directly onto its next planned edge even when that edge is
    NOT a real graph successor of the car's current edge — a teleport onto
    disconnected geometry. A mismatched plan must be abandoned instead."""
    from idm import IDMParams, idm_tick
    paths = make_synthetic_paths(n_paths=4, length=20.0)   # short: crosses fast
    anim = make_car_anim(1, paths, seed=0)
    anim["edge_idx"] = np.array([0], dtype=np.int64)
    anim["dist"]     = np.array([19.9])       # about to cross path 0's end
    anim["speed"]    = np.array([8.0])
    # path 0's REAL successor is path 1 (make_next_edges' ring), but the plan
    # claims the next hop is path 3 — not reachable from path 0.
    anim["planned_edges"]  = [np.array([0, 3], dtype=np.int64)]
    anim["planned_cursor"] = np.array([0], dtype=np.int64)
    nxt = make_next_edges(paths)   # ring: 0->1->2->3->0

    rng = np.random.default_rng(0)
    for _ in range(10):
        idm_tick(anim, paths, nxt, {}, dt=0.05, params=IDMParams(), rng=rng)

    assert int(anim["edge_idx"][0]) != 3, "car jumped straight onto a non-adjacent planned edge"
    assert int(anim["edge_idx"][0]) == 1, "car should fall back to its real graph successor (path 1)"
    assert anim["planned_edges"][0] is None, "the unreachable plan should have been abandoned"


@test
def idm_deadlock_exemption_covers_spillback_one_edge_upstream():
    """Regression: the red-light deadlock-teleport exemption used to check
    only the car's OWN edge. A car queued on the edge just upstream of a red
    light (a real spillback, not a deadlock) is legitimately stopped and must
    never be randomly teleported either."""
    from idm import IDMParams, idm_tick

    class _FakeLight:
        controlled_paths = frozenset({1})
        def can_enter(self, path_idx):
            return False   # permanently red for this test

    # path_b is short enough (3 m) that ANY position on it is within IDM's
    # braking zone for the red light at its far end — so a single car placed
    # right at its start (spilling back from the light) stays physically
    # pinned there for the whole test, a stable stand-in for "the queue has
    # backed up all the way to this edge's entrance" without needing to model
    # a full multi-car queue.
    path_a = {"u": 0, "v": 1, "length": 100.0, "maxspeed_ms": 10.0,
              "points": np.array([[0.0, 0.0, 0.5], [100.0, 0.0, 0.5]]),
              "cum_len": np.array([0.0, 100.0])}
    path_b = {"u": 1, "v": 2, "length": 3.0, "maxspeed_ms": 10.0,
              "points": np.array([[100.0, 0.0, 0.5], [103.0, 0.0, 0.5]]),
              "cum_len": np.array([0.0, 3.0])}
    paths = [path_a, path_b]
    nxt = [np.array([1], dtype=np.int64), np.empty(0, dtype=np.int64)]
    lights = {2: _FakeLight()}   # sits at the end of path_b (node 2), red

    anim = make_car_anim(2, paths, seed=1)
    anim["edge_idx"] = np.array([1, 0], dtype=np.int64)   # car0=on the red edge itself
    anim["dist"]     = np.array([0.5, 99.0])              # car1: 1 m from pathA's end
    anim["speed"]    = np.array([0.0, 0.0])

    rng = np.random.default_rng(0)
    for _ in range(400):   # 400 * 0.05s = 20s, well past the 8s teleport threshold
        idm_tick(anim, paths, nxt, lights, dt=0.05, params=IDMParams(), rng=rng)

    assert int(anim["edge_idx"][1]) == 0, (
        "car queued one edge upstream of a red light was teleported — "
        "spillback is not a deadlock"
    )
    assert float(anim.get("stuck_time", [0, 0])[1]) < 8.0


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


@test
def validation_tt_cong_falls_back_to_freeflow_not_implicit_one_second():
    """Regression: nx.shortest_path_length(weight="tt_cong") treats a MISSING
    edge attribute as cost 1 SECOND (networkx's default), not free-flow time.
    validate_congested only wrote "tt_cong" onto edges covered by car_paths —
    every other real street edge (e.g. anything the drivable-path extractor
    filtered out) was an exploitable "free" 1-second edge for shortest-path
    routing, making congested travel times far too low. Every edge must come
    out of validate_congested with a real tt_cong, seeded from free-flow "tt"."""
    import networkx as nx
    import validation as V

    g = nx.MultiDiGraph()
    g.graph["proj_str"] = "+proj=tmerc +lat_0=38 +lon_0=23.7 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
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

    all_edges = list(g.edges(keys=True))
    # Deliberately cover only HALF the edges with car_paths, like a real
    # drivable-path extractor that drops footways/filtered edges.
    car_paths = [{"u": u, "v": v, "length": 300.0, "maxspeed_ms": 10.0}
                 for u, v, k in all_edges[: len(all_edges) // 2]]
    car_anim = {"edge_idx": np.arange(len(car_paths), dtype=np.int64),
                "speed": np.full(len(car_paths), 6.0)}

    orig = V.osrm_duration
    V.osrm_duration = lambda o, d, host=None, timeout=None: 120.0
    try:
        V.validate_congested(g, car_anim, car_paths, n_pairs=5, seed=7)
    finally:
        V.osrm_duration = orig

    for u, v, k, data in g.edges(keys=True, data=True):
        assert "tt_cong" in data, f"edge ({u},{v},{k}) has no tt_cong — exploitable as a free 1s edge"
        # A 300 m edge at any plausible urban speed (2.8-41.7 m/s per
        # _edge_speed_ms's clip) takes at least 7s — nowhere near the
        # networkx implicit-missing-weight default of 1.
        assert data["tt_cong"] > 5.0, f"edge ({u},{v},{k}) tt_cong={data['tt_cong']} looks like the 1s default, not free-flow"


# ══════════════════════════════════════════════════════════════════════════════
# Demand model
# ══════════════════════════════════════════════════════════════════════════════

@test
def route_stats_overlay_called_once_with_full_summary():
    """Regression: _update_route_stats_overlay / _assign_selected_car_route /
    _update_pareto_chart used to be indented INSIDE the per-route-spec loop in
    _compute_and_render_routes, so they ran 3x on a partially-filled summary
    dict instead of once with all three routes (energy/joint/shortest)."""
    import types
    import networkx as nx
    from route_mixin import RouteMixin

    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=50.0, y=0.0)
    g.add_node(2, x=100.0, y=0.0)
    g.add_edge(0, 1, key=0, length=50.0, maxspeed_ms=10.0)
    g.add_edge(1, 2, key=0, length=50.0, maxspeed_ms=10.0)
    car_paths = [
        {"u": 0, "v": 1, "length": 50.0, "maxspeed_ms": 10.0},
        {"u": 1, "v": 2, "length": 50.0, "maxspeed_ms": 10.0},
    ]

    class _FakePlotter:
        def add_lines(self, *a, **kw):
            return object()
        def remove_actor(self, *a, **kw):
            pass

    host = types.SimpleNamespace(
        street_graph=g, car_paths=car_paths,
        scene_state={"scene_lat": 33.9, "scene_lon": 35.5, "hour": 12.0,
                     "route_alpha": 0.5, "route_hour": 12.0, "solar_fleet": False},
        route_state={"route_actors": [], "selected_car_idx": None},
        plotter=_FakePlotter(),
    )
    for name in ("_compute_and_render_routes", "_clear_route_actors", "_node_xy",
                 "_edge_best", "_route_polyline", "_route_stats", "_path_time_energy",
                 "_update_pareto_chart", "_remove_pareto_chart", "_build_edge_shadow_cache"):
        if hasattr(RouteMixin, name):
            setattr(host, name, types.MethodType(getattr(RouteMixin, name), host))

    calls = []
    host._update_route_stats_overlay = lambda summary: calls.append(dict(summary))

    host._compute_and_render_routes(0, 2)

    assert len(calls) == 1, f"_update_route_stats_overlay called {len(calls)}x, expected exactly once"
    assert set(calls[0].keys()) == {"energy", "joint", "shortest"}, (
        f"summary passed was incomplete: {calls[0].keys()}"
    )
    for name, stats in calls[0].items():
        assert stats["distance_m"] > 0, f"{name} route has zero distance — routing failed upstream"


@test
def survey_prep_downsamples_elevation_and_ortho_correctly():
    """render.survey_prep on tiny synthetic strip-organised GeoTIFFs with a
    known answer: NaN-aware block means, a non-divisible trailing row band,
    alpha-weighted RGB, coverage mask, and preserved georeferencing."""
    import tempfile
    from pathlib import Path
    import rasterio
    from rasterio.transform import from_origin
    from render.survey_prep import downsample_elevation, downsample_ortho

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        H, W = 23, 20          # 23 % 5 != 0 -> exercises the trailing padded band
        tr = from_origin(733000.0, 3753000.0, 0.05, 0.05)
        dem = np.arange(H * W, dtype=np.float32).reshape(H, W)
        dem[0:5, 0:5] = np.nan          # one fully-missing output cell
        dem[5, 5] = np.nan              # one partially-missing output cell
        prof = dict(driver="GTiff", width=W, height=H, crs="EPSG:32636", transform=tr,
                    tiled=False, blockysize=4)   # strip-organised like the real survey
        with rasterio.open(td / "dtm.tif", "w", count=1, dtype="float32", **prof) as d:
            d.write(dem[None])
        downsample_elevation(td / "dtm.tif", td / "dtm_out.tif", factor=5)
        with rasterio.open(td / "dtm_out.tif") as d:
            out = d.read(1)
            assert out.shape == (5, 4), out.shape
            assert np.isclose(d.transform.a, 0.25) and np.isclose(d.transform.c, 733000.0) \
                and np.isclose(d.transform.f, 3753000.0), d.transform
        assert np.isnan(out[0, 0]), "all-NaN block must stay NaN"
        blk = dem[5:10, 5:10]
        assert np.isclose(out[1, 1], np.nanmean(blk)), "NaN-aware mean wrong"
        assert np.isclose(out[4, 0], np.nanmean(dem[20:23, 0:5])), "trailing partial band wrong"

        rgba = np.zeros((4, 8, 8), dtype=np.uint8)
        rgba[0], rgba[1], rgba[2] = 200, 100, 50
        rgba[3, :, :] = 255
        rgba[:3, 0, 0] = 0; rgba[3, 0, 0] = 0          # one transparent pixel in block (0,0)
        rgba[3, 6:8, 6:8] = 0                            # fully transparent block (3,3)
        with rasterio.open(td / "rgb.tif", "w", count=4, dtype="uint8",
                           **{**prof, "width": 8, "height": 8}) as d:
            d.write(rgba)
        downsample_ortho(td / "rgb.tif", td / "rgb_out.tif", factor=2, quality=100)
        with rasterio.open(td / "rgb_out.tif") as d:
            rgb = d.read()
            mask = d.dataset_mask()
        assert rgb.shape == (3, 4, 4)
        assert abs(int(rgb[0, 0, 0]) - 200) <= 3, "transparent pixel leaked into the average (alpha weighting)"
        assert mask[0, 0] == 255 and mask[3, 3] == 0, "coverage mask wrong"


@test
def survey_scene_fuses_dtm_with_correct_datum_edge_feather_and_north_up_texture():
    """render.survey_ground on a synthetic UTM survey around Beirut: the fused
    sampler returns DTM - geoidN inside coverage, the base DEM outside, has no
    cliff at the coverage edge, and the ortho texture comes out north-up."""
    import json as _json
    import tempfile
    from pathlib import Path
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    import render.survey_ground as sg

    lat0, lon0 = 33.8944, 35.5227
    local_crs = (f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
                 f"+datum=WGS84 +units=m +no_defs")
    e0, n0 = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True).transform(lon0, lat0)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        res, W, H = 0.5, 400, 400                       # 200 m x 200 m survey, centred
        left, top = e0 - 100.0, n0 + 100.0
        tr = from_origin(left, top, res, res)
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        east = left + (cols + 0.5) * res
        dtm = (50.0 + 0.1 * (east - e0)).astype(np.float32)   # plane rising eastward
        dtm[:, : W // 4] = np.nan                               # western quarter uncovered
        prof = dict(driver="GTiff", width=W, height=H, crs="EPSG:32636", transform=tr)
        with rasterio.open(td / "dtm.tif", "w", count=1, dtype="float32", nodata=np.nan, **prof) as d:
            d.write(dtm[None])
        rgb = np.zeros((3, H, W), dtype=np.uint8)
        rgb[0, : H // 2, W // 2 :] = 255                        # NORTH-EAST quadrant red
        rgb[2, H // 2 :, : W // 2] = 255                        # south-west quadrant blue
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(td / "rgb.tif", "w", count=3, dtype="uint8", **prof) as d:
                d.write(rgb)
                d.write_mask(np.full((H, W), 255, dtype=np.uint8))
        (td / "survey_meta.json").write_text(_json.dumps({
            "sources": {"x": 1}, "crs": "EPSG:32636", "ortho_res_m": 0.5,
            "outputs": {"dtm": str(td / "dtm.tif"), "rgb": str(td / "rgb.tif")}}))

        base = lambda xy: np.full(np.asarray(xy).shape[0], 7.0)   # "Copernicus"
        orig_geoid = sg.geoid_undulation
        sg.geoid_undulation = lambda lon, lat: 20.0
        try:
            scene = sg.load_survey_scene(local_crs, (-150.0, 150.0, -150.0, 150.0), base,
                                         cache_dir=td, elev_res=0.5, tex_max_px=600, survey_dir=td)
        finally:
            sg.geoid_undulation = orig_geoid
        assert scene is not None
        # covered survey = 150 m x 200 m (east 3/4 of 200x200) over a 300x300 scene
        assert abs(scene.stats["coverage_frac"] - (150 * 200) / (300 * 300)) < 0.02, scene.stats

        # Well inside coverage (>> feather width from its edge): exact datum shift.
        z = scene.sampler(np.array([[40.0, 0.0]]))[0]
        e_at, _ = Transformer.from_crs(local_crs, "EPSG:32636", always_xy=True).transform(40.0, 0.0)
        expected = 50.0 + 0.1 * (e_at - e0) - 20.0
        assert abs(z - expected) < 0.05, f"datum-corrected DTM {z:.3f} != expected {expected:.3f}"
        # Outside survey footprint entirely -> base DEM.
        assert abs(scene.sampler(np.array([[140.0, 140.0]]))[0] - 7.0) < 1e-6
        # No cliff: along a west->east transect across the coverage edge the
        # height changes smoothly (feathered), never by metres in one 0.5 m step.
        xs = np.linspace(-60.0, 0.0, 241)
        prof_z = scene.sampler(np.column_stack([xs, np.zeros_like(xs)]))
        assert np.max(np.abs(np.diff(prof_z))) < 1.0, f"cliff at coverage edge: {np.max(np.abs(np.diff(prof_z))):.2f} m"

        tex = scene.texture
        h, w = tex.shape[:2]
        tx0, tx1, ty0, ty1 = scene.tex_extent
        def px(x, y):   # local (x, y) -> texture pixel, row 0 = north
            return tex[int((ty1 - y) / (ty1 - ty0) * h), int((x - tx0) / (tx1 - tx0) * w)]
        assert px(50.0, 50.0)[0] > 200 and px(50.0, 50.0)[2] < 50, f"NE should be red, got {px(50.0, 50.0)}"
        assert px(-50.0, -50.0)[2] > 200 and px(-50.0, -50.0)[0] < 50, f"SW should be blue, got {px(-50.0, -50.0)}"
        assert not scene.texture_mask[int((ty1 - 140.0) / (ty1 - ty0) * h), int((140.0 - tx0) / (tx1 - tx0) * w)], \
            "texture mask should be False outside the survey footprint"

        ground = sg.build_photoreal_ground(scene, mesh_res=5.0)
        assert ground is not None
        mesh, texobj = ground
        tc = np.asarray(mesh.active_texture_coordinates)
        assert tc.shape[0] == mesh.n_points and tc.min() >= -1e-6 and tc.max() <= 1 + 1e-6
        assert mesh.bounds[1] - mesh.bounds[0] < 300.0, "uncovered cells were not dropped"

        # RENDERED orientation (regression: a manual [::-1] on top of
        # pv.Texture's own row flip mirrored the whole ortho north<->south).
        import pyvista as pv
        pl = pv.Plotter(off_screen=True, window_size=(400, 400))
        pl.set_background("white")
        pl.add_mesh(mesh, texture=texobj, lighting=False)
        pl.camera_position = [(0.0, 0.0, 800.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]   # north up
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = 150.0
        pl.render()
        img = pl.screenshot(return_img=True)
        pl.close()
        def at(x, y):   # local metres -> screen pixel (parallel projection, 300 m across)
            return img[int(200 - y / 150.0 * 200), int(200 + x / 150.0 * 200)]
        ne, sw = at(50.0, 50.0), at(-40.0, -50.0)
        assert ne[0] > 180 and ne[2] < 80, f"rendered NE should be red (north-up), got {ne}"
        assert sw[2] > 180 and sw[0] < 80, f"rendered SW should be blue, got {sw}"


@test
def building_reconcile_decides_present_absent_containers_trees_and_no_survey():
    """render.building_reconstruct on a synthetic nDSM with known truth."""
    import pyvista as pv
    from editor_ops import make_building_mesh
    from render.survey_ground import SurveyScene
    from render.building_reconstruct import reconcile_buildings

    dx, half = 0.5, 100.0
    n = int(2 * half / dx) + 1
    xs = -half + np.arange(n) * dx
    X, Y = np.meshgrid(xs, xs)
    ndsm = np.zeros((n, n), dtype=np.float32)
    def box(cx, cy, w, d):
        return (np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= d / 2)
    ndsm[box(-48.0, -50.0, 20, 20)] = 14.0           # A: real 14 m roof, 2 m EAST of its footprint
    yard = box(50.0, -50.0, 40, 40)                   # C: container rows 2.4 m wide, 3 m aisles, stacked 5.2 m
    rows_mask = ((X - 30.0) % 5.4) < 2.4
    ndsm[yard & rows_mask] = 5.2
    ndsm[box(-50.0, 50.0, 15, 15)] = 8.0             # D: tree canopy (green in the ortho)
    ndsm[box(80.0, 80.0, 30, 30)] = np.nan           # E: outside survey coverage
    tex = np.full((n, n, 3), 128, dtype=np.uint8)     # neutral grey everywhere...
    tex[::-1][box(-50.0, 50.0, 15, 15)] = (40, 150, 40)   # ...green canopy (texture rows are north-up)
    scene = SurveyScene(x0=-half, y0=-half, dx=dx, elev=np.zeros_like(ndsm), coverage=np.isfinite(ndsm),
                        geoid_n=0.0, datum_source="", base_sampler=None, texture=tex,
                        texture_mask=np.ones((n, n), bool), tex_extent=(-half, half + dx, -half, half + dx),
                        ndsm=ndsm)

    specs = {"A": (-50.0, -50.0, 20, 20), "B": (50.0, 50.0, 20, 20), "C": (50.0, -50.0, 40, 40),
             "D": (-50.0, 50.0, 15, 15), "E": (80.0, 80.0, 20, 20)}
    parts = []
    for k, (cx, cy, w, d) in specs.items():
        m = make_building_mesh(cx, cy, width=w, depth=d, height=10.0).triangulate()
        m.cell_data["building_class"] = np.full(m.n_cells, 4, dtype=np.uint8)
        parts.append(m)
    mesh = parts[0]
    for m in parts[1:]:
        mesh = mesh.merge(m, merge_points=False)

    out, rep = reconcile_buildings(mesh, scene)
    status = {}
    for b in rep.buildings:
        name = min(specs, key=lambda k: (specs[k][0] - b.centroid[0]) ** 2 + (specs[k][1] - b.centroid[1]) ** 2)
        status[name] = b
    assert status["A"].status == "present", status["A"]
    assert abs(status["A"].new_height - 14.0) < 0.5, status["A"].new_height
    assert abs(status["A"].relocated_by[0] - 2.0) <= 0.5 and abs(status["A"].relocated_by[1]) <= 0.5, \
        f"footprint A not co-registered onto its roof: {status['A'].relocated_by}"
    assert status["B"].status == "absent", status["B"]
    assert status["C"].status == "absent" and status["C"].occupied_frac > 0.35, \
        f"container yard must be rejected on continuity despite occupancy: {status['C']}"
    assert status["D"].status == "absent", f"tree canopy mistaken for a roof: {status['D']}"
    assert status["E"].status == "no_survey" and status["E"].new_height == status["E"].old_height

    assert "building_class" in out.cell_data and out.n_cells > 0
    zmax = float(out.points[:, 2].max())
    assert abs(zmax - 14.0) < 0.5 or abs(zmax - 10.0) < 1e-6, zmax
    kept = out.connectivity()
    assert int(np.asarray(kept.cell_data["RegionId"]).max()) + 1 == 2, "expected A and E only"


@test
def survey_structures_extract_buildings_and_reject_bridges_walls_smear_and_known_footprints():
    """render.survey_structures on a synthetic nDSM with known truth: stepped
    massing splits by height band; a footbridge deck crossing a road, a thin
    wall, a smeared (textureless) blob and an already-modelled footprint are
    not extruded."""
    from shapely.geometry import LineString, box as sbox
    from render.survey_ground import SurveyScene
    from render.survey_structures import extract_structures

    dx, half = 0.5, 100.0
    n = int(2 * half / dx) + 1
    xs = -half + np.arange(n) * dx
    X, Y = np.meshgrid(xs, xs)
    def box(cx, cy, w, d):
        return (np.abs(X - cx) <= w / 2) & (np.abs(Y - cy) <= d / 2)
    specs = {
        "low":    (-60.0, -60.0, 12, 12, 9.0),   # stepped block: 9 m ...
        "high":   (-48.0, -60.0, 12, 12, 18.0),  # ... abutting 18 m, must stay a separate volume
        "bridge": (40.0, -60.0, 6, 30, 6.0),     # footbridge deck across the E-W road at y=-60
        "wall":   (-50.0, 40.0, 30, 2, 4.0),     # hoarding / boundary wall
        "smear":  (40.0, 40.0, 15, 15, 8.0),     # photogrammetry failure: no image detail
        "known":  (0.0, 60.0, 12, 12, 10.0),     # already in buildings_mesh
    }
    ndsm = np.zeros((n, n), dtype=np.float32)
    for cx, cy, w, d, h in specs.values():
        ndsm[box(cx, cy, w, d)] = h
    rng = np.random.default_rng(3)
    g = rng.integers(40, 220, size=(n, n), dtype=np.uint8)        # detailed (sharp) imagery
    tex = np.repeat(g[:, :, None], 3, axis=2)
    cx, cy, w, d, _ = specs["smear"]
    tex[::-1][box(cx, cy, w + 4, d + 4)] = 128                    # smeared: flat grey (rows north-up)
    scene = SurveyScene(x0=-half, y0=-half, dx=dx, elev=np.zeros_like(ndsm), coverage=np.ones((n, n), bool),
                        geoid_n=0.0, datum_source="", base_sampler=None, texture=tex,
                        texture_mask=np.ones((n, n), bool), tex_extent=(-half, half + dx, -half, half + dx),
                        ndsm=ndsm)
    road = LineString([(20.0, -60.0), (half, -60.0)])   # E-W road under the deck, clear of the block
    kx, ky, kw, kd, _ = specs["known"]
    known_fp = sbox(kx - kw / 2, ky - kd / 2, kx + kw / 2, ky + kd / 2)

    def found(roofs):
        cc = roofs.cell_centers().points
        h = np.asarray(roofs.cell_data["height"])
        out = {}
        for k, (sx, sy, sw, sd, _) in specs.items():
            inside = (np.abs(cc[:, 0] - sx) <= sw / 2) & (np.abs(cc[:, 1] - sy) <= sd / 2)
            out[k] = h[inside]
        return out

    walls, roofs = extract_structures(scene, exclude_footprints=[known_fp], road_lines=[road])
    f = found(roofs)
    assert f["low"].size and np.allclose(f["low"], 9.0, atol=0.5), f"9 m block: {f['low'][:5]}"
    assert f["high"].size and np.allclose(f["high"], 18.0, atol=0.5), \
        f"18 m block must be its own height-band volume: {f['high'][:5]}"
    for k in ("bridge", "wall", "smear", "known"):
        assert f[k].size == 0, f"'{k}' must not be extruded (got {f[k].size} roof faces)"
    tc = np.asarray(roofs.active_texture_coordinates)
    assert tc.min() >= -1e-6 and tc.max() <= 1 + 1e-6
    assert walls.n_cells > 0 and float(walls.points[:, 2].max()) <= 18.0 + 1e-6

    # Without the road network the deck is indistinguishable from a building:
    # proves the rejection above comes from the road-crossing test.
    _, roofs_nr = extract_structures(scene, exclude_footprints=[known_fp], road_lines=[])
    assert found(roofs_nr)["bridge"].size > 0, "control: bridge deck should extrude without road lines"


@test
def survey_edge_degradation_masks_smeared_boundary_keeps_water_and_smooth_interior():
    """render.survey_ground._edge_degradation_mask: low-detail imagery that is
    connected to the survey no-data boundary is degraded (smear), unless it is
    saturated water; smooth-but-valid interior surfaces are kept."""
    import render.survey_ground as sg

    res, n = 0.25, 800                                   # 200 m square
    rng = np.random.default_rng(5)
    g = rng.integers(40, 220, size=(n, n), dtype=np.uint8)
    tex = np.repeat(g[:, :, None], 3, axis=2)
    tmask = np.ones((n, n), bool)
    tmask[:, 700:] = False                               # east 25 m: outside the flight
    tex[~tmask] = 0
    rows = lambda y0, y1: slice(int(y0 / res), int(y1 / res))
    cols = lambda x0, x1: slice(int(x0 / res), int(x1 / res))
    tex[rows(20, 60), cols(135, 175)] = (117, 135, 147)  # desaturated blue-grey smear at the edge
    tex[rows(120, 160), cols(135, 175)] = (40, 150, 150) # turquoise basin water at the edge
    tex[rows(80, 110), cols(40, 70)] = (90, 90, 92)      # smooth asphalt, interior
    bad = sg._edge_degradation_mask(tex, tmask, res)
    frac = lambda r, c: float(bad[r, c].mean())
    assert frac(rows(25, 55), cols(140, 170)) > 0.95, f"edge smear not masked: {frac(rows(25, 55), cols(140, 170)):.2f}"
    assert frac(rows(125, 155), cols(140, 170)) < 0.05, "saturated water must not be masked"
    assert frac(rows(85, 105), cols(45, 65)) < 0.05, "smooth interior surface must not be masked"
    assert frac(rows(0, 200), cols(0, 100)) < 0.01, "detailed imagery must not be masked"
    assert not bad[~tmask].any(), "mask must stay inside the ortho coverage"


@test
def physical_sky_is_blue_reddens_at_low_sun_and_renders_in_scene_orientation():
    """render.sky: Rayleigh+Mie single scattering gives a blue zenith, a
    reddened low sun and a dark night; RENDERED through VTK (skybox and
    mirror-sphere IBL) the sun-side glow appears on the correct compass side
    in the scene frame (regression guard for VTK's mirrored equirect lookup)."""
    import pyvista as pv
    from render.sky import (sky_environment, sun_transmittance, configure_environment_basis,
                            environment_texture, make_skybox)

    from render.sky import sky_radiance
    sun = np.array([0.2, 0.0, 0.98]) / np.linalg.norm([0.2, 0.0, 0.98])
    away = np.array([[0.0, 0.7071, 0.7071]])              # 45 deg up, ~90 deg from the sun
    near = np.array([[0.2588, 0.0, 0.9659]])              # ~4 deg from the sun (aerosol aureole)
    blue, aureole = sky_radiance(away, sun)[0], sky_radiance(near, sun)[0]
    assert blue[2] > blue[1] > blue[0], f"sky away from the sun should be blue-dominant: {blue}"
    assert aureole.sum() > 3 * blue.sum() and aureole[2] / aureole[0] < blue[2] / blue[0], \
        f"aureole should be brighter and whiter than the open sky: {aureole} vs {blue}"
    horizon = sky_radiance(np.array([[0.0, -0.996, 0.087]]), sun)[0]
    assert horizon.sum() > blue.sum(), f"hazy horizon should be brighter than the sky at 45 deg: {horizon} vs {blue}"
    low = np.array([np.cos(np.radians(8.0)), 0.0, np.sin(np.radians(8.0))])
    t = sun_transmittance(low)
    assert t[0] > t[1] > t[2] and t[2] < 0.5 * t[0], f"low sun should redden: {t}"
    night = sky_environment(np.array([0.0, 0.3, -0.95]), 64, 32)
    assert float(night[:16].max()) < 0.01, "night sky should be dark"

    env = sky_environment(low, 256, 128)             # low sun in the EAST (+x)
    tex = environment_texture(env)

    def lum(img):
        return float(np.asarray(img[45:55, 45:55], float).mean())

    p = pv.Plotter(off_screen=True, window_size=(100, 100))
    configure_environment_basis(p.renderer)
    p.renderer.AddActor(make_skybox(tex))
    seen = {}
    for name, v in {"east": (1, 0, 0.12), "west": (-1, 0, 0.12)}.items():
        p.camera.position = (0, 0, 0); p.camera.focal_point = v; p.camera.up = (0, 0, 1)
        p.render(); seen[name] = lum(p.screenshot(return_img=True))
    p.close()
    assert seen["east"] > 1.2 * seen["west"], f"skybox: sun glow must be in the east {seen}"

    p = pv.Plotter(off_screen=True, window_size=(100, 100))
    p.set_environment_texture(tex, is_srgb=False)
    configure_environment_basis(p.renderer)
    p.remove_all_lights()
    p.add_mesh(pv.Sphere(radius=1.0, theta_resolution=90, phi_resolution=90),
               pbr=True, metallic=1.0, roughness=0.05, color="white")
    refl = {}
    for name, cam in {"east": (-6, 0, 0.7), "west": (6, 0, 0.7)}.items():
        # the sphere centre reflects the direction back toward the camera:
        # a camera in the WEST sees the EASTERN sky reflected
        p.camera.position = cam; p.camera.focal_point = (0, 0, 0); p.camera.up = (0, 0, 1)
        p.camera.view_angle = 8
        p.render(); refl["west" if name == "east" else "east"] = lum(p.screenshot(return_img=True))
    p.close()
    assert refl["east"] > 1.2 * refl["west"], f"IBL: reflected sun glow must come from the east {refl}"


@test
def photoreal_ground_drape_stays_above_coarse_draped_stylized_layers():
    """Regression: over concave terrain the coarse stylized ground polygons
    (draped at their vertices only) are chords that rose above the finely
    draped photo and showed through as dark patches. The photo drape must
    stay above them, and remain a plain DEM drape where they are absent."""
    import pyvista as pv
    from render.survey_mixin import SurveyMixin, _PHOTO_CLEARANCE, _PHOTO_DRAPE_BIAS

    bowl = lambda xy: 0.002 * (np.asarray(xy)[:, 0] ** 2 + np.asarray(xy)[:, 1] ** 2)   # concave DEM
    ground = pv.PolyData(np.array([[-50., -50, 0.15], [50, -50, 0.15], [50, 50, 0.15], [-50, 50, 0.15]]),
                         np.array([3, 0, 1, 2, 3, 0, 2, 3]))
    ground.points[:, 2] += bowl(ground.points[:, :2]) + 0.3      # draped at vertices only -> chord at 10 m
    photo = pv.Plane(center=(0, 0, 0.18), i_size=120, j_size=120, i_resolution=60, j_resolution=60)

    class _Host(SurveyMixin):
        pass
    host = _Host()
    host.plotter = pv.Plotter(off_screen=True)
    host.ground_mesh = ground
    host.scene_state = {"survey_ground_actor": host.plotter.add_mesh(photo)}
    host._drape_survey_ground(bowl)
    pd = pv.wrap(host.scene_state["survey_ground_actor"].GetMapper().GetInputDataObject(0, 0))
    z = np.asarray(pd.points[:, 2])
    xy = np.asarray(pd.points[:, :2])
    inside = (np.abs(xy[:, 0]) < 49) & (np.abs(xy[:, 1]) < 49)
    chord = 0.002 * 2 * 50 ** 2 + 0.45                              # flat chord height of the draped quad
    assert np.all(z[inside] >= chord + _PHOTO_CLEARANCE - 1e-6), \
        f"photo dips below the stylized chord by {float((chord + _PHOTO_CLEARANCE - z[inside]).max()):.2f} m"
    outside = (np.abs(xy[:, 0]) > 51) | (np.abs(xy[:, 1]) > 51)
    assert np.allclose(z[outside], 0.18 + bowl(xy[outside]) + _PHOTO_DRAPE_BIAS, atol=1e-6), \
        "outside the stylized layers the photo must be a plain DEM drape"
    assert np.allclose(host.scene_state["_orig_survey_ground_actor_z"], 0.18), "original z not stored for restore"
    host.plotter.close()


@test
def scene_materials_migrate_lit_actors_to_pbr_preserving_display_colours():
    """render.materials: lit Phong surfaces become PBR, and under the single
    exposure a fully lit face still shows the colour the call site asked for
    — for property colours, runtime SetColor changes and direct RGB scalars.
    Unlit actors and plain lines are untouched; app light resets re-expose."""
    import pyvista as pv
    from render.materials import SceneMaterials, PBR_EXPOSURE

    C, C2 = (0.8, 0.4, 0.2), (0.2, 0.5, 0.8)
    p = pv.Plotter(off_screen=True, window_size=(120, 40))
    p.set_background("black")
    p.remove_all_lights()
    light = pv.Light(position=(0, 0, 10), focal_point=(0, 0, 0), light_type="scene light", intensity=1.0)
    p.add_light(light)
    lit = p.add_mesh(pv.Plane(center=(-2, 0, 0), i_size=1.9, j_size=1.9), color=C, specular=0.0)
    m = pv.Plane(center=(0, 0, 0), i_size=1.9, j_size=1.9)
    m.point_data["rgb"] = np.tile(np.array([51, 204, 102], np.uint8), (m.n_points, 1))
    p.add_mesh(m, scalars="rgb", rgb=True)
    unlit = p.add_mesh(pv.Plane(center=(2, 0, 0), i_size=1.9, j_size=1.9), color=C, lighting=False)
    line = p.add_mesh(pv.Line((-3, 0.99, 0.1), (3, 0.99, 0.1)), color=C2, line_width=2)
    p.camera.parallel_projection = True
    p.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
    p.camera.parallel_scale = 1.0
    mats = SceneMaterials(p.renderer)
    mats.attach()
    p.render()
    img = p.screenshot(return_img=True)
    px = lambda x: img[20, int(60 + x / 3.0 * 60)].astype(int)
    target = lambda c: (np.array(c) * 255).astype(int)
    assert lit.GetProperty().GetInterpolationAsString() == "Physically based rendering"
    assert np.abs(px(-2) - target(C)).max() <= 16, f"migrated colour {px(-2)} vs asked {target(C)}"
    assert np.abs(px(0) - np.array([51, 204, 102])).max() <= 16, f"direct scalars {px(0)}"
    assert np.abs(px(2) - target(C)).max() <= 2 and not unlit.GetProperty().GetInterpolationAsString().startswith("Phys"), \
        "unlit actor must be untouched"
    assert line.GetProperty().GetInterpolationAsString() != "Physically based rendering", "plain lines stay legacy"
    assert abs(light.GetIntensity() - PBR_EXPOSURE) < 1e-6
    lit.GetProperty().SetColor(*C2)                       # app changes a colour at runtime
    light.SetIntensity(1.0)                               # app resets a light (legacy units)
    p.render()
    img = p.screenshot(return_img=True)
    assert np.abs(px(-2) - target(C2)).max() <= 16, f"runtime colour change {px(-2)} vs {target(C2)}"
    assert abs(light.GetIntensity() - PBR_EXPOSURE) < 1e-6, "reset light must be re-exposed"
    p.close()


@test
def instanced_fleet_draws_one_call_per_model_with_per_car_colour_and_motion():
    """render.instanced_cars: N cars of M models -> M instanced actors; each
    car shows its own colour at its own position; update() moves them and
    set_z() lifts them (terrain drape), with no new VTK actors."""
    import pyvista as pv
    from render.instanced_cars import InstancedFleet

    box = pv.Box(bounds=(-0.8, 0.8, -0.8, 0.8, 0.0, 0.5))
    wedge = pv.Cone(center=(0, 0, 0.3), direction=(1, 0, 0), height=1.6, radius=0.8, resolution=24)
    p = pv.Plotter(off_screen=True, window_size=(200, 80))
    p.set_background("black")
    pos = np.array([[-6.0, 0, 0], [-2.0, 0, 0], [2.0, 0, 0], [6.0, 0, 0]])
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    fleet = InstancedFleet(p, [box, wedge], [0, 1, 0, 1], cols, pos, np.zeros(4))
    assert len(fleet.actors) == 2, "one instanced draw per model"
    for a in fleet.actors:
        a.GetProperty().SetLighting(False)
    n_actors = p.renderer.GetActors().GetNumberOfItems()
    p.camera.parallel_projection = True
    p.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
    p.camera.parallel_scale = 4.0
    p.render()
    img = p.screenshot(return_img=True)
    at = lambda x, y=0.0: img[int(40 - y / 4.0 * 40), int(100 + x / 10.0 * 100)].astype(int)
    for (x, _, _), c in zip(pos, cols):
        assert np.abs(at(x) - np.array(c)).max() < 40, f"car at x={x}: {at(x)} vs {c}"
    fleet.update(pos + np.array([0.0, 2.5, 0.0]), np.zeros(4))
    p.render()
    img = p.screenshot(return_img=True)
    assert np.abs(at(-6.0, 2.5) - np.array(cols[0])).max() < 40 and at(-6.0).max() < 40, "update() must move cars"
    fleet.set_z(np.full(4, 3.0))
    assert np.allclose(fleet.positions[:, 2], 3.0)
    assert p.renderer.GetActors().GetNumberOfItems() == n_actors, "no actors created per tick"
    fleet.remove()
    assert p.renderer.GetActors().GetNumberOfItems() == n_actors - 2
    p.close()


@test
def procedural_facades_centre_bays_start_storeys_at_ground_and_light_windows_at_night():
    """render.facades: metric UVs put whole bays centred on each wall and
    storeys from ground level; rendered, glass is darker than the render by
    day, and lit windows glow at night while the wall stays dark."""
    import pyvista as pv
    from render.facades import STYLES, TILE_BAYS, TILE_STOREYS, facade_uvs, facade_textures, apply_facade

    st = STYLES[4]                                          # residential: 3.2 m bays, 3.1 m storeys
    L = 8 * st.bay_m + 1.0                                  # 8 bays + 0.5 m margin each side
    H = 8 * st.storey_m
    box = pv.Box(bounds=(0, L, 0, 6, 0, H)).triangulate().subdivide(2).compute_normals(
        split_vertices=True, point_normals=True, cell_normals=False)
    uv = facade_uvs(box, st, seed=1)
    tile_w, tile_h = TILE_BAYS * st.bay_m, TILE_STOREYS * st.storey_m
    pts = np.asarray(box.points)
    south = np.flatnonzero(np.isclose(pts[:, 1], 0.0) & (np.asarray(box.point_data["Normals"])[:, 1] < -0.9))
    u_m = uv[south, 0] * tile_w
    v_m = uv[south, 1] * tile_h
    # bay phase: the wall's first metre-mark sits at -margin modulo a bay
    phase = np.mod(u_m - pts[south, 0], st.bay_m)
    assert np.allclose(phase, np.mod(-0.5, st.bay_m), atol=1e-6), "bays not centred on the wall"
    r = np.mod(v_m - pts[south, 2], st.storey_m)
    assert np.allclose(np.minimum(r, st.storey_m - r), 0.0, atol=1e-6), "storeys must start at ground"

    tex = facade_textures(st, seed=1)
    shots = {}
    for tag, emis, light_on in (("day", 0.0, True), ("night", 1.0, False)):
        p = pv.Plotter(off_screen=True, window_size=(400, 400))
        p.set_background("black")
        p.remove_all_lights()
        # the app always has IBL (physical sky); VTK's PBR path without an
        # environment falls back to a bright default and drops emission
        from render.sky import environment_texture
        p.set_environment_texture(environment_texture(np.full((32, 64, 3), 0.002, np.float32)), is_srgb=False)
        if light_on:
            p.add_light(pv.Light(position=(L / 2, -100, H / 2), focal_point=(L / 2, 0, H / 2),
                                 light_type="scene light", intensity=np.pi))
        a = p.add_mesh(box.copy(), pbr=True, color="white")
        assert apply_facade(a, st, tex, seed=1)
        a.GetProperty().SetEmissiveFactor(emis, emis, emis)
        p.camera.parallel_projection = True
        p.camera_position = [(L / 2, -60, H / 2), (L / 2, 0, H / 2), (0, 0, 1)]
        p.camera.parallel_scale = H / 2
        p.render()
        shots[tag] = p.screenshot(return_img=True).astype(float)
        p.close()
    s = 400 / H                                            # px per metre (square view)
    def px(img, x, z):
        return img[int(200 - (z - H / 2) * s), int(200 + (x - L / 2) * s)].mean()
    x_win = 0.5 + 3 * st.bay_m + st.bay_m / 2              # centre of bay 3
    x_pier = 0.5 + 3 * st.bay_m + 0.15                     # wall between windows
    z_win = 2 * st.storey_m + st.sill_m + st.win_h / 2     # mid-window, storey 2
    day, night = shots["day"], shots["night"]
    assert px(day, x_win, z_win) < 0.6 * px(day, x_pier, z_win), \
        f"glass {px(day, x_win, z_win):.0f} should be darker than render {px(day, x_pier, z_win):.0f}"
    wins = [px(night, 0.5 + b * st.bay_m + st.bay_m / 2, k * st.storey_m + st.sill_m + st.win_h / 2)
            for b in range(8) for k in range(8)]
    assert max(wins) > 120 and px(night, x_pier, z_win) < 20, \
        f"night: lit windows max {max(wins):.0f}, wall {px(night, x_pier, z_win):.0f}"
    assert 0.1 < np.mean(np.array(wins) > 60) < 0.7, "a realistic share of windows should be lit"

    # Regression: a smooth cylinder (port tank) is one wall region with no
    # dominant direction; it was classed as roof, got constant UVs -> NaN
    # tangents -> rendered black. Curved walls now get arc-length UVs.
    cyl = pv.Cylinder(center=(0, 0, 10), direction=(0, 0, 1), radius=8, height=20, resolution=48).triangulate() \
        .compute_normals(split_vertices=True, point_normals=True, cell_normals=False)
    uvc = facade_uvs(cyl, STYLES[0])
    assert np.ptp(uvc[:, 0]) > 1.0, "curved wall must get along-wall UVs"
    p = pv.Plotter(off_screen=True, window_size=(300, 300))
    p.remove_all_lights()
    from render.sky import environment_texture
    p.set_environment_texture(environment_texture(np.full((32, 64, 3), 0.05, np.float32)), is_srgb=False)
    p.add_light(pv.Light(position=(0, -100, 40), focal_point=(0, 0, 10), light_type="scene light", intensity=np.pi))
    a = p.add_mesh(cyl, pbr=True, color="white")
    assert apply_facade(a, STYLES[0], facade_textures(STYLES[0]))
    p.camera_position = [(0, -60, 15), (0, 0, 10), (0, 0, 1)]
    p.render()
    tank = p.screenshot(return_img=True).astype(float)[100:200, 120:180].mean()
    p.close()
    assert tank > 60, f"curved facade renders black ({tank:.0f})"


@test
def overpass_outage_falls_back_to_mirrors_and_never_aborts_startup():
    """Regression: an overpass-api.de timeout in the OSM direction-graph
    fetch aborted scene startup ('Failed to fetch/build geometry'). Network
    failures now fall through the public mirrors; if all are down the
    direction graph degrades to None (callers use Overture one-ways)."""
    import osmnx as ox
    import requests
    import osm_net
    import turn_restrictions as tr

    tried = []
    def flaky():
        tried.append(ox.settings.overpass_url)
        if len(tried) < 3:
            raise requests.exceptions.ConnectTimeout("simulated")
        return "graph"
    orig = ox.settings.overpass_url
    osm_net._preferred = 0
    osm_net._down.clear()
    assert osm_net.with_overpass_fallback(flaky) == "graph"
    assert tried == list(osm_net.OVERPASS_ENDPOINTS[:3]), tried
    assert ox.settings.overpass_url == orig, "endpoint setting must be restored"
    # circuit breaker: the next call goes straight to the mirror that worked
    first = []
    osm_net.with_overpass_fallback(lambda: first.append(ox.settings.overpass_url) or "ok")
    assert first == [osm_net.OVERPASS_ENDPOINTS[2]], first
    # an endpoint that timed out is tried LAST even if a later call's
    # success elsewhere reset the preference (the real outage pattern)
    osm_net._preferred = 0
    order = []
    osm_net.with_overpass_fallback(lambda: order.append(ox.settings.overpass_url) or "ok")
    assert order == [osm_net.OVERPASS_ENDPOINTS[2]], f"down endpoints must be skipped first: {order}"
    osm_net._preferred = 0
    osm_net._down.clear()

    def no_data():
        raise ValueError("no data in area")          # not a network error: no retries
    calls = []
    try:
        osm_net.with_overpass_fallback(lambda: calls.append(1) or no_data())
        raise AssertionError("non-network errors must propagate")
    except ValueError:
        pass
    assert len(calls) == 1

    real = ox.graph_from_bbox
    ox.graph_from_bbox = lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("down"))
    try:
        g = tr.load_osm_direction_graph_cached((35.5, 33.89, 35.51, 33.9), None, None, False, "t")
    finally:
        ox.graph_from_bbox = real
    assert g is None, "all endpoints down must degrade to None, not raise"
    assert tr.resolve_osm_road_direction((0, 0), (1, 0), None) == (None, None)


@test
def flood_water_surface_follows_depth_with_beer_lambert_opacity():
    """render.water: surface z = base + offset + depth; opacity follows
    Beer-Lambert (turbid floodwater); dry cells are fully transparent and
    leave the ground untouched in the render; deep water covers it."""
    import pyvista as pv
    from render.water import FloodWaterSurface, water_alpha, E_FOLD_M
    from render.sky import environment_texture

    from render.water import WET_FILM_ALPHA, water_rgb_linear
    a = water_alpha(np.array([0.0, 0.004, 0.01, E_FOLD_M, 0.6]))
    assert a[0] == 0 and a[1] == 0, "dry / sub-5mm cells must be invisible"
    assert WET_FILM_ALPHA <= a[2] < a[3] < a[4] and a[4] > 0.97, "coverage must grow with depth from the wet-film level"
    rgb = water_rgb_linear(np.array([0.01, 0.6]))
    assert rgb[0].max() < 0.08 and rgb[1][0] > 0.2, "shallow = dark wet film, deep = bright muddy body"

    p = pv.Plotter(off_screen=True, window_size=(200, 100))
    p.set_background("black")
    p.set_environment_texture(environment_texture(np.full((32, 64, 3), 0.3, np.float32)), is_srgb=False)
    p.add_mesh(pv.Plane(center=(0, 0, 0), i_size=40, j_size=20), color=(0.9, 0.2, 0.2), lighting=False)
    nx, ny, dx = 80, 40, 0.5
    depth = np.zeros((ny, nx), np.float32)
    depth[:, nx // 2:] = 0.4                                  # east half flooded 40 cm
    w = FloodWaterSurface(p, nx, ny, -20, -10, dx, dx, 0.2, depth)
    z = np.asarray(w.surf.points)[:, 2]
    assert np.isclose(z.max(), 0.6) and np.isclose(z.min(), 0.2)
    w.set_base(np.full(nx * ny, 5.0))
    assert np.isclose(np.asarray(w.surf.points)[:, 2].min(), 5.2), "drape base not applied"
    w.set_base(None)
    p.camera.parallel_projection = True
    p.camera_position = [(0, 0, 50), (0, 0, 0), (0, 1, 0)]
    p.camera.parallel_scale = 10
    p.render()
    img = p.screenshot(return_img=True).astype(int)
    dry, wet = img[50, 40], img[50, 160]
    import vtk
    assert w.actor.GetMapper().GetColorMode() == vtk.VTK_COLOR_MODE_DIRECT_SCALARS, \
        "water colours must be direct RGBA (a lookup table drops the Beer-Lambert alpha)"
    assert dry[0] > 200 and dry[1] < 80, f"dry ground must show through untouched: {dry}"
    assert wet[0] < 150, f"40 cm of turbid water must cover the red ground: {wet}"
    assert wet[0] >= wet[2] and abs(int(wet[0]) - int(wet[1])) < 60, \
        f"covered ground should read as silt-brown/grey water, not a colormap hue: {wet}"
    p.close()


@test
def storm_hyetograph_drives_rain_density_and_wetness():
    """render.hyetograph + WeatherMixin: rain intensity/cumulative depth from
    the flood engine's storm file (mass-consistent with total_mm), and the
    rain streak count scales with intensity (cloudburst = all drops)."""
    import pyvista as pv
    from render.hyetograph import load_storm, intensity_mm_h, cumulative_mm, CLOUDBURST_MM_H
    from weather_mixin import WeatherMixin

    storm = load_storm("v1_nov2025")
    if storm is None:
        print("    (Beirut_Project-main storms not present — hyetograph file checks skipped)")
        storm = {"steps": np.array([[0, 450, 12.8], [450, 1350, 88.8], [1350, 1800, 12.8]]),
                 "duration": 3600.0, "total_mm": 25.4}
    assert intensity_mm_h(storm, 600) == 88.8 and intensity_mm_h(storm, 2000) == 0.0
    assert abs(cumulative_mm(storm, 1e9) - storm["total_mm"]) < 0.05, "hyetograph must integrate to total_mm"
    assert cumulative_mm(storm, 0) == 0.0 and 0 < cumulative_mm(storm, 450) < cumulative_mm(storm, 900)

    class _Host(WeatherMixin):
        def _apply_weather_car_factor(self, f):
            pass
        def _apply_wet_road(self, wf):
            self.wf = wf
    h = _Host()
    h.plotter = pv.Plotter(off_screen=True)
    h._init_weather()
    h._weather_bounds = (-50, 50, -50, 50, 0, 80)
    counts = {}
    for i_mm in (12.8, 88.8):
        h.weather["rain_scale"] = min(1.0, i_mm / CLOUDBURST_MM_H)
        h.weather["wet_override"] = 0.3
        h._tick_rain(0.05)
        counts[i_mm] = h.weather["rain_pd"].n_lines
    assert counts[88.8] > 5 * counts[12.8] > 0, counts
    assert h.wf == 0.3, "wetness must follow rain that has fallen, not a timer"
    h.plotter.close()


@test
def cache_writes_are_atomic_strip_samplers_and_corrupt_caches_are_misses():
    """Regression: the land-use re-save pickled the street graph WITH its
    closure terrain_sampler; the dump raised midway and left a truncated
    graph cache, and the next startup crashed with EOFError."""
    import pickle
    import tempfile
    from pathlib import Path
    import networkx as nx
    from app_core import _atomic_pickle, _load_pickle_or_none, _save_graph_cache

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        g = nx.MultiDiGraph()
        g.add_edge(1, 2)
        def _local_sampler(xy):                    # closure: not picklable
            return xy
        g.graph["terrain_sampler"] = _local_sampler
        g.graph["crs"] = "x"
        _save_graph_cache(g, td / "graph.pkl")
        assert g.graph["terrain_sampler"] is _local_sampler, "sampler must be restored after saving"
        back = _load_pickle_or_none(td / "graph.pkl", "graph")
        assert back is not None and back.number_of_edges() == 1 and "terrain_sampler" not in back.graph

        try:
            _atomic_pickle({"f": _local_sampler}, td / "bad.pkl")
            raise AssertionError("unpicklable object must raise")
        except (pickle.PicklingError, AttributeError, TypeError):
            pass
        assert not (td / "bad.pkl").exists() and not (td / "bad.pkl.tmp").exists(), \
            "a failed dump must not leave a (truncated) cache file"

        (td / "trunc.pkl").write_bytes(b"")
        assert _load_pickle_or_none(td / "trunc.pkl", "graph") is None
        assert not (td / "trunc.pkl").exists(), "corrupt cache must be removed so it gets rebuilt"


@test
def flood_overlays_stack_above_roads_sidewalks_and_photo_ground():
    """Regression: puddles (z 0.06) and corridor (0.04) sat BELOW the road
    (0.10) / sidewalk (0.15) surfaces and the photoreal ground (0.18), so
    floodwater was invisible exactly on the streets where it flows."""
    import flood_mixin as fm
    from render.survey_mixin import _PHOTO_Z
    ROAD_Z, SIDEWALK_Z = 0.10, 0.15
    for name, z in (("puddles", fm._PUDDLE_Z_OFFSET), ("corridor", fm._CORRIDOR_Z_OFFSET),
                    ("water", fm._WATER_Z_OFFSET)):
        assert z > max(ROAD_Z, SIDEWALK_Z, _PHOTO_Z), f"{name} overlay z={z} is under the ground layers"
    assert fm._PUDDLE_Z_OFFSET >= fm._CORRIDOR_Z_OFFSET, "puddles must draw over the corridor materials"


@test
def flood_regular_resampler_is_exact_volume_preserving_and_dry_outside():
    """flood_mixin._RegularResampler (replaced Delaunay/griddata): exact on
    linear fields, block-averages so a sub-cell flooded channel keeps its
    volume, is dry outside the solver domain, nearest for class rasters."""
    from pyproj import Transformer
    import flood_mixin as fm

    lat0, lon0 = 33.8944, 35.5227
    to_local = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
                                    always_xy=True)
    e0, n0 = Transformer.from_crs("EPSG:4326", fm.FLOOD_UTM_CRS, always_xy=True).transform(lon0, lat0)
    res, W, H = 0.5, 800, 600                                  # 400 m x 300 m solver grid
    tr = {"minx": e0 - 200.0, "maxy": n0 + 150.0, "res": res, "width": W, "height": H}
    rs = fm._RegularResampler(tr, to_local, radius=2000.0, target_res_m=2.0, max_grid_cells=400)
    assert rs.factor in (3, 4) and 150 <= rs.nx <= 215, (rs.factor, rs.nx)   # grid convergence widens bounds

    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    E = tr["minx"] + (cols + 0.5) * res
    lin = 0.1 + 0.001 * (E - tr["minx"])                     # linear ramp 0.1 .. 0.5 m
    out = rs.sample(lin, "mean")
    tx, ty = np.meshgrid(np.linspace(rs.x0, rs.x0 + rs.dx * (rs.nx - 1), rs.nx),
                         np.linspace(rs.y0, rs.y0 + rs.dy * (rs.ny - 1), rs.ny))
    lon, lat = to_local.transform(tx.ravel(), ty.ravel(), direction="INVERSE")
    te, tn = Transformer.from_crs("EPSG:4326", fm.FLOOD_UTM_CRS, always_xy=True).transform(lon, lat)
    te, tn = np.asarray(te), np.asarray(tn)
    expect = (0.1 + 0.001 * (te - tr["minx"])).reshape(rs.ny, rs.nx)
    # inside the (grid-convergence-rotated) solver domain, away from its edge
    inner = ((te > tr["minx"] + 3) & (te < tr["minx"] + W * res - 3)
             & (tn < tr["maxy"] - 3) & (tn > tr["maxy"] - H * res + 3)).reshape(rs.ny, rs.nx)
    assert inner.mean() > 0.8
    assert np.abs(out[inner] - expect[inner]).max() < 0.005, "linear field not reproduced"

    chan = np.zeros((H, W))
    chan[:, 401] = 1.0                                           # one 0.5 m column, 1 m deep
    v_src = chan.sum() * res * res
    o = rs.sample(chan, "mean")
    v_out = o.sum() * rs.dx * rs.dy
    assert abs(v_out - v_src) / v_src < 0.15, f"channel volume {v_out:.0f} vs {v_src:.0f} m^3"
    assert o.max() < 0.6, "block averaging should spread a sub-cell channel, not point-sample it"

    big = fm._RegularResampler(tr, to_local, radius=2000.0, target_res_m=2.0, max_grid_cells=400)
    big._col = big._col - 10000                                 # shift targets off the solver grid
    assert np.all(big.sample(lin, "mean") == 0.0), "outside the solver domain must be dry"

    cls = (cols // 200 + rows // 200).astype(np.uint8)          # classes 0..4
    oc = rs.sample(cls, "nearest")
    assert set(np.unique(oc)).issubset(set(np.unique(cls))), "nearest must not invent classes"


def _flat_inputs(n=40, res=2.0, slope=0.0, manning=0.03, infil=0.0, building=None, open_frac=None):
    from floodsim.engine import FloodInputs
    xs = np.arange(n) * res
    dem = np.tile(slope * xs[None, :], (n, 1)).astype(float)
    valid = np.ones((n, n), bool)
    return FloodInputs(dem, res, np.full((n, n), manning), np.full((n, n), float(infil)), np.ones((n, n)), valid,
                       np.zeros((n, n), bool), None, building, open_frac)


@test
def flood_engine_closes_mass_holds_still_water_and_matches_infiltration_theory():
    """floodsim.engine: (1) rain on a sloped plane drains off the open edge and the
    mass balance closes to rounding error; (2) a lake at rest over a rough bed
    stays at rest (well-balanced, no spurious current); (3) rain below the
    infiltration rate never ponds, above it ponds at (i - f) * t."""
    from floodsim.engine import FloodEngine, FloodInputs

    # (1) mass balance with outflow
    inp = _flat_inputs(40, 2.0, slope=0.02, manning=0.03)
    eng = FloodEngine(inp)
    r = eng.run([(0.0, 600.0, 60.0)], 900.0, save_every=300.0, chunk_s=1e9)
    assert abs(r.meta["closure_rel"]) < 1e-9, r.meta["closure_rel"]
    assert r.meta["vol_outflow_m3"] > 0.3 * r.meta["vol_rain_m3"], "a sloped plane must shed most of the rain"
    assert abs(r.meta["vol_rain_m3"] - 0.060 * (600.0 / 3600.0) * 38 * 38 * 4.0 * 1.0) < 40, "rain volume off"

    # (2) lake at rest: water surface at 1.0 m over a bumpy bed, closed domain
    rng = np.random.default_rng(1)
    n, res = 30, 2.0
    bed = 0.3 * rng.random((n, n))
    valid = np.ones((n, n), bool)
    water = np.zeros((n, n), bool)
    lake = FloodInputs(bed, res, np.full((n, n), 0.03), np.zeros((n, n)), np.zeros((n, n)), valid, water)
    eng = FloodEngine(lake)
    eng.keep[:] = 1.0                                           # closed boundary for this test
    eng.depth = np.maximum(1.0 - bed, 0.0)
    v0 = eng.stored_m3()
    for _ in range(400):
        eng.step(0.0, 1e9)
    assert abs(eng.stored_m3() - v0) / v0 < 1e-9
    speed = np.abs(eng.qx).max() / 0.5
    assert speed < 0.05, f"spurious current {speed:.3f} m/s in still water"

    # (3) infiltration theory: flat closed plane, rain 36 mm/h vs infiltration 18 mm/h (and 72)
    for infil, expect_pond in ((72.0, 0.0), (18.0, (36.0 - 18.0) / 3.6e6 * 600.0)):
        inp = _flat_inputs(20, 2.0, infil=infil)
        eng = FloodEngine(inp)
        eng.keep[:] = 1.0
        r = eng.run([(0.0, 600.0, 36.0)], 600.0, save_every=1e9, chunk_s=1e9)
        got = float(r.final_depth.mean())
        assert abs(got - expect_pond) < 1e-4 + 0.02 * expect_pond, f"infil {infil}: ponded {got:.5f} vs {expect_pond:.5f} m"
        assert abs(r.meta["closure_rel"]) < 1e-9


@test
def flood_engine_porosity_blocks_flow_and_conserves_volume():
    """Sub-grid porosity: a half-open face passes half the discharge, volumes use the
    open area, and mass still closes; a drain removes water at its capacity."""
    from floodsim.engine import FloodEngine
    n = 24
    for phi in (1.0, 0.5):
        of = np.ones((n, n))
        of[:, 12] = phi                                          # a half-blocked column
        inp = _flat_inputs(n, 2.0, slope=0.0, open_frac=of)
        eng = FloodEngine(inp)
        eng.keep[:] = 1.0
        eng.depth[:, :12] = 0.5                                  # dam-break onto the dry side
        v0 = eng.stored_m3()
        for _ in range(12):                                       # the dam-break transient (long runs equalise)
            eng.step(0.0, 1e9)
        moved = float((eng.depth[:, 13:] * eng.phi[:, 13:]).sum()) * eng.area
        assert abs(eng.stored_m3() - v0) / v0 < 1e-9, "porosity broke mass conservation"
        if phi == 1.0:
            full = moved
        else:
            assert 0.0 < moved < 0.8 * full, f"half-open gap passed {moved / full:.2f} of the open-gap flow"
    inp = _flat_inputs(10, 2.0)
    from floodsim.engine import FloodInputs
    inp = FloodInputs(inp.dem, inp.res, inp.manning, inp.infil_mmh, inp.rain_weight, inp.valid, inp.water,
                      (np.array([5]), np.array([5]), np.array([0.01])))
    eng = FloodEngine(inp)
    eng.keep[:] = 1.0
    r = eng.run([(0.0, 600.0, 100.0)], 600.0, save_every=1e9, chunk_s=1e9)
    assert abs(r.meta["vol_drained_m3"] - 0.01 * 600.0) < 0.02 * 6.0 + 1e-6 or r.meta["vol_drained_m3"] <= 0.01 * 600.0 + 1e-6
    assert abs(r.meta["closure_rel"]) < 1e-9


@test
def flood_design_material_blend_georef_rasterization_and_model():
    """floodsim.design / georef / model: material fractions blend infiltration and
    roughness by area and lower the bed for detention; a local-frame polygon
    rasterizes to its true area; the DesignModel turns trees/areas/buildings/drains
    into solver inputs."""
    from pyproj import Transformer
    from floodsim import design as D
    from floodsim.engine import FloodInputs
    from floodsim.georef import SolverGeoref
    from floodsim.model import DesignModel, DesignObject

    n = 30
    base = _flat_inputs(n, 2.0, manning=0.016, infil=5.0)
    base.meta["factor"] = 4
    half = np.zeros((n, n)); half[10:20, 10:20] = 0.5
    out = D.apply_design(base, {3: half})                         # 50 % bioswale
    assert abs(out.infil_mmh[15, 15] - (0.5 * 5.0 + 0.5 * 200.0)) < 1e-9
    assert abs(out.manning[15, 15] - (0.5 * 0.016 + 0.5 * 0.15)) < 1e-9
    assert abs((base.dem - out.dem)[15, 15] - 0.5 * 0.15) < 1e-12
    assert out.infil_mmh[0, 0] == 5.0 and out.dem[0, 0] == base.dem[0, 0]
    both = D.apply_design(base, {3: np.full((n, n), 0.8), 6: np.full((n, n), 0.8)})    # overlap shares the cell
    assert abs(both.infil_mmh[0, 0] - 0.5 * (200.0 + 250.0)) < 1e-9

    lat0, lon0 = 33.8944, 35.5227
    to_local = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
                                    always_xy=True)
    e0, n0 = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True).transform(lon0, lat0)
    tr = {"crs": "EPSG:32636", "minx": e0 - 100.0, "maxy": n0 + 100.0, "res": 2.0, "width": 100, "height": 100}
    geo = SolverGeoref(tr, to_local)
    sq = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]])          # 40 x 20 m
    (rs, cs), blk = geo.polygon_fraction(sq)
    assert abs(blk.sum() * 4.0 - 800.0) < 800.0 * 0.03, f"rasterized area {blk.sum() * 4.0:.0f} vs 800"
    r_, c_ = geo.local_to_cell(np.array([[0.0, 0.0]]))
    assert abs(int(r_[0]) - 50) <= 1 and abs(int(c_[0]) - 50) <= 1
    cell_xy = geo.xy[(r_[0]) * 100 + c_[0]]
    assert np.hypot(*cell_xy) < 2.0 * 1.5, "cell centre not at the origin"

    base2 = FloodInputs(np.zeros((100, 100)), 2.0, np.full((100, 100), 0.016), np.full((100, 100), 5.0),
                        np.ones((100, 100)), np.ones((100, 100), bool), np.zeros((100, 100), bool), None,
                        np.zeros((100, 100), bool), np.ones((100, 100)))
    base2.meta["factor"] = 4
    dm = DesignModel()
    dm.add(DesignObject("a1", "area", cls=6, geom=sq))
    dm.add(DesignObject("t1", "tree", geom=np.array([[40.0, 40.0]])))
    dm.add(DesignObject("d1", "drain", geom=np.array([[-40.0, -40.0]])))
    bld = np.array([[30.0, -40.0], [42.0, -40.0], [42.0, -30.0], [30.0, -30.0]])
    dm.add(DesignObject("b1", "building", geom=bld))
    inp = dm.to_inputs(base2, geo)
    assert inp.infil_mmh.max() >= 250.0 - 1e-9 and inp.dem.min() <= -0.39, "rain garden not applied"
    assert inp.drains is not None and len(inp.drains[0]) == 1
    base2.drains = (np.arange(600) % 100, np.arange(600) // 6, np.full(600, 0.03))      # the city's study inlets
    inp2 = dm.to_inputs(base2, geo)
    assert len(inp2.drains[0]) == 1 and abs(inp2.drains[2].sum() - 0.03) < 1e-12, \
        "a design must add only its own drains: the city's inlets are clogged in the studied storm"
    dm_empty = DesignModel(); dm_empty.add(DesignObject("a", "area", cls=5, geom=sq))
    assert dm_empty.to_inputs(base2, geo).drains is None
    assert inp.building.sum() >= 20 and inp.dem.max() > 10.0, "placed building must block"
    assert dm.summary().get("trees") == 1 and dm.summary().get("buildings") == 1
    assert dm.remove_missing({"a1", "t1"}) == 2 and len(dm.objects) == 2


@test
def flood_engine_reproduces_published_corridor_volumes_and_trend():
    """floodsim on the real Al-Masar terrain at 4 m vs the published 0.5 m GPU runs:
    rain volume and infiltration share agree, mass closes exactly, and the
    green corridor reduces the flooded area (skipped without the study data)."""
    from floodsim.terrain import terrain_available, OFFICIAL_MATERIAL
    from floodsim import validate as V
    if not terrain_available() or not OFFICIAL_MATERIAL.exists() or not (V.RUNS / "before_v1_nov2025" / "run_meta.json").exists():
        print("    (Beirut_Project-main study data not present — skipped)")
        return
    from floodsim.metrics import summarize
    out = {}
    for tag, corridor in (("before", False), ("after", True)):
        inp, r = V.run_case(4.0, "v1_nov2025", corridor)
        meta = json.loads((V.RUNS / f"{tag}_v1_nov2025" / "run_meta.json").read_text())
        s = summarize(r, inp)
        assert abs(s["volume_rain_m3"] / meta["vol_rain_m3"] - 1.0) < 0.01, "rain volume"
        ref_inf = 100 * meta["vol_infiltrated_m3"] / meta["vol_rain_m3"]
        assert abs(s["infiltrated_pct"] - ref_inf) < 0.15 * ref_inf + 0.5, f"{tag} infiltration {s['infiltrated_pct']:.1f} vs {ref_inf:.1f} %"
        assert abs(s["closure_rel"]) < 1e-9
        out[tag] = s
    assert out["after"]["flooded_area_ha"] < out["before"]["flooded_area_ha"], "corridor must reduce flooded area"
    assert out["after"]["infiltrated_pct"] > 2.0 * out["before"]["infiltrated_pct"], "corridor must at least double infiltration"


@test
def flood_run_thread_streams_snapshots_cancels_and_matches_direct_run():
    """floodsim.runner: the worker thread streams monotonically advancing snapshots
    while the caller polls, a cancel stops it promptly with a consistent partial result,
    and a completed threaded run is bit-identical to a direct engine run."""
    import time as _t
    from floodsim.engine import FloodEngine
    from floodsim.runner import FloodRun

    storm = {"steps": [(0.0, 300.0, 80.0)], "duration": 600.0, "name": "t"}
    inp = _flat_inputs(40, 2.0, slope=0.02, infil=5.0)
    direct = FloodEngine(inp).run(storm["steps"], storm["duration"], save_every=100.0, chunk_s=1e9)
    run = FloodRun(inp, storm, save_every=100.0, chunk_s=20.0).start()
    seen = []
    while not run.done.is_set():
        sn = run.snapshot()
        seen.append((sn["t"], sn["seq"]))
        _t.sleep(0.002)
    sn = run.snapshot()
    assert run.error is None and sn["finished"] and abs(sn["fraction"] - 1.0) < 1e-6
    ts = [t for t, _ in seen]
    assert ts == sorted(ts), "snapshot time must never go backwards"
    assert np.array_equal(run.result.max_depth, direct.max_depth), "threaded run differs from the direct run"
    assert abs(run.result.meta["closure_rel"]) < 1e-9 and len(run.result.frames) == len(direct.frames)

    big = _flat_inputs(120, 1.0, slope=0.01)
    r2 = FloodRun(big, {"steps": [(0.0, 3600.0, 60.0)], "duration": 3600.0}, chunk_s=5.0).start()
    _t.sleep(0.4)
    r2.cancel()
    assert r2.join(10.0), "cancel must stop the run"
    assert r2.result is not None and r2.result.meta["cancelled"] and r2.result.meta["duration"] < 3600.0
    assert abs(r2.result.meta["closure_rel"]) < 1e-9, "a cancelled run must still close its mass balance"


@test
def flood_worker_shares_the_numba_kernel_lock_with_other_parallel_kernels():
    """Regression: numba's workqueue layer aborts the process when two threads launch
    parallel kernels at once (it crashed the app when the background shadow worker
    overlapped a flood run). The engine must hold shadow_engine.NUMBA_KERNEL_LOCK per step."""
    import time as _t
    from numba import njit, prange
    from shadow_engine import NUMBA_KERNEL_LOCK
    from floodsim import engine as fe
    from floodsim.runner import FloodRun
    assert fe.KERNEL_LOCK is NUMBA_KERNEL_LOCK, "engine must use the app-wide kernel lock"

    @njit(parallel=True)
    def other(a):
        t = 0.0
        for i in prange(a.size):
            t += a[i] * 1.0001
        return t

    other(np.ones(10))
    run = FloodRun(_flat_inputs(60, 2.0, slope=0.01), {"steps": [(0.0, 120.0, 60.0)], "duration": 240.0},
                   chunk_s=1e9).start()
    n = 0
    t_end = _t.time() + 20.0
    while not run.done.is_set() and _t.time() < t_end:
        with NUMBA_KERNEL_LOCK:                                   # what the shadow engine does
            other(np.ones(200000))
        n += 1
    assert run.join(30.0) and run.error is None and n > 5, (n, run.error)


@test
def flood_lab_hotspots_skip_deep_pits_and_buildings_and_stay_separated():
    """flood_lab_mixin._lab_hotspots: ranks open street/ground cells wet 8 cm - 1.2 m by wet area in a
    30 m window — a 3 m courtyard pit and wet cells inside buildings are ignored, and returned
    hotspots keep their minimum separation."""
    from flood_lab_mixin import FloodLabMixin
    from floodsim.engine import FloodInputs

    n, res = 120, 2.0
    base = FloodInputs(np.zeros((n, n)), res, np.full((n, n), 0.03), np.zeros((n, n)), np.ones((n, n)),
                       np.ones((n, n), bool), np.zeros((n, n), bool), None, np.zeros((n, n), bool), np.ones((n, n)))

    class _Geo:
        xy = np.column_stack([np.tile(np.arange(n) * res, n), np.repeat(np.arange(n) * res, n)])

    class _Rec:
        pass

    class _Host(FloodLabMixin):
        def _lab_inputs(self):
            return base, _Geo()
        def _lab_best_record(self):
            return None

    host = _Host()
    host.flood_lab = {}
    depth = np.zeros((n, n))
    depth[20:30, 20:30] = 0.4                       # a real puddle (wet 20 m x 20 m)
    depth[60:70, 60:70] = 3.0                       # a courtyard pit: too deep to count
    depth[90:100, 30:40] = 0.3                      # second puddle, far from the first
    base.building[90:100, 30:34] = True            # part of it inside a building (ignored)
    spots = host._lab_hotspots(depth, k=5, sep_m=40.0)
    xs = [(x, y) for x, y, _ in spots]
    assert len(spots) == 2, spots
    assert all(abs(x - 50.0) < 25 and abs(y - 50.0) < 25 or abs(x - 70.0) < 25 and abs(y - 190.0) < 25 for x, y in xs), xs
    assert not any(abs(x - 130.0) < 20 and abs(y - 130.0) < 20 for x, y in xs), "deep pit must not be a hotspot"
    d = np.hypot(xs[0][0] - xs[1][0], xs[0][1] - xs[1][1])
    assert d >= 40.0, d
    assert spots[0][2] >= spots[1][2], "hotspots must be ranked by wet area"


@test
def postfx_tone_maps_the_3d_scene_but_leaves_ui_overlay_colours_exact():
    """render.postfx: the filmic curve compresses an over-bright 3D surface,
    the 2D control-panel overlay is drawn after it (colours unchanged), SSAO
    can be toggled by rebuilding, and disabling restores the default path."""
    import pyvista as pv
    from render.postfx import PostFX

    p = pv.Plotter(off_screen=True, window_size=(300, 200))
    p.set_background("black")
    p.add_mesh(pv.Plane(i_size=10, j_size=10), color=(1.0, 1.0, 1.0), lighting=False)
    p.add_text("PANEL", position=(5, 5), font_size=20, color=(0.2, 0.8, 0.2))
    p.camera_position = [(0, -15, 10), (0, 0, 0), (0, 0, 1)]
    p.render()
    base = p.screenshot(return_img=True).copy()
    fx = PostFX(p.renderer)
    fx.apply()
    p.render()
    toned = p.screenshot(return_img=True).copy()
    assert base[100, 150].min() >= 250 and toned[100, 150].max() < 235, \
        f"filmic tone mapping should compress white: {base[100, 150]} -> {toned[100, 150]}"
    text = np.all(np.abs(base.astype(int) - [51, 204, 51]) < 40, axis=2)
    assert text.sum() > 200, "panel text not found"
    assert np.abs(base[text].astype(int) - toned[text].astype(int)).max() <= 12, \
        "UI overlay must not be tone-mapped"
    fx.update(ssao=True, translucency="oit")
    p.render()
    assert p.renderer.GetPass() is not None and not p.renderer.GetUseSSAO()
    # VTK 9.6: OIT + SSAO renders translucent layers opaque white in the twin
    assert fx._passes["steps"].GetTranslucentPass().GetClassName() != "vtkOrderIndependentTranslucentPass"
    fx.update(ssao=False)
    assert fx._passes["steps"].GetTranslucentPass().GetClassName() == "vtkOrderIndependentTranslucentPass"
    # cached chains: switching back is instant (same pass graph, no rebuild)
    seq_a = p.renderer.GetPass()
    fx.update(ssao=True)
    fx.update(ssao=False)
    assert p.renderer.GetPass() is seq_a, "a used configuration must reuse its cached chain"
    fx.invalidate()
    assert p.renderer.GetPass() is not seq_a, "invalidate() must rebuild from fresh passes"
    fx.update(ssao=False, tone_mapping=False, fxaa=False)
    p.render()
    assert p.renderer.GetPass() is None
    assert np.array_equal(p.screenshot(return_img=True)[100, 150], base[100, 150])
    p.close()


@test
def greenspace_polygon_triangulation_handles_non_star_shaped_concave_outline():
    """Regression: make_greenspace_polygon fan-triangulated from vertex 0,
    which only covers a polygon correctly if EVERY point is visible from
    vertex 0 ("star-shaped from vertex 0"). A plus/cross outline is not —
    the fan must have produced triangles poking outside the drawn shape."""
    from shapely.geometry import Polygon as SPoly, Point as SPoint
    from editor_ops import make_greenspace_polygon

    # A 12-point plus/cross, CCW, starting at a concave-adjacent vertex.
    pts = [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2),
           (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1)]
    true_poly = SPoly(pts)
    assert true_poly.is_valid

    mesh = make_greenspace_polygon(pts, base_z=0.0)
    assert mesh.n_cells == len(pts) - 2, f"expected a valid fan-free triangulation ({len(pts)-2} tris), got {mesh.n_cells}"

    total_area = 0.0
    for i in range(mesh.n_cells):
        tri = mesh.get_cell(i).points[:, :2]
        centroid = tri.mean(axis=0)
        assert true_poly.buffer(1e-9).contains(SPoint(centroid)), (
            f"triangle {i} (verts {tri.tolist()}) centroid {centroid.tolist()} "
            f"falls OUTSIDE the drawn cross shape — triangulation leaked past the outline"
        )
        a, b, c = tri
        total_area += abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2.0

    assert abs(total_area - true_poly.area) < 1e-6, (
        f"triangulated area {total_area} != true polygon area {true_poly.area}"
    )


@test
def stairs_mesh_spans_full_requested_distance_and_handles_flat_case():
    """Regression: n_steps was derived from rise alone and step run capped at
    the physical `step_run` constant, so a shallow slope over a long span
    produced a short flight ending well before the second click point — and
    two same-elevation clicks produced n zero-height (invisible) boxes."""
    from editor_ops import make_stairs_mesh

    # Shallow slope: dz=1m over a 20m span. The old code's flight length was
    # min(step_run, ...) * n_from_rise ~= 0.29 * 6 = 1.7 m, nowhere near 20 m.
    mesh = make_stairs_mesh(0, 0, 0.0, 20, 0, 1.0, width=2.0)
    xmax = float(mesh.points[:, 0].max())
    assert xmax > 19.0, f"stairs flight only reached x={xmax:.2f}, expected ~20 (disconnected from the clicked endpoint)"
    zmax = float(mesh.points[:, 2].max())
    assert abs(zmax - 1.0) < 0.05, f"stairs flight top z={zmax:.2f}, expected ~1.0"

    # Flat case: both clicks at the same elevation must not produce a
    # degenerate (zero-volume, invisible) mesh.
    flat = make_stairs_mesh(0, 0, 2.0, 10, 0, 2.0005, width=2.0)
    zspan = float(flat.points[:, 2].max() - flat.points[:, 2].min())
    assert zspan > 0.05, f"flat-elevation stairs produced a near-zero-height mesh (z span={zspan:.4f})"

    # Coincident points: nothing sensible to build.
    try:
        make_stairs_mesh(5, 5, 1.0, 5, 5, 1.0)
        assert False, "expected ValueError for coincident start/end points"
    except ValueError:
        pass


@test
def stairs_mesh_bottom_face_normal_points_outward():
    """Regression: the bottom cap's vertex order matched the top cap's, which
    (by the right-hand rule) gives BOTH caps a +z normal — correct for the
    top, inverted (pointing back into the solid) for the bottom.

    Computed from the RAW vertex winding (cross product of consecutive
    triangle edges), not VTK's compute_normals(auto_orient_normals=True) —
    that flag silently repairs inconsistent winding, which would hide
    exactly the bug this test exists to catch."""
    from editor_ops import make_stairs_mesh

    mesh = make_stairs_mesh(0, 0, 0.0, 0, 0, 0.5, width=2.0)  # dz-only: one step, no horizontal drift
    pts = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape(-1, 4)   # post-triangulate(): each cell is [3, a, b, c]
    bottom_z = float(pts[:, 2].min())

    checked = 0
    for f in faces:
        a, b, c = pts[f[1]], pts[f[2]], pts[f[3]]
        if abs(a[2] - bottom_z) < 1e-6 and abs(b[2] - bottom_z) < 1e-6 and abs(c[2] - bottom_z) < 1e-6:
            n = np.cross(b - a, c - b)
            assert n[2] < 0, f"bottom face raw winding gives normal {n} — points upward (into the solid)"
            checked += 1
    assert checked > 0, "test setup: no triangle found at the bottom z"


@test
def demand_edge_map_rebuilt_after_editor_graph_rebuild():
    """Regression: _rebuild_traffic_and_arrows (called after any editor graph
    edit — road add/delete/reverse, roundabout, bridge) rebuilds car_paths
    from scratch, but never rebuilt demand_mixin's (u,v)->old-index map. Every
    demand-routed trip planned after ANY edit would silently follow whatever
    edge happened to occupy that stale index in the NEW car_paths list."""
    import types
    import networkx as nx
    from route_mixin import RouteMixin
    from demand_mixin import DemandMixin

    g = nx.MultiDiGraph()
    for n, (x, y) in {0: (0, 0), 1: (100, 0), 2: (200, 0), 3: (300, 0)}.items():
        g.add_node(n, x=float(x), y=float(y))
    g.add_edge(0, 1, key=0, length=100.0)
    g.add_edge(1, 2, key=0, length=100.0)
    g.add_edge(2, 3, key=0, length=100.0)

    def _mk_path(u, v):
        return {"u": u, "v": v, "length": 100.0, "maxspeed_ms": 10.0,
                "points": np.array([[float(u) * 100, 0.0, 0.5], [float(v) * 100, 0.0, 0.5]]),
                "cum_len": np.array([0.0, 100.0])}

    # "Before" extraction: paths in edge-insertion order [0->1, 1->2, 2->3].
    paths_before = [_mk_path(0, 1), _mk_path(1, 2), _mk_path(2, 3)]
    # "After" extraction (simulating an editor edit that reordered/changed the
    # graph): the SAME (1,2) edge now sits at a DIFFERENT index, and a new
    # edge (3,0) was added.
    paths_after = [_mk_path(2, 3), _mk_path(1, 2), _mk_path(0, 1), _mk_path(3, 0)]
    extraction_calls = {"n": 0}

    def _fake_extract(self, graph, z_level=0.5):
        extraction_calls["n"] += 1
        paths = paths_before if extraction_calls["n"] == 1 else paths_after
        outgoing = {}
        for i, p in enumerate(paths):
            outgoing.setdefault(p["u"], []).append(i)
        return paths, outgoing

    class _FakePlotter:
        def add_mesh(self, *a, **kw):
            return object()
        def remove_actor(self, *a, **kw):
            pass
        def __getattr__(self, _name):
            return lambda *a, **kw: None

    host = types.SimpleNamespace(
        street_graph=g,
        scene_state={},
        args=types.SimpleNamespace(traffic_speed=1.0),
        plotter=_FakePlotter(),
        car_rng=np.random.default_rng(0),
        car_anim={"edge_idx": np.zeros(2, dtype=np.int64)},
        demand=object(),   # non-None sentinel: "demand routing is in use"
    )
    host._extract_drivable_paths = types.MethodType(_fake_extract, host)
    host._set_actor_visibility = lambda *a, **kw: None
    host._stage = lambda *a, **kw: None
    host._rebuild_traffic_and_arrows = types.MethodType(RouteMixin._rebuild_traffic_and_arrows, host)
    host._route_to_edges = types.MethodType(DemandMixin._route_to_edges, host)

    host._rebuild_traffic_and_arrows()   # 1st extraction -> paths_before
    assert host._route_to_edges([1, 2]) == [1], "map wrong even on first build"

    host._rebuild_traffic_and_arrows()   # simulates a later editor edit -> paths_after
    edges = host._route_to_edges([1, 2])
    assert edges == [1], (
        f"demand route (1,2) resolved to car_paths index {edges}, but in the "
        f"POST-EDIT car_paths edge (1,2) is actually at index 1 "
        f"({[(p['u'], p['v']) for p in paths_after]}) — the map was not rebuilt"
    )
    assert host._route_to_edges([3, 0]) == [3], "new post-edit edge (3,0) not resolvable at all"


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
def cyclists_gated_by_bearing_not_by_colliding_car_path_index():
    """Regression: cyclist_mixin used to pass traffic_lights_dict straight to
    idm_tick for cyclist_paths — a DIFFERENT path list than the car_paths the
    lights were built from. can_enter(path_idx) then checked cyclist path
    index against car-path-index green groups: pure coincidence. Proves the
    new bearing-matched view (traffic_lights.build_external_light_views)
    gives the geometrically correct answer where the raw car light, indexed
    by the same colliding integer, would have gotten it backwards."""
    from traffic_lights import build_traffic_lights, build_external_light_views

    g = _make_cross_graph(arm_len=120.0)
    car_paths, _ = paths_from_graph(g)
    lights = build_traffic_lights(g, car_paths, min_degree=3)
    center = next(iter(lights))
    light = lights[center]
    assert light.state == "green" and len(light.green_groups) == 2

    green_group = light.green_groups[light.phase % 2]
    red_group = light.green_groups[(light.phase + 1) % 2]
    assert green_group and red_group, "test fixture needs both axes populated"
    green_idx = green_group[0]
    red_car_path = car_paths[red_group[0]]

    # A cyclist path list, unrelated to car_paths, whose one real entry (a
    # cyclist approaching `center` on the currently-RED axis) is deliberately
    # padded to land at the exact index a currently-GREEN car path occupies.
    dummy = {"u": -1, "v": -1}
    cyclist_paths = [dummy] * green_idx + [{"u": red_car_path["u"], "v": center}]
    real_idx = len(cyclist_paths) - 1
    assert real_idx == green_idx, "test setup: index collision must be exact"

    view = build_external_light_views(g, cyclist_paths, lights)
    assert view[center].can_enter(real_idx) is False, (
        "cyclist approaching on the red axis was allowed through — bearing gating broken"
    )
    assert light.can_enter(real_idx) is True, (
        "test setup: the raw car light must (wrongly) allow this index, proving the "
        "old direct-reuse pattern was a real bug, not just theoretical"
    )


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
