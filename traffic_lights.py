"""Traffic light FSM for city digital twin.

Each intersection node in the street graph gets a TrafficLight instance.
Cars query can_enter(edge_idx) before advancing onto a new edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv


# ── Phase timing (seconds) ────────────────────────────────────────────────
GREEN_DURATION  = 18.0
YELLOW_DURATION =  4.0
ALL_RED_PAUSE   =  1.5   # brief all-red between phases for safety

_PHASE_SEQ = ["green", "yellow", "all_red"]   # repeats per phase group


# ── Geometry helpers ──────────────────────────────────────────────────────

def _edge_bearing(graph, u, v) -> float:
    """Compass bearing (degrees, 0=north, clockwise) of edge u→v."""
    nu = graph.nodes[u]
    nv = graph.nodes[v]
    dx = float(nv["x"]) - float(nu["x"])
    dy = float(nv["y"]) - float(nu["y"])
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def _group_edges_by_axis(
    incoming_edges: list[tuple],   # [(path_idx, u, v, bearing), ...]
    n_phases: int = 2,
) -> list[list[int]]:
    """
    Split incoming edge path-indices into n_phases groups by bearing similarity.
    Works for 4-way (2 phases) and T-junctions (still 2 phases, one smaller).
    """
    if not incoming_edges:
        return [[] for _ in range(n_phases)]

    bearings = np.array([e[3] for e in incoming_edges], dtype=float)
    path_indices = [e[0] for e in incoming_edges]

    # Normalise to [0, 180) so opposite directions collapse to same axis
    axes = bearings % 180

    # k-means with k=n_phases on the circle [0, 180)
    # Simple approach: sort and split at largest gap
    order = np.argsort(axes)
    sorted_axes = axes[order]
    sorted_paths = [path_indices[i] for i in order]

    if n_phases == 1 or len(sorted_axes) <= 1:
        return [sorted_paths] + [[] for _ in range(n_phases - 1)]

    # Find the n_phases-1 largest circular gaps
    gaps = np.diff(np.concatenate([sorted_axes, sorted_axes[:1] + 180]))
    split_after = np.argsort(gaps)[::-1][:n_phases - 1]
    split_after = sorted(split_after)

    groups: list[list[int]] = []
    prev = 0
    for sp in split_after:
        groups.append(sorted_paths[prev : sp + 1])
        prev = sp + 1
    groups.append(sorted_paths[prev:])
    return groups


# ── Core FSM ─────────────────────────────────────────────────────────────

@dataclass
class TrafficLight:
    """
    FSM for one intersection node.

    green_groups[i] = list of path indices that may proceed during phase i.
    """

    node_id: object
    x: float
    y: float
    green_groups: List[List[int]]        # path indices per phase
    green_durations: List[float] = field(default_factory=lambda: [18.0, 18.0])
    yellow_durations: List[float] = field(default_factory=lambda: [4.0, 4.0])
    n_phases: int = 2
    phase: int = 0                       # current phase index
    state: str = "green"                 # "green" | "yellow" | "all_red"
    elapsed: float = 0.0
    offset: float = 0.0                  # stagger so not all lights change at once
    approach_points: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    # Mean approach bearing (deg, mod 180 so opposite directions match) of each
    # phase's car-path group, set by build_traffic_lights. Lets a DIFFERENT
    # agent class (cyclists, ...), whose own path list uses unrelated indices,
    # be gated by this same light via its geometry instead of a path index —
    # see can_enter_by_bearing / build_external_light_views.
    phase_axis_bearing: list = field(default_factory=list, repr=False)

    # Derived from green_groups at build time
    _allowed: set = field(default_factory=set, repr=False)
    controlled_paths: frozenset = field(init=False)

    def __post_init__(self):
        self.controlled_paths = frozenset(idx for group in self.green_groups for idx in group)
        self._refresh_allowed()
        # Advance by the full offset so lights start at different points in their cycle.
        # Use the actual per-phase durations (not the module-level defaults) so the modulo
        # is correct even when green_durations differ from GREEN_DURATION.
        _cycle = sum(self.green_durations) + sum(self.yellow_durations) + self.n_phases * ALL_RED_PAUSE
        self._advance(self.offset % max(_cycle, 1.0))

    def _refresh_allowed(self):
        if self.state == "green":
            idx = self.phase % len(self.green_groups)
            self._allowed = set(self.green_groups[idx])
        else:
            self._allowed = set()

    def _advance(self, dt: float):
        """Advance clock by dt, draining through as many state transitions as needed."""
        self.elapsed += dt
        for _ in range(100):
            old = self.elapsed
            self._tick_inner()
            if abs(self.elapsed - old) < 1e-9:
                break

    def _tick_inner(self):
        idx = self.phase % max(1, len(self.green_groups))
        gd = self.green_durations[idx] if idx < len(self.green_durations) else GREEN_DURATION
        yd = self.yellow_durations[idx] if idx < len(self.yellow_durations) else YELLOW_DURATION
        
        if self.state == "green":
            if self.elapsed >= gd:
                self.elapsed -= gd
                self.state = "yellow"
        elif self.state == "yellow":
            if self.elapsed >= yd:
                self.elapsed -= yd
                self.state = "all_red"
        elif self.state == "all_red":
            if self.elapsed >= ALL_RED_PAUSE:
                self.elapsed -= ALL_RED_PAUSE
                self.phase = (self.phase + 1) % self.n_phases
                self.state = "green"
        self._refresh_allowed()

    def tick(self, dt: float):
        """Advance the FSM by dt simulation seconds."""
        self.elapsed += dt
        self._tick_inner()

    def can_enter(self, path_idx: int) -> bool:
        """Return True if a car on path_idx may proceed through this intersection."""
        if not self._allowed:
            return False
        return path_idx in self._allowed

    def phase_for_bearing(self, bearing: float) -> int:
        """Nearest phase axis (by circular distance mod 180) to `bearing`."""
        if not self.phase_axis_bearing:
            return 0
        axis = bearing % 180
        diffs = [
            (min(abs(axis - a), 180 - abs(axis - a)) if a == a else float("inf"))  # a==a: not NaN
            for a in self.phase_axis_bearing
        ]
        return int(np.argmin(diffs))

    def can_enter_by_bearing(self, bearing: float) -> bool:
        """Like can_enter, but for an agent whose own path index space this
        light knows nothing about — classify its approach direction onto the
        nearest phase axis instead."""
        if self.state != "green" or not self.green_groups:
            return False
        return self.phase_for_bearing(bearing) == self.phase % len(self.green_groups)

    @property
    def color(self) -> str:
        if self.state == "green":
            return "green"
        if self.state == "yellow":
            return "yellow"
        return "red"

    def display_color_for_path(self, path_idx: int) -> str:
        """Signal color for one incoming approach path."""
        if self.state == "green" and path_idx in self._allowed:
            return "green"
        if self.state == "yellow":
            idx = self.phase % len(self.green_groups)
            if path_idx in set(self.green_groups[idx]):
                return "yellow"
        return "red"


# ── Smart phase timing ────────────────────────────────────────────────────

# Green-time bounds (seconds).  Even a tiny side street holds green long enough
# to clear a couple of cars; a fast arterial gets a generous window.
_GREEN_MIN = 14.0
_GREEN_MAX = 45.0
# Yellow-time bounds (seconds).  ITE clearance ≈ reaction + v / (2·decel).
_YELLOW_MIN = 3.0
_YELLOW_MAX = 5.5
_DECEL = 3.0          # m/s² assumed comfortable braking for yellow calc
_REACTION = 1.0       # s driver reaction time


def _group_speed_ms(group: list[int], car_paths: list[dict]) -> float:
    """Max posted speed (m/s) among the paths in one phase group.

    Falls back to 8.33 m/s (30 km/h) for an empty group or missing data.
    """
    best = 0.0
    for path_idx in group:
        if 0 <= path_idx < len(car_paths):
            v = float(car_paths[path_idx].get("maxspeed_ms", 0.0) or 0.0)
            if v > best:
                best = v
    return best if best > 0.0 else 8.33


def _phase_timings(
    groups: list[list[int]],
    car_paths: list[dict],
) -> tuple[list[float], list[float]]:
    """Return (green_durations, yellow_durations), one entry per phase group.

    Green time scales linearly with the group's share of total approach speed,
    so the faster axis at the junction holds green longer.  Yellow time follows
    the ITE clearance formula t_y = reaction + v / (2·decel), clamped to a sane
    urban band.
    """
    speeds = [_group_speed_ms(g, car_paths) for g in groups]
    total = sum(speeds) if speeds else 0.0

    green_durations: list[float] = []
    yellow_durations: list[float] = []
    for v in speeds:
        if total > 1e-6:
            share = v / total                       # 0..1, sums to 1 over phases
            green = _GREEN_MIN + (_GREEN_MAX - _GREEN_MIN) * share
        else:
            green = (_GREEN_MIN + _GREEN_MAX) * 0.5
        green_durations.append(float(np.clip(green, _GREEN_MIN, _GREEN_MAX)))

        yellow = _REACTION + v / (2.0 * _DECEL)
        yellow_durations.append(float(np.clip(yellow, _YELLOW_MIN, _YELLOW_MAX)))

    return green_durations, yellow_durations


# ── Build traffic lights from graph ──────────────────────────────────────

def build_traffic_lights(
    graph,
    car_paths: list[dict],
    min_degree: int = 3,
    traffic_speed: float = 1.0,
) -> Dict[object, TrafficLight]:
    """
    Create a TrafficLight for every intersection node with degree >= min_degree.

    car_paths  : the same list produced by _extract_drivable_paths in main.py
    min_degree : skip dead-ends and simple through-roads (degree < 3)

    Returns dict mapping node_id → TrafficLight.
    """
    # Index paths by their destination node
    paths_entering: dict[object, list[tuple]] = {}
    for idx, path in enumerate(car_paths):
        v = path["v"]
        u = path["u"]
        nu = graph.nodes.get(u, {})
        nv = graph.nodes.get(v, {})
        if "x" not in nu or "x" not in nv:
            continue
        bearing = _edge_bearing(graph, u, v)
        paths_entering.setdefault(v, []).append((idx, u, v, bearing))

    lights: Dict[object, TrafficLight] = {}
    rng = np.random.default_rng(0)

    for node_id, in_edges in paths_entering.items():
        ndata = graph.nodes.get(node_id, {})
        control_type = ndata.get("highway")
        
        # In this smarter logic, we only build traffic signals if explicitly tagged by OSM 
        # (or added via the editor). We ignore the min_degree rule unless it's a fallback.
        # But wait! If the user wants actual logic and Stop signs, we should allow them.
        if control_type == "traffic_signals":
            pass  # explicit OSM traffic signal
        elif control_type == "stop":
            continue  # stop signs handled by IDM stop-wait logic, not FSM
        else:
            # Fallback: auto-place a light at any significant intersection (3+ incoming paths)
            # that isn't explicitly tagged. This ensures lights appear regardless of OSM
            # tagging density in the loaded area.
            if len(in_edges) < 3:
                continue
            
        n_phases = 2 if len(in_edges) >= 3 else 1
        groups = _group_edges_by_axis(in_edges, n_phases=n_phases)
        if n_phases == 1:
            # Mid-block / straight-road signal (≤2 approaches on one axis,
            # e.g. an editor-placed light on a plain road).  A single-phase
            # light is green for the whole cycle minus the ~1.5 s all-red
            # pause — useless.  Model it as a pedestrian crossing instead:
            # alternate cars-green with an all-red phase.  The empty group
            # gets share=0 in _phase_timings → _GREEN_MIN (14 s) of red.
            groups = [groups[0], []]
            n_phases = 2

        # ── Smart time windows from real per-phase road speed ────────────────
        # car_paths IS available here (it's a parameter), so look up the actual
        # maxspeed_ms of each path in each phase group.  The faster/busier axis
        # gets a proportionally longer green; faster approaches get a longer
        # yellow (clearance time grows with approach speed).
        green_durations, yellow_durations = _phase_timings(groups, car_paths)

        # Mean bearing (mod 180, circular) of each phase's approaches, for
        # can_enter_by_bearing — computed from the SAME in_edges bearings
        # used to build `groups`, so it stays consistent even after the
        # single-phase -> pedestrian-crossing rewrite above (empty group ->
        # no bearing, phase_for_bearing just never selects it).
        _bearing_by_idx = {idx: b for idx, _u, _v, b in in_edges}
        phase_axis_bearing = []
        for group in groups:
            if not group:
                phase_axis_bearing.append(float("nan"))
                continue
            axes = np.array([_bearing_by_idx[i] % 180 for i in group], dtype=float)
            # Circular mean on [0, 180): double the angle, average, halve back.
            ang = np.deg2rad(axes * 2)
            m = math.degrees(math.atan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) / 2.0
            phase_axis_bearing.append(m % 180)

        ndata = graph.nodes.get(node_id, {})
        _cycle = sum(green_durations) + sum(yellow_durations) + n_phases * ALL_RED_PAUSE
        light = TrafficLight(
            node_id=node_id,
            x=float(ndata.get("x", 0.0)),
            y=float(ndata.get("y", 0.0)),
            green_groups=groups,
            green_durations=green_durations,
            yellow_durations=yellow_durations,
            n_phases=n_phases,
            offset=float(rng.uniform(0, _cycle)),
            phase_axis_bearing=phase_axis_bearing,
        )
        approach_points: dict[int, tuple[float, float, float]] = {}
        vx = float(ndata.get("x", 0.0))
        vy = float(ndata.get("y", 0.0))
        for path_idx, u, _v, _bearing in in_edges:
            udata = graph.nodes.get(u, {})
            ux = float(udata.get("x", vx))
            uy = float(udata.get("y", vy))
            vec = np.array([ux - vx, uy - vy], dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            approach_points[int(path_idx)] = (
                vx + float(vec[0]) * 4.0,
                vy + float(vec[1]) * 4.0,
                _LIGHT_Z,
            )
        light.approach_points = approach_points
        lights[node_id] = light

    print(f"[tl] {len(lights)} traffic lights created at degree≥{min_degree} nodes")
    return lights


class _ExternalPathLightView:
    """Adapts a real TrafficLight for an agent class with its OWN path index
    space (e.g. cyclist_paths, unrelated to car_paths) — gates it by matching
    each of its own path's approach bearing to the light's nearest phase axis
    instead of by (meaningless, cross-list-collision-prone) path index.

    Exposes exactly the interface idm.py's find_leaders needs
    (`.controlled_paths`, `.can_enter(path_idx)`), so it drops into the
    `traffic_lights` dict argument of idm_tick unchanged.
    """
    __slots__ = ("controlled_paths", "_light", "_bearing_by_idx")

    def __init__(self, light: "TrafficLight", bearing_by_idx: dict[int, float]):
        self._light = light
        self._bearing_by_idx = bearing_by_idx
        self.controlled_paths = frozenset(bearing_by_idx)

    def can_enter(self, path_idx: int) -> bool:
        bearing = self._bearing_by_idx.get(path_idx)
        if bearing is None:
            return True
        return self._light.can_enter_by_bearing(bearing)


def build_external_light_views(
    graph, paths: list[dict], lights: Dict[object, TrafficLight],
) -> Dict[object, _ExternalPathLightView]:
    """Build `traffic_lights`-shaped views of the existing car-light FSMs for
    a second agent class whose `paths` list (own u/v/index space) is unrelated
    to the car_paths the lights were built from. See _ExternalPathLightView."""
    entering: dict[object, dict[int, float]] = {}
    for idx, path in enumerate(paths):
        v = path["v"]
        if v not in lights:
            continue
        u = path["u"]
        if "x" not in graph.nodes.get(u, {}) or "x" not in graph.nodes.get(v, {}):
            continue
        entering.setdefault(v, {})[idx] = _edge_bearing(graph, u, v)

    return {
        node_id: _ExternalPathLightView(lights[node_id], bearing_by_idx)
        for node_id, bearing_by_idx in entering.items()
    }


def tick_all(lights: Dict[object, TrafficLight], dt: float, traffic_speed: float):
    """Advance every traffic light by dt * traffic_speed seconds."""
    sim_dt = dt * traffic_speed
    for light in lights.values():
        light.tick(sim_dt)


# ── PyVista glyph rendering ───────────────────────────────────────────────

_COLOR_MAP = {"green": [0, 200, 80], "yellow": [255, 210, 0], "red": [220, 40, 40]}
_POLE_Z   = 0.0
_LIGHT_Z  = 4.5   # top of pole, above buildings in typical 150 m radius scene


def build_light_mesh(lights: Dict[object, TrafficLight]) -> pv.PolyData:
    """Build a point cloud with RGB color array for glyph rendering."""
    if not lights:
        return pv.PolyData()

    pts_list: list[tuple[float, float, float]] = []
    color_list: list[list[int]] = []
    for light in lights.values():
        if light.approach_points:
            for path_idx, point in light.approach_points.items():
                pts_list.append(point)
                color_list.append(_COLOR_MAP[light.display_color_for_path(int(path_idx))])
        else:
            pts_list.append((light.x, light.y, _LIGHT_Z))
            color_list.append(_COLOR_MAP[light.color])

    pts = np.array(pts_list, dtype=float)
    colors = np.array(color_list, dtype=np.uint8)
    mesh = pv.PolyData(pts)
    mesh["colors"] = colors
    return mesh


def update_light_mesh(mesh: pv.PolyData, lights: Dict[object, TrafficLight]):
    """Update colors in-place — call each animation tick."""
    if mesh.n_points == 0:
        return
    colors = np.zeros((mesh.n_points, 3), dtype=np.uint8)
    i = 0
    for light in lights.values():
        if light.approach_points:
            for path_idx in light.approach_points:
                colors[i] = _COLOR_MAP[light.display_color_for_path(int(path_idx))]
                i += 1
        else:
            colors[i] = _COLOR_MAP[light.color]
            i += 1
    mesh["colors"] = colors


def build_light_glyphs(mesh: pv.PolyData, geom: pv.PolyData) -> pv.PolyData:
    """Glyph the light mesh and guarantee the 'colors' array survives.

    vtkGlyph3D silently drops point-data arrays on some VTK builds unless
    the array is the active scalars.  We set active scalars first, then
    broadcast manually as a fallback if the array is still missing.
    """
    if mesh.n_points == 0:
        return pv.PolyData()
    mesh.set_active_scalars("colors")
    glyphs = mesh.glyph(geom=geom, scale=False, factor=1.0)
    # Fallback: if vtkGlyph3D dropped the array, broadcast it ourselves.
    # Each input point maps to geom.n_points output points.
    if "colors" not in glyphs.point_data:
        n_per = geom.n_points
        src = np.asarray(mesh["colors"], dtype=np.uint8)   # (n_lights, 3)
        glyphs["colors"] = np.repeat(src, n_per, axis=0)  # (n_lights * n_per, 3)
    return glyphs
