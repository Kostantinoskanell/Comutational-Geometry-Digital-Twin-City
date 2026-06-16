"""turn_restrictions.py — Turn restriction filtering for car_next_edges.

Two data sources are supported and composed additively:

Overture
    Connector features carry ``prohibited_transitions``:
    ``[(from_segment_id, to_segment_id), ...]`` pairs at each junction.
    ``overture_source._fetch_overture_connectors`` stores these as
    ``graph.graph['prohibited_turn_pairs']``.

OSM (osmnx)
    Some graph edges carry a ``restriction`` attribute from the original
    OSM turn-restriction relation (e.g. ``no_left_turn``, ``no_u_turn``).
    Bearing arithmetic classifies the outgoing edge as left/right/straight/
    u-turn and the matching exits are filtered out.

Public API
----------
build_next_edges(car_paths, car_outgoing, graph) -> list[np.ndarray]
    Drop-in replacement for the car_next_edges construction in main.py.
    Applies all available restrictions then falls back to non-reverse-only
    filtering (identical to old behaviour) if no data is present.

build_forbidden_turns(car_paths, car_outgoing, graph) -> set[tuple[int,int]]
    Return the raw (from_path_idx, to_path_idx) forbidden set without
    touching car_next_edges.  Useful for diagnostics.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import osmnx as ox

if TYPE_CHECKING:
    import networkx as nx  # noqa: F401


# ── OSM restriction tag vocabulary ───────────────────────────────────────────
_PROHIBITING: frozenset[str] = frozenset({
    "no_left_turn", "no_right_turn", "no_straight_on",
    "no_u_turn", "no_entry", "no_turn",
})
_MANDATING: frozenset[str] = frozenset({
    "only_left_turn", "only_right_turn", "only_straight_on", "only_u_turn",
})


def _cache_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def load_osm_direction_graph_cached(
    bbox_wsen: tuple[float, float, float, float],
    target_crs,
    cache_dir: Path | None,
    use_cache: bool,
    cache_context_key: str,
) -> nx.MultiDiGraph:
    """Fetch and cache an OSM drive graph used only for road-direction lookup."""
    bbox_wsen = tuple(float(v) for v in bbox_wsen)
    crs_key = str(target_crs) if target_crs is not None else "default"
    cache_key = _cache_key("osm-direction", cache_context_key, bbox_wsen, crs_key)
    cache_path = None if cache_dir is None else cache_dir / f"osm_dir_{cache_key}.pkl"

    if use_cache and cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            graph = pickle.load(f)
        print(f"[turn-restr] Loaded OSM direction cache: {cache_path.name}")
        return graph

    raw = ox.graph_from_bbox(bbox_wsen, network_type="drive", retain_all=True)
    graph = ox.projection.project_graph(raw, to_crs=target_crs) if target_crs is not None else ox.projection.project_graph(raw)
    graph.graph["source_bbox_wsen"] = bbox_wsen
    graph.graph["direction_only"] = True

    if use_cache and cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"[turn-restr] Saved OSM direction cache: {cache_path.name}")

    return graph


def resolve_osm_road_direction(
    source_start_xy: tuple[float, float],
    source_end_xy: tuple[float, float],
    osm_direction_graph,
) -> tuple[bool, bool]:
    """Return (is_oneway, legal_forward) using the cached OSM direction graph.

    ``legal_forward`` is True when the source edge's stored direction already
    matches the legal OSM direction.  If it is False, callers should reverse
    the geometry before storing the edge so the graph's u→v edge is legal.
    """
    if osm_direction_graph is None:
        return None, None

    try:
        u_osm = ox.distance.nearest_nodes(
            osm_direction_graph,
            X=float(source_start_xy[0]),
            Y=float(source_start_xy[1]),
        )
        v_osm = ox.distance.nearest_nodes(
            osm_direction_graph,
            X=float(source_end_xy[0]),
            Y=float(source_end_xy[1]),
        )
    except Exception:
        return None, None

    forward_exists = bool(osm_direction_graph.has_edge(u_osm, v_osm))
    reverse_exists = bool(osm_direction_graph.has_edge(v_osm, u_osm))

    if forward_exists and reverse_exists:
        return False, True
    if forward_exists:
        return True, True
    if reverse_exists:
        return True, False
    return None, None


# ── Bearing helpers ───────────────────────────────────────────────────────────

def _bearing(x0: float, y0: float, x1: float, y1: float) -> float:
    """Compass bearing (0 = north, clockwise) of the vector (x0,y0)→(x1,y1)."""
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _turn_dir(in_bearing: float, out_bearing: float) -> str:
    """Classify a turn as 'left', 'right', 'straight', or 'u_turn'."""
    diff = (out_bearing - in_bearing + 360.0) % 360.0
    if diff < 30.0 or diff > 330.0:
        return "straight"
    if diff < 150.0:
        return "right"
    if diff > 210.0:
        return "left"
    return "u_turn"


# ── Core restriction builder ──────────────────────────────────────────────────

def build_forbidden_turns(
    car_paths:    list[dict],
    car_outgoing: dict,
    graph,
    include_overture: bool = True,
) -> set[tuple[int, int]]:
    """Return a set of (from_path_idx, to_path_idx) pairs that are forbidden.

    Sources tried in order when enabled:

    1. ``graph.graph['prohibited_turn_pairs']``
       Set of ``(from_segment_id, to_segment_id)`` str pairs stored by
       ``overture_source._fetch_overture_connectors``.  Mapped to path
       indices via the ``segment_id`` key on each car_path.

    2. ``data['restriction']`` on graph edges (OSM / osmnx).
       Each restriction string is resolved using bearing arithmetic against
       all outgoing edges from the via-node.

    ``include_overture=False`` disables source 1.  Falls back gracefully to an
    empty set if no enabled source has data.
    """
    forbidden: set[tuple[int, int]] = set()

    # ── Source 1: Overture prohibited_turn_pairs ──────────────────────────────
    overture_pairs: set[tuple[str, str]] = (
        graph.graph.get("prohibited_turn_pairs", set()) if include_overture else set()
    )
    if overture_pairs:
        seg_to_paths: dict[str, list[int]] = {}
        for i, p in enumerate(car_paths):
            sid = p.get("segment_id")
            if sid:
                seg_to_paths.setdefault(str(sid), []).append(i)

        n_before = len(forbidden)
        for from_seg, to_seg in overture_pairs:
            for fi in seg_to_paths.get(from_seg, []):
                for ti in seg_to_paths.get(to_seg, []):
                    forbidden.add((fi, ti))

        added = len(forbidden) - n_before
        print(
            f"[turn-restr] Overture: {len(overture_pairs)} connector transitions "
            f"→ {added} path-pair restrictions"
        )

    # ── Source 2: OSM 'restriction' edge attributes ───────────────────────────
    # Build lookup tables
    uv_to_paths: dict[tuple, list[int]] = {}
    for i, p in enumerate(car_paths):
        uv_to_paths.setdefault((p["u"], p["v"]), []).append(i)

    node_data: dict = {nid: d for nid, d in graph.nodes(data=True)}

    n_osm = 0
    for u, v, data in graph.edges(data=True):
        raw_restr = data.get("restriction")
        if not raw_restr:
            continue

        if isinstance(raw_restr, (list, tuple)):
            restr_list = [str(r).lower() for r in raw_restr]
        else:
            restr_list = [str(raw_restr).lower()]

        from_paths = uv_to_paths.get((u, v), [])
        if not from_paths:
            continue

        nu = node_data.get(u, {})
        nv = node_data.get(v, {})
        if "x" not in nu or "x" not in nv:
            continue

        in_bear = _bearing(
            float(nu["x"]), float(nu["y"]),
            float(nv["x"]), float(nv["y"]),
        )

        outgoing_from_v = list(car_outgoing.get(v, []))

        for restr in restr_list:
            if restr not in _PROHIBITING and restr not in _MANDATING:
                continue

            # Determine which direction is forbidden / mandatory
            if restr == "no_left_turn":
                forbidden_dir, mandatory_dir = "left", None
            elif restr == "no_right_turn":
                forbidden_dir, mandatory_dir = "right", None
            elif restr == "no_straight_on":
                forbidden_dir, mandatory_dir = "straight", None
            elif restr in {"no_u_turn", "no_turn"}:
                forbidden_dir, mandatory_dir = "u_turn", None
            elif restr == "no_entry":
                forbidden_dir, mandatory_dir = None, None   # block ALL exits
            elif restr == "only_straight_on":
                forbidden_dir, mandatory_dir = None, "straight"
            elif restr == "only_left_turn":
                forbidden_dir, mandatory_dir = None, "left"
            elif restr == "only_right_turn":
                forbidden_dir, mandatory_dir = None, "right"
            else:
                continue

            for to_path_idx in outgoing_from_v:
                tp = car_paths[to_path_idx]
                nw = node_data.get(tp["v"], {})
                if "x" not in nw:
                    continue

                out_bear = _bearing(
                    float(nv["x"]), float(nv["y"]),
                    float(nw["x"]), float(nw["y"]),
                )
                direction = _turn_dir(in_bear, out_bear)

                block = False
                if restr == "no_entry":
                    block = True
                elif forbidden_dir and direction == forbidden_dir:
                    block = True
                elif mandatory_dir and direction != mandatory_dir:
                    block = True

                if block:
                    for fi in from_paths:
                        forbidden.add((fi, to_path_idx))
                    n_osm += 1

    if n_osm > 0:
        print(f"[turn-restr] OSM: {n_osm} restricted exit(s) from 'restriction' edge attrs")
    elif not overture_pairs:
        print("[turn-restr] No turn restriction data found in graph — using non-reverse filter only")

    return forbidden


def build_next_edges(
    car_paths:    list[dict],
    car_outgoing: dict,
    graph,
    include_overture: bool = True,
) -> list[np.ndarray]:
    """Build car_next_edges with enabled turn restrictions applied.

    Replacement for the manual loop in main.py:

        # OLD
        car_next_edges = []
        for p in car_paths:
            candidates = list(car_outgoing.get(p["v"], []))
            if len(candidates) > 1:
                non_reverse = [j for j in candidates if car_paths[j]["v"] != p["u"]]
                if non_reverse:
                    candidates = non_reverse
            car_next_edges.append(np.asarray(candidates, dtype=np.int64))

        # NEW
        car_next_edges = turn_restrictions.build_next_edges(
            car_paths, car_outgoing, street_graph
        )

    Filtering order (most permissive → most restrictive):
        1. Non-reverse  — never go straight back the way you came (geometry)
        2. Explicit turn restrictions (OSM, plus Overture when enabled)
        Safety net: keep at least one exit only when filtering restrictions
        would otherwise remove every candidate.
    """
    forbidden = build_forbidden_turns(
        car_paths,
        car_outgoing,
        graph,
        include_overture=include_overture,
    )

    n_restricted = 0
    car_next_edges: list[np.ndarray] = []

    for i, p in enumerate(car_paths):
        candidates = list(car_outgoing.get(p["v"], []))

        # ── Step 1: non-reverse filter (U-turn suppression) ───────────────
        if len(candidates) > 1:
            non_rev = [j for j in candidates if car_paths[int(j)]["v"] != p["u"]]
            if non_rev:
                candidates = non_rev

        # ── Step 2: explicit turn restriction filter ───────────────────────
        if forbidden:
            allowed = [j for j in candidates if (i, j) not in forbidden]
            # Safety net: never leave a car with zero exits
            if allowed:
                n_restricted += len(candidates) - len(allowed)
                candidates = allowed

        car_next_edges.append(np.asarray(candidates, dtype=np.int64))

    total_exits = sum(len(arr) for arr in car_next_edges)
    print(
        f"[turn-restr] built car_next_edges: {len(car_paths)} paths, "
        f"{total_exits} exits, {n_restricted} turn(s) restricted"
    )
    return car_next_edges
