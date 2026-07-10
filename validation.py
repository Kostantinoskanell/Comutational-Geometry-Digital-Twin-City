"""validation.py — travel-time validation against an external routing engine.

Turns "the simulation looks plausible" into "the simulation is within X% of a
reference router."  Samples random origin–destination node pairs from the loaded
street graph, computes the model's free-flow route time, queries a routing
engine for the same trip, and reports error metrics.

Default engine is the public OSRM demo server (no API key, driving profile).
Mapbox / TomTom / HERE can be slotted in via `engine_fn` for traffic-aware
comparison, but those need keys.

Metrics
-------
  • MAPE   — mean absolute percentage error of travel time
  • RMSE   — root-mean-square error (seconds)
  • bias   — mean(sim − ref), seconds (sign shows systematic over/under-estimate)
  • r      — Pearson correlation of sim vs ref times
  • GEH    — borrowed from traffic engineering; GEH<5 on most pairs = good match

Graceful degradation: no network / no `requests` / engine errors → those pairs
are skipped and the reason is reported; the app never crashes.

Public API
----------
run_validation(graph, n_pairs=40, engine="osrm", osrm_host=..., seed=42,
               out_path=None) -> dict
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request

import numpy as np

try:
    import networkx as nx
    _NX_OK = True
except Exception:
    _NX_OK = False


# ── Edge free-flow speed (mirrors the app's highway-type fallback) ───────────

_HIGHWAY_SPEED_MS = {
    "motorway": 33.3, "trunk": 22.2, "primary": 13.9, "secondary": 13.9,
    "tertiary": 11.1, "residential": 8.3, "living_street": 5.6,
    "unclassified": 11.1, "service": 5.6,
}


def _parse_maxspeed_ms(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals = [_parse_maxspeed_ms(r) for r in raw]
        vals = [v for v in vals if v]
        return max(vals) if vals else None
    try:
        s = str(raw).strip().lower()
        if "mph" in s:
            return float(s.split()[0]) * 0.44704
        return float(s.split()[0]) / 3.6   # km/h → m/s
    except Exception:
        return None


def _edge_speed_ms(data: dict) -> float:
    ms = _parse_maxspeed_ms(data.get("maxspeed"))
    if ms is not None:
        return float(np.clip(ms, 2.8, 41.7))
    hw = data.get("highway")
    if isinstance(hw, (list, tuple)):
        hw = hw[0] if hw else None
    return _HIGHWAY_SPEED_MS.get(str(hw), 11.1)


def _annotate_traveltime(graph) -> None:
    """Add a 'tt' (free-flow seconds) weight to every edge, once."""
    for u, v, data in graph.edges(data=True):
        length = float(data.get("length", 0.0) or 0.0)
        if length <= 0.0:
            nu, nv = graph.nodes[u], graph.nodes[v]
            length = math.hypot(
                float(nv.get("x", 0.0)) - float(nu.get("x", 0.0)),
                float(nv.get("y", 0.0)) - float(nu.get("y", 0.0)),
            )
        data["tt"] = length / _edge_speed_ms(data)


# ── OSRM reference engine ────────────────────────────────────────────────────

def osrm_duration(o_lonlat, d_lonlat, host="https://router.project-osrm.org",
                  timeout=10.0) -> float | None:
    """Return driving duration (s) between two (lon, lat) points, or None."""
    coords = f"{o_lonlat[0]:.6f},{o_lonlat[1]:.6f};{d_lonlat[0]:.6f},{d_lonlat[1]:.6f}"
    url = f"{host}/route/v1/driving/{coords}?overview=false&alternatives=false"
    try:
        try:
            import requests
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except ImportError:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        return float(data["routes"][0]["duration"])
    except Exception:
        return None


# ── Metrics ──────────────────────────────────────────────────────────────────

def _metrics(sim: np.ndarray, ref: np.ndarray) -> dict:
    sim = np.asarray(sim, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = len(sim)
    if n == 0:
        return {"n": 0}
    err = sim - ref
    mape = float(np.mean(np.abs(err) / np.maximum(ref, 1e-6)) * 100.0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    r = float(np.corrcoef(sim, ref)[0, 1]) if n >= 2 and np.std(sim) > 0 and np.std(ref) > 0 else float("nan")
    geh = np.sqrt(2.0 * err ** 2 / np.maximum(sim + ref, 1e-6))
    geh_ok = float(np.mean(geh < 5.0) * 100.0)
    return {
        "n": n, "mape_pct": mape, "rmse_s": rmse, "bias_s": bias,
        "pearson_r": r, "geh_under5_pct": geh_ok,
        "sim_mean_s": float(np.mean(sim)), "ref_mean_s": float(np.mean(ref)),
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_validation(
    graph,
    n_pairs: int = 40,
    engine: str = "osrm",
    osrm_host: str = "https://router.project-osrm.org",
    seed: int = 42,
    min_dist_m: float = 200.0,
    max_dist_m: float = 2500.0,
    out_path: str | None = None,
    engine_fn=None,
) -> dict:
    """Compare model free-flow route times against a reference engine.

    Returns a report dict; also writes JSON to `out_path` if given.
    """
    if not _NX_OK:
        return {"error": "networkx not available"}

    crs = str(graph.graph.get("crs", "")) or str(graph.graph.get("proj_str", ""))
    if not crs:
        return {"error": "graph has no projected CRS — cannot georeference O-D pairs"}

    try:
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception as exc:
        return {"error": f"pyproj transform unavailable: {exc}"}

    _annotate_traveltime(graph)

    nodes = [n for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if len(nodes) < 4:
        return {"error": "graph too small to sample O-D pairs"}

    rng = np.random.default_rng(seed)
    xy = {n: (float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in nodes}

    ref_fn = engine_fn or (lambda o, d: osrm_duration(o, d, host=osrm_host))

    results = []
    attempts = 0
    max_attempts = n_pairs * 12
    print(f"[validate] sampling up to {n_pairs} O-D pairs, engine={engine} …")

    while len(results) < n_pairs and attempts < max_attempts:
        attempts += 1
        o, d = rng.choice(len(nodes), size=2, replace=False)
        o_id, d_id = nodes[int(o)], nodes[int(d)]
        ox_, oy_ = xy[o_id]; dx_, dy_ = xy[d_id]
        straight = math.hypot(dx_ - ox_, dy_ - oy_)
        if straight < min_dist_m or straight > max_dist_m:
            continue

        # Model free-flow route time
        try:
            sim_t = float(nx.shortest_path_length(graph, o_id, d_id, weight="tt"))
        except Exception:
            continue
        if not math.isfinite(sim_t) or sim_t <= 0.0:
            continue

        # Reference route time
        o_ll = to_wgs84.transform(ox_, oy_)
        d_ll = to_wgs84.transform(dx_, dy_)
        ref_t = ref_fn(o_ll, d_ll)
        if ref_t is None or ref_t <= 0.0:
            continue

        results.append({
            "o": str(o_id), "d": str(d_id),
            "straight_m": round(straight, 1),
            "sim_s": round(sim_t, 1), "ref_s": round(ref_t, 1),
        })
        # Be polite to the public OSRM demo server.
        if engine == "osrm" and engine_fn is None:
            time.sleep(0.05)

    report = {
        "engine": engine,
        "requested_pairs": n_pairs,
        "matched_pairs": len(results),
        "attempts": attempts,
        "metrics": _metrics(
            np.array([r["sim_s"] for r in results]),
            np.array([r["ref_s"] for r in results]),
        ),
        "pairs": results,
    }

    _print_report(report)

    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[validate] report written to {out_path}")
        except Exception as exc:
            print(f"[validate] could not write report: {exc}")

    return report


def _print_report(report: dict) -> None:
    m = report.get("metrics", {})
    print("\n" + "=" * 60)
    print("  TRAVEL-TIME VALIDATION  (model free-flow vs "
          f"{report['engine']})")
    print("=" * 60)
    if not m or m.get("n", 0) == 0:
        print("  No comparable O-D pairs (network/engine unavailable?).")
        print("=" * 60 + "\n")
        return
    print(f"  pairs matched : {m['n']}  (of {report['requested_pairs']} requested)")
    print(f"  sim mean      : {m['sim_mean_s']:.0f} s     ref mean: {m['ref_mean_s']:.0f} s")
    print(f"  MAPE          : {m['mape_pct']:.1f} %")
    print(f"  RMSE          : {m['rmse_s']:.0f} s")
    print(f"  bias (sim−ref): {m['bias_s']:+.0f} s")
    print(f"  Pearson r     : {m['pearson_r']:.3f}")
    print(f"  GEH < 5       : {m['geh_under5_pct']:.0f} % of pairs")
    print("=" * 60 + "\n")


# ── Road-class classification helper ────────────────────────────────────────

_ROAD_CLASS_MAP = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "motorway", "trunk_link": "motorway",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "residential", "living_street": "residential",
    "service": "residential", "unclassified": "residential",
}

def _road_class(data: dict) -> str:
    hw = data.get("highway", "")
    if isinstance(hw, (list, tuple)):
        hw = hw[0] if hw else ""
    return _ROAD_CLASS_MAP.get(str(hw), "other")


def _metrics_by_class(pairs: list[dict], graph) -> dict[str, dict]:
    """Group pair metrics by road class of the shortest path's dominant edge."""
    by_class: dict[str, list] = {}
    for p in pairs:
        try:
            path_nodes = nx.shortest_path(graph, p["o"], p["d"], weight="tt")
            classes = []
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                edata = graph.get_edge_data(a, b)
                if edata:
                    # MultiDiGraph: get best edge
                    if isinstance(edata, dict) and all(isinstance(v, dict) for v in edata.values()):
                        d = min(edata.values(), key=lambda x: x.get("tt", float("inf")))
                    else:
                        d = edata
                    classes.append(_road_class(d))
            dominant = max(set(classes), key=classes.count) if classes else "other"
        except Exception:
            dominant = "other"
        by_class.setdefault(dominant, []).append((p["sim_s"], p["ref_s"]))

    result = {}
    for cls, vals in by_class.items():
        sims = np.array([v[0] for v in vals])
        refs = np.array([v[1] for v in vals])
        result[cls] = _metrics(sims, refs)
    return result


# ── Congested validation ─────────────────────────────────────────────────────

def validate_congested(
    graph,
    car_anim: dict,
    car_paths: list,
    n_pairs: int = 30,
    engine: str = "osrm",
    osrm_host: str = "https://router.project-osrm.org",
    seed: int = 42,
    out_path: str | None = None,
) -> dict:
    """Validate simulated (IDM) travel times under active demand vs OSRM.

    Unlike run_validation (which uses free-flow times), this samples actual
    simulated speeds from the live car_anim snapshot to estimate congested
    travel time, then compares against OSRM.

    Returns a report dict; writes JSON to out_path if given.
    """
    if not _NX_OK:
        return {"error": "networkx not available"}

    crs = str(graph.graph.get("crs", "")) or str(graph.graph.get("proj_str", ""))
    if not crs:
        return {"error": "graph has no projected CRS"}

    try:
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception as exc:
        return {"error": f"pyproj: {exc}"}

    # Build speed-aware travel time weights from live car_anim
    # For each car_path, use the mean speed of cars on that path (or free-flow if empty)
    edge_idx_arr = np.asarray(car_anim.get("edge_idx", []), dtype=np.int64)
    speed_arr    = np.asarray(car_anim.get("speed",    []), dtype=float)

    path_mean_speed: dict[int, list[float]] = {}
    for i in range(len(edge_idx_arr)):
        e = int(edge_idx_arr[i])
        path_mean_speed.setdefault(e, []).append(float(speed_arr[i]))

    # Annotate edges with congested travel time
    _annotate_traveltime(graph)  # ensures "tt" exists (free-flow fallback)
    for pidx, path in enumerate(car_paths):
        u, v = path.get("u"), path.get("v")
        if u is None or v is None:
            continue
        edata = graph.get_edge_data(u, v)
        if edata is None:
            continue
        speeds = path_mean_speed.get(pidx, [])
        if speeds:
            mean_v = max(0.5, float(np.mean(speeds)))
        else:
            mean_v = float(path.get("maxspeed_ms", 8.3) or 8.3)
        length = float(path.get("length", 0.0) or 0.0)
        cong_tt = length / mean_v if mean_v > 0.01 else float("inf")
        if isinstance(edata, dict) and all(isinstance(val, dict) for val in edata.values()):
            for ed in edata.values():
                ed["tt_cong"] = cong_tt
        else:
            edata["tt_cong"] = cong_tt

    nodes = [n for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if len(nodes) < 4:
        return {"error": "graph too small"}

    rng = np.random.default_rng(seed)
    xy  = {n: (float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in nodes}
    ref_fn = lambda o, d: osrm_duration(o, d, host=osrm_host)

    results = []
    attempts = 0
    max_attempts = n_pairs * 12
    print(f"[validate-cong] sampling up to {n_pairs} O-D pairs under active demand …")

    while len(results) < n_pairs and attempts < max_attempts:
        attempts += 1
        o_i, d_i = rng.choice(len(nodes), size=2, replace=False)
        o_id, d_id = nodes[int(o_i)], nodes[int(d_i)]
        ox, oy = xy[o_id]; dx, dy = xy[d_id]
        straight = math.hypot(dx - ox, dy - oy)
        if straight < 200.0 or straight > 2500.0:
            continue

        try:
            sim_t = float(nx.shortest_path_length(graph, o_id, d_id, weight="tt_cong"))
            path_nodes = list(nx.shortest_path(graph, o_id, d_id, weight="tt_cong"))
        except Exception:
            continue
        if not math.isfinite(sim_t) or sim_t <= 0:
            continue

        # Classify dominant road class for this path
        classes = []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            ed = graph.get_edge_data(a, b)
            if ed:
                if isinstance(ed, dict) and all(isinstance(v, dict) for v in ed.values()):
                    d_ = min(ed.values(), key=lambda x: x.get("tt_cong", float("inf")))
                else:
                    d_ = ed
                classes.append(_road_class(d_))
        dominant_class = max(set(classes), key=classes.count) if classes else "other"

        o_ll = to_wgs84.transform(ox, oy)
        d_ll = to_wgs84.transform(dx, dy)
        ref_t = ref_fn(o_ll, d_ll)
        if ref_t is None or ref_t <= 0:
            continue

        results.append({
            "o": str(o_id), "d": str(d_id),
            "straight_m": round(straight, 1),
            "sim_s": round(sim_t, 1),
            "ref_s": round(ref_t, 1),
            "road_class": dominant_class,
        })
        if engine == "osrm":
            time.sleep(0.05)

    # Overall metrics
    sim_arr = np.array([r["sim_s"] for r in results])
    ref_arr = np.array([r["ref_s"] for r in results])
    overall = _metrics(sim_arr, ref_arr)

    # Per-road-class metrics
    by_class: dict[str, list] = {}
    for r in results:
        by_class.setdefault(r["road_class"], []).append((r["sim_s"], r["ref_s"]))
    per_class = {}
    for cls, vals in by_class.items():
        per_class[cls] = _metrics(
            np.array([v[0] for v in vals]),
            np.array([v[1] for v in vals]),
        )

    report = {
        "mode": "congested",
        "engine": engine,
        "requested_pairs": n_pairs,
        "matched_pairs": len(results),
        "metrics": overall,
        "metrics_by_class": per_class,
        "pairs": results,
    }

    _print_report(report)
    print(f"  Per-class MAPE:")
    for cls, m in per_class.items():
        n_cls = m.get("n", 0)
        if n_cls > 0:
            print(f"    {cls:12s}: MAPE={m.get('mape_pct', float('nan')):.1f}%  "
                  f"RMSE={m.get('rmse_s', float('nan')):.0f}s  "
                  f"GEH<5={m.get('geh_under5_pct', float('nan')):.0f}%  n={n_cls}")

    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[validate-cong] report → {out_path}")
        except Exception as exc:
            print(f"[validate-cong] write failed: {exc}")

    return report


# ── Engine comparison: IDM vs SUMO ──────────────────────────────────────────

def validate_engine_comparison(
    graph,
    car_paths_idm: list,
    car_anim_idm: dict,
    sumo_conn,          # SumoConnection or None
    n_pairs: int = 30,
    seed: int = 42,
    out_path: str | None = None,
) -> dict:
    """Compare travel-time estimates from IDM engine vs SUMO engine.

    IDM travel times are derived from mean car speed per edge (same as
    validate_congested).  SUMO travel times are derived from TraCI's
    edge mean-travel-time query (if available) or fall back to IDM times.

    Saves engine_comparison_<timestamp>.json and returns the report dict.
    Metric definitions follow the same MAPE/RMSE/GEH convention as
    run_validation so results are comparable for academic reporting.
    """
    if not _NX_OK:
        return {"error": "networkx not available"}

    _annotate_traveltime(graph)

    # ── IDM speeds from live car_anim ────────────────────────────────────────
    edge_idx_arr = np.asarray(car_anim_idm.get("edge_idx", []), dtype=np.int64)
    speed_arr    = np.asarray(car_anim_idm.get("speed",    []), dtype=float)
    path_mean_speed: dict[int, list[float]] = {}
    for i in range(len(edge_idx_arr)):
        e = int(edge_idx_arr[i])
        path_mean_speed.setdefault(e, []).append(float(speed_arr[i]))

    for pidx, path in enumerate(car_paths_idm):
        u, v = path.get("u"), path.get("v")
        if u is None or v is None:
            continue
        edata = graph.get_edge_data(u, v)
        if edata is None:
            continue
        speeds = path_mean_speed.get(pidx, [])
        mean_v = max(0.5, float(np.mean(speeds))) if speeds else float(path.get("maxspeed_ms", 8.3) or 8.3)
        length = float(path.get("length", 0.0) or 0.0)
        cong_tt = length / mean_v if mean_v > 0.01 else float("inf")
        if isinstance(edata, dict) and all(isinstance(val, dict) for val in edata.values()):
            for ed in edata.values():
                ed["tt_idm"] = cong_tt
        else:
            edata["tt_idm"] = cong_tt

    # ── SUMO travel times via TraCI ──────────────────────────────────────────
    sumo_edge_tt: dict = {}
    if sumo_conn is not None and getattr(sumo_conn, "ready", False):
        try:
            for eid in sumo_conn._traci.edge.getIDList():
                try:
                    tt = float(sumo_conn._traci.edge.getTraveltime(eid))
                    sumo_edge_tt[eid] = tt
                except Exception:
                    pass
            print(f"[validate-engines] SUMO: {len(sumo_edge_tt)} edge travel times from TraCI")
        except Exception as exc:
            print(f"[validate-engines] SUMO TraCI query failed: {exc}")

    # Annotate graph with SUMO travel times (fall back to IDM if missing)
    for u, v, key, edata in graph.edges(data=True, keys=True):
        osmid = str(edata.get("osmid", ""))
        sumo_tt = sumo_edge_tt.get(osmid) or sumo_edge_tt.get(f"{osmid}#{key}")
        edata["tt_sumo"] = float(sumo_tt) if sumo_tt else float(edata.get("tt_idm", edata.get("tt", 0.0)))

    nodes = [n for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if len(nodes) < 4:
        return {"error": "graph too small"}

    rng = np.random.default_rng(seed)
    xy  = {n: (float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in nodes}

    results = []
    attempts = 0
    max_attempts = n_pairs * 10
    print(f"[validate-engines] comparing IDM vs SUMO over {n_pairs} O-D pairs …")

    while len(results) < n_pairs and attempts < max_attempts:
        attempts += 1
        o_i, d_i = rng.choice(len(nodes), size=2, replace=False)
        o_id, d_id = nodes[int(o_i)], nodes[int(d_i)]
        ox, oy = xy[o_id]; dx, dy = xy[d_id]
        straight = math.hypot(dx - ox, dy - oy)
        if straight < 200.0 or straight > 2500.0:
            continue

        try:
            idm_t  = float(nx.shortest_path_length(graph, o_id, d_id, weight="tt_idm"))
            sumo_t = float(nx.shortest_path_length(graph, o_id, d_id, weight="tt_sumo"))
            path_nodes = list(nx.shortest_path(graph, o_id, d_id, weight="tt_idm"))
        except Exception:
            continue
        if not math.isfinite(idm_t) or not math.isfinite(sumo_t):
            continue
        if idm_t <= 0 or sumo_t <= 0:
            continue

        classes = []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            ed = graph.get_edge_data(a, b)
            if ed:
                if isinstance(ed, dict) and all(isinstance(val, dict) for val in ed.values()):
                    d_ = min(ed.values(), key=lambda x: x.get("tt_idm", float("inf")))
                else:
                    d_ = ed
                classes.append(_road_class(d_))
        dominant_class = max(set(classes), key=classes.count) if classes else "other"

        results.append({
            "o": str(o_id), "d": str(d_id),
            "straight_m": round(straight, 1),
            "idm_s":  round(idm_t,  1),
            "sumo_s": round(sumo_t, 1),
            "road_class": dominant_class,
        })

    idm_arr  = np.array([r["idm_s"]  for r in results])
    sumo_arr = np.array([r["sumo_s"] for r in results])
    overall  = _metrics(idm_arr, sumo_arr)

    by_class: dict[str, list] = {}
    for r in results:
        by_class.setdefault(r["road_class"], []).append((r["idm_s"], r["sumo_s"]))
    per_class = {}
    for cls, vals in by_class.items():
        per_class[cls] = _metrics(
            np.array([v[0] for v in vals]),
            np.array([v[1] for v in vals]),
        )

    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = f"engine_comparison_{ts}.json"
    out_path = out_path or default_out

    report = {
        "mode": "engine_comparison",
        "engines": ["idm", "sumo"],
        "note": (
            "sim=IDM  ref=SUMO. "
            "MAPE/RMSE/GEH follow standard traffic-engineering definitions: "
            "MAPE=mean|sim-ref|/ref×100, RMSE=sqrt(mean((sim-ref)^2)), "
            "GEH=sqrt(2(sim-ref)^2/(sim+ref)); GEH<5 is the TfL/Highway Capacity Manual 'acceptable' threshold."
        ),
        "requested_pairs": n_pairs,
        "matched_pairs": len(results),
        "metrics_overall": overall,
        "metrics_by_class": per_class,
        "pairs": results,
    }

    print("\n" + "=" * 60)
    print("  ENGINE COMPARISON  (IDM vs SUMO)")
    print("=" * 60)
    m = overall
    if m.get("n", 0) > 0:
        print(f"  pairs   : {m['n']}")
        print(f"  IDM mean: {m['sim_mean_s']:.0f}s   SUMO mean: {m['ref_mean_s']:.0f}s")
        print(f"  MAPE    : {m['mape_pct']:.1f}%   RMSE: {m['rmse_s']:.0f}s")
        print(f"  bias (IDM-SUMO): {m['bias_s']:+.0f}s   Pearson r: {m.get('pearson_r', float('nan')):.3f}")
        print(f"  GEH<5   : {m['geh_under5_pct']:.0f}% of pairs")
    print("=" * 60 + "\n")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[validate-engines] report → {out_path}")
    except Exception as exc:
        print(f"[validate-engines] write failed: {exc}")

    return report
