"""scenario_compare.py — save/compare traffic scenarios headlessly in SUMO.

A scenario = current street_graph topology + traffic-signal timings + demand
parameters. Scenarios are serialized to JSON and built into standalone SUMO
scenes; 'run comparison' executes both headlessly with identical seeds and
demand parameters, then reports per-edge travel-time deltas and network-wide
emission/noise deltas.

Public API
----------
edge_ids_for_graph(street_graph) -> list[(edge_id, u, v, key, data)]
snapshot_scenario(street_graph, traffic_lights, demand_params, name) -> dict
save_scenario(app, slot) -> bool
run_headless(cfg_path, duration_s, step_s, seed, sample_every_s) -> dict | None
compare(result_a, result_b, out_path=None) -> dict

Graceful degradation: everything that touches SUMO probes availability first
and prints a clear message instead of raising.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from emissions import POLLUTANTS, emission_rate_g_per_s

_PFX = "[scenario]"

# Noise model constants (simplified CNOSSOS point-source, same as the app)
_NOISE_CELL_M = 25.0
_NOISE_MAX_CELLS = 40_000
_NOISE_THRESHOLD_DB = 65.0


# ── shared edge-id generation ─────────────────────────────────────────────────

def edge_ids_for_graph(street_graph) -> list[tuple]:
    """Return [(edge_id, u, v, key, data), ...] using the exact id-generation
    scheme of the SUMO plain-XML exporter (graph_to_sumo_plainxml imports this
    helper too, so the two can never diverge).

    edge_id = "{osmid}#{key}" when the edge carries an osmid, else
              "e_{u}_{v}_{key}"; duplicates are deduped with "_{n}" suffixes
    in graph iteration order.
    """
    out: list[tuple] = []
    seen_ids: set[str] = set()
    for u, v, key, data in street_graph.edges(data=True, keys=True):
        osmid = data.get("osmid", "")
        raw_id = f"{osmid}#{key}" if osmid else f"e_{u}_{v}_{key}"
        edge_id = raw_id
        suffix = 0
        while edge_id in seen_ids:
            edge_id = f"{raw_id}_{suffix}"
            suffix += 1
        seen_ids.add(edge_id)
        out.append((edge_id, u, v, key, data))
    return out


# ── serialization ─────────────────────────────────────────────────────────────

def _json_native(val):
    """Coerce a value into a JSON-native type (str/int/float/bool/list/None)."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (list, tuple, set, frozenset)):
        return [_json_native(v) for v in val]
    return str(val)


def snapshot_scenario(street_graph, traffic_lights, demand_params, name) -> dict:
    """Serialize the current scenario (topology + signals + demand) to a
    JSON-native dict."""
    nodes = []
    for node_id, data in street_graph.nodes(data=True):
        entry = {
            "id": str(node_id),
            "x": float(data.get("x", 0.0)),
            "y": float(data.get("y", 0.0)),
        }
        hw = data.get("highway")
        if hw:
            entry["highway"] = _json_native(hw)
        nodes.append(entry)

    edges = []
    for u, v, key, data in street_graph.edges(data=True, keys=True):
        raw_speed = data.get("maxspeed_ms")
        if raw_speed is None:
            try:
                raw_speed = float(data.get("maxspeed", 50)) / 3.6
            except (TypeError, ValueError):
                raw_speed = 13.89
        entry = {
            "u": str(u),
            "v": str(v),
            "key": int(key),
            "length": float(data.get("length", 0.0) or 0.0),
            "lanes": max(1, int(data.get("lanes", 1) or 1)),
            "maxspeed_ms": float(raw_speed),
            "osmid": _json_native(data.get("osmid", "")),
            "oneway": bool(data.get("oneway", False)),
            "junction": _json_native(data.get("junction", "")),
        }
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            entry["geometry"] = [[float(c[0]), float(c[1])] for c in geom.coords]
        edges.append(entry)

    signals = []
    for node_id, tl in (traffic_lights or {}).items():
        signals.append({
            "node_id": str(node_id),
            "green_durations": [float(g) for g in getattr(tl, "green_durations", [])],
            "yellow_durations": [float(y) for y in getattr(tl, "yellow_durations", [])],
            "n_phases": int(getattr(tl, "n_phases", 2)),
        })

    demand_params = demand_params or {}
    demand = {
        "period": float(demand_params.get("period", 12.0)),
        "end": float(demand_params.get("end", 900.0)),
        "seed": int(demand_params.get("seed", 42)),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "signals": signals,
        "demand": demand,
        "meta": {
            "name": str(name),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    }


# ── scenario build ────────────────────────────────────────────────────────────

def save_scenario(app, slot: str) -> bool:
    """Snapshot the app's current scenario into scenarios/<slot>/ and build a
    standalone SUMO scene (net + routes + cfg) for headless comparison runs."""
    try:
        from sumo_network_patch import (
            graph_to_sumo_plainxml, rebuild_sumo_net, generate_routes,
        )
    except Exception as exc:
        print(f"{_PFX} sumo_network_patch unavailable: {exc}")
        return False

    try:
        slot = str(slot)
        scn_dir = Path(__file__).parent / "scenarios" / slot
        scn_dir.mkdir(parents=True, exist_ok=True)

        graph = app.street_graph
        lights = app.scene_state.get("traffic_lights", {}) or {}
        try:
            seed = int(app.args.seed)
        except Exception:
            seed = 42
        demand = {"period": 12.0, "end": 900.0, "seed": seed}

        print(f"{_PFX} snapshotting scenario {slot} → {scn_dir}")
        snap = snapshot_scenario(graph, lights, demand, name=f"scenario_{slot}")
        with open(scn_dir / "scenario.json", "w") as fh:
            json.dump(snap, fh, indent=1)
        print(f"{_PFX} scenario.json written "
              f"({len(snap['nodes'])} nodes, {len(snap['edges'])} edges, "
              f"{len(snap['signals'])} signals)")

        # SUMO build chain: plain-XML → netconvert → randomTrips
        try:
            from sumo_bridge import resolve_sumo_home
            resolve_sumo_home()   # sets $SUMO_HOME for the child tools
        except Exception:
            pass

        if not graph_to_sumo_plainxml(graph, scn_dir, prefix="scn"):
            print(f"{_PFX} plain-XML export failed — scenario {slot} not built")
            return False
        print(f"{_PFX} plain-XML exported")

        net_file = scn_dir / "scn.net.xml"
        if not rebuild_sumo_net("scn", scn_dir, net_file):
            print(f"{_PFX} netconvert failed — scenario {slot} not built")
            return False
        print(f"{_PFX} network rebuilt → {net_file.name}")

        if not generate_routes(
            str(net_file),
            str(scn_dir / "scn.trips.xml"),
            str(scn_dir / "scn.rou.xml"),
            period=demand["period"],
            end=demand["end"],
            seed=demand["seed"],
        ):
            print(f"{_PFX} route generation failed — scenario {slot} not built")
            return False
        print(f"{_PFX} demand generated (period={demand['period']}s, "
              f"end={demand['end']}s, seed={demand['seed']})")

        cfg = (
            "<configuration>\n"
            "  <input>\n"
            "    <net-file value=\"scn.net.xml\"/>\n"
            "    <route-files value=\"scn.rou.xml\"/>\n"
            "  </input>\n"
            "</configuration>\n"
        )
        (scn_dir / "scn.sumocfg").write_text(cfg)
        print(f"{_PFX} scenario {slot} saved and built ✓")
        return True

    except Exception as exc:
        print(f"{_PFX} save_scenario({slot!r}) failed: {exc}")
        return False


# ── headless run ─────────────────────────────────────────────────────────────

def _resolve_sumo_binary() -> str | None:
    """Locate the headless `sumo` binary (PATH, then $SUMO_HOME/bin)."""
    import os
    found = shutil.which("sumo")
    if found:
        return found
    home = os.environ.get("SUMO_HOME", "")
    if home:
        cand = os.path.join(home, "bin", "sumo")
        if os.path.exists(cand) or os.path.exists(cand + ".exe"):
            return cand
    return None


class _NoiseGrid:
    """Lazy 2-D analysis grid over the SUMO net bounds; accumulates
    cell-seconds above the dB(A) threshold from vehicle point sources."""

    def __init__(self, boundary) -> None:
        (xmin, ymin), (xmax, ymax) = boundary
        w = max(xmax - xmin, 1.0)
        h = max(ymax - ymin, 1.0)
        cell = _NOISE_CELL_M
        while (w / cell) * (h / cell) > _NOISE_MAX_CELLS:
            cell *= 1.5
        nx = max(1, int(math.ceil(w / cell)))
        ny = max(1, int(math.ceil(h / cell)))
        xs = xmin + (np.arange(nx) + 0.5) * cell
        ys = ymin + (np.arange(ny) + 0.5) * cell
        gx, gy = np.meshgrid(xs, ys)
        self.centers = np.column_stack([gx.ravel(), gy.ravel()])   # (nc, 2)
        self.cell = cell
        self.exceed_cell_seconds = 0.0

    def sample(self, pos: np.ndarray, v_ms: np.ndarray, dt: float) -> None:
        """pos (nv,2), v_ms (nv,) — energy-sum per cell, count breaches."""
        if pos.shape[0] == 0:
            return
        v_kmh = np.maximum(v_ms * 3.6, 1.0)
        l_w = 55.0 + 10.0 * np.log10(v_kmh / 50.0)                 # (nv,)
        d = np.linalg.norm(
            self.centers[:, None, :] - pos[None, :, :], axis=2)   # (nc, nv)
        l_recv = l_w[None, :] - 20.0 * np.log10(np.maximum(d, 1.0))
        energy = np.sum(10.0 ** (l_recv / 10.0), axis=1)           # (nc,)
        l_cell = 10.0 * np.log10(np.maximum(energy, 1e-12))
        self.exceed_cell_seconds += float(
            np.count_nonzero(l_cell > _NOISE_THRESHOLD_DB)) * dt


def run_headless(
    cfg_path,
    duration_s: float = 900.0,
    step_s: float = 0.5,
    seed: int = 42,
    sample_every_s: float = 5.0,
) -> dict | None:
    """Run one scenario headlessly in SUMO via traci, collecting per-edge
    travel times, network-wide emissions, and noise exceedance.

    Strictly sequential-safe: uses the default traci connection and always
    closes it in a finally block — run A fully, then run B.
    """
    try:
        from sumo_bridge import check_sumo
    except Exception as exc:
        print(f"{_PFX} sumo_bridge unavailable: {exc}")
        return None

    ok, msg = check_sumo()
    if not ok:
        print(f"{_PFX} {msg}")
        return None

    cfg_path = str(cfg_path)
    if not Path(cfg_path).exists():
        print(f"{_PFX} config not found: {cfg_path}")
        return None

    binary = _resolve_sumo_binary()
    if binary is None:
        print(f"{_PFX} could not locate the `sumo` binary on PATH or $SUMO_HOME/bin")
        return None

    try:
        import traci
    except Exception as exc:
        print(f"{_PFX} traci import failed: {exc}")
        return None

    cmd = [
        binary, "-c", cfg_path,
        "--step-length", str(step_s),
        "--seed", str(seed),
        "--no-warnings", "--no-step-log",
        "--start", "--quit-on-end",
    ]

    started = False
    try:
        print(f"{_PFX} headless run: {cfg_path} "
              f"(duration={duration_s}s, step={step_s}s, seed={seed})")
        traci.start(cmd)
        started = True

        edge_ids = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        tt_sum = np.zeros(len(edge_ids), dtype=float)
        tt_cnt = 0

        emissions = {p: 0.0 for p in POLLUTANTS}
        noise_grid: _NoiseGrid | None = None
        arrived = 0
        speed_sum = 0.0
        speed_n = 0

        t = 0.0
        next_sample = sample_every_s
        while t < duration_s:
            traci.simulationStep()
            t += step_s
            arrived += int(traci.simulation.getArrivedNumber())

            if t + 1e-9 < next_sample:
                continue
            next_sample += sample_every_s

            # ── vehicle state ────────────────────────────────────────────────
            ids = traci.vehicle.getIDList()
            speeds, classes, positions = [], [], []
            for vid in ids:
                try:
                    speeds.append(float(traci.vehicle.getSpeed(vid)))
                    vc = str(traci.vehicle.getVehicleClass(vid))
                    if vc in ("DEFAULT_VEHTYPE", "ignoring", ""):
                        vc = "passenger"
                    classes.append(vc)
                    positions.append(traci.vehicle.getPosition(vid))
                except Exception:
                    continue
            v_arr = np.asarray(speeds, dtype=float)
            pos_arr = (np.asarray(positions, dtype=float)
                       if positions else np.empty((0, 2)))

            speed_sum += float(v_arr.sum())
            speed_n += int(v_arr.size)

            # ── emissions (vectorized per class) ─────────────────────────────
            if v_arr.size:
                cls_arr = np.asarray(classes)
                for cls in set(classes):
                    mask = cls_arr == cls
                    for pol in POLLUTANTS:
                        rates = emission_rate_g_per_s(pol, v_arr[mask], cls)
                        emissions[pol] += float(np.sum(rates)) * sample_every_s

            # ── noise exceedance ─────────────────────────────────────────────
            if noise_grid is None:
                try:
                    noise_grid = _NoiseGrid(traci.simulation.getNetBoundary())
                except Exception as exc:
                    print(f"{_PFX} noise grid init failed: {exc}")
                    noise_grid = _NoiseGrid(((0.0, 0.0), (1.0, 1.0)))
            noise_grid.sample(pos_arr, v_arr, sample_every_s)

            # ── per-edge instantaneous travel time ───────────────────────────
            for i, eid in enumerate(edge_ids):
                try:
                    tt_sum[i] += float(traci.edge.getTraveltime(eid))
                except Exception:
                    pass
            tt_cnt += 1

        edge_tt = {}
        if tt_cnt > 0:
            for i, eid in enumerate(edge_ids):
                edge_tt[eid] = float(tt_sum[i] / tt_cnt)

        result = {
            "duration_s": float(duration_s),
            "step_s": float(step_s),
            "seed": int(seed),
            "edge_tt": edge_tt,
            "emissions_g": {p: float(emissions[p]) for p in POLLUTANTS},
            "noise_exceed_cell_seconds": float(
                noise_grid.exceed_cell_seconds if noise_grid else 0.0),
            "vehicles_arrived": int(arrived),
            "mean_speed_ms": float(speed_sum / speed_n) if speed_n else 0.0,
        }
        print(f"{_PFX} run complete: {arrived} arrived, "
              f"mean speed {result['mean_speed_ms']:.1f} m/s, "
              f"CO2 {result['emissions_g']['co2'] / 1000.0:.1f} kg")
        return result

    except Exception as exc:
        print(f"{_PFX} headless run failed: {exc}")
        return None
    finally:
        if started:
            try:
                traci.close()
            except Exception:
                pass


# ── comparison report ─────────────────────────────────────────────────────────

_METRIC_DEFINITIONS = {
    "edge_tt": (
        "Per-edge travel time [s] = time-mean of SUMO's instantaneous "
        "per-edge travel-time estimate (traci.edge.getTraveltime) sampled "
        "over the whole run. Delta = scenario B minus scenario A."
    ),
    "emissions_g": (
        "Total emitted mass [g] per pollutant = EEA/COPERT average-speed "
        "emission model (emissions.emission_rate_g_per_s) integrated over "
        "the sampled per-vehicle speeds for the whole run."
    ),
    "noise_exceed_cell_seconds": (
        "Noise exceedance [cell-seconds] = accumulated time-area above "
        "65 dB(A) on a 25 m analysis grid, with per-vehicle source levels "
        "from a simplified CNOSSOS point-source model "
        "(L_w = 55 + 10*log10(v_kmh/50), spherical 20*log10(d) spreading, "
        "energetic summation per cell)."
    ),
    "vehicles_arrived": "Number of vehicles that completed their trip during the run.",
    "mean_speed_ms": "Mean sampled vehicle speed over the whole run [m/s].",
}


def _pct(a: float, b: float) -> float | None:
    return None if abs(a) < 1e-12 else (b - a) / a * 100.0


def compare(result_a: dict, result_b: dict, out_path: str | None = None) -> dict:
    """Diff two run_headless() results: per-edge travel-time deltas plus a
    network-wide summary. Writes the report as JSON and prints a table."""
    tt_a = result_a.get("edge_tt", {})
    tt_b = result_b.get("edge_tt", {})
    common = sorted(set(tt_a) & set(tt_b))
    delta_tt = {eid: float(tt_b[eid]) - float(tt_a[eid]) for eid in common}

    mean_a = float(np.mean([tt_a[e] for e in common])) if common else 0.0
    mean_b = float(np.mean([tt_b[e] for e in common])) if common else 0.0

    em_a = result_a.get("emissions_g", {})
    em_b = result_b.get("emissions_g", {})
    em_delta = {p: float(em_b.get(p, 0.0)) - float(em_a.get(p, 0.0))
                for p in POLLUTANTS}
    em_pct = {p: _pct(float(em_a.get(p, 0.0)), float(em_b.get(p, 0.0)))
              for p in POLLUTANTS}

    noise_a = float(result_a.get("noise_exceed_cell_seconds", 0.0))
    noise_b = float(result_b.get("noise_exceed_cell_seconds", 0.0))

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "metric_definitions": dict(_METRIC_DEFINITIONS),
        "edge_tt_delta_s": delta_tt,
        "summary": {
            "edges_compared": len(common),
            "mean_edge_tt_a_s": mean_a,
            "mean_edge_tt_b_s": mean_b,
            "mean_edge_tt_delta_s": mean_b - mean_a,
            "emissions_a_g": {p: float(em_a.get(p, 0.0)) for p in POLLUTANTS},
            "emissions_b_g": {p: float(em_b.get(p, 0.0)) for p in POLLUTANTS},
            "emissions_delta_g": em_delta,
            "emissions_delta_pct": em_pct,
            "noise_exceed_a_cell_s": noise_a,
            "noise_exceed_b_cell_s": noise_b,
            "noise_exceed_delta_cell_s": noise_b - noise_a,
            "vehicles_arrived_a": int(result_a.get("vehicles_arrived", 0)),
            "vehicles_arrived_b": int(result_b.get("vehicles_arrived", 0)),
            "mean_speed_a_ms": float(result_a.get("mean_speed_ms", 0.0)),
            "mean_speed_b_ms": float(result_b.get("mean_speed_ms", 0.0)),
        },
    }

    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(__file__).parent / f"comparison_{stamp}.json")
    try:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"{_PFX} report written → {out_path}")
    except Exception as exc:
        print(f"{_PFX} could not write report: {exc}")

    s = report["summary"]

    def _fmt_pct(v):
        return f"{v:+.1f}%" if v is not None else "n/a"

    print(f"\n{_PFX} ── Scenario comparison (B − A) ─────────────────────────")
    print(f"{_PFX} {'metric':<28}{'A':>12}{'B':>12}{'delta':>14}")
    print(f"{_PFX} {'mean edge travel time [s]':<28}"
          f"{s['mean_edge_tt_a_s']:>12.2f}{s['mean_edge_tt_b_s']:>12.2f}"
          f"{s['mean_edge_tt_delta_s']:>+14.2f}")
    for p in POLLUTANTS:
        print(f"{_PFX} {p.upper() + ' [g]':<28}"
              f"{s['emissions_a_g'][p]:>12.1f}{s['emissions_b_g'][p]:>12.1f}"
              f"{s['emissions_delta_g'][p]:>+9.1f} "
              f"{_fmt_pct(s['emissions_delta_pct'][p]):>4}")
    print(f"{_PFX} {'noise >65 dB [cell·s]':<28}"
          f"{s['noise_exceed_a_cell_s']:>12.0f}{s['noise_exceed_b_cell_s']:>12.0f}"
          f"{s['noise_exceed_delta_cell_s']:>+14.0f}")
    print(f"{_PFX} {'vehicles arrived':<28}"
          f"{s['vehicles_arrived_a']:>12d}{s['vehicles_arrived_b']:>12d}"
          f"{s['vehicles_arrived_b'] - s['vehicles_arrived_a']:>+14d}")
    print(f"{_PFX} ({len(common)} edges compared)\n")

    return report
