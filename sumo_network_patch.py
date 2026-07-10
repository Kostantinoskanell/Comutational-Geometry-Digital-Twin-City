"""sumo_network_patch.py — rebuild a SUMO network from the current street_graph.

Used when the in-app road editor (reverse road, roundabout) modifies geometry
while --engine sumo is active.  The graph is exported as SUMO plain-XML, fed to
netconvert (fast: no OSM download), demand is regenerated, and the .sumocfg is
updated so the caller can restart the TraCI connection.

Public API
----------
graph_to_sumo_plainxml(street_graph, out_dir, prefix="patch") -> bool
rebuild_sumo_net(plain_prefix, out_dir, out_net, sumo_home="")   -> bool
generate_routes(net_file, out_trips, out_routes, period, end, sumo_home="", seed=42) -> bool
update_sumocfg(cfg_path, net_file, rou_file)                     -> bool
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ── graph → plain-XML ────────────────────────────────────────────────────────

def _parse_lane_count(raw) -> int | None:
    """Parse an OSM lanes value ('2', 2, '2;3', ['2','3']) → int or None."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals = [_parse_lane_count(r) for r in raw]
        vals = [v for v in vals if v]
        return max(vals) if vals else None
    try:
        s = str(raw).strip()
        if ";" in s:
            parts = [p for p in s.split(";") if p.strip()]
            vals = [_parse_lane_count(p) for p in parts]
            vals = [v for v in vals if v]
            return max(vals) if vals else None
        return max(1, int(float(s)))
    except (TypeError, ValueError):
        return None


def graph_to_sumo_plainxml(
    street_graph,
    out_dir: str | Path,
    prefix: str = "patch",
) -> bool:
    """Export current street_graph as SUMO plain-XML node/edge files.

    Coordinates are converted from local metres (proj_str) to lon/lat so
    netconvert can produce a geo-referenced network.  Returns True on success.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        print("[sumo-patch] pyproj missing — cannot export graph")
        return False

    proj_str = street_graph.graph.get("proj_str", "")
    if not proj_str:
        print("[sumo-patch] no proj_str in graph — cannot export")
        return False

    try:
        to_wgs = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    except Exception as exc:
        print(f"[sumo-patch] pyproj transform init failed: {exc}")
        return False

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── nodes ─────────────────────────────────────────────────────────────────
    nod = ET.Element("nodes")
    for node_id, data in street_graph.nodes(data=True):
        xm = float(data.get("x", 0.0))
        ym = float(data.get("y", 0.0))
        lon, lat = to_wgs.transform(xm, ym)
        el = ET.SubElement(nod, "node")
        el.set("id", str(node_id))
        el.set("x", f"{lon:.8f}")
        el.set("y", f"{lat:.8f}")
        hw = data.get("highway", "")
        if hw in ("traffic_signals", "traffic_lights"):
            el.set("type", "traffic_light")
        else:
            el.set("type", "priority")

    ET.indent(nod)
    ET.ElementTree(nod).write(
        str(out / f"{prefix}.nod.xml"),
        encoding="unicode",
        xml_declaration=True,
    )

    # ── edges ─────────────────────────────────────────────────────────────────
    # Edge ids come from the shared helper so the scenario-comparison overlay
    # (scenario_compare / scenario_mixin) can rebuild the exact same mapping.
    from scenario_compare import edge_ids_for_graph

    edg = ET.Element("edges")

    for edge_id, u, v, key, data in edge_ids_for_graph(street_graph):
        el = ET.SubElement(edg, "edge")
        el.set("id", edge_id)
        el.set("from", str(u))
        el.set("to", str(v))
        raw_lanes = _parse_lane_count(data.get("lanes"))
        oneway = bool(data.get("oneway", False))
        # OSM 'lanes' counts both directions on bidirectional roads; SUMO numLanes
        # is per directed edge. A reverse edge in the MultiDiGraph means two-way.
        has_reverse = street_graph.has_edge(v, u)
        if raw_lanes is None:
            lanes = 1
        elif oneway or not has_reverse:
            lanes = raw_lanes
        else:
            lanes = max(1, raw_lanes // 2)
        el.set("numLanes", str(lanes))

        raw_speed = data.get("maxspeed_ms")
        if raw_speed is None:
            raw_ms = data.get("maxspeed", 50)
            try:
                raw_speed = float(raw_ms) / 3.6
            except (TypeError, ValueError):
                raw_speed = 13.89
        el.set("speed", f"{float(raw_speed):.2f}")

        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            pts = []
            for c in geom.coords:
                lon_p, lat_p = to_wgs.transform(float(c[0]), float(c[1]))
                pts.append(f"{lon_p:.8f},{lat_p:.8f}")
            if len(pts) >= 2:
                el.set("shape", " ".join(pts))

    ET.indent(edg)
    ET.ElementTree(edg).write(
        str(out / f"{prefix}.edg.xml"),
        encoding="unicode",
        xml_declaration=True,
    )

    print(f"[sumo-patch] plain-XML written → {out / prefix}.{{nod,edg}}.xml")
    return True


# ── netconvert rebuild ────────────────────────────────────────────────────────

def _resolve_netconvert(sumo_home: str = "") -> str:
    """Return path to netconvert binary, searching SUMO_HOME and PATH."""
    if not sumo_home:
        sumo_home = os.environ.get("SUMO_HOME", "")

    # In the macOS Framework layout, SUMO_HOME is remapped to .../share/sumo
    # by _resolve_sumo_home() in build_sumo_scene.py.  The binary lives at
    # SUMO_HOME/bin/netconvert (already the correct path after remapping).
    roots = [sumo_home]
    # If SUMO_HOME still points at the framework root (not remapped), check
    # both framework-root/bin and share/sumo/bin.
    if sumo_home and not sumo_home.endswith(os.sep + "share" + os.sep + "sumo"):
        roots.append(os.path.join(sumo_home, "share", "sumo"))

    for root in roots:
        for cand in (
            os.path.join(root, "bin", "netconvert"),
            os.path.join(root, "bin", "netconvert.exe"),
        ):
            if os.path.isfile(cand):
                return cand

    return "netconvert"  # rely on PATH


def rebuild_sumo_net(
    plain_prefix: str,
    out_dir: str | Path,
    out_net: str | Path,
    sumo_home: str = "",
) -> bool:
    """Run netconvert on plain-XML files to produce a new .net.xml.

    Coordinates in the plain-XML are lon/lat (written by graph_to_sumo_plainxml),
    so we pass --proj.plain-geo to tell netconvert to interpret them as geographic.
    """
    out = Path(out_dir)
    nod_file = out / f"{plain_prefix}.nod.xml"
    edg_file = out / f"{plain_prefix}.edg.xml"

    if not nod_file.exists() or not edg_file.exists():
        print(f"[sumo-patch] plain-XML not found in {out}")
        return False

    binary = _resolve_netconvert(sumo_home)
    cmd = [
        binary,
        "--node-files",    str(nod_file),
        "--edge-files",    str(edg_file),
        "-o",              str(out_net),
        # Input node coords are lon/lat degrees.  --proj.utm makes netconvert
        # PROJECT them to UTM metres and store the projection in the net
        # (projParameter).  Without it, degrees are treated as metres — the
        # whole network collapses to a ~4 cm dot and sumolib's hasGeoProj()
        # is False, so SumoConnection refuses the net.
        "--proj.utm",
        "--proj.plain-geo",               # plain-XML output stays geographic
        "--geometry.remove",
        "--roundabouts.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--no-turnarounds",
    ]
    print(f"[sumo-patch] netconvert: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"[sumo-patch] netconvert failed (exit {result.returncode}):\n"
                  f"{result.stderr[-600:]}")
            return False
        print("[sumo-patch] netconvert OK")
        return True
    except FileNotFoundError:
        print(f"[sumo-patch] netconvert binary not found: {binary}")
        return False
    except Exception as exc:
        print(f"[sumo-patch] netconvert error: {exc}")
        return False


# ── demand generation ─────────────────────────────────────────────────────────

def generate_routes(
    net_file: str,
    out_trips: str,
    out_routes: str,
    period: float = 12.0,
    end: float = 3600.0,
    sumo_home: str = "",
    seed: int = 42,
) -> bool:
    """Run randomTrips.py to generate demand for the patched network."""
    if not sumo_home:
        sumo_home = os.environ.get("SUMO_HOME", "")

    # Search for randomTrips.py in canonical locations
    tool_dirs = []
    if sumo_home:
        tool_dirs.append(os.path.join(sumo_home, "tools"))
        tool_dirs.append(os.path.join(sumo_home, "share", "sumo", "tools"))
    try:
        import sumo as _sumo_pkg
        pkg_dir = os.path.dirname(os.path.abspath(_sumo_pkg.__file__))
        tool_dirs.append(os.path.join(pkg_dir, "tools"))
        tool_dirs.append(os.path.join(pkg_dir, "share", "sumo", "tools"))
    except Exception:
        pass

    random_trips = next(
        (os.path.join(d, "randomTrips.py") for d in tool_dirs
         if os.path.isfile(os.path.join(d, "randomTrips.py"))),
        None,
    )
    if random_trips is None:
        print("[sumo-patch] randomTrips.py not found — cannot generate demand")
        return False

    cmd = [
        sys.executable, random_trips,
        "-n",              str(net_file),
        "-o",              str(out_trips),
        "-r",              str(out_routes),
        "--end",           str(end),
        "--period",        f"{period:.3f}",
        "--validate",
        "--fringe-factor", "10",
        "--seed",          str(seed),
    ]
    print(f"[sumo-patch] randomTrips: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"[sumo-patch] randomTrips failed:\n{result.stderr[-400:]}")
            return False
        print("[sumo-patch] randomTrips OK")
        return True
    except Exception as exc:
        print(f"[sumo-patch] randomTrips error: {exc}")
        return False


# ── .sumocfg update ───────────────────────────────────────────────────────────

def update_sumocfg(
    cfg_path: str | Path,
    net_file: str | Path,
    rou_file: str | Path,
) -> bool:
    """Rewrite the .sumocfg to point at new network and route files."""
    try:
        cfg_path = Path(cfg_path)
        base = cfg_path.parent

        def _rel(p: Path) -> str:
            try:
                return os.path.relpath(str(p), str(base))
            except ValueError:
                return str(p)

        tree = ET.parse(str(cfg_path))
        root = tree.getroot()

        for tag, rel_val in [
            ("net-file",    _rel(Path(net_file))),
            ("route-files", _rel(Path(rou_file))),
        ]:
            node = root.find(f".//{tag}")
            if node is None:
                inp = root.find(".//input")
                if inp is None:
                    inp = ET.SubElement(root, "input")
                node = ET.SubElement(inp, tag)
            node.set("value", rel_val)

        ET.indent(tree)
        tree.write(str(cfg_path), encoding="unicode", xml_declaration=True)
        print(f"[sumo-patch] .sumocfg updated: net={_rel(Path(net_file))} rou={_rel(Path(rou_file))}")
        return True
    except Exception as exc:
        print(f"[sumo-patch] cfg update failed: {exc}")
        return False
