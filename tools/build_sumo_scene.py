#!/usr/bin/env python3
"""build_sumo_scene.py — generate a geo-referenced SUMO scene for co-simulation.

Produces a .net.xml, random demand, and a .sumocfg covering the SAME area you
load in the digital-twin app, so SUMO vehicles align with your rendered city.

Pipeline (all standard SUMO tools):
  1. osmGet.py    — download an OSM extract for the bbox  (skipped if --osm-file)
  2. netconvert   — OSM → geo-referenced SUMO network
  3. randomTrips.py — synthetic demand → routes
  4. write <name>.sumocfg

Requirements: a working SUMO install with $SUMO_HOME set (provides netconvert,
osmGet.py, randomTrips.py).

Examples
--------
  # Match an app scene at lat/lon with a 600 m radius
  python tools/build_sumo_scene.py --lat 37.9838 --lon 23.7275 --radius 600 \
        --out-dir sumo_athens --name athens

  # From an existing OSM extract
  python tools/build_sumo_scene.py --osm-file athens.osm.xml \
        --out-dir sumo_athens --name athens

Then run the app with:
  python main_ast6.py --lat 37.9838 --lon 23.7275 --radius 600 --n-cars 0 \
        --sumo-cfg sumo_athens/athens.sumocfg
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def _resolve_sumo_home() -> str:
    """SUMO_HOME from env, or inferred from the eclipse-sumo pip wheel.

    The eclipse-sumo pip wheel (v1.x) does NOT expose a ``sumo.SUMO_HOME``
    attribute.  Instead, its package directory *is* SUMO_HOME — it contains
    the ``tools/`` subdirectory with osmGet.py, randomTrips.py, etc., and a
    ``bin/`` subdirectory (or PATH-accessible binaries) for netconvert.

    Resolution order
    ----------------
    1. $SUMO_HOME env var (if set and is a real directory)
    2. ``sumo.SUMO_HOME`` attribute (older wheel layout)
    3. ``os.path.dirname(sumo.__file__)`` — package directory (v1.x wheel)
    4. Return ``""`` and rely on PATH for binaries
    """
    home = os.environ.get("SUMO_HOME", "")
    if home and os.path.isdir(home):
        # macOS Framework layout: SUMO_HOME points to the framework root
        # (e.g. .../EclipseSUMO) but netconvert expects data/typemap/ directly
        # under SUMO_HOME, which in the Framework lives under share/sumo/.
        _share = os.path.join(home, "share", "sumo")
        if (os.path.isdir(os.path.join(_share, "data")) and
                not os.path.isdir(os.path.join(home, "data"))):
            os.environ["SUMO_HOME"] = _share
            return _share
        return home
    try:
        import sumo as _sumo_pkg  # eclipse-sumo wheel

        # Approach A: explicit attribute (older wheels)
        pkg_home = getattr(_sumo_pkg, "SUMO_HOME", "")
        if pkg_home and os.path.isdir(pkg_home):
            os.environ["SUMO_HOME"] = pkg_home
            return pkg_home

        # Approach B: the package directory itself contains tools/ (v1.x wheels)
        pkg_dir = os.path.dirname(os.path.abspath(_sumo_pkg.__file__))
        if os.path.isdir(os.path.join(pkg_dir, "tools")):
            os.environ["SUMO_HOME"] = pkg_dir
            return pkg_dir
    except Exception:
        pass
    return home


def _sumo_tool(name: str) -> list[str]:
    """Return a runnable command prefix for a SUMO python tool or binary.

    Searches in the following order for Python tools (osmGet.py etc.):
      1. $SUMO_HOME/tools/                   (standard SUMO install)
      2. $SUMO_HOME/share/sumo/tools/        (macOS Framework layout)
      3. <pip wheel package dir>/tools/       (eclipse-sumo pip wheel)
      4. <pip wheel package dir>/share/sumo/tools/
    Netconvert is resolved from $SUMO_HOME/bin/ then PATH.
    """
    home = _resolve_sumo_home()

    # ── netconvert binary ──────────────────────────────────────────────────
    if name == "netconvert":
        if home:
            # If SUMO_HOME was remapped to .../share/sumo (macOS Framework),
            # the binary lives two levels up at the framework root's bin/.
            _roots = [home]
            if home.endswith(os.sep + "share" + os.sep + "sumo"):
                _roots.append(os.path.dirname(os.path.dirname(home)))
            for root in _roots:
                for cand in [
                    os.path.join(root, "bin", "netconvert"),
                    os.path.join(root, "bin", "netconvert.exe"),
                ]:
                    if os.path.isfile(cand):
                        return [cand]
        return ["netconvert"]

    # ── Python tools (osmGet.py, randomTrips.py, …) ───────────────────────
    tool_dirs: list[str] = []
    if home:
        tool_dirs.append(os.path.join(home, "tools"))                   # standard
        tool_dirs.append(os.path.join(home, "share", "sumo", "tools"))  # framework

    # Also probe the pip-wheel package directory directly — this works even
    # when $SUMO_HOME points to the system Framework that lacks tools/.
    try:
        import sumo as _sumo_pkg  # eclipse-sumo wheel
        pkg_dir = os.path.dirname(os.path.abspath(_sumo_pkg.__file__))
        tool_dirs.append(os.path.join(pkg_dir, "tools"))
        tool_dirs.append(os.path.join(pkg_dir, "share", "sumo", "tools"))
    except Exception:
        pass

    for tdir in tool_dirs:
        cand = os.path.join(tdir, name)
        if os.path.isfile(cand):
            return [sys.executable, cand]

    # Last-resort: try importing as a module (rarely works, but harmless)
    return [sys.executable, "-m", name.replace(".py", "")]


def _run(cmd: list[str], desc: str) -> bool:
    print(f"\n[build] {desc}\n[build] $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print(f"[build] ERROR: tool not found for: {desc}. Is $SUMO_HOME set?")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[build] ERROR: {desc} failed (exit {exc.returncode})")
        return False


def _bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # W, S, E, N


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a geo-referenced SUMO scene.")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--radius", type=float, default=600.0, help="metres (matches app --radius)")
    ap.add_argument("--osm-file", type=str, default="", help="use this OSM extract instead of downloading")
    ap.add_argument("--out-dir", type=str, default="sumo_scene")
    ap.add_argument("--name", type=str, default="scene")
    ap.add_argument("--vehicles", type=int, default=300, help="approx concurrent vehicles")
    ap.add_argument("--end", type=float, default=3600.0, help="sim end time (s) for demand")
    args = ap.parse_args()

    # Resolve SUMO_HOME now (may auto-detect from the pip wheel) so the
    # warning is accurate and child processes inherit the env var.
    _sumo_home = _resolve_sumo_home()
    if _sumo_home:
        os.environ["SUMO_HOME"] = _sumo_home
        print(f"[build] SUMO_HOME = {_sumo_home}")
    else:
        print("[build] WARNING: SUMO_HOME could not be resolved. "
              "Install eclipse-sumo (pip install eclipse-sumo) or set $SUMO_HOME.")

    # Verify osmGet.py is actually findable (helps diagnose tool-path issues).
    _osmget_cmd = _sumo_tool("osmGet.py")
    if len(_osmget_cmd) >= 2 and os.path.isfile(_osmget_cmd[1]):
        print(f"[build] osmGet.py = {_osmget_cmd[1]}")
    else:
        print("[build] WARNING: osmGet.py not found in any known SUMO tools dir. "
              "OSM download step will likely fail.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name
    osm_path = out / f"{name}.osm.xml"
    net_path = out / f"{name}.net.xml"
    rou_path = out / f"{name}.rou.xml"
    cfg_path = out / f"{name}.sumocfg"

    # 1. OSM extract ----------------------------------------------------------
    if args.osm_file:
        osm_path = Path(args.osm_file)
        if not osm_path.exists():
            print(f"[build] ERROR: --osm-file not found: {osm_path}")
            return 1
    else:
        if args.lat is None or args.lon is None:
            print("[build] ERROR: provide --lat/--lon (or --osm-file).")
            return 1
        w, s, e, n = _bbox(args.lat, args.lon, args.radius)
        cmd = _sumo_tool("osmGet.py") + [
            "--bbox", f"{w},{s},{e},{n}",
            "--output-dir", str(out),
            "--prefix", name,
        ]
        if not _run(cmd, "downloading OSM extract (osmGet.py)"):
            print("[build] If osmGet.py fails, download an .osm.xml manually and "
                  "re-run with --osm-file.")
            return 1
        # osmGet writes <prefix>_bbox.osm.xml
        cand = out / f"{name}_bbox.osm.xml"
        if cand.exists():
            osm_path = cand

    # 2. netconvert → geo-referenced network ---------------------------------
    cmd = _sumo_tool("netconvert") + [
        "--osm-files", str(osm_path),
        "-o", str(net_path),
        "--geometry.remove",            # simplify collinear geometry
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",             # merge clustered intersections
        "--tls.guess-signals",          # infer signals from OSM
        "--tls.discard-simple",
        "--tls.join",
        "--remove-edges.isolated",
        "--keep-edges.by-vclass", "passenger",
        "--no-turnarounds",
    ]
    if not _run(cmd, "converting OSM → SUMO network (netconvert)"):
        return 1

    # 3. randomTrips → demand -------------------------------------------------
    # period = end / vehicles  → roughly `vehicles` insertions spread over `end`
    period = max(0.2, float(args.end) / max(1, args.vehicles))
    cmd = _sumo_tool("randomTrips.py") + [
        "-n", str(net_path),
        "-o", str(out / f"{name}.trips.xml"),
        "-r", str(rou_path),
        "--end", str(args.end),
        "--period", f"{period:.3f}",
        "--validate",
        "--fringe-factor", "10",        # bias trips to enter from the edges
    ]
    if not _run(cmd, "generating random demand (randomTrips.py)"):
        return 1

    # 4. sumocfg --------------------------------------------------------------
    cfg_path.write_text(
        f"""<configuration>
    <input>
        <net-file value="{net_path.name}"/>
        <route-files value="{rou_path.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{int(args.end)}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="120"/>
    </processing>
</configuration>
""",
        encoding="utf-8",
    )

    print(f"\n[build] DONE. Scene written to {out}/")
    print(f"[build] Run the app with:\n"
          f"        python main_ast6.py --n-cars 0 --sumo-cfg {cfg_path}")
    if args.lat is not None:
        print(f"        (use the SAME --lat {args.lat} --lon {args.lon} "
              f"--radius {args.radius} you built this scene with)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
