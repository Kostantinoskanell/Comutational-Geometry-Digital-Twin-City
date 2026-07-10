#!/usr/bin/env python
"""Headless pedestrian-safety study over a real city extract.

Runs the IDM car + crossing-pedestrian co-simulation (no window, no VTK)
under different presets and prints an injury comparison table.

Examples
--------
    # Compare signals on vs off for the default city
    python tools/run_safety_study.py --address "Patras, Greece" --radius 250

    # Longer run, more agents, plus a 30-zone traffic-calming preset
    python tools/run_safety_study.py --duration 300 --n-cars 60 --n-peds 120 --calm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", default="Patras, Greece")
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--duration", type=float, default=120.0, help="Sim seconds per preset")
    ap.add_argument("--n-cars", type=int, default=40)
    ap.add_argument("--n-peds", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calm", action="store_true",
                    help="Add a traffic-calming preset (speed ×0.6, signals on)")
    args = ap.parse_args()

    from app_core import _load_or_fetch_osm_cached
    from safety import run_safety_study

    print(f"[safety-study] loading {args.address} (r={args.radius:.0f} m) …")
    _, graph, *_ = _load_or_fetch_osm_cached(
        address=args.address, radius=args.radius, extrusion_height=10.0,
        cache_dir=Path("cache"), use_cache=True,
    )
    print(f"[safety-study] graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    presets = [
        ("signals ON ", dict(signals=True)),
        ("signals OFF", dict(signals=False)),
    ]
    if args.calm:
        presets.append(("calmed 30-zone", dict(signals=True, traffic_speed=0.6)))

    results = []
    for name, kw in presets:
        print(f"[safety-study] running preset: {name} "
              f"({args.duration:.0f} s, {args.n_cars} cars, {args.n_peds} peds)")
        r = run_safety_study(
            graph, duration_s=args.duration, n_cars=args.n_cars,
            n_peds=args.n_peds, seed=args.seed, **kw,
        )
        results.append((name, r))

    print()
    print(f"{'preset':<16} {'lights':>6} {'crossings':>9} {'cross-events':>12} "
          f"{'injuries':>8} {'severe':>6}")
    print("-" * 62)
    for name, r in results:
        if "error" in r:
            print(f"{name:<16} ERROR: {r['error']}")
            continue
        print(f"{name:<16} {r['n_lights']:>6} {r['crossings_built']:>9} "
              f"{r['ped_crossing_events']:>12} {r['injuries']:>8} {r['severe']:>6}")

    base = results[0][1].get("injuries")
    for name, r in results[1:]:
        if "error" in r or base is None:
            continue
        delta = r["injuries"] - base
        sign = "+" if delta > 0 else ""
        print(f"\n  {name.strip()} vs {results[0][0].strip()}: "
              f"{sign}{delta} injuries over {args.duration:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
