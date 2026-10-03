#!/usr/bin/env python
"""Headless render benchmark per quality preset (WG performance gate).

Builds the real twin scene off-screen (no window, no startup menu), then
times steady-state frames for each post-processing configuration, with the
first frames after every pipeline change excluded (shader compilation).
Target: >= 30 fps at 'quality' for the corridor scene on the M1.

Examples
--------
    python tools/render_benchmark.py --address "Mar Mikhael, Beirut, Lebanon" --radius 250
    python tools/render_benchmark.py --size 2880x1800 --frames 60 --json bench.json

Off-screen timings approximate the interactive window at the same pixel
size (Retina windows have 2x the logical size in pixels: pass --size).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default="Mar Mikhael, Beirut, Lebanon")
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--size", default="1600x1000", help="render size WxH in pixels")
    ap.add_argument("--frames", type=int, default=30, help="timed frames per configuration")
    ap.add_argument("--warmup", type=int, default=5, help="untimed frames after each change")
    ap.add_argument("--terrain", action="store_true", help="benchmark with the terrain drape on")
    ap.add_argument("--json", default="", help="write results to this JSON file")
    args, passthrough = ap.parse_known_args()

    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.chdir(ROOT)
    import numpy as np
    import pyvista as pv

    w, h = (int(v) for v in args.size.lower().split("x"))
    results: dict = {"address": args.address, "radius": args.radius, "size": [w, h], "configs": {}}
    holder: dict = {}

    def bench(plotter) -> None:
        app = holder["app"]
        plotter.window_size = (w, h)
        plotter.ren_win.Render()        # realize the window and GPU resources
        if args.terrain:
            app._toggle_terrain()
        fx = getattr(app, "postfx", None)

        win = plotter.ren_win           # pyvista's render() is a no-op before first show

        def timed(tag: str) -> None:
            for _ in range(args.warmup):
                win.Render()
            t0 = time.perf_counter()
            for _ in range(args.frames):
                win.Render()
            ms = 1000.0 * (time.perf_counter() - t0) / args.frames
            results["configs"][tag] = {"ms_per_frame": round(ms, 2), "fps": round(1000.0 / ms, 1)}
            print(f"  {tag:<34} {ms:7.2f} ms/frame  {1000.0 / ms:6.1f} fps")

        print(f"\nrender benchmark {w}x{h}, {args.frames} frames/config"
              f"{' (terrain on)' if args.terrain else ''}:")
        timed("no post-processing")
        if fx is not None:
            fx.update(ssao=False, fxaa=True, tone_mapping=False)
            timed("performance (FXAA)")
            fx.update(ssao=True, fxaa=True, tone_mapping=False)
            timed(f"quality (SSAO k={fx.settings.ssao_kernel} + FXAA)")
        mats = getattr(app, "materials", None)
        results["actors"] = int(plotter.renderer.GetActors().GetNumberOfItems())
        results["materials"] = dict(mats.stats) if mats is not None else None
        print(f"  actors: {results['actors']}  materials: {results['materials']}")
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
            print(f"  wrote {args.json}")
        sys.stdout.flush()
        os._exit(0)          # the app keeps simulating after show(); stop here

    pv.Plotter.show = lambda self, *a, **k: bench(self)
    sys.argv = ["main_ast6.py", "--address", args.address, "--radius", str(args.radius), "--no-gui", *passthrough]
    import main_ast6
    app = main_ast6.DigitalTwinApp()
    holder["app"] = app
    if not app._init_ok:
        print("scene initialisation failed")
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
