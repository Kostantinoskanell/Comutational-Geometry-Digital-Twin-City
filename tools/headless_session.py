#!/usr/bin/env python
"""Drive the real twin headless: build the scene, then run a script against it.

    python tools/headless_session.py --script demo.py --address "beirut corridor" --radius 700

The script is exec'd with `app` (DigitalTwinApp, fully built), `plotter`, `np`,
`pv` and the helpers below in scope. No window or startup menu opens (the
session forces --no-gui and an off-screen plotter); the process exits when the
script returns. Output paths must be ABSOLUTE (the app chdirs to the project).

Helpers
-------
shot(path, camera=None, size=None, view_angle=30)     screenshot (camera = [pos, focal, up])
orbit_camera(app, azim_deg, elev_deg, dist, target)   camera tuple
Recorder(path, fps=12).grab()/close()                 GIF writer (imageio)
pump(app, seconds)                                    run UI timers (flood lab polling) for N s
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--script", required=True)
    ap.add_argument("--size", default="1600x1000")
    known, passthrough = ap.parse_known_args()

    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.chdir(ROOT)
    import numpy as np
    import pyvista as pv

    w, h = (int(v) for v in known.size.lower().split("x"))
    script = Path(known.script).read_text()

    def run(plotter):
        import main_ast6  # noqa: F401  (already imported)
        app = holder["app"]
        plotter.window_size = (w, h)
        try:
            plotter.camera.ParallelProjectionOff()
        except Exception:
            pass
        app.scene_state["interactive_ready"] = True        # what the 300 ms startup timer sets in the live app
        ns = dict(app=app, plotter=plotter, np=np, pv=pv, ROOT=ROOT, time=time)
        ns.update(_helpers(app, plotter, np, pv))
        try:
            exec(compile(script, known.script, "exec"), ns)
            rc = 0
        except SystemExit as e:
            rc = int(e.code or 0)
        except BaseException:
            import traceback
            traceback.print_exc()
            rc = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)

    holder: dict = {}
    pv.Plotter.show = lambda self, *a, **k: run(self)
    sys.argv = ["main_ast6.py", "--no-gui", *passthrough]
    import main_ast6
    app = main_ast6.DigitalTwinApp()
    holder["app"] = app
    if not app._init_ok:
        print("scene initialisation failed")
        return 1
    app.run()
    return 0


def _helpers(app, plotter, np, pv):
    def shot(path, camera=None, size=None, view_angle=30.0, parallel=False):
        if size:
            plotter.window_size = tuple(size)
        if camera is not None:
            plotter.camera_position = camera
        plotter.camera.view_angle = view_angle
        if parallel:
            plotter.camera.ParallelProjectionOn()
        else:
            plotter.camera.ParallelProjectionOff()
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
        img = plotter.screenshot(str(path), return_img=True)
        return img

    def orbit_camera(azim_deg, elev_deg, dist, target):
        a, e = np.radians(azim_deg), np.radians(elev_deg)
        tx, ty, tz = target
        pos = (tx + dist * np.cos(e) * np.sin(a), ty - dist * np.cos(e) * np.cos(a), tz + dist * np.sin(e))
        return [pos, tuple(target), (0, 0, 1)]

    class Recorder:
        """Video/GIF writer. .mp4 (H.264, small, smooth) or .gif (downscaled + subsampled so it
        stays shareable). Frames are grabbed at the plotter's size and resized to `width`."""
        def __init__(self, path, fps=12, width=1280, quality=8):
            import imageio.v2 as iio
            self.path = str(path)
            self.width = int(width)
            self.fps = fps
            self.n = 0
            if self.path.endswith(".mp4"):
                self.w = iio.get_writer(self.path, fps=fps, quality=quality, macro_block_size=2, codec="libx264",
                                        pixelformat="yuv420p")
            else:
                self.w = iio.get_writer(self.path, mode="I", duration=1000.0 / fps, loop=0, palettesize=128)
            self.gif = not self.path.endswith(".mp4")

        def _frame(self):
            from PIL import Image
            img = plotter.screenshot(return_img=True)
            h, w = img.shape[:2]
            if self.width and w != self.width:
                im = Image.fromarray(img[:, :, :3]).resize((self.width, int(round(h * self.width / w / 2)) * 2), Image.LANCZOS)
                img = np.asarray(im)
            return img[:, :, :3]

        def grab(self, camera=None, view_angle=30.0):
            if camera is not None:
                plotter.camera_position = camera
            plotter.camera.view_angle = view_angle
            plotter.camera.ParallelProjectionOff()
            plotter.renderer.ResetCameraClippingRange()
            plotter.render()
            self.w.append_data(self._frame())
            self.n += 1

        def close(self):
            self.w.close()

    def pump(seconds, render=True):
        """Let time pass for UI-driven features: run the app's timer callbacks."""
        t_end = time.time() + seconds
        while time.time() < t_end:
            for cb in getattr(app, "_headless_timers", []):
                cb(0)
            if render:
                plotter.render()
            time.sleep(0.02)

    return dict(shot=shot, orbit_camera=orbit_camera, Recorder=Recorder, pump=pump)


if __name__ == "__main__":
    sys.exit(main())
