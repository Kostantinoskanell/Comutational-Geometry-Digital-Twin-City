"""The flood-study story, scripted against the real app (run through tools/headless_session.py).

  python tools/headless_session.py --script tools/demo_flood_story.py --address "beirut corridor" --radius 700 --terrain-on

Writes screenshots and GIFs to $DEMO_OUT (default <project>/demo_out).
"""
import os
import numpy as np
from pathlib import Path

OUT = Path(os.environ.get("DEMO_OUT", str(ROOT / "demo_out")))
OUT.mkdir(exist_ok=True)
lab = app.flood_lab
assert lab is not None, "Flood Lab not available"
georef = lab["georef"][2.0]
gb = georef.bounds()
dem = app.street_graph.graph["terrain_sampler"]


def z_at(x, y):
    return float(dem(np.array([[x, y]]))[0])


app._build_flood_lab_panel()
if not app.scene_state.get("_terrain_drape_active"):
    app._toggle_terrain()
cx, cy = (gb[0] + gb[1]) / 2, (gb[2] + gb[3]) / 2
wide_cam = orbit_camera(18, 50, 900, (cx, cy, 20))
base_inp, _ = app._lab_inputs()

# ---- 0. a first quick run only to know where the water collects (hidden from the story)
quiet = app.flood_lab_run
import copy
from floodsim.engine import FloodEngine
_probe = FloodEngine(copy.copy(base_inp)); _probe.inp.drains = None
_pr = _probe.run(lab["storm_def"]["steps"] if lab["storm_def"] else app._lab_storm()["steps"], 3600.0, save_every=1e9, chunk_s=1e9)
spots = app._lab_hotspots(_pr.max_depth, k=6)
print("hotspots:", [(round(x), round(y), round(a)) for x, y, a in spots])
hx, hy, _ = spots[0]
hz = z_at(hx, hy)


def street_cam(az, dist=85.0, el=24.0, spot=None):
    x, y = (hx, hy) if spot is None else spot
    return app._lab_clear_camera(x, y, dist, az)


# ---- 1. the city as it is
shot(OUT / "01_city_before.png", wide_cam, view_angle=30)
shot(OUT / "01b_street_before.png", street_cam(30), view_angle=42)

# ---- 2. storm on the existing city: rain + sky + puddles forming, orbiting the hotspot street
CAM0 = street_cam(30, 95.0)
rec = Recorder(OUT / "02_storm_existing_city.mp4", fps=24)
frame = {"n": 0}


def grab_cb(snap):
    app._animate_weather(0)
    frame["n"] += 1
    if frame["n"] % 4 == 0:
        rec.grab(CAM0, view_angle=44 - 0.01 * frame["n"])


app.flood_lab_run(design=False, block=True, on_tick=grab_cb)
for k in range(8):
    rec.grab(CAM0, view_angle=38)
rec.close()
b = lab["runs"]["baseline"]
print("baseline:", {k: round(v, 3) for k, v in b["summary"].items()})
shot(OUT / "03_flood_existing_wide.png", orbit_camera(18, 62, 700, (cx, cy, 20)))
for i, (x, y, a) in enumerate(spots[:3]):
    shot(OUT / f"04_flood_existing_street{i + 1}.png", street_cam(40 + 50 * i, 85, 24, (x, y)), view_angle=42)
lab["view_idx"] = 2; app._lab_refresh_labels(); app._lab_show_current()
shot(OUT / "05_hazard_existing.png", orbit_camera(18, 60, 550, (hx, hy, hz)))
lab["view_idx"] = 1; app._lab_refresh_labels(); app._lab_show_current()
shot(OUT / "05b_depth_map_existing.png", orbit_camera(18, 60, 550, (hx, hy, hz)))
lab["view_idx"] = 0; app._lab_refresh_labels(); app._lab_show_current()

# ---- 3. the architect designs the corridor with the editor tools (same calls the clicks make)
import flood_lab_mixin as FL
cls_i = {c: i for i, (c, _) in enumerate(FL.MATERIAL_CYCLE)}
# street direction at the hotspot: along the principal axis of the wet cells nearby
dirv = np.array([1.0, 0.0])
ex, ey = hx, hy
perp = np.array([-dirv[1], dirv[0]])


def P(a, b):
    return (ex + a * dirv[0] + b * perp[0], ey + a * dirv[1] + b * perp[1])


app._set_editor_mode("greenspace")
lab["material_idx"] = cls_i[6]
app.scene_state["_greenspace_pts"] = [P(-14, -9), P(14, -9), P(14, 7), P(-14, 7)]
app._finalize_greenspace()
lab["material_idx"] = cls_i[3]
app._set_editor_mode("strip"); app.scene_state["_strip_pts"] = [P(-70, 12), P(-10, 11), P(40, 13), P(95, 15)]
app._finalize_strip()
lab["material_idx"] = cls_i[2]
app.scene_state["_strip_pts"] = [P(-70, 18), P(95, 21)]
app._finalize_strip()
lab["material_idx"] = cls_i[8]
app.scene_state["_strip_pts"] = [P(-70, -14), P(95, -12)]
app._finalize_strip()
for a in range(-66, 96, 12):
    app._add_tree_at(*P(a, 24))
for a in (-50, -10, 30, 70):
    app._add_drain_at(*P(a, 0))
print("design:", lab["design"].summary())
shot(OUT / "06_design_authored.png", street_cam(35, 110, 30), view_angle=42)
shot(OUT / "06b_design_authored_top.png", orbit_camera(20, 65, 260, (hx, hy, hz)), view_angle=30)

# ---- 4. same storm on the designed city
rec = Recorder(OUT / "08_storm_with_design.mp4", fps=24)
frame["n"] = 0
app.flood_lab_run(design=True, block=True, on_tick=grab_cb)
for k in range(8):
    rec.grab(CAM0, view_angle=38)
rec.close()
d = lab["runs"]["design"]
print("design:", {k: round(v, 3) for k, v in d["summary"].items()})
shot(OUT / "09_flood_with_design_street.png", CAM0, view_angle=38)
lab["view_idx"] = 3; app._lab_refresh_labels(); app._lab_show_current()
shot(OUT / "10_change_map_street.png", orbit_camera(25, 50, 300, (hx, hy, hz)), view_angle=30)
shot(OUT / "11_change_map_wide.png", orbit_camera(18, 62, 700, (cx, cy, 20)), view_angle=30)

# ---- 5. the official Al-Masar corridor design (architect's reference), same storm
lab["design"].clear()                                   # official corridor alone, to compare with the published study
app._lab_toggle_official()                              # loads the design and flies to the corridor
CORR = [tuple(plotter.camera.position), tuple(plotter.camera.focal_point), (0, 0, 1)]
shot(OUT / "12_official_corridor_design.png", CORR, view_angle=40)
rec = Recorder(OUT / "14_storm_official_corridor.mp4", fps=24)
frame["n"] = 0


def grab_corr(snap):
    app._animate_weather(0)
    frame["n"] += 1
    if frame["n"] % 4 == 0:
        rec.grab(CORR, view_angle=40)


app.flood_lab_run(design=True, block=True, on_tick=grab_cb if False else grab_corr)
for k in range(8):
    rec.grab(CORR, view_angle=40)
rec.close()
o = lab["runs"]["design"]
print("official:", {k: round(v, 3) for k, v in o["summary"].items()})
lab["view_idx"] = 3; app._lab_refresh_labels(); app._lab_show_current()
shot(OUT / "12b_change_map_official_corridor.png", CORR, view_angle=40)
shot(OUT / "12c_change_map_official_overview.png", orbit_camera(15, 64, 760, (cx, cy, 20)), view_angle=30)
lab["view_idx"] = 0; app._lab_refresh_labels(); app._lab_show_current()
shot(OUT / "13_flood_official_overview.png", orbit_camera(15, 64, 760, (cx, cy, 20)), view_angle=30)
print("EXPORT", app.flood_lab_export(str(OUT)))
print("DONE")
