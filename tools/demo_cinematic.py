"""Cinematic GIF: storm + puddles forming -> authoring the corridor (tools) -> same storm with the design.

  python tools/headless_session.py --script tools/demo_cinematic.py --size 1280x800 \
      --address "beirut corridor" --radius 700 --terrain-on
Writes $DEMO_OUT/cinematic_flood.gif (default <project>/demo_out).
"""
import os, copy
import numpy as np
from pathlib import Path

OUT = Path(os.environ.get("DEMO_OUT", str(ROOT / "demo_out"))); OUT.mkdir(exist_ok=True)
lab = app.flood_lab
import flood_lab_mixin as FL
from floodsim.engine import FloodEngine

app._build_flood_lab_panel()
if not app.scene_state.get("_terrain_drape_active"):
    app._toggle_terrain()
dem = app.street_graph.graph["terrain_sampler"]

# where the water collects (hidden probe run) -> the film's location
b = copy.copy(app._lab_inputs()[0]); b.drains = None
probe = FloodEngine(b).run(app._lab_storm()["steps"], 3600.0, save_every=1e9, chunk_s=1e9)
hx, hy, _ = app._lab_hotspots(probe.max_depth, k=1)[0]
hz = float(dem(np.array([[hx, hy]]))[0])
dirv = np.array([1.0, 0.0]); perp = np.array([0.0, 1.0])
P = lambda a, c: (hx + a, hy + c)

rec = Recorder(OUT / "cinematic_flood.gif", fps=10, width=480)


def clean_screen():
    """Cinematic framing: hide the control panels, widgets, key legend and mode bar; keep the
    flood HUD and our captions."""
    keep = ("lab_hud_", "cine_caption")
    for a in list(plotter.renderer.GetActors2D()):
        name = None
        for k in plotter.renderer._actors.keys():
            if plotter.renderer._actors[k] is a:
                name = k; break
        if name is None or not name.startswith(keep):
            a.SetVisibility(False)
    for fn in ("clear_button_widgets", "clear_slider_widgets"):
        try:
            getattr(plotter, fn)()
        except Exception:
            pass
    props = plotter.renderer.GetViewProps()
    props.InitTraversal()
    for _ in range(props.GetNumberOfItems()):          # button / slider widget representations
        pr = props.GetNextProp()
        if pr is not None and ("Representation" in pr.GetClassName() or "Button" in pr.GetClassName()):
            pr.SetVisibility(False)
    for k in list(plotter.renderer._actors.keys()):
        if k.startswith(("lab_lbl_", "lab_title", "lab_hdr", "panel_", "editor_mode", "key_legend", "road_info")):
            try:
                plotter.renderer._actors[k].SetVisibility(False)
            except Exception:
                pass


clean_screen()
state = {"az": 28.0, "dist": 105.0, "el": 27.0, "n": 0}


def cam():
    a, e = np.radians(state["az"]), np.radians(state["el"])
    d = state["dist"]
    return [(hx + d * np.cos(e) * np.sin(a), hy - d * np.cos(e) * np.cos(a), hz + d * np.sin(e) + 2.0), (hx, hy, hz + 0.5), (0, 0, 1)]


_clean = {"n": -1}


def clean_screen_once():
    """Re-hide UI that the editor tools re-create (mode bar, HUD hints) once per call burst."""
    n = len(list(plotter.renderer._actors.keys()))
    if n != _clean["n"]:
        _clean["n"] = n
        clean_screen()


def caption(text):
    plotter.add_text(text, position=(0.5, 0.075), viewport=True, name="cine_caption", font_size=17,
                     color="#ffffff", shadow=True, font="arial")


def frame(n=1, d_az=0.5, d_dist=0.0, d_el=0.0):
    clean_screen_once()
    for _ in range(n):
        state["az"] += d_az; state["dist"] += d_dist; state["el"] += d_el
        plotter.render()
        rec.grab(cam(), view_angle=45)


# --- intro: the dry street
caption("Mar Mikhael, Beirut  -  25 Nov 2025 storm")
for _ in range(10):
    frame(1, 0.8)

# --- 1. the storm, puddles forming
caption("Heavy rain - water collects on the street")
tick = {"n": 0}


def on_tick(snap):
    app._animate_weather(0)
    tick["n"] += 1
    if tick["n"] % 22 == 0:
        frame(1, 0.35)


app.flood_lab_run(design=False, block=True, on_tick=on_tick)
base_s = lab["runs"]["baseline"]["summary"]
for _ in range(10):
    frame(1, 0.5)

# --- 2. authoring the corridor with the real editor tools
caption("The architect designs the corridor ...")
plotter.add_text("", name="lab_dummy")
state["el"] = 40.0; state["dist"] = 130.0
mats = {c: i for i, (c, _) in enumerate(FL.MATERIAL_CYCLE)}


def hold(n=5, d_az=0.6):
    frame(n, d_az)


def mode_click(mode, pts):
    app._set_editor_mode(mode)
    for x, y in pts:
        app._unified_pick_callback((x, y, 0.0), picker="_deferred")


# rain garden
lab["material_idx"] = mats[6]; caption("Rain garden")
app._set_editor_mode("greenspace"); app.scene_state["_greenspace_pts"] = [P(-14, -9), P(14, -9), P(14, 7), P(-14, 7)]
app._finalize_greenspace(); hold(7)
# bioswale + bike lane + permeable paving strips
for cls, label, off0, off1 in ((3, "Bioswale", 12, 15), (2, "Permeable bike lane", 20, 23), (8, "Permeable paving", -14, -12)):
    lab["material_idx"] = mats[cls]; caption(label)
    app._set_editor_mode("strip")
    app.scene_state["_strip_pts"] = [P(-70, off0), P(-10, off0 - 1), P(40, off0 + 1), P(95, off1)]
    app._finalize_strip(); hold(6)
# trees one by one
caption("Street trees")
for i, a in enumerate(range(-66, 96, 12)):
    app._add_tree_at(*P(a, 26 + 0.03 * a)); 
    if i % 2 == 0:
        hold(2, 0.7)
hold(3)
# stairs (real two-click tool) beside the garden
caption("Stairs")
mode_click("stairs", [P(-30, 30), P(-30, 44)]); hold(6)
# drains
caption("Storm drains")
for a in (-50, -10, 30, 70):
    app._add_drain_at(*P(a, 0)); hold(2)
# a bridge (existing highway tool) between two nearby street nodes
caption("Footbridge / overpass")
try:
    nodes = [(n, d["x"], d["y"]) for n, d in app.street_graph.nodes(data=True) if "x" in d]
    near = sorted(nodes, key=lambda t: (t[1] - hx) ** 2 + (t[2] - hy) ** 2)[:60]
    pair = None
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            d = np.hypot(near[i][1] - near[j][1], near[i][2] - near[j][2])
            if 45 < d < 160:
                pair = (near[i], near[j]); break
        if pair: break
    if pair:
        mode_click("highway", [(pair[0][1], pair[0][2]), (pair[1][1], pair[1][2])])
        state["dist"] = 180.0; state["el"] = 34.0
        hold(8, 0.9)
except Exception as exc:
    print("bridge skipped:", exc)
app._set_editor_mode("view")
print("design:", lab["design"].summary())
state["dist"] = 125.0; state["el"] = 32.0
for _ in range(6):
    frame(1, 0.8, -1.0)

# --- 3. same storm, with the design
caption("Same storm - with the new corridor")
tick["n"] = 0
app.flood_lab_run(design=True, block=True, on_tick=on_tick)
des_s = lab["runs"]["design"]["summary"]
for _ in range(8):
    frame(1, 0.5)

# --- 4. what changed
lab["view_idx"] = 3; app._lab_refresh_labels(); app._lab_show_current()
fa = 100 * (des_s["flooded_area_ha"] - base_s["flooded_area_ha"]) / base_s["flooded_area_ha"]
caption(f"Change in flood depth  (blue = drier)  -  flooded area {fa:+.0f} %")
state["dist"] = 230.0; state["el"] = 55.0
for _ in range(22):
    frame(1, 0.9, -0.8)
rec.close()
print("GIF", OUT / "cinematic_flood.gif", rec.n, "frames")
