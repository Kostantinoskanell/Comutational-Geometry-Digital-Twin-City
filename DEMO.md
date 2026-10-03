# Flood Lab — presentation guide

**Story for the teacher:** *the city today → press a button → it rains, puddles form where the
water really collects → an architect designs the green corridor with the editor tools → run the
same storm again → see exactly what changed.*

## Start

```bash
python main_ast6.py --preset beirut-corridor
```

(or run `python main_ast6.py`, open the **Location & Source** tab and press **"Beirut flood demo —
Al-Masar corridor preset"**, then **Run**). First start on a new machine downloads the map data
(a few minutes); after that it is cached. Terrain is switched on automatically.

The window opens on the whole city. The **right panel is the Flood Lab**; the left panel is the
normal city control panel (traffic, weather, style, "Photoreal ground").

## 1 — The city as it is (30 s)
Orbit with the mouse (drag), zoom with the wheel. "Photoreal ground" (left panel) switches between
the real drone orthophoto and the stylised analysis ground. Mention: real 5 cm survey of Mar
Mikhael, real heights, buildings checked against the survey.

## 2 — Run the flood (≈ 45 s of waiting, all of it watchable)
Press **Run flood: existing city**.

* The camera moves in on the study area, the sky turns to heavy overcast, rain falls, the streets
  turn wet, and **water appears where it collects** while the clock at the top counts the
  hyetograph of the real **25 Nov 2025** storm (25 mm in an hour, 89 mm/h peak).
* Top line = clock, water stored, deepest water. After it ends: area flooded, depth, infiltration,
  buildings flooded.
* **View: …** cycles Water → Depth map → Hazard → Change. **Storm: …** changes the design storm
  (2-, 10-, 50-year, climate-uplift). **Replay last run** plays it again. **Fly to worst flooding**
  (key `H`) jumps to the next worst street (it picks a clear line of sight).

## 3 — Design the corridor (2–3 min, live)
Use the **DESIGN TOOLS** (the same editor the city already had):

| Tool | How |
|---|---|
| **Material:** … | press to cycle: garden, rain garden, bioswale, permeable paving, porous sidewalk, bike lane, terrace |
| **Tool: area** | click the corners of a polygon, **Enter** to finish — it is filled with the current material |
| **Tool: strip** (`S`) | click along a line (a swale, a bike lane, a sidewalk), **Enter** — buffered to a 3 m wide strip |
| **Tool: street trees** (`L`) | click to plant Mediterranean trees (ficus, jacaranda, olive, palm, pine) |
| **Tool: storm drain** (`D`) | click to place a gully inlet |
| **Tool: building** (`G`) | click to place a building (blocks the flow like the real ones) |
| stairs (`K`), roads, bridges (`Y`) | the existing tools; stairs shed water as impervious steps |
| `O` | undo the last edit (the design model follows) |
| **Official Masar corridor** | loads the architect's reference design and flies to it |

The panel footer shows what is in your design ("2 rain gardens, 14 trees, 4 drains …").

## 4 — Run the same storm with the design
Press **Run flood: with my design**. The HUD then shows both runs and the **CHANGE** line
(flooded area %, depth, hazard area, infiltration, buildings affected, flooded area within 60 m
of the design). **View: Change** paints it: **blue = shallower, red = deeper**.

## 5 — Take it away
**Export report (PNG + MD)** writes `reports/flood_<time>/` with a three-panel figure
(existing / design / change), a one-page report table and the raw depth rasters.

## What to say about the numbers
* The solver is a local-inertial shallow-water model (Bates et al. 2010) on the real terrain,
  same numerics as the validated GPU solver in `Beirut_Project-main`. On its native 0.5 m grid it
  reproduces that solver's flooded area with **IoU 0.86 and depth correlation 0.95**; the live
  2 m preview trades some fine detail for speed (design comparison, not sizing).
* The official corridor cuts flooded area by about **10 %** and raises infiltration from ~5 % to
  ~14 % of the rain (published study: −12 to −14 %).
* The city's 600 drain inlets are treated as **clogged** — that is what happened on 25 Nov 2025.
  A design adds only the drains you place.
* Mass balance closes to 1e-13: every cubic metre of rain is accounted for.

## If something goes wrong
| Symptom | Do |
|---|---|
| No Flood Lab panel on the right | the scene did not overlap the study area: start with `--preset beirut-corridor` |
| Slow first run | the first run after install compiles the solver once (~3 s) |
| A run is stuck | **Stop / clear results** |
| Everything looks too dark | the storm sky is on; **Stop / clear results** restores the clear day |
| Need the fast preview only | `--flood-lab-res preview` (default); `fine` = 1 m grid (minutes) |

## Prepared material (no GUI needed)
`python tools/headless_session.py --script tools/demo_flood_story.py --address "beirut corridor" --radius 700 --terrain-on`
regenerates screenshots and videos of the whole story in `demo_out/`.
