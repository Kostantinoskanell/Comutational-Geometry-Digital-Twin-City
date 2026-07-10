# Digital Twin City

> **Session context note** — this README is updated each session to track the current state of
> features, key bindings, known bugs, and pending work.  Claude should read this first before
> making any changes.  Last updated: 2026-07-04.

---

## Current State (what works, what was recently changed)

### Recently added / fixed
| Change | File(s) |
|---|---|
| **Pedestrian street crossings + injury tracking** — footway nodes facing each other across a road get synthetic crossing links (peds pause 1.5–3 s then cross; cars within 12 m yield); a moving car (> 2 m/s) within 1.4 m injures a ped (lies flat, frozen, red HUD counter `⚠ pedestrians injured: N`, severity ≥ 30 km/h = SEVERE); core ped tick extracted to `safety.py::advance_peds_core` shared by app + headless | `safety.py` (new), `ped_mixin.py`, `glyph_instance.py` (X-tilt) |
| **Headless safety study** — `python tools/run_safety_study.py --address ... --duration 120 --calm` runs the IDM+ped co-simulation with presets (signals on/off, 30-zone calming) and prints an injury comparison table; also callable as `safety.run_safety_study(graph, signals=..., traffic_speed=...)` | `tools/run_safety_study.py` (new), `safety.py` |
| **Route planner sees editor bridges** — cost cache key now includes network fingerprint (car_paths length + edge count); test proves a built highway wins by length and travel-time (energy route may legitimately avoid it — drag ∝ v²) | `route_mixin.py`, `tests/run_tests.py` |
| **pyvista default key bindings cleared** — `v` (isometric camera reset — broke centrality view), `b` (fake mouse press), `C` (cell picking), `Up`/`Down` (zoom fighting arrow pan) | `main_ast6.py` |
| **Centrality all-purple fix** — MultiDiGraph betweenness keys are `(u,v,key)` triples; `_collapse_edge_centrality` normalizes to pairs | `analysis_mixin.py` |
| **Noise model realism** — `_L_REF` 55→85 dB(A) (real car pass-by at 1 m); ~100 m halo/car, energy-scale stacking, breach counter meaningful; quiet cells transparent (NaN < 45 dB) | `analysis_mixin.py` |
| **Walk-mode blue screen fix** — sea plane (7× city) blew far clip to ~5 km, VTK forced near ≥ 5 m clipping all nearby geometry; walk modes pin `clipping_range=(0.4, 6000)`; 1p camera now applies mouse pitch | `walk_mixin.py` |
| **u/i/b keys always bound** — with SUMO/GTFS active they work; otherwise on-screen hint explains the needed startup flag (were silently unregistered) | `main_ast6.py`, `sumo_mixin.py`, `bus_mixin.py` |
| **Emergency vehicles drivable-only routing** — ambulances no longer route down footways/stairs (`_em_drivable_graph`); red-light running remains (siren priority) | `emergency_mixin.py` |
| **SUMO first-class engine mode** — `--engine {idm,sumo}` selects primary traffic source; sumo mode auto-builds+caches scene from `--address/--radius`; analysis overlays (heatmap, noise, AQ) read from SUMO snapshot; walk camera follows SUMO vehicles/persons; graceful IDM fallback if SUMO unavailable | `app_cli.py`, `sumo_engine.py` (new), `main_ast6.py`, `analysis_mixin.py`, `heatmap_mixin.py`, `walk_mixin.py`, `sumo_mixin.py` |
| **Physics/render decoupling** — physics runs at fixed 20 Hz accumulator; render uses linear interpolation between prev/curr physics states; TL colors ≤10 Hz; overlays ≤3 Hz | `car_mixin.py`, `ped_mixin.py`, `cyclist_mixin.py`, `main_ast6.py` |
| **Terrain draping** — pressing `g` lifts roads, sidewalks, TL anchors, parked cars, streetlight poles, and trees onto DEM surface; agents follow via precomputed path-height profiles (O(N) numpy); shadow cache cleared + recomputed on toggle; toggle-off restores flat z | `terrain_drape.py` (new), `terrain_mixin.py`, `car_mixin.py`, `ped_mixin.py`, `cyclist_mixin.py`, `sumo_mixin.py`, `ui_mixin.py`, `main_ast6.py` |
| **SUMO DEM vectorized** — batch DEM lookup for all vehicles/persons per frame (was one scipy call per vehicle) | `sumo_mixin.py` |
| Walk/follow camera — `f` cycles off→3p→1p→free-walk | `walk_mixin.py`, `main_ast6.py` |
| SUMO vehicle type meshes (moto, bicycle, truck, rail, bus_box, van, big, car) | `sumo_mixin.py` |
| SUMO pedestrian rendering (green cylinders via `conn.persons()`) | `sumo_mixin.py`, `sumo_bridge.py` |
| Synthetic street-tree fallback when OSM tree data is empty | `main_ast6.py` |
| Fill mesh palette → cool slate greys; base ground `#3a3f46` (was brown) | `osm_3d_buildings.py`, `main_ast6.py` |
| Fill mesh versioning (`_FILL_VER`) forces cache re-fetch on palette change | `app_core.py` |
| **Water/sea in fill mesh** — `natural=water`, `natural=bay`, `waterway=riverbank`, etc. | `osm_3d_buildings.py` |
| **Park mesh removed from sidewalk merge** — fill mesh handles parks as flat solid polygons; extruded park_mesh created crosshatch artefact | `app_core.py` |
| **Pedestrian fallback** — cylinder instances (r=0.22m, h=1.7m) instead of invisible point cloud | `ped_mixin.py` |
| **Car deadlock recovery** — stuck > 8 s at near-zero speed → teleport to random edge | `idm.py` |
| Environment checkboxes in left panel (Weather, Terrain, AQ overlay) at rows 14-17 | `main_ast6.py` |
| Centrality moved from `c` to `v` (`c` reserved for camera reset) | `main_ast6.py` |
| **IDM heterogeneity** — per-car T/a/b parameters sampled from normal distributions (seeded); speed-factor array added to `car_anim`; heterogeneity re-sampled on `_rebuild_traffic_and_arrows` | `idm.py`, `main_ast6.py`, `route_mixin.py` |
| **Congested validation** — `validate_congested()` samples active IDM speeds to estimate congested travel times, compares vs OSRM, reports MAPE/RMSE/GEH per road class; wired to `--validate-congested N` CLI flag | `validation.py`, `app_cli.py`, `main_ast6.py` |
| **Engine comparison** — `validate_engine_comparison()` compares IDM vs SUMO travel times over shared O-D pairs; wired to `--validate-engines N` CLI flag | `validation.py`, `app_cli.py`, `main_ast6.py` |
| **TL editor fix** — traffic light toggle in editor mode now properly calls `_phase_timings`, builds `approach_points`, stores `_tl_glyphs` in `scene_state`, and syncs `traffic_lights_dict` | `ui_mixin.py` |
| **Headless test suite** — `python tests/run_tests.py` runs 38 tests (IDM physics + chain braking + lane changing, TL FSM, emissions, validation metrics, demand model, terrain draping, SUMO export + live TraCI, scenario compare, streetlight GA + smart placement, functional behavior) with no VTK window; safe on macOS M1 | `tests/run_tests.py` (new) |
| **Lane changing ACTIVE** — `adj_left_paths`/`adj_right_paths` arrays built from `adjacent_left_path`/`adjacent_right_path` after `_extract_drivable_paths` and passed through `idm_tick` → `find_leaders`; slow cars (< 80 % of desired) with tight gaps (< 15 m) now merge to a clearer adjacent lane; verified by `idm_lane_change_moves_slow_car` test | `idm.py`, `main_ast6.py`, `route_mixin.py`, `car_mixin.py` |
| **Bug fix: deadlock teleport fired at red lights** — cars legitimately stopped at a red TL accumulated `stuck_time` and teleported after 8 s (front of any queue at a long red vanished); Step 8 now recomputes TL-blocked mask and exempts those cars | `idm.py` |
| **Bug fix: `vehicle_pos_xy` misleading name** — returned (lon, lat) despite the `_xy` suffix; renamed `vehicle_lonlat` before anything relied on it | `sumo_bridge.py` |
| **Night window lights** — after 20:00 / before 06:00, ≤300 warm point sprites (`#f5c842`) sampled from building-face centroids appear (seeded by hour, no flicker; rebuilt only when hour moves ≥ 1 h); wired into `_apply_tod_visuals` | `tod_mixin.py` |
| **GTFS-RT failure logging** — parse errors printed, consecutive fetch failures logged every 5th, stale-feed warning once when data older than 3× poll interval | `gtfs_realtime.py` |
| **Streetlight GA verified headless** — full pipeline (grid candidates → coverage matrix with building-occlusion LOS → GA optimize) and editor smart placement both covered by regression tests: right light count, in-bounds, ≥ 5 m spacing, deterministic per seed, GA cost monotone improvement | `tests/run_tests.py` |
| **Bug fix: road network lines not terrain-draped** — the `vehicle_actor`/`ped_actor` line layers stayed flat at z≈0 while terrain lifted everything else, so roads vanished under the DEM; both are now lifted via `_drape_actor_points` with a +0.6 m bias (line vertices sample the DEM at different points than ground triangles, so without the bias they z-fight on slopes) and restored on toggle-off | `terrain_mixin.py` |
| **Bug fix: magenta screen + freeze on editor clicks** — root cause was the SSAO pass: actor churn during rebuilds triggers renders whose interrupted SSAO pass presents its *normals G-buffer* (the flat magenta/lavender frame). Editor modes (roads/roundabouts/lights/stops) now defer their heavy work via a one-shot timer (`_defer_editor_work`), **suspend SSAO for the rebuild**, present a clean frame first, and re-enable SSAO after; the picker is also pinned to the software cell picker. Roundabout indicator cylinder is DEM-aware when terrain is draped | `ui_mixin.py` |
| **Bug fix: `pv` UnboundLocalError on roundabout insert** — `_rebuild_traffic_and_arrows` had a local `import pyvista as pv` *below* a `pv.Cylinder` usage, shadowing the module-level import for the whole function; local import removed | `route_mixin.py` |
| **Bug fix: roads buried under parks/squares with terrain on** — large fill polygons (plazas/parks) spanned curved terrain with single flat triangles that bulged above the finer road mesh; the fill mesh is now `triangulate().subdivide(2)`-ed before DEM lifting (RGB cell data preserved) and the ground mesh gets a +0.3 m bias | `terrain_mixin.py` |
| **Bug fix: route overlays flat under DEM** — the 3 route polylines (energy/joint/shortest) and source/target marker spheres now sample the terrain sampler and drape when terrain is active | `route_mixin.py` |
| **Bug fix: stale terrain profiles after editor rebuild** — inserting a roundabout changed the path count but height profiles weren't rebuilt (`profiles.shape[0] < n_paths` → slow live-DEM fallback each tick); `_rebuild_traffic_and_arrows` now refreshes profiles when draping is active | `route_mixin.py` |
| **Bridge/overpass logic** — OSM `bridge` + `layer` tags elevate edges: line layers and car-lane paths ride a ramped profile (0 → 5 m/layer over 18 m ramps, so vehicles climb on/off the deck); road network lines also raised to z=0.35 base so they always render above road/sidewalk surfaces | `app_core.py` (`_edge_bridge_height`, `_bridge_z_offsets`), `main_ast6.py` |
| **Editor lights unified with built-ins** — the lights editor now just toggles `highway="traffic_signals"` on the node and rebuilds: one bulb per incoming approach, proper phase grouping/timings, survives later rebuilds (old bespoke light object was wiped by any rebuild), and the SUMO export creates a real `type="traffic_light"` junction (fixes the "not a signalised junction — skipped" TraCI failure) | `ui_mixin.py` |
| **Bug fix: SUMO editor rebuild always failed with "no proj_str in graph"** — the fallback tmerc projection derived at SUMO connect time was never stored into `street_graph.graph`, so every editor-triggered network export aborted; now stored | `sumo_mixin.py` |
| **SSAO stays off after editor edits** — VTK's SSAO pass is corrupted by rebuild actor churn and then presents its normals G-buffer (persistent magenta) on every later frame; re-enabling right after the rebuild brought it straight back, so it now stays off with a console note (re-enable via the SSAO panel checkbox) | `ui_mixin.py` |
| **Crosswalks + lane markings now drape properly** — both baked the DEM into their geometry at build time (floating when terrain off, flat when the sampler loaded late, never re-toggling); now built flat, actors stored in `scene_state`, lifted/restored by the terrain toggle like every other layer (`+0.35` bias above the ground mesh); `stop_signs_actor` added to the drape list too | `main_ast6.py`, `terrain_mixin.py` |
| **Bug fix: ambulance spawn crashed on EVERY 'm' press** — `int(rng.choice(n, size=2))` raised TypeError (can't cast a 2-element array) outside the try block, so no emergency vehicle ever spawned; also stale "press 'x'" message → 'm'; ambulance routes now only take DEM z when terrain draping is active (they floated over the flat city before) | `emergency_mixin.py` |
| **Bug fix: buses teleported back to route start on spawn** — buses spawn at a random distance along their GTFS route but `stop_cursor` pointed at the stop at 0.0 m, so the first tick "arrived" there and teleported the bus back; cursor now starts at the first stop *ahead*, and stops far behind (wrapped cursor near route end) are skipped instead of teleporting | `bus_mixin.py` |
| **Bug fix: second GTFS arity crash** — the scipy-missing path returned `[]` instead of `([], [])` (same class as the networkx one fixed earlier) | `bus_mixin.py` |
| **Buses/bus stops no longer float when terrain is off** — route z is DEM-baked at load; bus positions now zero z unless draping is active, shelters are placed flat and lifted/restored by the terrain toggle (position-based, like parked cars) | `bus_mixin.py`, `terrain_mixin.py` |
| **MAGENTA MYSTERY SOLVED: VTK's built-in '3' key toggles anaglyph stereo** — pressing `3` (roundabout mode) fired VTK's default CharEvent handler which enables red/blue stereo rendering: the whole screen goes wobbly magenta and stays that way across mode changes. It was never the picker or SSAO. ALL of VTK's default char bindings are now removed at startup (`RemoveObservers("CharEvent")`) — this also stops `w` secretly toggling wireframe, `s` surface, `f` fly-to, and `e` exiting the app | `car_mixin.py` |
| **Bug fix: AQ overlay stuck at zero** — the colour scale started at 0.01 g/s and could only ratchet UP; with few vehicles (e.g. SUMO mode) per-cell emissions (~0.001 g/s) never crossed the 4 % display threshold, so the overlay stayed invisible forever. The scale now adapts downward too (instant jump on regime change, 10 %/update decay otherwise) | `heatmap_mixin.py` |
| **`b` key: bus routes + stops overlay** — coloured route polylines (one colour per GTFS route, tube-rendered) and the low-poly shelter OBJs (verified loading: 4 591-pt model) are built hidden and toggle together on `b` | `bus_mixin.py` |
| **Terrain surface lowered 0.5 m below roads** (was 8 cm) — the 130×130 terrain grid interpolates the DEM more coarsely than road vertices, bulging ~0.3 m above them on slopes; roads kept sinking into the hillside | `terrain_mixin.py` |
| **SUMO native emissions → AQ overlay** — in SUMO mode the `q` overlay now uses SUMO's own HBEFA4 per-vehicle emission model (`getCO2/NOx/PMxEmission`, accounts for acceleration — idling in queue vs cruising differ) instead of the speed-only EEA approximation; EEA remains the fallback | `sumo_bridge.py`, `heatmap_mixin.py` |
| **`u` key: live congestion overlay (SUMO)** — per-edge mean speed vs free-flow, colored green/amber/red tubes over the road network, rebuilt at 1 Hz from SUMO's own edge geometry (works with any net, terrain-aware) | `sumo_bridge.py`, `sumo_mixin.py` |
| **City-health KPI HUD (SUMO)** — third line of the SUMO overlay: mean delay vs free-flow (`getTimeLoss`), mean standstill wait, total arrived, live collision count | `sumo_bridge.py`, `sumo_mixin.py` |
| **`i` key: breakdown incident (SUMO)** — forces a random vehicle to a standstill for 60 s (red cone marker), jam propagates realistically, auto-released after the timer; combine with `u` to watch congestion spread | `sumo_bridge.py`, `sumo_mixin.py` |
| **CRITICAL bug fix: editor-rebuilt SUMO nets were 4 cm wide** — the plain-XML exporter writes lon/lat but netconvert was never told the input is geodetic (`projParameter="!"`, degrees treated as metres); every editor network rebuild produced a microscopic net and `hasGeoProj()` failed. `--proj.utm` added to netconvert flags | `sumo_network_patch.py` |
| **Bug fix: mid-block editor lights were green ~97 % of the cycle** — a light on a straight road (≤2 approaches, one axis) got `n_phases=1`: green for the full 45 s cycle with only the 1.5 s all-red pause as "red". Such lights are now modeled as pedestrian crossings — two phases alternating cars-green (45 s) with all-red (14 s + yellow), so they genuinely stop traffic | `traffic_lights.py` |
| **SUMO's real signal states rendered** — in SUMO mode the FSM bulbs showed colors SUMO's cars don't obey; now one bulb per controlled link (positioned at its incoming lane end, cached) colored from `getRedYellowGreenState`, refreshed 1 Hz; FSM bulbs auto-hidden while SUMO drives, restored on close | `sumo_bridge.py` (`tl_link_states`), `sumo_mixin.py` |
| **Click-routing ETA** — retargeting a SUMO vehicle now prints the router's estimated travel time (`simulation.findRoute`, no vehicle spawn needed) | `sumo_bridge.py` (`find_route_time`), `sumo_mixin.py` |
| **Pedestrian stages** — SUMO persons now carry their stage (waiting/walking/riding); the HUD shows "N ped waiting" when pedestrians are queued at stops | `sumo_bridge.py`, `sumo_mixin.py` |
| **First-person pedestrian mode upgraded** — click any pedestrian (IDM or SUMO, 3.5 m radius) to enter first-person follow of *that* person; press `f` to detach into free-walk with **WASD movement + mouse look** (trackball suspended, yaw/pitch from mouse deltas, pitch clamped ±70°, eye at 1.65 m, terrain-snapped); WASD falls back to its global actions (weather/pan) outside free-walk | `walk_mixin.py`, `ui_mixin.py`, `main_ast6.py` |
| **Bug fix: weather advanced 2 steps per press** — `w` was registered by both `main_ast6` and `weather_mixin` (pyvista stacks callbacks), so one press skipped from clear straight to snow; the lazy registration is now suppressed | `main_ast6.py` |
| **Bug fix: `f` walk mode showed only sky** — teleporting the camera from overview to street level left VTK's near/far clipping planes stale, clipping the entire city out of view; `ResetCameraClippingRange()` now runs on every walk-camera update and on restore | `walk_mixin.py` |
| **`o` key: editor undo** — every editor operation (road reversal, roundabout, traffic light, stop sign, streetlight add/remove) pushes a full graph snapshot first (up to 10 levels); `o` restores the snapshot, removes op-created actors, and rebuilds traffic + the SUMO network | `ui_mixin.py`, `main_ast6.py` |
| **Bug fix: dynamic sun/moon lights never applied** — `pv.Light` has no `.ambient` attribute (the correct properties are `ambient_color`/`diffuse_color`/`specular_color`); the assignment raised on every render pass, so the time-of-day scene lighting silently fell back to defaults since the feature was written | `ui_mixin.py` |
| **Bug fix: pick crash while in walk mode** — mouse-look used `renderer.SetInteractive(0)`, which broke pyvista's poked-renderer lookup (`RuntimeError: Poked renderer not found`); now swaps to a null interactor style instead (trackball suspended, picking intact), restored on exit | `walk_mixin.py` |
| **Bug fix: nested roundabouts** — clicking near an existing ring node built roundabouts ON ring nodes (`ra_ra_ra_…`); such clicks are now refused with a hint to undo | `ui_mixin.py` |
| **Solar fleet benefit study** — `solar_fleet_benefit_study` test sweeps a full day comparing fleet net energy solar vs conventional on a mixed-shade network: noon saving ≈4.5 %, daily average ≈1.8 %, exactly 0 at night, 0 harvest on fully-shadowed streets, diurnal bell shape asserted. The quantitative answer to "are solar cars worth it" | `tests/run_tests.py` |
| **FATAL crash fix: numba workqueue abort** — numba's default threading layer is not threadsafe; the background shadow worker and a main-thread edge-shadow recompute (editor click during time-lapse) entering `parallel=True` kernels concurrently aborted the whole process. `NUMBA_KERNEL_LOCK` (RLock) now serializes every parallel-kernel call site; the main-thread edge cache uses a non-blocking acquire and falls back to the nearest cached hour so the UI never freezes. Stress-tested with 4 concurrent threads | `shadow_engine.py`, `shadow_mixin.py` |
| **Weather slowdown strengthened + applied to SUMO** — rain 60 % / snow 30 % of normal speed (was 75 %/55 %, too subtle); SUMO vehicles previously **ignored weather entirely** — now `setSpeedFactor` is pushed to all vehicles at 1 Hz and reset on clear | `weather_mixin.py` |
| **Night lights visible again** — side effect of the `pv.Light` fix: with real scene lights finally applying, the night ground (semantic lit/unlit colors) was scene-lit at ~0.1 intensity → near-black, hiding the streetlight pools. Ground actors are now `lighting=False` (they're data visualization) and the moon light is brighter (0.45, warmer ambient) so buildings stay readable at night | `shadow_mixin.py`, `ui_mixin.py` |
| **BUILDING EDITOR (`g` mode)** — click anywhere to place a building: deterministic pseudo-random footprint/height per click position (12–26 m × 12–42 m tall), styled to the active palette, terrain-aware. The box merges into `buildings_mesh`, the shadow **octree rebuilds**, and shadow + solar-cost caches clear — so the new building **immediately casts shadows and cuts solar harvest** on adjacent streets (demo this against the solar study!). Undo with `o` restores the exact prior mesh + octree. Key `7` was taken by scenario-save-A, hence `g` | `editor_ops.py`, `ui_mixin.py`, `main_ast6.py`, `terrain_mixin.py` |
| **`--n-parked-cars` CLI flag added** — the code read `args.n_parked_cars` but the flag never existed in the CLI | `app_cli.py` |
| **Bug fix: GTFS loader arity** — `_load_gtfs_buses` returned `[]` instead of `([], [])` when networkx missing → unpacking crash | `bus_mixin.py` |
| **Bug fix: parking-lot cars not terrain-draped** — terrain toggle read `_parked_car_actors` (ultra-mode list) but parking sim stores its glyph pool under `_parked_cars_actor`; the glyph pool is now lifted/restored via `_drape_actor_points`, and the init-time DEM pre-lift only applies when draping is already active | `terrain_mixin.py`, `parking_mixin.py` |
| **Bug fix: MS Buildings height lookup lon/lat swap** — `transformer.transform(centroid.y, centroid.x)` passed (lat, lon) to an `always_xy=True` transformer, so the MS height override silently never matched; now (x, y) | `overture_source.py` |
| **Scenario comparison** — save scenario A (`7`) / B (`8`), `9` runs both headless in SUMO (15 min sim, identical seeds+demand), `0` toggles a 3D diff overlay coloring edges by travel-time delta (blue=better, red=worse) + summary panel (Δ mean travel time, Δ CO₂/NOx/PM, Δ noise exceedance); writes `comparison_<timestamp>.json` with explicit metric definitions | `scenario_compare.py` (new), `scenario_mixin.py` (new), `main_ast6.py` |
| **Git pre-commit hook** — `.githooks/pre-commit` runs the full headless test suite before every commit (installed via `git config core.hooksPath .githooks`, already configured; bypass with `--no-verify`) | `.githooks/pre-commit` (new) |
| **Sea-background plane** — base ground restored to dark slate `#3a3f46`; a dedicated sea plane at Z=−0.05 (3× scene extent) is added only when the scene contains water (water mesh or water-coloured fill cells), closing the open-sea gap without making inland gaps look flooded | `main_ast6.py` |
| **Roundabout → SUMO verified** — regression test builds an editor-style roundabout (string `ra_*` node IDs), exports plain-XML, and runs real netconvert (EclipseSUMO 1.27.1); accepted | `tests/run_tests.py` |
| **Emergency key moved `x` → `m`** — `x` fired both zoom-out and ambulance spawn (PyVista stacks callbacks per key) | `emergency_mixin.py` |
| **SUMO car-click routing** — in `--engine sumo` mode, clicking a SUMO vehicle selects it (green marker) and the next click retargets it via TraCI `changeTarget`; the remaining route renders as a cyan polyline | `sumo_bridge.py`, `sumo_mixin.py`, `ui_mixin.py` |

### Known remaining issues
- Tree data: OSM tagged trees often absent from old caches; synthetic fallback places ≤500 trees.
- ~~Lane changing is dormant~~ — resolved: `adj_left_paths`/`adj_right_paths` wired through `car_mixin` → `idm_tick` → `find_leaders` Phase 0b (2026-07-05).
- Scenario comparison requires SUMO (netconvert + randomTrips + traci); degrades with a clear message when missing.
- `fill_ver` is now **3**. Any cache built with fill_ver < 3 will be re-fetched automatically on next run.
- ~~Cyclist/human GLB T-pose~~ — resolved: models re-posed in Blender (2026-07-04).
- `streetlight_ga.py` creates a fresh ThreadPoolExecutor per fitness-evaluation call — works, but wastes pool setup/teardown; could be created once in `optimize()`.
- `demand_model._pick_weighted()` is dead code (only `_pick_weighted_index` is used).
- ~~`gtfs_realtime.py` silent failures~~ — resolved: parse errors, throttled consecutive-failure warnings, and stale-feed warning added (2026-07-05).

### Key bindings (current, complete)
**Weather / Terrain / AQ overlay have NO keyboard shortcuts — use their panel
checkboxes.** (`q` is dangerous: pyvista hard-binds it to close-window — that
binding is cleared at startup, and the key is left unbound. `w`/`g` were freed
for WASD walking and future use.)

| Key | Action |
|---|---|
| `t` | Play/pause day/night time-lapse |
| `h` | Toggle traffic density heatmap |
| `n` | Toggle noise-pollution map |
| `v` | Toggle network centrality heatmap |
| `c` | Reset camera to overview |
| `f` | Cycle walk modes: off → 3rd-person follow → 1st-person POV → free walk → off |
| Click a pedestrian | **Take direct control** of that pedestrian (detached from AI): arrows walk/turn, mouse looks, `f` exits |
| `b` | Toggle bus routes + stop shelters overlay |
| `u` | Toggle live congestion overlay (SUMO mode) |
| `i` | Trigger breakdown incident (SUMO mode, 60 s) |
| `m` | Spawn emergency vehicle (max 3; press again when full to clear all) |
| `o` | Undo last editor operation (up to 10 levels) |
| `w`/`a`/`s`/`d` | Free-walk movement (outside free-walk: `a`/`d`/`s` pan camera, `w` does nothing) |
| `e` / `r` | Rotate camera |
| `z` / `x` | Zoom in / out |
| Arrow keys | Pan camera (or walk forward/back + turn in free-walk mode) |
| Escape | Exit walk mode / clear road info overlay |
| `1`–`6` | Editor modes (view / roads / roundabouts / lights / stops / streetlights) |
| `g` | Buildings editor — click to place a building (casts shadows, cuts solar harvest; `o` undoes) |
| `y` | Highway editor — click two nodes to span an elevated red motorway bridge between them (auto-clears buildings under the span, support pillars, cars drive over it; `o` undoes) |
| `7` / `8` | Save scenario A / B (network + signals + demand → `scenarios/A|B/`) |
| `9` | Run headless A-vs-B comparison in SUMO (identical seeds) |
| `0` | Toggle travel-time diff overlay + summary panel |

Free keys for future features: `j`, `k`, `l` (and `q`, kept unbound on purpose).

### Walk / follow camera (`WalkMixin`)
- **Click any pedestrian** (within 3.5 m) → 1st-person follow of that person.
- `f` cycles: off → 3rd-person follow (nearest ped) → 1st-person → free walk → off
- **Free walk**: WASD to move (terrain-snapped, eye 1.65 m), **mouse to look**
  (trackball suspended, pitch ±70°), arrow keys as fallback.

Escape always exits walk mode.

### Hidden / power-user features (easy to forget these exist)

**CLI flags that unlock whole modes:**
- `--solo` — a single solar car with **live energy telemetry** (mechanical vs
  harvested Wh on screen). Great one-slide demo for the solar thesis.
- `--profile --profile-duration 60 [--profile-output report.json]` — full
  performance profiler: per-frame breakdown of IDM physics / VTK actors /
  render / overlays, exported to JSON (that's where `profile_report.json`
  came from).
- `--gtfs-rt-url` / `--gtfs-rt-key` / `--gtfs-rt-key-param` /
  `--gtfs-rt-interval` — **live real-world bus positions** from any
  GTFS-Realtime feed, polled in a background thread. If the city's transit
  agency publishes a feed, actual buses drive in the twin.
- `--car-detail low`, `--no-dem`, `--no-ms-buildings` — fast-startup switches
  for demos on weaker hardware.
- The validation trio: `--validate-od N` (free-flow vs OSRM),
  `--validate-congested N` (live congested times vs OSRM),
  `--validate-engines N` (IDM vs SUMO) — the scientific-rigor story.
- `--n-parked-cars N` — override the auto parked-car count.

**Ambient features running silently:**
- **Parking-lot life cycle**: parked cars vacate after ~30 s and return after
  45–90 s, simulating turnover without explicit agents.
- `p` (with `--debug-cars`) prints live car diagnostics in the console.
- `tools/build_sumo_scene.py` works **standalone** — pre-build SUMO scenes
  offline before a presentation instead of waiting for the auto-build.

### Scene Z-layer ordering
| Z | Layer |
|---|---|
| −0.10 | Base ground plane (`#3a3f46` dark slate) |
| 0.02 | Land-use fill mesh (water, parks, residential, etc.) |
| 0.10 | Road surface |
| 0.15 | Sidewalk surface |
| 0.20+ | Building bases |

### IDM — deadlock recovery (Step 8 of `idm_tick`)
Cars stuck at < 0.12 m/s with a gap < 25 m for > 8 continuous seconds are teleported to a random
new edge at 30 % of the new road's speed limit. `stuck_time` array lives in `car_anim`.

---

A 3D interactive digital twin of any city, built on **PyVista / VTK**. Load any address, explore a real-time traffic simulation with physically-based shadow casting, solar energy analysis, genetic-algorithm streetlight optimisation, and live analysis overlays (traffic heatmap, noise pollution, road centrality).

Traffic demand is **emergent**, not scripted: a **gravity O-D model** sends cars from residential to commercial zones in the morning peak and back in the evening, with distance-decayed destination choice and trip chaining (arrived cars get a fresh onward trip).

Also includes: **weather** (rain/snow with wet-road shading), a **day/night time-lapse**, **3D terrain** with elevation contours, a **parking simulation**, a live **air-quality heatmap** driven by a real speed-dependent **emissions model** (CO₂/NOx/PM with Gaussian dispersion), multi-modal agents (pedestrians, cyclists, buses), **GTFS-Realtime** live bus positions, **SUMO co-simulation** (let the validated SUMO engine drive the vehicles while this app renders them), and **travel-time validation** against an external routing engine (OSRM) so accuracy can be measured, not just asserted.

---

## Testing (headless — safe to run anywhere)

```
python tests/run_tests.py          # all 26 tests
python tests/run_tests.py -k idm   # only IDM tests
python tests/run_tests.py -v       # verbose
```

A git **pre-commit hook** (`.githooks/pre-commit`, activated via `git config core.hooksPath
.githooks` — already configured in this repo) runs the suite before every commit and blocks
on failure. Bypass once with `git commit --no-verify`.

Pure-logic tests with synthetic data, no VTK window, no network calls (OSRM is stubbed).
Covers: IDM physics (movement, collision-free following, red-light stopping, deadlock
teleport), per-car heterogeneity (variance + seed reproducibility), traffic-light FSM
(state cycling, phase exclusivity, editor-created light shape), emissions (U-shape,
class scaling), validation metrics (MAPE/GEH known values, per-road-class grouping,
congested-mode end-to-end on a synthetic grid), gravity demand distance decay, terrain
profile lookup, SUMO plain-XML export, roundabout export with string `ra_*` node IDs
through real netconvert (skips gracefully if SUMO absent), scenario snapshot round-trip,
scenario compare math + JSON report, edge-id mapping stability, GTFS loader return-arity
regression, solar physics sanity, and an import smoke test over all engine modules.
Exit code 0 = green.
Add new tests by decorating a function with `@test` in `tests/run_tests.py`.

---

## Running

```
python main_ast6.py [options]
```

**Common flags**

| Flag | Default | Description |
|---|---|---|
| `--address "Place Name"` | required | OSM address to load (e.g. `"Syntagma Square, Athens"`) |
| `--radius 300` | 300 | Scene radius in metres |
| `--mode view` | `view` | Startup editor mode |
| `--traffic-speed 1.0` | 1.0 | Car speed multiplier (0 = freeze) |
| `--n-cars 40` | auto | Active cars (-1 = auto from building density) |
| `--car-detail ultra` | `standard` | OBJ car models with per-part PBR |
| `--n-lights 60` | 60 | Target number of streetlights |
| `--light-strategy smart` | `smart` | Streetlight placement (`smart`/`random`) |
| `--seed 42` | 42 | RNG seed |
| `--n-peds -1` | auto | Pedestrian agents (-1 = auto, 50–200) |
| `--n-cyclists -1` | auto | Cyclist agents (-1 = auto, 10–60) |
| `--solo` | off | Single solar car with live energy telemetry overlay |
| `--no-cache` | off | Bypass geometry cache |
| `--gui` | off | Show Tkinter settings GUI before launch |

**Transit & co-simulation flags**

| Flag | Default | Description |
|---|---|---|
| `--gtfs PATH` | — | GTFS directory or `.zip` for simulated buses |
| `--n-buses 30` | auto | Max bus agents from GTFS |
| `--gtfs-rt-url URL` | — | GTFS-Realtime VehiclePositions feed → **live** bus positions |
| `--gtfs-rt-key KEY` | — | API key for the RT feed (header auth) |
| `--gtfs-rt-key-param NAME` | — | Send the key as this query parameter instead of a header |
| `--gtfs-rt-interval 15` | 15 | Seconds between RT feed polls |
| `--sumo-cfg PATH` | — | `.sumocfg` → enable **SUMO co-simulation** |
| `--sumo-net PATH` | — | Explicit geo-referenced `.net.xml` (else read from cfg) |
| `--sumo-binary sumo` | `sumo` | SUMO binary name/path |
| `--sumo-gui` | off | Also launch `sumo-gui` alongside |
| `--sumo-step 0.1` | 0.1 | SUMO step length (seconds) |
| `--sumo-port` | auto | Explicit TraCI port |
| `--engine {idm,sumo}` | `idm` | Primary traffic engine. `sumo` promotes SUMO to the main source: overlays (heatmap, noise, AQ) read SUMO vehicle data; walk camera follows SUMO agents; if `--sumo-cfg` is omitted the scene is auto-built from `--address/--radius` and cached. Degrades to `idm` with a warning if SUMO deps are missing. |

> GTFS-RT needs `pip install gtfs-realtime-bindings`. SUMO co-sim needs `pip install eclipse-sumo traci sumolib` (the `eclipse-sumo` wheel bundles the binaries and is auto-detected, so `$SUMO_HOME` need not be set). Build a matching SUMO scene with `python tools/build_sumo_scene.py --lat .. --lon .. --radius ..`, then run with `--n-cars 0 --sumo-cfg <scene>.sumocfg`. Or use `--engine sumo` to have the app auto-build and cache the scene. Both layers degrade gracefully to the built-in IDM simulation when their dependencies or feeds are absent.

**Validation flags** (Tier-1 accuracy)

| Flag | Default | Description |
|---|---|---|
| `--validate-od N` | 0 | Compare model free-flow route times vs a routing engine over N random O-D pairs, then write `validation_<engine>_Npairs.json`. Works with `--no-view`. |
| `--validate-engine osrm` | `osrm` | Reference engine (OSRM public demo server, no API key). |
| `--validate-osrm-host URL` | OSRM demo | Point at your own OSRM instance for heavy use. |
| `--validate-congested N` | 0 | Compare simulated congested travel times (active demand) vs OSRM over N O-D pairs, report MAPE/RMSE/GEH per road class. |
| `--validate-engines N` | 0 | Compare IDM vs SUMO travel times over N shared O-D pairs; saves `engine_comparison_<ts>.json`. |

> Example: `python main_ast6.py --lat 37.9838 --lon 23.7275 --radius 800 --validate-od 50 --no-view` prints MAPE / RMSE / bias / Pearson r / GEH and saves a per-pair JSON report.

---

## Interactive Controls

A key legend prints to the console at launch and sits in the bottom-right of the viewer. The left panel has layer toggles, style presets, and sliders.

| Key | Action |
|---|---|
| `w` | Cycle weather: clear → rain → snow |
| `t` | Play/pause day/night time-lapse |
| `q` | Cycle air-quality overlay: NOx → CO₂ → PM2.5 → off (real emissions + dispersion) |
| `g` | Toggle 3D terrain surface + contours |
| `h` | Toggle accumulated traffic heatmap |
| `n` | Toggle noise-pollution map |
| `c` | Reset camera view |
| `a` / `d` | Pan left / right |
| `z` / `x` | Zoom in / out |
| `e` / `r` | Rotate camera |
| `f` | Walk mode: off → 3rd-person → 1st-person → free walk → off |
| `m` | Spawn emergency vehicle (max 3; again to clear) |
| arrows | Pan camera |
| `1`–`6` | Editor modes (view / roads / roundabouts / lights / stops / streetlights) |
| `g` | Buildings editor — click to place a building (casts shadows, cuts solar harvest; `o` undoes) |
| `7`/`8`/`9`/`0` | Scenario compare: save A / save B / run / toggle diff |

---

## File Map

### Entry points
| File | Role |
|---|---|
| `main_ast6.py` | **Active file.** `DigitalTwinApp` OOP class wiring all mixins together. |
| `main.py` | Original monolith — reference only, not actively developed. |
| `app_cli.py` | CLI argument parsing + optional Tkinter config GUI (runs in subprocess). |

### Mixin classes
| File | Role |
|---|---|
| `car_mixin.py` | IDM car animation, parked cars, OBJ ultra mode, route tracking UI. |
| `demand_mixin.py` | Assigns gravity O-D trips to cars + re-trips arrivals (trip chaining). |
| `ped_mixin.py` | Pedestrian agents + shared GLB loader (arms auto-folded from T-pose). |
| `cyclist_mixin.py` | Cyclist agents on bike-friendly edges. |
| `bus_mixin.py` | GTFS buses, bus-stop shelters, transit route polylines, live ETA labels, GTFS-Realtime layer. |
| `emergency_mixin.py` | Emergency vehicle with right-of-way yielding. |
| `weather_mixin.py` | Rain/snow particles, wet-road shader, roof snow, weather-slowed traffic (`w`). |
| `tod_mixin.py` | Day/night time-lapse: sun arc, window glow, ambient + shadow updates (`t`). |
| `parking_mixin.py` | Parked cars in OSM lots, scheduled leave/enter wired to the IDM system. |
| `terrain_mixin.py` | DEM terrain surface + elevation contours, fill-mesh lifting (`g`). |
| `heatmap_mixin.py` | Live air-quality / noise overlay plane from car + bus density (`q`). |
| `sumo_mixin.py` | SUMO co-simulation render layer (vehicles driven by TraCI). |
| `shadow_mixin.py` | Background shadow pipeline, edge-shadow cache, ground mesh coloring. |
| `route_mixin.py` | Route planning, editor rebuild (`_rebuild_traffic_and_arrows`). |
| `ui_mixin.py` | All slider/checkbox/picker callbacks, SSAO, streetlight editor. |
| `analysis_mixin.py` | Traffic heatmap, noise pollution map, network centrality overlays. |
| `scenario_mixin.py` | Scenario compare UI: save A/B, run headless comparison, diff overlay + summary panel (`7`/`8`/`9`/`0`). |

### Engine modules
| File | Role |
|---|---|
| `app_core.py` | OSM fetch & cache, road geometry, HDRI/PBR environment setup. |
| `idm.py` | Intelligent Driver Model physics + turn-cornering + roundabout yield. |
| `traffic_lights.py` | Two-phase FSM traffic lights with road-class-aware green/yellow timing, glyph rendering, per-path approach points. |
| `shadow_engine.py` | Numba JIT ray-casting for shadow masks and spotlight coverage matrices. |
| `streetlight_ga.py` | Genetic algorithm for maximum-coverage streetlight placement. |
| `solar_physics.py` | Sun angles, clear-sky irradiance, panel power model. |
| `solar_routing.py` | Energy-optimal and Pareto-optimal route finding. |
| `turn_restrictions.py` | OSM turn restriction enforcement for car routing. |
| `overture_source.py` | Overture Maps building footprint + POI fetch. |
| `osm_3d_buildings.py` | OSM building extrusion to 3D mesh, land-use fill, water, DEM sampler. |
| `spatial_trees.py` | Octree for building geometry (shadow engine acceleration). |
| `gtfs_realtime.py` | GTFS-Realtime feed fetch + protobuf parse + background poller. |
| `sumo_bridge.py` | TraCI connection, stepping, vehicle extraction, XY→lon/lat conversion. |
| `sumo_engine.py` | Auto-build and cache a SUMO scene from lat/lon/radius when `--engine sumo` is used without an explicit `--sumo-cfg`. |
| `emissions.py` | Speed-dependent CO₂/NOx/PM emission factors (EEA average-speed shape, fuel-based CO₂). |
| `validation.py` | O-D travel-time validation vs OSRM with MAPE/RMSE/bias/GEH metrics. |
| `demand_model.py` | Gravity O-D demand: time-of-day direction, distance decay, cluster-weighted zones. |
| `scenario_compare.py` | Scenario snapshot/serialize, SUMO scene build per slot, headless traci run (edge travel times + emissions + noise), A-vs-B compare + JSON report. |
| `tools/build_sumo_scene.py` | Generate a geo-referenced SUMO scene (netconvert + randomTrips) for an `--lat/--lon/--radius`. |

---

## Architecture

```
DigitalTwinApp(CarMixin, PedMixin, CyclistMixin, BusMixin, EmergencyMixin,
               WeatherMixin, TODMixin, ParkingMixin, TerrainMixin, HeatmapMixin,
               SumoMixin, ShadowMixin, RouteMixin, UIMixin, AnalysisMixin)
  │
  ├─ __init__()
  │     _load_or_fetch_osm_cached   ← buildings, street graph, POIs (cached by radius+address)
  │     _warmup_numba_kernels
  │     _extract_drivable_paths     ← per-lane paths, maxspeed from OSM or highway-type fallback
  │     build_traffic_lights        ← two-phase FSM, staggered offsets, degree≥3 fallback
  │     _init_analysis              ← allocates 4 m analysis grid over scene bounds
  │
  └─ run()
        pv.Plotter created + actors added
        plotter.show(auto_close=False, interactive_update=True)   ← macOS non-blocking
        │
        └─ Manual macOS event loop (ProcessEvents + wall-clock)
               _animate_cars()          every ~16 ms  — IDM tick, TL tick, analysis overlay
               _poll_shadow_job()       every 250 ms  — apply bg shadow result to ground mesh
               _deferred_initial_render at 500 ms
               _deferred_ssao_enable    at 800 ms     — render() flush first (prevents dark frame)
```

---

## Traffic Simulation

### Intelligent Driver Model (IDM)
Cars follow `idm.py`'s symplectic Euler pipeline each tick:

1. **Snapshot** — read-only copy of `(edge_idx, dist, speed, desired_speed)`
2. **Turn pre-braking** — cars within 18 m of their edge end have `desired_speed` lowered to the cornering speed limit (30/20/14/10 km/h for 20°/45°/90°/135° turns). `desired_speed_base` is untouched so cars re-accelerate normally on the new edge.
3. **Leader finding** — three-priority hierarchy:
   - Same-edge leader (car directly ahead)
   - Cross-edge leader (frontmost car on next path)
   - Ghost leader injected at stop line for red/yellow traffic lights
   - Roundabout yield ghost when entering a ring edge
4. **IDM formula** — vectorised NumPy, no Python loop
5. **Position advance** — speed clamp at edge crossing enforces cornering limit as a hard cap
6. **Write-back** — results written to `car_anim` arrays

### Rush-hour traffic curve
`_rush_hour_multiplier(hour)` multiplies `traffic_speed` each tick:

| Time | Multiplier |
|---|---|
| 03:00 (quietest) | 0.60× |
| Midday | ~0.70× |
| 08:00 (morning peak) | 1.40× |
| 18:00 (evening peak) | 1.40× |

Gaussian peaks at 08 h and 18 h, floor of 0.60×.

### Speed limits
OSM `maxspeed` tag is used when present. When absent, inferred from `highway` type:

| Type | Speed |
|---|---|
| `motorway` | 120 km/h |
| `trunk` | 80 km/h |
| `primary` | 50 km/h |
| `secondary` | 40 km/h |
| `residential` / `tertiary` | 30 km/h |
| `service` / `living_street` | 20 km/h |

### Traffic lights
Two-phase FSM (`TrafficLight` dataclass in `traffic_lights.py`).

- Edges grouped into two phases by bearing axis
- Phase 0 green: **35 s**, phase 1 green: **15 s**, yellow: **4 s**, all-red: **1.5 s**
- Offset initialised from `rng.uniform(0, full_cycle)` and drained through transitions at startup — lights genuinely start scattered across the cycle rather than all green
- Rebuilt fresh whenever roads are edited (road flip, roundabout insert)
- Ghost-leader braking: IDM decelerates smoothly 18 m before a red stop line

### Agent interaction behaviours (previously undocumented)
These run every physics tick in `car_mixin._advance_cars` / `idm.find_leaders`:

| Behaviour | Rule |
|---|---|
| **Pedestrian crosswalk yield** | Cars within 12 m of an active crossing (a pedestrian currently crossing, published via `ped_anim["active_crossings"]`) are speed-capped to 1.4 m/s for that tick |
| **Emergency clearance** | Cars within 30 m of an active emergency vehicle slow to ≤0.5 m/s until it passes |
| **Junction conflict yield** | Cars from different edges converging on the same node within 20 m: only the closest proceeds, the rest get a ghost leader at the junction |
| **Stop signs** | Cars on a `control="stop"` edge brake to the line, wait 2 s at standstill, then proceed |
| **Roundabout yield** | Cars entering a ring edge get a yield ghost when ring traffic is within 15 m of the conflict point |
| **Weather slowdown** | Rain ×0.75, snow ×0.55 on traffic speed (restored on clear) |
| **Rush-hour curve** | Gaussian peaks at 08:00/18:00 (1.4×), floor 0.6× at ~03:00 |

### Parked cars (ultra mode)
Static OBJ cars placed at road-side offsets near residential POIs. Skips any placement within **20 m** of an intersection node.

### Route planning
Cars are assigned home→work routes via `nx.shortest_path`. Starting distance is `rng.uniform(0, path_length × 0.3)` so cars spawn scattered rather than all at edge position 0.

---

## Scenario Comparison (planning tool)

A scenario = current network edits + signal timings + demand parameters, serialized to JSON.

**Workflow:** edit the network with modes 2–5 → press `7` (save scenario A) → edit again →
`8` (save scenario B) → `9` (run comparison) → `0` (toggle diff overlay).

Each save builds a standalone SUMO scene in `scenarios/A|B/` (plain-XML export →
netconvert → randomTrips with the app's `--seed` → minimal `.sumocfg`). The comparison
runs both scenes **headless** for 15 sim-minutes with identical seeds and demand, collecting:

- **Per-edge mean travel time** — time-mean of SUMO's instantaneous per-edge estimate
- **Total CO₂ / NOx / PM** — `emissions.py` EEA average-speed model integrated over sampled vehicle speeds
- **Noise exceedance** — cell-seconds above 65 dB(A) on a 25 m grid (simplified CNOSSOS)

Results render as (1) a diff overlay coloring each edge by travel-time delta (diverging
colormap: blue = B better, red = B worse, clamped at the 95th-percentile |Δ|), and
(2) a summary panel with network-wide deltas. A `comparison_<timestamp>.json` is written
with explicit metric definitions. Everything runs in background threads; the UI stays live.

Files: `scenario_compare.py` (snapshot/build/run/compare), `scenario_mixin.py` (UI + diff render).

---

## Analysis Overlays

Three read-only overlays, all toggled by keyboard:

### `H` — Traffic density heatmap
Accumulates car XY positions into a rolling 2D histogram over the last 180 animation ticks (~3 s). Gaussian-blurred (σ = 10 m), normalised to [0, 1], written into a flat 4 m/cell grid mesh above the ground. Color ramp: `hot` (black → dark red → orange → white).

### `N` — Noise pollution map
For every 4 m grid cell, sums the incoherent dB contribution of every car using a simplified CNOSSOS-EU model:

```
L_emission = L_ref + 10·log₁₀(v / 50 km·h⁻¹)   per vehicle
L_at_cell  = L_emission − 20·log₁₀(d)            spherical spreading
L_total    = 10·log₁₀(Σ 10^(L_at_cell/10))       energy sum
```

Colour ramp: `RdYlGn_r` (green = quiet, red = loud), 35–80 dB. Live status bar counts cells breaching the 65 dB(A) WHO day-time limit.

### `V` — Network centrality
Runs `networkx.edge_betweenness_centrality(weight="length")` once, caches the result, and renders each road segment coloured by normalised centrality. Colour ramp: `plasma` (dark purple = quiet side street → bright yellow = backbone artery). Shows which streets carry the most structural traffic load and where jams propagate furthest.

*H and N share the same grid mesh — enabling one disables the other. V is independent.*

---

## Keyboard Reference

### Editor modes
| Key | Mode | Click action |
|---|---|---|
| `1` | View | Road info + route planning (in `--engine sumo` mode, clicks route SUMO vehicles: first click selects a nearby vehicle, second click retargets it) |
| `2` | Roads | Reverse road direction |
| `3` | Roundabouts | Insert roundabout at nearest node |
| `4` | Lights | Toggle traffic light at nearest node |
| `5` | Stops | Toggle stop sign at nearest node |
| `6` | Streetlights | Add / remove streetlight at click point |

### Analysis overlays
| Key | Action |
|---|---|
| `H` | Toggle traffic density heatmap |
| `N` | Toggle noise pollution map |
| `V` | Toggle network centrality |

### VTK built-ins
| Key | Action |
|---|---|
| `R` | Reset camera |
| `S` | Surface shading |
| `W` | Wireframe |
| `P` | Point picker |
| `Escape` | Clear road info overlay |

---

## Shadow Pipeline

```
_on_time_change(hour)
  → _request_shadow_render(hour, spot_radius)
      → _apply_visual_updates(...)              fast — background colour, lights, sun sphere
      → _schedule_shadow_job(...)               submit to ThreadPoolExecutor
           │  pre-extracts numpy arrays on main thread (VTK not thread-safe)
           └─ _shadow_bg_worker(...)            background thread, numpy + Numba only
                → Numba ray-cast or spotlight coverage
                → returns (mask_or_coverage, lit_ratio)
  ← _poll_shadow_job()  every 250 ms in event loop
      → _apply_shadow_result_to_ground(...)     main thread — rewrites ground mesh scalars
```

Day mode writes `day_surface_class` (4 classes: road shadow/lit, sidewalk shadow/lit).  
Night mode writes `night_surface_class` (6 classes: coverage 0/1/2 × road/sidewalk).

---

## Key `scene_state` Entries

| Key | Type | Description |
|---|---|---|
| `interactive_ready` | bool | Set True at ms=300; gates all callbacks |
| `hour` | float | Scene hour (0–24), drives sun + shadows |
| `is_night` | bool | True when hour outside daylight range |
| `ground_actor` | vtkActor | Ground mesh actor (replaced each shadow update) |
| `building_actor` | vtkActor | First PBR building actor (backward-compat) |
| `_building_actors_pbr` | dict | Per-class PBR actors (`concrete`, `brick`, `glass`, …) |
| `_shadow_future` | Future\|None | Running background shadow job |
| `traffic_lights` | dict | `node_id → TrafficLight` FSM instances |
| `_tl_mesh` | PolyData | Source point cloud for TL glyph colors |
| `_tl_glyphs` | PolyData | Glyph mesh; colors broadcast per tick |
| `_ultra_car_actors` | list | Per-car vtkActor/Assembly in OBJ mode |
| `_parked_car_actors` | list | Static parked car actors |
| `editor_mode` | str | `view`\|`roads`\|`roundabouts`\|`lights`\|`stops`\|`streetlights` |
| `heatmap_on` | bool | Traffic heatmap overlay active |
| `noise_on` | bool | Noise pollution overlay active |
| `centrality_on` | bool | Centrality overlay active |

---

## `car_anim` Dict

All arrays are aligned by car index `[0 … N-1]`:

| Key | Shape | Description |
|---|---|---|
| `enabled` | bool | False if n_cars=0 or no drivable paths |
| `edge_idx` | (N,) int64 | Current drivable path index per car |
| `dist` | (N,) float | Distance along current path (metres) |
| `speed` | (N,) float | Actual speed m/s |
| `desired_speed` | (N,) float | Free-flow target v₀ × effective_traffic_speed |
| `desired_speed_base` | (N,) float | Per-car v₀ before traffic_speed multiplier |
| `accel` | (N,) float | Last IDM acceleration m/s² |
| `model_idx` | (N,) int64 | OBJ template index per car |
| `car_len` | (N,) float | Bumper length for IDM gap calculation |
| `planned_edges` | list[list\|None] | Route plan per car (None = free roam) |
| `planned_cursor` | (N,) int64 | Current position within planned route |
| `stop_wait` | (N,) float | Stop-sign wait timer (s) |
| `pos` | (N, 3) float | XYZ positions, refreshed every tick |

---

## macOS M1 / Apple Silicon Notes

**Non-blocking event loop:**  
`plotter.show(auto_close=False, interactive_update=True)` + manual `ProcessEvents()` pump. `add_timer_event` callbacks don't fire under this mode — all timers are driven by wall-clock comparisons in the loop.

**VTK stereo key conflict:**  
VTK's built-in `'3'` key handler enables anaglyph stereo (turning the scene magenta). The roundabout mode key `'3'` calls `render_window.StereoRenderOff()` immediately after switching mode to cancel it.

**SSAO first-frame flicker:**  
`plotter.render()` is called before `SetUseSSAO(True)` to flush the VTK pipeline.

**OBJ importer:**  
`vtkOBJImporter` creates a render window at import time, crashing Cocoa before `show()`. Replaced by a custom parser `_load_obj_without_render_window()` in `main_ast6.py`.

**Tkinter NSApp conflict:**  
Tkinter acquires `NSApplication` and releases it on exit. If this happens in-process before PyVista, VTK deadlocks. Fixed by running the Tkinter GUI in a subprocess (`app_cli.py`).

---

## Caching

OSM data, road meshes, and spatial caches are stored in `~/.cache/digital_twin/` (or `--cache-dir`). Cache keys include address, radius, extrusion height, road buffer widths, and a schema version string. Changing `--radius` or `--address` automatically fetches fresh data.

Use `--no-cache` to force a full rebuild.
