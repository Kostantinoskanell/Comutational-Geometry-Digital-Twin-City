from __future__ import annotations
import time
import numpy as np
import pyvista as pv
import networkx as nx
from shapely.geometry import Point as _SPoint, LineString as _LS
from solar_routing import nearest_graph_node, find_pareto_routes, build_edge_costs, build_graph_with_costs
from traffic_lights import build_traffic_lights as _build_traffic_lights
from streetlight_ga import optimize_streetlights, build_sidewalk_polygon_from_street_graph, _smart_light_positions
from editor_ops import reverse_edge, insert_roundabout


class UIMixin:
    def _compute_smart_lights(self) -> np.ndarray:
        self._stage("Running smart streetlight placement...")
        return _smart_light_positions(
            ground_mesh=self.ground_mesh,
            street_graph=self.street_graph,
            n_lights=self.args.n_lights,
            grid_step=self.args.grid_step,
            seed=self.args.seed,
        )

    def _set_editor_mode(self, mode: str) -> None:
        self.scene_state["editor_mode"] = mode
        # Cancel any in-progress bridge placement when leaving highway mode
        if mode != "highway":
            _pending_mk = self.scene_state.pop("_bridge_start_actor", None)
            if _pending_mk is not None:
                try:
                    self.plotter.remove_actor("highway_start_marker")
                except Exception:
                    pass
            self.scene_state.pop("_bridge_start", None)
        # key '7' belongs to scenario-save-A; 'g' = buildings, 'y' = highway
        mode_keys = [("1", "view"), ("2", "roads"), ("3", "roundabouts"),
                     ("4", "lights"), ("5", "stops"), ("6", "streetlights"),
                     ("g", "buildings"), ("y", "highway")]
        text = "  |  ".join(f"[{k}] {m.capitalize()}" for k, m in mode_keys)
        self.plotter.add_text(f"Mode: {mode.upper()}   |   {text}", position=(10, 10), name="editor_mode_overlay", font_size=12, color="white")
        print(f"[editor] Switched to mode: {mode}")

    # ------------------------------------------------------------------
    # Editor undo — 'o' key.  Snapshot-based: every editor mutation pushes a
    # full graph copy first (city graphs are small — a copy is milliseconds),
    # so ANY operation (roundabout surgery, tag toggles, reversals) undoes
    # identically: restore the snapshot and rebuild.
    # ------------------------------------------------------------------

    def _editor_push_undo(self, desc: str, snapshot_buildings: bool = False) -> dict:
        stack = self.scene_state.setdefault("_editor_undo", [])
        entry = {
            "desc":   desc,
            "graph":  self.street_graph.copy(),
            "lights": (None if getattr(self, "best_positions", None) is None
                       else np.array(self.best_positions, copy=True)),
            "actors": [],   # actor names created by the op (removed on undo)
            # Buildings mesh snapshot only for building ops (it's the big one)
            "bmesh":  (self.buildings_mesh.copy()
                       if snapshot_buildings and getattr(self, "buildings_mesh", None) is not None
                       else None),
        }
        stack.append(entry)
        if len(stack) > 10:
            stack.pop(0)
        return entry

    def _editor_undo(self) -> None:
        stack = self.scene_state.get("_editor_undo") or []
        if not stack:
            print("[editor] nothing to undo")
            return
        entry = stack.pop()
        print(f"[editor] undo: {entry['desc']}")

        self.street_graph = entry["graph"]
        if entry["lights"] is not None:
            self.best_positions = entry["lights"]
            self.scene_state.pop("cached_night_coverage", None)
            self.scene_state.pop("cached_night_key", None)
            try:
                self._request_shadow_render(
                    float(self.scene_state["hour"]),
                    float(self.scene_state["spot_radius"]))
            except Exception:
                pass
        for _name in entry["actors"]:
            try:
                self.plotter.remove_actor(_name)
            except Exception:
                pass
            self.scene_state.pop(_name, None)   # drop drape registration too

        # Building op: restore the mesh, rebuild the octree, refresh shadows
        if entry.get("bmesh") is not None:
            self.buildings_mesh = entry["bmesh"]
            try:
                from app_core import _build_octree_from_buildings
                self.octree_root = _build_octree_from_buildings(self.buildings_mesh)
                self.scene_state["edge_shadow_cache"] = {}
                self.scene_state["edge_costs_cache"] = {}
                self._request_shadow_render(
                    float(self.scene_state["hour"]),
                    float(self.scene_state["spot_radius"]))
            except Exception as _b_exc:
                print(f"[editor] building undo shadow refresh failed: {_b_exc}")
            return   # building ops don't touch the street graph — no rebuild

        def _do_undo_rebuild():
            self._rebuild_traffic_and_arrows()
            if hasattr(self, "_sumo_geometry_rebuild"):
                self._sumo_geometry_rebuild()

        self._defer_editor_work(_do_undo_rebuild)

    def _defer_editor_work(self, fn) -> None:
        """Run heavy editor work AFTER the pick event finishes.

        Graph surgery + traffic rebuild + SUMO netconvert must never run inside
        the VTK picking callback: the hardware-selection pass renders the scene
        in flat ID colors, and blocking there leaves that (magenta) selection
        buffer on screen for the whole rebuild.  A one-shot timer lets the pick
        event return and the window restore its normal render first.
        """
        def _cb(_step=None):
            # Turn SSAO OFF for the edit and LEAVE it off: VTK's SSAO pass is
            # corrupted by the actor churn of a rebuild and then presents its
            # normals G-buffer (the flat magenta/lavender frame) on every
            # subsequent render — re-enabling it right after the rebuild brings
            # the broken pass straight back.  The user can re-enable SSAO from
            # the panel checkbox once the scene is stable.
            _ssao_was_on = False
            try:
                _ssao_was_on = bool(self.renderer.GetUseSSAO())
                if _ssao_was_on:
                    self.renderer.SetUseSSAO(False)
                    print("[editor] SSAO disabled during scene edit — "
                          "re-enable it from the SSAO checkbox if wanted")
            except Exception:
                _ssao_was_on = False
            # Present a clean frame FIRST so the user is not staring at the
            # last picking/partial frame during the whole blocking rebuild.
            try:
                self.plotter.render()
                _rw = self.plotter.ren_win
                if _rw is not None:
                    _rw.Render()
            except Exception:
                pass
            try:
                fn()
            except Exception as exc:
                print(f"[editor] deferred edit failed: {exc}")
            try:
                self.plotter.render()
            except Exception:
                pass
        try:
            self.plotter.add_timer_event(max_steps=1, duration=50, callback=_cb)
        except Exception:
            _cb()   # no timer support — run inline as fallback

    def _unified_pick_callback(self, point, picker=None):
        """Single surface picker — handles road info, car-click routing, and editor mode."""
        mode = self.scene_state.get("editor_mode", "view")

        # Heavy editor modes must not run inside the VTK pick event (see
        # _defer_editor_work) — defer a re-entry with the picked point.
        if (mode in ("roads", "roundabouts", "lights", "stops", "buildings", "highway")
                and picker != "_deferred"):
            _pt = (float(point[0]), float(point[1]),
                   float(point[2]) if len(point) > 2 else 0.0)
            self._defer_editor_work(
                lambda: self._unified_pick_callback(_pt, picker="_deferred"))
            return

        if mode == "roads":
            try:
                import osmnx as ox
                px, py = float(point[0]), float(point[1])
                u, v, key = ox.nearest_edges(self.street_graph, px, py)
                self._editor_push_undo(f"road reversal ({u} → {v})")
                reverse_edge(self.street_graph, u, v, key)

                print(f"[editor] Reversed road edge ({u} -> {v}) to ({v} -> {u})")
                self._rebuild_traffic_and_arrows()
                if hasattr(self, "_sumo_geometry_rebuild"):
                    self._sumo_geometry_rebuild()
            except Exception as exc:
                print(f"[editor] Road reversal failed: {exc}")
            return
        elif mode == "roundabouts":
            try:
                px, py = float(point[0]), float(point[1])
                _snap_node = nearest_graph_node(self.street_graph, px, py)

                # Refuse to build a roundabout ON a roundabout ring node —
                # repeated clicks near an existing ring nested them
                # (ra_ra_ra_… nodes) into unusable spaghetti.
                if str(_snap_node).startswith("ra_"):
                    print(f"[editor] node {_snap_node} is already part of a "
                          "roundabout — click elsewhere (or press 'o' to undo)")
                    return

                _undo = self._editor_push_undo(f"roundabout @ node {_snap_node}")
                _undo["actors"].append(f"roundabout_{_snap_node}")
                result = insert_roundabout(self.street_graph, _snap_node)
                self.ring_nodes = result["ring_nodes"]
                cx, cy = result["center"]
                radius = result["radius"]
                print(f"[editor] Generated roundabout at node {_snap_node}")

                # Visual indicator — draped onto terrain when active
                _rz = 0.5
                if self.scene_state.get("_terrain_drape_active"):
                    try:
                        _dem = self.street_graph.graph.get("terrain_sampler")
                        if _dem is not None:
                            _rz += float(np.asarray(_dem(np.array([[cx, cy]])), dtype=float)[0])
                    except Exception:
                        pass
                self.plotter.add_mesh(
                    pv.Cylinder(center=(cx, cy, _rz), direction=(0, 0, 1), radius=radius-2, height=0.2),
                    color="#808080",
                    pbr=False,
                    lighting=False,
                    name=f"roundabout_{_snap_node}"
                )
                self._rebuild_traffic_and_arrows()
                if hasattr(self, "_sumo_geometry_rebuild"):
                    self._sumo_geometry_rebuild()
            except Exception as exc:
                print(f"[editor] Roundabout generation failed: {exc}")
            return
        elif mode == "lights":
            # Toggle the OSM tag and rebuild: build_traffic_lights honors
            # highway="traffic_signals" regardless of degree, giving the editor
            # light the exact same behavior as built-ins — one bulb per
            # incoming approach, proper phase grouping/timings, AND it survives
            # later rebuilds (the old bespoke light object was wiped by the
            # next build_traffic_lights call).  The SUMO export also maps the
            # tag to a type="traffic_light" node, so the network rebuild
            # produces a real signalised junction instead of the TraCI
            # "not a signalised junction" skip.
            try:
                px, py = float(point[0]), float(point[1])
                _snap_node = nearest_graph_node(self.street_graph, px, py)
                ndata = self.street_graph.nodes[_snap_node]
                self._editor_push_undo(f"traffic light toggle @ node {_snap_node}")
                if ndata.get("highway") == "traffic_signals":
                    ndata["highway"] = None
                    print(f"[editor] Removed traffic light at node {_snap_node}")
                else:
                    _has_approach = any(p["v"] == _snap_node for p in self.car_paths)
                    if not _has_approach:
                        print(f"[editor] Cannot add light: no incoming paths at node {_snap_node}")
                        return
                    ndata["highway"] = "traffic_signals"
                    print(f"[editor] Added traffic light at node {_snap_node}")
                self._rebuild_traffic_and_arrows()
                if hasattr(self, "_sumo_geometry_rebuild"):
                    self._sumo_geometry_rebuild()
            except Exception as exc:
                print(f"[editor] Traffic light toggle failed: {exc}")
            return
        elif mode == "stops":
            try:
                px, py = float(point[0]), float(point[1])
                _snap_node = nearest_graph_node(self.street_graph, px, py)

                self._editor_push_undo(f"stop sign toggle @ node {_snap_node}")
                current_tag = self.street_graph.nodes[_snap_node].get("highway")
                if current_tag == "stop":
                    self.street_graph.nodes[_snap_node]["highway"] = None
                    print(f"[editor] Removed stop sign at node {_snap_node}")
                else:
                    self.street_graph.nodes[_snap_node]["highway"] = "stop"
                    print(f"[editor] Added stop sign at node {_snap_node}")

                self._rebuild_traffic_and_arrows()
                # Stop signs affect junction priorities; rebuild the SUMO network
                if hasattr(self, "_sumo_geometry_rebuild"):
                    self._sumo_geometry_rebuild()
            except Exception as exc:
                print(f"[editor] Stop sign toggle failed: {exc}")
            return
        elif mode == "streetlights":
            try:
                px, py = float(point[0]), float(point[1])
                self._editor_push_undo("streetlight add/remove")
                if self.best_positions is not None and self.best_positions.shape[0] > 0:
                    _dists = np.linalg.norm(self.best_positions[:, :2] - np.array([px, py]), axis=1)
                    _nearest_idx = int(np.argmin(_dists))
                    if float(_dists[_nearest_idx]) < 5.0:
                        self.best_positions = np.delete(self.best_positions, _nearest_idx, axis=0)
                        print(f"[editor] Removed streetlight near ({px:.1f}, {py:.1f})")
                    else:
                        self.best_positions = np.vstack([self.best_positions, [px, py]])
                        print(f"[editor] Added streetlight at ({px:.1f}, {py:.1f})")
                else:
                    self.best_positions = np.array([[px, py]])
                    print(f"[editor] Added first streetlight at ({px:.1f}, {py:.1f})")

                self.scene_state["show_streetlights"] = True
                self.scene_state.pop("cached_night_coverage", None)
                self.scene_state.pop("cached_night_key", None)
                self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))
            except Exception as exc:
                print(f"[editor] Streetlight toggle failed: {exc}")
            return

        elif mode == "buildings":
            try:
                px, py = float(point[0]), float(point[1])
                from editor_ops import make_building_mesh, building_size_for_click
                _w, _d, _h = building_size_for_click(px, py)

                # Base z follows terrain when draping is active
                _bz = 0.0
                if self.scene_state.get("_terrain_drape_active"):
                    try:
                        _dem = self.street_graph.graph.get("terrain_sampler")
                        if _dem is not None:
                            _bz = float(np.asarray(_dem(np.array([[px, py]])), dtype=float)[0])
                    except Exception:
                        pass

                _undo = self._editor_push_undo(
                    f"building @ ({px:.0f}, {py:.0f})", snapshot_buildings=True)
                _n = int(self.scene_state.get("_user_building_count", 0)) + 1
                self.scene_state["_user_building_count"] = _n
                _actor_name = f"user_building_{_n}"
                _undo["actors"].append(_actor_name)

                box = make_building_mesh(px, py, _w, _d, _h, base_z=_bz)

                # Visual actor (matches the active style preset's palette)
                try:
                    _style = self._style()
                except Exception:
                    _style = {}
                _ba = self.plotter.add_mesh(
                    box, color=str(_style.get("building", "#1e2d45")),
                    show_edges=True, edge_color=str(_style.get("building_edge", "#2a4a6b")),
                    smooth_shading=False, reset_camera=False, name=_actor_name,
                )
                # Registered under its name so the terrain drape toggle can
                # lift/restore it via _drape_actor_points like other layers
                self.scene_state[_actor_name] = _ba

                # Physics: merge into the building set + rebuild the shadow
                # octree so the new building casts shadows and cuts the solar
                # harvest on adjacent streets immediately.
                from app_core import _build_octree_from_buildings
                if getattr(self, "buildings_mesh", None) is not None:
                    self.buildings_mesh = self.buildings_mesh.merge(
                        box, merge_points=False)
                else:
                    self.buildings_mesh = box
                self.octree_root = _build_octree_from_buildings(self.buildings_mesh)
                self.scene_state["edge_shadow_cache"] = {}
                self.scene_state["edge_costs_cache"] = {}
                self._request_shadow_render(
                    float(self.scene_state["hour"]),
                    float(self.scene_state["spot_radius"]))

                print(f"[editor] Building #{_n} placed at ({px:.0f}, {py:.0f}) — "
                      f"{_w:.0f}×{_d:.0f} m, {_h:.0f} m tall; shadows + solar "
                      f"harvest recomputing ('o' to undo)")
            except Exception as exc:
                print(f"[editor] Building placement failed: {exc}")
            return

        elif mode == "highway":
            # Two-click bridge placement: first click sets start node (orange
            # sphere marker), second click finalises the bridge tube + graph edge.
            try:
                px, py = float(point[0]), float(point[1])
                _snap = nearest_graph_node(self.street_graph, px, py)
                _nx = float(self.street_graph.nodes[_snap]["x"])
                _ny = float(self.street_graph.nodes[_snap]["y"])

                pending_node = self.scene_state.get("_bridge_start")

                if pending_node is None:
                    # ── First click: mark the start ──
                    self.scene_state["_bridge_start"] = _snap
                    _mk = pv.Sphere(radius=3.5, center=(_nx, _ny, 7.0))
                    self.plotter.add_mesh(
                        _mk, color="#e85d04", render_points_as_spheres=False,
                        reset_camera=False, name="highway_start_marker",
                    )
                    self.scene_state["_bridge_start_actor"] = True
                    self._walk_hud(
                        f"Highway: start = node {_snap}  — click second endpoint  [y]=cancel")
                    print(f"[editor] Highway: start node {_snap} at ({_nx:.0f}, {_ny:.0f})")

                else:
                    # ── Second click: build the bridge ──
                    u = pending_node
                    v = _snap
                    if u == v:
                        print("[editor] Highway: same node for start and end — click elsewhere")
                        return
                    ux = float(self.street_graph.nodes[u]["x"])
                    uy = float(self.street_graph.nodes[u]["y"])
                    vx = float(self.street_graph.nodes[v]["x"])
                    vy = float(self.street_graph.nodes[v]["y"])

                    from editor_ops import (make_highway_tube,
                                            add_bridge_highway_edge,
                                            bridge_clearance_height)
                    # Deck must clear every building under the span; quantised
                    # to the 5 m/layer grid _edge_bridge_height uses so the
                    # cars' z profile matches the visual tube exactly.
                    _bpts = (self.buildings_mesh.points
                             if getattr(self, "buildings_mesh", None) is not None
                             else None)
                    _need = bridge_clearance_height(
                        _bpts, ux, uy, vx, vy, base_height=5.0)
                    import math
                    _layer = max(1, int(math.ceil(_need / 5.0)))
                    _BRIDGE_H = 5.0 * _layer

                    _undo = self._editor_push_undo(f"highway {u}→{v}")
                    add_bridge_highway_edge(
                        self.street_graph, u, v, ux, uy, vx, vy, layer=_layer)

                    _tube = make_highway_tube(ux, uy, vx, vy, height=_BRIDGE_H)
                    _n = int(self.scene_state.get("_user_highway_count", 0)) + 1
                    self.scene_state["_user_highway_count"] = _n
                    _hname = f"user_highway_{_n}"
                    _undo["actors"].append(_hname)
                    self.plotter.add_mesh(
                        _tube, color="#e85d04",
                        smooth_shading=True, reset_camera=False,
                        name=_hname,
                    )

                    # Also add vertical support pillars every ~30 m
                    _span = math.hypot(vx - ux, vy - uy)
                    _ramp = min(18.0, _span / 2.0)
                    _n_pillars = max(0, int(_span / 30) - 1)
                    for _pi in range(1, _n_pillars + 1):
                        _t = _pi / (_n_pillars + 1)
                        _px2 = ux + (vx - ux) * _t
                        _py2 = uy + (vy - uy) * _t
                        # Same ramped flat-deck profile as the tube
                        _along = min(_t * _span, (1.0 - _t) * _span)
                        _ph = _BRIDGE_H * min(1.0, _along / max(_ramp, 1e-6))
                        if _ph < 1.0:
                            continue   # skip degenerate stubs on the ramps
                        _cyl = pv.Cylinder(
                            center=(_px2, _py2, _ph / 2),
                            direction=(0, 0, 1),
                            radius=0.6, height=_ph)
                        _pname = f"{_hname}_pillar_{_pi}"
                        _undo["actors"].append(_pname)
                        self.plotter.add_mesh(
                            _cyl, color="#c0522a",
                            smooth_shading=True, reset_camera=False,
                            name=_pname,
                        )

                    # Remove start marker
                    try:
                        self.plotter.remove_actor("highway_start_marker")
                    except Exception:
                        pass
                    self.scene_state.pop("_bridge_start", None)
                    self.scene_state.pop("_bridge_start_actor", None)

                    self._rebuild_traffic_and_arrows()
                    if hasattr(self, "_sumo_geometry_rebuild"):
                        self._sumo_geometry_rebuild()

                    self._walk_hud("")
                    print(f"[editor] Highway bridge placed: node {u} → {v} "
                          f"({_span:.0f} m span, {_BRIDGE_H:.0f} m elevated)  "
                          f"('o' to undo)")
            except Exception as exc:
                import traceback
                print(f"[editor] Highway placement failed: {exc}")
                traceback.print_exc()
            return

        # ── Pedestrian click: enter 1st-person follow of THAT pedestrian ──
        #    (tight 3.5 m radius so cars/roads nearby still get their clicks)
        if self.scene_state.get("walk_mode") is None:
            try:
                if self._walk_select_ped_at(float(point[0]), float(point[1])):
                    return
            except Exception as _walk_exc:
                print(f"[walk] ped-click failed: {_walk_exc}")
        else:
            # Actively walking: clicks must not start route planning or select
            # cars underneath the walker — that hijacked the session mid-walk.
            return

        # ── SUMO vehicle routing: in sumo engine mode, clicks near SUMO vehicles
        #    select them; the follow-up click retargets the vehicle via TraCI. ──
        if (getattr(self.args, "engine", "idm") == "sumo"
                and getattr(self, "sumo", None) and self.sumo.get("enabled")):
            try:
                if self._sumo_route_click(float(point[0]), float(point[1])):
                    return
            except Exception as _sumo_rc_exc:
                print(f"[sumo-route] click failed: {_sumo_rc_exc}")

        # ── Car proximity check: if a car is within 8 m of the click, treat as car click ──
        _pos_arr = self.car_anim.get("pos")
        if _pos_arr is not None:
            _pos_arr = np.asarray(_pos_arr, dtype=float)
            if _pos_arr.shape[0] > 0 and _pos_arr.ndim == 2 and _pos_arr.shape[1] >= 2:
                _click_xy = np.array([float(point[0]), float(point[1])], dtype=float)
                _dists = np.linalg.norm(_pos_arr[:, :2] - _click_xy, axis=1)
                _nearest_car = int(np.argmin(_dists))
                if float(_dists[_nearest_car]) < 8.0:
                    # Delegate to the existing car-pick logic
                    try:
                        _e = int(self.car_anim["edge_idx"][_nearest_car])
                        _d = float(self.car_anim["dist"][_nearest_car])
                        _p3d = self._car_pose_on_path(self.car_paths[_e], _d)
                        _snap_node = nearest_graph_node(self.street_graph, _p3d[0], _p3d[1])
                        self._clear_route_actors()
                        self.route_state["source_node"] = _snap_node
                        _xy = self._node_xy(_snap_node)
                        if _xy is not None:
                            _src_actor = self.plotter.add_mesh(
                                pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], 2.0),
                                          theta_resolution=18, phi_resolution=18),
                                color="#33cc66", render=False,
                            )
                            self.route_state["route_actors"] = [_src_actor]
                        self._select_car_for_route(_nearest_car, _snap_node)
                        self.route_state["stage"] = 1
                        print(f"[route] Car {_nearest_car} selected as source — click target.")
                        if getattr(self.args, "solo", False):
                            self.scene_state["solo_solar_Wh"] = 0.0
                            self.scene_state["solo_mech_Wh"] = 0.0
                    except Exception as _exc:
                        print(f"[cars] Car pick failed: {_exc}")
                    return   # do NOT fall through to road-info logic

        # ── Road info + route planning (original _road_pick_callback logic) ──
        self._road_pick_callback(point)

    def _register_unified_picker(self) -> None:
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        # Prefer the software CELL picker (vtkCellPicker ray-cast).  The default
        # hardware picker renders the whole scene in flat ID colors (a magenta
        # frame) for every pick — on macOS that frame can linger on screen.
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._unified_pick_callback,
                show_message=False,
                show_point=False,
                tolerance=0.025,
                picker="cell",
            )
            print("[picker] unified surface picker registered (software cell picker)")
        except TypeError:
            try:
                self.plotter.enable_surface_point_picking(
                    callback=self._unified_pick_callback,
                    show_message=False,
                    show_point=False,
                    tolerance=0.025,
                )
                print("[picker] unified surface picker registered (default picker)")
            except Exception as _exc:
                print(f"[picker] enable_surface_point_picking not available: {_exc}")
        except Exception as _exc:
            print(f"[picker] enable_surface_point_picking not available: {_exc}")

    def _hex_to_rgb(self, h: str) -> list[int]:
            h = h.lstrip("#")
            return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

    def _poi_label(self, p: dict) -> str:
            cats = p["categories"]
            tag = next((f"[{self._CAT_ABBREV[c]}] " for c in cats if c in self._CAT_ABBREV), "")
            name = p["name"] or (cats[0] if cats else "place")
            # Strip non-ASCII: VTK's text renderer only handles ASCII
            name = name.encode("ascii", "ignore").decode("ascii").strip() or "place"
            max_name = 28 - len(tag)
            if len(name) > max_name:
                name = name[:max(max_name - 3, 4)] + "..."
            return tag + name

    def _apply_visual_updates(self, hour: float, spot_radius: float, is_night: bool, sun_dir, style: dict) -> None:
            """Apply all non-blocking visual updates (background, lights, sun sphere, actor colors).
                This runs on the main thread and returns immediately."""
            from app_core import _load_hdri, _build_spotlight_discs, _apply_atmosphere
            # Background / skybox
            has_skybox = False
            try:
                hdri_tex = _load_hdri(float(hour), str(self.scene_state["preset"]))
                self.plotter.set_environment_texture(hdri_tex)
                self.plotter.renderer.UseImageBasedLightingOn()
                try:
                    self.plotter.add_background_cube_map(hdri_tex)
                    has_skybox = True
                except Exception as _bg_exc:
                    print(f"[pbr] failed to set background cubemap: {_bg_exc}")
            except Exception as _exc:
                print(f"[pbr] dynamic HDRI unavailable ({_exc}); using gradient background")

            if not has_skybox:
                bg = style.get("day_bg" if not is_night else "night_bg")
                if isinstance(bg, list) and len(bg) == 2:
                    self.plotter.set_background(str(bg[0]), top=str(bg[1]))
                else:
                    self.plotter.set_background(str(bg))

            # Sun sphere
            try:
                self.plotter.remove_actor("sun_sphere", reset_camera=False)
            except Exception:
                pass
            if not is_night:
                try:
                    _sun_pos = np.asarray(sun_dir, dtype=float) * 500.0
                    _sun_sphere = pv.Sphere(
                        radius=15.0,
                        center=(float(_sun_pos[0]), float(_sun_pos[1]), float(_sun_pos[2])),
                    )
                    self.plotter.add_mesh(
                        _sun_sphere, color="#ffeb3b", emissive=True,
                        name="sun_sphere", lighting=False,
                    )
                except Exception as _exc:
                    print(f"[sun-sphere] failed: {_exc}")

            # Dynamic scene lights
            try:
                self.plotter.remove_all_lights()
                _sun_pos = np.asarray(sun_dir, dtype=float) * 500.0
                _sun_intensity = max(0.0, float(sun_dir[2])) * 0.9 + 0.1
                _sun_light = pv.Light(
                    light_type="scene light",
                    position=(float(_sun_pos[0]), float(_sun_pos[1]), float(_sun_pos[2])),
                    focal_point=(0.0, 0.0, 0.0),
                    intensity=float(_sun_intensity),
                )
                # pv.Light exposes *_color properties (plain .ambient does not
                # exist — assigning it raised every pass, so the dynamic
                # sun/moon lights silently never applied)
                _sun_light.ambient_color = (0.25, 0.25, 0.25)
                _sun_light.diffuse_color = (0.85, 0.85, 0.85)
                _sun_light.specular_color = (0.1, 0.1, 0.1)
                self.plotter.add_light(_sun_light)
                if is_night:
                    # Bright enough that buildings stay readable at night now
                    # that scene lights actually apply (the old broken
                    # `.ambient` meant VTK fell back to a full headlight)
                    _moon_light = pv.Light(
                        light_type="scene light",
                        position=(0.0, 0.0, 500.0),
                        intensity=0.45,
                    )
                    _moon_light.ambient_color = (0.30, 0.30, 0.36)
                    _moon_light.diffuse_color = (0.35, 0.35, 0.42)
                    self.plotter.add_light(_moon_light)
            except Exception as _exc:
                print(f"[light] dynamic scene light setup failed: {_exc}")

            # Actor colors
            b_actor = self.scene_state.get("building_actor")
            _pbr_actors = self.scene_state.get("_building_actors_pbr", {})
            _edge_rgb = pv.Color(str(style["building_edge"])).float_rgb
            if _pbr_actors:
                for _ba in _pbr_actors.values():
                    try:
                        _ba.GetProperty().SetEdgeColor(_edge_rgb)
                    except Exception:
                        pass
            elif b_actor is not None:
                try:
                    b_actor.GetProperty().SetColor(pv.Color(str(style["building"])).float_rgb)
                    b_actor.GetProperty().SetEdgeColor(_edge_rgb)
                except Exception:
                    pass
            v_actor = self.scene_state.get("vehicle_actor")
            if v_actor is not None:
                try:
                    v_actor.GetProperty().SetColor(pv.Color(str(style["vehicle"])).float_rgb)
                except Exception:
                    pass
            p_actor = self.scene_state.get("ped_actor")
            if p_actor is not None:
                try:
                    p_actor.GetProperty().SetColor(pv.Color(str(style["ped"])).float_rgb)
                except Exception:
                    pass
            car_actors = self.scene_state.get("car_actors")
            if isinstance(car_actors, dict):
                car_actor = car_actors.get("cars")
                if car_actor is not None:
                    try:
                        car_actor.GetProperty().SetColor(pv.Color(str(style["car"])).float_rgb)
                    except Exception:
                        pass

            # Street-light poles and bulbs
            try:
                light_color = str(style["light_night"]) if is_night else str(style["light_day"])

                if self.best_positions is not None and self.best_positions.shape[0] > 0:
                    _ph = float(self.args.pole_height)
                    # Terrain-drape: lift pole base and bulb by DEM height
                    if self.scene_state.get("_terrain_drape_active"):
                        _sl_dem = self.street_graph.graph.get("terrain_sampler")
                        if _sl_dem is not None:
                            try:
                                _base_h = np.asarray(
                                    _sl_dem(self.best_positions[:, :2]), dtype=float
                                )
                            except Exception:
                                _base_h = np.zeros(self.best_positions.shape[0])
                        else:
                            _base_h = np.zeros(self.best_positions.shape[0])
                    else:
                        _base_h = np.zeros(self.best_positions.shape[0])
                    lp = np.column_stack((
                        self.best_positions[:, 0],
                        self.best_positions[:, 1],
                        _base_h + _ph,
                    ))
                    poles_lines = []
                    for _pi, pt in enumerate(lp):
                        _bz = float(_base_h[_pi])
                        poles_lines.append(pv.Line(
                            (pt[0], pt[1], _bz), (pt[0], pt[1], _bz + _ph)
                        ))
                    poles_mesh = pv.MultiBlock(poles_lines).combine()

                    bulb_mesh = pv.PolyData(lp)
                    # Night: large emissive glow sphere visible from aerial view.
                    # Day: small opaque bulb to mark pole positions.
                    bulb_radius = 1.8 if is_night else 0.3
                    bulb_glyph = bulb_mesh.glyph(geom=pv.Sphere(radius=bulb_radius, theta_resolution=8, phi_resolution=8))

                    self.scene_state["lights_actor"] = self.plotter.add_mesh(
                        bulb_glyph,
                        color=light_color,
                        name="streetlight_points",
                        # lighting=False makes the sphere render at full colour
                        # regardless of scene-light intensity (simulates emissive).
                        lighting=not is_night,
                        opacity=1.0,
                    )
                    self.scene_state["poles_actor"] = self.plotter.add_mesh(
                        poles_mesh,
                        color="#555555",
                        line_width=3,
                        name="streetlight_poles",
                    )
                    self._set_actor_visibility(self.scene_state.get("lights_actor"), bool(self.scene_state["show_streetlights"]))
                    self._set_actor_visibility(self.scene_state.get("poles_actor"), bool(self.scene_state["show_streetlights"]))
            except Exception as _exc:
                print(f"[lights] failed to add lights actor: {_exc}")

            # Spotlight discs (night only) — ground halos that simulate light pooling
            if is_night:
                _bp = self.best_positions if self.best_positions is not None else np.empty((0, 2))
                discs = _build_spotlight_discs(_bp, spot_radius)
                if discs is not None:
                    self.scene_state["spotlight_actor"] = self.plotter.add_mesh(
                        discs,
                        color=str(style["disc"]),
                        opacity=0.55,      # more visible than 0.38
                        show_edges=False,
                        lighting=False,    # discs glow at full colour
                        name="spotlight_discs",
                    )
                self._set_actor_visibility(self.scene_state.get("spotlight_actor"), bool(self.scene_state["show_streetlights"]))
            else:
                try:
                    self.plotter.remove_actor("spotlight_discs", reset_camera=False)
                    self.scene_state["spotlight_actor"] = None
                except Exception:
                    pass

            self._set_actor_visibility(self.scene_state.get("vehicle_actor"), bool(self.scene_state["show_roads"]))
            self._set_actor_visibility(self.scene_state.get("ped_actor"), bool(self.scene_state["show_roads"]))
            self._render_cars()
            _apply_atmosphere(
                self.plotter,
                hour=float(hour),
                is_night=bool(is_night),
                preset=str(self.scene_state["preset"]),
            )

    def _on_time_change(self, value: float) -> None:
            self.scene_state["hour"] = float(value)
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))

    def _on_radius_change(self, value: float) -> None:
            self.scene_state["spot_radius"] = float(value)
            # Invalidate cached coverage since spotlight radius changed
            self.scene_state.pop("cached_night_coverage", None)
            self.scene_state.pop("cached_night_key", None)
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))

    def _toggle_roads(self, value: bool) -> None:
            self.scene_state["show_roads"] = bool(value)
            self._set_actor_visibility(self.scene_state.get("vehicle_actor"), bool(value))
            self._set_actor_visibility(self.scene_state.get("ped_actor"), bool(value))
            self.plotter.update()

    def _toggle_cars(self, value: bool) -> None:
            self.scene_state["show_cars"] = bool(value)
            car_actors = self.scene_state.get("car_actors")
            if isinstance(car_actors, dict):
                for actor in car_actors.values():
                    self._set_actor_visibility(actor, bool(value))
            for _ua in self.scene_state.get("_ultra_car_actors") or []:
                self._set_actor_visibility(_ua, bool(value))
            self.plotter.update()

    def _preset_mini(self, _: bool) -> None:
            self.scene_state["preset"] = "mini"
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))

    def _preset_coastal(self, _: bool) -> None:
            self.scene_state["preset"] = "coastal"
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))

    def _preset_sunset(self, _: bool) -> None:
            self.scene_state["preset"] = "sunset"
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            self._request_shadow_render(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]))

    def _toggle_arrows(self, value: bool) -> None:
            self.scene_state["show_arrows"] = bool(value)
            self._set_actor_visibility(self.scene_state.get("street_arrows_actor"), bool(value))
            self.plotter.update()

    def _toggle_streetlights(self, value: bool) -> None:
            self.scene_state["show_streetlights"] = bool(value)
            self._set_actor_visibility(self.scene_state.get("lights_actor"), bool(value))
            self._set_actor_visibility(self.scene_state.get("spotlight_actor"), bool(value))
            self.plotter.update()

    def _toggle_traffic_signals(self, value: bool) -> None:
            self.scene_state["show_traffic_signals"] = bool(value)
            self._set_actor_visibility(self.scene_state.get("tl_actor"), bool(value))
            self.plotter.update()

    def _do_ga_thread(self) -> None:
        try:
            from app_core import _build_grid_points_from_ground_mesh, _load_or_build_coverage_matrix_cached
            if self.args.light_strategy == "smart":
                self.best_positions = np.asarray(self._compute_smart_lights(), dtype=float)
                print(f"[lights] Smart placement complete: {self.best_positions.shape[0]} lights")
            else:
                cov_matrix_local = None
                sw_poly_local = None
                if self.args.fast_startup:
                    sw_poly_local = build_sidewalk_polygon_from_street_graph(self.street_graph)
                    gp = _build_grid_points_from_ground_mesh(
                        self.ground_mesh, self.args.grid_step, sidewalk_polygon=sw_poly_local
                    )
                    cov_matrix_local = _load_or_build_coverage_matrix_cached(
                        cache_dir=self.cache_dir,
                        use_cache=self.use_cache,
                        cache_context_key=self.cache_context_key,
                        grid_points=gp,
                        ground_mesh=self.ground_mesh,
                        octree_root=self._ensure_octree(),
                        street_graph=self.street_graph,
                        radius=self.args.light_radius,
                        pole_height=self.args.pole_height,
                        n_jobs=(None if self.args.coverage_jobs == 0 else self.args.coverage_jobs),
                    )
                ga_result = optimize_streetlights(
                    ground_mesh=self.ground_mesh,
                    n_lights=self.args.n_lights,
                    light_radius=self.args.light_radius,
                    w1=self.args.w1,
                    w2=self.args.w2,
                    grid_step=self.args.grid_step,
                    population_size=self.args.population,
                    generations=self.args.generations,
                    mutation_rate=self.args.mutation,
                    seed=self.args.seed,
                    octree_root=self._ensure_octree(),
                    pole_height=self.args.pole_height,
                    use_precomputed_coverage=bool(self.args.fast_startup),
                    precomputed_coverage_matrix=cov_matrix_local,
                    street_graph=self.street_graph,
                    sidewalk_polygon=sw_poly_local,
                    ga_jobs=max(1, int(self.args.ga_jobs)),
                    ga_progress_every=max(1, int(self.args.ga_progress_every)),
                    ga_verbose=True,
                )
                self.best_positions = np.asarray(ga_result["best_positions"], dtype=float)
                print(f"[ga] Optimization complete: cost={ga_result['best_cost']:.4f}, lit={ga_result['lit_ratio']:.3f}")
            self.scene_state.pop("cached_night_coverage", None)
            self.scene_state.pop("cached_night_key", None)
            self.scene_state["ga_done"] = True
        except Exception as _exc:
            print(f"[lights] Background placement failed: {_exc}")
            self.scene_state["ga_done"] = True
        finally:
            self.scene_state["ga_running"] = False

    def _on_optimize(self, _: bool) -> None:
            if not bool(self.scene_state.get("interactive_ready", False)):
                return
            if bool(self.scene_state.get("ga_running", False)):
                # Already running — ignore double-click
                return
            self.scene_state["ga_running"] = True
            self.plotter.add_text("Running streetlight placement...", position=(0.18, 0.02), name="status", font_size=9, viewport=True)
            self.plotter.update()

            import threading as _threading_ga
            _threading_ga.Thread(target=self._do_ga_thread, daemon=True).start()

    def _poll_ga_done(self, _: int) -> None:
            """Timer callback: fires every 500ms to flush the GA result into the viewer."""
            try:
                if not bool(self.scene_state.pop("ga_done", False)):
                    return
                try:
                    self._render_ground(float(self.scene_state["hour"]), float(self.scene_state["spot_radius"]), _use_update=True)
                except Exception as _exc:
                    print(f"[ga] Post-GA render failed: {_exc}")
            except Exception as _exc:
                print(f"[ga-poll ERROR] {_exc}")

    def _panel_text_color(self):
            # Choose text color for panel for readability
            bg_val = self._style()["day_bg"] if not self.scene_state.get("is_night") else self._style()["night_bg"]
            bg = str(bg_val[0] if isinstance(bg_val, list) else bg_val)
            # Simple luminance check
            bg = bg.lstrip("#")
            r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return "#f4f4f4" if luminance < 128 else "#222222"

    def _panel_desc_color(self):
            # Muted but still readable
            base = self._panel_text_color()
            if base == "#f4f4f4":
                return "#e0e0e0"
            return "#444444"

    def _cy(self, row: int) -> float:
            """Return checkbox_y_px for given row (0=top)."""
            y_px = 836 - row * 30
            return float(y_px)

    def _toggle_pois_lazy(self, val: bool) -> None:
        self.scene_state["show_pois"] = bool(val)
        actor = self.scene_state.get("poi_actor")
        mesh = self.scene_state.get("_poi_mesh")
        if val and actor is None and mesh is not None:
            self.scene_state["poi_actor"] = self.plotter.add_mesh(
                mesh,
                scalars="colors",
                rgb=True,
                style="points",
                point_size=14,
                render_points_as_spheres=True,
                lighting=False,
                reset_camera=False,
            )
            print("[poi] POI dots created")
        elif actor is not None:
            self._set_actor_visibility(actor, val)

    def _toggle_poi_names_lazy(self, val: bool) -> None:
        self.scene_state["show_poi_names"] = bool(val)
        actor = self.scene_state.get("poi_labels_actor")
        mesh = self.scene_state.get("_poi_mesh")
        if val and actor is None and mesh is not None:
            self.scene_state["poi_labels_actor"] = self.plotter.add_point_labels(
                mesh,
                "labels",
                point_size=1,
                font_size=14,
                text_color="white",
                render_points_as_spheres=False,
                always_visible=False,
                shape_opacity=0.55,
                shape_color="#111111",
                tolerance=0.01,
            )
            print("[poi] POI labels created")
        elif actor is not None:
            self._set_actor_visibility(actor, val)

    def _toggle_ssao(self, val: bool) -> None:
            try:
                self.renderer.SetUseSSAO(bool(val))
                self.plotter.update()
            except Exception:
                print("[ssao] SSAO unavailable")

    def _on_route_alpha_change(self, value: float) -> None:
        self.scene_state["route_alpha"] = float(np.clip(value, 0.0, 1.0))
        if (
            self.route_state.get("stage") == 2
            and self.route_state.get("source_node") is not None
            and self.route_state.get("target_node") is not None
        ):
            try:
                self._compute_and_render_routes(self.route_state["source_node"], self.route_state["target_node"])
                self.plotter.update()
            except Exception as _exc:
                print(f"[route] alpha update failed: {_exc}")

    def _on_route_hour_change(self, value: float) -> None:
        self.scene_state["route_hour"] = float(np.clip(value, 6.0, 18.0))
        self.scene_state["edge_shadow_cache"] = {}
        self.scene_state["edge_costs_cache"] = {}
        if (
            self.route_state.get("stage") == 2
            and self.route_state.get("source_node") is not None
            and self.route_state.get("target_node") is not None
        ):
            try:
                self._compute_and_render_routes(self.route_state["source_node"], self.route_state["target_node"])
                self.plotter.update()
            except Exception as _exc:
                print(f"[route] route hour update failed: {_exc}")

    def _on_panel_area_change(self, value: float) -> None:
        _old = self.scene_state.get("solar_params")
        from solar_physics import SolarParams
        if isinstance(_old, SolarParams):
            self.scene_state["solar_params"] = SolarParams(
                roof_area_m2=float(np.clip(value, 0.5, 3.0)),
                panel_efficiency=float(_old.panel_efficiency),
                temperature_derating=float(_old.temperature_derating),
                vehicle_mass_kg=float(_old.vehicle_mass_kg),
                rolling_coeff=float(_old.rolling_coeff),
                drag_coeff=float(_old.drag_coeff),
                frontal_area_m2=float(_old.frontal_area_m2),
            )
        else:
            self.scene_state["solar_params"] = SolarParams(roof_area_m2=float(np.clip(value, 0.5, 3.0)))
        self.scene_state["edge_costs_cache"] = {}
        if (
            self.route_state.get("stage") == 2
            and self.route_state.get("source_node") is not None
            and self.route_state.get("target_node") is not None
        ):
            try:
                self._compute_and_render_routes(self.route_state["source_node"], self.route_state["target_node"])
                self.plotter.update()
            except Exception as _exc:
                print(f"[route] panel area update failed: {_exc}")

    def _pan_camera(self, dx: float, dy: float) -> None:
            cam = self.plotter.camera
            pos = np.asarray(cam.position, dtype=float)
            focal = np.asarray(cam.focal_point, dtype=float)
            up = np.asarray(cam.up, dtype=float)
            view = focal - pos
            nv = float(np.linalg.norm(view))
            nu = float(np.linalg.norm(up))
            if nv <= 1e-9 or nu <= 1e-9:
                return
            view = view / nv
            up = up / nu
            right = np.cross(view, up)
            nr = float(np.linalg.norm(right))
            if nr <= 1e-9:
                return
            right = right / nr
            shift = (right * float(dx)) + (up * float(dy))
            cam.position = tuple(pos + shift)
            cam.focal_point = tuple(focal + shift)
            self.plotter.update()

    def _rotate_camera(self, azimuth_deg: float = 0.0, elevation_deg: float = 0.0) -> None:
            cam = self.plotter.camera
            if abs(float(azimuth_deg)) > 0.0:
                cam.Azimuth(float(azimuth_deg))
            if abs(float(elevation_deg)) > 0.0:
                cam.Elevation(float(elevation_deg))
            cam.OrthogonalizeViewUp()
            self.plotter.update()

    def _zoom_camera(self, factor: float) -> None:
            cam = self.plotter.camera
            cam.Dolly(float(factor))
            self.plotter.reset_camera_clipping_range()
            self.plotter.update()

    def _reset_camera_view(self) -> None:
            self.plotter.view_isometric()
            self.plotter.update()

    def _deferred_ssao_enable(self, _: int) -> None:
        try:
            self.plotter.render()   # flush pipeline before enabling SSAO to prevent dark first frame
            self.renderer.SetUseSSAO(True)
            self.plotter.update()
            print("[ssao] SSAO activated (deferred)")
        except Exception as _exc:
            print(f"[ssao] deferred SSAO failed: {_exc}")

    def _deferred_initial_render(self, _: int) -> None:
        try:
            print("[timer] _deferred_initial_render: triggering initial shadow computation")
            self._request_shadow_render(12.0, float(self.args.light_radius))
        except Exception as _exc:
            print(f"[init-render ERROR] {_exc}")
