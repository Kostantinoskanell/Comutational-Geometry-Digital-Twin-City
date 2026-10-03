from __future__ import annotations
import hashlib
import time
import numpy as np
import networkx as nx
import pyvista as pv
from shapely.geometry import Point as _SPoint, LineString as _LS
from solar_routing import (
    build_edge_costs,
    build_graph_with_costs,
    find_energy_optimal_route,
    find_joint_optimal_route,
    find_pareto_routes,
    nearest_graph_node,
)
from traffic_lights import build_traffic_lights as _build_traffic_lights
from turn_restrictions import build_next_edges as _build_next_edges


class RouteMixin:
    def _remove_pareto_chart(self) -> None:
            _chart = self.scene_state.get("pareto_chart")
            if _chart is None:
                return
            try:
                self.plotter.remove_chart(_chart)
            except Exception:
                pass
            self.scene_state["pareto_chart"] = None

    def _clear_route_actors(self, keep_markers: bool = False) -> None:
            actors = self.route_state.get("route_actors", [])
            start_idx = 2 if keep_markers else 0
            to_remove = list(actors)[start_idx:]
            for _a in list(actors):
                if _a not in to_remove:
                    continue
                try:
                    self.plotter.remove_actor(_a, reset_camera=False)
                except Exception:
                    pass
            if keep_markers:
                self.route_state["route_actors"] = list(actors)[:2]
            else:
                self.route_state["route_actors"] = []

    def _node_xy(self, _node_id: object) -> tuple[float, float] | None:
            _d = self.street_graph.nodes.get(_node_id, {})
            if "x" in _d and "y" in _d:
                return float(_d["x"]), float(_d["y"])
            return None

    def _edge_best(self, _g: nx.MultiDiGraph, _u: object, _v: object, _weight: str) -> dict | None:
            _ed = _g.get_edge_data(_u, _v)
            if not _ed:
                return None
            if isinstance(_ed, dict) and all(isinstance(v, dict) for v in _ed.values()):
                return min(_ed.values(), key=lambda x: float(x.get(_weight, np.inf)))
            return _ed

    def _route_polyline(self, _g: nx.MultiDiGraph, _nodes: list[object], _weight: str) -> np.ndarray | None:
            if len(_nodes) < 2:
                return None
            _pts: list[list[float]] = []
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _ed = self._edge_best(_g, _a, _b, _weight)
                if _ed is None:
                    continue
                _geom = _ed.get("geometry") if isinstance(_ed, dict) else None
                if _geom is not None and hasattr(_geom, "coords"):
                    _xy = np.asarray(_geom.coords, dtype=float)[:, :2]
                else:
                    _pa = self._node_xy(_a)
                    _pb = self._node_xy(_b)
                    if _pa is None or _pb is None:
                        continue
                    _xy = np.asarray([_pa, _pb], dtype=float)
                if _xy.shape[0] == 0:
                    continue
                for _i in range(_xy.shape[0]):
                    _p = [_xy[_i, 0], _xy[_i, 1], 1.5]
                    if _pts and _i == 0 and np.allclose(_pts[-1][:2], _p[:2]):
                        continue
                    _pts.append(_p)
            if len(_pts) < 2:
                return None
            _arr = np.asarray(_pts, dtype=float)
            # Drape onto terrain so routes stay visible above the lifted ground
            if self.scene_state.get("_terrain_drape_active"):
                try:
                    _dem = self.street_graph.graph.get("terrain_sampler")
                    if _dem is not None:
                        _arr[:, 2] += np.asarray(_dem(_arr[:, :2]), dtype=float)
                except Exception:
                    pass
            return _arr

    def _route_stats(self, _nodes: list[object], _uv_metrics: dict[tuple[object, object], dict[str, float]]) -> dict[str, float]:
            _dist = _time = _mech = _solar = _net = 0.0
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _m = _uv_metrics.get((_a, _b))
                if _m is None:
                    _edge_data = self._edge_best(self.street_graph, _a, _b, "length")
                    if _edge_data is not None:
                        _dist += float(_edge_data.get("length", 0.0))
                    continue
                _dist += float(_m.get("length_m", 0.0))
                _time += float(_m.get("travel_time_s", 0.0))
                _mech += float(_m.get("mechanical_J", 0.0))
                _solar += float(_m.get("solar_J", 0.0))
                _net += float(_m.get("net_energy_J", 0.0))
            return {
                "distance_m": _dist,
                "travel_time_s": _time,
                "mechanical_J": _mech,
                "solar_J": _solar,
                "net_energy_J": _net,
            }

    def _path_time_energy(self, _g: nx.MultiDiGraph, _nodes: list[object]) -> tuple[float, float]:
            _time = 0.0
            _energy = 0.0
            for _a, _b in zip(_nodes[:-1], _nodes[1:]):
                _tt = self._edge_best(_g, _a, _b, "travel_time_s")
                _ne = self._edge_best(_g, _a, _b, "net_energy_J")
                if _tt is not None:
                    _time += float(_tt.get("travel_time_s", 0.0))
                if _ne is not None:
                    _energy += float(_ne.get("net_energy_J", 0.0))
            return _time, _energy

    def _update_pareto_chart(self, _g: nx.MultiDiGraph, _src: object, _tgt: object) -> None:
            self._remove_pareto_chart()
            if not bool(self.scene_state.get("solar_fleet", False)):
                return
            _pareto = find_pareto_routes(_g, _src, _tgt, k=8)
            if len(_pareto) < 2:
                return

            _candidates: list[dict[str, object]] = []
            try:
                _gen = nx.shortest_simple_paths(_g, _src, _tgt, weight="travel_time_s")
                for _idx, _nodes in enumerate(_gen):
                    if _idx >= 8:
                        break
                    _t_s, _e_j = self._path_time_energy(_g, list(_nodes))
                    _candidates.append({
                        "idx": int(_idx),
                        "nodes": list(_nodes),
                        "time_s": float(_t_s),
                        "energy_J": float(_e_j),
                    })
            except nx.NetworkXNoPath:
                return

            if not _candidates:
                return

            _px = np.asarray([c["time_s"] / 60.0 for c in _candidates], dtype=float)
            _py = np.asarray([c["energy_J"] / 3600.0 for c in _candidates], dtype=float)

            _pareto_nodes = {tuple(r.get("nodes", [])) for r in _pareto}
            _p_idx = [int(c["idx"]) for c in _candidates if tuple(c["nodes"]) in _pareto_nodes]
            _p_x = np.asarray([_px[i] for i in _p_idx], dtype=float)
            _p_y = np.asarray([_py[i] for i in _p_idx], dtype=float)
            if _p_x.size < 2:
                return

            _chart = pv.Chart2D()
            _chart.title = "Route Pareto Front — time vs net energy"
            _chart.x_label = "Travel time (min)"
            _chart.y_label = "Net energy (Wh)"
            _chart.size = (260, 180)
            _chart.loc = (0.73, 0.02)

            _chart.scatter(_px, _py, color="#888888", size=6)
            _chart.scatter(_p_x, _p_y, color="#ffd54f", size=10)
            for _idx in _p_idx:
                try:
                    _chart.text(float(_px[_idx]), float(_py[_idx]), str(_idx), color="#ffd54f")
                except Exception:
                    pass

            try:
                self.plotter.add_chart(_chart)
                self.scene_state["pareto_chart"] = _chart
            except Exception as _exc:
                print(f"[route] pareto chart unavailable: {_exc}")

    def _line(self, _name: str, _s: dict[str, float]) -> str:
        _net = float(_s.get("net_energy_J", 0.0))
        _sign = "▼" if _net <= 0.0 else "▲"
        base = (
            f"{_name:<8}  d={_s.get('distance_m', 0.0):7.1f} m  "
            f"t={_s.get('travel_time_s', 0.0) / 60.0:6.2f} min  "
            f"mech={_s.get('mechanical_J', 0.0) / 3600.0:7.2f} Wh  "
        )
        if self._use_solar:
            return (
                f"{base}"
                f"solar={_s.get('solar_J', 0.0) / 3600.0:7.2f} Wh  "
                f"net={_net / 3600.0:7.2f} Wh {_sign}"
            )
        return f"{base}net={_net / 3600.0:7.2f} Wh {_sign}  (no solar)"

    def _update_route_stats_overlay(self, _summary: dict[str, dict[str, float]]) -> None:
            if not _summary:
                self.plotter.add_text(
                    "",
                    position=(0.55, 0.04),
                    name="route_stats_overlay",
                    font_size=8,
                    color="#f4f4f4",
                    viewport=True,
                )
                return

            self._use_solar = bool(self.scene_state.get("solar_fleet", False))

            txt = "\n".join([
                self._line("Energy", _summary.get("energy", {})),
                self._line("Joint", _summary.get("joint", {})),
                self._line("Shortest", _summary.get("shortest", {})),
            ])
            self.plotter.add_text(
                txt,
                position=(0.55, 0.04),
                name="route_stats_overlay",
                font_size=9,
                color="#ffffff",
                shadow=True,
                viewport=True,
            )

    def _cost_cache_key(
            self, _hour: float,
            _alpha: float,
            _params,
            _use_solar: bool,
    ) -> tuple[float, ...]:
            return (
                bool(_use_solar),
                round(float(_hour), 2),
                round(float(_alpha), 3),
                round(float(_params.roof_area_m2), 3),
                round(float(_params.panel_efficiency), 4),
                round(float(_params.temperature_derating), 4),
                round(float(_params.vehicle_mass_kg), 3),
                round(float(_params.rolling_coeff), 5),
                round(float(_params.drag_coeff), 5),
                round(float(_params.frontal_area_m2), 4),
            )

    def _compute_and_render_routes(self, _src: object, _tgt: object) -> None:
            _lat = float(self.scene_state.get("scene_lat", np.nan))
            _lon = float(self.scene_state.get("scene_lon", np.nan))
            if not np.isfinite(_lat) or not np.isfinite(_lon):
                print("[route] scene_lat/scene_lon unavailable; cannot build solar costs")
                return

            _alpha = float(self.scene_state.get("route_alpha", 0.5))
            _route_hour = float(self.scene_state.get("route_hour", self.scene_state.get("hour", 12.0)))
            _params_obj = self.scene_state.get("solar_params")
            from solar_physics import SolarParams
            _params = _params_obj if isinstance(_params_obj, SolarParams) else SolarParams()
            self._use_solar = bool(self.scene_state.get("solar_fleet", False))

            if self._use_solar:
                _shadow = self._build_edge_shadow_cache(_route_hour)
            else:
                _shadow = np.ones((len(self.car_paths),), dtype=float)

            _cost_cache = self.scene_state.get("edge_costs_cache")
            if not isinstance(_cost_cache, dict):
                _cost_cache = {}
                self.scene_state["edge_costs_cache"] = _cost_cache

            self.edge_shadow_frac = np.asarray(_shadow, dtype=float)
            # Include a shadow fingerprint so the cache is invalidated whenever
            # edge_shadow_frac changes (e.g. after the time-of-day slider moves).
            _shadow_fp = int(np.sum(self.edge_shadow_frac * 1000).round())  # cheap hash
            _costs_key = (
                round(float(self.scene_state.get("route_hour", _route_hour)), 2),
                round(float(self.scene_state.get("route_alpha", _alpha)), 3),
                _shadow_fp,
                # Network fingerprint: editor ops (bridges, reversals, roundabouts)
                # change car_paths — cached cost arrays would be mis-indexed.
                len(self.car_paths),
                self.street_graph.number_of_edges(),
            )
            _cached_costs = _cost_cache.get(_costs_key)
            if _cached_costs is None:
                _cached_costs = build_edge_costs(
                    car_paths=self.car_paths,
                    edge_shadow_frac=self.edge_shadow_frac,
                    lat_deg=_lat,
                    lon_deg=_lon,
                    hour_local=_route_hour,
                    params=_params,
                    alpha=_alpha,
                    use_solar=self._use_solar,
                )
                _cost_cache[_costs_key] = _cached_costs

            _g_cost = build_graph_with_costs(self.street_graph, self.car_paths, _cached_costs)

            # One-way streets can put the click targets in different strongly-
            # connected components — instead of "no path" ×3, remap the target
            # to the reachable node nearest to the click.
            if (_src in _g_cost and _tgt in _g_cost
                    and not nx.has_path(_g_cost, _src, _tgt)):
                _reach = nx.descendants(_g_cost, _src)
                if _reach:
                    _tx = float(self.street_graph.nodes[_tgt].get("x", np.nan))
                    _ty = float(self.street_graph.nodes[_tgt].get("y", np.nan))
                    _best, _best_d = None, float("inf")
                    for _n in _reach:
                        _nd = self.street_graph.nodes.get(_n, {})
                        if "x" not in _nd or "y" not in _nd:
                            continue
                        _dd = (float(_nd["x"]) - _tx) ** 2 + (float(_nd["y"]) - _ty) ** 2
                        if _dd < _best_d:
                            _best_d, _best = _dd, _n
                    if _best is not None and _best != _src:
                        print(f"[route] target node {_tgt} unreachable from {_src} "
                              f"(one-way network) — rerouted to nearest reachable "
                              f"node {_best} ({_best_d ** 0.5:.0f} m away)")
                        _tgt = _best
                        self.route_state["target_node"] = _tgt
                else:
                    print(f"[route] node {_src} has no outgoing connectivity — "
                          "pick a different source")

            _energy_nodes = find_energy_optimal_route(_g_cost, _src, _tgt)
            _joint_nodes = find_joint_optimal_route(_g_cost, _src, _tgt, alpha=_alpha)
            try:
                _short_nodes = list(nx.shortest_path(_g_cost, _src, _tgt, weight="length"))
            except nx.NetworkXNoPath:
                _short_nodes = []

            self._clear_route_actors(keep_markers=True)
            _route_specs = [
                ("energy", _energy_nodes, "#32cd32", "net_energy_J"),
                ("joint", _joint_nodes, "#ff9f1a", "combined_score"),
                ("shortest", _short_nodes, "#ffffff", "length"),
            ]
            for _name, _nodes, _color, _w in _route_specs:
                _pl = self._route_polyline(_g_cost, _nodes, _w)
                if _pl is None:
                    continue
                _actor = self.plotter.add_lines(_pl, color=_color, width=5, connected=True)
                self.route_state["route_actors"].append(_actor)

            _uv_metrics: dict[tuple[object, object], dict[str, float]] = {}
            for _i, _p in enumerate(self.car_paths):
                _k = (_p.get("u"), _p.get("v"))
                _row = {
                    "length_m": float(_p.get("length", 0.0)),
                    "travel_time_s": float(_cached_costs["travel_time_s"][_i]),
                    "mechanical_J": float(_cached_costs["mechanical_J"][_i]),
                    "solar_J": float(_cached_costs["solar_J"][_i]),
                    "net_energy_J": float(_cached_costs["net_energy_J"][_i]),
                }
                _old = _uv_metrics.get(_k)
                if _old is None or _row["net_energy_J"] < _old["net_energy_J"]:
                    _uv_metrics[_k] = _row

            _summary: dict[str, dict[str, float]] = {}
            for _name, _nodes, _color, _w in _route_specs:
                if len(_nodes) < 2:
                    print(f"[route] {_name}: no path")
                    _summary[_name] = {
                        "distance_m": 0.0,
                        "travel_time_s": 0.0,
                        "mechanical_J": 0.0,
                        "solar_J": 0.0,
                        "net_energy_J": 0.0,
                    }
                    continue
                _s = self._route_stats(_nodes, _uv_metrics)
                _summary[_name] = _s
                print(
                    f"[route] {_name}: distance={_s['distance_m']:.1f}m, "
                    f"time={_s['travel_time_s']:.1f}s, mech={_s['mechanical_J']:.1f}J, "
                    f"solar={_s['solar_J']:.1f}J, net={_s['net_energy_J']:.1f}J"
                )

            # Once per call, with the FULLY populated summary — these three
            # used to be indented inside the loop above and ran (redundantly,
            # on partial data) once per route spec instead of once overall.
            self._update_route_stats_overlay(_summary)
            if self.route_state.get("selected_car_idx") is not None and len(_joint_nodes) >= 2:
                self._assign_selected_car_route(_joint_nodes)
            self._update_pareto_chart(_g_cost, _src, _tgt)

    def _road_pick_callback(self, point) -> None:
            if point is None or len(point) < 2:
                return
            px, py = float(point[0]), float(point[1])
            pt = _SPoint(px, py)
            best = min(self._road_edge_data, key=lambda e: e["geom"].distance(pt), default=None)
            if best is None or best["geom"].distance(pt) > 20.0:
                return
            hw    = best["highway"] or "unclassified"
            speed = best["maxspeed"]
            lanes = best["lanes"]
            surf  = best["surface"] or "unknown"
            ow    = "one-way" if best["oneway"] else "two-way"
            wid   = f"{best['width_m']:.1f} m" if best["width_m"] else "unknown"
            spd_s = f"{speed:.0f} km/h" if speed else "no data"
            info = (
                f"Road type:   {hw}\n"
                f"Direction:   {ow}\n"
                f"Speed limit: {spd_s}\n"
                f"Lanes:       {lanes if lanes else 'unknown'}\n"
                f"Width:       {wid}\n"
                f"Surface:     {surf}"
            )
            self.plotter.add_text(
                info,
                position=(0.68, 0.90),
                name="road_info_overlay",
                font_size=9,
                color="white",
                font="courier",
            )

            # Click route workflow: 1st click = source, 2nd = target, 3rd = reset.
            try:
                nearest_geom = best["geom"].interpolate(best["geom"].project(pt))
                _snap_node = nearest_graph_node(self.street_graph, nearest_geom.x, nearest_geom.y)
                if self.route_state["stage"] == 1 and _snap_node == self.route_state["source_node"]:
                    _nodes_sorted = sorted(
                        self.street_graph.nodes(data=True),
                        key=lambda n: (float(n[1]["x"]) - nearest_geom.x)**2 + (float(n[1]["y"]) - nearest_geom.y)**2
                    )
                    for n_id, _ in _nodes_sorted:
                        if n_id != self.route_state["source_node"]:
                            _snap_node = n_id
                            print(f"[route] snapped to second-nearest node {_snap_node} (source node was the closest)")
                            break
            except Exception as _exc:
                print(f"[route] Node snap failed: {_exc}")
                return

            if self.route_state["stage"] == 2:
                self._clear_route_actors()
                self._remove_pareto_chart()
                self._clear_selected_car(clear_plan=True)
                self.route_state["stage"] = 0
                self.route_state["source_node"] = None
                self.route_state["target_node"] = None
                self._update_route_stats_overlay({})
                print("[route] Route selection reset — click again for a new source")
                return

            _xy = self._node_xy(_snap_node)
            if _xy is None:
                print(f"[route] Node {_snap_node} has no local coordinates")
                return

            # Marker z follows the terrain when draping is active
            _mz = 2.0
            if self.scene_state.get("_terrain_drape_active"):
                try:
                    _dem = self.street_graph.graph.get("terrain_sampler")
                    if _dem is not None:
                        _mz += float(np.asarray(_dem(np.array([[_xy[0], _xy[1]]])), dtype=float)[0])
                except Exception:
                    pass

            if self.route_state["stage"] == 0:
                self._clear_route_actors()
                self._clear_selected_car(clear_plan=True)
                self.route_state["source_node"] = _snap_node
                _src_actor = self.plotter.add_mesh(
                    pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], _mz), theta_resolution=18, phi_resolution=18),
                    color="#33cc66",
                    render=False,
                )
                self.route_state["route_actors"] = [_src_actor]
                self.route_state["stage"] = 1
                print(f"[route] Source set: node {_snap_node} — click target")
                return

            self.route_state["target_node"] = _snap_node
            _tgt_actor = self.plotter.add_mesh(
                pv.Sphere(radius=2.0, center=(_xy[0], _xy[1], _mz), theta_resolution=18, phi_resolution=18),
                color="#ff4d4d",
                render=False,
            )
            self.route_state.setdefault("route_actors", []).append(_tgt_actor)
            print(f"[route] Target set: node {_snap_node}")
            self._compute_and_render_routes(self.route_state["source_node"], self.route_state["target_node"])

            self.route_state["stage"] = 2

    def _rebuild_traffic_and_arrows(self) -> None:
        """Re-extracts paths, redraws arrows, and resets IDM state without replacing ground mesh."""
        from app_core import _street_direction_arrows
        self._stage("Rebuilding traffic and arrows...")
        self.car_paths, self.car_outgoing = self._extract_drivable_paths(self.street_graph, z_level=0.5)
        self.car_path_lengths = np.asarray([float(p["length"]) for p in self.car_paths], dtype=float)

        # demand_mixin._route_to_edges looks up (u,v) -> car_paths INDEX via
        # this map. car_paths was just rebuilt from scratch above (edges
        # inserted/removed/reordered by whatever editor op triggered this
        # rebuild), so the old map's indices now point at wrong-or-nonexistent
        # edges in the new list — every demand-routed trip planned afterwards
        # would silently follow a corrupted route. Only demand_mixin.
        # _init_demand builds this map in the first place, so only rebuild it
        # here if demand is actually in use.
        if getattr(self, "demand", None) is not None:
            self._demand_edge_map = {}
            for _i, _p in enumerate(self.car_paths):
                self._demand_edge_map.setdefault((_p.get("u"), _p.get("v")), _i)
        self.adj_left_paths  = np.array([p.get("adjacent_left_path",  -1) for p in self.car_paths], dtype=np.int64)
        self.adj_right_paths = np.array([p.get("adjacent_right_path", -1) for p in self.car_paths], dtype=np.int64)

        self.car_next_edges = _build_next_edges(
            self.car_paths,
            self.car_outgoing,
            self.street_graph,
            include_overture=False,
        )
        self.roundabout_yield_map = {}
        for p_idx, path in enumerate(self.car_paths):
            if path.get("junction") != "roundabout":
                for next_p_idx_raw in self.car_next_edges[p_idx]:
                    next_p = int(next_p_idx_raw)
                    if self.car_paths[next_p].get("junction") == "roundabout":
                        ring_paths = []
                        for rp_idx, rp in enumerate(self.car_paths):
                            if rp["v"] == path["v"] and rp.get("junction") == "roundabout":
                                ring_paths.append(rp_idx)
                        if ring_paths:
                            self.roundabout_yield_map[p_idx] = np.array(ring_paths, dtype=np.int64)

        if self.scene_state.get("street_arrows_actor") is not None:
            self.plotter.remove_actor(self.scene_state["street_arrows_actor"])
        arrows = _street_direction_arrows(self.street_graph)
        if arrows is not None:
            self.scene_state["street_arrows_actor"] = self.plotter.add_mesh(
                arrows, color="#00e5ff", opacity=0.7, smooth_shading=False, lighting=False, render=False
            )
            self._set_actor_visibility(self.scene_state["street_arrows_actor"], self.scene_state.get("show_arrows", True))

        # Render Stop Signs
        if self.scene_state.get("stop_signs_actor") is not None:
            self.plotter.remove_actor(self.scene_state["stop_signs_actor"])
        stop_nodes = [n for n, d in self.street_graph.nodes(data=True) if d.get("highway") == "stop"]
        if stop_nodes:
            stop_meshes = []
            for n in stop_nodes:
                ndata = self.street_graph.nodes[n]
                _snx, _sny = float(ndata.get("x", 0)), float(ndata.get("y", 0))
                # Red octagon (8-sided cylinder)
                cyl = pv.Cylinder(center=(_snx, _sny, 1.5), direction=(0, 0, 1), radius=1.2, height=0.3, resolution=8)
                stop_meshes.append(cyl)

            merged_stops = stop_meshes[0]
            for i in range(1, len(stop_meshes)):
                merged_stops = merged_stops.merge(stop_meshes[i])
            self.scene_state["stop_signs_actor"] = self.plotter.add_mesh(
                merged_stops, color="#cc0000", smooth_shading=False, lighting=True, render=False
            )
        else:
            self.scene_state["stop_signs_actor"] = None

        # Rebuild traffic lights from the updated graph so path indices stay valid
        # (pv is the module-level import — a local `import pyvista as pv` here
        #  would shadow it for the WHOLE function and crash line ~490)
        from traffic_lights import build_traffic_lights, build_light_mesh, build_light_glyphs
        self.traffic_lights_dict = build_traffic_lights(
            self.street_graph,
            self.car_paths,
            min_degree=3,
            traffic_speed=float(self.args.traffic_speed),
        )
        self.scene_state["traffic_lights"] = self.traffic_lights_dict
        self._tl_mesh = build_light_mesh(self.traffic_lights_dict)
        self.scene_state["_tl_mesh"] = self._tl_mesh
        if self._tl_mesh.n_points > 0:
            _tl_sphere = pv.Sphere(radius=1.4, theta_resolution=10, phi_resolution=10)
            _tl_glyphs = build_light_glyphs(self._tl_mesh, _tl_sphere)
            self.scene_state["_tl_glyphs"] = _tl_glyphs
            if self.scene_state.get("tl_actor"):
                self.plotter.remove_actor(self.scene_state["tl_actor"], reset_camera=False)
            self.scene_state["tl_actor"] = self.plotter.add_mesh(
                _tl_glyphs, scalars="colors", rgb=True,
                smooth_shading=True, pbr=True, metallic=0.1, roughness=0.4,
                lighting=True, render=False,
            )
            self._set_actor_visibility(
                self.scene_state["tl_actor"],
                bool(self.scene_state.get("show_traffic_signals", True))
            )

        # Redistribute cars across the new path set instead of stacking at edge 0
        n = len(self.car_anim["edge_idx"])
        if n > 0 and len(self.car_paths) > 0:
            new_ei = self.car_rng.choice(len(self.car_paths), size=n, replace=True).astype(np.int64)
            new_dist = np.array([
                float(self.car_rng.uniform(0.0, float(self.car_path_lengths[int(e)])))
                for e in new_ei
            ], dtype=float)
            new_base = np.array([
                self.car_paths[int(e)]["maxspeed_ms"] * float(self.car_rng.uniform(0.7, 1.0))
                for e in new_ei
            ], dtype=float)
            self.car_anim["edge_idx"] = new_ei
            self.car_anim["dist"] = new_dist
            self.car_anim["speed"] = (new_base * float(self.args.traffic_speed)).copy()
            self.car_anim["desired_speed_base"] = new_base.copy()
            self.car_anim["desired_speed"] = self.car_anim["speed"].copy()
        else:
            self.car_anim["edge_idx"] = np.zeros(n, dtype=np.int64)
            self.car_anim["dist"] = np.zeros(n, dtype=float)
            self.car_anim["speed"] = np.zeros(n, dtype=float)
        self.car_anim["stop_wait"] = np.zeros(n, dtype=float)
        self.car_anim["planned_edges"] = [None for _ in range(n)]
        self.car_anim["planned_cursor"] = np.zeros(n, dtype=np.int64)
        if n > 0:
            _hrng = self.car_rng
            self.car_anim["idm_T_arr"] = np.clip(_hrng.normal(1.4, 0.3, n), 0.8, 2.5).astype(float)
            self.car_anim["idm_a_arr"] = np.clip(_hrng.normal(1.6, 0.4, n), 0.9, 2.8).astype(float)
            self.car_anim["idm_b_arr"] = np.clip(_hrng.normal(2.0, 0.4, n), 1.0, 3.5).astype(float)
            self.car_anim["idm_speed_factor"] = np.clip(_hrng.normal(1.0, 0.12, n), 0.7, 1.3).astype(float)

        # Terrain drape active → path set changed, so height profiles are stale
        # (otherwise every car tick falls back to slow live-DEM lookups).
        if self.scene_state.get("_terrain_drape_active"):
            try:
                _dem = self.street_graph.graph.get("terrain_sampler")
                if _dem is not None and self.car_paths:
                    from terrain_drape import precompute_path_heights
                    _prof, _lens, _counts = precompute_path_heights(
                        self.car_paths, self._car_pose_on_path, _dem, spacing_m=5.0,
                    )
                    self.scene_state["_car_h_profiles"] = _prof
                    self.scene_state["_car_h_lens"]     = _lens
                    self.scene_state["_car_h_counts"]   = _counts
                    print(f"[terrain] height profiles refreshed for {len(self.car_paths)} paths")
            except Exception as _exc:
                print(f"[terrain] profile refresh failed: {_exc}")

        self.plotter.render()
        print("[editor] Traffic logic and arrows rebuilt successfully")
