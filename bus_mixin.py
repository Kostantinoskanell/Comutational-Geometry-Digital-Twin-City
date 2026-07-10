from __future__ import annotations
import csv
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

from ped_mixin import _load_glb_as_polydata

try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False


# ---------------------------------------------------------------------------
# GTFS helpers
# ---------------------------------------------------------------------------

def _gtfs_seconds(t: str) -> float:
    parts = t.strip().split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    except ValueError:
        return 0.0


def _csv_from_gtfs(gtfs_root, filename: str) -> list[dict]:
    p = Path(str(gtfs_root))
    if p.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(p, "r") as zf:
                match = next((n for n in zf.namelist() if n.endswith(filename)), None)
                if match is None:
                    return []
                data = zf.read(match).decode("utf-8-sig")
        except Exception:
            return []
    else:
        f = p / filename
        if not f.exists():
            return []
        data = f.read_text(encoding="utf-8-sig")
    try:
        return list(csv.DictReader(io.StringIO(data)))
    except Exception:
        return []


_MODELS_DIR = Path(__file__).parent / "assets" / "models"


def _build_bus_mesh() -> pv.PolyData | None:
    """Load low_poly_bus.obj, scaled to 12 m length along X."""
    geo = _load_glb_as_polydata(_MODELS_DIR / "low_poly_bus.obj")
    if geo is None:
        return pv.Box(bounds=(-6.0, 6.0, -1.275, 1.275, 0.0, 3.3))  # fallback
    x_len = float(geo.bounds[1] - geo.bounds[0])
    if x_len < 1e-6:
        return None
    scale = 12.0 / x_len
    geo.points *= scale
    b = geo.bounds
    cx = (b[0] + b[1]) / 2.0
    cy = (b[2] + b[3]) / 2.0
    cz = float(b[4])
    geo.points -= np.array([cx, cy, cz], dtype=float)
    b = geo.bounds
    print(f"[buses] bus mesh: L={b[1]-b[0]:.1f}m W={b[3]-b[2]:.1f}m H={b[5]-b[4]:.1f}m")
    return geo


def _build_bus_stop_mesh() -> pv.PolyData | None:
    """Load low_poly_bus_stop.obj, scaled to 2.8 m height."""
    geo = _load_glb_as_polydata(_MODELS_DIR / "low_poly_bus_stop.obj")
    if geo is None:
        return pv.Box(bounds=(-1.5, 1.5, -0.5, 0.5, 0.0, 2.8))  # fallback
    z_h = float(geo.bounds[5] - geo.bounds[4])
    if z_h < 1e-6:
        return None
    scale = 2.8 / z_h
    geo.points *= scale
    b = geo.bounds
    # Centre XY but keep feet at Z=0
    cx = (b[0] + b[1]) / 2.0
    cy = (b[2] + b[3]) / 2.0
    geo.points -= np.array([cx, cy, float(b[4])], dtype=float)
    return geo


def _load_gtfs_buses(gtfs_path, graph, proj_str: str, max_buses: int, rng
                     ) -> tuple[list[dict], list[np.ndarray]]:
    if not _NX_OK:
        print("[buses] networkx not available — buses disabled")
        return [], []

    routes_rows     = _csv_from_gtfs(gtfs_path, "routes.txt")
    trips_rows      = _csv_from_gtfs(gtfs_path, "trips.txt")
    stop_times_rows = _csv_from_gtfs(gtfs_path, "stop_times.txt")
    stops_rows      = _csv_from_gtfs(gtfs_path, "stops.txt")

    if not routes_rows or not trips_rows or not stop_times_rows or not stops_rows:
        print("[buses] GTFS files missing or empty — buses disabled")
        return [], []

    # Project stops to local CRS
    try:
        from pyproj import Transformer
        _to_local = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    except Exception as exc:
        print(f"[buses] CRS projection failed: {exc} — buses disabled")
        return [], []

    stop_xy: dict[str, tuple[float, float]] = {}
    for row in stops_rows:
        sid = row.get("stop_id", "").strip()
        try:
            lx, ly = _to_local.transform(float(row["stop_lon"]), float(row["stop_lat"]))
            stop_xy[sid] = (lx, ly)
        except (KeyError, ValueError, Exception):
            continue

    if not stop_xy:
        print("[buses] no valid stops — buses disabled")
        return [], []

    # Match stops to nearest OSM node
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[buses] scipy not available — buses disabled")
        return [], []

    nodes = list(graph.nodes(data=True))
    node_ids = np.array([n for n, _ in nodes])
    node_xy  = np.array(
        [[float(d.get("x", 0.0)), float(d.get("y", 0.0))] for _, d in nodes],
        dtype=float,
    )
    node_tree = cKDTree(node_xy)

    stop_node: dict[str, int] = {}
    for sid, (sx, sy) in stop_xy.items():
        dist_nn, idx = node_tree.query([sx, sy])
        if dist_nn < 300.0:
            stop_node[sid] = int(node_ids[idx])

    if not stop_node:
        print("[buses] no GTFS stops matched to road nodes — buses disabled")
        return [], []

    # Filter to bus routes (route_type = 3)
    bus_route_ids = {
        row["route_id"].strip()
        for row in routes_rows
        if str(row.get("route_type", "")).strip() == "3"
    }
    if not bus_route_ids:
        bus_route_ids = {row["route_id"].strip() for row in routes_rows}

    trip_route: dict[str, str] = {
        row["trip_id"].strip(): row["route_id"].strip()
        for row in trips_rows
        if row.get("route_id", "").strip() in bus_route_ids
    }

    # Build stop sequences per trip
    trip_stops: dict[str, list] = {}
    for row in stop_times_rows:
        tid = row.get("trip_id", "").strip()
        if tid not in trip_route:
            continue
        sid = row.get("stop_id", "").strip()
        if sid not in stop_node:
            continue
        try:
            seq   = int(row.get("stop_sequence", 0))
            arr   = _gtfs_seconds(row.get("arrival_time",   "00:00:00"))
            dep   = _gtfs_seconds(row.get("departure_time", "00:00:00"))
        except ValueError:
            continue
        trip_stops.setdefault(tid, []).append((seq, sid, max(0.0, dep - arr)))

    for tid in trip_stops:
        trip_stops[tid].sort(key=lambda x: x[0])

    valid_trips = [(tid, s) for tid, s in trip_stops.items() if len(s) >= 2]
    if not valid_trips:
        print("[buses] no valid bus trips — buses disabled")
        return [], []

    rng.shuffle(valid_trips)
    selected = valid_trips[:max_buses]

    dem_sampler = graph.graph.get("terrain_sampler")
    buses = []

    for tid, stops in selected:
        route_xy_parts: list[np.ndarray] = []
        stop_dists: list[tuple[float, float]] = []
        current_len = 0.0
        ok = True

        for k in range(len(stops)):
            if k == 0:
                # Mark first stop at dist 0
                stop_dists.append((0.0, float(stops[0][2])))
                continue
            _, sid_a, _      = stops[k - 1]
            _, sid_b, dwell_b = stops[k]
            na = stop_node[sid_a]
            nb = stop_node[sid_b]
            if na == nb:
                stop_dists.append((current_len, dwell_b))
                continue
            try:
                path_nodes = nx.shortest_path(graph, na, nb, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                ok = False
                break

            seg_xy = np.array(
                [[float(graph.nodes[n].get("x", 0.0)), float(graph.nodes[n].get("y", 0.0))]
                 for n in path_nodes],
                dtype=float,
            )
            keep = np.ones(seg_xy.shape[0], dtype=bool)
            keep[1:] = np.linalg.norm(seg_xy[1:] - seg_xy[:-1], axis=1) > 1e-6
            seg_xy = seg_xy[keep]
            if seg_xy.shape[0] < 2:
                continue

            route_xy_parts.append(seg_xy)
            seg_len = float(np.sum(np.linalg.norm(seg_xy[1:] - seg_xy[:-1], axis=1)))
            current_len += seg_len
            stop_dists.append((current_len, dwell_b))

        if not ok or not route_xy_parts or current_len < 20.0:
            continue

        full_xy = np.vstack(route_xy_parts)
        keep = np.ones(full_xy.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(full_xy[1:] - full_xy[:-1], axis=1) > 1e-6
        full_xy = full_xy[keep]
        if full_xy.shape[0] < 2:
            continue

        z_col = np.zeros(full_xy.shape[0], dtype=float)
        if dem_sampler is not None:
            try:
                z_col = dem_sampler(full_xy).astype(float)
            except Exception:
                pass

        route_pts = np.column_stack((full_xy, z_col))
        seg_lens  = np.linalg.norm(route_pts[1:, :2] - route_pts[:-1, :2], axis=1)
        route_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

        start_dist = float(rng.uniform(0.0, max(float(route_cum[-1]) - 1.0, 0.0)))

        # First stop AHEAD of the random start position — starting the cursor
        # at 0 (a stop at dist 0.0) made dist_to_stop negative on the first
        # tick, and the "arrived" branch teleported the bus back to the start.
        _cursor0 = 0
        for _si, (_sd, _dw) in enumerate(stop_dists):
            if _sd > start_dist:
                _cursor0 = _si
                break

        buses.append({
            "route_pts":   route_pts,
            "route_cum":   route_cum,
            "dist":        start_dist,
            "speed":       0.0,
            "dwell_timer": 0.0,
            "stop_dists":  stop_dists,
            "stop_cursor": _cursor0,
            "v0":          float(rng.uniform(6.5, 8.5)),
            "bus_len":     12.0,
            "route_name":  trip_route.get(tid, "?"),
            "trip_id":     tid,
        })

    # Collect unique matched stop positions for bus-stop shelter placement
    dem_sampler = graph.graph.get("terrain_sampler")
    seen_nodes: set[int] = set()
    stop_positions: list[np.ndarray] = []
    for sid, node_id in stop_node.items():
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nd = graph.nodes.get(node_id, {})
        sx, sy = float(nd.get("x", 0.0)), float(nd.get("y", 0.0))
        sz = 0.0
        if dem_sampler is not None:
            try:
                sz = float(dem_sampler(np.array([[sx, sy]]))[0])
            except Exception:
                pass
        stop_positions.append(np.array([sx, sy, sz], dtype=float))

    print(f"[buses] loaded {len(buses)} buses, {len(stop_positions)} stops from {len(valid_trips)} GTFS trips")
    return buses, stop_positions


# ---------------------------------------------------------------------------
# BusMixin
# ---------------------------------------------------------------------------

class BusMixin:

    def _init_buses(self, gtfs_path=None, n_buses: int = -1) -> None:
        self.buses: list[dict] = []
        self.bus_mesh: pv.PolyData | None = None
        self.bus_stop_mesh: pv.PolyData | None = None
        self.bus_stop_positions: list[np.ndarray] = []

        if not gtfs_path:
            print("[buses] no GTFS path provided — buses disabled")
            return

        if not _NX_OK:
            print("[buses] networkx not available — buses disabled")
            return

        seed = int(getattr(self.args, "seed", 42))
        self.bus_rng = np.random.default_rng(seed + 9371)

        # Derive proj_str
        proj_str = self.street_graph.graph.get("proj_str", "")
        if not proj_str:
            lat = float(getattr(self.args, "lat", 0.0))
            lon = float(getattr(self.args, "lon", 0.0))
            proj_str = (
                f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )

        max_buses = 30 if n_buses == -1 else max(0, n_buses)
        self.buses, self.bus_stop_positions = _load_gtfs_buses(
            gtfs_path, self.street_graph, proj_str, max_buses, self.bus_rng
        )
        if n_buses > 0 and len(self.buses) > n_buses:
            self.buses = self.buses[:n_buses]

        if not self.buses:
            return

        self.bus_mesh      = _build_bus_mesh()
        self.bus_stop_mesh = _build_bus_stop_mesh()
        print(f"[buses] {len(self.buses)} buses ready")

    # ------------------------------------------------------------------
    # Physics tick
    # ------------------------------------------------------------------

    def _advance_buses(self, dt: float) -> None:
        for bus in self.buses:
            route_len = float(bus["route_cum"][-1]) if len(bus["route_cum"]) else 0.0
            if route_len < 1.0:
                continue

            # Dwell at stop
            if bus["dwell_timer"] > 0.0:
                bus["dwell_timer"] = max(0.0, bus["dwell_timer"] - dt)
                bus["speed"] = 0.0
                continue

            stop_dists = bus["stop_dists"]
            cursor     = int(bus["stop_cursor"])

            # Find next stop ahead
            next_stop = None
            if stop_dists:
                cursor = cursor % len(stop_dists)
                next_stop = stop_dists[cursor]

            if next_stop is not None:
                dist_to_stop = float(next_stop[0]) - float(bus["dist"])
                if dist_to_stop < -2.0:
                    # Stop is far BEHIND (cursor wrapped past the last stop
                    # while the bus still heads to the route end, or a missed
                    # stop) — skip it instead of teleporting backwards.
                    bus["stop_cursor"] = (cursor + 1) % len(stop_dists)
                    continue
                if dist_to_stop < 1.0:
                    # Arrived at stop (snap ≤2 m is imperceptible)
                    bus["dist"]        = float(next_stop[0])
                    bus["dwell_timer"] = float(next_stop[1])
                    bus["stop_cursor"] = (cursor + 1) % len(stop_dists)
                    bus["speed"]       = 0.0
                    continue
                # Braking profile (b = 2 m/s²)
                v = float(bus["speed"])
                braking_dist = (v * v) / (2.0 * 2.0)
                if dist_to_stop < braking_dist + 2.0:
                    v_target = float(np.sqrt(max(0.0, 2.0 * 2.0 * max(0.0, dist_to_stop))))
                    bus["speed"] = max(0.0, min(v, v_target))
                else:
                    bus["speed"] = min(float(bus["v0"]), float(bus["speed"]) + 1.0 * dt)
            else:
                bus["speed"] = min(float(bus["v0"]), float(bus["speed"]) + 1.0 * dt)

            bus["dist"] = float(bus["dist"]) + float(bus["speed"]) * dt

            # Loop route
            if bus["dist"] >= route_len:
                bus["dist"]        = 0.0
                bus["stop_cursor"] = 0

    # ------------------------------------------------------------------
    # Position & heading helpers
    # ------------------------------------------------------------------

    def _sample_bus_positions(self) -> np.ndarray:
        n = len(self.buses)
        positions = np.zeros((n, 3), dtype=float)
        for i, bus in enumerate(self.buses):
            pts = bus["route_pts"]
            cum = bus["route_cum"]
            total = float(cum[-1]) if len(cum) else 0.0
            if total < 1e-6 or pts.shape[0] < 2:
                continue
            d = float(np.clip(bus["dist"], 0.0, max(total - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            t   = (d - float(cum[seg])) / max(1e-9, float(cum[seg + 1]) - float(cum[seg]))
            positions[i] = pts[seg] + (pts[seg + 1] - pts[seg]) * t
        # route_pts z carries the DEM elevation baked at load time — only use
        # it while terrain draping is active, else buses float above the city
        if not self.scene_state.get("_terrain_drape_active"):
            positions[:, 2] = 0.0
        return positions

    def _bus_headings_deg(self) -> np.ndarray:
        n = len(self.buses)
        headings = np.zeros(n, dtype=float)
        for i, bus in enumerate(self.buses):
            pts = bus["route_pts"]
            cum = bus["route_cum"]
            total = float(cum[-1]) if len(cum) else 0.0
            if total < 1e-6 or pts.shape[0] < 2:
                continue
            d = float(np.clip(bus["dist"], 0.0, max(total - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            p0, p1 = pts[seg], pts[seg + 1]
            h = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
            headings[i] = (h + 180.0) % 360.0
        return headings

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_buses(self) -> None:
        if not self.buses or self.bus_mesh is None:
            return

        positions = self._sample_bus_positions()
        headings  = self._bus_headings_deg()
        n = len(self.buses)

        actors = self.scene_state.get("_bus_vtk_actors")

        if not actors:
            actors = []
            for i in range(n):
                try:
                    actor = self.plotter.add_mesh(
                        self.bus_mesh.copy(),
                        color="#1565C0",
                        smooth_shading=True,
                        lighting=True,
                        reset_camera=False,
                    )
                    actor.SetPosition(
                        float(positions[i, 0]),
                        float(positions[i, 1]),
                        float(positions[i, 2]),
                    )
                    actor.SetOrientation(0.0, 0.0, float(headings[i]))
                    actors.append(actor)
                except Exception as exc:
                    print(f"[buses] actor {i} failed: {exc}")
            self.scene_state["_bus_vtk_actors"]       = actors
            self.scene_state["_bus_last_headings"]    = np.full(n, np.nan)
            print(f"[buses] {len(actors)} actors created")
        else:
            _last_h = self.scene_state.get("_bus_last_headings")
            if not isinstance(_last_h, np.ndarray) or _last_h.shape[0] != n:
                _last_h = np.full(n, np.nan)
            for i, actor in enumerate(actors):
                if i >= n:
                    break
                try:
                    actor.SetPosition(
                        float(positions[i, 0]),
                        float(positions[i, 1]),
                        float(positions[i, 2]),
                    )
                    h = float(headings[i])
                    if not np.isfinite(_last_h[i]) or abs(h - float(_last_h[i])) > 3.0:
                        actor.SetOrientation(0.0, 0.0, h)
                        _last_h[i] = h
                except Exception:
                    pass
            self.scene_state["_bus_last_headings"] = _last_h

    # ------------------------------------------------------------------
    # Bus stop rendering (static, one-time placement)
    # ------------------------------------------------------------------

    def _render_bus_stops(self) -> None:
        if self.scene_state.get("_bus_stops_rendered"):
            return
        stops = getattr(self, "bus_stop_positions", [])
        mesh  = getattr(self, "bus_stop_mesh", None)
        if not stops or mesh is None:
            self.scene_state["_bus_stops_rendered"] = True
            return

        placed = 0
        _stop_actors = []
        for pos in stops:
            try:
                actor = self.plotter.add_mesh(
                    mesh.copy(),
                    color="#607D8B",
                    smooth_shading=True,
                    lighting=True,
                    reset_camera=False,
                )
                # Placed FLAT — pos[2] is DEM-baked; the terrain drape toggle
                # lifts/restores these actors (like parked cars)
                actor.SetPosition(float(pos[0]), float(pos[1]), 0.0)
                # Hidden until the user presses 'b'
                if not self.scene_state.get("_transit_overlay_visible", False):
                    actor.VisibilityOff()
                _stop_actors.append(actor)
                placed += 1
            except Exception:
                pass

        self.scene_state["_bus_stop_actors"] = _stop_actors
        self.scene_state["_bus_stops_rendered"] = True
        print(f"[buses] {placed} bus stop shelters placed (press 'b' to show)")

    # ------------------------------------------------------------------
    # Transit overlay — route polylines
    # ------------------------------------------------------------------

    _ROUTE_COLORS = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00",
        "#8E24AA", "#00ACC1", "#E91E63", "#7CB342",
    ]

    def _render_transit_routes(self) -> None:
        """Draw a coloured polyline for each unique bus route (called once)."""
        if self.scene_state.get("_transit_routes_rendered"):
            return
        self.scene_state["_transit_routes_rendered"] = True

        if not self.buses:
            return

        seen: set[str] = set()
        _route_actors = []
        for bus in self.buses:
            rname = str(bus.get("route_name", "?"))
            if rname in seen:
                continue
            seen.add(rname)

            pts = bus["route_pts"].copy()
            # float 0.4 m above local ground / road; route z is DEM-baked so
            # only keep it when terrain draping is active
            if not self.scene_state.get("_terrain_drape_active"):
                pts[:, 2] = 0.0
            pts[:, 2] = np.maximum(pts[:, 2] + 0.4, 0.4)
            n = len(pts)
            if n < 2:
                continue

            color = self._ROUTE_COLORS[hash(rname) % len(self._ROUTE_COLORS)]
            conn = np.empty(n + 1, dtype=np.int64)
            conn[0] = n
            conn[1:] = np.arange(n, dtype=np.int64)
            mesh = pv.PolyData(pts)
            mesh.lines = conn
            try:
                actor = self.plotter.add_mesh(
                    mesh, color=color, line_width=4.0,
                    opacity=0.85, lighting=False, reset_camera=False,
                    render_lines_as_tubes=True,
                )
                _route_actors.append(actor)
            except Exception:
                pass

        # Hidden until the user presses 'b'
        if not self.scene_state.get("_transit_overlay_visible", False):
            for _a in _route_actors:
                try:
                    _a.VisibilityOff()
                except Exception:
                    pass

        self.scene_state["_transit_route_actors"] = _route_actors
        if _route_actors:
            print(f"[transit] {len(_route_actors)} route polylines drawn (press 'b' to show)")

    def _toggle_transit_overlay(self) -> None:
        """'b' — show/hide bus routes + stop shelters together."""
        # Render on first use (both are lazy one-time builders)
        self._render_transit_routes()
        self._render_bus_stops()

        visible = not bool(self.scene_state.get("_transit_overlay_visible", False))
        self.scene_state["_transit_overlay_visible"] = visible

        _routes = self.scene_state.get("_transit_route_actors") or []
        _stops  = self.scene_state.get("_bus_stop_actors") or []
        for actor in list(_routes) + list(_stops):
            try:
                if visible:
                    actor.VisibilityOn()
                else:
                    actor.VisibilityOff()
            except Exception:
                pass
        n_r, n_s = len(_routes), len(_stops)
        if n_r == 0 and n_s == 0:
            print("[transit] no bus routes/stops in this scene (needs --gtfs data)")
        else:
            print(f"[transit] overlay {'ON' if visible else 'OFF'} — "
                  f"{n_r} routes, {n_s} stop shelters")
        try:
            self.plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Transit overlay — ETA labels
    # ------------------------------------------------------------------

    def _bus_stop_world_xy(self, bus: dict) -> np.ndarray | None:
        """(N_stops, 2) local-CRS XY positions of stops on this bus's route."""
        pts = bus["route_pts"]
        cum = bus["route_cum"]
        if len(cum) == 0 or float(cum[-1]) < 1.0:
            return None
        stop_dists = bus.get("stop_dists", [])
        if not stop_dists:
            return None
        out = []
        for dist, _ in stop_dists:
            d = float(np.clip(dist, 0.0, float(cum[-1])))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1,
                               0, len(pts) - 2))
            t = (d - float(cum[seg])) / max(1e-9, float(cum[seg + 1]) - float(cum[seg]))
            pos = pts[seg] + (pts[seg + 1] - pts[seg]) * t
            out.append(pos[:2])
        return np.array(out, dtype=float)

    def _precompute_stop_bus_map(self) -> None:
        """Build stop_idx → [(bus_idx, route_stop_dist_idx)] for ETA queries."""
        if self.scene_state.get("_transit_stop_map") is not None:
            return
        if not self.buses or not self.bus_stop_positions:
            self.scene_state["_transit_stop_map"] = {}
            return
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self.scene_state["_transit_stop_map"] = {}
            return

        stop_arr  = np.array([p[:2] for p in self.bus_stop_positions], dtype=float)
        stop_tree = cKDTree(stop_arr)

        stop_map: dict[int, list] = {}
        for bus_idx, bus in enumerate(self.buses):
            bsxy = self._bus_stop_world_xy(bus)
            if bsxy is None or len(bsxy) == 0:
                continue
            for sd_idx, bs in enumerate(bsxy):
                dist, idx = stop_tree.query(bs, k=1)
                if float(dist) <= 300.0:
                    stop_map.setdefault(int(idx), []).append((bus_idx, sd_idx))

        self.scene_state["_transit_stop_map"] = stop_map
        n_links = sum(len(v) for v in stop_map.values())
        print(f"[transit] ETA map: {len(stop_map)} stops linked ({n_links} bus/stop pairs)")

    def _update_transit_eta_labels(self) -> None:
        """Show live ETA labels above stops when a bus is within ~5 min."""
        stop_map = self.scene_state.get("_transit_stop_map")
        if not stop_map:
            return

        # Throttle: update only every 3 s
        now = time.perf_counter()
        if now - float(self.scene_state.get("_transit_label_t", 0.0)) < 3.0:
            return
        self.scene_state["_transit_label_t"] = now

        eta_pts: list[np.ndarray] = []
        eta_labels: list[str]    = []

        for stop_idx, bus_list in stop_map.items():
            if stop_idx >= len(self.bus_stop_positions):
                continue
            min_eta   = float("inf")
            best_name = ""

            for bus_idx, sd_idx in bus_list:
                if bus_idx >= len(self.buses):
                    continue
                bus       = self.buses[bus_idx]
                sd        = bus.get("stop_dists", [])
                if sd_idx >= len(sd):
                    continue
                stop_d    = float(sd[sd_idx][0])
                cur_d     = float(bus["dist"])
                route_len = float(bus["route_cum"][-1]) if len(bus["route_cum"]) else 1.0
                remaining = stop_d - cur_d
                if remaining < 0.0:
                    remaining += route_len
                spd = max(float(bus.get("speed", 0.0)), float(bus.get("v0", 7.0)) * 0.3)
                eta = remaining / spd
                if eta < min_eta:
                    min_eta   = eta
                    best_name = str(bus.get("route_name", ""))

            if min_eta < 300.0:
                mm = int(min_eta) // 60
                ss = int(min_eta) % 60
                tag = f"{best_name} " if best_name and best_name != "?" else ""
                eta_labels.append(f"{tag}{mm}:{ss:02d}")
                sp  = np.asarray(self.bus_stop_positions[stop_idx], dtype=float).copy()
                sp[2] += 5.0
                eta_pts.append(sp)

        # Remove previous label actor
        old = self.scene_state.get("_transit_eta_actor")
        if old is not None:
            try:
                self.plotter.remove_actor(old, reset_camera=False)
            except Exception:
                pass
            self.scene_state["_transit_eta_actor"] = None

        if not eta_pts:
            return

        try:
            actor = self.plotter.add_point_labels(
                np.array(eta_pts, dtype=float),
                eta_labels,
                point_size=0,
                font_size=9,
                text_color="white",
                shape_color=(0.1, 0.1, 0.6),
                shape_opacity=0.72,
                always_visible=True,
                reset_camera=False,
            )
            self.scene_state["_transit_eta_actor"] = actor
        except Exception:
            pass

    # ==================================================================
    # GTFS-Realtime — LIVE vehicle positions
    # ==================================================================

    # Feed considered "live" only if the last successful fetch is within this
    # many seconds; otherwise we fall back to the simulated schedule.
    _RT_STALE_S = 90.0
    # Live vehicles farther than this beyond the scene bounds are ignored.
    _RT_SCENE_MARGIN = 250.0
    # Per-tick easing factor toward the latest reported position (0..1).
    _RT_LERP = 0.18

    def _init_gtfs_rt(self, feed_url: str = "", api_key: str = "",
                      api_key_param: str = "", interval: float = 15.0) -> None:
        """Start the GTFS-Realtime poller (no-op if no URL configured)."""
        self.gtfs_rt: dict = {
            "enabled":     False,
            "poller":      None,
            "transformer": None,
            "actors":      {},   # vehicle_id -> VTK actor
            "current":     {},   # vehicle_id -> np.array([x, y, z, bearing])
            "bounds":      None, # (x0, x1, y0, y1) scene clip box
            "live":        False,
        }
        if not feed_url:
            return

        try:
            from gtfs_realtime import GTFSRealtimePoller
        except Exception as exc:
            print(f"[gtfs-rt] module unavailable: {exc}")
            return

        # CRS transformer: WGS84 (lon/lat) -> local projected metres
        proj_str = self.street_graph.graph.get("proj_str", "")
        if not proj_str:
            lat = float(getattr(self.args, "lat", 0.0))
            lon = float(getattr(self.args, "lon", 0.0))
            proj_str = (
                f"+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
        except Exception as exc:
            print(f"[gtfs-rt] CRS transform unavailable: {exc}")
            return

        # Scene clip box from the buildings mesh
        bounds = None
        try:
            if self.buildings_mesh is not None and self.buildings_mesh.n_points > 0:
                bx0, bx1, by0, by1 = self.buildings_mesh.bounds[:4]
                m = self._RT_SCENE_MARGIN
                bounds = (bx0 - m, bx1 + m, by0 - m, by1 + m)
        except Exception:
            pass

        poller = GTFSRealtimePoller(
            feed_url,
            interval=interval,
            api_key=api_key or None,
            api_key_param=api_key_param or None,
        )
        if not poller.start():
            return   # bindings missing — message already printed

        # Ensure we have a bus glyph even if simulated buses were disabled
        if self.bus_mesh is None:
            self.bus_mesh = _build_bus_mesh()

        self.gtfs_rt.update({
            "enabled":     True,
            "poller":      poller,
            "transformer": transformer,
            "bounds":      bounds,
        })
        print("[gtfs-rt] live vehicle layer enabled")

    def _rt_is_live(self) -> bool:
        rt = getattr(self, "gtfs_rt", None)
        if not rt or not rt.get("enabled"):
            return False
        poller = rt.get("poller")
        if poller is None:
            return False
        return poller.age_seconds() <= self._RT_STALE_S

    def _render_live_vehicles(self) -> None:
        """Place/move VTK actors at the latest live vehicle positions."""
        rt = self.gtfs_rt
        poller      = rt["poller"]
        transformer = rt["transformer"]
        bounds      = rt["bounds"]
        actors      = rt["actors"]
        current     = rt["current"]

        vehicles = poller.snapshot()
        dem = self.street_graph.graph.get("terrain_sampler")

        seen: set[str] = set()
        for v in vehicles:
            vid = v["vehicle_id"]
            try:
                x, y = transformer.transform(v["lon"], v["lat"])
            except Exception:
                continue
            if bounds is not None:
                if not (bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]):
                    continue   # outside the loaded scene — skip

            z = 0.0
            if dem is not None:
                try:
                    z = float(dem(np.array([[x, y]]))[0])
                except Exception:
                    z = 0.0

            bearing = v.get("bearing")
            # GTFS bearing is compass (0=N, CW); VTK Z-rotation is CCW from +X.
            heading = (90.0 - float(bearing)) % 360.0 if bearing is not None else None

            seen.add(vid)
            target = np.array([x, y, z, heading if heading is not None else 0.0], dtype=float)

            if vid not in current:
                current[vid] = target.copy()
            else:
                cur = current[vid]
                cur[:3] += (target[:3] - cur[:3]) * self._RT_LERP
                if heading is not None:
                    # shortest-arc angular lerp
                    da = (target[3] - cur[3] + 180.0) % 360.0 - 180.0
                    cur[3] = (cur[3] + da * self._RT_LERP) % 360.0
                current[vid] = cur

            pos = current[vid]
            actor = actors.get(vid)
            if actor is None and self.bus_mesh is not None:
                try:
                    actor = self.plotter.add_mesh(
                        self.bus_mesh.copy(),
                        color="#00E5A0",          # live = bright teal (vs sim navy)
                        smooth_shading=True, lighting=True, reset_camera=False,
                    )
                    actors[vid] = actor
                except Exception:
                    actor = None
            if actor is not None:
                try:
                    actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
                    actor.SetOrientation(0.0, 0.0, float(pos[3]))
                except Exception:
                    pass

        # Retire actors for vehicles that dropped out of the feed
        for vid in list(actors.keys()):
            if vid not in seen:
                try:
                    self.plotter.remove_actor(actors[vid], reset_camera=False)
                except Exception:
                    pass
                actors.pop(vid, None)
                current.pop(vid, None)

        # HUD badge
        try:
            self.plotter.add_text(
                f"\U0001F6F0 LIVE transit — {len(seen)} vehicles",
                position=(0.02, 0.93), name="gtfs_rt_hud",
                font_size=10, viewport=True, color="#00E5A0",
            )
        except Exception:
            pass

    def _hide_sim_buses(self) -> None:
        """Hide simulated bus actors while live data is driving the scene."""
        for actor in self.scene_state.get("_bus_vtk_actors", []) or []:
            try:
                actor.VisibilityOff()
            except Exception:
                pass

    def _show_sim_buses(self) -> None:
        for actor in self.scene_state.get("_bus_vtk_actors", []) or []:
            try:
                actor.VisibilityOn()
            except Exception:
                pass

    def _clear_live_vehicles(self) -> None:
        rt = getattr(self, "gtfs_rt", None)
        if not rt:
            return
        for vid, actor in list(rt.get("actors", {}).items()):
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        rt["actors"].clear()
        rt["current"].clear()
        try:
            self.plotter.add_text("", position=(0.02, 0.93),
                                  name="gtfs_rt_hud", viewport=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animate_buses(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return

        rt_live = self._rt_is_live()

        # Nothing to do if neither simulated nor live buses are available
        if not self.buses and not rt_live:
            # If a live layer just went stale, retire its actors once
            if getattr(self, "gtfs_rt", {}).get("actors"):
                self._clear_live_vehicles()
                self._show_sim_buses()
            return

        try:
            now  = time.perf_counter()
            last = float(self.scene_state.get("_bus_anim_last_t", now))
            dt   = min(float(now - last), 0.25)
            self.scene_state["_bus_anim_last_t"] = now

            self._render_bus_stops()
            self._render_transit_routes()

            # 'b' key is bound centrally at startup in main_ast6 — registering
            # it here again would make each press fire twice (toggle = no-op).
            if not self.scene_state.get("_bus_key_registered"):
                self.scene_state["_bus_key_registered"] = True
                print("[transit] 'b' = toggle bus routes + stops overlay")

            if rt_live:
                # Live data drives the buses; pause the simulated layer.
                if not self.scene_state.get("_rt_was_live"):
                    self._hide_sim_buses()
                    self.scene_state["_rt_was_live"] = True
                self._render_live_vehicles()
            else:
                # Simulated schedule (default / fallback).
                if self.scene_state.get("_rt_was_live"):
                    self._clear_live_vehicles()
                    self._show_sim_buses()
                    self.scene_state["_rt_was_live"] = False
                if self.buses:
                    self._precompute_stop_bus_map()
                    self._advance_buses(dt)
                    self._render_buses()
                    self._update_transit_eta_labels()
        except Exception as exc:
            print(f"[buses] animate error: {exc}")
