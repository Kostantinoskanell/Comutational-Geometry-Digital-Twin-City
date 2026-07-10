from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from ped_mixin import _load_glb_as_polydata

_MODELS_DIR = Path(__file__).parent / "assets" / "models"

try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

_em_mesh_cache: pv.PolyData | None = None

def _build_emergency_mesh() -> pv.PolyData:
    """Load low_poly_ambulance.obj scaled to 5.5 m length; procedural fallback."""
    global _em_mesh_cache
    if _em_mesh_cache is not None:
        return _em_mesh_cache

    geo = _load_glb_as_polydata(_MODELS_DIR / "low_poly_ambulance.obj")
    if geo is not None:
        x_len = float(geo.bounds[1] - geo.bounds[0])
        if x_len > 1e-6:
            scale = 5.5 / x_len
            geo.points *= scale
            b = geo.bounds
            geo.points -= np.array([(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, float(b[4])],
                                   dtype=float)
            b = geo.bounds
            print(f"[emergency] ambulance mesh: L={b[1]-b[0]:.1f}m "
                  f"W={b[3]-b[2]:.1f}m H={b[5]-b[4]:.1f}m")
            _em_mesh_cache = geo
            return geo

    # Fallback: van body + light bar
    body = pv.Box(bounds=(-2.25, 2.25, -1.0, 1.0, 0.0, 2.2))
    bar  = pv.Box(bounds=(-0.8,  0.8,  -0.85, 0.85, 2.2, 2.55))
    merged = body.merge(bar, merge_points=False)
    _em_mesh_cache = merged
    return merged


# ---------------------------------------------------------------------------
# EmergencyMixin
# ---------------------------------------------------------------------------

class EmergencyMixin:

    def _init_emergency(self) -> None:
        self.emergency_anim: dict = {
            "vehicles":  [],
            "max_count": 3,
        }
        self._em_keys_registered = False

    def _register_emergency_keys(self) -> None:
        if self._em_keys_registered:
            return
        try:
            # 'm' (medic) — was 'x', which is also bound to zoom-out; both fired
            self.plotter.add_key_event("m", lambda: self._spawn_or_clear_emergency())
            self._em_keys_registered = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Spawn / clear
    # ------------------------------------------------------------------

    def _spawn_or_clear_emergency(self) -> None:
        vehicles = self.emergency_anim["vehicles"]
        if len(vehicles) >= int(self.emergency_anim["max_count"]):
            self._clear_all_emergency()
        else:
            self._spawn_one_emergency()

    def _em_drivable_graph(self):
        """Street graph restricted to car-drivable roads — an ambulance may
        run red lights, but it must not route down footways or stairs."""
        g = self.street_graph
        ped_tags = {"footway", "pedestrian", "path", "cycleway", "steps", "bridleway"}

        def _hw(d) -> str:
            hw = d.get("highway", "")
            if isinstance(hw, (list, tuple)):
                hw = hw[0] if hw else ""
            return str(hw).lower()

        try:
            keep = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
                    if _hw(d) not in ped_tags]
        except TypeError:   # plain DiGraph (no keys) — used by some tests
            keep = [(u, v) for u, v, d in g.edges(data=True)
                    if _hw(d) not in ped_tags]
        if not keep or len(keep) == g.number_of_edges():
            return g
        return g.edge_subgraph(keep)

    def _spawn_one_emergency(self) -> None:
        if not _NX_OK:
            print("[emergency] networkx not available")
            return
        g = self._em_drivable_graph()
        all_nodes = list(g.nodes())
        if len(all_nodes) < 2:
            return

        rng = np.random.default_rng(int(time.perf_counter() * 1e6) & 0xFFFFFFFF)

        route_pts = None
        for _ in range(20):
            _pick = rng.choice(len(all_nodes), size=2, replace=False)
            node_a = all_nodes[int(_pick[0])]
            node_b = all_nodes[int(_pick[1])]
            try:
                path_len = nx.shortest_path_length(g, node_a, node_b, weight="length")
                if path_len < 200.0:
                    continue
                path_nodes = nx.shortest_path(g, node_a, node_b, weight="length")
                route_pts = self._em_nodes_to_polyline(path_nodes)
                if route_pts is not None and route_pts.shape[0] >= 2:
                    break
            except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                continue

        if route_pts is None or route_pts.shape[0] < 2:
            # Fallback: random edge
            edges = list(g.edges())
            if not edges:
                return
            u, v = edges[int(rng.integers(len(edges)))]
            try:
                path_nodes = nx.shortest_path(g, u, v, weight="length")
                route_pts = self._em_nodes_to_polyline(path_nodes)
            except Exception:
                return
            if route_pts is None or route_pts.shape[0] < 2:
                return

        seg_lens  = np.linalg.norm(route_pts[1:, :2] - route_pts[:-1, :2], axis=1)
        route_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

        vehicle = {
            "route_pts":   route_pts,
            "route_cum":   route_cum,
            "dist":        0.0,
            "speed":       0.0,
            "v_target":    15.0,
            "heading":     0.0,
            "pos":         np.zeros(3, dtype=float),
            "actor":       None,
            "flash_t":     0.0,
            "flash_state": False,
        }
        self.emergency_anim["vehicles"].append(vehicle)
        n = len(self.emergency_anim["vehicles"])
        print(f"[emergency] vehicle spawned, total={n}  (press 'm' again to add more, {3-n} remaining)")

    def _em_nodes_to_polyline(self, path_nodes) -> np.ndarray | None:
        g = self.street_graph
        xy = np.array(
            [[float(g.nodes[n].get("x", 0.0)), float(g.nodes[n].get("y", 0.0))]
             for n in path_nodes],
            dtype=float,
        )
        keep = np.ones(xy.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > 1e-6
        xy = xy[keep]
        if xy.shape[0] < 2:
            return None

        # Elevate only when terrain draping is active — otherwise the ambulance
        # floats at DEM height above the flat city.
        z_col = np.zeros(xy.shape[0], dtype=float)
        if self.scene_state.get("_terrain_drape_active"):
            dem = self.street_graph.graph.get("terrain_sampler")
            if dem is not None:
                try:
                    z_col = dem(xy).astype(float)
                except Exception:
                    pass

        return np.column_stack((xy, z_col))

    def _clear_all_emergency(self) -> None:
        for v in self.emergency_anim["vehicles"]:
            actor = v.get("actor")
            if actor is not None:
                try:
                    actor.VisibilityOff()
                except Exception:
                    pass
        self.emergency_anim["vehicles"].clear()
        print("[emergency] all vehicles cleared")

    # ------------------------------------------------------------------
    # Physics tick
    # ------------------------------------------------------------------

    def _advance_emergency(self, dt: float) -> None:
        for v in self.emergency_anim["vehicles"]:
            route_len = float(v["route_cum"][-1]) if len(v["route_cum"]) else 0.0
            if route_len < 1.0:
                continue

            # Accelerate hard to target speed
            v["speed"] = min(float(v["v_target"]), float(v["speed"]) + 3.0 * dt)
            v["dist"]  = float(v["dist"]) + float(v["speed"]) * dt

            # Respawn on new route when end reached
            if v["dist"] >= route_len:
                self._em_respawn_route(v)
                continue

            # Interpolate world position
            pts = v["route_pts"]
            cum = v["route_cum"]
            d   = float(np.clip(v["dist"], 0.0, max(route_len - 1e-9, 0.0)))
            seg = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, pts.shape[0] - 2))
            t   = (d - float(cum[seg])) / max(1e-9, float(cum[seg + 1]) - float(cum[seg]))
            v["pos"] = pts[seg] + (pts[seg + 1] - pts[seg]) * t

            # Heading
            p0, p1 = pts[seg], pts[seg + 1]
            h = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
            v["heading"] = (h + 180.0) % 360.0

            # Flashing lights
            v["flash_t"] = float(v["flash_t"]) + dt
            if v["flash_t"] >= 0.30:
                v["flash_t"]     = 0.0
                v["flash_state"] = not bool(v["flash_state"])
                actor = v.get("actor")
                if actor is not None:
                    try:
                        if v["flash_state"]:
                            actor.GetProperty().SetColor(1.0, 0.9, 0.0)   # amber
                        else:
                            actor.GetProperty().SetColor(1.0, 0.0, 0.0)   # red
                        actor.GetProperty().SetSpecular(0.8)
                        actor.GetProperty().SetSpecularPower(80)
                    except Exception:
                        pass

    def _em_respawn_route(self, vehicle: dict) -> None:
        if not _NX_OK:
            return
        g = self._em_drivable_graph()
        all_nodes = list(g.nodes())
        if len(all_nodes) < 2:
            return

        # Start from the node nearest current position
        pos = np.asarray(vehicle["pos"], dtype=float)
        nodes_xy = np.array(
            [[float(g.nodes[n].get("x", 0.0)), float(g.nodes[n].get("y", 0.0))]
             for n in all_nodes],
            dtype=float,
        )
        dists = np.linalg.norm(nodes_xy - pos[:2], axis=1)
        start_node = all_nodes[int(np.argmin(dists))]

        rng = np.random.default_rng(int(time.perf_counter() * 1e6) & 0xFFFFFFFF)
        route_pts = None
        for _ in range(15):
            nb = all_nodes[int(rng.integers(len(all_nodes)))]
            if nb == start_node:
                continue
            try:
                path_nodes = nx.shortest_path(g, start_node, nb, weight="length")
                candidate  = self._em_nodes_to_polyline(path_nodes)
                if candidate is not None and candidate.shape[0] >= 2:
                    route_pts = candidate
                    break
            except Exception:
                continue

        if route_pts is None:
            vehicle["dist"] = 0.0
            return

        seg_lens = np.linalg.norm(route_pts[1:, :2] - route_pts[:-1, :2], axis=1)
        vehicle["route_pts"] = route_pts
        vehicle["route_cum"] = np.concatenate([[0.0], np.cumsum(seg_lens)])
        vehicle["dist"]      = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_emergency(self) -> None:
        for v in self.emergency_anim["vehicles"]:
            if v["actor"] is None:
                try:
                    mesh  = _build_emergency_mesh().copy()
                    actor = self.plotter.add_mesh(
                        mesh,
                        color=(1.0, 0.0, 0.0),
                        smooth_shading=True,
                        lighting=True,
                        reset_camera=False,
                    )
                    actor.GetProperty().SetSpecular(0.8)
                    actor.GetProperty().SetSpecularPower(80)
                    v["actor"] = actor
                except Exception as exc:
                    print(f"[emergency] actor create failed: {exc}")
                    continue

            actor = v["actor"]
            pos   = v["pos"]
            try:
                actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
                actor.SetOrientation(0.0, 0.0, float(v["heading"]))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animate_emergency(self, _: int) -> None:
        if not bool(self.scene_state.get("interactive_ready", False)):
            return
        # Register key on first live callback (plotter guaranteed ready here)
        self._register_emergency_keys()
        try:
            now  = time.perf_counter()
            last = float(self.scene_state.get("_em_anim_last_t", now))
            dt   = min(float(now - last), 0.25)
            self.scene_state["_em_anim_last_t"] = now
            if self.emergency_anim["vehicles"]:
                self._advance_emergency(dt)
                self._render_emergency()
        except Exception as exc:
            print(f"[emergency] animate error: {exc}")

    # ------------------------------------------------------------------
    # Public: positions for car yield check in car_mixin._advance_cars
    # ------------------------------------------------------------------

    def _em_active_positions(self) -> list[np.ndarray]:
        return [
            np.asarray(v["pos"], dtype=float)
            for v in self.emergency_anim.get("vehicles", [])
            if v.get("route_pts") is not None and np.any(v["pos"] != 0)
        ]
