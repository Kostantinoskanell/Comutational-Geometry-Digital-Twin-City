"""DemandMixin — wires the gravity O-D model into the IDM car layer.

What it changes vs the old behaviour
------------------------------------
Before: cars picked a uniform-random residential→commercial trip once at init,
then reverted to random next-edge turns forever after arriving.

After:
  • Initial trips are gravity- + time-of-day-weighted (demand_model.DemandModel).
  • Every tick a few cars whose route is finished (`planned_edges[i] is None`)
    are given a *fresh* onward trip from where they currently sit (trip chaining),
    so demand is continuous and the random-turn fallback essentially never fires.

The route is handed to the IDM via the existing `planned_edges` / `planned_cursor`
mechanism, so no IDM changes are needed.
"""
from __future__ import annotations

import numpy as np

try:
    import networkx as nx
    _NX_OK = True
except Exception:
    _NX_OK = False


class DemandMixin:

    _DEMAND_REASSIGN_PER_TICK = 5   # cap routing work per car tick

    # ------------------------------------------------------------------
    # Build the model from the graph's residential / commercial nodes
    # ------------------------------------------------------------------

    def _init_demand(self) -> None:
        self.demand = None
        self._demand_scan = 0
        if not _NX_OK:
            return
        if not hasattr(self, "car_anim") or not bool(self.car_anim.get("enabled")):
            return

        g = self.street_graph
        res_nodes, res_xy, com_nodes, com_xy = [], [], [], []
        for n, d in g.nodes(data=True):
            if "x" not in d or "y" not in d:
                continue
            xy = (float(d["x"]), float(d["y"]))
            if d.get("is_residential"):
                res_nodes.append(n); res_xy.append(xy)
            if d.get("is_commercial"):
                com_nodes.append(n); com_xy.append(xy)

        # Fallbacks so the model still works on sparsely-tagged areas:
        if not com_nodes and getattr(self, "places", None):
            try:
                from scipy.spatial import cKDTree
                ids = list(g.nodes)
                pts = np.array([[float(g.nodes[i]["x"]), float(g.nodes[i]["y"])] for i in ids])
                tree = cKDTree(pts)
                p_pts = np.array([[float(p["x"]), float(p["y"])] for p in self.places])
                _, idx = tree.query(p_pts)
                seen = set()
                for i in idx:
                    nid = ids[int(i)]
                    if nid not in seen:
                        seen.add(nid)
                        com_nodes.append(nid)
                        com_xy.append((float(g.nodes[nid]["x"]), float(g.nodes[nid]["y"])))
            except Exception:
                pass
        if not res_nodes:
            for n, d in g.nodes(data=True):
                if "x" in d and "y" in d:
                    res_nodes.append(n); res_xy.append((float(d["x"]), float(d["y"])))

        if not res_nodes or not com_nodes:
            print("[demand] no residential/commercial zones — gravity model disabled "
                  "(cars use random turns)")
            return

        # Distance-decay length scaled to the loaded scene size.
        try:
            radius = float(getattr(self.args, "radius", 600.0))
        except Exception:
            radius = 600.0
        decay = float(np.clip(radius * 1.5, 400.0, 3000.0))

        from demand_model import DemandModel
        self.demand = DemandModel(
            res_nodes, np.array(res_xy, dtype=float),
            com_nodes, np.array(com_xy, dtype=float),
            decay_length_m=decay,
        )

        # Cache the (u,v)→path-index map once for fast route→edge conversion.
        self._demand_edge_map = {}
        for i, p in enumerate(self.car_paths):
            self._demand_edge_map.setdefault((p.get("u"), p.get("v")), i)

        print(f"[demand] gravity O-D model: {len(res_nodes)} residential, "
              f"{len(com_nodes)} commercial zones, decay≈{decay:.0f} m")

        # Give every car a gravity trip up front (one-time; n_cars is capped).
        hour = float(self.scene_state.get("hour", 12.0))
        assigned = 0
        n = len(self.car_anim.get("edge_idx", []))
        for i in range(n):
            if self._assign_demand_trip(i, hour, snap=True):
                assigned += 1
        print(f"[demand] {assigned}/{n} cars assigned an initial gravity trip")

    # ------------------------------------------------------------------
    # Assign one trip to a car
    # ------------------------------------------------------------------

    def _route_to_edges(self, nodes: list) -> list[int]:
        if len(nodes) < 2:
            return []
        edges = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            idx = self._demand_edge_map.get((a, b))
            if idx is None:
                return []
            edges.append(int(idx))
        return edges

    def _assign_demand_trip(self, car_idx: int, hour: float, snap: bool) -> bool:
        """Sample a trip for `car_idx`, route it, install it as the car's plan.

        snap=True  : place the car at the route start (used at init).
        snap=False : trip-chaining — origin is the car's current node, so the
                     plan begins right where the car already is (no teleport).
        Returns True if a plan was installed.
        """
        if self.demand is None or not self.demand.usable:
            return False
        plans = self.car_anim.get("planned_edges")
        cursors = self.car_anim.get("planned_cursor")
        if not isinstance(plans, list) or car_idx >= len(plans):
            return False

        try:
            if snap:
                trip = self.demand.sample_trip(hour, self.car_rng)
                if trip is None:
                    return False
                origin, dest = trip
            else:
                cur_edge = int(self.car_anim["edge_idx"][car_idx])
                origin = self.car_paths[cur_edge].get("v")
                od = self.street_graph.nodes.get(origin, {})
                if "x" not in od:
                    return False
                dest = self.demand.pick_destination_from(
                    (float(od["x"]), float(od["y"])), hour, self.car_rng)
                if dest is None or dest == origin:
                    return False

            nodes = nx.shortest_path(self.street_graph, origin, dest, weight="length")
            edges = self._route_to_edges(nodes)
            if len(edges) < 1:
                return False

            plans[car_idx] = np.asarray(edges, dtype=np.int64)
            cursors[car_idx] = 0
            if snap:
                first = int(edges[0])
                self.car_anim["edge_idx"][car_idx] = first
                length0 = float(self.car_path_lengths[first])
                self.car_anim["dist"][car_idx] = float(
                    self.car_rng.uniform(0.0, max(length0 * 0.3, 0.0)))
            return True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Per-tick: re-trip finished cars (throttled)
    # ------------------------------------------------------------------

    def _reassign_finished_trips(self, hour: float) -> None:
        """Give a fresh onward trip to cars whose planned route is exhausted."""
        if self.demand is None or not self.demand.usable:
            return
        plans = self.car_anim.get("planned_edges")
        if not isinstance(plans, list) or not plans:
            return

        n = len(plans)
        budget = self._DEMAND_REASSIGN_PER_TICK
        # Round-robin scan so no car is starved.
        start = int(self._demand_scan) % n
        done = 0
        for k in range(n):
            if budget <= 0:
                break
            i = (start + k) % n
            if plans[i] is None:
                if self._assign_demand_trip(i, hour, snap=False):
                    budget -= 1
                    done += 1
        self._demand_scan = (start + n) % max(n, 1)
