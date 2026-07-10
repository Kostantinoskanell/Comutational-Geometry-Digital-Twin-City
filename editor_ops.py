"""editor_ops.py — pure graph-surgery operations behind the in-app road editor.

No VTK/PyVista imports. Each function mutates the given networkx MultiDiGraph
in place and returns data the caller needs for rendering.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString


def reverse_edge(street_graph, u, v, key) -> dict:
    """Move the u→v edge (at `key`) to v→u, flipping geometry and direction tags.

    Returns the new edge data dict. Raises KeyError if the edge is absent.
    """
    data = street_graph.get_edge_data(u, v, key)
    if data is None:
        raise KeyError(f"edge ({u} -> {v}, key={key}) not in graph")
    data = data.copy()

    if "geometry" in data:
        data["geometry"] = LineString(list(data["geometry"].coords)[::-1])

    # Flip direction tags if present
    if "oneway" in data:
        data["oneway"] = True
    data["oneway_legal_forward"] = not data.get("oneway_legal_forward", True)

    street_graph.remove_edge(u, v, key=key)
    street_graph.add_edge(v, u, key=key, **data)
    return data


def _nearest_ring_node(street_graph, ring_nodes, x, y):
    """Return the ring node closest to (x, y)."""
    dists = [((street_graph.nodes[rn]["x"] - x) ** 2
              + (street_graph.nodes[rn]["y"] - y) ** 2, rn)
             for rn in ring_nodes]
    return min(dists)[1]


def insert_roundabout(street_graph, node_id, radius: float = 14.0, n_pts: int = 8) -> dict:
    """Replace `node_id` with a one-way circulating ring of `n_pts` nodes.

    Ring nodes are named ``ra_{node_id}_{i}``; ring edges are tagged
    oneway=True, junction="roundabout".  Every in/out edge of the original
    node is rewired to its nearest ring node (with geometry endpoint fix and
    length update) and the original node is removed.

    Returns {"ring_nodes": [...], "center": (cx, cy), "radius": radius}.
    """
    cx = float(street_graph.nodes[node_id]["x"])
    cy = float(street_graph.nodes[node_id]["y"])
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    ring_nodes = []
    for i, a in enumerate(angles):
        nx_ = cx + radius * np.cos(a)
        ny_ = cy + radius * np.sin(a)
        n_id = f"ra_{node_id}_{i}"
        street_graph.add_node(n_id, x=nx_, y=ny_, osmid=n_id)
        ring_nodes.append(n_id)

    for i in range(n_pts):
        u = ring_nodes[i]
        v = ring_nodes[(i + 1) % n_pts]
        geom = LineString([(street_graph.nodes[u]["x"], street_graph.nodes[u]["y"]),
                           (street_graph.nodes[v]["x"], street_graph.nodes[v]["y"])])
        street_graph.add_edge(u, v, key=0, length=geom.length, geometry=geom,
                              oneway=True, junction="roundabout")

    in_edges = list(street_graph.in_edges(node_id, data=True, keys=True))
    out_edges = list(street_graph.out_edges(node_id, data=True, keys=True))
    for u, _, k, d in in_edges:
        if u == node_id:
            continue
        ux, uy = float(street_graph.nodes[u]["x"]), float(street_graph.nodes[u]["y"])
        rn = _nearest_ring_node(street_graph, ring_nodes, ux, uy)
        if "geometry" in d:
            coords = list(d["geometry"].coords)
            coords[-1] = (float(street_graph.nodes[rn]["x"]), float(street_graph.nodes[rn]["y"]))
            d["geometry"] = LineString(coords)
            d["length"] = d["geometry"].length
        street_graph.remove_edge(u, node_id, key=k)
        street_graph.add_edge(u, rn, key=k, **d)

    for _, v, k, d in out_edges:
        if v == node_id:
            continue
        vx, vy = float(street_graph.nodes[v]["x"]), float(street_graph.nodes[v]["y"])
        rn = _nearest_ring_node(street_graph, ring_nodes, vx, vy)
        if "geometry" in d:
            coords = list(d["geometry"].coords)
            coords[0] = (float(street_graph.nodes[rn]["x"]), float(street_graph.nodes[rn]["y"]))
            d["geometry"] = LineString(coords)
            d["length"] = d["geometry"].length
        street_graph.remove_edge(node_id, v, key=k)
        street_graph.add_edge(rn, v, key=k, **d)

    street_graph.remove_node(node_id)
    return {"ring_nodes": ring_nodes, "center": (cx, cy), "radius": radius}


def make_building_mesh(cx: float, cy: float,
                       width: float = 18.0, depth: float = 18.0,
                       height: float = 24.0, base_z: float = 0.0):
    """Extruded rectangular building footprint centred at (cx, cy).

    Returns a triangulated pv.PolyData ready to merge into buildings_mesh and
    feed the shadow octree.  Imported lazily so this module stays VTK-free for
    the pure graph operations above.
    """
    import pyvista as pv
    w2, d2 = float(width) / 2.0, float(depth) / 2.0
    box = pv.Box(bounds=(cx - w2, cx + w2,
                         cy - d2, cy + d2,
                         float(base_z), float(base_z) + float(height)))
    return box.triangulate()


def building_size_for_click(px: float, py: float) -> tuple[float, float, float]:
    """Deterministic pseudo-random building dimensions for a click position
    (same spot → same building, so undo/redo demos are reproducible)."""
    rng = np.random.default_rng(abs(int(px * 131 + py * 71)) % (2**32))
    width  = float(rng.uniform(12.0, 26.0))
    depth  = float(rng.uniform(12.0, 26.0))
    height = float(rng.uniform(12.0, 42.0))
    return width, depth, height


def make_highway_tube(x1: float, y1: float, x2: float, y2: float,
                      height: float = 6.0, n_pts: int = 48,
                      radius: float = 2.2,
                      ramp_len: float = 18.0) -> "pv.PolyData":
    """Smooth elevated tube for a highway/bridge segment.

    Ramped flat-deck profile: rises from ground over `ramp_len` metres at each
    end, flat at `height` in between — the SAME shape `_bridge_z_offsets` in
    app_core gives the car paths, so vehicles ride exactly on the deck.
    Returns a pyvista tube mesh coloured later by the caller.
    """
    import pyvista as pv
    t = np.linspace(0.0, 1.0, n_pts)
    xs = float(x1) + (float(x2) - float(x1)) * t
    ys = float(y1) + (float(y2) - float(y1)) * t
    total = float(np.hypot(x2 - x1, y2 - y1))
    ramp = min(float(ramp_len), total / 2.0) if total > 1e-6 else 1.0
    from_start = t * total
    from_end = total - from_start
    zs = float(height) * np.clip(
        np.minimum(from_start, from_end) / max(ramp, 1e-6), 0.0, 1.0)
    pts = np.column_stack([xs, ys, zs]).astype(float)
    # Upsample to a smooth spline then sweep a tube cross-section
    spline = pv.Spline(pts, n_points=max(60, n_pts * 3))
    return spline.tube(radius=float(radius))


def add_bridge_highway_edge(graph, u, v,
                            ux: float, uy: float,
                            vx: float, vy: float,
                            layer: int = 1) -> None:
    """Add bidirectional motorway/bridge edges between existing nodes u and v.

    Sets all attributes `_extract_drivable_paths` and `_street_line_layers`
    need: highway, bridge, layer, length, maxspeed, lanes, geometry.
    `layer` scales the elevation used by `_edge_bridge_height` (5 m per layer)
    — pass ceil(height/5) so cars ride at the visual deck height.
    """
    import math
    dist = math.hypot(vx - ux, vy - uy)
    _common = {
        "highway":  "motorway",
        "bridge":   True,
        "layer":    max(1, int(layer)),
        "oneway":   False,
        "length":   dist,
        "maxspeed": 100,
        "lanes":    2,
    }
    graph.add_edge(u, v, **_common,
                   geometry=LineString([(ux, uy), (vx, vy)]))
    graph.add_edge(v, u, **_common,
                   geometry=LineString([(vx, vy), (ux, uy)]))


def bridge_clearance_height(building_points, ux: float, uy: float,
                            vx: float, vy: float,
                            corridor: float = 12.0,
                            base_height: float = 6.0,
                            clearance: float = 3.0) -> float:
    """Deck height needed for a bridge to clear every building near its span.

    `building_points` is an (N, 3) array (e.g. buildings_mesh.points).  Any
    point within `corridor` metres of the u→v segment raises the deck to that
    building's top + `clearance`.  Never returns less than `base_height`.
    """
    if building_points is None or len(building_points) == 0:
        return float(base_height)
    pts = np.asarray(building_points, dtype=float)
    a = np.array([ux, uy], dtype=float)
    b = np.array([vx, vy], dtype=float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return float(base_height)
    t = np.clip((pts[:, :2] - a) @ ab / L2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(pts[:, :2] - proj, axis=1)
    near = d < float(corridor)
    if not near.any():
        return float(base_height)
    return max(float(base_height), float(pts[near, 2].max()) + float(clearance))
