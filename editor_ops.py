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


def _ear_clip_triangulate(verts_2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple (non-self-intersecting) polygon.

    Unlike a fan from a single fixed vertex, this is correct for ANY simple
    non-convex polygon — a fan only works when the whole polygon is "star
    shaped" from that one vertex, which a hand-clicked concave greenspace
    (an L-shape, a U-shape, ...) is not guaranteed to be; the fan would then
    emit triangles that poke outside the drawn outline.

    Returns index triples into `verts_2d` (O(n^2), fine for the modest vertex
    counts a click-editor produces).
    """
    n = verts_2d.shape[0]
    if n < 3:
        return []

    # The ear test below assumes CCW winding; flip the traversal order if the
    # input polygon is CW (shoelace formula sign).
    shoelace = np.sum(
        verts_2d[:, 0] * np.roll(verts_2d[:, 1], -1) - np.roll(verts_2d[:, 0], -1) * verts_2d[:, 1]
    )
    order = list(range(n)) if shoelace >= 0 else list(range(n))[::-1]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _in_tri(p, a, b, c):
        d1, d2, d3 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_neg and has_pos)

    tris: list[tuple[int, int, int]] = []
    remaining = list(order)
    guard = 0
    while len(remaining) > 3 and guard < 10000:
        guard += 1
        m = len(remaining)
        for k in range(m):
            i0, i1, i2 = remaining[(k - 1) % m], remaining[k], remaining[(k + 1) % m]
            a, b, c = verts_2d[i0], verts_2d[i1], verts_2d[i2]
            if _cross(a, b, c) <= 1e-12:
                continue   # reflex or degenerate vertex: cannot be an ear
            if any(j not in (i0, i1, i2) and _in_tri(verts_2d[j], a, b, c) for j in remaining):
                continue   # another vertex sits inside this candidate ear
            tris.append((i0, i1, i2))
            remaining.pop(k)
            break
        else:
            break   # no ear found (duplicate/degenerate points) — stop gracefully
    if len(remaining) == 3:
        tris.append((remaining[0], remaining[1], remaining[2]))
    return tris


def make_greenspace_polygon(points, base_z: float = 0.0) -> "pv.PolyData":
    """Flat filled polygon mesh from >=3 (x, y) click points, coloured green.

    Cleaned via shapely (handles a self-intersecting click order) then
    triangulated with ear-clipping (correct for non-convex outlines — see
    `_ear_clip_triangulate`). Returns a pv.PolyData ready to `plotter.add_mesh`
    the same way `make_building_mesh` is used.
    """
    import pyvista as pv
    from shapely.geometry import Polygon as _SPoly, MultiPolygon as _SMultiPoly

    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("make_greenspace_polygon needs at least 3 points")

    z = float(base_z)
    verts = pts
    try:
        poly = _SPoly(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if isinstance(poly, _SMultiPoly):
            # A self-intersecting click order ("figure 8") can clean into
            # several disjoint pieces — keep the largest rather than falling
            # through to triangulating the raw, still-self-intersecting input.
            poly = max(poly.geoms, key=lambda g: g.area)
        if not poly.is_empty and poly.geom_type == "Polygon":
            verts = np.asarray(poly.exterior.coords[:-1], dtype=float)
    except Exception:
        pass   # keep the raw click points; ear-clipping degrades gracefully

    tris = _ear_clip_triangulate(verts)
    n = verts.shape[0]
    vertices_3d = np.column_stack([verts[:, 0], verts[:, 1], np.full(n, z)])
    faces = np.hstack([[3, a, b, c] for a, b, c in tris]).astype(np.int64) if tris else np.empty(0, dtype=np.int64)
    return pv.PolyData(vertices_3d, faces)


def _oriented_box(cx: float, cy: float, bot_z: float, top_z: float,
                   half_run: float, w2: float, dirx: float, diry: float,
                   perpx: float, perpy: float) -> "pv.PolyData":
    """One step's solid box, correctly wound on every face (outward normals).

    Bottom face [0,3,2,1] deliberately reverses vertex 0,1,2,3's traversal —
    verified by the right-hand-rule cross product of consecutive edges: the
    naive [0,1,2,3] order (matching the top face's [4,5,6,7]) gives BOTH caps
    a +z normal, which is correct for the top but points the bottom's normal
    upward into the box instead of downward and out of it.
    """
    import pyvista as pv
    corners_local = np.array([
        [-half_run, -w2, bot_z], [half_run, -w2, bot_z],
        [half_run,  w2, bot_z], [-half_run,  w2, bot_z],
        [-half_run, -w2, top_z], [half_run, -w2, top_z],
        [half_run,  w2, top_z], [-half_run,  w2, top_z],
    ])
    rot = np.array([[dirx, perpx, 0.0], [diry, perpy, 0.0], [0.0, 0.0, 1.0]])
    world = corners_local @ rot.T + np.array([cx, cy, 0.0])
    return pv.PolyData(world, faces=np.hstack([
        [4, 0, 3, 2, 1], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5], [4, 2, 3, 7, 6], [4, 3, 0, 4, 7],
    ]).astype(np.int64)).triangulate()


def make_stairs_mesh(x1: float, y1: float, z1: float,
                     x2: float, y2: float, z2: float,
                     width: float = 2.0, step_rise: float = 0.17,
                     step_run: float = 0.29) -> "pv.PolyData":
    """Stepped ramp of boxes between two 3D points, oriented along (x1,y1)->(x2,y2).

    The flight always spans the FULL requested distance in both x/y and z —
    step count is the larger of "steps needed for a comfortable rise" and
    "steps needed to cover the horizontal span at a comfortable tread depth",
    so a shallow slope over a long span gets many gentle steps instead of a
    physically-correct-but-short flight that stops well before (x2, y2)
    leaving it visibly disconnected from the second click point.

    Two clicks at (near enough) the same elevation build a single flat
    walkway slab instead of `step_rise/n_steps == 0`-tall degenerate boxes.
    """
    dz = float(z2) - float(z1)
    horiz = float(np.hypot(x2 - x1, y2 - y1))
    if horiz < 1e-6 and abs(dz) < 1e-6:
        raise ValueError("make_stairs_mesh: start and end points coincide")

    if horiz > 1e-6:
        dirx, diry = (float(x2) - float(x1)) / horiz, (float(y2) - float(y1)) / horiz
    else:
        dirx, diry = 1.0, 0.0
    perpx, perpy = -diry, dirx
    w2 = float(width) / 2.0

    _FLAT_EPS = 0.02   # m: below this, "stairs" degenerate to a flat walkway
    if abs(dz) < _FLAT_EPS:
        thickness = 0.15
        cx, cy = float(x1) + dirx * horiz / 2.0, float(y1) + diry * horiz / 2.0
        half_run = max(horiz, width) / 2.0
        return _oriented_box(cx, cy, float(z1), float(z1) + thickness,
                              half_run, w2, dirx, diry, perpx, perpy)

    n_from_rise = max(1, int(round(abs(dz) / max(float(step_rise), 1e-3))))
    n_from_run = max(1, int(round(horiz / max(float(step_run), 1e-3)))) if horiz > 1e-6 else 1
    n_steps = max(n_from_rise, n_from_run)
    run = horiz / n_steps if horiz > 1e-6 else float(step_run)
    rise = dz / n_steps
    half_run = run / 2.0

    blocks = []
    for i in range(n_steps):
        cx = float(x1) + dirx * run * (i + 0.5)
        cy = float(y1) + diry * run * (i + 0.5)
        top_z = float(z1) + rise * (i + 1)
        bot_z = float(z1)   # steps are solid down to the base for a filled staircase look
        blocks.append(_oriented_box(cx, cy, bot_z, top_z, half_run, w2, dirx, diry, perpx, perpy))

    merged = blocks[0]
    for b in blocks[1:]:
        merged = merged.merge(b, merge_points=False)
    return merged


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
