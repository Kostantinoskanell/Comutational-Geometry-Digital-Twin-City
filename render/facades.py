"""Procedural PBR facades for building walls (WG).

Metric-scale window grids generated per building class as a full PBR set:
  albedo   (sRGB)    wall render with weathering streaks, glass, frames
  ORM      (linear)  R occlusion (recessed windows), G roughness (glass
                     ~0.05, render ~0.85), B metallic (aluminium frames)
  normal   (linear)  tangent-space bevel of the window recesses
  emissive (sRGB)    randomly lit windows at 2700-4000 K, faded in at dusk
                     by the time-of-day update (replaces the ambient hack,
                     which PBR ignores)

Glass is a low-roughness dielectric, so windows reflect the physical sky
with the correct Fresnel response instead of being painted dark.

UVs are metric: each planar wall (connected region after the sharp-edge
vertex split) gets u along the wall, v = height above ground, with the bay
grid centred on the wall so windows are not cut at its ends; storeys start
at ground level (the flat, pre-drape z), so the terrain drape keeps floors
attached to their building. Random whole-bay / whole-storey offsets per
wall decorrelate the lit-window pattern without breaking alignment.
Roof faces map to a plain-wall texel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PX_PER_M = 24.0
TILE_BAYS = 8
TILE_STOREYS = 8


@dataclass(frozen=True)
class FacadeStyle:
    name: str
    bay_m: float          # horizontal window pitch
    storey_m: float       # floor-to-floor height
    win_w: float
    win_h: float
    sill_m: float         # window bottom above the storey floor
    wall_rgb: tuple       # display sRGB 0..1
    glass_rgb: tuple
    frame_rgb: tuple
    lit_frac: float       # share of windows lit at night
    recess_m: float = 0.15
    slab_band: bool = True


STYLES = {
    # building_class ids used by the app's PBR classes (main_ast6 _PBR_CLASSES)
    0: FacadeStyle("concrete", 3.0, 3.0, 1.0, 1.25, 1.0, (0.66, 0.64, 0.60), (0.09, 0.11, 0.13), (0.30, 0.30, 0.30), 0.30),
    1: FacadeStyle("brick", 3.2, 3.1, 1.3, 1.5, 0.9, (0.63, 0.47, 0.37), (0.09, 0.11, 0.13), (0.85, 0.85, 0.82), 0.35),
    2: FacadeStyle("glass", 1.5, 3.6, 1.42, 2.9, 0.35, (0.28, 0.30, 0.33), (0.10, 0.16, 0.20), (0.28, 0.30, 0.33), 0.50,
                   recess_m=0.04, slab_band=False),
    3: FacadeStyle("commercial", 1.6, 3.8, 1.45, 2.4, 0.8, (0.74, 0.74, 0.73), (0.10, 0.14, 0.17), (0.35, 0.36, 0.38), 0.55),
    4: FacadeStyle("residential", 3.2, 3.1, 1.4, 1.5, 0.9, (0.82, 0.78, 0.70), (0.09, 0.11, 0.13), (0.90, 0.90, 0.88), 0.35),
}


def _value_noise(h: int, w: int, cell: int, rng) -> np.ndarray:
    """Smooth [0,1] value noise, tileable in both axes."""
    gh, gw = max(1, h // cell), max(1, w // cell)
    g = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = g[y0[:, None] % gh, x0[None, :] % gw]
    b = g[y0[:, None] % gh, (x0[None, :] + 1) % gw]
    c = g[(y0[:, None] + 1) % gh, x0[None, :] % gw]
    d = g[(y0[:, None] + 1) % gh, (x0[None, :] + 1) % gw]
    return (a * (1 - fx) + b * fx) * (1 - fy[:, None]) + (c * (1 - fx) + d * fx) * fy[:, None]


def facade_maps(style: FacadeStyle, seed: int = 0) -> dict[str, np.ndarray]:
    """Generate the PBR maps for one tile (TILE_BAYS x TILE_STOREYS), as
    uint8 (H, W, 3) arrays with row 0 = TOP of the tile (pv.Texture layout)."""
    rng = np.random.default_rng(seed)
    W = int(round(TILE_BAYS * style.bay_m * PX_PER_M))
    H = int(round(TILE_STOREYS * style.storey_m * PX_PER_M))
    xm = (np.arange(W) + 0.5) / PX_PER_M                  # metres, left -> right
    ym = (np.arange(H) + 0.5) / PX_PER_M                  # metres, bottom -> top
    bx = np.mod(xm, style.bay_m) - style.bay_m / 2.0      # centred in bay
    sy = np.mod(ym, style.storey_m)
    frame = 0.07
    in_x = np.abs(bx) <= style.win_w / 2
    in_y = (sy >= style.sill_m) & (sy <= style.sill_m + style.win_h)
    win = in_y[:, None] & in_x[None, :]
    glass = ((np.abs(bx) <= style.win_w / 2 - frame)[None, :]
             & ((sy >= style.sill_m + frame) & (sy <= style.sill_m + style.win_h - frame))[:, None])
    frm = win & ~glass
    slab = (np.zeros_like(sy, bool) if not style.slab_band else (sy < 0.22))[:, None] & ~win

    # albedo: render + weathering (streaks below windows, large-scale tone)
    tone = 0.92 + 0.10 * _value_noise(H, W, int(6 * PX_PER_M), rng)
    grain = 0.97 + 0.06 * _value_noise(H, W, 6, rng)
    streak = np.ones((H, W))
    below = (sy < style.sill_m)[:, None] & in_x[None, :]
    depth = np.clip((style.sill_m - sy) / max(style.sill_m, 1e-6), 0, 1)[:, None]
    streak = np.where(below, 1.0 - 0.10 * (1.0 - depth) * (0.6 + 0.4 * _value_noise(H, W, 10, rng)), streak)
    wall = np.asarray(style.wall_rgb)[None, None, :] * (tone * grain * streak)[..., None]
    alb = wall.copy()
    alb[slab] *= 0.88
    alb[frm] = style.frame_rgb
    gl = np.asarray(style.glass_rgb)[None, None, :] * (0.85 + 0.3 * _value_noise(H, W, 20, rng))[..., None]
    alb[glass] = gl[glass]

    # ORM
    occ = np.where(win, 0.62, 1.0)
    occ = np.where(frm, 0.8, occ)
    rough = np.where(glass, 0.05, np.where(frm, 0.4, 0.85 + 0.1 * (grain - 1.0)))
    metal = np.where(frm, 0.8, 0.0)
    orm = np.stack([occ, rough, metal], axis=-1)

    # normal from a height field (recess), central differences
    hgt = np.where(win, -style.recess_m, 0.0)
    hgt = np.where(frm, -style.recess_m * 0.5, hgt)
    gx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * PX_PER_M / 2.0
    gy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * PX_PER_M / 2.0
    n = np.stack([-gx, -gy, np.ones_like(hgt)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = (n + 1.0) / 2.0

    # emissive: per window random lit state, colour temperature, dimming
    col_id = np.floor(xm / style.bay_m).astype(int)
    row_id = np.floor(ym / style.storey_m).astype(int)
    n_c, n_r = int(col_id.max()) + 1, int(row_id.max()) + 1
    lit = rng.random((n_r, n_c)) < style.lit_frac
    warm = rng.random((n_r, n_c))
    level = 0.45 + 0.55 * rng.random((n_r, n_c))
    k_warm = np.array([1.0, 0.72, 0.42])
    k_cool = np.array([0.92, 0.92, 1.0])
    cw = warm[row_id[:, None], col_id[None, :]][..., None]
    ecol = k_warm * cw + k_cool * (1 - cw)
    emis = np.where((glass & lit[row_id[:, None], col_id[None, :]])[..., None],
                    ecol * level[row_id[:, None], col_id[None, :]][..., None], 0.0)

    to8 = lambda a: np.clip(np.round(np.flipud(a) * 255.0), 0, 255).astype(np.uint8)
    return {"albedo": to8(alb), "orm": to8(orm), "normal": to8(nrm), "emissive": to8(emis),
            "tile_m": (TILE_BAYS * style.bay_m, TILE_STOREYS * style.storey_m)}


def facade_textures(style: FacadeStyle, seed: int = 0) -> dict:
    """pv.Texture set for a style (albedo/emissive sRGB, ORM/normal linear)."""
    import pyvista as pv
    maps = facade_maps(style, seed)
    out = {"tile_m": maps["tile_m"]}
    for key, srgb in (("albedo", True), ("orm", False), ("normal", False), ("emissive", True)):
        t = pv.Texture(np.ascontiguousarray(maps[key]))
        t.SetUseSRGBColorSpace(srgb)
        t.SetMipmap(True)
        t.SetInterpolate(True)
        t.SetRepeat(True)
        try:
            t.SetMaximumAnisotropicFiltering(8.0)
        except Exception:
            pass
        out[key] = t
    return out


def facade_uvs(mesh, style: FacadeStyle, seed: int = 0, ground_z=None) -> np.ndarray:
    """(n_points, 2) metric facade texture coordinates for a building mesh
    whose vertices are split at sharp edges (each planar wall = one
    point-connected region). ground_z: per-point storey origin (default 0)."""
    import pyvista as pv
    mesh = pv.wrap(mesh)
    tile_w = TILE_BAYS * style.bay_m
    tile_h = TILE_STOREYS * style.storey_m
    pts = np.asarray(mesh.points, dtype=float)
    z0 = np.zeros(len(pts)) if ground_z is None else np.asarray(ground_z, dtype=float)
    uv = np.zeros((len(pts), 2))
    # plain-wall texel for roofs / unassigned: left margin of bay 0, below the first sill
    plain = np.array([0.05 / tile_w, 0.05 / tile_h])
    uv[:] = plain
    tri = mesh.triangulate() if not mesh.is_all_triangles else mesh
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:]
    if faces.size == 0:
        return uv
    # point regions = planar walls (vertices are split at sharp edges)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(pts), len(pts)))
    n_reg, rid_pt = connected_components(g, directed=False)
    fn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    freg = rid_pt[faces[:, 0]]
    nsum = np.zeros((n_reg, 3))
    np.add.at(nsum, freg, fn)                            # area-weighted region normal
    wall_area = np.zeros(n_reg)
    tot_area = np.zeros(n_reg)
    vertical = np.abs(fn[:, 2]) < 0.3 * np.maximum(area, 1e-12)
    np.add.at(wall_area, freg, area * vertical)
    np.add.at(tot_area, freg, area)
    rng = np.random.default_rng(seed)
    for r in range(n_reg):
        nrm = nsum[r]
        horiz = np.linalg.norm(nrm[:2])
        sel = np.flatnonzero(rid_pt == r)
        if horiz >= 0.7 * np.linalg.norm(nrm) + 1e-12:   # planar wall
            t = np.array([-nrm[1], nrm[0]]) / horiz        # along-wall tangent
            d = pts[sel, :2] @ t
        elif wall_area[r] > 0.5 * tot_area[r]:             # curved wall (towers, tanks)
            c = pts[sel, :2].mean(axis=0)
            rad = float(np.linalg.norm(pts[sel, :2] - c, axis=1).mean())
            d = np.arctan2(pts[sel, 1] - c[1], pts[sel, 0] - c[0]) * rad     # arc length
        else:                                              # roof / floor / sloped roof
            continue
        L = float(d.max() - d.min())
        margin = (L - np.floor(L / style.bay_m) * style.bay_m) / 2.0
        u_m = d - d.min() - margin + rng.integers(0, TILE_BAYS) * style.bay_m
        v_m = pts[sel, 2] - z0[sel] + rng.integers(0, TILE_STOREYS) * style.storey_m
        uv[sel, 0] = u_m / tile_w
        uv[sel, 1] = v_m / tile_h
    return uv


def apply_facade(actor, style: FacadeStyle, textures: dict, seed: int = 0) -> bool:
    """Give a (split-vertex, triangulated) building actor facade UVs,
    tangents and the PBR texture set. Returns False if not applicable."""
    import pyvista as pv
    import vtk
    mapper = actor.GetMapper()
    data = mapper.GetInputDataObject(0, 0) if mapper is not None else None
    if data is None or data.GetNumberOfPoints() == 0:
        return False
    mesh = pv.wrap(data)
    if "Normals" not in mesh.point_data:
        return False
    if not mesh.is_all_triangles:          # tangent generation needs triangles
        mesh = mesh.triangulate()
        mapper.SetInputData(mesh)
    mesh.active_texture_coordinates = facade_uvs(mesh, style, seed).astype(np.float32)
    tan = vtk.vtkPolyDataTangents()
    tan.SetInputData(mesh)
    tan.Update()
    t_arr = tan.GetOutput().GetPointData().GetTangents()
    if t_arr is not None:
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        t_np = vtk_to_numpy(t_arr).astype(np.float32).copy()
        # Constant-UV faces (roofs) give degenerate/NaN tangents, which shade
        # black: replace with any unit vector orthogonal to the normal.
        nrm = np.asarray(mesh.point_data["Normals"], dtype=np.float32)
        bad = ~np.isfinite(t_np).all(axis=1) | (np.linalg.norm(t_np, axis=1) < 1e-6)
        if bad.any():
            ref = np.where(np.abs(nrm[bad, 2:3]) < 0.9, np.array([[0, 0, 1]], np.float32), np.array([[1, 0, 0]], np.float32))
            alt = np.cross(nrm[bad], ref)
            t_np[bad] = alt / np.maximum(np.linalg.norm(alt, axis=1, keepdims=True), 1e-6)
        out = numpy_to_vtk(np.ascontiguousarray(t_np), deep=True)
        out.SetName("Tangents")
        mesh.GetPointData().SetTangents(out)
    prop = actor.GetProperty()
    prop.SetInterpolationToPBR()
    prop.SetBaseColorTexture(textures["albedo"])
    prop.SetORMTexture(textures["orm"])
    prop.SetNormalTexture(textures["normal"])
    prop.SetEmissiveTexture(textures["emissive"])
    prop.SetEmissiveFactor(0.0, 0.0, 0.0)
    prop.SetRoughness(1.0)          # ORM texture drives roughness/metallic
    prop.SetMetallic(1.0)
    prop.SetOcclusionStrength(1.0)
    prop.SetNormalScale(1.0)
    return True
