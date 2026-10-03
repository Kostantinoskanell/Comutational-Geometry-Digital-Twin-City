"""Physically based floodwater surface for the photoreal view (WG).

The analysis puddle overlay is a colour-mapped depth grid (unlit, legend
colours). This is its photoreal counterpart, built on the same flood grid:

  * geometry: the water SURFACE, z = ground + depth (the flood engine's h),
    updated in place during the time-lapse and lifted with the terrain drape;
  * optics (two regimes, see constants): a shallow film darkens the wet
    ground beneath it and adds sky sheen; with depth the turbid sediment
    body colour takes over (Beer–Lambert); dry cells are invisible;
  * surface: a smooth dielectric (IOR 1.33) that reflects the physical sky
    through IBL with the right Fresnel falloff, plus a tileable procedural
    ripple normal map drifting with the wind (rain/wind chop).

Colours are written already linear (array name ends in "_twinlin"), so the
scene material manager leaves this actor alone.
"""
from __future__ import annotations

import numpy as np

# Two optical regimes (what flood photographs show):
#  * shallow sheet flow (cm): the ground under a water film is DARKENED —
#    wet surfaces drop to ~50-70 % of their dry albedo (Lekner & Dorf 1988)
#    — plus the Fresnel sky sheen of a smooth surface;
#  * deeper water: the suspended-sediment body colour takes over
#    (Beer-Lambert over a turbid e-folding depth). Sediment-laden
#    floodwater is optically BRIGHT (irradiance reflectance ~0.15-0.3 in
#    the red): light muddy brown.
E_FOLD_M = 0.15                          # sediment-colour depth scale (urban stormwater)
DRY_M = 0.005
WET_FILM_ALPHA = 0.45                    # darkening of wet ground under any water film
WET_FILM_LIN = (0.02, 0.02, 0.022)       # near-black film => ~55 % of the ground albedo shows
TURBID_ALBEDO_LIN = (0.40, 0.24, 0.11)     # orange-brown, readable against grey asphalt and ortho (checked in the app)
ROUGHNESS = 0.04
RIPPLE_TILE_M = 6.0
RIPPLE_NORMAL_SCALE = 0.35
WIND_DRIFT_M_S = (0.35, 0.12)


def ripple_normal_map(px: int = 256, seed: int = 3) -> np.ndarray:
    """Tileable tangent-space normal map (uint8 RGB, row 0 = top) of a
    wind-chop height field: sum of sinusoids with INTEGER wave numbers over
    the tile, so it repeats seamlessly."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:px, 0:px] / px
    h = np.zeros((px, px))
    gx = np.zeros((px, px))
    gy = np.zeros((px, px))
    for _ in range(24):
        kx, ky = rng.integers(-9, 10, 2)
        if kx == 0 and ky == 0:
            continue
        k = np.hypot(kx, ky)
        amp = 1.0 / k ** 1.5                     # capillary/short-wave spectrum falloff
        ph = rng.uniform(0, 2 * np.pi)
        arg = 2 * np.pi * (kx * x + ky * y) + ph
        h += amp * np.sin(arg)
        gx += amp * 2 * np.pi * kx * np.cos(arg)
        gy += amp * 2 * np.pi * ky * np.cos(arg)
    s = 0.04                                      # slope scale
    n = np.stack([-gx * s, -gy * s, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip(np.round(np.flipud((n + 1) / 2) * 255), 0, 255).astype(np.uint8)


def sediment_fraction(depth: np.ndarray) -> np.ndarray:
    """Share of the water colour coming from the turbid body (Beer-Lambert)."""
    d = np.maximum(np.asarray(depth, dtype=np.float32), 0.0)
    return (1.0 - np.exp(-d / E_FOLD_M)).astype(np.float32)


def water_alpha(depth: np.ndarray) -> np.ndarray:
    """Coverage: wet-film darkening at any depth, rising to opaque with the
    sediment body; 0 (fully transparent) on dry cells."""
    d = np.asarray(depth, dtype=np.float32)
    f = sediment_fraction(d)
    a = WET_FILM_ALPHA + (1.0 - WET_FILM_ALPHA) * f
    return np.where(d > DRY_M, a, 0.0).astype(np.float32)


def water_rgb_linear(depth: np.ndarray) -> np.ndarray:
    """(N, 3) linear colour: dark wet film -> muddy body with depth."""
    f = sediment_fraction(depth)[:, None]
    return (np.asarray(WET_FILM_LIN)[None, :] * (1 - f) + np.asarray(TURBID_ALBEDO_LIN)[None, :] * f).astype(np.float32)


class FloodWaterSurface:
    def __init__(self, plotter, nx: int, ny: int, x0: float, y0: float, dx: float, dy: float,
                 z_offset: float, depth: np.ndarray, name: str = "flood_water_surface", xy: np.ndarray | None = None):
        """Axis-aligned grid from (x0, y0, dx, dy), or — for the solver's UTM-aligned
        raster in the twin's rotated local frame — pass `xy` (nx*ny, 2), row-major."""
        import pyvista as pv
        self.plotter = plotter
        if xy is not None:
            self._xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        else:
            xs = x0 + np.arange(nx) * dx
            ys = y0 + np.arange(ny) * dy
            gx, gy = np.meshgrid(xs, ys)
            self._xy = np.column_stack([gx.ravel(), gy.ravel()])
        self._z_offset = float(z_offset)
        self._base = np.zeros(len(self._xy))            # terrain drape base (0 when flat)
        self._depth = np.zeros(len(self._xy), dtype=np.float32)
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([self._xy, np.full(len(self._xy), self._z_offset)])
        grid.dimensions = (nx, ny, 1)
        rgba = np.zeros((len(self._xy), 4), dtype=np.float32)
        rgba[:, :3] = TURBID_ALBEDO_LIN
        grid.point_data["water_twinlin"] = rgba
        self._tc0 = (self._xy / RIPPLE_TILE_M).astype(np.float32)
        grid.point_data["ripple_tc"] = self._tc0.copy()
        grid.point_data.active_texture_coordinates_name = "ripple_tc"
        self.grid = grid
        surf = grid.extract_surface(algorithm="dataset_surface").triangulate()   # tangents need triangles
        surf = surf.compute_normals(point_normals=True, cell_normals=False, split_vertices=False)
        self.surf = surf
        self._surf_ids = np.asarray(surf.point_data["vtkOriginalPointIds"]) if "vtkOriginalPointIds" in surf.point_data \
            else np.arange(surf.n_points)
        tex = pv.Texture(np.ascontiguousarray(ripple_normal_map()))
        tex.SetRepeat(True)
        tex.SetMipmap(True)
        tex.SetInterpolate(True)
        self.actor = plotter.add_mesh(surf, scalars="water_twinlin", rgba=True, pbr=True, metallic=0.0,
                                      roughness=ROUGHNESS, smooth_shading=True, show_scalar_bar=False,
                                      reset_camera=False, name=name)
        # pyvista's rgba=True leaves FLOAT arrays in "Default" colour mode
        # (mapped through a lookup table: alpha and colour lost) — force
        # direct RGBA so the Beer-Lambert opacity is what is drawn.
        mapper = self.actor.GetMapper()
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("water_twinlin")
        prop = self.actor.GetProperty()
        prop.SetBaseIOR(1.33)
        try:
            import vtk
            t = vtk.vtkPolyDataTangents()
            t.SetInputData(surf)
            t.Update()
            if t.GetOutput().GetPointData().GetTangents() is not None:
                surf.GetPointData().SetTangents(t.GetOutput().GetPointData().GetTangents())
            prop.SetNormalTexture(tex)
            prop.SetNormalScale(RIPPLE_NORMAL_SCALE)
        except Exception:
            pass
        self.actor.PickableOff()
        self.set_depth(depth)

    # ── state updates ────────────────────────────────────────────────────
    def set_depth(self, depth: np.ndarray) -> None:
        self._depth = np.asarray(depth, dtype=np.float32).ravel()
        self._push()

    def set_base(self, base_z: np.ndarray | None) -> None:
        """Terrain drape base under each grid point (None/zeros = flat)."""
        self._base = np.zeros(len(self._xy)) if base_z is None else np.asarray(base_z, dtype=float).ravel()
        self._push()

    def advance(self, t_seconds: float) -> None:
        """Wind drift of the ripple pattern."""
        drift = np.asarray(WIND_DRIFT_M_S, dtype=np.float32) * float(t_seconds) / RIPPLE_TILE_M
        tc = self.surf.point_data["ripple_tc"]
        tc[:] = self._tc0[self._surf_ids] + drift
        self.surf.GetPointData().GetArray("ripple_tc").Modified()
        self.surf.Modified()

    def _push(self) -> None:
        d = self._depth[self._surf_ids]
        pts = self.surf.points
        pts[:, 2] = self._z_offset + self._base[self._surf_ids] + np.maximum(d, 0.0)
        self.surf.GetPoints().Modified()
        rgba = self.surf.point_data["water_twinlin"]
        rgba[:, :3] = water_rgb_linear(d)
        rgba[:, 3] = water_alpha(d)
        self.surf.GetPointData().GetArray("water_twinlin").Modified()
        self.surf.Modified()

    @property
    def xy(self) -> np.ndarray:
        return self._xy

    def set_visible(self, on: bool) -> None:
        self.actor.SetVisibility(bool(on))
