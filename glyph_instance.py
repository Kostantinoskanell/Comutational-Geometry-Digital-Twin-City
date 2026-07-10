"""glyph_instance.py — single-draw-call instanced rendering for agent classes.

One GlyphInstances object replaces N individual VTK actors. The mapper reads
one vtkPoints array (N xyz positions) and one per-point float3 array named
"rotation" (Euler angles 0,0,heading_deg) and renders the template geometry
at every point. Only those two arrays are mutated each tick via numpy; no
actors or meshes are created or deleted during animation.

Usage
-----
    gi = GlyphInstances(template_polydata, initial_n=50,
                        color="#d4a574", plotter=self.plotter)

    # each animation tick:
    gi.update(positions_array,   # shape (n, 3) float
              headings_deg)      # shape (n,)   float  — degrees around Z

    gi.set_visible(True)
    gi.remove(plotter)           # cleanup only
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

try:
    import vtk as _vtk
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False


_SENTINEL = 1e8          # off-screen coordinate for unused pool slots
_HEADROOM_FRAC = 0.25   # grow pool by 25 % when capacity is exceeded


class GlyphInstances:
    """Single-draw-call glyph mapper for a homogeneous group of agents.

    The underlying pipeline is:
        vtkPolyData (centers: N points + "rotation" float3 array)
            → vtkGlyph3DMapper  (source = template geometry, mode = Rotation)
            → vtkActor

    Parameters
    ----------
    template : pv.PolyData
        Source geometry placed at every agent position.  The template's
        "forward" direction should face +X and it should be centred at the
        origin (the mapper applies the Z-rotation before translation).
    initial_n : int
        Initial pool capacity.  The pool grows automatically when more
        agents arrive; it never shrinks.
    color : str
        Hex colour string passed to pv.Color.
    plotter : pv.Plotter
        The scene plotter whose renderer receives the actor.
    smooth : bool
        Phong interpolation on the actor property (default True).
    """

    def __init__(
        self,
        template: pv.PolyData,
        initial_n: int,
        color: str,
        plotter,
        smooth: bool = True,
    ) -> None:
        if not _HAS_VTK:
            raise ImportError("vtk is required for GlyphInstances")

        cap = max(1, initial_n + max(8, int(initial_n * _HEADROOM_FRAC)))

        # Buffers reused every tick — allocated once, grown on demand
        self._pts_buf = np.full((cap, 3), _SENTINEL, dtype=np.float64)
        self._rot_buf = np.zeros((cap, 3), dtype=np.float32)
        self._capacity = cap
        self._n_active = 0

        # ── Centers PolyData (PyVista for efficient numpy ↔ VTK) ─────────────
        self._centers = pv.PolyData(self._pts_buf.copy())
        self._centers["rotation"] = self._rot_buf.copy()

        # ── Keep a VTK copy of the template so Python GC can't drop it ────────
        self._src = _vtk.vtkPolyData()
        self._src.DeepCopy(template)

        # ── Mapper ─────────────────────────────────────────────────────────────
        self._mapper = _vtk.vtkGlyph3DMapper()
        self._mapper.SetInputData(self._centers)
        self._mapper.SetSourceData(self._src)
        self._mapper.SetScaleModeToNoDataScaling()
        self._mapper.SetOrientationModeToRotation()   # array → Euler (rx,ry,rz)
        self._mapper.SetOrientationArray("rotation")
        self._mapper.OrientOn()
        self._mapper.ScalarVisibilityOff()            # use actor property colour, not source arrays

        # ── Actor ──────────────────────────────────────────────────────────────
        self._actor = _vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        rgb = pv.Color(color).float_rgb
        prop = self._actor.GetProperty()
        prop.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        if smooth:
            prop.SetInterpolationToPhong()

        plotter.renderer.AddActor(self._actor)

    # ------------------------------------------------------------------
    # Per-tick update  (the only method called in the hot path)
    # ------------------------------------------------------------------

    def update(
        self,
        positions: np.ndarray,          # (n, 3) float  — agent XYZ positions
        headings_deg: np.ndarray | None = None,  # (n,) float  — heading °
        tilt_x_deg: np.ndarray | None = None,    # (n,) float — X tilt (90 = lying down)
    ) -> None:
        """Mutate the glyph centers in-place; no VTK objects are created."""
        n = int(positions.shape[0]) if positions is not None else 0

        if n > self._capacity:
            self._grow(n)

        self._n_active = n

        if n > 0:
            # Fill active slots — numpy ops, no Python loop
            np.copyto(self._pts_buf[:n], positions, casting="unsafe")
            if headings_deg is not None:
                h = np.asarray(headings_deg, dtype=np.float32)
                np.copyto(self._rot_buf[:n, 2], h[:n], casting="unsafe")
            else:
                self._rot_buf[:n, 2] = 0.0
            if tilt_x_deg is not None:
                t = np.asarray(tilt_x_deg, dtype=np.float32)
                np.copyto(self._rot_buf[:n, 0], t[:n], casting="unsafe")
            else:
                self._rot_buf[:n, 0] = 0.0

        # Blank unused pool slots (move them off-screen)
        self._pts_buf[n:] = _SENTINEL
        self._rot_buf[n:] = 0.0

        # Push to VTK — PyVista handles numpy → vtkFloatArray conversion + Modified
        self._centers.points = self._pts_buf
        self._centers["rotation"] = self._rot_buf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grow(self, needed: int) -> None:
        new_cap = needed + max(8, int(needed * _HEADROOM_FRAC))
        new_pts = np.full((new_cap, 3), _SENTINEL, dtype=np.float64)
        new_rot = np.zeros((new_cap, 3), dtype=np.float32)
        # Copy over current active data
        if self._n_active > 0:
            new_pts[:self._n_active] = self._pts_buf[:self._n_active]
            new_rot[:self._n_active] = self._rot_buf[:self._n_active]
        self._pts_buf  = new_pts
        self._rot_buf  = new_rot
        self._capacity = new_cap
        # Rebuild centers polydata with new capacity (rare — only on grow)
        self._centers = pv.PolyData(new_pts.copy())
        self._centers["rotation"] = new_rot.copy()
        self._mapper.SetInputData(self._centers)

    def set_visible(self, visible: bool) -> None:
        self._actor.SetVisibility(int(visible))

    def remove(self, plotter) -> None:
        try:
            plotter.renderer.RemoveActor(self._actor)
        except Exception:
            pass
