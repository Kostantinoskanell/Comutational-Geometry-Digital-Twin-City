"""GPU-instanced car fleet for the detailed ("ultra") car models (WG perf).

The ultra mode used one vtkActor (or vtkAssembly) per car — ~600 draw calls
and a Python SetPosition/SetOrientation loop over every car each tick.
InstancedFleet renders every car of one model with a single
vtkGlyph3DMapper (hardware instancing): per-instance position, heading
(rotation array) and body colour (direct RGB scalars). Multi-material
templates (the solar car: list of (poly, property, texture) parts) get one
instanced mapper per part, all sharing the same instance centres.

Draw calls: n_models (+ extra parts) instead of n_cars. Per tick: one numpy
gather per model; no VTK objects created.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk


def _instanced_actor(centres: pv.PolyData, source, colour_array: str | None):
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(centres)
    mapper.SetSourceData(source)
    mapper.SetScaleModeToNoDataScaling()
    mapper.SetOrientationModeToRotation()
    mapper.SetOrientationArray("rotation")
    mapper.OrientOn()
    if colour_array:
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(colour_array)
        mapper.SetColorModeToDirectScalars()
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


class InstancedFleet:
    """A fixed-size fleet of cars rendered with one instanced draw per model.

    templates : list of pv.PolyData (single material, tinted per car) or
                list[(poly, vtkProperty, vtkTexture|None)] (multi-material).
    model_idx : (n,) template index per car.
    colours   : (n, 3) uint8 display-RGB body colour per car (ignored for
                multi-material templates, which keep their own materials).
    """

    def __init__(self, plotter, templates, model_idx, colours, positions, headings_deg):
        self.plotter = plotter
        self.n = int(len(model_idx))
        self.model_idx = np.asarray(model_idx, dtype=np.int64) % max(1, len(templates))
        self._pos = np.asarray(positions, dtype=float).reshape(self.n, 3).copy()
        self._hdg = np.asarray(headings_deg, dtype=float).reshape(self.n).copy()
        self._groups = []            # (indices, centres, [actors])
        self.actors = []
        colours = np.asarray(colours, dtype=np.uint8).reshape(self.n, 3)
        for m, tmpl in enumerate(templates):
            idx = np.flatnonzero(self.model_idx == m)
            if idx.size == 0:
                continue
            centres = pv.PolyData(self._pos[idx].copy())
            rot = np.zeros((idx.size, 3), dtype=np.float32)
            rot[:, 2] = self._hdg[idx]
            centres.point_data["rotation"] = rot
            actors = []
            if isinstance(tmpl, list):                       # multi-material parts
                for poly, prop, tex in tmpl:
                    a = _instanced_actor(centres, poly, None)
                    a.SetProperty(prop)
                    if tex:
                        a.SetTexture(tex)
                    actors.append(a)
            else:
                centres.point_data["rgb"] = colours[idx]
                src = vtk.vtkPolyData()
                src.DeepCopy(tmpl)
                a = _instanced_actor(centres, src, "rgb")
                prop = a.GetProperty()
                prop.SetInterpolationToPhong()
                prop.SetColor(1.0, 1.0, 1.0)
                actors.append(a)
            for a in actors:
                plotter.renderer.AddActor(a)
            self._groups.append((idx, centres, actors))
            self.actors.extend(actors)

    # ── per tick ─────────────────────────────────────────────────────────
    def update(self, positions, headings_deg=None) -> None:
        pos = np.asarray(positions, dtype=float)
        n = min(self.n, pos.shape[0])
        self._pos[:n] = pos[:n]
        if headings_deg is not None:
            h = np.asarray(headings_deg, dtype=float)
            self._hdg[:n] = h[:n]
        self._push()

    def _push(self) -> None:
        for idx, centres, _ in self._groups:
            centres.points[:] = self._pos[idx]                     # views on the VTK buffers
            centres.point_data["rotation"][:, 2] = self._hdg[idx]
            centres.GetPoints().Modified()
            centres.GetPointData().GetArray("rotation").Modified()
            centres.Modified()

    @property
    def positions(self) -> np.ndarray:
        return self._pos

    def set_z(self, z: np.ndarray) -> None:
        """Absolute z per car (terrain drape / restore)."""
        self._pos[:, 2] = np.asarray(z, dtype=float)[: self.n]
        self._push()

    def set_visible(self, on: bool) -> None:
        for a in self.actors:
            a.SetVisibility(bool(on))

    def remove(self) -> None:
        for a in self.actors:
            self.plotter.renderer.RemoveActor(a)
        self.actors = []
        self._groups = []
