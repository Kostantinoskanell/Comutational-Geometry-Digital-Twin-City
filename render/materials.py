"""Scene-wide PBR material migration (WG).

The twin was built with Phong materials whose colours are DISPLAY values,
plus a few PBR actors. PBR and Phong respond differently to the same light
(Lambert/pi in linear space + gamma vs. a display-space dot product), and
glossy PBR surfaces reflect the environment at its radiometric level, so a
mixed scene can never be consistently exposed.

SceneMaterials migrates every lit surface actor to PBR at render time,
without touching the ~80 call sites that create them:

  * base colour = sRGB->linear(display colour), so a fully sunlit face
    renders exactly the colour the call site asked for;
  * one exposure (PBR_EXPOSURE = pi) applied to every scene light and to the
    image-based lighting, which is what makes linear Lambert/pi reproduce
    the legacy intensity response; sky, reflections and sun are therefore
    exposed identically (the skybox uses the same gain);
  * textures move to the PBR base-colour slot, flagged sRGB;
  * Phong specular power maps to GGX roughness, sqrt(2 / (n + 2)).

Left alone, by design: unlit actors (orthophoto, analysis colour layers,
emissive markers: their colours are final), plain line / point actors
(VTK does not light them in Phong, measured identical at any intensity),
the skybox, and 2-D overlays.

Event-driven, not per-frame scanning (~800 actors): new actors are handled
when the renderer's actor collection changes; colours the app changes at
runtime (weather, style presets, night) are caught by a ModifiedEvent
observer on each migrated property and re-linearized.
"""
from __future__ import annotations

import numpy as np

PBR_EXPOSURE = float(np.pi)
_TOL = 1e-4


def srgb_to_linear(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def phong_to_roughness(specular: float, specular_power: float) -> float:
    """GGX roughness equivalent of a Blinn-Phong lobe (Walter et al. 2007
    mapping alpha = sqrt(2 / (n + 2))); matte when there was no specular."""
    if specular <= 1e-3:
        return 0.9
    return float(np.clip(np.sqrt(2.0 / (max(specular_power, 0.0) + 2.0)), 0.05, 1.0))


def _has_surfaces(actor) -> bool:
    mapper = actor.GetMapper()
    if mapper is None:
        return False
    data = mapper.GetInputDataObject(0, 0)
    if data is None:
        return False
    if data.IsA("vtkPolyData"):
        return data.GetNumberOfPolys() > 0 or data.GetNumberOfStrips() > 0
    return data.GetNumberOfCells() > 0


class SceneMaterials:
    def __init__(self, renderer, exposure: float = PBR_EXPOSURE):
        self.renderer = renderer
        self.exposure = float(exposure)
        self._actors: dict[str, dict] = {}     # address -> state
        self._lights: dict[str, float] = {}    # address -> intensity we set
        self._props: set[str] = set()           # properties already migrated (may be shared)
        self._collection_mtime = -1
        self._light_mtime = -1
        self._observer = None
        self.stats = {"migrated": 0, "textured": 0, "skipped_unlit": 0, "skipped_lines": 0}

    # ── lifecycle ────────────────────────────────────────────────────────
    def attach(self) -> None:
        if self._observer is None:
            self._observer = self.renderer.AddObserver("StartEvent", self._on_start)
        self.sync()

    def detach(self) -> None:
        if self._observer is not None:
            self.renderer.RemoveObserver(self._observer)
            self._observer = None

    def _on_start(self, *_):
        try:
            self.sync()
        except Exception as exc:          # never break a frame
            print(f"[materials] sync failed: {exc}")

    def _surface_actors(self):
        """vtkActors to render, including parts of vtkAssembly props (which
        renderer.GetActors() does not return)."""
        props = self.renderer.GetViewProps()
        props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is None:
                continue
            if prop.IsA("vtkAssembly"):
                paths = prop.GetParts()
                paths.InitTraversal()
                for _ in range(paths.GetNumberOfItems()):
                    part = paths.GetNextProp3D()
                    if part is not None and part.IsA("vtkActor"):
                        yield part
            elif prop.IsA("vtkActor"):
                yield prop

    def sync(self) -> None:
        actors = self.renderer.GetViewProps()
        if actors.GetMTime() != self._collection_mtime:
            self._collection_mtime = actors.GetMTime()
            present = set()
            for a in self._surface_actors():
                addr = a.GetAddressAsString("vtkActor")
                present.add(addr)
                if addr not in self._actors:
                    self._migrate(a, addr)
            # removed actors: VTK may reuse their addresses for new ones
            for addr in [k for k in self._actors if k not in present]:
                del self._actors[addr]
        lights = self.renderer.GetLights()
        if lights.GetMTime() != self._light_mtime or any(
                abs(l.GetIntensity() - self._lights.get(l.GetAddressAsString("vtkLight"), -1.0)) > _TOL
                for l in lights):
            self._light_mtime = lights.GetMTime()
            seen = {}
            for light in lights:
                addr = light.GetAddressAsString("vtkLight")
                mine = self._lights.get(addr)
                if mine is None or abs(light.GetIntensity() - mine) > _TOL:
                    # new light, or the app set a new (legacy-unit) intensity
                    light.SetIntensity(light.GetIntensity() * self.exposure)
                seen[addr] = light.GetIntensity()
            self._lights = seen

    def display_color(self, actor) -> tuple:
        """The display (sRGB) colour the app assigned to an actor, even after
        migration replaced its property colour with the linear value."""
        st = self._actors.get(actor.GetAddressAsString("vtkActor"))
        if st and st.get("done"):
            return tuple(st["display"])
        return tuple(actor.GetProperty().GetColor())

    # ── actors ───────────────────────────────────────────────────────────
    def _migrate(self, actor, addr: str) -> None:
        state = {"done": False}
        self._actors[addr] = state
        if actor.IsA("vtkSkybox"):
            return
        prop = actor.GetProperty()
        paddr = prop.GetAddressAsString("vtkProperty")
        if paddr in self._props:           # shared property: already linear + observed
            return
        if not prop.GetLighting():
            self.stats["skipped_unlit"] += 1
            return
        tex = actor.GetTexture()
        if not _has_surfaces(actor) and not (prop.GetRenderLinesAsTubes() or prop.GetRenderPointsAsSpheres()):
            self.stats["skipped_lines"] += 1
            return
        if prop.GetInterpolation() != 3:                     # VTK_PBR == 3
            prop.SetMetallic(0.0)
            prop.SetRoughness(phong_to_roughness(prop.GetSpecular(), prop.GetSpecularPower()))
            prop.SetInterpolationToPBR()
        if tex is not None:
            tex.SetUseSRGBColorSpace(True)
            prop.SetBaseColorTexture(tex)
            actor.SetTexture(None)
            self.stats["textured"] += 1
        self._linearize_lut(actor)
        self._linearize_direct_scalars(actor)
        state["display"] = tuple(prop.GetColor())
        state["linear"] = tuple(srgb_to_linear(state["display"]))
        state["busy"] = False
        prop.SetColor(*state["linear"])
        state["obs"] = prop.AddObserver("ModifiedEvent", lambda o, e, s=state: self._on_prop_modified(o, s))
        state["done"] = True
        self._props.add(paddr)
        self.stats["migrated"] += 1

    def _on_prop_modified(self, prop, state) -> None:
        if state.get("busy"):
            return
        c = tuple(prop.GetColor())
        if max(abs(a - b) for a, b in zip(c, state["linear"])) <= _TOL:
            return
        # the app assigned a new display colour -> linearize it
        state["busy"] = True
        try:
            state["display"] = c
            state["linear"] = tuple(srgb_to_linear(c))
            prop.SetColor(*state["linear"])
        finally:
            state["busy"] = False

    @staticmethod
    def _linearize_direct_scalars(actor) -> None:
        """Direct RGB(A) scalars (e.g. the lighting-analysis ground colours)
        are display colours; point the mapper at a linearized float copy."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        mapper = actor.GetMapper()
        if mapper is None or not mapper.GetScalarVisibility() or mapper.GetColorMode() != vtk.VTK_COLOR_MODE_DIRECT_SCALARS:
            return
        data = mapper.GetInputDataObject(0, 0)
        if data is None:
            return
        flag = vtk.reference(0)
        arr = vtk.vtkAbstractMapper.GetScalars(data, mapper.GetScalarMode(), mapper.GetArrayAccessMode(),
                                               mapper.GetArrayId(), mapper.GetArrayName(), flag)
        if arr is None or arr.GetNumberOfComponents() not in (3, 4) or (arr.GetName() or "").endswith("_twinlin"):
            return
        a = vtk_to_numpy(arr).astype(np.float32)
        if arr.GetDataType() in (vtk.VTK_UNSIGNED_CHAR, vtk.VTK_CHAR, vtk.VTK_SIGNED_CHAR):
            a = a / 255.0
        a[:, :3] = srgb_to_linear(a[:, :3])
        out = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
        name = f"{arr.GetName() or 'rgb'}_twinlin"
        out.SetName(name)
        fields = data.GetCellData() if int(flag) == 1 else data.GetPointData()
        fields.AddArray(out)
        mapper.SetScalarMode(vtk.VTK_SCALAR_MODE_USE_CELL_FIELD_DATA if int(flag) == 1
                             else vtk.VTK_SCALAR_MODE_USE_POINT_FIELD_DATA)
        mapper.SelectColorArray(name)

    @staticmethod
    def _linearize_lut(actor) -> None:
        """Colormapped scalars: the LUT holds display colours; PBR reads them
        as linear albedo. Direct RGB scalars are left as-is (no VTK flag)."""
        mapper = actor.GetMapper()
        if mapper is None or not mapper.GetScalarVisibility():
            return
        lut = mapper.GetLookupTable()
        if lut is None or not lut.IsA("vtkLookupTable") or getattr(lut, "_twin_linear", False):
            return
        n = lut.GetNumberOfTableValues()
        for i in range(n):
            r, g, b, a = lut.GetTableValue(i)
            lr, lg, lb = srgb_to_linear((r, g, b))
            lut.SetTableValue(i, float(lr), float(lg), float(lb), a)
        try:
            lut._twin_linear = True
        except Exception:
            pass
