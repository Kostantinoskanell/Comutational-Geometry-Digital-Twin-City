"""Post-processing render-pass chain for the twin (WG lighting).

One owner for the renderer's pass graph, rebuilt from fresh VTK objects on
every change:

    3D image:  camera( render steps without overlay, translucency )
               -> SSAO (optional) -> filmic tone mapping -> FXAA
    then:      overlay pass (text / 2D widgets / control panel)

The UI overlay is drawn AFTER tone mapping and anti-aliasing, so panel
colours stay exact and text stays crisp while the 3D scene gets an HDR
filmic response (the physical sky and PBR lighting exceed [0, 1]).

Chains are built from fresh VTK objects and cached per configuration, so
switching (e.g. adaptive SSAO during navigation) is instant; invalidate()
rebuilds from scratch — this also sidesteps the renderer's built-in SSAO
failure mode (after heavy actor churn it presents its normals G-buffer).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PostFXSettings:
    ssao: bool = False
    tone_mapping: bool = True
    fxaa: bool = True
    exposure: float = 1.0
    ssao_radius: float = 4.0
    ssao_bias: float = 0.025
    ssao_kernel: int = 32         # measured: visually identical to 128 (mean |diff| 1.5/255) at ~1/4 the cost
    ssao_blur: bool = True
    # GenericFilmic with the Uncharted-2 preset: a filmic shoulder that keeps
    # bright sky/sunlit facades from clipping; ACES-like but hue-preserving.
    filmic_preset: str = "uncharted2"
    # "plain" = VTK's default translucent pass (same look as the legacy path);
    # "oit" = weighted blended order-independent translucency. OIT is never
    # combined with SSAO: in VTK 9.6 the pair renders translucent layers
    # (tube-rendered road lines, overlays) as opaque white (found in the twin).
    translucency: str = "plain"
    extra: dict = field(default_factory=dict)


class PostFX:
    def __init__(self, renderer, settings: PostFXSettings | None = None):
        self.renderer = renderer
        self.settings = settings or PostFXSettings()
        self._passes: dict[str, object] = {}
        self._chains: dict[tuple, dict] = {}    # built chains, reused on switch (no shader rebuild)

    def active(self) -> bool:
        s = self.settings
        return bool(s.ssao or s.tone_mapping or s.fxaa)

    def apply(self) -> None:
        """(Re)build the pass graph from the current settings."""
        import vtk
        r = self.renderer
        s = self.settings
        try:
            r.SetUseSSAO(False)       # the chain owns SSAO; never both
            r.SetUseFXAA(False)
        except Exception:
            pass
        if not self.active():
            r.SetPass(None)
            self._passes = {}
            return
        key = self._key()
        if key in self._chains:          # instant switch: GPU resources stay warm
            self._passes = self._chains[key]
            r.SetPass(self._passes["seq"])
            return

        steps = vtk.vtkRenderStepsPass()
        if s.translucency == "oit" and not s.ssao:
            oit = vtk.vtkOrderIndependentTranslucentPass()
            oit.SetTranslucentPass(steps.GetTranslucentPass())
            steps.SetTranslucentPass(oit)
        overlay = steps.GetOverlayPass()
        steps.SetOverlayPass(None)            # drawn after post-processing
        camera = vtk.vtkCameraPass()
        camera.SetDelegatePass(steps)
        current = camera

        if s.ssao:
            ssao = vtk.vtkSSAOPass()
            ssao.SetRadius(float(s.ssao_radius))
            ssao.SetBias(float(s.ssao_bias))
            ssao.SetKernelSize(int(s.ssao_kernel))
            ssao.SetBlur(bool(s.ssao_blur))
            ssao.SetDelegatePass(current)
            current = ssao
        if s.tone_mapping:
            tm = vtk.vtkToneMappingPass()
            tm.SetToneMappingType(vtk.vtkToneMappingPass.GenericFilmic)
            if s.filmic_preset == "uncharted2":
                tm.SetGenericFilmicUncharted2Presets()
            else:
                tm.SetGenericFilmicDefaultPresets()
            tm.SetExposure(float(s.exposure))
            tm.SetDelegatePass(current)
            current = tm
        if s.fxaa:
            aa = vtk.vtkOpenGLFXAAPass()
            aa.SetDelegatePass(current)
            current = aa

        passes = vtk.vtkRenderPassCollection()
        passes.AddItem(current)
        if overlay is not None:
            passes.AddItem(overlay)
        seq = vtk.vtkSequencePass()
        seq.SetPasses(passes)
        r.SetPass(seq)
        self._passes = {"steps": steps, "camera": camera, "final": current, "overlay": overlay, "seq": seq}
        self._chains[key] = self._passes

    def _key(self) -> tuple:
        s = self.settings
        return (bool(s.ssao), bool(s.tone_mapping), bool(s.fxaa), round(float(s.exposure), 4),
                float(s.ssao_radius), float(s.ssao_bias), int(s.ssao_kernel), bool(s.ssao_blur),
                s.filmic_preset, s.translucency if not s.ssao else "plain")

    def invalidate(self) -> None:
        """Release every cached chain (e.g. after heavy scene rebuilds) and
        rebuild the current one from fresh passes."""
        for chain in list(self._chains.values()):
            self._release(chain)
        self._chains = {}
        self.apply()

    def _release(self, chain: dict) -> None:
        """Free a chain's GPU resources before dropping it (VTK image passes
        otherwise leak their FBOs/textures)."""
        win = self.renderer.GetRenderWindow()
        for key in ("final", "camera", "steps", "overlay"):
            node = chain.get(key)
            if key == "final":
                # walk the image-processing chain back to the camera pass
                while node is not None and node is not chain.get("camera"):
                    if win is not None:
                        node.ReleaseGraphicsResources(win)
                    node = node.GetDelegatePass() if hasattr(node, "GetDelegatePass") else None
            elif node is not None and win is not None:
                node.ReleaseGraphicsResources(win)

    def update(self, **changes) -> None:
        for k, v in changes.items():
            if not hasattr(self.settings, k):
                raise AttributeError(f"unknown post-fx setting '{k}'")
            setattr(self.settings, k, v)
        self.apply()
