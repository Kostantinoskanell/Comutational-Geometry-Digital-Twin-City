"""LightingMixin — physical sky + post-processing for the twin (WG).

  _init_postfx()              after the plotter exists: owns the render-pass
                              chain (render.postfx); activated by the same
                              deferred timer that used to switch on SSAO.
  _apply_sky_environment()    per time-of-day update: physically based sky
                              (render.sky) as IBL environment + background
                              skybox, used when no local HDRI asset exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

QUALITY_PRESETS = {
    # SSAO is the expensive term; FXAA is a ~free full-screen pass.
    # Filmic tone mapping stays OFF here: much of the scene is still
    # display-referred (unlit orthophoto with baked sun, flat-colour
    # analysis layers), and every VTK filmic curve compresses those
    # (1.0 -> ~0.8, measured). It belongs to a fully scene-referred PBR
    # preset, not to this mixed pipeline.
    "performance": dict(ssao=False, tone_mapping=False, fxaa=True),
    "quality":     dict(ssao=True, tone_mapping=False, fxaa=True),
    "legacy":      dict(ssao=False, tone_mapping=False, fxaa=False),
}
# Legacy (no material migration) background display: the skybox is drawn straight into the display-referred
# framebuffer (no PBR gamma), so its texture is exposed and sRGB-encoded here.
# Exposure follows the app's convention that a sunlit white surface shows at
# ~0.9: that surface's radiance is T_sun * cos / pi ~ 0.72 / pi at noon, so
# gain = 0.9 * pi / 0.72 ~ 3.9. IBL keeps the linear radiometric env.
SKYBOX_DISPLAY_GAIN = 0.9 * np.pi / 0.72


# Deep clear-water diffuse reflectance (linear), from ocean-optics
# irradiance reflectance R ~ pi * Rrs (Rrs ~0.003/0.008/0.02 sr^-1 in R/G/B
# for oligotrophic Mediterranean water): dark and blue-dominant; the bright
# look at grazing angles comes from the Fresnel sky reflection.
WATER_ALBEDO = (0.01, 0.03, 0.07)
WATER_ROUGHNESS = 0.08              # wind-roughened sea, keeps the sun glint broad


class LightingMixin:

    def _init_postfx(self) -> None:
        from render.postfx import PostFX, PostFXSettings
        self.postfx = None
        self.materials = None
        quality = str(getattr(self.args, "render_quality", "quality"))
        if quality == "legacy" or self.plotter is None:
            return
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["quality"])
        self.postfx = PostFX(self.renderer, PostFXSettings(**preset))
        from render.materials import SceneMaterials
        self.materials = SceneMaterials(self.renderer)
        self.materials.attach()
        self._postfx_ssao_wanted = bool(preset["ssao"])
        # Chain starts with SSAO off; the deferred-activation timer turns the
        # whole chain on after the first frames (macOS first-render hang).
        self.postfx.settings.ssao = False
        print(f"[postfx] render quality '{quality}': "
              f"SSAO {'deferred' if preset['ssao'] else 'off'}, FXAA"
              + (", filmic tone mapping" if preset["tone_mapping"] else ""))

    def _postfx_activate(self) -> bool:
        """Deferred activation; True if the chain handled it."""
        fx = getattr(self, "postfx", None)
        if fx is None:
            return False
        fx.update(ssao=bool(getattr(self, "_postfx_ssao_wanted", False)))
        self._install_adaptive_quality()
        return True

    def _install_adaptive_quality(self) -> None:
        """Interactive LOD: SSAO (the dominant full-screen cost, ~10 ms at
        Retina size) is suspended while the user drags the camera and
        restored when the view settles; both chains are cached, so the
        switch is instant."""
        iren = getattr(self.plotter, "iren", None)
        if iren is None or getattr(self, "_adaptive_obs", None):
            return
        inter = getattr(iren, "interactor", iren)

        def start(*_):
            fx = getattr(self, "postfx", None)
            if fx is not None and fx.settings.ssao:
                self._adaptive_suspended = True
                fx.update(ssao=False)

        def end(*_):
            fx = getattr(self, "postfx", None)
            if fx is not None and getattr(self, "_adaptive_suspended", False):
                self._adaptive_suspended = False
                fx.update(ssao=bool(getattr(self, "_postfx_ssao_wanted", False)))
                try:
                    self.plotter.render()
                except Exception:
                    pass
        try:
            self._adaptive_obs = (inter.AddObserver("StartInteractionEvent", start),
                                  inter.AddObserver("EndInteractionEvent", end))
        except Exception as exc:
            print(f"[postfx] adaptive quality unavailable: {exc}")
            self._adaptive_obs = ()

    def _postfx_ssao_on(self) -> bool:
        fx = getattr(self, "postfx", None)
        if fx is not None:
            return bool(fx.settings.ssao)
        try:
            return bool(self.renderer.GetUseSSAO())
        except Exception:
            return False

    def _postfx_set_ssao(self, on: bool) -> bool:
        fx = getattr(self, "postfx", None)
        if fx is None:
            return False
        self._postfx_ssao_wanted = bool(on)
        self._adaptive_suspended = False
        fx.update(ssao=bool(on))
        return True

    # ── procedural facades (render.facades) ──────────────────────────────
    _FACADE_CLASS_BY_NAME = {"concrete": 0, "brick": 1, "glass": 2, "commercial": 3, "residential": 4}

    def _init_facades(self) -> None:
        """Procedural PBR facades on the building walls (after the building
        actors exist, before any terrain drape: storeys start at flat z=0)."""
        self._facade_actors = []           # (actor, class tint display rgb)
        # Needs PBR materials AND image-based lighting: VTK's PBR path without
        # an environment falls back to a bright default and drops emission.
        if (getattr(self, "materials", None) is None or bool(getattr(self.args, "no_facades", False))
                or bool(getattr(self.args, "no_physical_sky", False))):
            return
        from render.facades import STYLES, apply_facade, facade_textures
        tex_cache = {}

        def textures(cid):
            if cid not in tex_cache:
                tex_cache[cid] = facade_textures(STYLES[cid], seed=cid)
            return tex_cache[cid]

        pbr = self.scene_state.get("_building_actors_pbr") or {}
        for name, actor in pbr.items():
            cid = self._FACADE_CLASS_BY_NAME.get(str(name))
            if cid is None:
                continue
            tint = self.materials.display_color(actor)
            if apply_facade(actor, STYLES[cid], textures(cid), seed=cid):
                self._facade_actors.append((actor, tint))
        walls = self.scene_state.get("survey_struct_walls_actor")
        if walls is not None:
            tint = self.materials.display_color(walls)
            if apply_facade(walls, STYLES[4], textures(4), seed=104):
                self._facade_actors.append((walls, tint))
        self._set_facade_tint()
        if self._facade_actors:
            print(f"[facades] procedural PBR facades on {len(self._facade_actors)} building layer(s) "
                  f"({len(tex_cache)} styles)")

    def _set_facade_tint(self) -> None:
        """White (texture colour) in the photoreal view; the building-class
        colour as a tint in the analysis view, keeping class coding readable.
        Without a survey there is no analysis/photoreal split: realistic."""
        analysis = (self.scene_state.get("survey_ground_actor") is not None
                    and not bool(self.scene_state.get("photoreal_ground", False)))
        for actor, tint in getattr(self, "_facade_actors", []):
            actor.GetProperty().SetColor(*(tint if analysis else (1.0, 1.0, 1.0)))

    def _set_facade_emission(self, level: float) -> bool:
        """Lit-window emission 0..1 (time-of-day). True if facades exist."""
        acts = getattr(self, "_facade_actors", [])
        for actor, _ in acts:
            actor.GetProperty().SetEmissiveFactor(level, level, level)
        return bool(acts)

    def _water_to_pbr(self) -> None:
        """Water as a PBR dielectric once a physical environment exists:
        deep-water albedo, low roughness, so its look is dominated by the
        Fresnel reflection of the actual sky (bright at grazing angles, dark
        looking down) instead of a flat Phong colour that goes black when
        the sun is low. Idempotent."""
        for key in ("sea_actor", "water_actor"):
            actor = self.scene_state.get(key)
            if actor is None or self.scene_state.get(f"_{key}_pbr"):
                continue
            try:
                prop = actor.GetProperty()
                prop.SetInterpolationToPBR()
                prop.SetColor(*WATER_ALBEDO)
                prop.SetMetallic(0.0)
                prop.SetRoughness(WATER_ROUGHNESS)
                prop.SetBaseIOR(1.33)
                self.scene_state[f"_{key}_pbr"] = True
            except Exception as exc:
                print(f"[sky] water material unchanged ({key}): {exc}")

    def _apply_sky_environment(self, hour: float, sun_dir) -> bool:
        """Physical sky as environment lighting + background. True if applied."""
        if bool(getattr(self.args, "no_physical_sky", False)) or self.plotter is None:
            return False
        from render.sky import (VTK_SKY_SCALE, cached_sky_environment, configure_environment_basis,
                                environment_texture, make_skybox, srgb_encode)
        s = np.asarray(sun_dir, dtype=float)
        if not np.all(np.isfinite(s)) or np.linalg.norm(s) < 1e-9:
            return False
        cache = Path(getattr(self, "cache_dir", ".cache")) / "sky"
        try:
            # One exposure for sun, sky light and reflections once every lit
            # surface is PBR (render.materials); legacy path keeps unit IBL.
            from render.materials import PBR_EXPOSURE
            mats = getattr(self, "materials", None) is not None
            env = cached_sky_environment(s, cache_dir=cache) * VTK_SKY_SCALE * (PBR_EXPOSURE if mats else 1.0)
            oc = float(self.scene_state.get("overcast", 0.0))
            if oc > 0.01:
                from render.sky import overcast_environment
                night = float(s[2]) <= 0.0
                oce = overcast_environment(irradiance=0.34 * (0.15 if night else 1.0)) * VTK_SKY_SCALE * (PBR_EXPOSURE if mats else 1.0)
                env = ((1.0 - oc) * env + oc * oce).astype(np.float32)
            tex = environment_texture(env)
            ren = self.plotter.renderer
            configure_environment_basis(ren)
            self.plotter.set_environment_texture(tex, is_srgb=False)
            ren.UseImageBasedLightingOn()
            old = self.scene_state.get("_sky_box")
            if old is not None:
                ren.RemoveActor(old)
            box = make_skybox(environment_texture(srgb_encode(env if mats else env * SKYBOX_DISPLAY_GAIN)))
            box.PickableOff()          # never intercept editor / inspection picks
            ren.AddActor(box)
            self.scene_state["_sky_box"] = box
        except Exception as exc:
            print(f"[sky] physical sky unavailable: {exc}")
            return False
        self._water_to_pbr()
        if not self.scene_state.get("_sky_logged"):
            self.scene_state["_sky_logged"] = True
            print("[sky] physically based sky (Rayleigh+Mie single scattering) drives IBL + background")
        return True
