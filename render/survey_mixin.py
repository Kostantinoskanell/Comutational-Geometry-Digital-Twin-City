"""SurveyMixin — drone-survey terrain + photoreal ground for the twin (WG).

Two stages, matching the app's init order:

  _init_survey_terrain()   (end of data loading, before anything samples terrain)
      Replaces street_graph.graph["terrain_sampler"] with the fused 0.5 m
      survey-DTM sampler (datum-corrected, feathered into Copernicus). Every
      consumer — road/car/tree/building drape, flood overlays, editor
      objects, terrain surface — upgrades with no further changes. The
      original sampler is kept as "terrain_sampler_base".

  _init_photoreal_ground() (scene build, after the stylized ground layers)
      Adds the ortho-textured ground over survey coverage. It sits just above
      the stylized road/sidewalk ground (which carries the shadow and night
      streetlight ANALYSIS classes, so it is kept, not replaced) and toggles
      via the control panel. Built flat and draped by terrain_mixin like
      every other ground layer.
"""
from __future__ import annotations

import numpy as np

from render.survey_ground import build_photoreal_ground, build_roof_overlay, load_survey_scene, survey_available

_PHOTO_Z = 0.18            # above sidewalks (0.15) and roads (0.10)
_SCENE_PAD_M = 150.0       # same order as the terrain ring pad
_PHOTOREAL_LAYERS = ("survey_ground_actor", "survey_roofs_actor",
                     "survey_struct_walls_actor", "survey_struct_roofs_actor")
_HIDE_IN_PHOTOREAL =("lane_marks_cl_actor", "lane_marks_div_actor", "crosswalks_actor")


_PHOTO_DRAPE_BIAS = 0.3   # matches the ground-mesh road bias in terrain_mixin
_PHOTO_CLEARANCE = 0.08   # above the highest stylized layer (roads 0.10 / sidewalks 0.15 overlap)


def _surface_height_at(mesh, xy: np.ndarray):
    """Height of a (draped, 2.5-D) surface mesh at xy, NaN outside it.
    Probes a flattened copy that carries the draped z as a point scalar."""
    import pyvista as pv
    if mesh is None:
        return None
    mesh = pv.wrap(mesh)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return None
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    flat = pv.PolyData(np.column_stack([mesh.points[:, :2], np.zeros(mesh.n_points)]), mesh.faces)
    if flat.n_cells == 0:
        return None
    flat.point_data["_z"] = np.asarray(mesh.points[:, 2], dtype=float)
    probe = pv.PolyData(np.column_stack([xy, np.zeros(len(xy))])).sample(flat, tolerance=1e-3)
    z = np.asarray(probe.point_data["_z"], dtype=float)
    valid = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    z[~valid] = np.nan
    return z


class SurveyMixin:

    def _init_survey_terrain(self) -> None:
        self.survey_scene = None
        if bool(getattr(self.args, "no_survey", False)) or not survey_available():
            if not survey_available():
                print("[survey] no preprocessed drone survey (run: python -m render.survey_prep) — using base DEM")
            return
        bm = getattr(self, "buildings_mesh", None)
        g = getattr(self, "street_graph", None)
        if bm is None or bm.n_points == 0 or g is None or not g.graph.get("crs"):
            return
        bx0, bx1, by0, by1 = bm.bounds[:4]
        extent = (bx0 - _SCENE_PAD_M, bx1 + _SCENE_PAD_M, by0 - _SCENE_PAD_M, by1 + _SCENE_PAD_M)
        anchors = np.array([[d["x"], d["y"]] for _, d in g.nodes(data=True) if "x" in d and "y" in d], dtype=float)
        base = g.graph.get("terrain_sampler")
        try:
            scene = load_survey_scene(
                str(g.graph["crs"]), extent, base, anchor_xy=anchors,
                cache_dir=getattr(self, "cache_dir", None),
                elev_res=float(getattr(self.args, "survey_elev_res", 0.5)),
                tex_max_px=int(getattr(self.args, "survey_tex_px", 8192)),
            )
        except Exception as exc:
            print(f"[survey] survey terrain skipped: {exc}")
            return
        if scene is None:
            return
        self.survey_scene = scene
        g.graph["terrain_sampler_base"] = base
        g.graph["terrain_sampler"] = scene.sampler
        print(f"[survey] terrain sampler -> drone DTM ({scene.datum_source})")
        self._reconcile_buildings_with_survey()

    def _reconcile_buildings_with_survey(self) -> None:
        """Remove footprints the survey shows as empty (e.g. port warehouses
        destroyed in 2020), re-height the rest from measured roofs, and
        co-register them to the surveyed roofs (render.building_reconstruct)."""
        self.building_reconcile_report = None
        if bool(getattr(self.args, "no_building_reconcile", False)) or self.survey_scene is None:
            return
        from render.building_reconstruct import reconcile_buildings
        try:
            new_mesh, report = reconcile_buildings(self.buildings_mesh, self.survey_scene)
        except Exception as exc:
            print(f"[survey] building reconciliation skipped: {exc}")
            return
        if new_mesh.n_points == 0:
            print("[survey] building reconciliation removed everything — keeping source buildings")
            return
        self.buildings_mesh = new_mesh
        # (runs during data loading, before scene_state exists)
        self.building_reconcile_report = report
        print(f"[survey] buildings reconciled with survey nDSM: {report.summary()}")

    def _init_photoreal_ground(self) -> None:
        self.scene_state["photoreal_ground"] = False
        scene = getattr(self, "survey_scene", None)
        if scene is None or self.plotter is None:
            return
        built = build_photoreal_ground(scene, mesh_res=float(getattr(self.args, "survey_mesh_res", 2.0)), z=_PHOTO_Z)
        if built is None:
            return
        mesh, texture = built
        actor = self.plotter.add_mesh(
            mesh, texture=texture, lighting=False, show_scalar_bar=False,
            reset_camera=False, name="survey_ground",
        )
        self.scene_state["survey_ground_actor"] = actor
        self._survey_texture = texture   # shared by the roof overlay (one GPU texture)
        print(f"[survey] photoreal ground: {mesh.n_cells} cells, texture "
              f"{scene.texture.shape[1]}x{scene.texture.shape[0]}")
        self._set_photoreal_ground(not bool(getattr(self.args, "no_photoreal", False)))

    def _init_photoreal_roofs(self) -> None:
        """Ortho rooftops on the (survey-co-registered) buildings. Call after
        the building actors exist; drapes with them via terrain_mixin."""
        scene = getattr(self, "survey_scene", None)
        texture = getattr(self, "_survey_texture", None)
        if scene is None or texture is None or self.plotter is None:
            return
        roofs = build_roof_overlay(self.buildings_mesh, scene)
        if roofs is None:
            return
        # Unlit, like the ground: an orthophoto already carries the real sun
        # and shadows of the survey day; lighting it again double-counts.
        actor = self.plotter.add_mesh(roofs, texture=texture, lighting=False, show_scalar_bar=False,
                                      reset_camera=False, name="survey_roofs")
        self.scene_state["survey_roofs_actor"] = actor
        actor.SetVisibility(bool(self.scene_state.get("photoreal_ground", False)))
        print(f"[survey] photoreal rooftops: {roofs.n_cells} roof faces textured from the ortho")

    def _init_survey_structures(self) -> None:
        """Raised city fabric the source footprints miss, extracted from the
        survey nDSM (render.survey_structures). Visual-only: kept out of
        buildings_mesh so no analysis (shadows, damage, ...) counts it."""
        scene = getattr(self, "survey_scene", None)
        texture = getattr(self, "_survey_texture", None)
        if scene is None or texture is None or self.plotter is None or bool(getattr(self.args, "no_survey_structures", False)):
            return
        from render.building_reconstruct import building_footprints
        from render.survey_structures import extract_structures, road_centrelines
        g = getattr(self, "street_graph", None)
        built = extract_structures(
            scene,
            exclude_footprints=[fp for fp in building_footprints(self.buildings_mesh) if fp is not None],
            road_lines=road_centrelines(g) if g is not None else (),
        )
        if built is None:
            return
        walls, roofs = built
        try:
            walls = walls.compute_normals(cell_normals=False, point_normals=True, split_vertices=True,
                                          auto_orient_normals=True)
        except Exception:
            pass
        on = bool(self.scene_state.get("photoreal_ground", False))
        # Beirut's dominant facade material: rendered concrete / limestone.
        self.scene_state["survey_struct_walls_actor"] = self.plotter.add_mesh(
            walls, color="#cfc6b4", pbr=True, metallic=0.0, roughness=0.85, smooth_shading=True,
            show_scalar_bar=False, reset_camera=False, name="survey_struct_walls")
        self.scene_state["survey_struct_roofs_actor"] = self.plotter.add_mesh(
            roofs, texture=texture, lighting=False,          # baked ortho light (see roofs)
            show_scalar_bar=False, reset_camera=False, name="survey_struct_roofs")
        for key in ("survey_struct_walls_actor", "survey_struct_roofs_actor"):
            self.scene_state[key].SetVisibility(on)
        h = np.asarray(roofs.cell_data["height"])
        print(f"[survey] structures from nDSM: {roofs.n_cells} roof faces, median height {np.median(h):.1f} m "
              "(visual only, not in analyses)")

    def _drape_survey_ground(self, dem) -> None:
        """Drape the photo ground over the terrain, never below the draped
        stylized ground layers.

        Those layers (roads/sidewalks ground mesh, land-use fill) are large
        polygons draped at their vertices only, so over concave terrain their
        faces are chords that rise above the DEM — through the finely draped
        photo (dark patches). Lifting each photo vertex to at least their
        surface keeps the photo on top without touching the analysis meshes.
        Stores the same original-z key as terrain_mixin._drape_actor_points,
        so _restore_actor_points("survey_ground_actor") undoes it.
        """
        actor = self.scene_state.get("survey_ground_actor")
        if actor is None:
            return
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            pd = actor.GetMapper().GetInputDataObject(0, 0)
            if pd is None or pd.GetNumberOfPoints() == 0:
                return
            pts_vtk = pd.GetPoints()
            pts = vtk_to_numpy(pts_vtk.GetData())
            self.scene_state["_orig_survey_ground_actor_z"] = pts[:, 2].copy()
            xy = pts[:, :2].astype(float)
            z = pts[:, 2] + np.asarray(dem(xy), dtype=float) + _PHOTO_DRAPE_BIAS
            fill = self.scene_state.get("fill_actor")
            layers = [getattr(self, "ground_mesh", None),
                      fill.GetMapper().GetInputDataObject(0, 0) if fill is not None else None]
            raised = 0
            for layer in layers:
                top = _surface_height_at(layer, xy)
                if top is None:
                    continue
                need = np.isfinite(top) & (top + _PHOTO_CLEARANCE > z)
                z[need] = top[need] + _PHOTO_CLEARANCE
                raised += int(need.sum())
            pts[:, 2] = z
            pts_vtk.Modified()
            pd.Modified()
            if raised:
                print(f"[survey] photo ground kept above draped stylized layers at {raised} vertices")
        except Exception as exc:
            print(f"[terrain] 'survey_ground_actor' lift failed: {exc}")

    def _drape_over_ground(self, actor_key: str, dem) -> None:
        """Drape a flat overlay (z = its layer offset) so it stays at that
        offset above the draped GROUND it covers: the photo ground where it
        exists (itself lifted over coarse stylized chords), else DEM + the
        road bias. Stores the original z like _drape_actor_points."""
        actor = self.scene_state.get(actor_key)
        if actor is None:
            return
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            pd = actor.GetMapper().GetInputDataObject(0, 0)
            if pd is None or pd.GetNumberOfPoints() == 0:
                return
            pts_vtk = pd.GetPoints()
            pts = vtk_to_numpy(pts_vtk.GetData())
            self.scene_state[f"_orig_{actor_key}_z"] = pts[:, 2].copy()
            xy = pts[:, :2].astype(float)
            base = np.asarray(dem(xy), dtype=float) + _PHOTO_DRAPE_BIAS
            photo = self.scene_state.get("survey_ground_actor")
            if photo is not None:
                top = _surface_height_at(photo.GetMapper().GetInputDataObject(0, 0), xy)
                if top is not None:
                    ok = np.isfinite(top)
                    base[ok] = np.maximum(base[ok], top[ok] - _PHOTO_Z)
            pts[:, 2] = pts[:, 2] + base
            pts_vtk.Modified()
            pd.Modified()
        except Exception as exc:
            print(f"[terrain] '{actor_key}' lift failed: {exc}")

    def _set_wet_film(self, wet: float) -> None:
        """Wet-street film over the photo ground: a thin water layer (PBR
        dielectric, roughness ~0.08, near-black body) whose opacity follows
        wetness, so the orthophoto darkens and mirrors the sky at grazing
        angles like wet asphalt. Built lazily; shares the photo mesh."""
        wet = float(np.clip(wet, 0.0, 1.0))
        film = self.scene_state.get("survey_wet_film_actor")
        ground = self.scene_state.get("survey_ground_actor")
        if ground is None:
            return
        if film is None:
            if wet <= 0.0:
                return
            import vtk
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(ground.GetMapper().GetInputDataObject(0, 0))   # same (draped) geometry
            mapper.ScalarVisibilityOff()
            film = vtk.vtkActor()
            film.SetMapper(mapper)
            prop = film.GetProperty()
            prop.SetInterpolationToPBR()
            prop.SetColor(0.01, 0.01, 0.012)
            prop.SetMetallic(0.0)
            prop.SetRoughness(0.08)
            prop.SetBaseIOR(1.33)
            film.SetPosition(0.0, 0.0, 0.01)
            film.PickableOff()
            self.plotter.renderer.AddActor(film)
            self.scene_state["survey_wet_film_actor"] = film
        film.GetProperty().SetOpacity(0.45 * wet)
        film.SetVisibility(bool(wet > 0.01 and self.scene_state.get("photoreal_ground", False)))

    def _set_photoreal_ground(self, on: bool) -> None:
        actor = self.scene_state.get("survey_ground_actor")
        if actor is None:
            return
        self.scene_state["photoreal_ground"] = bool(on)
        if hasattr(self, "_set_facade_tint"):
            self._set_facade_tint()
        if hasattr(self, "_sync_flood_layers"):
            self._sync_flood_layers()
        for key in _PHOTOREAL_LAYERS:
            a = self.scene_state.get(key)
            if a is not None:
                a.SetVisibility(bool(on))
        # Painted OSM markings would double up (and misalign) with the real
        # ones in the photo; restore them in the stylized analysis view.
        for key in _HIDE_IN_PHOTOREAL:
            a = self.scene_state.get(key)
            if a is not None:
                a.SetVisibility(not on)

    def _toggle_photoreal_ground(self) -> None:
        self._set_photoreal_ground(not bool(self.scene_state.get("photoreal_ground", False)))
        try:
            self.plotter.render()
        except Exception:
            pass
        print(f"[survey] photoreal ground {'ON' if self.scene_state.get('photoreal_ground') else 'OFF (analysis view)'}")
