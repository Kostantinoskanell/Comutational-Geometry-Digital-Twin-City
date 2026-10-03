"""TerrainMixin — DEM-based 3D terrain mesh with elevation colouring and contour lines.

Active only when the Copernicus DEM GLO-30 sampler was loaded successfully
(requires pystac_client, planetary_computer, rasterio).  For sea-level / flat
cities the sampler is None and this mixin is a complete no-op.

What it adds:
  • Smooth terrain surface coloured by elevation (matplotlib "terrain" cmap)
  • Contour polylines every ~10 m elevation, earth-brown
  • DEM-lifted land-use fill mesh so OSM colour patches sit on the hillside
  • Key 'g' — toggle terrain surface visibility
"""
from __future__ import annotations

import numpy as np
import pyvista as pv


class TerrainMixin:

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @staticmethod
    def _hillshade_rgb(pts_xy: np.ndarray, elev: np.ndarray, dem,
                       z_min: float, z_range: float) -> np.ndarray:
        """Hillshaded muted-earth RGB per point (uint8, N×3).

        Lambertian shading from DEM finite differences (sun from the NW at
        ~45° elevation) multiplied into an earth-tone elevation palette —
        reads like a relief map instead of the rainbow 'terrain' cmap.
        """
        eps = 6.0
        try:
            h_e = np.asarray(dem(pts_xy + np.array([eps, 0.0])), dtype=float)
            h_w = np.asarray(dem(pts_xy - np.array([eps, 0.0])), dtype=float)
            h_n = np.asarray(dem(pts_xy + np.array([0.0, eps])), dtype=float)
            h_s = np.asarray(dem(pts_xy - np.array([0.0, eps])), dtype=float)
            gx = (h_e - h_w) / (2.0 * eps)
            gy = (h_n - h_s) / (2.0 * eps)
        except Exception:
            gx = np.zeros(pts_xy.shape[0])
            gy = np.zeros(pts_xy.shape[0])

        inv_n = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
        nx, ny, nz = -gx * inv_n, -gy * inv_n, inv_n
        # Light from NW, 45° above horizon
        lx, ly, lz = -0.5, 0.5, 0.7071
        shade = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        shade = 0.55 + 0.45 * shade          # keep shadows readable, not black

        # Earth palette stops: valley green → olive → tan → rock grey → light peak
        t = np.clip((np.asarray(elev, dtype=float) - z_min) / max(z_range, 1e-6), 0.0, 1.0)
        stops_t = np.array([0.00, 0.35, 0.60, 0.85, 1.00])
        stops_c = np.array([
            [0x3d, 0x5a, 0x3d],
            [0x6b, 0x7d, 0x4f],
            [0xa0, 0x8b, 0x5f],
            [0x8a, 0x83, 0x78],
            [0xc9, 0xc5, 0xbd],
        ], dtype=float)
        rgb = np.empty((t.shape[0], 3), dtype=float)
        for ch in range(3):
            rgb[:, ch] = np.interp(t, stops_t, stops_c[:, ch])
        rgb *= shade[:, None]

        # Sea: DEM cells at (or below) sea level are water, not earth — the
        # Copernicus grid stores 0 over ocean.  Paint them the scene sea blue
        # (flat colour, no hillshade) so coastal cities read correctly.
        sea = np.asarray(elev, dtype=float) <= 0.05
        if np.any(sea):
            rgb[sea] = np.array([0x2e, 0x6e, 0xa6], dtype=float)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _init_terrain(self) -> None:
        dem = self.street_graph.graph.get("terrain_sampler")
        if dem is None:
            print("[terrain] DEM sampler not found — terrain skipped")
            return

        if self.buildings_mesh is None or self.buildings_mesh.n_points == 0:
            print("[terrain] buildings mesh empty — terrain skipped")
            return

        bx0, bx1, by0, by1 = self.buildings_mesh.bounds[:4]
        pad    = 130.0
        GRID_N = 130          # 130×130 = ~17 k vertices, < 1 ms GPU

        xs = np.linspace(bx0 - pad, bx1 + pad, GRID_N)
        ys = np.linspace(by0 - pad, by1 + pad, GRID_N)
        XX, YY = np.meshgrid(xs, ys)

        xy_flat = np.column_stack([XX.ravel(), YY.ravel()])
        try:
            zz_flat = dem(xy_flat).astype(float)
        except Exception as exc:
            print(f"[terrain] DEM sampling failed: {exc}")
            return

        z_min   = float(zz_flat.min())
        z_max   = float(zz_flat.max())
        z_range = max(1.0, z_max - z_min)
        print(f"[terrain] elevation {z_min:.0f} – {z_max:.0f} m  (range {z_range:.0f} m)")

        if z_range < 3.0:
            print("[terrain] terrain is virtually flat — terrain mesh skipped")
            # Still try to lift fill mesh so it doesn't vanish underground
            self._lift_fill_mesh_to_dem(dem)
            return

        # Offset 0.5 m below the draped road surface: the 130×130 terrain grid
        # interpolates the DEM more coarsely than road vertices do, so on
        # slopes its triangles can bulge ~0.3 m above the roads — 8 cm was not
        # enough clearance and roads kept sinking into the hillside.
        ZZ = zz_flat.reshape(XX.shape) - 0.5

        grid = pv.StructuredGrid(XX, YY, ZZ)
        grid["elev"] = zz_flat.astype(float)

        # Clip terrain to the outer ring: hide cells inside the city building bounds
        # so the terrain mesh does not slice through buildings that are extruded
        # from Z=0 while the DEM has positive elevation in those areas.
        # The road/fill/sidewalk meshes already supply the in-city ground surface.
        _CITY_INSET = 20.0   # metres inside the building bbox to start clipping
        try:
            cc   = grid.cell_centers().points
            in_city = (
                (cc[:, 0] >= bx0 - _CITY_INSET) & (cc[:, 0] <= bx1 + _CITY_INSET) &
                (cc[:, 1] >= by0 - _CITY_INSET) & (cc[:, 1] <= by1 + _CITY_INSET)
            )
            if in_city.any() and not in_city.all():
                outer_idx = np.where(~in_city)[0]
                grid = grid.extract_cells(outer_idx).extract_surface()
                # Re-attach elevation as a point scalar for the clipped surface
                pts_xy = grid.points[:, :2]
                grid["elev"] = dem(pts_xy).astype(float)
                print(f"[terrain] city interior clipped — {int(outer_idx.size)} outer cells kept")
        except Exception as _clip_exc:
            print(f"[terrain] city clip skipped: {_clip_exc}")

        # Hillshaded muted-earth colouring (replaces the rainbow "terrain" cmap
        # that clashed with the slate/grey city palette)
        try:
            grid["RGB"] = self._hillshade_rgb(
                grid.points[:, :2], grid["elev"], dem, z_min, z_range
            )
        except Exception as _hs_exc:
            print(f"[terrain] hillshade failed ({_hs_exc}) — falling back to cmap")

        try:
            if "RGB" in grid.point_data:
                t_actor = self.plotter.add_mesh(
                    grid,
                    scalars="RGB",
                    rgb=True,
                    smooth_shading=True,
                    lighting=False,      # shading is baked into the colours
                    show_scalar_bar=False,
                    opacity=1.0,
                    reset_camera=False,
                    name="terrain_surface",
                )
            else:
                t_actor = self.plotter.add_mesh(
                    grid,
                    scalars="elev",
                    cmap="terrain",
                    clim=[z_min, z_max],
                    smooth_shading=True,
                    lighting=True,
                    show_scalar_bar=False,
                    opacity=0.90,
                    reset_camera=False,
                    name="terrain_surface",
                )
            # Built but HIDDEN: the scene starts flat everywhere (buildings,
            # roads, water all at z≈0) and the Terrain checkbox turns the
            # surface + drape on together.  Starting visible forced hilly
            # cities (e.g. San Francisco) into a broken half-draped view.
            t_actor.VisibilityOff()
            self.scene_state["terrain_actor"]   = t_actor
            self.scene_state["_terrain_visible"] = False
            print(f"[terrain] terrain surface ready (hidden): {grid.n_cells} cells "
                  f"— enable via the Terrain checkbox")
        except Exception as exc:
            print(f"[terrain] mesh render failed: {exc}")
            return

        # Contour lines — skip if barely any relief
        if z_range >= 15.0:
            n_contours = min(20, max(4, int(z_range / 10.0)))
            try:
                contours = grid.contour(
                    n_contours,
                    scalars="elev",
                    rng=[z_min, z_max],
                )
                if contours.n_points > 0:
                    c_actor = self.plotter.add_mesh(
                        contours,
                        color="#4a3c28",
                        opacity=0.30,        # subtle — hillshade carries the relief now
                        line_width=1.0,
                        lighting=False,
                        reset_camera=False,
                        name="terrain_contours",
                    )
                    c_actor.VisibilityOff()   # hidden with the terrain surface
                    self.scene_state["terrain_contour_actor"] = c_actor
                    print(
                        f"[terrain] {n_contours} contour lines"
                        f"  (Δz ≈ {z_range / n_contours:.0f} m each)"
                    )
            except Exception as exc:
                print(f"[terrain] contours failed: {exc}")

        # Terrain is toggled via its panel checkbox — no keyboard shortcut
        # ('g' freed for future use).

        # NOTE: _apply_terrain_drape is NOT called here because buildings and
        # trees haven't been rendered yet at this point in the init sequence.
        # The call site in main_ast6.py triggers draping after all geometry exists.

    # ------------------------------------------------------------------
    # Visibility toggle
    # ------------------------------------------------------------------

    def _toggle_terrain(self) -> None:
        visible = not bool(self.scene_state.get("_terrain_visible", True))
        self.scene_state["_terrain_visible"] = visible
        for key in ("terrain_actor", "terrain_contour_actor"):
            actor = self.scene_state.get(key)
            if actor is None:
                continue
            try:
                if visible:
                    actor.VisibilityOn()
                else:
                    actor.VisibilityOff()
            except Exception:
                pass
        dem = self.street_graph.graph.get("terrain_sampler")
        if visible:
            if dem is not None:
                self._apply_terrain_drape(dem)
        else:
            self._remove_terrain_drape()
        try:
            self.plotter.render()
        except Exception:
            pass
        print(f"[terrain] {'visible + draped' if visible else 'hidden + restored'}")

    # ------------------------------------------------------------------
    # Lift the OSM fill mesh to DEM elevation
    # ------------------------------------------------------------------

    def _lift_fill_mesh_to_dem(self, dem) -> None:
        """Replace the flat fill mesh with a DEM-elevated version.

        The fill mesh stored in the graph has all Z = 0.02 (FILL_Z).  For hilly
        cities this puts the land-use colour patches underground.  We re-sample
        the DEM at each fill-mesh vertex and place it 10 cm above the terrain.
        """
        fill_pts   = self.street_graph.graph.get("fill_pts")
        fill_faces = self.street_graph.graph.get("fill_faces")
        fill_rgb   = self.street_graph.graph.get("fill_rgb")
        if fill_pts is None or fill_faces is None or fill_rgb is None:
            return

        try:
            pts    = np.asarray(fill_pts,   dtype=float).copy()
            elevs  = dem(pts[:, :2]).astype(float)
            pts[:, 2] = elevs + 0.10   # 10 cm above terrain surface

            lifted = pv.PolyData(pts, np.asarray(fill_faces, dtype=np.int64))
            lifted.cell_data["RGB"] = np.asarray(fill_rgb, dtype=np.uint8)

            old = self.scene_state.get("fill_actor")
            if old is not None:
                try:
                    self.plotter.remove_actor(old, reset_camera=False)
                except Exception:
                    pass

            new_actor = self.plotter.add_mesh(
                lifted,
                scalars="RGB", rgb=True,
                smooth_shading=False, lighting=False,
                show_scalar_bar=False,
                reset_camera=False,
            )
            self.scene_state["fill_actor"] = new_actor
            print(f"[terrain] fill mesh DEM-lifted  ({lifted.n_cells} patches)")
        except Exception as exc:
            print(f"[terrain] fill mesh lift failed: {exc}")

    # ------------------------------------------------------------------
    # Terrain draping — lift static geometry onto DEM surface
    # ------------------------------------------------------------------

    def _apply_terrain_drape(self, dem) -> None:
        """Lift roads, sidewalks, TL anchors, and parked cars onto DEM surface.

        Idempotent — safe to call multiple times.  Precomputes per-path height
        profiles on first call so per-frame car z lookup is O(N) numpy.
        """
        if self.scene_state.get("_terrain_drape_active"):
            return

        # ── Precompute car path height profiles ───────────────────────────
        # Rebuild if profiles are missing OR path count has changed since last build.
        _existing_prof = self.scene_state.get("_car_h_profiles")
        _n_paths_now   = len(self.car_paths) if getattr(self, "car_paths", None) else 0
        _needs_rebuild = (
            _existing_prof is None
            or _existing_prof.shape[0] != _n_paths_now
        )
        if _needs_rebuild and _n_paths_now > 0:
            print(f"[terrain] building height profiles for {_n_paths_now} paths ...")
            try:
                from terrain_drape import precompute_path_heights
                _prof, _lens, _counts = precompute_path_heights(
                    self.car_paths, self._car_pose_on_path, dem, spacing_m=5.0,
                )
                self.scene_state["_car_h_profiles"] = _prof
                self.scene_state["_car_h_lens"]     = _lens
                self.scene_state["_car_h_counts"]   = _counts
                print(f"[terrain] height profiles ready: {_n_paths_now} paths, "
                      f"profile array shape {_prof.shape}")
            except Exception as _exc:
                print(f"[terrain] profile precompute failed: {_exc}")
        elif not _needs_rebuild:
            print(f"[terrain] reusing {_n_paths_now} existing height profiles")

        # ── Lift ground mesh (roads + sidewalks) per-vertex ───────────────
        # +0.3 bias keeps roads proud of the (coarser) fill mesh on slopes.
        gm = getattr(self, "ground_mesh", None)
        if gm is not None and gm.n_points > 0:
            _orig_gz = gm.points[:, 2].copy()
            self.scene_state["_orig_ground_z"] = _orig_gz
            _h = np.asarray(dem(gm.points[:, :2]), dtype=float)
            gm.points[:, 2] = _orig_gz + _h + 0.3
            gm.Modified()

        # ── Lift TL mesh and rebuild glyph actor ──────────────────────────
        self._drape_tl_glyphs(dem)

        # ── Lift the instanced parked-car fleet (render.instanced_cars) ────
        _pfleet = self.scene_state.get("_parked_fleet")
        if _pfleet is not None and _pfleet.n:
            _pz0 = _pfleet.positions[:, 2].copy()
            self.scene_state["_orig_parked_fleet_z"] = _pz0
            _pfleet.set_z(_pz0 + np.asarray(dem(_pfleet.positions[:, :2]), dtype=float))

        # ── Lift parked car actors (one DEM call for all) ─────────────────
        _parked = self.scene_state.get("_parked_car_actors") or []
        if _parked:
            _park_xy = np.array(
                [list(a.GetPosition())[:2] for a in _parked], dtype=float
            )
            _h_park = np.asarray(dem(_park_xy), dtype=float)
            _orig_pz = []
            for _i, _pa in enumerate(_parked):
                try:
                    _pos = _pa.GetPosition()
                    _orig_pz.append(float(_pos[2]))
                    _pa.SetPosition(float(_pos[0]), float(_pos[1]),
                                    float(_pos[2]) + float(_h_park[_i]))
                except Exception:
                    _orig_pz.append(0.0)
            self.scene_state["_orig_parked_z"] = _orig_pz

        # ── Lift fill mesh (non-water cells only) ────────────────────────
        self._drape_fill_mesh(dem)

        # ── Lift building actors per-vertex ──────────────────────────────
        self._drape_buildings(dem)

        # ── Lift trees (re-glyph from flat seeds + DEM offset) ───────────
        self._drape_trees(dem)

        # ── Lift street direction arrows ──────────────────────────────────
        self._drape_actor_points("street_arrows_actor", dem)

        # ── Lift road/pedestrian network line layers ──────────────────────
        # extra_z keeps the tubes above the interpolated ground triangles on
        # slopes (line vertices sample the DEM at different points than the
        # ground mesh, so without a bias they z-fight or sink between verts).
        self._drape_actor_points("vehicle_actor", dem, extra_z=0.6)
        self._drape_actor_points("ped_actor", dem, extra_z=0.6)

        # ── Lift road-surface decorations (built flat) ────────────────────
        # +0.35 rides just above the ground mesh (+0.3 bias) so paint stays
        # visible on slopes.
        self._drape_actor_points("crosswalks_actor", dem, extra_z=0.35)
        self._drape_actor_points("lane_marks_cl_actor", dem, extra_z=0.35)
        self._drape_actor_points("lane_marks_div_actor", dem, extra_z=0.35)
        self._drape_actor_points("stop_signs_actor", dem)

        # ── Lift user-placed buildings (editor 'g' mode) ──────────────────
        for _ub_i in range(int(self.scene_state.get("_user_building_count", 0))):
            self._drape_actor_points(f"user_building_{_ub_i + 1}", dem)

        # ── Lift user-placed green-corridor objects (editor 'j'/'k' modes) ─
        for _kind in ("greenspace", "strip", "drain"):
            for _i in range(int(self.scene_state.get(f"_user_{_kind}_count", 0))):
                _key = f"user_{_kind}_{_i + 1}"
                if self.scene_state.get(_key) is not None:
                    self._drape_over_ground(_key, dem)
        for _st_i in range(int(self.scene_state.get("_user_stairs_count", 0))):
            self._drape_actor_points(f"user_stairs_{_st_i + 1}", dem)

        # ── Lift flood/corridor overlays (flood_mixin, if built) ──────────




        # ── Lift the photoreal survey ground (render.survey_mixin) ────────
        # +0.3 matches the ground-mesh road bias, and the photo is further
        # kept above the coarse draped stylized layers (see SurveyMixin).
        self._drape_survey_ground(dem)
        # Flood/corridor overlays ride on the draped ground (photo or roads).
        self._drape_over_ground("flood_puddles_actor", dem)
        self._drape_over_ground("corridor_materials_actor", dem)
        # Flood Lab layers (live depth / change / hazard grid, water, official decal)
        if hasattr(self, "_lab_drape"):
            self._lab_drape(True)
            self._drape_over_ground("lab_official_actor", dem)
        # Physical flood water rides on the draped photo ground (after it).
        _fw = getattr(self, "_flood_water", None)
        if _fw is not None:
            _base = np.asarray(dem(_fw.xy), dtype=float) + 0.3
            _photo = self.scene_state.get("survey_ground_actor")
            if _photo is not None:
                from render.survey_mixin import _surface_height_at, _PHOTO_Z
                _top = _surface_height_at(_photo.GetMapper().GetInputDataObject(0, 0), _fw.xy)
                if _top is not None:
                    _ok = np.isfinite(_top)
                    _base[_ok] = np.maximum(_base[_ok], _top[_ok] - _PHOTO_Z)
            _fw.set_base(_base)
        # Rooftops follow the per-vertex building drape exactly (same xy, same dem).
        self._drape_actor_points("survey_roofs_actor", dem)
        self._drape_actor_points("building_outline_actor", dem)     # per-vertex, same as buildings
        # nDSM structures: heights are above the local DTM, so per-vertex drape.
        self._drape_actor_points("survey_struct_walls_actor", dem)
        self._drape_actor_points("survey_struct_roofs_actor", dem)

        # ── Lift parking-lot glyph cars (parking_mixin pool) ──────────────
        self._drape_actor_points("_parked_cars_actor", dem)

        # ── Lift bus stop shelters (position-based, like parked cars) ─────
        _bus_stops = self.scene_state.get("_bus_stop_actors") or []
        if _bus_stops:
            _bs_xy = np.array([list(a.GetPosition())[:2] for a in _bus_stops], dtype=float)
            _h_bs = np.asarray(dem(_bs_xy), dtype=float)
            _orig_bsz = []
            for _i, _ba in enumerate(_bus_stops):
                try:
                    _bp = _ba.GetPosition()
                    _orig_bsz.append(float(_bp[2]))
                    _ba.SetPosition(float(_bp[0]), float(_bp[1]),
                                    float(_bp[2]) + float(_h_bs[_i]))
                except Exception:
                    _orig_bsz.append(0.0)
            self.scene_state["_orig_bus_stop_z"] = _orig_bsz

        self.scene_state["_terrain_drape_active"] = True
        self._invalidate_shadow_caches()
        print("[terrain] draping applied — scene lifted to DEM surface")

    def _remove_terrain_drape(self) -> None:
        """Restore every layer to its original flat z.

        Idempotent — safe to call when draping is not active.
        """
        if not self.scene_state.get("_terrain_drape_active"):
            return

        # Restore ground mesh
        _orig_gz = self.scene_state.get("_orig_ground_z")
        gm = getattr(self, "ground_mesh", None)
        if gm is not None and _orig_gz is not None:
            gm.points[:, 2] = _orig_gz
            gm.Modified()

        # Restore TL glyphs
        self._restore_tl_glyphs()

        # Restore the instanced parked-car fleet
        _pfleet = self.scene_state.get("_parked_fleet")
        _pz0 = self.scene_state.pop("_orig_parked_fleet_z", None)
        if _pfleet is not None and _pz0 is not None:
            _pfleet.set_z(_pz0)

        # Restore parked cars
        _parked  = self.scene_state.get("_parked_car_actors") or []
        _orig_pz = self.scene_state.get("_orig_parked_z") or []
        for _i, _pa in enumerate(_parked):
            if _i < len(_orig_pz):
                try:
                    _pos = _pa.GetPosition()
                    _pa.SetPosition(float(_pos[0]), float(_pos[1]), float(_orig_pz[_i]))
                except Exception:
                    pass

        # Restore fill mesh to original flat z
        self._restore_fill_mesh()

        # Restore buildings to original z
        self._restore_buildings()

        # Restore trees to flat seed z
        self._restore_trees()

        # Restore street direction arrows
        self._restore_actor_points("street_arrows_actor")

        # Restore road/pedestrian network line layers
        self._restore_actor_points("vehicle_actor")
        self._restore_actor_points("ped_actor")

        # Restore road-surface decorations
        self._restore_actor_points("crosswalks_actor")
        self._restore_actor_points("lane_marks_cl_actor")
        self._restore_actor_points("lane_marks_div_actor")
        self._restore_actor_points("stop_signs_actor")

        # Restore user-placed buildings
        for _ub_i in range(int(self.scene_state.get("_user_building_count", 0))):
            self._restore_actor_points(f"user_building_{_ub_i + 1}")

        # Restore user-placed green-corridor objects
        for _kind in ("greenspace", "strip", "drain"):
            for _i in range(int(self.scene_state.get(f"_user_{_kind}_count", 0))):
                self._restore_actor_points(f"user_{_kind}_{_i + 1}")
        for _st_i in range(int(self.scene_state.get("_user_stairs_count", 0))):
            self._restore_actor_points(f"user_stairs_{_st_i + 1}")

        # Restore flood/corridor overlays
        self._restore_actor_points("flood_puddles_actor")
        _fw = getattr(self, "_flood_water", None)
        if _fw is not None:
            _fw.set_base(None)
        self._restore_actor_points("corridor_materials_actor")
        self._restore_actor_points("survey_ground_actor")
        self._restore_actor_points("survey_roofs_actor")
        self._restore_actor_points("building_outline_actor")
        if hasattr(self, "_lab_drape"):
            self._lab_drape(False)
            self._restore_actor_points("lab_official_actor")
        self._restore_actor_points("survey_struct_walls_actor")
        self._restore_actor_points("survey_struct_roofs_actor")

        # Restore parking-lot glyph cars
        self._restore_actor_points("_parked_cars_actor")

        # Restore bus stop shelters
        _bus_stops = self.scene_state.get("_bus_stop_actors") or []
        _orig_bsz  = self.scene_state.get("_orig_bus_stop_z") or []
        for _i, _ba in enumerate(_bus_stops):
            if _i < len(_orig_bsz):
                try:
                    _bp = _ba.GetPosition()
                    _ba.SetPosition(float(_bp[0]), float(_bp[1]), float(_orig_bsz[_i]))
                except Exception:
                    pass

        self.scene_state["_terrain_drape_active"] = False
        self._invalidate_shadow_caches()
        print("[terrain] draping removed — scene restored to flat z")

    # ------------------------------------------------------------------
    # Drape / restore helpers — one per scene layer
    # ------------------------------------------------------------------

    def _drape_actor_points(self, actor_key: str, dem, extra_z: float = 0.0) -> None:
        """Lift a scene_state actor's VTK mesh points by DEM (+extra_z bias); store originals."""
        actor = self.scene_state.get(actor_key)
        if actor is None:
            return
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            _vtk_pd  = actor.GetMapper().GetInputDataObject(0, 0)
            if _vtk_pd is None or _vtk_pd.GetNumberOfPoints() == 0:
                return
            _pts_vtk = _vtk_pd.GetPoints()
            _pts_np  = vtk_to_numpy(_pts_vtk.GetData())
            _orig_key = f"_orig_{actor_key}_z"
            self.scene_state[_orig_key] = _pts_np[:, 2].copy()
            _h = np.asarray(dem(_pts_np[:, :2]), dtype=float)
            _pts_np[:, 2] += _h + float(extra_z)
            _pts_vtk.Modified()
            _vtk_pd.Modified()
        except Exception as _exc:
            print(f"[terrain] '{actor_key}' lift failed: {_exc}")

    def _restore_actor_points(self, actor_key: str) -> None:
        """Restore a scene_state actor's VTK mesh z to values saved by _drape_actor_points."""
        actor = self.scene_state.get(actor_key)
        _orig_key = f"_orig_{actor_key}_z"
        _orig_z   = self.scene_state.get(_orig_key)
        if actor is None or _orig_z is None:
            return
        try:
            from vtk.util.numpy_support import vtk_to_numpy
            _vtk_pd  = actor.GetMapper().GetInputDataObject(0, 0)
            if _vtk_pd is None:
                return
            _pts_vtk = _vtk_pd.GetPoints()
            _pts_np  = vtk_to_numpy(_pts_vtk.GetData())
            _pts_np[:, 2] = _orig_z
            _pts_vtk.Modified()
            _vtk_pd.Modified()
        except Exception as _exc:
            print(f"[terrain] '{actor_key}' restore failed: {_exc}")

    def _drape_tl_glyphs(self, dem) -> None:
        """Lift TL seed mesh, then rebuild the glyph actor at new positions."""
        _tl_m = self.scene_state.get("_tl_mesh")
        if _tl_m is None or _tl_m.n_points == 0:
            return
        _orig_tlz = _tl_m.points[:, 2].copy()
        self.scene_state["_orig_tl_z"] = _orig_tlz
        _h_tl = np.asarray(dem(_tl_m.points[:, :2]), dtype=float)
        _tl_m.points[:, 2] = _orig_tlz + _h_tl
        _tl_m.Modified()
        self._rebuild_tl_glyph_actor()

    def _restore_tl_glyphs(self) -> None:
        """Restore TL seed mesh z, then rebuild the glyph actor at original positions."""
        _tl_m     = self.scene_state.get("_tl_mesh")
        _orig_tlz = self.scene_state.get("_orig_tl_z")
        if _tl_m is None or _orig_tlz is None:
            return
        _tl_m.points[:, 2] = _orig_tlz
        _tl_m.Modified()
        self._rebuild_tl_glyph_actor()

    def _rebuild_tl_glyph_actor(self) -> None:
        """Replace the traffic-light actor with fresh glyphs from current _tl_mesh."""
        _tl_m = self.scene_state.get("_tl_mesh")
        if _tl_m is None or _tl_m.n_points == 0:
            return
        try:
            from traffic_lights import build_light_glyphs
            _tl_sphere = pv.Sphere(radius=1.4, theta_resolution=10, phi_resolution=10)
            _old = self.scene_state.get("tl_actor")
            if _old is not None:
                try:
                    self.plotter.remove_actor(_old, reset_camera=False)
                except Exception:
                    pass
            _new_glyphs = build_light_glyphs(_tl_m, _tl_sphere)
            self.scene_state["_tl_glyphs"] = _new_glyphs
            _new_actor = self.plotter.add_mesh(
                _new_glyphs,
                scalars="colors", rgb=True,
                smooth_shading=True, pbr=True,
                metallic=0.1, roughness=0.4, lighting=True,
            )
            self.scene_state["tl_actor"] = _new_actor
            _show = bool(self.scene_state.get("show_traffic_signals", True))
            try:
                self._set_actor_visibility(_new_actor, _show)
            except Exception:
                pass
        except Exception as _exc:
            print(f"[terrain] TL rebuild failed: {_exc}")

    def _drape_fill_mesh(self, dem) -> None:
        """Replace flat fill actor with DEM-lifted version; water cells stay flat."""
        _fill_pts   = self.street_graph.graph.get("fill_pts")
        _fill_faces = self.street_graph.graph.get("fill_faces")
        _fill_rgb   = self.street_graph.graph.get("fill_rgb")
        if _fill_pts is None or _fill_faces is None or _fill_rgb is None:
            return
        try:
            _WATER_RGBS = {
                (46, 110, 166), (53, 120, 176), (42, 90, 138), (74, 120, 88),
            }
            _rgb = np.asarray(_fill_rgb, dtype=np.uint8)
            _is_water_cell = np.zeros(len(_rgb), dtype=bool)
            for _wr in _WATER_RGBS:
                _is_water_cell |= (
                    (_rgb[:, 0] == _wr[0]) &
                    (_rgb[:, 1] == _wr[1]) &
                    (_rgb[:, 2] == _wr[2])
                )

            # Subdivide before lifting: large plaza/park polygons otherwise
            # span curved terrain with a single flat triangle that bulges
            # above the finer road mesh, burying the roads that cross them.
            _base = pv.PolyData(
                np.asarray(_fill_pts, dtype=float),
                np.asarray(_fill_faces, dtype=np.int64),
            )
            _base.cell_data["RGB"] = _rgb
            _mesh = _base.triangulate()
            try:
                _sub = _mesh.subdivide(2, subfilter="linear")
                if "RGB" in _sub.cell_data and _sub.n_cells > 0:
                    _mesh = _sub
            except Exception:
                pass   # keep unsubdivided mesh

            _pts  = np.asarray(_mesh.points, dtype=float).copy()
            _rgb2 = np.asarray(_mesh.cell_data["RGB"], dtype=np.uint8)
            _is_water_cell2 = np.zeros(len(_rgb2), dtype=bool)
            for _wr in _WATER_RGBS:
                _is_water_cell2 |= (
                    (_rgb2[:, 0] == _wr[0]) &
                    (_rgb2[:, 1] == _wr[1]) &
                    (_rgb2[:, 2] == _wr[2])
                )

            # Mark which vertices belong to water cells (triangles: 4 ints/face)
            _tri = _mesh.faces.reshape(-1, 4)[:, 1:]
            _is_water_v = np.zeros(len(_pts), dtype=bool)
            if _is_water_cell2.any():
                _is_water_v[_tri[_is_water_cell2].ravel()] = True

            # Lift only non-water vertices
            _land_mask = ~_is_water_v
            if _land_mask.any():
                _h = np.asarray(dem(_pts[_land_mask, :2]), dtype=float)
                _pts[_land_mask, 2] += _h

            _lifted = pv.PolyData(_pts, _mesh.faces)
            _lifted.cell_data["RGB"] = _rgb2
            _is_water_cell = _is_water_cell2   # for the summary print below

            _old = self.scene_state.get("fill_actor")
            if _old is not None:
                try:
                    self.plotter.remove_actor(_old, reset_camera=False)
                except Exception:
                    pass

            self.scene_state["fill_actor"] = self.plotter.add_mesh(
                _lifted, scalars="RGB", rgb=True,
                smooth_shading=False, lighting=False,
                show_scalar_bar=False, reset_camera=False,
            )
            print(
                f"[terrain] fill: {int((~_is_water_cell).sum())} land cells lifted, "
                f"{int(_is_water_cell.sum())} water cells kept flat"
            )
        except Exception as _exc:
            print(f"[terrain] fill drape failed: {_exc}")

    def _restore_fill_mesh(self) -> None:
        """Restore fill actor from original flat data in the street graph."""
        _fill_pts   = self.street_graph.graph.get("fill_pts")
        _fill_faces = self.street_graph.graph.get("fill_faces")
        _fill_rgb   = self.street_graph.graph.get("fill_rgb")
        if _fill_pts is None or _fill_faces is None or _fill_rgb is None:
            return
        try:
            _flat = pv.PolyData(
                np.asarray(_fill_pts,  dtype=float),
                np.asarray(_fill_faces, dtype=np.int64),
            )
            _flat.cell_data["RGB"] = np.asarray(_fill_rgb, dtype=np.uint8)

            _old = self.scene_state.get("fill_actor")
            if _old is not None:
                try:
                    self.plotter.remove_actor(_old, reset_camera=False)
                except Exception:
                    pass

            self.scene_state["fill_actor"] = self.plotter.add_mesh(
                _flat, scalars="RGB", rgb=True,
                smooth_shading=False, lighting=False,
                show_scalar_bar=False, reset_camera=False,
            )
        except Exception as _exc:
            print(f"[terrain] fill restore failed: {_exc}")

    def _drape_buildings(self, dem) -> None:
        """Lift building actor geometry in-place via VTK point mutation."""
        try:
            from vtk.util.numpy_support import vtk_to_numpy
        except ImportError:
            print("[terrain] vtk numpy support unavailable — buildings not draped")
            return

        _pbr  = self.scene_state.get("_building_actors_pbr") or {}
        _orig: dict = {}

        def _lift_actor(actor, key: str) -> None:
            try:
                _vtk_pd = actor.GetMapper().GetInputDataObject(0, 0)
                if _vtk_pd is None or _vtk_pd.GetNumberOfPoints() == 0:
                    return
                _pts_vtk = _vtk_pd.GetPoints()
                _pts_np  = vtk_to_numpy(_pts_vtk.GetData())   # shared-memory view
                _orig[key] = _pts_np[:, 2].copy()
                _h = np.asarray(dem(_pts_np[:, :2]), dtype=float)
                _pts_np[:, 2] += _h
                _pts_vtk.Modified()
                _vtk_pd.Modified()
            except Exception as _exc:
                print(f"[terrain] building '{key}' lift failed: {_exc}")

        if _pbr:
            for _name, _act in _pbr.items():
                _lift_actor(_act, _name)
        else:
            _ba = self.scene_state.get("building_actor")
            if _ba is not None:
                _lift_actor(_ba, "_fallback")

        self.scene_state["_orig_building_actor_z"] = _orig
        print(f"[terrain] buildings: {len(_orig)} actor(s) lifted")

    def _restore_buildings(self) -> None:
        """Restore building actors to their pre-drape z values."""
        try:
            from vtk.util.numpy_support import vtk_to_numpy
        except ImportError:
            return

        _pbr  = self.scene_state.get("_building_actors_pbr") or {}
        _orig = self.scene_state.get("_orig_building_actor_z") or {}
        if not _orig:
            return

        def _restore_actor(actor, key: str) -> None:
            if key not in _orig:
                return
            try:
                _vtk_pd = actor.GetMapper().GetInputDataObject(0, 0)
                if _vtk_pd is None:
                    return
                _pts_vtk = _vtk_pd.GetPoints()
                _pts_np  = vtk_to_numpy(_pts_vtk.GetData())
                _pts_np[:, 2] = _orig[key]
                _pts_vtk.Modified()
                _vtk_pd.Modified()
            except Exception as _exc:
                print(f"[terrain] building '{key}' restore failed: {_exc}")

        if _pbr:
            for _name, _act in _pbr.items():
                _restore_actor(_act, _name)
        else:
            _ba = self.scene_state.get("building_actor")
            if _ba is not None:
                _restore_actor(_ba, "_fallback")

    def _drape_trees(self, dem) -> None:
        """Re-glyph tree actors with DEM-lifted seed points."""
        _seeds = self.scene_state.get("_tree_seeds_flat") or {}
        _tmpls = self.scene_state.get("_tree_templates")  or {}
        _kw    = self.scene_state.get("_tree_actor_kwargs") or {}
        if not _seeds:
            return
        for _tkey, _flat_seeds in _seeds.items():
            _tmpl = _tmpls.get(_tkey)
            if _tmpl is None:
                continue
            try:
                _lifted = _flat_seeds.copy()
                _lifted[:, 2] += np.asarray(dem(_lifted[:, :2]), dtype=float)
                _pd = pv.PolyData(_lifted)
                self.plotter.add_mesh(
                    _pd.glyph(geom=_tmpl, orient=False, scale=False),
                    name=f"_tree_{_tkey}",
                    **_kw.get(_tkey, {}),
                )
            except Exception as _exc:
                print(f"[terrain] tree '{_tkey}' drape failed: {_exc}")
        print(f"[terrain] trees: {len(_seeds)} glyph set(s) re-lifted")

    def _restore_trees(self) -> None:
        """Re-glyph tree actors from flat seed points (no DEM offset)."""
        _seeds = self.scene_state.get("_tree_seeds_flat") or {}
        _tmpls = self.scene_state.get("_tree_templates")  or {}
        _kw    = self.scene_state.get("_tree_actor_kwargs") or {}
        if not _seeds:
            return
        for _tkey, _flat_seeds in _seeds.items():
            _tmpl = _tmpls.get(_tkey)
            if _tmpl is None:
                continue
            try:
                _pd = pv.PolyData(_flat_seeds.copy())
                self.plotter.add_mesh(
                    _pd.glyph(geom=_tmpl, orient=False, scale=False),
                    name=f"_tree_{_tkey}",
                    **_kw.get(_tkey, {}),
                )
            except Exception as _exc:
                print(f"[terrain] tree '{_tkey}' restore failed: {_exc}")

    def _invalidate_shadow_caches(self) -> None:
        """Clear shadow triangle cache and recreate ground copies for correct shadows."""
        import traceback as _tb
        _caller = "".join(_tb.format_stack(limit=3)[0]).strip().split("\n")[0]
        print(f"[terrain] _invalidate_shadow_caches called — {_caller}")
        # The cache key (id, n_points, n_cells) doesn't change when only z
        # changes, so it MUST be cleared explicitly after any mesh point mutation.
        try:
            from shadow_engine import _GROUND_TRI_CACHE
            _GROUND_TRI_CACHE.clear()
        except Exception:
            pass

        # Discard any in-flight shadow result (it used stale geometry)
        self.scene_state["_shadow_future"]  = None
        self.scene_state["_shadow_pending"] = None

        # Recreate ground copies so shadow scalars land on the correct surface
        gm = getattr(self, "ground_mesh", None)
        if gm is not None and gm.n_points > 0:
            self.scene_state["day_ground_copy"]   = gm.copy()
            self.scene_state["night_ground_copy"] = gm.copy()

        # Schedule a fresh shadow computation if the app is interactive
        if self.scene_state.get("interactive_ready"):
            try:
                from app_core import _sun_dir_from_hour
                _hour  = float(self.scene_state.get("hour", 12.0))
                _sr    = float(self.scene_state.get("spot_radius", 15.0))
                _sdir  = _sun_dir_from_hour(_hour)
                _night = bool(_sdir[2] <= 0.0)
                self._schedule_shadow_job(_hour, _sr, _night, _sdir)
            except Exception as _exc:
                print(f"[terrain] shadow reschedule failed: {_exc}")
