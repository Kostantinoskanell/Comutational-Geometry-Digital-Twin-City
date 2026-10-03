"""Extract raised structures the source footprints are missing (WG).

Overture/OSM have no footprints for much of the dense Beirut fabric (or it
lies outside the scene's query radius), so there it would render as flat
photo. The survey nDSM resolves every raised surface directly:

  1. roof-evidence raster: nDSM > min_height, not vegetation (ortho ExG),
     minus the (dilated) analysis-building footprints already in the scene;
  2. light morphological cleanup (opening, small-hole fill);
  3. height-banded segmentation: connected regions of equal quantized
     height, so adjacent buildings of different heights stay separate
     volumes (stepped LoD1.5 massing rather than one blob per block);
  4. polygonize + simplify, extrude each region to its median roof height;
     two rejections keep non-buildings out:
       * thin regions (nothing farther than MIN_WIDTH_M/2 from an edge):
         walls, hoardings, and slivers left around the excluded footprints;
       * regions split in two by a vehicular road centreline: a footbridge
         deck or overpass sits above the DTM road beneath it, and extruding
         it would build a solid wall across the street.

These are VISUAL structures only: the extraction also finds container stacks
and similar non-buildings, so they are kept out of buildings_mesh and every
analysis (shadows, flood damage, ...) that relies on attributed buildings.
Returns separate wall and roof meshes; roofs carry ortho texture coordinates.
"""
from __future__ import annotations

import numpy as np

MIN_HEIGHT_M = 3.0
MIN_AREA_M2 = 20.0
BAND_M = 3.0          # ~one storey
SIMPLIFY_M = 0.6
FOOTPRINT_BUFFER_M = 1.5
MIN_SHARPNESS_FRAC = 0.25   # of the scene's median structure sharpness
MIN_WIDTH_M = 4.5    # narrower than this = bridge deck / wall / hoarding / sliver, not a building
MIN_CORE_FRAC = 0.05  # area share that must lie farther than MIN_WIDTH_M/2 from the outline
BRIDGE_SPLIT_FRAC = 0.1  # each side of a crossing road must hold this share of the region
VEHICULAR_HIGHWAYS = ("motorway", "trunk", "primary", "secondary", "tertiary",
                      "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link")


def _grid_transform(scene):
    """rasterio affine for the scene grid flipped north-up (row 0 = north)."""
    from rasterio.transform import from_origin
    ny, _ = scene.ndsm.shape
    return from_origin(scene.x0 - scene.dx / 2.0, scene.y0 + (ny - 0.5) * scene.dx, scene.dx, scene.dx)


def _crosses_as_bridge(poly, lines) -> bool:
    """True when a road centreline cuts `poly` into two substantial parts."""
    for line in lines:
        if not line.intersects(poly):
            continue
        parts = poly.difference(line.buffer(0.25))
        areas = sorted((g.area for g in getattr(parts, "geoms", [parts])), reverse=True)
        if len(areas) >= 2 and areas[1] >= BRIDGE_SPLIT_FRAC * poly.area:
            return True
    return False


def road_centrelines(street_graph, highways=VEHICULAR_HIGHWAYS):
    """Shapely centrelines of the vehicular street-graph edges."""
    from shapely.geometry import LineString
    out = []
    for u, v, d in street_graph.edges(data=True):
        hw = d.get("highway")
        hw = hw if isinstance(hw, (list, tuple)) else [hw]
        if not any(str(h) in highways for h in hw):
            continue
        geom = d.get("geometry")
        if geom is None:
            nu, nv = street_graph.nodes[u], street_graph.nodes[v]
            if "x" not in nu or "x" not in nv:
                continue
            geom = LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])
        out.append(geom)
    return out


def extract_structures(scene, exclude_footprints=(), road_lines=(), min_height: float = MIN_HEIGHT_M,
                       min_area: float = MIN_AREA_M2, band_m: float = BAND_M,
                       simplify_m: float = SIMPLIFY_M, min_sharpness: float | None = None):
    """Return (walls pv.PolyData, roofs pv.PolyData) or None."""
    import pyvista as pv
    from rasterio.features import rasterize, shapes
    from scipy import ndimage as ndi
    from shapely.geometry import shape
    from render.building_reconstruct import _Evidence

    if scene is None or getattr(scene, "ndsm", None) is None:
        return None
    ev = _Evidence(scene, min_height)
    built = ev.built[::-1].copy()                      # north-up for rasterio
    ndsm = np.nan_to_num(scene.ndsm[::-1], nan=0.0)
    transform = _grid_transform(scene)

    geoms = [fp.buffer(FOOTPRINT_BUFFER_M) for fp in exclude_footprints if fp is not None and not fp.is_empty]
    if geoms:
        occupied = rasterize(((g, 1) for g in geoms), out_shape=built.shape, transform=transform,
                             fill=0, dtype="uint8", all_touched=True).astype(bool)
        built &= ~occupied

    built = ndi.binary_opening(built, structure=np.ones((3, 3), bool))
    holes = ndi.binary_fill_holes(built) & ~built
    hl, nh = ndi.label(holes)
    if nh:
        small = np.bincount(hl.ravel())[1:] * scene.dx ** 2 < 4.0
        built |= np.isin(hl, np.flatnonzero(small) + 1)

    smooth = ndi.median_filter(ndsm, size=3)
    band = np.where(built, np.floor(smooth / band_m).astype(np.int32), 0)
    labels = np.zeros(built.shape, np.int32)
    next_id = 1
    for k in np.unique(band[built]):
        lab, n = ndi.label(built & (band == k))
        if n:
            labels[lab > 0] = lab[lab > 0] + next_id - 1
            next_id += n
    if next_id == 1:
        return None

    counts = np.bincount(labels.ravel())
    heights = ndi.median(smooth, labels=labels, index=np.arange(next_id))

    # Image sharpness per region (mean |Laplacian| of the ortho sampled on the
    # 0.5 m grid): real roofs carry detail (tanks, parapets, edges); regions
    # where the source photogrammetry failed are smeared blobs in both the
    # ortho and the DSM, and would extrude into "melted" volumes.
    ny, nx = scene.ndsm.shape
    gx = scene.x0 + np.arange(nx) * scene.dx
    gy = scene.y0 + np.arange(ny) * scene.dx
    GX, GY = np.meshgrid(gx, gy)
    gray = scene.rgb_at(np.column_stack([GX.ravel(), GY.ravel()])).astype(np.float32).mean(axis=1).reshape(ny, nx)[::-1]
    lap = np.abs(ndi.laplace(gray))
    sharpness = ndi.mean(lap, labels=labels, index=np.arange(next_id))
    if min_sharpness is None:
        big = np.flatnonzero(counts[1:] * scene.dx ** 2 >= min_area) + 1
        min_sharpness = MIN_SHARPNESS_FRAC * float(np.median(sharpness[big])) if big.size else 0.0

    road_lines = list(road_lines)
    road_index = None
    if road_lines:
        from shapely import STRtree
        road_index = STRtree(road_lines)

    wall_pts, wall_faces, roof_pts, roof_faces, roof_h, roof_s = [], [], [], [], [], []
    wp = rp = 0
    for geom, val in shapes(labels, mask=labels > 0, transform=transform, connectivity=4):
        lid = int(val)
        if counts[lid] * scene.dx ** 2 < min_area:
            continue
        h = float(heights[lid])
        if not np.isfinite(h) or h < min_height:
            continue
        if min_sharpness is not None and float(sharpness[lid]) < min_sharpness:
            continue
        poly = shape(geom).simplify(simplify_m, preserve_topology=True)
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area < min_area:
            continue
        if poly.buffer(-MIN_WIDTH_M / 2.0).area < MIN_CORE_FRAC * poly.area:
            continue
        if road_index is not None and _crosses_as_bridge(
                poly, [road_lines[i] for i in road_index.query(poly)]):
            continue
        ring = np.asarray(poly.exterior.coords)[:-1, :2]
        n = ring.shape[0]
        if n < 3:
            continue
        bottom = np.column_stack([ring, np.zeros(n)])
        top = np.column_stack([ring, np.full(n, h)])
        wall_pts.append(np.vstack([bottom, top]))
        i = np.arange(n)
        j = (i + 1) % n
        quads = np.column_stack([np.full(n, 4), wp + i, wp + j, wp + n + j, wp + n + i])
        wall_faces.append(quads.ravel())
        wp += 2 * n
        roof_pts.append(top)
        roof_faces.append(np.concatenate([[n], rp + i]))
        roof_h.append(h)
        roof_s.append(float(sharpness[lid]))
        rp += n

    if not roof_pts:
        return None
    walls = pv.PolyData(np.vstack(wall_pts), np.concatenate(wall_faces)).triangulate()
    roofs = pv.PolyData(np.vstack(roof_pts), np.concatenate(roof_faces))
    roofs.cell_data["height"] = np.asarray(roof_h, dtype=np.float32)
    roofs.cell_data["sharpness"] = np.asarray(roof_s, dtype=np.float32)
    roofs = roofs.triangulate()
    if scene.tex_extent is not None:
        tx0, tx1, ty0, ty1 = scene.tex_extent
        p = roofs.points
        roofs.active_texture_coordinates = np.column_stack(
            [(p[:, 0] - tx0) / (tx1 - tx0), (p[:, 1] - ty0) / (ty1 - ty0)]).astype(np.float32)
    return walls, roofs
