"""Reconcile the twin's building massing with the drone survey (WG).

Overture/OSM footprints come with a height (often a round-number fallback)
and reflect whatever was mapped — in Beirut that includes port warehouses
destroyed in August 2020. The survey's normalized DSM (surface minus bare
earth, datum-free) is direct evidence of what stands on each footprint.

Per building (a connected component of the extruded mesh):
  * sample nDSM on the survey grid inside the footprint;
  * exclude vegetation (ortho excess-green) so tree canopy is not a roof;
  * occupied fraction = share of valid samples with nDSM above min_height;
  * roof continuity  = largest 4-connected occupied region / valid samples.
    A real roof is ONE continuous raised surface over most of its footprint;
    stacked containers or parked trucks on a cleared lot are many separate
    blocks split by aisles — they can raise occupancy (tall stacks) but not
    continuity. This is what separates the post-2020 container yards from
    buildings in the Beirut port.
  * present   (continuity >= present_frac): height := robust roof height;
    absent    (continuity <  absent_frac):  removed — nothing stands there;
    uncertain (between):                    kept at source height, flagged;
    no_survey (too few valid samples):      untouched.

Before any of this the whole footprint layer is co-registered to the nDSM:
the (dx, dy) shift maximizing total roof overlap is found on a small search
grid and applied to the building geometry itself, since the surveyed roofs
are the positional truth (and the rendered massing then sits on the photo's
roofs). The shift is reported so it can be checked against expectations.

Footprint offsets are not uniform: source imagery tiles differ, and in the
Mar Mikhael blocks Overture sits ~5 m east of the survey while elsewhere it
agrees to ~1 m. So registration is a smooth LOCAL offset field rather than
one global shift (and rather than a free per-building search, which overfits
small footprints onto whatever raised structure is nearby): each building is
evaluated at the shift that maximizes the mean roof overlap of all compact
footprints within REGISTRATION_RADIUS_M of it, itself included.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_HEIGHT_M = 2.5
PRESENT_FRAC = 0.55
ABSENT_FRAC = 0.30
MIN_VALID_FRAC = 0.6
REGISTRATION_SEARCH_M = 4.0
LOCAL_SEARCH_M = 6.0
REGISTRATION_RADIUS_M = 80.0
MAX_VOTER_AREA_M2 = 2000.0
MIN_LOCAL_OVERLAP = 0.5
MIN_LOCAL_GAIN = 0.03
ROOF_PERCENTILE = 90.0
EXG_VEGETATION = 0.06


@dataclass
class BuildingEvidence:
    region: int
    status: str
    centroid: tuple
    area_m2: float
    old_height: float
    new_height: float
    occupied_frac: float
    valid_frac: float
    n_samples: int
    continuity: float = 0.0
    relocated_by: tuple = (0.0, 0.0)


@dataclass
class ReconcileReport:
    buildings: list = field(default_factory=list)
    shift_xy: tuple = (0.0, 0.0)

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for b in self.buildings:
            out[b.status] = out.get(b.status, 0) + 1
        return out

    def summary(self) -> str:
        c = self.counts()
        pres = [b for b in self.buildings if b.status == "present"]
        dh = np.array([b.new_height - b.old_height for b in pres]) if pres else np.array([0.0])
        return (f"{len(self.buildings)} buildings (footprints co-registered by "
                f"({self.shift_xy[0]:+.1f}, {self.shift_xy[1]:+.1f}) m): " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))
                + (f"; re-heighted median {np.median(dh):+.1f} m (|dh| p90 {np.percentile(np.abs(dh), 90):.1f} m)"
                   if pres else ""))


def _excess_green(rgb: np.ndarray) -> np.ndarray:
    """Chromatic excess-green 2g - r - b on normalized rgb (Woebbecke 1995)."""
    f = rgb.astype(np.float32)
    s = f.sum(axis=1) + 1e-6
    r, g, b = f[:, 0] / s, f[:, 1] / s, f[:, 2] / s
    return 2.0 * g - r - b


def _footprint(component) -> "object | None":
    """2D footprint = union of the component's top (roof) triangles."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    pts = np.asarray(component.points)
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    if zmax - zmin < 1e-6:
        return None
    faces = component.faces.reshape(-1, 4) if component.faces.size % 4 == 0 else None
    polys = []
    if faces is not None and np.all(faces[:, 0] == 3):
        tri = pts[faces[:, 1:]]
        top = np.all(tri[:, :, 2] > zmin + 0.5 * (zmax - zmin), axis=1)
        for t in tri[top]:
            p = Polygon(t[:, :2])
            if p.area > 1e-6:
                polys.append(p)
    if not polys:
        from shapely.geometry import MultiPoint
        return MultiPoint(pts[:, :2]).convex_hull
    return unary_union(polys).buffer(0)


def _footprint_samples(fp, scene):
    """Survey-grid cell centres inside a footprint, as (row, col) and xy."""
    import shapely
    minx, miny, maxx, maxy = fp.bounds
    gx = np.arange(np.floor((minx - scene.x0) / scene.dx), np.ceil((maxx - scene.x0) / scene.dx) + 1).astype(int)
    gy = np.arange(np.floor((miny - scene.y0) / scene.dx), np.ceil((maxy - scene.y0) / scene.dx) + 1).astype(int)
    CX, CY = np.meshgrid(gx, gy)
    X, Y = scene.x0 + CX * scene.dx, scene.y0 + CY * scene.dx
    inside = shapely.contains_xy(fp, X.ravel(), Y.ravel())
    return CY.ravel()[inside], CX.ravel()[inside]


def _built_mask(scene, rows, cols, min_height):
    ny, nx = scene.ndsm.shape
    ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    vals = np.full(rows.shape, np.nan)
    vals[ok] = scene.ndsm[rows[ok], cols[ok]]
    xy = np.column_stack([scene.x0 + cols * scene.dx, scene.y0 + rows * scene.dx])
    exg = _excess_green(scene.rgb_at(xy))
    valid = np.isfinite(vals)
    return vals, valid, valid & (vals > min_height) & (exg < EXG_VEGETATION)


def _continuity(rows, cols, built) -> float:
    """Largest 4-connected component of `built` over the footprint samples."""
    from scipy.ndimage import label
    if not built.any():
        return 0.0
    r0, c0 = rows.min(), cols.min()
    grid = np.zeros((rows.max() - r0 + 1, cols.max() - c0 + 1), dtype=bool)
    grid[rows[built] - r0, cols[built] - c0] = True
    lab, n = label(grid)
    return float(np.bincount(lab.ravel())[1:].max()) if n else 0.0


class _Evidence:
    """Scene-wide roof-evidence rasters, so every shift test is array indexing."""

    def __init__(self, scene, min_height):
        self.scene = scene
        ny, nx = scene.ndsm.shape
        self.valid = np.isfinite(scene.ndsm)
        gx = scene.x0 + np.arange(nx) * scene.dx
        gy = scene.y0 + np.arange(ny) * scene.dx
        GX, GY = np.meshgrid(gx, gy)
        exg = _excess_green(scene.rgb_at(np.column_stack([GX.ravel(), GY.ravel()]))).reshape(ny, nx)
        self.built = self.valid & (np.nan_to_num(scene.ndsm, nan=-1.0) > min_height) & (exg < EXG_VEGETATION)

    def at(self, rows, cols, di=0, dj=0):
        ny, nx = self.built.shape
        r, c = rows + di, cols + dj
        ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
        valid = np.zeros(r.shape, bool)
        built = np.zeros(r.shape, bool)
        valid[ok] = self.valid[r[ok], c[ok]]
        built[ok] = self.built[r[ok], c[ok]]
        return valid, built


def _shift_table(search_m: float, dx: float):
    """Candidate (di, dj) cell shifts within search_m, in the original scan
    order (di outer, dj inner, ascending), plus their squared norms."""
    k = int(round(search_m / dx))
    di, dj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
    di, dj = di.ravel(), dj.ravel()
    keep = (di * di + dj * dj) * dx ** 2 <= search_m ** 2 + 1e-9
    return di[keep], dj[keep], (di[keep] ** 2 + dj[keep] ** 2)


def _voter_scores(ev, samples, di, dj) -> np.ndarray:
    """(n_voters, n_shifts) built-overlap fraction of every voter footprint
    at every candidate shift: one padded fancy-index gather per voter (no
    Python loop over shifts). Equivalent to ev.at(rows, cols, di, dj)[1].mean()."""
    k = int(max(np.abs(di).max(), np.abs(dj).max())) if len(di) else 0
    if not hasattr(ev, "_padded") or ev._padded[0] < k:
        ev._padded = (k, np.pad(ev.built, k, constant_values=False))
    kk, B = ev._padded
    out = np.zeros((len(samples), len(di)), dtype=np.float64)
    for n, (rows, cols) in enumerate(samples):
        if rows.size == 0:
            continue
        hit = B[rows[:, None] + di[None, :] + kk, cols[:, None] + dj[None, :] + kk]
        out[n] = hit.sum(axis=0) / rows.size
    return out


def _argbest(score: np.ndarray, norm2: np.ndarray) -> int:
    """Index of the best shift with the original tie-break: highest score;
    within 1e-9 of it the smallest shift; then the earliest in scan order."""
    cand = np.flatnonzero(score >= score.max() - 1e-9)
    return int(cand[np.argmin(norm2[cand])])      # argmin returns the first (scan-order) minimum


def coregister_footprints(footprints, scene, min_height: float = MIN_HEIGHT_M,
                          search_m: float = REGISTRATION_SEARCH_M, max_area_m2: float = MAX_VOTER_AREA_M2,
                          min_occupancy: float = 0.5, evidence: "_Evidence | None" = None) -> tuple[float, float]:
    """Single global (dx, dy) for the layer (reported for reference): voted by
    compact footprints already clearly occupied at zero shift, equal weight."""
    ev = evidence or _Evidence(scene, min_height)
    voters = []
    for fp in footprints:
        if fp is None or fp.is_empty or fp.area > max_area_m2:
            continue
        rows, cols = _footprint_samples(fp, scene)
        if rows.size < 4:
            continue
        valid, built = ev.at(rows, cols)
        if valid.mean() > 0.9 and built.sum() / max(valid.sum(), 1) >= min_occupancy:
            voters.append((rows, cols))
    if len(voters) < 3:
        return 0.0, 0.0
    di, dj, n2 = _shift_table(search_m, ev.scene.dx)
    best = _argbest(_voter_scores(ev, voters, di, dj).sum(axis=0), n2)
    return dj[best] * ev.scene.dx, di[best] * ev.scene.dx


def _mean_overlap(voters, ev, dx, dy):
    di, dj = int(round(dy / ev.scene.dx)), int(round(dx / ev.scene.dx))
    return float(np.mean([ev.at(rows, cols, di, dj)[1].sum() / max(rows.size, 1) for rows, cols in voters]))


def local_offsets(footprints, scene, evidence, global_shift=(0.0, 0.0),
                  radius_m: float = REGISTRATION_RADIUS_M, search_m: float = LOCAL_SEARCH_M,
                  max_area_m2: float = MAX_VOTER_AREA_M2):
    """Per-footprint (dx, dy) from a neighbourhood-voted rigid co-registration.

    The local optimum is only trusted where the neighbourhood actually has
    roofs to register against (mean overlap >= MIN_LOCAL_OVERLAP and at least
    MIN_LOCAL_GAIN better than the global shift). Where the voters are all
    phantom footprints on cleared lots, any "best" shift is noise, so the
    global shift is used instead.

    Vectorized: every compact voter is scored once against every shift
    (V: voters x shifts); a neighbourhood's score is the sum of its voters'
    rows (sparse membership @ V). O(voters * shifts) gathers instead of
    O(buildings * neighbours * shifts) Python-level lookups (11 min -> seconds
    at corridor scale, ~1700 buildings)."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    samples, cents = {}, {}
    for r, fp in enumerate(footprints):
        if fp is None or fp.is_empty:
            continue
        rows, cols = _footprint_samples(fp, scene)
        if rows.size:
            samples[r] = (rows, cols)
            cents[r] = (fp.centroid.x, fp.centroid.y)
    if not samples:
        return {}
    di, dj, n2 = _shift_table(search_m, evidence.scene.dx)
    gi = int(round(global_shift[1] / evidence.scene.dx))
    gj = int(round(global_shift[0] / evidence.scene.dx))
    g_idx = np.flatnonzero((di == gi) & (dj == gj))
    compact = [r for r in samples if footprints[r].area <= max_area_m2]
    V = _voter_scores(evidence, [samples[j] for j in compact], di, dj) if compact else np.zeros((0, len(di)))
    tree = cKDTree(np.array([cents[j] for j in compact])) if compact else None
    order = list(samples)
    members = [tree.query_ball_point(cents[r], radius_m) if tree is not None else [] for r in order]
    out = {}
    lone = [n for n, m in enumerate(members) if not m]              # no compact voter nearby: vote alone
    if compact:
        rows_i = np.repeat(np.arange(len(order)), [len(m) for m in members])
        cols_i = np.concatenate([np.asarray(m, dtype=int) for m in members if m]) if len(rows_i) else np.zeros(0, int)
        A = csr_matrix((np.ones(len(rows_i)), (rows_i, cols_i)), shape=(len(order), len(compact)))
        S = np.asarray(A @ V)
        counts = np.array([max(len(m), 1) for m in members], dtype=float)
    if lone:
        V_lone = _voter_scores(evidence, [samples[order[n]] for n in lone], di, dj)
    lone_pos = {n: i for i, n in enumerate(lone)}
    gshift = tuple(global_shift)
    for n, r in enumerate(order):
        if n in lone_pos:
            score, cnt = V_lone[lone_pos[n]], 1.0
        else:
            score, cnt = S[n], counts[n]
        b = _argbest(score, n2)
        m_local = score[b] / cnt
        m_global = (score[g_idx[0]] / cnt) if g_idx.size else _mean_overlap(
            [samples[order[n]]] if n in lone_pos else [samples[compact[j]] for j in members[n]],
            evidence, *gshift)
        trusted = m_local >= MIN_LOCAL_OVERLAP and m_local - m_global >= MIN_LOCAL_GAIN
        out[r] = (float(dj[b] * evidence.scene.dx), float(di[b] * evidence.scene.dx)) if trusted else gshift
    return out


def _component_footprints(conn, rid):
    """Footprint per connected component, same definition as _footprint()
    (union of the top-half triangles; convex hull fallback; None if flat),
    computed from the whole mesh's arrays at once instead of one VTK
    extract per building (1700+ filter calls at corridor scale)."""
    import shapely
    from shapely.geometry import MultiPoint
    rid = np.asarray(rid)
    n_reg = int(rid.max()) + 1 if rid.size else 0
    faces = np.asarray(conn.faces)
    if n_reg == 0 or faces.size % 4 != 0 or not np.all(faces.reshape(-1, 4)[:, 0] == 3):
        return [_footprint(conn.extract_cells(np.flatnonzero(rid == r))
                           .extract_surface(algorithm="dataset_surface").triangulate())
                for r in range(n_reg)]
    pts = np.asarray(conn.points)
    tri = pts[faces.reshape(-1, 4)[:, 1:]]                      # (n_tri, 3, 3)
    zmin = np.full(n_reg, np.inf)
    zmax = np.full(n_reg, -np.inf)
    np.minimum.at(zmin, rid, tri[:, :, 2].min(axis=1))
    np.maximum.at(zmax, rid, tri[:, :, 2].max(axis=1))
    tall = (zmax - zmin) >= 1e-6
    thr = zmin + 0.5 * (zmax - zmin)
    top = tall[rid] & np.all(tri[:, :, 2] > thr[rid][:, None], axis=1)
    polys = shapely.polygons(tri[top][:, :, :2])
    ok = shapely.area(polys) > 1e-6
    t_rid = rid[top][ok]
    polys = polys[ok]
    order = np.argsort(t_rid, kind="stable")
    bounds = np.searchsorted(t_rid[order], np.arange(n_reg + 1))
    out = []
    for r in range(n_reg):
        if not tall[r]:
            out.append(None)
            continue
        idx = order[bounds[r]:bounds[r + 1]]
        if idx.size:
            out.append(shapely.union_all(polys[idx]).buffer(0))
        else:
            out.append(MultiPoint(tri[rid == r].reshape(-1, 3)[:, :2]).convex_hull)
    return out


def building_footprints(mesh):
    """Shapely footprint per connected building volume of `mesh` (None if degenerate)."""
    if mesh is None or mesh.n_cells == 0:
        return []
    conn = mesh.triangulate().connectivity()
    return _component_footprints(conn, np.asarray(conn.cell_data["RegionId"]))


def reconcile_buildings(mesh, scene, min_height: float = MIN_HEIGHT_M,
                        present_frac: float = PRESENT_FRAC, absent_frac: float = ABSENT_FRAC,
                        min_valid_frac: float = MIN_VALID_FRAC, coregister: bool = True):
    """Return (reconciled pv.PolyData, ReconcileReport). Cell data (e.g.
    building_class) is preserved; absent buildings are removed; present
    ones are rescaled vertically to the measured roof height; each building
    is translated by its local co-registration offset."""
    from shapely.affinity import translate
    if scene is None or getattr(scene, "ndsm", None) is None:
        return mesh, ReconcileReport()

    tri = mesh.triangulate()
    conn = tri.connectivity()
    rid = np.asarray(conn.cell_data["RegionId"])
    pts = np.array(conn.points, dtype=float)
    point_rid = np.asarray(conn.point_data["RegionId"])
    keep_cells = np.ones(conn.n_cells, dtype=bool)
    report = ReconcileReport()
    ev = _Evidence(scene, min_height)

    footprints = _component_footprints(conn, rid)

    report.shift_xy = coregister_footprints(footprints, scene, min_height, evidence=ev) if coregister else (0.0, 0.0)
    offsets = local_offsets(footprints, scene, ev, report.shift_xy) if coregister else {}

    for r, fp in enumerate(footprints):
        if fp is None or fp.is_empty:
            continue
        dx, dy = offsets.get(r, (0.0, 0.0))
        fp = translate(fp, dx, dy)
        cells = np.flatnonzero(rid == r)
        pmask = point_rid == r
        pts[pmask, 0] += dx
        pts[pmask, 1] += dy
        zb = pts[pmask, 2].min()
        h_old = float(pts[pmask, 2].max() - zb)

        rows, cols = _footprint_samples(fp, scene)
        if rows.size < 4:
            c = fp.representative_point()
            rows = np.array([int(round((c.y - scene.y0) / scene.dx))])
            cols = np.array([int(round((c.x - scene.x0) / scene.dx))])
        valid, built = ev.at(rows, cols)
        ny, nx = scene.ndsm.shape
        inb = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
        vals = np.full(rows.shape, np.nan)
        vals[inb] = scene.ndsm[rows[inb], cols[inb]]
        n_valid = max(int(valid.sum()), 1)
        valid_frac = float(valid.mean())
        occ = float(built.sum() / n_valid)
        cont = _continuity(rows, cols, built) / n_valid

        if valid_frac < min_valid_frac:
            status, h_new = "no_survey", h_old
        elif cont < absent_frac:
            status, h_new = "absent", 0.0
            keep_cells[cells] = False
        elif cont < present_frac:
            status, h_new = "uncertain", h_old
        else:
            status = "present"
            h_new = float(np.clip(np.percentile(vals[built], ROOF_PERCENTILE), min_height, 250.0))
            if h_old > 1e-6:
                zp = pts[pmask, 2]
                pts[pmask, 2] = zb + (zp - zb) * (h_new / h_old)

        ctr = fp.centroid
        report.buildings.append(BuildingEvidence(
            region=r, status=status, centroid=(float(ctr.x), float(ctr.y)), area_m2=float(fp.area),
            old_height=h_old, new_height=h_new, occupied_frac=occ, valid_frac=valid_frac,
            n_samples=int(rows.size), continuity=cont, relocated_by=(dx, dy)))

    conn.points = pts
    out = conn.extract_cells(np.flatnonzero(keep_cells)).extract_surface(algorithm="dataset_surface")
    for name in ("RegionId", "vtkOriginalCellIds", "vtkOriginalPointIds"):
        for data in (out.cell_data, out.point_data):
            if name in data:
                del data[name]
    return out, report
