"""Procedural Mediterranean street/park trees for the twin (WG).

Replaces the sphere-on-cylinder placeholder with species models of the
trees that actually line Beirut's streets and parks:

  ficus       Ficus microcarpa/nitida — dense rounded dark-glossy crown (the
              most common street tree)
  jacaranda   Jacaranda mimosifolia — open umbrella crown on a forked trunk
  olive       Olea europaea — small, irregular, silver-green, gnarled trunk
  palm        Washingtonia robusta — tall slender trunk, crown of fan fronds
  stone_pine  Pinus pinea — high flat umbrella over a bare forked trunk
              (Horsh Beirut, seafront)

Models are built from geometry, not textures: crowns are clusters of
noise-displaced icospheres (lumpy silhouettes that read as foliage under
PBR + SSAO), trunks are tapered, leaning cylinders with forks toward the
crown masses, palm fronds are drooping tapered blades. Several seeded
variants per species give the canopy variety while staying compatible
with the app's glyph/drape pipeline (fixed templates, flat seeds at z=0:
every template is anchored at its trunk base).

Species come from OSM species/genus/taxon tags when present, otherwise a
deterministic, position-hashed draw from a Beirut street-tree mix.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pyvista as pv

N_VARIANTS = 3


@dataclass(frozen=True)
class TreeSpecies:
    name: str
    crown: str            # round | umbrella | irregular | palm | pine
    height: float         # total height (m)
    trunk_h: float        # clear trunk to crown base
    trunk_r: float
    crown_r: float        # crown radius
    crown_rgb: str
    trunk_rgb: str
    roughness: float


SPECIES = {
    "ficus": TreeSpecies("ficus", "round", 7.5, 2.4, 0.28, 3.4, "#2c4a22", "#6d6456", 0.55),
    "jacaranda": TreeSpecies("jacaranda", "umbrella", 8.5, 3.2, 0.24, 4.2, "#4e6d34", "#5d4a3a", 0.8),
    "olive": TreeSpecies("olive", "irregular", 5.0, 1.4, 0.26, 2.6, "#7b8660", "#5f564a", 0.85),
    "palm": TreeSpecies("palm", "palm", 14.0, 12.5, 0.30, 2.8, "#52702f", "#7a6a55", 0.7),
    "stone_pine": TreeSpecies("stone_pine", "pine", 12.0, 7.5, 0.33, 5.0, "#36522a", "#6e4a33", 0.85),
}
# Relative frequency when OSM does not say (Beirut street/park planting)
DEFAULT_MIX = {"ficus": 0.35, "jacaranda": 0.20, "olive": 0.15, "palm": 0.15, "stone_pine": 0.15}

_TAG_RULES = (
    (("ficus",), "ficus"),
    (("jacaranda",), "jacaranda"),
    (("olea", "olive"), "olive"),
    (("washingtonia", "phoenix", "palm", "arecaceae", "syagrus", "trachycarpus"), "palm"),
    (("pinus", "pine", "cedrus", "cedar", "cupressus"), "stone_pine"),
)


def species_for(tag: str, x: float, y: float) -> str:
    """Species key from an OSM tag string, else a deterministic draw by position."""
    t = (tag or "").lower()
    for keys, sp in _TAG_RULES:
        if any(k in t for k in keys):
            return sp
    h = zlib.crc32(f"{round(x, 1)}:{round(y, 1)}".encode()) / 2 ** 32
    acc = 0.0
    for sp, w in DEFAULT_MIX.items():
        acc += w
        if h < acc:
            return sp
    return "ficus"


def variant_for(x: float, y: float) -> int:
    return zlib.crc32(f"v{round(x, 1)}:{round(y, 1)}".encode()) % N_VARIANTS


def _noise3(p: np.ndarray, rng, freq: float) -> np.ndarray:
    """Cheap smooth 3-D noise: sum of random-direction sinusoids."""
    out = np.zeros(len(p))
    for _ in range(6):
        k = rng.normal(size=3)
        k *= freq / np.linalg.norm(k)
        out += np.sin(p @ k + rng.uniform(0, 2 * np.pi))
    return out / 6.0


def _blob(center, radii, rng, subdiv=2, bump=0.22) -> pv.PolyData:
    s = pv.Icosphere(radius=1.0, nsub=subdiv)
    pts = s.points.copy()
    n = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    # two octaves: crown massing + leaf-cluster lumpiness
    pts = n * (1.0 + bump * _noise3(n * 1.7, rng, 3.0) + 0.35 * bump * _noise3(n, rng, 9.0))[:, None]
    pts = pts * np.asarray(radii)[None, :] + np.asarray(center)[None, :]
    s.points = pts
    return s


def _limb(p0, p1, r0, r1, res=7) -> pv.PolyData:
    """Tapered cylinder from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    c = pv.Cylinder(center=(0, 0, L / 2), direction=(0, 0, 1), radius=1.0, height=L, resolution=res, capping=False)
    pts = c.points.copy()
    t = pts[:, 2] / max(L, 1e-9)
    rr = r0 + (r1 - r0) * t
    pts[:, :2] *= rr[:, None]
    # rotate +z onto d
    z = np.array([0.0, 0.0, 1.0])
    u = d / max(L, 1e-9)
    v = np.cross(z, u)
    s, cth = np.linalg.norm(v), float(z @ u)
    if s > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
        pts = pts @ R.T
    elif cth < 0:
        pts[:, 2] *= -1
    c.points = pts + p0
    return c


def _merge(parts) -> pv.PolyData:
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, merge_points=False)
    return out.triangulate().clean()


def tree_parts(sp: TreeSpecies, variant: int = 0) -> tuple[pv.PolyData, pv.PolyData]:
    """(trunk, crown) meshes for one species variant, trunk base at the
    origin, z up, metres."""
    rng = np.random.default_rng(zlib.crc32(sp.name.encode()) + 7919 * variant)
    scale = 0.85 + 0.3 * rng.random()
    H, th, R = sp.height * scale, sp.trunk_h * scale, sp.crown_r * scale
    lean = rng.normal(0, 0.06, 2)
    top = np.array([lean[0] * th, lean[1] * th, th])
    trunk = [_limb((0, 0, 0), top, sp.trunk_r * scale, sp.trunk_r * scale * 0.7)]
    crown = []
    if sp.crown == "palm":
        top = np.array([lean[0] * th * 0.5, lean[1] * th * 0.5, th])
        trunk = [_limb((0, 0, 0), top, sp.trunk_r * scale * 1.15, sp.trunk_r * scale * 0.8, res=9)]
        crown.append(_blob(top + (0, 0, 0.3), (0.55, 0.55, 0.7), rng, subdiv=1, bump=0.1))
        n_fr = 16
        for i in range(n_fr):
            az = 2 * np.pi * (i + rng.random() * 0.4) / n_fr
            up = rng.uniform(-0.2, 0.6)
            L = R * rng.uniform(0.85, 1.1)
            s = np.linspace(0, 1, 7)
            dirv = np.array([np.cos(az), np.sin(az)])
            xs = top[:2][None, :] + dirv[None, :] * (s * L)[:, None]
            zs = top[2] + 0.4 + (up * L) * s - 0.9 * L * s ** 2 * 0.6      # arch and droop
            w = 0.55 * np.sin(np.pi * np.clip(s * 0.95 + 0.05, 0, 1)) + 0.05
            side = np.array([-dirv[1], dirv[0]])
            left = np.column_stack([xs + side * w[:, None], zs])
            right = np.column_stack([xs - side * w[:, None], zs - 0.08])
            pts = np.vstack([left, right])
            k = len(s)
            faces = []
            for j in range(k - 1):
                faces += [3, j, j + 1, k + j, 3, j + 1, k + j + 1, k + j]
            crown.append(pv.PolyData(pts, np.array(faces)))
        return _merge(trunk), _merge(crown)

    if sp.crown == "pine":
        fork = top
        for i in range(3):
            az = 2 * np.pi * i / 3 + rng.random()
            tip = fork + (np.cos(az) * R * 0.45, np.sin(az) * R * 0.45, (H - th) * 0.35)
            trunk.append(_limb(fork, tip, sp.trunk_r * scale * 0.6, sp.trunk_r * scale * 0.3))
        zc = th + (H - th) * 0.55
        for i in range(7):
            ang = 2 * np.pi * rng.random()
            rad = R * np.sqrt(rng.random()) * 0.65
            crown.append(_blob((np.cos(ang) * rad, np.sin(ang) * rad, zc + rng.normal(0, 0.25)),
                               (R * 0.45, R * 0.45, (H - th) * 0.28), rng))
        return _merge(trunk), _merge(crown)

    if sp.crown == "umbrella":
        n, zc, rz, spread = 6, th + (H - th) * 0.55, (H - th) * 0.32, 0.6
    elif sp.crown == "irregular":
        n, zc, rz, spread = 5, th + (H - th) * 0.5, (H - th) * 0.35, 0.55
    else:                                                    # round
        n, zc, rz, spread = 6, th + (H - th) * 0.5, (H - th) * 0.42, 0.45
    for i in range(n):
        ang = 2 * np.pi * i / n + rng.normal(0, 0.3)
        rad = R * spread * np.sqrt(rng.uniform(0.3, 1.0))
        c = np.array([np.cos(ang) * rad, np.sin(ang) * rad, zc + rng.normal(0, rz * 0.25)])
        rr = R * rng.uniform(0.45, 0.62)
        crown.append(_blob(c, (rr, rr, rz * rng.uniform(0.8, 1.05)), rng))
        if i % 2 == 0:                                       # limbs into the crown
            trunk.append(_limb(top, c * np.array([0.7, 0.7, 1.0]), sp.trunk_r * scale * 0.55,
                               sp.trunk_r * scale * 0.25))
    crown.append(_blob((0, 0, zc + rz * 0.35), (R * 0.55, R * 0.55, rz * 0.8), rng))
    return _merge(trunk), _merge(crown)


_CACHE: dict = {}


def templates(species: str, variant: int):
    key = (species, variant)
    if key not in _CACHE:
        _CACHE[key] = tree_parts(SPECIES[species], variant)
    return _CACHE[key]


def tree_keys(prefix: str, x: float, y: float, tag: str = "") -> tuple[str, str]:
    """(trunk_key, canopy_key) that add_tree_groups uses for a tree at (x, y)."""
    sp, v = species_for(tag, x, y), variant_for(x, y)
    return f"{prefix}_trunk_{sp}_{v}", f"{prefix}_canopy_{sp}_{v}"


def add_tree_groups(plotter, scene_state: dict, prefix: str, xy: np.ndarray, tags=None,
                    base_z: float = 0.0) -> int:
    """Register and render trees at xy (flat) grouped by (species, variant),
    using the app's drape contract: scene_state['_tree_seeds_flat' /
    '_tree_templates' / '_tree_actor_kwargs'][key], actor name '_tree_<key>'.
    Returns the number of actor groups created."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    tags = list(tags) if tags is not None else [""] * len(xy)
    seeds = scene_state.setdefault("_tree_seeds_flat", {})
    tmpls = scene_state.setdefault("_tree_templates", {})
    kws = scene_state.setdefault("_tree_actor_kwargs", {})
    groups: dict[tuple, list] = {}
    for (x, y), tag in zip(xy, tags):
        groups.setdefault((species_for(tag, x, y), variant_for(x, y)), []).append((x, y))
    for (sp, v), pts in groups.items():
        trunk, crown = templates(sp, v)
        spc = SPECIES[sp]
        flat = np.column_stack([np.asarray(pts), np.zeros(len(pts))])
        for part, tmpl, colour, rough in (("trunk", trunk, spc.trunk_rgb, 0.9),
                                          ("canopy", crown, spc.crown_rgb, spc.roughness)):
            key = f"{prefix}_{part}_{sp}_{v}"
            kw = dict(color=colour, smooth_shading=True, pbr=True, roughness=rough, metallic=0.0)
            seeds[key], tmpls[key], kws[key] = flat, tmpl, kw
            shown = flat.copy()
            shown[:, 2] += base_z          # e.g. placed while the terrain drape is active
            scene_state[f"_tree_{key}"] = plotter.add_mesh(
                pv.PolyData(shown).glyph(geom=tmpl, orient=False, scale=False),
                name=f"_tree_{key}", reset_camera=False, **kw)
    return len(groups)
