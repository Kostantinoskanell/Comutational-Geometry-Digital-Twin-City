"""Physically based sky environment maps (WG lighting).

Rayleigh + Mie scattering in an exponentially stratified atmosphere:
single scattering integrated along each view ray (Nishita et al. 1993)
plus multiple scattering from Hillaire's (EGSR 2020) Psi_ms LUT, with a
Beirut-realistic aerosol optical depth. Driven by the true sun
direction, it yields an HDR equirectangular radiance map used for
image-based lighting and reflections, so the twin needs no downloaded
HDRI assets and the sky always matches the analysed hour.

Frame: x = east, y = north, z = up (the scene's local AEQD frame).
Units: radiance relative to a unit-irradiance sun (see VTK_SKY_SCALE). The sun disc itself is left out: the
scene's directional light already carries direct sun, and a near-delta
peak in the environment would be double counted and alias in the
prefiltered specular maps.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

R_EARTH = 6_360e3
R_ATMOS = 6_420e3
BETA_R = np.array([5.802e-6, 13.558e-6, 33.1e-6])   # Rayleigh scattering, sea level (1/m), RGB 680/550/440 nm
H_R = 8_000.0
# Aerosols (Mie) from an aerosol optical depth rather than clean-air
# defaults: urban-coastal Beirut sits around AOD(550 nm) 0.2-0.4 (AERONET /
# MODIS climatology); clean-air textbook values (AOD ~0.005) give a
# too-deep-blue, too-dark sky with far too little diffuse light.
AEROSOL_AOD_550 = 0.25
H_M = 1_200.0                                       # aerosol scale height (boundary layer)
MIE_SSA = 0.9                                       # single-scattering albedo, urban aerosol
BETA_M_EXT = AEROSOL_AOD_550 / H_M                  # Mie extinction, sea level (1/m)
BETA_M = BETA_M_EXT * MIE_SSA                       # Mie scattering, sea level (1/m)
MIE_G = 0.76
OBSERVER_ALT_M = 50.0
GROUND_ALBEDO = 0.18
NIGHT_RADIANCE = np.array([0.0020, 0.0024, 0.0040])  # airglow + urban skyglow floor
# Absolute scale into the twin's VTK lighting units: 1.0, i.e. none — the
# model is in units of a unit-irradiance sun, and the app's noon sun light
# delivers ~0.83 (x0.85 diffuse) vs the model's transmitted ~0.72, close
# enough that no correction is warranted. VALIDATION (not a fit): a white
# Lambertian plane rendered by VTK under this sky alone vs under the app's
# sun alone gives diffuse/direct 0.30 at noon and 0.38 at 09:00 (June),
# i.e. diffuse fractions 0.23 / 0.28 — plausible clear-sky values for
# AOD ~0.25. (Clean-air aerosol would have needed a fudge factor of ~2.5.)
VTK_SKY_SCALE = 1.0


def _ray_sphere_far(origin: np.ndarray, d: np.ndarray, radius: float) -> np.ndarray:
    """Distance along unit rays d from origin (inside the sphere) to its surface."""
    b = np.einsum("...i,...i->...", origin, d)
    c = np.einsum("...i,...i->...", origin, origin) - radius * radius
    return -b + np.sqrt(np.maximum(b * b - c, 0.0))


def _ground_hit(origin: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance to the ground sphere, +inf when the ray misses it."""
    b = np.einsum("...i,...i->...", origin, d)
    c = np.einsum("...i,...i->...", origin, origin) - R_EARTH * R_EARTH
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc > 0.0) & (t > 0.0), t, np.inf)


def _quadratic_step(t_total: np.ndarray, i: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint and length of step i of n with quadratic spacing along a ray:
    dense near the origin, where the (1.2 km scale-height) aerosol lives —
    uniform steps along a ~600 km horizon ray would skip it entirely."""
    a, b = (i / n) ** 2, ((i + 1) / n) ** 2
    return t_total * (0.5 * (a + b)), t_total * (b - a)


def _optical_depth_to_sun(p: np.ndarray, sun: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rayleigh / Mie optical depth from points p to the top of the atmosphere
    along the sun direction; inf where the Earth blocks the sun."""
    sd = np.broadcast_to(sun, p.shape)
    t_top = _ray_sphere_far(p, sd, R_ATMOS)
    blocked = np.isfinite(_ground_hit(p, sd))
    od_r = np.zeros(p.shape[:-1])
    od_m = np.zeros(p.shape[:-1])
    for j in range(n):
        t_mid, ds = _quadratic_step(t_top, j, n)
        q = p + sd * t_mid[..., None]
        h = np.maximum(np.linalg.norm(q, axis=-1) - R_EARTH, 0.0)   # blocked rays dip underground
        od_r += np.exp(-h / H_R) * ds
        od_m += np.exp(-h / H_M) * ds
    od_r[blocked] = np.inf
    od_m[blocked] = np.inf
    return od_r, od_m


def _sigma_s(h: np.ndarray) -> np.ndarray:
    """Scattering coefficient (..., 3) at altitude h (m)."""
    h = np.asarray(h, dtype=float)[..., None]
    return BETA_R * np.exp(-h / H_R) + BETA_M * np.exp(-h / H_M)


def _sigma_t(h: np.ndarray) -> np.ndarray:
    """Extinction coefficient (..., 3) at altitude h (m)."""
    h = np.asarray(h, dtype=float)[..., None]
    return BETA_R * np.exp(-h / H_R) + BETA_M_EXT * np.exp(-h / H_M)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azim = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)], axis=-1)


MS_LUT_H, MS_LUT_MU, MS_DIRS, MS_STEPS = 32, 32, 64, 20
_MS_LUT = None


def multiple_scattering_lut() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hillaire (2020) multiple-scattering LUT Psi_ms(h, mu_sun), RGB.

    At each altitude / sun angle: second-order radiance L2 gathered over the
    sphere with an isotropic phase (incl. the sunlit Lambertian ground), and
    the transfer factor f_ms of one more isotropic bounce; infinite orders
    sum as a geometric series, Psi_ms = L2 / (1 - f_ms). Added to the single
    scattering it removes the too-dark, too-yellow horizon of single-scatter
    skies. Returns (heights, mus, lut[h, mu, 3]).
    """
    global _MS_LUT
    if _MS_LUT is not None:
        return _MS_LUT
    hs = (np.linspace(0.0, 1.0, MS_LUT_H) ** 2) * (R_ATMOS - R_EARTH)
    mus = np.linspace(-1.0, 1.0, MS_LUT_MU)
    dirs = _fibonacci_sphere(MS_DIRS)
    p_u = 1.0 / (4.0 * np.pi)
    lut = np.zeros((MS_LUT_H, MS_LUT_MU, 3))
    for ih, h in enumerate(hs):
        x = np.array([0.0, 0.0, R_EARTH + h + 1.0])
        o = np.broadcast_to(x, dirs.shape)
        t_ground = _ground_hit(o, dirs)
        hits_ground = np.isfinite(t_ground)
        t_end = np.minimum(_ray_sphere_far(o, dirs, R_ATMOS), t_ground)
        edges = (np.arange(MS_STEPS + 1) / MS_STEPS) ** 2                   # quadratic spacing
        ts = t_end[:, None] * (0.5 * (edges[:-1] + edges[1:]))[None, :]    # (D, S)
        dt = t_end[:, None] * np.diff(edges)[None, :]                      # (D, S)
        p = x + dirs[:, None, :] * ts[..., None]                           # (D, S, 3)
        hp = np.maximum(np.linalg.norm(p, axis=-1) - R_EARTH, 0.0)
        ext = _sigma_t(hp) * dt[..., None]                                 # (D, S, 3)
        tau = np.cumsum(ext, axis=1) - 0.5 * ext                           # midpoint optical depth
        t_view = np.exp(-tau)
        scat = _sigma_s(hp) * dt[..., None]
        f_ms = (t_view * scat).sum(axis=1).mean(axis=0)                    # (3,)  (4pi/D * p_u = 1/D)
        t_end_view = np.exp(-ext.sum(axis=1))                              # (D, 3)
        pg = x + dirs * np.where(hits_ground, t_ground, 0.0)[:, None]
        n_g = pg / np.linalg.norm(pg, axis=-1, keepdims=True)
        flat_p = p.reshape(-1, 3)
        for im, mu in enumerate(mus):
            sun = np.array([np.sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu])
            od_r, od_m = _optical_depth_to_sun(flat_p, sun, 8)
            t_sun = np.exp(-(BETA_R * od_r[:, None] + BETA_M_EXT * od_m[:, None])).reshape(p.shape)
            l2 = (t_view * scat * t_sun).sum(axis=1) * p_u                 # (D, 3)
            if hits_ground.any():
                gr, gm = _optical_depth_to_sun(pg[hits_ground], sun, 8)
                t_sun_g = np.exp(-(BETA_R * gr[:, None] + BETA_M_EXT * gm[:, None]))
                cos_g = np.clip(n_g[hits_ground] @ sun, 0.0, None)[:, None]
                l2[hits_ground] += t_end_view[hits_ground] * (GROUND_ALBEDO / np.pi) * cos_g * t_sun_g
            lut[ih, im] = l2.mean(axis=0) / np.maximum(1.0 - f_ms, 1e-6)
    _MS_LUT = (hs, mus, lut)
    return _MS_LUT


def _ms_lookup(h: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Bilinear Psi_ms at altitudes h and sun cosines mu -> (N, 3)."""
    hs, mus, lut = multiple_scattering_lut()
    u = np.sqrt(np.clip(h / (R_ATMOS - R_EARTH), 0.0, 1.0)) * (MS_LUT_H - 1)   # inverse of the h^2 spacing
    v = (np.clip(mu, -1.0, 1.0) + 1.0) / 2.0 * (MS_LUT_MU - 1)
    i0 = np.minimum(u.astype(np.int64), MS_LUT_H - 2)
    j0 = np.minimum(v.astype(np.int64), MS_LUT_MU - 2)
    fu, fv = (u - i0)[:, None], (v - j0)[:, None]
    return ((lut[i0, j0] * (1 - fv) + lut[i0, j0 + 1] * fv) * (1 - fu)
            + (lut[i0 + 1, j0] * (1 - fv) + lut[i0 + 1, j0 + 1] * fv) * fu)


def sky_radiance(dirs: np.ndarray, sun_dir: np.ndarray, n_view: int = 32, n_light: int = 12,
                 multiple_scattering: bool = True) -> np.ndarray:
    """Sky radiance (N, 3) for unit view directions (N, 3): single scattering
    plus (by default) Hillaire's multiple-scattering term."""
    d = np.asarray(dirs, dtype=float)
    s = np.asarray(sun_dir, dtype=float)
    s = s / np.linalg.norm(s)
    origin = np.array([0.0, 0.0, R_EARTH + OBSERVER_ALT_M])
    o = np.broadcast_to(origin, d.shape)
    t_max = np.minimum(_ray_sphere_far(o, d, R_ATMOS), _ground_hit(o, d))

    mu = d @ s
    phase_r = 3.0 / (16.0 * np.pi) * (1.0 + mu * mu)
    g2 = MIE_G * MIE_G
    phase_m = (3.0 / (8.0 * np.pi) * (1.0 - g2) * (1.0 + mu * mu)
               / ((2.0 + g2) * np.power(1.0 + g2 - 2.0 * MIE_G * mu, 1.5)))

    sum_r = np.zeros(d.shape)
    sum_m = np.zeros(d.shape)
    sum_ms = np.zeros(d.shape)
    od_r_view = np.zeros(d.shape[0])
    od_m_view = np.zeros(d.shape[0])
    for i in range(n_view):
        t_mid, ds = _quadratic_step(t_max, i, n_view)
        p = o + d * t_mid[:, None]
        r = np.linalg.norm(p, axis=-1)
        h = r - R_EARTH
        hr = np.exp(-h / H_R) * ds
        hm = np.exp(-h / H_M) * ds
        od_r_view += hr
        od_m_view += hm
        lr, lm = _optical_depth_to_sun(p, s, n_light)
        tau = BETA_R[None, :] * (od_r_view + lr)[:, None] + BETA_M_EXT * (od_m_view + lm)[:, None]
        att = np.exp(-tau)
        sum_r += att * hr[:, None]
        sum_m += att * hm[:, None]
        if multiple_scattering:
            t_view = np.exp(-(BETA_R[None, :] * od_r_view[:, None] + BETA_M_EXT * od_m_view[:, None]))
            scat = BETA_R[None, :] * hr[:, None] + BETA_M * hm[:, None]
            sum_ms += t_view * scat * _ms_lookup(h, (p @ s) / r)
    return sum_r * BETA_R[None, :] * phase_r[:, None] + sum_m * BETA_M * phase_m[:, None] + sum_ms


def sun_transmittance(sun_dir: np.ndarray) -> np.ndarray:
    """RGB transmittance of direct sunlight to the observer (reddening at low sun)."""
    s = np.asarray(sun_dir, dtype=float)
    s = s / np.linalg.norm(s)
    p = np.array([[0.0, 0.0, R_EARTH + OBSERVER_ALT_M]])
    od_r, od_m = _optical_depth_to_sun(p, s, 32)
    return np.exp(-(BETA_R * od_r[0] + BETA_M_EXT * od_m[0]))


def equirect_directions(width: int, height: int) -> np.ndarray:
    """(H, W, 3) WORLD direction that VTK samples at each texel of an
    equirectangular map passed as pv.Texture(array) (row 0 = zenith), with
    the renderer's environment basis set by configure_environment_basis().

    Calibrated by rendering (tests: skybox and mirror-sphere IBL agree):
    VTK's lookup mirrors x relative to the naive (cos phi, sin phi) layout.
    """
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = 2.0 * np.pi * u                   # azimuth
    theta = np.pi * v                       # 0 at zenith
    P, T = np.meshgrid(phi, theta)
    return np.stack([-np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)


def configure_environment_basis(renderer) -> None:
    """Environment lookups in the scene frame: z up, x (east) right. Applies
    to image-based lighting and to a sphere-projection vtkSkybox."""
    renderer.SetEnvironmentUp(0.0, 0.0, 1.0)
    renderer.SetEnvironmentRight(1.0, 0.0, 0.0)


def environment_texture(env: np.ndarray):
    """pv.Texture for an HDR equirect env (float32, linear radiance)."""
    import pyvista as pv
    tex = pv.Texture(np.ascontiguousarray(env, dtype=np.float32))
    tex.SetMipmap(True)
    tex.SetInterpolate(True)
    return tex


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> sRGB-encoded [0, 1] (IEC 61966-2-1)."""
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def make_skybox(texture):
    """Background sky from the same equirect env (sphere projection)."""
    import vtk
    sky = vtk.vtkSkybox()
    sky.SetTexture(texture)
    sky.SetProjectionToSphere()
    return sky


def sky_environment(sun_dir: np.ndarray, width: int = 512, height: int = 256) -> np.ndarray:
    """HDR equirectangular environment (H, W, 3) float32 for a sun direction.

    Upper hemisphere: single-scattered sky. Lower hemisphere: a Lambertian
    ground lit by the attenuated sun plus the sky's own irradiance, so IBL
    receives a plausible bounce from below instead of black.
    """
    dirs = equirect_directions(width, height)
    flat = dirs.reshape(-1, 3)
    s = np.asarray(sun_dir, dtype=float)
    s = s / np.linalg.norm(s)
    up = flat[:, 2] >= 0.0
    out = np.zeros_like(flat)
    # Rays exactly at/below the horizon hit the ground sphere immediately;
    # evaluate the horizon row slightly above so the sky meets the ground smoothly.
    sky_dirs = flat[up].copy()
    sky_dirs[:, 2] = np.maximum(sky_dirs[:, 2], 0.01)
    sky_dirs /= np.linalg.norm(sky_dirs, axis=1, keepdims=True)
    out[up] = sky_radiance(sky_dirs, s)

    # Sky irradiance on a horizontal plane (cosine-weighted, solid-angle-weighted).
    theta = np.arccos(np.clip(flat[:, 2], -1.0, 1.0))
    d_omega = (2.0 * np.pi / width) * (np.pi / height) * np.sin(theta)
    e_sky = (out[up] * (flat[up, 2] * d_omega[up])[:, None]).sum(axis=0)
    e_sun = sun_transmittance(s) * max(float(s[2]), 0.0)
    out[~up] = GROUND_ALBEDO * (e_sky + e_sun) / np.pi
    out += NIGHT_RADIANCE[None, :]
    return out.reshape(height, width, 3).astype(np.float32)


def sun_direction(lat_deg: float, lon_deg: float, hour_local: float, day_of_year: int = 172) -> np.ndarray:
    """Unit vector toward the sun in the local frame (x east, y north, z up)."""
    from solar_physics import sun_angles
    el, az = sun_angles(lat_deg, lon_deg, hour_local, day_of_year)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def cached_sky_environment(sun_dir: np.ndarray, cache_dir: Path | None = None,
                           width: int = 512, height: int = 256) -> np.ndarray:
    """sky_environment with an on-disk cache keyed by the (rounded) sun direction."""
    s = np.round(np.asarray(sun_dir, dtype=float) / np.linalg.norm(sun_dir), 3)
    key = hashlib.sha1(f"sky_v3_ms_aod|{s.tolist()}|{width}x{height}".encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"sky_{key}.npy" if cache_dir is not None else None
    if path is not None and path.exists():
        return np.load(path)
    env = sky_environment(s, width, height)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, env)
    return env


def overcast_environment(width: int = 512, height: int = 256, irradiance: float = 0.20,
                         tint=(0.90, 0.95, 1.0), ground_albedo: float = GROUND_ALBEDO) -> np.ndarray:
    """HDR equirect env of a heavy rain overcast: the CIE standard overcast sky,
    L(elevation) = Lz (1 + 2 sin e) / 3, scaled so the horizontal sky irradiance
    is `irradiance` (relative to a unit sun at normal incidence — about a fifth
    of a clear noon's total, typical of a rain storm). No sun, no aureole."""
    dirs = equirect_directions(width, height)
    sin_e = dirs[..., 2]
    lz = irradiance * 9.0 / (7.0 * np.pi)                    # E = pi * Lz * 7/9
    sky = lz * (1.0 + 2.0 * np.clip(sin_e, 0.0, 1.0)) / 3.0
    t = np.asarray(tint, dtype=np.float32)[None, None, :]
    env = np.where((sin_e >= 0.0)[..., None], sky[..., None] * t,
                   (ground_albedo * irradiance / np.pi) * t)
    return (env + NIGHT_RADIANCE[None, None, :] * 0.5).astype(np.float32)
