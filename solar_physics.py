from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_G = 9.81


@dataclass(frozen=True)
class SolarParams:
    """Bundled defaults for solar harvesting and vehicle energy model."""

    roof_area_m2: float = 1.6
    panel_efficiency: float = 0.22
    temperature_derating: float = 0.88
    vehicle_mass_kg: float = 1400.0
    rolling_coeff: float = 0.012
    drag_coeff: float = 0.30
    frontal_area_m2: float = 2.2


def sun_angles(
    lat_deg: float,
    lon_deg: float,
    hour_local: float,
    day_of_year: int = 172,
) -> tuple[float, float]:
    """Return true solar (elevation, azimuth) in radians.

    Uses Spencer declination/equation-of-time with hour-angle method.
    Azimuth is returned in radians clockwise from North in [0, 2*pi).
    """
    lat = np.deg2rad(float(lat_deg))
    lon = float(lon_deg)
    hour = float(hour_local)
    n = int(np.clip(day_of_year, 1, 366))

    gamma = 2.0 * np.pi * (n - 1) / 365.0

    # Spencer declination (radians)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    # Spencer equation of time (minutes)
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    # Approximate local standard meridian from longitude.
    lstm_deg = 15.0 * round(lon / 15.0)

    # True solar time (minutes) and hour angle (radians)
    tst_min = hour * 60.0 + eot_min + 4.0 * (lon - lstm_deg)
    hra = np.deg2rad((tst_min / 4.0) - 180.0)

    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hra)
    sin_el = float(np.clip(sin_el, -1.0, 1.0))
    elevation = float(np.arcsin(sin_el))

    # Azimuth from North, clockwise.
    x = -np.cos(decl) * np.sin(hra)
    y = np.sin(decl) * np.cos(lat) - np.cos(decl) * np.sin(lat) * np.cos(hra)
    azimuth = float(np.mod(np.arctan2(x, y) + 2.0 * np.pi, 2.0 * np.pi))

    return elevation, azimuth


def clear_sky_ghi(
    elevation_rad: float,
    extraterrestrial_irradiance: float = 1361.0,
) -> tuple[float, float, float]:
    """Return (GHI, DNI, DHI) in W/m^2 using Meinel + clear-sky Erbs fraction."""
    el = float(elevation_rad)
    if el <= 0.0:
        return 0.0, 0.0, 0.0

    sin_el = float(np.clip(np.sin(el), 1e-6, 1.0))
    air_mass = max(1.0, 1.0 / sin_el)

    dni = float(extraterrestrial_irradiance) * float(0.7 ** (air_mass ** 0.678))

    # Erbs clear-sky diffuse fraction for horizontal irradiance.
    # DHI = 0.0721 * GHI and GHI = DNI*sin(el) + DHI.
    ghi = (dni * sin_el) / (1.0 - 0.0721)
    dhi = 0.0721 * ghi

    return float(ghi), float(dni), float(dhi)


def panel_irradiance(
    ghi: float,
    dhi: float,
    dni: float,
    sun_elevation_rad: float,
) -> float:
    """Irradiance on a flat horizontal panel (tilt=0)."""
    _ = ghi  # retained for API completeness with caller-side decomposition
    sin_el = float(np.sin(float(sun_elevation_rad)))
    if sin_el <= 0.0:
        return 0.0
    g_panel = float(dni) * sin_el + float(dhi)
    return float(max(0.0, g_panel))


def solar_energy_joules(
    panel_irradiance_Wm2: float,
    shadow_fraction: float,
    travel_time_s: float,
    roof_area_m2: float = 1.6,
    panel_efficiency: float = 0.22,
    temperature_derating: float = 0.88,
) -> float:
    """Harvested solar energy over one traversal interval (J)."""
    shade = float(np.clip(shadow_fraction, 0.0, 1.0))
    t = max(0.0, float(travel_time_s))
    g = max(0.0, float(panel_irradiance_Wm2))

    return float(
        g
        * (1.0 - shade)
        * float(roof_area_m2)
        * float(panel_efficiency)
        * float(temperature_derating)
        * t
    )


def mechanical_energy_joules(
    length_m: float,
    speed_ms: float,
    vehicle_mass_kg: float = 1400.0,
    rolling_coeff: float = 0.012,
    drag_coeff: float = 0.30,
    frontal_area_m2: float = 2.2,
    rho_air: float = 1.20,
) -> float:
    """Mechanical traction energy for constant-speed travel over edge length (J)."""
    d = max(0.0, float(length_m))
    v = max(0.0, float(speed_ms))

    f_roll = float(rolling_coeff) * float(vehicle_mass_kg) * _G
    f_drag = 0.5 * float(rho_air) * float(drag_coeff) * float(frontal_area_m2) * (v ** 2)

    return float((f_roll + f_drag) * d)


def net_energy_joules(
    length_m: float,
    speed_ms: float,
    panel_irradiance_Wm2: float,
    shadow_fraction: float,
    travel_time_s: float,
    **kwargs,
) -> float:
    """Net edge energy = mechanical consumption - solar harvest (J)."""
    mech = mechanical_energy_joules(
        length_m=length_m,
        speed_ms=speed_ms,
        vehicle_mass_kg=kwargs.get("vehicle_mass_kg", 1400.0),
        rolling_coeff=kwargs.get("rolling_coeff", 0.012),
        drag_coeff=kwargs.get("drag_coeff", 0.30),
        frontal_area_m2=kwargs.get("frontal_area_m2", 2.2),
        rho_air=kwargs.get("rho_air", 1.20),
    )
    solar = solar_energy_joules(
        panel_irradiance_Wm2=panel_irradiance_Wm2,
        shadow_fraction=shadow_fraction,
        travel_time_s=travel_time_s,
        roof_area_m2=kwargs.get("roof_area_m2", 1.6),
        panel_efficiency=kwargs.get("panel_efficiency", 0.22),
        temperature_derating=kwargs.get("temperature_derating", 0.88),
    )
    return float(mech - solar)


__all__ = [
    "SolarParams",
    "sun_angles",
    "clear_sky_ghi",
    "panel_irradiance",
    "solar_energy_joules",
    "mechanical_energy_joules",
    "net_energy_joules",
]
