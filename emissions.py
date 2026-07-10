"""emissions.py — speed-dependent vehicle emission factors.

Replaces the "cars × weight" density proxy with physically grounded emission
rates so the air-quality overlay shows real pollutant mass, not just where
cars are.

Model
-----
Follows the EEA/COPERT *average-speed* methodology: emission factor per unit
distance is a U-shaped function of mean speed — high in stop-and-go congestion,
minimum around 60–70 km/h, rising again at high speed.

  • CO2 is derived from modelled fuel consumption (L/100 km), which is the
    physically solid quantity:  CO2 [g/km] = FC[L/100km]/100 × ρ_fuel.
  • NOx and PM2.5 use representative Euro-4/5 urban factors scaled by the same
    speed shape (and a cold-start-free "hot" assumption).

These are *representative* factors (petrol passenger car baseline, with class
multipliers for bus/truck/etc.), not a certified COPERT coefficient set — the
goal is a correct speed/class *relationship*, suitable for relative AQ mapping
and scenario comparison.  Swap in jurisdiction-specific COPERT tables here if
you need regulatory-grade absolute numbers.

Public API
----------
emission_rate_g_per_s(pollutant, v_ms, vclass="passenger") -> float | np.ndarray
    Per-vehicle emission rate.  Accepts scalar or numpy array of speeds (m/s);
    returns the same shape.  This is what the heatmap accumulates per cell.

POLLUTANTS = ("co2", "nox", "pm")
"""
from __future__ import annotations

import numpy as np

POLLUTANTS = ("co2", "nox", "pm")

# Fuel density → CO2: petrol ≈ 2.31 kg CO2 / L, diesel ≈ 2.64 kg/L.
_CO2_PER_L = {"petrol": 2310.0, "diesel": 2640.0}  # g CO2 per litre burned

# Optimal speed (km/h) where per-km emissions are minimised.
_V_OPT = 65.0

# Per-class scaling of the passenger-car baseline (engine size / mass / fuel).
# (fuel_mult, nox_mult, pm_mult, fuel_type)
_CLASS = {
    "passenger":  (1.00, 1.0,  1.0,  "petrol"),
    "private":    (1.00, 1.0,  1.0,  "petrol"),
    "taxi":       (1.10, 1.3,  1.2,  "diesel"),
    "bus":        (3.20, 9.0,  6.0,  "diesel"),
    "coach":      (3.20, 9.0,  6.0,  "diesel"),
    "truck":      (3.60, 11.0, 7.0,  "diesel"),
    "trailer":    (4.20, 13.0, 8.0,  "diesel"),
    "delivery":   (1.60, 3.0,  2.5,  "diesel"),
    "emergency":  (2.00, 4.0,  3.0,  "diesel"),
    "motorcycle": (0.45, 0.6,  1.4,  "petrol"),
    "moped":      (0.30, 0.4,  1.6,  "petrol"),
    "bicycle":    (0.00, 0.0,  0.0,  "petrol"),
}
_DEFAULT_CLASS = (1.00, 1.0, 1.0, "petrol")

# Baseline passenger-car factors at the optimal speed (hot running):
_FC_MIN_L_PER_100KM = 5.5    # L/100km at ~65 km/h, free-flow
_NOX_MIN_G_PER_KM   = 0.06   # g/km  (Euro-5 petrol, hot)
_PM_MIN_G_PER_KM    = 0.003  # g/km


def _speed_shape(v_kmh: np.ndarray) -> np.ndarray:
    """U-shaped per-km multiplier vs speed, normalised to 1.0 at V_OPT.

    Congestion (low v) and high-speed cruising both raise per-km emissions:
        f(v) = 1 + A·(V_OPT/v − 1)        for the low-speed congestion branch
                 + B·((v − V_OPT)/V_OPT)²  for the high-speed drag branch
    """
    v = np.maximum(np.asarray(v_kmh, dtype=float), 3.0)   # floor avoids blow-up
    low  = 0.95 * (_V_OPT / v - 1.0)          # large when v << V_OPT
    low  = np.maximum(low, 0.0)
    high = 0.55 * ((v - _V_OPT) / _V_OPT) ** 2  # grows above V_OPT
    return 1.0 + low + high


def _fuel_l_per_100km(v_kmh, fuel_mult: float) -> np.ndarray:
    return _FC_MIN_L_PER_100KM * fuel_mult * _speed_shape(v_kmh)


def emission_factor_g_per_km(pollutant: str, v_ms, vclass: str = "passenger"):
    """Emission factor in g per km for a vehicle of `vclass` at speed `v_ms`."""
    fuel_mult, nox_mult, pm_mult, fuel_type = _CLASS.get(vclass, _DEFAULT_CLASS)
    v_kmh = np.asarray(v_ms, dtype=float) * 3.6
    shape = _speed_shape(v_kmh)

    if pollutant == "co2":
        fc = _FC_MIN_L_PER_100KM * fuel_mult * shape          # L/100km
        return (fc / 100.0) * _CO2_PER_L[fuel_type]           # g/km
    if pollutant == "nox":
        return _NOX_MIN_G_PER_KM * nox_mult * shape
    if pollutant == "pm":
        return _PM_MIN_G_PER_KM * pm_mult * shape
    raise ValueError(f"unknown pollutant {pollutant!r} (use one of {POLLUTANTS})")


def emission_rate_g_per_s(pollutant: str, v_ms, vclass: str = "passenger"):
    """Per-vehicle emission rate in g/s.

    rate[g/s] = EF[g/km] × v[km/h] / 3600
    A stationary (idling) vehicle still emits, so a small idle floor is added.
    Accepts scalar or numpy array; returns the same shape.
    """
    v_ms_arr = np.asarray(v_ms, dtype=float)
    ef = emission_factor_g_per_km(pollutant, v_ms_arr, vclass)   # g/km
    v_kmh = np.maximum(v_ms_arr * 3.6, 0.0)
    moving = ef * v_kmh / 3600.0                                 # g/s while moving

    # Idle emission floor (g/s) so jammed traffic still shows up.
    fuel_mult, nox_mult, pm_mult, fuel_type = _CLASS.get(vclass, _DEFAULT_CLASS)
    idle = {
        "co2": 0.60 * fuel_mult,
        "nox": 0.0009 * nox_mult,
        "pm":  0.00004 * pm_mult,
    }.get(pollutant, 0.0)

    rate = np.where(v_ms_arr < 0.3, idle, moving + idle * 0.2)
    return rate if rate.shape else float(rate)
