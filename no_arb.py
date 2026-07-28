"""
Module 5: No-static-arbitrage conditions for SVI slices.

Butterfly arbitrage (within one expiry):
    - Necessary condition (Roger Lee moment formula): b*(1+|rho|) <= 4
    - Sufficient/rigorous condition (Gatheral-Jaeckel Durrleman function):
      g(k) >= 0 for all k, where g is built from w, w', w''.

Calendar arbitrage (across expiries):
    - Total variance w(k, T) must be non-decreasing in T for fixed k.
"""
from __future__ import annotations
import numpy as np


def lee_bound_check(b: float, rho: float) -> bool:
    """Cheap necessary condition for no butterfly arbitrage in the wings."""
    return b * (1 + abs(rho)) <= 4.0


def _svi_derivatives(k, a, b, rho, m, sigma):
    """w, w', w'' for raw SVI, evaluated analytically."""
    diff = k - m
    root = np.sqrt(diff ** 2 + sigma ** 2)
    w = a + b * (rho * diff + root)
    w_prime = b * (rho + diff / root)
    w_dprime = b * sigma ** 2 / root ** 3
    return w, w_prime, w_dprime


def durrleman_function(k, a, b, rho, m, sigma):
    """
    g(k) from Gatheral & Jaeckel (2014). g(k) >= 0 everywhere <=> no
    butterfly arbitrage for this slice (the density is non-negative).
    """
    w, wp, wpp = _svi_derivatives(k, a, b, rho, m, sigma)
    term1 = (1 - (k * wp) / (2 * w)) ** 2
    term2 = (wp ** 2) / 4 * (1 / w + 0.25)
    term3 = wpp / 2
    return term1 - term2 + term3


def butterfly_arbitrage_check(params: dict, k_grid: np.ndarray | None = None) -> dict:
    """Full check: cheap Lee bound + Durrleman density check on a k grid."""
    a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    if k_grid is None:
        k_grid = np.linspace(m - 3, m + 3, 400)

    lee_ok = lee_bound_check(b, rho)
    g_vals = durrleman_function(k_grid, a, b, rho, m, sigma)
    min_g = float(np.min(g_vals))
    density_ok = min_g >= -1e-8  # small numerical tolerance

    return {"lee_bound_ok": lee_ok, "min_durrleman_g": min_g, "no_butterfly_arb": lee_ok and density_ok}


def calendar_arbitrage_check(slices: list[dict], k_grid: np.ndarray | None = None) -> dict:
    """
    slices: list of {"T": float, "a":..,"b":..,"rho":..,"m":..,"sigma":..}
    sorted or unsorted; checked pairwise after sorting by T.
    Returns whether w(k,T) is non-decreasing in T across all k in k_grid.
    """
    from svi import svi_total_variance  # local import to avoid circularity in some layouts

    slices_sorted = sorted(slices, key=lambda s: s["T"])
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 300)

    violations = []
    for i in range(len(slices_sorted) - 1):
        s1, s2 = slices_sorted[i], slices_sorted[i + 1]
        w1 = svi_total_variance(k_grid, s1["a"], s1["b"], s1["rho"], s1["m"], s1["sigma"])
        w2 = svi_total_variance(k_grid, s2["a"], s2["b"], s2["rho"], s2["m"], s2["sigma"])
        bad = w2 < w1 - 1e-8
        if np.any(bad):
            violations.append((s1["T"], s2["T"], float(np.min(w2 - w1))))

    return {"no_calendar_arb": len(violations) == 0, "violations": violations}


def calendar_penalty(slices: list[dict], k_grid: np.ndarray | None = None) -> float:
    """
    Soft penalty to add to a joint calibration objective: sums squared
    violations of w2 >= w1 across adjacent expiries. Use this inside your
    calibration loop when fitting expiries jointly rather than independently.
    """
    from svi import svi_total_variance

    slices_sorted = sorted(slices, key=lambda s: s["T"])
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 300)

    penalty = 0.0
    for i in range(len(slices_sorted) - 1):
        s1, s2 = slices_sorted[i], slices_sorted[i + 1]
        w1 = svi_total_variance(k_grid, s1["a"], s1["b"], s1["rho"], s1["m"], s1["sigma"])
        w2 = svi_total_variance(k_grid, s2["a"], s2["b"], s2["rho"], s2["m"], s2["sigma"])
        violation = np.maximum(w1 - w2, 0.0)
        penalty += float(np.sum(violation ** 2))
    return penalty