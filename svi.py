"""
Module 4: Raw SVI smile parametrization and calibration.

w(k) = a + b*(rho*(k-m) + sqrt((k-m)**2 + sigma**2))

where w is TOTAL variance (sigma_BS^2 * T), k is log-moneyness ln(K/F).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m: float, sigma: float) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_iv(k: np.ndarray, T: float, a: float, b: float, rho: float,
           m: float, sigma: float) -> np.ndarray:
    """Convert SVI total variance back to Black-Scholes implied vol."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0.0) / T)


def _residuals(params, k, w_mkt, weights):
    a, b, rho, m, sigma = params
    w_model = svi_total_variance(k, a, b, rho, m, sigma)
    return weights * (w_model - w_mkt)


def calibrate_svi(k: np.ndarray, w_mkt: np.ndarray, weights: np.ndarray | None = None,
                   x0: tuple | None = None) -> dict:
    """
    Least-squares calibration of raw SVI to a single expiry's market total variances.

    k       : log-moneyness array
    w_mkt   : observed total variance array (iv_mkt^2 * T)
    weights : optional per-point weights (e.g. 1/bid-ask spread, or vega-based)
    x0      : optional initial guess (a, b, rho, m, sigma)
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    if weights is None:
        weights = np.ones_like(k)

    if x0 is None:
        # reasonable generic starting point
        a0 = max(w_mkt.min(), 1e-4)
        b0 = 0.1
        rho0 = -0.3
        m0 = float(k[np.argmin(w_mkt)])
        sigma0 = 0.1
        x0 = (a0, b0, rho0, m0, sigma0)

    # bounds: a>=0, b>=0, -1<rho<1, sigma>0 ; m free
    lower = [0.0, 0.0, -0.999, -2.0, 1e-4]
    upper = [5.0, 5.0, 0.999, 2.0, 5.0]

    result = least_squares(
        _residuals, x0=x0, args=(k, w_mkt, weights),
        bounds=(lower, upper), method="trf",
    )

    a, b, rho, m, sigma = result.x
    return {
        "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
        "success": result.success, "cost": result.cost,
    }