import numpy as np
from scipy.optimize import brentq
from pricing import OptionParams, black_scholes_price, greeks
 
 
class IVSolverError(Exception):         ## Error class
    pass
 
 
def implied_vol_newton(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    q: float = 0.0,
    sigma_init: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100,
    vega_floor: float = 1e-8,
) -> float:
    """
    Newton-Raphson IV solver. Raises IVSolverError if it fails to
    converge or vega collapses (caller should fall back to Brent).
    """
    sigma = sigma_init
 
    if abs(market_price) < 1e-10:       ## reject prices near 0
        raise IVSolverError("market_price too close to zero to invert reliably")
 
    for _ in range(max_iter):
        if sigma <= 0:
            raise IVSolverError("sigma went non-positive during iteration")
 
        p = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        price = black_scholes_price(p, option_type)         ## Calc theoretical price using BS
        vega_per_point = greeks(p, option_type)["vega"]     ## Calc Vega of this option
        vega = vega_per_point * 100  # undo the /100 scaling used for quoting
 
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
 
        if abs(vega) < vega_floor:
            raise IVSolverError(f"vega collapsed to {vega:.2e}; Newton unreliable here")
 
        sigma = sigma - diff / vega
 
    raise IVSolverError(f"Newton did not converge in {max_iter} iterations")
 
 
def implied_vol_brent(          ## Brent method as fallback
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    q: float = 0.0,
    sigma_bounds: tuple = (1e-6, 5.0),
) -> float:
    """
    Brent's method IV solver. Robust bracket-based fallback.
    Requires the market price to be within the no-arbitrage bounds
    implied by sigma_bounds, else raises ValueError from brentq
    (sign of g must differ at the two bounds).
    """
    def g(sigma):
        p = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        return black_scholes_price(p, option_type) - market_price
 
    lo, hi = sigma_bounds
    return brentq(g, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200)
 
 
def implied_vol(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Main entry point: try Newton first, fall back to Brent on failure.
    This mirrors what production pricing libraries actually do.
    """
    try:
        return implied_vol_newton(market_price, S, K, T, r, option_type, q)
    except IVSolverError:
        return implied_vol_brent(market_price, S, K, T, r, option_type, q)
 
