import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pricing import OptionParams, black_scholes_price, greeks


# ==========================================
# 1. SOLVER DEFINITIONS (From your original code)
# ==========================================

class IVSolverError(Exception):
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
    sigma = sigma_init

    if abs(market_price) < 1e-10:
        raise IVSolverError("market_price too close to zero to invert reliably")

    for _ in range(max_iter):
        if sigma <= 0:
            raise IVSolverError("sigma went non-positive during iteration")

        p = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        price = black_scholes_price(p, option_type)
        vega_per_point = greeks(p, option_type)["vega"]
        vega = vega_per_point * 100  # undo the /100 scaling used for quoting

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        if abs(vega) < vega_floor:
            raise IVSolverError(f"vega collapsed to {vega:.2e}; Newton unreliable here")

        sigma = sigma - diff / vega

    raise IVSolverError(f"Newton did not converge in {max_iter} iterations")


def implied_vol_brent(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    q: float = 0.0,
    sigma_bounds: tuple = (1e-6, 10.0),
) -> float:
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
    try:
        return implied_vol_newton(market_price, S, K, T, r, option_type, q)
    except IVSolverError:
        return implied_vol_brent(market_price, S, K, T, r, option_type, q)


# ==========================================
# 2. LOOP & PLOTTING SCRIPT
# ==========================================

# --- Inputs ---
S = 100.0        # Current Spot Price
T = 0.5          # Time to maturity (6 months)
r = 0.05         # 5% risk-free rate
q = 0.01         # 1% dividend yield

# --- Sample Market Data ---
strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
market_prices = np.array([21.50, 17.20, 13.10, 9.40, 6.20, 3.80, 2.10, 1.05, 0.48])

# --- Calculate Volatilities ---
implied_vols = []

for K, price in zip(strikes, market_prices):
    try:
        iv = implied_vol(
            market_price=price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type="call",
            q=q
        )
        implied_vols.append(iv)
    except Exception as e:
        print(f"Skipping Strike {K}: {e}")
        implied_vols.append(np.nan)

implied_vols = np.array(implied_vols)

# --- Plotting ---
plt.figure(figsize=(9, 5))
plt.plot(strikes, implied_vols * 100, marker='o', color='#1f77b4', linewidth=2, label="Implied Volatility")
plt.axvline(x=S, color='gray', linestyle='--', alpha=0.7, label=f"Spot Price (S={S})")

plt.title("Option Volatility Smile", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Strike Price ($K$)", fontsize=11)
plt.ylabel("Implied Volatility (%)", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()