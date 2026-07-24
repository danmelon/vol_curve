import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.stats import norm
import yfinance as yf

# ---------------------------------------------------------
# 1. Black-Scholes Formula & Root Finder
# ---------------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    """Calculates European Call price using Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_manual_iv(market_price, S, K, T, r=0.05):
    """Finds IV (sigma) using Brent's method to solve BS_Price - Market_Price = 0"""
    intrinsic = max(0.0, S - K * np.exp(-r * T))
    if market_price <= intrinsic:
        return np.nan  # Arbitrage violation or bad quote

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma) - market_price

    try:
        # Search for implied vol between 0.1% and 500%
        return brentq(objective_function, a=0.001, b=5.0)
    except (ValueError, RuntimeError):
        return np.nan

# ---------------------------------------------------------
# 2. Fetch Option Data & Compute Manual IV across Expirations
# ---------------------------------------------------------
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)

# Get current underlying spot price
spot_price = ticker.history(period="1d")["Close"].iloc[-1]
expirations = ticker.options
today = datetime.datetime.now()

records = []
r = 0.05  # Assumed risk-free rate

print("Calculating manual IV across expirations...")

# Loop over the first 8 expiration dates to build the full 3D surface
for exp in expirations[:8]:
    try:
        opt_chain = ticker.option_chain(exp)
        calls = opt_chain.calls.copy()

        exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d")
        dte = (exp_date - today).days

        if dte < 5:  # Skip expiring options (unstable IV)
            continue

        T = dte / 365.0  # Time to expiration in years

        # Calculate Mid-Price
        calls["mid_price"] = (calls["bid"] + calls["ask"]) / 2.0

        # Compute Manual Black-Scholes IV for every option in this expiration
        calls["manual_IV"] = calls.apply(
            lambda row: calculate_manual_iv(row["mid_price"], spot_price, row["strike"], T, r),
            axis=1
        )

        calls["DTE"] = dte
        records.append(calls)
    except Exception as e:
        print(f"Skipping {exp}: {e}")

# Combine into a single DataFrame
df = pd.concat(records, ignore_index=True)

# ---------------------------------------------------------
# 3. Clean Data & Construct Interpolation Grid
# ---------------------------------------------------------
# Filter for realistic liquid contracts near the spot price
clean_df = df[
    (df["manual_IV"] > 0.05) & 
    (df["manual_IV"] < 1.5) & 
    (df["volume"] > 5) &
    (df["strike"] >= spot_price * 0.7) &
    (df["strike"] <= spot_price * 1.3)
].dropna(subset=["manual_IV"])

X = clean_df["DTE"].values
Y = clean_df["strike"].values
Z = clean_df["manual_IV"].values

# Create a regular mesh grid
xi = np.linspace(X.min(), X.max(), 50)
yi = np.linspace(Y.min(), Y.max(), 50)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Interpolate manual IVs across the grid
Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method="cubic")

# ---------------------------------------------------------
# 4. Plot 3D Manual Volatility Surface
# ---------------------------------------------------------

plt.style.use('dark_background')

# 1. Grid interpolation for both IV columns
Z_manual = griddata((X, Y), clean_df["manual_IV"].values, (X_grid, Y_grid), method="cubic")
Z_yf = griddata((X, Y), clean_df["impliedVolatility"].values, (X_grid, Y_grid), method="cubic")

# 2. Setup side-by-side 3D figure
fig = plt.figure(figsize=(16, 7))

# --- Plot 1: Manual BS IV Surface ---
ax1 = fig.add_subplot(121, projection="3d")
surf1 = ax1.plot_surface(X_grid, Y_grid, Z_manual, cmap=cm.plasma, alpha=0.85, linewidth=0.2, edgecolors="k")
ax1.set_title("Manual Black-Scholes IV Surface", fontsize=12)
ax1.set_xlabel("DTE")
ax1.set_ylabel("Strike ($)")
ax1.set_zlabel("IV")
ax1.view_init(elev=25, azim=-125)

# --- Plot 2: yfinance IV Surface ---
ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(X_grid, Y_grid, Z_yf, cmap=cm.viridis, alpha=0.85, linewidth=0.2, edgecolors="k")
ax2.set_title("yfinance Backend IV Surface", fontsize=12)
ax2.set_xlabel("DTE")
ax2.set_ylabel("Strike ($)")
ax2.set_zlabel("IV")
ax2.view_init(elev=25, azim=-125)

plt.tight_layout()
plt.show()