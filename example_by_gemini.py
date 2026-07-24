import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import yfinance as yf

# ---------------------------------------------------------
# 1. Fetch Option Data via yfinance
# ---------------------------------------------------------
ticker_symbol = "AAPL"  # Highly liquid options yield cleaner surfaces
ticker = yf.Ticker(ticker_symbol)

# Get all available expiration dates
expirations = ticker.options
print(f"Found {len(expirations)} expiration dates for {ticker_symbol}.")

records = []
today = datetime.datetime.now()

# Gather call options across the first 10 expiration dates
for exp in expirations[:10]:
    try:
        opt_chain = ticker.option_chain(exp)
        calls = opt_chain.calls.copy()

        # Days to Expiration (DTE)
        exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d")
        dte = (exp_date - today).days

        # Skip expired or ultra-short-dated options (noisy IV)
        if dte < 5:
            continue

        calls["DTE"] = dte
        records.append(calls)
    except Exception as e:
        print(f"Skipping expiry {exp}: {e}")

df = pd.concat(records, ignore_index=True)

# ---------------------------------------------------------
# 2. Filter Out Noise & Outliers
# ---------------------------------------------------------
# Keep reasonably liquid, non-zero implied volatility contracts
clean_df = df[
    (df["impliedVolatility"] > 0.05)
    & (df["impliedVolatility"] < 1.5)
    & (df["volume"] > 5)  # Filter for liquid strikes
].copy()

# ---------------------------------------------------------
# 3. Grid & Interpolate for 3D Surface Rendering
# ---------------------------------------------------------
X = clean_df["DTE"].values
Y = clean_df["strike"].values
Z = clean_df["impliedVolatility"].values

# Create a uniform grid over Days to Expiration and Strike Prices
xi = np.linspace(X.min(), X.max(), 50)
yi = np.linspace(Y.min(), Y.max(), 50)
X_grid, Y_grid = np.meshgrid(xi, yi)

# Interpolate missing mesh points using cubic spline interpolation
Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method="cubic")

# ---------------------------------------------------------
# 4. Plot 3D Volatility Surface
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot surface
surf = ax.plot_surface(
    X_grid,
    Y_grid,
    Z_grid,
    cmap=cm.viridis,
    linewidth=0.2,
    edgecolors="k",
    alpha=0.85,
)

# Customize camera angle and axes
ax.view_init(elev=25, azim=-125)
ax.set_title(f"3D Volatility Surface / Smile ({ticker_symbol})", fontsize=14, pad=20)
ax.set_xlabel("Days to Expiration (DTE)", fontsize=10, labelpad=10)
ax.set_ylabel("Strike Price ($)", fontsize=10, labelpad=10)
ax.set_zlabel("Implied Volatility (IV)", fontsize=10, labelpad=10)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Implied Volatility", fontsize=10)

plt.tight_layout()
plt.show()