import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

@dataclass
class OptionParams:
    S: float    # spot
    K: float    # strike
    T: float    # Time to expiry (years)
    r: float    # risk-free rate
    sigma: float    # volatility
    q: float= 0.0   # dividend yield

def _d1_d2(p: OptionParams):
    d1 = (np.log(p.S/p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    return d1, d2

def black_scholes_price(p: OptionParams, option_type: str = "call") -> float:
    d1, d2 = _d1_d2(p)
    disc_r = np.exp(-p.r * p.T)
    disc_q = np.exp(-p.q * p.T)

    if option_type == "call":
        return p.S * norm.cdf(d1) - p.K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return p.K * disc_r * norm.cdf(-d2) - p.S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
def greeks(p: OptionParams, option_type: str = "call") -> dict:
    d1, d2 = _d1_d2(p)
    disc_r = np.exp(-p.r * p.T)
    disc_q = np.exp(-p.q * p.T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (p.S * p.sigma * np.sqrt(p.T))
    vega = p.S * disc_q * pdf_d1 * np.sqrt(p.T) / 100  # per 1 vol point

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-p.S * disc_q * pdf_d1 * p.sigma / (2 * np.sqrt(p.T))
                  - p.r * p.K * disc_r * norm.cdf(d2)
                  + p.q * p.S * disc_q * norm.cdf(d1)) / 365
        rho = p.K * p.T * disc_r * norm.cdf(d2) / 100
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (-p.S * disc_q * pdf_d1 * p.sigma / (2 * np.sqrt(p.T))
                  + p.r * p.K * disc_r * norm.cdf(-d2)
                  - p.q * p.S * disc_q * norm.cdf(-d1)) / 365
        rho = -p.K * p.T * disc_r * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
