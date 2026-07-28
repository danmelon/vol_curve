"""
Module 6: Stitch calibrated SVI slices (one per expiry) into a full
VolSurface exposing iv(strike, expiry).

Design choice: interpolate TOTAL VARIANCE in T at fixed k (not the raw
SVI params directly) — this is what keeps the calendar no-arbitrage
property intact under interpolation, and is standard market practice.
"""
from __future__ import annotations
import numpy as np
from svi import svi_total_variance


class VolSurface:
    def __init__(self, forward_curve):
        """
        forward_curve: callable T -> forward price F(T), so we can convert
        strike K to log-moneyness k = ln(K/F(T)) for any expiry.
        """
        self.forward_curve = forward_curve
        self.slices = []  # list of dicts: {"T", "a","b","rho","m","sigma"}

    def add_slice(self, T: float, params: dict):
        """params must contain a, b, rho, m, sigma for this expiry."""
        entry = {"T": T, **params}
        self.slices.append(entry)
        self.slices.sort(key=lambda s: s["T"])

    def _bracket(self, T: float):
        """Find the two calibrated expiries bracketing T (or the nearest one)."""
        Ts = [s["T"] for s in self.slices]
        if T <= Ts[0]:
            return self.slices[0], self.slices[0]
        if T >= Ts[-1]:
            return self.slices[-1], self.slices[-1]
        for i in range(len(Ts) - 1):
            if Ts[i] <= T <= Ts[i + 1]:
                return self.slices[i], self.slices[i + 1]
        raise RuntimeError("bracket not found")  # should not happen

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        """w(k,T) by linear interpolation in T of total variance at fixed k."""
        s1, s2 = self._bracket(T)
        w1 = svi_total_variance(k, s1["a"], s1["b"], s1["rho"], s1["m"], s1["sigma"])
        if s1["T"] == s2["T"]:
            return w1  # flat extrapolation at the edges
        weight = (T - s1["T"]) / (s2["T"] - s1["T"])
        w2 = svi_total_variance(k, s2["a"], s2["b"], s2["rho"], s2["m"], s2["sigma"])
        return (1 - weight) * w1 + weight * w2

    def iv(self, strike: float, expiry: float) -> float:
        """Black-Scholes implied vol at an arbitrary (strike, expiry)."""
        if not self.slices:
            raise RuntimeError("no calibrated slices in the surface yet")
        F = self.forward_curve(expiry)
        k = np.log(strike / F)
        w = self.total_variance(np.array([k]), expiry)[0]
        return float(np.sqrt(max(w, 0.0) / expiry))