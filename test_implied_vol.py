import numpy as np
from pricing import OptionParams, black_scholes_price
from implied_vol import implied_vol, implied_vol_newton, implied_vol_brent, IVSolverError


def test_recovers_true_vol_atm():
    true_sigma = 0.27
    p = OptionParams(S=100, K=100, T=0.5, r=0.03, sigma=true_sigma)
    price = black_scholes_price(p, "call")
    recovered = implied_vol(price, S=100, K=100, T=0.5, r=0.03, option_type="call")
    assert np.isclose(recovered, true_sigma, atol=1e-6)


def test_recovers_true_vol_otm_put():
    true_sigma = 0.35
    p = OptionParams(S=100, K=80, T=0.25, r=0.02, sigma=true_sigma)
    price = black_scholes_price(p, "put")
    recovered = implied_vol(price, S=100, K=80, T=0.25, r=0.02, option_type="put")
    assert np.isclose(recovered, true_sigma, atol=1e-6)


def test_deep_otm_near_expiry_rejects_underflowed_price():
    # This price underflows to ~0 in float64 - Newton must refuse
    # to "converge" spuriously at the initial guess.
    p = OptionParams(S=100, K=180, T=0.02, r=0.04, sigma=0.27)
    tiny_price = black_scholes_price(p, "call")
    try:
        implied_vol_newton(tiny_price, S=100, K=180, T=0.02, r=0.04, option_type="call")
        assert False, "Newton should have raised IVSolverError"
    except IVSolverError:
        pass

    # Brent fallback should still recover it via sign-based bracketing
    recovered = implied_vol_brent(tiny_price, S=100, K=180, T=0.02, r=0.04, option_type="call")
    assert np.isclose(recovered, 0.27, atol=1e-3)