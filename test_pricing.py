import numpy as np
from pricing import OptionParams, black_scholes_price

def test_put_call_parity():
    p = OptionParams(S=100, K=105, T=0.5, r=0.03, sigma=0.25)
    call = black_scholes_price(p, "call")
    put = black_scholes_price(p, "put")
    lhs = call - put
    rhs = p.S * np.exp(-p.q * p.T) - p.K * np.exp(-p.r * p.T)
    assert np.isclose(lhs, rhs, atol=1e-8)

def test_deep_itm_call_converges_to_intrinsic():
    p = OptionParams(S=200, K=100, T=0.01, r=0.03, sigma=0.2)
    price = black_scholes_price(p, "call")
    assert np.isclose(price, 100, atol=1.0)