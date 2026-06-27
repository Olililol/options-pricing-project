"""
Implied volatility solvers.

"Implied vol" is the σ that makes the Black-Scholes formula reproduce a given
market price. It is the market's consensus forecast of future realised vol,
compressed into a single number per (strike, maturity) pair.

Two methods:

1. Bisection  — bracket the root in [σ_lo, σ_hi] and halve the interval.
   Guaranteed to converge if the market price is in the no-arbitrage range.
   Convergence: linear, ~log₂((σ_hi - σ_lo)/tol) ≈ 50 iterations to 1e-8.

2. Newton-Raphson  — use vega as the derivative: σ_{n+1} = σ_n - (BS(σ) - price) / vega
   Convergence: quadratic (doubles correct digits each step) → ~10 iterations.
   Breaks when vega ≈ 0: deep ITM/OTM options near expiry have tiny vega and
   Newton can overshoot wildly. Guard with a vega floor and fallback to bisection.

When does Newton break?
  - Very short-dated deep OTM options: vega → 0, step size → ∞
  - Prices very close to intrinsic value: the BS function flattens against the
    lower bound and the root becomes ill-conditioned
  - σ → 0: the function value vanishes and the step can overshoot to negative σ

No-arbitrage bounds for a European option:
  Call: max(F·e^{-rT} - K·e^{-rT}, 0) ≤ C ≤ S·e^{-qT}
  Put:  max(K·e^{-rT} - F·e^{-rT}, 0) ≤ P ≤ K·e^{-rT}
  Market prices outside these bounds imply data issues, not real arbitrage.
"""
from __future__ import annotations

import math
from black_scholes import bs_price, bs_greeks

_IV_LO = 1e-6
_IV_HI = 10.0    # 1000% vol upper bracket — more than enough for any real market


def _no_arb_bounds(
    S: float, K: float, r: float, T: float, q: float, option_type: str
) -> tuple[float, float]:
    """Lower and upper no-arbitrage bounds for an option price."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float | None:
    """
    Find implied volatility by bisection.

    Returns None if market_price violates the no-arbitrage bounds or if
    the bracket cannot be established.
    """
    lower_arb, upper_arb = _no_arb_bounds(S, K, r, T, q, option_type)
    if market_price < lower_arb - 1e-6 or market_price >= upper_arb:
        return None

    lo, hi = _IV_LO, _IV_HI
    f_lo = bs_price(S, K, r, lo, T, q=q, option_type=option_type) - market_price
    f_hi = bs_price(S, K, r, hi, T, q=q, option_type=option_type) - market_price

    if f_lo * f_hi > 0:
        return None   # root not bracketed (shouldn't happen for valid inputs)

    for _ in range(max_iter):
        mid   = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, r, mid, T, q=q, option_type=option_type) - market_price

        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
    sigma0: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 50,
    vega_floor: float = 1e-8,
) -> float | None:
    """
    Newton-Raphson implied vol solver.

    Falls back to bisection automatically when vega is too small to trust.
    """
    lower_arb, upper_arb = _no_arb_bounds(S, K, r, T, q, option_type)
    if market_price < lower_arb - 1e-6 or market_price >= upper_arb:
        return None

    sigma = sigma0
    for _ in range(max_iter):
        price = bs_price(S, K, r, sigma, T, q=q, option_type=option_type)
        diff  = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_greeks(S, K, r, sigma, T, q=q, option_type=option_type)["vega"]

        if vega < vega_floor:
            # Low vega makes the Newton step unreliable — switch to bisection
            return implied_vol_bisection(market_price, S, K, r, T, q, option_type, tol)

        sigma = sigma - diff / vega
        sigma = max(_IV_LO, min(_IV_HI, sigma))   # stay in valid range

    # Didn't converge in Newton iterations; use bisection as backstop
    return implied_vol_bisection(market_price, S, K, r, T, q, option_type, tol)
