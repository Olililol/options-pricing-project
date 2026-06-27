"""
Black-Scholes (Merton 1973) closed-form pricer for European options.

Derivation sketch (risk-neutral expectation route):
  Under the risk-neutral measure Q, the stock follows GBM:
      dS = (r - q) S dt + σ S dW
  so
      S_T = S · exp((r - q - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)

  The European call price is the discounted expected payoff:
      C = e^{-rT} · E^Q[max(S_T - K, 0)]

  Evaluating the expectation by completing the square in the exponent yields:
      C = S·e^{-qT}·N(d₁) - K·e^{-rT}·N(d₂)
      P = K·e^{-rT}·N(-d₂) - S·e^{-qT}·N(-d₁)

  where  d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
         d₂ = d₁ - σ·√T
         q  = continuous dividend yield (0 recovers the original BS 1973 result)

Put-call parity:  C - P = S·e^{-qT} - K·e^{-rT}
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Primitives — implementing N() from first principles avoids scipy dependency
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF: Φ(x) = (1/2) erfc(-x/√2)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> tuple[float, float]:
    """Compute (d₁, d₂) — the two Black-Scholes arguments."""
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bs_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes European option price.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    r           : risk-free rate (continuous, annualised)
    sigma       : volatility (annualised)
    T           : time to expiry (years)
    q           : continuous dividend yield (default 0)
    option_type : 'call' or 'put'
    """
    if T <= 0.0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive; got {sigma}")

    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    elif option_type == "put":
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")


# ---------------------------------------------------------------------------
# Greeks (analytical)
# ---------------------------------------------------------------------------

def bs_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Analytical Black-Scholes Greeks.

    delta : ∂V/∂S       — hedge ratio; how much the option moves per $1 in stock
    gamma : ∂²V/∂S²     — convexity; rate of change of delta with spot
    vega  : ∂V/∂σ       — P&L per +1.0 move in vol (divide by 100 for per 1%)
    theta : -∂V/∂T      — value lost per year as time passes (negative for long options)
    rho   : ∂V/∂r       — sensitivity to the risk-free rate
    """
    if T <= 0.0:
        atm = abs(S - K) < 1e-10
        if option_type == "call":
            delta = 1.0 if S > K else (0.5 if atm else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if atm else 0.0)
        return dict(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    sqrtT  = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    nd1    = _norm_pdf(d1)           # φ(d₁)

    # Gamma and vega are the same for calls and puts
    gamma = disc_q * nd1 / (S * sigma * sqrtT)
    vega  = S * disc_q * nd1 * sqrtT     # per unit move in σ

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -disc_q * S * nd1 * sigma / (2.0 * sqrtT)
            + q * S * disc_q * _norm_cdf(d1)
            - r * K * disc_r * _norm_cdf(d2)
        )
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -disc_q * S * nd1 * sigma / (2.0 * sqrtT)
            - q * S * disc_q * _norm_cdf(-d1)
            + r * K * disc_r * _norm_cdf(-d2)
        )
        rho = -K * T * disc_r * _norm_cdf(-d2)

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Greeks (finite differences) — verification cross-check
# ---------------------------------------------------------------------------

def bs_greeks_fd(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
    h_S: float = 0.01,
    h_sigma: float = 1e-4,
    h_T: float = 1.0 / 365.0,
    h_r: float = 1e-4,
) -> dict[str, float]:
    """
    Central-difference approximations to the Greeks.
    Use this to verify the analytical expressions above.
    """
    def price(**kw) -> float:
        params = dict(S=S, K=K, r=r, sigma=sigma, T=T, q=q, option_type=option_type)
        params.update(kw)
        return bs_price(**params)

    p0     = price()
    delta  = (price(S=S + h_S) - price(S=S - h_S)) / (2 * h_S)
    gamma  = (price(S=S + h_S) - 2 * p0 + price(S=S - h_S)) / (h_S**2)
    vega   = (price(sigma=sigma + h_sigma) - price(sigma=sigma - h_sigma)) / (2 * h_sigma)
    rho    = (price(r=r + h_r) - price(r=r - h_r)) / (2 * h_r)
    # Negate to match the analytical convention: theta = -∂V/∂T (time decay is negative)
    if T > h_T:
        theta = -(price(T=T + h_T) - price(T=T - h_T)) / (2 * h_T)
    else:
        theta = -(price(T=T + h_T) - p0) / h_T

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
