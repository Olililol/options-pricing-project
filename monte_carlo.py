"""
Monte Carlo pricer for European options.

Under the risk-neutral measure Q, the stock price at maturity is:
    S_T = S · exp((r - q - σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)

The drift term is (r - q - σ²/2)·T:
  - (r - q) is the risk-neutral growth rate (net of dividends)
  - -σ²/2   is the Itô correction (Jensen's inequality on the exponential)

Variance reduction — antithetic variates:
  For each standard normal draw Z, also evaluate the payoff at -Z, then average
  the pair into a SINGLE observation: M_i = ½·(payoff(Z_i) + payoff(-Z_i)).
  The gain does NOT come from having more samples — it comes from the negative
  correlation between payoff(Z) and payoff(-Z). One leg is deep ITM exactly when
  the other is OTM, so the pair average has lower variance than two independent
  draws would:
        Var(M) = ½·Var(payoff)·(1 + ρ),   ρ = Corr(payoff(Z), payoff(-Z)) < 0
  For a vanilla call/put ρ is strongly negative (it would be -1 only if the
  payoff were linear in Z, which it isn't past the strike), so the reduction is
  large but not exactly a factor of two. It is effectively "free": no extra
  simulation cost beyond evaluating the payoff twice per Z draw.

Standard error:  SE = std(M) / √N_eff
  N_eff is the number of PAIRED observations, i.e. n_paths/2 — NOT n_paths.
  Each averaged pair M_i is one observation, and the n_base = n_paths//2 values
  in `combined` are those observations. The code computes the SE over exactly
  those n_base values (see `n_eff = len(discounted)`), so it is already correct;
  this note exists because the naive "divide by √n_paths" is a common and wrong
  shortcut that would understate the true standard error by √2.

Confidence interval: price ± 1.96 · SE  (95% CI under CLT)
"""
from __future__ import annotations

import math
import numpy as np


def mc_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 200_000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> dict[str, float]:
    """
    Monte Carlo European option price with optional antithetic variates.

    Returns
    -------
    dict with keys:
        price          : MC estimate
        std_error      : standard error of the estimate
        conf_95_low    : lower bound of 95% confidence interval
        conf_95_high   : upper bound of 95% confidence interval
        n_paths_used   : actual number of paths evaluated
    """
    rng = np.random.default_rng(seed)

    n_base = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(n_base)

    drift     = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)

    def payoff(z: np.ndarray) -> np.ndarray:
        ST = S * np.exp(drift + diffusion * z)
        if option_type == "call":
            return np.maximum(ST - K, 0.0)
        elif option_type == "put":
            return np.maximum(K - ST, 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")

    if antithetic:
        # Each pair (Z, -Z) gives one combined observation
        combined = 0.5 * (payoff(Z) + payoff(-Z))
    else:
        combined = payoff(Z)

    discount        = math.exp(-r * T)
    discounted      = discount * combined
    n_eff           = len(discounted)   # = n_paths//2 with antithetics, n_paths without

    price           = float(np.mean(discounted))
    std_error       = float(np.std(discounted, ddof=1) / math.sqrt(n_eff))
    half_width      = 1.96 * std_error

    return {
        "price":       price,
        "std_error":   std_error,
        "conf_95_low": price - half_width,
        "conf_95_high": price + half_width,
        "n_paths_used": n_paths,
    }


def convergence_table(
    S: float, K: float, r: float, sigma: float, T: float,
    q: float = 0.0,
    option_type: str = "call",
    path_counts: list[int] | None = None,
    seed: int = 0,
) -> list[dict]:
    """MC prices at increasing n_paths to show convergence and SE reduction."""
    if path_counts is None:
        path_counts = [1_000, 5_000, 20_000, 100_000, 500_000]
    from black_scholes import bs_price
    bs = bs_price(S, K, r, sigma, T, q=q, option_type=option_type)
    rows = []
    for n in path_counts:
        result = mc_price(S, K, r, sigma, T, q=q, option_type=option_type,
                          n_paths=n, seed=seed, antithetic=True)
        rows.append({
            "n_paths": n,
            "price": result["price"],
            "std_error": result["std_error"],
            "error_vs_bs": result["price"] - bs,
        })
    return rows
