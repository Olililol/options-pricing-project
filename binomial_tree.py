"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

The tree discretises the GBM stock process over N equal time steps Δt = T/N:

    u = exp(σ·√Δt)      [up factor]
    d = 1/u              [down factor; 1/u ensures the tree recombines]
    p = (exp((r-q)·Δt) - d) / (u - d)   [risk-neutral up-probability]

Algorithm (backward induction):
  1. Build terminal stock prices:  S_j = S · u^j · d^{N-j},  j = 0, …, N
  2. Compute terminal option payoffs V_j = max(S_j - K, 0) for calls, etc.
  3. Fold backwards one step at a time:
       V_j = e^{-rΔt} · (p · V_{j+1} + (1-p) · V_j)
  4. For American options, at each node take max(continuation, exercise value).
     This is the key feature Black-Scholes CANNOT handle: the right to exercise early.

Convergence: European CRR price → BS price as N → ∞ (rate ~ 1/N with oscillations;
a smoothed variant converges like 1/N² but plain CRR is enough for comparison).
"""
from __future__ import annotations

import math
import numpy as np


def binomial_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 500,
    q: float = 0.0,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """
    CRR binomial tree option price.

    Parameters
    ----------
    N        : number of time steps (accuracy improves with N; cost is O(N²))
    q        : continuous dividend yield
    american : if True, the holder can exercise early at any node
    """
    dt      = T / N
    u       = math.exp(sigma * math.sqrt(dt))
    d       = 1.0 / u
    disc    = math.exp(-r * dt)
    p       = (math.exp((r - q) * dt) - d) / (u - d)
    q_prob  = 1.0 - p

    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability {p:.4f} is outside (0,1). "
            "Check that r, sigma, T are sensible."
        )

    # --- Terminal stock prices ---
    # Node (N, j): stock went up j times and down (N-j) times
    j  = np.arange(N + 1, dtype=float)
    ST = S * (u ** j) * (d ** (N - j))

    # --- Terminal option payoffs ---
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")

    # --- Backward induction ---
    for step in range(N - 1, -1, -1):
        # Continuation value (one discounted expected step back)
        V = disc * (p * V[1:step + 2] + q_prob * V[0:step + 1])

        if american:
            # Stock price at each node of the current step
            j_step = np.arange(step + 1, dtype=float)
            S_step = S * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                exercise = np.maximum(S_step - K, 0.0)
            else:
                exercise = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])


def convergence_table(
    S: float, K: float, r: float, sigma: float, T: float,
    q: float = 0.0,
    option_type: str = "call",
    steps: list[int] | None = None,
) -> list[dict]:
    """Return a table of binomial prices at increasing N for convergence analysis."""
    if steps is None:
        steps = [10, 25, 50, 100, 200, 500, 1000]
    from black_scholes import bs_price
    bs = bs_price(S, K, r, sigma, T, q=q, option_type=option_type)
    rows = []
    for N in steps:
        price = binomial_price(S, K, r, sigma, T, N=N, q=q, option_type=option_type)
        rows.append({"N": N, "price": price, "error_vs_bs": price - bs})
    return rows
