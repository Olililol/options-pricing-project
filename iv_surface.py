"""
Implied volatility surface: build and interpret.

What the surface tells us:
  Black-Scholes assumes σ is constant and returns are lognormal.
  The market disagrees: different (K, T) pairs imply DIFFERENT σ values.

  Equity index options (e.g. SPY/SPX) typically show:
  - Negative skew: downside puts have higher IV than upside calls at the same |Δ|.
    Reason: investors pay a premium for crash protection; the real-world return
    distribution has fat left tails that the lognormal model ignores.
  - Term structure: short-dated vol tends to be higher than long-dated vol in
    calm markets (vol mean-reverts); the pattern inverts during crises.
  - The surface is thus a map of 'where BS is wrong and by how much'.

Put-call parity as a consistency check:
  C - P = S·e^{-qT} - K·e^{-rT}  (with continuous dividend yield q)
  Any deviation from parity in quoted prices indicates:
    (a) bid-ask spread (most common — use mid-prices to reduce it)
    (b) stale/illiquid quotes
    (c) dividends, borrow costs not captured by our simple model
    (d) NOT real arbitrage — if there were, market makers would close it instantly.
"""
from __future__ import annotations

import math
import warnings
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection

from implied_vol import implied_vol_newton


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def fetch_option_chain(ticker: str = "SPY") -> tuple[float, pd.DataFrame]:
    """
    Fetch a live option chain from Yahoo Finance.

    Returns (spot_price, raw_chain_df).
    Raises ImportError if yfinance is not installed.
    Raises RuntimeError if no data can be retrieved.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance:  pip install yfinance")

    tk      = yf.Ticker(ticker)
    hist    = tk.history(period="2d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}. Check your internet connection.")
    spot    = float(hist["Close"].iloc[-1])

    expirations = tk.options
    if not expirations:
        raise RuntimeError(f"No option chain available for {ticker}.")

    today   = datetime.date.today()
    records = []

    for exp_str in expirations[:10]:    # ≤10 expirations to stay within rate limits
        exp_date = datetime.date.fromisoformat(exp_str)
        T        = (exp_date - today).days / 365.0
        if T <= 0.02:                    # skip same-week expiry (too noisy)
            continue

        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                bid = row.get("bid", 0.0) or 0.0
                ask = row.get("ask", 0.0) or 0.0
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                records.append({
                    "expiry":      exp_str,
                    "T":           T,
                    "K":           float(row["strike"]),
                    "bid":         bid,
                    "ask":         ask,
                    "mid":         0.5 * (bid + ask),
                    "volume":      int(row.get("volume", 0) or 0),
                    "open_interest": int(row.get("openInterest", 0) or 0),
                    "option_type": opt_type,
                })

    if not records:
        raise RuntimeError("Option chain came back empty — try a different ticker.")

    return spot, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# IV computation
# ---------------------------------------------------------------------------

def compute_iv_dataframe(
    spot: float,
    chain: pd.DataFrame,
    r: float = 0.045,
    q: float = 0.013,
    option_type: str = "call",
    moneyness_lo: float = 0.75,
    moneyness_hi: float = 1.30,
    min_mid: float = 0.10,
) -> pd.DataFrame:
    """
    Compute implied vol for every option in the chain.

    Parameters
    ----------
    r, q            : risk-free rate and dividend yield
    option_type     : 'call' or 'put' (use calls for OTM region above spot,
                      puts for OTM region below spot; both for the full surface)
    moneyness_lo/hi : K/S range — discard deep ITM/OTM illiquid options
    min_mid         : discard options with mid price below this (near-zero = stale)
    """
    subset = chain[chain["option_type"] == option_type].copy()
    subset = subset[
        (subset["mid"] >= min_mid) &
        (subset["K"] / spot >= moneyness_lo) &
        (subset["K"] / spot <= moneyness_hi)
    ]

    ivs = []
    for _, row in subset.iterrows():
        iv = implied_vol_newton(
            row["mid"], spot, row["K"], r, row["T"],
            q=q, option_type=option_type,
        )
        if iv is not None and 0.02 < iv < 3.0:
            ivs.append(iv)
        else:
            ivs.append(float("nan"))

    subset = subset.copy()
    subset["iv"]         = ivs
    subset["moneyness"]  = subset["K"] / spot
    subset["log_money"]  = np.log(subset["K"] / spot)

    return subset.dropna(subset=["iv"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Put-call parity check
# ---------------------------------------------------------------------------

def check_put_call_parity(
    spot: float,
    chain: pd.DataFrame,
    r: float = 0.045,
    q: float = 0.013,
) -> pd.DataFrame:
    """
    Compute put-call parity errors: (C - P) - (S·e^{-qT} - K·e^{-rT}).

    Non-zero errors are explained by bid-ask spread, stale quotes, and dividends —
    NOT by exploitable arbitrage.
    """
    calls = (
        chain[chain["option_type"] == "call"]
        .set_index(["expiry", "K"])[["mid", "T", "bid", "ask"]]
        .rename(columns={"mid": "call_mid", "bid": "call_bid", "ask": "call_ask"})
    )
    puts = (
        chain[chain["option_type"] == "put"]
        .set_index(["expiry", "K"])[["mid"]]
        .rename(columns={"mid": "put_mid"})
    )
    parity = calls.join(puts, how="inner").reset_index()
    parity["parity_lhs"] = parity["call_mid"] - parity["put_mid"]
    parity["parity_rhs"] = (
        spot * np.exp(-q * parity["T"]) - parity["K"] * np.exp(-r * parity["T"])
    )
    parity["parity_error"]     = parity["parity_lhs"] - parity["parity_rhs"]
    parity["bid_ask_half_call"] = 0.5 * (parity["call_ask"] - parity["call_bid"])
    return parity.sort_values("T")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_iv_smile(iv_df: pd.DataFrame, spot: float, out_path: str = "plots/iv_smile.png") -> None:
    """
    2-D slice: IV vs. strike/moneyness for each available expiry.
    This shows the volatility smile/skew directly.
    """
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    expirations = sorted(iv_df["expiry"].unique())
    cmap        = plt.cm.viridis
    colors      = [cmap(i / max(len(expirations) - 1, 1)) for i in range(len(expirations))]

    for ax, x_col, xlabel in [
        (axes[0], "K",          "Strike ($)"),
        (axes[1], "log_money",  "Log-moneyness  ln(K/S)"),
    ]:
        for exp, col in zip(expirations, colors):
            sub = iv_df[iv_df["expiry"] == exp].sort_values(x_col)
            T   = sub["T"].iloc[0]
            ax.plot(sub[x_col], sub["iv"] * 100, "o-", ms=3, lw=1.2,
                    color=col, label=f"{exp}  (T={T:.2f}y)")
        if x_col == "K":
            ax.axvline(spot, color="grey", lw=1, ls="--", label=f"Spot = {spot:.1f}")
        else:
            ax.axvline(0, color="grey", lw=1, ls="--", label="ATM")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Implied vol (%)")
        ax.set_title("Volatility smile / skew")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Equity skew: downside puts carry higher IV than upside calls", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  IV smile saved → {out_path}")


def plot_iv_surface(iv_df: pd.DataFrame, out_path: str = "plots/iv_surface.png") -> None:
    """
    3-D implied volatility surface: IV as a function of log-moneyness and maturity.
    """
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        iv_df["log_money"],
        iv_df["T"],
        iv_df["iv"] * 100,
        c=iv_df["iv"] * 100,
        cmap="plasma",
        s=8,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="Implied vol (%)")

    ax.set_xlabel("Log-moneyness  ln(K/S)", labelpad=8)
    ax.set_ylabel("Time to expiry (years)", labelpad=8)
    ax.set_zlabel("Implied vol (%)", labelpad=8)
    ax.set_title("Implied Volatility Surface\n"
                 "(flat would mean Black-Scholes is right; it never is)")
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  IV surface saved → {out_path}")


def plot_parity_errors(parity_df: pd.DataFrame, out_path: str = "plots/parity_errors.png") -> None:
    """
    Histogram and scatter of put-call parity errors.
    Shows they are small and explained by bid-ask, not real arbitrage.
    """
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(parity_df["parity_error"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(0, color="red", lw=1.5, ls="--", label="Zero (perfect parity)")
    ax1.set_xlabel("Parity error  (C - P) − (S·e^{-qT} - K·e^{-rT})  ($)")
    ax1.set_ylabel("Count")
    ax1.set_title("Put-call parity errors")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.scatter(
        parity_df["bid_ask_half_call"],
        parity_df["parity_error"].abs(),
        alpha=0.4, s=12, color="darkorange",
    )
    ax2.set_xlabel("Half call bid-ask spread ($)")
    ax2.set_ylabel("|Parity error|  ($)")
    ax2.set_title("Parity error vs. bid-ask spread\n(correlation shows errors are transaction-cost noise)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parity error plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Synthetic fallback (no internet / market closed)
# ---------------------------------------------------------------------------

def generate_synthetic_surface(spot: float = 500.0) -> tuple[float, pd.DataFrame]:
    """
    Simulate a realistic SPY-like option chain with a negative skew and
    upward-sloping term structure, so plots can be produced without live data.

    The surface is parameterised by the SVI (Stochastic Volatility Inspired)
    approximation:
        σ(k) = a + b·(ρ·(k - m) + √((k-m)² + s²))
    where k = ln(K/F) is log-moneyness.
    """
    from black_scholes import bs_price

    # SVI-like parameters mimicking SPY 2024-2025 implied surface
    maturities = [1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12]
    strikes    = np.linspace(0.70, 1.30, 40) * spot

    # Term-structure: ATM vol rises and skew flattens with maturity
    atm_vols   = [0.120, 0.135, 0.145, 0.160, 0.170, 0.178, 0.185]
    skew_slopes = [-0.35, -0.30, -0.27, -0.23, -0.21, -0.19, -0.18]

    records = []
    today   = datetime.date.today()
    r, q    = 0.045, 0.013

    for T, atm, slope in zip(maturities, atm_vols, skew_slopes):
        exp_date = (today + datetime.timedelta(days=int(T * 365))).isoformat()
        F        = spot * math.exp((r - q) * T)

        for K in strikes:
            k  = math.log(K / F)
            # Simple linear skew approximation
            iv = max(atm + slope * k + 0.5 * 0.10 * k**2, 0.01)

            call_price = bs_price(spot, K, r, iv, T, q=q, option_type="call")
            put_price  = bs_price(spot, K, r, iv, T, q=q, option_type="put")

            spread = max(0.02, 0.01 * call_price)
            for opt_type, mid in [("call", call_price), ("put", put_price)]:
                records.append({
                    "expiry":      exp_date,
                    "T":           T,
                    "K":           K,
                    "bid":         max(mid - spread, 0.01),
                    "ask":         mid + spread,
                    "mid":         mid,
                    "volume":      100,
                    "open_interest": 500,
                    "option_type": opt_type,
                })

    return spot, pd.DataFrame(records)
