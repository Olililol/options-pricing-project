"""
Derivatives pricing and implied volatility surface — four layers.

Run:  python main.py
      python main.py --synthetic     (use synthetic data instead of live market)

Layer 1 — Black-Scholes closed form + Greeks (analytical vs finite-difference)
Layer 2 — Convergence: BS, binomial tree, Monte Carlo all agree on the same price
Layer 3 — Implied vol inversion: price → sigma (bisection and Newton-Raphson)
Layer 4 — Real option chain → IV surface → smile/skew interpretation
"""
from __future__ import annotations

import sys
import io
import math
import textwrap

# Force UTF-8 output so Greek letters and arrows print correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================
# Helpers
# ============================================================

def _section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _table(headers: list[str], rows: list, fmt: list[str]) -> None:
    # Accept rows as either lists/tuples or dicts (values extracted in insertion order)
    def _vals(r):
        return list(r.values()) if isinstance(r, dict) else list(r)
    data  = [_vals(r) for r in rows]
    col_w = [max(len(h), max(len(f % (row[i],)) for row in data))
             for i, (h, f) in enumerate(zip(headers, fmt))]
    sep   = "  ".join("-" * w for w in col_w)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)
    for row in data:
        print("  " + "  ".join((f % v).ljust(w) for f, v, w in zip(fmt, row, col_w)))


# ============================================================
# Layer 1: Black-Scholes and Greeks
# ============================================================

def layer1() -> None:
    from black_scholes import bs_price, bs_greeks, bs_greeks_fd

    _section("LAYER 1 -- Black-Scholes closed form + Greeks")

    # Reference contract used throughout the demo
    S, K, r, sigma, T, q = 500.0, 500.0, 0.045, 0.20, 0.5, 0.013

    call = bs_price(S, K, r, sigma, T, q=q, option_type="call")
    put  = bs_price(S, K, r, sigma, T, q=q, option_type="put")
    print(f"\n  Reference contract: S={S}, K={K}, r={r}, σ={sigma}, T={T}y, q={q}")
    print(f"\n  BS call price  = {call:8.4f}")
    print(f"  BS put  price  = {put:8.4f}")

    # Put-call parity check
    parity_lhs = call - put
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    print(f"\n  Put-call parity check:")
    print(f"    C - P              = {parity_lhs:8.4f}")
    print(f"    S·e^{{-qT}} - K·e^{{-rT}} = {parity_rhs:8.4f}")
    print(f"    Difference         = {abs(parity_lhs - parity_rhs):.2e}  (should be ~0)")

    # Greeks comparison
    greeks_a  = bs_greeks(S, K, r, sigma, T, q=q, option_type="call")
    greeks_fd = bs_greeks_fd(S, K, r, sigma, T, q=q, option_type="call")

    print(f"\n  Greeks (ATM call) — analytical vs finite difference:")
    rows = []
    for key in ["delta", "gamma", "vega", "theta", "rho"]:
        a  = greeks_a[key]
        fd = greeks_fd[key]
        rows.append([key, a, fd, abs(a - fd)])
    _table(
        ["Greek", "Analytical", "Finite-diff", "Abs error"],
        rows,
        ["%s", "%+.6f", "%+.6f", "%.2e"],
    )

    # Greeks across spot to show shapes
    print(f"\n  Delta and Gamma across spot (K={K}, σ={sigma}):")
    rows = []
    for spot in [400, 440, 470, 490, 500, 510, 530, 560, 600]:
        g = bs_greeks(float(spot), K, r, sigma, T, q=q, option_type="call")
        rows.append([spot, g["delta"], g["gamma"], g["vega"]])
    _table(["Spot", "Delta", "Gamma", "Vega"], rows, ["%d", "%.4f", "%.6f", "%.4f"])

    print(textwrap.dedent("""
    Interpretation:
      Delta → 0 deep OTM, → 1 deep ITM (probability of ending ITM, roughly).
      Gamma peaks at-the-money: the delta of an ATM option is most sensitive to
        spot moves — expensive to delta-hedge dynamically.
      Vega also peaks ATM: ATM options are most sensitive to vol uncertainty.
    """))


# ============================================================
# Layer 2: Three pricers must agree
# ============================================================

def layer2() -> None:
    from black_scholes import bs_price
    from binomial_tree import binomial_price, convergence_table as bt_convergence
    from monte_carlo   import mc_price, convergence_table as mc_convergence

    _section("LAYER 2 -- Convergence: Black-Scholes, Binomial Tree, Monte Carlo")

    S, K, r, sigma, T, q = 500.0, 500.0, 0.045, 0.20, 0.5, 0.013

    bs = bs_price(S, K, r, sigma, T, q=q, option_type="call")
    bt = binomial_price(S, K, r, sigma, T, N=1000, q=q, option_type="call")
    mc = mc_price(S, K, r, sigma, T, q=q, option_type="call", n_paths=500_000)

    print(f"\n  Reference contract: S={S}, K={K}, r={r}, σ={sigma}, T={T}y")
    print(f"\n  {'Method':<30} {'Price':>10}  {'Error vs BS':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Black-Scholes (closed form)':<30} {bs:10.5f}  {'—':>12}")
    print(f"  {'Binomial CRR (N=1000)':<30} {bt:10.5f}  {bt-bs:+12.5f}")
    print(f"  {'Monte Carlo (500k paths, AV)':<30} {mc['price']:10.5f}  {mc['price']-bs:+12.5f}")
    print(f"  {'MC 95% CI':<30}  [{mc['conf_95_low']:.5f}, {mc['conf_95_high']:.5f}]")

    # Binomial convergence table
    print(f"\n  Binomial tree convergence (European call):")
    bt_rows = bt_convergence(S, K, r, sigma, T, q=q, option_type="call",
                             steps=[10, 25, 50, 100, 200, 500, 1000])
    _table(["N", "Binomial price", "Error vs BS"], bt_rows,
           ["%d", "%.5f", "%+.6f"])

    # MC convergence table
    print(f"\n  Monte Carlo convergence (European call, antithetic variates):")
    mc_rows = mc_convergence(S, K, r, sigma, T, q=q, option_type="call",
                             path_counts=[1_000, 5_000, 20_000, 100_000, 500_000])
    _table(["N paths", "MC price", "Std error", "Error vs BS"], mc_rows,
           ["%d", "%.5f", "%.6f", "%+.6f"])

    # American put: BS can't do it, tree can
    print(f"\n  American vs European put (K={K}, S={S}) — early exercise premium:")
    for S_spot in [480.0, 500.0, 520.0]:
        eu = binomial_price(S_spot, K, r, sigma, T, N=1000, q=q,
                            option_type="put", american=False)
        am = binomial_price(S_spot, K, r, sigma, T, N=1000, q=q,
                            option_type="put", american=True)
        print(f"    S={S_spot:.0f}:  European={eu:.4f}  American={am:.4f}  "
              f"Premium={am-eu:.4f}")

    print(textwrap.dedent("""
    Key insight: three independent methods converge to the same European price.
    The American put premium is the value of the early-exercise right — larger
    when the option is deeply in-the-money and rates are high (because holding
    a deep ITM put means forgoing the interest earned on the strike).
    Black-Scholes has no closed form for this; the tree prices it naturally.
    """))


# ============================================================
# Layer 3: Implied vol inversion
# ============================================================

def layer3() -> None:
    from black_scholes import bs_price
    from implied_vol   import implied_vol_bisection, implied_vol_newton

    _section("LAYER 3 -- Implied volatility: inverting price -> sigma")

    S, K, r, T, q = 500.0, 500.0, 0.045, 0.5, 0.013

    print(f"\n  Recovering known σ from BS-generated prices:")
    print(f"  {'True σ':>8}  {'BS price':>10}  {'IV bisect':>10}  {'IV Newton':>10}  "
          f"{'Bisect err':>12}  {'Newton err':>12}")
    print(f"  {'-'*72}")

    for true_sigma in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80]:
        price  = bs_price(S, K, r, true_sigma, T, q=q, option_type="call")
        iv_bis = implied_vol_bisection(price, S, K, r, T, q=q, option_type="call")
        iv_nwt = implied_vol_newton(price, S, K, r, T, q=q, option_type="call",
                                    sigma0=0.3)
        err_b  = abs(iv_bis - true_sigma) if iv_bis else float("nan")
        err_n  = abs(iv_nwt - true_sigma) if iv_nwt else float("nan")
        print(f"  {true_sigma:>8.2f}  {price:>10.4f}  {iv_bis or float('nan'):>10.6f}  "
              f"{iv_nwt or float('nan'):>10.6f}  {err_b:>12.2e}  {err_n:>12.2e}")

    # Newton convergence speed vs bisection
    print(f"\n  Convergence demo (true σ = 0.25, price = {bs_price(S, K, r, 0.25, T, q=q):.4f}):")
    target = bs_price(S, K, r, 0.25, T, q=q, option_type="call")

    from black_scholes import bs_greeks

    sigma = 0.30
    print(f"  {'Iter':>5}  {'σ estimate':>12}  {'BS price':>10}  {'|Error|':>12}")
    print(f"  {'-'*48}")
    for i in range(8):
        p    = bs_price(S, K, r, sigma, T, q=q, option_type="call")
        vega = bs_greeks(S, K, r, sigma, T, q=q, option_type="call")["vega"]
        print(f"  {i:>5}  {sigma:>12.8f}  {p:>10.6f}  {abs(p - target):>12.2e}")
        if abs(p - target) < 1e-9:
            break
        sigma -= (p - target) / vega

    print(textwrap.dedent("""
    Newton-Raphson halves the number of correct digits each iteration (quadratic
    convergence). Six iterations reach machine precision; bisection needs ~50.
    The trade-off: Newton requires a vega evaluation per step and can diverge
    near zero vega (deep OTM short-dated options). The implementation above
    falls back to bisection automatically in those cases.
    """))


# ============================================================
# Layer 4: IV surface from real (or synthetic) market data
# ============================================================

def layer4(use_synthetic: bool = False) -> None:
    from iv_surface import (
        fetch_option_chain, compute_iv_dataframe, check_put_call_parity,
        plot_iv_smile, plot_iv_surface, plot_parity_errors,
        generate_synthetic_surface,
    )

    _section("LAYER 4 -- Implied volatility surface")

    r, q = 0.045, 0.013

    # `is_synthetic` tracks what we ACTUALLY priced, not what was requested.
    # A live-fetch failure silently falls back to synthetic, and every
    # interpretive claim below must branch on the real source — otherwise we'd
    # narrate "what the market thinks" about numbers no market produced.
    if use_synthetic:
        print("\n  [synthetic SPY-like surface — omit --synthetic to attempt live data]")
        spot, chain = generate_synthetic_surface(spot=500.0)
        is_synthetic = True
    else:
        print("\n  Fetching SPY option chain from Yahoo Finance...")
        try:
            spot, chain = fetch_option_chain("SPY")
            print(f"  SPY spot = ${spot:.2f}")
            print(f"  Loaded {len(chain)} option quotes across {chain['expiry'].nunique()} expirations")
            is_synthetic = False
        except Exception as exc:
            print(f"  Live data unavailable ({exc}); falling back to synthetic surface.")
            spot, chain = generate_synthetic_surface(spot=500.0)
            is_synthetic = True

    # Compute IV for both calls and puts; use OTM options (higher liquidity)
    call_iv = compute_iv_dataframe(spot, chain, r=r, q=q, option_type="call",
                                   moneyness_lo=1.00, moneyness_hi=1.25)
    put_iv  = compute_iv_dataframe(spot, chain, r=r, q=q, option_type="put",
                                   moneyness_lo=0.75, moneyness_hi=1.00)
    iv_df   = pd.concat([put_iv, call_iv], ignore_index=True)

    print(f"\n  Computed {len(iv_df)} implied vols  "
          f"({len(put_iv)} puts OTM, {len(call_iv)} calls OTM)")

    if is_synthetic:
        print(textwrap.dedent("""
      NOTE — this is a SELF-CONSISTENCY / PLUMBING test, not a market observation.
      generate_synthetic_surface() built these prices BY FEEDING a prescribed IV
      function into bs_price(). Inverting them here can only recover the skew that
      was injected — the result is circular by construction. What it legitimately
      checks: that the Newton/bisection inverter round-trips price → σ accurately,
      that calls and puts stitch into one continuous surface, and that the plots
      render. It says NOTHING about what any real market believes. Run without
      --synthetic for that."""))

    # Surface statistics
    print(f"\n  IV surface summary by expiry:")
    print(f"  {'Expiry':<12} {'T (y)':>6}  {'Min IV':>7}  {'ATM IV':>7}  {'Max IV':>7}  {'Skew (put25-call25)':>20}")
    print(f"  {'-'*70}")

    import numpy as np
    for exp in sorted(iv_df["expiry"].unique()):
        sub = iv_df[iv_df["expiry"] == exp]
        T   = sub["T"].iloc[0]
        atm = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]["iv"].values[0]
        lo  = sub["iv"].min()
        hi  = sub["iv"].max()
        # Skew: IV at 25Δ put moneyness (~0.90) minus 25Δ call moneyness (~1.10)
        otm_put  = sub[sub["moneyness"] < 0.95]["iv"].mean()
        otm_call = sub[sub["moneyness"] > 1.05]["iv"].mean()
        skew = (otm_put - otm_call) if (not math.isnan(otm_put) and not math.isnan(otm_call)) else float("nan")
        print(f"  {exp:<12} {T:>6.3f}  {lo*100:>6.1f}%  {atm*100:>6.1f}%  {hi*100:>6.1f}%  {skew*100:>18.2f}%")

    # Put-call parity
    parity = check_put_call_parity(spot, chain, r=r, q=q)
    print(f"\n  Put-call parity check:")
    print(f"    Median |error|  = ${parity['parity_error'].abs().median():.4f}")
    print(f"    Max    |error|  = ${parity['parity_error'].abs().max():.4f}")
    print(f"    Fraction within bid-ask: "
          f"{(parity['parity_error'].abs() <= parity['bid_ask_half_call']).mean()*100:.1f}%")
    if is_synthetic:
        print(f"\n  Interpretation (synthetic): calls and puts were both generated from")
        print(f"  bs_price() at the SAME σ, so parity holds to machine precision up to the")
        print(f"  injected bid-ask spread. This confirms the parity arithmetic is wired")
        print(f"  correctly — it is NOT evidence about real-market efficiency.")
    else:
        print(f"\n  Interpretation: most parity 'violations' sit within the bid-ask spread.")
        print(f"  That is the actual evidence they are transaction-cost noise rather than")
        print(f"  free money — see the error-vs-spread scatter in plots/parity_errors.png.")
        print(f"  Stale quotes and the discrete-dividend approximation explain the residual.")
        print(f"  (Parity is an equality only for European options; the American puts in")
        print(f"  Layer 2 would instead satisfy an inequality.)")

    # Plots
    print(f"\n  Generating plots...")
    plot_iv_smile(iv_df, spot)
    plot_iv_surface(iv_df)
    plot_parity_errors(parity)

    if is_synthetic:
        print(textwrap.dedent("""
        Reading the surface (synthetic):
          • The shape you see — negative skew, upward term structure — is exactly
            what generate_synthetic_surface() hard-coded via its atm_vols and
            skew_slopes parameters. We recovered our own inputs; that is the test
            passing, not a discovery.
          • Do NOT read market psychology into these curves. There is no crash-fear
            premium here because there is no market here — only Black-Scholes prices
            inverted back through Black-Scholes.
          • To see a surface the market actually produced (and that genuinely
            departs from BS in ways nobody designed), run without --synthetic.
        """))
    else:
        print(textwrap.dedent("""
        Reading the surface (live market):
          • The surface is NOT flat — if Black-Scholes were right, it would be.
          • Negative skew (left side higher): the market prices more probability into
            large downside moves than a lognormal distribution predicts. Memories of
            equity crashes (1987, 2008, 2020) push investors to pay up for put
            protection, lifting down-strike IV above ATM IV.
          • Term structure: short-dated IV is often higher (vol mean-reverts to a
            long-run level); in stressed markets this inverts as near-term fear spikes.
          • Caveat on what's really "market" here: a single flat r and q across all
            maturities, plus treating SPY's discrete dividends as a continuous yield,
            means some of the apparent term structure is OUR modeling error, not the
            market's signal. The surface is the market's correction to BS PLUS the
            residue of our own simplifications — honest analysis separates the two.
        """))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import pandas as pd    # needed in layer4; import here so error is clear
    synthetic = "--synthetic" in sys.argv

    layer1()
    layer2()
    layer3()
    layer4(use_synthetic=synthetic)

    print(f"\n{'=' * 72}")
    print("  All layers complete. Plots saved to  ./plots/")
    print(f"{'=' * 72}\n")
