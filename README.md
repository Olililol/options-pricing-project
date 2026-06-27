# Options Pricing & Implied Volatility Surface

A from-scratch implementation of derivatives pricing in four layers, ending with a real market implied volatility surface. No pricing libraries — only `math`, `numpy`, `pandas`, and `matplotlib`.

## Layers

### Layer 1 — Black-Scholes closed form + Greeks

`black_scholes.py` implements the Merton (1973) formula directly from the risk-neutral expectation derivation:

Under Q, the stock follows GBM: `dS = (r − q)S dt + σS dW`, so at expiry:

```
S_T = S · exp((r − q − σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)
```

The European call price is the discounted expected payoff, evaluated by completing the square:

```
C = S·e^{-qT}·N(d₁) − K·e^{-rT}·N(d₂)
P = K·e^{-rT}·N(−d₂) − S·e^{-qT}·N(−d₁)

d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d₂ = d₁ − σ·√T
```

`N()` is computed from `math.erfc` — no scipy. All five Greeks are implemented both analytically and by central finite differences, and verified to agree to < 1e-7:

| Greek | Definition | Intuition |
|-------|-----------|-----------|
| Delta | ∂V/∂S | Hedge ratio; ≈ risk-neutral probability of expiring ITM |
| Gamma | ∂²V/∂S² | Convexity; peaks ATM — most expensive node to delta-hedge |
| Vega | ∂V/∂σ | P&L per unit vol move; peaks ATM |
| Theta | −∂V/∂T | Value lost per year as time passes (negative for long options) |
| Rho | ∂V/∂r | Sensitivity to the risk-free rate |

### Layer 2 — Three independent pricers that must agree

`binomial_tree.py` implements the Cox-Ross-Rubinstein (CRR) tree:

```
u = exp(σ·√Δt),   d = 1/u,   p = (exp((r−q)·Δt) − d) / (u − d)
```

Backward induction folds terminal payoffs to the root. For American options, each node takes `max(continuation, exercise)` — capturing early exercise that Black-Scholes cannot handle.

`monte_carlo.py` simulates terminal stock prices and averages discounted payoffs, with **antithetic variates** for variance reduction (each Z draw is paired with −Z; the negative correlation between the two payoffs roughly halves the estimator variance at no extra simulation cost).

All three methods converge to the same European price. At N=1000 the tree is within 0.007 of BS; at 500k paths the MC is within 0.013 with a 95% CI width of ~0.19.

| Method | Price (ATM call, σ=20%) | Error vs BS |
|--------|------------------------|-------------|
| Black-Scholes | 31.9009 | — |
| Binomial (N=1000) | 31.8939 | −0.007 |
| Monte Carlo (500k, AV) | 31.9143 | +0.013 |

The American put premium (value of early exercise) is visible and increases as the put moves deeper ITM.

### Layer 3 — Implied volatility: price → sigma

`implied_vol.py` inverts the BS formula to find the σ that reproduces a given market price.

**Bisection** brackets the root in [1e-6, 10] and halves the interval each step. Guaranteed to converge; ~50 iterations for 1e-8 tolerance.

**Newton-Raphson** uses vega as the derivative:

```
σ_{n+1} = σ_n − (BS(σ_n) − market_price) / vega(σ_n)
```

Quadratically convergent: 3 iterations reach 1e-11 accuracy. Breaks when vega ≈ 0 (deep OTM near expiry); the implementation auto-falls back to bisection in those cases.

```
Iter    σ estimate    |Error|
   0    0.30000000    6.89e+00
   1    0.24998417    2.18e-03
   2    0.25000000    2.63e-11
```

### Layer 4 — Implied volatility surface

`iv_surface.py` fetches a live option chain (SPY via `yfinance`), computes IV across all (strike, maturity) pairs, and plots the surface.

```
python main.py              # live SPY data
python main.py --synthetic  # synthetic surface (no internet required)
```

**What the surface shows:**

The volatility surface is the market's running correction to every assumption Black-Scholes makes. If BS were correct, the surface would be flat. It never is.

- **Negative skew**: downside puts carry higher IV than equidistant upside calls. Equity crashes (1987, 2008, 2020) cause investors to pay a premium for crash protection, which BS's lognormal distribution cannot price. The skew is the market saying "fat left tails are real."
- **Term structure**: ATM vol tends to rise with maturity in calm markets (short-dated vol is low; longer-dated reflects uncertainty compounding). The term structure inverts during crises as near-term fear spikes above long-run expectations.
- **Put-call parity** is used as a consistency check: `C − P = S·e^{-qT} − K·e^{-rT}`. Apparent violations are entirely within the bid-ask spread — no exploitable arbitrage.

Sample surface statistics (synthetic, SPY-like):

| Expiry | T (y) | Min IV | ATM IV | Max IV | Skew (25Δ put − 25Δ call) |
|--------|-------|--------|--------|--------|--------------------------|
| 1 month | 0.08 | 10.3% | 11.8% | 15.2% | 4.4% |
| 3 months | 0.25 | 11.1% | 14.5% | 21.9% | 6.7% |
| 6 months | 0.50 | 11.6% | 16.5% | 23.0% | 6.9% |
| 1 year | 1.00 | 14.5% | 18.6% | 24.0% | 5.8% |

## Strengths

- **Transparent, from-scratch implementation.** Every formula is derived in the docstrings and implemented directly (`N()` from `math.erfc`, no scipy; no quant pricing libraries). For a learning or interview-portfolio context, this is the right call: nothing is hidden behind a black box.
- **Cross-validation by construction.** Three independent pricers (closed form, CRR tree, Monte Carlo) must agree on the European price, and the Greeks are checked analytically against central finite differences. Independent methods converging on the same number is the strongest evidence of correctness short of a formal test suite.
- **The IV solver handles its own failure mode.** Newton-Raphson is fast but breaks as vega → 0; the implementation detects low vega and falls back to bisection rather than overshooting to a garbage root. This is the kind of edge-case handling that separates a toy solver from a usable one.
- **Honest consistency checks.** Put-call parity is used as a sanity check, and parity "violations" are correctly attributed to bid-ask spread rather than free money.
- **Graceful degradation.** The synthetic surface lets the whole pipeline run offline when `yfinance` fails or the market is closed.

## Limitations

These are ordered roughly by how much they would matter if you tried to use this for real pricing rather than demonstration.

### Conceptual

- **The project proves Black-Scholes is wrong, then keeps using it.** Layer 4 builds an IV surface whose entire point is that volatility is *not* constant and returns are *not* lognormal. But every pricer in Layers 1–3 assumes exactly that. Nothing in the codebase consumes the surface it produces. A natural and significant next step is a model that the surface can't immediately falsify: local volatility (Dupire), stochastic volatility (Heston), or jump-diffusion (Merton). Without one, the surface is a diagnostic with no treatment.
- **The surface is not arbitrage-checked or fitted — it is a scatter plot.** Real surfaces must satisfy no-arbitrage constraints: total implied variance increasing in maturity (no calendar arbitrage) and convexity in strike (no butterfly arbitrage). The code plots raw `(K, T, IV)` points with no smoothing and no arb-free parameterization. SVI is used to *generate* the synthetic surface but never *fitted* to real data — so the live surface can contain points that imply negative densities, and you'd never know.

### Methodological

- **r and q are hard-coded flat constants across all maturities.** Using `r = 0.045`, `q = 0.013` for every expiry misprices the forward, and that mispricing contaminates the skew you're trying to measure. The standard fix is to *imply* the forward and discount factor per expiry directly from put-call parity (`C − P = D·(F − K)`), then compute IV off the forward. This removes the rate/dividend assumption entirely and is strictly better. As written, a wrong `q` will tilt the whole skew and you'll misread it as market structure.
- **Continuous dividend yield only.** SPY pays *discrete* dividends. Continuous-yield `q` is an approximation that matters most for American options and single-name equities — precisely the cases where early exercise is driven by dividend dates.
- **No maturity-matched rates.** A single flat `r` ignores the term structure of interest rates, which is observable and free to incorporate.

### Implementation

- **American options are tree-only, with no Greeks and no MC path.** The MC engine simulates only the terminal price, so it cannot price American (no Longstaff-Schwartz), Asian, or barrier options without restructuring to full path simulation. American Greeks aren't exposed anywhere.
- **The antithetic SE bookkeeping is subtly mislabeled.** The standard error itself is computed correctly over `n_paths/2` paired observations. But the docstring's claim of "n_paths effective independent observations" overstates it, and `n_paths_used` reports `n_paths` when only `n_paths/2` pairs were drawn. The number is right; the description of why is not.
- **IV computation loops with `iterrows()`.** Fine for an SPY-sized chain, but it won't scale, and the Newton seed is a fixed `sigma0 = 0.3` rather than an ATM approximation (e.g. Brenner-Subrahmanyam) — a poor start in the deep wings where you most need robustness.
- **Magic numbers are duplicated.** `0.045` and `0.013` appear independently in `main.py`, `iv_surface.py` defaults, and the synthetic generator. Nothing enforces consistency between them; change one and the others silently disagree.
- **`pd` scoping in `main.py` is fragile.** `layer4` uses `pd.concat`, but `pandas` is imported only inside the `__main__` block. Run the file and it works; `import main; main.layer4()` raises `NameError`. `np` is imported locally inside `layer4` but `pd` is not — inconsistent.
- **No automated tests.** "Verification" happens by printing cross-checks at runtime. There are no asserts that fail a build, so a regression would only surface if someone reads the output and notices.

### Data quality

- **`yfinance` quotes are often stale or wide**, especially bid/ask on illiquid strikes. The filters (`mid >= 0.10`, a moneyness band) are weak quality gates; `volume` and `open_interest` are recorded but never used to screen. Garbage strikes can enter the surface unflagged.

**Suggested priority if you extend this:** (1) imply the forward from parity per expiry — it's the highest-leverage correctness fix and removes two assumptions at once; (2) add a Heston or local-vol pricer so the surface feeds back into pricing; (3) add an arbitrage-free fit (SVI per slice with calendar/butterfly checks) before plotting; (4) add a real test suite.

## Project structure

```
options_pricing/
├── black_scholes.py    # BS formula, analytical Greeks, FD Greeks
├── binomial_tree.py    # CRR tree, European + American
├── monte_carlo.py      # MC with antithetic variates
├── implied_vol.py      # Bisection + Newton-Raphson IV solver
├── iv_surface.py       # Option chain fetch, IV computation, surface plots
├── main.py             # Runs all four layers end-to-end
├── requirements.txt
└── plots/              # Generated at runtime
    ├── iv_smile.png
    ├── iv_surface.png
    └── parity_errors.png
```

## Usage

```bash
pip install -r requirements.txt

python main.py              # fetches live SPY data, generates all plots
python main.py --synthetic  # uses a synthetic surface — works offline
```

Output is printed to stdout. Three plots are saved to `./plots/`.

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
yfinance >= 0.2.40      # only needed for live market data
```

The core pricing logic (`black_scholes.py`, `binomial_tree.py`, `monte_carlo.py`, `implied_vol.py`) has no external dependencies beyond the Python standard library and numpy.
