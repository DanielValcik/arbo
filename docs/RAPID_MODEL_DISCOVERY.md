# Rapid Model Discovery, Validation & Deployment

*How professional quantitative trading firms compress months of iteration into days — and how Arbo can adopt the same techniques.*

**Status:** research study
**Author:** CTO research (2026-04-12)
**Target system:** Arbo (Polymarket 3-strategy RDH + Strategy B3 5m / 15m momentum scalpers)
**Audience:** CEO + self (small-team operator)

---

## Executive Summary

Arbo's current research loop is *serial*: deploy one model → collect ~100 live trades (1–2 weeks at 5–15 trades/day) → analyse → re-optimise → redeploy. Each iteration takes 10–14 days, and over a quarter the system gets ~6–8 iterations. That is fine for a single stable strategy; it is ruinous when the regime changes, when a new hypothesis appears, or when one wants to run B3_5M, B3_15M, C, C2, B2 and D in parallel.

Professional quant shops — Renaissance, Two Sigma, Citadel, Jane Street — do not operate this way. They run **parallel portfolios of model variants**, use **Bayesian optimisation** instead of grid search, allocate capital dynamically with **multi-armed bandits**, handle regime change with **online learning + drift detectors**, and bridge the backtest-to-live gap with **shadow deployment infrastructure**. The result: they can identify and scale a new edge in *days*, not weeks or months.

This document is both a survey of those techniques and a concrete migration plan. The actionable output is in §12 and §14: how to turn Arbo's Watchdog into a **multi-variant champion-challenger orchestrator** that tests 5–10 configurations of B3_15M simultaneously, allocates the $175 live pool via Thompson Sampling, and refreshes the parameter search with Bayesian optimisation rather than 235k-config sweeps. Estimated speed-up: **5–10× faster iteration**, with equal or stronger statistical rigour (DSR + PBO baked in).

---

## 1. The Speed Problem in Strategy Research

Arbo's current rhythm is governed by the **minimum-N gate**: no model is trusted until it has produced a statistically meaningful sample, typically N=100 live trades, because the Bayesian shadow+live weighting only shifts the prior after real fills.

At Arbo's current rates:

| Strategy | Trades/day | Days to N=100 | Days to N=300 (DSR-relevant) |
|----------|-----------:|--------------:|-----------------------------:|
| B3_5M    |    8–15    |     7–13      |          20–38               |
| B3_15M   |    2–4     |    25–50      |          75–150              |
| C / C2   |    3–8     |    12–33      |          37–100              |
| B2       |    1–3     |    33–100     |         100–300              |

Each iteration also requires a decision gate (human or Watchdog) and potentially a redeployment. The real blocker is the **serial** structure:
one idea per strategy at a time. A quarter of effort yields 6–8 iterations of one hypothesis on one strategy — which is why Arbo still has a single champion config per layer after months of work.

Contrast this with how Renaissance reportedly operates: *"the Medallion model contains many thousands of signals that are constantly tested, recalibrated, and retired as alpha decays"* ([day-to-data analysis](https://www.day-to-data.com/p/ren-tech)). The unit of research is not "the strategy" but "the hypothesis" — and many hundreds live in production at once.

The gap between the two operating modes is entirely methodological; the techniques needed to close it are well-documented in the academic and industry literature.

---

## 2. Champion-Challenger Pattern — the Core Technique

The **champion-challenger** model, originally from credit-risk and decision-management literature (FICO popularised it commercially), is the skeleton on which every modern production ML and quant-trading system hangs ([FICO](https://www.fico.com/blogs/benefits-championchallenger-testing-decision-management), [Altair](https://altair.com/docs/default-source/resource-library/da_print_solutionsflyer_champion-challenger_letter.pdf)).

**The pattern:**

* **Champion** = current production model; receives the bulk of capital and makes real decisions.
* **Challengers** = 3–10 candidate models running **in parallel** against the same signal stream. Each receives a small sliver of capital (or runs shadow, with no capital).
* Every challenger sees the **same prediction request** the champion sees, so P&L comparisons are paired and not confounded by differences in signal flow (see Minitab's deployment pattern: *"Once a day, the same prediction requests are replayed against the challengers"* — [Minitab docs](https://support.minitab.com/en-us/model-ops/import-and-deploy-models/use-champion-challenger-models-in-a-deployment/)).
* A **promotion rule** — e.g. challenger's DSR exceeds champion's by ≥ τ over ≥ N paired trades, with drift detector not firing — swaps the challenger into the champion slot.

**Why this is faster than serial testing:** if you have 8 challengers running, you accumulate one-per-day's worth of *evidence per variant* plus the **paired-sample advantage** (same market conditions, same signal stream → much lower variance of the difference statistic, so fewer samples needed to decide).

**Industry examples:**

* Two Sigma's "head-to-head" development process keeps multiple candidates in trunk with gated validations, and "works closely with researchers, data teams, and platform engineers to transform innovative ideas into scalable, efficient, and robust production systems" ([Two Sigma Ventures — Model Review](https://twosigmaventures.com/how-we-help/model-review/), [Two Sigma — Head-to-Head, Part 1](https://www.twosigma.com/articles/introduction-to-head-to-head-development-part-1/)).
* Citadel QR: *"quantitative researchers develop and deploy automated strategies … observing patterns and forming hypotheses, and when the data supports a hypothesis, building out large-scale strategies"* ([Citadel Securities QR](https://www.citadelsecurities.com/careers/quantitative-research/)).
* AWS ML Lens explicitly lists champion-challenger alongside canary, blue-green and shadow as one of five core deployment patterns for model rollout ([AWS MLREL-11](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlrel-11.html)).

**Application to Arbo (concrete):**

Instead of "deploy one B3_15M config at $75" →
Deploy **5 B3_15M variants** simultaneously:

```
champion    : current (min_edge 4–11, sigma_scale 0.526, threshold 0.089)     $75
challenger_1: aggressive edge (min_edge 6–13, sigma_scale 0.500)              $25
challenger_2: tight threshold (threshold 0.100)                               $25
challenger_3: loose sigma     (sigma_scale 0.600)                             $25
challenger_4: BTC-filter off (stress test)                                    $25
```

Every signal from the market hits all 5 configs; the orchestrator logs `(variant_id, decision, fill_price, realised_pnl)` per signal. After 150 paired observations (≈ 10 days at B3_15M rates, because observations are **per-variant** not per-strategy), run DSR-adjusted pairwise tests and promote the winner.

This is what `arbo/core/b3_watchdog.py` should graduate to — instead of advising on one config, it becomes an orchestrator over a variant population.

---

## 3. Multi-Armed Bandit (MAB) for Capital Allocation

Once you have multiple challengers, the next question is: how do you **allocate capital** among them day-by-day? Fixed 5×$25 wastes capital on losers and under-invests in winners. The academic answer is the **multi-armed bandit**.

**Theory (brief):** each variant is an "arm"; reward is realised per-trade PnL (or PnL/σ). Three canonical algorithms:

### Epsilon-greedy
With probability ε explore (random arm), with probability 1–ε exploit (best-mean arm). Simple, but explores uniformly even after signal is obvious. Typical ε = 0.1 with decay schedule ε_t = 1/√t.

### UCB1
Select arm maximising `μ̂ᵢ + √(2 ln t / nᵢ)` — mean reward plus an exploration bonus that shrinks with visits. *"The key advantage of UCB1 is that it defines its own mix of exploration vs. exploitation without a user-supplied parameter"* ([James LeDoux blog](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/)).

### Thompson Sampling (recommended for Arbo)
Maintain a posterior over each arm's mean reward; at each decision, **sample** from the posterior and play the arm with the highest sample.

For Bernoulli rewards (won/lost): `Beta(α_i + successes, β_i + failures)`, update on each outcome.

> *"After observing successes and failures in plays of arm i, the algorithm updates the distribution on μᵢ as Beta(Sᵢ(t) + 1, Fᵢ(t) + 1). The algorithm then samples from these posterior distributions of the μᵢ's, and plays the arm according to the probability of its mean being the largest."* ([Stanford TS Tutorial, Russo & Van Roy](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf))

TS achieves **logarithmic regret** (Agrawal & Goyal 2012) — optimal up to constants.

**Trading-specific research:** Zhu & Kwok (2019), *"Adaptive Portfolio by Solving Multi-armed Bandit via Thompson Sampling"* ([arXiv 1911.05309](https://arxiv.org/abs/1911.05309)) show TS-based portfolio allocation outperforms uniform and mean-variance baselines. More recently, *"Combinatorial Adaptive Discounted Thompson Sampling (CADTS)"* reports a **20 % higher out-of-sample Sharpe** vs. classical methods ([arXiv 2410.04217](https://arxiv.org/html/2410.04217v2)).

Practitioner overview for trading: DayTrading.com's *Multi-Armed Bandit Methods in Trading* explicitly frames strategies as arms, with Sharpe or realised PnL as reward ([daytrading.com](https://www.daytrading.com/multi-armed-bandit)).

**Application to Arbo:** wrap variant-capital allocation in a Thompson Sampler.

```python
# every morning, before scanners start, reallocate capital
posterior = {v.id: Beta(1+v.wins, 1+v.losses) for v in active_variants}
draws = {vid: dist.rvs() for vid, dist in posterior.items()}
weights = softmax(draws, temperature=2.0)   # keep some diversification
capital_per_variant = TOTAL_LIVE_CAPITAL * weights
```

Use **discounted Thompson Sampling** (decay older rewards) to handle non-stationarity — CADTS paper gives the discount factor formulation. Library: `mabwiser` from Fidelity provides a production-grade, scikit-learn-style implementation ([Fidelity MABWiser](https://fidelity.github.io/mabwiser/), [PyPI](https://pypi.org/project/mabwiser/)).

---

## 4. Bayesian Optimisation (BO) for Parameter Search

Arbo's current autoresearch loop for Strategy C reportedly ran 6000+ experiments across 3 generations (grid / random sweep over ~235k configs). That's wasteful: most of the search space is obviously bad, and each experiment is expensive (real-price backtest on historical data).

**Bayesian optimisation** replaces the blind sweep with a smart iterative loop:

1. Build a **Gaussian Process** (GP) surrogate model of the objective f(params) → score.
2. Use an **acquisition function** — usually Expected Improvement (EI) — to pick the *next* most-informative point:

   `α_EI(x) = E[max(0, f(x) − f*)]` where f* is the current best.

3. Evaluate f(x), update the GP, repeat.

The surrogate **explicitly models uncertainty**, so it preferentially samples regions with high mean reward *or* high uncertainty — the exploration/exploitation tradeoff, now mechanised.

> *"Bayesian optimization builds a surrogate for the objective and quantifies the uncertainty in that surrogate using Gaussian process regression, and then uses an acquisition function defined from this surrogate to decide where to sample."* ([Wikipedia — Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization), [Frazier tutorial, arXiv 1807.02811](https://arxiv.org/abs/1807.02811))

**Empirical efficiency:** BO typically finds ≥ 95%-optimal configs in **50–100 evaluations** on smooth objectives, vs. 10³–10⁵ for grid search. For Arbo, that's a ~1000× reduction in compute time per autoresearch cycle.

**Tools:**

* `scikit-optimize` (skopt) — simplest, GP + EI/PI/UCB ([xgboosting.com guide](https://xgboosting.com/bayesian-optimization-of-xgboost-hyperparameters-with-scikit-optimize/)).
* `optuna` — TPE (Tree-structured Parzen Estimator), supports pruning, parallel trials ([Optuna](https://optuna.org/), [GitHub](https://github.com/optuna/optuna)).
* `hyperopt` — original TPE implementation.
* `SMAC3` — for mixed-integer / conditional parameter spaces.

**Application to Arbo:** replace the grid sweep in `research/innovations/sweep_b3_15min.py` and `sweep_emos_exit_fusion.py` with Optuna. Objective = OOS backtest score (or DSR) on a purged-CV split. Expected wall-clock speed-up for autoresearch: **30–100×**. And when the answer is "no config works", BO tells you that via flat GP variance, not a week of failed grid.

Caveat: BO assumes a reasonably smooth objective. Very noisy / discontinuous objectives (rare-event strategies) may need many more samples; in those cases `SMAC` with racing procedures is preferred.

---

## 5. Online Learning & Concept-Drift Handling

The elephant in Arbo's room is **regime change**. B3_5M performance in a calm-BTC regime looks nothing like it in a high-volatility regime. Waiting for N=100 trades to redetect this is slow and painful — and moreover, you *don't* want to re-optimise on a contaminated sample of mixed regimes.

**Online learning** updates model weights per-observation without retraining from scratch. The key library is **`river`** (merger of creme and scikit-multiflow):

> *"River is a Python library for online machine learning … supports regression, classification, unsupervised learning. It also handles ad-hoc tasks such as computing online metrics, and concept drift detection."* ([GitHub online-ml/river](https://github.com/online-ml/river), [arXiv 2012.04740](https://arxiv.org/abs/2012.04740))

**Drift detectors** — small algorithms that sit next to the model and raise a flag when input distribution or error rate has shifted:

* **ADWIN (Adaptive Windowing)** — maintains a sliding window; splits it into reference and test sub-windows; if their means differ by more than a Hoeffding bound, declares drift and shrinks the window accordingly ([OneUptime concept-drift guide](https://oneuptime.com/blog/post/2026-01-30-concept-drift-detection/view)).
* **Page-Hinkley (CUSUM)** — cumulative sum of deviations from expected mean; alarm when `mT − min(mT) > λ`. Simpler, very low memory.
* **DDM / EDDM** — monitor error rate mean and standard deviation.

**Application to Arbo:** run a Page-Hinkley detector on B3's *rolling 50-trade win rate* and on *realised slippage vs expected*. When either fires:

1. Watchdog pauses live capital allocation on affected variant.
2. Re-run BO on recent 30-day data only (discard pre-drift).
3. Promote new champion only after 20+ post-drift trades confirm.

This replaces the current "wait for global PnL drawdown trigger" — Page-Hinkley typically fires **3–5× faster** than a drawdown-based trigger.

Shadow C2 Dallas's paper-vs-live divergence (paper +$0.66, live –$0.76) is exactly the signal a drift detector on `realised_pnl − paper_pnl` would have caught in the first 10 trades, not the first 50.

---

## 6. Ensemble Methods — Run Many, Vote

*"If you can't tell which model is best, run all of them and average."* The ensemble result is typically (a) lower variance than any constituent, and (b) more regime-robust because different constituents win in different regimes.

**Bagging / boosting concepts in a trading context:**

* **Signal averaging:** take the mean of N variants' predicted probabilities; size the trade on the ensemble probability.
* **Voting (majority / soft):** trade only when ≥ k of N variants agree on direction.
* **Weighted ensemble:** weight by recent Sharpe, or by **Hierarchical Risk Parity** weights.

**Hierarchical Risk Parity (HRP)** — López de Prado 2016, *"Building Diversified Portfolios that Outperform Out-of-Sample"* ([SSRN 2708678](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), [Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity)) — uses graph-theoretic clustering of the covariance matrix rather than direct inversion, so it:

* works on **ill-conditioned / singular** covariance (few samples, many assets);
* produces **lower out-of-sample variance** than mean-variance in Monte Carlo tests;
* is robust to correlation-matrix noise.

For a small research program like Arbo, HRP is the right way to combine 5–10 variant P&L streams into a single aggregated capital allocation: the algorithm naturally down-weights highly-correlated variants and up-weights diversifying ones.

**Application to Arbo:** after running champion-challenger for a month, rather than kill losers, keep the top-3 and ensemble-combine their signals. A *quorum-3* rule ("enter only if all three B3_15M configs agree") historically kills ~40% of trades but lifts hit-rate by 5–10 pp in most published backtests. The trade-off (fewer trades, higher quality) is exactly what B3_15M's thin-liquidity oracle-lag edge needs.

Libraries: `PyPortfolioOpt` implements HRP in 3 lines; `riskfolio-lib` has richer variants.

---

## 7. Synthetic Data / Bootstrap Augmentation

When live N is small, **squeeze more information from existing samples** via resampling:

### Block bootstrap for time series
Standard bootstrap breaks time-dependence. Politis & Romano (1994) introduced the **stationary bootstrap** (random-length blocks) and **circular block bootstrap** (fixed-length, wrap-around) that preserve autocorrelation structure ([Politis & Romano 1994 PDF](https://www.ssc.wisc.edu/~bhansen/718/Politis%20Romano.pdf), [Politis & White 2004 — automatic block length](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf)).

Practitioner note: *"P-values were obtained using 100 000 bootstrap samples created with the circular block procedure of Politis and Romano (1994), with optimal block size chosen according to Politis and White (2004)"* — now the default for trading-strategy significance tests. Package: `arch.bootstrap` (Kevin Sheppard) in Python.

**Application to Arbo:** every DSR computation and every "is this config better than champion?" test should be done with a stationary block bootstrap, **not** assuming i.i.d. returns. Existing `deflated_sharpe.py` should wrap its returns in `arch.bootstrap.StationaryBootstrap(block_length=auto, returns)`.

### Synthetic market data (GANs / diffusion)

Recent academic work on **TimeGAN** (Yoon et al. 2019) and **QuantGAN** (Wiese et al. 2019) generates synthetic price paths. Limitations, per a 2025 survey ([Tandfonline, Generation of synthetic financial time series by diffusion models](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2528697)):

* TimeGAN "fails to convincingly replicate log-return nuances; users can't condition on inputs";
* QuantGAN "performs better than TimeGAN on shape replication and fat tails" but "skips cross-correlations among multiple series";
* Both struggle with the core **stylised facts** — fat tails, volatility clustering, leverage effect. Diffusion models (DDPMs) look more promising but are immature.

**Honest verdict for Arbo:** **do not use GAN-synthetic data for strategy validation yet.** For data augmentation inside an ML feature extractor, fine; for computing DSR or PBO, lean on block bootstrap instead. Synthetic lies, and we cannot afford false confidence at $175 live.

---

## 8. Shadow Deployment + A/B Testing Infrastructure

Arbo already has the idea (`shadow_exit_tracker.py`). Professional setups generalise this.

**Shadow deployment** (Microsoft, AWS, Google Cloud all document this pattern — [Microsoft playbook](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/shadow-testing/), [GCP](https://cloud.google.com/solutions/application-deployment-and-testing-strategies), [AWS MLREL-11](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlrel-11.html)):

* New version sees **full production traffic** (prediction requests, market signals) but its outputs are **not executed** — only logged.
* Enables apples-to-apples comparison on real data.
* Zero capital risk.

**A/B testing** for trading:

* Split live capital, not traffic: 80% champion / 20% challenger.
* Record fills for both; compute paired t-test or DSR-adjusted Welch test.
* Promotion triggered by pre-registered stopping rule (SPRT or group-sequential).

**Trading-specific infrastructure:** *"firms use canary or blue-green strategies, running two identical systems side by side — one live, one shadowing"* ([Appinventiv HFT guide](https://appinventiv.com/blog/high-frequency-trading-software-development-guide/)). Wallaroo.AI describes the same for ML/AI inference in production: *"the art of A/B testing and shadow deployments"* ([Wallaroo.AI](https://wallaroo.ai/ai-production-experiments-the-art-of-a-b-testing-and-shadow-deployments/)).

**Application to Arbo (concrete):** extend `arbo/strategies/b3_15m_shadow.py` into a generic `ShadowOrchestrator`:

```python
class ShadowOrchestrator:
    def run(self, signal: Signal) -> None:
        # 1. champion decides + executes real order (if live mode)
        champ_decision = self.champion.evaluate(signal)
        if champ_decision.trade:
            self.live_executor.execute(champ_decision)

        # 2. each challenger evaluates same signal in parallel
        for ch in self.challengers:
            ch_decision = ch.evaluate(signal)
            self.paper_engine.simulate(ch_decision, orderbook_snapshot=signal.book)
            self.log(variant_id=ch.id, decision=ch_decision, ...)
```

The paper engine is already validated against real fills (per current memory: *"Paper fill prices = live fill prices (validated)"* for non-inverted books), so shadow P&L ≈ hypothetical live P&L for the no-trade-size-impact regime Arbo operates in ($10–$50 tickets on markets with $1000s of liquidity).

---

## 9. Faster Resolution Signals — Mid-Trade Evaluation

Polymarket binary resolution ($1/$0) is a *terminal* reward signal; waiting 5–15 min for every B3 trade to settle discards the **intermediate information** the orderbook provides.

**Leading indicators of skill:**

* **Fair-value drift at t+30 s:** did the orderbook mid move in our direction? If yes, strategy is identifying real information edge; the final $1/$0 outcome is a noisier function of that edge plus settlement luck.
* **Cumulative marked-to-market at t+60 s:** realised microstructure P&L if we could close now.
* **Directional accuracy (sign only):** did mid move *up* when we bought YES? Computable in seconds, and a consistent estimator of the underlying skill parameter.

Economically, if `E[Δmid | signal] > 0` after 30 s, we have a directional alpha, even if variance of $1/$0 resolution is huge. This is basically the **information coefficient (IC)** concept from equity factor research — *"Rolling IC analysis shows whether a factor decays, swings wildly, or holds up through regime changes"* ([PyQuantNews — IC and Alphalens](https://www.pyquantnews.com/free-python-resources/real-factor-alpha-how-to-measure-it-with-information-coefficient-and-alphalens-in-python)).

**Application to Arbo:** add to every trade record:

```python
@dataclass
class TradeOutcome:
    entry_price: float
    entry_time: datetime
    mid_at_30s: float
    mid_at_60s: float
    mid_at_exit: float
    pnl_realised_terminal: float
    # derived
    directional_correct_30s: bool   # sign(mid_30s - entry) == signal_dir
    directional_correct_60s: bool
```

Then the Bayesian reward fed to the Thompson Sampler can be a *weighted combination* of `directional_correct_60s` (fast, precise) and `pnl_realised_terminal` (slow, definitive):

```python
reward = 0.4 * directional_correct_60s + 0.6 * normalise(pnl_realised)
```

The fast component lets the sampler converge in ~30 observations instead of 100. Formally, this is a **two-scale reward** in the bandit literature (Gittins-style composite reward); the 0.4/0.6 weighting can itself be BO-tuned against historical data.

---

## 10. The Renaissance / Two Sigma Style: Continuous Hypothesis Testing

Putting the above together, the mindset shift is from *"strategies"* to *"hypotheses"*:

> *"In an interview, Jim Simons commented on the importance of having good quant-infra systems to test market hypothesis, particularly in the early years of RenTech … some quants run AI 24/7, testing hundreds of signals a day."* ([HangukQuant — Alpha-Encoding](https://hanguk-quant.medium.com/quantitative-alpha-encoding-data-structures-e6f649cba682), [day-to-data](https://www.day-to-data.com/p/ren-tech))

> Renaissance's Medallion reportedly runs a **single unified model incorporating thousands of signals**, all simultaneously — cross-asset, cross-horizon — and new signals are tested, scaled or retired as **alpha decays** naturally. ([breakingthemarket.com — RenTec Part II](https://breakingthemarket.com/the-greatest-geometric-balancers-renaissance-technologies-part-ii/), [Wikipedia — RenTec](https://en.wikipedia.org/wiki/Renaissance_Technologies))

**Operational model:**

```
   idea
    │
    ▼
   shadow (no money) ────── drop if no edge
    │
    ▼
   incubate ($5–$25) ───── drop if DSR < threshold after 50 trades
    │
    ▼
   small live ($25–$100) ── scale via bandit if outperforming
    │
    ▼
   scaled ($100+)
    │
    ▼
   retire (alpha decayed, drift flag, or displaced by better variant)
```

Each phase gate has a pre-registered stopping rule. Many ideas die at "shadow" or "incubate"; the survivors compound. The **kill rate** (60% fail in incubation per [QuantifiedStrategies.com](https://www.quantifiedstrategies.com/does-quant-trading-work/)) is a feature — it's how the system avoids over-fitting. *"If you test enough random signals, you will eventually find one that looks amazing historically just by luck"* ([Maven Securities on alpha decay](https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/)) — so every survivor must cross a DSR-adjusted bar, not just a raw-Sharpe bar.

For Arbo, this translates to a **strategy lifecycle FSM** with explicit promotion rules, which the Watchdog already has the scaffolding for (Tier 1/2/3, audit log, adaptive_config). It just needs to operate on variants, not a single config.

---

## 11. Practical Toolchain for Arbo

Install once, use everywhere:

| Tool | Purpose | Install |
|------|---------|---------|
| `optuna` | Bayesian optimisation (TPE) — replaces grid sweeps | `pip install optuna` |
| `scikit-optimize` | GP-based BO with EI acquisition | `pip install scikit-optimize` |
| `mabwiser` | Multi-armed bandits (TS, UCB1, ε-greedy), parallel-safe | `pip install mabwiser` |
| `river` | Online learning + drift detectors (ADWIN, Page-Hinkley) | `pip install river` |
| `arch` | Block bootstrap, stationary bootstrap, HAC SE | `pip install arch` |
| `pyportfolioopt` | HRP, risk parity, efficient frontier | `pip install pyportfolioopt` |
| `hypothesis` | Property-based testing for model robustness | `pip install hypothesis` |
| `mlflow` (optional) | Experiment tracking for BO / champion-challenger | `pip install mlflow` |

Integration points inside Arbo:

* `arbo/core/b3_watchdog.py` → `arbo/core/strategy_orchestrator.py` (generalisation to multi-variant).
* `arbo/core/adaptive_config.py` → add `VariantConfig` dataclass; config becomes a *pool*, not a singleton.
* `research/innovations/*sweep*.py` → rewrite as Optuna study (`optuna.create_study(sampler=TPESampler())`).
* New `arbo/core/bandit_allocator.py` → MABWiser TS over variants; daily rebalance.
* New `arbo/core/drift_monitor.py` → river's `PageHinkley()` and `ADWIN()` over each variant's rolling win-rate and PnL-vs-paper.
* New `arbo/dashboard/variant_leaderboard.py` → Slack daily digest: rank of variants by posterior-median Sharpe, with 95% CIs.

---

## 12. Application to Current Arbo State (ACTIONABLE)

**Current state (2026-04-12):** B3_15M spec ready, not yet live. B3_5M live ($175 total). Strategy C / C2 live. Watchdog running (Tier 1 autonomous). Paper-vs-live pricing inversion partially mitigated but not fully (NegRisk spread issue).

**Proposal — "Project PARALLEL":**

### Phase 1 — Shadow Variant Matrix (week 1)

1. Build `ShadowOrchestrator` (see §8). It wraps the existing scanners and forks each signal to all active variants.
2. Define 8 B3_15M variants by varying `min_edge`, `sigma_scale`, `threshold`, `BTC_move_filter` across a small 3-level grid. No live capital — pure shadow.
3. Record per-variant fills using the paper engine with real CLOB snapshots.

**Expected data after 7 days:** ~120–160 shadow signals × 8 variants = 960–1280 paired observations. More than enough for a DSR-adjusted pairwise test *today*, vs ≥ 100 days under current serial process.

### Phase 2 — Bandit-Allocated Live (week 2)

1. Take top-3 variants from Phase 1 by DSR.
2. Allocate $75 total live capital via MABWiser `ThompsonSampling` with discount = 0.98/day.
3. Reward = §9's composite `0.4·dir_60s + 0.6·norm_pnl`.
4. Drift detector (`river.drift.PageHinkley`) on each variant's rolling win-rate. If fires → pause + re-shadow + BO sweep on recent data.

### Phase 3 — Continuous Autoresearch (week 3+)

1. Every Sunday, Optuna study runs on *last 60 days* of paper+live data; objective = OOS-DSR under purged CV (López de Prado CPCV — see [quantbeckman.com](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)).
2. Top 2 new configs auto-enrolled as challengers; bottom 2 retired.
3. Watchdog absorbs the orchestrator role: Tier 1 now includes *variant admission, variant retirement, bandit temperature*; Tier 3 still gates hard caps.

### Phase 4 — Cross-Strategy Bandit (month 2)

Extend bandit one level up: $1000 capital pool reallocated weekly across B3, C, C2, B2, D1-UFC, etc. using HRP for diversification-weighted top layer and TS for the bottom variant layer. This is the **Renaissance-shaped** end state.

**Estimated speed-up:** iteration cycle from 10–14 days → **1–2 days** for shadow-only hypothesis validation; **5–7 days** for a fully-promoted new champion. That's a 5–10× research-velocity improvement.

**File plan:**

```
arbo/
├── core/
│   ├── strategy_orchestrator.py      NEW (§8)
│   ├── variant_pool.py               NEW (VariantConfig + pool mgmt)
│   ├── bandit_allocator.py           NEW (MABWiser wrapper)
│   ├── drift_monitor.py              NEW (river detectors)
│   ├── fast_reward.py                NEW (§9 mid-trade features)
│   └── b3_watchdog.py                REFACTOR → multi-variant aware
├── research/
│   ├── bo_sweep_b3_15m.py            NEW (Optuna replacement for grid)
│   └── bo_sweep_c2.py                NEW
└── dashboard/
    └── variant_leaderboard.py        NEW (Slack digest)
docs/
└── RAPID_MODEL_DISCOVERY.md          (this file)
```

---

## 13. Tradeoffs and Risks

Honest list:

1. **Complexity.** Moving from 1 champion to 5–10 variants multiplies code surface, logging volume (×10), database schema (variant_id on every row), and alerting rules. For a single-operator system this is real. Mitigation: build the orchestrator *once*, make variants declarative (YAML per variant in `config/variants/*.yaml`), keep pipelines DRY.

2. **Bandit over-exploitation in small N.** Thompson Sampling with Beta(1,1) prior can lock onto an early-lucky variant and under-explore real champions. Mitigation: floor each arm's capital at ≥ 10% until it has ≥ 30 observations; use **discounted** TS so past luck decays; consider UCB1 (deterministic exploration bonus) for the first phase.

3. **BO assumes smoothness.** If the objective surface is pathological (e.g. Strategy B3 near the `min_edge = LIVE_MIN_EDGE` threshold has a cliff), GP-based BO misbehaves. Mitigation: use **Optuna TPE** (tree-structured, handles discontinuities better) rather than vanilla GP-EI; always verify BO's best config against a neighborhood random search before promoting.

4. **Synthetic data lies.** Already addressed in §7. Decision: don't use GAN-synthetic data for any promotion decision; block-bootstrap only.

5. **Multiple-testing inflation.** Testing 8 variants in parallel raises the probability of at least one looking significant by luck. Mitigation: use DSR (corrects for trial count); apply **Bonferroni or Benjamini-Hochberg** on promotion p-values; require an **out-of-sample validation window** before any live-capital change. López de Prado's [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) and [PBO framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) are explicitly designed for this.

6. **Speed vs robustness tradeoff.** Faster iteration = faster overfit. If we go from "decide in 100 trades" to "decide in 30 paired observations", we MUST compensate with more rigorous statistics (DSR / PBO / CPCV) and with lifecycle gates (shadow → incubate → live). Speed without rigour is a path to ruin.

7. **Watchdog autonomy creep.** Giving Watchdog power over variant admission/retirement expands Tier 1 scope. Keep variant creation behind a human/CEO gate initially; only let Watchdog kill / re-weight existing variants.

---

## 14. Decision Framework: When to Use Each Technique

| Situation | Recommended technique | Arbo example |
|-----------|----------------------|--------------|
| Lots of historical data, slow live data | Shadow deployment + Bayesian optimisation | B3_15M pre-launch (we have 1 year BTC data, 0 live trades) |
| Want to test 10 ideas at once | Champion–challenger with shadow | Compare 8 B3_15M configs before committing any capital |
| Capital allocation across variants of one strategy | Thompson Sampling (MABWiser) | Weekly rebalance of $75 over top-3 B3_15M variants |
| Capital allocation across different strategies | HRP on weekly PnL series | Month-2 step: split $1000 over B3 / C / C2 / B2 / D |
| Strategy has many continuous params | Optuna TPE / BO | Replace grid sweeps for C2, B3 weekly autoresearch |
| Strategy has discrete / conditional params | SMAC3 | Strategy D with its sport-specific conditional trees |
| Suspect regime change | Online learning + Page-Hinkley / ADWIN | Post-BTC-crash drift check on B3 |
| Limited live samples | Block bootstrap (stationary, Politis-Romano) | All DSR / hypothesis tests under N < 300 |
| New, untested idea | Shadow first, then small champion-challenger | Any new sub-strategy in D (D1/D2/D3) |
| Need fast feedback signal | Composite reward (§9): dir_accuracy + pnl | Everywhere — this is "free" once logged |
| Trying to find parameter surface cliffs | Latin Hypercube / Sobol seed + BO refine | First run on any new strategy before BO takes over |
| Want maximum theoretical efficiency | Discounted Thompson Sampling | Long-running variant pool with regime drift |
| Want maximum interpretability | UCB1 (deterministic confidence bound) | Regulatory / CEO-facing reports |
| Want highest single-model rigour | Combinatorial Purged Cross-Validation + DSR | Every promotion gate |

---

## Implementation Roadmap (one-page)

**Week 1 — Infrastructure.**
Build `ShadowOrchestrator`, `VariantConfig`, `variant_pool.py`. Add `variant_id` to paper_trades and live_trades tables (Alembic migration). Extend logging.

**Week 2 — B3_15M variant matrix.**
8 variants in shadow. Collect 1000+ paired observations. First leaderboard.

**Week 3 — BO for autoresearch.**
Convert one sweep (e.g. `sweep_emos_exit_fusion.py`) to Optuna. Validate same-or-better optimum in 1/30th of the time.

**Week 4 — Bandit allocator live.**
MABWiser Thompson Sampling on top-3 B3_15M variants with $75. Drift monitors on each.

**Week 5 — Composite reward.**
Log mid-at-30s / mid-at-60s on every trade; feed composite reward to bandit. Measure bandit convergence speed improvement.

**Week 6 — Watchdog integration.**
Tier 1 handles variant rewight / retirement autonomously. Tier 2 flags variant admission for CEO. Audit log + Slack digest.

**Week 7–8 — Extend to Strategy C2, B2.**
Same pattern. Lift cross-strategy bandit.

**Month 2 — HRP cross-strategy layer.**
Top-of-stack HRP on weekly PnL. $1000 pool. This is the endgame.

---

## Sources

Academic / foundational:
- [Bailey & López de Prado (2014) — *The Deflated Sharpe Ratio*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Bailey, Borwein, López de Prado, Zhu — *The Probability of Backtest Overfitting*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [López de Prado (2016) — *Building Diversified Portfolios that Outperform Out-of-Sample* (HRP)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [Wikipedia — Hierarchical Risk Parity](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity)
- [Frazier — *A Tutorial on Bayesian Optimization* (arXiv 1807.02811)](https://arxiv.org/abs/1807.02811)
- [Russo & Van Roy — *A Tutorial on Thompson Sampling* (Stanford)](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Agrawal & Goyal — *Analysis of Thompson Sampling for the Multi-armed Bandit*](http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf)
- [Zhu & Kwok — *Adaptive Portfolio by Solving Multi-armed Bandit via Thompson Sampling* (arXiv 1911.05309)](https://arxiv.org/abs/1911.05309)
- [*Improving Portfolio Optimization with Bandit Networks* (arXiv 2410.04217)](https://arxiv.org/html/2410.04217v2)
- [Politis & Romano — *The Stationary Bootstrap*](https://www.ssc.wisc.edu/~bhansen/718/Politis%20Romano.pdf)
- [Politis & White — *Automatic Block-Length Selection*](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf)
- [Wiese et al. — *Quant GANs: Deep Generation of Financial Time Series*](https://www.semanticscholar.org/paper/Quant-GANs:-deep-generation-of-financial-time-Wiese-Knobloch/3cf57cad75d71bffac9fc4589d7b294d90558a13)
- [*Generation of synthetic financial time series by diffusion models* (2025)](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2528697)
- [*River: machine learning for streaming data in Python* (arXiv 2012.04740)](https://arxiv.org/abs/2012.04740)
- [*The Probability of Backtest Overfitting*, Bailey et al. PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [*Combinatorial Purged Cross Validation for Optimization*, quantbeckman.com](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)

Industry / practice:
- [Two Sigma — *Treating Data as Code*](https://www.twosigma.com/articles/treating-data-as-code-at-two-sigma/)
- [Two Sigma — *Introduction to Head-to-Head Development*](https://www.twosigma.com/articles/introduction-to-head-to-head-development-part-1/)
- [Two Sigma Ventures — Model Review](https://twosigmaventures.com/how-we-help/model-review/)
- [Citadel Securities — Quantitative Research careers](https://www.citadelsecurities.com/careers/quantitative-research/)
- [Jane Street — Quantitative Research](https://www.janestreet.com/quantitative-research/)
- [Renaissance Technologies — Wikipedia](https://en.wikipedia.org/wiki/Renaissance_Technologies)
- [day-to-data — *Ren Tech deep-dive*](https://www.day-to-data.com/p/ren-tech)
- [breakingthemarket.com — *The Greatest Geometric Balancers: Renaissance Technologies, Part II*](https://breakingthemarket.com/the-greatest-geometric-balancers-renaissance-technologies-part-ii/)
- [Maven Securities — *Alpha decay*](https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/)
- [FICO — *Champion / Challenger testing in decision management*](https://www.fico.com/blogs/benefits-championchallenger-testing-decision-management)
- [Altair — *Champion/Challenger techniques improve AI*](https://altair.com/docs/default-source/resource-library/da_print_solutionsflyer_champion-challenger_letter.pdf)
- [Minitab — *Use champion/challenger models in a deployment*](https://support.minitab.com/en-us/model-ops/import-and-deploy-models/use-champion-challenger-models-in-a-deployment/)
- [AWS ML Lens — MLREL-11 *Deployment and Testing Strategy*](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/mlrel-11.html)
- [Wallaroo.AI — *A/B Testing and Shadow Deployments*](https://wallaroo.ai/ai-production-experiments-the-art-of-a-b-testing-and-shadow-deployments/)
- [Microsoft Engineering Fundamentals — Shadow Testing](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/shadow-testing/)
- [Appinventiv — *High-Frequency Trading Software Development Guide*](https://appinventiv.com/blog/high-frequency-trading-software-development-guide/)
- [DayTrading.com — *Multi-Armed Bandit Methods in Trading*](https://www.daytrading.com/multi-armed-bandit)
- [QuantifiedStrategies — *Does Quant Trading Work?*](https://www.quantifiedstrategies.com/does-quant-trading-work/)

Tools:
- [Optuna — hyperparameter optimization framework](https://optuna.org/) · [GitHub](https://github.com/optuna/optuna)
- [scikit-optimize](https://xgboosting.com/bayesian-optimization-of-xgboost-hyperparameters-with-scikit-optimize/)
- [MABWiser — Fidelity contextual MAB library](https://fidelity.github.io/mabwiser/) · [PyPI](https://pypi.org/project/mabwiser/)
- [river — online ML in Python](https://github.com/online-ml/river)
- [OneUptime — *How to Implement Concept Drift Detection*](https://oneuptime.com/blog/post/2026-01-30-concept-drift-detection/view)
- [Wikipedia — Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)
- [Wikipedia — Multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Wikipedia — Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling)
- [James LeDoux — *Bandit algorithms in Python*](https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/)
- [PyQuantNews — *Real Factor Alpha with Information Coefficient*](https://www.pyquantnews.com/free-python-resources/real-factor-alpha-how-to-measure-it-with-information-coefficient-and-alphalens-in-python)
