#!/usr/bin/env python3
"""Deep exploration from 315.05 baseline — focus on score components.

Score formula: avg_sharpe * sqrt(total_trades/40) * (1 - max_dd/30) * consistency

Current: S=191.5, T=122, DD=1.74%, WR=94.6%, consistency=5/5
Score breakdown:
  sharpe_term = 191.5
  trade_factor = sqrt(122/40) = 1.747
  dd_factor = 1 - 1.74/30 = 0.942
  consistency = 1.0
  -> 191.5 * 1.747 * 0.942 * 1.0 = 315.05

To improve: need either higher Sharpe or more trades (or both).
More trades with ~same win rate would increase sqrt(T/40).
Higher Sharpe means better avg return / std deviation ratio.
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy
from backtest_a_harness import WINDOWS, BASE_SEED, simulate_window, compute_composite_score

# Verify baseline
results = []
for w in WINDOWS:
    seed = BASE_SEED + w["seed_offset"]
    strategy.reset_state()
    r = simulate_window(seed, w["n_days"], w["label"])
    results.append(r)
scores = compute_composite_score(results)
BEST_SCORE = scores["composite_score"]
print(f"Verified baseline: {BEST_SCORE:.6f}")
print(f"  S={scores['avg_sharpe']:.1f} T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}%")
print("=" * 100)


def run_structural(label, compute_entry_fn=None, should_trade_fn=None, position_size_fn=None, check_exit_fn=None, **param_overrides):
    """Test structural changes."""
    originals = {}
    if compute_entry_fn:
        originals["compute_entry"] = strategy.compute_entry
        strategy.compute_entry = compute_entry_fn
    if should_trade_fn:
        originals["should_trade"] = strategy.should_trade
        strategy.should_trade = should_trade_fn
    if position_size_fn:
        originals["position_size"] = strategy.position_size
        strategy.position_size = position_size_fn
    if check_exit_fn:
        originals["check_exit"] = strategy.check_exit
        strategy.check_exit = check_exit_fn
    for k, v in param_overrides.items():
        originals[k] = getattr(strategy, k)
        setattr(strategy, k, v)
    try:
        results = []
        for w in WINDOWS:
            seed = BASE_SEED + w["seed_offset"]
            strategy.reset_state()
            r = simulate_window(seed, w["n_days"], w["label"])
            results.append(r)
        scores = compute_composite_score(results)
        cs = scores["composite_score"]
        delta = cs - BEST_SCORE
        marker = "***BETTER***" if delta > 0.1 else ("~same" if abs(delta) < 0.1 else "")
        print(f"{label}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% WR={scores['avg_win_rate']:.1f}% {marker}")
        return cs
    finally:
        for k, v in originals.items():
            if k in ("compute_entry", "should_trade", "position_size", "check_exit"):
                setattr(strategy, k, v)
            else:
                setattr(strategy, k, v)


# === SECTION 1: More trades via relaxed filters ===
print("=== Relaxed LP (wider price range = more trades) ===")
for lp in [0.092, 0.095, 0.098, 0.10, 0.105, 0.11, 0.12, 0.13, 0.15]:
    run_structural(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
print("=== Lower MIN_EDGE (more trades, lower quality) ===")
for me in [0.040, 0.042, 0.044, 0.046, 0.048, 0.050, 0.052, 0.054, 0.055]:
    run_structural(f"ME={me}", MIN_EDGE=me)

print()
print("=== Higher MAX_EDGE ===")
for maxe in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
    run_structural(f"MaxE={maxe}", MAX_EDGE=maxe)

print()
print("=== More concurrent positions ===")
for mc in [25, 30, 35, 40, 50]:
    run_structural(f"MC={mc}", MAX_CONCURRENT=mc)

print()
print("=== Higher capital deployed ===")
for mcd in [0.80, 0.85, 0.90, 0.95, 1.0]:
    run_structural(f"MCD={mcd}", MAX_CAPITAL_DEPLOYED_PCT=mcd)

print()
print("=== Wider resolution window ===")
for rdmin, rdmax in [(1, 21), (2, 25), (2, 30), (2, 35), (1, 30), (3, 21)]:
    run_structural(f"RD={rdmin}-{rdmax}", RESOLUTION_DAYS_MIN=rdmin, RESOLUTION_DAYS_MAX=rdmax)

print()
print("=== MIN_YES_PRICE ===")
for myp in [0.005, 0.008, 0.01, 0.012, 0.015]:
    run_structural(f"MYP={myp}", MIN_YES_PRICE=myp)

print()
# === SECTION 2: Better Sharpe via sizing optimization ===
print("=== Position size min/max (% of capital) ===")
for pmin, pmax in [(0.01, 0.05), (0.015, 0.05), (0.02, 0.04), (0.02, 0.06),
                    (0.02, 0.07), (0.02, 0.08), (0.02, 0.10),
                    (0.025, 0.05), (0.025, 0.06), (0.03, 0.06),
                    (0.015, 0.06), (0.015, 0.07)]:
    run_structural(f"PSZ={pmin}-{pmax}", POSITION_PCT_MIN=pmin, POSITION_PCT_MAX=pmax)

print()
print("=== Kelly fraction ===")
for kf in [0.028, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.038, 0.040]:
    run_structural(f"KF={kf}", KELLY_FRACTION=kf)

print()
print("=== Kelly multiplier ===")
for km in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    run_structural(f"KM={km}", KELLY_MULTIPLIER=km)

print()
# === SECTION 3: Stop loss optimization ===
print("=== Stop loss ===")
for sl in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]:
    run_structural(f"SL={sl}", STOP_LOSS_PCT=sl)

print()
print("=== Partial exit ===")
for pep, pes in [(0.30, 0.50), (0.40, 0.50), (0.50, 0.40), (0.50, 0.50), (0.50, 0.60),
                  (0.60, 0.50), (0.70, 0.50), (0.80, 0.50), (1.0, 0.50), (0.50, 0.30)]:
    run_structural(f"PE={pep}/{pes}", PARTIAL_EXIT_PROFIT_PCT=pep, PARTIAL_EXIT_SELL_PCT=pes)

print()
# === SECTION 4: Extreme combos ===
print("=== EXTREME COMBOS ===")
# More trades + higher Sharpe: relax LP and increase position max
for lp in [0.10, 0.12]:
    for me in [0.050, 0.055]:
        for pmax in [0.05, 0.06]:
            run_structural(f"LP={lp}+ME={me}+PM={pmax}",
                          LONGSHOT_PRICE_MAX=lp, MIN_EDGE=me, POSITION_PCT_MAX=pmax)

print()
# LP + DF re-optimization
print("=== LP + DF re-optimization ===")
for lp in [0.10, 0.12]:
    for df in [0.35, 0.36, 0.37, 0.375, 0.38, 0.40, 0.42]:
        run_structural(f"LP={lp}+DF={df}", LONGSHOT_PRICE_MAX=lp, DISCOUNT_FACTOR=df)

print()
# Different scale for different LP
print("=== LP + custom scale ===")
def make_entry_custom(a, b, lp):
    def compute_entry(price_yes, zscore):
        price_scale = a + b * (price_yes / lp)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

for lp in [0.10, 0.12, 0.15]:
    for a, b in [(-1.80, 2.75), (-1.50, 2.45), (-1.20, 2.15), (-2.0, 2.95),
                 (-1.60, 2.55), (-1.70, 2.65), (-1.90, 2.85)]:
        run_structural(f"LP={lp}+a={a},b={b}",
                      compute_entry_fn=make_entry_custom(a, b, lp),
                      LONGSHOT_PRICE_MAX=lp)
