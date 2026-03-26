#!/usr/bin/env python3
"""Sweep around ZT=2.96 + scale a=-1.3 (221.41) — find optimal combo."""
import sys
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
print("=" * 100)

orig_compute_entry = strategy.compute_entry

def make_compute_entry(a, b):
    def compute_entry(price_yes, zscore):
        price_scale = a + b * (price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        effective_discount -= strategy.DISCOUNT_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        effective_discount = max(0.05, min(effective_discount, 0.95))
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        edge += strategy.EDGE_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        return edge, model_no_prob
    return compute_entry


def run_scale(label, a, b, **extra_overrides):
    originals = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_compute_entry(a, b)
    for k, v in extra_overrides.items():
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
        strategy.compute_entry = originals["compute_entry"]
        for k in extra_overrides:
            setattr(strategy, k, originals[k])


# Fine-grid scale with ZT=2.96
print("--- ZT=2.96 + fine scale grid ---")
for a in [-1.50, -1.40, -1.35, -1.30, -1.25, -1.20, -1.15, -1.10, -1.00, -0.90, -0.80, -0.50, 0.0]:
    b = 1.0 - a
    run_scale(f"a={a:.2f}", a, b, ZSCORE_THRESHOLD=2.96)

print()
# Best scale + DF combos with ZT=2.96
print("--- ZT=2.96 + a=-1.3 + DF ---")
for df in [0.30, 0.32, 0.33, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38, 0.40]:
    run_scale(f"DF={df}", -1.3, 2.3, ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=df)

print()
# Best scale + ME combos with ZT=2.96
print("--- ZT=2.96 + a=-1.3 + ME ---")
for me in [0.02, 0.03, 0.04, 0.05, 0.055, 0.06]:
    run_scale(f"ME={me}", -1.3, 2.3, ZSCORE_THRESHOLD=2.96, MIN_EDGE=me)

print()
# Explore even flatter scale (a > -1.0)
print("--- ZT=2.96 + flat scales + DF grid ---")
for a in [-1.0, -0.8, -0.5, 0.0]:
    b = 1.0 - a
    for df in [0.30, 0.35, 0.40, 0.45, 0.50]:
        run_scale(f"a={a:.1f}+DF={df}", a, b, ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=df)

print()
# ZT fine-tune with best scale
print("--- a=-1.3 + ZT grid ---")
for zt in [2.92, 2.94, 2.95, 2.955, 2.96, 2.965, 2.97, 2.98, 3.0]:
    run_scale(f"ZT={zt}", -1.3, 2.3, ZSCORE_THRESHOLD=zt)

print()
# Scale + ZT + MV grid
print("--- a=-1.3 + ZT=2.96 + MV ---")
for mv in [12000, 13000, 14000, 15000]:
    run_scale(f"MV={mv}", -1.3, 2.3, ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=float(mv))

print()
# Triple: scale + ZT + DF + MV
print("--- a=-1.3 + ZT=2.96 + best DF + MV ---")
for df in [0.33, 0.34, 0.345, 0.35, 0.36]:
    for mv in [14000, 15000]:
        run_scale(f"DF={df}+MV={mv}", -1.3, 2.3, ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=df, MIN_VOLUME_24H=float(mv))

print()
# Try unconstrained b (not b = 1-a)
print("--- ZT=2.96 + unconstrained scale ---")
for a, b in [(-1.3, 2.0), (-1.3, 2.3), (-1.3, 2.5), (-1.3, 3.0),
             (-1.0, 1.5), (-1.0, 2.0), (-1.0, 2.5),
             (-0.8, 1.5), (-0.8, 1.8), (-0.8, 2.0),
             (-0.5, 1.0), (-0.5, 1.5), (-0.5, 2.0)]:
    run_scale(f"a={a},b={b}", a, b, ZSCORE_THRESHOLD=2.96)
