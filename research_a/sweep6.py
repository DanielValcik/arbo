#!/usr/bin/env python3
"""Sweep price scale coefficients and combos from 202.03 baseline."""
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
print(f"Verified baseline: {BEST_SCORE:.6f} (sharpe={scores['avg_sharpe']:.4f})")
print("=" * 100)

# We need to modify the compute_entry function to test different scale params
# Save original function
orig_compute_entry = strategy.compute_entry


def make_compute_entry(a, b):
    """Create compute_entry with price_scale = a + b * (price_yes / LONGSHOT_PRICE_MAX)"""
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
    """Test a specific price scale formula."""
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
        marker = "***BETTER***" if delta > 0.5 else ("~same" if abs(delta) < 0.5 else "")
        print(f"{label}: composite={cs:.6f} ({delta:+.2f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} {marker}")
        return cs
    finally:
        strategy.compute_entry = originals["compute_entry"]
        for k in extra_overrides:
            setattr(strategy, k, originals[k])


# Grid search on scale parameters a and b
# Current: a=-1.8, b=2.8 (so at LONGSHOT_PRICE_MAX, scale = -1.8+2.8 = 1.0)
# Constraint: a + b = 1.0 (so at max price, discount = DISCOUNT_FACTOR * 1.0)
# Actually no constraint — let's explore freely

print("--- Scale a+b grid (b = 1-a so scale=1 at LP_MAX) ---")
for a in [-3.0, -2.5, -2.2, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5, 1.0]:
    b = 1.0 - a
    run_scale(f"a={a:.1f},b={b:.1f}", a, b)

print()
print("--- Scale with b not constrained ---")
for a, b in [(-2.0, 3.0), (-1.8, 2.8), (-2.0, 2.5), (-1.5, 2.5),
             (-1.0, 2.0), (-2.0, 2.0), (-1.5, 2.0), (-0.5, 1.5),
             (-1.8, 3.0), (-1.8, 2.5), (-1.5, 3.0), (-2.5, 3.5),
             (-2.5, 3.0), (-3.0, 4.0), (-3.0, 3.5), (-1.0, 2.5),
             (-2.2, 3.2), (-2.2, 3.0), (-1.9, 2.9), (-1.7, 2.7),
             (-1.85, 2.85), (-1.75, 2.75), (-1.82, 2.82), (-1.78, 2.78)]:
    run_scale(f"a={a},b={b}", a, b)

print()
print("--- Best scale + DF combos ---")
for df in [0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.38, 0.40]:
    run_scale(f"a=-1.8,b=2.8,DF={df}", -1.8, 2.8, DISCOUNT_FACTOR=df)

print()
print("--- Best scale + KF combos ---")
for kf in [0.030, 0.031, 0.032, 0.033, 0.034]:
    run_scale(f"a=-1.8,b=2.8,KF={kf}", -1.8, 2.8, KELLY_FRACTION=kf)

print()
print("--- Scale + DF + KF combos ---")
for a_val in [-2.0, -1.8, -1.5]:
    b_val = 1.0 - a_val
    for df in [0.32, 0.33, 0.35]:
        for kf in [0.032, 0.033]:
            run_scale(f"a={a_val},b={b_val},DF={df},KF={kf}",
                      a_val, b_val, DISCOUNT_FACTOR=df, KELLY_FRACTION=kf)
