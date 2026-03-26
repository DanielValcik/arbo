#!/usr/bin/env python3
"""Sweep from 315.05 baseline (a=-1.80+2.75, DF=0.375, MV=13.5K, ME=0.055)."""
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
print(f"  DF={strategy.DISCOUNT_FACTOR}, KF={strategy.KELLY_FRACTION}, ME={strategy.MIN_EDGE}")
print("=" * 100)


def run_one(label, **overrides):
    originals = {}
    for k, v in overrides.items():
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
            setattr(strategy, k, v)


# DF ultra-fine
print("--- DF ---")
for df in [0.374, 0.375, 0.376, 0.377, 0.378, 0.380, 0.385]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# Scale ultra-fine around -1.80+2.75
print("--- SCALE ---")
orig_compute_entry = strategy.compute_entry

def make_compute_entry(a, b):
    def compute_entry(price_yes, zscore):
        price_scale = a + b * (price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        effective_discount -= strategy.DISCOUNT_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        edge += strategy.EDGE_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        return edge, model_no_prob
    return compute_entry

for a, b in [(-1.82, 2.77), (-1.82, 2.75), (-1.81, 2.76), (-1.81, 2.75),
             (-1.80, 2.75), (-1.80, 2.74), (-1.80, 2.76),
             (-1.79, 2.74), (-1.79, 2.75), (-1.78, 2.73), (-1.78, 2.75)]:
    originals_f = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_compute_entry(a, b)
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
        marker = "***BETTER***" if delta > 0.1 else ""
        print(f"a={a},b={b}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals_f["compute_entry"]

print()
# Try different DF with best scales
print("--- Scale + DF 2D fine ---")
for a, b in [(-1.80, 2.75), (-1.82, 2.77), (-1.79, 2.74)]:
    for df in [0.375, 0.376, 0.377, 0.378]:
        originals_f = {"compute_entry": strategy.compute_entry}
        strategy.compute_entry = make_compute_entry(a, b)
        orig_df = strategy.DISCOUNT_FACTOR
        strategy.DISCOUNT_FACTOR = df
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
            marker = "***BETTER***" if delta > 0.1 else ""
            print(f"a={a},b={b},DF={df}: {cs:.2f} ({delta:+.2f}) {marker}")
        finally:
            strategy.compute_entry = originals_f["compute_entry"]
            strategy.DISCOUNT_FACTOR = orig_df

print()
# MV fine
print("--- MV ---")
for mv in [13000, 13200, 13500, 13800, 14000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
# ME fine
print("--- ME ---")
for me in [0.054, 0.055, 0.056, 0.058]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# ZT fine
print("--- ZT ---")
for zt in [2.98, 2.99, 3.0, 3.01, 3.02]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# LP fine
print("--- LP ---")
for lp in [0.091, 0.092, 0.093]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)
