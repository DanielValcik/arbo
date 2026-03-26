#!/usr/bin/env python3
"""Sweep from 282.10 baseline (a=-1.9, ME=0.055, MV=14K, KF=0.032)."""
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


# === Fine-tune every parameter from new baseline ===

print("--- DISCOUNT_FACTOR fine ---")
for df in [0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
print("--- KELLY_FRACTION fine ---")
for kf in [0.031, 0.0315, 0.032, 0.0325, 0.033]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
print("--- MIN_EDGE fine ---")
for me in [0.050, 0.052, 0.054, 0.055, 0.056, 0.058, 0.060]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
print("--- MIN_VOLUME fine ---")
for mv in [12000, 13000, 13500, 14000, 14500, 15000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
print("--- LONGSHOT_PRICE_MAX fine ---")
for lp in [0.088, 0.090, 0.091, 0.092, 0.093, 0.094]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
print("--- ZSCORE_THRESHOLD fine ---")
for zt in [2.95, 2.96, 2.97, 2.98, 2.99, 3.0, 3.01, 3.02, 3.03]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# Scale fine-tune
print("--- SCALE fine ---")
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

for a in [-2.1, -2.0, -1.95, -1.9, -1.85, -1.8, -1.75, -1.7, -1.6]:
    b = 1.0 - a
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
        print(f"a={a:.2f}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals_f["compute_entry"]

print()
# Best single combos
print("--- TOP 2-WAY COMBOS ---")
# These are speculative — will fill based on above
run_one("DF=0.345+ME=0.054", DISCOUNT_FACTOR=0.345, MIN_EDGE=0.054)
run_one("DF=0.36+ME=0.054", DISCOUNT_FACTOR=0.36, MIN_EDGE=0.054)
run_one("DF=0.35+MV=13K", DISCOUNT_FACTOR=0.35, MIN_VOLUME_24H=13000.0)
run_one("DF=0.355+LP=0.091", DISCOUNT_FACTOR=0.355, LONGSHOT_PRICE_MAX=0.091)

print()
print("--- SPIKE params ---")
for slt in [18, 19, 20, 21, 22]:
    run_one(f"SLT={slt}", SPIKE_LOOKBACK_TICKS=slt)

print()
for mht in [10, 11, 12]:
    run_one(f"MHT={mht}", MIN_HISTORY_TICKS=mht)

print()
# Resolution days
for rdm in [19, 20, 21, 22]:
    run_one(f"RDM={rdm}", RESOLUTION_DAYS_MAX=rdm)

print()
# Days pivot
for zdp in [7, 8, 9, 10, 11, 12]:
    run_one(f"ZDP={zdp}", ZSCORE_DAYS_PIVOT=zdp)

print()
# Penalty coeff
for zpc in [0.0003, 0.0005, 0.0008, 0.001, 0.002]:
    run_one(f"ZPC={zpc}", ZSCORE_PENALTY_COEFF=zpc)
