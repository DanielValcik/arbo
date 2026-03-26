#!/usr/bin/env python3
"""Sweep around ZT=2.96 (273.60) — find best combos."""
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


# First verify ZT=2.96
print("--- ZT=2.96 baseline ---")
run_one("ZT=2.96", ZSCORE_THRESHOLD=2.96)

print()
# Fine-tune ZT very precisely
print("--- ZT ultra-fine ---")
for zt in [2.955, 2.958, 2.960, 2.962, 2.965, 2.968, 2.970]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# ZT=2.96 + single params
print("--- ZT=2.96 + singles ---")
run_one("ZT=2.96+MV=14K", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0)
run_one("ZT=2.96+MV=13K", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=13000.0)
run_one("ZT=2.96+DF=0.345", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.96+DF=0.34", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.34)
run_one("ZT=2.96+DF=0.35", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.35)
run_one("ZT=2.96+DF=0.355", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.355)
run_one("ZT=2.96+ME=0.05", ZSCORE_THRESHOLD=2.96, MIN_EDGE=0.05)
run_one("ZT=2.96+ME=0.055", ZSCORE_THRESHOLD=2.96, MIN_EDGE=0.055)
run_one("ZT=2.96+ME=0.04", ZSCORE_THRESHOLD=2.96, MIN_EDGE=0.04)
run_one("ZT=2.96+LP=0.092", ZSCORE_THRESHOLD=2.96, LONGSHOT_PRICE_MAX=0.092)
run_one("ZT=2.96+LP=0.091", ZSCORE_THRESHOLD=2.96, LONGSHOT_PRICE_MAX=0.091)
run_one("ZT=2.96+LP=0.088", ZSCORE_THRESHOLD=2.96, LONGSHOT_PRICE_MAX=0.088)
run_one("ZT=2.96+LP=0.085", ZSCORE_THRESHOLD=2.96, LONGSHOT_PRICE_MAX=0.085)
run_one("ZT=2.96+KF=0.033", ZSCORE_THRESHOLD=2.96, KELLY_FRACTION=0.033)
run_one("ZT=2.96+KF=0.031", ZSCORE_THRESHOLD=2.96, KELLY_FRACTION=0.031)

print()
# ZT=2.96 + top doubles
print("--- ZT=2.96 + doubles ---")
run_one("ZT=2.96+MV=14K+DF=0.345", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.96+MV=14K+ME=0.05", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, MIN_EDGE=0.05)
run_one("ZT=2.96+MV=14K+LP=0.092", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, LONGSHOT_PRICE_MAX=0.092)
run_one("ZT=2.96+DF=0.345+ME=0.05", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.345, MIN_EDGE=0.05)
run_one("ZT=2.96+DF=0.345+LP=0.092", ZSCORE_THRESHOLD=2.96, DISCOUNT_FACTOR=0.345, LONGSHOT_PRICE_MAX=0.092)

print()
# ZT=2.96 + triples
print("--- ZT=2.96 + triples ---")
run_one("ZT=2.96+MV=14K+DF=0.345+ME=0.05", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345, MIN_EDGE=0.05)
run_one("ZT=2.96+MV=14K+DF=0.345+LP=0.092", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345, LONGSHOT_PRICE_MAX=0.092)
run_one("ZT=2.96+MV=14K+ME=0.05+LP=0.092", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092)

print()
# ZT=2.96 + quad
print("--- ZT=2.96 + quad ---")
run_one("ZT=2.96+MV=14K+DF=0.345+ME=0.05+LP=0.092", ZSCORE_THRESHOLD=2.96, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092)

print()
# New ideas: scale coefficients with ZT=2.96
print("--- ZT=2.96 + scale coefficients ---")
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

for a in [-2.0, -1.8, -1.75, -1.7, -1.65, -1.6, -1.5, -1.3]:
    b = 1.0 - a
    originals = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_compute_entry(a, b)
    strategy_orig_zt = strategy.ZSCORE_THRESHOLD
    strategy.ZSCORE_THRESHOLD = 2.96
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
        print(f"ZT=2.96+a={a:.2f}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals["compute_entry"]
        strategy.ZSCORE_THRESHOLD = strategy_orig_zt
