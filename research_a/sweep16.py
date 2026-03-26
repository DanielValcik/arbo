#!/usr/bin/env python3
"""Quick sweep from current state to find improvements."""
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
print(f"  LP={strategy.LONGSHOT_PRICE_MAX}, MV={strategy.MIN_VOLUME_24H}, DF={strategy.DISCOUNT_FACTOR}")
print(f"  KF={strategy.KELLY_FRACTION}, ME={strategy.MIN_EDGE}, ZT={strategy.ZSCORE_THRESHOLD}")
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


# Quick test critical params
print("--- MIN_VOLUME ---")
for mv in [13000, 13500, 14000, 14500, 15000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
print("--- DISCOUNT_FACTOR ---")
for df in [0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
print("--- MIN_EDGE ---")
for me in [0.05, 0.054, 0.055, 0.056, 0.058, 0.06]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
print("--- ZSCORE_THRESHOLD ---")
for zt in [2.96, 2.98, 3.0, 3.01, 3.02]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
print("--- KELLY_FRACTION ---")
for kf in [0.031, 0.032, 0.033]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
print("--- TOP COMBOS ---")
run_one("MV=14K+DF=0.34", MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.34)
run_one("MV=14K+ME=0.054", MIN_VOLUME_24H=14000.0, MIN_EDGE=0.054)
run_one("DF=0.34+ME=0.054", DISCOUNT_FACTOR=0.34, MIN_EDGE=0.054)
run_one("DF=0.34+MV=14K+ME=0.054", DISCOUNT_FACTOR=0.34, MIN_VOLUME_24H=14000.0, MIN_EDGE=0.054)
