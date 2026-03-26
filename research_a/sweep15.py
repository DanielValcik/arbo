#!/usr/bin/env python3
"""Sweep from ~284 baseline (a=-1.83, ME=0.055, MV=14K)."""
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
print(f"  LP={strategy.LONGSHOT_PRICE_MAX}, MV={strategy.MIN_VOLUME_24H}, DF={strategy.DISCOUNT_FACTOR}")
print(f"  KF={strategy.KELLY_FRACTION}, ME={strategy.MIN_EDGE}, ZT={strategy.ZSCORE_THRESHOLD}")
print(f"  SLT={strategy.SPIKE_LOOKBACK_TICKS}, MHT={strategy.MIN_HISTORY_TICKS}")
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


# === Key params sweep ===
print("--- DISCOUNT_FACTOR ---")
for df in [0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
print("--- MIN_EDGE ---")
for me in [0.050, 0.052, 0.054, 0.055, 0.056, 0.058, 0.060, 0.065]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
print("--- MIN_VOLUME ---")
for mv in [13000, 13500, 14000, 14500, 15000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
print("--- ZSCORE_THRESHOLD ---")
for zt in [2.96, 2.98, 3.0, 3.01, 3.02, 3.03]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# New structural ideas
print("--- ZSCORE_PENALTY_COEFF ---")
for zpc in [0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003]:
    run_one(f"ZPC={zpc}", ZSCORE_PENALTY_COEFF=zpc)

print()
print("--- ZSCORE_DAYS_PIVOT ---")
for zdp in [6, 7, 8, 9, 10, 11, 12]:
    run_one(f"ZDP={zdp}", ZSCORE_DAYS_PIVOT=zdp)

print()
# Edge-zscore bonus: this adds extra edge for stronger z-scores
print("--- EDGE_ZSCORE_BONUS ---")
for ezb in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
    run_one(f"EZB={ezb}", EDGE_ZSCORE_BONUS=ezb)

print()
# Discount-zscore bonus: reduce discount for stronger z-scores
print("--- DISCOUNT_ZSCORE_BONUS ---")
for dzb in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
    run_one(f"DZB={dzb}", DISCOUNT_ZSCORE_BONUS=dzb)

print()
# Try combinations of best winners
print("--- 2-WAY COMBOS ---")
# Test ZPC=0.001 with various
run_one("ZPC=0.001+ZDP=8", ZSCORE_PENALTY_COEFF=0.001, ZSCORE_DAYS_PIVOT=8)
run_one("ZPC=0.001+ZDP=9", ZSCORE_PENALTY_COEFF=0.001, ZSCORE_DAYS_PIVOT=9)
run_one("ZPC=0.001+DF=0.355", ZSCORE_PENALTY_COEFF=0.001, DISCOUNT_FACTOR=0.355)
run_one("ZPC=0.001+ME=0.058", ZSCORE_PENALTY_COEFF=0.001, MIN_EDGE=0.058)
run_one("ZPC=0.002+ZDP=8", ZSCORE_PENALTY_COEFF=0.002, ZSCORE_DAYS_PIVOT=8)
run_one("ZPC=0.002+ZDP=9", ZSCORE_PENALTY_COEFF=0.002, ZSCORE_DAYS_PIVOT=9)

print()
# RESOLUTION_DAYS_MIN
print("--- RESOLUTION_DAYS_MIN ---")
for rdmin in [1, 2, 3, 4]:
    run_one(f"RDmin={rdmin}", RESOLUTION_DAYS_MIN=rdmin)

print()
# MIN_YES_PRICE
print("--- MIN_YES_PRICE ---")
for myp in [0.005, 0.01, 0.015, 0.02]:
    run_one(f"MYP={myp}", MIN_YES_PRICE=myp)

print()
# MAX_EDGE
print("--- MAX_EDGE ---")
for mxe in [0.30, 0.40, 0.50, 0.60, 0.70, 1.0]:
    run_one(f"MXE={mxe}", MAX_EDGE=mxe)
