#!/usr/bin/env python3
"""Combo sweep from 214.73 baseline. Top singles: ZT=2.95(+27), ME=0.05(+21), LP=0.092(+7), MV=14K(+4), DF=0.345(+1)."""
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


# === Top 2 combos ===
print("--- TOP 2-WAY COMBOS ---")
# ZT=2.95 + ME=0.05
run_one("ZT=2.95+ME=0.05", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05)
# ZT=2.95 + LP=0.092
run_one("ZT=2.95+LP=0.092", ZSCORE_THRESHOLD=2.95, LONGSHOT_PRICE_MAX=0.092)
# ZT=2.95 + MV=14K
run_one("ZT=2.95+MV=14K", ZSCORE_THRESHOLD=2.95, MIN_VOLUME_24H=14000.0)
# ZT=2.95 + DF=0.345
run_one("ZT=2.95+DF=0.345", ZSCORE_THRESHOLD=2.95, DISCOUNT_FACTOR=0.345)
# ZT=2.95 + KF=0.033
run_one("ZT=2.95+KF=0.033", ZSCORE_THRESHOLD=2.95, KELLY_FRACTION=0.033)
# ME=0.05 + LP=0.092
run_one("ME=0.05+LP=0.092", MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092)
# ME=0.05 + MV=14K
run_one("ME=0.05+MV=14K", MIN_EDGE=0.05, MIN_VOLUME_24H=14000.0)
# ME=0.05 + DF=0.345
run_one("ME=0.05+DF=0.345", MIN_EDGE=0.05, DISCOUNT_FACTOR=0.345)
# LP=0.092 + MV=14K
run_one("LP=0.092+MV=14K", LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0)

print()
# === Top 3 combos ===
print("--- TOP 3-WAY COMBOS ---")
run_one("ZT=2.95+ME=0.05+LP=0.092", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092)
run_one("ZT=2.95+ME=0.05+MV=14K", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, MIN_VOLUME_24H=14000.0)
run_one("ZT=2.95+ME=0.05+DF=0.345", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.95+LP=0.092+MV=14K", ZSCORE_THRESHOLD=2.95, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0)
run_one("ZT=2.95+LP=0.092+DF=0.345", ZSCORE_THRESHOLD=2.95, LONGSHOT_PRICE_MAX=0.092, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.95+MV=14K+DF=0.345", ZSCORE_THRESHOLD=2.95, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)
run_one("ME=0.05+LP=0.092+MV=14K", MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0)
run_one("ME=0.05+LP=0.092+DF=0.345", MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, DISCOUNT_FACTOR=0.345)

print()
# === Top 4 combos ===
print("--- TOP 4-WAY COMBOS ---")
run_one("ZT=2.95+ME=0.05+LP=0.092+MV=14K", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0)
run_one("ZT=2.95+ME=0.05+LP=0.092+DF=0.345", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.95+ME=0.05+MV=14K+DF=0.345", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.95+LP=0.092+MV=14K+DF=0.345", ZSCORE_THRESHOLD=2.95, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)

print()
# === Top 5 combo ===
print("--- TOP 5-WAY COMBO ---")
run_one("ZT=2.95+ME=0.05+LP=0.092+MV=14K+DF=0.345", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)
run_one("ZT=2.95+ME=0.05+LP=0.092+MV=14K+KF=0.033", ZSCORE_THRESHOLD=2.95, MIN_EDGE=0.05, LONGSHOT_PRICE_MAX=0.092, MIN_VOLUME_24H=14000.0, KELLY_FRACTION=0.033)

print()
# === Fine-tune ZT around 2.95 ===
print("--- ZT fine-tune ---")
for zt in [2.92, 2.93, 2.94, 2.95, 2.96, 2.97, 2.98]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# === Fine-tune ME around 0.05 ===
print("--- ME fine-tune ---")
for me in [0.04, 0.045, 0.048, 0.05, 0.052, 0.055, 0.06, 0.07, 0.08]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# === Fine-tune LP around 0.092 ===
print("--- LP fine-tune ---")
for lp in [0.091, 0.092, 0.093, 0.094]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)
