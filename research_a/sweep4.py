#!/usr/bin/env python3
"""Deep sweep around DF=0.35+KF=0.033 (116.25)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy
from backtest_a_harness import WINDOWS, BASE_SEED, simulate_window, compute_composite_score

# First verify the baseline
results = []
for w in WINDOWS:
    seed = BASE_SEED + w["seed_offset"]
    strategy.reset_state()
    r = simulate_window(seed, w["n_days"], w["label"])
    results.append(r)
scores = compute_composite_score(results)
print(f"Current file baseline: {scores['composite_score']:.6f}")
print("=" * 100)

BEST_SCORE = 116.251386


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
        marker = "***BETTER***" if delta > 0.01 else ("=SAME=" if abs(delta) < 0.01 else "")
        print(f"{label}: composite={cs:.6f} (delta={delta:+.4f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} wr={scores['avg_win_rate']:.1f} {marker}")
        return cs
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


# Fine-tune DF around 0.35 with KF=0.033
print("--- DF sweep with KF=0.033 ---")
for df in [0.30, 0.32, 0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37, 0.38, 0.40, 0.42, 0.45]:
    run_one(f"DF={df}+KF=0.033", DISCOUNT_FACTOR=df, KELLY_FRACTION=0.033)

print()
# Fine-tune KF with DF=0.35
print("--- KF sweep with DF=0.35 ---")
for kf in [0.030, 0.031, 0.032, 0.0325, 0.033, 0.0335, 0.034, 0.035, 0.036, 0.037, 0.038, 0.040]:
    run_one(f"DF=0.35+KF={kf}", DISCOUNT_FACTOR=0.35, KELLY_FRACTION=kf)

print()
# 2D grid around DF=0.35, KF=0.033
print("--- 2D grid DF x KF ---")
for df in [0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37]:
    for kf in [0.032, 0.0325, 0.033, 0.0335, 0.034]:
        run_one(f"DF={df}+KF={kf}", DISCOUNT_FACTOR=df, KELLY_FRACTION=kf)

print()
# Best DF+KF + MIN_EDGE
print("--- DF=0.35+KF=0.033 + MIN_EDGE ---")
for me in [0.02, 0.025, 0.028, 0.030, 0.032, 0.035, 0.04]:
    run_one(f"DF=0.35+KF=0.033+ME={me}", DISCOUNT_FACTOR=0.35, KELLY_FRACTION=0.033, MIN_EDGE=me)

print()
# Best DF+KF + MIN_VOLUME
print("--- DF=0.35+KF=0.033 + MIN_VOL ---")
for mv in [12000, 13000, 14000, 15000, 16000, 18000, 20000]:
    run_one(f"DF=0.35+KF=0.033+MV={mv}", DISCOUNT_FACTOR=0.35, KELLY_FRACTION=0.033, MIN_VOLUME_24H=float(mv))

print()
# Best DF+KF + LONGSHOT
print("--- DF=0.35+KF=0.033 + LONGSHOT ---")
for lp in [0.075, 0.080, 0.085, 0.090, 0.095, 0.10]:
    run_one(f"DF=0.35+KF=0.033+LP={lp}", DISCOUNT_FACTOR=0.35, KELLY_FRACTION=0.033, LONGSHOT_PRICE_MAX=lp)

print()
# Best DF+KF + ZSCORE
print("--- DF=0.35+KF=0.033 + ZSCORE ---")
for zt in [2.85, 2.88, 2.90, 2.92, 2.95]:
    run_one(f"DF=0.35+KF=0.033+ZT={zt}", DISCOUNT_FACTOR=0.35, KELLY_FRACTION=0.033, ZSCORE_THRESHOLD=zt)

print()
# Quad combos
print("--- QUAD COMBOS ---")
for df, kf, me, mv in [
    (0.35, 0.033, 0.028, 14000),
    (0.35, 0.033, 0.03, 14000),
    (0.35, 0.033, 0.028, 15000),
    (0.35, 0.033, 0.03, 16000),
    (0.36, 0.033, 0.028, 14000),
    (0.34, 0.033, 0.028, 14000),
    (0.35, 0.034, 0.028, 14000),
    (0.35, 0.032, 0.028, 14000),
    (0.36, 0.034, 0.028, 14000),
    (0.36, 0.033, 0.03, 14000),
]:
    run_one(f"DF={df}+KF={kf}+ME={me}+MV={mv}",
            DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_EDGE=me, MIN_VOLUME_24H=float(mv))
