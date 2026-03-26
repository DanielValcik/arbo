#!/usr/bin/env python3
"""Sweep from 192.81 baseline (DF=0.35, KF=0.028, price-scale -2+3x)."""
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
print(f"Verified baseline: {BEST_SCORE:.6f} (sharpe={scores['avg_sharpe']:.4f})")
print(f"  DF={strategy.DISCOUNT_FACTOR}, KF={strategy.KELLY_FRACTION}")
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
        marker = "***BETTER***" if delta > 0.5 else ("~same" if abs(delta) < 0.5 else "")
        print(f"{label}: composite={cs:.6f} ({delta:+.2f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} {marker}")
        return cs
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


# DF sweep
print("--- DISCOUNT_FACTOR ---")
for df in [0.30, 0.32, 0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.38, 0.40]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
print("--- KELLY_FRACTION ---")
for kf in [0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.031, 0.032, 0.033]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
print("--- MIN_EDGE ---")
for me in [0.01, 0.015, 0.02, 0.025, 0.028, 0.03, 0.035, 0.04, 0.05]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
print("--- MIN_VOLUME ---")
for mv in [12000, 13000, 14000, 14500, 15000, 16000, 18000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
print("--- LONGSHOT_PRICE_MAX ---")
for lp in [0.070, 0.075, 0.080, 0.082, 0.085, 0.088, 0.090]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
print("--- BEST 2-PARAM COMBOS ---")
# DF + KF grid (fine)
for df in [0.34, 0.345, 0.35, 0.355, 0.36]:
    for kf in [0.027, 0.028, 0.029, 0.030]:
        run_one(f"DF={df}+KF={kf}", DISCOUNT_FACTOR=df, KELLY_FRACTION=kf)

print()
# Add MV=14000
for df in [0.34, 0.345, 0.35, 0.355, 0.36]:
    for kf in [0.028, 0.029, 0.030]:
        run_one(f"DF={df}+KF={kf}+MV=14000",
                DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_VOLUME_24H=14000.0)

print()
# Add ME=0.028
for df in [0.34, 0.345, 0.35, 0.355, 0.36]:
    run_one(f"DF={df}+ME=0.028", DISCOUNT_FACTOR=df, MIN_EDGE=0.028)

print()
# Triple: DF + KF + MV + ME
for df, kf in [(0.355, 0.028), (0.355, 0.029), (0.35, 0.029), (0.36, 0.028)]:
    for me in [0.02, 0.028, 0.03]:
        for mv in [14000, 15000]:
            run_one(f"DF={df}+KF={kf}+ME={me}+MV={mv}",
                    DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_EDGE=me, MIN_VOLUME_24H=float(mv))
