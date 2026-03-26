#!/usr/bin/env python3
"""Focused sweep with price-dependent discount as new baseline (96.57)."""
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
BEST_SCORE = scores["composite_score"]
print(f"Verified baseline: {BEST_SCORE:.6f}")
print(f"  sharpe={scores['avg_sharpe']:.4f}, trades={scores['num_trades']}, dd={scores['max_drawdown_pct']:.2f}")
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
        marker = "***BETTER***" if delta > 0.001 else ""
        print(f"{label}: composite={cs:.6f} (delta={delta:+.4f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} wr={scores['avg_win_rate']:.1f} {marker}")
        return cs
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


# DISCOUNT_FACTOR sweep (with the price-scale model)
print("--- DISCOUNT_FACTOR (with price-scale model) ---")
for df in [0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35, 0.38, 0.40, 0.45, 0.50]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
print("--- KELLY_FRACTION ---")
for kf in [0.028, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.038, 0.040, 0.042, 0.045]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
print("--- MIN_EDGE ---")
for me in [0.01, 0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030, 0.035]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
print("--- LONGSHOT_PRICE_MAX ---")
for lp in [0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
print("--- ZSCORE_THRESHOLD ---")
for zt in [2.7, 2.8, 2.85, 2.88, 2.90, 2.92, 2.95, 3.0]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
print("--- MIN_VOLUME_24H ---")
for mv in [8000, 10000, 12000, 14000, 15000, 16000, 18000, 20000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
print("--- RESOLUTION_DAYS ---")
for rdmax in [15, 18, 20, 21, 22, 25, 28]:
    run_one(f"RD_MAX={rdmax}", RESOLUTION_DAYS_MAX=rdmax)

for rdmin in [1, 2, 3, 4]:
    run_one(f"RD_MIN={rdmin}", RESOLUTION_DAYS_MIN=rdmin)

print()
print("--- LOOKBACK / HISTORY ---")
for lb in [15, 17, 19, 20, 21, 23, 25]:
    run_one(f"LB={lb}", SPIKE_LOOKBACK_TICKS=lb)

for mh in [8, 9, 10, 11, 12, 13, 14]:
    run_one(f"MH={mh}", MIN_HISTORY_TICKS=mh)

print()
print("--- POSITION_PCT_MIN ---")
for pm in [0.010, 0.015, 0.018, 0.020, 0.022, 0.025]:
    run_one(f"PCT_MIN={pm}", POSITION_PCT_MIN=pm)

print()
print("--- BEST COMBOS ---")
# DF + KF
for df, kf in [(0.31, 0.032), (0.32, 0.032), (0.33, 0.032), (0.31, 0.033),
                (0.31, 0.034), (0.32, 0.033), (0.35, 0.032), (0.35, 0.033)]:
    run_one(f"DF={df}+KF={kf}", DISCOUNT_FACTOR=df, KELLY_FRACTION=kf)

# DF + MV
for df, mv in [(0.31, 14000), (0.32, 14000), (0.30, 14000), (0.31, 16000),
                (0.30, 16000), (0.30, 12000)]:
    run_one(f"DF={df}+MV={mv}", DISCOUNT_FACTOR=df, MIN_VOLUME_24H=float(mv))

# DF + LP
for df, lp in [(0.31, 0.090), (0.31, 0.095), (0.31, 0.100), (0.30, 0.090)]:
    run_one(f"DF={df}+LP={lp}", DISCOUNT_FACTOR=df, LONGSHOT_PRICE_MAX=lp)

# Triple combos
for df, kf, mv in [(0.31, 0.033, 14000), (0.32, 0.033, 14000), (0.30, 0.033, 14000)]:
    run_one(f"DF={df}+KF={kf}+MV={mv}",
            DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_VOLUME_24H=float(mv))
