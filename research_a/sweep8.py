#!/usr/bin/env python3
"""Sweep from 217.16 baseline (LP=0.09, MV=12K, scale -1.7+2.7)."""
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
print(f"  SL={strategy.STOP_LOSS_PCT}, PP={strategy.PARTIAL_EXIT_PROFIT_PCT}")
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


# === DISCOUNT_FACTOR sweep ===
print("--- DISCOUNT_FACTOR ---")
for df in [0.30, 0.31, 0.32, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37, 0.38, 0.40]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# === KELLY_FRACTION sweep ===
print("--- KELLY_FRACTION ---")
for kf in [0.028, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.038, 0.040]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
# === MIN_VOLUME sweep ===
print("--- MIN_VOLUME_24H ---")
for mv in [8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 18000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
# === LONGSHOT_PRICE_MAX sweep ===
print("--- LONGSHOT_PRICE_MAX ---")
for lp in [0.080, 0.085, 0.088, 0.090, 0.092, 0.095, 0.10, 0.11, 0.12]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
# === MIN_EDGE sweep ===
print("--- MIN_EDGE ---")
for me in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# === ZSCORE_THRESHOLD sweep ===
print("--- ZSCORE_THRESHOLD ---")
for zt in [2.5, 2.6, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.1, 3.2]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# === STOP_LOSS_PCT sweep ===
print("--- STOP_LOSS_PCT ---")
for sl in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50, 1.0]:
    run_one(f"SL={sl}", STOP_LOSS_PCT=sl)

print()
# === PARTIAL_EXIT_PROFIT_PCT sweep ===
print("--- PARTIAL_EXIT_PROFIT_PCT ---")
for pp in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]:
    run_one(f"PP={pp}", PARTIAL_EXIT_PROFIT_PCT=pp)

print()
# === POSITION_PCT_MIN sweep ===
print("--- POSITION_PCT_MIN ---")
for pm in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
    run_one(f"PM={pm}", POSITION_PCT_MIN=pm)

print()
# === POSITION_PCT_MAX sweep ===
print("--- POSITION_PCT_MAX ---")
for px in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15]:
    run_one(f"PX={px}", POSITION_PCT_MAX=px)

print()
# === MAX_CONCURRENT sweep ===
print("--- MAX_CONCURRENT ---")
for mc in [10, 15, 20, 25, 30, 40, 50]:
    run_one(f"MC={mc}", MAX_CONCURRENT=mc)

print()
# === RESOLUTION_DAYS range ===
print("--- RESOLUTION_DAYS_MAX ---")
for rd in [14, 18, 21, 25, 28, 30, 35, 45]:
    run_one(f"RDmax={rd}", RESOLUTION_DAYS_MAX=rd)

print()
# === SPIKE_LOOKBACK_TICKS ===
print("--- SPIKE_LOOKBACK_TICKS ---")
for sl in [12, 15, 18, 20, 22, 25, 30]:
    run_one(f"SLT={sl}", SPIKE_LOOKBACK_TICKS=sl)

print()
# === MIN_HISTORY_TICKS ===
print("--- MIN_HISTORY_TICKS ---")
for mh in [6, 8, 10, 11, 12, 14, 16]:
    run_one(f"MHT={mh}", MIN_HISTORY_TICKS=mh)

print()
# === MAX_CAPITAL_DEPLOYED_PCT ===
print("--- MAX_CAPITAL_DEPLOYED_PCT ---")
for md in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]:
    run_one(f"MCD={md}", MAX_CAPITAL_DEPLOYED_PCT=md)
