#!/usr/bin/env python3
"""Sweep23: MHT=9 breakthrough — explore combos from 326.59 baseline.

MHT=9 scores 326.59 (+11.54 over 315.05).
Now combine MHT=9 with other parameter variations.
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy
from backtest_a_harness import WINDOWS, BASE_SEED, simulate_window, compute_composite_score

# Set MHT=9 as new baseline
strategy.MIN_HISTORY_TICKS = 9

results = []
for w in WINDOWS:
    seed = BASE_SEED + w["seed_offset"]
    strategy.reset_state()
    r = simulate_window(seed, w["n_days"], w["label"])
    results.append(r)
scores = compute_composite_score(results)
BEST_SCORE = scores["composite_score"]
print(f"MHT=9 baseline: {BEST_SCORE:.6f}")
print(f"  S={scores['avg_sharpe']:.1f} T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% WR={scores['avg_win_rate']:.1f}%")
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


def run_structural(label, compute_entry_fn=None, **param_overrides):
    originals = {}
    if compute_entry_fn:
        originals["compute_entry"] = strategy.compute_entry
        strategy.compute_entry = compute_entry_fn
    for k, v in param_overrides.items():
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
            if k == "compute_entry":
                setattr(strategy, k, v)
            else:
                setattr(strategy, k, v)


# === 1. Fine-tune MHT itself ===
print("=== MHT fine-tune (from MHT=9 baseline) ===")
for mht in [7, 8, 9, 10]:
    run_one(f"MHT={mht}", MIN_HISTORY_TICKS=mht)

print()
# === 2. DF sweep from MHT=9 ===
print("=== DF with MHT=9 ===")
for df in [0.370, 0.372, 0.374, 0.375, 0.376, 0.378, 0.380, 0.385]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# === 3. KF sweep from MHT=9 ===
print("=== KF with MHT=9 ===")
for kf in [0.030, 0.031, 0.0315, 0.032, 0.0325, 0.033, 0.034]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
# === 4. ME sweep from MHT=9 ===
print("=== ME with MHT=9 ===")
for me in [0.050, 0.052, 0.054, 0.055, 0.056, 0.058, 0.060]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# === 5. Scale sweep from MHT=9 ===
print("=== Scale with MHT=9 ===")
def make_entry(a, b):
    def compute_entry(price_yes, zscore):
        price_scale = a + b * (price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

for a, b in [(-1.82, 2.77), (-1.81, 2.76), (-1.80, 2.75),
             (-1.795, 2.745), (-1.79, 2.74), (-1.78, 2.73),
             (-1.83, 2.78), (-1.85, 2.80)]:
    run_structural(f"a={a},b={b}", compute_entry_fn=make_entry(a, b))

print()
# === 6. LP sweep from MHT=9 ===
print("=== LP with MHT=9 ===")
for lp in [0.090, 0.091, 0.092, 0.093, 0.094]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
# === 7. MV sweep from MHT=9 ===
print("=== MV with MHT=9 ===")
for mv in [12000, 13000, 13500, 14000, 15000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
# === 8. ZT sweep from MHT=9 ===
print("=== ZT with MHT=9 ===")
for zt in [2.90, 2.95, 2.98, 3.0, 3.02, 3.05, 3.10]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# === 9. SLT fine-tune from MHT=9 ===
print("=== SLT with MHT=9 ===")
for slt in [18, 19, 20, 21, 22]:
    run_one(f"SLT={slt}", SPIKE_LOOKBACK_TICKS=slt)

print()
# === 10. Position sizing from MHT=9 ===
print("=== Position sizing with MHT=9 ===")
for pmin, pmax in [(0.02, 0.04), (0.02, 0.05), (0.02, 0.06), (0.02, 0.07)]:
    run_one(f"PSZ={pmin}-{pmax}", POSITION_PCT_MIN=pmin, POSITION_PCT_MAX=pmax)

print()
# === 11. Resolution days from MHT=9 ===
print("=== Resolution days with MHT=9 ===")
for rdmin, rdmax in [(1, 21), (2, 18), (2, 21), (2, 24), (3, 21)]:
    run_one(f"RD={rdmin}-{rdmax}", RESOLUTION_DAYS_MIN=rdmin, RESOLUTION_DAYS_MAX=rdmax)

print()
# === 12. Top combos from MHT=9 ===
print("=== TOP COMBOS from MHT=9 ===")
# Try top single-param winners together
for df in [0.375, 0.376]:
    for kf in [0.032, 0.0325]:
        for me in [0.055, 0.056]:
            if df == 0.375 and kf == 0.032 and me == 0.055:
                continue  # Skip baseline
            run_one(f"DF={df}+KF={kf}+ME={me}",
                   DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_EDGE=me)

# SL adjustments
print()
print("=== Stop loss with MHT=9 ===")
for sl in [0.15, 0.20, 0.25, 0.30]:
    run_one(f"SL={sl}", STOP_LOSS_PCT=sl)

print()
# === 13. ZSCORE penalty configs from MHT=9 ===
print("=== Zscore penalty from MHT=9 ===")
for pivot, power, coeff in [(10, 2, 0.0005), (10, 2, 0.001), (8, 2, 0.0005),
                              (10, 2, 0.0003), (12, 2, 0.0005), (10, 2, 0.0)]:
    run_one(f"ZP={pivot}+pw={power}+c={coeff}",
           ZSCORE_DAYS_PIVOT=pivot, ZSCORE_PENALTY_POWER=power, ZSCORE_PENALTY_COEFF=coeff)
