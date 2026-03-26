#!/usr/bin/env python3
"""Fresh sweep from 238.37 baseline (LP=0.092, ZT=3.0, SL=0.10, no clamp)."""
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
print(f"  SL={strategy.STOP_LOSS_PCT}")
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


# === DISCOUNT_FACTOR ===
print("--- DISCOUNT_FACTOR ---")
for df in [0.30, 0.32, 0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37, 0.38]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# === KELLY_FRACTION ===
print("--- KELLY_FRACTION ---")
for kf in [0.030, 0.031, 0.032, 0.033, 0.034, 0.035]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
# === ZSCORE_THRESHOLD fine-tune around 3.0 ===
print("--- ZSCORE_THRESHOLD ---")
for zt in [2.92, 2.94, 2.95, 2.96, 2.97, 2.98, 2.99, 3.0, 3.01, 3.02, 3.03, 3.05]:
    run_one(f"ZT={zt}", ZSCORE_THRESHOLD=zt)

print()
# === MIN_VOLUME ===
print("--- MIN_VOLUME ---")
for mv in [12000, 13000, 14000, 14500, 15000, 16000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
# === LONGSHOT_PRICE_MAX ===
print("--- LONGSHOT_PRICE_MAX ---")
for lp in [0.088, 0.090, 0.091, 0.092, 0.093, 0.094, 0.095]:
    run_one(f"LP={lp}", LONGSHOT_PRICE_MAX=lp)

print()
# === MIN_EDGE ===
print("--- MIN_EDGE ---")
for me in [0.02, 0.03, 0.04, 0.05, 0.055, 0.06]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# === STOP_LOSS_PCT ===
print("--- STOP_LOSS_PCT ---")
for sl in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    run_one(f"SL={sl}", STOP_LOSS_PCT=sl)

print()
# === SPIKE_LOOKBACK ===
print("--- SPIKE_LOOKBACK ---")
for slt in [15, 18, 19, 20, 21, 22, 25]:
    run_one(f"SLT={slt}", SPIKE_LOOKBACK_TICKS=slt)

print()
# === MIN_HISTORY ===
print("--- MIN_HISTORY ---")
for mht in [9, 10, 11, 12, 13]:
    run_one(f"MHT={mht}", MIN_HISTORY_TICKS=mht)

print()
# === RESOLUTION_DAYS_MAX ===
print("--- RES_DAYS_MAX ---")
for rdm in [18, 20, 21, 22, 24, 25]:
    run_one(f"RDM={rdm}", RESOLUTION_DAYS_MAX=rdm)

print()
# === Best combo: top winners ===
print("--- TOP COMBOS ---")
# Will fill based on above results... test common good combos
run_one("MV=14K+DF=0.345", MIN_VOLUME_24H=14000.0, DISCOUNT_FACTOR=0.345)
run_one("MV=14K+ME=0.05", MIN_VOLUME_24H=14000.0, MIN_EDGE=0.05)
run_one("MV=14K+KF=0.033", MIN_VOLUME_24H=14000.0, KELLY_FRACTION=0.033)
run_one("DF=0.345+ME=0.05", DISCOUNT_FACTOR=0.345, MIN_EDGE=0.05)
run_one("DF=0.345+KF=0.033", DISCOUNT_FACTOR=0.345, KELLY_FRACTION=0.033)
run_one("ME=0.05+KF=0.033", MIN_EDGE=0.05, KELLY_FRACTION=0.033)

print()
# Scale coefficient exploration
print("--- SCALE COEFFICIENTS ---")
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

for a in [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.0]:
    b = 1.0 - a
    originals = {"compute_entry": strategy.compute_entry}
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
        print(f"a={a:.1f}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals["compute_entry"]
