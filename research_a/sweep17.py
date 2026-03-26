#!/usr/bin/env python3
"""Sweep from 308.26 baseline (DF=0.37, a=-1.85+2.80, MV=13.5K, ME=0.055)."""
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
print(f"  DF={strategy.DISCOUNT_FACTOR}, KF={strategy.KELLY_FRACTION}, ME={strategy.MIN_EDGE}")
print(f"  LP={strategy.LONGSHOT_PRICE_MAX}, MV={strategy.MIN_VOLUME_24H}, ZT={strategy.ZSCORE_THRESHOLD}")
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


# Fine-tune DF further
print("--- DF fine-tune ---")
for df in [0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.40, 0.42, 0.45]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# KF sweep
print("--- KF ---")
for kf in [0.030, 0.031, 0.032, 0.033]:
    run_one(f"KF={kf}", KELLY_FRACTION=kf)

print()
# ME fine-tune
print("--- ME ---")
for me in [0.050, 0.054, 0.055, 0.056, 0.058, 0.060]:
    run_one(f"ME={me}", MIN_EDGE=me)

print()
# MV fine-tune
print("--- MV ---")
for mv in [12000, 13000, 13500, 14000, 15000]:
    run_one(f"MV={mv}", MIN_VOLUME_24H=float(mv))

print()
# Combos
print("--- TOP COMBOS ---")
run_one("DF=0.38+KF=0.031", DISCOUNT_FACTOR=0.38, KELLY_FRACTION=0.031)
run_one("DF=0.39+KF=0.031", DISCOUNT_FACTOR=0.39, KELLY_FRACTION=0.031)
run_one("DF=0.40+KF=0.031", DISCOUNT_FACTOR=0.40, KELLY_FRACTION=0.031)
run_one("DF=0.38+ME=0.054", DISCOUNT_FACTOR=0.38, MIN_EDGE=0.054)
run_one("DF=0.39+ME=0.054", DISCOUNT_FACTOR=0.39, MIN_EDGE=0.054)
run_one("KF=0.031+ME=0.054", KELLY_FRACTION=0.031, MIN_EDGE=0.054)
run_one("KF=0.031+ME=0.058", KELLY_FRACTION=0.031, MIN_EDGE=0.058)

print()
# Scale fine-tune
print("--- SCALE ---")
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

for a, b in [(-2.0, 3.0), (-1.9, 2.9), (-1.85, 2.85), (-1.85, 2.80), (-1.85, 2.75),
             (-1.80, 2.80), (-1.80, 2.75), (-1.80, 2.70),
             (-1.75, 2.75), (-1.75, 2.70), (-1.70, 2.70),
             (-1.90, 2.80), (-1.90, 2.85), (-1.95, 2.85)]:
    originals_f = {"compute_entry": strategy.compute_entry}
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
        print(f"a={a},b={b}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals_f["compute_entry"]
