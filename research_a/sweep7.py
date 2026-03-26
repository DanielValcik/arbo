#!/usr/bin/env python3
"""Deep sweep from 203.47 baseline (scale -1.7+2.7)."""
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
print("=" * 100)

orig_compute_entry = strategy.compute_entry


def make_compute_entry(a, b):
    def compute_entry(price_yes, zscore):
        price_scale = a + b * (price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        effective_discount -= strategy.DISCOUNT_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        effective_discount = max(0.05, min(effective_discount, 0.95))
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        edge += strategy.EDGE_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        return edge, model_no_prob
    return compute_entry


def run_scale(label, a, b, **extra_overrides):
    originals = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_compute_entry(a, b)
    for k, v in extra_overrides.items():
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
        print(f"{label}: {cs:.6f} ({delta:+.3f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f} {marker}")
        return cs
    finally:
        strategy.compute_entry = originals["compute_entry"]
        for k in extra_overrides:
            setattr(strategy, k, originals[k])


# Fine grid around a=-1.7
print("--- Fine scale grid ---")
for a in [-1.80, -1.75, -1.72, -1.71, -1.70, -1.69, -1.68, -1.65, -1.60, -1.55, -1.50]:
    b = 1.0 - a
    run_scale(f"a={a:.2f}", a, b)

print()
# Now: scale + DF combos (fine)
print("--- scale=-1.7+2.7, DF combos ---")
for df in [0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.37, 0.38]:
    run_scale(f"DF={df}", -1.7, 2.7, DISCOUNT_FACTOR=df)

print()
# Best DF with different scales
print("--- scale + DF 2D grid ---")
best_cs = 0
best_params = ""
for a in [-1.75, -1.72, -1.70, -1.68, -1.65, -1.60]:
    b = 1.0 - a
    for df in [0.33, 0.34, 0.35, 0.36, 0.37, 0.38]:
        cs = run_scale(f"a={a:.2f},DF={df}", a, b, DISCOUNT_FACTOR=df)
        if cs > best_cs:
            best_cs = cs
            best_params = f"a={a:.2f},DF={df}"

print(f"\nBest: {best_params} = {best_cs:.6f}")

print()
# scale + MV
print("--- scale=-1.7, MV combos ---")
for mv in [13000, 14000, 14500, 15000, 16000]:
    run_scale(f"MV={mv}", -1.7, 2.7, MIN_VOLUME_24H=float(mv))

print()
# scale + LP
print("--- scale=-1.7, LP combos ---")
for lp in [0.075, 0.080, 0.085, 0.090]:
    run_scale(f"LP={lp}", -1.7, 2.7, LONGSHOT_PRICE_MAX=lp)

print()
# Nonlinear price scaling ideas
print("--- Nonlinear: power scaling ---")
def make_power_entry(power):
    def compute_entry(price_yes, zscore):
        ratio = price_yes / strategy.LONGSHOT_PRICE_MAX
        effective_discount = strategy.DISCOUNT_FACTOR * (ratio ** power)
        effective_discount -= strategy.DISCOUNT_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        effective_discount = max(0.05, min(effective_discount, 0.95))
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        edge += strategy.EDGE_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        return edge, model_no_prob
    return compute_entry

for power in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]:
    originals = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_power_entry(power)
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
        print(f"power={power}: {cs:.6f} ({delta:+.3f}) S={scores['avg_sharpe']:.1f} {marker}")
    finally:
        strategy.compute_entry = originals["compute_entry"]

print()
# Log scaling
print("--- Log scaling ---")
def make_log_entry():
    def compute_entry(price_yes, zscore):
        ratio = max(0.01, price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * max(0, 1 + math.log(ratio))
        effective_discount -= strategy.DISCOUNT_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        effective_discount = max(0.05, min(effective_discount, 0.95))
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        edge += strategy.EDGE_ZSCORE_BONUS * max(0, zscore - strategy.ZSCORE_THRESHOLD)
        return edge, model_no_prob
    return compute_entry

originals = {"compute_entry": strategy.compute_entry}
strategy.compute_entry = make_log_entry()
try:
    results = []
    for w in WINDOWS:
        seed = BASE_SEED + w["seed_offset"]
        strategy.reset_state()
        r = simulate_window(seed, w["n_days"], w["label"])
        results.append(r)
    scores = compute_composite_score(results)
    print(f"log: {scores['composite_score']:.6f} S={scores['avg_sharpe']:.1f}")
finally:
    strategy.compute_entry = originals["compute_entry"]
