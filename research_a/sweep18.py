#!/usr/bin/env python3
"""Fine sweep from 311.23 baseline (DF=0.375, a=-1.85+2.80, MV=13.5K)."""
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


# Ultra-fine DF
print("--- DF ultra-fine ---")
for df in [0.370, 0.372, 0.374, 0.375, 0.376, 0.378, 0.380]:
    run_one(f"DF={df}", DISCOUNT_FACTOR=df)

print()
# Scale with DF=0.375
print("--- SCALE with DF=0.375 ---")
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

for a, b in [(-1.85, 2.80), (-1.85, 2.78), (-1.85, 2.82),
             (-1.83, 2.78), (-1.83, 2.80), (-1.83, 2.75),
             (-1.80, 2.75), (-1.80, 2.78), (-1.80, 2.80),
             (-1.78, 2.75), (-1.78, 2.78),
             (-1.75, 2.70), (-1.75, 2.75),
             (-1.87, 2.80), (-1.87, 2.82)]:
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

print()
# 2D: scale + DF
print("--- Scale + DF 2D ---")
for a, b in [(-1.80, 2.75), (-1.80, 2.78), (-1.83, 2.78), (-1.85, 2.78)]:
    for df in [0.37, 0.375, 0.38, 0.385]:
        originals_f = {"compute_entry": strategy.compute_entry}
        strategy.compute_entry = make_compute_entry(a, b)
        orig_df = strategy.DISCOUNT_FACTOR
        strategy.DISCOUNT_FACTOR = df
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
            print(f"a={a},b={b},DF={df}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
                  f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
        finally:
            strategy.compute_entry = originals_f["compute_entry"]
            strategy.DISCOUNT_FACTOR = orig_df
