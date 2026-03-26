#!/usr/bin/env python3
"""Robust sweep: test all changes against BOTH KF=0.032 and KF=0.033 to handle race conditions."""
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
print(f"Verified baseline: {BEST_SCORE:.6f} (KF={strategy.KELLY_FRACTION})")
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
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
        return cs
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


# Test top candidates with BOTH KF=0.032 and KF=0.033 to be safe
print("=== ME=0.055 with both KF values ===")
run_one("ME=0.055+KF=0.032", MIN_EDGE=0.055, KELLY_FRACTION=0.032)
run_one("ME=0.055+KF=0.033", MIN_EDGE=0.055, KELLY_FRACTION=0.033)

print()
print("=== ME=0.05 with both KF values ===")
run_one("ME=0.05+KF=0.032", MIN_EDGE=0.05, KELLY_FRACTION=0.032)
run_one("ME=0.05+KF=0.033", MIN_EDGE=0.05, KELLY_FRACTION=0.033)

print()
print("=== MV=14K with both KF values ===")
run_one("MV=14K+KF=0.032", MIN_VOLUME_24H=14000.0, KELLY_FRACTION=0.032)
run_one("MV=14K+KF=0.033", MIN_VOLUME_24H=14000.0, KELLY_FRACTION=0.033)

print()
print("=== scale a=-1.9 with both KF values ===")
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

for kf in [0.032, 0.033]:
    originals = {"compute_entry": strategy.compute_entry, "KELLY_FRACTION": strategy.KELLY_FRACTION}
    strategy.compute_entry = make_compute_entry(-1.9, 2.9)
    strategy.KELLY_FRACTION = kf
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
        print(f"a=-1.9+KF={kf}: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals["compute_entry"]
        strategy.KELLY_FRACTION = originals["KELLY_FRACTION"]

print()
# Test combinations
print("=== Multi-param combos (with both KF) ===")
for kf in [0.032, 0.033]:
    run_one(f"MV=14K+ME=0.055+KF={kf}", MIN_VOLUME_24H=14000.0, MIN_EDGE=0.055, KELLY_FRACTION=kf)
    run_one(f"MV=14K+ME=0.05+KF={kf}", MIN_VOLUME_24H=14000.0, MIN_EDGE=0.05, KELLY_FRACTION=kf)

print()
# DF sweep around 0.35 with both KF
print("=== DF sweep with KF=0.032 ===")
for df in [0.345, 0.35, 0.355, 0.36]:
    run_one(f"DF={df}+KF=0.032", DISCOUNT_FACTOR=df, KELLY_FRACTION=0.032)

print("=== DF sweep with KF=0.033 ===")
for df in [0.345, 0.35, 0.355, 0.36]:
    run_one(f"DF={df}+KF=0.033", DISCOUNT_FACTOR=df, KELLY_FRACTION=0.033)

print()
# ME fine-tune with KF=0.032 (main target)
print("=== ME fine-tune with KF=0.032 ===")
for me in [0.048, 0.050, 0.052, 0.054, 0.055, 0.056, 0.058, 0.060]:
    run_one(f"ME={me}+KF=0.032", MIN_EDGE=me, KELLY_FRACTION=0.032)

print()
# ME fine-tune with KF=0.033 (in case other process keeps it)
print("=== ME fine-tune with KF=0.033 ===")
for me in [0.048, 0.050, 0.052, 0.054, 0.055, 0.056, 0.058, 0.060]:
    run_one(f"ME={me}+KF=0.033", MIN_EDGE=me, KELLY_FRACTION=0.033)

print()
# New idea: scale + ME combos
print("=== scale a=-1.9 + ME combos (KF=0.032) ===")
for me in [0.02, 0.04, 0.05, 0.055, 0.06]:
    originals = {"compute_entry": strategy.compute_entry}
    strategy.compute_entry = make_compute_entry(-1.9, 2.9)
    orig_me = strategy.MIN_EDGE
    orig_kf = strategy.KELLY_FRACTION
    strategy.MIN_EDGE = me
    strategy.KELLY_FRACTION = 0.032
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
        print(f"a=-1.9+ME={me}+KF=0.032: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
              f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
    finally:
        strategy.compute_entry = originals["compute_entry"]
        strategy.MIN_EDGE = orig_me
        strategy.KELLY_FRACTION = orig_kf

print()
# Triple combo: a=-1.9 + ME=0.055 + MV=14K
print("=== Triple: a=-1.9 + ME=0.055 + MV=14K (KF=0.032) ===")
originals = {"compute_entry": strategy.compute_entry}
strategy.compute_entry = make_compute_entry(-1.9, 2.9)
orig_vals = {k: getattr(strategy, k) for k in ["MIN_EDGE", "MIN_VOLUME_24H", "KELLY_FRACTION"]}
strategy.MIN_EDGE = 0.055
strategy.MIN_VOLUME_24H = 14000.0
strategy.KELLY_FRACTION = 0.032
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
    print(f"a=-1.9+ME=0.055+MV=14K+KF=0.032: {cs:.2f} ({delta:+.2f}) S={scores['avg_sharpe']:.1f} "
          f"T={scores['num_trades']} DD={scores['max_drawdown_pct']:.2f}% {marker}")
finally:
    strategy.compute_entry = originals["compute_entry"]
    for k, v in orig_vals.items():
        setattr(strategy, k, v)
