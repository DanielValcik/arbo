#!/usr/bin/env python3
"""Structural experiments from 315.05 baseline."""
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


def run_structural(label, compute_entry_fn=None, should_trade_fn=None, position_size_fn=None, check_exit_fn=None, **param_overrides):
    """Test structural changes."""
    originals = {}
    if compute_entry_fn:
        originals["compute_entry"] = strategy.compute_entry
        strategy.compute_entry = compute_entry_fn
    if should_trade_fn:
        originals["should_trade"] = strategy.should_trade
        strategy.should_trade = should_trade_fn
    if position_size_fn:
        originals["position_size"] = strategy.position_size
        strategy.position_size = position_size_fn
    if check_exit_fn:
        originals["check_exit"] = strategy.check_exit
        strategy.check_exit = check_exit_fn
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
            if k in ("compute_entry", "should_trade", "position_size", "check_exit"):
                setattr(strategy, k, v)
            else:
                setattr(strategy, k, v)


# 1. Zscore-dependent discount (more zscore = more confident in NO = lower discount)
print("=== Zscore-dependent discount ===")
for factor in [0.005, 0.01, 0.015, 0.02]:
    def make_entry(f):
        def compute_entry(price_yes, zscore):
            price_scale = -1.80 + 2.75 * (price_yes / strategy.LONGSHOT_PRICE_MAX)
            # Reduce discount for stronger z-scores (more confident in spike = more edge)
            zscore_adj = f * max(0, zscore - strategy.ZSCORE_THRESHOLD)
            effective_discount = (strategy.DISCOUNT_FACTOR - zscore_adj) * price_scale
            model_yes_prob = price_yes * effective_discount
            model_no_prob = 1.0 - model_yes_prob
            no_price = 1.0 - price_yes
            edge = model_no_prob - no_price
            return edge, model_no_prob
        return compute_entry
    run_structural(f"ZsDF={factor}", compute_entry_fn=make_entry(factor))

print()
# 2. Volume-weighted edge (higher volume = higher confidence)
print("=== Volume edge bonus ===")
for enabled, bonus in [(True, 0.005), (True, 0.01), (True, -0.005)]:
    run_structural(f"VEB={bonus}", VOLUME_EDGE_ENABLED=enabled, VOLUME_EDGE_REDUCTION=bonus)

print()
# 3. Short-date bonus (lower z-score threshold for near-resolution)
print("=== Short-date bonus ===")
for sdb in [0.01, 0.02, 0.03, 0.05, 0.1]:
    run_structural(f"SDB={sdb}", ZSCORE_SHORT_DATE_BONUS=sdb)

print()
# 4. Resolution-time sizing (bigger positions closer to resolution)
print("=== Resolution sizing ===")
for enabled, bonus in [(True, 0.005), (True, 0.01), (True, 0.002)]:
    run_structural(f"RSB={bonus}", RESOLUTION_SIZE_ENABLED=enabled, RESOLUTION_SIZE_BONUS_PCT=bonus)

print()
# 5. Trailing stop
print("=== Trailing stop ===")
for pct in [0.05, 0.10, 0.15, 0.20]:
    for act in [0.03, 0.05, 0.10]:
        run_structural(f"TS={pct},A={act}", TRAILING_STOP_ENABLED=True, TRAILING_STOP_PCT=pct, TRAILING_STOP_ACTIVATION=act)

print()
# 6. Time exit
print("=== Time exit ===")
for days in [0.5, 1, 2]:
    run_structural(f"TE={days}", TIME_EXIT_ENABLED=True, TIME_EXIT_DAYS_BEFORE=days)

print()
# 7. Different spike cooldown
print("=== Spike cooldown ===")
for cd in [0, 1, 2, 3, 5]:
    run_structural(f"CD={cd}", SPIKE_COOLDOWN_TICKS=cd)

print()
# 8. Quadratic price scale
print("=== Nonlinear price scale ===")
def make_power_entry(power):
    def compute_entry(price_yes, zscore):
        ratio = price_yes / strategy.LONGSHOT_PRICE_MAX
        # Interpolate: at ratio=0 → scale=-1.80, at ratio=1 → scale=0.95 (=-1.80+2.75)
        effective_discount = strategy.DISCOUNT_FACTOR * (-1.80 + 2.75 * (ratio ** power))
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

for power in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    run_structural(f"power={power}", compute_entry_fn=make_power_entry(power))

print()
# 9. Sqrt-scaled edge
print("=== Edge transformation ===")
def make_sqrt_edge():
    def compute_entry(price_yes, zscore):
        price_scale = -1.80 + 2.75 * (price_yes / strategy.LONGSHOT_PRICE_MAX)
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        # Sqrt compression: reduces large edges, preserves small ones
        if edge > 0:
            edge = math.sqrt(edge) * 0.15
        return edge, model_no_prob
    return compute_entry

run_structural("sqrt_edge", compute_entry_fn=make_sqrt_edge())

print()
# 10. Kelly with edge-adaptive fraction
print("=== Edge-adaptive sizing ===")
def make_edge_sizing():
    def position_size(edge, no_price, available_capital, total_capital):
        if no_price <= 0 or no_price >= 1 or edge <= 0 or total_capital <= 0:
            return 0.0
        odds_minus_1 = (1.0 / no_price) - 1.0
        if odds_minus_1 <= 0:
            return 0.0
        kelly_raw = edge / odds_minus_1
        kelly_raw = max(0.0, min(kelly_raw, strategy.KELLY_CAP))
        # Adaptive: higher edge = slightly higher kelly fraction
        adaptive_kf = strategy.KELLY_FRACTION * (1.0 + 0.5 * min(edge, 0.10))
        kelly_adjusted = kelly_raw * adaptive_kf * strategy.KELLY_MULTIPLIER
        size = available_capital * kelly_adjusted
        min_size = total_capital * strategy.POSITION_PCT_MIN
        max_size = total_capital * strategy.POSITION_PCT_MAX
        size = max(min_size, min(max_size, size))
        size = min(size, available_capital * 0.95)
        if size < min_size:
            return 0.0
        return round(size, 2)
    return position_size

run_structural("edge_adaptive_sizing", position_size_fn=make_edge_sizing())
