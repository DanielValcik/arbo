#!/usr/bin/env python3
"""Sweep22: Fundamentally different entry models.

The current model uses: price_scale = -1.80 + 2.75 * (price_yes / LP)
This is a linear function of normalized price. What if we use completely
different model structures?

Also: LP=0.092 is critical because it determines the normalization.
The scale function equals 0.95 at LP boundary. What if LP is just
selecting the RIGHT price range and the scale is the real driver?

Let's try:
1. Two-segment piecewise entry model
2. Exponential discount model
3. Probability-based entry (Bayesian-like)
4. Edge = constant margin (fixed edge regardless of price)
5. LP=0.092 but with different a,b sweeps (re-center around optimum)
6. Ultra-fine KF + DF 2D grid
"""
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


def run_structural(label, compute_entry_fn=None, should_trade_fn=None, **param_overrides):
    originals = {}
    if compute_entry_fn:
        originals["compute_entry"] = strategy.compute_entry
        strategy.compute_entry = compute_entry_fn
    if should_trade_fn:
        originals["should_trade"] = strategy.should_trade
        strategy.should_trade = should_trade_fn
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
            if k in ("compute_entry", "should_trade"):
                setattr(strategy, k, v)
            else:
                setattr(strategy, k, v)


# === 1. Ultra-fine a,b sweep centered on optimum ===
print("=== Ultra-fine a,b near (-1.80, 2.75) ===")
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

# Fine grid around -1.80, 2.75
for a in [-1.82, -1.81, -1.805, -1.800, -1.795, -1.79, -1.78]:
    for b in [2.73, 2.74, 2.745, 2.750, 2.755, 2.76, 2.77]:
        run_structural(f"a={a},b={b}", compute_entry_fn=make_entry(a, b))

print()
# === 2. Ultra-fine DF × KF 2D grid ===
print("=== DF × KF ultra-fine ===")
for df in [0.374, 0.3745, 0.375, 0.3755, 0.376]:
    for kf in [0.0318, 0.0319, 0.032, 0.0321, 0.0322]:
        run_structural(f"DF={df},KF={kf}", DISCOUNT_FACTOR=df, KELLY_FRACTION=kf)

print()
# === 3. Exponential discount model ===
print("=== Exponential discount ===")
def make_exp_entry(base, exp_rate):
    def compute_entry(price_yes, zscore):
        ratio = price_yes / strategy.LONGSHOT_PRICE_MAX
        effective_discount = base * math.exp(exp_rate * ratio)
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

# Calibrate exp to match linear at endpoints:
# Linear: at ratio=0: -1.80*0.375=-0.675, at ratio=1: 0.95*0.375=0.35625
# exp: at ratio=0: base, at ratio=1: base*exp(rate)
# So base = -0.675, but exp of negative is still positive...
# Let's just sweep different forms
for base, rate in [(-0.68, 1.65), (-0.70, 1.65), (-0.65, 1.60),
                   (-0.72, 1.70), (-0.68, 1.55), (-0.68, 1.70),
                   (-0.60, 1.50), (-0.75, 1.75)]:
    run_structural(f"exp({base},{rate})", compute_entry_fn=make_exp_entry(base, rate))

print()
# === 4. Piecewise model (two segments) ===
print("=== Piecewise entry model ===")
def make_piecewise(a1, b1, a2, b2, split):
    """Two different linear segments, split at normalized price ratio."""
    def compute_entry(price_yes, zscore):
        ratio = price_yes / strategy.LONGSHOT_PRICE_MAX
        if ratio < split:
            price_scale = a1 + b1 * ratio
        else:
            price_scale = a2 + b2 * ratio
        effective_discount = strategy.DISCOUNT_FACTOR * price_scale
        model_yes_prob = price_yes * effective_discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

# Split at midpoint, try different slopes for each half
for a1, b1, a2, b2, split in [
    (-2.0, 3.5, -1.5, 2.45, 0.5),
    (-2.2, 3.8, -1.4, 2.35, 0.5),
    (-1.5, 2.3, -2.0, 3.15, 0.5),
    (-2.0, 3.0, -1.6, 2.55, 0.5),
    (-2.5, 4.5, -1.2, 2.15, 0.5),
    (-1.80, 2.75, -1.80, 2.75, 0.5),  # Sanity: should match baseline
    # Split at 0.3 (cheap vs moderate)
    (-2.5, 4.5, -1.5, 2.45, 0.3),
    (-3.0, 6.0, -1.5, 2.45, 0.3),
    # Split at 0.7
    (-1.80, 2.75, -1.5, 2.45, 0.7),
    (-1.80, 2.75, -2.0, 3.0, 0.7),
]:
    run_structural(f"PW({a1},{b1},{a2},{b2}@{split})",
                  compute_entry_fn=make_piecewise(a1, b1, a2, b2, split))

print()
# === 5. Constant edge model (bypasses price-dependent discount) ===
print("=== Constant edge ===")
def make_const_edge(fixed_edge):
    def compute_entry(price_yes, zscore):
        no_price = 1.0 - price_yes
        model_no_prob = no_price + fixed_edge
        return fixed_edge, model_no_prob
    return compute_entry

for fe in [0.055, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]:
    run_structural(f"FixE={fe}", compute_entry_fn=make_const_edge(fe))

print()
# === 6. Logistic discount ===
print("=== Logistic discount ===")
def make_logistic(L, k, x0):
    """Logistic function: L / (1 + exp(-k*(x-x0))) mapped from ratio to discount."""
    def compute_entry(price_yes, zscore):
        ratio = price_yes / strategy.LONGSHOT_PRICE_MAX
        # Logistic: goes from ~0 to L
        discount = L / (1 + math.exp(-k * (ratio - x0)))
        model_yes_prob = price_yes * discount
        model_no_prob = 1.0 - model_yes_prob
        no_price = 1.0 - price_yes
        edge = model_no_prob - no_price
        return edge, model_no_prob
    return compute_entry

for L, k, x0 in [(1.0, 5.0, 0.5), (0.95, 5.0, 0.5), (1.0, 8.0, 0.5),
                   (1.0, 3.0, 0.5), (0.95, 5.0, 0.4), (0.95, 5.0, 0.6),
                   (0.80, 5.0, 0.5), (1.2, 5.0, 0.5)]:
    run_structural(f"Log(L={L},k={k},x0={x0})", compute_entry_fn=make_logistic(L, k, x0))

print()
# === 7. Quality gate with different z-score configurations ===
print("=== Quality gate: zscore penalty configs ===")
for pivot, power, coeff in [
    (10, 2, 0.0005),  # baseline
    (10, 1, 0.001),   # linear penalty
    (10, 2, 0.001),   # stronger quadratic
    (10, 2, 0.0),     # no penalty
    (10, 3, 0.0002),  # cubic but weaker
    (8, 2, 0.0005),   # lower pivot
    (12, 2, 0.0005),  # higher pivot
    (7, 2, 0.001),    # low pivot, stronger penalty
    (15, 2, 0.0002),  # higher pivot, weaker
]:
    run_structural(f"ZP={pivot},pw={power},c={coeff}",
                  ZSCORE_DAYS_PIVOT=pivot, ZSCORE_PENALTY_POWER=power, ZSCORE_PENALTY_COEFF=coeff)

print()
# === 8. LP=0.093 (just barely wider) with DF/scale retuning ===
print("=== LP=0.093 with retuning ===")
for a, b in [(-1.80, 2.75), (-1.78, 2.73), (-1.82, 2.77), (-1.75, 2.70), (-1.85, 2.80)]:
    for df in [0.375, 0.380, 0.385, 0.370]:
        run_structural(f"LP=0.093+a={a},b={b},DF={df}",
                      compute_entry_fn=make_entry(a, b),
                      LONGSHOT_PRICE_MAX=0.093, DISCOUNT_FACTOR=df)

print()
# === 9. MIN_HISTORY_TICKS sensitivity ===
print("=== MIN_HISTORY_TICKS ===")
for mht in [8, 9, 10, 11, 12, 13, 14, 15]:
    run_structural(f"MHT={mht}", MIN_HISTORY_TICKS=mht)

print()
# === 10. SPIKE_LOOKBACK_TICKS fine-tune ===
print("=== SPIKE_LOOKBACK fine ===")
for slt in [17, 18, 19, 20, 21, 22, 23, 24, 25]:
    run_structural(f"SLT={slt}", SPIKE_LOOKBACK_TICKS=slt)
