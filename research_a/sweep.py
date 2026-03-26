#!/usr/bin/env python3
"""
Parameter sweep for Strategy A.
Imports the strategy module, overrides parameters, runs backtest.
Does NOT modify strategy_a_experiment.py.
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy
from backtest_a_harness import WINDOWS, BASE_SEED, simulate_window, compute_composite_score

BEST_SCORE = 74.878352


def run_one(label: str, **overrides):
    """Run backtest with parameter overrides, print result."""
    # Save originals
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
        marker = "***BETTER***" if delta > 0 else ""
        print(f"{label}: composite={cs:.6f} (delta={delta:+.3f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} {marker}")
    finally:
        # Restore
        for k, v in originals.items():
            setattr(strategy, k, v)


if __name__ == "__main__":
    print(f"Baseline best: {BEST_SCORE}")
    print("=" * 80)

    # Partial exit experiments
    for pep in [0.30, 0.40, 0.60, 0.70, 0.80, 1.00, 2.00, 10.0]:
        run_one(f"PARTIAL_EXIT_PROFIT_PCT={pep}", PARTIAL_EXIT_PROFIT_PCT=pep)

    print()
    # Partial exit sell percentage
    for pesp in [0.25, 0.30, 0.40, 0.60, 0.70, 0.80]:
        run_one(f"PARTIAL_EXIT_SELL_PCT={pesp}", PARTIAL_EXIT_SELL_PCT=pesp)

    print()
    # Stop loss experiments
    for sl in [0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30, 0.40, 0.50, 1.00]:
        run_one(f"STOP_LOSS_PCT={sl}", STOP_LOSS_PCT=sl)

    print()
    # Position sizing min
    for pm in [0.015, 0.018, 0.022, 0.025, 0.03]:
        run_one(f"POSITION_PCT_MIN={pm}", POSITION_PCT_MIN=pm)

    print()
    # Position sizing max
    for pm in [0.03, 0.04, 0.06, 0.08, 0.10]:
        run_one(f"POSITION_PCT_MAX={pm}", POSITION_PCT_MAX=pm)

    print()
    # Kelly fraction
    for kf in [0.028, 0.030, 0.031, 0.033, 0.034, 0.035, 0.036, 0.038, 0.040]:
        run_one(f"KELLY_FRACTION={kf}", KELLY_FRACTION=kf)

    print()
    # MIN_EDGE
    for me in [0.01, 0.015, 0.018, 0.022, 0.025, 0.03]:
        run_one(f"MIN_EDGE={me}", MIN_EDGE=me)

    print()
    # MAX_EDGE
    for me in [0.20, 0.30, 0.40, 0.60]:
        run_one(f"MAX_EDGE={me}", MAX_EDGE=me)

    print()
    # DISCOUNT_FACTOR
    for df in [0.20, 0.22, 0.25, 0.27, 0.28, 0.29, 0.31, 0.32, 0.33, 0.35]:
        run_one(f"DISCOUNT_FACTOR={df}", DISCOUNT_FACTOR=df)

    print()
    # MAX_CONCURRENT
    for mc in [15, 20, 30, 35, 40]:
        run_one(f"MAX_CONCURRENT={mc}", MAX_CONCURRENT=mc)

    print()
    # MAX_CAPITAL_DEPLOYED_PCT
    for mcd in [0.50, 0.60, 0.70, 0.90, 1.00]:
        run_one(f"MAX_CAPITAL_DEPLOYED_PCT={mcd}", MAX_CAPITAL_DEPLOYED_PCT=mcd)

    print()
    # LONGSHOT_PRICE_MAX
    for lpm in [0.060, 0.070, 0.075, 0.080, 0.090, 0.095, 0.100]:
        run_one(f"LONGSHOT_PRICE_MAX={lpm}", LONGSHOT_PRICE_MAX=lpm)

    print()
    # MIN_YES_PRICE
    for myp in [0.005, 0.008, 0.015, 0.02, 0.03]:
        run_one(f"MIN_YES_PRICE={myp}", MIN_YES_PRICE=myp)

    print()
    # RESOLUTION_DAYS_MIN
    for rdm in [1, 3, 4, 5]:
        run_one(f"RESOLUTION_DAYS_MIN={rdm}", RESOLUTION_DAYS_MIN=rdm)

    print()
    # RESOLUTION_DAYS_MAX
    for rdm in [14, 18, 24, 28, 30, 35]:
        run_one(f"RESOLUTION_DAYS_MAX={rdm}", RESOLUTION_DAYS_MAX=rdm)

    print()
    # MIN_VOLUME_24H
    for mv in [5000, 8000, 10000, 12000, 18000, 20000, 25000]:
        run_one(f"MIN_VOLUME_24H={mv}", MIN_VOLUME_24H=float(mv))

    print()
    # ZSCORE_THRESHOLD
    for zt in [2.5, 2.6, 2.7, 2.8, 2.85, 2.95, 3.0, 3.1, 3.2]:
        run_one(f"ZSCORE_THRESHOLD={zt}", ZSCORE_THRESHOLD=zt)

    print()
    # Two-parameter combos
    print("=== TWO-PARAMETER COMBOS ===")

    # MIN_EDGE + KELLY combos
    for me, kf in [(0.01, 0.035), (0.01, 0.040), (0.015, 0.035), (0.015, 0.040),
                    (0.01, 0.032), (0.018, 0.032), (0.025, 0.032)]:
        run_one(f"MIN_EDGE={me}+KELLY={kf}", MIN_EDGE=me, KELLY_FRACTION=kf)

    # DISCOUNT + MIN_EDGE combos
    for df, me in [(0.28, 0.02), (0.28, 0.015), (0.32, 0.02), (0.32, 0.015),
                    (0.25, 0.015), (0.25, 0.01)]:
        run_one(f"DISCOUNT={df}+MIN_EDGE={me}", DISCOUNT_FACTOR=df, MIN_EDGE=me)

    # POSITION sizing combos
    for pm_min, pm_max in [(0.015, 0.06), (0.018, 0.06), (0.025, 0.08),
                            (0.025, 0.05), (0.03, 0.06), (0.03, 0.08)]:
        run_one(f"PCT_MIN={pm_min}+PCT_MAX={pm_max}",
                POSITION_PCT_MIN=pm_min, POSITION_PCT_MAX=pm_max)

    # Kelly + Position combos
    for kf, pm_max in [(0.035, 0.06), (0.040, 0.06), (0.035, 0.08),
                        (0.040, 0.08), (0.045, 0.08)]:
        run_one(f"KELLY={kf}+PCT_MAX={pm_max}",
                KELLY_FRACTION=kf, POSITION_PCT_MAX=pm_max)

    print()
    print("=== THREE-PARAMETER COMBOS ===")
    for me, kf, pm_max in [(0.01, 0.035, 0.06), (0.01, 0.040, 0.06),
                            (0.015, 0.035, 0.06), (0.015, 0.040, 0.08),
                            (0.01, 0.032, 0.06), (0.01, 0.035, 0.08)]:
        run_one(f"EDGE={me}+KELLY={kf}+MAX={pm_max}",
                MIN_EDGE=me, KELLY_FRACTION=kf, POSITION_PCT_MAX=pm_max)

    # Discount + Kelly + min_edge combos
    for df, kf, me in [(0.28, 0.035, 0.015), (0.28, 0.040, 0.01),
                        (0.32, 0.035, 0.015), (0.25, 0.040, 0.01)]:
        run_one(f"DISC={df}+KELLY={kf}+EDGE={me}",
                DISCOUNT_FACTOR=df, KELLY_FRACTION=kf, MIN_EDGE=me)
