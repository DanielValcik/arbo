#!/usr/bin/env python3
"""Focused sweep around DISCOUNT_FACTOR=0.31 and combinations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_a_experiment as strategy
from backtest_a_harness import WINDOWS, BASE_SEED, simulate_window, compute_composite_score

BEST_SCORE = 74.977098  # DISCOUNT_FACTOR=0.31


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
        marker = "***BETTER***" if delta > 0 else ""
        print(f"{label}: composite={cs:.6f} (delta={delta:+.4f}) sharpe={scores['avg_sharpe']:.4f} "
              f"trades={scores['num_trades']} dd={scores['max_drawdown_pct']:.2f} wr={scores['avg_win_rate']:.1f} {marker}")
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


if __name__ == "__main__":
    print(f"Target best: {BEST_SCORE}")
    print("=" * 100)

    # Fine-tune DISCOUNT_FACTOR around 0.31
    print("--- DISCOUNT_FACTOR fine-tune ---")
    for df in [0.300, 0.302, 0.305, 0.308, 0.310, 0.312, 0.315, 0.318, 0.320]:
        run_one(f"DF={df}", DISCOUNT_FACTOR=df)

    print()
    print("--- DF=0.31 + KELLY combos ---")
    for kf in [0.030, 0.031, 0.032, 0.033, 0.034, 0.035]:
        run_one(f"DF=0.31+KF={kf}", DISCOUNT_FACTOR=0.31, KELLY_FRACTION=kf)

    print()
    print("--- DF=0.31 + MIN_EDGE combos ---")
    for me in [0.015, 0.018, 0.020, 0.022, 0.025]:
        run_one(f"DF=0.31+ME={me}", DISCOUNT_FACTOR=0.31, MIN_EDGE=me)

    print()
    print("--- DF=0.31 + ZSCORE combos ---")
    for zt in [2.85, 2.88, 2.90, 2.92, 2.95]:
        run_one(f"DF=0.31+ZT={zt}", DISCOUNT_FACTOR=0.31, ZSCORE_THRESHOLD=zt)

    print()
    print("--- DF=0.31 + STOP_LOSS combos ---")
    for sl in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30]:
        run_one(f"DF=0.31+SL={sl}", DISCOUNT_FACTOR=0.31, STOP_LOSS_PCT=sl)

    print()
    print("--- DF=0.31 + MIN_VOLUME combos ---")
    for mv in [12000, 14000, 15000, 16000, 18000]:
        run_one(f"DF=0.31+MV={mv}", DISCOUNT_FACTOR=0.31, MIN_VOLUME_24H=float(mv))

    print()
    print("--- DF=0.31 + RESOLUTION combos ---")
    for rdmax in [18, 19, 20, 21, 22]:
        run_one(f"DF=0.31+RD_MAX={rdmax}", DISCOUNT_FACTOR=0.31, RESOLUTION_DAYS_MAX=rdmax)

    print()
    print("--- DF=0.31 + LOOKBACK combos ---")
    for lb in [15, 17, 19, 20, 21, 23, 25]:
        run_one(f"DF=0.31+LB={lb}", DISCOUNT_FACTOR=0.31, SPIKE_LOOKBACK_TICKS=lb)

    print()
    print("--- DF=0.31 + MIN_HISTORY combos ---")
    for mh in [8, 9, 10, 11, 12, 13]:
        run_one(f"DF=0.31+MH={mh}", DISCOUNT_FACTOR=0.31, MIN_HISTORY_TICKS=mh)

    print()
    print("--- DF=0.31 + LONGSHOT_PRICE combos ---")
    for lp in [0.080, 0.082, 0.085, 0.088, 0.090]:
        run_one(f"DF=0.31+LP={lp}", DISCOUNT_FACTOR=0.31, LONGSHOT_PRICE_MAX=lp)

    print()
    print("--- TRIPLE COMBOS (DF=0.31 base) ---")
    for kf, me in [(0.031, 0.020), (0.032, 0.018), (0.033, 0.020),
                    (0.031, 0.022), (0.033, 0.022), (0.034, 0.020)]:
        run_one(f"DF=0.31+KF={kf}+ME={me}",
                DISCOUNT_FACTOR=0.31, KELLY_FRACTION=kf, MIN_EDGE=me)

    # DF + KELLY + ZSCORE
    for kf, zt in [(0.032, 2.88), (0.032, 2.92), (0.033, 2.88), (0.033, 2.92),
                    (0.031, 2.92), (0.031, 2.88)]:
        run_one(f"DF=0.31+KF={kf}+ZT={zt}",
                DISCOUNT_FACTOR=0.31, KELLY_FRACTION=kf, ZSCORE_THRESHOLD=zt)

    # Quad combos around best single
    for kf, me, zt in [(0.032, 0.020, 2.90), (0.033, 0.020, 2.92),
                        (0.031, 0.020, 2.88), (0.033, 0.022, 2.88),
                        (0.032, 0.022, 2.92)]:
        run_one(f"DF=0.31+KF={kf}+ME={me}+ZT={zt}",
                DISCOUNT_FACTOR=0.31, KELLY_FRACTION=kf, MIN_EDGE=me, ZSCORE_THRESHOLD=zt)
