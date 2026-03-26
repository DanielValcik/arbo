#!/usr/bin/env python3
"""
Exit Strategy Parameter Sweep for Strategy C
=============================================

Sweeps exit parameters in strategy_experiment_v2 and compares
each variant against the baseline (EXIT_ENABLED=False = no exits).

Run: python3 research/sweep_exits.py

Output: greppable results with delta vs baseline for each parameter combination.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment_v2 as strategy
from backtest_harness_v2 import (
    download_historical_data, load_data, compute_monthly_normals,
    walk_forward_evaluate,
)


BASELINE = 0.0  # Set after first run


def run_one(label: str, **overrides):
    """Run backtest with parameter overrides, print result vs baseline."""
    global BASELINE
    originals = {}
    for k, v in overrides.items():
        originals[k] = getattr(strategy, k)
        setattr(strategy, k, v)

    try:
        results = walk_forward_evaluate(DATA, NORMALS)
        cs = results["composite_score"]
        delta = cs - BASELINE
        marker = " ***" if delta > 0.5 else ""
        exits = results["total_exits"]
        saves = results["total_saves"]
        regrets = results["total_regrets"]
        rolls = results["total_rolls"]
        trades = results["total_trades"]
        wr = results["avg_win_rate"]
        dd = results["max_drawdown_pct"]

        print(f"  {label:<45} cs={cs:>10.4f} (Δ{delta:>+8.3f})  "
              f"trades={trades:>4} wr={wr:>5.1f}% dd={dd:>5.2f}% "
              f"exits={exits:>3} saves={saves:>3} regrets={regrets:>3} "
              f"rolls={rolls:>3}{marker}")
        return cs
    finally:
        for k, v in originals.items():
            setattr(strategy, k, v)


def section(title):
    print(f"\n{'─' * 90}")
    print(f"  {title}")
    print(f"{'─' * 90}")


if __name__ == "__main__":
    print("=" * 90)
    print("  EXIT STRATEGY PARAMETER SWEEP — Strategy C")
    print("=" * 90)

    # Load data once
    download_historical_data()
    DATA = load_data()
    NORMALS = compute_monthly_normals(DATA)

    # ── Baseline: no exits ──
    section("BASELINE (no exits)")
    BASELINE = run_one("EXIT_ENABLED=False", EXIT_ENABLED=False)
    # Re-enable for subsequent tests
    strategy.EXIT_ENABLED = True

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Single-parameter sweeps
    # ══════════════════════════════════════════════════════════════════

    # ── MIN_HOLD_EDGE: when to exit based on updated edge ──
    section("MIN_HOLD_EDGE (edge-based exit threshold)")
    for mhe in [-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
        run_one(f"MIN_HOLD_EDGE={mhe}", MIN_HOLD_EDGE=mhe, STOP_LOSS_PCT=1.0,
                PROB_EXIT_FLOOR=0.0, PARTIAL_PROFIT_ENABLED=False, ROLL_ENABLED=False)

    # ── STOP_LOSS_PCT: price-based safety net ──
    section("STOP_LOSS_PCT (price-based stop-loss)")
    for sl in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]:
        run_one(f"STOP_LOSS_PCT={sl}", STOP_LOSS_PCT=sl, MIN_HOLD_EDGE=-99.0,
                PROB_EXIT_FLOOR=0.0, PARTIAL_PROFIT_ENABLED=False, ROLL_ENABLED=False)

    # ── PROB_EXIT_FLOOR: absolute probability minimum ──
    section("PROB_EXIT_FLOOR (minimum probability to hold)")
    for pef in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        run_one(f"PROB_EXIT_FLOOR={pef}", PROB_EXIT_FLOOR=pef, MIN_HOLD_EDGE=-99.0,
                STOP_LOSS_PCT=1.0, PARTIAL_PROFIT_ENABLED=False, ROLL_ENABLED=False)

    # ── PARTIAL_PROFIT: take profit on winners ──
    section("PARTIAL_PROFIT (profit-taking threshold + fraction)")
    run_one("PARTIAL_PROFIT=off", PARTIAL_PROFIT_ENABLED=False, MIN_HOLD_EDGE=-99.0,
            STOP_LOSS_PCT=1.0, PROB_EXIT_FLOOR=0.0, ROLL_ENABLED=False)
    for thresh in [0.15, 0.20, 0.30, 0.40, 0.50]:
        for frac in [0.30, 0.50, 0.70]:
            run_one(f"PARTIAL +{thresh*100:.0f}% sell {frac*100:.0f}%",
                    PARTIAL_PROFIT_ENABLED=True,
                    PARTIAL_PROFIT_THRESHOLD=thresh,
                    PARTIAL_PROFIT_FRACTION=frac,
                    MIN_HOLD_EDGE=-99.0, STOP_LOSS_PCT=1.0,
                    PROB_EXIT_FLOOR=0.0, ROLL_ENABLED=False)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Combinations of best single params
    # ══════════════════════════════════════════════════════════════════

    section("COMBINATIONS: MIN_HOLD_EDGE + STOP_LOSS")
    for mhe in [-0.05, 0.0, 0.05]:
        for sl in [0.25, 0.35, 0.50, 1.0]:
            run_one(f"MHE={mhe} + SL={sl}",
                    MIN_HOLD_EDGE=mhe, STOP_LOSS_PCT=sl,
                    PROB_EXIT_FLOOR=0.0, PARTIAL_PROFIT_ENABLED=False,
                    ROLL_ENABLED=False)

    section("COMBINATIONS: MIN_HOLD_EDGE + PROB_FLOOR")
    for mhe in [-0.05, 0.0, 0.05]:
        for pef in [0.0, 0.25, 0.35]:
            run_one(f"MHE={mhe} + PEF={pef}",
                    MIN_HOLD_EDGE=mhe, PROB_EXIT_FLOOR=pef,
                    STOP_LOSS_PCT=1.0, PARTIAL_PROFIT_ENABLED=False,
                    ROLL_ENABLED=False)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Rolling (hedging via adjacent bucket rotation)
    # ══════════════════════════════════════════════════════════════════

    section("ROLLING: exit + enter adjacent bucket")
    for mhe in [-0.05, 0.0, 0.05]:
        for mre in [0.04, 0.06, 0.08, 0.10]:
            run_one(f"ROLL MHE={mhe} MRE={mre}",
                    MIN_HOLD_EDGE=mhe, ROLL_ENABLED=True, MIN_ROLL_EDGE=mre,
                    STOP_LOSS_PCT=1.0, PROB_EXIT_FLOOR=0.0,
                    PARTIAL_PROFIT_ENABLED=False)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Full combos (best from each phase)
    # ══════════════════════════════════════════════════════════════════

    section("FULL COMBOS (narrow search around best params)")
    for mhe in [-0.05, 0.0, 0.05]:
        for sl in [0.35, 0.50, 1.0]:
            for roll in [False, True]:
                mre = 0.06 if roll else 0.08
                label = f"MHE={mhe} SL={sl} ROLL={'Y' if roll else 'N'}"
                run_one(label,
                        MIN_HOLD_EDGE=mhe, STOP_LOSS_PCT=sl,
                        ROLL_ENABLED=roll, MIN_ROLL_EDGE=mre,
                        PROB_EXIT_FLOOR=0.0, PARTIAL_PROFIT_ENABLED=False)

    print("\n" + "=" * 90)
    print(f"  BASELINE: {BASELINE:.6f}")
    print("=" * 90)
