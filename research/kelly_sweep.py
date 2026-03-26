"""
Kelly Fraction Sweep — find optimal sizing for production use.

Tests multiple KELLY_FRACTION values while keeping all other v7 parameters fixed.
Also tests with/without the 0.35 multiplier to match production code.

Usage: python3 research/kelly_sweep.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import backtest_harness as harness
import strategy_experiment as strategy

# Save original values
ORIG_KELLY = strategy.KELLY_FRACTION

# Kelly values to test (these are the EFFECTIVE kelly — after any multiplier)
# Production code: kelly_adjusted = kelly_raw * KELLY_FRACTION (no multiplier)
# So KELLY_FRACTION IS the effective fraction
KELLY_VALUES = [
    # (label, kelly_fraction, use_035_multiplier)
    ("v7_baseline", 0.01, True),     # Original v7: 0.01 * 0.35 = 0.0035 effective
    ("kelly_0.02", 0.02, False),     # Current production
    ("kelly_0.04", 0.04, False),
    ("kelly_0.06", 0.06, False),
    ("kelly_0.08", 0.08, False),
    ("kelly_0.10", 0.10, False),
    ("kelly_0.15", 0.15, False),
    ("kelly_0.20", 0.20, False),
    ("kelly_0.25", 0.25, False),     # Quarter-Kelly
]


def estimate_trade_size(kelly_eff, capital=1000.0, edge=0.15, price=0.35):
    """Estimate typical trade size for a given kelly."""
    prob = price + edge
    odds = (1.0 / price) - 1.0
    kelly_raw = (prob * odds - (1 - prob)) / odds
    kelly_raw = min(kelly_raw, 0.40)
    available = capital * 0.80  # MAX_TOTAL_EXPOSURE_PCT
    return available * kelly_raw * kelly_eff


def main():
    t0 = time.time()

    harness.download_historical_data()
    data = harness.load_data()
    normals = harness.compute_monthly_normals(data)

    print(f"Data: {len(data)} cities loaded\n")
    print("=" * 100)
    print(f"{'Label':<16} {'Eff.Kelly':>10} {'~Size($)':>9} {'Composite':>10} "
          f"{'Sharpe':>8} {'WinRate':>8} {'Trades':>7} {'MaxDD%':>7} "
          f"{'PnL%':>8} {'PF':>7} {'Windows':>8}")
    print("=" * 100)

    results = []

    for label, kelly_frac, use_mult in KELLY_VALUES:
        # Patch strategy module
        strategy.KELLY_FRACTION = kelly_frac

        # Monkey-patch position_size to optionally remove 0.35 multiplier
        if not use_mult:
            # Override position_size to NOT use 0.35 multiplier
            orig_fn = strategy.position_size.__code__

            def patched_position_size(edge, market_price, available_capital, total_capital,
                                      *, city=None, _kf=kelly_frac):
                if market_price <= 0 or market_price >= 1 or edge <= 0:
                    return 0.0
                prob = market_price + edge
                if prob <= 0 or prob >= 1:
                    return 0.0
                odds = (1.0 / market_price) - 1.0
                kelly_raw = (prob * odds - (1.0 - prob)) / odds
                if kelly_raw <= 0:
                    return 0.0
                kelly_raw = min(kelly_raw, 0.40)
                kelly_adjusted = kelly_raw * _kf  # NO 0.35 multiplier
                size = available_capital * kelly_adjusted
                max_size = total_capital * strategy.MAX_POSITION_PCT
                size = min(size, max_size)
                if size < 1.0:
                    return 0.0
                return round(size, 2)

            strategy.position_size = patched_position_size
        else:
            # Restore original with 0.35 multiplier
            def orig_position_size(edge, market_price, available_capital, total_capital,
                                   *, city=None, _kf=kelly_frac):
                if market_price <= 0 or market_price >= 1 or edge <= 0:
                    return 0.0
                prob = market_price + edge
                if prob <= 0 or prob >= 1:
                    return 0.0
                odds = (1.0 / market_price) - 1.0
                kelly_raw = (prob * odds - (1.0 - prob)) / odds
                if kelly_raw <= 0:
                    return 0.0
                kelly_raw = min(kelly_raw, 0.40)
                kelly_adjusted = kelly_raw * _kf * 0.35  # WITH 0.35 multiplier
                size = available_capital * kelly_adjusted
                max_size = total_capital * strategy.MAX_POSITION_PCT
                size = min(size, max_size)
                if size < 1.0:
                    return 0.0
                return round(size, 2)

            strategy.position_size = orig_position_size

        # Effective kelly (what actually gets multiplied)
        eff_kelly = kelly_frac * (0.35 if use_mult else 1.0)
        est_size = estimate_trade_size(eff_kelly)

        # Run walk-forward
        r = harness.walk_forward_evaluate(data, normals)

        row = {
            "label": label,
            "eff_kelly": eff_kelly,
            "est_size": round(est_size, 2),
            "composite": r["composite_score"],
            "sharpe": r["avg_sharpe"],
            "win_rate": r["avg_win_rate"],
            "trades": r["total_trades"],
            "max_dd": r["max_drawdown_pct"],
            "pnl_pct": r["avg_pnl_pct"],
            "profit_factor": r["avg_profit_factor"],
            "profitable_windows": r["profitable_windows"],
            "total_windows": r["total_windows"],
        }
        results.append(row)

        win_str = f"{r['profitable_windows']}/{r['total_windows']}"
        print(f"{label:<16} {eff_kelly:>10.4f} {est_size:>9.2f} {r['composite_score']:>10.2f} "
              f"{r['avg_sharpe']:>8.2f} {r['avg_win_rate']:>7.1f}% {r['total_trades']:>7d} "
              f"{r['max_drawdown_pct']:>6.2f}% {r['avg_pnl_pct']:>7.1f}% "
              f"{r['avg_profit_factor']:>7.2f} {win_str:>8}")

    # Restore
    strategy.KELLY_FRACTION = ORIG_KELLY

    # Find best composite
    best = max(results, key=lambda r: r["composite"])
    print("\n" + "=" * 100)
    print(f"BEST: {best['label']} — composite={best['composite']:.2f}, "
          f"~${best['est_size']:.2f}/trade, Sharpe={best['sharpe']:.2f}, "
          f"WR={best['win_rate']:.1f}%, MaxDD={best['max_dd']:.2f}%")

    # Find best with reasonable size ($5+)
    reasonable = [r for r in results if r["est_size"] >= 5.0]
    if reasonable:
        best_reas = max(reasonable, key=lambda r: r["composite"])
        print(f"BEST ($5+): {best_reas['label']} — composite={best_reas['composite']:.2f}, "
              f"~${best_reas['est_size']:.2f}/trade, Sharpe={best_reas['sharpe']:.2f}")

    # Save results
    out_path = Path(__file__).parent / "kelly_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
