"""
Baseline OOS comparison — runs the ORIGINAL (pre-optimization) parameters
on the same OOS data to measure the true improvement from autoresearch.
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtest_harness import (
    CITIES, INITIAL_CAPITAL, MAX_SIZING_CAPITAL, SLIPPAGE_PCT,
    generate_buckets, generate_market, simulate_forecast, bucket_contains,
    run_single_backtest, calculate_metrics, compute_monthly_normals,
)

# Monkey-patch strategy_experiment with BASELINE parameters
import strategy_experiment as strategy

# Save optimized values
_opt_sigma = strategy.FORECAST_SIGMA.copy()
_opt_city_sigma = {k: v.copy() for k, v in strategy.CITY_SIGMA.items()}
_opt_days_out = strategy.DAYS_OUT_TO_TRADE[:]
_opt_min_edge = strategy.MIN_EDGE
_opt_min_price = strategy.MIN_PRICE
_opt_max_price = strategy.MAX_PRICE
_opt_distribution = strategy.DISTRIBUTION
_opt_prob_sharpening = strategy.PROB_SHARPENING
_opt_min_forecast_prob = strategy.MIN_FORECAST_PROB
_opt_kelly = strategy.KELLY_FRACTION
_opt_conviction = strategy.CONVICTION_RATIO

# Set BASELINE values (what production Strategy C uses)
BASELINE = {
    "FORECAST_SIGMA": {0: 1.8, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.0, 6: 7.0},
    "CITY_SIGMA": {},  # no per-city sigma
    "DAYS_OUT_TO_TRADE": [0, 1, 2],
    "MIN_EDGE": 0.10,
    "MIN_PRICE": 0.05,
    "MAX_PRICE": 0.85,
    "DISTRIBUTION": "student_t",
    "PROB_SHARPENING": 1.0,
    "MIN_FORECAST_PROB": 0.50,
    "KELLY_FRACTION": 0.25,
    "CONVICTION_RATIO": 1.2,
    "STUDENT_T_DF": 5,
    "MIN_VOLUME": 1000.0,
    "MIN_LIQUIDITY": 200.0,
}


def apply_params(params):
    strategy.FORECAST_SIGMA = params["FORECAST_SIGMA"]
    strategy.CITY_SIGMA = params.get("CITY_SIGMA", {})
    strategy.DAYS_OUT_TO_TRADE = params["DAYS_OUT_TO_TRADE"]
    strategy.MIN_EDGE = params["MIN_EDGE"]
    strategy.MIN_PRICE = params["MIN_PRICE"]
    strategy.MAX_PRICE = params["MAX_PRICE"]
    strategy.DISTRIBUTION = params["DISTRIBUTION"]
    strategy.PROB_SHARPENING = params["PROB_SHARPENING"]
    strategy.MIN_FORECAST_PROB = params["MIN_FORECAST_PROB"]
    strategy.KELLY_FRACTION = params["KELLY_FRACTION"]
    strategy.CONVICTION_RATIO = params["CONVICTION_RATIO"]
    strategy.STUDENT_T_DF = params.get("STUDENT_T_DF", 5)
    strategy.MIN_VOLUME = params.get("MIN_VOLUME", 1000.0)
    strategy.MIN_LIQUIDITY = params.get("MIN_LIQUIDITY", 200.0)


DATA_DIR = Path(__file__).parent / "data"
BASE_SEED = 12345


def run_comparison():
    # Load data
    with open(DATA_DIR / "weather_history.json") as f:
        cached = json.load(f)
    normals = compute_monthly_normals(cached)

    with open(DATA_DIR / "weather_2026.json") as f:
        data_2026 = json.load(f)

    periods = [
        ("Oct-Dec 2025", cached, "2025-10-01", "2025-12-31", BASE_SEED),
        ("Jan-Mar 2026", data_2026, "2026-01-01", "2026-03-10", BASE_SEED + 99999),
    ]

    for label in ["BASELINE (pre-optimization)", "OPTIMIZED (autoresearch best)"]:
        if "BASELINE" in label:
            apply_params(BASELINE)
        else:
            # Restore optimized
            strategy.FORECAST_SIGMA = _opt_sigma
            strategy.CITY_SIGMA = _opt_city_sigma
            strategy.DAYS_OUT_TO_TRADE = _opt_days_out
            strategy.MIN_EDGE = _opt_min_edge
            strategy.MIN_PRICE = _opt_min_price
            strategy.MAX_PRICE = _opt_max_price
            strategy.DISTRIBUTION = _opt_distribution
            strategy.PROB_SHARPENING = _opt_prob_sharpening
            strategy.MIN_FORECAST_PROB = _opt_min_forecast_prob
            strategy.KELLY_FRACTION = _opt_kelly
            strategy.CONVICTION_RATIO = _opt_conviction

        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"{'=' * 78}")

        all_trades = []
        for p_label, data, start, end, seed in periods:
            trades, cap = run_single_backtest(data, normals, start, end, seed)
            m = calculate_metrics(trades, INITIAL_CAPITAL)
            all_trades.extend(trades)
            print(f"  {p_label:15s}:  {m['num_trades']:4d} trades  |  "
                  f"Win {m['win_rate_pct']:5.1f}%  |  "
                  f"Sharpe {m['sharpe_ratio']:>8.2f}  |  "
                  f"PnL {m['total_pnl_pct']:>6.1f}%  |  "
                  f"MaxDD {m['max_drawdown_pct']:>5.2f}%")

        combined = calculate_metrics(all_trades, INITIAL_CAPITAL)
        print(f"  {'COMBINED':15s}:  {combined['num_trades']:4d} trades  |  "
              f"Win {combined['win_rate_pct']:5.1f}%  |  "
              f"Sharpe {combined['sharpe_ratio']:>8.2f}  |  "
              f"PnL {combined['total_pnl_pct']:>6.1f}%  |  "
              f"MaxDD {combined['max_drawdown_pct']:>5.2f}%")


if __name__ == "__main__":
    run_comparison()
