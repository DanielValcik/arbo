"""
Out-of-Sample Validation for Strategy C Autoresearch
=====================================================

Tests the optimized strategy on data it has NEVER seen:
1. Oct-Dec 2025: In the cached dataset but NOT in any walk-forward test window
2. Jan-Mar 2026: Fresh download from Open-Meteo — completely unseen data

This validates that 168+ composite_score is real edge, not overfitting.

Usage: python3 research/validate_oos.py
"""

import json
import math
import os
import random
import ssl
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy

# Re-use constants from the harness
from backtest_harness import (
    CITIES, INITIAL_CAPITAL, MAX_SIZING_CAPITAL, SLIPPAGE_PCT,
    BUCKET_WIDTH_C, NUM_RANGE_BUCKETS,
    MM_FORECAST_NOISE, MM_SIGMA, MM_PRICING_NOISE, OUR_FORECAST_NOISE,
    generate_buckets, generate_market, simulate_forecast, bucket_contains,
    run_single_backtest, calculate_metrics, Trade,
    compute_monthly_normals,
)

DATA_DIR = Path(__file__).parent / "data"
OOS_CACHE = DATA_DIR / "weather_2026.json"
BASE_SEED = 12345  # Different from training seed (42) to ensure independence


def download_2026_data():
    """Download Jan 1 - Mar 10 2026 weather data from Open-Meteo archive."""
    if OOS_CACHE.exists():
        size_kb = OOS_CACHE.stat().st_size / 1024
        print(f"OOS data: cached at {OOS_CACHE} ({size_kb:.0f} KB)")
        with open(OOS_CACHE) as f:
            return json.load(f)

    print("OOS data: downloading 2026 weather from Open-Meteo archive...")
    all_data = {}
    start = "2026-01-01"
    end = "2026-03-10"

    for city_id, coords in CITIES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={coords['lat']}&longitude={coords['lon']}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_max,temperature_2m_min"
            f"&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  {city_id}: FAILED ({e})")
            continue

        daily = data.get("daily", {})
        city_data = {}
        for i, dt in enumerate(daily.get("time", [])):
            high = daily["temperature_2m_max"][i]
            low = daily["temperature_2m_min"][i]
            if high is not None and low is not None:
                city_data[dt] = {"high_c": high, "low_c": low}
        all_data[city_id] = city_data
        print(f"  {city_id}: {len(city_data)} days")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OOS_CACHE, "w") as f:
        json.dump(all_data, f)
    print(f"OOS data: saved to {OOS_CACHE}")
    return all_data


def run_oos_period(data, normals, label, start, end, seed):
    """Run a single OOS period and report metrics."""
    trades, final_cap = run_single_backtest(data, normals, start, end, seed)
    metrics = calculate_metrics(trades, INITIAL_CAPITAL)

    # Per-city breakdown
    city_trades = {}
    for t in trades:
        if t.city not in city_trades:
            city_trades[t.city] = []
        city_trades[t.city].append(t)

    print(f"\n{'=' * 78}")
    print(f"OOS PERIOD: {label} ({start} -> {end})")
    print(f"{'=' * 78}")
    print(f"  Composite: {metrics['composite_score']:.4f}  |  "
          f"Sharpe: {metrics['sharpe_ratio']:.3f}  |  "
          f"PnL: {metrics['total_pnl_pct']:.1f}%  |  "
          f"Win: {metrics['win_rate_pct']:.1f}%  |  "
          f"Trades: {metrics['num_trades']}  |  "
          f"MaxDD: {metrics['max_drawdown_pct']:.2f}%")

    print(f"\n  Per-city breakdown:")
    for city_id in sorted(city_trades.keys()):
        ct = city_trades[city_id]
        wins = sum(1 for t in ct if t.won)
        pnl = sum(t.pnl for t in ct)
        wr = wins / len(ct) * 100 if ct else 0
        print(f"    {city_id:15s}:  {len(ct):4d} trades  |  "
              f"Win {wr:5.1f}%  |  PnL ${pnl:>8.2f}")

    # Days-out breakdown
    d0_trades = [t for t in trades if t.days_out == 0]
    d1_trades = [t for t in trades if t.days_out == 1]
    print(f"\n  Days-out breakdown:")
    for label_d, dt in [("Day 0", d0_trades), ("Day 1", d1_trades)]:
        if dt:
            wins = sum(1 for t in dt if t.won)
            pnl = sum(t.pnl for t in dt)
            wr = wins / len(dt) * 100
            print(f"    {label_d}:  {len(dt):4d} trades  |  "
                  f"Win {wr:5.1f}%  |  PnL ${pnl:>8.2f}")

    return metrics, trades


def main():
    t_start = time.time()

    # ── Period 1: Oct-Dec 2025 (in cached data, NOT in any test window) ──
    print("Loading cached 2024-2025 data...")
    cached_file = DATA_DIR / "weather_history.json"
    if not cached_file.exists():
        print("ERROR: Run 'python3 research/backtest_harness.py' first to download data")
        return

    with open(cached_file) as f:
        cached_data = json.load(f)
    normals_cached = compute_monthly_normals(cached_data)

    print(f"Cached data: {len(cached_data)} cities")

    m1, t1 = run_oos_period(
        cached_data, normals_cached,
        "Oct-Dec 2025 (holdout from cached data)",
        "2025-10-01", "2025-12-31",
        BASE_SEED,
    )

    # ── Period 2: Jan-Mar 2026 (completely fresh data) ──
    data_2026 = download_2026_data()

    # Compute normals from 2024-2025 data (what we'd know at the time)
    # This is more realistic — we wouldn't have 2026 normals in production
    normals_for_2026 = normals_cached

    m2, t2 = run_oos_period(
        data_2026, normals_for_2026,
        "Jan-Mar 2026 (completely unseen data)",
        "2026-01-01", "2026-03-10",
        BASE_SEED + 99999,
    )

    # ── Combined OOS summary ──
    all_trades = t1 + t2
    combined = calculate_metrics(all_trades, INITIAL_CAPITAL)

    t_end = time.time()

    print(f"\n{'=' * 78}")
    print("OUT-OF-SAMPLE VALIDATION SUMMARY")
    print(f"{'=' * 78}")

    # Compare against walk-forward in-sample results
    print(f"\n  Walk-forward (in-sample, 5 windows avg):")
    print(f"    composite_score:  168.67  (best from 293 experiments)")

    print(f"\n  OOS Oct-Dec 2025 (holdout):")
    print(f"    composite_score:  {m1['composite_score']:.4f}")
    print(f"    sharpe:           {m1['sharpe_ratio']:.3f}")
    print(f"    win_rate:         {m1['win_rate_pct']:.1f}%")
    print(f"    num_trades:       {m1['num_trades']}")
    print(f"    max_drawdown:     {m1['max_drawdown_pct']:.2f}%")

    print(f"\n  OOS Jan-Mar 2026 (completely unseen):")
    print(f"    composite_score:  {m2['composite_score']:.4f}")
    print(f"    sharpe:           {m2['sharpe_ratio']:.3f}")
    print(f"    win_rate:         {m2['win_rate_pct']:.1f}%")
    print(f"    num_trades:       {m2['num_trades']}")
    print(f"    max_drawdown:     {m2['max_drawdown_pct']:.2f}%")

    print(f"\n  Combined OOS (6 months total):")
    print(f"    composite_score:  {combined['composite_score']:.4f}")
    print(f"    sharpe:           {combined['sharpe_ratio']:.3f}")
    print(f"    win_rate:         {combined['win_rate_pct']:.1f}%")
    print(f"    num_trades:       {combined['num_trades']}")
    print(f"    max_drawdown:     {combined['max_drawdown_pct']:.2f}%")
    print(f"    profit_factor:    {combined['profit_factor']:.3f}")

    # Overfitting ratio: OOS / IS performance
    if m1['composite_score'] > 0 and m2['composite_score'] > 0:
        avg_oos = (m1['composite_score'] + m2['composite_score']) / 2
        ratio = avg_oos / 168.67
        print(f"\n  Overfitting ratio (OOS avg / IS):  {ratio:.3f}")
        if ratio > 0.7:
            print(f"    -> STRONG: Strategy generalizes well (>70% retained)")
        elif ratio > 0.4:
            print(f"    -> MODERATE: Some overfitting but strategy has real edge")
        else:
            print(f"    -> WEAK: Significant overfitting detected (<40% retained)")

    # Greppable output
    print("\n---")
    print(f"oos_oct_dec_2025_composite:   {m1['composite_score']:.6f}")
    print(f"oos_oct_dec_2025_sharpe:      {m1['sharpe_ratio']:.4f}")
    print(f"oos_oct_dec_2025_trades:      {m1['num_trades']}")
    print(f"oos_jan_mar_2026_composite:   {m2['composite_score']:.6f}")
    print(f"oos_jan_mar_2026_sharpe:      {m2['sharpe_ratio']:.4f}")
    print(f"oos_jan_mar_2026_trades:      {m2['num_trades']}")
    print(f"combined_oos_composite:       {combined['composite_score']:.6f}")
    print(f"combined_oos_sharpe:          {combined['sharpe_ratio']:.4f}")
    print(f"combined_oos_win_rate:        {combined['win_rate_pct']:.1f}")
    print(f"combined_oos_trades:          {combined['num_trades']}")
    print(f"combined_oos_max_dd:          {combined['max_drawdown_pct']:.2f}")
    print(f"validation_seconds:           {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
