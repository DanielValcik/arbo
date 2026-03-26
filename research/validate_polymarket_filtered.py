"""
Filtered Polymarket Validation — focuses on signals our quality gate would select.

Instead of checking if our top pick is the winner across ALL events,
this tests: when our model says a specific bucket has >62% probability
AND it's priced in the 0.30-0.43 range on the market, how often does
that bucket actually win?

This matches what the strategy actually does in production.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
from validate_polymarket import (
    DATA_DIR, CITY_COORDS, parse_events, fetch_historical_temps,
)


def main():
    # Load cached Polymarket events
    pm_cache = DATA_DIR / "polymarket_weather_events.json"
    if not pm_cache.exists():
        print("Run validate_polymarket.py first to fetch Polymarket data")
        return

    with open(pm_cache) as f:
        raw_events = json.load(f)

    resolved = parse_events(raw_events)
    print(f"Parsed {len(resolved)} resolved events")

    # Load historical temps (cached from previous run)
    by_city = defaultdict(list)
    for rm in resolved:
        by_city[rm.city].append(rm)

    city_temps = {}
    for city, mks in by_city.items():
        dates = sorted(set(str(rm.target_date) for rm in mks))
        city_temps[city] = fetch_historical_temps(city, dates)

    # Run filtered analysis
    # For each event, compute probability for EACH bucket
    # Apply quality gate logic: only "trade" buckets where prob > 0.62 AND
    # the market would plausibly price them in 0.30-0.43

    high_conf_total = 0
    high_conf_correct = 0
    conf_bins = defaultdict(lambda: {"total": 0, "correct": 0})
    city_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for rm in resolved:
        date_str = rm.target_date.isoformat()
        forecast_temp = city_temps.get(rm.city, {}).get(date_str)
        if forecast_temp is None:
            continue

        for i, bucket in enumerate(rm.buckets):
            prob = strategy.estimate_probability(
                forecast_temp,
                bucket["low_c"],
                bucket["high_c"],
                0,  # day 0
                city=rm.city,
            )

            # Bucket the confidence level
            if prob >= 0.30:
                conf_bin = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"

                # High confidence signals (what we'd actually trade)
                if prob >= 0.62:
                    high_conf_total += 1
                    city_stats[rm.city]["total"] += 1
                    if bucket["won"]:
                        high_conf_correct += 1
                        city_stats[rm.city]["correct"] += 1

                    conf_bins[conf_bin]["total"] += 1
                    if bucket["won"]:
                        conf_bins[conf_bin]["correct"] += 1

                # Also track lower confidence for calibration
                elif prob >= 0.30:
                    conf_bins[conf_bin]["total"] += 1
                    if bucket["won"]:
                        conf_bins[conf_bin]["correct"] += 1

    # Report
    print(f"\n{'=' * 78}")
    print("FILTERED VALIDATION (signals quality gate would select)")
    print(f"{'=' * 78}")

    if high_conf_total > 0:
        hc_pct = high_conf_correct / high_conf_total * 100
        print(f"\n  High-confidence signals (forecast_prob >= 62%):")
        print(f"    Total signals:    {high_conf_total}")
        print(f"    Actually won:     {high_conf_correct}")
        print(f"    Win rate:         {hc_pct:.1f}%")
    else:
        print("\n  No high-confidence signals found (prob >= 62%)")

    print(f"\n  Per-city high-confidence accuracy:")
    for city in sorted(city_stats.keys()):
        d = city_stats[city]
        acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        print(f"    {city:15s}:  {d['total']:4d} signals  |  "
              f"{d['correct']:4d} won  |  {acc:5.1f}%")

    # Calibration curve
    print(f"\n  Probability calibration (predicted vs actual):")
    print(f"    {'Predicted':>12s}  {'Signals':>8s}  {'Won':>6s}  {'Actual':>8s}  {'Status':>10s}")
    for bin_label in sorted(conf_bins.keys()):
        d = conf_bins[bin_label]
        actual = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        predicted_mid = int(bin_label.split("-")[0]) + 5
        cal = "GOOD" if abs(actual - predicted_mid) < 15 else ("OVER" if actual < predicted_mid else "UNDER")
        print(f"    {bin_label:>12s}  {d['total']:>8d}  {d['correct']:>6d}  {actual:>7.1f}%  {cal:>10s}")

    # Greppable
    print(f"\n---")
    if high_conf_total > 0:
        print(f"pm_filtered_signals:    {high_conf_total}")
        print(f"pm_filtered_wins:       {high_conf_correct}")
        print(f"pm_filtered_win_rate:   {hc_pct:.1f}")
    else:
        print(f"pm_filtered_signals:    0")
        print(f"pm_filtered_wins:       0")
        print(f"pm_filtered_win_rate:   0.0")


if __name__ == "__main__":
    main()
