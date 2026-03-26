"""
Data Source Alignment Analysis
==============================

Measures the temperature disagreement between Open-Meteo (our forecast source)
and Weather Underground (Polymarket's resolution source) by checking where the
winning bucket falls relative to our forecast.

If Open-Meteo and WU agree within 1°C, our model should be highly accurate.
If they disagree by 3-5°C, our model fails even when perfectly calibrated.

This tells us which cities have reliable data alignment for production.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_polymarket import (
    DATA_DIR, CITY_COORDS, parse_events, fetch_historical_temps, f_to_c,
)


def main():
    pm_cache = DATA_DIR / "polymarket_weather_events.json"
    with open(pm_cache) as f:
        raw_events = json.load(f)

    resolved = parse_events(raw_events)

    by_city = defaultdict(list)
    for rm in resolved:
        by_city[rm.city].append(rm)

    city_temps = {}
    for city, mks in by_city.items():
        dates = sorted(set(str(rm.target_date) for rm in mks))
        city_temps[city] = fetch_historical_temps(city, dates)

    # For each event, compute:
    # 1. Open-Meteo forecast temperature
    # 2. Winning bucket midpoint (proxy for Weather Underground actual temp)
    # 3. Difference between the two

    city_errors = defaultdict(list)
    city_within_bucket = defaultdict(lambda: {"total": 0, "within": 0})

    for rm in resolved:
        date_str = rm.target_date.isoformat()
        om_temp = city_temps.get(rm.city, {}).get(date_str)
        if om_temp is None:
            continue

        winning = rm.buckets[rm.winning_bucket_idx]

        # Estimate Weather Underground actual temp from winning bucket
        if winning["type"] == "range" or winning["type"] == "exact":
            if winning["low_c"] is not None and winning["high_c"] is not None:
                wu_midpoint = (winning["low_c"] + winning["high_c"]) / 2
            else:
                continue
        elif winning["type"] == "below":
            if winning["high_c"] is not None:
                wu_midpoint = winning["high_c"] - 1.5  # Estimate below bucket midpoint
            else:
                continue
        elif winning["type"] == "above":
            if winning["low_c"] is not None:
                wu_midpoint = winning["low_c"] + 1.5  # Estimate above bucket midpoint
            else:
                continue
        else:
            continue

        error = om_temp - wu_midpoint
        city_errors[rm.city].append(error)

        # Check if Open-Meteo temp falls within the winning bucket
        in_bucket = True
        if winning["low_c"] is not None and om_temp < winning["low_c"]:
            in_bucket = False
        if winning["high_c"] is not None and om_temp >= winning["high_c"]:
            in_bucket = False

        city_within_bucket[rm.city]["total"] += 1
        if in_bucket:
            city_within_bucket[rm.city]["within"] += 1

    # Report
    print(f"\n{'=' * 78}")
    print("DATA SOURCE ALIGNMENT: Open-Meteo vs Weather Underground (Polymarket)")
    print(f"{'=' * 78}")

    print(f"\n  Temperature Error (Open-Meteo − Weather Underground estimated midpoint):")
    print(f"    {'City':15s}  {'N':>5s}  {'Mean Error':>11s}  {'Std Dev':>8s}  {'MAE':>6s}  {'In Bucket':>10s}")

    all_errors = []
    for city in sorted(city_errors.keys()):
        errors = city_errors[city]
        all_errors.extend(errors)
        n = len(errors)
        mean_err = sum(errors) / n
        mae = sum(abs(e) for e in errors) / n
        variance = sum((e - mean_err) ** 2 for e in errors) / max(1, n - 1)
        std = math.sqrt(variance)

        wb = city_within_bucket[city]
        in_pct = wb["within"] / wb["total"] * 100 if wb["total"] > 0 else 0

        print(f"    {city:15s}  {n:5d}  {mean_err:>+10.2f}°C  {std:>7.2f}°C  {mae:>5.2f}°C  "
              f"{wb['within']:>3d}/{wb['total']:<3d} ({in_pct:.0f}%)")

    # Overall
    if all_errors:
        n = len(all_errors)
        mean_err = sum(all_errors) / n
        mae = sum(abs(e) for e in all_errors) / n
        variance = sum((e - mean_err) ** 2 for e in all_errors) / max(1, n - 1)
        std = math.sqrt(variance)

        total_wb = sum(d["within"] for d in city_within_bucket.values())
        total_n = sum(d["total"] for d in city_within_bucket.values())
        in_pct = total_wb / total_n * 100 if total_n > 0 else 0

        print(f"    {'ALL':15s}  {n:5d}  {mean_err:>+10.2f}°C  {std:>7.2f}°C  {mae:>5.2f}°C  "
              f"{total_wb:>3d}/{total_n:<3d} ({in_pct:.0f}%)")

    # Error distribution
    print(f"\n  Error Distribution (all cities):")
    if all_errors:
        for threshold in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            count = sum(1 for e in all_errors if abs(e) <= threshold)
            pct = count / len(all_errors) * 100
            print(f"    Within ±{threshold:.1f}°C:  {count:4d}/{len(all_errors)}  ({pct:.0f}%)")

    # Implications
    print(f"\n  Implications for Strategy C:")
    print(f"    Typical Polymarket bucket width: ~2.5°C (range) or 1°C (exact)")
    for city in sorted(city_errors.keys()):
        errors = city_errors[city]
        mae = sum(abs(e) for e in errors) / len(errors)
        wb = city_within_bucket[city]
        in_pct = wb["within"] / wb["total"] * 100 if wb["total"] > 0 else 0

        if in_pct > 40:
            status = "STRONG — data sources well aligned"
        elif in_pct > 25:
            status = "MODERATE — some alignment, tradeable"
        else:
            status = "WEAK — data sources disagree too much"

        print(f"    {city:15s}: MAE {mae:.1f}°C, {in_pct:.0f}% in-bucket → {status}")

    # Production recommendation
    print(f"\n  Production Data Source Recommendations:")
    for city in sorted(city_errors.keys()):
        wb = city_within_bucket[city]
        in_pct = wb["within"] / wb["total"] * 100 if wb["total"] > 0 else 0
        if city in ("nyc", "chicago"):
            print(f"    {city}: Use NOAA (production) — likely better than Open-Meteo ({in_pct:.0f}% baseline)")
        elif city == "london":
            print(f"    {city}: Use Met Office (production) — likely better than Open-Meteo ({in_pct:.0f}% baseline)")
        else:
            print(f"    {city}: Open-Meteo only — {in_pct:.0f}% baseline. Consider Weather Underground API if available")


if __name__ == "__main__":
    main()
