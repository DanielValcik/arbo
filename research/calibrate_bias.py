#!/usr/bin/env python3
"""Calibrate forecast bias against actual METAR observations (IEM).

Compares our forecast sources (NOAA, Open-Meteo, Met Office) against actual
observed temperatures from the same airport stations Polymarket uses for resolution.

Outputs:
  - Per-city bias (mean forecast error in °C)
  - Per-city sigma (std of forecast error — use for _FORECAST_SIGMA)
  - Updated WU_BIAS_CORRECTION values for weather_models.py
  - Comparison table showing current vs computed corrections

Usage:
    python3 research/calibrate_bias.py [--days 60] [--cities nyc,seoul,london]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp

from arbo.connectors.weather_iem import STATION_MAP, IEMClient
from arbo.connectors.weather_models import (
    CITY_SOURCE_MAP,
    City,
    WeatherSource,
    WU_BIAS_CORRECTION,
)


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Output directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class CityCalibration:
    """Calibration results for one city."""

    city: City
    source: str
    n_days: int
    bias_c: float          # Mean(forecast - actual), positive = forecast reads too high
    sigma_c: float         # Std(forecast - actual)
    mae_c: float           # Mean absolute error
    current_correction: float  # Current WU_BIAS_CORRECTION value
    computed_correction: float  # What it should be (-bias)
    sample_errors: list[float]  # Individual errors for analysis


async def fetch_open_meteo_archive(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> dict[str, list[float]]:
    """Fetch Open-Meteo historical archive (ERA5 reanalysis) data."""
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }
    try:
        async with session.get(OPEN_METEO_ARCHIVE_URL, params=params) as resp:
            if resp.status != 200:
                print(f"  Open-Meteo archive error: {resp.status}")
                return {}
            data = await resp.json()
            daily = data.get("daily", {})
            return {
                "dates": daily.get("time", []),
                "max_temps": daily.get("temperature_2m_max", []),
                "min_temps": daily.get("temperature_2m_min", []),
            }
    except Exception as e:
        print(f"  Open-Meteo archive error: {e}")
        return {}


async def calibrate_city(
    city: City,
    days: int,
    iem_client: IEMClient,
    session: aiohttp.ClientSession,
) -> CityCalibration | None:
    """Calibrate one city by comparing forecast vs METAR actuals."""
    from arbo.connectors.weather_models import CITY_COORDS

    station = STATION_MAP.get(city)
    if station is None:
        return None

    coords = CITY_COORDS.get(city)
    if coords is None:
        return None

    source = CITY_SOURCE_MAP.get(city, WeatherSource.OPEN_METEO)
    source_name = source.value

    # Date range: end = 6 days ago (to allow data finalization), start = days before that
    end_date = date.today() - timedelta(days=6)
    start_date = end_date - timedelta(days=days)

    print(f"\n{'='*60}")
    print(f"  {city.value.upper()} ({station.icao}) — {source_name}")
    print(f"  Range: {start_date} to {end_date} ({days} days)")
    print(f"{'='*60}")

    # 1. Fetch METAR actuals from IEM
    print(f"  Fetching METAR observations from IEM...")
    try:
        metar_obs = await iem_client.get_historical_range(city, start_date, end_date)
    except Exception as e:
        print(f"  IEM error: {e}")
        return None

    if not metar_obs:
        print(f"  No METAR data available")
        return None

    metar_by_date = {obs.date: obs for obs in metar_obs}
    print(f"  Got {len(metar_obs)} days of METAR data")

    # 2. Fetch forecast source data (Open-Meteo archive for now)
    print(f"  Fetching Open-Meteo archive data...")
    archive = await fetch_open_meteo_archive(
        session, coords[0], coords[1], start_date, end_date,
    )

    if not archive or not archive.get("dates"):
        print(f"  No archive data available")
        return None

    # Build archive lookup
    archive_by_date: dict[date, float] = {}
    for i, date_str in enumerate(archive["dates"]):
        d = date.fromisoformat(date_str)
        max_temp = archive["max_temps"][i]
        if max_temp is not None:
            archive_by_date[d] = float(max_temp)

    print(f"  Got {len(archive_by_date)} days of archive data")

    # 3. Compare: forecast (archive) vs actual (METAR)
    errors: list[float] = []  # forecast - actual (positive = forecast too high)
    for d, metar in metar_by_date.items():
        forecast_temp = archive_by_date.get(d)
        if forecast_temp is None:
            continue
        # Use max_temp_c from METAR as actual
        error = forecast_temp - metar.max_temp_c
        errors.append(error)

    if len(errors) < 5:
        print(f"  Not enough matching data points ({len(errors)})")
        return None

    # 4. Compute statistics
    import statistics

    bias = statistics.mean(errors)
    sigma = statistics.stdev(errors)
    mae = statistics.mean([abs(e) for e in errors])

    current_corr = WU_BIAS_CORRECTION.get(city, 0.0)
    # Correction should be -bias (if forecast reads low, add positive correction)
    computed_corr = round(-bias, 2)

    print(f"  Matched days: {len(errors)}")
    print(f"  Bias (forecast - actual): {bias:+.2f}°C")
    print(f"  Sigma: {sigma:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  Current correction: {current_corr:+.1f}°C")
    print(f"  Computed correction: {computed_corr:+.1f}°C")

    return CityCalibration(
        city=city,
        source=source_name,
        n_days=len(errors),
        bias_c=round(bias, 3),
        sigma_c=round(sigma, 3),
        mae_c=round(mae, 3),
        current_correction=current_corr,
        computed_correction=computed_corr,
        sample_errors=errors,
    )


async def main(days: int, cities: list[str] | None) -> None:
    """Run calibration for all or selected cities."""
    import ssl
    import certifi

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    print("=" * 60)
    print("  FORECAST BIAS CALIBRATION vs METAR (IEM)")
    print(f"  {days}-day lookback, comparing Open-Meteo archive vs METAR actuals")
    print("=" * 60)

    iem_client = IEMClient()
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as session:
        # Determine which cities to calibrate
        if cities:
            city_list = [c for c in City if c.value in cities]
        else:
            city_list = list(City)

        results: list[CityCalibration] = []
        for city in city_list:
            try:
                result = await calibrate_city(city, days, iem_client, session)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  Error calibrating {city.value}: {e}")

    await iem_client.close()

    if not results:
        print("\nNo calibration results. Check IEM/Open-Meteo connectivity.")
        return

    # Summary table
    print("\n" + "=" * 90)
    print("  CALIBRATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'City':<16} {'Source':<12} {'Days':>5} {'Bias':>8} {'Sigma':>8} "
          f"{'MAE':>8} {'Current':>9} {'Computed':>9} {'Delta':>8}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: abs(x.bias_c), reverse=True):
        delta = r.computed_correction - r.current_correction
        print(
            f"  {r.city.value:<16} {r.source:<12} {r.n_days:>5} "
            f"{r.bias_c:>+8.2f} {r.sigma_c:>8.2f} {r.mae_c:>8.2f} "
            f"{r.current_correction:>+9.1f} {r.computed_correction:>+9.1f} "
            f"{delta:>+8.1f}"
        )

    # Output Python code for updated corrections
    print("\n" + "=" * 60)
    print("  UPDATED WU_BIAS_CORRECTION (copy to weather_models.py)")
    print("=" * 60)
    print("WU_BIAS_CORRECTION: dict[City, float] = {")
    for r in sorted(results, key=lambda x: x.city.value):
        print(f"    City.{r.city.name}: {r.computed_correction},  "
              f"# measured: bias={r.bias_c:+.2f}°C, σ={r.sigma_c:.2f}°C, n={r.n_days}")
    print("}")

    # Output sigma recommendations
    print("\n" + "=" * 60)
    print("  RECOMMENDED _FORECAST_SIGMA (copy to weather_scanner.py)")
    print("=" * 60)
    print("_FORECAST_SIGMA: dict[str, dict[int, float]] = {")
    for r in sorted(results, key=lambda x: x.city.value):
        # Sigma at d0 should be the measured forecast error std
        # Current values are from autoresearch optimization, so only update if significantly different
        print(f'    "{r.city.value}": {{0: {r.sigma_c:.2f}}},  # measured from {r.n_days} days')
    print("}")

    # Save raw results to JSON
    output_path = DATA_DIR / "calibration_results.json"
    output_data = [
        {
            "city": r.city.value,
            "source": r.source,
            "n_days": r.n_days,
            "bias_c": r.bias_c,
            "sigma_c": r.sigma_c,
            "mae_c": r.mae_c,
            "current_correction": r.current_correction,
            "computed_correction": r.computed_correction,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate forecast bias vs METAR")
    parser.add_argument("--days", type=int, default=60, help="Number of days to look back")
    parser.add_argument("--cities", type=str, help="Comma-separated city list (e.g., nyc,seoul)")
    args = parser.parse_args()

    city_list = args.cities.split(",") if args.cities else None
    asyncio.run(main(args.days, city_list))
