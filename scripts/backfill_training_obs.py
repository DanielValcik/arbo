"""Backfill training observations from GEFS ensemble + Open-Meteo actuals.

Creates (forecast, actual) pairs per city per date by:
1. GEFS ensemble_mean = forecast (from ensemble_stats)
2. Open-Meteo archive = actual observed temperature

Stores to weather_training_obs table for EMOSEnsembleModel fitting.

Usage (on VPS):
    /opt/arbo/.venv/bin/python scripts/backfill_training_obs.py
"""

import json
import os
import ssl
import time
import urllib.request
from datetime import date, timedelta

from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

CITIES = {
    "nyc": (40.71, -74.01), "chicago": (41.88, -87.63),
    "london": (51.51, -0.13), "seoul": (37.57, 126.98),
    "ankara": (39.93, 32.86), "sao_paulo": (-23.55, -46.63),
    "miami": (25.76, -80.19), "paris": (48.86, 2.35),
    "dallas": (32.78, -96.80), "seattle": (47.61, -122.33),
    "munich": (48.14, 11.58), "tokyo": (35.68, 139.65),
    "toronto": (43.65, -79.38), "atlanta": (33.75, -84.39),
    "wellington": (-41.29, 174.78), "buenos_aires": (-34.60, -58.38),
    "tel_aviv": (32.09, 34.78), "lucknow": (26.85, 80.95),
    "los_angeles": (34.05, -118.24),
}


def fetch_actual_tmax(lat: float, lon: float, start: str, end: str) -> dict[str, float]:
    """Fetch actual observed TMAX from Open-Meteo archive API."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max"
        f"&timezone=auto"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
        data = json.loads(resp.read().decode())

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    result = {}
    for d, t in zip(dates, temps):
        if t is not None:
            result[d] = round(t, 2)
    return result


def main():
    import psycopg2

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip("/") or "arbo",
        user=parsed.username or "arbo",
        password=parsed.password or "",
    )

    # Create training_obs table if not exists
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_training_obs (
            city VARCHAR(50) NOT NULL,
            target_date VARCHAR(10) NOT NULL,
            forecast_temp FLOAT NOT NULL,
            actual_temp FLOAT NOT NULL,
            source VARCHAR(20) DEFAULT 'gefs_openmeteo',
            PRIMARY KEY (city, target_date)
        )
    """)
    conn.commit()

    # Load GEFS ensemble_mean as "forecast"
    cur.execute("SELECT city, target_date, ensemble_mean FROM ensemble_stats ORDER BY city, target_date")
    gefs_data: dict[str, dict[str, float]] = {}
    for city, dt, mean in cur.fetchall():
        if city not in gefs_data:
            gefs_data[city] = {}
        gefs_data[city][dt] = mean

    print(f"GEFS forecasts: {sum(len(v) for v in gefs_data.values())} across {len(gefs_data)} cities")

    # Fetch actuals from Open-Meteo per city
    total_inserted = 0
    for city, (lat, lon) in CITIES.items():
        if city not in gefs_data:
            continue

        dates = sorted(gefs_data[city].keys())
        start_date = dates[0]
        # Open-Meteo archive has ~5 day lag
        end_date = (date.today() - timedelta(days=5)).isoformat()

        print(f"  {city}: fetching actuals {start_date} to {end_date}...", end=" ", flush=True)
        try:
            actuals = fetch_actual_tmax(lat, lon, start_date, end_date)
        except Exception as e:
            print(f"FAILED ({e})")
            time.sleep(1)
            continue

        # Match GEFS forecast → Open-Meteo actual
        matched = 0
        for dt, forecast in gefs_data[city].items():
            actual = actuals.get(dt)
            if actual is not None:
                cur.execute(
                    """INSERT INTO weather_training_obs (city, target_date, forecast_temp, actual_temp)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (city, target_date) DO UPDATE SET
                        forecast_temp = EXCLUDED.forecast_temp,
                        actual_temp = EXCLUDED.actual_temp
                    """,
                    (city, dt, forecast, actual),
                )
                matched += 1

        conn.commit()
        total_inserted += matched
        print(f"{matched}/{len(dates)} matched")
        time.sleep(0.5)  # Rate limit

    print(f"\nTotal: {total_inserted} training observations inserted")

    # Verify
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT city) FROM weather_training_obs")
    count, cities = cur.fetchone()
    print(f"Verified: {count} rows, {cities} cities")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
