"""Download GEFS Ensemble TMAX Data from AWS S3.

Downloads GFS ensemble (31 members) temperature forecasts for 20 cities,
extracts TMAX at nearest grid point, stores in SQLite for EMOS backtesting.

Source: s3://noaa-gefs-pds/ (free, no auth)
Format: GRIB2 with .idx index files
Members: gec00 (control) + gep01-gep30 (30 perturbed)

Usage:
    python3 research/innovations/download_gefs_ensemble.py
    python3 research/innovations/download_gefs_ensemble.py --start 2026-01-01 --end 2026-03-16
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import ssl
import struct
import sys
import tempfile
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

# ── Constants ────────────────────────────────────────────────────────

S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# Pattern: gefs.YYYYMMDD/HH/atmos/pgrb2sp25/gepNN.tHHz.pgrb2s.0p25.f024
# For TMAX we want f024 (24-hour forecast) from the 00Z run

CITIES = {
    "nyc": (40.71, -74.01),
    "chicago": (41.88, -87.63),
    "london": (51.51, -0.13),
    "seoul": (37.57, 126.98),
    "ankara": (39.93, 32.86),
    "sao_paulo": (-23.55, -46.63),
    "miami": (25.76, -80.19),
    "paris": (48.86, 2.35),
    "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33),
    "munich": (48.14, 11.58),
    "tokyo": (35.68, 139.65),
    "toronto": (43.65, -79.38),
    "atlanta": (33.75, -84.39),
    "wellington": (-41.29, 174.78),
    "buenos_aires": (-34.60, -58.38),
    "tel_aviv": (32.09, 34.78),
    "lucknow": (26.85, 80.95),
    "los_angeles": (34.05, -118.24),
}

MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]  # 31 total

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "gefs_ensemble.sqlite"

# ── Database ─────────────────────────────────────────────────────────


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_forecasts (
            city TEXT NOT NULL,
            target_date TEXT NOT NULL,
            member TEXT NOT NULL,
            tmax_c REAL,
            PRIMARY KEY (city, target_date, member)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ensemble_stats (
            city TEXT NOT NULL,
            target_date TEXT NOT NULL,
            ensemble_mean REAL,
            ensemble_std REAL,
            ensemble_min REAL,
            ensemble_max REAL,
            n_members INTEGER,
            PRIMARY KEY (city, target_date)
        )
    """)
    conn.commit()
    return conn


# ── GRIB2 Download & Parse ───────────────────────────────────────────


def _fetch_idx(run_date: str, member: str, cycle: str = "00", fhour: int = 24) -> str | None:
    """Fetch .idx index file content."""
    fname = f"{member}.t{cycle}z.pgrb2s.0p25.f{fhour:03d}.idx"
    url = f"{S3_BASE}/gefs.{run_date}/{cycle}/atmos/pgrb2sp25/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _find_tmax_range(idx_content: str) -> tuple[int, int] | None:
    """Parse .idx to find TMAX byte range."""
    lines = idx_content.strip().split("\n")
    for i, line in enumerate(lines):
        if ":TMAX:" in line:
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            if end is None:
                return None
            return (start, end)
    return None


def download_tmax_member(
    run_date: str,
    member: str,
    cities: dict[str, tuple[float, float]],
    cycle: str = "00",
    fhour: int = 24,
) -> dict[str, float]:
    """Download TMAX for one member using Range request (~400KB vs 17MB).

    Returns: {city: temp_celsius}
    """
    import warnings
    warnings.filterwarnings("ignore")

    # 1. Get .idx to find byte range
    idx = _fetch_idx(run_date, member, cycle, fhour)
    if idx is None:
        return {}

    byte_range = _find_tmax_range(idx)
    if byte_range is None:
        return {}

    # 2. Download just TMAX bytes
    fname = f"{member}.t{cycle}z.pgrb2s.0p25.f{fhour:03d}"
    url = f"{S3_BASE}/gefs.{run_date}/{cycle}/atmos/pgrb2sp25/{fname}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "ArboResearch/1.0",
        "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
    })

    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            tmax_data = resp.read()
    except Exception:
        return {}

    # 3. Parse GRIB2 message
    tmp = Path(tempfile.mkdtemp()) / "tmax.grib2"
    tmp.write_bytes(tmax_data)

    try:
        import cfgrib
        datasets = cfgrib.open_datasets(str(tmp))
        results = {}
        for ds in datasets:
            if "tmax" in ds.data_vars:
                tmax = ds["tmax"]
                for city, (lat, lon) in cities.items():
                    try:
                        val = float(tmax.sel(
                            latitude=lat, longitude=lon % 360, method="nearest"
                        ).values)
                        results[city] = round(val - 273.15, 2)
                    except Exception:
                        pass
                ds.close()
                break
        return results
    except Exception:
        return {}
    finally:
        try:
            tmp.unlink()
            tmp.parent.rmdir()
        except OSError:
            pass


# ── Main Pipeline ────────────────────────────────────────────────────


def download_date(
    conn: sqlite3.Connection,
    target_date: str,
    cities: dict,
) -> int:
    """Download all 31 members for one target date.

    run_date = target_date - 1 day (we want yesterday's forecast for today).
    Returns number of members successfully downloaded.
    """
    # Run date = day before target (forecast made yesterday for today)
    td = date.fromisoformat(target_date)
    run_date = (td - timedelta(days=1)).strftime("%Y%m%d")

    # Check if already done
    existing = conn.execute(
        "SELECT COUNT(DISTINCT member) FROM ensemble_forecasts WHERE target_date=?",
        (target_date,),
    ).fetchone()[0]
    if existing >= 25:  # At least 25 of 31 members
        return existing

    success = 0
    member_temps: dict[str, dict[str, float]] = {}  # member -> {city -> temp}

    for member in MEMBERS:
        temps = download_tmax_member(run_date, member, cities)
        if temps:
            member_temps[member] = temps
            success += 1

            # Store individual member values
            for city, temp in temps.items():
                conn.execute(
                    "INSERT OR REPLACE INTO ensemble_forecasts "
                    "(city, target_date, member, tmax_c) VALUES (?, ?, ?, ?)",
                    (city, target_date, member, temp),
                )

        time.sleep(0.05)  # Be nice to S3

    # Compute and store ensemble stats per city
    if member_temps:
        for city in cities:
            vals = [
                member_temps[m][city]
                for m in member_temps
                if city in member_temps[m]
            ]
            if len(vals) >= 5:
                import statistics
                conn.execute(
                    "INSERT OR REPLACE INTO ensemble_stats "
                    "(city, target_date, ensemble_mean, ensemble_std, "
                    "ensemble_min, ensemble_max, n_members) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        city,
                        target_date,
                        round(statistics.mean(vals), 3),
                        round(statistics.stdev(vals), 3),
                        round(min(vals), 3),
                        round(max(vals), 3),
                        len(vals),
                    ),
                )

    conn.commit()
    return success


def main():
    parser = argparse.ArgumentParser(description="Download GEFS ensemble data")
    parser.add_argument("--start", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-16", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cities", default=None, help="Comma-separated city list")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db(DB_PATH)

    cities = CITIES
    if args.cities:
        selected = set(args.cities.split(","))
        cities = {k: v for k, v in CITIES.items() if k in selected}

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    total_days = (end - start).days + 1

    print(f"GEFS Ensemble Download")
    print(f"  Period: {args.start} to {args.end} ({total_days} days)")
    print(f"  Cities: {len(cities)}")
    print(f"  Members: {len(MEMBERS)} per day")
    print(f"  DB: {DB_PATH}")
    print()

    t_start = time.time()
    completed = 0

    current = start
    while current <= end:
        target = current.isoformat()
        n = download_date(conn, target, cities)
        completed += 1

        if n > 0:
            print(f"  {target}: {n}/{len(MEMBERS)} members OK  "
                  f"[{completed}/{total_days}]", flush=True)
        else:
            print(f"  {target}: FAILED (no data)  [{completed}/{total_days}]", flush=True)

        current += timedelta(days=1)
        time.sleep(0.1)  # Be nice to S3

    # Summary
    elapsed = time.time() - t_start
    total_forecasts = conn.execute("SELECT COUNT(*) FROM ensemble_forecasts").fetchone()[0]
    total_stats = conn.execute("SELECT COUNT(*) FROM ensemble_stats").fetchone()[0]

    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Forecasts: {total_forecasts:,} (member-level)")
    print(f"  Stats: {total_stats:,} (city-date-level)")

    conn.close()


if __name__ == "__main__":
    main()
