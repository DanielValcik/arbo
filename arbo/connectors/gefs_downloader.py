"""GEFS Ensemble Downloader for Production.

Downloads daily TMAX ensemble forecasts (31 GFS members) from NOAA S3.
Uses HTTP Range requests with .idx files to extract only TMAX (~400KB vs 17MB).

Designed to run as daily cron on VPS:
    0 8 * * * cd /opt/arbo && python3 -m arbo.connectors.gefs_downloader

Stores results in PostgreSQL (ensemble_forecasts, ensemble_stats tables).
"""

from __future__ import annotations

import asyncio
import ssl
import statistics
import tempfile
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]  # 31 total

# Cities with their coordinates (matching weather_scanner.py)
CITIES: dict[str, tuple[float, float]] = {
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

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()


def _fetch_idx(run_date: str, member: str) -> str | None:
    """Fetch .idx index file to find TMAX byte range."""
    fname = f"{member}.t00z.pgrb2s.0p25.f024.idx"
    url = f"{S3_BASE}/gefs.{run_date}/00/atmos/pgrb2sp25/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
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
            return (start, end) if end else None
    return None


def download_tmax_member(
    run_date: str,
    member: str,
    cities: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Download TMAX for one member via Range request (~400KB)."""
    import warnings
    warnings.filterwarnings("ignore")

    idx = _fetch_idx(run_date, member)
    if idx is None:
        return {}

    byte_range = _find_tmax_range(idx)
    if byte_range is None:
        return {}

    fname = f"{member}.t00z.pgrb2s.0p25.f024"
    url = f"{S3_BASE}/gefs.{run_date}/00/atmos/pgrb2sp25/{fname}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Arbo/1.0",
        "Range": f"bytes={byte_range[0]}-{byte_range[1]}",
    })

    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            tmax_data = resp.read()
    except Exception:
        return {}

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


async def download_today(
    session: AsyncSession | None = None,
) -> dict[str, dict[str, float]]:
    """Download ensemble forecasts for today.

    run_date = yesterday (00Z forecast for today's TMAX).
    Returns {city: {member: temp_celsius}}.
    """
    today = date.today()
    run_date = (today - timedelta(days=1)).strftime("%Y%m%d")
    target_date = today.isoformat()

    log = logger.bind(target_date=target_date, run_date=run_date)
    log.info("gefs_download_start", n_members=len(MEMBERS))

    # Download in executor to not block event loop
    loop = asyncio.get_event_loop()
    member_temps: dict[str, dict[str, float]] = {}
    success = 0

    for member in MEMBERS:
        temps = await loop.run_in_executor(
            None, download_tmax_member, run_date, member, CITIES
        )
        if temps:
            member_temps[member] = temps
            success += 1

    log.info("gefs_download_complete", members_ok=success, total=len(MEMBERS))

    # Compute ensemble stats per city
    ensemble_stats: dict[str, dict[str, float]] = {}
    for city in CITIES:
        vals = [
            member_temps[m][city]
            for m in member_temps
            if city in member_temps[m]
        ]
        if len(vals) >= 5:
            ensemble_stats[city] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "n_members": len(vals),
            }

    # Store to PostgreSQL if session provided
    if session is not None:
        await _store_to_db(session, target_date, member_temps, ensemble_stats)

    log.info(
        "gefs_stats_computed",
        cities_with_stats=len(ensemble_stats),
        avg_std=round(
            statistics.mean(s["std"] for s in ensemble_stats.values()), 3
        ) if ensemble_stats else 0,
    )

    return ensemble_stats


async def _store_to_db(
    session: AsyncSession,
    target_date: str,
    member_temps: dict[str, dict[str, float]],
    ensemble_stats: dict[str, dict[str, float]],
) -> None:
    """Store ensemble data to PostgreSQL."""
    from sqlalchemy import text

    # Store individual member forecasts
    for member, temps in member_temps.items():
        for city, temp in temps.items():
            await session.execute(
                text("""
                    INSERT INTO ensemble_forecasts (city, target_date, member, tmax_c)
                    VALUES (:city, :date, :member, :temp)
                    ON CONFLICT (city, target_date, member) DO UPDATE SET tmax_c = :temp
                """),
                {"city": city, "date": target_date, "member": member, "temp": temp},
            )

    # Store ensemble stats
    for city, stats in ensemble_stats.items():
        await session.execute(
            text("""
                INSERT INTO ensemble_stats
                    (city, target_date, ensemble_mean, ensemble_std,
                     ensemble_min, ensemble_max, n_members)
                VALUES (:city, :date, :mean, :std, :min, :max, :n)
                ON CONFLICT (city, target_date) DO UPDATE SET
                    ensemble_mean = :mean, ensemble_std = :std,
                    ensemble_min = :min, ensemble_max = :max, n_members = :n
            """),
            {
                "city": city, "date": target_date,
                "mean": stats["mean"], "std": stats["std"],
                "min": stats["min"], "max": stats["max"],
                "n": stats["n_members"],
            },
        )

    await session.commit()
    logger.info("gefs_stored_to_db", target_date=target_date)


def get_ensemble_std(
    city: str,
    target_date: str,
    ensemble_cache: dict[str, dict[str, float]],
) -> float | None:
    """Get ensemble_std for use in EMOSEnsembleModel.

    Called by weather_scanner during signal evaluation.
    """
    city_stds = ensemble_cache.get(city, {})
    if target_date in city_stds:
        return city_stds[target_date]
    # Fallback: nearest prior date
    prior = [d for d in sorted(city_stds.keys()) if d <= target_date]
    return city_stds[prior[-1]] if prior else None


# ── CLI entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    async def _main():
        stats = await download_today()
        print(f"\nEnsemble stats for {date.today()}:")
        for city in sorted(stats, key=lambda c: stats[c]["std"]):
            s = stats[city]
            print(f"  {city:<14} mean={s['mean']:>6.1f}°C  "
                  f"std={s['std']:>5.2f}°C  [{s['min']:.1f}, {s['max']:.1f}]")

    asyncio.run(_main())
