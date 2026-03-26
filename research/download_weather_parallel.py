"""Parallel weather data download — 5 city groups for 5× speed.

Splits 20 cities into 5 groups of 4 and downloads each group in a
separate subprocess. Each worker handles 4 cities sequentially.

~1700 RPM available (2000 Ultra - ~300 used by sports download).
5 workers × ~340 RPM each = safe.

Usage:
    PYTHONPATH=. python3 research/download_weather_parallel.py
    PYTHONPATH=. python3 research/download_weather_parallel.py --workers 4
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CITIES = [
    "nyc", "chicago", "london", "seoul", "buenos_aires",
    "atlanta", "toronto", "ankara", "sao_paulo", "miami",
    "paris", "dallas", "seattle", "wellington", "tokyo",
    "munich", "los_angeles", "dc", "tel_aviv", "lucknow",
]

PYTHON = sys.executable
SCRIPT = str(Path(__file__).parent / "download_weather_pmd.py")
LOG_DIR = Path(__file__).parent / "data"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resolution", default="1m")
    args = parser.parse_args()

    # Split cities into groups
    chunk_size = math.ceil(len(CITIES) / args.workers)
    groups = [CITIES[i:i + chunk_size] for i in range(0, len(CITIES), chunk_size)]

    print(f"{'='*50}")
    print(f"  Parallel Weather Download — {args.workers} workers")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*50}")
    for i, group in enumerate(groups):
        print(f"  Worker {i}: {', '.join(group)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    start = time.time()

    for i, group in enumerate(groups):
        log_path = LOG_DIR / f"weather_worker_{i}.log"
        # Each worker downloads its cities sequentially
        # The download_weather_pmd.py --city flag takes one city at a time
        # So we chain them in a shell command
        city_cmds = " && ".join(
            f"PYTHONPATH=. {PYTHON} {SCRIPT} --city {city} --resolution {args.resolution}"
            for city in group
        )
        p = subprocess.Popen(
            ["bash", "-c", city_cmds],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        processes.append((i, p, group, log_path))
        print(f"  Started worker {i} (PID {p.pid}): {', '.join(group)}")

    # Monitor
    print(f"\nAll {len(processes)} workers running. Monitoring...")
    while True:
        time.sleep(60)
        alive = sum(1 for _, p, _, _ in processes if p.poll() is None)
        elapsed_h = (time.time() - start) / 3600

        # Count from progress file
        progress_path = LOG_DIR / "weather_pmd_progress.txt"
        done = 0
        if progress_path.exists():
            done = len([l for l in progress_path.read_text().splitlines() if l.strip()])

        print(f"  [{elapsed_h:.1f}h] Done: {done}/18,700 "
              f"({done/18700*100:.1f}%) Workers: {alive}")

        if alive == 0:
            break

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"  DONE in {elapsed/3600:.1f} hours")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
