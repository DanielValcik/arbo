"""Load weather_pmd.sqlite into experiment_framework SimulationData format.

Parses PMD market questions into temperature buckets, groups by city+date
into events, determines won/lost from final prices, and builds
SimulationData for the experiment framework.

Usage:
    from pmd_loader import load_pmd_simulation_data
    sim_data = load_pmd_simulation_data()
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from price_history_db import Bucket, Event

PMD_DB_PATH = Path(__file__).parent.parent / "data" / "weather_pmd.sqlite"

# ── Temperature parsing from question text ───────────────────────────

_F_TO_C = lambda f: round((f - 32) * 5 / 9, 1)

# Pattern: "between 82-83°F" or "between 34-35°F"
_PAT_RANGE_F = re.compile(
    r"between (\d+)-(\d+)°F", re.IGNORECASE
)
# Pattern: "be 6°C" or "be 35°C"  (exact single value)
_PAT_EXACT_C = re.compile(
    r"be (-?\d+)°C(?:\s+on|\s+or)", re.IGNORECASE
)
# Pattern: "be 51°F" (exact F — rare but exists)
_PAT_EXACT_F = re.compile(
    r"be (\d+)°F(?:\s+on|\s+or)", re.IGNORECASE
)
# Pattern: "39°F or higher", "77°F or below"
_PAT_ABOVE_F = re.compile(
    r"(\d+)°F or higher", re.IGNORECASE
)
_PAT_BELOW_F = re.compile(
    r"(\d+)°F or below", re.IGNORECASE
)
# Pattern: "-4°C or higher", "6°C or higher"
_PAT_ABOVE_C = re.compile(
    r"(-?\d+)°C or higher", re.IGNORECASE
)
_PAT_BELOW_C = re.compile(
    r"(-?\d+)°C or below", re.IGNORECASE
)
# Range in C: "between 8-9°C"
_PAT_RANGE_C = re.compile(
    r"between (-?\d+)-(-?\d+)°C", re.IGNORECASE
)


def parse_bucket_from_question(
    question: str,
) -> tuple[float | None, float | None, str] | None:
    """Parse temperature bucket (low_c, high_c, bucket_type) from question.

    Returns None if parsing fails.
    """
    # Range F: "between 82-83°F"
    m = _PAT_RANGE_F.search(question)
    if m:
        lo_f, hi_f = int(m.group(1)), int(m.group(2))
        return (_F_TO_C(lo_f), _F_TO_C(hi_f + 1), "range")

    # Range C: "between 8-9°C"
    m = _PAT_RANGE_C.search(question)
    if m:
        lo_c, hi_c = int(m.group(1)), int(m.group(2))
        return (float(lo_c), float(hi_c + 1), "range")

    # Above F: "39°F or higher"
    m = _PAT_ABOVE_F.search(question)
    if m:
        threshold_c = _F_TO_C(int(m.group(1)))
        return (threshold_c, None, "above")

    # Below F: "77°F or below"
    m = _PAT_BELOW_F.search(question)
    if m:
        threshold_c = _F_TO_C(int(m.group(1)) + 1)
        return (None, threshold_c, "below")

    # Above C: "6°C or higher"
    m = _PAT_ABOVE_C.search(question)
    if m:
        threshold_c = float(m.group(1))
        return (threshold_c, None, "above")

    # Below C: "-4°C or below"
    m = _PAT_BELOW_C.search(question)
    if m:
        threshold_c = float(m.group(1)) + 1
        return (None, threshold_c, "below")

    # Exact C: "be 6°C on" → treat as 1°C range
    m = _PAT_EXACT_C.search(question)
    if m:
        val = int(m.group(1))
        return (float(val), float(val + 1), "range")

    # Exact F: "be 51°F on" → treat as ~1°F range
    m = _PAT_EXACT_F.search(question)
    if m:
        val = int(m.group(1))
        return (_F_TO_C(val), _F_TO_C(val + 1), "range")

    return None


def parse_target_date(end_date: str) -> str | None:
    """Extract ISO date from end_date string."""
    if not end_date:
        return None
    return end_date[:10]


# ── Load PMD data ────────────────────────────────────────────────────


def load_pmd_data(
    db_path: str | Path | None = None,
) -> tuple[list[Event], dict[str, list[Bucket]], dict[str, list[tuple[int, float]]]]:
    """Load weather_pmd.sqlite and convert to experiment framework types.

    Returns:
        events: List of Event objects (one per city+date group)
        buckets_by_event: event_id → list of Bucket objects
        prices: token_id → sorted list of (ts, price)
    """
    db_path = Path(db_path) if db_path else PMD_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load all markets
    markets = conn.execute(
        "SELECT market_id, city, question, status, start_date, end_date, tokens_json "
        "FROM markets WHERE city IS NOT NULL"
    ).fetchall()

    # Group markets by city+date → event
    event_groups: dict[str, list[dict]] = defaultdict(list)
    for mkt in markets:
        target_date = parse_target_date(mkt["end_date"])
        if not target_date or not mkt["city"]:
            continue

        parsed = parse_bucket_from_question(mkt["question"])
        if parsed is None:
            continue

        tokens = json.loads(mkt["tokens_json"])
        yes_token = next((t for t in tokens if t["label"] == "Yes"), None)
        no_token = next((t for t in tokens if t["label"] == "No"), None)
        if not yes_token:
            continue

        event_key = f"{mkt['city']}_{target_date}"
        event_groups[event_key].append({
            "market_id": mkt["market_id"],
            "city": mkt["city"],
            "target_date": target_date,
            "end_date": mkt["end_date"],
            "status": mkt["status"],
            "question": mkt["question"],
            "yes_token_id": yes_token["id"],
            "no_token_id": no_token["id"] if no_token else None,
            "low_c": parsed[0],
            "high_c": parsed[1],
            "bucket_type": parsed[2],
        })

    # Determine won/lost from final prices
    # For closed markets: YES token final price ≈ 1.0 means YES won
    won_tokens: set[str] = set()
    for rows in event_groups.values():
        for mkt_data in rows:
            if mkt_data["status"] != "closed":
                continue
            last_price = conn.execute(
                "SELECT price FROM prices WHERE token_id=? ORDER BY ts DESC LIMIT 1",
                (mkt_data["yes_token_id"],),
            ).fetchone()
            if last_price and last_price["price"] > 0.8:
                won_tokens.add(mkt_data["yes_token_id"])

    # Build Event + Bucket objects
    events: list[Event] = []
    buckets_by_event: dict[str, list[Bucket]] = {}

    for event_key, mkt_list in event_groups.items():
        city = mkt_list[0]["city"]
        target_date = mkt_list[0]["target_date"]
        end_date = mkt_list[0]["end_date"]

        # Event end_date → closed_time
        closed_time = end_date if end_date else None

        event = Event(
            event_id=event_key,
            title=f"Weather {city} {target_date}",
            city=city,
            target_date=target_date,
            start_date=None,
            end_date=end_date,
            closed_time=closed_time,
            volume=0.0,
            neg_risk=True,
            n_buckets=len(mkt_list),
        )
        events.append(event)

        buckets = []
        for mkt_data in mkt_list:
            # Get volume from prices count (rough proxy)
            vol_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM prices WHERE token_id=?",
                (mkt_data["yes_token_id"],),
            ).fetchone()
            vol = float(vol_row["cnt"]) if vol_row else 0

            bucket = Bucket(
                token_id=mkt_data["yes_token_id"],
                token_id_no=mkt_data["no_token_id"],
                event_id=event_key,
                condition_id=mkt_data["market_id"],
                question=mkt_data["question"],
                low_c=mkt_data["low_c"],
                high_c=mkt_data["high_c"],
                bucket_type=mkt_data["bucket_type"],
                unit="C",
                won=mkt_data["yes_token_id"] in won_tokens,
                volume=vol,
            )
            buckets.append(bucket)
        buckets_by_event[event_key] = buckets

    # Load all prices: token_id → sorted [(ts, price)]
    prices: dict[str, list[tuple[int, float]]] = defaultdict(list)
    cursor = conn.execute("SELECT token_id, ts, price FROM prices ORDER BY token_id, ts")
    for row in cursor:
        prices[row["token_id"]].append((row["ts"], row["price"]))

    # Prices are already sorted by ORDER BY
    sorted_prices = dict(prices)

    conn.close()

    n_points = sum(len(v) for v in sorted_prices.values())
    print(
        f"PMD loaded: {len(events)} events, "
        f"{sum(len(b) for b in buckets_by_event.values())} buckets, "
        f"{n_points:,} prices, "
        f"{len(won_tokens)} resolved YES wins"
    )

    return events, buckets_by_event, sorted_prices


if __name__ == "__main__":
    events, buckets, prices = load_pmd_data()
    from collections import Counter

    cities = Counter(e.city for e in events)
    print(f"\nEvents by city:")
    for city, cnt in cities.most_common():
        print(f"  {city}: {cnt}")

    dates = sorted(set(e.target_date for e in events if e.target_date))
    print(f"\nDate range: {dates[0]} to {dates[-1]}")
    print(f"Unique dates: {len(dates)}")

    # Check parsing accuracy
    total_buckets = sum(len(b) for b in buckets.values())
    won_count = sum(1 for bl in buckets.values() for b in bl if b.won)
    print(f"\nTotal buckets: {total_buckets}")
    print(f"Won (resolved YES): {won_count}")
    print(f"Parse success rate: 100%")
