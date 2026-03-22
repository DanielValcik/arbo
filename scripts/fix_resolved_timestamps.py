"""Fix resolved_at timestamps for ghost trades using Gamma API closedTime.

The ghost trade fix script set all resolved_at to the same time (script runtime).
This script fetches the actual closedTime from Gamma API for each resolved trade.
"""

import asyncio
from datetime import datetime, timezone

import aiohttp
import sqlalchemy as sa

from arbo.utils.db import PaperTrade as PT
from arbo.utils.db import get_session_factory


async def fix_timestamps(dry_run: bool = True) -> None:
    """Update resolved_at using Gamma API closedTime."""
    factory = get_session_factory()

    async with factory() as session:
        # Find trades resolved by ghost fix (all have same resolved_at within 1 minute)
        result = await session.execute(
            sa.select(PT.id, PT.token_id, PT.resolved_at, PT.status)
            .where(PT.status.in_(["won", "lost"]))
            .where(PT.resolved_at.isnot(None))
            .order_by(PT.resolved_at.desc())
        )
        trades = result.fetchall()

        # Group by resolved_at to find the batch (all same timestamp)
        from collections import Counter

        ts_counts = Counter()
        for t in trades:
            if t[2]:
                ts_key = t[2].strftime("%Y-%m-%d %H:%M")
                ts_counts[ts_key] += 1

        # Find batch timestamp (the one with most trades)
        if not ts_counts:
            print("No resolved trades found")
            return

        batch_ts = ts_counts.most_common(1)[0][0]
        batch_count = ts_counts.most_common(1)[0][1]
        print(f"Batch timestamp: {batch_ts} ({batch_count} trades)")

        # Get trades with that batch timestamp
        batch_trades = [
            t for t in trades if t[2] and t[2].strftime("%Y-%m-%d %H:%M") == batch_ts
        ]
        print(f"Fixing {len(batch_trades)} trades\n")

        updated = 0
        failed = 0

        async with aiohttp.ClientSession() as http:
            for t in batch_trades:
                trade_id, token_id = t[0], t[1]

                try:
                    url = "https://gamma-api.polymarket.com/markets"
                    params = {"clob_token_ids": token_id, "limit": "1"}
                    async with http.get(url, params=params) as resp:
                        if resp.status != 200:
                            failed += 1
                            continue
                        data = await resp.json()
                        if not isinstance(data, list) or not data:
                            failed += 1
                            continue

                    market = data[0]
                    closed_time_str = market.get("closedTime")
                    end_date_str = market.get("endDate")

                    # Use closedTime first, fallback to endDate
                    actual_resolved = None
                    for ts_str in [closed_time_str, end_date_str]:
                        if ts_str:
                            try:
                                actual_resolved = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                )
                                break
                            except (ValueError, TypeError):
                                continue

                    if actual_resolved is None:
                        question = market.get("question", "")[:50]
                        print(f"  #{trade_id}: no timestamp ({question})")
                        failed += 1
                        continue

                    question = market.get("question", "")[:50]
                    print(
                        f"  #{trade_id}: {actual_resolved.strftime('%Y-%m-%d %H:%M')} "
                        f"({question})"
                    )

                    if not dry_run:
                        await session.execute(
                            sa.update(PT)
                            .where(PT.id == trade_id)
                            .values(resolved_at=actual_resolved)
                        )
                    updated += 1

                except Exception as e:
                    print(f"  #{trade_id}: ERROR {e}")
                    failed += 1

        if not dry_run:
            await session.commit()

        print(f"\n{'DRY RUN — ' if dry_run else ''}Updated: {updated}, Failed: {failed}")


if __name__ == "__main__":
    import sys

    dry = "--apply" not in sys.argv
    if dry:
        print("=== DRY RUN (use --apply to write to DB) ===\n")
    else:
        print("=== APPLYING CHANGES TO DB ===\n")
    asyncio.run(fix_timestamps(dry_run=dry))
