"""One-time fix: resolve ghost trades stuck as 'open' in paper_trades DB.

These trades were resolved in-memory (position removed) but the DB update
failed because self._trades was empty after VPS restart (bug fixed in f6946a4).

This script:
1. Finds ghost trades (open in paper_trades, no matching paper_position)
2. Fetches market state from Gamma API for each
3. Determines outcome (closed market → price, or METAR for weather)
4. Updates paper_trades DB with correct status/pnl/resolved_at
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import aiohttp
import sqlalchemy as sa

from arbo.utils.db import PaperTrade as PT
from arbo.utils.db import get_session_factory
from arbo.utils.logger import get_logger

logger = get_logger("fix_ghost_trades")


async def fix_ghost_trades(dry_run: bool = True) -> None:
    """Fix ghost trades by fetching market state from Gamma API."""
    factory = get_session_factory()

    async with factory() as session:
        # Find ghost trades: open in paper_trades but no paper_position
        result = await session.execute(
            sa.text("""
                SELECT pt.id, pt.strategy, pt.token_id, pt.placed_at,
                       pt.price, pt.size, pt.side, pt.trade_details,
                       pt.market_condition_id
                FROM paper_trades pt
                LEFT JOIN paper_positions pp ON pt.token_id = pp.token_id
                WHERE pt.status = 'open' AND pp.token_id IS NULL
                ORDER BY pt.placed_at ASC
            """)
        )
        ghosts = result.fetchall()
        print(f"Found {len(ghosts)} ghost trades")

        if not ghosts:
            return

        resolved = 0
        failed = 0

        async with aiohttp.ClientSession() as http:
            for g in ghosts:
                trade_id, strategy, token_id, placed_at = g[0], g[1], g[2], g[3]
                entry_price, size, side = g[4], g[5], g[6]
                trade_details = g[7] or {}

                # Fetch from Gamma API
                try:
                    url = "https://gamma-api.polymarket.com/markets"
                    params = {"clob_token_ids": token_id, "limit": "1"}
                    async with http.get(url, params=params) as resp:
                        if resp.status != 200:
                            print(f"  #{trade_id}: Gamma API HTTP {resp.status}")
                            failed += 1
                            continue
                        data = await resp.json()
                        if not isinstance(data, list) or not data:
                            print(f"  #{trade_id}: Gamma returned empty")
                            failed += 1
                            continue

                    market = data[0]
                    closed = market.get("closed", False)
                    question = market.get("question", "")[:60]

                    import json

                    outcome_prices = market.get("outcomePrices", "[]")
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)

                    clob_token_ids = market.get("clobTokenIds", "[]")
                    if isinstance(clob_token_ids, str):
                        clob_token_ids = json.loads(clob_token_ids)

                    if not closed:
                        print(f"  #{trade_id}: market still open — skipping ({question})")
                        failed += 1
                        continue

                    # Determine outcome
                    yes_price = float(outcome_prices[0]) if outcome_prices else None
                    token_yes = clob_token_ids[0] if clob_token_ids else None
                    token_no = clob_token_ids[1] if len(clob_token_ids) > 1 else None

                    if yes_price is None:
                        print(f"  #{trade_id}: no price data")
                        failed += 1
                        continue

                    yes_won = yes_price > 0.5
                    if token_id == token_yes:
                        winning = yes_won
                    elif token_id == token_no:
                        winning = not yes_won
                    else:
                        print(f"  #{trade_id}: token_id doesn't match YES or NO")
                        failed += 1
                        continue

                    # Calculate P&L (same as paper_engine.resolve_market)
                    entry_dec = Decimal(str(entry_price))
                    size_dec = Decimal(str(size))
                    shares = size_dec / entry_dec if entry_dec > 0 else Decimal("0")

                    if side == "BUY":
                        if winning:
                            pnl = shares * Decimal("1") - size_dec
                        else:
                            pnl = -size_dec
                    else:
                        if winning:
                            pnl = -shares * (Decimal("1") - entry_dec)
                        else:
                            pnl = shares * entry_dec

                    status_str = "won" if winning else "lost"
                    pnl_float = float(pnl)

                    print(
                        f"  #{trade_id} strat={strategy} {status_str} "
                        f"pnl=${pnl_float:+.2f} ({question})"
                    )

                    if not dry_run:
                        await session.execute(
                            sa.update(PT)
                            .where(PT.id == trade_id)
                            .values(
                                status=status_str,
                                actual_pnl=pnl,
                                resolved_at=datetime.now(UTC),
                            )
                        )
                    resolved += 1

                except Exception as e:
                    print(f"  #{trade_id}: ERROR {e}")
                    failed += 1

        if not dry_run:
            await session.commit()

        print(f"\n{'DRY RUN — ' if dry_run else ''}Resolved: {resolved}, Failed: {failed}")


if __name__ == "__main__":
    import sys

    dry = "--apply" not in sys.argv
    if dry:
        print("=== DRY RUN (use --apply to write to DB) ===\n")
    else:
        print("=== APPLYING CHANGES TO DB ===\n")
    asyncio.run(fix_ghost_trades(dry_run=dry))
