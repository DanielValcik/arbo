"""One-shot: force-close all B2 pre_reset live positions on CLOB.

Sells every "above" market position still held in the wallet to free up
real USDC. Uses live_executor.sell with taker fallback so thin maker
liquidity doesn't block the exit. Logs each result with realized PnL.

After this runs, wallet_balance should rise by ~sum(shares × current_bid),
B2 can resume trading with meaningful per-trade sizing, and the pre_reset
chapter is fully closed on-chain (no more drift between DB and wallet).

Run on the VPS inside the arbo venv:

    sudo -u arbo /opt/arbo/.venv/bin/python scripts/force_close_b2_pre_reset.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

from arbo.connectors.polymarket_client import PolymarketClient
from arbo.core.live_executor import LiveExecutor
from arbo.utils.logger import get_logger

logger = get_logger("force_close_b2")

DATA_API = "https://data-api.polymarket.com"


async def fetch_b2_positions(funder: str) -> list[dict]:
    """Return all B2-style 'above' positions currently held in the wallet."""
    url = f"{DATA_API}/positions?user={funder}&limit=500"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    b2 = [
        p for p in data
        if "above" in (p.get("title", "") or "").lower()
        and float(p.get("size", 0)) > 0
    ]
    return b2


async def main() -> int:
    import os
    funder = os.getenv("POLY_FUNDER_ADDRESS", "")
    if not funder:
        print("ERROR: POLY_FUNDER_ADDRESS not set", file=sys.stderr)
        return 1

    print(f"Funder: {funder}")
    print("Fetching wallet positions...")
    positions = await fetch_b2_positions(funder)
    print(f"Found {len(positions)} B2 'above' positions in wallet")
    if not positions:
        print("Nothing to close.")
        return 0

    total_current_value = 0.0
    for p in positions:
        sz = float(p.get("size", 0))
        cur = float(p.get("curPrice", 0) or 0)
        total_current_value += sz * cur
    print(f"Total current value: ${total_current_value:.2f}")
    print()

    # Initialize live executor. PolymarketClient requires explicit
    # initialize() before get_price will work; otherwise _get_prices
    # returns ("Client not initialized") and every sell fails upstream.
    poly = PolymarketClient()
    await poly.initialize()
    executor = LiveExecutor(poly)
    await executor._ensure_clob()  # populates _shares_owned too

    print("=== Selling each position with taker fallback ===")
    total_filled_value = 0.0
    total_shares_sold = 0
    sold_count = 0
    skipped_count = 0
    failed_count = 0

    for i, p in enumerate(positions, 1):
        token_id = p["asset"]
        title = p.get("title", "?")[:50]
        held_size = int(float(p.get("size", 0)))
        cur_price = float(p.get("curPrice", 0) or 0)

        print(f"[{i}/{len(positions)}] {title}")
        print(f"  token={token_id[:30]}... shares={held_size} curr=${cur_price:.3f}")

        # live_executor.sell needs shares in its _shares_owned dict
        # force-populate from API data even if sync missed it
        executor._shares_owned[token_id] = held_size

        try:
            fill = await executor.sell(
                token_id=token_id,
                price=cur_price,  # hint; executor fetches fresh prices anyway
                neg_risk=False,  # B2 markets are non-NegRisk
                tick_size="0.01",
                maker_timeout_s=20,  # shorter than default — accept taker quickly
                skip_sync=True,
            )
            filled = fill.shares_filled
            avg_price = (
                fill.usdc_spent / filled if filled > 0 else 0
            )
            if filled > 0:
                sold_count += 1
                total_filled_value += fill.usdc_spent
                total_shares_sold += filled
                print(f"  → SOLD {filled}/{held_size} avg=${avg_price:.3f} "
                      f"USDC=${fill.usdc_spent:.2f} type={fill.order_type}")
            elif fill.status == "skipped":
                skipped_count += 1
                print(f"  → SKIPPED: {fill.error}")
            else:
                failed_count += 1
                print(f"  → FAILED: status={fill.status} err={fill.error}")
        except Exception as e:
            failed_count += 1
            print(f"  → EXCEPTION: {e}")

    print()
    print("=== Summary ===")
    print(f"Positions sold:   {sold_count}/{len(positions)}")
    print(f"Positions skipped: {skipped_count}")
    print(f"Positions failed:  {failed_count}")
    print(f"Total shares sold: {total_shares_sold}")
    print(f"Total USDC recovered: ${total_filled_value:.2f}")
    print(f"Expected from wallet snapshot: ${total_current_value:.2f}")

    # Check new balance
    try:
        new_bal = await executor.get_balance()
        print(f"Wallet balance after: ${new_bal:.2f}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
