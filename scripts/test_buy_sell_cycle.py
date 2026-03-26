"""Test complete BUY → SELL cycle on live Polymarket.

Buys $5 of a cheap weather token, waits 10s, then sells.
Verifies the entire flow works before enabling C2 live strategy.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()

from arbo.connectors.polymarket_client import PolymarketClient
from arbo.core.live_executor import LiveExecutor


async def main():
    # 1. Init
    print("Initializing...")
    client = PolymarketClient()
    await client.initialize()
    executor = LiveExecutor(client)

    # Force init clob + sync positions
    await executor._ensure_clob()
    print(f"Current positions: {len(executor.shares_owned)}")
    for tid, shares in list(executor.shares_owned.items())[:5]:
        print(f"  {tid[:30]}... = {shares} shares")

    # 2. Find a cheap weather token to test
    from arbo.utils.db import PaperTrade, get_session_factory
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PaperTrade)
            .where(PaperTrade.strategy == "C2")
            .where(PaperTrade.status == "open")
            .limit(1)
        )
        t = result.scalars().first()

    if not t:
        print("No open C2 position to test with")
        return

    td = t.trade_details or {}
    token = t.token_id
    city = td.get("city", "?")
    print(f"\nTest token: {city} ({token[:30]}...)")

    # 3. BUY $5
    print(f"\n=== BUY $5 ===")
    buy_fill = await executor.buy(
        token_id=token,
        price=0.05,  # Paper price (will be replaced by taker price)
        size_usdc=5.0,
        neg_risk=True,
    )
    print(f"Status: {buy_fill.status}")
    print(f"Shares filled: {buy_fill.shares_filled}")
    print(f"Fill price: {buy_fill.fill_price}")
    print(f"USDC spent: ${buy_fill.usdc_spent:.2f}")
    print(f"Latency: {buy_fill.latency_ms}ms")
    if buy_fill.error:
        print(f"Error: {buy_fill.error}")

    if buy_fill.shares_filled <= 0:
        print("\nBUY failed — cannot test SELL")
        await client.close()
        return

    # 4. Wait briefly
    print(f"\nOwned after buy: {executor.shares_owned.get(token, 0)} shares")
    print("Waiting 10s before SELL...")
    await asyncio.sleep(10)

    # 5. SELL all
    print(f"\n=== SELL ===")
    sell_fill = await executor.sell(
        token_id=token,
        price=0.04,  # Paper price
        neg_risk=True,
    )
    print(f"Status: {sell_fill.status}")
    print(f"Shares sold: {sell_fill.shares_filled}")
    print(f"Fill price: {sell_fill.fill_price}")
    print(f"USDC received: ${sell_fill.usdc_spent:.2f}")
    print(f"Latency: {sell_fill.latency_ms}ms")
    if sell_fill.error:
        print(f"Error: {sell_fill.error}")

    # 6. P&L
    if buy_fill.shares_filled > 0 and sell_fill.shares_filled > 0:
        buy_cost = buy_fill.usdc_spent
        sell_revenue = sell_fill.usdc_spent
        pnl = sell_revenue - buy_cost
        print(f"\n=== RESULT ===")
        print(f"Bought: {buy_fill.shares_filled} shares @ ${float(buy_fill.fill_price):.4f} = ${buy_cost:.2f}")
        print(f"Sold:   {sell_fill.shares_filled} shares @ ${float(sell_fill.fill_price):.4f} = ${sell_revenue:.2f}")
        print(f"P&L: ${pnl:+.2f}")
        print(f"Owned after: {executor.shares_owned.get(token, 0)} shares")

    await client.close()
    print("\nTest complete.")


asyncio.run(main())
