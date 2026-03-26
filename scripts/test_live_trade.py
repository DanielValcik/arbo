"""Test live trade — buy $1 of cheapest YES token on a weather market.

This is a REAL trade on Polymarket. Uses $1 USDC.e from the funder wallet.
Run manually to verify auth + execution before enabling C2 live mode.
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from arbo.connectors.polymarket_client import PolymarketClient
from arbo.core.live_executor import LiveExecutor


async def main():
    # 1. Initialize authenticated CLOB client
    print("Initializing Polymarket client...")
    client = PolymarketClient()
    await client.initialize()
    print("Auth OK")

    # 2. Check balance
    try:
        balance = await client.get_balance_allowance()
        print(f"Balance/allowance: {balance}")
    except Exception as e:
        print(f"Balance check: {e}")

    # 3. Find a cheap weather YES token to test
    # Use Gamma API to get current weather markets
    import urllib.request, json, ssl, certifi
    ctx = ssl.create_default_context(cafile=certifi.where())

    url = "https://gamma-api.polymarket.com/events?tag=weather&closed=false&limit=5&order=volume24hr&ascending=false"
    req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=10, context=ctx).read())

    # Find a token with YES price around $0.05 (cheap, safe to test)
    test_token = None
    test_price = None
    for event in data:
        markets = event.get("markets", [])
        for mkt in markets:
            tokens = json.loads(mkt.get("clobTokenIds", "[]"))
            prices = json.loads(mkt.get("outcomePrices", "[]"))
            if len(tokens) >= 1 and len(prices) >= 1:
                yes_price = float(prices[0])
                if 0.01 < yes_price < 0.10:  # Cheap token
                    test_token = tokens[0]
                    test_price = yes_price
                    neg_risk = mkt.get("enableNegRisk", False)
                    question = mkt.get("question", "")[:60]
                    print(f"\nTest target: {question}")
                    print(f"  Token: {test_token[:40]}...")
                    print(f"  YES price: ${test_price}")
                    print(f"  NegRisk: {neg_risk}")
                    break
        if test_token:
            break

    if not test_token:
        print("No suitable test token found!")
        return

    # 4. Execute $1 test buy
    print(f"\n=== EXECUTING $1 TEST BUY ===")
    executor = LiveExecutor(client)

    fill = await executor.buy(
        token_id=test_token,
        price=test_price + 0.01,  # Slightly above market for quick fill
        size_usdc=1.0,
        neg_risk=neg_risk,
    )

    print(f"\nResult:")
    print(f"  Status: {fill.status}")
    print(f"  Order ID: {fill.order_id}")
    print(f"  Fill price: {fill.fill_price}")
    print(f"  Shares: {fill.shares_filled}")
    print(f"  Latency: {fill.latency_ms}ms")
    if fill.error:
        print(f"  Error: {fill.error}")
    if fill.raw_response:
        print(f"  Raw: {json.dumps(fill.raw_response, indent=2)[:500]}")

    # 5. If filled, try selling immediately
    if fill.status == "filled" and fill.shares_filled > 0:
        print(f"\n=== EXECUTING SELL (return shares) ===")
        sell_fill = await executor.sell(
            token_id=test_token,
            price=max(0.01, test_price - 0.01),  # Slightly below for quick fill
            shares=float(fill.shares_filled),
            neg_risk=neg_risk,
        )
        print(f"  Sell status: {sell_fill.status}")
        print(f"  Sell price: {sell_fill.fill_price}")
        if sell_fill.error:
            print(f"  Error: {sell_fill.error}")

    await client.close()
    print("\nTest complete.")


asyncio.run(main())
