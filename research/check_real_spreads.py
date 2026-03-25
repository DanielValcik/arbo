"""Check Real Bid-Ask Spreads on Polymarket Weather Markets.

Connects to Polymarket CLOB API (no auth) to fetch live orderbooks
for current weather temperature markets. Measures spreads and depth
to answer: is 0.5% exit slippage realistic?

Usage:
    python3 research/check_real_spreads.py
"""

import json
import ssl
import sys
import time
import urllib.request
from dataclasses import dataclass, field

try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
REQUEST_DELAY = 0.25  # conservative rate limiting


def fetch_json(url: str) -> dict | list | None:
    """Fetch JSON from URL with SSL context and rate limiting."""
    time.sleep(REQUEST_DELAY)
    req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/1.0"})
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR fetching {url[:80]}...: {e}")
        return None


@dataclass
class MarketSpread:
    """Spread data for a single market/token."""

    event_title: str
    market_question: str
    token_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread_cents: float = 0.0
    spread_pct: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    total_bids: int = 0
    total_asks: int = 0
    bid_depth_3lvl_usd: float = 0.0
    ask_depth_3lvl_usd: float = 0.0


def analyze_orderbook(book: dict, event_title: str, question: str, token_id: str) -> MarketSpread | None:
    """Extract spread metrics from an orderbook response."""
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return None

    # Sort: bids descending by price, asks ascending
    bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
    asks_sorted = sorted(asks, key=lambda x: float(x["price"]))

    best_bid = float(bids_sorted[0]["price"])
    best_ask = float(asks_sorted[0]["price"])

    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return None

    mid = (best_bid + best_ask) / 2
    spread_cents = (best_ask - best_bid) * 100
    spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

    # Depth at best bid/ask (in $)
    bid_depth = float(bids_sorted[0]["size"]) * best_bid
    ask_depth = float(asks_sorted[0]["size"]) * best_ask

    # Depth across top 3 levels
    bid_depth_3 = sum(float(b["size"]) * float(b["price"]) for b in bids_sorted[:3])
    ask_depth_3 = sum(float(a["size"]) * float(a["price"]) for a in asks_sorted[:3])

    ms = MarketSpread(
        event_title=event_title,
        market_question=question,
        token_id=token_id,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread_cents=spread_cents,
        spread_pct=spread_pct,
        bid_depth_usd=bid_depth,
        ask_depth_usd=ask_depth,
        total_bids=len(bids_sorted),
        total_asks=len(asks_sorted),
        bid_depth_3lvl_usd=bid_depth_3,
        ask_depth_3lvl_usd=ask_depth_3,
    )
    return ms


def main():
    print("=" * 80)
    print("POLYMARKET WEATHER MARKETS — LIVE SPREAD ANALYSIS")
    print("=" * 80)
    print()

    # Step 1: Discover weather temperature events via Gamma API
    # NOTE: must use tag_slug=weather (not tag=weather), and filter for "temperature"
    print("Fetching weather temperature events from Gamma API...")
    events_url = (
        f"{GAMMA_BASE}/events?tag_slug=weather&active=true&closed=false&limit=100"
    )
    events = fetch_json(events_url)

    if not events:
        print("ERROR: No events returned from Gamma API")
        sys.exit(1)

    # Filter to temperature events only
    temp_events = [
        e for e in events
        if "temperature" in (e.get("title", "") or "").lower()
    ]
    print(f"Found {len(events)} weather events, {len(temp_events)} are temperature events")
    print()

    if not temp_events:
        print("No temperature events found!")
        sys.exit(1)

    # Step 2: Collect all YES token IDs from markets
    all_spreads: list[MarketSpread] = []
    markets_checked = 0
    markets_with_book = 0
    markets_empty = 0

    for event in temp_events:
        event_title = event.get("title", "Unknown")
        markets = event.get("markets", [])

        if not markets:
            continue

        for mkt in markets:
            # Skip closed/inactive markets
            if mkt.get("closed", False) or not mkt.get("active", True):
                continue

            question = mkt.get("question", mkt.get("groupItemTitle", ""))

            # Get YES token ID — clobTokenIds is a JSON-encoded string
            clob_raw = mkt.get("clobTokenIds", "[]")
            if isinstance(clob_raw, str):
                try:
                    token_ids = json.loads(clob_raw)
                except json.JSONDecodeError:
                    token_ids = []
            else:
                token_ids = clob_raw

            if not token_ids:
                continue

            # First token is YES
            yes_token = token_ids[0]
            markets_checked += 1

            # Fetch orderbook
            book_url = f"{CLOB_BASE}/book?token_id={yes_token}"
            book = fetch_json(book_url)

            if not book:
                markets_empty += 1
                continue

            result = analyze_orderbook(book, event_title, question, yes_token)
            if result:
                markets_with_book += 1
                all_spreads.append(result)
                # Print inline progress
                print(
                    f"  [{markets_with_book:3d}] {question[:60]:60s} "
                    f"bid={result.best_bid:.3f} ask={result.best_ask:.3f} "
                    f"spread={result.spread_cents:.1f}c ({result.spread_pct:.1f}%)"
                )
            else:
                markets_empty += 1

    print()
    print(f"Markets checked: {markets_checked}")
    print(f"Markets with orderbook: {markets_with_book}")
    print(f"Markets empty/invalid: {markets_empty}")
    print()

    if not all_spreads:
        print("No markets with valid orderbooks found!")
        sys.exit(1)

    # Step 3: Group by price range
    cheap = [s for s in all_spreads if 0.01 <= s.mid_price <= 0.15]
    mid = [s for s in all_spreads if 0.15 < s.mid_price <= 0.40]
    expensive = [s for s in all_spreads if 0.40 < s.mid_price <= 0.80]
    extreme = [s for s in all_spreads if s.mid_price > 0.80]

    def print_group(name: str, group: list[MarketSpread]):
        if not group:
            print(f"\n{'─' * 80}")
            print(f"  {name}: (no markets)")
            return

        spreads_c = [s.spread_cents for s in group]
        spreads_pct = [s.spread_pct for s in group]
        bid_depths = [s.bid_depth_usd for s in group]
        ask_depths = [s.ask_depth_usd for s in group]
        bid_depths_3 = [s.bid_depth_3lvl_usd for s in group]
        ask_depths_3 = [s.ask_depth_3lvl_usd for s in group]

        avg_spread_c = sum(spreads_c) / len(spreads_c)
        med_spread_c = sorted(spreads_c)[len(spreads_c) // 2]
        avg_spread_pct = sum(spreads_pct) / len(spreads_pct)
        med_spread_pct = sorted(spreads_pct)[len(spreads_pct) // 2]
        avg_bid_d = sum(bid_depths) / len(bid_depths)
        avg_ask_d = sum(ask_depths) / len(ask_depths)
        avg_bid_d3 = sum(bid_depths_3) / len(bid_depths_3)
        avg_ask_d3 = sum(ask_depths_3) / len(ask_depths_3)

        print(f"\n{'─' * 80}")
        print(f"  {name} ({len(group)} markets)")
        print(f"{'─' * 80}")
        print(f"  Spread (cents):   avg={avg_spread_c:.1f}c  median={med_spread_c:.1f}c  "
              f"min={min(spreads_c):.1f}c  max={max(spreads_c):.1f}c")
        print(f"  Spread (%):       avg={avg_spread_pct:.1f}%  median={med_spread_pct:.1f}%  "
              f"min={min(spreads_pct):.1f}%  max={max(spreads_pct):.1f}%")
        print(f"  Depth best lvl:   bid=${avg_bid_d:.0f}  ask=${avg_ask_d:.0f}")
        print(f"  Depth top-3 lvl:  bid=${avg_bid_d3:.0f}  ask=${avg_ask_d3:.0f}")

        # Is 0.5% slippage realistic?
        pct_under_05 = sum(1 for s in group if s.spread_pct <= 1.0) / len(group) * 100
        pct_under_1 = sum(1 for s in group if s.spread_pct <= 2.0) / len(group) * 100
        print(f"  Markets with spread <= 1%: {pct_under_05:.0f}%")
        print(f"  Markets with spread <= 2%: {pct_under_1:.0f}%")

    print()
    print("=" * 80)
    print("SPREAD ANALYSIS BY PRICE RANGE")
    print("=" * 80)

    print_group("CHEAP (0.01 - 0.15)", cheap)
    print_group("MID (0.15 - 0.40)", mid)
    print_group("EXPENSIVE (0.40 - 0.80)", expensive)
    print_group("EXTREME (0.80+)", extreme)

    # Step 4: Overall summary
    all_pct = [s.spread_pct for s in all_spreads]
    all_cents = [s.spread_cents for s in all_spreads]

    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Total markets analyzed: {len(all_spreads)}")
    print(f"  Spread (cents):  avg={sum(all_cents)/len(all_cents):.1f}c  "
          f"median={sorted(all_cents)[len(all_cents)//2]:.1f}c")
    print(f"  Spread (%):      avg={sum(all_pct)/len(all_pct):.1f}%  "
          f"median={sorted(all_pct)[len(all_pct)//2]:.1f}%")

    # Key question
    print()
    print("=" * 80)
    print("KEY QUESTION: Is 0.5% exit slippage realistic?")
    print("=" * 80)

    # For exit slippage, what matters is: can you cross the spread for < 0.5% cost?
    # Exit slippage ~ half-spread (you sell at bid, which is mid - half_spread)
    # So half-spread % = spread_pct / 2
    half_spreads = [s.spread_pct / 2 for s in all_spreads]
    under_05 = sum(1 for h in half_spreads if h <= 0.5)
    under_1 = sum(1 for h in half_spreads if h <= 1.0)
    under_2 = sum(1 for h in half_spreads if h <= 2.0)

    print(f"  Half-spread (= exit slippage if crossing at market):")
    print(f"    avg = {sum(half_spreads)/len(half_spreads):.2f}%")
    print(f"    median = {sorted(half_spreads)[len(half_spreads)//2]:.2f}%")
    print(f"    Markets with exit slippage <= 0.5%: {under_05}/{len(all_spreads)} "
          f"({under_05/len(all_spreads)*100:.0f}%)")
    print(f"    Markets with exit slippage <= 1.0%: {under_1}/{len(all_spreads)} "
          f"({under_1/len(all_spreads)*100:.0f}%)")
    print(f"    Markets with exit slippage <= 2.0%: {under_2}/{len(all_spreads)} "
          f"({under_2/len(all_spreads)*100:.0f}%)")

    # Breakdown for our typical trading range
    our_range = [s for s in all_spreads if 0.05 <= s.mid_price <= 0.70]
    if our_range:
        our_half = [s.spread_pct / 2 for s in our_range]
        our_under = sum(1 for h in our_half if h <= 0.5)
        print()
        print(f"  In OUR trading range (0.05-0.70, quality-gate prices):")
        print(f"    {len(our_range)} markets")
        print(f"    Half-spread avg = {sum(our_half)/len(our_half):.2f}%")
        print(f"    Half-spread median = {sorted(our_half)[len(our_half)//2]:.2f}%")
        print(f"    Exit slippage <= 0.5%: {our_under}/{len(our_range)} "
              f"({our_under/len(our_range)*100:.0f}%)")

    # Show worst spreads for inspection
    print()
    print("─" * 80)
    print("TOP 10 WIDEST SPREADS:")
    print("─" * 80)
    widest = sorted(all_spreads, key=lambda s: s.spread_pct, reverse=True)[:10]
    for i, s in enumerate(widest, 1):
        print(f"  {i:2d}. {s.market_question[:55]:55s} "
              f"mid={s.mid_price:.3f} spread={s.spread_cents:.1f}c ({s.spread_pct:.1f}%) "
              f"bids={s.total_bids} asks={s.total_asks}")

    # Show tightest spreads
    print()
    print("─" * 80)
    print("TOP 10 TIGHTEST SPREADS:")
    print("─" * 80)
    tightest = sorted(all_spreads, key=lambda s: s.spread_pct)[:10]
    for i, s in enumerate(tightest, 1):
        print(f"  {i:2d}. {s.market_question[:55]:55s} "
              f"mid={s.mid_price:.3f} spread={s.spread_cents:.1f}c ({s.spread_pct:.1f}%) "
              f"depth_bid=${s.bid_depth_3lvl_usd:.0f} depth_ask=${s.ask_depth_3lvl_usd:.0f}")

    print()
    print("=" * 80)
    print("VERDICT:")
    median_half = sorted(half_spreads)[len(half_spreads) // 2]
    if median_half <= 0.5:
        print(f"  0.5% exit slippage is REALISTIC (median half-spread = {median_half:.2f}%)")
    elif median_half <= 1.0:
        print(f"  0.5% exit slippage is TIGHT but possible with limit orders "
              f"(median half-spread = {median_half:.2f}%)")
    elif median_half <= 2.0:
        print(f"  0.5% exit slippage is OPTIMISTIC — budget 1-2% instead "
              f"(median half-spread = {median_half:.2f}%)")
    else:
        print(f"  0.5% exit slippage is UNREALISTIC — spreads are wide "
              f"(median half-spread = {median_half:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
