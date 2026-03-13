#!/usr/bin/env python3
"""
Fetch historical crypto data for Strategy B backtesting.
========================================================

Primary: Binance public API (free, no auth, full history, includes trade count).
Fallback: CoinGecko (limited to 365 days on demo plan).
Optional: Santiment DAA data (--santiment flag).

Usage:
    python3 research_b/fetch_data.py              # Binance (default)
    python3 research_b/fetch_data.py --santiment   # + Santiment DAA

Output: research_b/data/crypto_history.json
"""

import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

DATA_DIR = Path(__file__).parent / "data"
CACHE_FILE = DATA_DIR / "crypto_history.json"

# Coin ID -> (symbol, Binance trading pair)
COINS = {
    "bitcoin":      ("BTC",  "BTCUSDT"),
    "ethereum":     ("ETH",  "ETHUSDT"),
    "solana":       ("SOL",  "SOLUSDT"),
    "ripple":       ("XRP",  "XRPUSDT"),
    "cardano":      ("ADA",  "ADAUSDT"),
    "dogecoin":     ("DOGE", "DOGEUSDT"),
    "avalanche-2":  ("AVAX", "AVAXUSDT"),
    "polkadot":     ("DOT",  "DOTUSDT"),
    "chainlink":    ("LINK", "LINKUSDT"),
    "matic-network":("MATIC","MATICUSDT"),
    "uniswap":      ("UNI",  "UNIUSDT"),
    "cosmos":       ("ATOM", "ATOMUSDT"),
    "litecoin":     ("LTC",  "LTCUSDT"),
    "near":         ("NEAR", "NEARUSDT"),
    "arbitrum":     ("ARB",  "ARBUSDT"),
    "optimism":     ("OP",   "OPUSDT"),
    "aptos":        ("APT",  "APTUSDT"),
    "sui":          ("SUI",  "SUIUSDT"),
    "filecoin":     ("FIL",  "FILUSDT"),
    "aave":         ("AAVE", "AAVEUSDT"),
}

# Target date range: 2024-01-01 to 2025-12-31
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
TARGET_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
TARGET_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def fetch_binance(symbol: str) -> list[dict]:
    """
    Fetch daily klines from Binance public API.

    Returns daily OHLCV + trade count for 2024-2025.
    Binance kline format: [open_time, open, high, low, close, volume,
                           close_time, quote_volume, num_trades, ...]
    """
    entries = []
    current_start = START_MS

    while current_start < END_MS:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1d"
            f"&startTime={current_start}&endTime={END_MS}&limit=1000"
        )

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Arbo-Research/1.0")

        resp = urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30)
        klines = json.loads(resp.read().decode())

        if not klines:
            break

        for k in klines:
            open_time_ms = k[0]
            dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
            if dt < TARGET_START or dt > TARGET_END:
                continue

            entries.append({
                "date": dt.strftime("%Y-%m-%d"),
                "price": float(k[4]),           # Close price
                "volume": float(k[7]),          # Quote asset volume (USDT)
                "num_trades": int(k[8]),        # Number of trades (great DAA proxy!)
                "high": float(k[2]),
                "low": float(k[3]),
            })

        # Move past the last candle
        last_close_time = klines[-1][6]
        current_start = last_close_time + 1

        if len(klines) < 1000:
            break  # No more data

    return entries


def fetch_santiment_daa(coin_symbol: str) -> dict[str, float]:
    """Fetch daily active addresses from Santiment GraphQL API (free tier)."""
    slug_map = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
        "AVAX": "avalanche", "DOT": "polkadot", "LINK": "chainlink",
        "MATIC": "polygon", "UNI": "uniswap", "ATOM": "cosmos",
        "LTC": "litecoin", "NEAR": "near-protocol", "ARB": "arbitrum",
        "OP": "optimism", "APT": "aptos", "SUI": "sui",
        "FIL": "filecoin", "AAVE": "aave",
    }

    slug = slug_map.get(coin_symbol)
    if not slug:
        return {}

    query = """
    {
      getMetric(metric: "daily_active_addresses") {
        timeseriesData(
          slug: "%s"
          from: "2024-01-01T00:00:00Z"
          to: "2025-12-31T23:59:59Z"
          interval: "1d"
        ) {
          datetime
          value
        }
      }
    }
    """ % slug

    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        "https://api.santiment.net/graphql",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Arbo-Research/1.0"},
    )

    try:
        resp = urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30)
        data = json.loads(resp.read().decode())
        series = data.get("data", {}).get("getMetric", {}).get("timeseriesData", [])
        return {s["datetime"][:10]: s["value"] for s in series}
    except Exception as e:
        print(f"  WARNING: Santiment failed for {coin_symbol}: {e}")
        return {}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    use_santiment = "--santiment" in sys.argv

    print("Data source: Binance public API (free, no auth)")
    print(f"Date range: 2024-01-01 to 2025-12-31")

    all_data = {}
    total = len(COINS)

    for i, (coin_id, (symbol, binance_pair)) in enumerate(COINS.items(), 1):
        print(f"[{i}/{total}] Fetching {symbol} ({binance_pair})...", end="", flush=True)

        try:
            entries = fetch_binance(binance_pair)
            print(f" {len(entries)} days", end="")

            # Optionally add Santiment DAA
            if use_santiment:
                time.sleep(1)
                daa_data = fetch_santiment_daa(symbol)
                if daa_data:
                    for entry in entries:
                        daa_val = daa_data.get(entry["date"])
                        if daa_val is not None:
                            entry["daa"] = round(daa_val, 2)
                    daa_count = sum(1 for e in entries if "daa" in e)
                    print(f" + {daa_count} DAA", end="")

            all_data[coin_id] = entries
            print(" OK")

        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Binance rate limit: 1200 req/min — very generous
        if i < total:
            time.sleep(0.5)

    # Save
    with open(CACHE_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    total_entries = sum(len(v) for v in all_data.values())
    print(f"\nDone! Saved {len(all_data)} coins, {total_entries} total entries")
    print(f"File: {CACHE_FILE} ({CACHE_FILE.stat().st_size / 1024:.1f} KB)")

    # Summary
    for coin_id, entries in all_data.items():
        if entries:
            symbol = COINS[coin_id][0]
            has_trades = "num_trades" in entries[0]
            trades_info = f", avg {sum(e.get('num_trades', 0) for e in entries) // len(entries)} trades/day" if has_trades else ""
            print(f"  {symbol:5s}: {entries[0]['date']} to {entries[-1]['date']} ({len(entries)} days{trades_info})")


if __name__ == "__main__":
    main()
