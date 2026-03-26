#!/usr/bin/env python3
"""
Fetch real resolved Polymarket crypto markets for backtest calibration.
======================================================================

Queries Gamma API for closed/resolved crypto markets to extract:
- Realistic volume distributions
- Liquidity distributions
- Fee model parameters
- Resolution outcomes
- Market pricing patterns

Usage: python3 research_b/fetch_polymarket_markets.py

Output: research_b/data/polymarket_crypto_markets.json
"""

import json
import re
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
OUTPUT_FILE = DATA_DIR / "polymarket_crypto_markets.json"

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Crypto-related tags on Polymarket
CRYPTO_TAGS = ["crypto", "bitcoin", "ethereum", "solana", "cryptocurrency"]

# Coins we care about (for matching market questions)
COIN_PATTERNS = {
    "BTC": [r"bitcoin", r"\bBTC\b", r"btc"],
    "ETH": [r"ethereum", r"\bETH\b", r"eth"],
    "SOL": [r"solana", r"\bSOL\b"],
    "XRP": [r"\bXRP\b", r"ripple"],
    "ADA": [r"cardano", r"\bADA\b"],
    "DOGE": [r"dogecoin", r"\bDOGE\b"],
    "AVAX": [r"avalanche", r"\bAVAX\b"],
    "DOT": [r"polkadot", r"\bDOT\b"],
    "LINK": [r"chainlink", r"\bLINK\b"],
    "MATIC": [r"polygon", r"\bMATIC\b"],
    "UNI": [r"uniswap", r"\bUNI\b"],
    "ATOM": [r"cosmos", r"\bATOM\b"],
    "LTC": [r"litecoin", r"\bLTC\b"],
    "NEAR": [r"\bNEAR\b"],
    "ARB": [r"arbitrum", r"\bARB\b"],
    "OP": [r"optimism", r"\bOP\b"],
    "APT": [r"aptos", r"\bAPT\b"],
    "SUI": [r"\bSUI\b"],
    "FIL": [r"filecoin", r"\bFIL\b"],
    "AAVE": [r"\bAAVE\b"],
}


def fetch_gamma_markets(tag: str, offset: int = 0, limit: int = 100) -> list[dict]:
    """Fetch markets from Gamma API."""
    url = (
        f"{GAMMA_BASE}/markets"
        f"?tag={tag}&closed=true&limit={limit}&offset={offset}"
    )

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Arbo-Research/1.0")

    resp = urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30)
    return json.loads(resp.read().decode())


def identify_coin(question: str) -> str | None:
    """Identify which coin a market question is about."""
    q_lower = question.lower()
    for coin, patterns in COIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower):
                return coin
    return None


def extract_threshold(question: str) -> float | None:
    """Try to extract a price threshold from market question."""
    # Patterns like "$65,000", "$3,500.50", "$100K"
    patterns = [
        r"\$([0-9,]+(?:\.[0-9]+)?)\s*[kK]",  # $65K
        r"\$([0-9,]+(?:\.[0-9]+)?)",            # $65,000
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            if question.lower().find("k") > -1 and val < 1000:
                val *= 1000
            return val
    return None


def parse_market(raw: dict) -> dict | None:
    """Parse a Gamma API market into our format."""
    question = raw.get("question", "")
    coin = identify_coin(question)
    if not coin:
        return None

    # Extract market data
    volume = float(raw.get("volume", 0) or 0)
    liquidity = float(raw.get("liquidity", 0) or 0)
    outcome_prices = raw.get("outcomePrices", "")

    # Parse outcome prices
    try:
        if isinstance(outcome_prices, str) and outcome_prices:
            prices = json.loads(outcome_prices)
            yes_price = float(prices[0]) if prices else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
        elif isinstance(outcome_prices, list):
            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 1 - yes_price
        else:
            yes_price = 0.5
            no_price = 0.5
    except (ValueError, IndexError, json.JSONDecodeError):
        yes_price = 0.5
        no_price = 0.5

    # Resolution
    resolved = raw.get("resolved", False)
    resolution = raw.get("resolution", "")

    # Fee info
    fee_rate = 0.02 if raw.get("enableOrderBook", True) else 0.0  # Default crypto fee

    # Timestamps
    created = raw.get("createdAt", "")
    end_date = raw.get("endDate", "") or raw.get("endDateIso", "")
    closed_at = raw.get("closedTime", "") or raw.get("closeTime", "")

    threshold = extract_threshold(question)

    return {
        "question": question,
        "coin": coin,
        "condition_id": raw.get("conditionId", ""),
        "volume": volume,
        "liquidity": liquidity,
        "yes_price": yes_price,
        "no_price": no_price,
        "resolved": resolved,
        "resolution": resolution,
        "threshold": threshold,
        "fee_rate": fee_rate,
        "neg_risk": raw.get("enableNegRisk", False),
        "created_at": created,
        "end_date": end_date,
        "closed_at": closed_at,
        "outcomes": raw.get("outcomes", ""),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_markets = []
    seen_ids = set()

    for tag in CRYPTO_TAGS:
        print(f"Fetching tag='{tag}'...", end="", flush=True)
        offset = 0
        tag_count = 0

        while True:
            try:
                batch = fetch_gamma_markets(tag, offset=offset)
            except Exception as e:
                print(f" ERROR at offset {offset}: {e}")
                break

            if not batch:
                break

            for raw in batch:
                cid = raw.get("conditionId", "")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)

                parsed = parse_market(raw)
                if parsed:
                    all_markets.append(parsed)
                    tag_count += 1

            offset += len(batch)
            if len(batch) < 100:
                break

            time.sleep(0.5)  # Rate limit: 500 req/10s

        print(f" {tag_count} crypto markets")

    # Sort by volume
    all_markets.sort(key=lambda m: m["volume"], reverse=True)

    # Compute calibration stats
    volumes = [m["volume"] for m in all_markets if m["volume"] > 0]
    liquidities = [m["liquidity"] for m in all_markets if m["liquidity"] > 0]
    yes_prices = [m["yes_price"] for m in all_markets if 0.01 < m["yes_price"] < 0.99]

    calibration = {}
    if volumes:
        volumes_sorted = sorted(volumes)
        calibration["volume"] = {
            "count": len(volumes),
            "min": volumes_sorted[0],
            "p10": volumes_sorted[len(volumes_sorted) // 10],
            "p25": volumes_sorted[len(volumes_sorted) // 4],
            "median": volumes_sorted[len(volumes_sorted) // 2],
            "p75": volumes_sorted[3 * len(volumes_sorted) // 4],
            "p90": volumes_sorted[9 * len(volumes_sorted) // 10],
            "max": volumes_sorted[-1],
            "mean": sum(volumes) / len(volumes),
        }
    if liquidities:
        liq_sorted = sorted(liquidities)
        calibration["liquidity"] = {
            "count": len(liquidities),
            "min": liq_sorted[0],
            "p10": liq_sorted[len(liq_sorted) // 10],
            "p25": liq_sorted[len(liq_sorted) // 4],
            "median": liq_sorted[len(liq_sorted) // 2],
            "p75": liq_sorted[3 * len(liq_sorted) // 4],
            "p90": liq_sorted[9 * len(liq_sorted) // 10],
            "max": liq_sorted[-1],
            "mean": sum(liquidities) / len(liquidities),
        }
    if yes_prices:
        calibration["yes_price"] = {
            "mean": sum(yes_prices) / len(yes_prices),
            "count": len(yes_prices),
        }

    # Coin distribution
    coin_counts = {}
    for m in all_markets:
        coin_counts[m["coin"]] = coin_counts.get(m["coin"], 0) + 1
    calibration["coin_distribution"] = coin_counts

    # Resolution stats
    resolved_markets = [m for m in all_markets if m["resolved"]]
    calibration["resolution"] = {
        "total_resolved": len(resolved_markets),
        "total_unresolved": len(all_markets) - len(resolved_markets),
    }

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(all_markets),
        "calibration": calibration,
        "markets": all_markets,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! {len(all_markets)} crypto markets saved")
    print(f"File: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
    print(f"\nCalibration stats:")
    if "volume" in calibration:
        v = calibration["volume"]
        print(f"  Volume:    p10=${v['p10']:.0f}, median=${v['median']:.0f}, "
              f"p90=${v['p90']:.0f}, max=${v['max']:.0f}")
    if "liquidity" in calibration:
        l = calibration["liquidity"]
        print(f"  Liquidity: p10=${l['p10']:.0f}, median=${l['median']:.0f}, "
              f"p90=${l['p90']:.0f}, max=${l['max']:.0f}")
    print(f"  Coins:     {coin_counts}")


if __name__ == "__main__":
    main()
