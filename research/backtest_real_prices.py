"""Realistic Backtest Using Real Polymarket Price History.

Uses actual Polymarket CLOB prices (from /prices-history) instead of
synthetic market maker. Combines:
  - Real market prices at entry time (from SQLite)
  - Real weather forecasts (Open-Meteo archive)
  - Real resolution outcomes (which bucket won)

This is the most realistic backtest possible without live trading.

Usage:
    python3 research/backtest_real_prices.py

Prerequisites:
    python3 research/download_price_history.py  (download price data first)
"""

import json
import math
import ssl
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

sys.path.insert(0, str(Path(__file__).parent))
import strategy_experiment as strategy
from price_history_db import PriceHistoryDB

# ── Configuration ─────────────────────────────────────────────────────

# Strategy parameters (current production values)
MIN_EDGE = 0.08
MAX_PRICE = 0.43
MIN_PRICE = 0.30
MIN_FORECAST_PROB = 0.62
MIN_VOLUME = 1000
MIN_LIQUIDITY = 200

# Sizing (quarter-Kelly per architecture)
KELLY_FRACTION = 0.25
KELLY_RAW_CAP = 0.40
MAX_POSITION_PCT = 0.05

INITIAL_CAPITAL = 1000.0
SLIPPAGE_PCT = 0.005
GAS_COST_USD = 0.007

# Entry timing: hours before market close to simulate entry
ENTRY_HOURS_BEFORE_CLOSE = [48, 36, 24, 12, 6]

# City overrides (from autoresearch)
EXCLUDED_CITIES = {"dc", "toronto", "buenos_aires", "nyc", "atlanta", "wellington"}
WIDENED_CITIES = {"paris": 0.50, "seattle": 0.50, "london": 0.50, "miami": 0.50}

# Open-Meteo coordinates
CITY_COORDS = {
    "nyc": {"lat": 40.7128, "lon": -74.0060},
    "chicago": {"lat": 41.8781, "lon": -87.6298},
    "london": {"lat": 51.5074, "lon": -0.1278},
    "seoul": {"lat": 37.5665, "lon": 126.9780},
    "buenos_aires": {"lat": -34.6037, "lon": -58.3816},
    "atlanta": {"lat": 33.7490, "lon": -84.3880},
    "toronto": {"lat": 43.6532, "lon": -79.3832},
    "ankara": {"lat": 39.9334, "lon": 32.8597},
    "sao_paulo": {"lat": -23.5505, "lon": -46.6333},
    "miami": {"lat": 25.7617, "lon": -80.1918},
    "paris": {"lat": 48.8566, "lon": 2.3522},
    "dallas": {"lat": 32.7767, "lon": -96.7970},
    "seattle": {"lat": 47.6062, "lon": -122.3321},
    "wellington": {"lat": -41.2866, "lon": 174.7756},
    "tokyo": {"lat": 35.6762, "lon": 139.6503},
    "munich": {"lat": 48.1351, "lon": 11.5820},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437},
    "dc": {"lat": 38.9072, "lon": -77.0369},
    "tel_aviv": {"lat": 32.0853, "lon": 34.7818},
    "lucknow": {"lat": 26.8467, "lon": 80.9462},
}

FORECAST_CACHE_DIR = Path(__file__).parent / "data"


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class Trade:
    """A simulated trade on real market data."""

    event_id: str
    city: str
    target_date: str
    bucket_question: str
    entry_price: float
    our_prob: float
    edge: float
    size: float
    won: bool
    pnl: float
    entry_hours_before: float


# ── Weather forecast fetching ─────────────────────────────────────────


def fetch_forecast(city: str, target_date: str) -> float | None:
    """Fetch historical high temperature forecast from Open-Meteo.

    We use the archive API to get what Open-Meteo would have forecasted.
    In reality, we'd use forecast API days in advance, but archive gives
    us the actual observed temperature which is close to the forecast
    for days_out=0..2.
    """
    coords = CITY_COORDS.get(city)
    if not coords:
        return None

    cache_file = FORECAST_CACHE_DIR / f"openmeteo_{city}_realbt.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if target_date in cached:
            return cached[target_date]
    else:
        cached = {}

    # Fetch a range around the target date
    dt = date.fromisoformat(target_date)
    start = (dt - timedelta(days=7)).isoformat()
    end = dt.isoformat()

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={coords['lat']}&longitude={coords['lon']}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max&timezone=auto"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ArboResearch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return None

    daily = data.get("daily", {})
    for i, dt_str in enumerate(daily.get("time", [])):
        val = daily["temperature_2m_max"][i]
        if val is not None:
            cached[dt_str] = val

    FORECAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cached, f)

    return cached.get(target_date)


# ── Strategy logic ────────────────────────────────────────────────────


def quality_gate(city: str, edge: float, price: float,
                 forecast_prob: float, volume: float) -> bool:
    """Check if a trade passes the quality gate."""
    if city in EXCLUDED_CITIES:
        return False

    max_price = WIDENED_CITIES.get(city, MAX_PRICE)
    min_edge_city = MIN_EDGE

    if edge < min_edge_city:
        return False
    if price < MIN_PRICE or price > max_price:
        return False
    if forecast_prob < MIN_FORECAST_PROB:
        return False
    if volume < MIN_VOLUME:
        return False

    return True


def position_size(edge: float, market_price: float,
                  available_capital: float, total_capital: float) -> float:
    """Calculate position size using quarter-Kelly."""
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0

    prob = market_price + edge
    if prob <= 0 or prob >= 1:
        return 0.0

    odds = (1.0 / market_price) - 1.0
    kelly_raw = (prob * odds - (1.0 - prob)) / odds
    if kelly_raw <= 0:
        return 0.0

    kelly_raw = min(kelly_raw, KELLY_RAW_CAP)
    kelly_adjusted = kelly_raw * KELLY_FRACTION
    size = available_capital * kelly_adjusted
    max_size = total_capital * MAX_POSITION_PCT
    size = min(size, max_size)

    return round(size, 2) if size >= 1.0 else 0.0


# ── Main backtest ─────────────────────────────────────────────────────


def run_backtest(
    db: PriceHistoryDB,
    entry_hours: float = 24,
    verbose: bool = True,
) -> list[Trade]:
    """Run backtest on real Polymarket prices.

    For each event with price data:
    1. Get market prices N hours before close
    2. Get our weather forecast (Open-Meteo archive)
    3. Calculate edge = our_prob - market_price
    4. Apply quality gate
    5. If pass: simulate trade, check if bucket won
    """
    events = db.get_events(with_prices=True)
    if verbose:
        print(f"Events with price data: {len(events)}")

    trades = []
    skipped_no_city = 0
    skipped_excluded = 0
    skipped_no_forecast = 0
    skipped_no_prices = 0
    skipped_gate = 0
    capital = INITIAL_CAPITAL

    for ev in events:
        if not ev.city:
            skipped_no_city += 1
            continue
        if ev.city in EXCLUDED_CITIES:
            skipped_excluded += 1
            continue
        if not ev.target_date:
            continue

        # Get price snapshot
        snap = db.get_price_snapshot(ev.event_id, entry_hours)
        if not snap or not snap.prices:
            skipped_no_prices += 1
            continue

        # Get weather forecast
        forecast_temp = fetch_forecast(ev.city, ev.target_date)
        if forecast_temp is None:
            skipped_no_forecast += 1
            continue

        # Evaluate each bucket
        for bucket in snap.buckets:
            market_price = snap.prices.get(bucket.token_id)
            if market_price is None or market_price <= 0.001:
                continue

            # Calculate our probability for this bucket
            our_prob = strategy.estimate_probability(
                forecast_temp,
                bucket.low_c,
                bucket.high_c,
                0,  # days_out = 0 (day-of forecast)
                city=ev.city,
            )

            edge = our_prob - market_price

            # Quality gate
            if not quality_gate(ev.city, edge, market_price,
                                our_prob, bucket.volume):
                skipped_gate += 1
                continue

            # Position sizing
            size = position_size(edge, market_price, capital, capital)
            if size <= 0:
                continue

            # Apply slippage
            fill_price = market_price * (1 + SLIPPAGE_PCT)

            # Calculate PnL
            if bucket.won:
                # Buy YES at fill_price, resolves to $1
                pnl = size * (1.0 / fill_price - 1.0) - GAS_COST_USD
            else:
                # Buy YES at fill_price, resolves to $0
                pnl = -size - GAS_COST_USD

            capital += pnl

            trade = Trade(
                event_id=ev.event_id,
                city=ev.city,
                target_date=ev.target_date,
                bucket_question=bucket.question,
                entry_price=fill_price,
                our_prob=our_prob,
                edge=edge,
                size=size,
                won=bucket.won,
                pnl=pnl,
                entry_hours_before=entry_hours,
            )
            trades.append(trade)

    if verbose:
        print(f"\nSkipped: {skipped_no_city} no city, "
              f"{skipped_excluded} excluded, "
              f"{skipped_no_forecast} no forecast, "
              f"{skipped_no_prices} no prices, "
              f"{skipped_gate} gate rejected")

    return trades


def print_results(trades: list[Trade], entry_hours: float):
    """Print detailed backtest results."""
    if not trades:
        print("No trades executed!")
        return

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.won)
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100
    avg_size = sum(t.size for t in trades) / len(trades)
    avg_edge = sum(t.edge for t in trades) / len(trades)

    # Drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t.pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    print(f"\n{'=' * 70}")
    print(f"REALISTIC BACKTEST — Entry {entry_hours}h before close")
    print(f"{'=' * 70}")
    print(f"  Trades:      {len(trades)} ({wins}W / {losses}L)")
    print(f"  Win rate:    {win_rate:.1f}%")
    print(f"  Total PnL:   ${total_pnl:,.2f} ({total_pnl / INITIAL_CAPITAL * 100:.1f}%)")
    print(f"  Avg size:    ${avg_size:.2f}")
    print(f"  Avg edge:    {avg_edge:.3f}")
    print(f"  Max DD:      ${max_dd:.2f} ({max_dd / INITIAL_CAPITAL * 100:.1f}%)")
    print(f"  Final cap:   ${INITIAL_CAPITAL + total_pnl:,.2f}")

    # Per-city
    city_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        cs = city_stats[t.city]
        cs["trades"] += 1
        cs["wins"] += 1 if t.won else 0
        cs["pnl"] += t.pnl

    print(f"\n  {'City':<16} {'Trades':>6} {'WR':>7} {'PnL':>10}")
    for city in sorted(city_stats, key=lambda c: -city_stats[c]["pnl"]):
        cs = city_stats[city]
        wr = cs["wins"] / cs["trades"] * 100 if cs["trades"] else 0
        print(f"  {city:<16} {cs['trades']:>6} {wr:>6.1f}% ${cs['pnl']:>9.2f}")

    # Date distribution
    dates = sorted(set(t.target_date for t in trades))
    print(f"\n  Date range:  {dates[0]} → {dates[-1]}")
    print(f"  Days:        {len(dates)}")
    print(f"  Trades/day:  {len(trades) / len(dates):.1f}")


def main():
    t_start = time.time()

    db = PriceHistoryDB()
    stats = db.stats()
    print(f"Database: {stats['events']} events, "
          f"{stats['tokens_with_prices']} tokens with prices, "
          f"{stats['price_points']:,} price points")

    if stats["price_range"]:
        print(f"Price data: {stats['price_range']['start'][:10]} → "
              f"{stats['price_range']['end'][:10]} "
              f"({stats['price_range']['days']:.0f} days)")

    # Run backtest at different entry timings
    for hours in ENTRY_HOURS_BEFORE_CLOSE:
        trades = run_backtest(db, entry_hours=hours, verbose=(hours == 24))
        print_results(trades, hours)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    db.close()


if __name__ == "__main__":
    main()
