"""Load crypto_price_pmd.sqlite into experiment_framework SimulationData format.

Bridges crypto price prediction market data into the existing portfolio
simulation infrastructure. The key mapping:

    Weather concept    →  Crypto concept
    ─────────────────────────────────────
    city               →  asset (BTC, ETH)
    forecast_temp      →  exchange_price (Binance close)
    bucket.low_c       →  strike_price
    bucket.high_c      →  None (not used — crypto is binary above/below)
    bucket.bucket_type →  direction (above/below)
    forecasts[city]    →  binance_prices[asset] (ts → price)

The autoresearch sweep monkey-patches compute_prob() to use the volatility
model instead of the weather Normal CDF.

Usage:
    from crypto_price_loader import load_crypto_simulation_data
    sim_data = load_crypto_simulation_data()
"""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Add parent dirs for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from crypto_price_db import CryptoPriceDB
from price_history_db import Bucket, Event

CRYPTO_DB_PATH = Path(__file__).parent.parent / "data" / "crypto_price_pmd.sqlite"


def load_crypto_simulation_data(
    db_path: str | Path | None = None,
    asset: str | None = None,
    market_type: str | None = None,
) -> "CryptoSimulationData":
    """Load crypto price data into simulation-ready format.

    Args:
        db_path: Path to crypto_price_pmd.sqlite.
        asset: Filter by asset ("BTC", "ETH"). None = all.
        market_type: Filter by type ("daily_above", "monthly_hit"). None = all.

    Returns:
        CryptoSimulationData object ready for simulate_portfolio().
    """
    db_path = Path(db_path) if db_path else CRYPTO_DB_PATH

    import json
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load markets directly from PMD schema (each market = one event + one bucket)
    query = "SELECT * FROM markets WHERE won IS NOT NULL"
    q_params: list = []
    if asset:
        query += " AND asset = ?"
        q_params.append(asset)
    if market_type:
        query += " AND market_type = ?"
        q_params.append(market_type)
    query += " ORDER BY end_date"

    markets = conn.execute(query, q_params).fetchall()

    events: list[Event] = []
    buckets_by_event: dict[str, list[Bucket]] = {}
    token_ids_needed: set[str] = set()

    for m in markets:
        market_id = m["market_id"]
        asset_name = m["asset"] or "BTC"

        # Parse tokens from JSON
        try:
            tokens = json.loads(m["tokens_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            tokens = []
        yes_token = ""
        no_token = None
        for t in tokens:
            if t.get("label") == "Yes":
                yes_token = t.get("id", "")
            elif t.get("label") == "No":
                no_token = t.get("id")
        if not yes_token:
            continue

        # Event (city = asset for experiment_framework compatibility)
        event = Event(
            event_id=market_id,
            title=m["question"] or "",
            city=asset_name,  # city = asset
            target_date=m["end_date"][:10] if m["end_date"] else None,
            start_date=m["start_date"],
            end_date=m["end_date"],
            closed_time=m["end_date"],
            volume=0,
            neg_risk=False,
            n_buckets=1,
        )
        events.append(event)

        # Bucket (low_c = strike, bucket_type = direction)
        bucket = Bucket(
            token_id=yes_token,
            token_id_no=no_token,
            event_id=market_id,
            condition_id=market_id,
            question=m["question"] or "",
            low_c=m["strike_price"],  # strike stored as low_c
            high_c=None,
            bucket_type=m["direction"] or "above",  # direction stored as bucket_type
            unit="USD",
            won=bool(m["won"]),
            volume=0,
        )
        buckets_by_event[market_id] = [bucket]
        token_ids_needed.add(yes_token)
        if no_token:
            token_ids_needed.add(no_token)

    # Bulk load Polymarket prices
    prices_dict: dict[str, list[tuple[int, float]]] = defaultdict(list)
    conn.row_factory = None
    for row in conn.execute(
        "SELECT token_id, ts, price FROM prices ORDER BY token_id, ts"
    ).fetchall():
        tid = row[0]
        if tid in token_ids_needed:
            prices_dict[tid].append((row[1], row[2]))
    conn.row_factory = sqlite3.Row

    # Load Binance klines
    binance_prices: dict[str, list[tuple[int, float]]] = {}
    conn.row_factory = None
    for row in conn.execute(
        "SELECT symbol, ts, close FROM binance_klines ORDER BY symbol, ts"
    ).fetchall():
        symbol = row[0]
        if symbol not in binance_prices:
            binance_prices[symbol] = []
        binance_prices[symbol].append((row[1], row[2]))
    conn.row_factory = sqlite3.Row

    # Build forecasts: asset → {date_str → exchange_price}
    forecasts: dict[str, dict[str, float]] = {}
    assets_seen: set[str] = set()
    for ev in events:
        a = ev.city  # city = asset
        if a in assets_seen:
            continue
        assets_seen.add(a)
        symbol = f"{a}USDT"
        if symbol in binance_prices:
            day_prices: dict[str, float] = {}
            for ts, price in binance_prices[symbol]:
                dt = datetime.utcfromtimestamp(ts)
                day_str = dt.strftime("%Y-%m-%d")
                day_prices[day_str] = price
            forecasts[a] = day_prices

    n_points = sum(len(v) for v in prices_dict.values())
    n_klines = sum(len(v) for v in binance_prices.values())
    print(
        f"Crypto data loaded: {len(events)} events, "
        f"{sum(len(b) for b in buckets_by_event.values())} buckets, "
        f"{n_points:,} Polymarket prices, {n_klines:,} Binance klines"
    )

    conn.close()

    return CryptoSimulationData(
        events=events,
        buckets_by_event=buckets_by_event,
        prices=dict(prices_dict),
        forecasts=forecasts,
        binance_prices=binance_prices,
    )


class CryptoSimulationData:
    """Pre-loaded crypto data with indexes for fast portfolio simulation.

    Extends the experiment_framework.SimulationData interface with
    Binance kline lookups for real-time exchange price queries.
    """

    def __init__(
        self,
        events: list[Event],
        buckets_by_event: dict[str, list[Bucket]],
        prices: dict[str, list[tuple[int, float]]],
        forecasts: dict[str, dict[str, float]],
        binance_prices: dict[str, list[tuple[int, float]]],
    ):
        self.events = events
        self.buckets_by_event = buckets_by_event
        self._prices = prices  # token_id → sorted [(ts, price)]
        self.forecasts = forecasts  # asset → {date_str → exchange_price}
        self._binance = binance_prices  # symbol → sorted [(ts, close_price)]

        # Parse close timestamps + build lookups
        self._close_ts: dict[str, int] = {}
        self._events_by_id: dict[str, Event] = {}
        for ev in events:
            self._events_by_id[ev.event_id] = ev
            if ev.closed_time:
                try:
                    ct = ev.closed_time.replace("Z", "+00:00")
                    self._close_ts[ev.event_id] = int(
                        datetime.fromisoformat(ct).timestamp()
                    )
                except (ValueError, AttributeError):
                    pass

    def get_price(self, token_id: str, hour_ts: int) -> float | None:
        """Get most recent Polymarket price at or before hour_ts."""
        pts = self._prices.get(token_id)
        if not pts:
            return None
        idx = bisect.bisect_right(pts, (hour_ts, float("inf"))) - 1
        if idx < 0:
            return None
        if hour_ts - pts[idx][0] > 7 * 86400:
            return None
        return pts[idx][1]

    def get_exchange_price(self, symbol: str, hour_ts: int) -> float | None:
        """Get Binance close price at or before hour_ts."""
        pts = self._binance.get(symbol)
        if not pts:
            return None
        idx = bisect.bisect_right(pts, (hour_ts, float("inf"))) - 1
        if idx < 0:
            return None
        if hour_ts - pts[idx][0] > 3600:  # Max 1 hour stale for crypto
            return None
        return pts[idx][1]

    def get_exchange_prices_window(
        self, symbol: str, end_ts: int, window_hours: int,
    ) -> list[float]:
        """Get Binance close prices for the last N hours ending at end_ts.

        Returns list of close prices (for volatility computation).
        """
        pts = self._binance.get(symbol)
        if not pts:
            return []
        start_ts = end_ts - window_hours * 3600
        i_start = bisect.bisect_left(pts, (start_ts,))
        i_end = bisect.bisect_right(pts, (end_ts, float("inf")))
        return [p for _, p in pts[i_start:i_end]]

    def get_close_ts(self, event_id: str) -> int | None:
        return self._close_ts.get(event_id)

    def build_entry_index(
        self, entry_hours: float | list[float],
    ) -> dict[int, list[tuple[Event, int]]]:
        """Map: entry_hour_ts -> [(event, days_out), ...]."""
        if isinstance(entry_hours, (int, float)):
            entry_hours = [entry_hours]

        index: dict[int, list[tuple[Event, int]]] = defaultdict(list)
        for hours in entry_hours:
            offset = int(hours * 3600)
            days_out = max(0, int(hours / 24))
            for ev in self.events:
                close_ts = self._close_ts.get(ev.event_id)
                if close_ts is None:
                    continue
                entry_ts = close_ts - offset
                entry_hour = (entry_ts // 3600) * 3600
                index[entry_hour].append((ev, days_out))
        return dict(index)

    def build_close_index(self) -> dict[int, list[Event]]:
        """Map: close_hour_ts -> events resolving that hour."""
        index: dict[int, list[Event]] = defaultdict(list)
        for ev in self.events:
            close_ts = self._close_ts.get(ev.event_id)
            if close_ts is None:
                continue
            close_hour = (close_ts // 3600) * 3600
            index[close_hour].append(ev)
        return dict(index)

    def get_period(self, entry_hours: float | list[float] = 48) -> tuple[int, int]:
        """Get simulation period (earliest entry, latest close)."""
        all_close_ts = list(self._close_ts.values())
        if not all_close_ts:
            return (0, 0)
        max_hours = max(entry_hours) if isinstance(entry_hours, list) else entry_hours
        offset = int(max_hours * 3600)
        start = min(all_close_ts) - offset
        end = max(all_close_ts)
        return ((start // 3600) * 3600, ((end // 3600) + 1) * 3600)

    def filter_events(
        self, min_date: str | None = None, max_date: str | None = None,
    ) -> "CryptoSimulationData":
        """Return new CryptoSimulationData with events filtered by date."""
        filtered = [
            ev for ev in self.events
            if ev.target_date
            and (min_date is None or ev.target_date >= min_date)
            and (max_date is None or ev.target_date <= max_date)
        ]
        return CryptoSimulationData(
            filtered, self.buckets_by_event, self._prices,
            self.forecasts, self._binance,
        )


if __name__ == "__main__":
    sim = load_crypto_simulation_data()
    from collections import Counter

    assets = Counter(e.city for e in sim.events)  # city = asset
    print(f"\nEvents by asset:")
    for asset, cnt in assets.most_common():
        print(f"  {asset}: {cnt}")

    dates = sorted(set(e.target_date for e in sim.events if e.target_date))
    if dates:
        print(f"\nDate range: {dates[0]} to {dates[-1]}")
        print(f"Unique dates: {len(dates)}")

    total_buckets = sum(len(b) for b in sim.buckets_by_event.values())
    won_count = sum(1 for bl in sim.buckets_by_event.values() for b in bl if b.won)
    print(f"\nTotal buckets: {total_buckets}")
    print(f"Won (resolved YES): {won_count}")
