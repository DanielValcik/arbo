"""Crypto Price Database — Query Interface.

Provides access to downloaded Polymarket crypto price prediction market
data + Binance klines stored in SQLite. Use this module in backtests
and autoresearch scripts.

Usage:
    from crypto_price_db import CryptoPriceDB

    db = CryptoPriceDB()
    events = db.get_events(asset="BTC", market_type="daily_above")
    for ev in events:
        buckets = db.get_buckets(ev.event_id)
        for b in buckets:
            prices = db.get_price_history(b.token_id)

    # Exchange price at any timestamp
    btc_price = db.get_exchange_price_at("BTCUSDT", target_ts)
"""

import bisect
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "crypto_price_pmd.sqlite"


@dataclass
class CryptoEvent:
    """A resolved crypto price prediction event."""

    event_id: str
    title: str
    asset: str | None  # BTC, ETH, SOL
    market_type: str | None  # daily_above, monthly_hit, 5min, 15min
    resolution_date: str | None  # ISO date
    start_date: str | None
    end_date: str | None
    closed_time: str | None
    volume: float
    neg_risk: bool
    n_buckets: int


@dataclass
class CryptoBucket:
    """A price strike bucket (YES/NO market) within an event."""

    token_id: str
    token_id_no: str | None
    event_id: str
    condition_id: str | None
    question: str
    strike_price: float | None  # Strike in USD (e.g. 88000.0)
    direction: str | None  # "above" or "below"
    won: bool
    volume: float
    fee_enabled: bool


@dataclass
class PricePoint:
    """A single Polymarket price observation."""

    timestamp: int  # Unix timestamp
    price: float  # YES price (0.0 to 1.0)

    @property
    def dt(self) -> datetime:
        return datetime.utcfromtimestamp(self.timestamp)


@dataclass
class KlinePoint:
    """A single Binance OHLCV candle."""

    timestamp: int  # Unix timestamp (open time)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class EventSnapshot:
    """Complete market state at a point in time for one crypto event."""

    event: CryptoEvent
    buckets: list[CryptoBucket]
    prices: dict[str, float]  # token_id → Polymarket price at snapshot time
    exchange_price: float | None  # Binance price at snapshot time
    snapshot_time: datetime
    hours_before_close: float


class CryptoPriceDB:
    """Query interface for crypto price prediction market data."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Run: python3 research/download_crypto_price_history.py"
            )
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # Cache for bisect-based price lookups
        self._kline_cache: dict[str, list[tuple[int, float]]] = {}

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Events ────────────────────────────────────────────────────

    def get_events(
        self,
        asset: str | None = None,
        market_type: str | None = None,
        with_prices: bool = True,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[CryptoEvent]:
        """Get events, optionally filtered by asset and market type.

        Note: PMD schema uses 'markets' table (each market = one event/bucket).
        Groups markets by date into logical events.
        """
        query = "SELECT * FROM markets WHERE 1=1"
        params: list = []

        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if market_type:
            query += " AND market_type = ?"
            params.append(market_type)
        if min_date:
            query += " AND end_date >= ?"
            params.append(min_date)
        if max_date:
            query += " AND end_date <= ?"
            params.append(max_date)
        if with_prices:
            query += """ AND market_id IN (
                SELECT DISTINCT m2.market_id FROM markets m2
                JOIN json_each(m2.tokens_json) t
                JOIN prices p ON p.token_id = json_extract(t.value, '$.id')
            )"""

        query += " ORDER BY end_date"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_event(self, event_id: str) -> CryptoEvent | None:
        """Get a single event/market by ID."""
        row = self._conn.execute(
            "SELECT * FROM markets WHERE market_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    # ── Buckets ───────────────────────────────────────────────────

    def get_buckets(self, event_id: str) -> list[CryptoBucket]:
        """Get bucket for a market. Each market IS one bucket in PMD schema."""
        row = self._conn.execute(
            "SELECT * FROM markets WHERE market_id = ?", (event_id,)
        ).fetchone()
        if not row:
            return []
        return [self._row_to_bucket(row)]

    # ── Polymarket Prices ─────────────────────────────────────────

    def get_price_history(self, token_id: str) -> list[PricePoint]:
        """Get full Polymarket price history for a token."""
        rows = self._conn.execute(
            "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
            (token_id,),
        ).fetchall()
        return [PricePoint(timestamp=r["ts"], price=r["price"]) for r in rows]

    def get_price_at(self, token_id: str, target_ts: int) -> float | None:
        """Get Polymarket price closest to (but not after) target timestamp."""
        row = self._conn.execute(
            """SELECT price FROM prices
               WHERE token_id = ? AND ts <= ?
               ORDER BY ts DESC LIMIT 1""",
            (token_id, target_ts),
        ).fetchone()
        return row["price"] if row else None

    # ── Binance Klines ────────────────────────────────────────────

    def get_klines(
        self, symbol: str, start_ts: int | None = None, end_ts: int | None = None,
    ) -> list[KlinePoint]:
        """Get Binance klines for a symbol, optionally within time range."""
        query = "SELECT ts, open, high, low, close, volume FROM binance_klines WHERE symbol = ?"
        params: list = [symbol]
        if start_ts is not None:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND ts <= ?"
            params.append(end_ts)
        query += " ORDER BY ts"

        rows = self._conn.execute(query, params).fetchall()
        return [
            KlinePoint(
                timestamp=r["ts"], open=r["open"], high=r["high"],
                low=r["low"], close=r["close"], volume=r["volume"],
            )
            for r in rows
        ]

    def get_exchange_price_at(self, symbol: str, target_ts: int) -> float | None:
        """Get Binance close price closest to (but not after) target timestamp.

        Uses cached sorted list + bisect for O(log n) lookups.
        """
        if symbol not in self._kline_cache:
            rows = self._conn.execute(
                "SELECT ts, close FROM binance_klines WHERE symbol = ? ORDER BY ts",
                (symbol,),
            ).fetchall()
            self._kline_cache[symbol] = [(r["ts"], r["close"]) for r in rows]

        data = self._kline_cache[symbol]
        if not data:
            return None

        # Binary search for closest timestamp <= target_ts
        idx = bisect.bisect_right(data, (target_ts, float("inf"))) - 1
        if idx < 0:
            return None
        return data[idx][1]

    def get_exchange_prices_bulk(self, symbol: str) -> list[tuple[int, float]]:
        """Get all (timestamp, close_price) pairs for a symbol. For preloading."""
        if symbol not in self._kline_cache:
            rows = self._conn.execute(
                "SELECT ts, close FROM binance_klines WHERE symbol = ? ORDER BY ts",
                (symbol,),
            ).fetchall()
            self._kline_cache[symbol] = [(r["ts"], r["close"]) for r in rows]
        return self._kline_cache[symbol]

    # ── Snapshots ─────────────────────────────────────────────────

    def get_price_snapshot(
        self, event_id: str, hours_before_close: float = 24,
    ) -> EventSnapshot | None:
        """Get prices for market N hours before close.

        Also includes the Binance exchange price at that time.
        """
        event = self.get_event(event_id)
        if not event or not event.closed_time:
            return None

        try:
            closed_dt = datetime.fromisoformat(
                event.closed_time.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None
        snapshot_dt = closed_dt - timedelta(hours=hours_before_close)
        snapshot_ts = int(snapshot_dt.timestamp())

        buckets = self.get_buckets(event_id)
        prices: dict[str, float] = {}
        for b in buckets:
            if b.token_id:
                p = self.get_price_at(b.token_id, snapshot_ts)
                if p is not None:
                    prices[b.token_id] = p

        if not prices:
            return None

        # Get exchange price at snapshot time
        symbol = f"{event.asset}USDT" if event.asset else None
        exchange_price = None
        if symbol:
            exchange_price = self.get_exchange_price_at(symbol, snapshot_ts)

        return EventSnapshot(
            event=event,
            buckets=buckets,
            prices=prices,
            exchange_price=exchange_price,
            snapshot_time=snapshot_dt,
            hours_before_close=hours_before_close,
        )

    def get_all_snapshots(
        self,
        asset: str | None = None,
        market_type: str | None = None,
        hours_before_close: float = 24,
    ) -> list[EventSnapshot]:
        """Get price snapshots for all events, sorted by resolution date."""
        events = self.get_events(asset=asset, market_type=market_type, with_prices=True)
        snapshots = []
        for ev in events:
            snap = self.get_price_snapshot(ev.event_id, hours_before_close)
            if snap and snap.prices:
                snapshots.append(snap)
        return snapshots

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get database statistics."""
        n_events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_buckets = self._conn.execute("SELECT COUNT(*) FROM buckets").fetchone()[0]
        n_prices = self._conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        n_tokens = self._conn.execute(
            "SELECT COUNT(DISTINCT token_id) FROM prices"
        ).fetchone()[0]
        n_klines = self._conn.execute(
            "SELECT COUNT(*) FROM binance_klines"
        ).fetchone()[0]

        row = self._conn.execute(
            "SELECT MIN(ts), MAX(ts) FROM prices"
        ).fetchone()
        price_range = None
        if row[0]:
            price_range = {
                "start": datetime.utcfromtimestamp(row[0]).isoformat(),
                "end": datetime.utcfromtimestamp(row[1]).isoformat(),
                "days": (row[1] - row[0]) / 86400,
            }

        return {
            "events": n_events,
            "buckets": n_buckets,
            "price_points": n_prices,
            "tokens_with_prices": n_tokens,
            "binance_klines": n_klines,
            "price_range": price_range,
        }

    def get_assets(self) -> list[dict]:
        """Get summary of available assets."""
        rows = self._conn.execute("""
            SELECT e.asset, e.market_type,
                   COUNT(DISTINCT e.event_id) as total_events,
                   COUNT(DISTINCT b.token_id) as total_buckets,
                   MIN(e.resolution_date) as first_date,
                   MAX(e.resolution_date) as last_date,
                   SUM(e.volume) as total_volume
            FROM events e
            LEFT JOIN buckets b ON b.event_id = e.event_id
            WHERE e.asset IS NOT NULL
            GROUP BY e.asset, e.market_type
            ORDER BY total_volume DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row) -> CryptoEvent:
        """Convert PMD 'markets' row to CryptoEvent."""
        return CryptoEvent(
            event_id=row["market_id"],
            title=row["question"] or "",
            asset=row["asset"],
            market_type=row["market_type"],
            resolution_date=row["end_date"],  # end_date = resolution time
            start_date=row["start_date"],
            end_date=row["end_date"],
            closed_time=row["end_date"],  # PMD uses end_date as close time
            volume=0,  # PMD doesn't provide volume
            neg_risk=False,  # Crypto markets are NOT NegRisk
            n_buckets=1,  # Each market = one bucket in PMD
        )

    def _row_to_bucket(self, row) -> CryptoBucket:
        """Convert PMD 'markets' row to CryptoBucket."""
        # Extract token IDs from tokens_json
        import json
        tokens_json = row["tokens_json"] or "[]"
        try:
            tokens = json.loads(tokens_json)
        except (json.JSONDecodeError, TypeError):
            tokens = []

        yes_token = ""
        no_token = None
        for t in tokens:
            if t.get("label") == "Yes":
                yes_token = t.get("id", "")
            elif t.get("label") == "No":
                no_token = t.get("id")

        won_val = row["won"] if "won" in row.keys() else None

        return CryptoBucket(
            token_id=yes_token,
            token_id_no=no_token,
            event_id=row["market_id"],
            condition_id=row["market_id"],
            question=row["question"] or "",
            strike_price=row["strike_price"],
            direction=row["direction"],
            won=bool(won_val) if won_val is not None else False,
            volume=0,
            fee_enabled=row["market_type"] != "monthly_hit",  # Daily = fee, monthly = no fee
        )
