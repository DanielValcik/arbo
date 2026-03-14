"""Price History Database — Query Interface.

Provides easy access to downloaded Polymarket price history data
stored in SQLite. Use this module in backtests and research scripts.

Usage:
    from price_history_db import PriceHistoryDB

    db = PriceHistoryDB()
    events = db.get_events(city="paris")
    for ev in events:
        prices = db.get_prices(ev.event_id)
        for token_id, ts, price in prices:
            ...

    # Or get a complete snapshot at a specific time
    snapshot = db.get_price_snapshot(event_id, hours_before_close=24)
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "price_history.sqlite"


@dataclass
class Event:
    """A resolved weather temperature event."""

    event_id: str
    title: str
    city: str | None
    target_date: str | None  # ISO date
    start_date: str | None
    end_date: str | None
    closed_time: str | None
    volume: float
    neg_risk: bool
    n_buckets: int


@dataclass
class Bucket:
    """A temperature bucket (YES/NO market) within an event."""

    token_id: str
    token_id_no: str | None
    event_id: str
    condition_id: str | None
    question: str
    low_c: float | None  # Lower bound in Celsius
    high_c: float | None  # Upper bound in Celsius
    bucket_type: str | None  # "range", "below", "above", "exact"
    unit: str | None  # "F" or "C"
    won: bool
    volume: float


@dataclass
class PricePoint:
    """A single price observation."""

    timestamp: int  # Unix timestamp
    price: float  # YES price (0.0 to 1.0)

    @property
    def dt(self) -> datetime:
        return datetime.utcfromtimestamp(self.timestamp)


@dataclass
class EventSnapshot:
    """Complete market state at a point in time for one event."""

    event: Event
    buckets: list[Bucket]
    prices: dict[str, float]  # token_id → price at snapshot time
    snapshot_time: datetime
    hours_before_close: float


class PriceHistoryDB:
    """Query interface for Polymarket price history."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Run: python3 research/download_price_history.py"
            )
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Events ────────────────────────────────────────────────────────

    def get_events(
        self,
        city: str | None = None,
        with_prices: bool = True,
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> list[Event]:
        """Get events, optionally filtered by city and date range.

        Args:
            city: Filter by city ID (e.g. "paris", "chicago").
            with_prices: Only return events that have price data.
            min_date: Minimum target_date (ISO format).
            max_date: Maximum target_date (ISO format).
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if city:
            query += " AND city = ?"
            params.append(city)
        if min_date:
            query += " AND target_date >= ?"
            params.append(min_date)
        if max_date:
            query += " AND target_date <= ?"
            params.append(max_date)
        if with_prices:
            if self._has_goldsky():
                query += """ AND event_id IN (
                    SELECT DISTINCT b.event_id FROM buckets b
                    LEFT JOIN prices p ON p.token_id = b.token_id
                    LEFT JOIN goldsky_trades g ON g.token_id = b.token_id
                    WHERE p.token_id IS NOT NULL OR g.token_id IS NOT NULL
                )"""
            else:
                query += """ AND event_id IN (
                    SELECT DISTINCT b.event_id FROM buckets b
                    JOIN prices p ON p.token_id = b.token_id
                )"""

        query += " ORDER BY target_date"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_event(self, event_id: str) -> Event | None:
        """Get a single event by ID."""
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_cities(self) -> list[dict]:
        """Get summary of available cities with event counts."""
        rows = self._conn.execute("""
            SELECT e.city,
                   COUNT(DISTINCT e.event_id) as total_events,
                   COUNT(DISTINCT CASE WHEN p.token_id IS NOT NULL
                         THEN e.event_id END) as events_with_prices,
                   MIN(e.target_date) as first_date,
                   MAX(e.target_date) as last_date
            FROM events e
            LEFT JOIN buckets b ON b.event_id = e.event_id
            LEFT JOIN prices p ON p.token_id = b.token_id
            WHERE e.city IS NOT NULL
            GROUP BY e.city
            ORDER BY total_events DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # ── Buckets ───────────────────────────────────────────────────────

    def get_buckets(self, event_id: str) -> list[Bucket]:
        """Get all buckets for an event."""
        rows = self._conn.execute(
            "SELECT * FROM buckets WHERE event_id = ? ORDER BY low_c",
            (event_id,),
        ).fetchall()
        return [self._row_to_bucket(r) for r in rows]

    # ── Prices ────────────────────────────────────────────────────────

    def _has_goldsky(self) -> bool:
        """Check if goldsky_trades table exists."""
        if not hasattr(self, "_goldsky_available"):
            row = self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='goldsky_trades'"
            ).fetchone()
            self._goldsky_available = row is not None
        return self._goldsky_available

    def get_price_history(self, token_id: str) -> list[PricePoint]:
        """Get full price history for a token.

        Merges data from both CLOB /prices-history and Goldsky on-chain
        trades. Deduplicates by hour, preferring CLOB data when both exist.
        """
        if self._has_goldsky():
            rows = self._conn.execute(
                """SELECT ts, price FROM (
                    SELECT ts, price, 1 as priority FROM prices
                        WHERE token_id = ?
                    UNION ALL
                    SELECT ts, price, 2 as priority FROM goldsky_trades
                        WHERE token_id = ?
                ) sub
                GROUP BY ts
                ORDER BY ts""",
                (token_id, token_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts",
                (token_id,),
            ).fetchall()
        return [PricePoint(timestamp=r["ts"], price=r["price"]) for r in rows]

    def get_price_at(self, token_id: str, target_ts: int) -> float | None:
        """Get the price closest to (but not after) a target timestamp.

        Searches both CLOB prices and Goldsky trades, returns the closest.
        """
        if self._has_goldsky():
            row = self._conn.execute(
                """SELECT price FROM (
                    SELECT ts, price FROM prices
                        WHERE token_id = ? AND ts <= ?
                    UNION ALL
                    SELECT ts, price FROM goldsky_trades
                        WHERE token_id = ? AND ts <= ?
                ) sub
                ORDER BY ts DESC LIMIT 1""",
                (token_id, target_ts, token_id, target_ts),
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT price FROM prices
                   WHERE token_id = ? AND ts <= ?
                   ORDER BY ts DESC LIMIT 1""",
                (token_id, target_ts),
            ).fetchone()
        return row["price"] if row else None

    def get_price_snapshot(
        self, event_id: str, hours_before_close: float = 24
    ) -> EventSnapshot | None:
        """Get prices for all buckets at a specific time before market close.

        This is the key method for backtesting: it tells you what each
        bucket's price was N hours before the market resolved.

        Args:
            event_id: The event ID.
            hours_before_close: Hours before market closed to snapshot.
                E.g., 24 = prices 1 day before close.
        """
        event = self.get_event(event_id)
        if not event or not event.closed_time:
            return None

        closed_dt = datetime.fromisoformat(
            event.closed_time.replace("Z", "+00:00")
        )
        snapshot_dt = closed_dt - timedelta(hours=hours_before_close)
        snapshot_ts = int(snapshot_dt.timestamp())

        buckets = self.get_buckets(event_id)
        prices = {}
        for b in buckets:
            p = self.get_price_at(b.token_id, snapshot_ts)
            if p is not None:
                prices[b.token_id] = p

        if not prices:
            return None

        return EventSnapshot(
            event=event,
            buckets=buckets,
            prices=prices,
            snapshot_time=snapshot_dt,
            hours_before_close=hours_before_close,
        )

    def get_all_snapshots(
        self,
        city: str | None = None,
        hours_before_close: float = 24,
    ) -> list[EventSnapshot]:
        """Get price snapshots for all events (optionally filtered by city).

        Returns snapshots sorted by target_date. Useful for walk-forward
        backtesting on real market data.
        """
        events = self.get_events(city=city, with_prices=True)
        snapshots = []
        for ev in events:
            snap = self.get_price_snapshot(ev.event_id, hours_before_close)
            if snap and snap.prices:
                snapshots.append(snap)
        return snapshots

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get database statistics."""
        n_events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_buckets = self._conn.execute("SELECT COUNT(*) FROM buckets").fetchone()[0]
        n_clob = self._conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        n_clob_tokens = self._conn.execute(
            "SELECT COUNT(DISTINCT token_id) FROM prices"
        ).fetchone()[0]

        n_goldsky = 0
        n_goldsky_tokens = 0
        if self._has_goldsky():
            n_goldsky = self._conn.execute(
                "SELECT COUNT(*) FROM goldsky_trades"
            ).fetchone()[0]
            n_goldsky_tokens = self._conn.execute(
                "SELECT COUNT(DISTINCT token_id) FROM goldsky_trades"
            ).fetchone()[0]

        # Combined price range
        if self._has_goldsky():
            row = self._conn.execute("""
                SELECT MIN(ts), MAX(ts) FROM (
                    SELECT ts FROM prices
                    UNION ALL
                    SELECT ts FROM goldsky_trades
                )
            """).fetchone()
        else:
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
            "price_points": n_clob + n_goldsky,
            "clob_prices": n_clob,
            "goldsky_prices": n_goldsky,
            "tokens_with_prices": n_clob_tokens,
            "goldsky_tokens": n_goldsky_tokens,
            "price_range": price_range,
        }

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row) -> Event:
        return Event(
            event_id=row["event_id"],
            title=row["title"],
            city=row["city"],
            target_date=row["target_date"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            closed_time=row["closed_time"],
            volume=row["volume"] or 0,
            neg_risk=bool(row["neg_risk"]),
            n_buckets=row["n_buckets"] or 0,
        )

    @staticmethod
    def _row_to_bucket(row) -> Bucket:
        return Bucket(
            token_id=row["token_id"],
            token_id_no=row["token_id_no"],
            event_id=row["event_id"],
            condition_id=row["condition_id"],
            question=row["question"],
            low_c=row["low_c"],
            high_c=row["high_c"],
            bucket_type=row["bucket_type"],
            unit=row["unit"],
            won=bool(row["won"]),
            volume=row["volume"] or 0,
        )
