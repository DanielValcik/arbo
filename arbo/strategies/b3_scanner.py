"""B3 Scanner — Discover and scan BTC 5-min Up/Down markets.

Fetches active BTC Up/Down events from Gamma API, determines which are
in the entry window (minute 2), computes signal/market fair value from
Binance price + realized volatility, and produces B3Signal objects.

Unlike B2 which uses market_discovery.py, B3 manages its own event
lifecycle because 5-min markets need precise timing tracking.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp

from arbo.strategies.b3_quality_gate import (
    CONTRARIAN,
    ENTRY_THRESHOLD,
    MAX_ENTRY_MIN,
    MAX_ENTRY_MKT_FV,
    MIN_ENTRY_MIN,
    MIN_ENTRY_MKT_FV,
    SIGMA_FLOOR,
    SIGMA_SCALE,
    WINDOW_MIN,
)
from arbo.utils.logger import get_logger

logger = get_logger("b3_scanner")

GAMMA_URL = "https://gamma-api.polymarket.com"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass
class B3Signal:
    """A B3 scalping signal for a 5-min BTC Up/Down event."""

    condition_id: str
    token_id_up: str
    token_id_down: str
    question: str

    # Event timing
    event_start_ts: float       # Unix timestamp of event start
    event_end_ts: float         # Unix timestamp of resolution (start + 5 min)
    minutes_elapsed: float      # Minutes since event start

    # BTC prices
    btc_at_start: float         # BTC price at event start
    btc_now: float              # Current BTC price

    # Fair values
    signal_fv_up: float         # P(Up) with sigma_scale (entry trigger)
    market_fv_up: float         # P(Up) with sigma_scale=1.0 (realistic price)

    # Trade
    direction: int              # +1 = long Up, -1 = long Down
    entry_price: float          # market_fv for our direction + spread
    edge: float                 # |signal_fv - 0.50|

    # Volatility
    sigma_per_min: float


@dataclass
class B3Event:
    """Tracked 5-min BTC Up/Down event."""

    condition_id: str
    token_id_up: str
    token_id_down: str
    question: str
    start_ts: float
    end_ts: float
    btc_at_start: float | None = None  # Set when event starts
    traded: bool = False               # Already entered this event


class B3Scanner:
    """Discovers and tracks BTC 5-min Up/Down markets."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._events: dict[str, B3Event] = {}  # condition_id → event
        self._last_fetch_ts: float = 0
        self._fetch_interval_s: float = 120  # Refetch events every 2 min

    async def init(self) -> None:
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_events(self) -> None:
        """Fetch active BTC Up/Down events from Gamma API.

        Uses predictive slug generation because Gamma /events endpoint
        with tag_slug filters does NOT reliably return current 5-min events.
        Slug format: btc-updown-5m-{unix_start_ts} where start_ts is
        aligned to 5-minute boundaries.

        Fetches the current window + next 2 windows (covers 15 min ahead).
        """
        now = time.time()
        if now - self._last_fetch_ts < self._fetch_interval_s:
            return
        self._last_fetch_ts = now

        if not self._session:
            return

        # Generate slugs for current + upcoming windows
        window_start = int(now // 300) * 300  # Round down to 5-min boundary
        slugs_to_check = [
            f"btc-updown-5m-{window_start + offset * 300}"
            for offset in range(-1, 3)  # Previous, current, next 2
        ]

        new_count = 0
        for slug in slugs_to_check:
            # Skip if we already have this event (by slug → start_ts)
            try:
                slug_ts = int(slug.split("-")[-1])
            except ValueError:
                continue
            # Check if already tracked by start_ts
            if any(
                abs(ev.start_ts - slug_ts) < 10
                for ev in self._events.values()
            ):
                continue

            try:
                async with self._session.get(
                    f"{GAMMA_URL}/events", params={"slug": slug},
                ) as resp:
                    if resp.status != 200:
                        continue
                    events = await resp.json()

                if not events:
                    continue
                event = events[0] if isinstance(events, list) else events

                title = (event.get("title", "") or "").lower()
                if "bitcoin" not in title:
                    continue

                nested = event.get("markets", [])
                if not isinstance(nested, list) or not nested:
                    continue

                raw_market = nested[0]
                if not isinstance(raw_market, dict):
                    continue

                cid = raw_market.get("conditionId", "")
                if not cid or cid in self._events:
                    continue

                # Parse token IDs
                clob_ids = raw_market.get("clobTokenIds", [])
                if isinstance(clob_ids, str):
                    try:
                        clob_ids = json.loads(clob_ids)
                    except Exception:
                        clob_ids = []
                if len(clob_ids) < 2:
                    continue

                # Parse end date → compute start as end - 5min
                end_date_str = raw_market.get("endDate")
                if not end_date_str:
                    continue

                try:
                    end_dt = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                    end_ts = end_dt.timestamp()
                    start_ts = end_ts - WINDOW_MIN * 60
                except Exception:
                    continue

                question = raw_market.get("question", event.get("title", ""))

                # Fetch BTC price at event start from Binance klines API.
                # This matches the backtest's closes[start_idx] exactly.
                btc_start = await self._fetch_btc_at_start(start_ts)

                self._events[cid] = B3Event(
                    condition_id=cid,
                    token_id_up=clob_ids[0],
                    token_id_down=clob_ids[1],
                    question=question,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    btc_at_start=btc_start,
                )
                new_count += 1
                if btc_start:
                    logger.info(
                        "b3_event_btc_start",
                        slug=slug,
                        btc_at_start=f"${btc_start:,.2f}",
                    )

            except Exception as e:
                logger.warning("b3_slug_fetch_error", slug=slug, error=str(e))

        # Retry btc_at_start for events where Binance kline wasn't ready yet
        # (happens for future windows fetched before their kline exists)
        for ev in self._events.values():
            if ev.btc_at_start is None and now >= ev.start_ts + 60:
                # Kline should be available now (event started > 1 min ago)
                btc_start = await self._fetch_btc_at_start(ev.start_ts)
                if btc_start:
                    ev.btc_at_start = btc_start
                    logger.info(
                        "b3_btc_start_retry_ok",
                        question=ev.question[:50],
                        btc_at_start=f"${btc_start:,.2f}",
                    )

        # Clean expired events (ended > 10 min ago)
        expired = [
            cid for cid, ev in self._events.items()
            if now - ev.end_ts > 600
        ]
        for cid in expired:
            del self._events[cid]

        if new_count > 0 or expired:
            logger.info(
                "b3_events_updated",
                new=new_count,
                expired=len(expired),
                active=len(self._events),
            )

    async def _fetch_btc_at_start(self, event_start_ts: float) -> float | None:
        """Fetch BTC close price at event start from Binance 1-min klines.

        Matches backtest's `S_start = closes[start_idx]` — the close price
        of the 1-minute candle at the event start timestamp.

        Args:
            event_start_ts: Unix timestamp of the event start.

        Returns:
            BTC close price at event start, or None if unavailable.
        """
        if not self._session:
            return None
        try:
            start_ms = int(event_start_ts * 1000)
            async with self._session.get(
                BINANCE_KLINES_URL,
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": str(start_ms),
                    "limit": "1",
                },
            ) as resp:
                if resp.status != 200:
                    return None
                klines = await resp.json()
                if klines and len(klines) > 0:
                    # Kline format: [open_time, open, high, low, close, ...]
                    return float(klines[0][4])  # close price
        except Exception as e:
            logger.warning("b3_binance_kline_error", error=str(e))
        return None

    async def fetch_resolution(self, event_start_ts: float) -> bool | None:
        """Fetch market resolution from Polymarket Gamma API (Chainlink oracle).

        Returns True if UP won, False if DOWN won, None if not yet resolved.
        """
        if not self._session:
            return None
        slug = f"btc-updown-5m-{int(event_start_ts)}"
        try:
            async with self._session.get(
                f"{GAMMA_URL}/events", params={"slug": slug},
            ) as resp:
                if resp.status != 200:
                    return None
                events = await resp.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            markets = event.get("markets", [])
            if not markets:
                return None
            m = markets[0]
            if not m.get("closed"):
                return None
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))
            for i, o in enumerate(outcomes):
                if o.lower() == "up" and i < len(prices):
                    if prices[i] == "1":
                        return True   # UP won
                    # Check if DOWN explicitly won
                    for j, o2 in enumerate(outcomes):
                        if o2.lower() == "down" and j < len(prices) and prices[j] == "1":
                            return False  # DOWN won
                    return None  # Not yet resolved (both "0")
        except Exception as e:
            logger.warning("b3_resolution_fetch_error", slug=slug, error=str(e))
        return None

    def scan(
        self,
        btc_price: float,
        sigma_per_min: float,
    ) -> list[B3Signal]:
        """Scan active events for entry signals.

        Args:
            btc_price: Current BTC price from Binance.
            sigma_per_min: Per-minute realized volatility.

        Returns:
            List of B3Signal objects for events with entry triggers.
        """
        now = time.time()
        signals: list[B3Signal] = []
        sigma_per_min = max(sigma_per_min, SIGMA_FLOOR)

        for ev in self._events.values():
            if ev.traded:
                continue

            # How many minutes have elapsed since event start?
            elapsed_s = now - ev.start_ts
            elapsed_min = elapsed_s / 60.0

            # Only check at the entry window
            if elapsed_min < MIN_ENTRY_MIN or elapsed_min >= MAX_ENTRY_MIN + 1:
                continue

            # btc_at_start must be set by fetch_events via Binance klines API.
            # If missing (API failed), skip this event — wrong S_start causes
            # bad FV computation and 3x larger stop losses.
            if ev.btc_at_start is None:
                continue

            btc_start = ev.btc_at_start
            if btc_start <= 0:
                continue

            t_remaining = WINDOW_MIN - elapsed_min
            if t_remaining <= 0:
                continue

            log_ratio = math.log(btc_price / btc_start)
            sqrt_t = math.sqrt(t_remaining)

            # Market fair value (sigma_scale=1.0) — realistic price
            sigma_rem_true = sigma_per_min * sqrt_t
            if sigma_rem_true > 1e-12:
                market_fv_up = _norm_cdf(log_ratio / sigma_rem_true)
            else:
                market_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)
            market_fv_up = max(0.02, min(0.98, market_fv_up))

            # Signal fair value (sigma_scale from params) — entry trigger
            sigma_rem_model = sigma_per_min * SIGMA_SCALE * sqrt_t
            if sigma_rem_model > 1e-12:
                signal_fv_up = _norm_cdf(log_ratio / sigma_rem_model)
            else:
                signal_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)
            signal_fv_up = max(0.01, min(0.99, signal_fv_up))

            # Check entry trigger
            signal_dev = signal_fv_up - 0.50
            if abs(signal_dev) < ENTRY_THRESHOLD:
                continue

            # Direction: momentum (follow BTC direction)
            direction = (
                (-1 if signal_dev > 0 else 1)
                if CONTRARIAN
                else (1 if signal_dev > 0 else -1)
            )

            # Entry at MARKET price
            entry_mkt_fv = market_fv_up if direction == 1 else (1.0 - market_fv_up)

            # Price bounds check
            if entry_mkt_fv < MIN_ENTRY_MKT_FV or entry_mkt_fv > MAX_ENTRY_MKT_FV:
                continue

            signals.append(B3Signal(
                condition_id=ev.condition_id,
                token_id_up=ev.token_id_up,
                token_id_down=ev.token_id_down,
                question=ev.question,
                event_start_ts=ev.start_ts,
                event_end_ts=ev.end_ts,
                minutes_elapsed=elapsed_min,
                btc_at_start=btc_start,
                btc_now=btc_price,
                signal_fv_up=signal_fv_up,
                market_fv_up=market_fv_up,
                direction=direction,
                entry_price=entry_mkt_fv,
                edge=abs(signal_dev),
                sigma_per_min=sigma_per_min,
            ))

        return signals

    def mark_traded(self, condition_id: str) -> None:
        """Mark an event as traded (no re-entry for this event)."""
        ev = self._events.get(condition_id)
        if ev:
            ev.traded = True

    def set_btc_at_start(self, condition_id: str, price: float) -> None:
        """Set the BTC price at event start for more accurate FV."""
        ev = self._events.get(condition_id)
        if ev and ev.btc_at_start is None:
            ev.btc_at_start = price

    @property
    def active_event_count(self) -> int:
        """Number of actively tracked events."""
        return len(self._events)
