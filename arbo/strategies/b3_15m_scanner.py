"""B3_15M Scanner — Discover and scan BTC 15-min Up/Down markets.

Fork of b3_scanner.py for the 15-minute variant. Differences:
- Window boundary: 900s (vs 300s)
- Slug format: btc-updown-15m-{ts} (vs btc-updown-5m-{ts})
- Entry minutes 4-11 (vs 1-3) via MIN/MAX_ENTRY_MIN from 15m quality gate
- Quality gate constants sourced from b3_15m_quality_gate

Shared architecture: Chainlink RTDS buffer, Binance kline fallback,
Gamma API event discovery, momentum direction logic.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp

from arbo.strategies.b3_15m_quality_gate import (
    CONTRARIAN,
    ENTRY_THRESHOLD,
    MAX_ENTRY_MIN,
    MIN_ENTRY_MIN,
    MIN_ENTRY_MKT_FV,
    SIGMA_FLOOR,
    SIGMA_SCALE,
    WINDOW_MIN,
)
from arbo.utils.logger import get_logger

logger = get_logger("b3_15m_scanner")

GAMMA_URL = "https://gamma-api.polymarket.com"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

WINDOW_SECONDS = WINDOW_MIN * 60  # 900
SLUG_PREFIX = "btc-updown-15m"

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass
class B315MSignal:
    """A B3_15M scalping signal for a 15-min BTC Up/Down event."""

    condition_id: str
    token_id_up: str
    token_id_down: str
    question: str

    event_start_ts: float
    event_end_ts: float
    minutes_elapsed: float

    btc_at_start: float
    btc_now: float

    signal_fv_up: float
    market_fv_up: float

    direction: int
    entry_price: float
    edge: float

    sigma_per_min: float


@dataclass
class B315MEvent:
    """Tracked 15-min BTC Up/Down event."""

    condition_id: str
    token_id_up: str
    token_id_down: str
    question: str
    start_ts: float
    end_ts: float
    btc_at_start: float | None = None
    traded: bool = False
    entry_count: int = 0  # Track entries for re-entry cooldown


class B315MScanner:
    """Discovers and tracks BTC 15-min Up/Down markets."""

    def __init__(self, rtds_feed=None) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._events: dict[str, B315MEvent] = {}
        self._last_fetch_ts: float = 0
        self._fetch_interval_s: float = 180  # Refetch every 3 min (events every 15 min)
        self._rtds_feed = rtds_feed
        # 15-min buffer: keep 20 min of prices (covers entire event + margin)
        self._cl_price_buffer: list[tuple[float, float]] = []
        self._cl_buffer_max = 1200  # 20 min
        # Track entry-minute firing: (cond_id, minute) → True
        self._minute_fired: dict[tuple[str, int], bool] = {}

    async def init(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )

    def record_cl_price(self) -> None:
        """Record current Chainlink price with timestamp. Call every 5-15s."""
        if not self._rtds_feed:
            return
        price = self._rtds_feed.get_price("btc/usd")
        if price and price > 1000:
            now = time.time()
            self._cl_price_buffer.append((now, price))
            cutoff = now - self._cl_buffer_max
            self._cl_price_buffer = [(t, p) for t, p in self._cl_price_buffer if t >= cutoff]

    def _lookup_cl_price(self, target_ts: float) -> float | None:
        """Find Chainlink price closest to target timestamp from buffer."""
        if not self._cl_price_buffer:
            return None
        best = None
        best_diff = float('inf')
        for ts, price in self._cl_price_buffer:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = price
        if best_diff <= 15:  # Within 15s (vs 10s on 5-min — 15-min less time-sensitive)
            return best
        return None

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_events(self) -> None:
        """Fetch active BTC 15-min Up/Down events from Gamma API."""
        now = time.time()
        if now - self._last_fetch_ts < self._fetch_interval_s:
            return
        self._last_fetch_ts = now

        if not self._session:
            return

        window_start = int(now // WINDOW_SECONDS) * WINDOW_SECONDS
        slugs_to_check = [
            f"{SLUG_PREFIX}-{window_start + offset * WINDOW_SECONDS}"
            for offset in range(-1, 3)
        ]

        new_count = 0
        for slug in slugs_to_check:
            try:
                slug_ts = int(slug.split("-")[-1])
            except ValueError:
                continue
            if any(abs(ev.start_ts - slug_ts) < 10 for ev in self._events.values()):
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

                clob_ids = raw_market.get("clobTokenIds", [])
                if isinstance(clob_ids, str):
                    try:
                        clob_ids = json.loads(clob_ids)
                    except Exception:
                        clob_ids = []
                if len(clob_ids) < 2:
                    continue

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

                btc_start = await self._fetch_btc_at_start(start_ts)

                self._events[cid] = B315MEvent(
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
                        "b3_15m_event_btc_start",
                        slug=slug,
                        btc_at_start=f"${btc_start:,.2f}",
                    )

            except Exception as e:
                logger.warning("b3_15m_slug_fetch_error", slug=slug, error=str(e))

        # Retry btc_at_start for events where kline wasn't ready
        for ev in self._events.values():
            if ev.btc_at_start is None and now >= ev.start_ts + 60:
                btc_start = await self._fetch_btc_at_start(ev.start_ts)
                if btc_start:
                    ev.btc_at_start = btc_start
                    logger.info(
                        "b3_15m_btc_start_retry_ok",
                        question=ev.question[:50],
                        btc_at_start=f"${btc_start:,.2f}",
                    )

        # Clean expired events (ended > 20 min ago)
        expired = [
            cid for cid, ev in self._events.items()
            if now - ev.end_ts > 1200
        ]
        for cid in expired:
            del self._events[cid]
            # Clean minute_fired entries for expired events
            self._minute_fired = {
                k: v for k, v in self._minute_fired.items() if k[0] != cid
            }

        if new_count > 0 or expired:
            logger.info(
                "b3_15m_events_updated",
                new=new_count,
                expired=len(expired),
                active=len(self._events),
            )

    async def _fetch_btc_at_start(self, event_start_ts: float) -> float | None:
        """Fetch BTC price at event start — Chainlink buffer preferred, Binance fallback."""
        now = time.time()
        age = now - event_start_ts

        buffer_price = self._lookup_cl_price(event_start_ts)
        if buffer_price:
            logger.info(
                "b3_15m_btc_start_cl_buffer",
                price=f"${buffer_price:,.2f}",
                age_s=f"{age:.0f}",
            )
            return buffer_price

        if 0 <= age < 30 and self._rtds_feed:
            cl_price = self._rtds_feed.get_price("btc/usd")
            if cl_price and cl_price > 1000:
                logger.info(
                    "b3_15m_btc_start_chainlink",
                    price=f"${cl_price:,.2f}",
                    age_s=f"{age:.0f}",
                )
                return cl_price

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
                    price = float(klines[0][4])
                    logger.info(
                        "b3_15m_btc_start_binance_fallback",
                        price=f"${price:,.2f}",
                        age_s=f"{age:.0f}",
                    )
                    return price
        except Exception as e:
            logger.warning("b3_15m_binance_kline_error", error=str(e))
        return None

    async def fetch_resolution(self, event_start_ts: float) -> bool | None:
        """Fetch market resolution from Gamma API (Chainlink oracle)."""
        if not self._session:
            return None
        slug = f"{SLUG_PREFIX}-{int(event_start_ts)}"
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
                        return True
                    for j, o2 in enumerate(outcomes):
                        if o2.lower() == "down" and j < len(prices) and prices[j] == "1":
                            return False
                    return None
        except Exception as e:
            logger.warning("b3_15m_resolution_fetch_error", slug=slug, error=str(e))
        return None

    def scan(
        self,
        btc_price: float,
        sigma_per_min: float,
        *,
        params: dict | None = None,
        variant_id: str = "default",
    ) -> list[B315MSignal]:
        """Scan active events for entry signals (15-min variant, minutes 4-11).

        Args:
            btc_price: current BTC mid price (Binance).
            sigma_per_min: realized per-minute volatility.
            params: optional override dict — keys mirror module-level constants
                (SIGMA_FLOOR, SIGMA_SCALE, ENTRY_THRESHOLD, MIN_ENTRY_MIN,
                MAX_ENTRY_MIN, CONTRARIAN, MIN_ENTRY_MKT_FV). If None, falls
                back to module constants (champion's params, backward compat).
            variant_id: identifier so multiple variants can scan the same event
                without colliding in `_minute_fired` deduplication. Champion
                uses "default"; ShadowOrchestrator passes per-variant IDs.

        Returns the list of B315MSignal objects this variant would have fired.
        Mutates `_minute_fired[(cond_id, minute, variant_id)] = True` for each
        emitted signal so we don't re-fire the same minute for the same variant.
        """
        # Resolve per-variant params (fall back to module constants for champion)
        p = params or {}
        sigma_floor_eff = p.get("SIGMA_FLOOR", SIGMA_FLOOR)
        sigma_scale_eff = p.get("SIGMA_SCALE", SIGMA_SCALE)
        entry_threshold_eff = p.get("ENTRY_THRESHOLD", ENTRY_THRESHOLD)
        min_entry_min_eff = p.get("MIN_ENTRY_MIN", MIN_ENTRY_MIN)
        max_entry_min_eff = p.get("MAX_ENTRY_MIN", MAX_ENTRY_MIN)
        contrarian_eff = p.get("CONTRARIAN", CONTRARIAN)
        min_entry_mkt_fv_eff = p.get("MIN_ENTRY_MKT_FV", MIN_ENTRY_MKT_FV)
        # WINDOW_MIN is a strategy-level constant, not variant-overridable

        now = time.time()
        signals: list[B315MSignal] = []
        sigma_per_min = max(sigma_per_min, sigma_floor_eff)

        for ev in self._events.values():
            if ev.traded:
                continue

            elapsed_s = now - ev.start_ts
            elapsed_min = elapsed_s / 60.0

            entry_minute = int(elapsed_min)
            if entry_minute < min_entry_min_eff or entry_minute > max_entry_min_eff:
                continue
            # Fire once per integer minute per event PER VARIANT
            # (variant_id in key prevents collision when champion + challengers
            # scan the same event in parallel)
            key = (ev.condition_id, entry_minute, variant_id)
            if self._minute_fired.get(key):
                continue
            frac = elapsed_min - entry_minute
            if frac >= 0.50:
                continue

            if ev.btc_at_start is None:
                continue
            btc_start = ev.btc_at_start
            if btc_start <= 0:
                continue

            t_remaining = WINDOW_MIN - entry_minute
            if t_remaining <= 0:
                continue

            log_ratio = math.log(btc_price / btc_start)
            sqrt_t = math.sqrt(t_remaining)

            sigma_rem_true = sigma_per_min * sqrt_t
            if sigma_rem_true > 1e-12:
                market_fv_up = _norm_cdf(log_ratio / sigma_rem_true)
            else:
                market_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)
            market_fv_up = max(0.02, min(0.98, market_fv_up))

            sigma_rem_model = sigma_per_min * sigma_scale_eff * sqrt_t
            if sigma_rem_model > 1e-12:
                signal_fv_up = _norm_cdf(log_ratio / sigma_rem_model)
            else:
                signal_fv_up = 1.0 if log_ratio > 0 else (0.0 if log_ratio < 0 else 0.5)
            signal_fv_up = max(0.01, min(0.99, signal_fv_up))

            signal_dev = signal_fv_up - 0.50
            if abs(signal_dev) < entry_threshold_eff:
                continue

            direction = (
                (-1 if signal_dev > 0 else 1)
                if contrarian_eff
                else (1 if signal_dev > 0 else -1)
            )

            entry_mkt_fv = market_fv_up if direction == 1 else (1.0 - market_fv_up)

            if entry_mkt_fv < min_entry_mkt_fv_eff:
                continue

            self._minute_fired[key] = True

            signals.append(B315MSignal(
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
        ev = self._events.get(condition_id)
        if ev:
            ev.entry_count += 1
            # Allow up to REENTRY_COOLDOWN entries per event (from quality gate)
            from arbo.strategies.b3_15m_quality_gate import REENTRY_COOLDOWN
            if ev.entry_count >= REENTRY_COOLDOWN:
                ev.traded = True

    def set_btc_at_start(self, condition_id: str, price: float) -> None:
        ev = self._events.get(condition_id)
        if ev and ev.btc_at_start is None:
            ev.btc_at_start = price

    @property
    def active_event_count(self) -> int:
        return len(self._events)
