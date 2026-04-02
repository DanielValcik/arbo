"""B3 15-min Shadow Scanner — collects real orderbook data without trading.

Scans 15-min BTC Up/Down events, fetches CLOB orderbook on every signal,
records what the fill WOULD have been, tracks Chainlink resolution.
No orders placed — pure data collection.

Data stored in `shadow_15m_signals` table for analysis.
"""

from __future__ import annotations

import json
import math
import time

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("b3_15m_shadow")

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
WINDOW_MIN = 15
_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


class B3_15mShadow:
    """Shadow scanner for 15-min BTC Up/Down markets."""

    def __init__(self, binance_ws=None, rtds_feed=None):
        self._binance_ws = binance_ws
        self._rtds_feed = rtds_feed
        self._session: aiohttp.ClientSession | None = None
        self._events: dict[str, dict] = {}  # condition_id → event info
        self._last_fetch_ts: float = 0
        self._signals: list[dict] = []  # collected shadow signals
        self._sigma_prices: list[float] = []  # for volatility
        self._last_vol_ts: float = 0

    async def init(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )
        # Bootstrap sigma from Binance
        try:
            async with self._session.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": "1440"},
            ) as resp:
                if resp.status == 200:
                    klines = await resp.json()
                    self._sigma_prices = [float(k[4]) for k in klines]
                    logger.info("b3_15m_sigma_bootstrapped", n=len(self._sigma_prices))
        except Exception as e:
            logger.warning("b3_15m_sigma_bootstrap_error", error=str(e))

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    def _compute_sigma(self) -> float:
        if len(self._sigma_prices) < 30:
            return 0.0005
        log_returns = []
        for i in range(1, len(self._sigma_prices)):
            if self._sigma_prices[i] > 0 and self._sigma_prices[i-1] > 0:
                log_returns.append(math.log(self._sigma_prices[i] / self._sigma_prices[i-1]))
        if len(log_returns) < 10:
            return 0.0005
        n = len(log_returns)
        mean = sum(log_returns) / n
        var = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        return max(math.sqrt(var), 0.00005)

    async def fetch_events(self) -> None:
        """Fetch active 15-min BTC Up/Down events from Gamma API."""
        now = time.time()
        if now - self._last_fetch_ts < 120:
            return
        self._last_fetch_ts = now

        if not self._session:
            return

        window_start = int(now // 900) * 900
        slugs = [f"btc-updown-15m-{window_start + offset * 900}" for offset in range(-1, 2)]

        new_count = 0
        for slug in slugs:
            try:
                slug_ts = int(slug.split("-")[-1])
            except ValueError:
                continue

            if any(abs(ev.get("start_ts", 0) - slug_ts) < 10 for ev in self._events.values()):
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
                if not nested:
                    continue
                raw = nested[0]
                cid = raw.get("conditionId", "")
                clob_ids = json.loads(raw.get("clobTokenIds", "[]"))
                if len(clob_ids) < 2:
                    continue

                end_date = raw.get("endDate", "")
                if not end_date:
                    continue
                from datetime import datetime
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                end_ts = end_dt.timestamp()
                start_ts = end_ts - WINDOW_MIN * 60

                # Fetch BTC at start from Binance
                btc_start = None
                try:
                    start_ms = int(start_ts * 1000)
                    async with self._session.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": "BTCUSDT", "interval": "1m",
                                "startTime": str(start_ms), "limit": "1"},
                    ) as kresp:
                        if kresp.status == 200:
                            klines = await kresp.json()
                            if klines:
                                btc_start = float(klines[0][4])
                except Exception:
                    pass

                self._events[cid] = {
                    "condition_id": cid,
                    "token_up": clob_ids[0],
                    "token_down": clob_ids[1],
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "btc_at_start": btc_start,
                    "slug": slug,
                    "title": event.get("title", ""),
                    "scanned": False,
                }
                new_count += 1

            except Exception as e:
                logger.debug("b3_15m_fetch_error", slug=slug, error=str(e))

        # Expire old events
        expired = [cid for cid, ev in self._events.items() if now > ev["end_ts"] + 300]
        for cid in expired:
            self._events.pop(cid)

        if new_count > 0:
            logger.info("b3_15m_events", new=new_count, active=len(self._events))

    async def scan(self) -> None:
        """Scan active 15-min events and record shadow signals."""
        now = time.time()

        # Update sigma once per minute
        if now - self._last_vol_ts > 60 and self._binance_ws:
            price = self._binance_ws.get_price("BTCUSDT")
            if price:
                self._sigma_prices.append(price)
                if len(self._sigma_prices) > 2000:
                    self._sigma_prices = self._sigma_prices[-1500:]
                self._last_vol_ts = now

        btc_price = self._binance_ws.get_price("BTCUSDT") if self._binance_ws else None
        if not btc_price:
            return

        chainlink_price = None
        if self._rtds_feed:
            chainlink_price = self._rtds_feed.get_price("btc/usd")

        sigma = self._compute_sigma()

        for cid, ev in list(self._events.items()):
            if not ev.get("btc_at_start"):
                continue

            elapsed_min = (now - ev["start_ts"]) / 60.0

            # Check at integer minutes 4-11 (15-min entry window from sweep)
            entry_minute = int(elapsed_min)
            if entry_minute < 4 or entry_minute > 11:
                continue
            # Already scanned this minute?
            if ev.get(f"scanned_{entry_minute}"):
                continue
            frac = elapsed_min - entry_minute
            if frac >= 0.50:
                continue

            btc_start = ev["btc_at_start"]
            t_remaining = WINDOW_MIN - entry_minute
            btc_move = btc_price - btc_start
            abs_move = abs(btc_move)

            log_ratio = math.log(btc_price / btc_start) if btc_start > 0 else 0
            sqrt_t = math.sqrt(max(t_remaining, 0.01))

            # Market FV
            sigma_rem = sigma * sqrt_t
            market_up = _norm_cdf(log_ratio / sigma_rem) if sigma_rem > 1e-12 else 0.5
            market_up = max(0.02, min(0.98, market_up))

            # Signal FV (sigma_scale=0.526 from 15-min sweep)
            sigma_model = sigma * 0.526 * sqrt_t
            signal_up = _norm_cdf(log_ratio / sigma_model) if sigma_model > 1e-12 else 0.5
            signal_dev = signal_up - 0.50
            edge = abs(signal_dev)

            if edge < 0.02:  # Low threshold — collect ALL data for analysis
                continue

            direction = 1 if signal_dev > 0 else -1
            entry_fv = market_up if direction == 1 else (1 - market_up)

            # Fetch REAL orderbook for this token
            token_id = ev["token_up"] if direction == 1 else ev["token_down"]
            book_data = await self._fetch_orderbook(token_id)

            signal = {
                "timestamp": now,
                "event_slug": ev["slug"],
                "condition_id": cid,
                "direction": "up" if direction == 1 else "down",
                "entry_minute": entry_minute,
                "time_remaining": t_remaining,
                "btc_at_start": btc_start,
                "btc_now": btc_price,
                "btc_chainlink": chainlink_price,
                "btc_move": round(btc_move, 2),
                "btc_abs_move": round(abs_move, 2),
                "sigma": sigma,
                "model_fv": round(entry_fv, 4),
                "signal_fv": round(signal_up if direction == 1 else (1 - signal_up), 4),
                "edge": round(edge, 4),
                "market_fv_up": round(market_up, 4),
                # Orderbook data (what fill would be)
                "book_best_bid": book_data.get("best_bid", 0),
                "book_best_ask": book_data.get("best_ask", 0),
                "book_spread": book_data.get("spread", 0),
                "book_bid_depth_usd": book_data.get("bid_depth", 0),
                "book_ask_depth_usd": book_data.get("ask_depth", 0),
                "would_fill_at": book_data.get("best_bid", 0),  # PostOnly = buy at bid
                "market_gap": round(abs(entry_fv - book_data.get("best_bid", entry_fv)), 4),
                # Resolution (filled later)
                "event_end_ts": ev["end_ts"],
                "event_start_ts": ev["start_ts"],
                "resolution": None,  # filled after event ends
            }

            self._signals.append(signal)
            ev[f"scanned_{entry_minute}"] = True

            logger.info(
                "b3_15m_shadow_signal",
                direction=signal["direction"],
                edge=f"{edge:.3f}",
                btc_move=f"${abs_move:.0f}",
                model_fv=f"{entry_fv:.3f}",
                would_fill=f"${signal['would_fill_at']:.3f}",
                spread=f"${signal['book_spread']:.3f}",
                gap=f"{signal['market_gap']:.3f}",
                time_left=f"{t_remaining}min",
            )

    async def check_resolutions(self) -> None:
        """Check resolutions for completed events."""
        now = time.time()
        for sig in self._signals:
            if sig["resolution"] is not None:
                continue
            if now < sig["event_end_ts"] + 30:
                continue

            # Fetch resolution from Gamma API
            start_ts = int(sig["event_start_ts"])
            slug = f"btc-updown-15m-{start_ts}"
            try:
                async with self._session.get(
                    f"{GAMMA_URL}/events", params={"slug": slug},
                ) as resp:
                    if resp.status != 200:
                        continue
                    events = await resp.json()
                if not events:
                    continue
                m = events[0].get("markets", [{}])[0]
                outcomes = json.loads(m.get("outcomes", "[]"))
                prices = json.loads(m.get("outcomePrices", "[]"))
                up_won = None
                for i, o in enumerate(outcomes):
                    if o.lower() == "up" and i < len(prices):
                        if prices[i] == "1":
                            up_won = True
                            break
                        for j, o2 in enumerate(outcomes):
                            if o2.lower() == "down" and j < len(prices) and prices[j] == "1":
                                up_won = False
                                break
                        break

                if up_won is None:
                    continue

                direction = sig["direction"]
                won = (direction == "up" and up_won) or (direction == "down" and not up_won)
                fill = sig["would_fill_at"]
                pnl = (1.0 - fill) if won else -fill

                sig["resolution"] = "win" if won else "loss"
                sig["would_pnl_per_share"] = round(pnl, 4)

                logger.info(
                    "b3_15m_shadow_resolved",
                    direction=direction,
                    result=sig["resolution"],
                    fill=f"${fill:.3f}",
                    pnl=f"${pnl:+.3f}/share",
                    btc_move=f"${sig['btc_abs_move']:.0f}",
                )

            except Exception as e:
                logger.debug("b3_15m_resolution_error", error=str(e))

    async def _fetch_orderbook(self, token_id: str) -> dict:
        """Fetch orderbook for a token."""
        try:
            async with self._session.get(
                f"{CLOB_URL}/book", params={"token_id": token_id},
            ) as resp:
                if resp.status != 200:
                    return {}
                book = await resp.json()

            asks = book.get("asks", [])
            bids = book.get("bids", [])

            best_ask = min(float(a["price"]) for a in asks) if asks else 0
            best_bid = max(float(b["price"]) for b in bids) if bids else 0
            spread = round(best_ask - best_bid, 4) if best_ask and best_bid else 0
            bid_depth = sum(float(b["size"]) * float(b["price"]) for b in bids[:10]) if bids else 0
            ask_depth = sum(float(a["size"]) * float(a["price"]) for a in asks[:10]) if asks else 0

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "bid_depth": round(bid_depth, 0),
                "ask_depth": round(ask_depth, 0),
            }
        except Exception:
            return {}

    def get_stats(self) -> dict:
        """Get summary stats for dashboard/logging."""
        resolved = [s for s in self._signals if s["resolution"]]
        if not resolved:
            return {"signals": len(self._signals), "resolved": 0}

        wins = sum(1 for s in resolved if s["resolution"] == "win")
        fills = [s["would_fill_at"] for s in resolved if s["would_fill_at"] > 0]
        gaps = [s["market_gap"] for s in resolved]
        moves = [s["btc_abs_move"] for s in resolved]

        return {
            "signals": len(self._signals),
            "resolved": len(resolved),
            "wins": wins,
            "wr": round(wins / len(resolved) * 100, 1),
            "avg_fill": round(sum(fills) / len(fills), 3) if fills else 0,
            "avg_gap": round(sum(gaps) / len(gaps), 3) if gaps else 0,
            "avg_btc_move": round(sum(moves) / len(moves), 0) if moves else 0,
            "avg_spread": round(sum(s["book_spread"] for s in resolved) / len(resolved), 3),
        }

    def get_all_signals(self) -> list[dict]:
        """Get all collected signals for export/analysis."""
        return self._signals
