"""Live order executor v6 — Maker-first strategy.

ENTRY (BUY): PostOnly at BUY price (maker bid)
  - Same price as paper model → no spread cost
  - 0% fee + 20% rebate
  - Wait up to 3 min for fill, cancel if not filled

EXIT (SELL): PostOnly at SELL price (maker ask) first
  - Wait 2 min for fill
  - If not filled → fallback to taker at BUY price (instant but pays spread)

SPREAD FILTER: Skip if spread > MAX_SPREAD_PCT

Why maker > taker for weather:
  Paper buys at BUY price (low), sells at SELL price (high) → earns spread
  Taker buys at SELL price (high), sells at BUY price (low) → pays spread
  Maker buys at BUY price (low), sells at SELL price (high) → matches paper!
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("live_executor")

UTC = timezone.utc
MIN_ORDER_USD = 1.5
# Polymarket CLOB enforces a per-order minimum of 5 shares regardless of
# price or tick size. Observed error: "Size (4) lower than the minimum: 5".
# Orders below this are rejected BEFORE any state change on-chain, but they
# still burn a signal (wasted live fill attempt + strategy debounce side
# effects). Enforce client-side to fail fast and cleanly.
MIN_ORDER_SHARES = 5
MAKER_FILL_TIMEOUT_ENTRY_S = 30  # 30s for entry (keep poll loop fast)
MAKER_FILL_TIMEOUT_EXIT_S = 30  # 30s for exit (then fallback to taker)
FILL_POLL_INTERVAL_S = 5  # Check fill every 5s
MAX_SPREAD_PCT = 0.20  # Skip if spread > 20% of mid
DATA_API = "https://data-api.polymarket.com"

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


@dataclass
class LiveFill:
    token_id: str
    side: str
    price: Decimal
    size: Decimal
    order_id: str = ""
    status: str = "pending"
    fill_price: Decimal | None = None
    shares_requested: int = 0
    shares_filled: int = 0
    usdc_spent: float = 0.0
    order_type: str = ""  # "maker" or "taker"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: int = 0
    error: str | None = None

    def to_monitoring_dict(self) -> dict:
        return {
            "live_order_id": self.order_id,
            "live_submitted_price": float(self.price),
            "live_fill_price": float(self.fill_price) if self.fill_price else None,
            "live_shares_requested": self.shares_requested,
            "live_shares_filled": self.shares_filled,
            "live_usdc_spent": self.usdc_spent,
            "live_order_type": self.order_type,
            "live_latency_ms": self.latency_ms,
            "live_status": self.status,
            "live_error": self.error,
            "live_timestamp": self.timestamp.isoformat(),
            "live_gas_usd": 0.0,
        }


class LiveExecutor:
    def __init__(self, poly_client: Any) -> None:
        self._poly_client = poly_client
        self._clob: Any = None
        self._fills: list[LiveFill] = []
        self._shares_owned: dict[str, int] = {}
        self._funder: str = ""

    async def _ensure_clob(self) -> Any:
        if self._clob is not None:
            return self._clob
        import os
        from py_clob_client.client import ClobClient as _ClobClient
        self._funder = os.getenv("POLY_FUNDER_ADDRESS", "")
        self._clob = _ClobClient(
            host="https://clob.polymarket.com", chain_id=137,
            key=os.getenv("POLY_PRIVATE_KEY", ""), signature_type=2,
            funder=self._funder or None,
        )
        creds = self._clob.create_or_derive_api_creds()
        self._clob.set_api_creds(creds)
        await self._sync_positions()
        logger.info("live_executor_ready", positions=len(self._shares_owned))
        return self._clob

    async def _sync_positions(self) -> None:
        if not self._funder:
            return
        try:
            url = f"{DATA_API}/positions?user={self._funder}"
            req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: json.loads(
                urllib.request.urlopen(req, timeout=10, context=_SSL_CTX).read()
            ))
            self._shares_owned.clear()
            for pos in data:
                token = pos.get("asset", "")
                size = int(float(pos.get("size", 0)))
                if token and size > 0:
                    self._shares_owned[token] = size
        except Exception as e:
            logger.warning("positions_sync_failed", error=str(e))

    async def get_balance(self) -> float:
        """Get available USDC balance from CLOB API. Returns 0 on failure."""
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
            clob = await self._ensure_clob()
            loop = asyncio.get_event_loop()
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await loop.run_in_executor(
                None, lambda: clob.get_balance_allowance(params)
            )
            if isinstance(result, dict):
                raw = int(result.get("balance", 0))
                return raw / 1_000_000  # USDC has 6 decimals
        except Exception as e:
            logger.warning("balance_fetch_failed", error=str(e))
        return 0.0

    async def _get_prices(self, token_id: str) -> tuple[float | None, float | None]:
        """Get BUY (bid) and SELL (ask) prices. Returns (buy_price, sell_price)."""
        try:
            buy_p = float(await self._poly_client.get_price(token_id, "BUY"))
            sell_p = float(await self._poly_client.get_price(token_id, "SELL"))
            return buy_p, sell_p
        except Exception as e:
            logger.warning("prices_fetch_failed", error=str(e))
            return None, None

    async def buy(
        self, token_id: str, price: float, size_usdc: float,
        neg_risk: bool = True, tick_size: str = "0.01",
        maker_timeout_s: int | None = None,
        max_price: float | None = None,
        fallback_to_taker: bool = False,
    ) -> LiveFill:
        """MAKER BUY at BUY price (same as paper). 0% fee + rebate.

        If `max_price` is set, fail BEFORE posting the order when the
        current CLOB BUY price (the level we'd post the maker at) exceeds
        the cap. This protects against CLOB drift between the strategy's
        scan and our execution — we don't want to fill ITM tokens that
        the strategy already considers too expensive.

        If `fallback_to_taker` is True and the maker leg returns no fill
        within the timeout, cancels the maker and submits a taker BUY at
        `sell_price` (ask level, what paper's mirror entry uses when
        PAPER_MATCH_LIVE is on). This gives paper-live parity for thin
        markets where maker never gets crossed — at the cost of paying
        the bid/ask spread plus taker fees. Default False preserves the
        existing B3 maker-only behavior.
        """
        buy_price, sell_price = await self._get_prices(token_id)
        if buy_price is None or sell_price is None:
            return self._fail(token_id, "BUY", price, size_usdc, "No prices")

        # CLOB drift safety: reject if current bid would exceed strategy cap.
        # Done BEFORE posting any order — no money at risk.
        if max_price is not None and buy_price > max_price:
            logger.info(
                "live_buy_clob_drift_skip",
                buy_price=buy_price,
                max_price=max_price,
                token=token_id[:20],
            )
            return self._fail(
                token_id, "BUY", buy_price, size_usdc,
                f"CLOB bid {buy_price:.3f} > max_price {max_price:.3f}",
            )

        # Spread check
        mid = (buy_price + sell_price) / 2
        spread_pct = (sell_price - buy_price) / mid if mid > 0 else 1.0
        if spread_pct > MAX_SPREAD_PCT:
            return self._fail(token_id, "BUY", buy_price, size_usdc,
                              f"Spread {spread_pct:.0%} > {MAX_SPREAD_PCT:.0%}")

        # Maker: buy at BUY price (bid level — same as paper)
        maker_price = buy_price
        shares = int(size_usdc / maker_price) if maker_price > 0 else 0
        if shares < MIN_ORDER_SHARES:
            return self._fail(
                token_id, "BUY", maker_price, size_usdc,
                f"shares {shares} < {MIN_ORDER_SHARES} (Polymarket minimum)",
            )
        if shares * maker_price < MIN_ORDER_USD:
            return self._fail(token_id, "BUY", maker_price, size_usdc, "Below minimum")

        logger.info("live_buy_maker", price=maker_price, spread_pct=round(spread_pct, 3),
                     shares=shares, token=token_id[:20], fallback=fallback_to_taker)

        fill = LiveFill(
            token_id=token_id, side="BUY", price=Decimal(str(maker_price)),
            size=Decimal(str(size_usdc)), shares_requested=shares, order_type="maker",
        )

        t0 = time.monotonic()
        try:
            clob = await self._ensure_clob()
            order_id, immediate = await self._post_order(
                clob, token_id, maker_price, shares, "BUY", neg_risk, tick_size,
            )
            fill.order_id = order_id

            # Wait for maker fill
            timeout = maker_timeout_s if maker_timeout_s is not None else MAKER_FILL_TIMEOUT_ENTRY_S
            filled = await self._poll_fill(clob, order_id, immediate,
                                            timeout_s=timeout)

            # Always cancel maker remainder before any fallback
            await self._cancel(order_id, clob)

            if filled > 0:
                fill.status = "filled" if filled >= shares else "partial"
                fill.shares_filled = filled
                fill.fill_price = Decimal(str(maker_price))
                fill.usdc_spent = filled * maker_price
                self._shares_owned[token_id] = self._shares_owned.get(token_id, 0) + filled

            # Taker fallback for the unfilled remainder. Pays sell_price (ask)
            # — same level paper uses under PAPER_MATCH_LIVE, so paper/live
            # prices stay in parity. Re-check shares-floor on the remainder.
            if fallback_to_taker and filled < shares:
                remaining = shares - filled
                if remaining >= MIN_ORDER_SHARES:
                    taker_price = sell_price
                    # CLOB drift recheck — don't eat through a cap
                    if max_price is not None and taker_price > max_price:
                        logger.info(
                            "live_buy_taker_cap_skip", taker_price=taker_price,
                            max_price=max_price, token=token_id[:20],
                        )
                    else:
                        logger.info("live_buy_taker_fallback",
                                    price=taker_price, shares=remaining,
                                    token=token_id[:20])
                        try:
                            oid2, imm2 = await self._post_order(
                                clob, token_id, taker_price, remaining,
                                "BUY", neg_risk, tick_size,
                            )
                            filled2 = await self._poll_fill(clob, oid2, imm2, timeout_s=10)
                            await self._cancel(oid2, clob)
                            if filled2 > 0:
                                fill.shares_filled += filled2
                                # Blended avg fill price when partial maker + taker
                                total_cost = (filled * maker_price) + (filled2 * taker_price)
                                fill.usdc_spent = total_cost
                                fill.fill_price = Decimal(str(
                                    total_cost / fill.shares_filled if fill.shares_filled else taker_price
                                ))
                                fill.order_type = "maker+taker" if filled > 0 else "taker"
                                self._shares_owned[token_id] = (
                                    self._shares_owned.get(token_id, 0) + filled2
                                )
                                fill.status = "filled" if fill.shares_filled >= shares else "partial"
                        except Exception as e2:
                            logger.warning(
                                "live_buy_taker_fallback_error",
                                token=token_id[:20], error=str(e2),
                            )

            if fill.shares_filled == 0:
                fill.status = "failed"
                fill.error = "No fill (maker timeout, no taker fallback or taker also failed)"

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info("live_buy_done", token=token_id[:20], price=maker_price,
                        requested=shares, filled=fill.shares_filled, status=fill.status,
                        type=fill.order_type, latency=fill.latency_ms)

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_buy_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def sell(
        self, token_id: str, price: float, shares: float | None = None,
        neg_risk: bool = True, tick_size: str = "0.01",
        maker_timeout_s: int | None = None,
        skip_sync: bool = False,
    ) -> LiveFill:
        """MAKER SELL at SELL price first, fallback to TAKER at BUY price."""
        if not skip_sync:
            await self._sync_positions()
        actual = self._shares_owned.get(token_id, 0)
        if actual <= 0:
            return self._fail(token_id, "SELL", price, 0, "No shares owned")
        if actual < MIN_ORDER_SHARES:
            # Sub-minimum positions (shouldn't happen after buy-side fix, but
            # legacy positions exist). Skip sell — Polymarket would reject.
            # Shares auto-resolve at event end to $1 or $0.
            fill = LiveFill(
                token_id=token_id, side="SELL", price=Decimal("0"),
                size=Decimal(str(actual)), shares_requested=actual,
                status="skipped", order_type="below_min",
                error=f"shares {actual} < {MIN_ORDER_SHARES}, waiting for resolution",
            )
            fill.latency_ms = 0
            logger.info("live_sell_skip_below_min",
                        token=token_id[:20], shares=actual)
            self._fills.append(fill)
            return fill

        buy_price, sell_price = await self._get_prices(token_id)
        if buy_price is None or sell_price is None:
            return self._fail(token_id, "SELL", price, 0, "No prices")

        # Clamp prices to CLOB limits (0.01-0.99)
        sell_price = max(0.01, min(sell_price, 0.99))
        buy_price = max(0.01, min(buy_price, 0.99))

        # If token is near-worthless, skip sell — let it resolve at $0
        if sell_price <= 0.02 and buy_price <= 0.01:
            logger.info("live_sell_skip_worthless", token=token_id[:20],
                        sell_price=sell_price, buy_price=buy_price)
            fill = LiveFill(
                token_id=token_id, side="SELL", price=Decimal("0"),
                size=Decimal(str(actual)), shares_requested=actual,
                status="skipped", order_type="resolution",
                error="Token near-zero, will auto-resolve",
            )
            fill.latency_ms = 0
            self._fills.append(fill)
            return fill

        sell_shares = actual

        logger.info("live_sell_start", maker_price=sell_price, taker_price=buy_price,
                     shares=sell_shares, token=token_id[:20])

        fill = LiveFill(
            token_id=token_id, side="SELL", price=Decimal(str(sell_price)),
            size=Decimal(str(sell_shares)), shares_requested=sell_shares,
        )

        t0 = time.monotonic()
        try:
            clob = await self._ensure_clob()

            # Phase 1: Maker sell at SELL price (ask level — same as paper exit)
            fill.order_type = "maker"
            order_id, immediate = await self._post_order(
                clob, token_id, sell_price, sell_shares, "SELL", neg_risk, tick_size,
            )
            fill.order_id = order_id

            timeout = maker_timeout_s if maker_timeout_s is not None else MAKER_FILL_TIMEOUT_EXIT_S
            filled = await self._poll_fill(clob, order_id, immediate,
                                            timeout_s=timeout)
            await self._cancel(order_id, clob)

            if filled >= sell_shares:
                # Full maker fill — best case
                fill.status = "filled"
                fill.shares_filled = filled
                fill.fill_price = Decimal(str(sell_price))
                fill.usdc_spent = filled * sell_price
                self._shares_owned.pop(token_id, None)
            elif filled > 0:
                # Partial maker fill
                remaining = sell_shares - filled
                fill.shares_filled = filled
                fill.fill_price = Decimal(str(sell_price))
                fill.usdc_spent = filled * sell_price
                self._shares_owned[token_id] = remaining

                # Phase 2: Taker sell remainder at BUY price
                logger.info("live_sell_taker_fallback", remaining=remaining, token=token_id[:20])
                fill.order_type = "maker+taker"
                order_id2, imm2 = await self._post_order(
                    clob, token_id, buy_price, remaining, "SELL", neg_risk, tick_size,
                )
                filled2 = await self._poll_fill(clob, order_id2, imm2, timeout_s=10)
                await self._cancel(order_id2, clob)

                fill.shares_filled += filled2
                fill.usdc_spent += filled2 * buy_price
                new_remaining = remaining - filled2
                if new_remaining <= 0:
                    self._shares_owned.pop(token_id, None)
                else:
                    self._shares_owned[token_id] = new_remaining

                fill.status = "filled" if new_remaining <= 0 else "partial"
            else:
                # No maker fill — taker fallback
                fill.order_type = "taker"
                logger.info("live_sell_taker_only", shares=sell_shares, token=token_id[:20])
                order_id2, imm2 = await self._post_order(
                    clob, token_id, buy_price, sell_shares, "SELL", neg_risk, tick_size,
                )
                filled2 = await self._poll_fill(clob, order_id2, imm2, timeout_s=10)
                await self._cancel(order_id2, clob)

                fill.shares_filled = filled2
                fill.fill_price = Decimal(str(buy_price))
                fill.usdc_spent = filled2 * buy_price
                remaining = sell_shares - filled2
                if remaining <= 0:
                    self._shares_owned.pop(token_id, None)
                else:
                    self._shares_owned[token_id] = remaining
                fill.status = "filled" if remaining <= 0 else "partial"

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            avg_price = fill.usdc_spent / fill.shares_filled if fill.shares_filled > 0 else 0
            logger.info("live_sell_done", token=token_id[:20],
                        requested=sell_shares, filled=fill.shares_filled,
                        avg_price=round(avg_price, 4), type=fill.order_type,
                        status=fill.status, latency=fill.latency_ms)

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_sell_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def _post_order(
        self, clob: Any, token_id: str, price: float, shares: int,
        side: str, neg_risk: bool, tick_size: str,
    ) -> tuple[str, int]:
        """Post GTC order. Returns (order_id, immediate_fill_count)."""
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY as _BUY, SELL as _SELL

        loop = asyncio.get_event_loop()
        args = OrderArgs(
            token_id=token_id, price=price, size=shares,
            side=_BUY if side == "BUY" else _SELL,
        )
        opts = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

        def _post():
            signed = clob.create_order(args, opts)
            return clob.post_order(signed, OrderType.GTC)

        result = await loop.run_in_executor(None, _post)
        order_id = result.get("orderID", result.get("id", ""))
        immediate = int(float(result.get("takingAmount", "0") or "0"))
        return order_id, immediate

    async def _poll_fill(
        self, clob: Any, order_id: str, immediate: int, timeout_s: int,
    ) -> int:
        """Poll order for fills. Returns total shares filled."""
        if not order_id:
            return immediate

        loop = asyncio.get_event_loop()
        deadline = time.monotonic() + timeout_s
        best = immediate

        while time.monotonic() < deadline:
            await asyncio.sleep(FILL_POLL_INTERVAL_S)
            try:
                oid = order_id
                info = await loop.run_in_executor(None, lambda: clob.get_order(oid))
                if isinstance(info, dict):
                    sm = int(float(info.get("size_matched", "0") or "0"))
                    best = max(best, sm)
                    status = info.get("status", "")
                    if status in ("MATCHED", "matched", "filled"):
                        return best  # Fully filled
                    if status in ("CANCELLED", "cancelled", "expired"):
                        return best
            except Exception:
                pass

        return best

    async def _cancel(self, order_id: str, clob: Any) -> None:
        if not order_id:
            return
        try:
            oid = order_id
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: clob.cancel(oid))
        except Exception:
            pass

    def _fail(self, token_id: str, side: str, price: float, size: float, error: str) -> LiveFill:
        logger.warning("live_order_skip", side=side, error=error, token=token_id[:20])
        return LiveFill(
            token_id=token_id, side=side, price=Decimal(str(price)),
            size=Decimal(str(size)), status="failed", error=error,
        )

    async def _get_taker_price(self, token_id: str, side: str) -> float | None:
        """Legacy — used by ExitManager."""
        prices = await self._get_prices(token_id)
        if side == "SELL":
            return prices[0]  # BUY price = taker sell
        return prices[1]  # SELL price = taker buy

    @property
    def shares_owned(self) -> dict[str, int]:
        return dict(self._shares_owned)

    @property
    def recent_fills(self) -> list[LiveFill]:
        return self._fills[-100:]
