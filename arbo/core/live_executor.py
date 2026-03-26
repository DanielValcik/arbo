"""Live order executor for Strategy C2 — v5 (production-ready).

Handles Polymarket CLOB realities:
1. Posts GTC at taker price → checks immediate fill → cancels remainder
2. Tracks actual shares via Polymarket Data API (survives restarts)
3. Sells only confirmed owned shares
4. All errors handled gracefully (no crashes)

NegRisk pricing (CONFIRMED 2026-03-26):
- BUY order: price = /price?side=SELL (taker ask) → instant match
- SELL order: price = /price?side=BUY (taker bid) → instant match

Gas: $0 (Polymarket gasless relay).
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
MIN_ORDER_USD = 1.5  # Polymarket min ~$1, add buffer
FILL_CHECK_DELAY_S = 3  # Wait before checking fill status
DATA_API = "https://data-api.polymarket.com"

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


@dataclass
class LiveFill:
    """Result of a live order execution."""

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
            "live_latency_ms": self.latency_ms,
            "live_status": self.status,
            "live_error": self.error,
            "live_timestamp": self.timestamp.isoformat(),
            "live_gas_usd": 0.0,
        }


class LiveExecutor:
    """Production-ready Polymarket CLOB executor."""

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
            host="https://clob.polymarket.com",
            chain_id=137,
            key=os.getenv("POLY_PRIVATE_KEY", ""),
            signature_type=2,
            funder=self._funder or None,
        )
        creds = self._clob.create_or_derive_api_creds()
        self._clob.set_api_creds(creds)

        # Load actual positions from Polymarket
        await self._sync_positions()

        logger.info(
            "live_executor_ready",
            api_key=creds.api_key[:12] + "...",
            positions=len(self._shares_owned),
            funder=self._funder[:12] + "...",
        )
        return self._clob

    async def _sync_positions(self) -> None:
        """Load actual share positions from Polymarket Data API."""
        if not self._funder:
            return
        try:
            url = f"{DATA_API}/positions?user={self._funder}"
            req = urllib.request.Request(url, headers={"User-Agent": "Arbo/1.0"})
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: json.loads(
                    urllib.request.urlopen(req, timeout=10, context=_SSL_CTX).read()
                ),
            )
            self._shares_owned.clear()
            for pos in data:
                token = pos.get("asset", "")
                size = int(float(pos.get("size", 0)))
                if token and size > 0:
                    self._shares_owned[token] = size

            logger.info("live_positions_synced", count=len(self._shares_owned))
        except Exception as e:
            logger.warning("live_positions_sync_failed", error=str(e))

    async def _get_taker_price(self, token_id: str, side: str) -> float | None:
        """Get taker fill price. BUY→SELL price, SELL→BUY price."""
        try:
            opposite = "SELL" if side == "BUY" else "BUY"
            price = await self._poly_client.get_price(token_id, opposite)
            return float(price)
        except Exception as e:
            logger.warning("live_taker_price_failed", side=side, error=str(e))
            return None

    async def buy(
        self,
        token_id: str,
        price: float,
        size_usdc: float,
        neg_risk: bool = True,
        tick_size: str = "0.01",
    ) -> LiveFill:
        """BUY: fetch taker price, post GTC, verify fill, cancel remainder."""
        taker_price = await self._get_taker_price(token_id, "BUY")
        if taker_price is None:
            return self._fail(token_id, "BUY", price, size_usdc, "No taker price")

        taker_price = min(taker_price, 0.99)
        shares = int(size_usdc / taker_price)
        if shares * taker_price < MIN_ORDER_USD:
            return self._fail(token_id, "BUY", taker_price, size_usdc, f"Below min ${MIN_ORDER_USD}")

        logger.info("live_buy_start", paper=price, taker=taker_price, shares=shares, token=token_id[:20])

        fill = LiveFill(
            token_id=token_id, side="BUY",
            price=Decimal(str(taker_price)), size=Decimal(str(size_usdc)),
            shares_requested=shares,
        )

        t0 = time.monotonic()
        try:
            clob = await self._ensure_clob()
            size_matched = await self._post_and_verify(
                clob, token_id, taker_price, shares, "BUY", neg_risk, tick_size,
                fill,
            )

            if size_matched > 0:
                fill.status = "filled" if size_matched >= shares else "partial"
                fill.shares_filled = size_matched
                fill.fill_price = Decimal(str(taker_price))
                fill.usdc_spent = size_matched * taker_price
                self._shares_owned[token_id] = self._shares_owned.get(token_id, 0) + size_matched
            else:
                fill.status = "failed"
                fill.error = fill.error or "No shares filled"

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "live_buy_done", token=token_id[:20], price=taker_price,
                requested=shares, filled=size_matched,
                status=fill.status, latency=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_buy_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def sell(
        self,
        token_id: str,
        price: float,
        shares: float | None = None,
        neg_risk: bool = True,
        tick_size: str = "0.01",
    ) -> LiveFill:
        """SELL: use actual owned shares, taker price, verify fill."""
        # Refresh positions from API to get accurate count
        await self._sync_positions()

        actual = self._shares_owned.get(token_id, 0)
        if actual <= 0:
            return self._fail(token_id, "SELL", price, 0, "No shares owned (synced)")

        taker_price = await self._get_taker_price(token_id, "SELL")
        if taker_price is None:
            return self._fail(token_id, "SELL", price, 0, "No taker price")

        taker_price = round(max(0.01, taker_price), 4)
        sell_shares = actual

        logger.info("live_sell_start", paper=price, taker=taker_price, shares=sell_shares, token=token_id[:20])

        fill = LiveFill(
            token_id=token_id, side="SELL",
            price=Decimal(str(taker_price)), size=Decimal(str(sell_shares)),
            shares_requested=sell_shares,
        )

        t0 = time.monotonic()
        try:
            clob = await self._ensure_clob()
            size_matched = await self._post_and_verify(
                clob, token_id, taker_price, sell_shares, "SELL", neg_risk, tick_size,
                fill,
            )

            if size_matched > 0:
                fill.status = "filled" if size_matched >= sell_shares else "partial"
                fill.shares_filled = size_matched
                fill.fill_price = Decimal(str(taker_price))
                fill.usdc_spent = size_matched * taker_price
                remaining = actual - size_matched
                if remaining <= 0:
                    self._shares_owned.pop(token_id, None)
                else:
                    self._shares_owned[token_id] = remaining
            else:
                fill.status = "failed"
                fill.error = fill.error or "No shares sold"

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "live_sell_done", token=token_id[:20], price=taker_price,
                requested=sell_shares, filled=size_matched,
                status=fill.status, latency=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_sell_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def _post_and_verify(
        self,
        clob: Any,
        token_id: str,
        price: float,
        shares: int,
        side: str,
        neg_risk: bool,
        tick_size: str,
        fill: LiveFill,
    ) -> int:
        """Post GTC order, wait, check fill, cancel remainder. Returns shares filled."""
        from py_clob_client.clob_types import (
            OrderArgs, OrderType, PartialCreateOrderOptions,
        )
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
        fill.order_id = order_id

        # Check immediate fill from response
        immediate_fill = int(float(result.get("takingAmount", "0") or "0"))

        # Wait briefly then check actual fill count
        await asyncio.sleep(FILL_CHECK_DELAY_S)

        size_matched = immediate_fill
        if order_id:
            oid_check = order_id  # Capture for lambda
            try:
                order_info = await loop.run_in_executor(
                    None, lambda: clob.get_order(oid_check)
                )
                if isinstance(order_info, dict):
                    sm = order_info.get("size_matched", "0") or "0"
                    size_matched = max(size_matched, int(float(sm)))
            except Exception as e:
                logger.debug("live_order_check_failed", error=str(e))

            # ALWAYS cancel order (removes unfilled remainder from book)
            oid = order_id  # Capture in local variable for lambda
            try:
                await loop.run_in_executor(None, lambda: clob.cancel(oid))
                logger.info("live_order_cancelled", order_id=oid[:16], filled=size_matched)
            except Exception as e:
                logger.debug("live_cancel_skip", order_id=oid[:16], error=str(e))

        return size_matched

    def _fail(self, token_id: str, side: str, price: float, size: float, error: str) -> LiveFill:
        logger.warning("live_order_rejected", side=side, error=error, token=token_id[:20])
        return LiveFill(
            token_id=token_id, side=side, price=Decimal(str(price)),
            size=Decimal(str(size)), status="failed", error=error,
        )

    @property
    def shares_owned(self) -> dict[str, int]:
        return dict(self._shares_owned)

    @property
    def recent_fills(self) -> list[LiveFill]:
        return self._fills[-100:]
