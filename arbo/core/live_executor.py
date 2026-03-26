"""Live order executor for Strategy C2 — v4 (partial fill handling).

Handles the reality of illiquid weather markets:
- Posts GTC order at taker price (SELL price for BUY, BUY price for SELL)
- Waits briefly for fill, checks size_matched
- Cancels unfilled remainder
- Tracks only actually filled shares

NegRisk pricing:
- /price?side=SELL = what you pay to BUY as taker
- /price?side=BUY = what you receive when SELL as taker

Gas: $0 (Polymarket gasless relay model).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("live_executor")

UTC = timezone.utc
FILL_WAIT_S = 5  # Wait this long after post before checking fill
MIN_ORDER_USD = 1.0  # Polymarket minimum


@dataclass
class LiveFill:
    """Result of a live order execution."""

    token_id: str
    side: str
    price: Decimal
    size: Decimal
    order_id: str = ""
    status: str = "pending"  # filled, partial, failed
    fill_price: Decimal | None = None
    shares_requested: int = 0
    shares_filled: int = 0
    usdc_spent: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: int = 0
    error: str | None = None
    raw_response: dict | None = None

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
    """Execute real trades on Polymarket CLOB.

    Creates own ClobClient for order signing. Uses PolymarketClient
    for price reads. Tracks actual shares owned per token.
    """

    def __init__(self, poly_client: Any) -> None:
        self._poly_client = poly_client
        self._clob: Any = None
        self._fills: list[LiveFill] = []
        self._shares_owned: dict[str, int] = {}

    async def _ensure_clob(self) -> Any:
        if self._clob is not None:
            return self._clob
        import os
        from py_clob_client.client import ClobClient as _ClobClient

        self._clob = _ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=os.getenv("POLY_PRIVATE_KEY", ""),
            signature_type=2,
            funder=os.getenv("POLY_FUNDER_ADDRESS") or None,
        )
        creds = self._clob.create_or_derive_api_creds()
        self._clob.set_api_creds(creds)
        logger.info("live_executor_ready", api_key=creds.api_key[:12] + "...")
        return self._clob

    async def buy(
        self,
        token_id: str,
        price: float,
        size_usdc: float,
        neg_risk: bool = True,
        tick_size: str = "0.01",
    ) -> LiveFill:
        """BUY token: fetch taker price, post GTC, verify fill, cancel remainder."""
        # 1. Get real taker price (SELL price = what we pay)
        taker_price = price
        try:
            sell_price = await self._poly_client.get_price(token_id, "SELL")
            taker_price = float(sell_price)
            logger.info("live_buy_price", paper=price, taker=taker_price, token=token_id[:20])
        except Exception as e:
            logger.warning("live_buy_price_failed", error=str(e))

        taker_price = min(taker_price, 0.99)
        shares = int(size_usdc / taker_price)

        if shares * taker_price < MIN_ORDER_USD:
            return self._fail(token_id, "BUY", taker_price, size_usdc, "Order below $1 minimum")

        fill = LiveFill(
            token_id=token_id, side="BUY",
            price=Decimal(str(taker_price)), size=Decimal(str(size_usdc)),
            shares_requested=shares,
        )

        t0 = time.monotonic()
        try:
            from py_clob_client.clob_types import (
                OrderArgs, OrderType, PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import BUY as _BUY

            clob = await self._ensure_clob()
            loop = asyncio.get_event_loop()

            args = OrderArgs(token_id=token_id, price=taker_price, size=shares, side=_BUY)
            opts = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

            def _post():
                signed = clob.create_order(args, opts)
                return clob.post_order(signed, OrderType.GTC)

            result = await loop.run_in_executor(None, _post)
            fill.order_id = result.get("orderID", result.get("id", ""))
            fill.raw_response = result

            # 2. Check immediate result
            success = result.get("success", False)
            taking = int(float(result.get("takingAmount", "0")))

            if success and taking > 0:
                # Some shares matched immediately
                fill.shares_filled = taking
                fill.fill_price = Decimal(str(taker_price))
                fill.usdc_spent = taking * taker_price
            else:
                # Wait and check
                await asyncio.sleep(FILL_WAIT_S)

            # 3. Verify via get_order
            if fill.order_id:
                order_info = await loop.run_in_executor(
                    None, lambda: clob.get_order(fill.order_id)
                )
                if isinstance(order_info, dict):
                    size_matched = int(float(order_info.get("size_matched", "0")))
                    original_size = int(float(order_info.get("original_size", str(shares))))
                    order_status = order_info.get("status", "")

                    fill.shares_filled = size_matched
                    fill.usdc_spent = size_matched * taker_price

                    if size_matched >= original_size:
                        fill.status = "filled"
                    elif size_matched > 0:
                        fill.status = "partial"
                        # Cancel unfilled remainder
                        await self._cancel(fill.order_id, clob)
                        logger.info(
                            "live_buy_partial",
                            filled=size_matched, requested=original_size,
                            token=token_id[:20],
                        )
                    else:
                        fill.status = "failed"
                        fill.error = "No shares filled"
                        await self._cancel(fill.order_id, clob)

            # 4. Track actual shares
            if fill.shares_filled > 0:
                self._shares_owned[token_id] = self._shares_owned.get(token_id, 0) + fill.shares_filled
                fill.fill_price = Decimal(str(taker_price))
                if fill.status == "pending":
                    fill.status = "filled"

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "live_buy_done",
                token=token_id[:20], price=taker_price,
                requested=shares, filled=fill.shares_filled,
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
        """SELL token: use actual owned shares, taker price, verify fill."""
        actual = self._shares_owned.get(token_id, 0)
        if actual <= 0:
            return self._fail(token_id, "SELL", price, 0, "No shares owned")

        sell_shares = actual  # Sell all owned

        # Get taker price (BUY price = what buyers bid)
        taker_price = price
        try:
            buy_price = await self._poly_client.get_price(token_id, "BUY")
            taker_price = float(buy_price)
            logger.info("live_sell_price", paper=price, taker=taker_price, token=token_id[:20])
        except Exception as e:
            logger.warning("live_sell_price_failed", error=str(e))

        taker_price = round(max(0.01, taker_price), 4)

        fill = LiveFill(
            token_id=token_id, side="SELL",
            price=Decimal(str(taker_price)), size=Decimal(str(sell_shares)),
            shares_requested=sell_shares,
        )

        t0 = time.monotonic()
        try:
            from py_clob_client.clob_types import (
                OrderArgs, OrderType, PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import SELL as _SELL

            clob = await self._ensure_clob()
            loop = asyncio.get_event_loop()

            args = OrderArgs(
                token_id=token_id, price=taker_price, size=sell_shares, side=_SELL,
            )
            opts = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

            def _post():
                signed = clob.create_order(args, opts)
                return clob.post_order(signed, OrderType.GTC)

            result = await loop.run_in_executor(None, _post)
            fill.order_id = result.get("orderID", result.get("id", ""))
            fill.raw_response = result

            success = result.get("success", False)
            taking = int(float(result.get("takingAmount", "0")))

            if not (success and taking > 0):
                await asyncio.sleep(FILL_WAIT_S)

            # Verify fill
            if fill.order_id:
                order_info = await loop.run_in_executor(
                    None, lambda: clob.get_order(fill.order_id)
                )
                if isinstance(order_info, dict):
                    size_matched = int(float(order_info.get("size_matched", "0")))

                    if size_matched > 0:
                        fill.shares_filled = size_matched
                        fill.fill_price = Decimal(str(taker_price))
                        fill.usdc_spent = size_matched * taker_price
                        fill.status = "filled" if size_matched >= sell_shares else "partial"

                        # Update owned shares
                        remaining = actual - size_matched
                        if remaining <= 0:
                            self._shares_owned.pop(token_id, None)
                        else:
                            self._shares_owned[token_id] = remaining
                    else:
                        fill.status = "failed"
                        fill.error = "No shares sold"

                    # Cancel remainder if partial
                    if size_matched < sell_shares:
                        await self._cancel(fill.order_id, clob)

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "live_sell_done",
                token=token_id[:20], price=taker_price,
                requested=sell_shares, filled=fill.shares_filled,
                status=fill.status, latency=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_sell_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def _cancel(self, order_id: str, clob: Any) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: clob.cancel(order_id))
            logger.info("live_order_cancelled", order_id=order_id[:16])
        except Exception as e:
            logger.debug("live_cancel_failed", error=str(e))

    def _fail(self, token_id: str, side: str, price: float, size: float, error: str) -> LiveFill:
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
