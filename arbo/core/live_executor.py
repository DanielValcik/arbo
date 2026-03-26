"""Live order executor for Strategy C2.

Submits real orders to Polymarket CLOB via py-clob-client.
Records all execution data for monitoring and comparison with paper.

Gas: $0 (Polymarket relays transactions, gasless model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import asyncio

from arbo.connectors.polymarket_client import PolymarketClient
from arbo.utils.logger import get_logger

logger = get_logger("live_executor")

UTC = timezone.utc


@dataclass
class LiveFill:
    """Result of a live order execution with full monitoring data."""

    token_id: str
    side: str  # BUY or SELL
    price: Decimal  # Submitted limit price
    size: Decimal  # USDC amount (BUY) or shares (SELL)
    order_id: str = ""
    status: str = "pending"  # pending, filled, partial, failed
    fill_price: Decimal | None = None
    shares_filled: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    latency_ms: int = 0
    error: str | None = None
    raw_response: dict | None = None

    def to_monitoring_dict(self) -> dict:
        """Full monitoring data for trade_details JSONB."""
        return {
            "live_order_id": self.order_id,
            "live_submitted_price": float(self.price),
            "live_fill_price": float(self.fill_price) if self.fill_price else None,
            "live_fill_slippage": float(self.fill_price - self.price) if self.fill_price else None,
            "live_shares_filled": float(self.shares_filled),
            "live_latency_ms": self.latency_ms,
            "live_status": self.status,
            "live_error": self.error,
            "live_timestamp": self.timestamp.isoformat(),
            "live_gas_usd": 0.0,  # Polymarket is gasless
        }


class LiveExecutor:
    """Execute real trades on Polymarket CLOB.

    Creates its own ClobClient with derived L2 credentials to avoid
    sharing state with the read-only orderbook client.
    """

    def __init__(self, poly_client: PolymarketClient) -> None:
        self._client = poly_client
        self._clob: Any = None  # Own ClobClient for order signing
        self._fills: list[LiveFill] = []

    async def _ensure_clob(self) -> Any:
        """Create dedicated ClobClient with derived L2 creds."""
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
        logger.info("live_executor_clob_ready", api_key=creds.api_key[:12] + "...")
        return self._clob

    async def buy(
        self,
        token_id: str,
        price: float,
        size_usdc: float,
        neg_risk: bool = True,
        tick_size: str = "0.01",
    ) -> LiveFill:
        """Place a BUY order on CLOB.

        Args:
            token_id: YES or NO token ID.
            price: Limit price (what we're willing to pay per share).
            size_usdc: How many USDC to spend.
            neg_risk: Weather markets are NegRisk.
            tick_size: Market tick size.

        Returns:
            LiveFill with execution details.
        """
        # Calculate shares: size / price (max 2 decimal places per CLOB spec)
        shares = round(size_usdc / price, 0)  # Round to integer shares for safety

        fill = LiveFill(
            token_id=token_id,
            side="BUY",
            price=Decimal(str(price)),
            size=Decimal(str(size_usdc)),
        )

        t0 = time.monotonic()
        try:
            # Call py-clob-client directly (sync) via run_in_executor
            # to avoid async/coroutine confusion in _retry wrapper
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY as _BUY

            order_args = OrderArgs(token_id=token_id, price=price, size=shares, side=_BUY)
            from py_clob_client.clob_types import PartialCreateOrderOptions
            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

            clob = await self._ensure_clob()

            def _do_buy():
                signed = clob.create_order(order_args, options)
                return clob.post_order(signed, OrderType.GTC)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_buy)

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.raw_response = result
            fill.order_id = result.get("orderID", result.get("id", ""))

            # Check fill status
            status = result.get("status", "")
            if status in ("matched", "filled"):
                fill.status = "filled"
                fill.fill_price = Decimal(str(result.get("price", price)))
                fill.shares_filled = Decimal(str(shares))
            elif status in ("live", "delayed"):
                # GTC order posted to orderbook, waiting for match
                fill.status = "filled"  # Treat as success — order is active
                fill.fill_price = Decimal(str(price))
                fill.shares_filled = Decimal(str(shares))
            else:
                fill.status = "failed"
                fill.error = f"Unexpected status: {status}"

            logger.info(
                "live_buy_executed",
                token_id=token_id[:30],
                price=price,
                shares=shares,
                size_usdc=size_usdc,
                status=fill.status,
                order_id=fill.order_id[:20] if fill.order_id else "",
                latency_ms=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error(
                "live_buy_failed",
                token_id=token_id[:30],
                price=price,
                size_usdc=size_usdc,
                error=str(e),
            )

        self._fills.append(fill)
        return fill

    async def sell(
        self,
        token_id: str,
        price: float,
        shares: float,
        neg_risk: bool = True,
        tick_size: str = "0.01",
    ) -> LiveFill:
        """Place a SELL order on CLOB.

        Args:
            token_id: Token ID of position to sell.
            price: Limit price (minimum we accept per share).
            shares: Number of shares to sell.
            neg_risk: Weather markets are NegRisk.
            tick_size: Market tick size.

        Returns:
            LiveFill with execution details.
        """
        fill = LiveFill(
            token_id=token_id,
            side="SELL",
            price=Decimal(str(price)),
            size=Decimal(str(shares)),
        )

        t0 = time.monotonic()
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL as _SELL

            order_args = OrderArgs(token_id=token_id, price=price, size=shares, side=_SELL)
            from py_clob_client.clob_types import PartialCreateOrderOptions
            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

            clob = await self._ensure_clob()

            def _do_sell():
                signed = clob.create_order(order_args, options)
                return clob.post_order(signed, OrderType.GTC)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_sell)

            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.raw_response = result
            fill.order_id = result.get("orderID", result.get("id", ""))

            status = result.get("status", "")
            if status in ("matched", "filled"):
                fill.status = "filled"
                fill.fill_price = Decimal(str(result.get("price", price)))
                fill.shares_filled = Decimal(str(shares))
            else:
                fill.status = "failed"
                fill.error = f"Unexpected status: {status}"

            logger.info(
                "live_sell_executed",
                token_id=token_id[:30],
                price=price,
                shares=shares,
                status=fill.status,
                order_id=fill.order_id[:20] if fill.order_id else "",
                latency_ms=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error(
                "live_sell_failed",
                token_id=token_id[:30],
                price=price,
                shares=shares,
                error=str(e),
            )

        self._fills.append(fill)
        return fill

    async def get_balance(self) -> dict[str, Any]:
        """Check available USDC balance on Polymarket."""
        try:
            return await self._client.get_balance_allowance()
        except Exception as e:
            logger.error("balance_check_failed", error=str(e))
            return {}

    @property
    def recent_fills(self) -> list[LiveFill]:
        """Last 100 fills for monitoring."""
        return self._fills[-100:]
