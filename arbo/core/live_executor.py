"""Live order executor for Strategy C2.

Submits real orders to Polymarket CLOB. Designed to match paper trading
behavior exactly:
- BUY: taker at ask price → instant fill (same as paper's clob_fill + slippage)
- SELL: taker at bid price → instant fill (same as paper's best_bid)

Key rules:
- Always cross the spread for instant fills (taker, not maker)
- Verify fill status before tracking position
- Track actual shares owned for correct sell sizing
- Cancel unfilled orders immediately (no resting orders)
- Gas: $0 (Polymarket gasless relay model)
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

# No buffer needed — strategy already computes correct ask/bid from orderbook
# NegRisk markets have inverted spread (bid > ask), so buffer would worsen price
TAKER_BUFFER = 0.0


@dataclass
class LiveFill:
    """Result of a live order execution with full monitoring data."""

    token_id: str
    side: str
    price: Decimal  # Submitted limit price
    size: Decimal  # USDC (BUY) or shares (SELL)
    order_id: str = ""
    status: str = "pending"
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
            "live_gas_usd": 0.0,
        }


class LiveExecutor:
    """Execute real trades on Polymarket CLOB.

    Creates its own ClobClient with derived L2 credentials.
    All orders are taker (cross the spread) for instant fills.
    """

    def __init__(self, poly_client: Any) -> None:
        self._poly_client = poly_client
        self._clob: Any = None
        self._fills: list[LiveFill] = []
        # Track actual shares owned per token_id (for correct sell sizing)
        self._shares_owned: dict[str, float] = {}

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
        """BUY as taker — price at ask + buffer for instant fill.

        Submits limit order above the ask to cross the spread and get
        matched immediately. Same as paper trading behavior.
        """
        # Price above ask to guarantee taker fill
        taker_price = round(price + TAKER_BUFFER, 4)
        taker_price = min(taker_price, 0.99)  # Cap at 0.99
        shares = int(size_usdc / taker_price)  # Integer shares
        if shares < 1:
            return LiveFill(token_id=token_id, side="BUY", price=Decimal(str(price)),
                            size=Decimal(str(size_usdc)), status="failed", error="shares < 1")

        fill = LiveFill(
            token_id=token_id, side="BUY",
            price=Decimal(str(taker_price)), size=Decimal(str(size_usdc)),
        )

        t0 = time.monotonic()
        try:
            from py_clob_client.clob_types import (
                OrderArgs, OrderType, PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import BUY as _BUY

            order_args = OrderArgs(
                token_id=token_id, price=taker_price, size=shares, side=_BUY,
            )
            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
            clob = await self._ensure_clob()
            loop = asyncio.get_event_loop()

            def _post():
                signed = clob.create_order(order_args, options)
                return clob.post_order(signed, OrderType.GTC)

            result = await loop.run_in_executor(None, _post)
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.order_id = result.get("orderID", result.get("id", ""))
            fill.raw_response = result

            status = result.get("status", "")

            if status in ("matched", "filled"):
                # Instant taker fill — exactly what we want
                fill.status = "filled"
                fill.fill_price = Decimal(str(result.get("price", taker_price)))
                fill.shares_filled = Decimal(str(shares))
                self._shares_owned[token_id] = self._shares_owned.get(token_id, 0) + shares
            elif status in ("live", "delayed"):
                # Order sitting in book — not good for taker strategy.
                # Wait briefly then cancel if not filled.
                matched = await self._wait_for_fill(fill.order_id, timeout_s=10, clob=clob)
                if matched:
                    fill.status = "filled"
                    fill.fill_price = Decimal(str(taker_price))
                    fill.shares_filled = Decimal(str(shares))
                    self._shares_owned[token_id] = self._shares_owned.get(token_id, 0) + shares
                else:
                    await self._cancel_order(fill.order_id, clob)
                    fill.status = "failed"
                    fill.error = "Not matched as taker — cancelled (price may have moved)"
            else:
                fill.status = "failed"
                fill.error = f"Unexpected: {status}"

            logger.info(
                "live_buy",
                token=token_id[:20], price=taker_price, shares=shares,
                status=fill.status, latency=fill.latency_ms,
                order_id=fill.order_id[:16] if fill.order_id else "",
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
        """SELL as taker — price at bid - buffer for instant fill.

        Uses actual owned shares (not paper amount) to avoid
        'not enough balance' errors.
        """
        # Use actual owned shares, not paper estimate
        actual_shares = self._shares_owned.get(token_id, 0)
        if actual_shares <= 0:
            return LiveFill(token_id=token_id, side="SELL", price=Decimal(str(price)),
                            size=Decimal("0"), status="failed", error="No shares owned")

        sell_shares = int(actual_shares)  # Integer
        # Price below bid to guarantee taker fill
        taker_price = round(max(0.01, price - TAKER_BUFFER), 4)

        fill = LiveFill(
            token_id=token_id, side="SELL",
            price=Decimal(str(taker_price)), size=Decimal(str(sell_shares)),
        )

        t0 = time.monotonic()
        try:
            from py_clob_client.clob_types import (
                OrderArgs, OrderType, PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import SELL as _SELL

            order_args = OrderArgs(
                token_id=token_id, price=taker_price, size=sell_shares, side=_SELL,
            )
            options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)
            clob = await self._ensure_clob()
            loop = asyncio.get_event_loop()

            def _post():
                signed = clob.create_order(order_args, options)
                return clob.post_order(signed, OrderType.GTC)

            result = await loop.run_in_executor(None, _post)
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.order_id = result.get("orderID", result.get("id", ""))
            fill.raw_response = result

            status = result.get("status", "")

            if status in ("matched", "filled"):
                fill.status = "filled"
                fill.fill_price = Decimal(str(result.get("price", taker_price)))
                fill.shares_filled = Decimal(str(sell_shares))
                self._shares_owned.pop(token_id, None)
            elif status in ("live", "delayed"):
                matched = await self._wait_for_fill(fill.order_id, timeout_s=10, clob=clob)
                if matched:
                    fill.status = "filled"
                    fill.fill_price = Decimal(str(taker_price))
                    fill.shares_filled = Decimal(str(sell_shares))
                    self._shares_owned.pop(token_id, None)
                else:
                    await self._cancel_order(fill.order_id, clob)
                    fill.status = "failed"
                    fill.error = "Sell not matched — cancelled"
            else:
                fill.status = "failed"
                fill.error = f"Unexpected: {status}"

            logger.info(
                "live_sell",
                token=token_id[:20], price=taker_price, shares=sell_shares,
                status=fill.status, latency=fill.latency_ms,
            )

        except Exception as e:
            fill.latency_ms = int((time.monotonic() - t0) * 1000)
            fill.status = "failed"
            fill.error = str(e)
            logger.error("live_sell_error", token=token_id[:20], error=str(e))

        self._fills.append(fill)
        return fill

    async def _wait_for_fill(
        self, order_id: str, timeout_s: int = 10, clob: Any = None,
    ) -> bool:
        """Poll order until matched or timeout. Returns True if filled."""
        if not order_id or not clob:
            return False
        loop = asyncio.get_event_loop()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            try:
                order = await loop.run_in_executor(
                    None, lambda: clob.get_order(order_id)
                )
                status = order.get("status", "") if isinstance(order, dict) else ""
                if status in ("matched", "filled"):
                    return True
                if status in ("cancelled", "expired"):
                    return False
            except Exception:
                pass
        return False

    async def _cancel_order(self, order_id: str, clob: Any) -> None:
        """Cancel an open order (best effort)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: clob.cancel(order_id))
            logger.info("live_order_cancelled", order_id=order_id[:16])
        except Exception as e:
            logger.debug("live_cancel_failed", error=str(e))

    async def get_balance(self) -> dict[str, Any]:
        """Check available USDC balance."""
        try:
            return await self._poly_client.get_balance_allowance()
        except Exception:
            return {}

    @property
    def shares_owned(self) -> dict[str, float]:
        """Current shares per token_id."""
        return dict(self._shares_owned)

    @property
    def recent_fills(self) -> list[LiveFill]:
        return self._fills[-100:]
