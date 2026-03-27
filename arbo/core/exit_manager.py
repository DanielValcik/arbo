"""Persistent Exit Manager — ensures ALL shares are sold, never holds to resolution.

When C2 triggers an exit (edge_lost, profit_take, prob_floor), the exit
manager takes ownership and keeps trying until every share is sold.

Strategy:
1. Taker sell (aggressive, immediate fill at BUY price) — repeat every 60s
2. After 5 failed taker attempts → switch to maker sell (post limit at higher price)
3. Maker order stays in book for 5 min, then re-price if not filled
4. Track progress: original shares → remaining → 0 = done

Never gives up. Never holds to resolution.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("exit_manager")

UTC = timezone.utc
TAKER_RETRY_INTERVAL_S = 60  # Retry taker sell every 60s
TAKER_MAX_ATTEMPTS = 5  # After this many taker fails → switch to maker
MAKER_REPRICE_INTERVAL_S = 300  # Re-price maker order every 5 min
MAKER_PRICE_BUMP = 0.01  # Increase sell price by 1 tick each reprice


@dataclass
class PendingExit:
    """An exit that needs to be completed."""

    token_id: str
    city: str  # For weather; for crypto, use label instead
    exit_reason: str
    original_shares: int
    remaining_shares: int
    entry_price: float
    taker_attempts: int = 0
    maker_order_id: str | None = None
    maker_price: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    last_attempt_at: float = 0.0
    total_sold: int = 0
    total_revenue: float = 0.0
    neg_risk: bool = True  # False for B2 crypto markets
    label: str = ""  # Display label (e.g. "BTC_88000" for B2)


class ExitManager:
    """Manages persistent sell-down of positions until fully exited."""

    def __init__(self, live_executor: Any) -> None:
        self._executor = live_executor
        self._pending: dict[str, PendingExit] = {}

    def register_exit(
        self, token_id: str, city: str, exit_reason: str, shares: int, entry_price: float,
    ) -> None:
        """Register a new exit to be completed."""
        if token_id in self._pending:
            return  # Already being managed

        self._pending[token_id] = PendingExit(
            token_id=token_id,
            city=city,
            exit_reason=exit_reason,
            original_shares=shares,
            remaining_shares=shares,
            entry_price=entry_price,
        )
        logger.info(
            "exit_registered",
            city=city,
            shares=shares,
            reason=exit_reason,
            token=token_id[:20],
        )

    async def process_exits(self) -> list[dict]:
        """Process all pending exits. Called every 60s from exit monitor.

        Returns list of completed exits with P&L info.
        """
        completed = []

        for token_id, pe in list(self._pending.items()):
            now = time.monotonic()

            # Rate limit: don't hammer the same token
            if now - pe.last_attempt_at < TAKER_RETRY_INTERVAL_S:
                continue

            pe.last_attempt_at = now

            # Sync actual shares from API
            await self._executor._sync_positions()
            actual = self._executor._shares_owned.get(token_id, 0)
            if actual <= 0:
                # All shares sold (by previous attempt or resolution)
                pe.remaining_shares = 0
                completed.append(self._complete_exit(pe))
                continue

            pe.remaining_shares = actual

            if pe.taker_attempts < TAKER_MAX_ATTEMPTS:
                # Phase 1: Taker sell
                sold = await self._taker_sell(pe)
                pe.taker_attempts += 1
                if sold > 0:
                    pe.total_sold += sold
                    pe.remaining_shares = max(0, actual - sold)
                    revenue = sold * pe.maker_price if pe.maker_price > 0 else 0
                    pe.total_revenue += revenue

                    logger.info(
                        "exit_taker_progress",
                        city=pe.city,
                        sold=sold,
                        remaining=pe.remaining_shares,
                        attempt=pe.taker_attempts,
                        token=token_id[:20],
                    )
            else:
                # Phase 2: Maker sell (post limit order and wait)
                sold = await self._maker_sell(pe)
                if sold > 0:
                    pe.total_sold += sold
                    pe.remaining_shares = max(0, actual - sold)
                    pe.total_revenue += sold * pe.maker_price

                    logger.info(
                        "exit_maker_progress",
                        city=pe.city,
                        sold=sold,
                        remaining=pe.remaining_shares,
                        token=token_id[:20],
                    )

            # Check if fully exited
            if pe.remaining_shares <= 0:
                completed.append(self._complete_exit(pe))

        return completed

    async def _taker_sell(self, pe: PendingExit) -> int:
        """Aggressive taker sell at BUY price."""
        fill = await self._executor.sell(
            token_id=pe.token_id,
            price=pe.entry_price,  # Will be replaced by taker price inside executor
            neg_risk=pe.neg_risk,
        )
        if fill.fill_price:
            pe.maker_price = float(fill.fill_price)
        return fill.shares_filled

    async def _maker_sell(self, pe: PendingExit) -> int:
        """Patient maker sell — post limit at slightly better price and wait."""
        # Get current sell price and add a small premium (maker = better price)
        try:
            taker_price = await self._executor._get_taker_price(pe.token_id, "SELL")
            if taker_price is None:
                return 0
            # Maker price: slightly above taker bid (attracts buyers)
            maker_price = round(taker_price + MAKER_PRICE_BUMP, 4)
        except Exception:
            return 0

        # Cancel existing maker order if any
        if pe.maker_order_id:
            clob = await self._executor._ensure_clob()
            await self._executor._cancel(pe.maker_order_id, clob)
            pe.maker_order_id = None

        # Post maker sell order
        actual = self._executor._shares_owned.get(pe.token_id, 0)
        if actual <= 0:
            return 0

        try:
            from py_clob_client.clob_types import (
                OrderArgs, OrderType, PartialCreateOrderOptions,
            )
            from py_clob_client.order_builder.constants import SELL as _SELL

            clob = await self._executor._ensure_clob()
            loop = asyncio.get_event_loop()

            args = OrderArgs(
                token_id=pe.token_id, price=maker_price, size=actual, side=_SELL,
            )
            opts = PartialCreateOrderOptions(tick_size="0.01", neg_risk=True)

            def _post():
                signed = clob.create_order(args, opts)
                return clob.post_order(signed, OrderType.GTC)

            result = await loop.run_in_executor(None, _post)
            pe.maker_order_id = result.get("orderID", result.get("id", ""))
            pe.maker_price = maker_price

            # Wait longer for maker fills (5 min)
            await asyncio.sleep(min(MAKER_REPRICE_INTERVAL_S, 60))

            # Check how many filled
            if pe.maker_order_id:
                order_info = await loop.run_in_executor(
                    None, lambda oid=pe.maker_order_id: clob.get_order(oid)
                )
                if isinstance(order_info, dict):
                    sm = int(float(order_info.get("size_matched", "0") or "0"))
                    # Cancel remainder
                    await self._executor._cancel(pe.maker_order_id, clob)
                    pe.maker_order_id = None

                    if sm > 0:
                        self._executor._shares_owned[pe.token_id] = max(
                            0, self._executor._shares_owned.get(pe.token_id, 0) - sm
                        )
                    return sm

            logger.info(
                "exit_maker_posted",
                city=pe.city,
                price=maker_price,
                shares=actual,
                token=pe.token_id[:20],
            )

        except Exception as e:
            logger.warning("exit_maker_error", city=pe.city, error=str(e))

        return 0

    def _complete_exit(self, pe: PendingExit) -> dict:
        """Mark exit as complete and remove from pending."""
        self._pending.pop(pe.token_id, None)

        avg_sell_price = pe.total_revenue / pe.total_sold if pe.total_sold > 0 else 0
        pnl = pe.total_revenue - (pe.total_sold * pe.entry_price)

        logger.info(
            "exit_complete",
            city=pe.city,
            reason=pe.exit_reason,
            sold=pe.total_sold,
            avg_price=round(avg_sell_price, 4),
            pnl=round(pnl, 2),
            token=pe.token_id[:20],
        )

        return {
            "token_id": pe.token_id,
            "city": pe.city,
            "exit_reason": pe.exit_reason,
            "shares_sold": pe.total_sold,
            "avg_sell_price": avg_sell_price,
            "entry_price": pe.entry_price,
            "pnl": pnl,
        }

    @property
    def pending_exits(self) -> dict[str, PendingExit]:
        return dict(self._pending)

    @property
    def has_pending(self) -> bool:
        return len(self._pending) > 0
