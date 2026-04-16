"""Tests for LiveExecutor — focused on platform-rejection guards.

Polymarket's CLOB rejects orders with <5 shares. These tests ensure we
fail fast client-side (skip the order attempt) rather than submitting
and being rejected.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.core.live_executor import MIN_ORDER_SHARES, LiveExecutor


@pytest.mark.asyncio
async def test_buy_rejects_below_min_shares() -> None:
    """Buy with size_usdc that produces <5 shares must fail client-side.

    Regression: before this guard, live_executor posted 4-share orders
    and received "Size (4) lower than the minimum: 5" rejection from
    Polymarket — wasted signal + log noise (19 events observed Apr 15).
    """
    poly = MagicMock()
    # buy/sell price 0.78 → 3 shares for $2.5, rejected
    poly.get_price = AsyncMock(side_effect=lambda _t, side: Decimal("0.78"))
    executor = LiveExecutor(poly)

    fill = await executor.buy(token_id="tok", price=0.78, size_usdc=2.5)

    assert fill.status == "failed"
    assert str(MIN_ORDER_SHARES) in (fill.error or "")


@pytest.mark.asyncio
async def test_buy_rejects_below_min_shares_high_price() -> None:
    """At price 0.80, $4 produces int(4/0.8) = 5 shares (OK).

    Boundary test: verify the shares_floor formula (5 * entry_price) in
    strategy dynamic_min is sufficient.
    """
    poly = MagicMock()
    poly.get_price = AsyncMock(side_effect=lambda _t, side: Decimal("0.80"))
    poly.get_orderbook = AsyncMock()
    executor = LiveExecutor(poly)

    # $3.90 at 0.80 → 4 shares (rejected)
    fill = await executor.buy(token_id="tok", price=0.80, size_usdc=3.90)
    assert fill.status == "failed"
    assert "5" in (fill.error or "")


@pytest.mark.asyncio
async def test_sell_skips_below_min_shares() -> None:
    """Sell of <5 shares must skip (Polymarket would reject).

    These orphan shares auto-resolve at event end — no manual sell needed.
    """
    poly = MagicMock()
    poly.get_price = AsyncMock(side_effect=lambda _t, side: Decimal("0.50"))
    executor = LiveExecutor(poly)
    executor._shares_owned["tok"] = 3  # <MIN_ORDER_SHARES

    fill = await executor.sell(
        token_id="tok", price=0.50, skip_sync=True,
    )

    assert fill.status == "skipped"
    assert fill.order_type == "below_min"
    assert "5" in (fill.error or "")
