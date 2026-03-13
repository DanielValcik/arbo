"""Tests for OrderbookProvider — NegRisk-aware CLOB price integration."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.connectors.orderbook_provider import (
    OrderbookProvider,
    OrderbookSnapshot,
    available_depth,
    estimate_fill_price,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(price: str, size: str) -> MagicMock:
    """Create a mock OrderbookEntry."""
    e = MagicMock()
    e.price = Decimal(price)
    e.size = Decimal(size)
    return e


def _make_orderbook(
    token_id: str,
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
) -> MagicMock:
    """Create a mock Orderbook DTO."""
    book = MagicMock()
    book.token_id = token_id
    book.bids = [_make_entry(p, s) for p, s in bids]
    book.asks = [_make_entry(p, s) for p, s in asks]
    book.midpoint = None
    book.spread = None
    return book


def _make_snapshot(
    bids: list[tuple[str, str]] | None = None,
    asks: list[tuple[str, str]] | None = None,
    is_negrisk: bool = False,
) -> OrderbookSnapshot:
    """Create an OrderbookSnapshot for testing."""
    bid_list = [(Decimal(p), Decimal(s)) for p, s in (bids or [])]
    ask_list = [(Decimal(p), Decimal(s)) for p, s in (asks or [])]

    bid_depth = sum(p * s for p, s in bid_list)
    ask_depth = sum(p * s for p, s in ask_list)

    return OrderbookSnapshot(
        token_id="test_token",
        best_bid=bid_list[0][0] if bid_list else None,
        best_ask=ask_list[0][0] if ask_list else None,
        midpoint=(
            (bid_list[0][0] + ask_list[0][0]) / 2
            if bid_list and ask_list
            else None
        ),
        spread=(
            ask_list[0][0] - bid_list[0][0]
            if bid_list and ask_list
            else None
        ),
        bid_depth_usdc=bid_depth,
        ask_depth_usdc=ask_depth,
        bids=bid_list,
        asks=ask_list,
        fetched_at=time.monotonic(),
        is_negrisk=is_negrisk,
    )


# ---------------------------------------------------------------------------
# estimate_fill_price tests
# ---------------------------------------------------------------------------


class TestEstimateFillPrice:
    """Tests for estimate_fill_price()."""

    def test_buy_walks_asks(self) -> None:
        """BUY should walk through asks from best to worst."""
        snap = _make_snapshot(
            asks=[("0.40", "100"), ("0.42", "200"), ("0.45", "500")],
        )
        # Buy $40 worth — fits entirely at first level (100 shares * 0.40 = $40)
        fill = estimate_fill_price(snap, "BUY", Decimal("40"))
        assert fill is not None
        assert fill == Decimal("0.4000")

    def test_buy_walks_multiple_levels(self) -> None:
        """BUY exceeding first level walks into next."""
        snap = _make_snapshot(
            asks=[("0.40", "50"), ("0.50", "100")],
        )
        # First level: 50 shares * 0.40 = $20
        # Need $30 more at 0.50 = 60 shares
        # VWAP = $50 / (50 + 60) = 0.4545...
        fill = estimate_fill_price(snap, "BUY", Decimal("50"))
        assert fill is not None
        assert Decimal("0.45") < fill < Decimal("0.50")

    def test_sell_walks_bids(self) -> None:
        """SELL should walk through bids from best to worst."""
        snap = _make_snapshot(
            bids=[("0.60", "200"), ("0.58", "300")],
        )
        fill = estimate_fill_price(snap, "SELL", Decimal("50"))
        assert fill is not None
        assert fill == Decimal("0.6000")

    def test_empty_book_returns_none(self) -> None:
        """Empty orderbook returns None."""
        snap = _make_snapshot(asks=[], bids=[])
        assert estimate_fill_price(snap, "BUY", Decimal("10")) is None

    def test_insufficient_depth(self) -> None:
        """When depth < size, fills what's available."""
        snap = _make_snapshot(
            asks=[("0.40", "10")],  # Only $4 of depth
        )
        fill = estimate_fill_price(snap, "BUY", Decimal("100"))
        assert fill is not None
        # Should fill at 0.40 (all available)
        assert fill == Decimal("0.4000")

    def test_zero_size(self) -> None:
        """Zero size returns None."""
        snap = _make_snapshot(asks=[("0.40", "100")])
        fill = estimate_fill_price(snap, "BUY", Decimal("0"))
        assert fill is None

    def test_negrisk_returns_best_ask_directly(self) -> None:
        """NegRisk snap returns best_ask for BUY without walking."""
        snap = _make_snapshot(
            bids=[("0.38", "500")],
            asks=[("0.42", "500")],
            is_negrisk=True,
        )
        fill = estimate_fill_price(snap, "BUY", Decimal("100"))
        assert fill == Decimal("0.42")

    def test_negrisk_returns_best_bid_for_sell(self) -> None:
        """NegRisk snap returns best_bid for SELL."""
        snap = _make_snapshot(
            bids=[("0.38", "500")],
            asks=[("0.42", "500")],
            is_negrisk=True,
        )
        fill = estimate_fill_price(snap, "SELL", Decimal("100"))
        assert fill == Decimal("0.38")


# ---------------------------------------------------------------------------
# available_depth tests
# ---------------------------------------------------------------------------


class TestAvailableDepth:
    """Tests for available_depth()."""

    def test_buy_returns_ask_depth(self) -> None:
        """BUY side checks ask depth."""
        snap = _make_snapshot(
            asks=[("0.40", "100"), ("0.42", "200")],
            bids=[("0.38", "50")],
        )
        depth = available_depth(snap, "BUY")
        # 100*0.40 + 200*0.42 = 40 + 84 = 124
        assert depth == Decimal("124.00")

    def test_sell_returns_bid_depth(self) -> None:
        """SELL side checks bid depth."""
        snap = _make_snapshot(
            bids=[("0.38", "50")],
            asks=[("0.40", "100")],
        )
        depth = available_depth(snap, "SELL")
        # 50*0.38 = 19
        assert depth == Decimal("19.00")

    def test_empty_depth(self) -> None:
        """Empty book has zero depth."""
        snap = _make_snapshot()
        assert available_depth(snap, "BUY") == Decimal("0")


# ---------------------------------------------------------------------------
# OrderbookProvider tests
# ---------------------------------------------------------------------------


class TestOrderbookProvider:
    """Tests for OrderbookProvider."""

    def test_no_client_returns_none(self) -> None:
        """Without client, get_snapshot returns None."""
        import asyncio

        provider = OrderbookProvider(poly_client=None)
        result = asyncio.get_event_loop().run_until_complete(
            provider.get_snapshot("token123")
        )
        assert result is None

    def test_no_client_batch_returns_empty(self) -> None:
        """Without client, batch returns empty dict."""
        import asyncio

        provider = OrderbookProvider(poly_client=None)
        result = asyncio.get_event_loop().run_until_complete(
            provider.get_snapshots_batch(["t1", "t2"])
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_negrisk_uses_price_endpoint(self) -> None:
        """NegRisk snapshot uses get_price() BUY/SELL instead of orderbook."""
        mock_client = AsyncMock()
        mock_client.get_price = AsyncMock(side_effect=[
            Decimal("0.42"),  # BUY price (ask)
            Decimal("0.38"),  # SELL price (bid)
        ])

        provider = OrderbookProvider(poly_client=mock_client)
        snap = await provider.get_snapshot("tok1", neg_risk=True)

        assert snap is not None
        assert snap.best_ask == Decimal("0.42")
        assert snap.best_bid == Decimal("0.38")
        assert snap.spread == Decimal("0.04")
        assert snap.is_negrisk is True
        # Should NOT have called get_orderbook
        mock_client.get_orderbook.assert_not_called()
        # Should have called get_price twice
        assert mock_client.get_price.call_count == 2

    @pytest.mark.asyncio
    async def test_standard_uses_orderbook(self) -> None:
        """Standard (non-NegRisk) snapshot uses get_orderbook()."""
        mock_client = AsyncMock()
        book = _make_orderbook(
            "tok1",
            bids=[("0.45", "100")],
            asks=[("0.48", "200")],
        )
        mock_client.get_orderbook = AsyncMock(return_value=book)

        provider = OrderbookProvider(poly_client=mock_client)
        snap = await provider.get_snapshot("tok1", neg_risk=False)

        assert snap is not None
        assert snap.best_bid == Decimal("0.45")
        assert snap.best_ask == Decimal("0.48")
        assert snap.is_negrisk is False

    @pytest.mark.asyncio
    async def test_caching(self) -> None:
        """Second call within TTL returns cached result."""
        mock_client = AsyncMock()
        mock_client.get_price = AsyncMock(side_effect=[
            Decimal("0.42"), Decimal("0.38"),
        ])

        provider = OrderbookProvider(poly_client=mock_client, cache_ttl_s=60)

        snap1 = await provider.get_snapshot("tok1", neg_risk=True)
        snap2 = await provider.get_snapshot("tok1", neg_risk=True)

        assert snap1 is snap2  # Same object from cache
        assert mock_client.get_price.call_count == 2  # Only 1 API round

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self) -> None:
        """API error returns None gracefully."""
        mock_client = AsyncMock()
        mock_client.get_price = AsyncMock(side_effect=Exception("network"))

        provider = OrderbookProvider(poly_client=mock_client)
        snap = await provider.get_snapshot("tok1", neg_risk=True)

        assert snap is None

    @pytest.mark.asyncio
    async def test_zero_price_returns_none(self) -> None:
        """Zero price from NegRisk endpoint returns None."""
        mock_client = AsyncMock()
        mock_client.get_price = AsyncMock(side_effect=[
            Decimal("0"),  # BUY = 0
            Decimal("0"),  # SELL = 0
        ])

        provider = OrderbookProvider(poly_client=mock_client)
        snap = await provider.get_snapshot("tok1", neg_risk=True)

        assert snap is None
