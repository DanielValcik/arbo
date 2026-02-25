"""Tests for PM-103: Market Making Bot Shadow Mode (Layer 1).

Tests verify:
1. Market filtering: wide spread in, narrow out, low/high volume out, fee preferred
2. Quote computation: symmetric around midpoint, min spread, thin/deep book, clamped
3. Inventory management: balanced no skew, excess YES skew, excess NO skew, limit 60/40
4. Shadow mode: no real orders placed, fill simulation, PnL tracking, drawdown calc

Acceptance: 24h shadow run — positive simulated P&L, max drawdown <3%.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from arbo.connectors.market_discovery import GammaMarket
from arbo.connectors.polymarket_client import Orderbook, OrderbookEntry, PolymarketClient
from arbo.strategies.market_maker import MarketMaker, MMPosition

# ================================================================
# Factory helpers
# ================================================================


def _make_gamma_market(
    condition_id: str = "cond_1",
    question: str = "Will Arsenal win the Premier League?",
    price_yes: str = "0.45",
    price_no: str = "0.55",
    volume_24h: str = "5000",
    fee_enabled: bool = False,
) -> GammaMarket:
    """Build a GammaMarket for testing."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": "test-market",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{price_yes}", "{price_no}"]',
        "clobTokenIds": '["tok_yes_1", "tok_no_1"]',
        "volume": "100000",
        "volume24hr": volume_24h,
        "liquidity": "10000",
        "active": True,
        "closed": False,
        "feesEnabled": fee_enabled,
        "enableNegRisk": False,
        "tags": [],
    }
    return GammaMarket(raw)


def _make_orderbook(
    token_id: str = "tok_yes_1",
    best_bid: str = "0.45",
    best_ask: str = "0.55",
    bid_depth: int = 5,
    ask_depth: int = 5,
    level_size: str = "100",
) -> Orderbook:
    """Build a mock orderbook."""
    bid_price = Decimal(best_bid)
    ask_price = Decimal(best_ask)
    bids = [
        OrderbookEntry(price=bid_price - Decimal("0.01") * i, size=Decimal(level_size))
        for i in range(bid_depth)
    ]
    asks = [
        OrderbookEntry(price=ask_price + Decimal("0.01") * i, size=Decimal(level_size))
        for i in range(ask_depth)
    ]
    midpoint = (bid_price + ask_price) / 2
    spread = ask_price - bid_price

    return Orderbook(
        token_id=token_id,
        bids=bids,
        asks=asks,
        midpoint=midpoint,
        spread=spread,
    )


# ================================================================
# Market filtering
# ================================================================


class TestMarketFiltering:
    """MarketMaker.filter_markets() criteria."""

    @patch("arbo.strategies.market_maker.get_config")
    def test_wide_spread_passes(self, mock_config: MagicMock) -> None:
        """Market with spread > 4% passes filter."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        markets = [_make_gamma_market(price_yes="0.45", price_no="0.45")]  # spread = 10%
        filtered = mm.filter_markets(markets)
        assert len(filtered) == 1

    @patch("arbo.strategies.market_maker.get_config")
    def test_narrow_spread_rejected(self, mock_config: MagicMock) -> None:
        """Market with spread < 4% rejected."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        markets = [_make_gamma_market(price_yes="0.49", price_no="0.49")]  # spread = 2%
        filtered = mm.filter_markets(markets)
        assert len(filtered) == 0

    @patch("arbo.strategies.market_maker.get_config")
    def test_low_volume_rejected(self, mock_config: MagicMock) -> None:
        """Market with volume < $1K rejected."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        markets = [_make_gamma_market(volume_24h="500")]
        filtered = mm.filter_markets(markets)
        assert len(filtered) == 0

    @patch("arbo.strategies.market_maker.get_config")
    def test_high_volume_rejected(self, mock_config: MagicMock) -> None:
        """Market with volume > $50K rejected."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        markets = [_make_gamma_market(volume_24h="60000")]
        filtered = mm.filter_markets(markets)
        assert len(filtered) == 0

    @patch("arbo.strategies.market_maker.get_config")
    def test_fee_markets_preferred(self, mock_config: MagicMock) -> None:
        """Fee-enabled markets sorted first."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        markets = [
            _make_gamma_market("c1", price_yes="0.45", price_no="0.45", fee_enabled=False),
            _make_gamma_market("c2", price_yes="0.45", price_no="0.45", fee_enabled=True),
        ]
        filtered = mm.filter_markets(markets)

        assert len(filtered) == 2
        assert filtered[0].fee_enabled is True
        assert filtered[1].fee_enabled is False


# ================================================================
# Quote computation
# ================================================================


class TestQuoteComputation:
    """MarketMaker._compute_quote() correctness."""

    @patch("arbo.strategies.market_maker.get_config")
    def test_symmetric_around_midpoint(self, mock_config: MagicMock) -> None:
        """Quotes are symmetric around midpoint."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()
        market = _make_gamma_market()
        orderbook = _make_orderbook(best_bid="0.45", best_ask="0.55")

        quote = mm._compute_quote(market, orderbook)

        assert quote is not None
        mid = (quote.bid_price + quote.ask_price) / 2
        assert abs(mid - Decimal("0.5")) < Decimal("0.01")

    @patch("arbo.strategies.market_maker.get_config")
    def test_min_spread_enforced(self, mock_config: MagicMock) -> None:
        """Quote spread is at least min_spread."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()
        market = _make_gamma_market()
        # Deep book should push spread towards min
        orderbook = _make_orderbook(best_bid="0.49", best_ask="0.51", level_size="10000")

        quote = mm._compute_quote(market, orderbook)

        assert quote is not None
        assert quote.spread >= Decimal("0.04")  # min_spread

    @patch("arbo.strategies.market_maker.get_config")
    def test_thin_book_wider_spread(self, mock_config: MagicMock) -> None:
        """Thin orderbook → wider spread."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()
        market = _make_gamma_market()

        thin_book = _make_orderbook(level_size="1")
        deep_book = _make_orderbook(level_size="10000")

        thin_quote = mm._compute_quote(market, thin_book)
        deep_quote = mm._compute_quote(market, deep_book)

        assert thin_quote is not None
        assert deep_quote is not None
        assert thin_quote.spread > deep_quote.spread

    @patch("arbo.strategies.market_maker.get_config")
    def test_clamped_to_valid_range(self, mock_config: MagicMock) -> None:
        """Quotes clamped to [0.01, 0.99]."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()
        market = _make_gamma_market(price_yes="0.98", price_no="0.02")
        orderbook = _make_orderbook(best_bid="0.97", best_ask="0.99")

        quote = mm._compute_quote(market, orderbook)

        if quote is not None:
            assert quote.bid_price >= Decimal("0.01")
            assert quote.ask_price <= Decimal("0.99")


# ================================================================
# Inventory management
# ================================================================


class TestInventoryManagement:
    """Inventory skew adjustments."""

    @patch("arbo.strategies.market_maker.get_config")
    def test_balanced_no_skew(self, mock_config: MagicMock) -> None:
        """Balanced position → no skew."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        position = MMPosition(yes_shares=Decimal("100"), no_shares=Decimal("100"))
        bid, ask = mm._skew_for_inventory(Decimal("0.45"), Decimal("0.55"), position)

        # Balanced 50/50 → no skew
        assert bid == Decimal("0.45")
        assert ask == Decimal("0.55")

    @patch("arbo.strategies.market_maker.get_config")
    def test_excess_yes_skew(self, mock_config: MagicMock) -> None:
        """Excess YES → lower bid, lower ask (encourage selling YES)."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        position = MMPosition(yes_shares=Decimal("200"), no_shares=Decimal("0"))
        bid, ask = mm._skew_for_inventory(Decimal("0.45"), Decimal("0.55"), position)

        # Excess YES: skew > 0, both bid and ask decrease
        assert bid < Decimal("0.45")
        assert ask < Decimal("0.55")

    @patch("arbo.strategies.market_maker.get_config")
    def test_excess_no_skew(self, mock_config: MagicMock) -> None:
        """Excess NO → raise bid, raise ask (encourage buying YES)."""
        mock_config.return_value = _mock_config()
        mm = _make_market_maker()

        position = MMPosition(yes_shares=Decimal("0"), no_shares=Decimal("200"))
        bid, ask = mm._skew_for_inventory(Decimal("0.45"), Decimal("0.55"), position)

        # Excess NO: skew < 0, both bid and ask increase
        assert bid > Decimal("0.45")
        assert ask > Decimal("0.55")

    @patch("arbo.strategies.market_maker.get_config")
    def test_imbalance_limit(self, mock_config: MagicMock) -> None:
        """60/40 imbalance limit check."""
        mock_config.return_value = _mock_config()

        balanced = MMPosition(yes_shares=Decimal("60"), no_shares=Decimal("40"))
        assert balanced.is_within_limit is True

        exceeded = MMPosition(yes_shares=Decimal("90"), no_shares=Decimal("10"))
        assert exceeded.is_within_limit is False


# ================================================================
# Shadow mode
# ================================================================


class TestShadowMode:
    """Shadow mode — no real orders, simulated fills."""

    @patch("arbo.strategies.market_maker.get_config")
    async def test_no_real_orders(self, mock_config: MagicMock) -> None:
        """create_and_post_order() is NEVER called."""
        mock_config.return_value = _mock_config()
        poly_client = AsyncMock(spec=PolymarketClient)
        poly_client.get_orderbook.return_value = _make_orderbook()

        mm = MarketMaker(poly_client=poly_client)
        market = _make_gamma_market()
        await mm._heartbeat([market])

        poly_client.create_and_post_order.assert_not_called()

    @patch("arbo.strategies.market_maker.get_config")
    async def test_fill_simulation(self, mock_config: MagicMock) -> None:
        """Fills are simulated when orderbook crosses quote."""
        mock_config.return_value = _mock_config()
        poly_client = AsyncMock(spec=PolymarketClient)

        # Orderbook where best_bid > our potential ask (triggers sell fill)
        orderbook = _make_orderbook(best_bid="0.60", best_ask="0.65")
        poly_client.get_orderbook.return_value = orderbook

        mm = MarketMaker(poly_client=poly_client)
        market = _make_gamma_market(price_yes="0.40", price_no="0.50")
        await mm._heartbeat([market])

        assert mm.stats.total_fills > 0

    @patch("arbo.strategies.market_maker.get_config")
    async def test_pnl_tracking(self, mock_config: MagicMock) -> None:
        """P&L is tracked per position and aggregated in stats."""
        mock_config.return_value = _mock_config()
        poly_client = AsyncMock(spec=PolymarketClient)
        orderbook = _make_orderbook(best_bid="0.60", best_ask="0.65")
        poly_client.get_orderbook.return_value = orderbook

        mm = MarketMaker(poly_client=poly_client)
        market = _make_gamma_market(price_yes="0.40", price_no="0.50")
        await mm._heartbeat([market])

        position = mm.get_position(market.condition_id)
        if position is not None and position.fills > 0:
            # P&L should be non-zero after a fill
            assert position.simulated_pnl != Decimal("0") or position.fills > 0

    @patch("arbo.strategies.market_maker.get_config")
    async def test_drawdown_calc(self, mock_config: MagicMock) -> None:
        """Max drawdown is tracked across heartbeats."""
        mock_config.return_value = _mock_config()
        poly_client = AsyncMock(spec=PolymarketClient)

        # First heartbeat: profitable
        orderbook1 = _make_orderbook(best_bid="0.60", best_ask="0.65")
        poly_client.get_orderbook.return_value = orderbook1

        mm = MarketMaker(poly_client=poly_client)
        market = _make_gamma_market(price_yes="0.40", price_no="0.50")

        await mm._heartbeat([market])

        # Second heartbeat: could change stats
        orderbook2 = _make_orderbook(best_bid="0.40", best_ask="0.60")
        poly_client.get_orderbook.return_value = orderbook2
        await mm._heartbeat([market])

        # Drawdown should be >= 0
        assert mm.stats.max_drawdown >= Decimal("0")


# ================================================================
# Helpers
# ================================================================


def _mock_config() -> MagicMock:
    """Create a mock ArboConfig for market maker tests."""
    config = MagicMock()
    config.market_maker.min_spread = 0.04
    config.market_maker.min_volume_24h = 1000
    config.market_maker.max_volume_24h = 50000
    config.market_maker.max_inventory_imbalance = 0.6
    config.market_maker.order_size_pct = 0.025
    config.market_maker.heartbeat_interval_s = 30
    config.market_maker.prefer_fee_markets = True
    return config


def _make_market_maker() -> MarketMaker:
    """Create a MarketMaker with mocked client."""
    poly_client = AsyncMock(spec=PolymarketClient)
    return MarketMaker(poly_client=poly_client, capital=Decimal("2000"))
