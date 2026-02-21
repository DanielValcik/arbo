"""Market making bot — Layer 1, Shadow Mode (PM-103).

Simulates market making on Polymarket by reading orderbooks and computing
quotes. SHADOW MODE: no real orders are placed. Only get_orderbook() is called.
Fills are simulated against live orderbook data and tracked internally.

See brief Layer 1 for full specification.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from arbo.config.settings import get_config
from arbo.connectors.market_discovery import GammaMarket  # noqa: TC001
from arbo.connectors.polymarket_client import Orderbook, PolymarketClient  # noqa: TC001
from arbo.utils.logger import get_logger

logger = get_logger("market_maker")


@dataclass
class MMQuote:
    """A market maker quote (bid + ask)."""

    market_condition_id: str
    token_id_yes: str
    token_id_no: str
    bid_price: Decimal
    ask_price: Decimal
    size: Decimal
    spread: Decimal


@dataclass
class MMPosition:
    """Market maker position tracking per market."""

    yes_shares: Decimal = Decimal("0")
    no_shares: Decimal = Decimal("0")
    total_invested: Decimal = Decimal("0")
    simulated_pnl: Decimal = Decimal("0")
    fills: int = 0

    @property
    def imbalance(self) -> Decimal:
        """Inventory imbalance as ratio. 0 = balanced, 1 = fully one-sided."""
        total = self.yes_shares + self.no_shares
        if total == 0:
            return Decimal("0")
        return abs(self.yes_shares - self.no_shares) / total

    @property
    def is_within_limit(self) -> bool:
        """Check if position is within 60/40 imbalance limit."""
        return self.imbalance <= Decimal("0.6")


@dataclass
class MMStats:
    """Aggregate market making statistics."""

    markets_quoted: int = 0
    total_quotes: int = 0
    total_fills: int = 0
    simulated_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    peak_pnl: Decimal = Decimal("0")
    hours_running: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MarketMaker:
    """Layer 1: Market Making Bot (Shadow Mode).

    SHADOW MODE: Only reads orderbook data from PolymarketClient.
    create_and_post_order() is NEVER called.

    Simulates fills against live orderbook and tracks P&L internally.
    """

    def __init__(
        self,
        poly_client: PolymarketClient,
        capital: Decimal = Decimal("2000"),
    ) -> None:
        config = get_config()
        self._poly_client = poly_client
        self._capital = capital
        self._mm_config = config.market_maker
        self._positions: dict[str, MMPosition] = {}
        self._stats = MMStats()
        self._active_markets: list[GammaMarket] = []
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self, markets: list[GammaMarket]) -> None:
        """Start market making on filtered markets.

        Args:
            markets: Candidate markets to filter and quote.
        """
        self._active_markets = self.filter_markets(markets)
        if not self._active_markets:
            logger.warning("mm_no_markets", msg="No markets passed filtering")
            return

        self._running = True
        self._stats = MMStats(markets_quoted=len(self._active_markets))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "mm_started",
            markets=len(self._active_markets),
            heartbeat_s=self._mm_config.heartbeat_interval_s,
        )

    async def stop(self) -> None:
        """Stop market making and log final stats."""
        self._running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        logger.info(
            "mm_stopped",
            total_quotes=self._stats.total_quotes,
            total_fills=self._stats.total_fills,
            simulated_pnl=str(self._stats.simulated_pnl),
            max_drawdown=str(self._stats.max_drawdown),
        )

    async def _heartbeat_loop(self) -> None:
        """Run heartbeat every N seconds."""
        try:
            while self._running:
                await self._heartbeat(self._active_markets)
                await asyncio.sleep(self._mm_config.heartbeat_interval_s)
        except asyncio.CancelledError:
            raise

    async def _heartbeat(self, markets: list[GammaMarket]) -> None:
        """Single heartbeat: compute quotes + simulate fills for each market."""
        for market in markets:
            if not market.token_id_yes:
                continue

            try:
                orderbook = await self._poly_client.get_orderbook(market.token_id_yes)
            except Exception as e:
                logger.debug("mm_orderbook_error", condition_id=market.condition_id, error=str(e))
                continue

            quote = self._compute_quote(market, orderbook)
            if quote is None:
                continue

            self._stats.total_quotes += 1
            self._simulate_fill(quote, orderbook)

    def filter_markets(self, markets: list[GammaMarket]) -> list[GammaMarket]:
        """Filter markets suitable for market making.

        Criteria:
        - Spread > min_spread (default 4%)
        - Volume between min and max (default $1K-$50K)
        - Has valid token IDs
        - Prefer fee-enabled markets (maker rebates)

        Args:
            markets: Candidate markets.

        Returns:
            Filtered markets, sorted by preference.
        """
        candidates: list[GammaMarket] = []
        for m in markets:
            if not m.token_id_yes:
                continue

            spread = m.spread
            if spread is None or spread < Decimal(str(self._mm_config.min_spread)):
                continue

            if m.volume_24h < Decimal(str(self._mm_config.min_volume_24h)):
                continue

            if m.volume_24h > Decimal(str(self._mm_config.max_volume_24h)):
                continue

            candidates.append(m)

        # Sort: fee-enabled first (maker rebates), then by volume
        if self._mm_config.prefer_fee_markets:
            candidates.sort(key=lambda m: (not m.fee_enabled, -float(m.volume_24h)))

        return candidates

    def _compute_quote(
        self,
        market: GammaMarket,
        orderbook: Orderbook,
    ) -> MMQuote | None:
        """Compute bid/ask quotes around the midpoint.

        Uses dynamic spread based on orderbook depth and
        adjusts for inventory skew.
        """
        if orderbook.midpoint is None:
            return None

        midpoint = orderbook.midpoint
        spread = self._dynamic_spread(orderbook)
        half_spread = spread / 2

        bid_price = midpoint - half_spread
        ask_price = midpoint + half_spread

        # Apply inventory skew
        position = self._positions.get(market.condition_id)
        if position is not None:
            bid_price, ask_price = self._skew_for_inventory(bid_price, ask_price, position)

        # Clamp to valid range
        bid_price = max(Decimal("0.01"), min(Decimal("0.99"), bid_price))
        ask_price = max(Decimal("0.01"), min(Decimal("0.99"), ask_price))

        # Ensure bid < ask
        if bid_price >= ask_price:
            return None

        size = self._capital * Decimal(str(self._mm_config.order_size_pct))
        size = size.quantize(Decimal("0.01"))

        if size <= Decimal("0"):
            return None

        return MMQuote(
            market_condition_id=market.condition_id,
            token_id_yes=market.token_id_yes or "",
            token_id_no=market.token_id_no or "",
            bid_price=bid_price.quantize(Decimal("0.0001")),
            ask_price=ask_price.quantize(Decimal("0.0001")),
            size=size,
            spread=ask_price - bid_price,
        )

    def _dynamic_spread(self, orderbook: Orderbook) -> Decimal:
        """Calculate spread based on orderbook depth.

        Thinner books get wider spreads.
        Formula: spread = max(min_spread, base_factor / sqrt(depth))
        """
        min_spread = Decimal(str(self._mm_config.min_spread))
        base_factor = Decimal("0.5")

        # Depth = total volume in top 5 levels on both sides
        bid_depth = sum(e.size for e in orderbook.bids[:5]) if orderbook.bids else Decimal("0")
        ask_depth = sum(e.size for e in orderbook.asks[:5]) if orderbook.asks else Decimal("0")
        total_depth = bid_depth + ask_depth

        if total_depth <= Decimal("0"):
            return min_spread * 2  # Very wide for empty books

        # sqrt(depth) scaling
        depth_float = float(total_depth)
        dynamic = base_factor / Decimal(str(math.sqrt(depth_float)))

        return max(min_spread, dynamic)

    def _skew_for_inventory(
        self,
        bid: Decimal,
        ask: Decimal,
        position: MMPosition,
    ) -> tuple[Decimal, Decimal]:
        """Adjust bid/ask for inventory skew.

        If too much YES: lower bid (buy less YES), raise ask (sell YES more aggressively).
        If too much NO: raise bid (buy more YES), lower ask (sell less YES).
        """
        if position.yes_shares == 0 and position.no_shares == 0:
            return bid, ask

        total = position.yes_shares + position.no_shares
        if total == 0:
            return bid, ask

        # Skew factor: positive if excess YES, negative if excess NO
        yes_ratio = position.yes_shares / total
        skew = (yes_ratio - Decimal("0.5")) * Decimal("0.02")  # Max 1% skew

        return bid - skew, ask - skew

    def _simulate_fill(self, quote: MMQuote, orderbook: Orderbook) -> None:
        """Simulate fill against live orderbook.

        In shadow mode, our quotes are tighter than the book's spread.
        If our bid improves on the best bid, incoming sellers would fill us.
        If our ask improves on the best ask, incoming buyers would fill us.
        """
        condition_id = quote.market_condition_id
        if condition_id not in self._positions:
            self._positions[condition_id] = MMPosition()

        position = self._positions[condition_id]

        # Sell fill: our ask < book's best ask → we're offering a better price to buyers
        if orderbook.asks:
            best_ask = orderbook.asks[0].price
            if quote.ask_price < best_ask:
                shares = (quote.size / quote.ask_price).quantize(Decimal("0.01"))
                mid = orderbook.midpoint or quote.ask_price
                pnl = shares * (quote.ask_price - mid)
                position.no_shares += shares
                position.total_invested += quote.size
                position.simulated_pnl += pnl
                position.fills += 1
                self._stats.total_fills += 1

                logger.debug(
                    "mm_fill_sell",
                    condition_id=condition_id,
                    price=str(quote.ask_price),
                    pnl=str(pnl),
                )

        # Buy fill: our bid > book's best bid → we're offering a better price to sellers
        if orderbook.bids:
            best_bid = orderbook.bids[0].price
            if quote.bid_price > best_bid:
                shares = (quote.size / quote.bid_price).quantize(Decimal("0.01"))
                mid = orderbook.midpoint or quote.bid_price
                pnl = shares * (mid - quote.bid_price)
                position.yes_shares += shares
                position.total_invested += quote.size
                position.simulated_pnl += pnl
                position.fills += 1
                self._stats.total_fills += 1

                logger.debug(
                    "mm_fill_buy",
                    condition_id=condition_id,
                    price=str(quote.bid_price),
                    pnl=str(pnl),
                )

        # Update aggregate stats
        self._stats.simulated_pnl = sum(p.simulated_pnl for p in self._positions.values())
        self._stats.peak_pnl = max(self._stats.peak_pnl, self._stats.simulated_pnl)
        drawdown = self._stats.peak_pnl - self._stats.simulated_pnl
        self._stats.max_drawdown = max(self._stats.max_drawdown, drawdown)

    def get_position(self, condition_id: str) -> MMPosition | None:
        """Get position for a market."""
        return self._positions.get(condition_id)

    @property
    def stats(self) -> MMStats:
        """Get current statistics."""
        elapsed = (datetime.now(UTC) - self._stats.started_at).total_seconds()
        self._stats.hours_running = elapsed / 3600
        return self._stats
