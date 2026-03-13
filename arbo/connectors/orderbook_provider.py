"""NegRisk-aware orderbook provider for realistic paper trading.

Polymarket weather markets use NegRisk (multi-outcome) architecture.
Individual token orderbooks (GET /book?token_id=...) show only dust
(bid=0.001, ask=0.999) for NegRisk markets. Real prices live in:
- GET /price?token_id=...&side=BUY  → executable buy price
- GET /price?token_id=...&side=SELL → executable sell price
- GET /midpoint?token_id=...        → midpoint

This module detects NegRisk markets and uses the correct price endpoints.
For non-NegRisk markets, it uses the actual orderbook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from arbo.utils.logger import get_logger

if TYPE_CHECKING:
    from arbo.connectors.polymarket_client import PolymarketClient

logger = get_logger("orderbook_provider")


@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook state for a single token."""

    token_id: str
    best_bid: Decimal | None
    best_ask: Decimal | None
    midpoint: Decimal | None
    spread: Decimal | None
    bid_depth_usdc: Decimal
    ask_depth_usdc: Decimal
    bids: list[tuple[Decimal, Decimal]]  # (price, size) pairs
    asks: list[tuple[Decimal, Decimal]]  # (price, size) pairs
    fetched_at: float  # monotonic timestamp
    is_negrisk: bool = False


class OrderbookProvider:
    """Provides orderbook snapshots with NegRisk awareness.

    For NegRisk markets: uses CLOB /price endpoint (BUY/SELL) to get
    real executable prices. Constructs a synthetic 1-level orderbook.

    For standard markets: uses full orderbook from /book endpoint.

    Features:
    - TTL-based caching (default 30s)
    - Batch fetching with concurrency limit
    - Graceful fallback on API errors
    """

    def __init__(
        self,
        poly_client: PolymarketClient | None = None,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._client = poly_client
        self._cache_ttl = cache_ttl_s
        self._cache: dict[str, OrderbookSnapshot] = {}

    async def get_snapshot(
        self,
        token_id: str,
        neg_risk: bool = True,
    ) -> OrderbookSnapshot | None:
        """Get orderbook snapshot for a token.

        Args:
            token_id: CLOB token ID.
            neg_risk: Whether this is a NegRisk market.

        Returns:
            OrderbookSnapshot or None on error.
        """
        if self._client is None:
            return None

        # Check cache
        cached = self._cache.get(token_id)
        if cached is not None:
            age = time.monotonic() - cached.fetched_at
            if age < self._cache_ttl:
                return cached

        try:
            if neg_risk:
                snap = await self._fetch_negrisk_prices(token_id)
            else:
                snap = await self._fetch_orderbook(token_id)

            if snap is not None:
                self._cache[token_id] = snap
            return snap

        except Exception as e:
            logger.warning(
                "orderbook_fetch_error",
                token_id=token_id[:20],
                neg_risk=neg_risk,
                error=str(e),
            )
            return None

    async def get_snapshots_batch(
        self,
        token_ids: list[str],
        neg_risk: bool = True,
    ) -> dict[str, OrderbookSnapshot]:
        """Get orderbook snapshots for multiple tokens.

        Args:
            token_ids: List of CLOB token IDs.
            neg_risk: Whether these are NegRisk markets.

        Returns:
            Dict mapping token_id → OrderbookSnapshot.
        """
        if self._client is None:
            return {}

        result: dict[str, OrderbookSnapshot] = {}
        for token_id in token_ids:
            snap = await self.get_snapshot(token_id, neg_risk=neg_risk)
            if snap is not None:
                result[token_id] = snap

        return result

    async def _fetch_negrisk_prices(self, token_id: str) -> OrderbookSnapshot | None:
        """Fetch real prices for a NegRisk token via /price endpoint.

        The CLOB /price endpoint returns the real executable price for
        NegRisk markets, unlike /book which shows dust orders.

        BUY price = what you'd pay to buy (effective ask)
        SELL price = what you'd receive selling (effective bid)
        """
        assert self._client is not None

        buy_price = await self._client.get_price(token_id, "BUY")
        sell_price = await self._client.get_price(token_id, "SELL")

        # Validate prices are real (not dust)
        if buy_price <= Decimal("0") or sell_price <= Decimal("0"):
            logger.debug(
                "negrisk_zero_price",
                token_id=token_id[:20],
                buy=str(buy_price),
                sell=str(sell_price),
            )
            return None

        midpoint = (buy_price + sell_price) / 2
        spread = buy_price - sell_price  # BUY > SELL for normal markets

        # Construct synthetic orderbook with 1 level
        # BUY price = effective ask (what buyer pays)
        # SELL price = effective bid (what seller receives)
        # Synthetic depth: we don't know real depth, use conservative estimate
        synthetic_depth = Decimal("500")  # $500 per side (conservative)
        bids = [(sell_price, synthetic_depth / sell_price if sell_price > 0 else Decimal("0"))]
        asks = [(buy_price, synthetic_depth / buy_price if buy_price > 0 else Decimal("0"))]

        return OrderbookSnapshot(
            token_id=token_id,
            best_bid=sell_price,
            best_ask=buy_price,
            midpoint=midpoint,
            spread=spread,
            bid_depth_usdc=synthetic_depth,
            ask_depth_usdc=synthetic_depth,
            bids=bids,
            asks=asks,
            fetched_at=time.monotonic(),
            is_negrisk=True,
        )

    async def _fetch_orderbook(self, token_id: str) -> OrderbookSnapshot | None:
        """Fetch full orderbook for a standard (non-NegRisk) token."""
        assert self._client is not None

        book = await self._client.get_orderbook(token_id)

        bids = [(e.price, e.size) for e in book.bids]
        asks = [(e.price, e.size) for e in book.asks]

        bid_depth = sum(p * s for p, s in bids)
        ask_depth = sum(p * s for p, s in asks)

        return OrderbookSnapshot(
            token_id=token_id,
            best_bid=bids[0][0] if bids else None,
            best_ask=asks[0][0] if asks else None,
            midpoint=book.midpoint,
            spread=book.spread,
            bid_depth_usdc=bid_depth,
            ask_depth_usdc=ask_depth,
            bids=bids,
            asks=asks,
            fetched_at=time.monotonic(),
            is_negrisk=False,
        )


def available_depth(snapshot: OrderbookSnapshot, side: str) -> Decimal:
    """Get available depth in USDC for a given side.

    Args:
        snapshot: OrderbookSnapshot.
        side: "BUY" or "SELL".

    Returns:
        Available depth in USDC.
    """
    if side == "BUY":
        return snapshot.ask_depth_usdc
    return snapshot.bid_depth_usdc


def estimate_fill_price(
    snapshot: OrderbookSnapshot,
    side: str,
    size_usdc: Decimal,
) -> Decimal | None:
    """Estimate fill price by walking the orderbook (VWAP).

    For NegRisk markets (synthetic 1-level book), returns the
    best_ask (BUY) or best_bid (SELL) directly — this IS the
    real executable price from the CLOB /price endpoint.

    For standard markets, walks multiple levels for VWAP.

    Args:
        snapshot: OrderbookSnapshot.
        side: "BUY" or "SELL".
        size_usdc: Trade size in USDC.

    Returns:
        Estimated fill price, or None if book is empty / size is 0.
    """
    if size_usdc <= 0:
        return None

    levels = snapshot.asks if side == "BUY" else snapshot.bids
    if not levels:
        return None

    # For NegRisk synthetic books, return the price directly
    if snapshot.is_negrisk:
        if side == "BUY":
            return snapshot.best_ask
        return snapshot.best_bid

    # Walk the book for standard orderbooks
    total_cost = Decimal("0")
    total_shares = Decimal("0")
    remaining = size_usdc

    for price, size in levels:
        level_value = price * size  # USDC value at this level
        if level_value >= remaining:
            shares_at_level = remaining / price
            total_cost += remaining
            total_shares += shares_at_level
            remaining = Decimal("0")
            break
        else:
            total_cost += level_value
            total_shares += size
            remaining -= level_value

    if total_shares <= 0:
        return None

    # VWAP = total cost / total shares
    vwap = (total_cost / total_shares).quantize(Decimal("0.0001"))
    return vwap
