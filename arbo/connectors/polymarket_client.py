"""Polymarket CLOB client wrapper with production-grade reliability.

Wraps py-clob-client with retry logic, rate limiting, error handling,
and structured logging. All Polymarket API interactions go through this module.

See brief PM-001 for full specification.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    BookParams,
    OpenOrderParams,
    OrderArgs,
    OrderType,
    TradeParams,
)
from py_clob_client.order_builder.constants import BUY, SELL

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("polymarket_client")


# ================================================================
# Exceptions
# ================================================================


class PolymarketError(Exception):
    """Base exception for Polymarket client errors."""


class PolymarketAuthError(PolymarketError):
    """Authentication failed."""


class PolymarketRateLimitError(PolymarketError):
    """Rate limit exceeded."""


class PolymarketNetworkError(PolymarketError):
    """Network connectivity issue."""


# ================================================================
# DTOs
# ================================================================


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderbookEntry:
    """Single price level in an orderbook."""

    price: Decimal
    size: Decimal


@dataclass
class Orderbook:
    """Full orderbook for a token."""

    token_id: str
    bids: list[OrderbookEntry]
    asks: list[OrderbookEntry]
    midpoint: Decimal | None = None
    spread: Decimal | None = None


@dataclass
class MarketPrice:
    """Current price data for a market."""

    token_id: str
    midpoint: Decimal
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None


# ================================================================
# Rate Limiter
# ================================================================


class TokenBucketLimiter:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# ================================================================
# Client
# ================================================================


class PolymarketClient:
    """Production-grade wrapper around py-clob-client.

    Provides:
    - Retry logic with exponential backoff (max 3 retries, base 1s)
    - Rate limiting
    - Typed exceptions
    - Structured logging for every API call
    """

    def __init__(self) -> None:
        config = get_config()
        self._config = config.polymarket
        self._client: ClobClient | None = None
        self._read_limiter = TokenBucketLimiter(rate=100, burst=150)  # GET endpoints
        self._write_limiter = TokenBucketLimiter(rate=200, burst=350)  # POST endpoints
        self._max_retries = self._config.max_retries

    async def initialize(self) -> None:
        """Initialize the CLOB client with L1 + L2 authentication."""
        config = get_config()

        if not config.poly_private_key:
            logger.warning("polymarket_no_credentials", msg="No private key — read-only mode")
            self._client = ClobClient(self._config.clob_url)
            return

        try:
            self._client = ClobClient(
                host=self._config.clob_url,
                chain_id=self._config.chain_id,
                key=config.poly_private_key,
                signature_type=self._config.signature_type,
                funder=config.poly_funder_address or None,
            )

            # Set L2 credentials if available
            if config.poly_api_key and config.poly_secret and config.poly_passphrase:
                self._client.set_api_creds(
                    {
                        "apiKey": config.poly_api_key,
                        "secret": config.poly_secret,
                        "passphrase": config.poly_passphrase,
                    }
                )
                logger.info("polymarket_auth_l2", msg="L2 API credentials set")
            else:
                # Derive L2 credentials
                creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(creds)
                logger.info("polymarket_auth_derived", msg="L2 API credentials derived")

            logger.info("polymarket_initialized", host=self._config.clob_url)

        except Exception as e:
            logger.error("polymarket_init_failed", error=str(e))
            raise PolymarketAuthError(f"Failed to initialize Polymarket client: {e}") from e

    @property
    def client(self) -> ClobClient:
        """Get the underlying ClobClient. Raises if not initialized."""
        if self._client is None:
            raise PolymarketError("Client not initialized. Call initialize() first.")
        return self._client

    async def _retry(self, func: Any, *args: Any, is_write: bool = False, **kwargs: Any) -> Any:
        """Execute a function with retry logic and rate limiting.

        Args:
            func: The function to call.
            is_write: Whether this is a write operation (different rate limit).

        Returns:
            The function result.
        """
        limiter = self._write_limiter if is_write else self._read_limiter
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await limiter.acquire()
            start = time.monotonic()
            try:
                # py-clob-client is synchronous, run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

                elapsed = time.monotonic() - start
                logger.debug(
                    "api_call",
                    func=func.__name__,
                    elapsed_ms=round(elapsed * 1000),
                    attempt=attempt + 1,
                )
                return result

            except Exception as e:
                elapsed = time.monotonic() - start
                last_error = e
                error_str = str(e).lower()

                if "rate limit" in error_str or "429" in error_str:
                    logger.warning(
                        "rate_limited",
                        func=func.__name__,
                        attempt=attempt + 1,
                        elapsed_ms=round(elapsed * 1000),
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    raise PolymarketRateLimitError(str(e)) from e

                if "auth" in error_str or "401" in error_str or "403" in error_str:
                    raise PolymarketAuthError(str(e)) from e

                if attempt < self._max_retries:
                    backoff = min(2**attempt, 30)
                    logger.warning(
                        "api_retry",
                        func=func.__name__,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "api_failed",
                        func=func.__name__,
                        attempts=self._max_retries + 1,
                        error=str(e),
                    )

        raise PolymarketNetworkError(f"Failed after {self._max_retries + 1} attempts: {last_error}")

    # ================================================================
    # Public Read Methods (L0 — no auth needed)
    # ================================================================

    async def get_markets(self, next_cursor: str = "") -> dict[str, Any]:
        """Get paginated market list from CLOB."""
        return await self._retry(self.client.get_markets, next_cursor)

    async def get_simplified_markets(self, next_cursor: str = "") -> dict[str, Any]:
        """Get simplified paginated market list."""
        return await self._retry(self.client.get_simplified_markets, next_cursor)

    async def get_orderbook(self, token_id: str) -> Orderbook:
        """Get full orderbook for a token.

        Args:
            token_id: The CLOB token ID for a specific outcome.

        Returns:
            Orderbook with bids, asks, midpoint, and spread.
        """
        obs = await self._retry(self.client.get_order_book, token_id)

        # py-clob-client returns OrderBookSummary with OrderSummary objects
        bids = [
            OrderbookEntry(price=Decimal(str(b.price)), size=Decimal(str(b.size)))
            for b in (obs.bids or [])
        ]
        asks = [
            OrderbookEntry(price=Decimal(str(a.price)), size=Decimal(str(a.size)))
            for a in (obs.asks or [])
        ]

        midpoint = None
        spread = None
        if bids and asks:
            best_bid = bids[0].price
            best_ask = asks[0].price
            midpoint = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

        return Orderbook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            midpoint=midpoint,
            spread=spread,
        )

    async def get_orderbooks(self, token_ids: list[str]) -> list[Orderbook]:
        """Get orderbooks for multiple tokens in batch."""
        params = [BookParams(token_id=tid) for tid in token_ids]
        obs_list = await self._retry(self.client.get_order_books, params)

        books = []
        for obs in obs_list:
            token_id = obs.asset_id or ""
            bids = [
                OrderbookEntry(price=Decimal(str(b.price)), size=Decimal(str(b.size)))
                for b in (obs.bids or [])
            ]
            asks = [
                OrderbookEntry(price=Decimal(str(a.price)), size=Decimal(str(a.size)))
                for a in (obs.asks or [])
            ]
            midpoint = None
            spread = None
            if bids and asks:
                midpoint = (bids[0].price + asks[0].price) / 2
                spread = asks[0].price - bids[0].price

            books.append(
                Orderbook(token_id=token_id, bids=bids, asks=asks, midpoint=midpoint, spread=spread)
            )
        return books

    async def get_midpoint(self, token_id: str) -> Decimal:
        """Get current midpoint price for a token."""
        raw = await self._retry(self.client.get_midpoint, token_id)
        # py-clob-client returns {"mid": "0.50"}
        if isinstance(raw, dict):
            return Decimal(str(raw["mid"]))
        return Decimal(str(raw))

    async def get_price(self, token_id: str, side: str = "BUY") -> Decimal:
        """Get best available price for a side."""
        raw = await self._retry(self.client.get_price, token_id, side)
        # py-clob-client returns {"price": "0.50"}
        if isinstance(raw, dict):
            return Decimal(str(raw["price"]))
        return Decimal(str(raw))

    async def get_tick_size(self, token_id: str) -> str:
        """Get tick size for a token."""
        return await self._retry(self.client.get_tick_size, token_id)

    async def get_last_trade_price(self, token_id: str) -> Decimal:
        """Get last trade price for a token."""
        raw = await self._retry(self.client.get_last_trade_price, token_id)
        # py-clob-client returns {"price": "0.50"}
        if isinstance(raw, dict):
            return Decimal(str(raw["price"]))
        return Decimal(str(raw))

    # ================================================================
    # Authenticated Methods (L1 + L2)
    # ================================================================

    async def create_and_post_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        tick_size: str = "0.01",
        neg_risk: bool = False,
        order_type: str = "GTC",
    ) -> dict[str, Any]:
        """Create, sign, and post an order in one call.

        Args:
            token_id: CLOB token ID.
            price: Limit price (0-1).
            size: Number of shares.
            side: "BUY" or "SELL".
            tick_size: Market tick size.
            neg_risk: Whether this is a NegRisk market.
            order_type: GTC, GTD, FOK, or FAK.

        Returns:
            Order response from CLOB.
        """
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY if side == "BUY" else SELL,
        )

        options = {"tick_size": tick_size, "neg_risk": neg_risk}

        ot = getattr(OrderType, order_type, OrderType.GTC)

        async def _create_and_post() -> dict[str, Any]:
            signed = self.client.create_order(order_args, options)
            return self.client.post_order(signed, ot)

        result = await self._retry(_create_and_post, is_write=True)

        logger.info(
            "order_posted",
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            order_type=order_type,
        )
        return result

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a specific order."""
        result = await self._retry(self.client.cancel, order_id, is_write=True)
        logger.info("order_cancelled", order_id=order_id)
        return result

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Cancel all open orders. Used for emergency shutdown."""
        result = await self._retry(self.client.cancel_all, is_write=True)
        logger.warning("all_orders_cancelled")
        return result

    async def get_open_orders(self, market: str | None = None) -> list[dict[str, Any]]:
        """Get all open orders, optionally filtered by market."""
        params = OpenOrderParams(market=market) if market else OpenOrderParams()
        return await self._retry(self.client.get_orders, params)

    async def get_trades(self, market: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Get trade history."""
        params = TradeParams(market=market) if market else TradeParams()
        return await self._retry(self.client.get_trades, params)

    async def get_balance_allowance(self) -> dict[str, Any]:
        """Check USDC balance and token allowances."""
        return await self._retry(self.client.get_balance_allowance)

    # ================================================================
    # Health Check
    # ================================================================

    async def health_check(self) -> bool:
        """Verify connectivity to Polymarket CLOB."""
        try:
            result = await self._retry(self.client.get_ok)
            return result == "OK"
        except Exception:
            return False

    async def close(self) -> None:
        """Cleanup resources."""
        logger.info("polymarket_client_closed")
