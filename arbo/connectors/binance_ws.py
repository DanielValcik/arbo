"""Binance WebSocket feed for real-time crypto prices.

Subscribes to miniTicker streams for BTC/ETH (and optionally others).
Provides instant price lookups via get_price(). Used by Strategy B2
for live trading edge computation.

Usage:
    feed = BinanceWSFeed(symbols=["BTCUSDT", "ETHUSDT"])
    await feed.start()
    price = feed.get_price("BTCUSDT")  # Returns latest close
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("binance_ws")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]  # Exponential backoff
STALE_THRESHOLD_S = 30.0  # Price considered stale after this


class BinanceWSFeed:
    """Real-time Binance price feed via WebSocket miniTicker stream."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols = [s.lower() for s in (symbols or ["btcusdt", "ethusdt"])]
        self._prices: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and receiving data."""
        if self._ws is None or self._ws.closed:
            return False
        # Check if we've received data recently
        if not self._last_update:
            return False
        latest = max(self._last_update.values())
        return (time.time() - latest) < STALE_THRESHOLD_S

    def get_price(self, symbol: str) -> float | None:
        """Get latest close price for a symbol. Returns None if stale or unavailable."""
        symbol_upper = symbol.upper()
        price = self._prices.get(symbol_upper)
        if price is None:
            return None
        last_ts = self._last_update.get(symbol_upper, 0)
        if time.time() - last_ts > STALE_THRESHOLD_S:
            return None  # Stale
        return price

    def get_prices(self) -> dict[str, float]:
        """Get all cached prices (only fresh ones)."""
        now = time.time()
        return {
            sym: price
            for sym, price in self._prices.items()
            if now - self._last_update.get(sym, 0) < STALE_THRESHOLD_S
        }

    def get_price_age(self, symbol: str) -> float:
        """Get age of price in seconds. Returns inf if no data."""
        symbol_upper = symbol.upper()
        last_ts = self._last_update.get(symbol_upper, 0)
        if last_ts == 0:
            return float("inf")
        return time.time() - last_ts

    async def start(self) -> None:
        """Start the WebSocket connection in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("binance_ws_started", symbols=self._symbols)

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("binance_ws_stopped")

    async def _run_loop(self) -> None:
        """Main connection loop with reconnect logic."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                delay_idx = min(self._reconnect_count, len(RECONNECT_DELAYS) - 1)
                delay = RECONNECT_DELAYS[delay_idx]
                self._reconnect_count += 1
                logger.warning(
                    "binance_ws_reconnecting",
                    error=str(e),
                    delay_s=delay,
                    attempt=self._reconnect_count,
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        """Connect to Binance WS and process messages."""
        # Build combined stream URL
        streams = "/".join(f"{s}@miniTicker" for s in self._symbols)
        url = f"{BINANCE_WS_BASE}/{streams}"

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, heartbeat=20)
            self._reconnect_count = 0
            logger.info("binance_ws_connected", url=url)

            async for msg in self._ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                await self._session.close()

    def _handle_message(self, data: str) -> None:
        """Parse miniTicker message and update price cache."""
        try:
            msg = json.loads(data)
            # miniTicker format: {"e":"24hrMiniTicker","s":"BTCUSDT","c":"87500.00",...}
            symbol = msg.get("s", "")
            close_str = msg.get("c", "")
            if symbol and close_str:
                price = float(close_str)
                self._prices[symbol] = price
                self._last_update[symbol] = time.time()
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
