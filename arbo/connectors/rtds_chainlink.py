"""RTDS Chainlink — Real-time Chainlink resolution prices from Polymarket.

Connects to Polymarket's RTDS WebSocket and subscribes to the
crypto_prices_chainlink topic. This provides the EXACT price feed
that Polymarket uses for resolving 5-min Up/Down markets.

Usage:
    feed = RTDSChainlinkFeed()
    await feed.start()
    price = feed.get_price("btc/usd")  # Returns latest Chainlink price
    binance = feed.get_binance_price("btcusdt")  # Binance price via same WS
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

import aiohttp

from arbo.utils.logger import get_logger

logger = get_logger("rtds_chainlink")

RTDS_URL = "wss://ws-live-data.polymarket.com"
PING_INTERVAL_S = 5
STALE_THRESHOLD_S = 30.0
RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]


class RTDSChainlinkFeed:
    """Real-time Chainlink + Binance price feed from Polymarket RTDS."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._reconnect_count = 0

        # Chainlink prices (resolution source)
        self._chainlink_prices: dict[str, float] = {}
        self._chainlink_ts: dict[str, float] = {}

        # Binance prices (fast signal, via same RTDS WS)
        self._binance_prices: dict[str, float] = {}
        self._binance_ts: dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and receiving data."""
        if self._ws is None or self._ws.closed:
            return False
        if not self._chainlink_ts:
            return False
        latest = max(self._chainlink_ts.values())
        return (time.time() - latest) < STALE_THRESHOLD_S

    def get_price(self, symbol: str) -> float | None:
        """Get latest Chainlink resolution price. Returns None if stale."""
        price = self._chainlink_prices.get(symbol)
        if price is None:
            return None
        last = self._chainlink_ts.get(symbol, 0)
        if time.time() - last > STALE_THRESHOLD_S:
            return None
        return price

    def get_binance_price(self, symbol: str) -> float | None:
        """Get latest Binance price from RTDS. Returns None if stale."""
        price = self._binance_prices.get(symbol)
        if price is None:
            return None
        last = self._binance_ts.get(symbol, 0)
        if time.time() - last > STALE_THRESHOLD_S:
            return None
        return price

    def get_both(self, chainlink_sym: str = "btc/usd", binance_sym: str = "btcusdt") -> tuple[float | None, float | None]:
        """Get both Chainlink and Binance prices for comparison."""
        return self.get_price(chainlink_sym), self.get_binance_price(binance_sym)

    async def start(self) -> None:
        """Start the RTDS WebSocket connection."""
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._task = asyncio.create_task(self._run_loop(), name="rtds_chainlink")
        logger.info("rtds_chainlink_starting")

    async def stop(self) -> None:
        """Stop the RTDS WebSocket connection."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("rtds_chainlink_stopped")

    async def _run_loop(self) -> None:
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                delay_idx = min(self._reconnect_count, len(RECONNECT_DELAYS) - 1)
                delay = RECONNECT_DELAYS[delay_idx]
                self._reconnect_count += 1
                logger.warning(
                    "rtds_reconnecting",
                    error=str(e)[:100],
                    delay=delay,
                    attempt=self._reconnect_count,
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        """Connect to RTDS, subscribe, and process messages."""
        if not self._session:
            return

        async with self._session.ws_connect(RTDS_URL) as ws:
            self._ws = ws
            self._reconnect_count = 0
            logger.info("rtds_connected")

            # Subscribe to both Chainlink (resolution) and Binance (signal)
            sub_msg = {
                "action": "subscribe",
                "subscriptions": [
                    {"topic": "crypto_prices_chainlink", "type": "*", "filters": ""},
                    {"topic": "crypto_prices", "type": "*", "filters": ""},
                ],
            }
            await ws.send_json(sub_msg)

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws))

            try:
                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if msg.data in ("pong", ""):
                            continue
                        with contextlib.suppress(json.JSONDecodeError, ValueError):
                            self._process_message(json.loads(msg.data))
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
            finally:
                ping_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ping_task

    def _process_message(self, data: dict) -> None:
        """Process a single RTDS message."""
        topic = data.get("topic", "")
        payload = data.get("payload")
        if not payload:
            return

        symbol = payload.get("symbol", "")
        value = payload.get("value")
        if not symbol or value is None:
            return

        now = time.time()
        price = float(value)

        if topic == "crypto_prices_chainlink":
            self._chainlink_prices[symbol] = price
            self._chainlink_ts[symbol] = now
        elif topic == "crypto_prices":
            self._binance_prices[symbol] = price
            self._binance_ts[symbol] = now

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send text 'ping' every 5 seconds (RTDS protocol requirement)."""
        try:
            while not ws.closed:
                await asyncio.sleep(PING_INTERVAL_S)
                if not ws.closed:
                    await ws.send_str("ping")
        except asyncio.CancelledError:
            pass
