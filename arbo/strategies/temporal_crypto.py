"""Temporal crypto scanner — Layer 6 (PM-106).

Connects to Binance WebSocket for real-time spot prices, compares against
Polymarket 15-minute crypto resolution markets to find divergences.

Uses existing MarketDiscovery.get_crypto_15min_markets() for market discovery.

See brief Layer 6 for full specification.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import aiohttp

from arbo.config.settings import get_config
from arbo.connectors.market_discovery import GammaMarket, MarketDiscovery  # noqa: TC001
from arbo.core.scanner import Signal, SignalDirection
from arbo.utils.logger import get_logger

logger = get_logger("temporal_crypto")

# Symbol extraction keyword map
SYMBOL_MAP: dict[str, str] = {
    "bitcoin": "btcusdt",
    "btc": "btcusdt",
    "ethereum": "ethusdt",
    "eth": "ethusdt",
    "solana": "solusdt",
    "sol": "solusdt",
    "dogecoin": "dogeusdt",
    "doge": "dogeusdt",
}

# Regex patterns for strike price extraction
_STRIKE_PATTERNS = [
    re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)"),  # $95,000 or $95,000.50
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*k", re.IGNORECASE),  # $95k or $95.5k
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)"),  # $95000 or $95000.50
]

BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"


@dataclass
class SpotPrice:
    """Real-time spot price from exchange."""

    symbol: str
    price: Decimal
    source: str
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BinanceWSFeed:
    """WebSocket feed for Binance spot prices.

    Connects to Binance combined stream for real-time ticker data.
    Stale prices (>60s) are filtered out automatically.
    """

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols = symbols or ["btcusdt", "ethusdt", "solusdt"]
        self._prices: dict[str, SpotPrice] = {}
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task[None] | None = None
        self._connected = False
        self._stale_threshold_s = 60.0

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is connected."""
        return self._connected

    async def start(self) -> None:
        """Connect to Binance combined stream."""
        streams = "/".join(f"{s}@ticker" for s in self._symbols)
        url = f"{BINANCE_WS_URL}?streams={streams}"

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, heartbeat=30)
            self._connected = True
            self._task = asyncio.create_task(self._read_loop())
            logger.info("binance_ws_connected", symbols=self._symbols)
        except Exception as e:
            logger.error("binance_ws_connect_failed", error=str(e))
            self._connected = False

    async def stop(self) -> None:
        """Disconnect from Binance."""
        self._connected = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("binance_ws_disconnected")

    def get_price(self, symbol: str) -> SpotPrice | None:
        """Get latest price for a symbol. Returns None if stale (>60s)."""
        price = self._prices.get(symbol.lower())
        if price is None:
            return None

        age = (datetime.now(UTC) - price.received_at).total_seconds()
        if age > self._stale_threshold_s:
            return None

        return price

    async def _read_loop(self) -> None:
        """Read messages from WebSocket."""
        if self._ws is None:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("binance_ws_read_error", error=str(e))
        finally:
            self._connected = False

    def _handle_message(self, raw: str) -> None:
        """Parse a Binance combined stream ticker message.

        Combined stream format: {"stream": "btcusdt@ticker", "data": {...}}
        Ticker data includes: "s" (symbol), "c" (close/last price).
        """
        try:
            data = json.loads(raw)
            # Combined stream format
            ticker = data.get("data", data)

            symbol = ticker.get("s", "").lower()
            last_price = ticker.get("c")  # Close price (last price)

            if symbol and last_price:
                self._prices[symbol] = SpotPrice(
                    symbol=symbol,
                    price=Decimal(str(last_price)),
                    source="binance",
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("binance_parse_error", error=str(e))


def extract_symbol(question: str) -> str | None:
    """Extract trading symbol from market question.

    Args:
        question: Market question like "Bitcoin 15 min price >= $95,000?"

    Returns:
        Binance symbol like "btcusdt" or None if not recognized.
    """
    q_lower = question.lower()
    for keyword, symbol in SYMBOL_MAP.items():
        if keyword in q_lower:
            return symbol
    return None


def extract_strike_price(question: str) -> Decimal | None:
    """Extract strike price from market question.

    Handles formats: $95,000 | $95k | $95.5k | $95000

    Args:
        question: Market question text.

    Returns:
        Strike price as Decimal or None if not found.
    """
    for pattern in _STRIKE_PATTERNS:
        match = pattern.search(question)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                value = Decimal(value_str)
                # Handle k notation — check original match region for 'k'
                full_match = question[match.start() : match.end() + 2]
                if re.search(r"k\b", full_match, re.IGNORECASE):
                    value *= 1000
                return value
            except Exception:
                continue
    return None


class TemporalCryptoScanner:
    """Layer 6: Temporal crypto arbitrage scanner.

    Compares real-time spot prices from Binance against Polymarket
    15-minute crypto resolution markets to find pricing divergences.

    Strategy: if spot price strongly implies YES/NO but Polymarket
    contract hasn't moved, buy the underpriced side.
    """

    def __init__(
        self,
        discovery: MarketDiscovery,
        spot_feed: BinanceWSFeed | None = None,
    ) -> None:
        config = get_config()
        self._discovery = discovery
        self._spot_feed = spot_feed or BinanceWSFeed()
        self._deviation_threshold = Decimal(str(config.temporal_crypto.price_deviation_threshold))
        self._max_trades_per_hour = config.temporal_crypto.max_trades_per_hour
        self._use_postonly = config.temporal_crypto.use_postonly
        self._trades_this_hour: list[float] = []
        self._scan_count = 0

    async def initialize(self) -> None:
        """Start spot price feed."""
        await self._spot_feed.start()

    async def close(self) -> None:
        """Stop spot price feed."""
        await self._spot_feed.stop()

    async def scan(self) -> list[Signal]:
        """Scan for temporal crypto arbitrage opportunities.

        1. Get 15-min crypto markets from discovery
        2. For each: extract symbol + strike, get spot price
        3. Compute divergence between implied prob and contract price
        4. Emit signal if divergence > threshold

        Returns:
            List of Layer 6 signals.
        """
        self._scan_count += 1
        self._prune_trade_timestamps()

        markets = self._discovery.get_crypto_15min_markets()
        if not markets:
            logger.debug("temporal_no_crypto_markets")
            return []

        signals: list[Signal] = []
        for market in markets:
            signal = self._evaluate_market(market)
            if signal is not None:
                if len(self._trades_this_hour) < self._max_trades_per_hour:
                    signals.append(signal)
                    self._trades_this_hour.append(time.monotonic())
                else:
                    logger.debug("temporal_rate_limited", max=self._max_trades_per_hour)
                    break

        logger.info(
            "temporal_scan_complete",
            scan_number=self._scan_count,
            crypto_markets=len(markets),
            signals=len(signals),
        )

        return signals

    def _evaluate_market(self, market: GammaMarket) -> Signal | None:
        """Evaluate a single crypto market for divergence."""
        symbol = extract_symbol(market.question)
        if symbol is None:
            return None

        strike = extract_strike_price(market.question)
        if strike is None:
            return None

        spot = self._spot_feed.get_price(symbol)
        if spot is None:
            return None

        if market.price_yes is None or not market.token_id_yes:
            return None

        contract_price = market.price_yes
        divergence = self._compute_divergence(spot.price, strike, contract_price)

        if abs(divergence) < self._deviation_threshold:
            return None

        # If spot > strike, implied YES prob should be high
        # If contract is still low, BUY YES
        if divergence > 0:
            direction = SignalDirection.BUY_YES
            token_id = market.token_id_yes
        else:
            direction = SignalDirection.BUY_NO
            token_id = market.token_id_no or market.token_id_yes

        return Signal(
            layer=6,
            market_condition_id=market.condition_id,
            token_id=token_id,
            direction=direction,
            edge=abs(divergence),
            confidence=min(Decimal("0.9"), Decimal("0.5") + abs(divergence) * 3),
            details={
                "symbol": symbol,
                "spot_price": str(spot.price),
                "strike_price": str(strike),
                "contract_price": str(contract_price),
                "divergence": str(divergence),
                "use_postonly": self._use_postonly,
                "question": market.question[:100],
            },
        )

    def _compute_divergence(
        self,
        spot: Decimal,
        strike: Decimal,
        contract_price: Decimal,
    ) -> Decimal:
        """Compute divergence between spot-implied and contract price.

        Positive divergence means contract is underpriced (spot implies YES).
        Negative divergence means contract is overpriced (spot implies NO).
        """
        if strike == 0:
            return Decimal("0")

        # Distance from spot to strike as fraction
        spot_distance = (spot - strike) / strike

        # Implied probability: if spot is well above strike, prob close to 1
        # Simple linear mapping with clamping
        if spot_distance > Decimal("0.01"):
            implied_prob = min(Decimal("0.95"), Decimal("0.5") + spot_distance * 10)
        elif spot_distance < Decimal("-0.01"):
            implied_prob = max(Decimal("0.05"), Decimal("0.5") + spot_distance * 10)
        else:
            # Very close to strike — high uncertainty
            implied_prob = Decimal("0.5")

        return implied_prob - contract_price

    def _prune_trade_timestamps(self) -> None:
        """Remove trade timestamps older than 1 hour."""
        cutoff = time.monotonic() - 3600
        self._trades_this_hour = [t for t in self._trades_this_hour if t > cutoff]
