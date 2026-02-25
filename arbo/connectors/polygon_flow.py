"""Order flow monitor — Layer 7 (PM-105).

Monitors Polygon blockchain for CTF Exchange OrderFilled events via
Alchemy HTTP polling (eth_getLogs). Tracks volume, buy/sell imbalance,
and z-score spikes to detect smart money activity.

CU budget: ~5.1K CU/hour (eth_blockNumber ~10 CU + eth_getLogs ~75 CU, 1/min).

See brief Layer 7 for full specification.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections import deque
from collections.abc import Callable  # noqa: TC003 — used at runtime in __init__
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import aiohttp
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: TC003 — used at runtime

from arbo.config.settings import get_config
from arbo.core.scanner import Signal, SignalDirection
from arbo.utils.logger import get_logger

logger = get_logger("polygon_flow")

# OrderFilled(bytes32 indexed orderHash, address indexed maker, address taker,
#   uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled,
#   uint256 takerAmountFilled, uint256 fee)
ORDER_FILLED_TOPIC = "0x" + "a".zfill(64)  # Placeholder — set at runtime via _compute_topic()


def _compute_topic() -> str:
    """Compute OrderFilled event topic hash using web3."""
    try:
        from web3 import Web3

        sig = "OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
        return "0x" + Web3.keccak(text=sig).hex()
    except ImportError:
        # Fallback if web3 not installed
        return ORDER_FILLED_TOPIC


# USDC.e amount divisor (6 decimals)
_USDC_DECIMALS = Decimal("1000000")

# Incremental block tracking constants
_COLD_START_BLOCKS = 1000  # ~30 min of Polygon blocks on first run
_DB_KEY = "l7_last_processed_block"  # DB persistence key
_CU_PER_POLL = 85  # estimated CU per poll (blockNumber 10 + getLogs 75)
_MAX_BLOCK_GAP = 10_000  # discard restored block if >10K behind (~5h)


@dataclass
class OrderFilledEvent:
    """Parsed OrderFilled event from CTF Exchange."""

    order_hash: str
    maker: str
    taker: str
    maker_asset_id: str
    taker_asset_id: str
    maker_amount: Decimal
    taker_amount: Decimal
    fee: Decimal
    is_buy: bool
    token_id: str
    timestamp: datetime
    block_number: int


@dataclass
class FlowMetrics:
    """Aggregated flow metrics for a token."""

    token_id: str
    volume_1h: Decimal = Decimal("0")
    volume_4h: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")
    buy_volume_1h: Decimal = Decimal("0")
    sell_volume_1h: Decimal = Decimal("0")
    cumulative_delta: Decimal = Decimal("0")
    flow_imbalance: Decimal = Decimal("0")
    event_count: int = 0


@dataclass
class FlowEntry:
    """Single order flow entry."""

    value: Decimal
    is_buy: bool
    timestamp: float  # monotonic time


class RollingWindow:
    """Time-windowed rolling statistics for order flow.

    Tracks events in a deque, supports volume queries over arbitrary windows,
    and computes z-scores using 5-minute bucket aggregation.
    """

    BUCKET_SIZE_S = 300  # 5 minutes

    def __init__(self, max_window_s: int = 86400) -> None:
        self._entries: deque[FlowEntry] = deque()
        self._max_window_s = max_window_s
        self._time_fn = time.monotonic

    def add(self, value: Decimal, is_buy: bool, timestamp: float | None = None) -> None:
        """Add a flow entry."""
        ts = timestamp if timestamp is not None else self._time_fn()
        self._entries.append(FlowEntry(value=value, is_buy=is_buy, timestamp=ts))
        self._prune()

    def get_volume(self, window_s: int | None = None) -> Decimal:
        """Get total volume in window."""
        cutoff = self._cutoff(window_s)
        return sum((e.value for e in self._entries if e.timestamp >= cutoff), Decimal("0"))

    def get_buy_volume(self, window_s: int | None = None) -> Decimal:
        """Get buy volume in window."""
        cutoff = self._cutoff(window_s)
        return sum(
            (e.value for e in self._entries if e.timestamp >= cutoff and e.is_buy), Decimal("0")
        )

    def get_sell_volume(self, window_s: int | None = None) -> Decimal:
        """Get sell volume in window."""
        cutoff = self._cutoff(window_s)
        return sum(
            (e.value for e in self._entries if e.timestamp >= cutoff and not e.is_buy), Decimal("0")
        )

    def get_zscore(self, window_s: int = 3600) -> float:
        """Compute z-score of current 5-min bucket vs rolling buckets.

        Args:
            window_s: Time window for computing rolling stats.

        Returns:
            Z-score. 0.0 if insufficient data.
        """
        now = self._time_fn()
        cutoff = now - window_s

        # Build 5-min buckets
        buckets: dict[int, Decimal] = {}
        for e in self._entries:
            if e.timestamp < cutoff:
                continue
            bucket_idx = int((now - e.timestamp) / self.BUCKET_SIZE_S)
            buckets[bucket_idx] = buckets.get(bucket_idx, Decimal("0")) + e.value

        if len(buckets) < 2:
            return 0.0

        # Current bucket = index 0
        current = float(buckets.get(0, Decimal("0")))

        # Historical buckets (excluding current)
        historical = [float(v) for k, v in buckets.items() if k > 0]
        if not historical:
            return 0.0

        mean = sum(historical) / len(historical)
        if len(historical) < 2:
            return 0.0

        variance = sum((x - mean) ** 2 for x in historical) / (len(historical) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return 0.0

        return (current - mean) / std

    def _cutoff(self, window_s: int | None) -> float:
        """Get cutoff time for a window."""
        if window_s is None:
            return 0.0
        return self._time_fn() - window_s

    def _prune(self) -> None:
        """Remove entries older than max window."""
        cutoff = self._time_fn() - self._max_window_s
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()


def parse_order_filled(log: dict[str, Any]) -> OrderFilledEvent | None:
    """Parse raw Ethereum log into OrderFilledEvent.

    Decodes ABI-encoded data from eth_subscribe log result.

    Args:
        log: Raw log dict with 'topics', 'data', 'blockNumber' fields.

    Returns:
        Parsed event or None if invalid.
    """
    try:
        topics = log.get("topics", [])
        data = log.get("data", "")

        if len(topics) < 3:
            return None

        # topics[1] = orderHash (bytes32), topics[2] = maker (address, padded)
        order_hash = topics[1]
        maker = "0x" + topics[2][-40:]

        # Decode non-indexed data fields (each 32 bytes = 64 hex chars)
        data_hex = data[2:] if data.startswith("0x") else data

        if len(data_hex) < 384:  # 6 fields x 64 chars
            return None

        taker = "0x" + data_hex[24:64]
        maker_asset_id = int(data_hex[64:128], 16)
        taker_asset_id = int(data_hex[128:192], 16)
        maker_amount_raw = int(data_hex[192:256], 16)
        taker_amount_raw = int(data_hex[256:320], 16)
        fee_raw = int(data_hex[320:384], 16)

        block_hex = log.get("blockNumber", "0x0")
        block_number = int(block_hex, 16) if isinstance(block_hex, str) else int(block_hex)

        # Determine buy/sell: the conditional token is the larger asset ID
        # (USDC.e address as uint256 is much smaller than conditional token IDs)
        is_buy = maker_asset_id > taker_asset_id
        token_id = str(maker_asset_id) if is_buy else str(taker_asset_id)

        # Scale amounts (USDC has 6 decimals)
        maker_amount = Decimal(maker_amount_raw) / _USDC_DECIMALS
        taker_amount = Decimal(taker_amount_raw) / _USDC_DECIMALS
        fee = Decimal(fee_raw) / _USDC_DECIMALS

        return OrderFilledEvent(
            order_hash=order_hash,
            maker=maker,
            taker=taker,
            maker_asset_id=str(maker_asset_id),
            taker_asset_id=str(taker_asset_id),
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            fee=fee,
            is_buy=is_buy,
            token_id=token_id,
            timestamp=datetime.now(UTC),
            block_number=block_number,
        )
    except (ValueError, IndexError, TypeError) as e:
        logger.debug("parse_order_filled_error", error=str(e))
        return None


class OrderFlowMonitor:
    """Layer 7: Smart money order flow monitor.

    Polls CTF Exchange OrderFilled events via Alchemy HTTP (eth_getLogs).
    Tracks volume flow per token, detects z-score spikes and imbalance
    convergence to emit trading signals.

    CU budget: ~5.1K CU/hour (eth_blockNumber ~10 CU + eth_getLogs ~75 CU per poll).

    Signal convergence requires >=2 of:
    - Volume z-score > threshold (default 2.0)
    - Flow imbalance > threshold (default 0.65)
    - Cumulative delta trending (consistent directional flow)
    """

    def __init__(
        self,
        on_signal: Callable[[Signal], None] | None = None,
        poll_interval: int = 60,
        session_factory: async_sessionmaker | None = None,
    ) -> None:
        config = get_config()
        self._http_url = config.polygon_rpc_url
        self._ctf_exchange = config.order_flow.ctf_exchange
        self._zscore_threshold = config.order_flow.volume_zscore_threshold
        self._imbalance_threshold = Decimal(str(config.order_flow.flow_imbalance_threshold))
        self._min_converging = config.order_flow.min_converging_signals
        self._on_signal = on_signal
        self._poll_interval = poll_interval
        self._session_factory = session_factory
        self._windows: dict[str, RollingWindow] = {}
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_block = 0
        self._matched_tokens: set[str] = set()
        self._total_events = 0
        self._signals_emitted = 0
        self._topic: str = ""

    async def initialize(self) -> None:
        """Compute event topic hash."""
        self._topic = _compute_topic()
        if not self._http_url:
            logger.warning("order_flow_no_rpc_url", msg="DRPC_API_URL not set")
        logger.info(
            "order_flow_initialized",
            ctf_exchange=self._ctf_exchange,
            poll_interval=self._poll_interval,
        )

    async def start(self) -> None:
        """Start polling loop (non-blocking, creates background task).

        Block selection on start:
        1. Try restoring last processed block from DB
        2. If found and within _MAX_BLOCK_GAP of latest: resume from restored
        3. If found but stale (>_MAX_BLOCK_GAP behind): cold start
        4. If not found: cold start (latest - _COLD_START_BLOCKS)
        """
        if not self._http_url:
            logger.warning("order_flow_cannot_start", msg="No Polygon RPC URL configured")
            return
        if self._running:
            return

        self._session = aiohttp.ClientSession()
        self._running = True

        latest = await self._get_block_number()
        restored = await self._restore_last_block()

        if restored is not None and latest - restored <= _MAX_BLOCK_GAP:
            self._last_block = restored
            logger.info(
                "order_flow_restored_block",
                restored_block=restored,
                latest_block=latest,
                gap=latest - restored,
            )
        elif restored is not None:
            self._last_block = latest - _COLD_START_BLOCKS
            logger.warning(
                "order_flow_stale_block_discarded",
                restored_block=restored,
                latest_block=latest,
                gap=latest - restored,
            )
        else:
            self._last_block = latest - _COLD_START_BLOCKS
            logger.info("order_flow_cold_start", from_block=self._last_block, latest=latest)

        self._task = asyncio.create_task(self._poll_loop())
        logger.info("order_flow_polling_started", from_block=self._last_block)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        logger.info(
            "order_flow_stopped",
            total_events=self._total_events,
            signals=self._signals_emitted,
        )

    def update_matched_tokens(self, token_ids: set[str]) -> None:
        """Update the set of token IDs to filter on (from matched markets).

        Args:
            token_ids: Token IDs from discovery/matching. Empty = process all.
        """
        self._matched_tokens = token_ids

    async def _poll_loop(self) -> None:
        """Poll eth_getLogs every interval."""
        while self._running:
            try:
                current_block = await self._get_block_number()
                if current_block > self._last_block:
                    from_block = self._last_block + 1
                    logs = await self._get_logs(from_block, current_block)
                    for log in logs:
                        self._handle_log(log)
                    self._last_block = current_block
                    await self._save_last_block()
                    logger.debug(
                        "order_flow_poll_ok",
                        from_block=from_block,
                        to_block=current_block,
                        logs=len(logs),
                        blocks_scanned=current_block - from_block + 1,
                        cu_estimate=_CU_PER_POLL,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("order_flow_poll_error", error=str(e))
            # Sleep in 1s increments for responsive shutdown
            for _ in range(self._poll_interval):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def _get_block_number(self) -> int:
        """eth_blockNumber RPC call (~10 CU)."""
        if self._session is None:
            raise RuntimeError("Session not initialized")
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
        async with self._session.post(self._http_url, json=payload) as resp:
            data = await resp.json()
            return int(data["result"], 16)

    async def _get_logs(self, from_block: int, to_block: int) -> list[dict[str, Any]]:
        """eth_getLogs for OrderFilled events (~75 CU per call)."""
        if self._session is None:
            raise RuntimeError("Session not initialized")
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getLogs",
            "params": [
                {
                    "address": self._ctf_exchange,
                    "topics": [self._topic],
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                }
            ],
        }
        async with self._session.post(self._http_url, json=payload) as resp:
            data = await resp.json()
            return data.get("result", [])

    async def _save_last_block(self) -> None:
        """Persist _last_block to SystemState table (best-effort, non-blocking)."""
        if self._session_factory is None:
            return
        try:
            from sqlalchemy import select

            from arbo.utils.db import SystemState

            async with self._session_factory() as session:
                result = await session.execute(
                    select(SystemState).where(SystemState.key == _DB_KEY)
                )
                row = result.scalar_one_or_none()
                if row is not None:
                    row.value = str(self._last_block)
                else:
                    session.add(SystemState(key=_DB_KEY, value=str(self._last_block)))
                await session.commit()
        except Exception as e:
            logger.debug("order_flow_save_block_error", error=str(e))

    async def _restore_last_block(self) -> int | None:
        """Read last processed block from SystemState table.

        Returns:
            Block number or None if not found / no DB.
        """
        if self._session_factory is None:
            return None
        try:
            from sqlalchemy import select

            from arbo.utils.db import SystemState

            async with self._session_factory() as session:
                result = await session.execute(
                    select(SystemState.value).where(SystemState.key == _DB_KEY)
                )
                row = result.scalar_one_or_none()
                if row is not None:
                    return int(row)
        except Exception as e:
            logger.debug("order_flow_restore_block_error", error=str(e))
        return None

    def _handle_log(self, log: dict[str, Any]) -> None:
        """Parse log, filter by matched tokens, feed to RollingWindow."""
        event = parse_order_filled(log)
        if event is None:
            return

        # Filter: only process events for markets we're tracking
        if self._matched_tokens and event.token_id not in self._matched_tokens:
            return

        self._total_events += 1

        # Get or create rolling window for this token
        if event.token_id not in self._windows:
            self._windows[event.token_id] = RollingWindow()

        window = self._windows[event.token_id]
        volume = event.taker_amount if event.is_buy else event.maker_amount
        window.add(volume, event.is_buy)

        # Check convergence and emit signal
        signal = self._check_convergence(event.token_id)
        if signal is not None:
            self._signals_emitted += 1
            logger.info(
                "order_flow_signal",
                token_id=signal.token_id[:20],
                edge=str(signal.edge),
            )
            if self._on_signal is not None:
                self._on_signal(signal)

    def _check_convergence(self, token_id: str) -> Signal | None:
        """Check if multiple flow signals converge for a token.

        Requires >=2 of: z-score spike, flow imbalance, delta trending.
        """
        window = self._windows.get(token_id)
        if window is None:
            return None

        converging = 0

        # Signal 1: Volume z-score spike
        zscore = window.get_zscore(window_s=3600)
        if zscore > self._zscore_threshold:
            converging += 1

        # Signal 2: Flow imbalance
        buy_vol = window.get_buy_volume(window_s=3600)
        sell_vol = window.get_sell_volume(window_s=3600)
        total_vol = buy_vol + sell_vol
        imbalance = Decimal("0")
        if total_vol > 0:
            imbalance = abs(buy_vol - sell_vol) / total_vol
        if imbalance > self._imbalance_threshold:
            converging += 1

        # Signal 3: Cumulative delta trending (consistent direction)
        delta = buy_vol - sell_vol
        vol_4h = window.get_volume(window_s=14400)
        if vol_4h > 0 and abs(delta) / vol_4h > Decimal("0.3"):
            converging += 1

        if converging < self._min_converging:
            return None

        # Direction: net buyers → BUY, net sellers → SELL
        is_net_buy = buy_vol > sell_vol
        direction = SignalDirection.BUY_YES if is_net_buy else SignalDirection.SELL_YES

        edge = Decimal(str(min(0.15, zscore * 0.03)))
        confidence = Decimal(str(min(0.9, 0.4 + converging * 0.15)))

        return Signal(
            layer=7,
            market_condition_id="",  # Token-level signal, condition resolved downstream
            token_id=token_id,
            direction=direction,
            edge=edge,
            confidence=confidence,
            details={
                "zscore": round(zscore, 2),
                "imbalance": str(imbalance.quantize(Decimal("0.001"))),
                "delta": str(delta.quantize(Decimal("0.01"))),
                "buy_volume_1h": str(buy_vol.quantize(Decimal("0.01"))),
                "sell_volume_1h": str(sell_vol.quantize(Decimal("0.01"))),
                "converging_signals": converging,
            },
        )

    def get_metrics(self, token_id: str) -> FlowMetrics | None:
        """Get aggregated flow metrics for a token."""
        window = self._windows.get(token_id)
        if window is None:
            return None

        buy_1h = window.get_buy_volume(window_s=3600)
        sell_1h = window.get_sell_volume(window_s=3600)
        total_1h = buy_1h + sell_1h
        imbalance = abs(buy_1h - sell_1h) / total_1h if total_1h > 0 else Decimal("0")

        return FlowMetrics(
            token_id=token_id,
            volume_1h=window.get_volume(window_s=3600),
            volume_4h=window.get_volume(window_s=14400),
            volume_24h=window.get_volume(window_s=86400),
            buy_volume_1h=buy_1h,
            sell_volume_1h=sell_1h,
            cumulative_delta=buy_1h - sell_1h,
            flow_imbalance=imbalance,
            event_count=len(window._entries),
        )

    @property
    def is_healthy(self) -> bool:
        """Check if the monitor is running and its poll task is alive.

        Returns False if:
        - Not started (_running is False)
        - No background task exists
        - Background task has completed/crashed
        """
        if not self._running:
            return False
        if self._task is None:
            return False
        if self._task.done():
            return False
        return True

    @property
    def stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return {
            "total_events": self._total_events,
            "signals_emitted": self._signals_emitted,
            "active_tokens": len(self._windows),
        }
