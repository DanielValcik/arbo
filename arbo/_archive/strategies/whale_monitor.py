"""Whale position monitor via Polymarket Data API (PM-202).

Polls tracked whale wallets for position changes, detects new/increased/
decreased/closed positions, and generates Layer 4 trading signals.

Target: <10s from whale position change to signal emission.

See brief Layer 4 for full specification.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import aiohttp
import certifi

from arbo.config.settings import get_config
from arbo.core.scanner import Signal, SignalDirection
from arbo.strategies.whale_discovery import WhaleDiscovery  # noqa: TC001
from arbo.utils.logger import get_logger

logger = get_logger("whale_monitor")


class ChangeType:
    NEW = "NEW"
    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    CLOSED = "CLOSED"


@dataclass
class WalletPosition:
    """A single position held by a tracked whale wallet."""

    address: str
    condition_id: str
    token_id: str
    size: Decimal
    avg_price: Decimal
    outcome: str
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PositionChange:
    """Detected change in a whale's position."""

    address: str
    condition_id: str
    token_id: str
    change_type: str  # NEW, INCREASED, DECREASED, CLOSED
    old_size: Decimal
    new_size: Decimal
    delta: Decimal
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class WhaleMonitor:
    """Real-time whale position monitor.

    Polls the Data API every whale_poll_interval_s (default 4s) for each
    tracked wallet, diffs against stored positions, and generates Layer 4
    signals for new/increased positions.

    Multi-whale detection: ≥2 distinct whales in same market → boosted confidence.
    """

    def __init__(self, discovery: WhaleDiscovery) -> None:
        self._discovery = discovery
        config = get_config()
        self._data_url = config.polymarket.data_url
        self._poll_interval = config.confluence.whale_poll_interval_s
        self._session: aiohttp.ClientSession | None = None
        # Stored positions: {address: {token_id: WalletPosition}}
        self._positions: dict[str, dict[str, WalletPosition]] = {}

    async def initialize(self) -> None:
        """Create HTTP session."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Accept": "application/json"},
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )
        logger.info("whale_monitor_initialized")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def poll_cycle(self) -> list[Signal]:
        """Run one poll cycle across all tracked wallets.

        Fetches current positions for each whale, diffs against stored
        snapshot, generates signals for meaningful changes.

        Returns:
            List of Layer 4 signals from detected position changes.
        """
        wallets = self._discovery.get_tracked_wallets()
        if not wallets:
            return []

        all_changes: list[PositionChange] = []
        for wallet in wallets:
            new_positions = await self._fetch_positions(wallet.address)
            changes = self._diff_positions(wallet.address, new_positions)
            all_changes.extend(changes)

        if not all_changes:
            return []

        signals = self._changes_to_signals(all_changes)
        multi_signals = self._check_multi_whale(all_changes)
        signals.extend(multi_signals)

        logger.info(
            "whale_poll_complete",
            wallets_polled=len(wallets),
            changes=len(all_changes),
            signals=len(signals),
        )

        return signals

    def get_tracked_positions(self) -> dict[str, list[WalletPosition]]:
        """Get all tracked positions grouped by address."""
        return {addr: list(positions.values()) for addr, positions in self._positions.items()}

    async def _fetch_positions(self, address: str) -> list[WalletPosition]:
        """Fetch current positions for a wallet from Data API.

        Endpoint: GET {data_url}/positions?address={addr}
        """
        if not self._session:
            raise RuntimeError("WhaleMonitor not initialized. Call initialize() first.")

        url = f"{self._data_url}/positions"
        params = {"address": address}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.debug("whale_positions_error", address=address[:10], status=resp.status)
                    return []

                data = await resp.json()
                if not isinstance(data, list):
                    return []

                return self._parse_positions(address, data)

        except Exception as e:
            logger.debug("whale_positions_exception", address=address[:10], error=str(e))
            return []

    def _parse_positions(self, address: str, data: list[dict[str, Any]]) -> list[WalletPosition]:
        """Parse raw position data into WalletPosition objects."""
        positions: list[WalletPosition] = []
        for entry in data:
            try:
                size = Decimal(str(entry.get("size", "0") or "0"))
                if size <= 0:
                    continue

                positions.append(
                    WalletPosition(
                        address=address,
                        condition_id=entry.get("conditionId", "") or entry.get("condition_id", ""),
                        token_id=entry.get("tokenId", "") or entry.get("token_id", ""),
                        size=size,
                        avg_price=Decimal(
                            str(entry.get("avgPrice", "0") or entry.get("avg_price", "0") or "0")
                        ),
                        outcome=entry.get("outcome", "") or entry.get("side", ""),
                    )
                )
            except (ValueError, TypeError) as e:
                logger.debug("position_parse_error", error=str(e))
                continue

        return positions

    def _diff_positions(
        self, address: str, new_positions: list[WalletPosition]
    ) -> list[PositionChange]:
        """Diff new positions against stored snapshot for an address.

        Detects: NEW, INCREASED, DECREASED, CLOSED positions.
        Updates stored snapshot after diffing.

        D2 fix: On first poll (no baseline), seed positions and return empty —
        prevents mass signals from treating all existing positions as NEW.
        """
        old_map = self._positions.get(address, {})
        new_map = {p.token_id: p for p in new_positions}

        # D2: First run — seed baseline, emit no signals
        if not old_map:
            self._positions[address] = new_map
            if new_map:
                logger.info(
                    "whale_baseline_seeded",
                    address=address[:10],
                    positions=len(new_map),
                )
            return []

        changes: list[PositionChange] = []

        # Check for NEW and INCREASED positions
        for token_id, new_pos in new_map.items():
            if token_id not in old_map:
                changes.append(
                    PositionChange(
                        address=address,
                        condition_id=new_pos.condition_id,
                        token_id=token_id,
                        change_type=ChangeType.NEW,
                        old_size=Decimal("0"),
                        new_size=new_pos.size,
                        delta=new_pos.size,
                    )
                )
            else:
                old_pos = old_map[token_id]
                delta = new_pos.size - old_pos.size
                if delta > Decimal("0"):
                    changes.append(
                        PositionChange(
                            address=address,
                            condition_id=new_pos.condition_id,
                            token_id=token_id,
                            change_type=ChangeType.INCREASED,
                            old_size=old_pos.size,
                            new_size=new_pos.size,
                            delta=delta,
                        )
                    )
                elif delta < Decimal("0"):
                    changes.append(
                        PositionChange(
                            address=address,
                            condition_id=new_pos.condition_id,
                            token_id=token_id,
                            change_type=ChangeType.DECREASED,
                            old_size=old_pos.size,
                            new_size=new_pos.size,
                            delta=delta,
                        )
                    )

        # Check for CLOSED positions
        for token_id in old_map:
            if token_id not in new_map:
                old_pos = old_map[token_id]
                changes.append(
                    PositionChange(
                        address=address,
                        condition_id=old_pos.condition_id,
                        token_id=token_id,
                        change_type=ChangeType.CLOSED,
                        old_size=old_pos.size,
                        new_size=Decimal("0"),
                        delta=-old_pos.size,
                    )
                )

        # Update stored snapshot
        self._positions[address] = new_map

        return changes

    def _changes_to_signals(self, changes: list[PositionChange]) -> list[Signal]:
        """Convert position changes to Layer 4 signals.

        Only NEW and INCREASED changes generate signals.
        DECREASED and CLOSED are logged but not signaled.
        """
        signals: list[Signal] = []

        for change in changes:
            if change.change_type not in (ChangeType.NEW, ChangeType.INCREASED):
                logger.debug(
                    "whale_position_change",
                    address=change.address[:10],
                    change_type=change.change_type,
                    token_id=change.token_id[:10] if change.token_id else "",
                )
                continue

            signals.append(
                Signal(
                    layer=4,
                    market_condition_id=change.condition_id,
                    token_id=change.token_id,
                    direction=SignalDirection.BUY_YES,  # Mirrors whale side
                    edge=Decimal("0.03"),  # Conservative edge estimate
                    confidence=Decimal("0.65"),
                    details={
                        "whale_address": change.address,
                        "change_type": change.change_type,
                        "old_size": str(change.old_size),
                        "new_size": str(change.new_size),
                        "delta": str(change.delta),
                    },
                )
            )

        return signals

    def _check_multi_whale(self, changes: list[PositionChange]) -> list[Signal]:
        """Check for multi-whale confluence (≥2 whales in same market).

        When multiple whales are buying the same market, confidence is boosted
        from 0.65 to 0.85.
        """
        # Group BUY changes by condition_id
        market_whales: dict[str, set[str]] = {}
        market_token: dict[str, str] = {}
        for change in changes:
            if change.change_type not in (ChangeType.NEW, ChangeType.INCREASED):
                continue
            cid = change.condition_id
            if cid not in market_whales:
                market_whales[cid] = set()
                market_token[cid] = change.token_id
            market_whales[cid].add(change.address)

        signals: list[Signal] = []
        for cid, addresses in market_whales.items():
            if len(addresses) >= 2:
                signals.append(
                    Signal(
                        layer=4,
                        market_condition_id=cid,
                        token_id=market_token[cid],
                        direction=SignalDirection.BUY_YES,
                        edge=Decimal("0.05"),
                        confidence=Decimal("0.85"),
                        details={
                            "multi_whale": True,
                            "whale_count": len(addresses),
                            "addresses": list(addresses),
                        },
                    )
                )

        return signals
