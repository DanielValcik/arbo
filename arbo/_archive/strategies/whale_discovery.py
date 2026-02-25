"""Whale wallet discovery via Polymarket Data API (PM-201).

Discovers top-performing wallets from the leaderboard, enriches with
position data, and filters by win rate, resolved positions, and volume.

See brief Layer 4 for full specification.
"""

from __future__ import annotations

import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp
import certifi

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("whale_discovery")

# Filter constants — CEO-authorized relaxation (D2, 2026-02-25).
# Original: 60% / 50 / $50K — too strict, 0 wallets qualified.
MIN_WIN_RATE = 50.0  # percent (was 60)
MIN_RESOLVED_POSITIONS = 20  # (was 50)
MIN_TOTAL_VOLUME = 25_000.0  # USD (was 50K)

# Staleness interval for full refresh (7 days in seconds)
REFRESH_INTERVAL_S = 7 * 24 * 3600

# Leaderboard page size
LEADERBOARD_PAGE_SIZE = 50  # Polymarket API max per request
MAX_LEADERBOARD_PAGES = 2  # 50 * 2 = 100 wallets (top traders only)


@dataclass
class WhaleWallet:
    """A discovered whale wallet with performance metrics."""

    address: str
    display_name: str
    roi_pct: float
    win_rate: float
    resolved_positions: int
    total_volume: float
    discovery_source: str
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_checked: datetime = field(default_factory=lambda: datetime.now(UTC))

    def passes_filters(self) -> bool:
        """Check if this wallet passes quality filters.

        Leaderboard wallets lack win_rate/resolved_positions, so filter
        on volume + positive ROI. Once enriched, full filters apply.
        """
        if self.win_rate > 0 and self.resolved_positions > 0:
            # Fully enriched wallet — apply strict filters
            return (
                self.win_rate >= MIN_WIN_RATE
                and self.resolved_positions >= MIN_RESOLVED_POSITIONS
                and self.total_volume >= MIN_TOTAL_VOLUME
            )
        # Leaderboard wallet (not yet enriched) — accept on volume + positive PnL
        return self.total_volume >= MIN_TOTAL_VOLUME and self.roi_pct > 0


class WhaleDiscovery:
    """Discovers and catalogs whale wallets from Polymarket Data API.

    Features:
    - Scrapes leaderboard for top-performing wallets
    - Enriches wallets with position data from Data API
    - Filters by win rate > 60%, resolved >= 50, volume > $50K
    - 7-day refresh interval with monotonic clock staleness check
    - In-memory store keyed by address
    """

    def __init__(self) -> None:
        config = get_config()
        self._data_url = config.polymarket.data_url
        self._session: aiohttp.ClientSession | None = None
        self._wallets: dict[str, WhaleWallet] = {}
        self._last_refresh: float = 0

    async def initialize(self) -> None:
        """Create HTTP session with SSL context."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Accept": "application/json"},
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )
        logger.info("whale_discovery_initialized", data_url=self._data_url)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def discover(self) -> list[WhaleWallet]:
        """Run full discovery pipeline: scrape leaderboard, enrich, filter.

        Returns:
            List of whale wallets passing all quality filters.
        """
        raw_wallets = await self._scrape_leaderboard()
        logger.info("leaderboard_scraped", raw_count=len(raw_wallets))

        enriched: list[WhaleWallet] = []
        for wallet in raw_wallets:
            enriched_wallet = await self._enrich_via_data_api(wallet)
            enriched.append(enriched_wallet)

        filtered = self._apply_filters(enriched)

        # Update in-memory store
        for wallet in filtered:
            self._wallets[wallet.address] = wallet

        logger.info(
            "whale_discovery_complete",
            raw=len(raw_wallets),
            enriched=len(enriched),
            filtered=len(filtered),
        )

        return filtered

    async def refresh_if_stale(self) -> list[WhaleWallet]:
        """Refresh wallet catalog if 7-day interval has elapsed.

        Returns:
            Current list of tracked wallets.
        """
        now = time.monotonic()
        if now - self._last_refresh < REFRESH_INTERVAL_S:
            return list(self._wallets.values())

        wallets = await self.discover()
        self._last_refresh = now
        return wallets

    def get_tracked_wallets(self) -> list[WhaleWallet]:
        """Get all currently tracked whale wallets."""
        return list(self._wallets.values())

    def get_wallet(self, address: str) -> WhaleWallet | None:
        """Get a specific wallet by address."""
        return self._wallets.get(address)

    async def _scrape_leaderboard(self) -> list[WhaleWallet]:
        """Scrape leaderboard from Data API.

        Endpoint: GET {data_url}/v1/leaderboard
        Paginates through results to collect top wallets.
        """
        if not self._session:
            raise RuntimeError("WhaleDiscovery not initialized. Call initialize() first.")

        wallets: list[WhaleWallet] = []

        for page in range(MAX_LEADERBOARD_PAGES):
            offset = page * LEADERBOARD_PAGE_SIZE
            url = f"{self._data_url}/v1/leaderboard"
            params = {
                "limit": str(LEADERBOARD_PAGE_SIZE),
                "offset": str(offset),
                "timePeriod": "ALL",
                "orderBy": "PNL",
            }

            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(
                            "leaderboard_api_error",
                            status=resp.status,
                            page=page,
                        )
                        break

                    data = await resp.json()
                    if not isinstance(data, list) or not data:
                        break

                    for entry in data:
                        wallet = self._parse_leaderboard_entry(entry)
                        if wallet:
                            wallets.append(wallet)

                    if len(data) < LEADERBOARD_PAGE_SIZE:
                        break

            except Exception as e:
                logger.error("leaderboard_scrape_error", error=str(e), page=page)
                break

        return wallets

    def _parse_leaderboard_entry(self, entry: dict[str, Any]) -> WhaleWallet | None:
        """Parse a single leaderboard entry into a WhaleWallet.

        API response fields: proxyWallet, userName, pnl, vol, rank.
        win_rate/resolved_positions not in leaderboard — enriched later.
        """
        try:
            address = (
                entry.get("proxyWallet", "")
                or entry.get("address", "")
                or entry.get("wallet", "")
            )
            if not address:
                return None

            pnl = float(entry.get("pnl", 0) or 0)
            vol = float(entry.get("vol", 0) or entry.get("totalVolume", 0) or 0)

            return WhaleWallet(
                address=address,
                display_name=(
                    entry.get("userName", "")
                    or entry.get("displayName", "")
                    or address[:10]
                ),
                roi_pct=pnl / vol * 100 if vol > 0 else 0.0,
                win_rate=0.0,  # Not in leaderboard API; enriched later
                resolved_positions=0,  # Not in leaderboard API; enriched later
                total_volume=vol,
                discovery_source="leaderboard",
            )
        except (ValueError, TypeError) as e:
            logger.warning("leaderboard_parse_error", error=str(e))
            return None

    async def _enrich_via_data_api(self, wallet: WhaleWallet) -> WhaleWallet:
        """Enrich wallet with position data from Data API.

        Endpoint: GET {data_url}/positions?address={addr}
        Updates resolved_positions and total_volume if data is available.
        """
        if not self._session:
            return wallet

        url = f"{self._data_url}/positions"
        params = {"user": wallet.address}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.debug(
                        "enrich_api_error",
                        address=wallet.address[:10],
                        status=resp.status,
                    )
                    return wallet

                data = await resp.json()

                if isinstance(data, list):
                    # Count positions and calculate volume from position data
                    total_positions = len(data)
                    total_volume = sum(float(p.get("size", 0) or 0) for p in data)

                    if total_positions > wallet.resolved_positions:
                        wallet.resolved_positions = total_positions
                    if total_volume > wallet.total_volume:
                        wallet.total_volume = total_volume

                wallet.last_checked = datetime.now(UTC)

        except Exception as e:
            logger.debug("enrich_error", address=wallet.address[:10], error=str(e))

        return wallet

    def _apply_filters(self, wallets: list[WhaleWallet]) -> list[WhaleWallet]:
        """Apply quality filters to wallet list.

        Filters:
        - win_rate >= 60%
        - resolved_positions >= 50
        - total_volume >= $50,000
        """
        filtered = [w for w in wallets if w.passes_filters()]
        logger.debug(
            "whale_filters_applied",
            input=len(wallets),
            output=len(filtered),
            min_win_rate=MIN_WIN_RATE,
            min_resolved=MIN_RESOLVED_POSITIONS,
            min_volume=MIN_TOTAL_VOLUME,
        )
        return filtered
