"""Market discovery module using Polymarket Gamma API.

Automatically discovers and catalogs active markets with filtering
by category, liquidity, fee status, and NegRisk flag.

See brief PM-002 for full specification.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from decimal import Decimal
from typing import Any

import aiohttp
import certifi

from arbo.config.settings import get_config
from arbo.utils.logger import get_logger

logger = get_logger("market_discovery")

# Category mapping for Polymarket markets
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "soccer": ["soccer", "football", "epl", "premier league", "la liga", "bundesliga", "serie a"],
    "politics": ["president", "election", "congress", "senate", "democrat", "republican", "trump"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "token"],
    "esports": ["esports", "league of legends", "cs2", "dota", "valorant"],
    "entertainment": ["oscar", "grammy", "movie", "tv show", "celebrity"],
    "attention_markets": ["mindshare", "attention", "kaito"],
}


def categorize_market(question: str, tags: list[str] | None = None) -> str:
    """Categorize a market based on its question text and tags.

    Args:
        question: The market question string.
        tags: Optional list of Gamma API tags.

    Returns:
        Category string (soccer, politics, crypto, etc.) or "other".
    """
    q_lower = question.lower()
    tag_str = " ".join(tags).lower() if tags else ""
    combined = f"{q_lower} {tag_str}"

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return category

    return "other"


class GammaMarket:
    """Parsed market data from Gamma API response."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.condition_id: str = raw.get("conditionId", "")
        self.question: str = raw.get("question", "")
        self.slug: str = raw.get("slug", "")
        self.category: str = ""
        self.outcomes: list[str] = self._parse_json_list(raw.get("outcomes", []))
        self.outcome_prices: list[str] = self._parse_json_list(raw.get("outcomePrices", []))
        self.clob_token_ids: list[str] = self._parse_json_list(raw.get("clobTokenIds", []))
        self.fee_enabled: bool = raw.get("feesEnabled", False) or raw.get("feeType") is not None
        self.neg_risk: bool = raw.get("enableNegRisk", False)
        self.active: bool = raw.get("active", False)
        self.closed: bool = raw.get("closed", False)
        self.end_date: str | None = raw.get("endDate")
        self.volume: Decimal = Decimal(str(raw.get("volume", "0") or "0"))
        self.volume_24h: Decimal = Decimal(str(raw.get("volume24hr", "0") or "0"))
        self.liquidity: Decimal = Decimal(str(raw.get("liquidity", "0") or "0"))
        self.market_maker_address: str | None = raw.get("marketMakerAddress")
        self.description: str = raw.get("description", "")
        self.image: str = raw.get("image", "")

        # Categorize
        tags = raw.get("tags", [])
        if isinstance(tags, list):
            tag_labels = [t.get("label", "") if isinstance(t, dict) else str(t) for t in tags]
        else:
            tag_labels = []
        self.category = categorize_market(self.question, tag_labels)

    @staticmethod
    def _parse_json_list(value: Any) -> list[str]:
        """Parse a field that may be a JSON-encoded string or an actual list.

        Gamma API returns clobTokenIds, outcomes, and outcomePrices as
        JSON-encoded strings (e.g. '["Yes", "No"]') rather than native lists.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
            return []
        if isinstance(value, list):
            return [str(x) for x in value]
        return []

    @property
    def price_yes(self) -> Decimal | None:
        """Current YES price."""
        if self.outcome_prices and len(self.outcome_prices) > 0:
            try:
                return Decimal(str(self.outcome_prices[0]))
            except Exception:
                return None
        return None

    @property
    def price_no(self) -> Decimal | None:
        """Current NO price."""
        if self.outcome_prices and len(self.outcome_prices) > 1:
            try:
                return Decimal(str(self.outcome_prices[1]))
            except Exception:
                return None
        return None

    @property
    def token_id_yes(self) -> str | None:
        """CLOB token ID for YES outcome."""
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def token_id_no(self) -> str | None:
        """CLOB token ID for NO outcome."""
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None

    @property
    def spread(self) -> Decimal | None:
        """Price spread between YES and NO (should sum to ~1.0)."""
        yes = self.price_yes
        no = self.price_no
        if yes is not None and no is not None:
            return abs(Decimal("1") - yes - no)
        return None

    def to_db_dict(self) -> dict[str, Any]:
        """Convert to dict for DB insertion."""
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "slug": self.slug,
            "category": self.category,
            "outcomes": self.outcomes,
            "clob_token_ids": self.clob_token_ids,
            "fee_enabled": self.fee_enabled,
            "neg_risk": self.neg_risk,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "end_date": self.end_date,
            "active": self.active and not self.closed,
            "last_price_yes": float(self.price_yes) if self.price_yes else None,
            "last_price_no": float(self.price_no) if self.price_no else None,
        }


class MarketDiscovery:
    """Discovers and catalogs active Polymarket markets via Gamma API.

    Features:
    - Paginated fetching of all active markets
    - Filtering by category, volume, fee status
    - Categorization: soccer, politics, crypto, esports, entertainment, attention_markets
    - 15-minute refresh interval
    """

    def __init__(self) -> None:
        config = get_config()
        self._gamma_url = config.polymarket.gamma_url
        self._session: aiohttp.ClientSession | None = None
        self._markets: dict[str, GammaMarket] = {}
        self._last_refresh: float = 0
        self._refresh_interval = config.polling.market_discovery

    async def initialize(self) -> None:
        """Create HTTP session."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Accept": "application/json"},
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )
        logger.info("market_discovery_initialized", gamma_url=self._gamma_url)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _fetch_page(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        order: str = "volume24hr",
    ) -> list[dict[str, Any]]:
        """Fetch a single page of markets from Gamma API.

        Args:
            limit: Results per page (max ~100).
            offset: Pagination offset.
            active: Only active markets.
            order: Sort field.

        Returns:
            List of raw market dicts.
        """
        if not self._session:
            raise RuntimeError("MarketDiscovery not initialized. Call initialize() first.")

        params: dict[str, str] = {
            "limit": str(limit),
            "offset": str(offset),
            "order": order,
            "ascending": "false",
        }
        if active:
            params["active"] = "true"
            params["closed"] = "false"

        url = f"{self._gamma_url}/markets"

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(
                        "gamma_api_error",
                        status=resp.status,
                        url=url,
                    )
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error("gamma_api_exception", error=str(e))
            return []

    async def fetch_all_active_markets(self, max_pages: int = 50) -> list[GammaMarket]:
        """Fetch all active markets with pagination.

        Args:
            max_pages: Maximum pages to fetch (safety limit).

        Returns:
            List of parsed GammaMarket objects.
        """
        all_markets: list[GammaMarket] = []
        offset = 0
        page_size = 100

        for page in range(max_pages):
            raw_markets = await self._fetch_page(limit=page_size, offset=offset)

            if not raw_markets:
                break

            for raw in raw_markets:
                market = GammaMarket(raw)
                if market.condition_id and market.active and not market.closed:
                    all_markets.append(market)

            logger.debug(
                "gamma_page_fetched",
                page=page + 1,
                count=len(raw_markets),
                total=len(all_markets),
            )

            if len(raw_markets) < page_size:
                break

            offset += page_size
            await asyncio.sleep(0.1)  # Rate limit courtesy

        logger.info("markets_fetched", total=len(all_markets))
        return all_markets

    async def refresh(self) -> list[GammaMarket]:
        """Refresh market catalog if refresh interval has elapsed.

        Returns:
            Updated list of markets.
        """
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return list(self._markets.values())

        markets = await self.fetch_all_active_markets()

        self._markets = {m.condition_id: m for m in markets}
        self._last_refresh = now

        # Log category breakdown
        categories: dict[str, int] = {}
        for m in markets:
            categories[m.category] = categories.get(m.category, 0) + 1
        logger.info("market_catalog_refreshed", total=len(markets), categories=categories)

        return markets

    def get_all(self) -> list[GammaMarket]:
        """Get all cached markets."""
        return list(self._markets.values())

    def get_by_category(self, category: str) -> list[GammaMarket]:
        """Get markets filtered by category."""
        return [m for m in self._markets.values() if m.category == category]

    def get_by_condition_id(self, condition_id: str) -> GammaMarket | None:
        """Get a specific market by condition ID."""
        return self._markets.get(condition_id)

    def filter_markets(
        self,
        min_volume_24h: Decimal = Decimal("0"),
        min_liquidity: Decimal = Decimal("0"),
        category: str | None = None,
        fee_enabled: bool | None = None,
        neg_risk: bool | None = None,
    ) -> list[GammaMarket]:
        """Filter markets by multiple criteria.

        Args:
            min_volume_24h: Minimum 24h volume in USDC.
            min_liquidity: Minimum liquidity in USDC.
            category: Filter by category (soccer, crypto, etc.).
            fee_enabled: Filter by fee status.
            neg_risk: Filter by NegRisk flag.

        Returns:
            Filtered list of markets.
        """
        result = []
        for m in self._markets.values():
            if m.volume_24h < min_volume_24h:
                continue
            if m.liquidity < min_liquidity:
                continue
            if category is not None and m.category != category:
                continue
            if fee_enabled is not None and m.fee_enabled != fee_enabled:
                continue
            if neg_risk is not None and m.neg_risk != neg_risk:
                continue
            result.append(m)
        return result

    def get_mm_candidates(self) -> list[GammaMarket]:
        """Get markets suitable for market making (Layer 1).

        Criteria from brief:
        - Volume $1K-$50K/day
        - Active and not closed
        - Prefer fee-enabled markets (maker rebates)
        """
        return self.filter_markets(
            min_volume_24h=Decimal("1000"),
        )

    def get_crypto_15min_markets(self) -> list[GammaMarket]:
        """Get 15-minute crypto markets for Layer 6 (Temporal Crypto Arb)."""
        return [
            m
            for m in self._markets.values()
            if m.category == "crypto"
            and m.fee_enabled
            and ("15" in m.question.lower() or "minute" in m.question.lower())
        ]

    def get_negrisk_events(self) -> list[GammaMarket]:
        """Get NegRisk multi-outcome markets for Layer 3/5 monitoring."""
        return [m for m in self._markets.values() if m.neg_risk]
