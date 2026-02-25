"""Market discovery module using Polymarket Gamma API.

Automatically discovers and catalogs active markets with filtering
by category, liquidity, fee status, and NegRisk flag.

See brief PM-002 for full specification.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
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
    "weather": [
        "temperature",
        "weather",
        "degrees",
        "fahrenheit",
        "celsius",
        "high temperature",
        "low temperature",
        "°f",
        "°c",
    ],
    "attention_markets": [
        "mindshare",
        "attention",
        "kaito",
        "sentiment",
        "social",
        "trending",
        "popularity",
        "followers",
        "views",
        "engagement",
    ],
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


# ---------------------------------------------------------------------------
# Crypto market parsing (reuses patterns from temporal_crypto.py)
# ---------------------------------------------------------------------------

# Symbol extraction keyword map (mirrored from temporal_crypto.py:SYMBOL_MAP)
_CRYPTO_SYMBOL_MAP: dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "eth": "ETHUSDT",
    "solana": "SOLUSDT",
    "sol": "SOLUSDT",
    "dogecoin": "DOGEUSDT",
    "doge": "DOGEUSDT",
    "xrp": "XRPUSDT",
    "cardano": "ADAUSDT",
    "ada": "ADAUSDT",
    "avalanche": "AVAXUSDT",
    "avax": "AVAXUSDT",
    "chainlink": "LINKUSDT",
    "link": "LINKUSDT",
    "polkadot": "DOTUSDT",
    "dot": "DOTUSDT",
}

# Strike price patterns (mirrored from temporal_crypto.py:_STRIKE_PATTERNS)
_STRIKE_PATTERNS = [
    re.compile(r"\$([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)"),  # $95,000
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)\s*k", re.IGNORECASE),  # $95k
    re.compile(r"\$([0-9]+(?:\.[0-9]+)?)"),  # $95000
]

# Date patterns for expiry parsing
_DATE_PATTERNS = [
    # "March 1" / "February 28" / "March 1, 2026"
    re.compile(
        r"(?:by|on|before)?\s*"
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
        re.IGNORECASE,
    ),
    # "Feb 28" / "Mar 1"
    re.compile(
        r"(?:by|on|before)?\s*"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
        re.IGNORECASE,
    ),
]

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# 5-min market detection
_5MIN_PATTERN = re.compile(r"5\s*min|up or down", re.IGNORECASE)


@dataclass
class CryptoMarketInfo:
    """Parsed metadata from a crypto price-threshold market question."""

    asset: str  # e.g. "BTC"
    symbol: str  # Binance symbol e.g. "BTCUSDT"
    strike: Decimal
    expiry: datetime | None
    direction: str  # "above" or "below"
    is_5min: bool


def categorize_crypto_market(question: str) -> CryptoMarketInfo | None:
    """Parse a crypto market question into structured CryptoMarketInfo.

    Handles:
    - "Will BTC be above $100,000 on March 1?"
    - "Will ETH be above $4,000 by February 28?"
    - "Bitcoin Up or Down - February 23, 1:25PM" (5-min markets)
    - "Will Solana be above $200 on March 15?"

    Args:
        question: Market question text.

    Returns:
        CryptoMarketInfo or None if not a parseable crypto market.
    """
    q_lower = question.lower()

    # 1. Extract asset/symbol
    symbol: str | None = None
    asset: str = ""
    for keyword, sym in _CRYPTO_SYMBOL_MAP.items():
        if keyword in q_lower:
            symbol = sym
            asset = sym.replace("USDT", "")
            break

    if symbol is None:
        return None

    # 2. Extract strike price
    strike: Decimal | None = None
    for pattern in _STRIKE_PATTERNS:
        match = pattern.search(question)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                value = Decimal(value_str)
                # Handle k notation
                full_match = question[match.start() : match.end() + 2]
                if re.search(r"k\b", full_match, re.IGNORECASE):
                    value *= 1000
                strike = value
                break
            except Exception:
                continue

    # 5-min markets may not have a strike
    is_5min = bool(_5MIN_PATTERN.search(question))

    if strike is None and not is_5min:
        return None

    # 3. Parse expiry date
    expiry: datetime | None = None
    for pattern in _DATE_PATTERNS:
        match = pattern.search(question)
        if match:
            month_str = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else datetime.now().year
            month = _MONTH_MAP.get(month_str)
            if month:
                with contextlib.suppress(ValueError):
                    expiry = datetime(year, month, day)
            break

    # 4. Direction
    direction = "below" if "below" in q_lower or "down" in q_lower else "above"

    return CryptoMarketInfo(
        asset=asset,
        symbol=symbol,
        strike=strike or Decimal("0"),
        expiry=expiry,
        direction=direction,
        is_5min=is_5min,
    )


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
        # Parse end_date string to datetime for PostgreSQL
        end_date_dt = None
        if self.end_date:
            try:
                end_date_dt = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

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
            "end_date": end_date_dt,
            "active": self.active and not self.closed,
            "last_price_yes": float(self.price_yes) if self.price_yes else None,
            "last_price_no": float(self.price_no) if self.price_no else None,
        }


# Sports series IDs from Gamma API /sports endpoint (verified 2026-02-22)
SPORTS_SERIES: dict[str, int] = {
    "epl": 10188,
    "la_liga": 10193,
    "bundesliga": 10194,
    "serie_a": 10203,
    "ligue_1": 10195,
    "ucl": 10204,
    "europa_league": 10209,
}


class MarketDiscovery:
    """Discovers and catalogs active Polymarket markets via Gamma API.

    Features:
    - Paginated fetching of all active markets
    - Sports match-level events via /events endpoint (series-based)
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

    async def _fetch_sports_events(self) -> list[GammaMarket]:
        """Fetch match-level sports events via /events endpoint.

        Sports markets use a series-based system on the Events endpoint,
        NOT the Markets endpoint. Each soccer league has a series ID.
        Events contain nested market arrays with moneyline/spreads/totals.

        Returns:
            List of GammaMarket objects from sports events.
        """
        if not self._session:
            return []

        all_markets: list[GammaMarket] = []
        for league, series_id in SPORTS_SERIES.items():
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "series_id": str(series_id),
                    "limit": "50",
                }
                url = f"{self._gamma_url}/events"
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "sports_events_error",
                            league=league,
                            status=resp.status,
                        )
                        continue
                    events = await resp.json()
                    if not isinstance(events, list):
                        continue

                    for event in events:
                        nested = event.get("markets", [])
                        if not isinstance(nested, list):
                            continue
                        for raw_market in nested:
                            if not isinstance(raw_market, dict):
                                continue
                            # Skip closed/inactive
                            if raw_market.get("closed", False):
                                continue
                            if not raw_market.get("active", True):
                                continue
                            market = GammaMarket(raw_market)
                            # Force soccer category for sports events
                            market.category = "soccer"
                            if market.condition_id and market.active:
                                all_markets.append(market)

                logger.debug(
                    "sports_events_fetched",
                    league=league,
                    events=len(events),
                    markets=sum(1 for e in events for _ in (e.get("markets") or [])),
                )
            except Exception as e:
                logger.warning("sports_events_exception", league=league, error=str(e))
            await asyncio.sleep(0.1)

        if all_markets:
            logger.info(
                "sports_match_markets_found",
                total=len(all_markets),
                leagues=len(SPORTS_SERIES),
            )
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

        # Fetch sports match-level events (moneyline, spreads, totals)
        sports_markets = await self._fetch_sports_events()
        markets.extend(sports_markets)

        self._markets = {m.condition_id: m for m in markets}
        self._last_refresh = now

        # Sync to DB for dashboard JOINs
        await self._sync_markets_to_db(markets)

        # Log category breakdown
        categories: dict[str, int] = {}
        for m in markets:
            categories[m.category] = categories.get(m.category, 0) + 1
        logger.info("market_catalog_refreshed", total=len(markets), categories=categories)

        return markets

    async def _sync_markets_to_db(self, markets: list[GammaMarket]) -> None:
        """Sync discovered markets to the PostgreSQL markets table.

        Uses upsert (INSERT ON CONFLICT UPDATE) so dashboard JOINs with
        PaperTrade/PaperPosition can resolve question + category.
        """
        try:
            import sqlalchemy as sa

            from arbo.utils.db import Market, get_session_factory

            factory = get_session_factory()
            async with factory() as session:
                for m in markets:
                    db_dict = m.to_db_dict()
                    stmt = (
                        sa.dialects.postgresql.insert(Market)
                        .values(**db_dict)
                        .on_conflict_do_update(
                            index_elements=["condition_id"],
                            set_={
                                "question": db_dict["question"],
                                "category": db_dict["category"],
                                "volume_24h": db_dict["volume_24h"],
                                "liquidity": db_dict["liquidity"],
                                "active": db_dict["active"],
                                "last_price_yes": db_dict["last_price_yes"],
                                "last_price_no": db_dict["last_price_no"],
                            },
                        )
                    )
                    await session.execute(stmt)
                await session.commit()
            logger.info("markets_synced_to_db", count=len(markets))
        except Exception as e:
            logger.warning("market_db_sync_failed", error=str(e))

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

    def get_crypto_markets(self) -> list[GammaMarket]:
        """Get active crypto markets with price in tradeable range (5-95 cents).

        Filters for category=crypto, active, binary, and price within 0.05-0.95.
        """
        return [
            m
            for m in self._markets.values()
            if m.category == "crypto"
            and m.active
            and not m.closed
            and m.price_yes is not None
            and Decimal("0.05") <= m.price_yes <= Decimal("0.95")
            and len(m.outcomes) == 2
        ]

    def get_politics_markets(self) -> list[GammaMarket]:
        """Get active politics markets with price in tradeable range (5-95 cents).

        Filters for category=politics, active, binary, and price within 0.05-0.95.
        """
        return [
            m
            for m in self._markets.values()
            if m.category == "politics"
            and m.active
            and not m.closed
            and m.price_yes is not None
            and Decimal("0.05") <= m.price_yes <= Decimal("0.95")
            and len(m.outcomes) == 2
        ]

    def get_soccer_match_markets(self) -> list[GammaMarket]:
        """Get match-level soccer markets (moneyline: 'Will X win on date?').

        These come from the /events endpoint via sports series IDs.
        Filters for category=soccer, active, binary, price 5-95 cents,
        and question pattern indicating a match (not seasonal/outright).
        """
        match_keywords = ("win on", "beat", "end in a draw", "spread:", "o/u ", "both teams")
        return [
            m
            for m in self._markets.values()
            if m.category == "soccer"
            and m.active
            and not m.closed
            and m.price_yes is not None
            and Decimal("0.05") <= m.price_yes <= Decimal("0.95")
            and len(m.outcomes) == 2
            and any(kw in m.question.lower() for kw in match_keywords)
        ]

    async def fetch_by_token_id(self, token_id: str) -> GammaMarket | None:
        """Fetch a specific market by CLOB token ID from Gamma API.

        Works regardless of active/closed status. Used by resolution checker
        to detect resolved (closed) markets dropped from the active-only cache.
        """
        if not self._session:
            return None
        try:
            url = f"{self._gamma_url}/markets"
            params = {"clob_token_ids": token_id, "limit": "1"}
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if isinstance(data, list) and data:
                    return GammaMarket(data[0])
        except Exception as e:
            logger.debug("fetch_by_token_id_error", token_id=token_id[:20], error=str(e))
        return None

    def get_all_active(self) -> list[GammaMarket]:
        """Get all active, non-closed markets."""
        return [m for m in self._markets.values() if m.active and not m.closed]

    def get_negrisk_events(self) -> list[GammaMarket]:
        """Get NegRisk multi-outcome markets for Layer 3/5 monitoring."""
        return [m for m in self._markets.values() if m.neg_risk]
