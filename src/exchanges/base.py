"""Base exchange client ABC and shared Pydantic DTOs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime  # noqa: TC003 — Pydantic needs at runtime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

# --- Enums ---


class Side(StrEnum):
    BACK = "back"
    LAY = "lay"


class BetStatus(StrEnum):
    PENDING = "pending"
    MATCHED = "matched"
    PARTIAL = "partial"
    SETTLED = "settled"
    CANCELLED = "cancelled"


class EventStatus(StrEnum):
    UPCOMING = "upcoming"
    LIVE = "live"
    SETTLED = "settled"
    CANCELLED = "cancelled"


# --- DTOs ---


class RunnerPrice(BaseModel):
    """A single price point on a runner."""

    odds: Decimal
    available_amount: Decimal
    side: Side


class Runner(BaseModel):
    """A selectable outcome within a market (e.g., Home, Away, Draw)."""

    runner_id: int
    name: str
    prices: list[RunnerPrice] = Field(default_factory=list)

    @property
    def best_back(self) -> RunnerPrice | None:
        """Highest back price available."""
        backs = [p for p in self.prices if p.side == Side.BACK]
        return max(backs, key=lambda p: p.odds) if backs else None

    @property
    def best_lay(self) -> RunnerPrice | None:
        """Lowest lay price available."""
        lays = [p for p in self.prices if p.side == Side.LAY]
        return min(lays, key=lambda p: p.odds) if lays else None


class Market(BaseModel):
    """A betting market on an event (e.g., Match Odds, Over/Under 2.5)."""

    market_id: int
    name: str
    market_type: str  # h2h, spreads, totals
    runners: list[Runner] = Field(default_factory=list)


class ExchangeEvent(BaseModel):
    """An event from an exchange."""

    event_id: int
    name: str
    sport: str
    league: str | None = None
    home_team: str
    away_team: str
    start_time: datetime
    status: EventStatus = EventStatus.UPCOMING
    markets: list[Market] = Field(default_factory=list)


class BetParams(BaseModel):
    """Parameters for placing a bet."""

    event_id: int
    market_id: int
    runner_id: int
    side: Side
    odds: Decimal
    stake: Decimal


class BetResult(BaseModel):
    """Result of placing a bet."""

    offer_id: str
    status: BetStatus
    odds: Decimal
    stake: Decimal
    matched_amount: Decimal = Decimal("0")
    fill_pct: Decimal = Decimal("0")


# --- ABC ---


class BaseExchangeClient(ABC):
    """Abstract base class for exchange clients."""

    @abstractmethod
    async def login(self) -> None:
        """Authenticate with the exchange."""

    @abstractmethod
    async def get_events(
        self, sport: str, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> list[ExchangeEvent]:
        """Fetch events for a sport within a date range."""

    @abstractmethod
    async def get_markets(self, event_id: int) -> list[Market]:
        """Fetch markets for an event."""

    @abstractmethod
    async def get_prices(self, event_id: int, market_id: int) -> list[Runner]:
        """Fetch runner prices for a market."""

    @abstractmethod
    async def place_bet(self, params: BetParams) -> BetResult:
        """Place a bet on the exchange."""

    @abstractmethod
    async def cancel_bet(self, offer_id: str) -> bool:
        """Cancel an unmatched bet."""

    @abstractmethod
    async def get_open_bets(self) -> list[BetResult]:
        """Get all open/unmatched bets."""
