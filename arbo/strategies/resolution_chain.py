"""Resolution chaining engine for Strategy C.

Manages capital flow across weather markets: when one city's weather market
resolves, capital is automatically redeployed to the next city in the chain.
City rotation follows timezone order for daily settlement coverage.

Chain: NYC → Chicago → London → Seoul → Buenos Aires
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from arbo.connectors.weather_models import City
from arbo.utils.logger import get_logger

logger = get_logger("resolution_chain")

# City chain ordered by timezone (earliest UTC offset first for daily coverage)
CITY_CHAIN_ORDER: list[City] = [
    City.BUENOS_AIRES,  # UTC-3
    City.NYC,           # UTC-5
    City.CHICAGO,       # UTC-6
    City.LONDON,        # UTC+0
    City.SEOUL,         # UTC+9
]


class ChainStatus(str, Enum):
    """Status of a resolution chain."""

    ACTIVE = "active"
    WAITING_RESOLUTION = "waiting_resolution"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    HALTED = "halted"


@dataclass
class ChainResolution:
    """Record of a single resolution within a chain."""

    city: City
    market_id: str
    resolved_at: datetime
    pnl: Decimal
    capital_before: Decimal
    capital_after: Decimal


@dataclass
class ResolutionChainState:
    """State of a single resolution chain."""

    chain_id: str
    city_sequence: list[City]
    current_city_index: int = 0
    initial_capital: Decimal = Decimal("0")
    current_capital: Decimal = Decimal("0")
    cumulative_pnl: Decimal = Decimal("0")
    num_resolutions: int = 0
    status: ChainStatus = ChainStatus.ACTIVE
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    resolutions: list[ChainResolution] = field(default_factory=list)
    active_market_id: str | None = None

    @property
    def current_city(self) -> City | None:
        """Current city in the chain, or None if completed."""
        if self.current_city_index >= len(self.city_sequence):
            return None
        return self.city_sequence[self.current_city_index]

    @property
    def next_city(self) -> City | None:
        """Next city in the chain, or None if this is the last."""
        next_idx = self.current_city_index + 1
        if next_idx >= len(self.city_sequence):
            return None
        return self.city_sequence[next_idx]

    @property
    def is_complete(self) -> bool:
        """Whether chain has cycled through all cities."""
        return self.current_city_index >= len(self.city_sequence)

    @property
    def roi(self) -> Decimal:
        """Return on investment for this chain."""
        if self.initial_capital == 0:
            return Decimal("0")
        return self.cumulative_pnl / self.initial_capital

    def to_db_dict(self) -> dict:
        """Convert to dict for DB persistence."""
        return {
            "chain_id": self.chain_id,
            "city_sequence": [c.value for c in self.city_sequence],
            "current_city_index": self.current_city_index,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "cumulative_pnl": self.cumulative_pnl,
            "num_resolutions": self.num_resolutions,
            "status": self.status.value,
        }


class ResolutionChainEngine:
    """Manages resolution chains for weather strategy capital rotation."""

    def __init__(self) -> None:
        self._chains: dict[str, ResolutionChainState] = {}

    def start_chain(
        self,
        capital: Decimal,
        city_sequence: list[City] | None = None,
    ) -> ResolutionChainState:
        """Start a new resolution chain.

        Args:
            capital: Initial capital to deploy.
            city_sequence: Custom city order, or default CITY_CHAIN_ORDER.

        Returns:
            New ResolutionChainState.
        """
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        sequence = city_sequence or CITY_CHAIN_ORDER.copy()

        chain = ResolutionChainState(
            chain_id=chain_id,
            city_sequence=sequence,
            initial_capital=capital,
            current_capital=capital,
            status=ChainStatus.ACTIVE,
        )
        self._chains[chain_id] = chain

        logger.info(
            "chain_started",
            chain_id=chain_id,
            capital=str(capital),
            cities=[c.value for c in sequence],
            first_city=sequence[0].value if sequence else "none",
        )
        return chain

    def set_active_market(self, chain_id: str, market_id: str) -> bool:
        """Mark which market the chain is currently deployed in.

        Args:
            chain_id: Chain identifier.
            market_id: The Polymarket condition_id.

        Returns:
            True if updated, False if chain not found.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            return False

        chain.active_market_id = market_id
        chain.status = ChainStatus.WAITING_RESOLUTION
        logger.info(
            "chain_deployed",
            chain_id=chain_id,
            market_id=market_id,
            city=chain.current_city.value if chain.current_city else "none",
        )
        return True

    def resolve(
        self,
        chain_id: str,
        market_id: str,
        pnl: Decimal,
    ) -> City | None:
        """Handle a market resolution within a chain.

        Records the resolution, updates capital, and advances to next city.

        Args:
            chain_id: Chain identifier.
            market_id: The resolved market's condition_id.
            pnl: Realized P&L from the resolution.

        Returns:
            Next City to deploy to, or None if chain is complete.

        Raises:
            ValueError: If chain not found or market mismatch.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found")

        if chain.active_market_id and chain.active_market_id != market_id:
            logger.warning(
                "chain_market_mismatch",
                chain_id=chain_id,
                expected=chain.active_market_id,
                got=market_id,
            )

        current_city = chain.current_city
        capital_before = chain.current_capital

        # Record resolution
        resolution = ChainResolution(
            city=current_city or City.NYC,
            market_id=market_id,
            resolved_at=datetime.now(timezone.utc),
            pnl=pnl,
            capital_before=capital_before,
            capital_after=capital_before + pnl,
        )
        chain.resolutions.append(resolution)

        # Update state
        chain.current_capital = capital_before + pnl
        chain.cumulative_pnl += pnl
        chain.num_resolutions += 1
        chain.active_market_id = None

        logger.info(
            "chain_resolution",
            chain_id=chain_id,
            city=current_city.value if current_city else "unknown",
            pnl=str(pnl),
            cumulative_pnl=str(chain.cumulative_pnl),
            capital=str(chain.current_capital),
        )

        # Advance to next city
        chain.current_city_index += 1

        if chain.is_complete:
            chain.status = ChainStatus.COMPLETED
            chain.completed_at = datetime.now(timezone.utc)
            logger.info(
                "chain_completed",
                chain_id=chain_id,
                total_pnl=str(chain.cumulative_pnl),
                roi=f"{chain.roi*100:.1f}%",
                num_resolutions=chain.num_resolutions,
            )
            return None

        # Check if capital is still viable (> $1)
        if chain.current_capital < Decimal("1"):
            chain.status = ChainStatus.HALTED
            logger.warning(
                "chain_halted_no_capital",
                chain_id=chain_id,
                remaining=str(chain.current_capital),
            )
            return None

        next_city = chain.current_city
        chain.status = ChainStatus.DEPLOYING
        logger.info(
            "chain_advancing",
            chain_id=chain_id,
            next_city=next_city.value if next_city else "none",
            capital=str(chain.current_capital),
        )
        return next_city

    def get_chain(self, chain_id: str) -> ResolutionChainState | None:
        """Get current state of a chain."""
        return self._chains.get(chain_id)

    def get_active_chains(self) -> list[ResolutionChainState]:
        """Get all chains that are not completed or halted."""
        return [
            c
            for c in self._chains.values()
            if c.status not in (ChainStatus.COMPLETED, ChainStatus.HALTED)
        ]

    def get_all_chains(self) -> list[ResolutionChainState]:
        """Get all chains."""
        return list(self._chains.values())

    def halt_chain(self, chain_id: str, reason: str) -> bool:
        """Halt a chain (e.g., due to strategy kill switch).

        Returns True if halted, False if chain not found.
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            return False

        chain.status = ChainStatus.HALTED
        logger.warning("chain_halted", chain_id=chain_id, reason=reason)
        return True
