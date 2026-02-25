"""Temperature laddering logic for Strategy C.

Places parallel bets across adjacent temperature buckets based on forecast
distribution. Uses Quarter-Kelly sizing per bucket, respecting per-strategy
allocation limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from arbo.core.risk_manager import KELLY_FRACTION
from arbo.strategies.weather_scanner import WeatherSignal
from arbo.utils.logger import get_logger

logger = get_logger("weather_ladder")

# Maximum concurrent ladder positions per city per day
MAX_LADDER_POSITIONS = 3
# Minimum edge to include a bucket in the ladder
MIN_LADDER_EDGE = 0.03


@dataclass
class LadderPosition:
    """A single position in a temperature ladder."""

    signal: WeatherSignal
    size_usdc: Decimal
    kelly_fraction: Decimal
    priority: int  # Lower = higher priority (1 = best edge)


@dataclass
class TemperatureLadder:
    """A set of positions across adjacent temperature buckets for one city/date."""

    city: str
    target_date: str
    positions: list[LadderPosition]
    total_size_usdc: Decimal

    @property
    def num_positions(self) -> int:
        return len(self.positions)


def calculate_kelly_size(
    edge: float,
    price: float,
    available_capital: Decimal,
    kelly_fraction: Decimal = KELLY_FRACTION,
    max_position_size: Decimal | None = None,
) -> Decimal:
    """Calculate Quarter-Kelly position size.

    Kelly formula: f* = (p*b - q) / b
    where p = probability of winning, b = odds, q = 1 - p

    For binary markets:
        p = forecast_probability (our edge-adjusted probability)
        b = (1/price) - 1  (decimal odds minus 1)

    We use Quarter-Kelly (KELLY_FRACTION = 0.25) for conservative sizing.

    Args:
        edge: Expected edge (forecast_prob - market_price).
        price: Market price of the YES token.
        available_capital: Capital available for this strategy.
        kelly_fraction: Kelly fraction multiplier (default 0.25).
        max_position_size: Hard cap on position size (from risk manager).

    Returns:
        Position size in USDC.
    """
    if price <= 0 or price >= 1 or edge <= 0:
        return Decimal("0")

    # Binary market Kelly
    prob = price + edge  # Our estimated true probability
    if prob <= 0 or prob >= 1:
        return Decimal("0")

    odds = (1.0 / price) - 1.0  # Decimal odds minus 1
    kelly_raw = (prob * odds - (1 - prob)) / odds

    if kelly_raw <= 0:
        return Decimal("0")

    # Apply Kelly fraction (Quarter-Kelly)
    kelly_adjusted = kelly_raw * float(kelly_fraction)

    # Calculate size
    size = available_capital * Decimal(str(kelly_adjusted))

    # Cap at max position size (risk manager limit)
    if max_position_size is not None and size > max_position_size:
        size = max_position_size

    # Round down to 2 decimal places (USDC precision)
    return size.quantize(Decimal("0.01"))


def build_temperature_ladder(
    signals: list[WeatherSignal],
    available_capital: Decimal,
    max_positions: int = MAX_LADDER_POSITIONS,
    min_edge: float = MIN_LADDER_EDGE,
    max_position_size: Decimal | None = None,
) -> TemperatureLadder | None:
    """Build a temperature ladder from a set of weather signals.

    Groups signals by city+date, takes the top N by edge, and sizes each
    position using Quarter-Kelly.

    Args:
        signals: Weather signals (should be for the same city/date).
        available_capital: Capital available for this strategy.
        max_positions: Maximum positions in the ladder.
        min_edge: Minimum edge to include.

    Returns:
        TemperatureLadder or None if no valid positions.
    """
    if not signals:
        return None

    # Filter by minimum edge
    valid = [s for s in signals if s.edge >= min_edge]
    if not valid:
        return None

    # Sort by edge descending, take top N
    valid.sort(key=lambda s: s.edge, reverse=True)
    selected = valid[:max_positions]

    # Calculate position sizes
    positions = []
    total_size = Decimal("0")
    remaining_capital = available_capital

    for i, signal in enumerate(selected):
        size = calculate_kelly_size(
            edge=signal.edge,
            price=signal.market.market_price,
            available_capital=remaining_capital,
            max_position_size=max_position_size,
        )

        if size < Decimal("1"):  # Minimum $1 position
            continue

        position = LadderPosition(
            signal=signal,
            size_usdc=size,
            kelly_fraction=KELLY_FRACTION,
            priority=i + 1,
        )
        positions.append(position)
        total_size += size
        remaining_capital -= size

    if not positions:
        return None

    city = positions[0].signal.market.city.value
    target_date = str(positions[0].signal.market.target_date)

    logger.info(
        "temperature_ladder_built",
        city=city,
        date=target_date,
        positions=len(positions),
        total_size=str(total_size),
    )

    return TemperatureLadder(
        city=city,
        target_date=target_date,
        positions=positions,
        total_size_usdc=total_size,
    )


def build_ladders_by_city(
    signals: list[WeatherSignal],
    available_capital: Decimal,
    max_positions_per_city: int = MAX_LADDER_POSITIONS,
    max_position_size: Decimal | None = None,
) -> list[TemperatureLadder]:
    """Build temperature ladders for each city/date combination.

    Args:
        signals: All weather signals across cities.
        available_capital: Total capital available for weather strategy.
        max_positions_per_city: Max positions per city.

    Returns:
        List of TemperatureLadder objects.
    """
    # Group signals by (city, date)
    groups: dict[tuple[str, str], list[WeatherSignal]] = {}
    for signal in signals:
        key = (signal.market.city.value, str(signal.market.target_date))
        groups.setdefault(key, []).append(signal)

    # Build ladders, distributing capital proportionally
    ladders = []
    num_groups = len(groups)
    if num_groups == 0:
        return []

    per_group_capital = available_capital / num_groups

    for (city, date_str), group_signals in groups.items():
        ladder = build_temperature_ladder(
            signals=group_signals,
            available_capital=per_group_capital,
            max_positions=max_positions_per_city,
            max_position_size=max_position_size,
        )
        if ladder:
            ladders.append(ladder)

    logger.info(
        "ladders_built",
        total_ladders=len(ladders),
        total_groups=num_groups,
    )
    return ladders
