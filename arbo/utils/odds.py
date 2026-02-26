"""Odds conversion, Kelly sizing, and Polymarket fee utilities.

All functions use Decimal for precision in financial calculations.
"""

from __future__ import annotations

from decimal import Decimal


def decimal_to_implied(odds: Decimal) -> Decimal:
    """Convert decimal odds to implied probability. 2.00 → 0.50."""
    if odds <= 0:
        raise ValueError(f"Decimal odds must be positive, got {odds}")
    return Decimal(1) / odds


def implied_to_decimal(prob: Decimal) -> Decimal:
    """Convert implied probability to decimal odds. 0.50 → 2.00."""
    if prob <= 0 or prob > 1:
        raise ValueError(f"Implied probability must be in (0, 1], got {prob}")
    return Decimal(1) / prob


def american_to_decimal(american: int) -> Decimal:
    """Convert American odds to decimal. +150 → 2.50, -200 → 1.50."""
    if american == 0:
        raise ValueError("American odds cannot be zero")
    if american > 0:
        return Decimal(1) + Decimal(american) / Decimal(100)
    return Decimal(1) + Decimal(100) / Decimal(abs(american))


def overround(probs: list[Decimal]) -> Decimal:
    """Calculate overround (sum of implied probabilities). >1 means bookmaker margin."""
    if not probs:
        raise ValueError("Probabilities list cannot be empty")
    return sum(probs, Decimal(0))


def remove_vig(probs: list[Decimal]) -> list[Decimal]:
    """Remove vigorish by normalizing probabilities to sum to 1."""
    if not probs:
        raise ValueError("Probabilities list cannot be empty")
    total = sum(probs, Decimal(0))
    if total == 0:
        raise ValueError("Sum of probabilities cannot be zero")
    return [p / total for p in probs]


def polymarket_price_to_decimal_odds(price: Decimal) -> Decimal:
    """Convert Polymarket price (0-1) to decimal odds. 0.45 → 2.222."""
    if price <= 0 or price >= 1:
        raise ValueError(f"Polymarket price must be in (0, 1), got {price}")
    return Decimal(1) / price


def half_kelly(
    model_prob: Decimal,
    market_price: Decimal,
    max_position_pct: Decimal = Decimal("0.05"),
) -> Decimal:
    """Calculate half-Kelly position size for Polymarket.

    Formula: full_kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
             position_size = min(full_kelly * 0.5, max_position_pct)

    Args:
        model_prob: Our estimated probability (0-1).
        market_price: Polymarket price (0-1).
        max_position_pct: Maximum position as fraction of capital.

    Returns:
        Fraction of capital to allocate (0 to max_position_pct).
    """
    if market_price <= 0 or market_price >= 1:
        return Decimal(0)
    if model_prob <= 0 or model_prob >= 1:
        return Decimal(0)

    decimal_odds = Decimal(1) / market_price
    full_kelly = (model_prob * decimal_odds - Decimal(1)) / (decimal_odds - Decimal(1))

    if full_kelly <= 0:
        return Decimal(0)

    half = full_kelly * Decimal("0.5")
    return min(half, max_position_pct)


def quarter_kelly(
    model_prob: Decimal,
    market_price: Decimal,
    max_position_pct: Decimal = Decimal("0.05"),
) -> Decimal:
    """Calculate quarter-Kelly position size for Polymarket.

    More conservative than half-Kelly — used by Strategy A (Theta Decay)
    and Strategy C (Compound Weather) for smaller, higher-frequency positions.

    Args:
        model_prob: Our estimated probability (0-1).
        market_price: Polymarket price (0-1).
        max_position_pct: Maximum position as fraction of capital.

    Returns:
        Fraction of capital to allocate (0 to max_position_pct).
    """
    if market_price <= 0 or market_price >= 1:
        return Decimal(0)
    if model_prob <= 0 or model_prob >= 1:
        return Decimal(0)

    decimal_odds = Decimal(1) / market_price
    full_kelly = (model_prob * decimal_odds - Decimal(1)) / (decimal_odds - Decimal(1))

    if full_kelly <= 0:
        return Decimal(0)

    quarter = full_kelly * Decimal("0.25")
    return min(quarter, max_position_pct)
