"""Odds conversion and arbitrage math utilities.

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


def fractional_to_decimal(num: int, den: int) -> Decimal:
    """Convert fractional odds to decimal. 5/2 → 3.50."""
    if den <= 0:
        raise ValueError(f"Denominator must be positive, got {den}")
    if num < 0:
        raise ValueError(f"Numerator must be non-negative, got {num}")
    return Decimal(1) + Decimal(num) / Decimal(den)


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


def commission_adjusted_odds(odds: Decimal, commission: Decimal) -> Decimal:
    """Adjust exchange odds for commission. Net odds after commission.

    Formula: 1 + (odds - 1) * (1 - commission)
    Example: odds=2.10, commission=0.04 → 1 + 1.10 * 0.96 = 2.056
    """
    return Decimal(1) + (odds - Decimal(1)) * (Decimal(1) - commission)


def arb_margin(
    back_odds: Decimal,
    lay_odds: Decimal,
    commission: Decimal,
) -> Decimal:
    """Calculate arbitrage margin between back and lay odds.

    Returns: 1/back - 1/adjusted_lay. Negative value means arb exists.
    """
    adj_lay = commission_adjusted_odds(lay_odds, commission)
    return Decimal(1) / back_odds + Decimal(1) / adj_lay - Decimal(1)


def half_kelly(edge: Decimal, odds: Decimal) -> Decimal:
    """Calculate half-Kelly stake fraction.

    Formula: 0.5 * edge / (odds - 1)
    Returns fraction of bankroll to stake.
    """
    if odds <= 1:
        raise ValueError(f"Odds must be > 1 for Kelly calculation, got {odds}")
    return Decimal("0.5") * edge / (odds - Decimal(1))
