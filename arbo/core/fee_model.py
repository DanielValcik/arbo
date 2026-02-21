"""Polymarket fee model implementation.

Dynamic fees apply ONLY to fee-enabled markets. Most sports markets have 0% fee.
Fee formula: fee = price * (1 - price) * fee_rate

See brief Section 3 for full specification.
"""

from __future__ import annotations

from decimal import Decimal

# Maximum taker fee rate (as of February 2026)
DEFAULT_FEE_RATE = Decimal("0.0315")

# Sports-specific fee parameters
SPORTS_FEE_RATE = Decimal("0.0175")
SPORTS_EXPONENT = 1

# Crypto-specific fee parameters
CRYPTO_FEE_RATE = Decimal("0.25")
CRYPTO_EXPONENT = 2

# Maker rebate percentages (of total taker fees)
SPORTS_MAKER_REBATE_PCT = Decimal("0.25")
CRYPTO_MAKER_REBATE_PCT = Decimal("0.20")


def calculate_taker_fee(
    price: Decimal,
    fee_enabled: bool,
    fee_rate: Decimal = DEFAULT_FEE_RATE,
) -> Decimal:
    """Calculate Polymarket taker fee for a given price.

    Args:
        price: Market price (0 to 1).
        fee_enabled: Whether this market has fees enabled.
        fee_rate: Maximum taker fee rate.

    Returns:
        Fee amount in USDC per share.

    Examples:
        At p=0.50: 0.50 * 0.50 * 0.0315 = 0.007875 (0.79%)
        At p=0.95: 0.95 * 0.05 * 0.0315 = 0.001496 (0.15%)
        At p=0.05: 0.05 * 0.95 * 0.0315 = 0.001496 (0.15%)
    """
    if not fee_enabled:
        return Decimal("0")

    if price <= 0 or price >= 1:
        return Decimal("0")

    return price * (Decimal("1") - price) * fee_rate


def calculate_edge_after_fee(
    model_prob: Decimal,
    market_price: Decimal,
    fee_enabled: bool,
    fee_rate: Decimal = DEFAULT_FEE_RATE,
) -> Decimal:
    """Calculate edge after accounting for fees.

    Args:
        model_prob: Our estimated probability.
        market_price: Current Polymarket price.
        fee_enabled: Whether this market has fees.
        fee_rate: Fee rate for fee-enabled markets.

    Returns:
        Edge after fee deduction. Positive = profitable opportunity.
    """
    raw_edge = abs(model_prob - market_price)
    fee = calculate_taker_fee(market_price, fee_enabled, fee_rate)
    return raw_edge - fee


def estimate_maker_rebate(
    price: Decimal,
    size: Decimal,
    market_type: str = "sports",
) -> Decimal:
    """Estimate maker rebate for providing liquidity.

    Args:
        price: Order price.
        size: Order size in USDC.
        market_type: "sports" or "crypto" for rebate rate selection.

    Returns:
        Estimated rebate in USDC (daily, proportional to volume share).
    """
    if market_type == "crypto":
        fee_rate = CRYPTO_FEE_RATE
        rebate_pct = CRYPTO_MAKER_REBATE_PCT
    else:
        fee_rate = SPORTS_FEE_RATE
        rebate_pct = SPORTS_MAKER_REBATE_PCT

    # Fee equivalent for this order
    fee_equivalent = price * (Decimal("1") - price) * fee_rate * size

    # Rebate is a percentage of total taker fees in the pool
    # This is an estimate — actual depends on your share of total liquidity
    return fee_equivalent * rebate_pct


def is_fee_favorable(price: Decimal, fee_enabled: bool) -> bool:
    """Check if fee at this price is acceptable for trading.

    At extreme probabilities (>0.95 or <0.05), fees drop below 0.3%,
    making even fee-enabled markets viable for latency arb.
    """
    if not fee_enabled:
        return True

    fee = calculate_taker_fee(price, fee_enabled=True)
    fee_pct = fee / price if price > 0 else Decimal("1")
    return fee_pct < Decimal("0.003")  # 0.3% threshold
