"""Volatility-based probability model for crypto price prediction markets.

Estimates P(price >= K by time T) using log-normal returns with realized
volatility. Supports two market types:

1. Daily "Above": Standard GBM CDF — P(S_T >= K)
2. Monthly "Hit": First-passage (barrier) probability — P(max(S_t) >= K for t in [0,T])

This replaces EMOS (weather forecast errors) for Strategy B2.
The analogy: closer to expiry → sigma shrinks → probability clarifies,
just like weather forecasts become more accurate near resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Use math.erf for normal CDF to avoid scipy dependency in production
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


# ── Volatility estimation ─────────────────────────────────────────


def compute_realized_vol(
    prices: list[float],
    window: int = 24,
    method: str = "realized",
) -> float:
    """Compute annualized volatility from price series.

    Args:
        prices: List of prices (most recent last). At least 2 required.
        window: Number of observations to use (from the end).
        method: "realized" (std of log returns), "ewma" (exponentially weighted),
                or "garch" (simplified GARCH(1,1)).

    Returns:
        Annualized volatility (sigma_daily expressed as daily proportion).
        For crypto: annualized vol of ~60% → daily vol of ~3.8%.
    """
    if len(prices) < 2:
        return 0.03  # Default ~3% daily vol for crypto

    # Use the last `window` prices
    p = prices[-window:] if len(prices) > window else prices

    # Compute log returns
    log_returns: list[float] = []
    for i in range(1, len(p)):
        if p[i] > 0 and p[i - 1] > 0:
            log_returns.append(math.log(p[i] / p[i - 1]))

    if not log_returns:
        return 0.03

    if method == "realized":
        return _realized_vol(log_returns)
    elif method == "ewma":
        return _ewma_vol(log_returns)
    elif method == "garch":
        return _garch_vol(log_returns)
    else:
        return _realized_vol(log_returns)


def _realized_vol(returns: list[float]) -> float:
    """Standard deviation of log returns (annualized to daily)."""
    n = len(returns)
    if n < 2:
        return 0.03
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(max(var, 1e-10))


def _ewma_vol(returns: list[float], alpha: float = 0.06) -> float:
    """Exponentially Weighted Moving Average volatility (RiskMetrics style).

    More responsive to recent changes than realized vol.
    Alpha=0.06 corresponds to ~16 observation half-life.
    """
    if not returns:
        return 0.03
    var = returns[0] ** 2
    for r in returns[1:]:
        var = alpha * r * r + (1 - alpha) * var
    return math.sqrt(max(var, 1e-10))


def _garch_vol(
    returns: list[float],
    omega: float = 1e-6,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> float:
    """Simplified GARCH(1,1) volatility estimation.

    σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    Captures volatility clustering typical of crypto markets.
    """
    if not returns:
        return 0.03
    var = sum(r * r for r in returns) / len(returns)  # Initialize with sample variance
    for r in returns:
        var = omega + alpha * r * r + beta * var
    return math.sqrt(max(var, 1e-10))


# ── Probability estimation ────────────────────────────────────────


def estimate_daily_prob(
    current_price: float,
    strike: float,
    hours_to_expiry: float,
    sigma_per_hour: float,
    sigma_scale: float = 1.0,
) -> float:
    """Estimate P(price >= strike at expiry) for daily "Above" markets.

    Uses log-normal (GBM) model:
        z = ln(K/S) / (σ·√T)
        P(S_T >= K) = 1 - Φ(z)    [zero drift assumed]

    Args:
        current_price: Current exchange price (e.g. Binance BTC/USDT).
        strike: Strike price threshold (e.g. 88000).
        hours_to_expiry: Hours until market resolves.
        sigma_per_hour: Hourly volatility (log-return std per hour).
        sigma_scale: Multiplier for sigma calibration (from autoresearch).

    Returns:
        Probability between 0 and 1.
    """
    if current_price <= 0 or strike <= 0 or hours_to_expiry <= 0:
        return 0.5

    sigma = sigma_per_hour * sigma_scale * math.sqrt(hours_to_expiry)

    if sigma < 1e-8:
        # Essentially deterministic
        return 1.0 if current_price >= strike else 0.0

    z = math.log(strike / current_price) / sigma
    return 1.0 - _norm_cdf(z)


def estimate_hit_prob(
    current_price: float,
    strike: float,
    hours_remaining: float,
    sigma_per_hour: float,
    sigma_scale: float = 1.0,
    direction: str = "above",
) -> float:
    """Estimate P(max(S_t) >= strike at any point in [0, T]) for monthly "Hit" markets.

    Uses the reflection principle for Brownian motion first-passage time:
        P(max(S_t) >= K, 0<=t<=T) ≈ 2·(1 - Φ(|ln(K/S)| / (σ√T)))

    This is the barrier option probability — "will price TOUCH K at any point?"
    Always >= daily_prob since the barrier is checked continuously.

    For downward barriers (direction="below"):
        P(min(S_t) <= K) = 2·(1 - Φ(|ln(S/K)| / (σ√T)))

    Args:
        current_price: Current exchange price.
        strike: Strike/barrier price.
        hours_remaining: Hours remaining in the period.
        sigma_per_hour: Hourly volatility.
        sigma_scale: Multiplier for calibration.
        direction: "above" (upward barrier) or "below" (downward barrier).

    Returns:
        Probability between 0 and 1.
    """
    if current_price <= 0 or strike <= 0 or hours_remaining <= 0:
        return 0.5

    # Already touched?
    if direction == "above" and current_price >= strike:
        return 1.0
    if direction == "below" and current_price <= strike:
        return 1.0

    sigma = sigma_per_hour * sigma_scale * math.sqrt(hours_remaining)

    if sigma < 1e-8:
        if direction == "above":
            return 1.0 if current_price >= strike else 0.0
        else:
            return 1.0 if current_price <= strike else 0.0

    # Distance to barrier in log space
    if direction == "above":
        log_dist = abs(math.log(strike / current_price))
    else:
        log_dist = abs(math.log(current_price / strike))

    # Reflection principle: P(first passage) = 2 * P(terminal > barrier)
    prob = 2.0 * (1.0 - _norm_cdf(log_dist / sigma))

    return max(0.0, min(1.0, prob))


def estimate_crypto_prob(
    current_price: float,
    strike: float,
    hours_to_expiry: float,
    sigma_per_hour: float,
    market_type: str = "daily_above",
    sigma_scale: float = 1.0,
    direction: str = "above",
) -> float:
    """Unified probability estimator for all crypto price market types.

    Args:
        current_price: Current exchange price (Binance).
        strike: Strike price threshold.
        hours_to_expiry: Hours until resolution.
        sigma_per_hour: Hourly volatility.
        market_type: "daily_above" or "monthly_hit".
        sigma_scale: Calibration scale factor (from autoresearch).
        direction: "above" or "below".

    Returns:
        Probability between 0 and 1.
    """
    if market_type == "monthly_hit":
        return estimate_hit_prob(
            current_price, strike, hours_to_expiry,
            sigma_per_hour, sigma_scale, direction,
        )
    else:
        # daily_above: standard terminal probability
        prob = estimate_daily_prob(
            current_price, strike, hours_to_expiry,
            sigma_per_hour, sigma_scale,
        )
        if direction == "below":
            prob = 1.0 - prob
        return prob


# ── Volatility Estimator (stateful, for live trading) ─────────────


@dataclass
class VolatilityEstimator:
    """Maintains rolling price buffer and computes cached sigma estimates.

    For live trading: feed prices via update(), read sigma via get_sigma().
    For backtesting: use compute_realized_vol() directly with price arrays.
    """

    window: int = 24  # Number of hourly observations
    method: str = "realized"  # realized, ewma, garch
    _prices: dict[str, list[float]] = field(default_factory=dict)
    _sigma_cache: dict[str, tuple[float, float]] = field(default_factory=dict)  # symbol -> (sigma, ts)
    _cache_ttl: float = 60.0  # Recompute after 60 seconds

    def update(self, symbol: str, price: float, timestamp: float | None = None) -> None:
        """Add a new price observation for a symbol."""
        if symbol not in self._prices:
            self._prices[symbol] = []
        self._prices[symbol].append(price)
        # Keep buffer bounded
        max_buffer = self.window * 3
        if len(self._prices[symbol]) > max_buffer:
            self._prices[symbol] = self._prices[symbol][-max_buffer:]
        # Invalidate cache
        self._sigma_cache.pop(symbol, None)

    def get_sigma(self, symbol: str, current_time: float = 0.0) -> float:
        """Get hourly volatility for a symbol.

        Returns cached value if fresh, otherwise recomputes.
        """
        if symbol in self._sigma_cache:
            cached_sigma, cached_ts = self._sigma_cache[symbol]
            if current_time - cached_ts < self._cache_ttl:
                return cached_sigma

        prices = self._prices.get(symbol, [])
        sigma = compute_realized_vol(prices, self.window, self.method)
        self._sigma_cache[symbol] = (sigma, current_time)
        return sigma

    def get_sigma_daily(self, symbol: str, current_time: float = 0.0) -> float:
        """Get daily volatility (= hourly * sqrt(24))."""
        return self.get_sigma(symbol, current_time) * math.sqrt(24.0)

    def load_historical(self, symbol: str, prices: list[float]) -> None:
        """Bulk-load historical prices for backtesting initialization."""
        self._prices[symbol] = list(prices)
        self._sigma_cache.pop(symbol, None)


# ── Utility: convert between volatility timeframes ────────────────


def hourly_to_daily(sigma_hourly: float) -> float:
    """Convert hourly volatility to daily (assuming 24h trading)."""
    return sigma_hourly * math.sqrt(24.0)


def daily_to_hourly(sigma_daily: float) -> float:
    """Convert daily volatility to hourly."""
    return sigma_daily / math.sqrt(24.0)


def annualized_to_hourly(sigma_annual: float) -> float:
    """Convert annualized volatility to hourly (365 * 24 hours)."""
    return sigma_annual / math.sqrt(365.0 * 24.0)
