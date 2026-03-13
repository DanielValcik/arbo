"""
Strategy C Weather Experiment — MODIFY THIS FILE
=================================================

This file contains all tunable parameters and strategy logic for
Strategy C (Compound Weather Resolution Chaining).

The AI agent modifies this file to find the best parameter combination.
The backtest harness (backtest_harness.py) calls functions from this file.

Run: python3 research/backtest_harness.py
"""

import math

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — Edit these to experiment
# ═══════════════════════════════════════════════════════════════════════════════

# Which lead times to evaluate (days before resolution)
# For each city/date, the best edge across these days_out values is selected.
DAYS_OUT_TO_TRADE = [0, 1]

# Forecast uncertainty model: sigma (°C) by days_out
# Controls the width of our probability distribution.
# Lower = more confident (narrower distribution), Higher = less confident (wider).
FORECAST_SIGMA = {
    0: 1.22,
    1: 3.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}

# Per-city sigma overrides — METAR-calibrated (60-day IEM vs Open-Meteo archive)
# Data source quality: NOAA (nyc, chicago) > Met Office (london) > Open-Meteo (rest)
CITY_SIGMA = {
    "paris":         {0: 0.75},
    "seattle":       {0: 0.91},
    "london":        {0: 0.92, 1: 2.8},
    "lucknow":       {0: 0.97},
    "miami":         {0: 1.00},
    "wellington":    {0: 1.07},
    "tel_aviv":      {0: 1.14},
    "nyc":           {0: 1.15},
    "chicago":       {0: 1.15, 1: 3.0},
    "dallas":        {0: 1.24},
    "seoul":         {0: 1.32, 1: 2.5},
    "atlanta":       {0: 1.32},
    "sao_paulo":     {0: 1.36},
    "toronto":       {0: 1.38},
    "buenos_aires":  {0: 1.43, 1: 3.0},
    "ankara":        {0: 1.44},
    "munich":        {0: 1.49},
    "tokyo":         {0: 1.66},
    "dc":            {0: 1.67},
    "los_angeles":   {0: 2.05},
}

# Per-city bias corrections (°C) — measured forecast error vs METAR actual.
# Positive = forecast reads LOW vs actual (add correction to forecast).
# Negative = forecast reads HIGH (subtract). Run: python3 research/calibrate_bias.py --days 60
CITY_BIAS = {
    "buenos_aires": 2.58,
    "dc":           1.67,
    "nyc":          1.53,
    "wellington":   1.43,
    "atlanta":      1.30,
    "sao_paulo":    1.24,
    "toronto":      0.98,
    "chicago":      0.92,
    "seoul":        0.87,
    "dallas":       0.73,
    "miami":        0.56,
    "tel_aviv":     0.52,
    "seattle":      0.41,
    "london":       0.24,
    "paris":        0.16,
    "tokyo":        -0.09,
    "los_angeles":  -0.31,
    "munich":       -0.45,
    "lucknow":      -0.74,
    "ankara":       -0.78,
}

# Per-city quality gate overrides — optimized via sweep_per_city_v2.py
# Exclude 6 unprofitable/marginal cities, widen price range for top 6
CITY_OVERRIDES = {
    "dc":            {"min_edge": 0.99},
    "toronto":       {"min_edge": 0.99},
    "buenos_aires":  {"min_edge": 0.99},
    "nyc":           {"min_edge": 0.99},
    "atlanta":       {"min_edge": 0.99},
    "wellington":    {"min_edge": 0.99},
    "paris":         {"max_price": 0.50},
    "seattle":       {"max_price": 0.50},
    "london":        {"max_price": 0.50},
    "lucknow":       {"max_price": 0.50},
    "miami":         {"max_price": 0.50},
    "tel_aviv":      {"max_price": 0.50},
}

# Probability distribution type
# "normal" = Gaussian CDF
# "student_t" = Student-t CDF (heavier tails, more conservative)
DISTRIBUTION = "normal"
STUDENT_T_DF = 5  # Degrees of freedom (lower = heavier tails)

# Probability sharpening: raise raw prob to this power (>1 = more decisive, <1 = softer)
PROB_SHARPENING = 1.05

# ── Quality Gate Thresholds (global defaults) ──
MIN_EDGE = 0.08             # Minimum edge to trade
MIN_PRICE = 0.30            # Skip extreme longshots
MAX_PRICE = 0.43            # Skip near-certainties
MIN_VOLUME = 1000.0         # Minimum 24h volume ($)
MIN_LIQUIDITY = 200.0      # Minimum liquidity ($)
CONVICTION_RATIO = 0.0      # forecast_prob / market_price must exceed this
MIN_FORECAST_PROB = 0.62    # Minimum absolute probability to trade

# ── Position Sizing ──
KELLY_FRACTION = 0.01       # Ultra Conservative Kelly
MAX_POSITION_PCT = 0.05     # Max 5% of capital per trade
MAX_TOTAL_EXPOSURE_PCT = 0.80  # Max 80% capital deployed at once


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sigma(days_out, city=None):
    """Get forecast sigma for a given days_out and optional city."""
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))


def _get_threshold(name, city=None):
    """Get a quality gate threshold, with optional per-city override."""
    if city and city in CITY_OVERRIDES:
        override = CITY_OVERRIDES[city].get(name)
        if override is not None:
            return override
    return globals()[name.upper()]


def _normal_cdf(x, mu=0.0, sigma=1.0):
    """Standard normal CDF."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _student_t_cdf(x, mu, sigma, df=5):
    """Approximate Student-t CDF via scaled normal.

    Scales sigma by sqrt(df/(df-2)) to match Student-t variance.
    Accurate for df >= 3. For heavy tails, use lower df.
    """
    if df <= 2:
        df = 3
    adjusted_sigma = sigma * math.sqrt(df / (df - 2))
    return _normal_cdf(x, mu, adjusted_sigma)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY MODEL — Edit to experiment with structural changes
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_probability(forecast_temp_c, bucket_low_c, bucket_high_c, days_out,
                         *, city=None):
    """
    Estimate probability that actual temperature falls in [bucket_low_c, bucket_high_c).

    Args:
        forecast_temp_c: Our forecast temperature in Celsius.
        bucket_low_c: Lower bound of bucket (None for "below X" type).
        bucket_high_c: Upper bound of bucket (None for "above X" type).
        days_out: Days until resolution.
        city: City identifier ("nyc", "chicago", "london", "seoul", "buenos_aires").
              Enables per-city sigma and model customization.

    Returns:
        Probability in [0, 1].
    """
    sigma = _get_sigma(days_out, city)

    if DISTRIBUTION == "student_t":
        cdf = lambda x: _student_t_cdf(x, forecast_temp_c, sigma, STUDENT_T_DF)
    else:
        cdf = lambda x: _normal_cdf(x, forecast_temp_c, sigma)

    if bucket_low_c is None and bucket_high_c is not None:
        # "Below X" bucket
        raw = cdf(bucket_high_c)
    elif bucket_high_c is None and bucket_low_c is not None:
        # "Above X" bucket
        raw = 1.0 - cdf(bucket_low_c)
    elif bucket_low_c is not None and bucket_high_c is not None:
        # Range bucket
        raw = cdf(bucket_high_c) - cdf(bucket_low_c)
    else:
        return 0.0

    # Bayesian shrinkage: blend with uniform prior (reduces overconfidence)
    uniform_prior = 0.125  # 1/8 buckets
    raw = raw * 0.97 + uniform_prior * 0.03

    # Sharpen: push probabilities toward extremes
    if PROB_SHARPENING != 1.0 and raw > 0:
        raw = raw ** PROB_SHARPENING
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE — Edit to experiment with filtering logic
# ═══════════════════════════════════════════════════════════════════════════════

def should_trade(edge, forecast_prob, market_price, days_out, volume, liquidity,
                 *, city=None):
    """
    Quality gate: decide whether to trade this signal.

    Args:
        city: City identifier — enables per-city threshold overrides via CITY_OVERRIDES.

    Returns True if signal passes all quality checks.
    """
    if edge < _get_threshold("min_edge", city):
        return False
    if edge > 0.42:  # Suspiciously high edge — likely pricing anomaly
        return False
    if market_price < _get_threshold("min_price", city):
        return False
    if market_price > _get_threshold("max_price", city):
        return False
    if volume < _get_threshold("min_volume", city):
        return False
    if liquidity < _get_threshold("min_liquidity", city):
        return False
    if forecast_prob < _get_threshold("min_forecast_prob", city):
        return False
    # Conviction check
    conviction = _get_threshold("conviction_ratio", city)
    if conviction > 0 and market_price > 0:
        if forecast_prob / market_price < conviction:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING — Edit to experiment with sizing strategies
# ═══════════════════════════════════════════════════════════════════════════════

def position_size(edge, market_price, available_capital, total_capital,
                  *, city=None):
    """
    Calculate position size using Kelly criterion.

    Args:
        edge: Our edge (forecast_prob - market_price).
        market_price: Current market YES price.
        available_capital: Capital available for this trade.
        total_capital: Total portfolio capital.
        city: City identifier — enables per-city sizing adjustments.

    Returns:
        Position size in USDC (float). 0 if no trade.
    """
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0

    prob = market_price + edge
    if prob <= 0 or prob >= 1:
        return 0.0

    # Kelly: f* = (p*b - q) / b
    odds = (1.0 / market_price) - 1.0
    kelly_raw = (prob * odds - (1.0 - prob)) / odds

    if kelly_raw <= 0:
        return 0.0

    # Cap kelly_raw to reduce variance from high-edge trades
    kelly_raw = min(kelly_raw, 0.40)

    # Fixed 0.35x scaling, equal weight all cities
    kelly_adjusted = kelly_raw * KELLY_FRACTION * 0.35
    size = available_capital * kelly_adjusted

    # Cap at max position size
    max_size = total_capital * MAX_POSITION_PCT
    size = min(size, max_size)

    if size < 1.0:
        return 0.0

    return round(size, 2)
