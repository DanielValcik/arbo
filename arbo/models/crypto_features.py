"""Feature engineering for the crypto XGBoost model (PM-401).

Extracts 12 features from Binance OHLCV data, funding rates,
and Polymarket market metadata for crypto price-threshold markets.

Feature set:
  spot_vs_strike, time_to_expiry, time_to_expiry_log, volatility_24h,
  volatility_7d, volume_24h_log, volume_trend, funding_rate, rsi_14,
  momentum_24h, distance_pct, polymarket_mid
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Ordered feature columns — must match model training order
CRYPTO_FEATURE_COLUMNS: list[str] = [
    "spot_vs_strike",
    "time_to_expiry",
    "time_to_expiry_log",
    "volatility_24h",
    "volatility_7d",
    "volume_24h_log",
    "volume_trend",
    "funding_rate",
    "rsi_14",
    "momentum_24h",
    "distance_pct",
    "polymarket_mid",
]


@dataclass
class CryptoFeatures:
    """Raw feature inputs for a single crypto market prediction.

    All numeric fields. NaN-safe — XGBoost handles missing values natively.
    """

    spot_vs_strike: float | None = None  # spot_price / strike_price ratio
    time_to_expiry: float | None = None  # hours until resolution
    volatility_24h: float | None = None  # annualized vol from 24h bars
    volatility_7d: float | None = None  # annualized vol from 7d bars
    volume_24h_log: float | None = None  # log(1 + 24h_quote_volume)
    volume_trend: float | None = None  # recent_vol / avg_vol ratio
    funding_rate: float | None = None  # latest perp funding rate
    rsi_14: float | None = None  # RSI-14 from hourly bars
    momentum_24h: float | None = None  # 24h price change %
    distance_pct: float | None = None  # |spot - strike| / strike
    polymarket_mid: float | None = None  # current Polymarket YES price


def extract_crypto_feature_vector(features: CryptoFeatures) -> dict[str, float]:
    """Convert CryptoFeatures into a flat dict for model input.

    Missing values are encoded as NaN (XGBoost handles natively).
    """
    nan = float("nan")

    spot_vs_strike = features.spot_vs_strike if features.spot_vs_strike is not None else nan
    time_to_expiry = features.time_to_expiry if features.time_to_expiry is not None else nan
    time_to_expiry_log = math.log1p(time_to_expiry) if time_to_expiry == time_to_expiry else nan

    return {
        "spot_vs_strike": spot_vs_strike,
        "time_to_expiry": time_to_expiry,
        "time_to_expiry_log": time_to_expiry_log,
        "volatility_24h": features.volatility_24h if features.volatility_24h is not None else nan,
        "volatility_7d": features.volatility_7d if features.volatility_7d is not None else nan,
        "volume_24h_log": (features.volume_24h_log if features.volume_24h_log is not None else nan),
        "volume_trend": features.volume_trend if features.volume_trend is not None else nan,
        "funding_rate": features.funding_rate if features.funding_rate is not None else nan,
        "rsi_14": features.rsi_14 if features.rsi_14 is not None else nan,
        "momentum_24h": features.momentum_24h if features.momentum_24h is not None else nan,
        "distance_pct": features.distance_pct if features.distance_pct is not None else nan,
        "polymarket_mid": (features.polymarket_mid if features.polymarket_mid is not None else nan),
    }


def crypto_features_to_dataframe(
    feature_dicts: list[dict[str, float]],
) -> pd.DataFrame:
    """Convert list of feature dicts into a DataFrame with correct column order."""
    df = pd.DataFrame(feature_dicts)
    for col in CRYPTO_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[CRYPTO_FEATURE_COLUMNS]


def compute_spot_vs_strike(spot_price: float, strike_price: float) -> float:
    """Compute spot/strike ratio.

    Args:
        spot_price: Current spot price.
        strike_price: Market strike price.

    Returns:
        Ratio spot/strike. Values >1 mean spot is above strike.
    """
    if strike_price <= 0:
        return float("nan")
    return spot_price / strike_price


def compute_distance_pct(spot_price: float, strike_price: float) -> float:
    """Compute absolute distance from spot to strike as fraction.

    Args:
        spot_price: Current spot price.
        strike_price: Market strike price.

    Returns:
        |spot - strike| / strike as a fraction.
    """
    if strike_price <= 0:
        return float("nan")
    return abs(spot_price - strike_price) / strike_price
