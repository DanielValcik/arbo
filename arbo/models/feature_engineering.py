"""Feature engineering for the XGBoost value model.

Extracts and transforms raw market data into feature vectors
for probability prediction. Features are designed to work across
all Polymarket categories (soccer, crypto, politics, etc.).

See PM-101 specification for feature list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# Category encoding for XGBoost (numeric, not native categorical)
CATEGORY_ENCODING: dict[str, int] = {
    "soccer": 0,
    "politics": 1,
    "crypto": 2,
    "esports": 3,
    "entertainment": 4,
    "attention_markets": 5,
    "other": 6,
}

# Ordered feature columns — must match model training order
FEATURE_COLUMNS: list[str] = [
    "pinnacle_prob",
    "polymarket_mid",
    "price_divergence",
    "time_to_event_hours",
    "time_log",
    "volume_24h_log",
    "volume_trend",
    "liquidity_log",
    "category",
    "spread",
    "fee_enabled",
    "team_form_5g",
    "h2h_home_win_pct",
]


@dataclass
class MarketFeatures:
    """Raw feature inputs for a single market prediction.

    All fields optional except category — XGBoost handles NaN natively.
    """

    pinnacle_prob: float | None = None
    polymarket_mid: float | None = None
    time_to_event_hours: float | None = None
    category: str = "other"
    volume_24h: float = 0.0
    volume_30d_avg: float = 0.0
    liquidity: float = 0.0
    spread: float | None = None
    fee_enabled: bool = False
    # Soccer-specific (optional)
    team_form_5g: float | None = None
    h2h_home_win_pct: float | None = None


def extract_feature_vector(features: MarketFeatures) -> dict[str, float]:
    """Convert MarketFeatures into a flat dict for model input.

    Missing values are encoded as NaN (XGBoost handles natively).
    """
    nan = float("nan")

    pinnacle = features.pinnacle_prob if features.pinnacle_prob is not None else nan
    poly_mid = features.polymarket_mid if features.polymarket_mid is not None else nan

    # Price divergence is the key signal
    if features.pinnacle_prob is not None and features.polymarket_mid is not None:
        price_divergence = features.pinnacle_prob - features.polymarket_mid
    else:
        price_divergence = nan

    # Time features
    time_hours = features.time_to_event_hours if features.time_to_event_hours is not None else nan
    time_log = math.log1p(time_hours) if time_hours == time_hours else nan  # NaN != NaN

    # Volume features
    vol_24h_log = math.log1p(features.volume_24h) if features.volume_24h > 0 else 0.0
    vol_trend = (
        (features.volume_24h / features.volume_30d_avg) if features.volume_30d_avg > 0 else 1.0
    )
    liq_log = math.log1p(features.liquidity) if features.liquidity > 0 else 0.0

    return {
        "pinnacle_prob": pinnacle,
        "polymarket_mid": poly_mid,
        "price_divergence": price_divergence,
        "time_to_event_hours": time_hours,
        "time_log": time_log,
        "volume_24h_log": vol_24h_log,
        "volume_trend": vol_trend,
        "liquidity_log": liq_log,
        "category": float(CATEGORY_ENCODING.get(features.category, 6)),
        "spread": features.spread if features.spread is not None else nan,
        "fee_enabled": 1.0 if features.fee_enabled else 0.0,
        "team_form_5g": features.team_form_5g if features.team_form_5g is not None else nan,
        "h2h_home_win_pct": (
            features.h2h_home_win_pct if features.h2h_home_win_pct is not None else nan
        ),
    }


def features_to_dataframe(feature_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Convert list of feature dicts into a DataFrame with correct column order."""
    df = pd.DataFrame(feature_dicts)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[FEATURE_COLUMNS]
