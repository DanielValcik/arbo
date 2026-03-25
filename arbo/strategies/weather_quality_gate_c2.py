"""Quality gate for Strategy C2 — EMOS + Edge Exit Fusion.

Autoresearch-optimized params from sweep_emos_exit_fusion (2026-03-25):
  - Score: 138.1 (IS), 121.0 (OOS)
  - 1,878 trades, 54.1% WR, $15,512 PnL, Sharpe 9.44, DD 8.3%
  - Walk-forward: 3 folds all profitable ($5,030 / $3,567 / $1,637)

Key differences from C (C1f-ensemble):
  - Looser entry (min_edge=0.03 vs 0.10, min_price=0.03 vs 0.08)
  - Edge-based exit (min_hold_edge=0.05) — sells when EMOS says edge lost
  - EMOS adaptive sigma (rolling_mae, window=21, scale=0.6)
  - 3 cities excluded (São Paulo, Tel Aviv, Tokyo)
  - Per-city overrides for Dallas and Miami
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from arbo.connectors.weather_models import City
from arbo.strategies.weather_scanner import WeatherSignal
from arbo.utils.logger import get_logger

logger = get_logger("weather_quality_gate_c2")

# ── Global thresholds (EMOS+Exit fusion autoresearch) ──
MIN_EDGE = 0.03
MAX_EDGE = 0.90
MIN_PRICE = 0.03
MAX_PRICE = 0.45
MIN_VOLUME_24H = 0.0
MIN_CONFIDENCE = 0.3
MAX_FORECAST_AGE_HOURS = 6
MIN_LIQUIDITY = 100.0
MIN_FORECAST_PROB = 0.03

# ── EMOS parameters ──
EMOS_TRAINING_WINDOW = 21
EMOS_SIGMA_METHOD = "rolling_mae"
EMOS_BIAS_METHOD = "ewma"
EMOS_SIGMA_FLOOR = 0.7
EMOS_SIGMA_SCALE = 0.6
EMOS_EWMA_ALPHA = 0.15

# ── Exit parameters ──
EXIT_ENABLED = True
MIN_HOLD_EDGE = 0.05          # Exit when EMOS-computed edge < 5%
EXIT_SLIPPAGE_PCT = 0.06      # 6% slippage on exit sell (real weather market median ~7%)
PROB_EXIT_FLOOR = 0.10        # Also exit when prob < 10%
PROFIT_TAKE_ALSO = True       # Also trigger profit take at absolute target
PROFIT_TARGET_ABS = 0.15      # +$0.15 above entry → take profit

# ── Sizing ──
KELLY_RAW_CAP = 0.30
PROB_SHARPENING = 0.85
SHRINKAGE = 0.0

# ── Excluded cities ──
EXCLUDED_CITIES: set[str] = {
    "sao_paulo",
    "tel_aviv",
    "tokyo",
    "lucknow",
}

# ── Per-city overrides ──
CITY_OVERRIDES: dict[str, dict[str, float]] = {
    "dallas": {
        "min_edge": 0.01,
        "max_price": 0.40,
        "min_price": 0.01,
        "kelly_raw_cap": 0.15,
        "prob_sharpening": 0.90,
    },
    "miami": {
        "min_edge": 0.08,
        "max_price": 0.40,
        "min_price": 0.02,
        "kelly_raw_cap": 0.10,
        "prob_sharpening": 1.05,
    },
}


@dataclass
class QualityDecision:
    """Result of quality gate check."""

    passed: bool
    reason: str
    signal: WeatherSignal | None = None


def _get_threshold(name: str, city: str | None = None) -> float:
    """Get threshold with per-city override."""
    if city and city in CITY_OVERRIDES:
        override = CITY_OVERRIDES[city].get(name)
        if override is not None:
            return override
    defaults: dict[str, float] = {
        "min_edge": MIN_EDGE,
        "max_edge": MAX_EDGE,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
        "min_volume": MIN_VOLUME_24H,
        "min_liquidity": MIN_LIQUIDITY,
        "min_forecast_prob": MIN_FORECAST_PROB,
        "kelly_raw_cap": KELLY_RAW_CAP,
        "prob_sharpening": PROB_SHARPENING,
    }
    return defaults.get(name, 0.0)


def check_signal_quality(
    signal: WeatherSignal,
    forecast_fetched_at: datetime,
    min_edge: float | None = None,
    min_volume: float | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    max_forecast_age_hours: int = MAX_FORECAST_AGE_HOURS,
    min_liquidity: float | None = None,
) -> QualityDecision:
    """Validate signal against C2 quality thresholds."""
    city = signal.market.city.value if signal.market.city else None

    # Excluded cities
    if city and city in EXCLUDED_CITIES:
        return QualityDecision(False, f"excluded_city:{city}")

    # Edge
    eff_min_edge = min_edge if min_edge is not None else _get_threshold("min_edge", city)
    eff_max_edge = _get_threshold("max_edge", city)
    if signal.edge < eff_min_edge:
        return QualityDecision(False, f"edge_too_low:{signal.edge:.4f}<{eff_min_edge}")
    if signal.edge > eff_max_edge:
        return QualityDecision(False, f"edge_anomaly:{signal.edge:.4f}>{eff_max_edge}")

    # Price range
    price = signal.market.market_price
    eff_min_price = _get_threshold("min_price", city)
    eff_max_price = _get_threshold("max_price", city)
    if price < eff_min_price:
        return QualityDecision(False, f"price_too_low:{price:.4f}")
    if price > eff_max_price:
        return QualityDecision(False, f"price_too_high:{price:.4f}")

    # Volume
    eff_min_vol = min_volume if min_volume is not None else _get_threshold("min_volume", city)
    if signal.market.volume_24h < eff_min_vol:
        return QualityDecision(False, f"low_volume:{signal.market.volume_24h}")

    # Liquidity
    eff_min_liq = min_liquidity if min_liquidity is not None else _get_threshold("min_liquidity", city)
    if signal.market.liquidity < eff_min_liq:
        return QualityDecision(False, f"low_liquidity:{signal.market.liquidity}")

    # Fee-free only (weather markets should be 0% fee)
    if getattr(signal.market, "fee_enabled", False):
        return QualityDecision(False, "fee_enabled")

    # Confidence
    if signal.confidence < min_confidence:
        return QualityDecision(False, f"low_confidence:{signal.confidence}")

    # Forecast probability floor
    min_prob = _get_threshold("min_forecast_prob", city)
    if signal.forecast_probability < min_prob:
        return QualityDecision(False, f"prob_too_low:{signal.forecast_probability:.4f}")

    # Forecast freshness
    now = datetime.now(timezone.utc)
    age_hours = (now - forecast_fetched_at).total_seconds() / 3600
    if age_hours > max_forecast_age_hours:
        return QualityDecision(False, f"stale_forecast:{age_hours:.1f}h")

    return QualityDecision(True, "passed", signal)


def filter_signals(
    signals: list[WeatherSignal],
    forecast_fetched_at: datetime,
) -> list[WeatherSignal]:
    """Filter signals through C2 quality gate. Returns qualified signals."""
    qualified = []
    for sig in signals:
        decision = check_signal_quality(sig, forecast_fetched_at)
        if decision.passed:
            qualified.append(sig)
        else:
            city = sig.market.city.value if sig.market.city else "?"
            logger.debug(
                "c2_signal_rejected",
                city=city,
                reason=decision.reason,
                edge=round(sig.edge, 4),
                price=round(sig.market.market_price, 4),
            )
    return qualified
