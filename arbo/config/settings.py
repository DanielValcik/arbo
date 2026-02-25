"""Configuration management using Pydantic Settings with YAML overlay.

Loading priority: .env → config/settings.yaml → config/settings.{MODE}.yaml
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


# --- Nested config models ---


class PolymarketConfig(BaseModel):
    """Polymarket CLOB API configuration."""

    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    data_url: str = "https://data-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137
    signature_type: int = 0  # 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE
    heartbeat_interval_s: int = 10
    max_retries: int = 3
    timeout_seconds: int = 10


class MarketMakerConfig(BaseModel):
    """Layer 1: Market Making parameters."""

    min_spread: float = 0.04
    min_volume_24h: int = 1000
    max_volume_24h: int = 50000
    max_inventory_imbalance: float = 0.6
    order_size_pct: float = 0.025
    heartbeat_interval_s: int = 30
    prefer_fee_markets: bool = True


class ValueModelConfig(BaseModel):
    """Layer 2: Value Betting parameters."""

    edge_threshold: float = 0.03
    scan_interval_s: int = 300
    min_training_samples: int = 200
    xgboost_weight: float = 0.5
    llm_weight: float = 0.3
    historical_weight: float = 0.2
    kelly_fraction: float = 0.5
    max_position_pct: float = 0.05
    crypto_edge_threshold: float = 0.03
    politics_edge_threshold: float = 0.04
    max_politics_per_scan: int = 10


class ConfluenceConfig(BaseModel):
    """Layer 4: Confluence scoring parameters."""

    min_score: int = 2
    standard_size_pct: float = 0.025
    double_size_pct: float = 0.05
    whale_min_confidence: float = 0.6
    whale_poll_interval_s: int = 4
    whale_detection_target_s: int = 10


class LogicalArbConfig(BaseModel):
    """Layer 5: Logical arbitrage parameters."""

    min_pricing_violation: float = 0.03
    scan_interval_s: int = 900
    llm_model: str = "gemini-2.5-flash"
    max_llm_calls_per_hour: int = 40
    negrisk_sum_threshold: float = 0.03


class BinanceConfig(BaseModel):
    """Binance public API configuration."""

    base_url: str = "https://api.binance.com"
    futures_url: str = "https://fapi.binance.com"
    max_requests_per_min: int = 100
    cache_ttl: int = 300


class TemporalCryptoConfig(BaseModel):
    """Layer 6: Temporal crypto arbitrage parameters."""

    spot_sources: list[str] = ["binance_ws", "coinbase_ws"]
    price_deviation_threshold: float = 0.005
    market_lag_window_s: int = 60
    use_postonly: bool = True
    max_trades_per_hour: int = 20


class OrderFlowConfig(BaseModel):
    """Layer 7: Smart money order flow parameters."""

    ctf_exchange: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    volume_zscore_threshold: float = 2.0
    flow_imbalance_threshold: float = 0.65
    rolling_windows: list[int] = [3600, 14400, 86400]
    min_converging_signals: int = 2


class AttentionMarketsConfig(BaseModel):
    """Layer 8: Attention Markets parameters."""

    sentiment_scan_interval_s: int = 1800
    min_divergence: float = 0.05
    max_position_pct: float = 0.05
    sources: list[str] = ["twitter", "reddit", "tiktok"]
    llm_model: str = "gemini-2.5-flash"


class SportsLatencyConfig(BaseModel):
    """Layer 9: Live sports latency parameters."""

    max_trade_size: int = 200
    max_trades_per_hour: int = 10
    min_probability_extreme: float = 0.95
    max_fee_pct: float = 0.003
    poll_interval_s: int = 5


class WeatherConfig(BaseModel):
    """Weather connectors configuration (Strategy C)."""

    noaa_cache_ttl_s: int = 3600
    metoffice_cache_ttl_s: int = 3600
    openmeteo_cache_ttl_s: int = 3600
    scan_interval_s: int = 1800  # 30 min between weather scans
    forecast_max_age_hours: int = 6
    min_edge_threshold: float = 0.05  # 5% minimum edge to trade
    min_volume_24h: float = 10000.0  # $10K minimum 24h volume
    min_liquidity: float = 5000.0  # $5K minimum liquidity
    min_confidence: float = 0.5  # Minimum forecast confidence
    max_ladder_positions: int = 3  # Max positions per city per day
    temperature_sigma_c: float = 2.5  # Forecast uncertainty σ in °C


class ThetaDecayConfig(BaseModel):
    """Strategy A: Theta Decay configuration."""

    zscore_threshold: float = 3.0  # 3σ peak optimism threshold
    rolling_window_hours: int = 4  # 4h rolling window for z-score
    longshot_price_max: float = 0.15  # YES must be < $0.15
    min_volume_24h: float = 10000.0  # $10K minimum 24h volume
    min_age_hours: int = 24  # Market must exist for 24h+
    resolution_window_days_min: int = 3  # At least 3 days to resolution
    resolution_window_days_max: int = 30  # At most 30 days to resolution
    partial_exit_pct: float = 0.50  # Sell 50% at NO +50%
    stop_loss_pct: float = 0.30  # Exit all at NO -30%
    position_size_min: float = 20.0  # $20 minimum position
    position_size_max: float = 50.0  # $50 maximum position
    max_concurrent_positions: int = 10  # Max 10 concurrent theta positions
    excluded_categories: list[str] = ["crypto"]  # Skip finance/crypto categories
    snapshot_interval_s: int = 300  # 5 min taker flow snapshots


class OddsApiConfig(BaseModel):
    """The Odds API configuration — DEPRECATED (RDH pivot, subscription cancelled)."""

    base_url: str = "https://api.the-odds-api.com/v4"
    regions: str = "eu"
    markets: str = "h2h"
    odds_format: str = "decimal"
    min_remaining_quota: int = 100
    cache_ttl: int = 900


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-haiku-4-5-20251001"
    max_calls_per_hour: int = 60


class RiskConfig(BaseModel):
    """Documentation-only config. Actual limits are HARDCODED in arbo/core/risk_manager.py."""

    max_position_pct: float = 0.05
    daily_loss_pct: float = 0.10
    weekly_loss_pct: float = 0.20
    whale_copy_max_pct: float = 0.025
    max_market_type_pct: float = 0.30
    max_confluence_double_pct: float = 0.05
    min_paper_weeks: int = 4


class DashboardConfig(BaseModel):
    """Web dashboard configuration."""

    port: int = 8080
    host: str = "0.0.0.0"


class OrchestratorConfig(BaseModel):
    """Orchestrator runtime configuration."""

    health_check_interval_s: int = 30
    heartbeat_timeout_s: int = 120
    max_restart_count: int = 10
    signal_batch_timeout_s: float = 2.0
    dashboard_update_interval_s: int = 60
    snapshot_interval_s: int = 3600
    daily_report_hour_utc: int = 23
    weekly_report_day: int = 6  # Sunday
    weekly_report_hour_utc: int = 20


class PollingConfig(BaseModel):
    """Polling intervals for various data sources."""

    market_discovery: int = 900  # 15 min
    odds_api_batch: int = 300  # 5 min
    value_scan: int = 300  # 5 min
    news: int = 1800  # 30 min


class RSSFeed(BaseModel):
    url: str
    name: str


class NewsFeedsConfig(BaseModel):
    rss: list[RSSFeed] = []
    reddit_subreddits: list[str] = ["soccer", "sportsbook", "nba", "PremierLeague"]


# --- Main config class ---


class ArboConfig(BaseSettings):
    """Main configuration for Arbo Polymarket trading system."""

    # Runtime
    mode: str = Field(default="paper", alias="MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bankroll
    bankroll: float = 2000.00
    currency: str = "EUR"

    # Polymarket credentials
    poly_private_key: str = Field(default="", alias="POLY_PRIVATE_KEY")
    poly_funder_address: str = Field(default="", alias="POLY_FUNDER_ADDRESS")
    poly_api_key: str = Field(default="", alias="POLY_API_KEY")
    poly_secret: str = Field(default="", alias="POLY_SECRET")
    poly_passphrase: str = Field(default="", alias="POLY_PASSPHRASE")

    # External API keys
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    alchemy_key: str = Field(default="", alias="ALCHEMY_KEY")
    polygon_rpc_url: str = Field(default="", alias="DRPC_API_URL")
    metoffice_api_key: str = Field(default="", alias="METOFFICE_API_KEY")

    # Slack
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_app_token: str = Field(default="", alias="SLACK_APP_TOKEN")
    slack_channel_id: str = Field(default="", alias="SLACK_CHANNEL_ID")
    slack_daily_brief_channel_id: str = Field(default="", alias="SLACK_DAILY_BRIEF_CHANNEL_ID")
    slack_review_queue_channel_id: str = Field(default="", alias="SLACK_REVIEW_QUEUE_CHANNEL_ID")
    slack_weekly_report_channel_id: str = Field(default="", alias="SLACK_WEEKLY_REPORT_CHANNEL_ID")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://arbo:password@localhost:5432/arbo",
        alias="DATABASE_URL",
    )

    # Web dashboard
    dashboard_user: str = Field(default="arbo", alias="DASHBOARD_USER")
    dashboard_password: str = Field(default="", alias="DASHBOARD_PASSWORD")

    # Nested config (loaded from YAML)
    polymarket: PolymarketConfig = PolymarketConfig()
    market_maker: MarketMakerConfig = MarketMakerConfig()
    value_model: ValueModelConfig = ValueModelConfig()
    confluence: ConfluenceConfig = ConfluenceConfig()
    logical_arb: LogicalArbConfig = LogicalArbConfig()
    binance: BinanceConfig = BinanceConfig()
    temporal_crypto: TemporalCryptoConfig = TemporalCryptoConfig()
    order_flow: OrderFlowConfig = OrderFlowConfig()
    attention_markets: AttentionMarketsConfig = AttentionMarketsConfig()
    sports_latency: SportsLatencyConfig = SportsLatencyConfig()
    weather: WeatherConfig = WeatherConfig()
    theta_decay: ThetaDecayConfig = ThetaDecayConfig()
    odds_api: OddsApiConfig = OddsApiConfig()
    llm: LLMConfig = LLMConfig()
    risk: RiskConfig = RiskConfig()
    dashboard: DashboardConfig = DashboardConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    polling: PollingConfig = PollingConfig()
    news_feeds: NewsFeedsConfig = NewsFeedsConfig()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overlay into base dict. Overlay values win."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_config() -> ArboConfig:
    """Load and return the singleton ArboConfig.

    Loading priority: .env → settings.yaml → settings.{MODE}.yaml
    """
    mode = os.getenv("MODE", "paper")

    # Load YAML configs
    base_yaml = _load_yaml(_CONFIG_DIR / "settings.yaml")
    mode_yaml = _load_yaml(_CONFIG_DIR / f"settings.{mode}.yaml")

    # Deep merge: base + mode overlay
    merged = _deep_merge(base_yaml, mode_yaml)

    # Create config (env vars take priority via pydantic-settings)
    return ArboConfig(**merged)
