"""Configuration management using Pydantic Settings with YAML overlay.

Loading priority: .env → settings.yaml → settings.{MODE}.yaml
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


class SportConfig(BaseModel):
    leagues: list[str] = []
    markets: list[str] = ["h2h", "spreads", "totals"]


class PollingConfig(BaseModel):
    matchbook: int = 8
    odds_api_batch: int = 300
    news: int = 300
    betinasia: int = 60


class WeightsConfig(BaseModel):
    arb: float = 1.0
    value: float = 0.0
    situational: float = 0.0


class LLMConfig(BaseModel):
    provider: str = "gemini"
    gemini_model: str = "gemini-2.0-flash"
    claude_model: str = "claude-haiku-4-5-20251001"
    max_calls_per_day: int = 100


class ThresholdsConfig(BaseModel):
    min_edge: float = 0.02
    arb_margin: float = 0.04
    value_gap: float = 0.05
    llm_confidence: float = 0.7
    llm_magnitude: int = 5


class MatchbookConfig(BaseModel):
    base_url: str = "https://api.matchbook.com/edge/rest"
    auth_url: str = "https://api.matchbook.com/bpapi/rest"
    commission_pct: float = 0.04
    session_ttl_seconds: int = 18000
    max_retries: int = 3
    timeout_seconds: int = 10
    unmatched_cancel_after: int = 30
    min_fill_pct: float = 0.60


class RiskConfig(BaseModel):
    """Documentation-only config. Actual limits are HARDCODED in src/engine/risk.py."""

    max_bet_pct: float = 0.05
    daily_loss_pct: float = 0.10
    weekly_loss_pct: float = 0.20
    max_concurrent_bets: int = 3
    max_per_event: int = 1
    max_sport_exposure_pct: float = 0.40


class RSSFeed(BaseModel):
    url: str
    name: str


class NewsFeedsConfig(BaseModel):
    rss: list[RSSFeed] = []
    reddit_subreddits: list[str] = ["soccer", "sportsbook", "nba", "PremierLeague"]
    gnews_enabled: bool = False


# --- Main config class ---


class ArboConfig(BaseSettings):
    # Runtime
    mode: str = Field(default="paper", alias="MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bankroll
    bankroll: float = 2000.00
    currency: str = "EUR"

    # Platform credentials
    matchbook_username: str = Field(default="", alias="MATCHBOOK_USERNAME")
    matchbook_password: str = Field(default="", alias="MATCHBOOK_PASSWORD")
    betinasia_username: str = Field(default="", alias="BETINASIA_USERNAME")
    betinasia_password: str = Field(default="", alias="BETINASIA_PASSWORD")
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")

    # LLM keys
    google_ai_api_key: str = Field(default="", alias="GOOGLE_AI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gnews_api_key: str = Field(default="", alias="GNEWS_API_KEY")

    # Reddit
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="arbo/1.0", alias="REDDIT_USER_AGENT")

    # Slack
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_app_token: str = Field(default="", alias="SLACK_APP_TOKEN")
    slack_channel_id: str = Field(default="", alias="SLACK_CHANNEL_ID")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://arbo:password@localhost:5432/arbo",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Nested config (loaded from YAML)
    sports: dict[str, SportConfig] = {}
    polling: PollingConfig = PollingConfig()
    weights: WeightsConfig = WeightsConfig()
    llm: LLMConfig = LLMConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    matchbook: MatchbookConfig = MatchbookConfig()
    risk: RiskConfig = RiskConfig()
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
