"""Tests for L4/L5/L8 fixes (PM-407)."""

from __future__ import annotations

from pathlib import Path

import yaml

from arbo.connectors.market_discovery import CATEGORY_KEYWORDS, categorize_market
from arbo.strategies.attention_markets import ATTENTION_KEYWORDS

# ---------------------------------------------------------------------------
# L4: Whale wallet config
# ---------------------------------------------------------------------------


class TestWhaleWalletConfig:
    def test_whale_wallets_yaml_exists(self) -> None:
        """config/whale_wallets.yaml exists and is valid YAML."""
        path = Path("config/whale_wallets.yaml")
        assert path.exists(), f"Expected {path} to exist"
        data = yaml.safe_load(path.read_text())
        assert "wallets" in data

    def test_whale_wallets_has_entries(self) -> None:
        """Whale wallets file has at least 1 seed entry."""
        path = Path("config/whale_wallets.yaml")
        data = yaml.safe_load(path.read_text())
        wallets = data["wallets"]
        assert len(wallets) >= 1
        assert "address" in wallets[0]


# ---------------------------------------------------------------------------
# L8: Attention markets keywords
# ---------------------------------------------------------------------------


class TestAttentionKeywords:
    def test_attention_keywords_expanded(self) -> None:
        """ATTENTION_KEYWORDS includes new terms."""
        for kw in [
            "sentiment",
            "social",
            "trending",
            "popularity",
            "followers",
            "views",
            "engagement",
        ]:
            assert kw in ATTENTION_KEYWORDS, f"Missing keyword: {kw}"

    def test_attention_keywords_in_category_map(self) -> None:
        """CATEGORY_KEYWORDS['attention_markets'] includes expanded keywords."""
        cat_kws = CATEGORY_KEYWORDS["attention_markets"]
        for kw in [
            "sentiment",
            "social",
            "trending",
            "popularity",
            "followers",
            "views",
            "engagement",
        ]:
            assert kw in cat_kws, f"Missing in CATEGORY_KEYWORDS: {kw}"

    def test_categorize_market_attention_sentiment(self) -> None:
        """Markets with 'sentiment' are categorized as attention_markets."""
        assert categorize_market("Market sentiment index above 75?") == "attention_markets"

    def test_categorize_market_attention_social(self) -> None:
        """Markets with 'social' are categorized as attention_markets."""
        assert categorize_market("Social media mentions above 10k?") == "attention_markets"
