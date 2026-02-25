"""Tests for Opportunity Scanner (PM-005).

Tests verify:
1. Layer 1 (MM): wide spread detection, volume filtering
2. Layer 2 (Value): Pinnacle divergence detection (with mock odds)
3. Layer 3 (Arb): NegRisk pricing violation detection
4. Layer 6 (Crypto): 15-min crypto market discovery
5. scan_all runs all layers
6. Confluence scoring across layers
7. Signal output format and DB dict conversion
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.connectors.market_discovery import GammaMarket
from arbo.core.scanner import OpportunityScanner, Signal, SignalDirection, ThetaDecaySignal

# ================================================================
# Factory helper
# ================================================================


def _make_market(
    condition_id: str = "cond_1",
    question: str = "Test market?",
    outcome_prices: list[str] | None = None,
    clob_token_ids: list[str] | None = None,
    volume_24h: str = "5000",
    liquidity: str = "10000",
    fee_enabled: bool = False,
    neg_risk: bool = False,
    active: bool = True,
) -> GammaMarket:
    """Build a GammaMarket from minimal params."""
    raw = {
        "conditionId": condition_id,
        "question": question,
        "slug": question.lower().replace(" ", "-"),
        "outcomes": ["Yes", "No"],
        "outcomePrices": outcome_prices or ["0.50", "0.50"],
        "clobTokenIds": clob_token_ids or ["tok_yes", "tok_no"],
        "volume": "100000",
        "volume24hr": volume_24h,
        "liquidity": liquidity,
        "active": active,
        "closed": False,
        "feesEnabled": fee_enabled,
        "enableNegRisk": neg_risk,
        "tags": [],
    }
    return GammaMarket(raw)


@pytest.fixture
def scanner() -> OpportunityScanner:
    return OpportunityScanner()


# ================================================================
# Layer 1: Market Making
# ================================================================


class TestLayer1MM:
    """Layer 1: wide spread market detection."""

    def test_wide_spread_detected(self, scanner: OpportunityScanner) -> None:
        """Market with spread > 4% and volume in range → signal."""
        # Prices don't sum to 1 → spread = |1 - 0.45 - 0.50| = 0.05
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            volume_24h="5000",
        )
        signals = scanner.scan_layer1_mm([market])
        assert len(signals) == 1
        assert signals[0].layer == 1
        assert signals[0].edge == Decimal("0.05") / 2

    def test_narrow_spread_ignored(self, scanner: OpportunityScanner) -> None:
        """Market with spread < 4% → no signal."""
        market = _make_market(outcome_prices=["0.49", "0.50"])
        signals = scanner.scan_layer1_mm([market])
        assert len(signals) == 0

    def test_low_volume_ignored(self, scanner: OpportunityScanner) -> None:
        """Market with volume < $1K → no signal."""
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            volume_24h="500",
        )
        signals = scanner.scan_layer1_mm([market])
        assert len(signals) == 0

    def test_high_volume_ignored(self, scanner: OpportunityScanner) -> None:
        """Market with volume > $50K → no signal (too competitive)."""
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            volume_24h="60000",
        )
        signals = scanner.scan_layer1_mm([market])
        assert len(signals) == 0

    def test_fee_enabled_boosts_confidence(self, scanner: OpportunityScanner) -> None:
        """Fee-enabled markets get higher confidence (maker rebates)."""
        market_no_fee = _make_market(
            condition_id="c1",
            outcome_prices=["0.45", "0.50"],
            volume_24h="5000",
            fee_enabled=False,
        )
        market_fee = _make_market(
            condition_id="c2",
            outcome_prices=["0.45", "0.50"],
            volume_24h="5000",
            fee_enabled=True,
        )
        signals_no_fee = scanner.scan_layer1_mm([market_no_fee])
        signals_fee = scanner.scan_layer1_mm([market_fee])
        assert signals_fee[0].confidence > signals_no_fee[0].confidence

    def test_missing_spread_skipped(self, scanner: OpportunityScanner) -> None:
        """Market without price data → no signal."""
        raw = {
            "conditionId": "c1",
            "question": "Test?",
            "slug": "test",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [],
            "clobTokenIds": ["tok_yes", "tok_no"],
            "volume": "0",
            "volume24hr": "5000",
            "liquidity": "10000",
            "active": True,
            "closed": False,
            "feesEnabled": False,
            "enableNegRisk": False,
            "tags": [],
        }
        market = GammaMarket(raw)
        signals = scanner.scan_layer1_mm([market])
        assert len(signals) == 0


# ================================================================
# Layer 2: Value Betting
# ================================================================


class TestLayer2Value:
    """Layer 2: Pinnacle vs Polymarket divergence."""

    def test_no_pinnacle_odds_returns_empty(self, scanner: OpportunityScanner) -> None:
        """Without Pinnacle odds, no signals produced."""
        market = _make_market()
        signals = scanner.scan_layer2_value([market])
        assert len(signals) == 0

    def test_divergence_detected(self, scanner: OpportunityScanner) -> None:
        """Large Pinnacle-Poly divergence → signal."""
        market = _make_market(
            condition_id="c1",
            outcome_prices=["0.50", "0.50"],
        )
        pinnacle_odds = {"c1": Decimal("0.60")}  # 10% divergence
        signals = scanner.scan_layer2_value([market], pinnacle_odds)
        assert len(signals) == 1
        assert signals[0].layer == 2
        assert signals[0].direction == SignalDirection.BUY_YES

    def test_small_divergence_ignored(self, scanner: OpportunityScanner) -> None:
        """Divergence < 3% threshold → no signal."""
        market = _make_market(
            condition_id="c1",
            outcome_prices=["0.50", "0.50"],
        )
        pinnacle_odds = {"c1": Decimal("0.52")}  # only 2% divergence
        signals = scanner.scan_layer2_value([market], pinnacle_odds)
        assert len(signals) == 0

    def test_buy_no_when_pinnacle_lower(self, scanner: OpportunityScanner) -> None:
        """When Pinnacle says lower prob, direction is BUY_NO."""
        market = _make_market(
            condition_id="c1",
            outcome_prices=["0.60", "0.40"],
        )
        pinnacle_odds = {"c1": Decimal("0.50")}  # Pinnacle says 50%, Poly at 60%
        signals = scanner.scan_layer2_value([market], pinnacle_odds)
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY_NO

    def test_fee_reduces_edge(self, scanner: OpportunityScanner) -> None:
        """Fee-enabled market should have edge reduced by fee."""
        market_no_fee = _make_market(
            condition_id="c1",
            outcome_prices=["0.50", "0.50"],
            fee_enabled=False,
        )
        market_fee = _make_market(
            condition_id="c2",
            outcome_prices=["0.50", "0.50"],
            fee_enabled=True,
        )
        pinnacle = {"c1": Decimal("0.60"), "c2": Decimal("0.60")}
        sigs_no_fee = scanner.scan_layer2_value([market_no_fee], pinnacle)
        sigs_fee = scanner.scan_layer2_value([market_fee], pinnacle)
        if sigs_no_fee and sigs_fee:
            assert sigs_fee[0].edge < sigs_no_fee[0].edge


# ================================================================
# Layer 3: NegRisk Arb
# ================================================================


class TestLayer3Arb:
    """Layer 3: NegRisk pricing violations."""

    def test_underpriced_negrisk_detected(self, scanner: OpportunityScanner) -> None:
        """Sum < 0.97 → BUY arbitrage signal."""
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            neg_risk=True,
        )
        signals = scanner.scan_layer3_arb([market])
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY_YES
        assert signals[0].edge == Decimal("0.05")  # 1 - 0.95

    def test_overpriced_negrisk_detected(self, scanner: OpportunityScanner) -> None:
        """Sum > 1.03 → SELL arbitrage signal."""
        market = _make_market(
            outcome_prices=["0.55", "0.50"],
            neg_risk=True,
        )
        signals = scanner.scan_layer3_arb([market])
        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.SELL_YES
        assert signals[0].edge == Decimal("0.05")  # 1.05 - 1

    def test_non_negrisk_ignored(self, scanner: OpportunityScanner) -> None:
        """Non-NegRisk markets are skipped."""
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            neg_risk=False,
        )
        signals = scanner.scan_layer3_arb([market])
        assert len(signals) == 0

    def test_within_threshold_ignored(self, scanner: OpportunityScanner) -> None:
        """Deviation < 3% → no signal."""
        market = _make_market(
            outcome_prices=["0.49", "0.50"],
            neg_risk=True,
        )
        signals = scanner.scan_layer3_arb([market])
        assert len(signals) == 0

    def test_high_confidence(self, scanner: OpportunityScanner) -> None:
        """Arb signals should have high confidence."""
        market = _make_market(
            outcome_prices=["0.45", "0.50"],
            neg_risk=True,
        )
        signals = scanner.scan_layer3_arb([market])
        assert signals[0].confidence >= Decimal("0.85")


# ================================================================
# Layer 6: Temporal Crypto
# ================================================================


class TestLayer6Crypto:
    """Layer 6: 15-min crypto market discovery."""

    def test_crypto_15min_detected(self, scanner: OpportunityScanner) -> None:
        """Crypto market with '15' in question + fee-enabled → signal."""
        market = _make_market(
            question="Bitcoin 15 minute above $95k?",
            fee_enabled=True,
            outcome_prices=["0.96", "0.04"],  # extreme price → fee favorable
        )
        signals = scanner.scan_layer6_crypto([market])
        assert len(signals) == 1
        assert signals[0].layer == 6

    def test_non_crypto_ignored(self, scanner: OpportunityScanner) -> None:
        """Non-crypto markets with '15min' are ignored."""
        market = _make_market(
            question="Soccer 15 minute goal?",
            fee_enabled=True,
            outcome_prices=["0.96", "0.04"],
        )
        signals = scanner.scan_layer6_crypto([market])
        assert len(signals) == 0

    def test_no_fee_ignored(self, scanner: OpportunityScanner) -> None:
        """Fee-free crypto markets are ignored for layer 6."""
        market = _make_market(
            question="Bitcoin 15 minute above $95k?",
            fee_enabled=False,
            outcome_prices=["0.96", "0.04"],
        )
        signals = scanner.scan_layer6_crypto([market])
        assert len(signals) == 0

    def test_unfavorable_fee_ignored(self, scanner: OpportunityScanner) -> None:
        """Midpoint price → fee not favorable → no signal."""
        market = _make_market(
            question="Bitcoin 15 minute above $95k?",
            fee_enabled=True,
            outcome_prices=["0.50", "0.50"],  # midpoint → high fee
        )
        signals = scanner.scan_layer6_crypto([market])
        assert len(signals) == 0


# ================================================================
# scan_all
# ================================================================


class TestScanAll:
    """Full scan across all layers."""

    def test_scan_all_combines_layers(self, scanner: OpportunityScanner) -> None:
        """scan_all should run L1, L3, L6 and combine results."""
        markets = [
            # L1 candidate: wide spread, good volume
            _make_market(
                condition_id="c1",
                outcome_prices=["0.45", "0.50"],
                volume_24h="5000",
            ),
            # L3 candidate: NegRisk violation
            _make_market(
                condition_id="c2",
                outcome_prices=["0.45", "0.50"],
                neg_risk=True,
                volume_24h="60000",  # too high for L1
            ),
        ]
        signals = scanner.scan_all(markets)
        layers = {s.layer for s in signals}
        # c1 should trigger L1, c2 should trigger L3
        assert 1 in layers
        assert 3 in layers

    def test_scan_all_empty_markets(self, scanner: OpportunityScanner) -> None:
        """Empty market list → no signals."""
        signals = scanner.scan_all([])
        assert len(signals) == 0


# ================================================================
# Confluence scoring
# ================================================================


class TestConfluence:
    """Confluence scoring across layers."""

    def test_single_layer_confluence_1(self, scanner: OpportunityScanner) -> None:
        """Market with signal from one layer → confluence = 1."""
        signals = [
            Signal(
                layer=1,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert updated[0].confluence_score == 1

    def test_multi_layer_confluence(self, scanner: OpportunityScanner) -> None:
        """Same market from 2 layers → confluence = 2 for both."""
        signals = [
            Signal(
                layer=1,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
            ),
            Signal(
                layer=3,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.03"),
                confidence=Decimal("0.85"),
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert all(s.confluence_score == 2 for s in updated)

    def test_different_markets_independent(self, scanner: OpportunityScanner) -> None:
        """Signals from different markets → independent confluence."""
        signals = [
            Signal(
                layer=1,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
            ),
            Signal(
                layer=1,
                market_condition_id="c2",
                token_id="tok_2",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.04"),
                confidence=Decimal("0.6"),
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert all(s.confluence_score == 1 for s in updated)


# ================================================================
# Signal format
# ================================================================


class TestSignalFormat:
    """Signal dataclass and serialization."""

    def test_signal_to_db_dict(self) -> None:
        sig = Signal(
            layer=2,
            market_condition_id="cond_123",
            token_id="tok_123",
            direction=SignalDirection.BUY_YES,
            edge=Decimal("0.05"),
            confidence=Decimal("0.8"),
            confluence_score=2,
            details={"question": "Test?"},
        )
        db = sig.to_db_dict()
        assert db["layer"] == 2
        assert db["strategy"] == ""
        assert db["direction"] == "BUY_YES"
        assert db["edge"] == 0.05
        assert db["confluence_score"] == 2
        assert "question" in db["details"]

    def test_signal_with_strategy(self) -> None:
        """Signal with strategy field set."""
        sig = Signal(
            layer=7,
            market_condition_id="cond_456",
            token_id="tok_456",
            direction=SignalDirection.BUY_NO,
            edge=Decimal("0.08"),
            confidence=Decimal("0.7"),
            strategy="A",
        )
        assert sig.strategy == "A"
        db = sig.to_db_dict()
        assert db["strategy"] == "A"
        assert db["layer"] == 7

    def test_signal_direction_values(self) -> None:
        assert SignalDirection.BUY_YES.value == "BUY_YES"
        assert SignalDirection.BUY_NO.value == "BUY_NO"
        assert SignalDirection.SELL_YES.value == "SELL_YES"
        assert SignalDirection.SELL_NO.value == "SELL_NO"


# ================================================================
# ThetaDecaySignal
# ================================================================


class TestThetaDecaySignal:
    """Strategy A signal subtype."""

    def test_inherits_signal(self) -> None:
        """ThetaDecaySignal is a Signal."""
        sig = ThetaDecaySignal(
            layer=7,
            market_condition_id="cond_td",
            token_id="tok_no",
            direction=SignalDirection.BUY_NO,
            edge=Decimal("0.10"),
            confidence=Decimal("0.75"),
            strategy="A",
            z_score=3.5,
            taker_ratio=0.85,
        )
        assert isinstance(sig, Signal)
        assert sig.strategy == "A"
        assert sig.z_score == 3.5
        assert sig.taker_ratio == 0.85

    def test_to_db_dict_includes_theta_fields(self) -> None:
        """DB dict includes z_score and taker_ratio in details."""
        sig = ThetaDecaySignal(
            layer=7,
            market_condition_id="cond_td",
            token_id="tok_no",
            direction=SignalDirection.BUY_NO,
            edge=Decimal("0.10"),
            confidence=Decimal("0.75"),
            strategy="A",
            z_score=3.5,
            taker_ratio=0.85,
            details={"question": "Will X happen?"},
        )
        db = sig.to_db_dict()
        assert db["strategy"] == "A"
        assert db["details"]["z_score"] == 3.5
        assert db["details"]["taker_ratio"] == 0.85
        assert db["details"]["question"] == "Will X happen?"

    def test_default_values(self) -> None:
        """Defaults for z_score and taker_ratio."""
        sig = ThetaDecaySignal(
            layer=7,
            market_condition_id="cond_td",
            token_id="tok_no",
            direction=SignalDirection.BUY_NO,
            edge=Decimal("0.10"),
            confidence=Decimal("0.75"),
        )
        assert sig.z_score == 0.0
        assert sig.taker_ratio == 0.0
        assert sig.strategy == ""


# ================================================================
# Confluence with strategy field
# ================================================================


class TestConfluenceWithStrategy:
    """Confluence scoring uses strategy field when available."""

    def test_same_strategy_same_market(self, scanner: OpportunityScanner) -> None:
        """Two signals from same strategy → confluence = 1."""
        signals = [
            Signal(
                layer=7,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_NO,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
                strategy="A",
            ),
            Signal(
                layer=7,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_NO,
                edge=Decimal("0.08"),
                confidence=Decimal("0.7"),
                strategy="A",
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert all(s.confluence_score == 1 for s in updated)

    def test_different_strategies_same_market(self, scanner: OpportunityScanner) -> None:
        """Two signals from different strategies → confluence = 2."""
        signals = [
            Signal(
                layer=7,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_NO,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
                strategy="A",
            ),
            Signal(
                layer=2,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.08"),
                confidence=Decimal("0.7"),
                strategy="B",
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert all(s.confluence_score == 2 for s in updated)

    def test_mixed_legacy_and_strategy(self, scanner: OpportunityScanner) -> None:
        """Legacy (no strategy) + strategy signals → distinct sources."""
        signals = [
            Signal(
                layer=1,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_YES,
                edge=Decimal("0.05"),
                confidence=Decimal("0.6"),
            ),
            Signal(
                layer=7,
                market_condition_id="c1",
                token_id="tok_1",
                direction=SignalDirection.BUY_NO,
                edge=Decimal("0.08"),
                confidence=Decimal("0.7"),
                strategy="A",
            ),
        ]
        updated = scanner.compute_confluence(signals)
        assert all(s.confluence_score == 2 for s in updated)
