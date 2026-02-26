"""Tests for Reflexivity quality gate (RDH-303).

Tests verify:
1. Approves Phase 2 (BOOM) signal with sufficient divergence
2. Approves Phase 3 (PEAK) signal with sufficient divergence
3. Rejects Phase 1 (START) — no trading
4. Rejects insufficient divergence
5. Rejects low volume/liquidity
6. Allocation check works
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from arbo.core.risk_manager import RiskManager
from arbo.strategies.reflexivity_gate import (
    check_reflexivity_allocation,
    check_reflexivity_signal,
)
from arbo.strategies.reflexivity_surfer import Phase


@dataclass
class MockGateMarket:
    """Mock market for gate tests."""

    condition_id: str = "cond_gate_1"
    question: str = "Will event happen?"
    category: str = "politics"
    price_yes: Decimal | None = Decimal("0.60")
    price_no: Decimal | None = Decimal("0.40")
    token_id_yes: str = "tok_yes_g"
    token_id_no: str = "tok_no_g"
    fee_enabled: bool = False
    volume_24h: Decimal = Decimal("20000")
    liquidity: Decimal = Decimal("10000")
    active: bool = True
    closed: bool = False


class TestReflexivitySignalGate:
    """Reflexivity signal quality gate checks."""

    def test_approves_phase2_signal(self) -> None:
        """Phase 2 (BOOM) signal with divergence < -10% passes."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.15)
        assert decision.passed is True
        assert "BOOM" in decision.reason

    def test_approves_phase3_signal(self) -> None:
        """Phase 3 (PEAK) signal with divergence > +20% passes."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.PEAK, 0.25)
        assert decision.passed is True
        assert "PEAK" in decision.reason

    def test_rejects_start_phase(self) -> None:
        """Phase 1 (START) is always rejected — no trading."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.START, 0.05)
        assert decision.passed is False
        assert "START" in decision.reason

    def test_rejects_insufficient_boom_divergence(self) -> None:
        """Phase 2 rejected if divergence not negative enough."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.05)  # Above -10%
        assert decision.passed is False
        assert "threshold" in decision.reason.lower()

    def test_rejects_insufficient_peak_divergence(self) -> None:
        """Phase 3 rejected if divergence not positive enough."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.PEAK, 0.15)  # Below +20%
        assert decision.passed is False
        assert "threshold" in decision.reason.lower()

    def test_rejects_low_volume(self) -> None:
        """Market with low volume is rejected."""
        mkt = MockGateMarket(volume_24h=Decimal("1000"))
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.15)
        assert decision.passed is False
        assert "volume" in decision.reason.lower()

    def test_rejects_low_liquidity(self) -> None:
        """Market with low liquidity is rejected."""
        mkt = MockGateMarket(liquidity=Decimal("500"))
        decision = check_reflexivity_signal(mkt, Phase.PEAK, 0.25)
        assert decision.passed is False
        assert "liquidity" in decision.reason.lower()

    def test_rejects_inactive_market(self) -> None:
        """Inactive market is rejected."""
        mkt = MockGateMarket(active=False)
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.15)
        assert decision.passed is False
        assert "inactive" in decision.reason.lower()

    def test_rejects_extreme_price(self) -> None:
        """Extreme prices (near 0 or 1) are rejected."""
        mkt = MockGateMarket(price_yes=Decimal("0.01"))
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.15)
        assert decision.passed is False
        assert "extreme" in decision.reason.lower()

    def test_rejects_no_price(self) -> None:
        """Market without price data is rejected."""
        mkt = MockGateMarket(price_yes=None)
        decision = check_reflexivity_signal(mkt, Phase.PEAK, 0.25)
        assert decision.passed is False

    def test_rejects_missing_tokens(self) -> None:
        """Market without token IDs is rejected."""
        mkt = MockGateMarket(token_id_yes="", token_id_no="")
        decision = check_reflexivity_signal(mkt, Phase.BOOM, -0.15)
        assert decision.passed is False
        assert "token" in decision.reason.lower()

    def test_bust_phase_approved(self) -> None:
        """Phase 4 (BUST) with any divergence passes (holding)."""
        mkt = MockGateMarket()
        decision = check_reflexivity_signal(mkt, Phase.BUST, 0.05)
        assert decision.passed is True


class TestReflexivityAllocation:
    """Allocation check for Strategy B."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    def test_allocation_available(self, risk: RiskManager) -> None:
        """Fresh allocation passes."""
        decision = check_reflexivity_allocation(risk, "B")
        assert decision.passed is True
        assert "$" in decision.reason

    def test_allocation_halted(self, risk: RiskManager) -> None:
        """Halted strategy is rejected."""
        state = risk.get_strategy_state("B")
        if state is not None:
            state.is_halted = True
        decision = check_reflexivity_allocation(risk, "B")
        assert decision.passed is False
        assert "halted" in decision.reason.lower()

    def test_allocation_exhausted(self, risk: RiskManager) -> None:
        """Exhausted capital is rejected."""
        state = risk.get_strategy_state("B")
        if state is not None:
            state.deployed = state.allocated  # Deploy everything
        decision = check_reflexivity_allocation(risk, "B")
        assert decision.passed is False
        assert "exhausted" in decision.reason.lower()
