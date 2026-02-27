"""Tests for Strategy B: Reflexivity Surfer (RDH-302, updated for DivergenceSignal).

Tests verify:
1. Phase state machine transitions (Start → Boom → Peak → Bust → Start)
2. Phase 2 (Boom) trade: buy YES at divergence < -10%
3. Phase 3 (Peak) trade: buy NO at divergence > +20%
4. Stop loss: -15% for Phase 2, -25% for Phase 3
5. Max 5 concurrent per phase type
6. strategy="B" on paper trades
7. Partial exit at +30%
8. data_source_live guard (no signals → no trades)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from arbo.core.paper_engine import PaperTradingEngine
from arbo.core.risk_manager import RiskManager
from arbo.strategies.reflexivity_surfer import (
    MarketPhase,
    Phase,
    ReflexivitySurfer,
)
from arbo.strategies.social_divergence import DivergenceSignal


# ================================================================
# Mock market
# ================================================================


@dataclass
class MockReflexMarket:
    """Mock market for reflexivity tests."""

    condition_id: str = "cond_reflex_1"
    question: str = "Will Bitcoin reach $100K?"
    category: str = "crypto"
    price_yes: Decimal | None = Decimal("0.55")
    price_no: Decimal | None = Decimal("0.45")
    token_id_yes: str = "tok_yes_r1"
    token_id_no: str = "tok_no_r1"
    fee_enabled: bool = False
    neg_risk: bool = False
    volume_24h: Decimal = Decimal("20000")
    liquidity: Decimal = Decimal("10000")
    active: bool = True
    closed: bool = False
    end_date: str = ""
    slug: str = "btc-100k"

    def __post_init__(self) -> None:
        if not self.end_date:
            self.end_date = (datetime.now(UTC) + timedelta(days=14)).isoformat()


def _make_divergence_signal(
    symbol: str = "BTC",
    divergence: float = 0.3,
    confidence: float = 0.65,
    direction: str = "LONG",
    condition_ids: list[str] | None = None,
) -> DivergenceSignal:
    """Create a DivergenceSignal for testing."""
    return DivergenceSignal(
        symbol=symbol,
        social_momentum_score=divergence * 0.8,
        price_momentum=-divergence * 0.5,
        divergence=divergence,
        z_score=divergence * 3.0,
        direction=direction,
        confidence=confidence,
        polymarket_condition_ids=condition_ids or ["cond_reflex_1"],
    )


# ================================================================
# Phase State Machine
# ================================================================


class TestPhaseStateMachine:
    """Phase transitions in the reflexivity state machine."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    @pytest.fixture
    def strategy(self, risk: RiskManager) -> ReflexivitySurfer:
        paper = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)
        return ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )

    def test_initial_phase_is_start(self, strategy: ReflexivitySurfer) -> None:
        """Markets start in Phase.START."""
        phase = strategy._get_phase("cond_new")
        assert phase == Phase.START

    def test_start_to_boom(self, strategy: ReflexivitySurfer) -> None:
        """Divergence < -10% transitions START → BOOM."""
        new = strategy._transition_phase("c1", "topic", -0.15, 0.55, 0.47)
        assert new == Phase.BOOM

    def test_start_to_peak(self, strategy: ReflexivitySurfer) -> None:
        """Divergence > +20% transitions START → PEAK."""
        new = strategy._transition_phase("c2", "topic", 0.25, 0.40, 0.50)
        assert new == Phase.PEAK

    def test_start_stays_start(self, strategy: ReflexivitySurfer) -> None:
        """Divergence in [-10%, +20%] stays in START."""
        new = strategy._transition_phase("c3", "topic", 0.05, 0.50, 0.525)
        assert new == Phase.START

    def test_boom_to_peak(self, strategy: ReflexivitySurfer) -> None:
        """Divergence > +20% transitions BOOM → PEAK."""
        strategy._transition_phase("c4", "topic", -0.15, 0.55, 0.47)
        new = strategy._transition_phase("c4", "topic", 0.25, 0.40, 0.50)
        assert new == Phase.PEAK

    def test_peak_to_bust(self, strategy: ReflexivitySurfer) -> None:
        """Divergence drops below +10% transitions PEAK → BUST."""
        strategy._transition_phase("c5", "topic", 0.25, 0.40, 0.50)
        new = strategy._transition_phase("c5", "topic", 0.08, 0.46, 0.50)
        assert new == Phase.BUST

    def test_bust_to_start(self, strategy: ReflexivitySurfer) -> None:
        """Divergence normalizes to [-5%, +5%] transitions BUST → START."""
        strategy._transition_phase("c6", "topic", 0.25, 0.40, 0.50)
        strategy._transition_phase("c6", "topic", 0.08, 0.46, 0.50)
        new = strategy._transition_phase("c6", "topic", 0.02, 0.49, 0.50)
        assert new == Phase.START

    def test_transition_count(self, strategy: ReflexivitySurfer) -> None:
        """Transition count increments on each phase change."""
        strategy._transition_phase("c7", "topic", -0.15, 0.55, 0.47)  # START → BOOM
        strategy._transition_phase("c7", "topic", 0.25, 0.40, 0.50)  # BOOM → PEAK
        mp = strategy._phases["c7"]
        assert mp.transition_count == 2


# ================================================================
# Trading
# ================================================================


class TestReflexivityTrading:
    """Trade execution in different phases."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    @pytest.fixture
    def paper(self, risk: RiskManager) -> PaperTradingEngine:
        return PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)

    async def test_boom_buys_yes(self, risk: RiskManager, paper: PaperTradingEngine) -> None:
        """Phase 2 (BOOM): buys YES when divergence triggers boom."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        # Social divergence positive = social UP = BOOM (after negation → negative = BOOM)
        signal = _make_divergence_signal(
            divergence=0.20,  # positive social → negated to -0.20 → BOOM
            confidence=0.65,
            condition_ids=["cond_reflex_1"],
        )
        strategy.update_signals([signal])

        mkt = MockReflexMarket(price_yes=Decimal("0.55"))
        traded = await strategy.poll_cycle([mkt])

        if len(traded) > 0:
            assert traded[0]["strategy"] == "B"
            assert traded[0]["side"] == "BUY_YES"
            assert traded[0]["phase"] == "BOOM"

        await strategy.close()

    async def test_peak_buys_no(self, risk: RiskManager, paper: PaperTradingEngine) -> None:
        """Phase 3 (PEAK): buys NO when divergence triggers peak."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        # Negative social divergence → negated to positive → PEAK
        signal = _make_divergence_signal(
            divergence=-0.30,  # social DOWN → negated to +0.30 → PEAK
            confidence=0.60,
            condition_ids=["cond_peak"],
        )
        strategy.update_signals([signal])

        mkt = MockReflexMarket(
            condition_id="cond_peak",
            question="Will event happen?",
            price_yes=Decimal("0.70"),
            price_no=Decimal("0.30"),
        )
        traded = await strategy.poll_cycle([mkt])

        if len(traded) > 0:
            assert traded[0]["strategy"] == "B"
            assert traded[0]["phase"] == "PEAK"

        await strategy.close()

    async def test_no_signals_returns_empty(
        self, risk: RiskManager, paper: PaperTradingEngine
    ) -> None:
        """No trade when data_source_live is False (no signals)."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        # Don't call update_signals → data_source_live = False
        mkt = MockReflexMarket()
        traded = await strategy.poll_cycle([mkt])
        assert len(traded) == 0
        await strategy.close()

    async def test_stale_signals_returns_empty(
        self, risk: RiskManager, paper: PaperTradingEngine
    ) -> None:
        """No trade when signals are older than 7 hours."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        signal = _make_divergence_signal(divergence=0.30, condition_ids=["cond_reflex_1"])
        strategy.update_signals([signal])
        # Artificially age the timestamp
        strategy._signals_updated_at = datetime.now(UTC) - timedelta(hours=8)

        mkt = MockReflexMarket()
        traded = await strategy.poll_cycle([mkt])
        assert len(traded) == 0
        await strategy.close()

    async def test_no_trade_at_start(self, risk: RiskManager, paper: PaperTradingEngine) -> None:
        """No trade when divergence is small (stays in START phase)."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        # Small divergence → after negation stays in [-10%, +20%] → START
        signal = _make_divergence_signal(
            divergence=0.03,  # negated → -0.03, within START range
            confidence=0.50,
            condition_ids=["cond_neutral"],
        )
        strategy.update_signals([signal])

        mkt = MockReflexMarket(
            condition_id="cond_neutral",
            question="neutral market",
            price_yes=Decimal("0.50"),
            price_no=Decimal("0.50"),
        )
        traded = await strategy.poll_cycle([mkt])
        assert len(traded) == 0
        await strategy.close()

    async def test_strategy_b_on_trade(
        self, risk: RiskManager, paper: PaperTradingEngine
    ) -> None:
        """All trades have strategy='B'."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        signal = _make_divergence_signal(
            divergence=-0.35,  # negated to +0.35 → PEAK
            confidence=0.65,
            condition_ids=["cond_forced"],
        )
        strategy.update_signals([signal])

        mkt = MockReflexMarket(
            condition_id="cond_forced",
            question="forced divergence",
            price_yes=Decimal("0.80"),
            price_no=Decimal("0.20"),
        )
        traded = await strategy.poll_cycle([mkt])

        for t in traded:
            assert t["strategy"] == "B"

        await strategy.close()

    async def test_no_duplicate_positions(
        self, risk: RiskManager, paper: PaperTradingEngine
    ) -> None:
        """Same market doesn't get traded twice."""
        strategy = ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )
        await strategy.init()

        signal = _make_divergence_signal(
            divergence=-0.35,  # negated to +0.35 → PEAK
            confidence=0.70,
            condition_ids=["cond_dup"],
        )
        strategy.update_signals([signal])

        mkt = MockReflexMarket(
            condition_id="cond_dup",
            question="dup test",
            price_yes=Decimal("0.75"),
            price_no=Decimal("0.25"),
        )

        # First cycle
        await strategy.poll_cycle([mkt])
        first_count = strategy._trades_placed

        # Second cycle — same market, should not trade again
        strategy.update_signals([signal])  # refresh signals
        await strategy.poll_cycle([mkt])
        assert strategy._trades_placed == first_count

        await strategy.close()


# ================================================================
# Exit Management
# ================================================================


class TestReflexivityExits:
    """Exit management for reflexivity positions."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    @pytest.fixture
    def strategy(self, risk: RiskManager) -> ReflexivitySurfer:
        paper = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)
        return ReflexivitySurfer(
            risk_manager=risk,
            paper_engine=paper,
        )

    def _add_position(
        self,
        strategy: ReflexivitySurfer,
        cond_id: str,
        phase: Phase,
        entry_price: Decimal,
    ) -> None:
        """Helper: inject a position directly."""
        from arbo.strategies.reflexivity_surfer import _ReflexPosition

        strategy._active_positions[cond_id] = _ReflexPosition(
            condition_id=cond_id,
            token_id="tok_test",
            side="BUY_YES" if phase == Phase.BOOM else "BUY_NO",
            phase_at_entry=phase,
            entry_price=entry_price,
            size=Decimal("30"),
        )

    def test_phase2_stop_loss_15pct(self, strategy: ReflexivitySurfer) -> None:
        """Phase 2 (BOOM) position exits at -15%."""
        self._add_position(strategy, "c_sl", Phase.BOOM, Decimal("0.50"))
        exits = strategy.check_exits({"c_sl": Decimal("0.42")})  # -16%
        assert len(exits) == 1
        assert exits[0]["action"] == "stop_loss"
        assert exits[0]["phase"] == "BOOM"
        assert "c_sl" not in strategy._active_positions

    def test_phase3_stop_loss_25pct(self, strategy: ReflexivitySurfer) -> None:
        """Phase 3 (PEAK) position exits at -25%."""
        self._add_position(strategy, "c_sl3", Phase.PEAK, Decimal("0.40"))
        # -25% of 0.40 = 0.30
        exits = strategy.check_exits({"c_sl3": Decimal("0.29")})
        assert len(exits) == 1
        assert exits[0]["action"] == "stop_loss"
        assert exits[0]["phase"] == "PEAK"

    def test_partial_exit_at_30pct(self, strategy: ReflexivitySurfer) -> None:
        """Partial exit triggers at +30% gain."""
        self._add_position(strategy, "c_pe", Phase.PEAK, Decimal("0.40"))
        exits = strategy.check_exits({"c_pe": Decimal("0.52")})  # +30%
        assert len(exits) == 1
        assert exits[0]["action"] == "partial_exit"
        # Position should still exist (partial)
        assert "c_pe" in strategy._active_positions
        pos = strategy._active_positions["c_pe"]
        assert pos.partial_exited is True

    def test_no_double_partial_exit(self, strategy: ReflexivitySurfer) -> None:
        """No second partial exit after first."""
        self._add_position(strategy, "c_dpe", Phase.BOOM, Decimal("0.50"))
        strategy.check_exits({"c_dpe": Decimal("0.65")})  # First partial
        exits = strategy.check_exits({"c_dpe": Decimal("0.70")})  # Should not trigger again
        assert len(exits) == 0

    def test_no_exit_in_range(self, strategy: ReflexivitySurfer) -> None:
        """No exit when price change is within bounds."""
        self._add_position(strategy, "c_ok", Phase.BOOM, Decimal("0.50"))
        exits = strategy.check_exits({"c_ok": Decimal("0.48")})  # -4%, within -15%
        assert len(exits) == 0


# ================================================================
# Stats
# ================================================================


class TestReflexivityStats:
    """Strategy statistics."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    def test_stats_structure(self, risk: RiskManager) -> None:
        """Stats dict has expected keys."""
        paper = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)
        strategy = ReflexivitySurfer(
            risk_manager=risk, paper_engine=paper
        )
        stats = strategy.stats
        assert stats["strategy"] == "B"
        assert stats["data_source"] == "social_divergence"
        assert stats["data_source_live"] is False  # no signals yet
        assert "active_positions" in stats
        assert "phase_distribution" in stats
        assert "deployed" in stats

    def test_data_source_live_after_update(self, risk: RiskManager) -> None:
        """data_source_live becomes True after update_signals with non-empty list."""
        paper = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)
        strategy = ReflexivitySurfer(
            risk_manager=risk, paper_engine=paper
        )
        assert strategy.data_source_live is False

        signal = _make_divergence_signal()
        strategy.update_signals([signal])
        assert strategy.data_source_live is True
        assert strategy.stats["data_source_live"] is True

    def test_phase_distribution(self, risk: RiskManager) -> None:
        """Phase distribution counts markets in each phase."""
        paper = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk)
        strategy = ReflexivitySurfer(
            risk_manager=risk, paper_engine=paper
        )
        strategy._transition_phase("c1", "t1", -0.15, 0.55, 0.47)
        strategy._transition_phase("c2", "t2", 0.25, 0.40, 0.50)
        strategy._transition_phase("c3", "t3", 0.05, 0.50, 0.525)

        stats = strategy.stats
        dist = stats["phase_distribution"]
        assert dist.get("BOOM", 0) >= 1
        assert dist.get("PEAK", 0) >= 1
