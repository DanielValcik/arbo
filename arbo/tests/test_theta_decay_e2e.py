"""E2E integration tests for Strategy A: Theta Decay (RDH-211).

Tests verify the full pipeline:
1. Mock market → register with flow tracker
2. Inject 3σ YES taker spike
3. Theta decay detects peak optimism → signal
4. Quality gate validates market criteria
5. Paper trade placed (buy NO) with strategy="A"
6. Risk manager updated with Strategy A accounting
7. Exit management (partial exit, stop loss)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from arbo.connectors.polygon_flow import MarketFlowTracker, PeakOptimismResult
from arbo.core.paper_engine import PaperTradingEngine
from arbo.core.risk_manager import RiskManager
from arbo.core.scanner import Signal, SignalDirection, ThetaDecaySignal
from arbo.strategies.theta_decay import ThetaDecay
from arbo.strategies.theta_decay_gate import check_allocation, check_market_quality


@dataclass
class MockGammaMarket:
    """Mock market matching GammaMarket interface for E2E tests."""

    condition_id: str = "cond_e2e_longshot"
    question: str = "Will underdog win?"
    category: str = "politics"
    price_yes: Decimal | None = Decimal("0.10")
    price_no: Decimal | None = Decimal("0.90")
    token_id_yes: str = "tok_yes_e2e"
    token_id_no: str = "tok_no_e2e"
    fee_enabled: bool = False
    neg_risk: bool = False
    volume_24h: Decimal = Decimal("50000")
    liquidity: Decimal = Decimal("25000")
    active: bool = True
    closed: bool = False
    end_date: str = ""
    slug: str = "longshot-e2e"

    def __post_init__(self) -> None:
        if not self.end_date:
            self.end_date = (datetime.now(UTC) + timedelta(days=10)).isoformat()


def _inject_3sigma_spike(
    flow_tracker: MarketFlowTracker,
    condition_id: str,
) -> None:
    """Inject a 3σ YES taker flow spike for a registered market."""
    flow = flow_tracker.get_market_flow(condition_id)
    if flow is None:
        return

    now = time.monotonic()
    flow.yes_window._time_fn = lambda: now

    # Varying historical volume in 5-min buckets
    for i in range(1, 13):
        vol = Decimal(str(8 + i))
        flow.yes_window.add(vol, is_buy=True, timestamp=now - i * 300 - 1)

    # Massive spike in current bucket → 3σ+
    flow.yes_window.add(Decimal("2000"), is_buy=True, timestamp=now - 1)


class TestE2EFullPipeline:
    """Full E2E: flow spike → theta decay → quality gate → paper trade."""

    @pytest.fixture
    def risk(self) -> RiskManager:
        RiskManager.reset()
        return RiskManager(capital=Decimal("2000"))

    @pytest.fixture
    def paper(self, risk: RiskManager) -> PaperTradingEngine:
        return PaperTradingEngine(
            initial_capital=Decimal("2000"),
            risk_manager=risk,
        )

    @pytest.fixture
    def flow_tracker(self) -> MarketFlowTracker:
        return MarketFlowTracker()

    @pytest.fixture
    def strategy(
        self,
        risk: RiskManager,
        flow_tracker: MarketFlowTracker,
        paper: PaperTradingEngine,
    ) -> ThetaDecay:
        return ThetaDecay(
            risk_manager=risk,
            flow_tracker=flow_tracker,
            paper_engine=paper,
        )

    async def test_e2e_spike_to_trade(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper: PaperTradingEngine,
        risk: RiskManager,
    ) -> None:
        """3σ spike on longshot → buy NO trade → strategy='A' in paper engine."""
        await strategy.init()

        mkt = MockGammaMarket()

        # Poll 1: register market with flow tracker
        await strategy.poll_cycle([mkt])
        assert mkt.condition_id in strategy._registered_markets

        # Inject 3σ spike
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)

        # Poll 2: detect spike and trade
        traded = await strategy.poll_cycle([mkt])

        # Verify trade was placed
        assert len(traded) >= 1
        assert traded[0]["side"] == "BUY_NO"
        assert traded[0]["strategy"] == "A"
        assert traded[0]["size"] > 0

        # Paper engine has the trade
        assert len(paper.trade_history) >= 1
        trade = paper.trade_history[0]
        assert trade.strategy == "A"
        assert trade.side == "BUY"
        assert trade.token_id == "tok_no_e2e"

        # Risk manager Strategy A capital deployed
        state_a = risk.get_strategy_state("A")
        assert state_a is not None
        assert state_a.deployed > Decimal("0")
        assert state_a.position_count >= 1

        await strategy.close()

    async def test_e2e_quality_gate_before_trade(
        self,
        risk: RiskManager,
    ) -> None:
        """Quality gate approves valid market + rejects bad ones."""
        mkt = MockGammaMarket()
        peak = PeakOptimismResult(
            is_peak=True,
            zscore=3.5,
            yes_ratio=0.85,
            condition_id=mkt.condition_id,
        )

        # Valid market passes
        decision = check_market_quality(mkt, peak=peak)
        assert decision.passed is True

        # Allocation check passes with fresh capital
        alloc = check_allocation(risk, "A")
        assert alloc.passed is True

        # Bad market: crypto category rejected
        bad_mkt = MockGammaMarket(category="crypto")
        decision = check_market_quality(bad_mkt, peak=peak)
        assert decision.passed is False
        assert "excluded" in decision.reason.lower()

        # Bad market: too high price (not longshot)
        expensive = MockGammaMarket(price_yes=Decimal("0.50"))
        decision = check_market_quality(expensive, peak=peak)
        assert decision.passed is False

    async def test_e2e_no_duplicate_positions(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper: PaperTradingEngine,
    ) -> None:
        """Same market doesn't get traded twice."""
        await strategy.init()
        mkt = MockGammaMarket()

        # First: register + spike + trade
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)
        await strategy.poll_cycle([mkt])
        first_count = len(paper.trade_history)
        assert first_count >= 1

        # Second spike: should not trade again (already have position)
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)
        await strategy.poll_cycle([mkt])
        assert len(paper.trade_history) == first_count

        await strategy.close()

    async def test_e2e_exit_management(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper: PaperTradingEngine,
    ) -> None:
        """Partial exit at +50%, stop loss at -30%."""
        await strategy.init()
        mkt = MockGammaMarket()

        # Trade
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)
        await strategy.poll_cycle([mkt])
        assert mkt.condition_id in strategy._active_positions

        pos = strategy._active_positions[mkt.condition_id]
        entry = pos.entry_price

        # Partial exit at +50%
        exits = strategy.check_exits({mkt.condition_id: entry * Decimal("1.50")})
        assert len(exits) == 1
        assert exits[0]["action"] == "partial_exit"
        assert pos.partial_exited is True

        # No double partial exit
        exits = strategy.check_exits({mkt.condition_id: entry * Decimal("1.55")})
        assert len(exits) == 0

        # Stop loss at -30% (from entry)
        exits = strategy.check_exits({mkt.condition_id: entry * Decimal("0.70")})
        assert len(exits) == 1
        assert exits[0]["action"] == "stop_loss"
        assert mkt.condition_id not in strategy._active_positions

        await strategy.close()

    async def test_e2e_strategy_stats(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
    ) -> None:
        """Stats reflect signals and trades after pipeline run."""
        await strategy.init()
        mkt = MockGammaMarket()

        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)
        await strategy.poll_cycle([mkt])

        stats = strategy.stats
        assert stats["strategy"] == "A"
        assert stats["signals_generated"] >= 1
        assert stats["trades_placed"] >= 1
        assert stats["active_positions"] >= 1

        await strategy.close()

    async def test_e2e_theta_decay_signal_dto(self) -> None:
        """ThetaDecaySignal subtype works as a Signal with extra fields."""
        sig = ThetaDecaySignal(
            layer=7,
            market_condition_id="cond_e2e",
            token_id="tok_no_e2e",
            direction=SignalDirection.BUY_NO,
            edge=Decimal("0.10"),
            confidence=Decimal("0.75"),
            strategy="A",
            z_score=3.5,
            taker_ratio=0.85,
        )
        assert isinstance(sig, Signal)
        assert sig.strategy == "A"

        db_dict = sig.to_db_dict()
        assert db_dict["strategy"] == "A"
        assert db_dict["details"]["z_score"] == 3.5
        assert db_dict["details"]["taker_ratio"] == 0.85

    async def test_e2e_capital_isolation(
        self,
        strategy: ThetaDecay,
        flow_tracker: MarketFlowTracker,
        paper: PaperTradingEngine,
        risk: RiskManager,
    ) -> None:
        """Strategy A trades don't affect Strategy C capital."""
        await strategy.init()

        # Get initial states
        state_a_before = risk.get_strategy_state("A")
        state_c_before = risk.get_strategy_state("C")
        assert state_a_before is not None
        assert state_c_before is not None
        c_deployed_before = state_c_before.deployed

        # Execute Strategy A trade
        mkt = MockGammaMarket()
        await strategy.poll_cycle([mkt])
        _inject_3sigma_spike(flow_tracker, mkt.condition_id)
        await strategy.poll_cycle([mkt])

        # Strategy A deployed should increase
        state_a_after = risk.get_strategy_state("A")
        assert state_a_after is not None
        assert state_a_after.deployed > Decimal("0")

        # Strategy C should be unchanged
        state_c_after = risk.get_strategy_state("C")
        assert state_c_after is not None
        assert state_c_after.deployed == c_deployed_before

        await strategy.close()
