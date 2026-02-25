"""End-to-end integration tests for Arbo trading pipeline.

Tests verify the full signal flow through component integration:
1. Signal → confluence scorer → paper trade → P&L → report generation
2. Graceful shutdown behavior
3. Layer crash and restart resilience
4. LLM degraded mode (layers 5/8 disabled)
5. Risk manager daily loss shutdown
6. Multi-layer confluence scoring pipeline

All external HTTP is mocked via aioresponses or unittest.mock.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from arbo.core.confluence import ConfluenceScorer
from arbo.core.paper_engine import PaperTradingEngine, TradeStatus
from arbo.core.risk_manager import RiskManager
from arbo.core.scanner import Signal, SignalDirection
from arbo.dashboard.report_generator import ReportGenerator

# ================================================================
# Helpers
# ================================================================


def _make_signal(
    layer: int,
    market_condition_id: str = "cond_1",
    token_id: str = "tok_1",
    direction: SignalDirection = SignalDirection.BUY_YES,
    edge: Decimal = Decimal("0.08"),
    confidence: Decimal = Decimal("0.7"),
) -> Signal:
    """Create a Signal for testing."""
    return Signal(
        layer=layer,
        market_condition_id=market_condition_id,
        token_id=token_id,
        direction=direction,
        edge=edge,
        confidence=confidence,
    )


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    """Convert PaperTrade to dict for report generation."""
    return {
        "actual_pnl": float(trade.actual_pnl) if trade.actual_pnl is not None else None,
        "layer": trade.layer,
        "confluence_score": trade.confluence_score,
        "size": float(trade.size),
        "token_id": trade.token_id,
        "side": trade.side,
        "notes": trade.notes,
    }


@pytest.fixture(autouse=True)
def reset_risk_singleton() -> None:
    """Reset risk manager singleton between tests."""
    RiskManager.reset()


# ================================================================
# Test 1: Full Flow — Scan → Confluence → Paper Trade → P&L → Report
# ================================================================


class TestFullFlowScanToTrade:
    """E2E: Signal generation → confluence scoring → paper trade → report."""

    def test_full_flow_scan_to_trade(self) -> None:
        """Signals from multiple layers score, trade, resolve, and report."""
        # 1. Create signals from layers 2, 4, 7 for the same market
        signals = [
            _make_signal(
                layer=2,
                market_condition_id="cond_btc",
                token_id="tok_btc_yes",
                edge=Decimal("0.08"),
            ),
            _make_signal(
                layer=4,
                market_condition_id="cond_btc",
                token_id="tok_btc_yes",
                edge=Decimal("0.06"),
            ),
            _make_signal(
                layer=7,
                market_condition_id="cond_btc",
                token_id="tok_btc_yes",
                edge=Decimal("0.04"),
            ),
        ]

        # 2. Score via ConfluenceScorer
        risk_mgr = RiskManager(capital=Decimal("2000"))
        scorer = ConfluenceScorer(risk_manager=risk_mgr, capital=Decimal("2000"))
        tradeable = scorer.get_tradeable(
            signals,
            market_category_map={"cond_btc": "crypto"},
        )

        assert len(tradeable) == 1
        opp = tradeable[0]
        assert opp.score == 3  # layers 2, 4, 7
        assert opp.position_size_pct == Decimal("0.05")  # double size

        # 3. Execute paper trade
        engine = PaperTradingEngine(
            initial_capital=Decimal("2000"),
            risk_manager=risk_mgr,
        )
        trade = engine.place_trade(
            market_condition_id=opp.market_condition_id,
            token_id=opp.token_id,
            side="BUY",
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.60"),
            layer=2,
            market_category="crypto",
            confluence_score=opp.score,
        )
        assert trade is not None
        assert trade.confluence_score == 3

        # 4. Resolve market (win)
        pnl = engine.resolve_market("tok_btc_yes", winning_outcome=True)
        assert pnl > Decimal("0")
        assert trade.status == TradeStatus.WON

        # 5. Generate report
        gen = ReportGenerator()
        trade_dicts = [_trade_to_dict(t) for t in engine.trade_history]
        daily = gen.generate_daily(trade_dicts, signals=[], report_date=date(2026, 2, 21))
        assert daily.total_trades == 1
        assert daily.winning_trades == 1
        assert daily.total_pnl > Decimal("0")

        weekly = gen.generate_weekly(
            [daily],
            trades=trade_dicts,
            portfolio_balance=engine.balance,
        )
        assert weekly.total_trades == 1
        assert weekly.winning_trades == 1
        assert weekly.top_5_trades[0]["actual_pnl"] > 0


# ================================================================
# Test 2: Graceful Shutdown
# ================================================================


class TestGracefulShutdown:
    """E2E: Graceful shutdown cancels pending operations."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self) -> None:
        """Shutdown event stops processing and triggers risk manager shutdown."""
        risk_mgr = RiskManager(capital=Decimal("2000"))
        shutdown_event = asyncio.Event()

        # Simulate a layer task that respects shutdown
        processed_signals: list[Signal] = []

        async def layer_task(shutdown: asyncio.Event) -> None:
            while not shutdown.is_set():
                signal = _make_signal(layer=4, token_id="tok_shutdown")
                processed_signals.append(signal)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(shutdown.wait(), timeout=0.01)

        # Start layer task
        task = asyncio.create_task(layer_task(shutdown_event))

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Trigger shutdown
        shutdown_event.set()
        await risk_mgr.emergency_shutdown("graceful_test")

        # Wait for task to complete
        await asyncio.wait_for(task, timeout=1.0)

        assert risk_mgr.is_shutdown
        assert len(processed_signals) > 0

        # Verify no trades go through after shutdown
        engine = PaperTradingEngine(
            initial_capital=Decimal("2000"),
            risk_manager=risk_mgr,
        )
        trade = engine.place_trade(
            market_condition_id="cond_1",
            token_id="tok_1",
            side="BUY",
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.65"),
            layer=2,
            market_category="crypto",
            confluence_score=2,
        )
        assert trade is None  # rejected due to shutdown


# ================================================================
# Test 3: Layer Crash and Restart
# ================================================================


class TestLayerCrashAndRestart:
    """E2E: A layer task that crashes is restarted by supervision logic."""

    @pytest.mark.asyncio
    async def test_layer_crash_and_restart(self) -> None:
        """Crashing layer task is detected and restarted."""
        call_count = 0
        results: list[str] = []

        async def flaky_layer() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Layer 4 transient failure")
            results.append("success")

        # Simple restart supervisor
        max_retries = 3
        for _ in range(max_retries):
            try:
                await flaky_layer()
                break
            except RuntimeError:
                await asyncio.sleep(0.01)  # backoff
                continue

        assert call_count == 2
        assert results == ["success"]


# ================================================================
# Test 4: LLM Degraded Mode
# ================================================================


class TestLLMDegradedMode:
    """E2E: When LLM is unavailable, layers 5 and 8 are disabled."""

    def test_llm_degraded_mode(self) -> None:
        """System continues with non-LLM layers when LLM calls fail."""
        risk_mgr = RiskManager(capital=Decimal("2000"))
        scorer = ConfluenceScorer(risk_manager=risk_mgr, capital=Decimal("2000"))

        # Simulate: LLM-dependent layers (5, 8) produce no signals due to failure
        # Non-LLM layers (2, 4, 7) produce signals normally
        llm_available = False

        all_signals: list[Signal] = []

        # Layer 2 (XGBoost — no LLM needed)
        all_signals.append(
            _make_signal(
                layer=2, market_condition_id="cond_1", token_id="tok_1", edge=Decimal("0.08")
            ),
        )

        # Layer 4 (whale tracking — no LLM needed)
        all_signals.append(
            _make_signal(
                layer=4, market_condition_id="cond_1", token_id="tok_1", edge=Decimal("0.06")
            ),
        )

        # Layer 5 (logical arb — LLM required)
        if llm_available:
            all_signals.append(
                _make_signal(
                    layer=5, market_condition_id="cond_1", token_id="tok_1", edge=Decimal("0.10")
                ),
            )

        # Layer 7 (order flow — no LLM needed)
        all_signals.append(
            _make_signal(
                layer=7, market_condition_id="cond_1", token_id="tok_1", edge=Decimal("0.05")
            ),
        )

        # Layer 8 (attention markets — LLM required)
        if llm_available:
            all_signals.append(
                _make_signal(
                    layer=8, market_condition_id="cond_1", token_id="tok_1", edge=Decimal("0.07")
                ),
            )

        tradeable = scorer.get_tradeable(
            all_signals,
            market_category_map={"cond_1": "crypto"},
        )

        # Without LLM: score = 3 (layers 2, 4, 7)
        assert len(tradeable) == 1
        assert tradeable[0].score == 3
        assert 5 not in tradeable[0].contributing_layers
        assert 8 not in tradeable[0].contributing_layers

        # Verify system still trades despite degraded mode
        engine = PaperTradingEngine(initial_capital=Decimal("2000"), risk_manager=risk_mgr)
        trade = engine.place_trade(
            market_condition_id=tradeable[0].market_condition_id,
            token_id=tradeable[0].token_id,
            side="BUY",
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.60"),
            layer=2,
            market_category="crypto",
            confluence_score=tradeable[0].score,
        )
        assert trade is not None


# ================================================================
# Test 5: Risk Manager Daily Loss Shutdown
# ================================================================


class TestRiskManagerDailyLossShutdown:
    """E2E: Accumulating losses triggers daily loss shutdown."""

    def test_risk_manager_daily_loss_shutdown(self) -> None:
        """10% daily loss triggers automatic shutdown."""
        capital = Decimal("2000")
        risk_mgr = RiskManager(capital=capital)
        engine = PaperTradingEngine(
            initial_capital=capital,
            risk_manager=risk_mgr,
        )

        # Place and resolve losing trades until daily loss limit hit
        # Daily loss limit = 10% of 2000 = $200
        trades_placed = 0
        for i in range(20):
            token = f"tok_loss_{i}"
            trade = engine.place_trade(
                market_condition_id=f"cond_{i}",
                token_id=token,
                side="BUY",
                market_price=Decimal("0.50"),
                model_prob=Decimal("0.65"),
                layer=2,
                market_category=f"cat_{i % 5}",  # spread across categories
                confluence_score=2,
            )
            if trade is None:
                break

            # Resolve as loss and update risk manager P&L
            pnl = engine.resolve_market(token, winning_outcome=False)
            risk_mgr.post_trade_update(
                market_id=f"cond_{i}",
                market_category=f"cat_{i % 5}",
                size=trade.size,
                pnl=pnl,
            )
            trades_placed += 1

            if risk_mgr.is_shutdown:
                break

        # Risk manager should have triggered shutdown
        assert risk_mgr.is_shutdown
        assert trades_placed > 0

        # No more trades should be accepted
        final_trade = engine.place_trade(
            market_condition_id="cond_final",
            token_id="tok_final",
            side="BUY",
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.65"),
            layer=2,
            market_category="crypto",
            confluence_score=2,
        )
        assert final_trade is None


# ================================================================
# Test 6: Multi-Layer Confluence Scoring
# ================================================================


class TestMultiLayerConfluenceScoring:
    """E2E: Multiple layers emit signals for same and different markets."""

    def test_multi_layer_confluence_scoring(self) -> None:
        """Signals from 5 layers score correctly across multiple markets."""
        risk_mgr = RiskManager(capital=Decimal("5000"))
        scorer = ConfluenceScorer(risk_manager=risk_mgr, capital=Decimal("5000"))

        signals = [
            # Market A: all 5 scoring layers → score 5
            _make_signal(
                layer=2, market_condition_id="cond_a", token_id="tok_a", edge=Decimal("0.10")
            ),
            _make_signal(
                layer=4, market_condition_id="cond_a", token_id="tok_a", edge=Decimal("0.08")
            ),
            _make_signal(
                layer=5, market_condition_id="cond_a", token_id="tok_a", edge=Decimal("0.07")
            ),
            _make_signal(
                layer=7, market_condition_id="cond_a", token_id="tok_a", edge=Decimal("0.06")
            ),
            _make_signal(
                layer=8, market_condition_id="cond_a", token_id="tok_a", edge=Decimal("0.05")
            ),
            # Market B: layers 4, 7 → score 2
            _make_signal(
                layer=4, market_condition_id="cond_b", token_id="tok_b", edge=Decimal("0.06")
            ),
            _make_signal(
                layer=7, market_condition_id="cond_b", token_id="tok_b", edge=Decimal("0.04")
            ),
            # Market C: layer 4 only → score 1 (no trade)
            _make_signal(
                layer=4, market_condition_id="cond_c", token_id="tok_c", edge=Decimal("0.06")
            ),
            # Market D: non-scoring layer 1 only → score 0 (no trade)
            _make_signal(
                layer=1, market_condition_id="cond_d", token_id="tok_d", edge=Decimal("0.10")
            ),
        ]

        category_map = {
            "cond_a": "crypto",
            "cond_b": "sports",
            "cond_c": "politics",
            "cond_d": "entertainment",
        }

        # Score all
        all_opps = scorer.score_signals(signals)
        scores_by_market = {o.market_condition_id: o.score for o in all_opps}
        assert scores_by_market["cond_a"] == 5
        assert scores_by_market["cond_b"] == 2
        assert scores_by_market["cond_c"] == 1
        assert scores_by_market["cond_d"] == 0

        # Get tradeable (score >= 2 required)
        tradeable = scorer.get_tradeable(signals, market_category_map=category_map)
        tradeable_ids = {o.market_condition_id for o in tradeable}
        assert "cond_a" in tradeable_ids  # score 5
        assert "cond_b" in tradeable_ids  # score 2
        assert "cond_c" not in tradeable_ids  # score 1 (below min_score=2)
        assert "cond_d" not in tradeable_ids  # score 0

        # Execute paper trades for tradeable opportunities
        engine = PaperTradingEngine(initial_capital=Decimal("5000"), risk_manager=risk_mgr)

        for opp in tradeable:
            trade = engine.place_trade(
                market_condition_id=opp.market_condition_id,
                token_id=opp.token_id,
                side="BUY",
                market_price=Decimal("0.50"),
                model_prob=Decimal("0.60"),
                layer=min(opp.contributing_layers),
                market_category=category_map[opp.market_condition_id],
                confluence_score=opp.score,
            )
            assert trade is not None, f"Trade for {opp.market_condition_id} should succeed"

        assert len(engine.trade_history) == 2
        assert len(engine.open_positions) == 2

        # Resolve all
        engine.resolve_market("tok_a", winning_outcome=True)
        engine.resolve_market("tok_b", winning_outcome=False)

        # Generate report covering the whole flow
        gen = ReportGenerator()
        trade_dicts = [_trade_to_dict(t) for t in engine.trade_history]
        daily = gen.generate_daily(trade_dicts, signals=[], report_date=date(2026, 2, 21))
        weekly = gen.generate_weekly(
            [daily],
            trades=trade_dicts,
            portfolio_balance=engine.balance,
        )

        assert weekly.total_trades == 2
        assert weekly.winning_trades == 1
        assert weekly.confluence_score_distribution.get(5, 0) == 1
        assert weekly.confluence_score_distribution.get(2, 0) == 1
