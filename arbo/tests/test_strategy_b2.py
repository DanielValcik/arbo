"""Tests for Strategy B2 — Crypto Price Edge.

Focused on the paper-live mirror invariant (PAPER_MATCH_LIVE). Previous
paper BUY used min(bid, ask) — the "low side" that effectively earned the
spread. Live must pay best_ask (high side); the gap systematically
inflates paper PnL. This test pins the flag to prevent regression.
"""
from __future__ import annotations

from arbo.strategies import crypto_quality_gate


def test_paper_match_live_enabled_by_default() -> None:
    """Guards against regression to the spread-earning paper model.

    Whoever flips this back to False is inflating paper PnL vs live.
    If you need the old behavior for backtest/research, create a named
    variant — do not change the module default.
    """
    assert crypto_quality_gate.PAPER_MATCH_LIVE is True


def test_paper_match_live_constant_exists() -> None:
    """Contract: the flag must exist and be a bool."""
    assert hasattr(crypto_quality_gate, "PAPER_MATCH_LIVE")
    assert isinstance(crypto_quality_gate.PAPER_MATCH_LIVE, bool)


def test_mirror_cancel_debounce_constant_exists() -> None:
    """Debounce prevents cascade re-entries after live fill failures."""
    assert hasattr(crypto_quality_gate, "MIRROR_CANCEL_DEBOUNCE")
    assert crypto_quality_gate.MIRROR_CANCEL_DEBOUNCE >= 60


def test_paper_and_live_slot_pools_are_separated() -> None:
    """Paper positions must NOT count against the live slot budget.

    Regression: before this split, 10 legacy paper positions from pre-dual
    deploy blocked any new dual-mode entry because both pools shared the
    MAX_POSITIONS_PER_STRATEGY=10 cap. Paper → data collection (no capital
    at risk) must not starve live (real capital).
    """
    from decimal import Decimal

    from arbo.core.risk_manager import (
        MAX_LIVE_POSITIONS_PER_STRATEGY,
        RiskManager,
        TradeRequest,
    )

    RiskManager.reset()
    risk = RiskManager(capital=Decimal("5000"))

    # Fill up paper pool with 10 historic paper positions (as B2 does)
    for _ in range(10):
        risk.strategy_post_trade("B2", Decimal("40"), is_live_capital=False)

    ss = risk.get_strategy_state("B2")
    assert ss is not None
    assert ss.position_count == 10
    assert ss.live_position_count == 0

    # A live request must still be approved (live pool is empty)
    live_req = TradeRequest(
        market_id="m1", token_id="t1", side="BUY",
        price=Decimal("0.50"), size=Decimal("5"), layer=0,
        market_category="crypto", strategy="B2",
        is_live_capital=True,
    )
    decision = risk.pre_trade_check(live_req)
    assert decision.approved, (
        f"Live request should pass despite 10 paper positions, got: {decision.reason}"
    )

    # Fill live pool too; after MAX_LIVE it should reject
    for _ in range(MAX_LIVE_POSITIONS_PER_STRATEGY):
        risk.strategy_post_trade("B2", Decimal("5"), is_live_capital=True)

    decision2 = risk.pre_trade_check(live_req)
    assert not decision2.approved
    assert "LIVE" in decision2.reason


def test_b2_accepts_dual_mode_config() -> None:
    """B2 constructor must accept live_capital and daily_loss_limit for dual mode.

    Without these, main_rdh can't wire up the dual execution path for B2.
    """
    from unittest.mock import MagicMock

    from arbo.strategies.strategy_b2 import StrategyB2

    s = StrategyB2(
        risk_manager=MagicMock(),
        execution_mode="dual",
        live_executor=MagicMock(),
        live_capital=100.0,
        live_capital_fallback=100.0,
        live_entry_timeout_s=30,
        live_daily_loss_limit=20.0,
    )
    assert s._execution_mode == "dual"
    assert s._live_capital == 100.0
    assert s._live_daily_loss_limit == 20.0
    # Debounce dict should exist for mirror-cancel debounce
    assert hasattr(s, "_last_mirror_attempt")
    assert isinstance(s._last_mirror_attempt, dict)
