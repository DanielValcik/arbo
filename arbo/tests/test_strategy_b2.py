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
