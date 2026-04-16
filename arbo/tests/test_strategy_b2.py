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
