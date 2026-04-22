"""Unit tests for per-strategy health check (core/health_check.py).

Covers the pure-function evaluator (no DB) — full DB-backed flow is covered
in the deploy smoke test (health_checks table on VPS).
"""

from __future__ import annotations

import pytest

from arbo.core.health_check import (
    DORMANT_AFTER_DAYS,
    RETIRED_AFTER_DAYS,
    STRATEGY_BASELINES,
    _escalate,
    _evaluate_strategy,
)


def _stats(**overrides) -> dict:
    base = {
        "strategy": "C",
        "status": "active",
        "window_trades": 3,
        "window_wins": 1,
        "window_losses": 2,
        "window_resolved": 3,
        "window_pnl": 0.0,
        "total_trades": 100,
        "total_resolved": 80,
        "total_wins": 30,
        "total_losses": 50,
        "total_wr": 0.375,  # matches AR0134 baseline
        "total_pnl": 50.0,
        "open_positions": 1,
        "days_active": 20,
        "days_since_last_trade": 0.5,
        "daily_trade_rate": 5.0,
        "first_trade_at": None,
        "last_trade_at": None,
    }
    base.update(overrides)
    return base


class TestEscalate:
    def test_takes_worse(self) -> None:
        assert _escalate("ok", "needs_attention") == "needs_attention"
        assert _escalate("needs_attention", "bug_detected") == "bug_detected"
        assert _escalate("bug_detected", "ok") == "bug_detected"
        assert _escalate("ok", "ok") == "ok"


class TestEvaluateStrategy:
    def test_retired_is_silent(self) -> None:
        """Retired strategies produce ok verdict + no notes — operator disabled them."""
        v, notes = _evaluate_strategy(
            _stats(status="retired", days_since_last_trade=25), window_hours=12
        )
        assert v == "ok"
        assert notes == []

    def test_never_traded_is_silent(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(status="never_traded", days_since_last_trade=None), window_hours=12
        )
        assert v == "ok"
        assert notes == []

    def test_dormant_notes_but_ok_verdict(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(status="dormant", days_since_last_trade=5.2), window_hours=12
        )
        assert v == "ok"
        assert len(notes) == 1
        assert "5.2" in notes[0]

    def test_active_healthy_c_no_warnings(self) -> None:
        v, notes = _evaluate_strategy(_stats(), window_hours=12)
        assert v == "ok"
        assert notes == []

    def test_active_zero_window_triggers_bug(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(window_trades=0, window_resolved=0, days_active=10),
            window_hours=12,
        )
        assert v == "bug_detected"
        assert any("zadna aktivita" in n for n in notes)

    def test_wr_drift_on_baseline_strategy(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(total_wr=0.582, total_resolved=80), window_hours=12
        )
        assert v == "needs_attention"
        assert any("win rate" in n.lower() for n in notes)

    def test_wr_drift_ignored_below_min_comparison(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(total_wr=0.9, total_resolved=10), window_hours=12
        )
        # Not enough resolved — WR check skipped, verdict stays ok.
        assert v == "ok"

    def test_no_baseline_strategy_skips_wr_and_rate_checks(self) -> None:
        # B3 has no entry in STRATEGY_BASELINES — only activity check applies.
        assert "B3" not in STRATEGY_BASELINES
        v, notes = _evaluate_strategy(
            _stats(
                strategy="B3",
                total_wr=0.05,  # would trigger on baseline, but no baseline for B3
                total_resolved=500,
                daily_trade_rate=0.1,  # low rate ignored without baseline
            ),
            window_hours=12,
        )
        assert v == "ok"
        assert notes == []

    def test_trade_rate_warning_on_baseline_strategy(self) -> None:
        v, notes = _evaluate_strategy(
            _stats(daily_trade_rate=0.2, days_active=10), window_hours=12
        )
        assert v == "needs_attention"
        assert any("malo obchodu" in n for n in notes)

    def test_pnl_trajectory_flags_big_loss(self) -> None:
        # AR0134 oos_daily_pnl = 26.8 → 10 days expected = 268. Trigger: pnl < -134.
        v, notes = _evaluate_strategy(
            _stats(total_pnl=-500.0, days_active=10), window_hours=12
        )
        assert v == "needs_attention"
        assert any("pod ocekavanim" in n for n in notes)


class TestThresholdsSanity:
    def test_retired_threshold_gte_dormant(self) -> None:
        assert RETIRED_AFTER_DAYS >= DORMANT_AFTER_DAYS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
