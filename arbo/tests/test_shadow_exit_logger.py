"""Integration tests for shadow-exit telemetry logger.

Critical invariants:
  1. Logger NEVER changes exit behavior — champion rule still fires
     identically to the EXIT_POLICY=None baseline.
  2. Logger calls _log_shadow_exit_decision for:
     a) First tick where ML says should_exit (event_type='ml_first_exit')
     b) Every real exit (event_type='real_exit')
  3. When SHADOW_EXIT_LOG_ENABLED=False, NO logger calls happen.
  4. When model fails to load, system survives silently (no crash).

These tests mock the DB insert path (_shadow_insert_async) and verify
call counts + parameter shapes. Real DB integration is tested post-deploy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────


def _make_core(shadow_enabled: bool, model_path: str | None = None,
               model_path_exists: bool = True):
    """Build a test core with or without shadow logging enabled."""
    from arbo.strategies.strategy_d_core import StrategyDCore

    default_path = (
        str(REPO_ROOT / "arbo/data/models/strategy_d_exit_v1.ubj")
        if model_path_exists else "/nonexistent/path.ubj"
    )

    class _TestCore(StrategyDCore):
        SPORT_NAME = "test_nba"
        STRATEGY_NAME = "D_TEST"
        SHADOW_EXIT_LOG_ENABLED = shadow_enabled
        SHADOW_EXIT_MODEL_PATH = model_path or default_path
        SHADOW_EXIT_THRESHOLD = 6658.3

    core = _TestCore(
        risk_manager=MagicMock(),
        paper_engine=None,
        live_executor=None,
        orderbook_provider=None,
        elo_ratings={},
        pinnacle_odds={},
    )
    return core


def _make_pos(core, side: str = "yes", entry_price: float = 0.45,
              entry_time: float | None = None):
    from arbo.strategies.strategy_d_core import DPosition

    pos = DPosition(
        sport="test_nba",
        condition_id="test_cond_" + side,
        token_id="test_tok_" + side + "_" + str(time.time_ns()),
        side=side,
        entry_price=entry_price,
        entry_time=entry_time or (time.time() - 600),
        model_prob=0.60,
        edge=0.60 - entry_price,
        shares=30,
        cost=entry_price * 30,
        question="Test",
        team_a="LAL",
        team_b="BOS",
        game_date="2026-04-20",
    )
    core._positions[pos.condition_id] = pos
    return pos


# ── Tests ────────────────────────────────────────────────────────────


def test_shadow_disabled_logs_nothing():
    """When SHADOW_EXIT_LOG_ENABLED=False, logger NEVER called."""
    core = _make_core(shadow_enabled=False)
    pos = _make_pos(core, side="yes", entry_price=0.45)

    with patch.object(core, "_log_shadow_exit_decision") as mock_log:
        for price in [0.48, 0.52, 0.56, 0.60]:
            core.check_exits({pos.token_id: price})
        assert mock_log.call_count == 0, \
            f"Logger called {mock_log.call_count} times with shadow disabled"
    print(f"  shadow_disabled=True → 0 logger calls: OK")


def test_shadow_disabled_exit_behavior_unchanged():
    """With shadow disabled, exit fires at GB target like always."""
    core = _make_core(shadow_enabled=False)
    pos = _make_pos(core, side="yes", entry_price=0.45)
    # Ramp up to trigger GB (entry + 0.17 = 0.62)
    exits = []
    for price in [0.50, 0.55, 0.60, 0.62, 0.63]:
        exits.extend(core.check_exits({pos.token_id: price}))
    assert len(exits) == 1
    assert exits[0].exit_reason == "green_book"
    print(f"  disabled + champion rule: exit='{exits[0].exit_reason}' OK")


def test_shadow_enabled_logs_real_exit():
    """When real exit fires, 'real_exit' event is logged."""
    core = _make_core(shadow_enabled=True)
    pos = _make_pos(core, side="yes", entry_price=0.45)

    with patch.object(core, "_log_shadow_exit_decision") as mock_log:
        # Force a clean real exit via time_exit (set entry_time far past)
        pos.entry_time = time.time() - (2.5 * 3600 * 1.0)  # beyond MAX_HOLD×GAME_DUR
        core.check_exits({pos.token_id: 0.48})

        # Find 'real_exit' calls
        real_exit_calls = [c for c in mock_log.call_args_list
                          if c.kwargs.get("event_type") == "real_exit"]
        assert len(real_exit_calls) == 1, \
            f"Expected 1 real_exit log, got {len(real_exit_calls)}"
        assert real_exit_calls[0].kwargs["real_exit_reason"] == "time_exit"
        print(f"  real_exit logged with reason=time_exit: OK")


def test_shadow_enabled_logs_ml_first_exit_only_once():
    """'ml_first_exit' logs at most once per position (dedup)."""
    core = _make_core(shadow_enabled=True)
    pos = _make_pos(core, side="yes", entry_price=0.45)

    # Mock ML policy to always say should_exit=True
    fake_decision = MagicMock()
    fake_decision.should_exit = True
    fake_decision.pred_log_t = 9999.9
    fake_decision.reason = "ml_hazard_v1"
    fake_policy = MagicMock()
    fake_policy.decide = MagicMock(return_value=fake_decision)

    with patch.object(core, "_get_shadow_exit_policy", return_value=fake_policy):
        with patch.object(core, "_log_shadow_exit_decision") as mock_log:
            # 5 ticks where ML says "exit", none of which triggers real GB/SL
            for price in [0.50, 0.51, 0.52, 0.53, 0.54]:
                core.check_exits({pos.token_id: price})

            ml_first_calls = [c for c in mock_log.call_args_list
                              if c.kwargs.get("event_type") == "ml_first_exit"]
            assert len(ml_first_calls) == 1, \
                f"Expected exactly 1 ml_first_exit, got {len(ml_first_calls)}"
    print(f"  ml_first_exit dedup: 1 log for 5 qualifying ticks: OK")


def test_shadow_enabled_does_not_affect_exit_behavior():
    """Even when shadow says exit, real exit is unchanged (rule still rules)."""
    core = _make_core(shadow_enabled=True)
    pos = _make_pos(core, side="yes", entry_price=0.45)

    # Mock shadow policy to say should_exit=True immediately
    fake_decision = MagicMock()
    fake_decision.should_exit = True
    fake_decision.pred_log_t = 9999.9
    fake_decision.reason = "ml_hazard_v1"
    fake_policy = MagicMock()
    fake_policy.decide = MagicMock(return_value=fake_decision)

    with patch.object(core, "_get_shadow_exit_policy", return_value=fake_policy):
        with patch.object(core, "_log_shadow_exit_decision"):
            # Prices below GB target (0.62), no SL trigger — no real exit
            for price in [0.50, 0.52, 0.54, 0.56]:
                exits = core.check_exits({pos.token_id: price})
                # Position stays open
                assert not exits, f"Shadow saying exit should NOT cause real exit"
            assert pos.condition_id in core._positions
    print(f"  shadow enabled + says exit → real position STAYS OPEN: OK")


def test_shadow_logger_survives_missing_model():
    """When model file doesn't exist, logger cleanly returns None — no crash."""
    core = _make_core(shadow_enabled=True, model_path="/nonexistent/xyz.ubj")
    pos = _make_pos(core, side="yes", entry_price=0.45)

    with patch.object(core, "_log_shadow_exit_decision") as mock_log:
        # This should not crash even though model won't load
        for price in [0.50, 0.55, 0.60]:
            core.check_exits({pos.token_id: price})
    # We might get calls for real_exit via time_exit path but no crash
    print(f"  missing model → no crash: OK")


def test_dedup_cleared_after_real_exit():
    """After real exit fires, dedup entry is removed (so if re-entry happens
    later, ml_first_exit can log again)."""
    core = _make_core(shadow_enabled=True)
    pos = _make_pos(core, side="yes", entry_price=0.45)
    pos.entry_time = time.time() - (2.5 * 3600 * 1.0)  # force time_exit

    core._shadow_ml_first_logged.add((pos.token_id, pos.side))
    assert (pos.token_id, pos.side) in core._shadow_ml_first_logged

    with patch.object(core, "_log_shadow_exit_decision"):
        core.check_exits({pos.token_id: 0.48})

    assert (pos.token_id, pos.side) not in core._shadow_ml_first_logged, \
        "dedup entry should be cleaned after real exit"
    print(f"  dedup cleanup after real exit: OK")


# ── Run ──────────────────────────────────────────────────────────────


def _run_all():
    tests = [
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
