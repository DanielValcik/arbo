"""Integration tests for ML exit policy in strategy_d_core.

Checks:
  1. Import + no-op behavior when EXIT_POLICY=None (champion unchanged).
  2. _get_ml_exit_policy returns None when unset.
  3. price_history gets populated on check_exits ticks (regardless of
     whether ML policy is active).
  4. With EXIT_POLICY="ml_hazard_v1" and a real model file, decide()
     is queryable.

Run:
  PYTHONPATH=. python3 -m pytest arbo/tests/test_strategy_d_exit_integration.py -v
Or:
  PYTHONPATH=. python3 arbo/tests/test_strategy_d_exit_integration.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────


def _make_core(exit_policy_attr: str | None = None,
               exit_model_path: str | None = None,
               exit_threshold: float = 6658.3):
    """Build a minimal StrategyDCore with mocked dependencies.

    Optionally seed class attrs so _p() returns chosen exit policy.
    """
    from arbo.strategies.strategy_d_core import StrategyDCore

    class _TestCore(StrategyDCore):
        SPORT_NAME = "test_nba"
        STRATEGY_NAME = "D_TEST"
        EXIT_POLICY = exit_policy_attr
        EXIT_MODEL_PATH = exit_model_path
        EXIT_ML_THRESHOLD = exit_threshold

    return _TestCore(
        risk_manager=MagicMock(),
        paper_engine=None,
        live_executor=None,
        orderbook_provider=None,
        elo_ratings={},
        pinnacle_odds={},
    )


def _make_pos(
    core,
    side: str = "yes",
    entry_price: float = 0.45,
    entry_time: float | None = None,
) -> "DPosition":  # type: ignore[name-defined]
    """Attach a synthetic open position to core._positions and return it."""
    from arbo.strategies.strategy_d_core import DPosition

    now = time.time()
    pos = DPosition(
        sport="test_nba",
        condition_id="test_condition",
        token_id="test_token_xxx",
        side=side,
        entry_price=entry_price,
        entry_time=entry_time or (now - 600),  # 10 min ago
        model_prob=0.60,
        edge=0.60 - entry_price,
        shares=30,
        cost=entry_price * 30,
        question="Test game",
        team_a="LAL",
        team_b="BOS",
        game_date="2026-04-20",
    )
    core._positions[pos.condition_id] = pos
    return pos


# ── Tests ────────────────────────────────────────────────────────────


def test_import_succeeds():
    """Core + model module import without error."""
    from arbo.strategies.strategy_d_core import StrategyDCore, DPosition
    from arbo.models.strategy_d_exit import ExitPolicyModel, compute_exit_features
    # DPosition must have new fields
    fields = {f.name for f in DPosition.__dataclass_fields__.values()}
    assert "price_history" in fields
    # Core must have new class attrs
    assert hasattr(StrategyDCore, "EXIT_POLICY")
    assert hasattr(StrategyDCore, "EXIT_MODEL_PATH")
    assert hasattr(StrategyDCore, "EXIT_ML_THRESHOLD")
    assert StrategyDCore.EXIT_POLICY is None  # Default: no ML
    print("  imports + class attrs OK")


def test_get_ml_policy_returns_none_when_unset():
    """With EXIT_POLICY=None, _get_ml_exit_policy must return None."""
    core = _make_core(exit_policy_attr=None)
    assert core._get_ml_exit_policy() is None, \
        "_get_ml_exit_policy should return None when EXIT_POLICY unset"
    print("  _get_ml_exit_policy returns None when unset: OK")


def test_get_ml_policy_unknown_name_warns_and_returns_none():
    """Unknown EXIT_POLICY string should warn + return None (no crash)."""
    core = _make_core(exit_policy_attr="unknown_policy_v9000")
    assert core._get_ml_exit_policy() is None
    print("  _get_ml_exit_policy None on unknown name: OK")


def test_get_ml_policy_missing_path_returns_none():
    """EXIT_POLICY=ml_hazard_v1 but no EXIT_MODEL_PATH → None (no crash)."""
    core = _make_core(exit_policy_attr="ml_hazard_v1", exit_model_path=None)
    assert core._get_ml_exit_policy() is None
    print("  _get_ml_exit_policy None on missing path: OK")


def test_check_exits_populates_price_history():
    """check_exits must append (ts, price) to pos.price_history each tick."""
    core = _make_core(exit_policy_attr=None)
    pos = _make_pos(core, side="yes", entry_price=0.45)
    # Check 1: initial history is empty
    assert pos.price_history == []
    # Simulate 3 ticks
    core.check_exits({pos.token_id: 0.46})
    core.check_exits({pos.token_id: 0.47})
    core.check_exits({pos.token_id: 0.48})
    # Position still open (none of GB/SL/time_exit triggered)
    assert pos.condition_id in core._positions
    assert len(pos.price_history) == 3
    assert pos.price_history[0][1] == 0.46
    assert pos.price_history[-1][1] == 0.48
    print(f"  price_history populated: {len(pos.price_history)} ticks")


def test_check_exits_bounds_price_history():
    """price_history must be capped at PRICE_HISTORY_MAXLEN."""
    core = _make_core(exit_policy_attr=None)
    pos = _make_pos(core, side="yes", entry_price=0.45)
    # Inject MAXLEN+10 entries manually
    from arbo.strategies.strategy_d_core import StrategyDCore
    maxlen = StrategyDCore.PRICE_HISTORY_MAXLEN
    pos.price_history = [(float(i), 0.45) for i in range(maxlen + 10)]
    core.check_exits({pos.token_id: 0.46})
    assert len(pos.price_history) == maxlen, \
        f"Expected {maxlen} bounded, got {len(pos.price_history)}"
    print(f"  price_history bounded at {maxlen}: OK")


def test_check_exits_no_behavior_change_when_policy_none():
    """With EXIT_POLICY=None, learned_early must NEVER fire."""
    core = _make_core(exit_policy_attr=None)
    pos = _make_pos(core, side="yes", entry_price=0.45)
    # Price stays below GB target (0.45 + 0.17 = 0.62) and above SL (0.30)
    # for many ticks → no exit should fire at all
    for i in range(20):
        exits = core.check_exits({pos.token_id: 0.50})
    # No exit reason should be "learned_early"
    if pos.exit_reason:
        assert pos.exit_reason != "learned_early"
    print("  no learned_early when EXIT_POLICY=None: OK")


def test_ml_policy_loads_and_decides_with_real_model():
    """If model file exists, ExitPolicyModel loads and decides() returns
    structured output (should_exit may be True or False, but no crash)."""
    from arbo.models.strategy_d_exit import ExitPolicyModel

    model_path = REPO_ROOT / "arbo/data/models/strategy_d_exit_v1.ubj"
    if not model_path.exists():
        print(f"  SKIPPED: model file not found at {model_path}")
        return

    policy = ExitPolicyModel(model_path=model_path, threshold_log_t=6658.3)
    # Build synthetic trajectory of 10 ticks rising from 0.45 → 0.54
    now = time.time()
    trajectory = [(now - (10 - i) * 60, 0.45 + 0.01 * i) for i in range(10)]
    decision = policy.decide(
        trajectory=trajectory,
        entry_price=0.45,
        target=0.45 + 0.17,
        stop_loss_price=0.45 - 0.15,
        side="yes",
        model_prob=0.60,
        edge_at_entry=0.15,
    )
    assert hasattr(decision, "should_exit")
    assert isinstance(decision.should_exit, bool)
    assert hasattr(decision, "pred_log_t")
    assert isinstance(decision.pred_log_t, float)
    print(f"  real model decides: should_exit={decision.should_exit}, "
          f"pred_log_t={decision.pred_log_t:.1f}, reason={decision.reason}")


def test_ml_policy_rejects_unprofitable_position():
    """When position is underwater (unrealized < 0), ExitPolicyModel must
    short-circuit to NOT exit (avoid locking in losses)."""
    from arbo.models.strategy_d_exit import ExitPolicyModel

    model_path = REPO_ROOT / "arbo/data/models/strategy_d_exit_v1.ubj"
    policy = ExitPolicyModel(
        model_path=model_path if model_path.exists() else "/nonexistent.ubj",
        threshold_log_t=6658.3,
    )
    now = time.time()
    # Yes-side position underwater: entry 0.60, current 0.50
    trajectory = [(now - (10 - i) * 60, 0.60 - 0.01 * i) for i in range(10)]
    decision = policy.decide(
        trajectory=trajectory,
        entry_price=0.60,
        target=0.77,
        stop_loss_price=0.45,
        side="yes",
        model_prob=0.65,
        edge_at_entry=0.05,
    )
    assert not decision.should_exit
    assert decision.reason == "not_profitable"
    print(f"  unprofitable short-circuit: OK (reason={decision.reason})")


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
