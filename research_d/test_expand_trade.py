"""Integration tests for build_exit_timing_set.expand_trade.

Verifies survival label correctness on synthetic trades.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.build_exit_timing_set import expand_trade


def _mk_trade(
    prices: list[float],
    entry_price: float,
    target: float,
    stop_loss_price: float,
    side: str = "yes",
    exit_idx: int | None = None,
    green_booked: bool | None = None,
    exit_reason: str = "resolution",
    pnl: float = 0.0,
) -> dict:
    """Build a synthetic trade dict matching evaluate() output schema."""
    trajectory = [(i * 60, p) for i, p in enumerate(prices)]
    if exit_idx is None:
        exit_idx = len(prices) - 1
    if green_booked is None:
        # Auto-derive: check if any price past entry crosses target
        gb = False
        for i, p in enumerate(prices):
            if i == 0:
                continue
            if side == "yes" and p >= target:
                gb = True
                break
            if side == "no" and p <= target:
                gb = True
                break
        green_booked = gb
    return {
        "token_id": "test_tok",
        "sport": "nba",
        "game_date": "2025-01-15",
        "team_a": "LAL",
        "team_b": "BOS",
        "model_prob": 0.60,
        "entry_price": entry_price,
        "edge": 0.60 - entry_price,
        "position_usd": 15.0,
        "n_contracts": 30,
        "won": 1,
        "green_booked": green_booked,
        "exit_price": prices[exit_idx],
        "pnl": pnl,
        "max_price": max(prices),
        "min_price": min(prices),
        "n_prices": len(prices),
        "hold_fraction": exit_idx / max(len(prices) - 1, 1),
        "side": side,
        "target": target,
        "stop_loss_price": stop_loss_price,
        "exit_idx": exit_idx,
        "exit_reason": exit_reason,
        "entry_ts": 0,
        "exit_ts": exit_idx * 60,
        "trajectory": trajectory,
    }


def test_expand_gb_trade_yes_side():
    """Trajectory rises to target at t=4. Rows should have event=1 for t<=4."""
    trade = _mk_trade(
        prices=[0.40, 0.42, 0.45, 0.50, 0.55],  # hits 0.55 at idx 4
        entry_price=0.40,
        target=0.55,
        stop_loss_price=0.25,
        side="yes",
        exit_idx=4,
        exit_reason="green_book",
        pnl=15.0,
    )
    rows = expand_trade(trade, min_ticks_before_end=0)

    # Should have rows for t=0..4 (t=4 is terminal GB tick, for_training=0).
    ts = [r["t"] for r in rows]
    assert ts == [0, 1, 2, 3, 4], f"Expected t=[0,1,2,3,4], got {ts}"

    # All should have event=1 (GB at or after this tick)
    for r in rows:
        assert r["event"] == 1, f"t={r['t']}: event should be 1"

    # Training-eligibility flag: 1 for t<4, 0 for terminal t=4
    for r in rows:
        if r["t"] < 4:
            assert r["for_training"] == 1, f"t={r['t']}: should be trainable"
            expected_tte = 4 - r["t"]
            assert r["time_to_event"] == expected_tte
        else:
            assert r["for_training"] == 0, f"t={r['t']}: terminal tick, skip training"
            assert r["time_to_event"] == 0.5  # marker for terminal


def test_expand_censored_trade():
    """Trajectory never hits target — all rows should be censored (event=0)."""
    trade = _mk_trade(
        prices=[0.40, 0.42, 0.41, 0.43, 0.42],  # never reaches 0.55
        entry_price=0.40,
        target=0.55,
        stop_loss_price=0.25,
        side="yes",
        exit_idx=4,
        green_booked=False,
        exit_reason="time_exit",
    )
    rows = expand_trade(trade, min_ticks_before_end=0)

    # With min_ticks_before_end=0, should have t=0..3 (t=4 has time_to_event=0)
    # But t=4 is degenerate (time_to_event = 4-4 = 0) — skipped.
    ts = [r["t"] for r in rows]
    assert ts == [0, 1, 2, 3]

    # All should be censored
    for r in rows:
        assert r["event"] == 0, f"t={r['t']}: should be censored"

    # time_to_event = exit_idx - t for each censored row (trajectory len = 5)
    for r in rows:
        expected_tte = 4 - r["t"]  # trajectory len - 1 - t = 5 - 1 - t = 4 - t
        assert r["time_to_event"] == expected_tte


def test_expand_no_side_gb():
    """NO side trade where price drops to target."""
    trade = _mk_trade(
        prices=[0.60, 0.55, 0.50, 0.45],  # hits 0.45 at idx 3
        entry_price=0.60,
        target=0.45,
        stop_loss_price=0.75,
        side="no",
        exit_idx=3,
        exit_reason="green_book",
    )
    rows = expand_trade(trade, min_ticks_before_end=0)

    # GB fires at t=3. Expand emits t=0..3 (t=3 terminal).
    ts = [r["t"] for r in rows]
    assert ts == [0, 1, 2, 3]

    for r in rows:
        assert r["event"] == 1
        if r["t"] < 3:
            assert r["for_training"] == 1
            assert r["time_to_event"] == 3 - r["t"]
        else:
            assert r["for_training"] == 0
            assert r["time_to_event"] == 0.5


def test_expand_skips_trades_with_inconsistent_gb_flag():
    """If green_booked=True but target never crossed in trajectory, skip."""
    trade = _mk_trade(
        prices=[0.40, 0.42, 0.41, 0.40],
        entry_price=0.40,
        target=0.55,
        stop_loss_price=0.25,
        side="yes",
        green_booked=True,  # LIE — price never touched 0.55
        exit_idx=3,
    )
    rows = expand_trade(trade)
    assert rows == []


def test_min_ticks_before_end_trimming():
    """min_ticks_before_end should trim last N rows."""
    trade = _mk_trade(
        prices=[0.40, 0.41, 0.42, 0.43, 0.44],
        entry_price=0.40,
        target=0.55,  # never hit
        stop_loss_price=0.25,
        side="yes",
        exit_idx=4,
        green_booked=False,
        exit_reason="time_exit",
    )
    # min_ticks_before_end=0: rows t=0..3 (t=4 has tte=0, skipped)
    rows0 = expand_trade(trade, min_ticks_before_end=0)
    # min_ticks_before_end=2: last_usable_t = 4 - 2 = 2 → rows t=0,1,2
    rows2 = expand_trade(trade, min_ticks_before_end=2)

    t0 = [r["t"] for r in rows0]
    t2 = [r["t"] for r in rows2]
    assert t0 == [0, 1, 2, 3]
    assert t2 == [0, 1, 2]


def test_row_schema_keys():
    """Each row must have identity + label + all feature columns."""
    from research_d.exit_timing_features import FEATURE_COLUMNS

    trade = _mk_trade(
        prices=[0.40, 0.45, 0.55],  # hits target at idx 2
        entry_price=0.40,
        target=0.55,
        stop_loss_price=0.25,
        side="yes",
        exit_idx=2,
        exit_reason="green_book",
    )
    rows = expand_trade(trade, min_ticks_before_end=0)
    assert len(rows) >= 1

    for r in rows:
        # Identity
        assert "trade_id" in r
        assert "token_id" in r
        assert "game_date" in r
        assert "sport" in r
        assert "t" in r
        assert "price_now" in r
        assert "side" in r
        # Label
        assert "event" in r
        assert "time_to_event" in r
        # All FEATURE_COLUMNS present
        for f in FEATURE_COLUMNS:
            assert f in r, f"Missing feature column: {f}"


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
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
