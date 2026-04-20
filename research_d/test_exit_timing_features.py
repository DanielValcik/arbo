"""Unit tests for exit_timing_features — enforce no-lookahead + math correctness.

Run:
  PYTHONPATH=. python3 -m pytest research_d/test_exit_timing_features.py -v
Or simply:
  PYTHONPATH=. python3 research_d/test_exit_timing_features.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.exit_timing_features import (
    rolling_std,
    rolling_slope,
    pct_up,
    autocorr_1,
    find_first_gb_hit,
    compute_features_at,
    FEATURE_COLUMNS,
    MONOTONIC_CONSTRAINTS,
    get_monotone_vector,
)


# ── Rolling statistics ────────────────────────────────────────────────


def test_rolling_std_edge_cases():
    assert rolling_std([]) == 0.0
    assert rolling_std([1.0]) == 0.0
    # Known variance: [1, 2, 3] → sample σ = 1.0
    assert abs(rolling_std([1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_rolling_std_constant_is_zero():
    assert rolling_std([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_rolling_slope_edge_cases():
    assert rolling_slope([]) == 0.0
    assert rolling_slope([1.0]) == 0.0


def test_rolling_slope_linear_increase():
    # Slope of [0, 1, 2, 3] vs x=[0,1,2,3] = 1
    assert abs(rolling_slope([0.0, 1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_rolling_slope_flat_is_zero():
    assert rolling_slope([5.0] * 10) == 0.0


def test_rolling_slope_decreasing():
    # Slope of [3, 2, 1, 0] = -1
    assert abs(rolling_slope([3.0, 2.0, 1.0, 0.0]) - (-1.0)) < 1e-9


def test_pct_up_basics():
    assert pct_up([]) == 0.5
    assert pct_up([1.0]) == 0.5
    # [1, 2, 3, 4] — 3 up transitions of 3 transitions → 1.0
    assert pct_up([1.0, 2.0, 3.0, 4.0]) == 1.0
    # [1, 0, 1, 0] — 1 up of 3 transitions → 1/3
    assert abs(pct_up([1.0, 0.0, 1.0, 0.0]) - 1 / 3) < 1e-9


def test_autocorr_edge_cases():
    assert autocorr_1([]) == 0.0
    assert autocorr_1([1.0, 2.0]) == 0.0  # n<3
    # Constant returns → 0 variance → return 0
    assert autocorr_1([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_autocorr_positive():
    # Persistent returns: [+1, +1, +1] should have high positive autocorr (but
    # variance-constant so 0). Use non-constant:
    ac = autocorr_1([1.0, 1.0, 1.0, 2.0, 2.0])
    # Should be high positive (trend continuation)
    assert ac > 0


# ── First passage time ───────────────────────────────────────────────


def test_find_first_gb_hit_yes_side():
    traj = [(0, 0.40), (60, 0.45), (120, 0.55), (180, 0.60)]
    # Target 0.55 on YES → hits at index 2
    assert find_first_gb_hit(traj, 0.55, "yes") == 2
    # Target 0.70 never hit
    assert find_first_gb_hit(traj, 0.70, "yes") is None
    # Target <= entry → hits at index 0
    assert find_first_gb_hit(traj, 0.40, "yes") == 0


def test_find_first_gb_hit_no_side():
    traj = [(0, 0.60), (60, 0.55), (120, 0.45), (180, 0.40)]
    # Target 0.45 on NO → hits at index 2 (price drops below target)
    assert find_first_gb_hit(traj, 0.45, "no") == 2
    # Target 0.30 never hit
    assert find_first_gb_hit(traj, 0.30, "no") is None


# ── Feature extraction — no lookahead guarantee ──────────────────────


def _make_traj(prices: list[float]) -> list[tuple[int, float]]:
    """Helper: prices → trajectory with ts = 60s increments from 0."""
    return [(i * 60, p) for i, p in enumerate(prices)]


def test_features_use_only_past():
    """Identical features when trajectory is truncated at t — proves no lookahead."""
    full = _make_traj([0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.58])
    truncated = full[:4]  # prices up to index 3

    features_full = compute_features_at(
        trajectory=full, t=3,
        entry_price=0.40, target=0.55, stop_loss_price=0.25, side="yes",
        model_prob=0.60, edge_at_entry=0.20,
    )
    features_trunc = compute_features_at(
        trajectory=truncated, t=3,
        entry_price=0.40, target=0.55, stop_loss_price=0.25, side="yes",
        model_prob=0.60, edge_at_entry=0.20,
    )

    # Every computed feature that depends on prices[0..3] must match.
    # Exception: elapsed_frac depends on total_len which differs — skip it.
    for key in features_full:
        if key in ("elapsed_frac", "remaining_frac"):
            continue
        assert features_full[key] == features_trunc[key], (
            f"LOOKAHEAD in {key}: full={features_full[key]} "
            f"vs trunc={features_trunc[key]}"
        )


def test_features_at_entry_t0():
    """At t=0, most features should be 0 / degenerate."""
    traj = _make_traj([0.40, 0.45, 0.50])
    f = compute_features_at(
        trajectory=traj, t=0,
        entry_price=0.40, target=0.55, stop_loss_price=0.25, side="yes",
        model_prob=0.50, edge_at_entry=0.10,
    )
    assert f["price_return_since_entry"] == 0.0
    assert f["max_since_entry"] == 0.40
    assert f["min_since_entry"] == 0.40
    assert f["max_minus_now"] == 0.0
    assert f["drawdown_from_entry"] == 0.0
    assert f["unrealized_edge"] == 0.0
    assert f["peak_profit"] == 0.0
    assert f["vol_5"] == 0.0      # n=1 → std = 0
    assert f["elapsed_ticks"] == 0
    assert abs(f["gb_distance"] - (0.55 - 0.40)) < 1e-9
    assert f["gb_already_touched"] == 0.0
    assert f["entry_price_level"] == 0.40


def test_features_yes_side_profitable_t5():
    """Verify feature values at a specific timestep after price rose."""
    # entry=0.40, then rises linearly to 0.60 at t=5
    traj = _make_traj([0.40, 0.45, 0.50, 0.52, 0.55, 0.60])
    f = compute_features_at(
        trajectory=traj, t=5,
        entry_price=0.40, target=0.55, stop_loss_price=0.25, side="yes",
        model_prob=0.50, edge_at_entry=0.10,
    )
    assert abs(f["price_return_since_entry"] - 0.20) < 1e-9
    assert abs(f["max_since_entry"] - 0.60) < 1e-9
    assert abs(f["min_since_entry"] - 0.40) < 1e-9
    assert abs(f["peak_profit"] - 0.20) < 1e-9
    assert f["unrealized_edge"] > 0  # yes side, price up
    assert f["gb_already_touched"] == 1.0  # 0.60 > 0.55 earlier
    assert abs(f["gb_distance"] - (0.55 - 0.60)) < 1e-9  # negative — past target
    # Slope should be positive (increasing)
    assert f["slope_15"] > 0


def test_features_no_side_profitable():
    """NO side: profit when price DROPS."""
    # entry=0.60, drops linearly to 0.40 at t=5
    traj = _make_traj([0.60, 0.55, 0.52, 0.50, 0.45, 0.40])
    # For NO side: target is entry - delta = 0.60 - 0.15 = 0.45
    # stop_loss is entry + sl_delta = 0.60 + 0.15 = 0.75
    f = compute_features_at(
        trajectory=traj, t=5,
        entry_price=0.60, target=0.45, stop_loss_price=0.75, side="no",
        model_prob=0.40, edge_at_entry=0.10,
    )
    assert abs(f["peak_profit"] - (0.60 - 0.40)) < 1e-9  # entry - min = 0.20
    assert f["unrealized_edge"] > 0  # no side profits when price falls
    assert f["gb_already_touched"] == 1.0  # 0.40 < 0.45 earlier
    # For no side: gb_distance = price - target = 0.40 - 0.45 = -0.05
    assert abs(f["gb_distance"] - (0.40 - 0.45)) < 1e-9


def test_features_no_side_losing():
    """NO side trade where price goes wrong direction (up)."""
    # entry=0.60, rises to 0.70 (losing for NO side)
    traj = _make_traj([0.60, 0.65, 0.70])
    f = compute_features_at(
        trajectory=traj, t=2,
        entry_price=0.60, target=0.45, stop_loss_price=0.75, side="no",
        model_prob=0.40, edge_at_entry=0.10,
    )
    assert f["unrealized_edge"] < 0  # losing
    assert f["gb_already_touched"] == 0.0  # never touched 0.45
    # gb_distance = price - target = 0.70 - 0.45 = 0.25 (far from target)
    assert abs(f["gb_distance"] - 0.25) < 1e-9


def test_vol_is_positive_when_volatile():
    """Volatile prices → vol_15 > 0; flat prices → vol_15 = 0."""
    flat_traj = _make_traj([0.50] * 20)
    f_flat = compute_features_at(
        flat_traj, 10, entry_price=0.50, target=0.60, stop_loss_price=0.40,
        side="yes", model_prob=0.55, edge_at_entry=0.05,
    )
    assert f_flat["vol_5"] == 0.0
    assert f_flat["vol_15"] == 0.0

    vol_prices = [0.50, 0.55, 0.48, 0.52, 0.58, 0.51, 0.49, 0.54, 0.56, 0.52, 0.50]
    vol_traj = _make_traj(vol_prices)
    f_vol = compute_features_at(
        vol_traj, 10, entry_price=0.50, target=0.60, stop_loss_price=0.40,
        side="yes", model_prob=0.55, edge_at_entry=0.05,
    )
    assert f_vol["vol_5"] > 0.0
    assert f_vol["vol_15"] > 0.0


def test_gb_distance_norm_handles_zero_vol():
    """When vol=0, norm should not be INF or NaN."""
    traj = _make_traj([0.50] * 5)  # flat → vol=0
    f = compute_features_at(
        traj, 4, entry_price=0.50, target=0.60, stop_loss_price=0.40,
        side="yes", model_prob=0.55, edge_at_entry=0.05,
    )
    # Vol=0 but gb_distance > 0 → capped value, not NaN/inf
    assert not math.isnan(f["gb_distance_norm"])
    assert not math.isinf(f["gb_distance_norm"])
    assert abs(f["gb_distance_norm"]) <= 100.0


def test_monotone_vector_alignment():
    """Monotonic constraints vector has correct length and values."""
    vec = get_monotone_vector(FEATURE_COLUMNS)
    assert len(vec) == len(FEATURE_COLUMNS)
    # Spot-check a few
    idx_gb = FEATURE_COLUMNS.index("gb_distance")
    assert vec[idx_gb] == 1
    idx_vol = FEATURE_COLUMNS.index("vol_15")
    assert vec[idx_vol] == -1
    idx_elapsed = FEATURE_COLUMNS.index("elapsed_frac")
    assert vec[idx_elapsed] == -1
    # Unlisted feature should be 0
    idx_entry_px = FEATURE_COLUMNS.index("entry_price_level")
    assert vec[idx_entry_px] == 0


def test_feature_columns_no_identity_leakage():
    """FEATURE_COLUMNS must not include identity / label columns."""
    forbidden = {"trade_id", "game_date", "token_id", "sport", "team_a",
                 "team_b", "event", "time_to_event", "side", "t", "ts",
                 "price_now"}
    leaks = [f for f in FEATURE_COLUMNS if f in forbidden]
    assert not leaks, f"Feature leakage — these belong in identity/label: {leaks}"


def test_features_deterministic():
    """Same inputs → same outputs (no hidden state)."""
    traj = _make_traj([0.40, 0.42, 0.48, 0.55, 0.52])
    f1 = compute_features_at(traj, 4, 0.40, 0.55, 0.25, "yes", 0.55, 0.15)
    f2 = compute_features_at(traj, 4, 0.40, 0.55, 0.25, "yes", 0.55, 0.15)
    assert f1 == f2


def test_elapsed_frac_with_total_len():
    """elapsed_frac uses total_len_expected when provided."""
    traj = _make_traj([0.40, 0.42, 0.44, 0.46])
    # t=3 out of trajectory of length 4 → default elapsed_frac = 3/3 = 1.0
    f_default = compute_features_at(
        traj, 3, 0.40, 0.55, 0.25, "yes", 0.55, 0.15,
    )
    assert f_default["elapsed_frac"] == 1.0

    # With total_len_expected=10 → 3/9 = 0.333
    f_with_len = compute_features_at(
        traj, 3, 0.40, 0.55, 0.25, "yes", 0.55, 0.15,
        total_len_expected=10,
    )
    assert abs(f_with_len["elapsed_frac"] - 3 / 9) < 1e-9


# ── Run as script ────────────────────────────────────────────────────


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
