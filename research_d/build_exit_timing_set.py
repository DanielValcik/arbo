"""Strategy D — Exit-timing training-set builder.

For each simulated trade, expand its post-entry trajectory into multiple
per-timestep rows with (features at time t, survival label).

Survival label per timestep t:
  event  = 1 if green-book target is crossed at some t' > t, else 0
  time_to_event = (t' - t) if event=1, else (exit_idx - t) [censored]

Features per timestep are computed by exit_timing_features.compute_features_at
which guarantees no lookahead (see research_d/test_exit_timing_features.py).

Usage:
  PYTHONPATH=. python3 research_d/build_exit_timing_set.py
  PYTHONPATH=. python3 research_d/build_exit_timing_set.py --db /mnt/arbo-data/sports_backtest.sqlite --limit 500
  PYTHONPATH=. python3 research_d/build_exit_timing_set.py --params-preset v4_winner --output /tmp/ets.parquet

Output:
  research_d/data/exit_timing_set_v1.parquet  (per-timestep rows)
  research_d/data/exit_timing_set_v1.meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.prepare import PARAMS, DEFAULT_DB_PATH, evaluate
from research_d.exit_timing_features import (
    compute_features_at,
    find_first_gb_hit,
    FEATURE_COLUMNS,
)

OUTPUT_DIR = Path(__file__).parent / "data"


# ── Parameter presets ─────────────────────────────────────────────────


_PRESETS = {
    "v4_winner": {
        # Champion params — most realistic distribution of trades
        **PARAMS,
        "min_edge": 0.16,
        "green_book_delta_nba": 0.15,
        "stop_loss_delta": 0.15,
        "max_hold_fraction": 0.50,
        "stop_loss_enabled": True,
        "both_sides": True,
    },
    "wide_edge": {
        # Broader trade pool → more training rows (sacrifices realism)
        **PARAMS,
        "min_edge": 0.03,
        "max_edge": 0.40,
        "min_price": 0.10,
        "max_price": 0.85,
        "both_sides": True,
    },
}


# ── Expansion: one trade → N timestep rows ────────────────────────────


def expand_trade(trade: dict, min_ticks_before_end: int = 1) -> list[dict]:
    """Expand a single trade into per-timestep rows.

    Args:
        trade: dict from evaluate(return_trades=True, capture_trajectory=True)
        min_ticks_before_end: skip the last N ticks (degenerate time_to_event=0)

    Returns:
        List of per-timestep dicts with features + survival label + identity.
    """
    trajectory = trade.get("trajectory")
    if trajectory is None or len(trajectory) < 2:
        return []

    entry_price = trade["entry_price"]
    target = trade["target"]
    stop_loss_price = trade["stop_loss_price"]
    side = trade["side"]
    model_prob = trade["model_prob"]
    edge_entry = trade["edge"]
    exit_idx = trade["exit_idx"]
    # Expected total length = what the rule WOULD have considered (includes
    # ticks past exit_idx if backtest held longer). For the ML label we only
    # have ticks up to exit_idx in trajectory.
    total_len_expected = trade["n_prices"]

    # Identity pieces (shared by all timesteps of this trade)
    trade_id = f"{trade['token_id']}__{side}"
    game_date = trade["game_date"]
    sport = trade["sport"]

    # First GB hit (inside trajectory — trajectory is truncated at exit)
    # For green_booked trades, this should equal exit_idx.
    # For non-GB trades, expected None (GB never fired in captured window).
    gb_hit_idx = find_first_gb_hit(trajectory, target, side)

    # Sanity: if the backtest said green_booked=True, gb_hit_idx must be set
    # (and typically == exit_idx, since backtest breaks on first GB hit).
    if trade.get("green_booked") and gb_hit_idx is None:
        # Data inconsistency — emit warning later but skip this trade
        return []

    rows: list[dict] = []
    last_usable_t = len(trajectory) - 1 - min_ticks_before_end
    # If GB hits inside the trajectory, emit rows up to and including that
    # tick (so eval simulator can observe the crossing). Training loop
    # filters `for_training=1` only.
    max_t = min(last_usable_t, gb_hit_idx) if gb_hit_idx is not None else last_usable_t
    max_t = max(max_t, 0)
    for t in range(0, max_t + 1):
        # Survival label + training eligibility
        if gb_hit_idx is not None and gb_hit_idx > t:
            event = 1
            time_to_event = float(gb_hit_idx - t)
            for_training = 1
        elif gb_hit_idx is not None and gb_hit_idx == t:
            # Event fires AT this step — emit for eval use; mark as
            # NON-training (degenerate horizon). Use time_to_event=0.5 so
            # the column stays positive for schema consistency.
            event = 1
            time_to_event = 0.5
            for_training = 0
        else:
            # Censored (GB never fires in captured window).
            event = 0
            time_to_event = float(len(trajectory) - 1 - t)
            for_training = 1 if time_to_event > 0 else 0

        if time_to_event <= 0:
            continue

        features = compute_features_at(
            trajectory=trajectory,
            t=t,
            entry_price=entry_price,
            target=target,
            stop_loss_price=stop_loss_price,
            side=side,
            model_prob=model_prob,
            edge_at_entry=edge_entry,
            total_len_expected=total_len_expected,
        )

        row = {
            # Identity
            "trade_id": trade_id,
            "token_id": trade["token_id"],
            "game_date": game_date,
            "sport": sport,
            "t": t,
            "ts": trajectory[t][0],
            "price_now": trajectory[t][1],
            "side": side,
            # Survival label
            "event": event,
            "time_to_event": time_to_event,
            "for_training": for_training,  # 0 for degenerate rows, 1 for training
            # Features
            **features,
            # Metadata (for analytics, not used as model input)
            "exit_reason_of_trade": trade.get("exit_reason", "unknown"),
            "trade_green_booked": int(trade.get("green_booked", False)),
            "trade_pnl_usd": float(trade.get("pnl", 0.0)),
        }
        rows.append(row)

    return rows


# ── Build training set ────────────────────────────────────────────────


def build(
    db_path: Path,
    params: dict,
    limit: int = 0,
    output_path: Path | None = None,
    min_ticks_before_end: int = 1,
) -> dict:
    """Run backtest with trajectories captured, expand to per-timestep rows,
    save as parquet.
    """
    import pandas as pd

    t0 = time.time()

    print(f"Running backtest (capture_trajectory=True, limit={limit})...",
          flush=True)
    results = evaluate(
        params, db_path, limit=limit,
        return_trades=True, capture_trajectory=True,
    )
    trades = results.get("trades", [])
    print(f"  Backtest finished: {len(trades)} trades in "
          f"{time.time() - t0:.1f}s", flush=True)

    if not trades:
        print("ERROR: no trades generated. Adjust params or --limit.")
        return {"n_trades": 0, "n_rows": 0, "error": "no trades"}

    # Trajectory length distribution
    traj_lens = [
        len(t["trajectory"]) if t.get("trajectory") else 0
        for t in trades
    ]
    if traj_lens:
        print(f"\nTrajectory length stats:")
        print(f"  min={min(traj_lens)}  max={max(traj_lens)}  "
              f"mean={sum(traj_lens) / len(traj_lens):.1f}  "
              f"median={sorted(traj_lens)[len(traj_lens) // 2]}")

    # Exit-reason breakdown
    reason_counts: dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "?")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    print(f"\nExit-reason distribution:")
    for r, n in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {r:>16}: {n:>6} ({100 * n / len(trades):.1f}%)")

    # Expand
    print(f"\nExpanding {len(trades)} trades to per-timestep rows...",
          flush=True)
    all_rows: list[dict] = []
    skipped = 0
    for trade in trades:
        rows = expand_trade(trade, min_ticks_before_end=min_ticks_before_end)
        if not rows:
            skipped += 1
        all_rows.extend(rows)

    if skipped:
        print(f"  Skipped {skipped} trades (too short or data inconsistency)")
    print(f"  Total rows: {len(all_rows)}", flush=True)

    if not all_rows:
        print("ERROR: expansion produced 0 rows. Check trade trajectories.")
        return {"n_trades": len(trades), "n_rows": 0, "error": "empty expansion"}

    df = pd.DataFrame(all_rows)

    # Sanity: required columns present
    required = {"trade_id", "event", "time_to_event"} | set(FEATURE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Label stats
    event_rate = float(df["event"].mean())
    mean_tte = float(df["time_to_event"].mean())
    median_tte = float(df["time_to_event"].median())

    print(f"\nLabel stats:")
    print(f"  Event rate (GB in future): {event_rate:.4f} ({event_rate:.2%})")
    print(f"  Mean time_to_event (ticks): {mean_tte:.2f}")
    print(f"  Median time_to_event: {median_tte:.2f}")

    # Per-trade-outcome event rate breakdown
    print(f"\nRows by original trade exit reason:")
    by_reason = df.groupby("exit_reason_of_trade").agg(
        n_rows=("event", "size"),
        event_rate=("event", "mean"),
    )
    print(by_reason.to_string())

    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "exit_timing_set_v1.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved: {output_path}  shape={df.shape}")

    # Metadata
    param_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "params": params,
        "param_hash": param_hash,
        "n_trades": len(trades),
        "n_rows": len(df),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "event_rate": event_rate,
        "mean_time_to_event": mean_tte,
        "median_time_to_event": median_tte,
        "trajectory_length_mean": sum(traj_lens) / max(len(traj_lens), 1),
        "exit_reason_counts": reason_counts,
        "min_ticks_before_end": min_ticks_before_end,
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"✓ Meta:  {meta_path}")

    return {
        "n_trades": len(trades),
        "n_rows": len(df),
        "event_rate": event_rate,
        "output_path": str(output_path),
        "meta_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--params-preset", choices=list(_PRESETS.keys()),
                        default="v4_winner")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit candidate markets scanned (0 = all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet path")
    parser.add_argument("--min-ticks-before-end", type=int, default=1,
                        help="Skip last N ticks per trajectory (avoid degenerate horizon)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}")
        sys.exit(1)

    params = _PRESETS[args.params_preset]
    print(f"Preset: {args.params_preset}  DB: {args.db}")

    build(args.db, params, args.limit, args.output, args.min_ticks_before_end)


if __name__ == "__main__":
    main()
