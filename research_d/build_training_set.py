"""Strategy D — Training-set builder for ML meta-labeler.

Generates a labeled (X, y) dataset from the existing backtest harness.
Each row = one simulated trade with features known at entry + outcome label.

Labels (triple-barrier inspired):
  y_profitable = 1 if pnl > 0 else 0   (binary — did this trade profit?)
  y_gb_hit     = 1 if green_booked else 0   (binary — did green book trigger?)
  y_return     = pnl / position_usd   (regression — return as % of position)

Features at entry (no leakage — nothing post-entry):
  model_prob       — Elo+Pinnacle ensemble P(yes wins)
  edge             — model_prob - entry_price (signed)
  edge_magnitude   — |edge|
  entry_price      — Polymarket YES price at entry
  price_dist_50    — |entry_price - 0.5|  (favorite-longshot proxy)
  prob_confidence  — |model_prob - 0.5|
  logit_price      — log(p/(1-p))
  logit_model      — log(m/(1-m))
  hour_of_day      — from game_date (0-23)
  day_of_week      — 0=Mon..6=Sun
  month            — 1-12
  season_phase     — early/mid/late NBA regular+playoffs
  n_prices_total   — liquidity proxy (how many minute-bars we have)
  elo_prob         — Elo-only probability (separate from ensemble)
  pinnacle_prob    — Pinnacle-only probability (if available)
  ensemble_disagreement — |elo_prob - pinnacle_prob| (uncertainty proxy)
  elo_available    — 1 if Elo rating exists for both teams
  pinnacle_available — 1 if Pinnacle line exists for game

Usage:
  PYTHONPATH=. python3 research_d/build_training_set.py
  PYTHONPATH=. python3 research_d/build_training_set.py --params-preset v4_winner
  PYTHONPATH=. python3 research_d/build_training_set.py --limit 100 --output /tmp/ts.parquet

Output:
  research_d/data/training_set_v1.parquet  (default)
  research_d/data/training_set_v1_meta.json  (metadata: params, db, counts, hash)

Note on scope: runs against local sports_backtest.sqlite (small sample).
For production training set (~958M prices), run on arbo-download VPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.prepare import (
    PARAMS,
    DEFAULT_DB_PATH,
    evaluate,
    load_pinnacle,
    load_ratings,
    compute_model_prob,
    _find_pinnacle_game_id,
    _get_rating,
)

OUTPUT_DIR = Path(__file__).parent / "data"


def _season_phase(game_date: str) -> str:
    """Map YYYY-MM-DD to NBA season phase."""
    try:
        month = int(game_date[5:7])
    except (ValueError, IndexError):
        return "unknown"
    # NBA seasons: Oct-Dec = early, Jan-Feb = mid, Mar-Apr = late, May-Jun = playoffs
    if month in (10, 11, 12):
        return "early_regular"
    if month in (1, 2):
        return "mid_regular"
    if month in (3, 4):
        return "late_regular"
    if month in (5, 6):
        return "playoffs"
    return "offseason"


_SEASON_PHASES = ["early_regular", "mid_regular", "late_regular", "playoffs", "offseason", "unknown"]


def _logit(p: float) -> float:
    """Safe logit transform."""
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _compute_entry_features(
    trade: dict,
    ratings: dict,
    pinnacle: dict,
    params: dict,
) -> dict:
    """Compute features available AT ENTRY (no post-entry leakage).

    Input is a dict from evaluate(return_trades=True)'s per-trade records.
    """
    # Base features from the trade record
    model_prob = float(trade["model_prob"])
    entry_price = float(trade["entry_price"])
    edge = float(trade["edge"])
    game_date = trade["game_date"]
    sport = trade["sport"]
    team_a = trade["team_a"]
    team_b = trade["team_b"]

    # Temporal
    hour = 0
    day_of_week = -1
    month = 0
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        day_of_week = dt.weekday()
        month = dt.month
    except ValueError:
        pass

    # Elo (separated from ensemble blend)
    rating_a = _get_rating(ratings, team_a, game_date)
    rating_b = _get_rating(ratings, team_b, game_date)
    elo_prob = None
    elo_diff = None
    if rating_a and rating_b:
        elo_diff = rating_a.elo - rating_b.elo
        elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    # Pinnacle (separated from ensemble blend)
    pin_gid = _find_pinnacle_game_id(pinnacle, sport, game_date, team_a, team_b)
    pinnacle_prob = None
    if pin_gid:
        hp, ap = pinnacle[pin_gid]
        parts = pin_gid.split("_")
        if len(parts) >= 4 and parts[2] == team_a:
            pinnacle_prob = hp
        elif len(parts) >= 4 and parts[3] == team_a:
            pinnacle_prob = ap
        else:
            pinnacle_prob = hp

    # Ensemble disagreement
    ensemble_disagreement = None
    if elo_prob is not None and pinnacle_prob is not None:
        ensemble_disagreement = abs(elo_prob - pinnacle_prob)

    features = {
        # Trade identity (for reference, not model input)
        "token_id": trade["token_id"],
        "game_date": game_date,
        "sport": sport,
        "team_a": team_a,
        "team_b": team_b,
        # Price/probability features
        "model_prob": model_prob,
        "entry_price": entry_price,
        "edge": edge,
        "edge_magnitude": abs(edge),
        "price_dist_50": abs(entry_price - 0.5),
        "prob_confidence": abs(model_prob - 0.5),
        "logit_price": _logit(entry_price),
        "logit_model": _logit(model_prob),
        # Temporal
        "day_of_week": day_of_week,
        "month": month,
        "season_phase": _season_phase(game_date),
        # Liquidity proxy
        "n_prices_total": int(trade["n_prices"]),
        "position_usd": float(trade["position_usd"]),
        # Decomposed model components
        "elo_prob": elo_prob if elo_prob is not None else model_prob,
        "elo_diff": elo_diff if elo_diff is not None else 0.0,
        "pinnacle_prob": pinnacle_prob if pinnacle_prob is not None else model_prob,
        "ensemble_disagreement": (
            ensemble_disagreement if ensemble_disagreement is not None else 0.0
        ),
        "elo_available": 1 if elo_prob is not None else 0,
        "pinnacle_available": 1 if pinnacle_prob is not None else 0,
        "both_models_available": (
            1 if (elo_prob is not None and pinnacle_prob is not None) else 0
        ),
        # Labels (for training)
        "y_profitable": 1 if trade["pnl"] > 0 else 0,
        "y_gb_hit": 1 if trade["green_booked"] else 0,
        "y_return": (
            trade["pnl"] / trade["position_usd"]
            if trade["position_usd"] > 0 else 0.0
        ),
        "y_pnl_usd": float(trade["pnl"]),
    }
    return features


def build_training_set(
    db_path: Path,
    params: dict,
    limit: int = 0,
    output_path: Path | None = None,
) -> dict:
    """Generate training set as pandas DataFrame; save parquet + metadata.

    Returns dict with summary stats.
    """
    import pandas as pd

    t0 = time.time()

    # Run backtest with return_trades=True
    print(f"Running backtest to extract trades (limit={limit})...", flush=True)
    results = evaluate(params, db_path, limit=limit, return_trades=True)
    trades = results.get("trades", [])
    print(f"  Generated {len(trades)} trades in {time.time() - t0:.1f}s", flush=True)

    if not trades:
        print("ERROR: No trades generated — cannot build training set.", flush=True)
        return {"n_trades": 0, "error": "no trades"}

    # Load lookup tables once for feature enrichment (readonly-safe).
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro&immutable=1", uri=True,
        )
    ratings = load_ratings(conn)
    pinnacle = load_pinnacle(conn)
    conn.close()

    # Enrich each trade with entry-time features
    print(f"\nEnriching {len(trades)} trades with features...", flush=True)
    rows = []
    enrichment_errors = 0
    for i, trade in enumerate(trades):
        try:
            row = _compute_entry_features(trade, ratings, pinnacle, params)
            rows.append(row)
        except Exception as e:
            enrichment_errors += 1
            if enrichment_errors < 5:
                print(f"  WARN: enrichment failed for trade {i}: {e}", flush=True)

    if enrichment_errors:
        print(f"  {enrichment_errors} enrichment errors (see warnings)", flush=True)

    df = pd.DataFrame(rows)

    # Write parquet
    if output_path is None:
        output_path = OUTPUT_DIR / "training_set_v1.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved training set: {output_path}", flush=True)
    print(f"  Shape: {df.shape}", flush=True)

    # Summary stats on labels
    label_summary = {
        "total": len(df),
        "profitable_rate": float(df["y_profitable"].mean()),
        "gb_hit_rate": float(df["y_gb_hit"].mean()),
        "mean_return": float(df["y_return"].mean()),
        "median_return": float(df["y_return"].median()),
        "mean_pnl_usd": float(df["y_pnl_usd"].mean()),
        "total_pnl_usd": float(df["y_pnl_usd"].sum()),
    }

    print(f"\nLabel summary:", flush=True)
    for k, v in label_summary.items():
        if isinstance(v, float):
            print(f"  {k:>20}: {v:.4f}", flush=True)
        else:
            print(f"  {k:>20}: {v}", flush=True)

    # Per-sport breakdown
    print(f"\nPer-sport:", flush=True)
    by_sport = df.groupby("sport").agg(
        n=("y_profitable", "size"),
        profitable_rate=("y_profitable", "mean"),
        gb_hit_rate=("y_gb_hit", "mean"),
        mean_return=("y_return", "mean"),
        total_pnl=("y_pnl_usd", "sum"),
    )
    print(by_sport.to_string(), flush=True)

    # Feature stats
    feat_cols = [c for c in df.columns if not c.startswith("y_") and c not in (
        "token_id", "game_date", "team_a", "team_b", "sport", "season_phase"
    )]
    print(f"\nNumeric feature summary:", flush=True)
    print(df[feat_cols].describe().T[["mean", "std", "min", "max"]].round(4).to_string(), flush=True)

    # Metadata
    param_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "params": params,
        "param_hash": param_hash,
        "n_rows": len(df),
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "label_summary": label_summary,
        "enrichment_errors": enrichment_errors,
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✓ Saved metadata: {meta_path}", flush=True)

    return {**label_summary, "output_path": str(output_path), "meta_path": str(meta_path)}


# Pre-sets for common parameter choices
_PRESETS = {
    "v4_winner": {
        **PARAMS,
        "min_edge": 0.16,
        "green_book_delta_nba": 0.15,
        "stop_loss_delta": 0.15,
        "max_hold_fraction": 0.50,
        "stop_loss_enabled": True,
        "both_sides": True,
    },
    "v4_current": {
        **PARAMS,
    },
    "wide_edge": {
        # Lowered edge + widened price band to capture MORE trades
        # (useful for training — we want signal diversity, not just winners)
        **PARAMS,
        "min_edge": 0.03,
        "max_edge": 0.40,
        "min_price": 0.10,
        "max_price": 0.85,
        "both_sides": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="SQLite DB path")
    parser.add_argument("--params-preset", choices=list(_PRESETS.keys()),
                        default="wide_edge",
                        help="Parameter preset (wide_edge = more signals for training)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit markets scanned (0 = all)")
    parser.add_argument("--output", type=Path,
                        help="Output parquet path (default: research_d/data/training_set_v1.parquet)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", flush=True)
        sys.exit(1)

    params = _PRESETS[args.params_preset]
    print(f"Using preset '{args.params_preset}' with {len(params)} params", flush=True)
    print(f"DB: {args.db}", flush=True)

    build_training_set(args.db, params, args.limit, args.output)


if __name__ == "__main__":
    main()
