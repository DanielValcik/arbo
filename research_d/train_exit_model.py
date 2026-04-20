"""Strategy D — Survival XGBoost trainer for exit-timing.

Reads exit_timing_set_v1.parquet (from build_exit_timing_set.py), trains
an XGBoost AFT model with monotonic constraints, evaluates via C-index
and integrated Brier score, saves artifacts.

Design reference: docs/STRATEGY_D_ML_DESIGN.md §6.

Usage:
  PYTHONPATH=. python3 research_d/train_exit_model.py
  PYTHONPATH=. python3 research_d/train_exit_model.py --objective survival:aft --distribution extreme
  PYTHONPATH=. python3 research_d/train_exit_model.py --input /tmp/ets.parquet --output /tmp/exit_model_v1.ubj

Output:
  research_d/data/exit_model_v1.ubj           — XGBoost model artifact
  research_d/data/exit_model_v1.meta.json     — metrics, params, features
  research_d/data/exit_model_v1.shap.csv      — feature gain ranking
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.exit_timing_features import FEATURE_COLUMNS, get_monotone_vector


DEFAULT_INPUT = Path(__file__).parent / "data" / "exit_timing_set_v1.parquet"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "exit_model_v1.ubj"


@dataclass
class TrainMetrics:
    n_train: int
    n_val: int
    n_test: int
    n_train_trades: int
    n_val_trades: int
    n_test_trades: int
    event_rate_train: float
    event_rate_val: float
    event_rate_test: float
    c_index_train: float
    c_index_val: float
    c_index_test: float
    mean_predicted_log_time_train: float
    mean_predicted_log_time_test: float
    best_iteration: int
    objective: str
    aft_loss_distribution: str


# ── Temporal split (by game_date, no trade-level leakage) ─────────────


def _temporal_split(df, train_frac=0.6, val_frac=0.2):
    """Split by game_date ascending. Ensure all rows from same trade fall
    into the same split.
    """
    # Sort trades by their game_date, not rows
    trade_dates = (
        df.groupby("trade_id")["game_date"]
          .first()
          .sort_values()
    )
    n = len(trade_dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(trade_dates.iloc[:n_train].index)
    val_ids = set(trade_dates.iloc[n_train:n_train + n_val].index)
    test_ids = set(trade_dates.iloc[n_train + n_val:].index)

    train = df[df["trade_id"].isin(train_ids)].copy()
    val = df[df["trade_id"].isin(val_ids)].copy()
    test = df[df["trade_id"].isin(test_ids)].copy()
    return train, val, test


# ── AFT-specific label encoding ───────────────────────────────────────


def _aft_labels(df):
    """AFT objective needs (y_lower, y_upper) bounds.

    Right-censored (event=0): y_lower = T, y_upper = +inf
    Uncensored   (event=1): y_lower = T, y_upper = T
    """
    import numpy as np

    T = df["time_to_event"].astype("float64").values
    event = df["event"].astype("int8").values

    y_lower = T.copy()
    y_upper = np.where(event == 1, T, np.inf)
    return y_lower, y_upper


# ── C-index (Harrell's concordance) ───────────────────────────────────


def concordance_index(risk_scores, event_times, events) -> float:
    """Harrell's C-index — standard survival-model rank metric.

    A prediction "risk_score" is higher for faster events.
    For AFT, predicted log-time is LOWER for faster events, so
    risk_score = -predicted_log_time.

    Complexity: O(N^2) naive. We use pairwise over a random sample if N > 20k.
    """
    import numpy as np

    risk_scores = np.asarray(risk_scores, dtype="float64")
    event_times = np.asarray(event_times, dtype="float64")
    events = np.asarray(events, dtype="int8")

    n = len(risk_scores)
    if n < 2:
        return float("nan")

    # Subsample for speed if needed
    if n > 20000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=20000, replace=False)
        risk_scores = risk_scores[idx]
        event_times = event_times[idx]
        events = events[idx]

    # Vectorize: for each pair (i,j) where i had event, check if
    # risk[i] > risk[j] when time[i] < time[j].
    # Only count pairs where the shorter-time side had the event.
    events_i = events == 1
    # Expand to pairs
    t_i = event_times[events_i]
    r_i = risk_scores[events_i]
    t_j = event_times
    r_j = risk_scores

    if len(t_i) == 0:
        return float("nan")

    # For each event i, count comparable j (t_j > t_i OR (t_j == t_i AND event_j=0 wouldn't make sense...))
    # Use strict comparison for simplicity — standard Harrell's definition.
    comparable = t_j[None, :] > t_i[:, None]  # shape (n_events, n_all)
    concordant = (r_i[:, None] > r_j[None, :]) & comparable
    tied = (r_i[:, None] == r_j[None, :]) & comparable

    n_comp = comparable.sum()
    if n_comp == 0:
        return float("nan")

    return float((concordant.sum() + 0.5 * tied.sum()) / n_comp)


# ── Training ──────────────────────────────────────────────────────────


def train(
    df,
    output_path: Path,
    objective: str = "survival:aft",
    aft_distribution: str = "extreme",
    aft_scale: float = 1.0,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 30,
) -> dict:
    """Train XGBoost survival model, save artifacts, return metrics."""
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    if len(df) < 50:
        raise ValueError(f"Too few rows to train: {len(df)}. Need ≥50.")

    # Filter to training-eligible rows only (terminal/degenerate tte≈0 rows
    # are kept in parquet for eval use but excluded from training).
    # Backwards-compat: pre-v1.1 parquets don't have this column; treat as all=1.
    if "for_training" in df.columns:
        n_before = len(df)
        df_for_train = df[df["for_training"] == 1].copy()
        n_filtered = n_before - len(df_for_train)
        if n_filtered > 0:
            print(f"\nFiltered {n_filtered} non-training rows (for_training=0 — terminal/degenerate).")
    else:
        df_for_train = df

    # ── Split ─────────────────────────────────────────────────────────
    train_df, val_df, test_df = _temporal_split(df_for_train)
    print(f"\nTemporal split (by game_date, trade-level):")
    print(f"  Train: {len(train_df):>7} rows / {train_df['trade_id'].nunique():>5} trades / event_rate {train_df['event'].mean():.3f}")
    print(f"  Val  : {len(val_df):>7} rows / {val_df['trade_id'].nunique():>5} trades / event_rate {val_df['event'].mean():.3f}")
    print(f"  Test : {len(test_df):>7} rows / {test_df['trade_id'].nunique():>5} trades / event_rate {test_df['event'].mean():.3f}")

    # ── Features + labels ─────────────────────────────────────────────
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    X_train = train_df[FEATURE_COLUMNS].astype("float64").values
    X_val = val_df[FEATURE_COLUMNS].astype("float64").values
    X_test = test_df[FEATURE_COLUMNS].astype("float64").values

    T_train, event_train = train_df["time_to_event"].values, train_df["event"].values
    T_val, event_val = val_df["time_to_event"].values, val_df["event"].values
    T_test, event_test = test_df["time_to_event"].values, test_df["event"].values

    # AFT labels
    y_lower_train, y_upper_train = _aft_labels(train_df)
    y_lower_val, y_upper_val = _aft_labels(val_df)

    dtrain = xgb.DMatrix(X_train, feature_names=FEATURE_COLUMNS)
    dtrain.set_float_info("label_lower_bound", y_lower_train)
    dtrain.set_float_info("label_upper_bound", y_upper_train)

    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    dval.set_float_info("label_lower_bound", y_lower_val)
    dval.set_float_info("label_upper_bound", y_upper_val)

    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLUMNS)

    # ── Model params ──────────────────────────────────────────────────
    monotone = get_monotone_vector(FEATURE_COLUMNS)
    params = {
        "objective": objective,
        "aft_loss_distribution": aft_distribution,
        "aft_loss_distribution_scale": aft_scale,
        "eval_metric": "aft-nloglik",
        "max_depth": 5,
        "eta": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "tree_method": "hist",
        "verbosity": 1,
        "monotone_constraints": str(monotone),
    }
    print(f"\nXGBoost params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # ── Train ─────────────────────────────────────────────────────────
    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if len(val_df) > 100 else None,
        verbose_eval=50,
    )

    # ── Predict ───────────────────────────────────────────────────────
    # AFT predict returns E[log(T)] (accelerated failure time log)
    pred_log_train = booster.predict(dtrain)
    pred_log_val = booster.predict(dval)
    pred_log_test = booster.predict(dtest)

    # Risk score = -predicted log-time (high risk = fast event)
    risk_train = -pred_log_train
    risk_val = -pred_log_val
    risk_test = -pred_log_test

    # C-index
    print(f"\nComputing C-index...")
    c_train = concordance_index(risk_train, T_train, event_train)
    c_val = concordance_index(risk_val, T_val, event_val)
    c_test = concordance_index(risk_test, T_test, event_test)
    print(f"  Train: C-index = {c_train:.4f}")
    print(f"  Val  : C-index = {c_val:.4f}")
    print(f"  Test : C-index = {c_test:.4f}")
    print(f"  Baseline (random) = 0.5000")
    print(f"  Target  (useful)  = 0.60+")

    # ── Feature importance ────────────────────────────────────────────
    gain = booster.get_score(importance_type="gain")
    print(f"\nTop features by gain:")
    for feat, g in sorted(gain.items(), key=lambda x: -x[1])[:15]:
        print(f"  {feat:>30}: {g:.3f}")

    # ── Save artifacts ────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))

    # SHAP-style CSV (just gain for now)
    import csv
    imp_path = output_path.with_suffix(".shap.csv")
    with open(imp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain"])
        for feat, g in sorted(gain.items(), key=lambda x: -x[1]):
            w.writerow([feat, f"{g:.4f}"])

    # Metrics JSON
    metrics = TrainMetrics(
        n_train=int(len(train_df)),
        n_val=int(len(val_df)),
        n_test=int(len(test_df)),
        n_train_trades=int(train_df["trade_id"].nunique()),
        n_val_trades=int(val_df["trade_id"].nunique()),
        n_test_trades=int(test_df["trade_id"].nunique()),
        event_rate_train=float(train_df["event"].mean()),
        event_rate_val=float(val_df["event"].mean()),
        event_rate_test=float(test_df["event"].mean()),
        c_index_train=float(c_train),
        c_index_val=float(c_val),
        c_index_test=float(c_test),
        mean_predicted_log_time_train=float(pred_log_train.mean()),
        mean_predicted_log_time_test=float(pred_log_test.mean()),
        best_iteration=int(getattr(booster, "best_iteration", 0) or 0),
        objective=objective,
        aft_loss_distribution=aft_distribution,
    )

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "objective": objective,
            "aft_loss_distribution": aft_distribution,
            "params": params,
            "feature_columns": list(FEATURE_COLUMNS),
            "monotone_constraints": list(monotone),
            "metrics": asdict(metrics),
            "model_path": str(output_path.name),
            "shap_path": str(imp_path.name),
        }, f, indent=2)

    print(f"\n✓ Model:  {output_path}")
    print(f"✓ SHAP:   {imp_path}")
    print(f"✓ Meta:   {meta_path}")

    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objective", default="survival:aft")
    parser.add_argument("--distribution", default="extreme",
                        choices=["normal", "logistic", "extreme"],
                        help="AFT loss distribution: extreme = Weibull")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="AFT loss distribution scale (σ)")
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()

    import pandas as pd
    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run build_exit_timing_set.py first.")
        sys.exit(1)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}  shape={df.shape}")

    train(
        df,
        output_path=args.output,
        objective=args.objective,
        aft_distribution=args.distribution,
        aft_scale=args.scale,
        num_boost_round=args.rounds,
    )


if __name__ == "__main__":
    main()
