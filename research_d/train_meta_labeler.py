"""Strategy D — ML Meta-Labeler training (López de Prado pattern).

The primary signal (Elo+Pinnacle ensemble edge rule) stays unchanged.
This script trains a secondary model that predicts P(trade is profitable)
given features available AT ENTRY, used to filter out low-confidence
signals and scale sizing by predicted probability.

Design choices (per research findings 2026-04-20):
  - Calibrated XGBoost (not deep learning, not logistic alone)
  - Triple-barrier inspired binary label: y = 1 if pnl > 0
  - Platt scaling for N<2000; isotonic otherwise
  - CPCV-style sequential time splits (no random shuffle — leakage risk)
  - Monotonic constraints on edge-like features (larger edge → not lower P)
  - SHAP feature importance dump for interpretability

Usage:
  PYTHONPATH=. python3 research_d/train_meta_labeler.py
  PYTHONPATH=. python3 research_d/train_meta_labeler.py --input research_d/data/training_set_v1.parquet
  PYTHONPATH=. python3 research_d/train_meta_labeler.py --label y_gb_hit

Output:
  research_d/data/meta_labeler_v1.json  — model + calibrator + metrics
  research_d/data/meta_labeler_v1_shap.csv  — feature importance
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Features used as model input (drop identity + label columns)
# Categorical features get one-hot encoded before training.
NUMERIC_FEATURES = [
    "model_prob", "entry_price", "edge", "edge_magnitude",
    "price_dist_50", "prob_confidence", "logit_price", "logit_model",
    "day_of_week", "month", "n_prices_total",
    "elo_prob", "elo_diff", "pinnacle_prob", "ensemble_disagreement",
    "elo_available", "pinnacle_available", "both_models_available",
]
CATEGORICAL_FEATURES = ["season_phase", "sport"]
IDENTITY_COLS = ["token_id", "game_date", "team_a", "team_b"]
LABEL_COLS = ["y_profitable", "y_gb_hit", "y_return", "y_pnl_usd"]

# Monotonic constraints: features where we expect prediction to be
# non-decreasing (1) or non-increasing (-1) in the feature value.
# Unlisted features get 0 (no constraint).
MONOTONIC_CONSTRAINTS = {
    "edge": 1,           # larger edge ⇒ not worse trade
    "edge_magnitude": 1, # larger |edge| ⇒ not worse trade
    "prob_confidence": 1, # farther from 0.5 ⇒ more decisive ⇒ not worse
    "both_models_available": 1,  # both model signals align ⇒ not worse
}


@dataclass
class TrainMetrics:
    """Aggregate training/validation metrics."""
    n_train: int
    n_val: int
    n_test: int
    train_auc: float
    val_auc: float
    test_auc: float
    train_brier: float
    val_brier: float
    test_brier: float
    train_logloss: float
    val_logloss: float
    test_logloss: float
    baseline_profitable_rate: float   # naive P(y=1) from data
    positive_rate_train: float
    positive_rate_test: float
    calibrator: str


def _prepare_features(df):
    """Split DataFrame into X (feature matrix) and y (label vector).

    One-hot encodes categoricals, drops identity columns.
    """
    import pandas as pd

    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], dummy_na=False,
                           prefix_sep="__", dtype="int8")
    X = pd.concat([X_num, X_cat], axis=1)

    # Ensure no object dtype leaks through
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X


def _monotone_constraint_vector(feature_names: list[str]) -> tuple[int, ...]:
    """Build XGBoost monotone_constraints tuple aligned with feature_names."""
    return tuple(MONOTONIC_CONSTRAINTS.get(f, 0) for f in feature_names)


def _temporal_split(df, train_frac: float = 0.6, val_frac: float = 0.2):
    """Split by game_date ascending: first train_frac → train, next val_frac →
    val, rest → test.

    Prevents look-ahead leakage. Requires 'game_date' column.
    """
    df_sorted = df.sort_values("game_date").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df_sorted.iloc[:n_train].copy()
    val = df_sorted.iloc[n_train:n_train + n_val].copy()
    test = df_sorted.iloc[n_train + n_val:].copy()
    return train, val, test


def train(
    df,
    label_col: str = "y_profitable",
    output_path: Path | None = None,
    calibrator: str | None = None,
) -> dict:
    """Train XGBoost meta-labeler with Platt/isotonic calibration.

    Returns dict of metrics + paths to saved artifacts.
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression

    if label_col not in df.columns:
        raise ValueError(f"Label {label_col} not in dataframe columns: {list(df.columns)}")
    if len(df) < 10:
        raise ValueError(f"Too few rows to train: {len(df)}. Need ≥10 for any signal.")

    print(f"\nTraining meta-labeler:")
    print(f"  Label: {label_col}")
    print(f"  Total rows: {len(df)}")

    # Temporal split (no leakage)
    train_df, val_df, test_df = _temporal_split(df)
    if len(train_df) < 5 or len(test_df) < 3:
        warnings.warn(
            f"Tiny sample: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}. "
            f"Results will NOT be reliable. Scale training set on VPS first."
        )

    # Feature matrices
    X_train = _prepare_features(train_df)
    X_val = _prepare_features(val_df)
    X_test = _prepare_features(test_df)

    # Align columns (in case one split is missing a dummy level)
    all_cols = sorted(set(X_train.columns) | set(X_val.columns) | set(X_test.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_val = X_val.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values

    print(f"  Train: {len(X_train)} rows × {X_train.shape[1]} features, positive rate {y_train.mean():.2%}")
    print(f"  Val  : {len(X_val)} rows, positive rate {y_val.mean():.2%}" if len(y_val) else "  Val  : (empty)")
    print(f"  Test : {len(X_test)} rows, positive rate {y_test.mean():.2%}" if len(y_test) else "  Test : (empty)")

    # Decide calibrator: Platt for small N (≤ 2000 calibration samples)
    if calibrator is None:
        calibrator = "platt" if len(X_train) <= 2000 else "isotonic"

    # Build XGBoost booster
    monotone = _monotone_constraint_vector(list(X_train.columns))
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 4,          # shallow — avoid overfitting on few rows
        "eta": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "monotone_constraints": str(monotone),
        "tree_method": "hist",
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val) if len(X_val) else None
    dtest = xgb.DMatrix(X_test, label=y_test) if len(X_test) else None

    evals = [(dtrain, "train")]
    if dval is not None and len(X_val) >= 3:
        evals.append((dval, "val"))

    print(f"\n  XGBoost params: {params}")
    print(f"  Calibrator   : {calibrator}")
    print(f"  Training...")

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=25 if dval is not None and len(X_val) >= 10 else None,
        verbose_eval=50,
    )

    # Raw predictions
    train_raw = booster.predict(dtrain)
    val_raw = booster.predict(dval) if dval is not None else np.array([])
    test_raw = booster.predict(dtest) if dtest is not None else np.array([])

    # Calibration
    if calibrator == "platt":
        # Platt = logistic regression on raw scores
        from sklearn.linear_model import LogisticRegression
        calib_X = train_raw.reshape(-1, 1) if not len(val_raw) else val_raw.reshape(-1, 1)
        calib_y = y_train if not len(y_val) else y_val
        if len(calib_X) >= 5 and len(set(calib_y)) == 2:
            calib_model = LogisticRegression().fit(calib_X, calib_y)
            train_cal = calib_model.predict_proba(train_raw.reshape(-1, 1))[:, 1]
            val_cal = calib_model.predict_proba(val_raw.reshape(-1, 1))[:, 1] if len(val_raw) else val_raw
            test_cal = calib_model.predict_proba(test_raw.reshape(-1, 1))[:, 1] if len(test_raw) else test_raw
        else:
            train_cal, val_cal, test_cal = train_raw, val_raw, test_raw
            print(f"  WARN: Platt calibration skipped (not enough data or single-class)")
    else:  # isotonic
        calib_X = val_raw if len(val_raw) else train_raw
        calib_y = y_val if len(y_val) else y_train
        if len(calib_X) >= 20 and len(set(calib_y)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip").fit(calib_X, calib_y)
            train_cal = iso.transform(train_raw)
            val_cal = iso.transform(val_raw) if len(val_raw) else val_raw
            test_cal = iso.transform(test_raw) if len(test_raw) else test_raw
        else:
            train_cal, val_cal, test_cal = train_raw, val_raw, test_raw
            print(f"  WARN: Isotonic skipped (need ≥20 calib samples and both classes)")

    # Metrics — safely handle single-class splits
    def _safe_auc(y_true, y_prob):
        if len(y_true) == 0 or len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    def _safe_brier(y_true, y_prob):
        if len(y_true) == 0:
            return float("nan")
        return float(brier_score_loss(y_true, y_prob))

    def _safe_logloss(y_true, y_prob):
        if len(y_true) == 0 or len(set(y_true)) < 2:
            return float("nan")
        eps = 1e-7
        y_clip = np.clip(y_prob, eps, 1 - eps)
        return float(log_loss(y_true, y_clip, labels=[0, 1]))

    metrics = TrainMetrics(
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        train_auc=_safe_auc(y_train, train_cal),
        val_auc=_safe_auc(y_val, val_cal),
        test_auc=_safe_auc(y_test, test_cal),
        train_brier=_safe_brier(y_train, train_cal),
        val_brier=_safe_brier(y_val, val_cal),
        test_brier=_safe_brier(y_test, test_cal),
        train_logloss=_safe_logloss(y_train, train_cal),
        val_logloss=_safe_logloss(y_val, val_cal),
        test_logloss=_safe_logloss(y_test, test_cal),
        baseline_profitable_rate=float(df[label_col].mean()),
        positive_rate_train=float(np.mean(y_train)),
        positive_rate_test=float(np.mean(y_test)) if len(y_test) else 0.0,
        calibrator=calibrator,
    )

    # Print metrics
    print(f"\n  === Results ===")
    print(f"  AUC:     train={metrics.train_auc:.3f} val={metrics.val_auc:.3f} test={metrics.test_auc:.3f}")
    print(f"  Brier:   train={metrics.train_brier:.3f} val={metrics.val_brier:.3f} test={metrics.test_brier:.3f}")
    print(f"  LogLoss: train={metrics.train_logloss:.3f} val={metrics.val_logloss:.3f} test={metrics.test_logloss:.3f}")
    print(f"  Baseline (random guess by prior): AUC=0.50, Brier={metrics.baseline_profitable_rate * (1 - metrics.baseline_profitable_rate):.3f}")

    # Feature importance (gain)
    importance = booster.get_score(importance_type="gain")
    print(f"\n  Top features by gain:")
    for feat, gain in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"    {feat:>30}: {gain:.2f}")

    # Save model artifact
    if output_path is None:
        output_path = Path(__file__).parent / "data" / f"meta_labeler_{label_col}_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = output_path.with_suffix(".ubj")
    booster.save_model(str(model_path))

    # Save importance as CSV
    import csv
    imp_path = output_path.with_suffix(".shap.csv")
    with open(imp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "gain", "monotone_constraint"])
        for feat, gain in sorted(importance.items(), key=lambda x: -x[1]):
            w.writerow([feat, f"{gain:.4f}", MONOTONIC_CONSTRAINTS.get(feat, 0)])

    # Save metadata JSON
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "label_col": label_col,
            "calibrator": calibrator,
            "xgboost_params": params,
            "feature_columns": list(X_train.columns),
            "metrics": asdict(metrics),
            "model_path": str(model_path.name),
            "importance_path": str(imp_path.name),
        }, f, indent=2)

    print(f"\n  ✓ Model:      {model_path}")
    print(f"  ✓ Importance: {imp_path}")
    print(f"  ✓ Metadata:   {meta_path}")

    return {
        "metrics": asdict(metrics),
        "importance": dict(importance),
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).parent / "data" / "training_set_v1.parquet",
                        help="Input parquet (from build_training_set.py)")
    parser.add_argument("--label", default="y_profitable",
                        choices=LABEL_COLS,
                        help="Label column to train on")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: research_d/data/meta_labeler_<label>_v1.json)")
    parser.add_argument("--calibrator", choices=["platt", "isotonic", "none"], default=None,
                        help="Calibration method (auto by N if unset)")
    args = parser.parse_args()

    import pandas as pd
    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run build_training_set.py first.", flush=True)
        sys.exit(1)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    calibrator = None if args.calibrator in (None, "none") else args.calibrator
    train(df, label_col=args.label, output_path=args.output, calibrator=calibrator)


if __name__ == "__main__":
    main()
