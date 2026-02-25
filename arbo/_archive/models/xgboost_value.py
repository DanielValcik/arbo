"""XGBoost value model for Polymarket probability prediction.

Predicts calibrated P(YES) for binary markets using features
from Polymarket prices, Pinnacle odds, and market metadata.

Workflow:
  1. tune_hyperparameters() — Optuna search (50 trials)
  2. ValueModel.train() — Train with Platt calibration
  3. ValueModel.predict_proba() — Calibrated predictions
  4. backtest_strategy() — Validate ROI on historical data

See PM-101 specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

from arbo.models.calibration import PlattCalibrator, brier_score
from arbo.models.feature_engineering import (
    FEATURE_COLUMNS,
    MarketFeatures,
    extract_feature_vector,
    features_to_dataframe,
)
from arbo.utils.logger import get_logger

logger = get_logger("value_model")

# Default hyperparameters (overridden by Optuna tuning)
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 300,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "tree_method": "hist",
}


class ValueModel:
    """XGBoost-based value model with Platt calibration.

    Acceptance: Brier < 0.22 on holdout, ROI > 2% on 200+ backtest bets.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._model: xgb.XGBClassifier | None = None
        self._calibrator = PlattCalibrator()
        self._is_trained = False
        self._brier_score: float | None = None
        self._feature_importance: dict[str, float] | None = None

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    @property
    def brier_score_val(self) -> float | None:
        """Brier score on calibration set from last training."""
        return self._brier_score

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Feature importance dict {name: importance}."""
        return self._feature_importance

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        calibration_fraction: float = 0.15,
        stratify_col: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the model on labeled data.

        Splits data into train + calibration sets. Trains XGBoost,
        then applies Platt scaling on the calibration set.

        Args:
            X: Feature DataFrame (columns must match FEATURE_COLUMNS).
            y: Binary labels (0 or 1).
            calibration_fraction: Fraction of data held out for Platt scaling.
            stratify_col: Optional array for stratified splitting (e.g. league).

        Returns:
            Dict with training metrics.
        """
        # Split train vs calibration
        indices = np.arange(len(X))
        if stratify_col is not None:
            train_idx, cal_idx = train_test_split(
                indices, test_size=calibration_fraction, stratify=stratify_col, random_state=42
            )
        else:
            train_idx, cal_idx = train_test_split(
                indices, test_size=calibration_fraction, stratify=y, random_state=42
            )

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_cal, y_cal = X.iloc[cal_idx], y[cal_idx]

        # Train XGBoost
        n_estimators = self._params.pop("n_estimators", 300)
        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            **self._params,
            random_state=42,
        )
        self._params["n_estimators"] = n_estimators

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_cal, y_cal)],
            verbose=False,
        )

        # Raw predictions for metrics
        raw_train_probs = self._model.predict_proba(X_train)[:, 1]
        raw_cal_probs = self._model.predict_proba(X_cal)[:, 1]

        brier_train = brier_score(y_train, raw_train_probs)
        brier_cal_raw = brier_score(y_cal, raw_cal_probs)

        # Apply Platt scaling on calibration set (raw probs → calibrated)
        self._calibrator.fit(raw_cal_probs, y_cal)
        cal_probs = self._calibrator.predict(raw_cal_probs)
        brier_cal = brier_score(y_cal, cal_probs)

        self._brier_score = brier_cal
        self._is_trained = True

        # Feature importance
        importance = self._model.feature_importances_
        self._feature_importance = dict(zip(FEATURE_COLUMNS, importance.tolist(), strict=True))

        metrics = {
            "brier_train": brier_train,
            "brier_cal_raw": brier_cal_raw,
            "brier_cal_calibrated": brier_cal,
            "n_train": len(X_train),
            "n_cal": len(X_cal),
            "calibration_improvement": brier_cal_raw - brier_cal,
        }

        logger.info(
            "model_trained",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        )
        return metrics

    def predict_proba(self, features: MarketFeatures | list[MarketFeatures]) -> np.ndarray:
        """Predict calibrated P(YES) for one or more markets.

        Args:
            features: Single or list of MarketFeatures.

        Returns:
            Array of calibrated probabilities [0, 1].
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(features, MarketFeatures):
            features = [features]

        feature_dicts = [extract_feature_vector(f) for f in features]
        df = features_to_dataframe(feature_dicts)
        return self.predict_proba_df(df)

    def predict_proba_df(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated P(YES) from a feature DataFrame.

        Args:
            X: DataFrame with columns matching FEATURE_COLUMNS.

        Returns:
            Array of calibrated probabilities.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        raw_probs = self._model.predict_proba(X)[:, 1]
        if self._calibrator.is_fitted:
            return self._calibrator.predict(raw_probs)
        return raw_probs

    def predict_single(self, features: MarketFeatures) -> float:
        """Predict P(YES) for a single market. Convenience wrapper."""
        return float(self.predict_proba(features)[0])

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Evaluate model on a test set.

        Returns:
            Dict with brier_score, accuracy, n_samples, etc.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained.")

        probs = self.predict_proba_df(X)
        preds = (probs >= 0.5).astype(int)

        return {
            "brier_score": brier_score(y, probs),
            "accuracy": float((preds == y).mean()),
            "n_samples": len(y),
            "mean_predicted_prob": float(probs.mean()),
            "positive_rate": float(y.mean()),
        }

    def save(self, path: str | Path) -> None:
        """Save model + calibrator to disk via joblib."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "calibrator": self._calibrator,
            "params": self._params,
            "brier_score": self._brier_score,
            "feature_importance": self._feature_importance,
        }
        joblib.dump(data, path)
        logger.info("model_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load model + calibrator from disk."""
        path = Path(path)
        data = joblib.load(path)
        self._model = data["model"]
        self._calibrator = data["calibrator"]
        self._params = data["params"]
        self._brier_score = data["brier_score"]
        self._feature_importance = data["feature_importance"]
        self._is_trained = True
        logger.info("model_loaded", path=str(path), brier=self._brier_score)


def tune_hyperparameters(
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 50,
    stratify_col: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter tuning.

    Optimizes Brier score on 3-fold cross-validation.

    Args:
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials (PM-101 spec: 50).
        stratify_col: For stratified CV.

    Returns:
        Best hyperparameters dict.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
        }

        cv_strat = stratify_col if stratify_col is not None else y
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        brier_scores = []
        for train_idx, val_idx in cv.split(X, cv_strat):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            n_est = params.pop("n_estimators")
            model = xgb.XGBClassifier(n_estimators=n_est, **params, random_state=42)
            params["n_estimators"] = n_est

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            brier_scores.append(brier_score(y_val, probs))

        return float(np.mean(brier_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "logloss"
    best["tree_method"] = "hist"

    logger.info(
        "optuna_complete",
        n_trials=n_trials,
        best_brier=round(study.best_value, 4),
    )

    return best


def backtest_strategy(
    model: ValueModel,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    market_prices: np.ndarray,
    fee_rates: np.ndarray,
    edge_threshold: float = 0.03,
    kelly_fraction: float = 0.5,
    initial_bankroll: float = 10000.0,
    flat_staking: bool = False,
) -> dict[str, Any]:
    """Simulate betting strategy on historical data.

    For each sample where edge > threshold, place a bet sized by Kelly.

    Args:
        model: Trained ValueModel.
        X_test: Test features DataFrame.
        y_test: Actual outcomes (0/1).
        market_prices: Polymarket YES prices at time of bet.
        fee_rates: Fee for each market.
        edge_threshold: Minimum edge to place bet (PM-101: 0.03).
        kelly_fraction: Kelly fraction (hardcoded 0.5).
        initial_bankroll: Starting capital.
        flat_staking: If True, stake relative to initial_bankroll (not compound).

    Returns:
        Dict with roi, n_bets, sharpe, max_drawdown, etc.
    """
    probs = model.predict_proba_df(X_test)

    bankroll = initial_bankroll
    peak = bankroll
    max_dd = 0.0
    returns: list[float] = []
    edges: list[float] = []
    n_bets = 0

    for i in range(len(X_test)):
        prob = float(probs[i])
        price = float(market_prices[i])
        fee = float(fee_rates[i])

        edge = prob - price - fee
        if edge < edge_threshold:
            continue

        # Kelly sizing
        if price <= 0 or price >= 1:
            continue
        b = (1.0 / price) - 1  # Net odds
        q = 1 - prob
        kelly = (prob * b - q) / b if b > 0 else 0
        if kelly <= 0:
            continue

        stake_pct = min(kelly * kelly_fraction, 0.05)  # Hard cap 5%
        stake_base = initial_bankroll if flat_staking else bankroll
        stake = stake_base * stake_pct

        # Resolve bet
        outcome = int(y_test[i])
        pnl = stake * ((1.0 / price) - 1) if outcome == 1 else -stake

        old_bankroll = bankroll
        bankroll += pnl
        if old_bankroll > 0:
            returns.append(pnl / old_bankroll)
        edges.append(edge)
        n_bets += 1

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    roi = (bankroll - initial_bankroll) / initial_bankroll
    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if len(returns) > 1 and np.std(returns) > 0
        else 0.0
    )

    return {
        "roi": roi,
        "n_bets": n_bets,
        "final_bankroll": bankroll,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_edge": float(np.mean(edges)) if edges else 0.0,
    }
