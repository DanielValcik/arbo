"""Run full backtest pipeline: load data, train model, evaluate, generate CEO report.

Usage:
    python3 scripts/run_backtest.py [--tune] [--output path] [--input path]

Pipeline:
    1. Load BacktestDataset from JSON
    2. Build training data (3 entries per match)
    3. Time-series split (70/30 chronological)
    4. Optional Optuna hyperparameter tuning
    5. Train ValueModel
    6. Evaluate on test set
    7. Run backtest_strategy simulation
    8. Generate CEO report with pass/fail gates
    9. Save model to data/models/backtest_model.joblib

See Sprint 4 Phase A specification.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbo.models.xgboost_value import (
    ValueModel,
    backtest_strategy,
    tune_hyperparameters,
)
from arbo.utils.logger import get_logger
from scripts.backfill_data import BacktestDataset
from scripts.process_data import build_training_data

logger = get_logger("backtest")

# ================================================================
# Pass/fail gates for CEO report
# ================================================================

GATES = {
    "brier_score": {"threshold": 0.22, "direction": "below", "label": "Brier Score < 0.22"},
    "roi": {"threshold": 0.02, "direction": "above", "label": "ROI > 2%"},
    "n_bets": {"threshold": 200, "direction": "above", "label": "N Bets >= 200"},
    "max_drawdown": {"threshold": 0.30, "direction": "below", "label": "Max Drawdown < 30%"},
}


def check_gate(metric_name: str, value: float) -> bool:
    """Check if a metric passes its gate.

    Args:
        metric_name: Key from GATES dict.
        value: Metric value to check.

    Returns:
        True if the gate is passed.
    """
    gate = GATES.get(metric_name)
    if gate is None:
        return True

    if gate["direction"] == "below":
        return value < gate["threshold"]
    return value >= gate["threshold"]


# ================================================================
# Time-series split
# ================================================================


def time_series_split(
    dataset: BacktestDataset,
    train_fraction: float = 0.7,
) -> tuple[BacktestDataset, BacktestDataset]:
    """Split dataset chronologically (NO random split — prevents leakage).

    Args:
        dataset: Full BacktestDataset.
        train_fraction: Fraction of matches for training.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    sorted_matches = sorted(dataset.matches, key=lambda m: m.date)
    n_train = int(len(sorted_matches) * train_fraction)

    train_matches = sorted_matches[:n_train]
    test_matches = sorted_matches[n_train:]

    train_ds = BacktestDataset(
        n_matches=len(train_matches),
        leagues=dataset.leagues,
        seasons=dataset.seasons,
        matches=train_matches,
    )
    test_ds = BacktestDataset(
        n_matches=len(test_matches),
        leagues=dataset.leagues,
        seasons=dataset.seasons,
        matches=test_matches,
    )

    return train_ds, test_ds


# ================================================================
# Report generation
# ================================================================


def generate_report(
    eval_metrics: dict[str, float],
    backtest_results: dict[str, Any],
    train_metrics: dict[str, float],
    n_train_matches: int,
    n_test_matches: int,
    tuned: bool,
    edge_threshold: float = 0.03,
    kelly_fraction: float = 0.5,
) -> dict[str, Any]:
    """Generate CEO report with pass/fail gates.

    Args:
        eval_metrics: Model evaluation metrics on test set.
        backtest_results: Backtest simulation results.
        train_metrics: Training metrics.
        n_train_matches: Number of matches in training set.
        n_test_matches: Number of matches in test set.
        tuned: Whether Optuna tuning was used.
        edge_threshold: Edge threshold used in backtest.
        kelly_fraction: Kelly fraction used in backtest.

    Returns:
        Report dict suitable for JSON serialization.
    """
    gates_results: dict[str, dict[str, Any]] = {}

    # Brier score gate
    brier = eval_metrics.get("brier_score", 1.0)
    gates_results["brier_score"] = {
        "value": round(brier, 4),
        "passed": check_gate("brier_score", brier),
        "gate": GATES["brier_score"]["label"],
    }

    # ROI gate
    roi = backtest_results.get("roi", 0.0)
    gates_results["roi"] = {
        "value": round(roi, 4),
        "passed": check_gate("roi", roi),
        "gate": GATES["roi"]["label"],
    }

    # N bets gate
    n_bets = backtest_results.get("n_bets", 0)
    gates_results["n_bets"] = {
        "value": n_bets,
        "passed": check_gate("n_bets", n_bets),
        "gate": GATES["n_bets"]["label"],
    }

    # Max drawdown gate
    max_dd = backtest_results.get("max_drawdown", 1.0)
    gates_results["max_drawdown"] = {
        "value": round(max_dd, 4),
        "passed": check_gate("max_drawdown", max_dd),
        "gate": GATES["max_drawdown"]["label"],
    }

    all_passed = all(g["passed"] for g in gates_results.values())

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "overall_verdict": "PASS" if all_passed else "FAIL",
        "gates": gates_results,
        "evaluation": {
            "brier_score": round(eval_metrics.get("brier_score", 0), 4),
            "accuracy": round(eval_metrics.get("accuracy", 0), 4),
            "n_test_samples": eval_metrics.get("n_samples", 0),
        },
        "backtest": {
            "roi": round(backtest_results.get("roi", 0), 4),
            "n_bets": backtest_results.get("n_bets", 0),
            "final_bankroll": round(backtest_results.get("final_bankroll", 0), 2),
            "max_drawdown": round(backtest_results.get("max_drawdown", 0), 4),
            "sharpe": round(backtest_results.get("sharpe", 0), 4),
            "avg_edge": round(backtest_results.get("avg_edge", 0), 4),
        },
        "training": {
            "brier_train": round(train_metrics.get("brier_train", 0), 4),
            "brier_calibrated": round(train_metrics.get("brier_cal_calibrated", 0), 4),
            "n_train_samples": train_metrics.get("n_train", 0),
            "n_cal_samples": train_metrics.get("n_cal", 0),
        },
        "data": {
            "n_train_matches": n_train_matches,
            "n_test_matches": n_test_matches,
        },
        "config": {
            "optuna_tuned": tuned,
            "time_series_split": "70/30 chronological",
            "edge_threshold": edge_threshold,
            "kelly_fraction": kelly_fraction,
        },
    }

    return report


# ================================================================
# Main pipeline
# ================================================================


def run_pipeline(
    input_path: Path,
    output_path: Path,
    tune: bool = False,
    n_trials: int = 50,
    edge_threshold: float = 0.03,
    kelly_fraction: float = 0.5,
    flat_staking: bool = False,
) -> dict[str, Any]:
    """Run the full backtest pipeline.

    Args:
        input_path: Path to backtest_dataset.json.
        output_path: Path to save model .joblib.
        tune: Whether to run Optuna hyperparameter tuning.
        n_trials: Number of Optuna trials.
        edge_threshold: Minimum edge to place bet.
        kelly_fraction: Kelly fraction for position sizing.
        flat_staking: Use flat staking (no compounding).

    Returns:
        CEO report dict.
    """
    # 1. Load dataset
    with open(input_path) as f:
        data = json.load(f)
    dataset = BacktestDataset.model_validate(data)
    logger.info("dataset_loaded", n_matches=dataset.n_matches)

    # 2. Time-series split
    train_ds, test_ds = time_series_split(dataset, train_fraction=0.7)
    logger.info(
        "split_done",
        train_matches=train_ds.n_matches,
        test_matches=test_ds.n_matches,
    )

    # 3. Build training data
    X_train, y_train, _prices_train, _fees_train = build_training_data(train_ds, seed=42)
    X_test, y_test, prices_test, fees_test = build_training_data(test_ds, seed=43)
    logger.info(
        "training_data_built",
        train_samples=len(y_train),
        test_samples=len(y_test),
    )

    if len(y_train) == 0 or len(y_test) == 0:
        logger.error("insufficient_data")
        return {"overall_verdict": "FAIL", "error": "Insufficient data"}

    # 4. Optional Optuna tuning
    params = None
    if tune:
        logger.info("optuna_tuning_start", n_trials=n_trials)
        params = tune_hyperparameters(X_train, y_train, n_trials=n_trials)
        logger.info("optuna_tuning_done", best_params=params)

    # 5. Train model
    model = ValueModel(params=params)
    train_metrics = model.train(X_train, y_train)
    logger.info("model_trained", **train_metrics)

    # 6. Evaluate on test set
    eval_metrics = model.evaluate(X_test, y_test)
    logger.info("model_evaluated", **eval_metrics)

    # 7. Run backtest simulation
    backtest_results = backtest_strategy(
        model=model,
        X_test=X_test,
        y_test=y_test,
        market_prices=prices_test,
        fee_rates=fees_test,
        edge_threshold=edge_threshold,
        kelly_fraction=kelly_fraction,
        flat_staking=flat_staking,
    )
    logger.info("backtest_done", **backtest_results)

    # 8. Generate CEO report
    report = generate_report(
        eval_metrics=eval_metrics,
        backtest_results=backtest_results,
        train_metrics=train_metrics,
        n_train_matches=train_ds.n_matches,
        n_test_matches=test_ds.n_matches,
        tuned=tune,
        edge_threshold=edge_threshold,
        kelly_fraction=kelly_fraction,
    )

    # 9. Save model
    model.save(output_path)
    logger.info("model_saved", path=str(output_path))

    # Save report
    report_path = output_path.parent / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("report_saved", path=str(report_path))

    # Print summary
    verdict = report["overall_verdict"]
    logger.info(
        "pipeline_complete",
        verdict=verdict,
        brier=report["evaluation"]["brier_score"],
        roi=report["backtest"]["roi"],
        n_bets=report["backtest"]["n_bets"],
    )

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run full backtest pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/backtest_dataset.json",
        help="Input dataset JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/backtest_model.joblib",
        help="Output model path",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum edge to place bet (default 0.05)",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Kelly fraction for position sizing (default 0.25)",
    )
    parser.add_argument(
        "--flat-staking",
        action="store_true",
        help="Use flat staking (% of initial bankroll, no compounding)",
    )
    args = parser.parse_args()

    report = run_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        tune=args.tune,
        n_trials=args.n_trials,
        edge_threshold=args.edge_threshold,
        kelly_fraction=args.kelly_fraction,
        flat_staking=args.flat_staking,
    )

    sys.exit(0 if report.get("overall_verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
