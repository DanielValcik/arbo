"""Process backfill dataset into XGBoost-ready training arrays.

Transforms HistoricalMatch records into 3 binary entries per match
(Home Win, Away Win, Draw) with simulated Polymarket prices and
feature vectors matching FEATURE_COLUMNS.

Usage:
    Imported by run_backtest.py. Can also be used standalone for inspection.

    python3 scripts/process_data.py [--input path] [--stats]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbo.models.feature_engineering import (
    FEATURE_COLUMNS,
    MarketFeatures,
    extract_feature_vector,
)
from arbo.utils.logger import get_logger
from scripts.backfill_data import BacktestDataset

logger = get_logger("process_data")


def build_training_data(
    dataset: BacktestDataset,
    polymarket_noise_std: float = 0.07,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Transform match dataset into XGBoost training arrays.

    Creates 3 binary entries per match: Home Win, Away Win, Draw.
    Each entry gets a simulated Polymarket price (opening_prob + noise).

    Args:
        dataset: BacktestDataset with historical matches.
        polymarket_noise_std: Std of Gaussian noise for simulated prices.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y, market_prices, fee_rates):
        - X: DataFrame with FEATURE_COLUMNS
        - y: Binary outcomes (1 if this outcome occurred)
        - market_prices: Simulated Polymarket YES prices
        - fee_rates: Fee rates (0.0 for soccer)
    """
    rng = np.random.default_rng(seed)

    feature_dicts: list[dict[str, float]] = []
    outcomes: list[int] = []
    prices: list[float] = []
    fees: list[float] = []

    for match in dataset.matches:
        # Get odds for implied probabilities
        home_odds = match.fd_home_odds
        draw_odds = match.fd_draw_odds
        away_odds = match.fd_away_odds

        # Use opening odds from Odds API if available
        if match.opening_odds:
            if match.opening_odds.home_win:
                home_odds = match.opening_odds.home_win
            if match.opening_odds.draw:
                draw_odds = match.opening_odds.draw
            if match.opening_odds.away_win:
                away_odds = match.opening_odds.away_win

        if home_odds is None or draw_odds is None or away_odds is None:
            continue
        if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
            continue

        # Convert odds to implied probabilities (with vig removal)
        raw_home = 1.0 / home_odds
        raw_draw = 1.0 / draw_odds
        raw_away = 1.0 / away_odds
        total = raw_home + raw_draw + raw_away

        if total <= 0:
            continue

        prob_home = raw_home / total
        prob_draw = raw_draw / total
        prob_away = raw_away / total

        # Calculate odds movement if we have both opening and closing
        odds_movement = None
        closing_vs_opening = None
        if (
            match.opening_odds
            and match.closing_odds
            and match.opening_odds.home_win
            and match.closing_odds.home_win
        ):
            odds_movement = (1.0 / match.closing_odds.home_win) - (
                1.0 / match.opening_odds.home_win
            )
            if match.opening_odds.home_win > 0:
                closing_vs_opening = match.closing_odds.home_win / match.opening_odds.home_win

        # Total goals line
        total_goals_line = None
        if match.opening_odds and match.opening_odds.over_25:
            total_goals_line = 2.5  # Standard line

        # Create 3 entries per match: Home Win, Away Win, Draw
        for outcome_name, pinnacle_prob, actual_result in [
            ("H", prob_home, match.result),
            ("A", prob_away, match.result),
            ("D", prob_draw, match.result),
        ]:
            # Simulated Polymarket price: opening prob + noise
            noise = rng.normal(0, polymarket_noise_std)
            sim_price = float(np.clip(pinnacle_prob + noise, 0.02, 0.98))

            # Binary outcome: 1 if this outcome occurred
            y_val = 1 if actual_result == outcome_name else 0

            # Build features
            mf = MarketFeatures(
                pinnacle_prob=pinnacle_prob,
                polymarket_mid=sim_price,
                time_to_event_hours=24.0,  # Simulated: betting 24h before
                category="soccer",
                volume_24h=0.0,  # Not available for historical
                volume_30d_avg=0.0,
                liquidity=0.0,
                spread=abs(pinnacle_prob - sim_price),
                fee_enabled=False,
                odds_movement_24h=odds_movement,
                total_goals_line=total_goals_line,
                closing_vs_opening_ratio=closing_vs_opening,
            )

            feature_dicts.append(extract_feature_vector(mf))
            outcomes.append(y_val)
            prices.append(sim_price)
            fees.append(0.0)  # Soccer: 0% fee on Polymarket

    # Build DataFrame
    X = pd.DataFrame(feature_dicts)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = float("nan")
    X = X[FEATURE_COLUMNS]

    y = np.array(outcomes, dtype=np.int64)
    market_prices = np.array(prices, dtype=np.float64)
    fee_rates = np.array(fees, dtype=np.float64)

    logger.info(
        "training_data_built",
        n_matches=len(dataset.matches),
        n_samples=len(y),
        positive_rate=float(y.mean()) if len(y) > 0 else 0.0,
    )

    return X, y, market_prices, fee_rates


def main() -> None:
    """CLI entry point for data inspection."""
    parser = argparse.ArgumentParser(description="Process backfill data into training arrays")
    parser.add_argument(
        "--input",
        type=str,
        default="data/backtest_dataset.json",
        help="Input dataset JSON path",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("dataset_not_found", path=str(input_path))
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)
    dataset = BacktestDataset.model_validate(data)

    X, y, market_prices, fee_rates = build_training_data(dataset)

    if args.stats:
        print(f"Samples: {len(y)}")
        print(f"Features: {list(X.columns)}")
        print(f"Positive rate: {y.mean():.3f}")
        print(f"Price range: [{market_prices.min():.3f}, {market_prices.max():.3f}]")
        print(f"Fee rates unique: {np.unique(fee_rates)}")
        print(f"Shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
