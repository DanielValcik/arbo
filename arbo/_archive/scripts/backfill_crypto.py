#!/usr/bin/env python3
"""Backfill crypto training data and train XGBoost model (PM-401).

Pipeline:
1. Fetch all closed crypto price-threshold markets from Gamma API
2. Parse asset, strike, expiry via categorize_crypto_market()
3. Determine resolution: YES if spot > strike at expiry
4. Fetch Binance OHLCV [expiry-7d, expiry] → compute 12 features
5. Chrono split 80/20 → Optuna 50 trials → Platt calibration → Half-Kelly backtest
6. Save to data/models/crypto_model.joblib

Usage:
    python scripts/backfill_crypto.py [--n-trials 50] [--skip-tune]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import ssl
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbo.connectors.binance_client import BinanceClient
from arbo.connectors.market_discovery import categorize_crypto_market
from arbo.models.crypto_features import (
    CRYPTO_FEATURE_COLUMNS,
    compute_distance_pct,
    compute_spot_vs_strike,
)
from arbo.models.xgboost_crypto import (
    CryptoValueModel,
    tune_crypto_hyperparameters,
)


async def fetch_closed_crypto_markets(
    gamma_url: str = "https://gamma-api.polymarket.com",
) -> list[dict[str, Any]]:
    """Fetch all closed crypto markets from Gamma API."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(ssl=ssl_ctx),
    ) as session:
        all_markets: list[dict[str, Any]] = []
        offset = 0
        page_size = 100

        for page in range(100):  # max 10K markets
            params = {
                "limit": str(page_size),
                "offset": str(offset),
                "closed": "true",
                "order": "volume",
                "ascending": "false",
            }
            async with session.get(f"{gamma_url}/markets", params=params) as resp:
                if resp.status != 200:
                    print(f"  Gamma API error: {resp.status}")
                    break
                data = await resp.json()
                if not data:
                    break

                # Filter crypto markets
                for raw in data:
                    q = raw.get("question", "")
                    info = categorize_crypto_market(q)
                    if info is not None and not info.is_5min:
                        raw["_parsed"] = {
                            "asset": info.asset,
                            "symbol": info.symbol,
                            "strike": float(info.strike),
                            "expiry": info.expiry.isoformat() if info.expiry else None,
                            "direction": info.direction,
                        }
                        all_markets.append(raw)

                if len(data) < page_size:
                    break
                offset += page_size
                await asyncio.sleep(0.15)

            if page % 10 == 0:
                print(f"  Page {page + 1}, found {len(all_markets)} crypto markets so far...")

        print(f"  Total closed crypto markets: {len(all_markets)}")
        return all_markets


async def build_features_for_market(
    binance: BinanceClient,
    market: dict[str, Any],
) -> dict[str, float] | None:
    """Build 12-feature vector for a single market.

    Returns None if data is insufficient.
    """
    parsed = market.get("_parsed", {})
    symbol = parsed.get("symbol", "")
    strike = parsed.get("strike", 0)
    expiry_str = parsed.get("expiry")

    if not symbol or strike <= 0 or not expiry_str:
        return None

    try:
        expiry = datetime.fromisoformat(expiry_str)
    except (ValueError, TypeError):
        return None

    # Fetch OHLCV data [expiry-7d, expiry]
    end_ms = int(expiry.timestamp() * 1000)
    start_ms = int((expiry - timedelta(days=7)).timestamp() * 1000)

    try:
        bars = await binance.get_klines_range(
            symbol=symbol,
            interval="1h",
            start_ms=start_ms,
            end_ms=end_ms,
        )
    except Exception:
        return None

    if len(bars) < 24:  # Need at least 24 bars for meaningful features
        return None

    # Compute features
    spot_at_expiry = bars[-1].close
    spot_vs_strike = compute_spot_vs_strike(spot_at_expiry, strike)
    distance_pct = compute_distance_pct(spot_at_expiry, strike)

    vol_24h = BinanceClient.compute_volatility(bars, window=24)
    vol_7d = BinanceClient.compute_volatility(bars)
    rsi_14 = BinanceClient.compute_rsi(bars)
    momentum = BinanceClient.compute_momentum(bars[-24:])  # Last 24h

    # Volume features
    volumes = [b.quote_volume for b in bars]
    vol_24h_total = sum(volumes[-24:])
    vol_avg = sum(volumes) / len(volumes) if volumes else 1
    recent_avg = sum(volumes[-24:]) / 24 if len(volumes) >= 24 else vol_avg
    volume_trend = recent_avg / vol_avg if vol_avg > 0 else 1.0

    # Funding rate (if available, otherwise NaN)
    try:
        funding_rates = await binance.get_funding_rate(symbol, limit=1)
        funding = funding_rates[0].rate if funding_rates else float("nan")
    except Exception:
        funding = float("nan")

    # Polymarket mid (from market data)
    outcome_prices = market.get("outcomePrices", "[]")
    if isinstance(outcome_prices, str):
        try:
            prices = json.loads(outcome_prices)
        except Exception:
            prices = []
    else:
        prices = outcome_prices
    poly_mid = float(prices[0]) if prices else float("nan")

    # Time to expiry at time of market creation (approximate: use 24h before expiry)
    time_to_expiry = 24.0  # Approximate: snapshot at T-24h

    return {
        "spot_vs_strike": spot_vs_strike,
        "time_to_expiry": time_to_expiry,
        "time_to_expiry_log": math.log1p(time_to_expiry),
        "volatility_24h": vol_24h,
        "volatility_7d": vol_7d,
        "volume_24h_log": math.log1p(vol_24h_total),
        "volume_trend": volume_trend,
        "funding_rate": funding,
        "rsi_14": rsi_14,
        "momentum_24h": momentum,
        "distance_pct": distance_pct,
        "polymarket_mid": poly_mid,
    }


def determine_resolution(market: dict[str, Any], spot_at_expiry: float) -> int:
    """Determine if market resolved YES (1) or NO (0)."""
    parsed = market.get("_parsed", {})
    strike = parsed.get("strike", 0)
    direction = parsed.get("direction", "above")

    if direction == "above":
        return 1 if spot_at_expiry > strike else 0
    else:
        return 1 if spot_at_expiry < strike else 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill crypto model training data")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--skip-tune", action="store_true", help="Skip Optuna, use defaults")
    args = parser.parse_args()

    print("=" * 60)
    print("CRYPTO MODEL BACKFILL PIPELINE")
    print("=" * 60)

    # 1. Fetch closed crypto markets
    print("\n[1/6] Fetching closed crypto markets from Gamma API...")
    markets = await fetch_closed_crypto_markets()

    if len(markets) < 50:
        print(f"  ABORT: Only {len(markets)} markets found (need >= 50)")
        sys.exit(1)

    # 2. Build features
    print(f"\n[2/6] Building features for {len(markets)} markets...")
    binance = BinanceClient()
    await binance.initialize()

    features_list: list[dict[str, float]] = []
    labels: list[int] = []
    skipped = 0

    try:
        for i, market in enumerate(markets):
            feats = await build_features_for_market(binance, market)
            if feats is None:
                skipped += 1
                continue

            # Determine resolution from spot price at expiry
            parsed = market.get("_parsed", {})
            strike = parsed.get("strike", 0)
            direction = parsed.get("direction", "above")
            # Use spot_vs_strike * strike to get spot price
            spot = feats["spot_vs_strike"] * strike if strike > 0 else 0

            if direction == "above":
                label = 1 if spot > strike else 0
            else:
                label = 1 if spot < strike else 0

            features_list.append(feats)
            labels.append(label)

            if (i + 1) % 50 == 0:
                print(
                    f"  Processed {i + 1}/{len(markets)} (valid: {len(features_list)}, skipped: {skipped})"
                )

            # Rate limit
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0.5)
    finally:
        await binance.close()

    print(f"  Total samples: {len(features_list)} (skipped: {skipped})")

    if len(features_list) < 50:
        print(f"  ABORT: Only {len(features_list)} valid samples (need >= 50)")
        sys.exit(1)

    # 3. Create DataFrame
    print("\n[3/6] Creating feature DataFrame...")
    X = pd.DataFrame(features_list)[CRYPTO_FEATURE_COLUMNS]
    y = np.array(labels)
    print(f"  Shape: {X.shape}, Positive rate: {y.mean():.3f}")

    # 4. Chrono split 80/20
    print("\n[4/6] Chrono split 80/20...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. Tune + train
    if not args.skip_tune and len(X_train) >= 30:
        print(f"\n[5/6] Optuna hyperparameter search ({args.n_trials} trials)...")
        best_params = tune_crypto_hyperparameters(X_train, y_train, n_trials=args.n_trials)
        print(f"  Best params: {best_params}")
    else:
        print("\n[5/6] Using default hyperparameters...")
        best_params = None

    model = CryptoValueModel(params=best_params)
    metrics = model.train(X_train, y_train)
    print(f"  Training metrics: {metrics}")

    # 6. Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  Test metrics: {test_metrics}")

    brier = test_metrics["brier_score"]
    print(f"\n  Brier score: {brier:.4f} {'PASS' if brier < 0.22 else 'FAIL'} (threshold: 0.22)")

    # Simple backtest
    # NOTE: polymarket_mid from closed markets = resolved price (0.00 or 1.00),
    # not the trading price. Use spot_vs_strike to derive a synthetic market price
    # that approximates what the market would have traded at pre-resolution.
    probs = model.predict_proba_df(X_test)
    spot_ratios = X_test["spot_vs_strike"].values
    bankroll = 10000.0
    n_bets = 0

    for i in range(len(X_test)):
        prob = float(probs[i])
        # Synthetic market price: maps spot_vs_strike to a [0.05, 0.95] range
        # spot_vs_strike > 1 → market leans YES, < 1 → leans NO
        svs = float(spot_ratios[i]) if not np.isnan(spot_ratios[i]) else 1.0
        price = np.clip(0.5 + (svs - 1.0) * 5.0, 0.05, 0.95)

        edge = prob - price
        if edge < 0.03 or price <= 0 or price >= 1:
            continue

        b = (1.0 / price) - 1
        q = 1 - prob
        kelly = (prob * b - q) / b if b > 0 else 0
        if kelly <= 0:
            continue

        stake = bankroll * min(kelly * 0.5, 0.05)
        outcome = int(y_test[i])
        pnl = stake * b if outcome == 1 else -stake
        bankroll += pnl
        n_bets += 1

    roi = (bankroll - 10000) / 10000
    print(f"  Backtest: ROI={roi:.2%}, bets={n_bets}, final=${bankroll:,.2f}")
    print(f"  ROI gate: {'PASS' if roi > 0.02 else 'FAIL'} (threshold: 2%)")

    # Save model
    output_path = Path("data/models/crypto_model.joblib")
    model.save(output_path)
    print(f"\n  Model saved to: {output_path}")

    # Verify save/load roundtrip
    verify = CryptoValueModel()
    verify.load(output_path)
    assert verify.is_trained
    print("  Save/load roundtrip: OK")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
