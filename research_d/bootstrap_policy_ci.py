"""Block-bootstrap confidence interval on learned vs baseline PnL difference.

Tests if the +30.6% PnL lift is statistically significant given paired
observations across 440 test trades.

Uses paired block bootstrap (same trade → same resample) to preserve
correlation between policies' decisions on each trade.

Usage:
  PYTHONPATH=. python3 research_d/bootstrap_policy_ci.py
  PYTHONPATH=. python3 research_d/bootstrap_policy_ci.py --ets research_d/data/ets_nba_v2.parquet --model research_d/data/exit_model_nba_v2.ubj
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.exit_timing_features import FEATURE_COLUMNS
from research_d.eval_exit_policy import (
    _simulate_fixed_policy,
    _simulate_learned_policy,
)


DEFAULT_ETS = Path(__file__).parent / "data" / "ets_nba_v2.parquet"
DEFAULT_MODEL = Path(__file__).parent / "data" / "exit_model_nba_v2.ubj"


def run(
    ets_path: Path,
    model_path: Path,
    threshold_log_t: float = 6658.3,
    gb_delta: float = 0.15,
    sl_delta: float = 0.15,
    max_hold_fraction: float = 0.50,
    n_contracts: float = 30.0,
    n_bootstrap: int = 2000,
    split: str = "test",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Paired-bootstrap CI on (learned - baseline) total PnL."""
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    print(f"Loading ETS: {ets_path}")
    df = pd.read_parquet(ets_path)

    # Temporal split (match train_exit_model.py)
    trade_dates = df.groupby("trade_id")["game_date"].first().sort_values()
    n = len(trade_dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if split == "test":
        chosen_ids = set(trade_dates.iloc[n_train + n_val:].index)
    elif split == "val":
        chosen_ids = set(trade_dates.iloc[n_train:n_train + n_val].index)
    else:
        chosen_ids = set(trade_dates.index)

    eval_df = df[df["trade_id"].isin(chosen_ids)].sort_values(["trade_id", "t"])
    trade_ids = list(eval_df["trade_id"].unique())
    n_trades = len(trade_ids)
    print(f"  Split={split}: {n_trades} trades, {len(eval_df)} rows")

    # Load model
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Compute per-trade PnL under both policies (one pass)
    print(f"\nComputing per-trade PnL (baseline + learned, threshold={threshold_log_t:.1f})...")
    pnl_baseline = {}
    pnl_learned = {}
    for tid, group in eval_df.groupby("trade_id"):
        meta = group.iloc[0].to_dict()
        _, _, _, pps_a = _simulate_fixed_policy(
            group, meta, gb_delta, sl_delta, max_hold_fraction,
        )
        _, _, _, pps_b = _simulate_learned_policy(
            group, meta, booster, threshold_log_t,
            gb_delta, sl_delta, max_hold_fraction,
        )
        pnl_baseline[tid] = pps_a * n_contracts
        pnl_learned[tid] = pps_b * n_contracts

    arr_base = np.array([pnl_baseline[t] for t in trade_ids])
    arr_learn = np.array([pnl_learned[t] for t in trade_ids])
    diff = arr_learn - arr_base

    point_base = float(arr_base.sum())
    point_learn = float(arr_learn.sum())
    point_diff = float(diff.sum())

    print(f"\n=== Point estimates ===")
    print(f"  Baseline total PnL: ${point_base:+.2f}")
    print(f"  Learned  total PnL: ${point_learn:+.2f}")
    print(f"  Diff (learned-base): ${point_diff:+.2f}")

    # Bootstrap — paired by trade (trades are independent; ticks within trade are NOT)
    # This tests "does the learned vs baseline decision per trade systematically help?"
    rng = np.random.default_rng(seed)
    print(f"\nRunning {n_bootstrap} paired bootstraps of {n_trades} trades...")
    boot_diffs = np.empty(n_bootstrap)
    boot_bases = np.empty(n_bootstrap)
    boot_learns = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_trades, size=n_trades)
        boot_diffs[b] = diff[idx].sum()
        boot_bases[b] = arr_base[idx].sum()
        boot_learns[b] = arr_learn[idx].sum()

    # CIs via percentile method
    ci_lo, ci_hi = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    p_gt_zero = float((boot_diffs > 0).mean())

    print(f"\n=== Bootstrap results (N={n_bootstrap}, paired by trade) ===")
    print(f"  Diff bootstrap mean: ${float(boot_diffs.mean()):+.2f}")
    print(f"  Diff bootstrap std : ${float(boot_diffs.std()):.2f}")
    print(f"  95% CI on diff     : [${ci_lo:+.2f}, ${ci_hi:+.2f}]")
    print(f"  P(learned beats baseline): {p_gt_zero:.3f} ({p_gt_zero:.1%})")

    # Per-trade sign test (number of trades where learned > baseline)
    n_win = int((diff > 0).sum())
    n_lose = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    print(f"\n=== Per-trade outcome ===")
    print(f"  Learned strictly beats baseline: {n_win}/{n_trades} ({100*n_win/n_trades:.1f}%)")
    print(f"  Tied:   {n_tie}/{n_trades} ({100*n_tie/n_trades:.1f}%)")
    print(f"  Baseline wins: {n_lose}/{n_trades} ({100*n_lose/n_trades:.1f}%)")

    # Binomial test (ignoring ties) — is win-rate > 50%?
    from scipy import stats
    if n_win + n_lose > 0:
        binom = stats.binomtest(n_win, n_win + n_lose, p=0.5, alternative="greater")
        p_val_sign = binom.pvalue
        print(f"\n  Sign test P(learned wins > 50%): p = {p_val_sign:.4f}")
    else:
        p_val_sign = float("nan")

    # Verdict per Project PARALLEL promotion engine (P(better) >= 0.75)
    print(f"\n=== Promotion gate ===")
    print(f"  Framework §7 requires P(learned beats baseline) ≥ 0.75")
    if p_gt_zero >= 0.75:
        print(f"  ✓ PASS — P = {p_gt_zero:.3f} ≥ 0.75")
    else:
        print(f"  ✗ BELOW threshold — P = {p_gt_zero:.3f} < 0.75")

    return {
        "n_trades": n_trades,
        "point_baseline_pnl": point_base,
        "point_learned_pnl": point_learn,
        "point_diff": point_diff,
        "bootstrap_n": n_bootstrap,
        "bootstrap_mean_diff": float(boot_diffs.mean()),
        "bootstrap_std_diff": float(boot_diffs.std()),
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "p_learned_better": p_gt_zero,
        "n_learned_wins": n_win,
        "n_learned_ties": n_tie,
        "n_learned_losses": n_lose,
        "sign_test_p_value": p_val_sign,
        "promotion_gate_pass": p_gt_zero >= 0.75,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ets", type=Path, default=DEFAULT_ETS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=6658.3,
                        help="Threshold from val-set tuning")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-contracts", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = run(
        args.ets, args.model,
        threshold_log_t=args.threshold,
        n_bootstrap=args.n_bootstrap,
        n_contracts=args.n_contracts,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
