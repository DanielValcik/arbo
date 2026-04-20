"""Strategy D — Compare learned exit policy vs fixed GREEN_BOOK_DELTA rule.

For each held-out trade (with captured trajectory), re-simulate the
trade under TWO policies:

  (A) Baseline (fixed): the champion rule already in strategy_d_nba.py.
      Exit when price crosses entry+GB_DELTA, entry-SL_DELTA, or
      elapsed >= MAX_HOLD_FRACTION * game_duration.

  (B) Learned: at each tick, query the AFT model for expected log(T) to
      next GB. Exit when model says "event is far AND we're already
      profitable" — take profit instead of holding.

Aggregate results: total PnL, Sharpe, GB-rate, avg hold time, missed
upside. Report side-by-side.

Promotion gate (§7 of design doc):
  B must beat A on test PnL AND Sharpe within 10%, DD within 20%.

Usage:
  PYTHONPATH=. python3 research_d/eval_exit_policy.py
  PYTHONPATH=. python3 research_d/eval_exit_policy.py --ets /tmp/ets.parquet --model /tmp/exit_model.ubj
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_d.exit_timing_features import (
    FEATURE_COLUMNS,
    compute_features_at,
    find_first_gb_hit,
)


DEFAULT_ETS = Path(__file__).parent / "data" / "exit_timing_set_v1.parquet"
DEFAULT_MODEL = Path(__file__).parent / "data" / "exit_model_v1.ubj"


@dataclass
class PolicyResult:
    """Aggregate stats for a single exit policy over all evaluated trades."""
    name: str
    n_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    gb_exits: int = 0
    sl_exits: int = 0
    time_exits: int = 0
    early_exits: int = 0  # only learned: exited before any hard barrier
    hold_to_end_exits: int = 0
    hold_ticks_total: int = 0
    max_upside_captured: float = 0.0
    missed_upside_total: float = 0.0
    pnl_series: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.n_trades, 1)

    @property
    def avg_hold_ticks(self) -> float:
        return self.hold_ticks_total / max(self.n_trades, 1)

    @property
    def sharpe(self) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        mean = sum(self.pnl_series) / len(self.pnl_series)
        var = sum((p - mean) ** 2 for p in self.pnl_series) / (len(self.pnl_series) - 1)
        return mean / math.sqrt(var) if var > 1e-9 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown as fraction of peak cumulative."""
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in self.pnl_series:
            cum += p
            peak = max(peak, cum)
            if peak > 0:
                max_dd = max(max_dd, (peak - cum) / peak)
        return max_dd


def _simulate_fixed_policy(
    trade_group,
    trade_meta: dict,
    gb_delta: float,
    sl_delta: float,
    max_hold_fraction: float,
) -> tuple[int, float, str, float]:
    """Simulate fixed-rule exit on a single trade's trajectory.

    trade_group is the dataframe slice for one trade_id, sorted by t.
    Each row has (t, price_now, gb_already_touched, ...).

    Returns:
        (exit_t, exit_price, exit_reason, pnl_per_share)
    """
    entry_price = float(trade_meta["entry_price_level"])
    side = trade_meta["side"]

    # GB target + SL threshold (side-adjusted)
    if side == "yes":
        target = entry_price + gb_delta
        sl_price = entry_price - sl_delta
    else:
        target = entry_price - gb_delta
        sl_price = entry_price + sl_delta

    # Expected trajectory length (from any row — it's a trade-level constant)
    n_prices = trade_meta.get("n_prices_total", len(trade_group))
    max_hold_idx = int(n_prices * max_hold_fraction)

    # Walk through each tick
    for _, row in trade_group.iterrows():
        t = int(row["t"])
        price = float(row["price_now"])

        if side == "yes":
            gb_hit = price >= target
            sl_hit = price <= sl_price
        else:
            gb_hit = price <= target
            sl_hit = price >= sl_price

        if gb_hit:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "green_book", pnl_per_share
        if sl_hit:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "stop_loss", pnl_per_share
        if t >= max_hold_idx:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "time_exit", pnl_per_share

    # Fell through — use last price
    last_row = trade_group.iloc[-1]
    t = int(last_row["t"])
    price = float(last_row["price_now"])
    pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
    return t, price, "hold_to_end", pnl_per_share


def _aft_hazard_next_k(pred_mu: float, k: int, aft_scale: float) -> float:
    """P(event in next k ticks | state) under Weibull AFT.

    S(t|μ) = exp(-(t / exp(μ))^(1/σ)) where σ=aft_scale.
    Hazard-in-[0, k] = 1 - S(k).
    """
    if pred_mu <= 0 or k <= 0 or aft_scale <= 0:
        return 0.0
    alpha = math.exp(pred_mu)
    shape = 1.0 / aft_scale
    try:
        ratio = (k / alpha) ** shape
    except OverflowError:
        return 0.0
    # P(event in [0,k]) = 1 - exp(-ratio)
    return 1.0 - math.exp(-ratio) if ratio > 0 else 0.0


def _simulate_learned_policy(
    trade_group,
    trade_meta: dict,
    model,
    threshold_log_t: float,
    gb_delta: float,
    sl_delta: float,
    max_hold_fraction: float,
    policy_type: str = "raw_log_t",
    horizon_k: int = 10,
    aft_scale: float = 1.0,
) -> tuple[int, float, str, float]:
    """Simulate learned-policy exit.

    Logic:
      1. If hard GB target is hit — exit (safety net, same as baseline).
      2. If SL target is hit — exit.
      3. At each tick: query model for predicted log(T).
         policy_type='raw_log_t': exit if pred_log_t > threshold AND profit>0
         policy_type='hazard':    exit if P(event in next k) < threshold AND profit>0
      4. Else HOLD.
      5. Time exit at max_hold_fraction.
    """
    import xgboost as xgb

    entry_price = float(trade_meta["entry_price_level"])
    side = trade_meta["side"]

    if side == "yes":
        target = entry_price + gb_delta
        sl_price = entry_price - sl_delta
    else:
        target = entry_price - gb_delta
        sl_price = entry_price + sl_delta

    n_prices = trade_meta.get("n_prices_total", len(trade_group))
    max_hold_idx = int(n_prices * max_hold_fraction)

    # Pre-compute DMatrix for all ticks (batch predict for efficiency)
    X = trade_group[FEATURE_COLUMNS].astype("float64").values
    dmat = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    pred_log_t = model.predict(dmat)

    for idx, (_, row) in enumerate(trade_group.iterrows()):
        t = int(row["t"])
        price = float(row["price_now"])

        # 1. Safety net: hard GB / SL barriers
        if side == "yes":
            gb_hit = price >= target
            sl_hit = price <= sl_price
        else:
            gb_hit = price <= target
            sl_hit = price >= sl_price

        if gb_hit:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "green_book", pnl_per_share
        if sl_hit:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "stop_loss", pnl_per_share

        # 2. Learned early-exit signal
        unrealized_profit = (
            (price - entry_price) if side == "yes" else (entry_price - price)
        )
        if unrealized_profit > 0:
            if policy_type == "raw_log_t":
                # Exit when predicted log(T) > threshold (event far)
                if pred_log_t[idx] > threshold_log_t:
                    return t, price, "learned_early", unrealized_profit
            elif policy_type == "hazard":
                # Exit when P(event in next horizon_k) < threshold
                p_event = _aft_hazard_next_k(
                    pred_log_t[idx], horizon_k, aft_scale,
                )
                if p_event < threshold_log_t:
                    return t, price, "learned_early", unrealized_profit

        # 3. Time exit
        if t >= max_hold_idx:
            pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
            return t, price, "time_exit", pnl_per_share

    # Fell through
    last_row = trade_group.iloc[-1]
    t = int(last_row["t"])
    price = float(last_row["price_now"])
    pnl_per_share = (price - entry_price) if side == "yes" else (entry_price - price)
    return t, price, "hold_to_end", pnl_per_share


def _record_trade_result(
    policy: PolicyResult,
    t: int,
    exit_price: float,
    reason: str,
    pnl_per_share: float,
    n_contracts: float,
    max_price_seen: float,
    min_price_seen: float,
    entry_price: float,
    side: str,
) -> None:
    """Aggregate a single-trade result into the PolicyResult."""
    policy.n_trades += 1
    pnl_usd = pnl_per_share * n_contracts
    policy.total_pnl += pnl_usd
    policy.pnl_series.append(pnl_usd)
    if pnl_usd > 0:
        policy.wins += 1
    else:
        policy.losses += 1
    policy.hold_ticks_total += t

    if reason == "green_book":
        policy.gb_exits += 1
    elif reason == "stop_loss":
        policy.sl_exits += 1
    elif reason == "time_exit":
        policy.time_exits += 1
    elif reason == "learned_early":
        policy.early_exits += 1
    elif reason == "hold_to_end":
        policy.hold_to_end_exits += 1

    # Max upside captured (what could we have gotten?)
    if side == "yes":
        max_possible = max_price_seen - entry_price
    else:
        max_possible = entry_price - min_price_seen
    captured = max(pnl_per_share, 0.0)
    missed = max_possible - captured if max_possible > 0 else 0.0
    policy.max_upside_captured += captured
    policy.missed_upside_total += missed


# ── Main eval loop ────────────────────────────────────────────────────


def evaluate(
    ets_path: Path,
    model_path: Path,
    split: str = "test",
    threshold_log_t: float | None = None,
    gb_delta: float = 0.15,
    sl_delta: float = 0.15,
    max_hold_fraction: float = 0.50,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    n_contracts: float = 30.0,  # Nominal size for PnL scaling
) -> dict:
    """Run head-to-head eval of fixed vs learned policy."""
    import pandas as pd
    import xgboost as xgb

    # Load data
    print(f"Loading ETS: {ets_path}")
    df = pd.read_parquet(ets_path)
    print(f"  {len(df)} rows, {df['trade_id'].nunique()} unique trades")

    # Temporal split by game_date (match train_exit_model.py)
    trade_dates = df.groupby("trade_id")["game_date"].first().sort_values()
    n = len(trade_dates)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    if split == "train":
        chosen_ids = set(trade_dates.iloc[:n_train].index)
    elif split == "val":
        chosen_ids = set(trade_dates.iloc[n_train:n_train + n_val].index)
    elif split == "test":
        chosen_ids = set(trade_dates.iloc[n_train + n_val:].index)
    elif split == "all":
        chosen_ids = set(trade_dates.index)
    else:
        raise ValueError(f"Unknown split: {split}")

    eval_df = df[df["trade_id"].isin(chosen_ids)].sort_values(["trade_id", "t"])
    print(f"\nEvaluating on split={split}: {eval_df['trade_id'].nunique()} trades, {len(eval_df)} rows")

    if eval_df.empty:
        print("ERROR: no trades in this split")
        return {}

    # Load model
    print(f"Loading model: {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Tune threshold on val set if not provided — use QUANTILES of actual
    # pred_log_t distribution (scale varies widely with AFT Weibull).
    if threshold_log_t is None and split == "test":
        print("\nTuning learned-policy threshold on VAL split...")
        val_ids = set(trade_dates.iloc[n_train:n_train + n_val].index)
        val_df = df[df["trade_id"].isin(val_ids)].sort_values(["trade_id", "t"])

        # Compute predictions on val split to determine threshold grid
        X_val = val_df[FEATURE_COLUMNS].astype("float64").values
        dmat_val = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
        pred_val = booster.predict(dmat_val)

        # Use distribution quantiles as candidate thresholds
        import numpy as np
        quantiles = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.99]
        candidates = [float(np.quantile(pred_val, q)) for q in quantiles]
        candidates = sorted(set(candidates))  # dedupe
        print(f"  pred_log_t distribution: min={pred_val.min():.1f} max={pred_val.max():.1f} median={np.median(pred_val):.1f}")
        print(f"  Candidate thresholds (quantiles): {[f'{c:.1f}' for c in candidates]}")

        best_thr = None
        best_pnl = -float("inf")
        for thr in candidates:
            pnl = 0.0
            for trade_id, group in val_df.groupby("trade_id"):
                meta = group.iloc[0].to_dict()
                t, p, reason, pps = _simulate_learned_policy(
                    group, meta, booster, thr,
                    gb_delta, sl_delta, max_hold_fraction,
                )
                pnl += pps * n_contracts
            print(f"  threshold={thr:>10.1f}: PnL=${pnl:+.2f}")
            if pnl > best_pnl:
                best_pnl = pnl
                best_thr = thr
        threshold_log_t = best_thr
        print(f"  Best threshold: {threshold_log_t:.1f}  (val PnL=${best_pnl:+.2f})")
    elif threshold_log_t is None:
        threshold_log_t = 2.5  # safe default

    # Run both policies on eval split
    print(f"\nRunning head-to-head on {split} split with threshold={threshold_log_t:.2f}...")
    baseline = PolicyResult(name="fixed (champion)")
    learned = PolicyResult(name=f"learned (thr={threshold_log_t:.2f})")

    for trade_id, group in eval_df.groupby("trade_id"):
        meta = group.iloc[0].to_dict()
        max_price_seen = float(group["price_now"].max())
        min_price_seen = float(group["price_now"].min())
        entry_price = float(meta["entry_price_level"])
        side = meta["side"]

        # A: fixed
        t_a, px_a, reason_a, pps_a = _simulate_fixed_policy(
            group, meta, gb_delta, sl_delta, max_hold_fraction,
        )
        _record_trade_result(
            baseline, t_a, px_a, reason_a, pps_a,
            n_contracts, max_price_seen, min_price_seen, entry_price, side,
        )

        # B: learned
        t_b, px_b, reason_b, pps_b = _simulate_learned_policy(
            group, meta, booster, threshold_log_t,
            gb_delta, sl_delta, max_hold_fraction,
        )
        _record_trade_result(
            learned, t_b, px_b, reason_b, pps_b,
            n_contracts, max_price_seen, min_price_seen, entry_price, side,
        )

    # Report
    print(f"\n{'═' * 72}")
    print(f" HEAD-TO-HEAD RESULTS — split={split}")
    print(f"{'═' * 72}")
    print(f"{'Metric':<26} {'Baseline (fixed)':>20} {'Learned':>20}")
    print("-" * 72)
    rows = [
        ("n_trades",         baseline.n_trades, learned.n_trades),
        ("total_pnl_usd",    f"${baseline.total_pnl:.2f}", f"${learned.total_pnl:.2f}"),
        ("win_rate",         f"{baseline.win_rate:.3f}", f"{learned.win_rate:.3f}"),
        ("sharpe_per_trade", f"{baseline.sharpe:.3f}", f"{learned.sharpe:.3f}"),
        ("max_drawdown",     f"{baseline.max_drawdown:.3f}", f"{learned.max_drawdown:.3f}"),
        ("avg_hold_ticks",   f"{baseline.avg_hold_ticks:.1f}", f"{learned.avg_hold_ticks:.1f}"),
        ("gb_exits",         baseline.gb_exits, learned.gb_exits),
        ("sl_exits",         baseline.sl_exits, learned.sl_exits),
        ("time_exits",       baseline.time_exits, learned.time_exits),
        ("early_exits",      baseline.early_exits, learned.early_exits),
        ("hold_to_end",      baseline.hold_to_end_exits, learned.hold_to_end_exits),
        ("captured_upside",  f"${baseline.max_upside_captured:.2f}", f"${learned.max_upside_captured:.2f}"),
        ("missed_upside",    f"${baseline.missed_upside_total:.2f}", f"${learned.missed_upside_total:.2f}"),
    ]
    for label, a, b in rows:
        print(f"{label:<26} {str(a):>20} {str(b):>20}")

    # Verdict
    print(f"\n{'─' * 72}")
    delta_pnl = learned.total_pnl - baseline.total_pnl
    delta_pct = 100 * delta_pnl / abs(baseline.total_pnl) if baseline.total_pnl != 0 else 0.0
    print(f"PnL delta: ${delta_pnl:+.2f}  ({delta_pct:+.1f}% vs baseline)")
    if delta_pnl > 0 and learned.max_drawdown <= baseline.max_drawdown * 1.20:
        print("✓ LEARNED BEATS BASELINE on PnL + DD within 20%.")
    elif delta_pnl > 0:
        print("⚠ PnL improved but DD worsened > 20% — consider rejecting.")
    else:
        print("✗ LEARNED DOES NOT beat baseline. Do NOT promote.")

    return {
        "split": split,
        "threshold_log_t": threshold_log_t,
        "baseline": {
            k: v for k, v in asdict(baseline).items() if k != "pnl_series"
        } | {
            "win_rate": baseline.win_rate,
            "sharpe": baseline.sharpe,
            "max_drawdown": baseline.max_drawdown,
            "avg_hold_ticks": baseline.avg_hold_ticks,
        },
        "learned": {
            k: v for k, v in asdict(learned).items() if k != "pnl_series"
        } | {
            "win_rate": learned.win_rate,
            "sharpe": learned.sharpe,
            "max_drawdown": learned.max_drawdown,
            "avg_hold_ticks": learned.avg_hold_ticks,
        },
        "delta_pnl": delta_pnl,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ets", type=Path, default=DEFAULT_ETS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="test")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold on predicted log(T). Auto-tuned on val if not set.")
    parser.add_argument("--gb-delta", type=float, default=0.15)
    parser.add_argument("--sl-delta", type=float, default=0.15)
    parser.add_argument("--max-hold-fraction", type=float, default=0.50)
    parser.add_argument("--n-contracts", type=float, default=30.0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results JSON here")
    args = parser.parse_args()

    for p in [args.ets, args.model]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    results = evaluate(
        ets_path=args.ets,
        model_path=args.model,
        split=args.split,
        threshold_log_t=args.threshold,
        gb_delta=args.gb_delta,
        sl_delta=args.sl_delta,
        max_hold_fraction=args.max_hold_fraction,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_contracts=args.n_contracts,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved: {args.output}")


if __name__ == "__main__":
    main()
