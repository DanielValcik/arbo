"""Optuna TPE Bayesian Optimization sweep — Phase 3.2.

Generic wrapper that replaces the brute-force grid sweeps in
research/innovations/sweep_*.py. Runs Optuna's TPE sampler over a
declarative param search space, scoring each trial against shadow +
composite-reward data already in the database.

Usage:
    python research/bo_sweep.py --strategy B3_15M --n-trials 200 \
        --metric composite_reward --window-days 14

Requires:
    pip install optuna>=3.5

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 3.2 + §11 (Rapid Mode)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from typing import Any

# Allow running as a script from repo root
sys.path.insert(0, ".")


# Per-strategy search space. Keep modest — Optuna TPE shines with 5-15 dims.
SEARCH_SPACES: dict[str, dict[str, tuple[str, float, float]]] = {
    "B3": {
        "LIVE_MIN_EDGE":       ("float", 0.20, 0.55),
        "LIVE_MIN_BTC_MOVE":   ("float", 25.0, 60.0),
        "LIVE_MAX_VELOCITY":   ("float", 30.0, 90.0),
        "LIVE_MAX_DIR_DELTA":  ("float", 5.0, 25.0),
        "LIVE_MAX_FILL_PRICE": ("float", 0.55, 0.90),
    },
    "B3_15M": {
        "LIVE_MIN_EDGE":       ("float", 0.20, 0.55),
        "LIVE_MAX_BTC_MOVE":   ("float", 50.0, 150.0),
        "LIVE_MAX_MARKET_GAP": ("float", 0.10, 0.40),
        "LIVE_MAX_FILL_PRICE": ("float", 0.55, 1.05),
    },
    # Project PARALLEL extension
    "B2": {
        "MIN_EDGE":              ("float", 0.05, 0.20),
        "MIN_PRICE":             ("float", 0.03, 0.10),
        "MAX_PRICE":             ("float", 0.40, 0.75),
        "MIN_TIME_TO_EXPIRY_H":  ("float", 4.0, 36.0),
        "MAX_TIME_TO_EXPIRY_H":  ("float", 96.0, 240.0),
    },
    "D": {
        "MIN_EDGE":          ("float", 0.10, 0.25),
        "MAX_EDGE":          ("float", 0.20, 0.40),
        "MIN_PRICE":         ("float", 0.15, 0.35),
        "MAX_PRICE":         ("float", 0.55, 0.80),
        "GREEN_BOOK_DELTA":  ("float", 0.10, 0.25),
        "STOP_LOSS_DELTA":   ("float", 0.08, 0.20),
        "MAX_HOLD_FRACTION": ("float", 0.30, 0.80),
    },
}


async def _load_signals(strategy: str, window_days: int) -> list[dict[str, Any]]:
    """Load resolved shadow signals (any variant) for replay-scoring."""
    from datetime import datetime, timedelta, timezone
    from arbo.utils.db import get_session_factory
    import sqlalchemy as sa

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
    factory = get_session_factory()
    rows: list[dict[str, Any]] = []
    async with factory() as session:
        # Distinct candidates: each (condition_id, signal_ts) once with
        # uniform features across variants. Pick one variant arbitrarily.
        result = await session.execute(
            sa.text("""
                SELECT DISTINCT ON (condition_id, signal_ts)
                    condition_id, signal_ts, direction, edge, btc_move,
                    market_gap, velocity, dir_delta, would_fill_at,
                    resolution_outcome, entry_price, model_prob, meta_json
                FROM shadow_variant_signals
                WHERE strategy = :s
                  AND signal_ts >= :cutoff
                  AND resolution_outcome IS NOT NULL
                  AND would_fill_at IS NOT NULL
                  AND would_fill_at > 0
                ORDER BY condition_id, signal_ts, variant_id
            """),
            {"s": strategy, "cutoff": cutoff},
        )
        for r in result.mappings():
            rows.append(dict(r))
    return rows


def _evaluate_params(
    params: dict[str, float], signals: list[dict[str, Any]], strategy: str,
) -> dict[str, float]:
    """Replay all signals through these gate params; return aggregate metrics."""
    n_qual = 0
    n_wins = 0
    pnl_sum = 0.0
    pnls: list[float] = []

    if strategy == "B3":
        for s in signals:
            edge = float(s["edge"] or 0)
            btc_move = abs(float(s["btc_move"] or 0))
            vel = float(s["velocity"] or 0)
            dd = abs(float(s["dir_delta"] or 0))
            fill = float(s["would_fill_at"] or 0)
            if edge < params["LIVE_MIN_EDGE"]: continue
            if btc_move < params["LIVE_MIN_BTC_MOVE"]: continue
            if vel > params["LIVE_MAX_VELOCITY"]: continue
            if dd > params["LIVE_MAX_DIR_DELTA"]: continue
            if fill > params["LIVE_MAX_FILL_PRICE"]: continue
            up_won = bool(s["resolution_outcome"])
            won = (s["direction"] == "up" and up_won) or (s["direction"] == "down" and not up_won)
            pnl = (1.0 - fill) if won else -fill
            n_qual += 1
            if won: n_wins += 1
            pnl_sum += pnl
            pnls.append(pnl)
    elif strategy == "B3_15M":
        for s in signals:
            edge = float(s["edge"] or 0)
            btc_move = abs(float(s["btc_move"] or 0))
            gap = float(s["market_gap"] or 0)
            fill = float(s["would_fill_at"] or 0)
            if edge < params["LIVE_MIN_EDGE"]: continue
            if btc_move > params["LIVE_MAX_BTC_MOVE"]: continue
            if gap > params["LIVE_MAX_MARKET_GAP"]: continue
            if fill > params["LIVE_MAX_FILL_PRICE"]: continue
            up_won = bool(s["resolution_outcome"])
            won = (s["direction"] == "up" and up_won) or (s["direction"] == "down" and not up_won)
            pnl = (1.0 - fill) if won else -fill
            n_qual += 1
            if won: n_wins += 1
            pnl_sum += pnl
            pnls.append(pnl)
    elif strategy == "B2":
        # Features: edge (signed), entry_price, would_fill_at, sigma,
        # event_end_ts → derive hours_to_expiry approximately from
        # signal_ts diff (already in seconds since epoch in row).
        # For simplicity, hours_to_expiry stored in meta_json.hours_to_expiry
        # (set at write time); fallback to 24h if missing.
        import json
        for s in signals:
            edge_abs = abs(float(s["edge"] or 0))
            price = float(s["entry_price"] or 0)
            fill = float(s["would_fill_at"] or 0)
            try:
                meta = json.loads(s.get("meta_json") or "{}") if isinstance(s.get("meta_json"), str) else (s.get("meta_json") or {})
                hours = float(meta.get("hours_to_expiry", 24.0))
            except Exception:
                hours = 24.0
            if edge_abs < params["MIN_EDGE"]: continue
            if price < params["MIN_PRICE"]: continue
            if price > params["MAX_PRICE"]: continue
            if hours < params["MIN_TIME_TO_EXPIRY_H"]: continue
            if hours > params["MAX_TIME_TO_EXPIRY_H"]: continue
            yes_won = bool(s["resolution_outcome"])
            won = (s["direction"] == "above" and yes_won) or (s["direction"] == "below" and not yes_won)
            pnl = (1.0 - fill) if won else -fill
            n_qual += 1
            if won: n_wins += 1
            pnl_sum += pnl
            pnls.append(pnl)
    elif strategy == "D":
        for s in signals:
            edge = float(s["edge"] or 0)
            price = float(s["entry_price"] or 0)
            fill = float(s["would_fill_at"] or 0)
            if edge < params["MIN_EDGE"]: continue
            if edge > params["MAX_EDGE"]: continue
            if price < params["MIN_PRICE"]: continue
            if price > params["MAX_PRICE"]: continue
            # Note: GREEN_BOOK_DELTA + STOP_LOSS_DELTA + MAX_HOLD_FRACTION
            # affect EXIT, not entry — replay here uses resolution_outcome
            # only (terminal PnL). Exit-aware replay would need mid-trade
            # price history (out of scope for first BO iteration).
            yes_won = bool(s["resolution_outcome"])
            won = (s["direction"] == "yes" and yes_won) or (s["direction"] == "no" and not yes_won)
            pnl = (1.0 - fill) if won else -fill
            n_qual += 1
            if won: n_wins += 1
            pnl_sum += pnl
            pnls.append(pnl)
    else:
        return {"n_qual": 0, "wr": 0, "pnl": 0, "sharpe": 0, "score": 0}

    if n_qual == 0:
        return {"n_qual": 0, "wr": 0, "pnl": 0, "sharpe": 0, "score": -1.0}

    wr = n_wins / n_qual
    avg = pnl_sum / n_qual
    var = sum((p - avg) ** 2 for p in pnls) / max(n_qual - 1, 1)
    sd = math.sqrt(var) if var > 0 else 0
    sharpe = avg / sd if sd > 0 else 0

    # Composite score: balance Sharpe + volume penalty for over-tightening
    # Penalize n_qual < 30 (statistical insufficiency)
    volume_penalty = 1.0 if n_qual >= 30 else max(0.3, n_qual / 30.0)
    score = sharpe * volume_penalty

    return {
        "n_qual": n_qual,
        "wr": round(100 * wr, 1),
        "pnl": round(pnl_sum, 4),
        "sharpe": round(sharpe, 3),
        "score": round(score, 4),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["B3", "B3_15M", "B2", "D"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--out", default=None,
                        help="Output YAML path; auto if omitted")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: pip install optuna>=3.5", file=sys.stderr)
        return 1

    strategy = args.strategy
    space = SEARCH_SPACES.get(strategy)
    if space is None:
        print(f"ERROR: no search space for {strategy}", file=sys.stderr)
        return 1

    print(f"Loading signals for {strategy} (window={args.window_days}d)...")
    signals = await _load_signals(strategy, args.window_days)
    print(f"Loaded {len(signals)} resolved signals.")
    if len(signals) < 30:
        print(f"WARNING: only {len(signals)} signals — Optuna BO may overfit.")

    # Tuning history from concurrent runs is OK — Optuna handles it
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: Any) -> float:
        p: dict[str, float] = {}
        for name, (kind, lo, hi) in space.items():
            if kind == "float":
                p[name] = trial.suggest_float(name, lo, hi)
            elif kind == "int":
                p[name] = trial.suggest_int(name, int(lo), int(hi))
        m = _evaluate_params(p, signals, strategy)
        trial.set_user_attr("metrics", m)
        return float(m["score"])

    print(f"Running Optuna TPE: {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = study.best_trial
    metrics = best.user_attrs.get("metrics", {})
    print()
    print(f"Best score: {best.value}")
    print(f"Best params: {best.params}")
    print(f"Metrics: {metrics}")

    out_yaml = {
        "strategy": strategy,
        "method": "optuna_tpe",
        "n_trials": args.n_trials,
        "window_days": args.window_days,
        "n_signals": len(signals),
        "best_score": best.value,
        "best_params": best.params,
        "best_metrics": metrics,
        "all_top10": [
            {
                "params": t.params,
                "score": t.value,
                "metrics": t.user_attrs.get("metrics", {}),
            }
            for t in sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:10]
        ],
    }
    out_path = args.out or f"research/bo_sweep_{strategy.lower()}_result.json"
    with open(out_path, "w") as f:
        json.dump(out_yaml, f, indent=2, default=str)
    print(f"Result written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
