"""Paper Validation — Compare backtest results vs real paper trades.

Loads exported paper trades and runs the same parameters through the
experiment framework to see how well backtest predicts reality.

Output: confidence score (0-100) and detailed comparison report.

Usage:
    python3 research/paper_validation.py
    python3 research/paper_validation.py --sweep research/data/experiments/sweep_20260314_1841.json
    python3 research/paper_validation.py --params '{"min_edge": 0.08, ...}'
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).parent))
from experiment_framework import (
    ExperimentResult,
    SimulationData,
    experiment_score,
    preload_data,
    simulate_portfolio,
)
from price_history_db import PriceHistoryDB

PAPER_TRADES_FILE = Path(__file__).parent / "data" / "paper_trades_export.json"
DB_PATH = Path(__file__).parent / "data" / "price_history.sqlite"


def load_paper_trades(path: Path = PAPER_TRADES_FILE) -> list[dict]:
    """Load exported paper trades."""
    if not path.exists():
        print(f"Paper trades not found: {path}")
        print("Run: python3 research/export_paper_trades.py")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    print(f"Loaded {len(trades)} paper trades")

    meta = data.get("meta", {})
    if meta:
        print(f"  Exported: {meta.get('exported_at', '?')}")
        print(f"  Source: {meta.get('source', '?')}")

    return trades


def paper_trade_stats(trades: list[dict]) -> dict:
    """Compute aggregate stats from paper trades."""
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    if not pnls:
        return {"count": 0}

    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = 100 * wins / len(pnls) if pnls else 0

    return {
        "count": len(pnls),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(pnls),
        "median_pnl": median(pnls),
        "win_rate": win_rate,
        "wins": wins,
        "losses": len(pnls) - wins,
    }


def compute_confidence(
    paper_stats: dict,
    bt_result: ExperimentResult,
) -> dict:
    """Compute confidence score (0-100) from paper vs backtest comparison.

    Components (each 0-20, total 0-100):
    1. Direction match: both positive or both negative PnL
    2. Win rate similarity: |bt_wr - paper_wr| proximity
    3. PnL per trade similarity: absolute difference
    4. Trade count ratio: how close bt trade count matches paper
    5. Edge consistency: are the aggregate stats in same ballpark

    Returns dict with score and breakdown.
    """
    if paper_stats.get("count", 0) < 3:
        return {"score": 0, "reason": "Too few paper trades (< 3)"}

    breakdown = {}

    # 1. Direction match (0-20)
    paper_pnl = paper_stats["total_pnl"]
    bt_pnl = bt_result.total_pnl
    if (paper_pnl > 0) == (bt_pnl > 0):
        breakdown["direction"] = 20
    elif abs(paper_pnl) < 5 or abs(bt_pnl) < 5:
        breakdown["direction"] = 10  # one near zero
    else:
        breakdown["direction"] = 0

    # 2. Win rate similarity (0-20)
    paper_wr = paper_stats["win_rate"]
    bt_wr = bt_result.win_rate
    wr_diff = abs(paper_wr - bt_wr)
    if wr_diff < 5:
        breakdown["win_rate"] = 20
    elif wr_diff < 10:
        breakdown["win_rate"] = 15
    elif wr_diff < 20:
        breakdown["win_rate"] = 10
    elif wr_diff < 30:
        breakdown["win_rate"] = 5
    else:
        breakdown["win_rate"] = 0

    # 3. PnL per trade similarity (0-20)
    paper_avg = paper_stats["avg_pnl"]
    bt_avg = bt_result.avg_pnl_per_trade
    if paper_avg == 0 and bt_avg == 0:
        breakdown["pnl_per_trade"] = 20
    elif paper_avg == 0 or bt_avg == 0:
        breakdown["pnl_per_trade"] = 5
    else:
        ratio = min(paper_avg, bt_avg) / max(paper_avg, bt_avg) if max(abs(paper_avg), abs(bt_avg)) > 0 else 0
        # Same sign check
        if (paper_avg > 0) != (bt_avg > 0):
            ratio = -abs(ratio)
        breakdown["pnl_per_trade"] = max(0, int(20 * max(0, ratio)))

    # 4. Trade count ratio (0-20)
    paper_n = paper_stats["count"]
    bt_n = bt_result.trades
    if paper_n == 0 or bt_n == 0:
        breakdown["trade_count"] = 0
    else:
        count_ratio = min(paper_n, bt_n) / max(paper_n, bt_n)
        breakdown["trade_count"] = int(20 * count_ratio)

    # 5. Edge consistency (0-20) — overall reasonableness
    # Penalize if backtest is much better than paper (overfitting signal)
    if bt_pnl > 0 and paper_pnl > 0:
        # Both profitable: how close
        if bt_result.roi_pct < paper_stats.get("total_pnl", 0) / 10 * 1.5:
            breakdown["edge_consistency"] = 20  # BT within 50% of paper
        else:
            # BT outperforms paper: potential overfitting
            overfit_ratio = bt_result.roi_pct / max(0.1, paper_stats["total_pnl"] / 10)
            breakdown["edge_consistency"] = max(0, int(20 - 5 * (overfit_ratio - 1)))
    elif bt_pnl > 0 and paper_pnl <= 0:
        breakdown["edge_consistency"] = 0  # BT says profit, paper says loss
    elif bt_pnl <= 0 and paper_pnl > 0:
        breakdown["edge_consistency"] = 5  # Conservative BT
    else:
        breakdown["edge_consistency"] = 15  # Both negative, at least consistent

    total = sum(breakdown.values())
    return {
        "score": total,
        "breakdown": breakdown,
        "interpretation": _interpret_score(total),
    }


def _interpret_score(score: int) -> str:
    """Human-readable interpretation of confidence score."""
    if score >= 80:
        return "Vynikajici — backtest dobre predikuje realitu"
    elif score >= 60:
        return "Dobry — merne odchylky, ale smer sedi"
    elif score >= 40:
        return "Prumerny — nektere metriky se rozchazeji"
    elif score >= 20:
        return "Slab — backtest se vyrazne lisi od reality"
    else:
        return "Neduveryhodny — backtest neodpovida realite"


def run_backtest_for_period(
    db: PriceHistoryDB,
    params: dict,
    start_date: str,
    end_date: str,
    entry_hours: float = 24,
) -> ExperimentResult:
    """Run backtest for the same period as paper trades."""
    from experiment_framework import load_forecasts

    events = db.get_events(start_date=start_date, end_date=end_date)
    print(f"  Backtest events: {len(events)} ({start_date} to {end_date})")

    forecasts = load_forecasts(events)
    sim = preload_data(db, events, forecasts)
    result = simulate_portfolio(
        sim, params, entry_hours=entry_hours,
        experiment_id="paper_validation",
        record_equity=False,
    )
    return result


def print_comparison(paper_stats: dict, bt_result: ExperimentResult, confidence: dict) -> None:
    """Print side-by-side comparison."""
    print("\n" + "=" * 60)
    print("  PAPER TRADES vs BACKTEST  ".center(60, "="))
    print("=" * 60)

    def row(label, paper_val, bt_val):
        print(f"  {label:<25} {'Paper':>12}  {'Backtest':>12}")
        print(f"  {'':<25} {str(paper_val):>12}  {str(bt_val):>12}")

    print(f"\n  {'Metrika':<25} {'Paper':>12}  {'Backtest':>12}")
    print(f"  {'-'*25} {'-'*12}  {'-'*12}")

    p = paper_stats
    b = bt_result

    print(f"  {'Trades':<25} {p.get('count', 0):>12}  {b.trades:>12}")
    print(f"  {'Wins':<25} {p.get('wins', 0):>12}  {b.wins:>12}")
    print(f"  {'Win Rate':<25} {p.get('win_rate', 0):>11.1f}%  {b.win_rate:>11.1f}%")
    print(f"  {'Total PnL':<25} ${p.get('total_pnl', 0):>11.2f}  ${b.total_pnl:>11.2f}")
    print(f"  {'Avg PnL/trade':<25} ${p.get('avg_pnl', 0):>11.2f}  ${b.avg_pnl_per_trade:>11.2f}")
    print(f"  {'Median PnL':<25} ${p.get('median_pnl', 0):>11.2f}  ${b.median_pnl:>11.2f}")

    print(f"\n  {'Confidence Score':>25}: {confidence['score']}/100")
    if "breakdown" in confidence:
        for k, v in confidence["breakdown"].items():
            print(f"    {k:<23}: {v}/20")
    print(f"\n  {confidence.get('interpretation', '')}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trade validation")
    parser.add_argument(
        "--paper", default=str(PAPER_TRADES_FILE),
        help="Path to paper trades JSON",
    )
    parser.add_argument(
        "--sweep", default=None,
        help="Path to sweep JSON (uses top config)",
    )
    parser.add_argument(
        "--params", default=None,
        help="JSON string with experiment params",
    )
    parser.add_argument(
        "--entry-hours", type=float, default=24,
        help="Entry timing in hours (default: 24)",
    )
    args = parser.parse_args()

    # Load paper trades
    paper_trades = load_paper_trades(Path(args.paper))
    if not paper_trades:
        print("No paper trades to validate.")
        return

    paper_stats = paper_trade_stats(paper_trades)
    print(f"\nPaper trade summary:")
    print(f"  Trades: {paper_stats['count']}")
    print(f"  Win rate: {paper_stats.get('win_rate', 0):.1f}%")
    print(f"  Total PnL: ${paper_stats.get('total_pnl', 0):.2f}")

    # Get params
    if args.params:
        params = json.loads(args.params)
    elif args.sweep:
        with open(args.sweep) as f:
            sweep = json.load(f)
        top = sweep.get("top_results", sweep.get("all_results", []))
        if not top:
            print("No results in sweep file")
            return
        params = top[0]["params"]
        print(f"\nUsing top config from sweep: {top[0].get('experiment_id', '?')}")
        print(f"  Score: {top[0].get('score', '?')}")
    else:
        # Default production params
        params = {
            "min_edge": 0.08,
            "max_edge": 0.42,
            "max_price": 0.43,
            "min_price": 0.30,
            "min_prob": 0.62,
            "min_volume": 1000,
            "kelly_raw_cap": 0.40,
            "prob_sharpening": 1.05,
            "shrinkage": 0.03,
            "exit_enabled": True,
            "min_hold_edge": 0.05,
        }
        print("\nUsing default production params")

    # Determine paper trade period
    dates = []
    for t in paper_trades:
        if t.get("placed_at"):
            dates.append(t["placed_at"][:10])
        if t.get("resolved_at"):
            dates.append(t["resolved_at"][:10])

    if not dates:
        print("Cannot determine paper trade period (no dates)")
        return

    start_date = min(dates)
    end_date = max(dates)
    print(f"\nPaper trade period: {start_date} to {end_date}")

    # Run backtest for same period
    print("\nRunning backtest...")
    db = PriceHistoryDB(str(DB_PATH))
    bt_result = run_backtest_for_period(
        db, params, start_date, end_date,
        entry_hours=args.entry_hours,
    )

    # Compute confidence
    confidence = compute_confidence(paper_stats, bt_result)

    # Print comparison
    print_comparison(paper_stats, bt_result, confidence)

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_period": {"start": start_date, "end": end_date},
        "params": params,
        "paper_stats": paper_stats,
        "backtest_stats": {
            "trades": bt_result.trades,
            "wins": bt_result.wins,
            "win_rate": bt_result.win_rate,
            "total_pnl": bt_result.total_pnl,
            "avg_pnl_per_trade": bt_result.avg_pnl_per_trade,
            "median_pnl": bt_result.median_pnl,
            "roi_pct": bt_result.roi_pct,
            "sharpe": bt_result.sharpe,
            "max_drawdown_pct": bt_result.max_drawdown_pct,
        },
        "confidence": confidence,
    }

    report_path = Path(__file__).parent / "data" / "paper_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
