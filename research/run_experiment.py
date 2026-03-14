"""Single experiment runner for autoresearch.

Takes JSON params as argument, runs the chronological portfolio simulator,
and prints greppable results.

Usage:
    python3 research/run_experiment.py '{"min_edge": 0.08, ...}'
    python3 research/run_experiment.py --file params.json
    python3 research/run_experiment.py --baseline
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_framework import (
    experiment_score,
    load_forecasts,
    preload_data,
    simulate_portfolio,
    walk_forward_validate,
)
from price_history_db import PriceHistoryDB

TRAIN_END = "2026-01-31"

# Current production baseline
BASELINE_PARAMS = {
    "min_edge": 0.10,
    "max_edge": 0.70,
    "max_price": 0.50,
    "min_price": 0.05,
    "min_prob": 0.06,
    "min_volume": 510,
    "kelly_raw_cap": 0.40,
    "prob_sharpening": 0.90,
    "shrinkage": 0.03,
    "excluded_cities": ["chicago", "seoul"],
    "city_overrides": {
        "seattle": {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.15},
        "toronto": {"max_price": 0.4, "min_edge": 0.005, "min_price": 0.05},
        "atlanta": {"max_price": 0.55, "min_edge": 0.02, "min_price": 0.05},
        "ankara": {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.08},
        "buenos_aires": {"max_price": 0.7, "min_edge": 0.02, "min_price": 0.05},
        "dallas": {"max_price": 0.8, "min_edge": 0.05, "min_price": 0.05},
        "wellington": {"max_price": 0.5, "min_edge": 0.02, "min_price": 0.05},
        "miami": {"max_price": 0.4, "min_edge": 0.05, "min_price": 0.08},
    },
    "exit_enabled": True,
    "min_hold_edge": 0.0,
    "prob_exit_floor": 0.25,
    "profit_take_enabled": True,
    "profit_take_threshold": 1.5,
    "profit_take_min_hours": 4,
    "reentry_enabled": False,
    "reentry_cooldown_h": 1,
    "stop_loss_pct": 1.0,
}

# Cache for preloaded data (module-level singleton)
_sim_cache = {}


def get_sim_data():
    """Load and cache simulation data."""
    if "sim" not in _sim_cache:
        db = PriceHistoryDB()
        events = db.get_events(with_prices=True)
        forecasts = load_forecasts(events)
        sim_all = preload_data(db, events, forecasts)
        sim_train = sim_all.filter_events(max_date=TRAIN_END)
        _sim_cache["sim"] = sim_train
        _sim_cache["sim_all"] = sim_all
        _sim_cache["db"] = db
    return _sim_cache["sim"]


def run(params: dict, entry_hours: float = 24, do_wf: bool = False) -> dict:
    """Run single experiment, return results dict."""
    sim = get_sim_data()

    t0 = time.time()
    result = simulate_portfolio(
        sim, params, entry_hours=entry_hours,
        experiment_id="autoresearch",
        record_equity=False,
    )
    experiment_score(result)
    elapsed = time.time() - t0

    out = {
        "score": result.score,
        "trades": result.trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "roi_pct": result.roi_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe": result.sharpe,
        "capital_utilization": result.capital_utilization,
        "avg_pnl_per_hour": result.avg_pnl_per_hour,
        "total_exits": result.total_exits,
        "exit_saves": result.exit_saves,
        "exit_regrets": result.exit_regrets,
        "concurrent_positions": result.concurrent_positions,
        "avg_price": result.avg_price,
        "avg_edge": result.avg_edge,
        "elapsed_s": round(elapsed, 2),
    }

    if do_wf:
        wf = walk_forward_validate(sim, params, entry_hours=entry_hours, n_folds=3)
        out["wf_oos_pnl"] = wf["oos_pnl"]
        out["wf_oos_score"] = wf["score"]
        out["wf_folds"] = wf.get("folds", [])

    return out


def print_results(out: dict):
    """Print greppable results."""
    print("---")
    for key in [
        "score", "trades", "win_rate", "total_pnl", "roi_pct",
        "max_drawdown_pct", "sharpe", "capital_utilization",
        "avg_pnl_per_hour", "total_exits", "exit_saves", "exit_regrets",
        "concurrent_positions", "avg_price", "avg_edge", "elapsed_s",
    ]:
        val = out.get(key, 0)
        if isinstance(val, float):
            print(f"{key}: {val:.2f}")
        else:
            print(f"{key}: {val}")
    if "wf_oos_pnl" in out:
        print(f"wf_oos_pnl: {out['wf_oos_pnl']:.2f}")
        print(f"wf_oos_score: {out['wf_oos_score']:.2f}")
    print("---")


def main():
    if "--baseline" in sys.argv:
        params = BASELINE_PARAMS
        print("Running baseline params...", file=sys.stderr)
    elif "--file" in sys.argv:
        idx = sys.argv.index("--file")
        with open(sys.argv[idx + 1]) as f:
            params = json.load(f)
    elif len(sys.argv) > 1 and sys.argv[1].startswith("{"):
        params = json.loads(sys.argv[1])
    else:
        print("Usage: python3 run_experiment.py '{...}' | --baseline | --file params.json",
              file=sys.stderr)
        sys.exit(1)

    # Convert excluded_cities list to set
    if isinstance(params.get("excluded_cities"), list):
        params["excluded_cities"] = set(params["excluded_cities"])

    entry_hours = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24
    do_wf = "--wf" in sys.argv

    out = run(params, entry_hours=entry_hours, do_wf=do_wf)
    print_results(out)


if __name__ == "__main__":
    main()
