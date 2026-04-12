"""Strategy D — Parameter Sweep (Autoresearch).

Systematic search over key parameters using fast cache backtest.
Each experiment: ~10-30 seconds.

Sweeps: min_edge × green_book_delta × stop_loss × price_range × sizing
~500 experiments → ~3-4 hours.

Usage:
    PYTHONPATH=. python3 research_d/sweep_strategy_d.py
    PYTHONPATH=. python3 research_d/sweep_strategy_d.py --max-experiments 100
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use streaming prepare.py (works on 291GB DB directly)
from research_d.prepare import evaluate, DEFAULT_DB_PATH

RESULTS_FILE = Path(__file__).parent / "data" / "sweep_d_results.tsv"


def build_param_grid() -> list[dict]:
    """Generate parameter combinations to test."""
    base = {
        "initial_capital": 1000.0,
        "elo_weight": 0.40,
        "pinnacle_weight": 0.60,
        "elo_only_weight_elo": 0.45,
        "elo_only_weight_glicko": 0.55,
        "green_book_enabled": True,
        "green_book_delta_epl": 0.10,
        "green_book_delta_nfl": 0.08,
        "green_book_delta_ufc": 0.12,
        "green_book_delta_default": 0.08,
        "min_volume": 0,
        "min_prices": 10,
        "competitive_threshold": 999,
        "excluded_teams": [],
        "wf_train_months": 3,
        "wf_test_months": 1,
    }

    # SWEEP v4 — BOTH SIDES + ALWAYS CLOSE
    # both_sides=True confirmed +33% trades, +70% PnL, lower DD
    # SL=0.15 fixed (confirmed better with time exit)
    combos = []
    for me in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for gbd in [0.13, 0.15, 0.17, 0.20]:
            for mhf in [0.50, 0.60, 0.70, 0.80]:
                params = dict(base)
                params.update({
                    "min_edge": me,
                    "max_edge": 0.25,
                    "green_book_delta_nba": gbd,
                    "stop_loss_enabled": True,
                    "stop_loss_delta": 0.15,
                    "min_price": 0.20,
                    "max_price": 0.65,
                    "kelly_fraction": 0.15,
                    "kelly_raw_cap": 0.10,
                    "max_position_pct": 0.03,
                    "enabled_sports": ["nba"],
                    "max_hold_fraction": mhf,
                    "both_sides": True,
                })
                combos.append(params)

    return combos


def run_sweep(max_experiments: int = 500, worker: int = 0, n_workers: int = 1) -> None:
    grid = build_param_grid()
    print(f"Total parameter combinations: {len(grid)}")

    # Sample if too many
    if len(grid) > max_experiments:
        import random
        random.seed(42)
        grid = random.sample(grid, max_experiments)
        print(f"Sampled {max_experiments} experiments")

    # Shard for parallel workers
    if n_workers > 1 and worker > 0:
        grid = [g for i, g in enumerate(grid) if (i % n_workers) == (worker - 1)]
        print(f"Worker {worker}/{n_workers}: {len(grid)} experiments")

    # Resume from existing results
    done_hashes: set[str] = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                done_hashes.add(row.get("param_hash", ""))
        print(f"Resuming: {len(done_hashes)} experiments already done")

    # Write header if new
    write_header = not RESULTS_FILE.exists()
    results_fd = open(RESULTS_FILE, "a", newline="")
    writer = csv.writer(results_fd, delimiter="\t")
    if write_header:
        writer.writerow([
            "experiment", "score", "pnl", "trades", "win_rate", "gb_rate",
            "stop_rate", "sharpe", "max_dd", "pf", "turnover", "avg_edge",
            "min_edge", "gb_delta", "sl_on", "sl_delta", "max_hold_frac",
            "min_price", "max_price", "kelly_f", "kelly_cap", "max_pos",
            "sports", "param_hash",
        ])
        results_fd.flush()

    best_score = -999
    best_params = None
    t0 = time.time()

    for i, params in enumerate(grid):
        # Hash for dedup
        key_parts = [
            params["min_edge"], params["green_book_delta_nba"],
            params["stop_loss_enabled"], params["stop_loss_delta"],
            params.get("max_hold_fraction", 1.0),
            params["min_price"], params["max_price"],
            params["kelly_fraction"], params["max_position_pct"],
            str(params["enabled_sports"]),
        ]
        phash = str(hash(tuple(str(x) for x in key_parts)))
        if phash in done_hashes:
            continue

        # Run experiment
        try:
            results = evaluate(params, DEFAULT_DB_PATH)
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}", flush=True)
            continue

        score = results["score"]
        pnl = results["total_pnl"]
        trades = results["n_trades"]

        # Log
        elapsed = time.time() - t0
        avg_time = elapsed / (i + 1) if i > 0 else 0
        eta = avg_time * (len(grid) - i - 1) / 60

        marker = ""
        if score > best_score and trades >= 30:
            best_score = score
            best_params = params
            marker = " ★ NEW BEST"

        print(
            f"  [{i+1}/{len(grid)}] score={score:.1f} pnl=${pnl:.0f} "
            f"trades={trades} wr={results['win_rate']:.0%} "
            f"gb={results['green_book_rate']:.0%} dd={results['max_drawdown']:.0%} "
            f"| edge={params['min_edge']} delta={params['green_book_delta_nba']} "
            f"sl={'Y' if params['stop_loss_enabled'] else 'N'}"
            f"{marker} (ETA {eta:.0f}min)",
            flush=True,
        )

        # Save
        writer.writerow([
            i + 1, f"{score:.2f}", f"{pnl:.2f}", trades,
            f"{results['win_rate']:.4f}", f"{results['green_book_rate']:.4f}",
            f"{results.get('stop_rate', 0):.4f}", f"{results['sharpe']:.2f}",
            f"{results['max_drawdown']:.4f}", f"{results['profit_factor']:.3f}",
            f"{results['turnover']:.1f}", f"{results['avg_edge']:.4f}",
            params["min_edge"], params["green_book_delta_nba"],
            params["stop_loss_enabled"], params["stop_loss_delta"],
            params.get("max_hold_fraction", 1.0),
            params["min_price"], params["max_price"],
            params["kelly_fraction"], params["kelly_raw_cap"],
            params["max_position_pct"],
            "+".join(params["enabled_sports"]), phash,
        ])
        results_fd.flush()

    results_fd.close()

    total = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"SWEEP COMPLETE ({total/60:.1f} min, {len(grid)} experiments)")
    print(f"{'═'*60}")

    if best_params:
        print(f"\nBest score: {best_score:.1f}")
        print(f"Best params:")
        for k in ["min_edge", "green_book_delta_nba", "stop_loss_enabled",
                   "stop_loss_delta", "min_price", "max_price",
                   "kelly_fraction", "max_position_pct", "enabled_sports"]:
            print(f"  {k}: {best_params[k]}")
    else:
        print("\nNo profitable configuration found with >= 30 trades")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=500)
    parser.add_argument("--worker", type=int, default=0, help="Worker index (0=all, 1-N=specific shard)")
    parser.add_argument("--n-workers", type=int, default=1, help="Total number of workers")
    args = parser.parse_args()
    run_sweep(args.max_experiments, worker=args.worker, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
