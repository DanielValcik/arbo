"""EPL v2 sweep — Dixon-Coles model parameter search.

Uses prepare_epl_v2.py (DC model with ρ=-0.10).

Since DC model already gave +425% CLV at baseline, sweep focuses on:
  - min_edge (lower might unlock more trades)
  - delta, SL (green book thresholds)
  - max_hold_fraction (time vs resolution exit)
  - rho parameter for Dixon-Coles (test -0.05 to -0.15)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_d.prepare_epl_v2 import DEFAULT_DB_PATH, DEFAULT_PARAMS, evaluate
from research_d import prepare_epl_v2

RESULTS_FILE = Path(__file__).parent / "data" / "sweep_epl_v2_results.tsv"


def build_param_grid():
    base = dict(DEFAULT_PARAMS)
    combos = []
    # Core params sweep (Dixon-Coles ρ fixed to -0.10 as proven EPL value)
    for me in [0.02, 0.03, 0.05, 0.07, 0.10]:
        for gbd in [0.10, 0.15, 0.20, 0.25]:
            for sl_d in [0.15, 0.20, 0.25, 0.30]:
                for mhf in [0.50, 0.70, 1.0]:
                    p = dict(base)
                    p.update({
                        "min_edge": me,
                        "green_book_delta": gbd,
                        "stop_loss_delta": sl_d,
                        "max_hold_fraction": mhf,
                    })
                    combos.append(p)
    return combos


def run_sweep(max_experiments=500, worker=0, n_workers=1):
    grid = build_param_grid()
    print(f"Total combos: {len(grid)}")

    if len(grid) > max_experiments:
        import random
        random.seed(42)
        grid = random.sample(grid, max_experiments)

    if n_workers > 1 and worker > 0:
        grid = [g for i, g in enumerate(grid) if (i % n_workers) == (worker - 1)]
        print(f"Worker {worker}/{n_workers}: {len(grid)} experiments")

    done = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                done.add(row.get("param_hash", ""))
        print(f"Resuming: {len(done)} done")

    write_header = not RESULTS_FILE.exists()
    fd = open(RESULTS_FILE, "a", newline="")
    writer = csv.writer(fd, delimiter="\t")
    if write_header:
        writer.writerow([
            "experiment", "score", "pnl", "trades", "win_rate", "gb_rate",
            "sharpe", "max_dd", "pf", "turnover", "avg_edge", "avg_clv",
            "min_edge", "gb_delta", "sl_d", "mhf", "param_hash",
        ])
        fd.flush()

    best_score = -999
    best = None
    t0 = time.time()

    for i, params in enumerate(grid):
        phash = str(hash((
            params["min_edge"], params["green_book_delta"],
            params["stop_loss_delta"], params["max_hold_fraction"],
        )))
        if phash in done:
            continue

        try:
            r = evaluate(params)
        except Exception as e:
            print(f"[{i+1}] ERR: {e}", flush=True)
            continue

        score = r["score"]
        pnl = r["total_pnl"]
        trades = r["n_trades"]
        clv = r.get("avg_clv", 0) * 100
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(grid) - i - 1) / 60

        marker = ""
        if score > best_score and trades >= 30:
            best_score = score
            best = params
            marker = " ★"

        print(
            f"[{i+1}/{len(grid)}] score={score:.1f} pnl=${pnl:.0f} trades={trades} "
            f"wr={r['win_rate']:.0%} clv={clv:.2f}¢ dd={r['max_drawdown']:.0%} "
            f"| e={params['min_edge']} d={params['green_book_delta']} "
            f"sl={params['stop_loss_delta']} mhf={params['max_hold_fraction']}"
            f"{marker} (ETA {eta:.0f}m)",
            flush=True,
        )

        writer.writerow([
            i + 1, f"{score:.2f}", f"{pnl:.2f}", trades,
            f"{r['win_rate']:.4f}", f"{r['green_book_rate']:.4f}",
            f"{r['sharpe']:.2f}", f"{r['max_drawdown']:.4f}",
            f"{r['profit_factor']:.3f}", f"{r['turnover']:.1f}",
            f"{r['avg_edge']:.4f}", f"{r.get('avg_clv', 0):.4f}",
            params["min_edge"], params["green_book_delta"],
            params["stop_loss_delta"], params["max_hold_fraction"], phash,
        ])
        fd.flush()

    fd.close()
    print(f"\nDone ({(time.time()-t0)/60:.1f}min)")
    if best:
        print(f"BEST score={best_score:.1f}")
        for k in ["min_edge", "green_book_delta", "stop_loss_delta", "max_hold_fraction"]:
            print(f"  {k}: {best[k]}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-experiments", type=int, default=500)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    args = p.parse_args()
    run_sweep(args.max_experiments, args.worker, args.n_workers)


if __name__ == "__main__":
    main()
