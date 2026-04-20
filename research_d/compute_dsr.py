"""Strategy D — Deflated Sharpe Ratio (DSR) analysis of 344 sweep experiments.

Tests whether the "Sharpe 7.03" headline from sweep v4 survives after
deflating for multiple testing + non-normality.

References:
  Bailey, López de Prado (2014) "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality."
  J. Portfolio Management, 40(5).

What it computes:
  PSR (Probabilistic Sharpe Ratio) — P(true SR > 0 | observed SR)
  DSR (Deflated Sharpe Ratio)       — P(true SR > 0 | observed SR, N trials)

What we need but don't have:
  Per-trade returns for each trial → needed for skewness / kurtosis
  adjustment and for PBO (Combinatorially Symmetric CV). The TSV has
  only summary stats, so this script uses normal-return assumption as
  the FIRST approximation. The result is an UPPER BOUND on DSR — if
  this already fails, we know the strategy is overfit without needing
  the full return matrix.

Usage:
  PYTHONPATH=. python3 research_d/compute_dsr.py
  PYTHONPATH=. python3 research_d/compute_dsr.py --tsv research_d/data/sweep_d_results.tsv
  PYTHONPATH=. python3 research_d/compute_dsr.py --all-sweeps    # combine v1-v4

Output:
  Observed Sharpe, expected max under null, PSR, DSR, verdict
  Sensitivity table: rank-k configs with DSR > 0.95 / 0.75 / 0.50
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance

from scipy import stats  # type: ignore

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_TSV = DATA_DIR / "sweep_d_results.tsv"
ALL_SWEEPS = [
    DATA_DIR / "sweep_d_results_v1.tsv",
    DATA_DIR / "sweep_d_results_v2.tsv",
    DATA_DIR / "sweep_d_results_v3.tsv",
    DATA_DIR / "sweep_d_results.tsv",
]

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class Trial:
    experiment: int
    score: float
    pnl: float
    trades: int
    win_rate: float
    sharpe: float  # annualized
    max_dd: float
    min_edge: float
    gb_delta: float
    sl_delta: float
    max_hold_frac: float
    source: str  # which sweep file


def _parse_tsv(path: Path) -> list[Trial]:
    trials: list[Trial] = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                trials.append(
                    Trial(
                        experiment=int(row["experiment"]),
                        score=float(row["score"]),
                        pnl=float(row["pnl"]),
                        trades=int(row["trades"]),
                        win_rate=float(row["win_rate"]),
                        sharpe=float(row["sharpe"]),
                        max_dd=float(row["max_dd"]),
                        min_edge=float(row["min_edge"]),
                        gb_delta=float(row["gb_delta"]),
                        sl_delta=float(row["sl_delta"]),
                        max_hold_frac=float(row.get("max_hold_frac", 1.0)),
                        source=path.stem,
                    )
                )
            except (KeyError, ValueError):
                continue
    return trials


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR_n] under null (i.i.d. trials with equal true SR=0).

    Bailey & López de Prado (2014) eq. 8.
        E[max_n SR_n] ≈ sqrt(V[SR]) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
    where γ = Euler-Mascheroni ≈ 0.5772.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    sr_std = math.sqrt(sr_variance)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def psr(observed_sr: float, benchmark_sr: float, n_obs: int,
        skew: float = 0.0, kurt_excess: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio — P(true SR > benchmark | observed SR).

    Bailey & López de Prado (2012) eq. 2.
        PSR(SR*) = Φ( (SR_hat - SR*) · sqrt(T-1)
                     / sqrt(1 - γ3·SR_hat + (γ4-1)/4·SR_hat²) )
    where γ3 is skewness, γ4 is kurtosis (not excess — note convention).
    We accept excess kurtosis and convert internally.
    """
    if n_obs < 2:
        return float("nan")
    kurt = kurt_excess + 3.0  # convert excess kurtosis to raw kurtosis
    denom_sq = 1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sr(observed_sr: float, sr_variance_across_trials: float,
                n_trials: int, n_obs: int,
                skew: float = 0.0, kurt_excess: float = 0.0) -> tuple[float, float]:
    """DSR = PSR with benchmark = E[max SR under null]."""
    sr_max_null = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return psr(observed_sr, sr_max_null, n_obs, skew, kurt_excess), sr_max_null


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────

def analyze(trials: list[Trial], label: str) -> None:
    print(f"\n{'═'*72}")
    print(f" {label}")
    print(f"{'═'*72}")
    print(f"Trials: {len(trials)}")

    if not trials:
        print("  (empty)")
        return

    sharpes = [t.sharpe for t in trials]
    sr_mean = mean(sharpes)
    sr_var = pvariance(sharpes)
    sr_std = math.sqrt(sr_var) if sr_var > 0 else 0.0

    # Sort by Sharpe descending for rank analysis
    sorted_trials = sorted(trials, key=lambda t: t.sharpe, reverse=True)
    best = sorted_trials[0]

    print(f"\nSharpe distribution:")
    print(f"  mean  : {sr_mean:>7.3f}")
    print(f"  stdev : {sr_std:>7.3f}")
    print(f"  max   : {best.sharpe:>7.3f}   (experiment #{best.experiment} from {best.source})")
    print(f"  min   : {sorted_trials[-1].sharpe:>7.3f}")

    print(f"\nBest config details:")
    print(f"  min_edge={best.min_edge}, gb_delta={best.gb_delta}, sl_delta={best.sl_delta}")
    print(f"  max_hold_frac={best.max_hold_frac}")
    print(f"  trades={best.trades}, WR={best.win_rate:.1%}, DD={best.max_dd:.1%}, PnL=${best.pnl:.0f}")

    # E[max SR_n] under null
    sr_max_null = expected_max_sharpe(len(trials), sr_var)
    print(f"\nMultiple-testing adjustment:")
    print(f"  N trials                : {len(trials)}")
    print(f"  V[SR] across trials     : {sr_var:.4f}")
    print(f"  E[max SR under null]    : {sr_max_null:.3f}")
    print(f"  Observed max - E[max]   : {best.sharpe - sr_max_null:+.3f}  (positive ⇒ survived)")

    # PSR (benchmark = 0, no non-normality adjustment)
    _psr_zero = psr(best.sharpe, 0.0, best.trades, skew=0.0, kurt_excess=0.0)

    # DSR under normal-return assumption (upper bound)
    dsr_normal, _sr_max = deflated_sr(
        best.sharpe, sr_var, len(trials), best.trades,
        skew=0.0, kurt_excess=0.0,
    )

    # DSR with typical non-normality assumption (sports betting returns have
    # fat-tailed losses from blowouts → negative skew, excess kurt).
    # Use conservative estimates until we can measure from actual returns.
    dsr_conservative, _ = deflated_sr(
        best.sharpe, sr_var, len(trials), best.trades,
        skew=-0.5, kurt_excess=2.0,
    )

    print(f"\nSharpe-survival statistics (best config):")
    print(f"  PSR vs 0   (normal)     : {_psr_zero:>7.3%}   P(true SR > 0)")
    print(f"  DSR        (normal)     : {dsr_normal:>7.3%}   upper bound")
    print(f"  DSR        (fat-tail)   : {dsr_conservative:>7.3%}   skew=-0.5, xkurt=+2")

    # Verdict
    print(f"\nVerdict:")
    if dsr_conservative >= 0.95:
        print(f"  ✓ PASS  — DSR > 0.95 even with conservative non-normality. Edge is real.")
    elif dsr_conservative >= 0.75:
        print(f"  ⚠ BORDERLINE — DSR in [0.75, 0.95). Some configs overfit; investigate.")
    elif dsr_conservative >= 0.50:
        print(f"  ⚠ WEAK — DSR in [0.50, 0.75). Most of headline Sharpe is selection bias.")
    else:
        print(f"  ✗ FAIL  — DSR < 0.50. Observed max Sharpe not distinguishable from noise.")

    # Sensitivity: how many configs have DSR > threshold?
    print(f"\nRank sensitivity (how many configs survive deflation):")
    thresholds = [(0.95, "strong"), (0.75, "borderline"), (0.50, "weak")]
    for thr, name in thresholds:
        n_surv = 0
        for t in sorted_trials:
            dsr_t, _ = deflated_sr(
                t.sharpe, sr_var, len(trials), t.trades,
                skew=-0.5, kurt_excess=2.0,
            )
            if dsr_t >= thr:
                n_surv += 1
        frac = n_surv / len(trials)
        print(f"  DSR ≥ {thr:.2f} ({name:>10}): {n_surv:>4}/{len(trials)} ({frac:.1%})")

    # Distribution of Sharpes — quick histogram
    print(f"\nSharpe histogram (width 0.5):")
    if sharpes:
        lo, hi = min(sharpes), max(sharpes)
        bins: dict[float, int] = {}
        for s in sharpes:
            b = math.floor(s * 2) / 2
            bins[b] = bins.get(b, 0) + 1
        for b in sorted(bins.keys()):
            bar = "█" * min(60, bins[b])
            print(f"  {b:>5.1f}..{b+0.5:<5.1f} | {bins[b]:>4} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV,
                        help="Single sweep TSV file (default: v4)")
    parser.add_argument("--all-sweeps", action="store_true",
                        help="Combine v1+v2+v3+v4 (full 344-experiment universe)")
    args = parser.parse_args()

    if args.all_sweeps:
        combined: list[Trial] = []
        for p in ALL_SWEEPS:
            if p.exists():
                combined.extend(_parse_tsv(p))
            else:
                print(f"⚠ missing: {p}")
        analyze(combined, f"COMBINED SWEEPS (v1+v2+v3+v4) — {len(combined)} trials")
    else:
        trials = _parse_tsv(args.tsv)
        analyze(trials, f"SWEEP: {args.tsv.name}")


if __name__ == "__main__":
    main()
