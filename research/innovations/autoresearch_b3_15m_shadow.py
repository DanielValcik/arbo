"""Autoresearch grid sweep for B3 15-min filters on 144 real shadow signals.

Loads research/b3_15m_shadow_export.csv (exported from PG b3_15m_shadow_signals)
and searches the filter parameter space with 5-fold time-ordered CV.

Score metric: mean fold PnL/trade × sqrt(mean_N) × (1 - cv_std/cv_mean)
  → rewards per-trade profitability, volume, and fold stability.

Output: top-20 robust configs with per-fold variance.
"""
from __future__ import annotations

import csv
import itertools
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "research" / "b3_15m_shadow_export.csv"

GRID = {
    "min_edge":   [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "min_move":   [0, 15, 20, 25, 30, 40, 50],
    "max_move":   [80, 100, 120, 150, 9999],
    "max_gap":    [0.10, 0.15, 0.20, 0.30, 9.0],
    "min_fill":   [0.00, 0.30, 0.50, 0.65],
    "max_fill":   [0.75, 0.85, 0.95, 1.01],
    "min_minute": [4, 5, 6],
    "max_minute": [9, 10, 11, 12],
}

N_FOLDS = 5
MIN_TRADES_PER_FOLD = 4  # every fold must have at least this many trades


@dataclass
class Signal:
    entry_minute: int
    btc_abs_move: float
    edge: float
    would_fill_at: float
    market_gap: float
    pnl: float
    event_start_ts: int


@dataclass
class Config:
    min_edge: float
    min_move: float
    max_move: float
    max_gap: float
    min_fill: float
    max_fill: float
    min_minute: int
    max_minute: int

    def matches(self, s: Signal) -> bool:
        return (
            s.edge >= self.min_edge
            and s.btc_abs_move >= self.min_move
            and s.btc_abs_move <= self.max_move
            and s.market_gap <= self.max_gap
            and s.would_fill_at >= self.min_fill
            and s.would_fill_at <= self.max_fill
            and s.entry_minute >= self.min_minute
            and s.entry_minute <= self.max_minute
        )


@dataclass
class FoldResult:
    n: int
    total_pnl: float
    avg_pnl: float
    wr: float
    max_dd: float


@dataclass
class Evaluation:
    cfg: Config
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def total_n(self) -> int:
        return sum(f.n for f in self.folds)

    @property
    def total_pnl(self) -> float:
        return sum(f.total_pnl for f in self.folds)

    @property
    def mean_avg_pnl(self) -> float:
        pnls = [f.avg_pnl for f in self.folds if f.n > 0]
        return statistics.fmean(pnls) if pnls else 0.0

    @property
    def std_avg_pnl(self) -> float:
        pnls = [f.avg_pnl for f in self.folds if f.n > 0]
        return statistics.pstdev(pnls) if len(pnls) > 1 else 0.0

    @property
    def mean_wr(self) -> float:
        wrs = [f.wr for f in self.folds if f.n > 0]
        return statistics.fmean(wrs) if wrs else 0.0

    @property
    def valid_folds(self) -> int:
        return sum(1 for f in self.folds if f.n >= MIN_TRADES_PER_FOLD)

    @property
    def min_fold_n(self) -> int:
        return min((f.n for f in self.folds), default=0)

    @property
    def score(self) -> float:
        # Require EVERY fold to be valid (no empty folds masking instability)
        if self.valid_folds < N_FOLDS:
            return -1e9
        if self.total_n < 30:
            return -1e9
        # Every fold must be net positive (robustness)
        if any(f.total_pnl <= 0 for f in self.folds):
            return -1e9
        mu = self.mean_avg_pnl
        sd = self.std_avg_pnl
        stability = max(0.0, 1.0 - (sd / abs(mu))) if mu > 0 else 0.0
        return mu * math.sqrt(self.total_n) * (0.4 + 0.6 * stability)


def load_signals() -> list[Signal]:
    rows: list[Signal] = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    Signal(
                        entry_minute=int(r["entry_minute"]),
                        btc_abs_move=float(r["btc_abs_move"]),
                        edge=float(r["edge"]),
                        would_fill_at=float(r["would_fill_at"]),
                        market_gap=float(r["market_gap"]),
                        pnl=float(r["would_pnl_per_share"]),
                        event_start_ts=int(r["event_start_ts"]),
                    )
                )
            except (ValueError, TypeError, KeyError):
                continue
    rows.sort(key=lambda s: s.event_start_ts)
    return rows


def time_folds(rows: list[Signal], k: int) -> list[list[Signal]]:
    n = len(rows)
    fold_size = n // k
    return [rows[i * fold_size : (i + 1) * fold_size if i < k - 1 else n] for i in range(k)]


def eval_fold(cfg: Config, rows: list[Signal]) -> FoldResult:
    matched = [s for s in rows if cfg.matches(s)]
    if not matched:
        return FoldResult(0, 0.0, 0.0, 0.0, 0.0)
    pnls = [s.pnl for s in matched]
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return FoldResult(
        n=len(matched),
        total_pnl=total,
        avg_pnl=total / len(matched),
        wr=100.0 * wins / len(matched),
        max_dd=max_dd,
    )


def evaluate(cfg: Config, folds: list[list[Signal]]) -> Evaluation:
    ev = Evaluation(cfg=cfg)
    for fold in folds:
        ev.folds.append(eval_fold(cfg, fold))
    return ev


def main() -> None:
    rows = load_signals()
    print(f"Loaded {len(rows)} signals")
    folds = time_folds(rows, N_FOLDS)
    print(f"Fold sizes: {[len(f) for f in folds]}")

    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())
    print(f"Total configs to evaluate: {len(combos)}")

    results: list[Evaluation] = []
    for combo in combos:
        cfg = Config(**dict(zip(keys, combo)))
        ev = evaluate(cfg, folds)
        if ev.score > -1e8:
            results.append(ev)

    print(f"\nEvaluated {len(results)} configs (all 5 folds positive, >=30 trades)")
    results.sort(key=lambda e: e.score, reverse=True)

    # Deduplicate: same (N, total_pnl, mean_avg) → pick tightest params
    seen: set[tuple] = set()
    deduped: list[Evaluation] = []
    for ev in results:
        sig = (ev.total_n, round(ev.total_pnl, 2), round(ev.mean_avg_pnl, 4))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(ev)
    results = deduped
    print(f"After dedup: {len(results)} unique configs")

    print("\n" + "=" * 140)
    print(
        f"{'rank':<5}{'score':<8}{'N':<5}{'PnL':<8}{'avg':<7}{'WR':<6}{'σ':<6}"
        f"{'edge':<6}{'mv_min':<7}{'mv_max':<7}{'gap':<6}{'fill_min':<9}{'fill_max':<9}"
        f"{'mn_min':<7}{'mn_max':<7}"
    )
    print("=" * 140)
    for rank, ev in enumerate(results[:25], 1):
        c = ev.cfg
        print(
            f"{rank:<5}{ev.score:<8.2f}{ev.total_n:<5}{ev.total_pnl:<8.2f}"
            f"{ev.mean_avg_pnl:<7.3f}{ev.mean_wr:<6.1f}{ev.std_avg_pnl:<6.3f}"
            f"{c.min_edge:<6.2f}{c.min_move:<7.0f}{c.max_move:<7.0f}"
            f"{c.max_gap:<6.2f}{c.min_fill:<9.2f}{c.max_fill:<9.2f}"
            f"{c.min_minute:<7}{c.max_minute:<7}"
        )

    print("\n" + "=" * 140)
    print("Top-5 per-fold detail (PnL / N / WR):")
    print("=" * 140)
    for rank, ev in enumerate(results[:5], 1):
        c = ev.cfg
        print(
            f"\n#{rank}: edge>={c.min_edge}, move=[{c.min_move:.0f},{c.max_move:.0f}], "
            f"gap<={c.max_gap}, fill=[{c.min_fill:.2f},{c.max_fill:.2f}], "
            f"min=[{c.min_minute},{c.max_minute}]"
        )
        for i, f in enumerate(ev.folds, 1):
            print(
                f"  Fold {i}: N={f.n:3d}  PnL={f.total_pnl:+6.2f}  "
                f"avg={f.avg_pnl:+6.3f}  WR={f.wr:5.1f}%  DD={f.max_dd:5.2f}"
            )


if __name__ == "__main__":
    main()
