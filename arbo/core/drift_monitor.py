"""Drift monitor — Phase 4.2.

Page-Hinkley test per variant to detect when a variant's PnL distribution
shifts from its historical baseline. Firing the PH test triggers Watchdog
auto-revert (if recent promotion) or a new BO sweep.

The test is one-sided: we care about DOWNWARD shifts (degrading performance),
not upward (improvement is fine).

Implementation: classical Page-Hinkley. Rolling computed from persisted
shadow_variant_signals + paper_trades, so no in-memory state across restarts.

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("drift_monitor")

# PH parameters
_PH_DELTA = 0.05   # allowed tolerance (5pp WR drift)
_PH_LAMBDA = 20.0  # fire threshold — higher = slower to fire, fewer false alarms
_MIN_N_FOR_TEST = 50


@dataclass
class DriftResult:
    variant_id: str
    n_samples: int
    ph_stat: float            # (m_n - M_n) — fires when > _PH_LAMBDA
    firing: bool
    fire_reason: str | None   # explanation if firing, else None
    running_mean: float       # running avg of the signal


async def evaluate_strategy_drift(
    strategy: str, window_days: int = 21
) -> list[DriftResult]:
    """Run PH test on each variant's PnL series. Return per-variant results."""
    from arbo.core.variant_pool import load_variants
    from arbo.utils.db import get_session_factory
    import sqlalchemy as sa

    pool = load_variants(strategy)
    active = [v for v in pool if v.status in {"champion", "challenger"}]
    if not active:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
    factory = get_session_factory()
    results: list[DriftResult] = []

    async with factory() as session:
        for v in active:
            # Use per-trade pnl_per_share as the signal (normalized)
            # Dedupe per (condition_id, direction): one market = one sample.
            # Raw table has 400× duplicates from periodic re-scans; PH stat
            # on raw data explodes (seen 101+ on duplicated losing signal
            # that was really 3 unique markets).
            r = await session.execute(
                sa.text("""
                    SELECT would_pnl_per_share FROM (
                        SELECT DISTINCT ON (condition_id, direction)
                            condition_id, direction, signal_ts, would_pnl_per_share
                        FROM shadow_variant_signals
                        WHERE strategy = :s
                          AND variant_id = :v
                          AND qualified = true
                          AND would_pnl_per_share IS NOT NULL
                          AND signal_ts >= :c
                        ORDER BY condition_id, direction, signal_ts ASC
                    ) t
                    ORDER BY signal_ts ASC
                """),
                {"s": strategy, "v": v.variant_id, "c": cutoff},
            )
            series = [float(row[0]) for row in r.fetchall()]
            results.append(_page_hinkley(v.variant_id, series))

    return results


def _page_hinkley(variant_id: str, series: list[float]) -> DriftResult:
    """Run one-sided PH test on a pnl series (negative drift = degradation)."""
    n = len(series)
    if n < _MIN_N_FOR_TEST:
        return DriftResult(
            variant_id=variant_id,
            n_samples=n,
            ph_stat=0.0,
            firing=False,
            fire_reason=None,
            running_mean=round(sum(series) / n, 4) if series else 0.0,
        )

    # Rolling mean for reference; PH uses running mean up to each step
    m_n = 0.0
    M_n = 0.0
    running_sum = 0.0
    running_count = 0
    for x in series:
        running_count += 1
        running_sum += x
        mean_so_far = running_sum / running_count
        # Negative drift: we care when x < mean - delta
        # Per classical PH:  m_n = m_{n-1} + (mean - x - delta)  (for downward test)
        m_n += (mean_so_far - x - _PH_DELTA)
        if m_n < M_n:
            M_n = m_n

    ph = round(m_n - M_n, 3)
    firing = ph > _PH_LAMBDA
    reason = None
    if firing:
        reason = (
            f"Page-Hinkley statistic {ph:.2f} > λ={_PH_LAMBDA} "
            f"over N={n} samples (δ={_PH_DELTA}). Signal drifting below "
            f"baseline."
        )
    return DriftResult(
        variant_id=variant_id,
        n_samples=n,
        ph_stat=ph,
        firing=firing,
        fire_reason=reason,
        running_mean=round(running_sum / running_count, 4),
    )
