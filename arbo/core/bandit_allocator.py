"""Thompson Sampling capital allocator — Phase 4.1.

Daily capital rebalance across variants using Beta-Bernoulli Thompson
Sampling on win-rate posteriors. Output: capital_per_variant dict.

Algorithm:
1. For each variant, maintain Beta(α=wins+1, β=losses+1) posterior (shadow
   + live combined).
2. Draw N samples from each posterior; compute win probability estimate.
3. Softmax over posteriors with temperature τ=0.25 → allocation weights.
4. UCB bonus for young variants (<30 trades): +10% weight to prevent
   starvation.
5. Apply floor + cap (min 5% for active, max 70% total to any one).

Caller: watchdog._eval_cycle calls get_allocations(strategy) once/day;
orchestrator uses returned weights × total_capital as per-variant cap.

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 4.1
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("bandit_allocator")

SOFTMAX_TEMPERATURE = 0.25
UCB_YOUNG_THRESHOLD = 30
UCB_YOUNG_BONUS_PCT = 10.0      # % added to young variant's weight
MIN_ALLOCATION_PCT = 5.0         # floor for any active variant
MAX_ALLOCATION_PCT = 70.0        # cap for any single variant
N_TS_SAMPLES = 500


@dataclass
class VariantAllocation:
    variant_id: str
    status: str
    wins: int
    losses: int
    posterior_mean: float      # Beta posterior mean = α/(α+β)
    ts_sample_mean: float      # mean of N_TS_SAMPLES posterior draws
    weight_raw: float          # softmax output before floor/cap
    weight_pct: float          # final allocation percent after floor/cap/bonus
    is_young: bool


async def get_allocations(
    strategy: str, window_days: int = 14
) -> list[VariantAllocation]:
    """Compute TS-based allocation for each active variant in strategy."""
    from arbo.core.variant_pool import load_variants
    from arbo.utils.db import get_session_factory
    import sqlalchemy as sa

    pool = load_variants(strategy)
    active = [v for v in pool if v.status in {"champion", "challenger"}]
    if not active:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
    factory = get_session_factory()
    rng = random.Random(42)

    stats: list[tuple[Any, int, int]] = []  # (variant, wins, losses)
    async with factory() as session:
        for v in active:
            # Combined shadow + live wins/losses
            r = await session.execute(
                sa.text("""
                    WITH dedup AS (
                        SELECT DISTINCT ON (condition_id, direction)
                            qualified, would_pnl_per_share
                        FROM shadow_variant_signals
                        WHERE strategy = :s AND variant_id = :v
                          AND signal_ts >= :c
                        ORDER BY condition_id, direction, signal_ts ASC
                    )
                    SELECT
                        COUNT(*) FILTER (
                            WHERE qualified AND would_pnl_per_share > 0
                        ) AS s_wins,
                        COUNT(*) FILTER (
                            WHERE qualified AND would_pnl_per_share < 0
                        ) AS s_losses
                    FROM dedup
                """),
                {"s": strategy, "v": v.variant_id, "c": cutoff},
            )
            srow = r.first()
            s_wins = int(srow.s_wins) if srow else 0
            s_losses = int(srow.s_losses) if srow else 0

            # Live wins/losses (champion's real trades)
            lr = await session.execute(
                sa.text("""
                    SELECT
                        COUNT(*) FILTER (
                            WHERE (trade_details->>'live_exit_price')::float > 0.5
                              AND trade_details->>'live_fill_status' IN ('filled','partial')
                        ) AS l_wins,
                        COUNT(*) FILTER (
                            WHERE (trade_details->>'live_exit_price')::float <= 0.5
                              AND trade_details->>'live_fill_status' IN ('filled','partial')
                        ) AS l_losses
                    FROM paper_trades
                    WHERE strategy = :s
                      AND trade_details->>'variant_id' = :v
                      AND placed_at >= :c
                """),
                {"s": strategy, "v": v.variant_id, "c": cutoff},
            )
            lrow = lr.first()
            l_wins = int(lrow.l_wins) if lrow else 0
            l_losses = int(lrow.l_losses) if lrow else 0
            stats.append((v, s_wins + l_wins, s_losses + l_losses))

    # Compute TS posterior draw mean per variant
    logits: list[float] = []
    posterior_means: list[float] = []
    ts_means: list[float] = []
    for _, wins, losses in stats:
        alpha = wins + 1.0
        beta = losses + 1.0
        post_mean = alpha / (alpha + beta)
        samples = [_beta_sample(alpha, beta, rng) for _ in range(N_TS_SAMPLES)]
        ts_mean = sum(samples) / len(samples)
        posterior_means.append(post_mean)
        ts_means.append(ts_mean)
        logits.append(ts_mean / SOFTMAX_TEMPERATURE)

    # Softmax
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps) or 1.0
    weights_raw = [e / total for e in exps]

    # UCB bonus for young variants (n < 30)
    adjusted: list[float] = []
    young_flags: list[bool] = []
    for (v, wins, losses), w in zip(stats, weights_raw):
        n = wins + losses
        young = n < UCB_YOUNG_THRESHOLD
        young_flags.append(young)
        bonus = UCB_YOUNG_BONUS_PCT / 100.0 if young else 0.0
        adjusted.append(w + bonus)
    # Renormalize
    adj_total = sum(adjusted) or 1.0
    adjusted = [x / adj_total for x in adjusted]

    # Apply floor + cap
    weights_pct = [w * 100.0 for w in adjusted]
    # Floor
    for i, w in enumerate(weights_pct):
        if w < MIN_ALLOCATION_PCT:
            weights_pct[i] = MIN_ALLOCATION_PCT
    # Cap
    for i, w in enumerate(weights_pct):
        if w > MAX_ALLOCATION_PCT:
            weights_pct[i] = MAX_ALLOCATION_PCT
    # Final renorm to 100
    total_pct = sum(weights_pct)
    if total_pct > 0:
        weights_pct = [w * 100.0 / total_pct for w in weights_pct]

    out: list[VariantAllocation] = []
    for (v, wins, losses), post_mean, ts_mean, raw, pct, young in zip(
        stats, posterior_means, ts_means, weights_raw, weights_pct, young_flags
    ):
        out.append(VariantAllocation(
            variant_id=v.variant_id,
            status=v.status,
            wins=wins,
            losses=losses,
            posterior_mean=round(post_mean, 3),
            ts_sample_mean=round(ts_mean, 3),
            weight_raw=round(raw, 4),
            weight_pct=round(pct, 1),
            is_young=young,
        ))
    out.sort(key=lambda a: a.weight_pct, reverse=True)
    return out


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Sample from Beta(alpha, beta). Uses Gamma-ratio method."""
    # random.gammavariate is not on random.Random — use inverse transform
    # via Gamma(k, θ=1) ratios. For small shape params random.Random works.
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    total = x + y
    return x / total if total > 0 else 0.5
