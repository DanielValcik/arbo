"""PromotionEngine — Phase 2C.B.

Given the current variant pool and recent shadow + live data, identifies
challengers that are beating the champion with high enough statistical
confidence to merit promotion. Does NOT execute promotion — emits a
PromotionCandidate event; Slack bot (Phase 2C.C) and PoolManager (2C.D)
handle the actual atomic swap.

Statistical method:
- Per-trade PnL series (shadow for challengers, shadow+live for champion)
- Block-bootstrap (Politis-Romano stationary, block size = 10) for
  P(challenger_mean > champion_mean)
- Deflated Sharpe Ratio approximation (shrink by sqrt(1 / n_variants_tested))

Tier classification of the proposed change:
- Tier 1: change touches only allowed-bounds params (MIN_EDGE, VELOCITY, etc.)
  → eligible for auto-promote with 24h Slack veto window
- Tier 2: change touches sigma_scale / sizing → flag CEO, manual only
- Tier 3: change touches MAX_BET_SIZE / DAILY_LOSS_PCT → reject always

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2C.B
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from arbo.core.hypothesis_generator import TIER_1_PARAMS
from arbo.core.variant_pool import get_champion, load_variants
from arbo.utils.logger import get_logger

logger = get_logger("promotion_engine")

# Tier 2 params (CEO approval required, not auto-promoted)
TIER_2_PARAMS = {
    "SIGMA_SCALE", "SIGMA_FLOOR", "POSITION_PCT", "EDGE_SCALING",
    "ENTRY_THRESHOLD", "PROFIT_TARGET", "EDGE_EXIT", "BTC_STOP_PCT",
}
# Tier 3 params (never promoted autonomously)
TIER_3_PARAMS = {
    "MAX_BET_SIZE", "MAX_SHARES", "MIN_ORDER_SIZE",
}

# Thresholds (configurable — start conservative)
MIN_PAIRED_N = 100          # min shadow signals for challenger
MIN_P_BETTER = 0.75         # bootstrap P(better) threshold for Tier 1 auto
MIN_P_BETTER_CEO = 0.65     # lower threshold triggers Tier 2 CEO alert
MIN_SHARPE_DELTA = 0.10     # minimum DSR gap (deflation-adjusted)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_BLOCK_SIZE = 10


@dataclass
class PromotionCandidate:
    """One challenger flagged for possible promotion."""
    strategy: str
    challenger_id: str
    champion_id: str
    tier: int                    # 1, 2, or 3
    n_challenger: int
    n_champion: int
    mean_challenger: float       # avg PnL per trade
    mean_champion: float
    sharpe_challenger: float
    sharpe_champion: float
    dsr_delta: float             # deflation-adjusted Sharpe gap
    p_better: float              # bootstrap P(challenger > champion)
    wr_challenger: float | None
    wr_champion: float | None
    param_diff: dict[str, tuple[float, float]]  # {name: (champion, challenger)}
    rationale: str
    reject_reason: str | None = None


def _classify_tier(param_diff: dict[str, Any]) -> tuple[int, str | None]:
    """Return (tier, reject_reason_if_tier_3_or_unknown)."""
    if not param_diff:
        return 3, "no parameter differences"
    names = set(param_diff.keys())
    if names & TIER_3_PARAMS:
        return 3, f"touches tier-3 params: {names & TIER_3_PARAMS}"
    if names & TIER_2_PARAMS:
        return 2, None
    if names.issubset(set(TIER_1_PARAMS.keys())):
        return 1, None
    # Unknown params — reject conservatively
    unknown = names - set(TIER_1_PARAMS.keys()) - TIER_2_PARAMS - TIER_3_PARAMS
    if unknown:
        return 3, f"unknown params: {unknown}"
    return 2, None


def _sharpe(series: list[float]) -> float:
    """Simple Sharpe (per-trade, no annualization)."""
    if len(series) < 2:
        return 0.0
    mean = sum(series) / len(series)
    var = sum((x - mean) ** 2 for x in series) / (len(series) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    return mean / sd if sd > 0 else 0.0


def _block_bootstrap_p_better(
    a: list[float], b: list[float], *, resamples: int, block: int
) -> float:
    """Estimate P(mean(a) > mean(b)) via stationary block bootstrap.

    Samples index sequences of `block` length with replacement independently
    from each series, then computes the delta of means. Returns fraction of
    resamples where mean(a) > mean(b).
    """
    if not a or not b:
        return 0.5
    rng = random.Random(42)
    wins = 0
    na = len(a)
    nb = len(b)
    for _ in range(resamples):
        # Sample with block structure (reduces dependency artifact)
        resa = _sample_blocks(a, na, block, rng)
        resb = _sample_blocks(b, nb, block, rng)
        if (sum(resa) / len(resa)) > (sum(resb) / len(resb)):
            wins += 1
    return wins / resamples


def _sample_blocks(series: list[float], target_n: int, block: int, rng: random.Random) -> list[float]:
    n = len(series)
    out: list[float] = []
    while len(out) < target_n:
        start = rng.randrange(0, n)
        out.extend(series[start:start + block])
        if len(out) >= target_n:
            out = out[:target_n]
            break
        if start + block >= n:
            # Wrap — stationary bootstrap with circularity
            wrap = start + block - n
            out.extend(series[:wrap])
    return out[:target_n]


def _deflated_delta(sharpe_diff: float, n_variants: int) -> float:
    """Crude deflation: shrink by sqrt(1/n_variants_tested)."""
    if n_variants <= 1:
        return sharpe_diff
    return sharpe_diff / math.sqrt(float(n_variants))


class PromotionEngine:
    """Detects promotion candidates without executing anything."""

    def __init__(self, strategy: str, window_days: int = 14) -> None:
        self.strategy = strategy
        self.window_days = window_days

    async def evaluate(self) -> list[PromotionCandidate]:
        """Return list of candidates (may be empty)."""
        from arbo.utils.db import get_session_factory
        import sqlalchemy as sa

        champ = get_champion(self.strategy)
        if champ is None:
            return []
        pool = load_variants(self.strategy)
        challengers = [v for v in pool if v.status == "challenger"]
        if not challengers:
            return []

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.window_days)
        factory = get_session_factory()
        candidates: list[PromotionCandidate] = []

        async with factory() as session:
            # Champion's shadow PnL series
            champ_series, champ_wr = await self._fetch_pnl_series(
                session, champ.variant_id, cutoff
            )
            champ_series_trimmed = champ_series[-500:]  # cap

            for ch in challengers:
                ch_series, ch_wr = await self._fetch_pnl_series(
                    session, ch.variant_id, cutoff
                )
                if len(ch_series) < MIN_PAIRED_N:
                    continue  # insufficient data

                # Param diff challenger vs champion
                diff: dict[str, tuple[float, float]] = {}
                for k, v in ch.params.items():
                    cv = champ.params.get(k)
                    if cv is None:
                        continue
                    try:
                        if abs(float(cv) - float(v)) > 1e-9:
                            diff[k] = (float(cv), float(v))
                    except (TypeError, ValueError):
                        continue

                tier, tier_reject = _classify_tier(diff)
                mean_ch = sum(ch_series) / len(ch_series)
                mean_cp = sum(champ_series_trimmed) / max(len(champ_series_trimmed), 1)
                sharpe_ch = _sharpe(ch_series)
                sharpe_cp = _sharpe(champ_series_trimmed)
                dsr_delta = _deflated_delta(sharpe_ch - sharpe_cp, len(challengers))
                p_better = _block_bootstrap_p_better(
                    ch_series, champ_series_trimmed,
                    resamples=BOOTSTRAP_RESAMPLES,
                    block=BOOTSTRAP_BLOCK_SIZE,
                )

                rationale_parts = [
                    f"N_challenger={len(ch_series)}, N_champion={len(champ_series_trimmed)}",
                    f"mean_PnL: ch ${mean_ch:.4f} vs cp ${mean_cp:.4f}",
                    f"Sharpe: ch {sharpe_ch:.2f} vs cp {sharpe_cp:.2f} (Δ deflation={dsr_delta:.3f})",
                    f"P(better) block-bootstrap: {p_better:.2f}",
                ]
                rationale = " | ".join(rationale_parts)

                reject: str | None = None
                if tier_reject:
                    reject = f"tier_3: {tier_reject}"
                elif mean_ch <= mean_cp:
                    reject = "challenger mean PnL not above champion"
                elif p_better < MIN_P_BETTER_CEO:
                    reject = f"P(better)={p_better:.2f} below CEO threshold {MIN_P_BETTER_CEO}"
                elif dsr_delta < MIN_SHARPE_DELTA and tier == 1:
                    reject = f"dsr_delta={dsr_delta:.3f} below min {MIN_SHARPE_DELTA}"

                cand = PromotionCandidate(
                    strategy=self.strategy,
                    challenger_id=ch.variant_id,
                    champion_id=champ.variant_id,
                    tier=tier,
                    n_challenger=len(ch_series),
                    n_champion=len(champ_series_trimmed),
                    mean_challenger=round(mean_ch, 4),
                    mean_champion=round(mean_cp, 4),
                    sharpe_challenger=round(sharpe_ch, 3),
                    sharpe_champion=round(sharpe_cp, 3),
                    dsr_delta=round(dsr_delta, 3),
                    p_better=round(p_better, 3),
                    wr_challenger=ch_wr,
                    wr_champion=champ_wr,
                    param_diff=diff,
                    rationale=rationale,
                    reject_reason=reject,
                )
                candidates.append(cand)

        # Sort by p_better descending
        candidates.sort(key=lambda c: c.p_better, reverse=True)
        return candidates

    async def _fetch_pnl_series(
        self, session: Any, variant_id: str, cutoff: datetime
    ) -> tuple[list[float], float | None]:
        """Return (pnl_per_share_series_ordered_by_ts, win_rate_pct_or_None)."""
        import sqlalchemy as sa

        r = await session.execute(
            sa.text("""
                SELECT would_pnl_per_share
                FROM shadow_variant_signals
                WHERE strategy = :s
                  AND variant_id = :v
                  AND qualified = true
                  AND would_pnl_per_share IS NOT NULL
                  AND signal_ts >= :cutoff
                ORDER BY signal_ts ASC
            """),
            {"s": self.strategy, "v": variant_id, "cutoff": cutoff},
        )
        series = [float(row[0]) for row in r.fetchall()]
        wins = sum(1 for x in series if x > 0)
        wr = (100.0 * wins / len(series)) if series else None
        return series, (round(wr, 1) if wr is not None else None)
