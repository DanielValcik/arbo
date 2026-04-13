"""PerformanceAnalyzer — Phase 2C.A.

Mines champion's paper_trades + shadow_variant_signals for failure-mode
buckets. Output feeds the HypothesisGenerator (Gemini) which proposes
single-parameter challenger mutations.

Inputs (per strategy):
- paper_trades rows for the champion (last 7-30 days, live fills only)
- shadow_variant_signals for every variant (all recent, resolved only)

Output: FailureMode dataclass list — ranked by loss concentration.

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2C.A
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("performance_analyzer")


@dataclass
class FailureMode:
    """One loss-concentration bucket in paper_trades."""
    feature: str        # e.g. "velocity", "fill_price", "btc_move"
    condition: str      # human-readable: "velocity > 45"
    sql_predicate: str  # machine: "(trade_details->>'velocity_paper')::float > 45"
    n_total: int        # total trades matching bucket
    n_losses: int       # losing trades matching bucket
    loss_rate: float    # n_losses / n_total
    avg_pnl: float      # average PnL inside bucket
    impact: float       # loss_rate × n_losses (higher = more painful overall)


@dataclass
class VariantSnapshot:
    """One variant's aggregate stats (shadow + live combined)."""
    variant_id: str
    status: str  # "champion" | "challenger" | "retired"
    live_n: int
    live_wins: int
    live_pnl: float
    live_wr: float | None
    shadow_n_qualified: int
    shadow_n_resolved: int
    shadow_wins: int
    shadow_pnl_per_share: float
    shadow_wr: float | None
    # Phase 3.3: mid-trade composite reward signal (None if no mid_at_60s yet)
    avg_mid_at_60s_drift: float | None = None  # avg (mid_60s - entry)*direction
    avg_composite_reward: float | None = None  # mean composite over resolved trades
    composite_reward_n: int = 0                # how many trades had both signals


@dataclass
class PerformanceReport:
    """Top-level output consumed by HypothesisGenerator."""
    strategy: str
    window_days: int
    generated_at: str
    champion_variant_id: str
    champion_live_n: int
    champion_live_pnl: float
    champion_live_wr: float | None
    variants: list[VariantSnapshot]
    failure_modes: list[FailureMode]
    stagnation_flag: bool    # champion below breakeven for N trades
    stagnation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class PerformanceAnalyzer:
    """Runs bucketized SQL on paper_trades + shadow_variant_signals."""

    def __init__(self, strategy: str, window_days: int = 14) -> None:
        self.strategy = strategy
        self.window_days = window_days

    async def analyze(self) -> PerformanceReport | None:
        from arbo.utils.db import get_session_factory
        from arbo.core.variant_pool import load_variants, get_champion
        import sqlalchemy as sa

        try:
            pool = load_variants(self.strategy)
            champ = get_champion(self.strategy)
        except Exception as e:
            logger.warning("perf_pool_load_error", strategy=self.strategy, error=str(e))
            return None
        if champ is None:
            logger.warning("perf_no_champion", strategy=self.strategy)
            return None

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.window_days)
        factory = get_session_factory()
        variants_snap: list[VariantSnapshot] = []
        failure_modes: list[FailureMode] = []
        champ_n = 0
        champ_pnl = 0.0
        champ_wr: float | None = None

        async with factory() as session:
            for v in pool:
                snap = await self._variant_stats(session, v, cutoff)
                variants_snap.append(snap)
                if v.variant_id == champ.variant_id:
                    champ_n = snap.live_n
                    champ_pnl = snap.live_pnl
                    champ_wr = snap.live_wr

            failure_modes = await self._compute_failure_modes(
                session, champ.variant_id, cutoff
            )

        # Stagnation heuristic: champion has ≥ 20 live trades in window AND
        # cumulative PnL < 0 (below breakeven). This is the trigger for
        # requesting a new hypothesis from Gemini.
        stagnation = champ_n >= 20 and champ_pnl < 0
        stag_reason = None
        if stagnation:
            stag_reason = f"champion {champ_n} trades, cumulative PnL ${champ_pnl:.2f} < 0"

        return PerformanceReport(
            strategy=self.strategy,
            window_days=self.window_days,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            champion_variant_id=champ.variant_id,
            champion_live_n=champ_n,
            champion_live_pnl=round(champ_pnl, 2),
            champion_live_wr=champ_wr,
            variants=variants_snap,
            failure_modes=failure_modes,
            stagnation_flag=stagnation,
            stagnation_reason=stag_reason,
        )

    async def _variant_stats(
        self, session: Any, variant: Any, cutoff: datetime
    ) -> VariantSnapshot:
        import sqlalchemy as sa

        live = await session.execute(
            sa.text("""
                SELECT
                    COUNT(*) FILTER (
                        WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                          AND (trade_details->>'live_entry_shares')::float > 0
                    ) AS n_live,
                    COUNT(*) FILTER (
                        WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                          AND (trade_details->>'live_entry_shares')::float > 0
                          AND (trade_details->>'live_exit_price')::float > 0.5
                    ) AS wins,
                    COALESCE(SUM(
                        ((trade_details->>'live_exit_price')::float
                         - (trade_details->>'live_entry_price')::float)
                        * (trade_details->>'live_entry_shares')::float
                    ) FILTER (
                        WHERE trade_details->>'live_fill_status' IN ('filled','partial')
                          AND (trade_details->>'live_entry_shares')::float > 0
                          AND trade_details->>'live_exit_price' IS NOT NULL
                          AND trade_details->>'live_exit_status' IN ('resolution','filled','partial')
                    ), 0) AS pnl
                FROM paper_trades
                WHERE strategy = :s
                  AND trade_details->>'variant_id' = :v
                  AND placed_at >= :cutoff
            """),
            {"s": self.strategy, "v": variant.variant_id, "cutoff": cutoff},
        )
        lrow = live.first()
        n_live = int(lrow.n_live) if lrow else 0
        wins = int(lrow.wins) if lrow else 0
        pnl = float(lrow.pnl) if lrow else 0.0
        wr = (100.0 * wins / n_live) if n_live > 0 else None

        shadow = await session.execute(
            sa.text("""
                SELECT
                    COUNT(*) FILTER (WHERE qualified) AS n_qualified,
                    COUNT(*) FILTER (
                        WHERE qualified AND resolution_outcome IS NOT NULL
                    ) AS n_resolved,
                    COUNT(*) FILTER (
                        WHERE qualified AND would_pnl_per_share > 0
                    ) AS s_wins,
                    COALESCE(SUM(would_pnl_per_share) FILTER (
                        WHERE qualified AND would_pnl_per_share IS NOT NULL
                    ), 0) AS s_pnl
                FROM shadow_variant_signals
                WHERE strategy = :s
                  AND variant_id = :v
                  AND signal_ts >= :cutoff
            """),
            {"s": self.strategy, "v": variant.variant_id, "cutoff": cutoff},
        )
        srow = shadow.first()
        s_qual = int(srow.n_qualified) if srow else 0
        s_resolved = int(srow.n_resolved) if srow else 0
        s_wins = int(srow.s_wins) if srow else 0
        s_pnl = float(srow.s_pnl) if srow else 0.0
        s_decided = s_wins + max(0, s_resolved - s_wins)
        s_wr = (100.0 * s_wins / s_resolved) if s_resolved > 0 else None

        # Phase 3.3: mid-trade composite reward stats (live trades only)
        mid_drift_avg: float | None = None
        composite_avg: float | None = None
        composite_n = 0
        try:
            mid_r = await session.execute(
                sa.text("""
                    SELECT
                        AVG(
                            ((trade_details->>'mid_at_60s')::float
                             - (trade_details->>'live_entry_price')::float)
                            * CASE
                                WHEN trade_details->>'direction' = 'down' THEN -1
                                ELSE 1
                              END
                        ) AS avg_drift,
                        COUNT(*) FILTER (
                            WHERE trade_details->>'mid_at_60s' IS NOT NULL
                              AND trade_details->>'live_exit_price' IS NOT NULL
                        ) AS n_with_signal
                    FROM paper_trades
                    WHERE strategy = :s
                      AND trade_details->>'variant_id' = :v
                      AND trade_details->>'mid_at_60s' IS NOT NULL
                      AND trade_details->>'live_entry_price' IS NOT NULL
                      AND placed_at >= :cutoff
                """),
                {"s": self.strategy, "v": variant.variant_id, "cutoff": cutoff},
            )
            mrow = mid_r.first()
            if mrow:
                mid_drift_avg = (
                    round(float(mrow.avg_drift), 4)
                    if mrow.avg_drift is not None else None
                )
                composite_n = int(mrow.n_with_signal or 0)
        except Exception as e:
            logger.debug("perf_mid_query_error", error=str(e))

        # Composite reward needs Python evaluation per row (CASE complexity)
        if composite_n > 0:
            try:
                from arbo.core.mid_sampler import composite_reward
                comp_r = await session.execute(
                    sa.text("""
                        SELECT
                            CASE WHEN trade_details->>'direction' = 'down'
                                 THEN -1 ELSE 1 END AS direction,
                            (trade_details->>'live_entry_price')::float AS entry_price,
                            (trade_details->>'mid_at_60s')::float AS mid_60s,
                            ((trade_details->>'live_exit_price')::float
                             - (trade_details->>'live_entry_price')::float) AS pnl
                        FROM paper_trades
                        WHERE strategy = :s
                          AND trade_details->>'variant_id' = :v
                          AND trade_details->>'mid_at_60s' IS NOT NULL
                          AND trade_details->>'live_entry_price' IS NOT NULL
                          AND trade_details->>'live_exit_price' IS NOT NULL
                          AND placed_at >= :cutoff
                    """),
                    {"s": self.strategy, "v": variant.variant_id, "cutoff": cutoff},
                )
                vals: list[float] = []
                for row in comp_r.fetchall():
                    cr = composite_reward(
                        direction=int(row[0]),
                        entry_price=float(row[1]),
                        mid_at_60s=float(row[2]),
                        pnl_per_share=float(row[3]),
                    )
                    if cr is not None:
                        vals.append(cr)
                if vals:
                    composite_avg = round(sum(vals) / len(vals), 4)
            except Exception as e:
                logger.debug("perf_composite_error", error=str(e))

        return VariantSnapshot(
            variant_id=variant.variant_id,
            status=variant.status,
            live_n=n_live,
            live_wins=wins,
            live_pnl=round(pnl, 2),
            live_wr=round(wr, 1) if wr is not None else None,
            shadow_n_qualified=s_qual,
            shadow_n_resolved=s_resolved,
            shadow_wins=s_wins,
            shadow_pnl_per_share=round(s_pnl, 4),
            shadow_wr=round(s_wr, 1) if s_wr is not None else None,
            avg_mid_at_60s_drift=mid_drift_avg,
            avg_composite_reward=composite_avg,
            composite_reward_n=composite_n,
        )

    async def _compute_failure_modes(
        self, session: Any, champion_id: str, cutoff: datetime
    ) -> list[FailureMode]:
        """Build list of loss-concentration buckets.

        Returns empty list if fewer than 10 losing trades — can't draw
        reliable bucket signal below that.
        """
        import sqlalchemy as sa

        if self.strategy == "B3":
            buckets = [
                ("velocity", "velocity_paper > 50",
                 "(trade_details->>'velocity_paper')::float > 50"),
                ("velocity", "velocity_paper > 45",
                 "(trade_details->>'velocity_paper')::float > 45"),
                ("dir_delta", "abs_dir_delta_paper > 12",
                 "(trade_details->>'abs_dir_delta_paper')::float > 12"),
                ("fill_price", "live_entry_price > 0.70",
                 "(trade_details->>'live_entry_price')::float > 0.70"),
                ("fill_price", "live_entry_price > 0.65",
                 "(trade_details->>'live_entry_price')::float > 0.65"),
                ("btc_move", "btc_abs_move > 80",
                 "(trade_details->>'btc_abs_move')::float > 80"),
                ("edge", "edge < 0.35",
                 "(trade_details->>'edge')::float < 0.35"),
                ("market_gap", "market_gap > 0.15",
                 "(trade_details->>'market_gap')::float > 0.15"),
            ]
        elif self.strategy == "B3_15M":
            buckets = [
                ("fill_price", "live_entry_price > 0.75",
                 "(trade_details->>'live_entry_price')::float > 0.75"),
                ("fill_price", "live_entry_price > 0.85",
                 "(trade_details->>'live_entry_price')::float > 0.85"),
                ("market_gap", "market_gap > 0.20",
                 "(trade_details->>'market_gap')::float > 0.20"),
                ("btc_move", "btc_abs_move > 60",
                 "(trade_details->>'btc_abs_move')::float > 60"),
                ("edge", "edge < 0.35",
                 "(trade_details->>'edge')::float < 0.35"),
                ("entry_minute", "entry_minute > 8",
                 "(trade_details->>'entry_minutes_elapsed')::float > 8"),
            ]
        else:
            return []

        modes: list[FailureMode] = []
        # Total losing trades (reference denom)
        total_result = await session.execute(
            sa.text("""
                SELECT
                    COUNT(*) AS n_total,
                    COUNT(*) FILTER (
                        WHERE (trade_details->>'live_exit_price')::float <= 0.5
                    ) AS n_losses
                FROM paper_trades
                WHERE strategy = :s
                  AND trade_details->>'variant_id' = :v
                  AND trade_details->>'live_fill_status' IN ('filled','partial')
                  AND (trade_details->>'live_entry_shares')::float > 0
                  AND trade_details->>'live_exit_price' IS NOT NULL
                  AND placed_at >= :cutoff
            """),
            {"s": self.strategy, "v": champion_id, "cutoff": cutoff},
        )
        trow = total_result.first()
        if not trow or int(trow.n_losses or 0) < 10:
            return []

        for feature, cond, pred in buckets:
            try:
                r = await session.execute(
                    sa.text(f"""
                        SELECT
                            COUNT(*) AS n_total,
                            COUNT(*) FILTER (
                                WHERE (trade_details->>'live_exit_price')::float <= 0.5
                            ) AS n_losses,
                            COALESCE(AVG(
                                ((trade_details->>'live_exit_price')::float
                                 - (trade_details->>'live_entry_price')::float)
                                * (trade_details->>'live_entry_shares')::float
                            ), 0) AS avg_pnl
                        FROM paper_trades
                        WHERE strategy = :s
                          AND trade_details->>'variant_id' = :v
                          AND trade_details->>'live_fill_status' IN ('filled','partial')
                          AND (trade_details->>'live_entry_shares')::float > 0
                          AND trade_details->>'live_exit_price' IS NOT NULL
                          AND placed_at >= :cutoff
                          AND {pred}
                    """),
                    {"s": self.strategy, "v": champion_id, "cutoff": cutoff},
                )
                brow = r.first()
            except Exception as e:
                logger.debug("perf_bucket_sql_error", bucket=cond, error=str(e))
                continue
            if not brow:
                continue
            n_total = int(brow.n_total or 0)
            n_losses = int(brow.n_losses or 0)
            avg_pnl = float(brow.avg_pnl or 0)
            if n_total < 3:
                continue
            loss_rate = n_losses / n_total if n_total > 0 else 0.0
            impact = loss_rate * n_losses
            modes.append(FailureMode(
                feature=feature,
                condition=cond,
                sql_predicate=pred,
                n_total=n_total,
                n_losses=n_losses,
                loss_rate=round(loss_rate, 3),
                avg_pnl=round(avg_pnl, 3),
                impact=round(impact, 2),
            ))

        modes.sort(key=lambda m: m.impact, reverse=True)
        return modes[:5]
