"""DWatchdog — Project PARALLEL extension for D_NBA.

Mirrors B2Watchdog architecture: lightweight daemon running shadow-only
optimization cycles (autochallenger + promotion + drift). Additionally
calls the strategy's shadow_resolution_sweep on cadence (D's events
resolve via Polymarket Gamma after game ends).

Spec: docs/PROJECT_PARALLEL_B2_DNBA_PLAN.md §4.8
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("d_watchdog")

_EVAL_INTERVAL_S = 6 * 3600
_RESOLUTION_SWEEP_INTERVAL_S = 300  # 5 min — NBA games end relatively often
_WARMUP_S = 60 * 30


class DWatchdog:
    """Project PARALLEL daemon for D (NBA Green Book)."""

    STRATEGY_NAME = "D"

    def __init__(
        self,
        gemini_agent: Any | None = None,
        slack_bot: Any | None = None,
        strategy_ref: Any | None = None,  # StrategyDNba instance for sweep
    ) -> None:
        self._gemini = gemini_agent
        self._slack = slack_bot
        self._strategy_ref = strategy_ref
        self._running = False
        self._last_eval_ts: float = 0.0
        self._last_autochallenger_ts: float = 0.0
        self._last_resolution_sweep_ts: float = 0.0
        self._promotion_posted: dict[str, float] = {}

    def attach_strategy(self, strategy_ref: Any) -> None:
        """Late-bind strategy reference (orchestrator may init us first)."""
        self._strategy_ref = strategy_ref

    async def run(self) -> None:
        """Main daemon loop. Runs until stop()."""
        self._running = True
        logger.info("d_watchdog_started")

        # Warmup
        for _ in range(_WARMUP_S):
            if not self._running:
                return
            await asyncio.sleep(1)

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("d_watchdog_cycle_error", error=str(e))

            # Sleep 60s, check resolution sweep more often than full eval
            for _ in range(60):
                if not self._running:
                    return
                await asyncio.sleep(60)

    async def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False

    async def _tick(self) -> None:
        """Per-tick work: resolution sweep always, full eval on cadence."""
        now = time.time()

        # Resolution sweep (every 5 min)
        if now - self._last_resolution_sweep_ts >= _RESOLUTION_SWEEP_INTERVAL_S:
            self._last_resolution_sweep_ts = now
            if self._strategy_ref is not None:
                try:
                    await self._strategy_ref.sweep_shadow_resolutions_d()
                except Exception as e:
                    logger.debug("d_resolution_sweep_error", error=str(e))

        # Full eval (every 6h)
        if now - self._last_eval_ts >= _EVAL_INTERVAL_S:
            self._last_eval_ts = now
            await self._autochallenger_cycle()
            await self._promotion_cycle()
            await self._drift_cycle()

    async def _autochallenger_cycle(self) -> None:
        """PerformanceAnalyzer → Gemini → PoolManager pipeline."""
        now = time.time()
        if now - self._last_autochallenger_ts < 24 * 3600:
            return

        try:
            from arbo.core.performance_analyzer import PerformanceAnalyzer
            from arbo.core.hypothesis_generator import HypothesisGenerator
            from arbo.core import pool_manager
        except Exception as e:
            logger.warning("d_autochallenger_import_error", error=str(e))
            return

        report = await PerformanceAnalyzer(self.STRATEGY_NAME, window_days=21).analyze()
        if report is None or not report.stagnation_flag:
            return

        hypo = await HypothesisGenerator(llm_agent=self._gemini).propose(report)
        if hypo is None:
            return

        ok, reason = await pool_manager.commit(hypo, self.STRATEGY_NAME)
        if not ok:
            logger.info("d_autochallenger_not_committed", reason=reason)
            return

        self._last_autochallenger_ts = now
        if self._slack is not None:
            try:
                changes = ", ".join(f"{k}={v}" for k, v in hypo.param_changes.items())
                msg = (
                    f":robot_face: *D (NBA) Auto-Challenger generated*\n"
                    f"Variant: `{hypo.variant_id}` (parent: `{hypo.parent_variant_id}`)\n"
                    f"Změna: `{changes}`\n"
                    f"Důvod: {hypo.rationale}\n"
                    f"Trigger: {report.stagnation_reason}\n"
                    f"Status: *shadow* (žádný kapitál, 24h CEO veto window)\n"
                    f"Veto: `/arbo veto {hypo.variant_id}`"
                )
                await self._slack._post(self._d_channel(), text=msg)
            except Exception as e:
                logger.warning("d_autochallenger_slack_error", error=str(e))

    async def _promotion_cycle(self) -> None:
        """Detect promotion candidates → post to Slack."""
        try:
            from arbo.core.promotion_engine import PromotionEngine, MIN_P_BETTER
            from arbo.dashboard.slack_promotion import post_candidate
        except Exception as e:
            logger.warning("d_promotion_import_error", error=str(e))
            return

        cands = await PromotionEngine(self.STRATEGY_NAME).evaluate()
        now = time.time()
        for cand in cands:
            if cand.reject_reason:
                continue
            if cand.tier == 1 and cand.p_better < MIN_P_BETTER:
                continue
            last_ts = self._promotion_posted.get(cand.challenger_id, 0.0)
            if now - last_ts < 24 * 3600:
                continue
            ok = await post_candidate(self._slack, cand)
            if ok:
                self._promotion_posted[cand.challenger_id] = now
                logger.info(
                    "d_promotion_emitted",
                    challenger=cand.challenger_id,
                    tier=cand.tier, p_better=cand.p_better,
                )

    async def _drift_cycle(self) -> None:
        """Page-Hinkley drift test → Slack alert on firing."""
        try:
            from arbo.core.drift_monitor import evaluate_strategy_drift
        except Exception as e:
            logger.warning("d_drift_import_error", error=str(e))
            return

        results = await evaluate_strategy_drift(self.STRATEGY_NAME)
        firing = [r for r in results if r.firing]
        if not firing:
            return
        for r in firing:
            logger.warning(
                "d_drift_detected",
                variant_id=r.variant_id, ph_stat=r.ph_stat, n_samples=r.n_samples,
            )
        if self._slack is not None:
            lines = [":rotating_light: *D (NBA) Drift Alert*"]
            for r in firing:
                lines.append(
                    f"• `{r.variant_id}` PH={r.ph_stat} (N={r.n_samples}, mean={r.running_mean})"
                )
            try:
                await self._slack._post(self._d_channel(), text="\n".join(lines))
            except Exception as e:
                logger.warning("d_drift_slack_error", error=str(e))

    def _d_channel(self) -> str:
        from arbo.dashboard.slack_promotion import STRATEGY_CHANNELS
        return STRATEGY_CHANNELS.get(
            "D",
            getattr(self._slack, "_channel_id", "") or "C0APX4K8Z2N",
        )
