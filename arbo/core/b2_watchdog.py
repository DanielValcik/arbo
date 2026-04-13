"""B2Watchdog — Project PARALLEL extension.

Lightweight daemon that runs the strategy-agnostic Project PARALLEL
cycles for B2 (crypto price edge):
  - autochallenger (PerformanceAnalyzer → Gemini → PoolManager)
  - promotion engine (block-bootstrap + DSR + tier classification)
  - drift monitor (Page-Hinkley per variant)

Does NOT include B3-style anomaly detection / runtime config tuning —
B2 has its own monitoring track via crypto_quality_gate's day/night
regime, and the promotion workflow is the only thing we need here.

Spec: docs/PROJECT_PARALLEL_B2_DNBA_PLAN.md §3.8
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("b2_watchdog")

# Cadence — same as B3Watchdog
_EVAL_INTERVAL_S = 6 * 3600
_WARMUP_S = 60 * 30  # 30-min warmup before first eval


class B2Watchdog:
    """Project PARALLEL daemon for B2."""

    STRATEGY_NAME = "B2"

    def __init__(
        self,
        gemini_agent: Any | None = None,
        slack_bot: Any | None = None,
    ) -> None:
        self._gemini = gemini_agent
        self._slack = slack_bot
        self._running = False
        self._last_eval_ts: float = 0.0
        self._last_autochallenger_ts: float = 0.0
        self._promotion_posted: dict[str, float] = {}

    async def run(self) -> None:
        """Main daemon loop. Runs until stop()."""
        self._running = True
        logger.info("b2_watchdog_started")

        # Warmup — let trades accumulate before first eval
        for _ in range(_WARMUP_S):
            if not self._running:
                return
            await asyncio.sleep(1)

        while self._running:
            try:
                await self._eval_cycle()
            except Exception as e:
                logger.error("b2_watchdog_cycle_error", error=str(e))

            # Sleep in 60s chunks for responsive shutdown
            for _ in range(60):
                if not self._running:
                    return
                await asyncio.sleep(60)

    async def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False

    async def _eval_cycle(self) -> None:
        """One eval pass — autochallenger + promotion + drift."""
        now = time.time()
        if now - self._last_eval_ts < _EVAL_INTERVAL_S:
            return
        self._last_eval_ts = now

        await self._autochallenger_cycle()
        await self._promotion_cycle()
        await self._drift_cycle()

    async def _autochallenger_cycle(self) -> None:
        """PerformanceAnalyzer → Gemini → PoolManager pipeline."""
        now = time.time()
        if now - self._last_autochallenger_ts < 24 * 3600:
            return  # 1 per day max

        try:
            from arbo.core.performance_analyzer import PerformanceAnalyzer
            from arbo.core.hypothesis_generator import HypothesisGenerator
            from arbo.core import pool_manager
        except Exception as e:
            logger.warning("b2_autochallenger_import_error", error=str(e))
            return

        report = await PerformanceAnalyzer(self.STRATEGY_NAME, window_days=14).analyze()
        if report is None or not report.stagnation_flag:
            return

        hypo = await HypothesisGenerator(llm_agent=self._gemini).propose(report)
        if hypo is None:
            return

        ok, reason = await pool_manager.commit(hypo, self.STRATEGY_NAME)
        if not ok:
            logger.info(
                "b2_autochallenger_not_committed",
                reason=reason,
            )
            return

        self._last_autochallenger_ts = now
        if self._slack is not None:
            try:
                changes = ", ".join(f"{k}={v}" for k, v in hypo.param_changes.items())
                msg = (
                    f":robot_face: *B2 Auto-Challenger generated*\n"
                    f"Variant: `{hypo.variant_id}` (parent: `{hypo.parent_variant_id}`)\n"
                    f"Změna: `{changes}`\n"
                    f"Důvod: {hypo.rationale}\n"
                    f"Trigger: {report.stagnation_reason}\n"
                    f"Status: *shadow* (žádný kapitál, 24h CEO veto window)\n"
                    f"Veto: `/arbo veto {hypo.variant_id}`"
                )
                channel = self._b2_channel()
                await self._slack._post(channel, text=msg)
            except Exception as e:
                logger.warning("b2_autochallenger_slack_error", error=str(e))

    async def _promotion_cycle(self) -> None:
        """Detect promotion candidates → post to Slack."""
        try:
            from arbo.core.promotion_engine import PromotionEngine, MIN_P_BETTER
            from arbo.dashboard.slack_promotion import post_candidate
        except Exception as e:
            logger.warning("b2_promotion_import_error", error=str(e))
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
                    "b2_promotion_emitted",
                    challenger=cand.challenger_id,
                    tier=cand.tier,
                    p_better=cand.p_better,
                )

    async def _drift_cycle(self) -> None:
        """Page-Hinkley drift test → Slack alert on firing."""
        try:
            from arbo.core.drift_monitor import evaluate_strategy_drift
        except Exception as e:
            logger.warning("b2_drift_import_error", error=str(e))
            return

        results = await evaluate_strategy_drift(self.STRATEGY_NAME)
        firing = [r for r in results if r.firing]
        if not firing:
            return
        for r in firing:
            logger.warning(
                "b2_drift_detected",
                variant_id=r.variant_id,
                ph_stat=r.ph_stat,
                n_samples=r.n_samples,
            )
        if self._slack is not None:
            lines = [":rotating_light: *B2 Drift Alert*"]
            for r in firing:
                lines.append(
                    f"• `{r.variant_id}` PH={r.ph_stat} (N={r.n_samples}, mean={r.running_mean})"
                )
            lines.append("Consider re-running BO sweep or manual review.")
            try:
                await self._slack._post(self._b2_channel(), text="\n".join(lines))
            except Exception as e:
                logger.warning("b2_drift_slack_error", error=str(e))

    def _b2_channel(self) -> str:
        """Channel for B2 Slack messages."""
        from arbo.dashboard.slack_promotion import STRATEGY_CHANNELS
        return STRATEGY_CHANNELS.get(
            "B2",
            getattr(self._slack, "_channel_id", "") or "C0APX4K8Z2N",
        )
