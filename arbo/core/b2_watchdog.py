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
        # Drift dedup is now persisted via `arbo.utils.alert_state` so
        # restarts don't re-emit the same alert. See
        # docs/KNOWLEDGE_BASE.md ("B2 Drift Alert" row) for the
        # suppression rules the operator should expect.

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
        """One eval pass — autochallenger + promotion + drift + incubate."""
        now = time.time()
        if now - self._last_eval_ts < _EVAL_INTERVAL_S:
            return
        self._last_eval_ts = now

        await self._autochallenger_cycle()
        await self._incubate_cycle()  # canary escalation/revert BEFORE promotion flags
        await self._promotion_cycle()
        await self._drift_cycle()

    async def _incubate_cycle(self) -> None:
        """Canary decision: escalate incubating variant → champion, or
        revert → challenger, based on LIVE performance.

        See docs/CANARY_PROMOTION_SPEC.md for the full spec. Summary:
        - fetch live PnL series per-variant from `paper_trades` (filtered
          by trade_details->>'variant_id', status='sold',
          notes NOT LIKE 'pre_reset%%')
        - if both sides have N ≥ MIN_PAIRED_N_LIVE, run block-bootstrap
          P(incubate_mean > champion_mean)
        - escalate to champion if P > P_ESCALATE (default 0.70)
        - revert to challenger if P < P_REVERT (default 0.30) AND at
          least MIN_N_BEFORE_REVERT live trades seen
        - otherwise: hold, accumulate more data
        """
        try:
            from arbo.core.variant_pool import get_champion, load_variants
            from arbo.core import pool_manager
            from arbo.core.promotion_engine import (
                _block_bootstrap_p_better,
                BOOTSTRAP_RESAMPLES,
                BOOTSTRAP_BLOCK_SIZE,
            )
        except Exception as e:
            logger.warning("b2_incubate_import_error", error=str(e))
            return

        # Thresholds — conservative defaults. B2 generates only a few
        # live trades per day, so escalation won't typically fire in
        # under a week. That's intentional: we want real confidence.
        MIN_PAIRED_N_LIVE = 15
        P_ESCALATE = 0.70
        P_REVERT = 0.30
        MIN_N_BEFORE_REVERT = 20

        pool = load_variants(self.STRATEGY_NAME)
        incubate = next((v for v in pool if v.status == "incubate"), None)
        if incubate is None:
            return  # no canary active
        champion = get_champion(self.STRATEGY_NAME)
        if champion is None:
            logger.warning("b2_incubate_no_champion")
            return

        # Fetch live per-variant PnL series
        try:
            from arbo.utils.db import get_session_factory
            import sqlalchemy as sa
        except Exception as e:
            logger.warning("b2_incubate_db_import_error", error=str(e))
            return

        factory = get_session_factory()
        async with factory() as session:
            ch_series = await self._fetch_live_pnl(
                session, sa, incubate.variant_id,
            )
            cp_series = await self._fetch_live_pnl(
                session, sa, champion.variant_id,
            )

        if len(ch_series) < MIN_PAIRED_N_LIVE or len(cp_series) < MIN_PAIRED_N_LIVE:
            logger.info(
                "b2_incubate_insufficient_live_data",
                incubate=incubate.variant_id,
                n_incubate=len(ch_series),
                n_champion=len(cp_series),
                min_required=MIN_PAIRED_N_LIVE,
            )
            return

        # Block bootstrap P(incubate better than champion)
        p_better = _block_bootstrap_p_better(
            ch_series, cp_series,
            resamples=BOOTSTRAP_RESAMPLES,
            block=BOOTSTRAP_BLOCK_SIZE,
        )
        mean_ch = sum(ch_series) / len(ch_series)
        mean_cp = sum(cp_series) / len(cp_series)
        logger.info(
            "b2_incubate_eval",
            incubate=incubate.variant_id,
            champion=champion.variant_id,
            n_incubate=len(ch_series),
            n_champion=len(cp_series),
            mean_incubate=round(mean_ch, 4),
            mean_champion=round(mean_cp, 4),
            p_better=round(p_better, 3),
        )

        # Decision
        if p_better >= P_ESCALATE:
            ok, reason = await pool_manager.promote(
                incubate.variant_id, self.STRATEGY_NAME, approved_by="watchdog_auto",
            )
            if ok:
                await self._post_incubate_decision(
                    "escalated", incubate.variant_id, champion.variant_id,
                    len(ch_series), len(cp_series), p_better, mean_ch, mean_cp,
                )
            else:
                logger.warning("b2_incubate_escalate_failed", reason=reason)
        elif p_better <= P_REVERT and len(ch_series) >= MIN_N_BEFORE_REVERT:
            ok, reason = await pool_manager.revert_incubate_to_challenger(
                incubate.variant_id, self.STRATEGY_NAME,
                reason=f"live_p_better={p_better:.2f} below {P_REVERT} after N={len(ch_series)}",
                decided_by="watchdog_auto",
            )
            if ok:
                await self._post_incubate_decision(
                    "reverted", incubate.variant_id, champion.variant_id,
                    len(ch_series), len(cp_series), p_better, mean_ch, mean_cp,
                )
            else:
                logger.warning("b2_incubate_revert_failed", reason=reason)
        else:
            logger.info(
                "b2_incubate_holding",
                p_better=round(p_better, 3),
                n_incubate=len(ch_series),
                n_champion=len(cp_series),
            )

    async def _fetch_live_pnl(
        self, session: Any, sa: Any, variant_id: str,
    ) -> list[float]:
        """Ordered live actual_pnl series for a B2 variant.

        Filters: strategy='B2', status='sold', trade_details.variant_id
        matches, and NOT pre_reset. Ordered by resolved_at (natural
        time order) for block-bootstrap validity.
        """
        result = await session.execute(
            sa.text("""
                SELECT actual_pnl
                FROM paper_trades
                WHERE strategy = 'B2'
                  AND status = 'sold'
                  AND actual_pnl IS NOT NULL
                  AND COALESCE(notes, '') NOT LIKE 'pre_reset%%'
                  AND trade_details ->> 'variant_id' = :v
                ORDER BY resolved_at ASC NULLS LAST
            """),
            {"v": variant_id},
        )
        return [float(row[0]) for row in result.fetchall()]

    async def _post_incubate_decision(
        self,
        decision: str,
        incubate_id: str,
        champion_id: str,
        n_incubate: int,
        n_champion: int,
        p_better: float,
        mean_incubate: float,
        mean_champion: float,
    ) -> None:
        if self._slack is None:
            return
        if decision == "escalated":
            title = ":trophy: B2 CANARY ESCALATED — new champion"
            details = (
                f"`{incubate_id}` promoted to champion.\n"
                f"Replaced: `{champion_id}`"
            )
        else:
            title = ":arrows_counterclockwise: B2 CANARY REVERTED"
            details = (
                f"`{incubate_id}` reverted to challenger (shadow only).\n"
                f"Champion `{champion_id}` remains active."
            )
        msg = (
            f"{title}\n\n"
            f"{details}\n\n"
            f"*Live paired stats:*\n"
            f"• `{incubate_id}`: N={n_incubate}, mean=${mean_incubate:.4f}\n"
            f"• `{champion_id}`: N={n_champion}, mean=${mean_champion:.4f}\n"
            f"• P(incubate better): `{p_better:.3f}` (block-bootstrap 1000)"
        )
        try:
            await self._slack._post(self._b2_channel(), text=msg)
        except Exception as e:
            logger.warning("b2_incubate_slack_error", error=str(e))

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
        """Detect promotion candidates → auto-approve or post to Slack.

        Design rules that reduce decision fatigue for the operator:
        1. If any variant is already in `incubate`, skip emitting NEW
           candidates entirely — only one canary at a time (matches
           `pool_manager.promote_to_incubate` guard).
        2. Rank candidates by `p_better` desc; emit only the TOP-1.
        3. If the top candidate's evidence is very strong
           (see AUTO_APPROVE_* thresholds in promotion_engine), call
           `promote_to_incubate` directly — operator sees a notification,
           not a decision.
        4. Otherwise post a single-option Slack message for manual
           approval. Dedup per challenger for 24h.
        """
        try:
            from arbo.core.promotion_engine import PromotionEngine, MIN_P_BETTER
            from arbo.core.variant_pool import load_variants
            from arbo.core import pool_manager
            from arbo.dashboard.slack_promotion import post_candidate
        except Exception as e:
            logger.warning("b2_promotion_import_error", error=str(e))
            return

        # One-canary-at-a-time guard: if an incubate already exists,
        # don't emit any other candidates (they would fail downstream).
        pool = load_variants(self.STRATEGY_NAME)
        active_incubate = next(
            (v for v in pool if v.status == "incubate"), None,
        )
        if active_incubate is not None:
            logger.debug(
                "b2_promotion_skipped_incubate_active",
                incubating=active_incubate.variant_id,
            )
            return

        cands = await PromotionEngine(self.STRATEGY_NAME).evaluate()
        # Filter out rejected + below user-approval threshold, then
        # take the TOP-1 by p_better.
        eligible = [
            c for c in cands
            if c.reject_reason is None
            and not (c.tier == 1 and c.p_better < MIN_P_BETTER)
        ]
        if not eligible:
            return
        eligible.sort(key=lambda c: c.p_better, reverse=True)
        top = eligible[0]

        now = time.time()
        last_ts = self._promotion_posted.get(top.challenger_id, 0.0)
        if now - last_ts < 24 * 3600:
            return  # already acted/posted within 24h

        # Auto-approve path — strong evidence, safe change, no human
        # in the loop. Pool manager will still apply the one-incubate
        # guard, so worst case this no-ops and logs.
        if top.auto_approve:
            ok, reason = await pool_manager.promote_to_incubate(
                top.challenger_id, self.STRATEGY_NAME,
                capital_pct=0.20,
                approved_by="system_auto",
            )
            if ok:
                self._promotion_posted[top.challenger_id] = now
                logger.info(
                    "b2_promotion_auto_approved",
                    challenger=top.challenger_id,
                    p_better=top.p_better,
                    dsr_delta=top.dsr_delta,
                    n=top.n_challenger,
                )
                if self._slack is not None:
                    msg = (
                        f":robot_face: *B2 AUTO-APPROVED CANARY*\n\n"
                        f"System promoted `{top.challenger_id}` → *incubate* "
                        f"(20% live capital) without manual review.\n\n"
                        f"Why: very strong shadow evidence — "
                        f"P(better)={top.p_better:.2f}, "
                        f"Sharpe Δ={top.dsr_delta:+.2f}, "
                        f"N={top.n_challenger}.\n\n"
                        f"Watchdog will auto-escalate or revert based on "
                        f"≥15 paired live trades."
                    )
                    try:
                        await self._slack._post(self._b2_channel(), text=msg)
                    except Exception as e:
                        logger.warning("b2_auto_approve_slack_error", error=str(e))
            else:
                logger.warning(
                    "b2_promotion_auto_approve_failed",
                    challenger=top.challenger_id,
                    reason=reason,
                )
            return

        # Manual approval path — post single card to Slack
        ok = await post_candidate(self._slack, top)
        if ok:
            self._promotion_posted[top.challenger_id] = now
            logger.info(
                "b2_promotion_emitted",
                challenger=top.challenger_id,
                tier=top.tier,
                p_better=top.p_better,
                also_eligible_count=len(eligible) - 1,
            )

    async def _drift_cycle(self) -> None:
        """Detect shadow-PnL drift → interpret → react + post human alert.

        Design (evolved from dedup-only to analysis+action):
          1. Run Page-Hinkley across variants.
          2. Compare current means to the previous alert snapshot
             (persistent, via alert_state) to compute deltas.
          3. Classify severity:
               - minor: all variants down <10%
               - moderate: champion down 10-25%, or any variant >25%
               - severe: champion down >25% AND canary also drifting
          4. Autonomous response:
               - minor/moderate: log + post human Slack interpretation
               - severe + live regression also present: flag for
                 operator attention in the message; do NOT auto-pause
                 (that's a capital decision — needs operator)
          5. Check the canary: if champion drifts but incubating
             variant does NOT drift, note this as evidence the
             canary is a regime-robust candidate. The incubate_cycle
             (not this one) still makes the actual promotion call —
             drift only adds context.

        Slack message is Gemini-synthesised into plain Czech with
        interpretation + recommended action. If LLM unavailable, a
        tight human-readable template fallback is used.
        """
        try:
            from arbo.core.drift_monitor import evaluate_strategy_drift
            from arbo.core.variant_pool import load_variants, get_champion
        except Exception as e:
            logger.warning("b2_drift_import_error", error=str(e))
            return

        from arbo.utils.alert_state import (
            should_alert, record_alert, clear_alert,
            _read_state, _write_state,
        )

        results = await evaluate_strategy_drift(self.STRATEGY_NAME)
        firing = sorted(
            [r for r in results if r.firing], key=lambda r: r.variant_id,
        )
        if not firing:
            clear_alert("b2_drift")
            return

        fingerprint = "|".join(
            f"{r.variant_id}:{r.n_samples}" for r in firing
        )
        fire = should_alert("b2_drift", fingerprint, cooldown_s=24 * 3600)

        for r in firing:
            logger.warning(
                "b2_drift_detected",
                variant_id=r.variant_id,
                ph_stat=r.ph_stat,
                n_samples=r.n_samples,
                dedup_suppressed=not fire,
            )
        if not fire:
            return

        # --- Historical means (for delta computation) ---
        # We persist the previous means snapshot alongside the fingerprint
        # under a side-key so human messages can say "průměr poklesl z
        # $X na $Y" instead of just the current number.
        state = _read_state()
        prev = state.get("b2_drift_means", {}) or {}
        current_means = {r.variant_id: float(r.running_mean) for r in firing}
        current_ns = {r.variant_id: int(r.n_samples) for r in firing}
        deltas: dict[str, tuple[float, float]] = {}  # id → (prev_mean, curr_mean)
        for vid, m in current_means.items():
            pm = prev.get(vid, {}).get("mean") if isinstance(prev.get(vid), dict) else None
            if pm is not None:
                deltas[vid] = (float(pm), float(m))

        # Canary context
        pool = load_variants(self.STRATEGY_NAME)
        champion = get_champion(self.STRATEGY_NAME)
        incubate = next((v for v in pool if v.status == "incubate"), None)
        firing_ids = {r.variant_id for r in firing}
        canary_in_drift = incubate is not None and incubate.variant_id in firing_ids
        champion_in_drift = champion is not None and champion.variant_id in firing_ids

        # Severity classification
        champion_drop_pct = None
        if champion and champion.variant_id in deltas:
            pm, cm = deltas[champion.variant_id]
            champion_drop_pct = (cm - pm) / pm if pm > 0 else 0.0

        if (
            champion_drop_pct is not None
            and champion_drop_pct < -0.25
            and canary_in_drift
        ):
            severity = "severe"
        elif champion_drop_pct is not None and champion_drop_pct < -0.10:
            severity = "moderate"
        else:
            severity = "minor"

        # Log everything structured for retrospective
        logger.warning(
            "b2_drift_analyzed",
            severity=severity,
            champion_drop_pct=(
                round(champion_drop_pct, 3)
                if champion_drop_pct is not None else None
            ),
            champion_in_drift=champion_in_drift,
            canary_in_drift=canary_in_drift,
            canary_id=incubate.variant_id if incubate else None,
        )

        record_alert("b2_drift", fingerprint)
        # Persist means snapshot for NEXT drift delta comparison.
        state = _read_state()
        state["b2_drift_means"] = {
            vid: {"mean": m, "n": current_ns.get(vid, 0)}
            for vid, m in current_means.items()
        }
        _write_state(state)

        if self._slack is None:
            return

        # Compose the Slack message. Prefer Gemini synthesis; fall
        # back to a tight human-readable template.
        text = self._render_drift_message(
            firing=firing,
            deltas=deltas,
            severity=severity,
            champion_id=champion.variant_id if champion else None,
            canary_id=incubate.variant_id if incubate else None,
            canary_in_drift=canary_in_drift,
        )

        try:
            await self._slack._post(self._b2_channel(), text=text)
        except Exception as e:
            logger.warning("b2_drift_slack_error", error=str(e))

    def _render_drift_message(
        self,
        *,
        firing: list[Any],
        deltas: dict[str, tuple[float, float]],
        severity: str,
        champion_id: str | None,
        canary_id: str | None,
        canary_in_drift: bool,
    ) -> str:
        """Produce a plain-Czech drift message. Tries Gemini first."""
        # Build a compact data block for the LLM prompt
        data_lines = []
        for r in firing:
            delta_str = ""
            if r.variant_id in deltas:
                prev_m, curr_m = deltas[r.variant_id]
                delta_str = f" (minule {prev_m:.4f})"
            role = []
            if r.variant_id == champion_id:
                role.append("champion")
            if r.variant_id == canary_id:
                role.append("kanárek")
            role_str = f" [{', '.join(role)}]" if role else ""
            data_lines.append(
                f"- {r.variant_id}{role_str}: {r.n_samples} obchodů, "
                f"průměr ${float(r.running_mean):.4f}{delta_str}"
            )
        canary_note = "kanárek v driftu" if canary_in_drift else (
            "kanárek drží stabilně" if canary_id else "žádný kanárek neinkubuje"
        )

        prompt = f"""Napiš krátkou Slack zprávu česky, uživatelsky přátelsky, o tom
že u strategie B2 systém detekoval drift. Žádné technické zkratky, ŽÁDNÉ "Page-Hinkley",
"fingerprint", "PH=X". Vysvětli co se stalo a co systém dělá.

KONTEXT:
Severita: {severity} (minor = do 10%, moderate = 10-25%, severe = >25% drop + kanárek taky v driftu)
Drift u těchto variant:
{chr(10).join(data_lines)}
Kanárek: {canary_note}

POŽADAVEK NA VÝSTUP — přesně v tomto formátu:

Úvodní řádek: ikona podle severity + krátký jednovětý titulek.
- minor → 📉 B2: mírné zhoršení
- moderate → ⚠️ B2: znatelné zhoršení
- severe → 🚨 B2: výrazné zhoršení, potřeba pozornost

*Co se stalo*
2 věty, konkrétně čísla (pokles průměrného zisku z $X na $Y), přeložené do lidské řeči.

*Co to znamená*
1-2 věty — co drift říká o strategii v aktuálním tržním režimu.

*Co dělá systém*
1-2 věty — co autonomně udělá nebo sleduje. Pokud je kanárek stabilní, zdůrazni to jako dobrou zprávu.

*Co musíš udělat*
"Nic — systém to hlídá." NEBO konkrétní akce. Pouze pokud severity=severe navrhni zvážit pauzu.

Max 150 slov. Žádné bullet pointy, hezká próza."""

        # Try LLM
        try:
            import os
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.3, "max_output_tokens": 4000},
                    request_options={"timeout": 30},
                )
                text = (resp.text or "").strip()
                if text:
                    return text
        except Exception as e:
            logger.debug("b2_drift_llm_failed", error=str(e))

        # Fallback: tight template
        icon = {"minor": "📉", "moderate": "⚠️", "severe": "🚨"}.get(severity, "📊")
        header = {
            "minor": f"{icon} *B2: mírné zhoršení*",
            "moderate": f"{icon} *B2: znatelné zhoršení*",
            "severe": f"{icon} *B2: výrazné zhoršení*",
        }[severity]
        lines = [header, ""]
        if champion_id and champion_id in deltas:
            pm, cm = deltas[champion_id]
            pct = (cm - pm) / pm * 100 if pm else 0
            lines.append(
                f"Průměrný zisk strategie poklesl z `${pm:.3f}` na `${cm:.3f}` "
                f"({pct:+.0f}%) za obchod."
            )
        if canary_id and not canary_in_drift:
            lines.append(
                f"Kanárek `{canary_id}` se v tomto zhoršení drží — dobrý signál."
            )
        lines.append(
            "_Systém sleduje dál. Detail v ranním briefingu._"
        )
        return "\n".join(lines)

    def _b2_channel(self) -> str:
        """Channel for B2 Slack messages."""
        from arbo.dashboard.slack_promotion import STRATEGY_CHANNELS
        return STRATEGY_CHANNELS.get(
            "B2",
            getattr(self._slack, "_channel_id", "") or "C0APX4K8Z2N",
        )
