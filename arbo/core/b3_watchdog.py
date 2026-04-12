"""B3 Watchdog — Autonomous strategy optimizer daemon.

Runs as an asyncio background task. Periodically evaluates B3 performance
via b3_metrics, detects anomalies, calls Gemini Flash for root cause
analysis and parameter recommendations, and autonomously applies changes
within safety bounds via adaptive_config.

Key design:
- Evaluates every 50 trades OR 6 hours (whichever comes first)
- 3-tier autonomy: Tier 1 (apply), Tier 2 (apply+flag), Tier 3 (escalate)
- Auto-revert: checks each change after 50 trades
- Self-learning: logs decisions + outcomes

See: docs/B3_WATCHDOG_SPEC.md
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from arbo.core.adaptive_config import PARAM_BOUNDS, AdaptiveConfig
from arbo.core.b3_metrics import B3MetricsSnapshot, fetch_b3_metrics
from arbo.utils.logger import get_logger

logger = get_logger("b3_watchdog")

# Evaluation triggers
_EVAL_INTERVAL_S = 6 * 3600  # 6 hours
_EVAL_TRADE_THRESHOLD = 50   # Or after 50 new trades
_MIN_TRADES_FOR_EVAL = 30    # Minimum trades before any action
_AUTO_REVERT_WINDOW = 50     # Trades after change before evaluation
_AUTO_REVERT_THRESHOLD = 0.05  # 5pp WR drop triggers revert
_MAX_CONSECUTIVE_REVERTS = 3   # 3 reverts → observation mode
_OBSERVATION_MODE_TRADES = 200  # Trades in observation mode

# Anomaly thresholds
_WR_DROP_PAPER_THRESHOLD = 0.10    # 10pp drop from baseline
_WR_DROP_LIVE_THRESHOLD = 0.10     # 10pp drop for live
_CONSECUTIVE_LOSSES_PAPER = 8
_CONSECUTIVE_LOSSES_LIVE = 5
_ECE_THRESHOLD = 0.15
_PSI_THRESHOLD = 0.20
_DAILY_PNL_CRASH = -20.0  # $20 daily loss


class B3Watchdog:
    """Autonomous B3 strategy optimizer.

    Usage::

        watchdog = B3Watchdog(
            session_factory=get_session_factory(),
            adaptive_config=adaptive_config,
            gemini_agent=gemini_agent,
            slack_bot=slack_bot,
            ta_provider=ta_provider,
        )
        asyncio.create_task(watchdog.run())
    """

    def __init__(
        self,
        session_factory: Any,
        adaptive_config: AdaptiveConfig,
        gemini_agent: Any | None = None,
        slack_bot: Any | None = None,
        ta_provider: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._config = adaptive_config
        self._gemini = gemini_agent
        self._slack = slack_bot
        self._ta_provider = ta_provider

        self._running = False
        self._last_eval_time: float = 0
        self._last_eval_trade_count: int = 0
        self._baseline: B3MetricsSnapshot | None = None
        self._observation_mode = False
        self._observation_start_trades = 0

    async def run(self) -> None:
        """Main daemon loop. Runs until stopped."""
        self._running = True
        logger.info("b3_watchdog_started")

        # Wait 30 min before first evaluation (let trades accumulate)
        for _ in range(1800):
            if not self._running:
                return
            await asyncio.sleep(1)

        while self._running:
            try:
                await self._eval_cycle()
            except Exception as e:
                logger.error("b3_watchdog_cycle_error", error=str(e))

            # Sleep in 60s chunks for responsive shutdown
            for _ in range(60):
                if not self._running:
                    return
                await asyncio.sleep(60)  # Check every 60 min; cycle itself checks thresholds

    async def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False

    async def _eval_cycle(self) -> None:
        """Single evaluation cycle: fetch metrics → detect → decide → apply."""
        metrics = await fetch_b3_metrics(self._session_factory)
        if metrics is None:
            return

        now = time.time()
        time_elapsed = now - self._last_eval_time > _EVAL_INTERVAL_S
        trades_elapsed = metrics.total_trades - self._last_eval_trade_count >= _EVAL_TRADE_THRESHOLD

        if not time_elapsed and not trades_elapsed:
            return  # Not yet time to evaluate

        self._last_eval_time = now
        self._last_eval_trade_count = metrics.total_trades

        # Set baseline on first evaluation
        if self._baseline is None:
            self._baseline = metrics
            logger.info(
                "b3_watchdog_baseline_set",
                trades=metrics.total_trades,
                wr_paper=metrics.rolling_wr_paper,
                wr_live=metrics.rolling_wr_live,
            )
            await self._send_info_report(metrics)
            return

        # Check observation mode (after 3 consecutive reverts)
        if self._observation_mode:
            trades_since = metrics.total_trades - self._observation_start_trades
            if trades_since < _OBSERVATION_MODE_TRADES:
                logger.info(
                    "b3_watchdog_observation_mode",
                    trades_remaining=_OBSERVATION_MODE_TRADES - trades_since,
                )
                await self._send_info_report(metrics)
                return
            self._observation_mode = False
            logger.info("b3_watchdog_observation_mode_ended")

        # 1. Auto-revert check for active changes
        await self._check_auto_reverts(metrics)

        # 2. Detect anomalies
        anomalies = self._detect_anomalies(metrics)

        if not anomalies:
            # No anomalies — send periodic info report
            await self._send_info_report(metrics)
            return

        # 3. Call Gemini for analysis (if available)
        if self._gemini is not None and metrics.total_trades >= _MIN_TRADES_FOR_EVAL:
            decision = await self._get_gemini_decision(metrics, anomalies)
            if decision:
                await self._execute_decision(decision, metrics)
        else:
            # No Gemini — just report anomalies
            await self._send_anomaly_report(metrics, anomalies)

    def _detect_anomalies(self, metrics: B3MetricsSnapshot) -> list[dict[str, Any]]:
        """Detect anomalies by comparing current metrics to baseline."""
        anomalies: list[dict[str, Any]] = []
        bl = self._baseline
        if bl is None:
            return anomalies

        # T1: Rolling WR drop (paper)
        wr_drop_paper = bl.rolling_wr_paper - metrics.rolling_wr_paper
        if wr_drop_paper > _WR_DROP_PAPER_THRESHOLD and metrics.rolling_n >= 50:
            anomalies.append({
                "trigger": "T1_wr_drop_paper",
                "severity": "WARNING",
                "current": metrics.rolling_wr_paper,
                "baseline": bl.rolling_wr_paper,
                "drop": round(wr_drop_paper, 4),
            })

        # T2: Rolling WR drop (live)
        wr_drop_live = bl.rolling_wr_live - metrics.rolling_wr_live
        if wr_drop_live > _WR_DROP_LIVE_THRESHOLD and metrics.rolling_n >= 30:
            anomalies.append({
                "trigger": "T2_wr_drop_live",
                "severity": "CRITICAL",
                "current": metrics.rolling_wr_live,
                "baseline": bl.rolling_wr_live,
                "drop": round(wr_drop_live, 4),
            })

        # T3: Consecutive losses (live)
        if metrics.max_consecutive_losses_live >= _CONSECUTIVE_LOSSES_LIVE:
            anomalies.append({
                "trigger": "T3_consecutive_losses_live",
                "severity": "CRITICAL",
                "streak": metrics.max_consecutive_losses_live,
            })

        # T4: Consecutive losses (paper)
        if metrics.max_consecutive_losses_paper >= _CONSECUTIVE_LOSSES_PAPER:
            anomalies.append({
                "trigger": "T4_consecutive_losses_paper",
                "severity": "WARNING",
                "streak": metrics.max_consecutive_losses_paper,
            })

        # T9: ECE calibration breakdown
        if metrics.ece > _ECE_THRESHOLD and metrics.rolling_n >= 50:
            anomalies.append({
                "trigger": "T9_calibration_breakdown",
                "severity": "CRITICAL",
                "ece": metrics.ece,
            })

        # T6/T8/T14: PSI drift
        for feature, psi_val in metrics.psi_scores.items():
            if psi_val > _PSI_THRESHOLD:
                anomalies.append({
                    "trigger": f"T_psi_drift_{feature}",
                    "severity": "WARNING",
                    "feature": feature,
                    "psi": psi_val,
                })

        # T13: Daily PnL crash
        if metrics.daily_pnl:
            latest_day = metrics.daily_pnl[0]
            if latest_day.get("pnl", 0) < _DAILY_PNL_CRASH:
                anomalies.append({
                    "trigger": "T13_daily_pnl_crash",
                    "severity": "CRITICAL",
                    "daily_pnl": latest_day["pnl"],
                })

        # T5: Regime WR collapse (any bucket < 40% with N > 20)
        for regime_name, regime_data in [
            ("sigma", metrics.sigma_regime),
            ("velocity", metrics.velocity_regime),
            ("spread", metrics.spread_regime),
            ("adx", metrics.adx_regime),
        ]:
            for bucket, stats in regime_data.items():
                if stats["n"] >= 20 and stats["wr"] < 0.40:
                    anomalies.append({
                        "trigger": f"T5_regime_collapse_{regime_name}_{bucket}",
                        "severity": "WARNING",
                        "bucket": bucket,
                        "wr": stats["wr"],
                        "n": stats["n"],
                    })

        return anomalies

    async def _get_gemini_decision(
        self,
        metrics: B3MetricsSnapshot,
        anomalies: list[dict],
    ) -> dict[str, Any] | None:
        """Ask Gemini Flash for analysis and decision."""
        context = self._build_gemini_context(metrics, anomalies)
        prompt = _GEMINI_SYSTEM_PROMPT + "\n\nDATA:\n" + json.dumps(context, default=str, indent=2)

        try:
            result = await self._gemini.raw_query(prompt)
            if result and isinstance(result, dict):
                logger.info(
                    "b3_watchdog_gemini_response",
                    verdict=result.get("verdict"),
                    confidence=result.get("confidence"),
                )
                return result
        except Exception as e:
            logger.warning("b3_watchdog_gemini_error", error=str(e))

        return None

    def _build_gemini_context(
        self,
        metrics: B3MetricsSnapshot,
        anomalies: list[dict],
    ) -> dict[str, Any]:
        """Build context packet for Gemini analysis."""
        ta_context = {}
        if self._ta_provider:
            ta = self._ta_provider.get("BTCUSDT")
            if ta:
                ta_context = {
                    "btc_5m": {"rsi": ta.rsi_5m, "adx": ta.adx_5m, "regime": ta.adx_regime},
                    "btc_1h": {"rsi": ta.rsi_1h, "adx": ta.adx_1h},
                    "multi_tf_aligned": ta.multi_tf_aligned,
                }

        return {
            "anomalies": anomalies,
            "current_metrics": {
                "rolling_n": metrics.rolling_n,
                "wr_paper": metrics.rolling_wr_paper,
                "wr_live": metrics.rolling_wr_live,
                "avg_pnl": metrics.rolling_avg_pnl,
                "sharpe": metrics.rolling_sharpe,
                "ece": metrics.ece,
            },
            "baseline_metrics": {
                "wr_paper": self._baseline.rolling_wr_paper if self._baseline else None,
                "wr_live": self._baseline.rolling_wr_live if self._baseline else None,
            },
            "regime_breakdown": {
                "sigma": metrics.sigma_regime,
                "velocity": metrics.velocity_regime,
                "spread": metrics.spread_regime,
                "adx": metrics.adx_regime,
                "rsi": metrics.rsi_regime,
            },
            "psi_drift": metrics.psi_scores,
            "ta_context": ta_context,
            "market_structure": {
                "avg_spread": metrics.avg_spread,
                "avg_cl_delta": metrics.avg_cl_delta,
                "avg_fill_price": metrics.avg_fill_price,
                "avg_liquidity": metrics.avg_liquidity,
            },
            "current_overrides": self._config.get_all_overrides(),
            "recent_decisions": [
                {
                    "param": c.param_name,
                    "old": c.old_value,
                    "new": c.new_value,
                    "reason": c.reason,
                    "status": c.status,
                }
                for c in self._config.get_change_log(limit=5)
            ],
            "available_params": {
                name: {"tier": b.tier, "range": [b.autonomous_min, b.autonomous_max]}
                for name, b in PARAM_BOUNDS.items()
            },
        }

    async def _execute_decision(
        self,
        decision: dict[str, Any],
        metrics: B3MetricsSnapshot,
    ) -> None:
        """Execute a Gemini decision within safety bounds."""
        verdict = decision.get("verdict", "MONITOR")

        if verdict == "MONITOR":
            logger.info("b3_watchdog_verdict_monitor", reason=decision.get("root_cause"))
            await self._send_info_report(metrics, note=decision.get("root_cause"))
            return

        if verdict == "ESCALATE":
            logger.info("b3_watchdog_verdict_escalate", reason=decision.get("root_cause"))
            await self._send_escalation(metrics, decision)
            return

        if verdict == "REVERT":
            action = decision.get("action", {})
            param = action.get("param")
            if param:
                reverted = self._config.revert(param, reason=decision.get("root_cause", "Gemini revert"))
                if reverted:
                    await self._send_action_report(metrics, decision, "REVERTED")
            return

        if verdict == "APPLY":
            action = decision.get("action", {})
            param = action.get("param")
            new_value = action.get("new_value")

            if not param or new_value is None:
                logger.warning("b3_watchdog_apply_missing_param", decision=decision)
                return

            # Cascade prevention: max 1 active Tier 1 change
            bounds = PARAM_BOUNDS.get(param)
            if bounds and bounds.tier == 1 and self._config.has_active_tier1_change():
                logger.info(
                    "b3_watchdog_cascade_blocked",
                    param=param,
                    msg="Another Tier 1 change is being evaluated",
                )
                return

            # Apply change
            success = self._config.set(
                param=param,
                value=float(new_value),
                reason=decision.get("root_cause", "Gemini recommendation"),
                trade_count=metrics.total_trades,
                wr=metrics.rolling_wr_paper,
            )

            if success:
                await self._send_action_report(metrics, decision, "APPLIED")

                # Check consecutive reverts → observation mode
                if self._config.consecutive_reverts() >= _MAX_CONSECUTIVE_REVERTS:
                    self._observation_mode = True
                    self._observation_start_trades = metrics.total_trades
                    logger.warning(
                        "b3_watchdog_observation_mode_entered",
                        msg=f"{_MAX_CONSECUTIVE_REVERTS} consecutive reverts — entering observation mode",
                    )

    async def _check_auto_reverts(self, metrics: B3MetricsSnapshot) -> None:
        """Check if any active changes should be auto-reverted."""
        for change in self._config.get_active_changes():
            trades_since = metrics.total_trades - change.trade_count_at_change
            if trades_since < _AUTO_REVERT_WINDOW:
                continue  # Not enough trades yet

            # Compare WR after vs before change
            wr_now = metrics.rolling_wr_paper
            wr_at_change = change.wr_at_change
            wr_drop = wr_at_change - wr_now

            if wr_drop > _AUTO_REVERT_THRESHOLD:
                # Auto-revert
                self._config.revert(
                    change.param_name,
                    reason=f"Auto-revert: WR dropped {wr_drop:.1%} after {trades_since} trades",
                )
                logger.warning(
                    "b3_watchdog_auto_revert",
                    param=change.param_name,
                    old=change.new_value,
                    reverted_to=change.old_value,
                    wr_drop=f"{wr_drop:.1%}",
                )
                await self._send_revert_report(change, wr_drop, metrics)
            elif abs(wr_drop) <= 0.02:
                # Neutral — extend evaluation
                pass
            else:
                # Positive — promote
                self._config.promote(change.param_name)
                logger.info(
                    "b3_watchdog_promoted",
                    param=change.param_name,
                    value=change.new_value,
                    wr_improvement=f"{-wr_drop:.1%}",
                )

    # ── Slack Reporting ──────────────────────────────────────────────

    async def _send_info_report(
        self, metrics: B3MetricsSnapshot, note: str | None = None,
    ) -> None:
        """Send periodic INFO report to Slack."""
        if self._slack is None:
            return

        lines = [
            "━━━ B3 Watchdog Report ━━━",
            f"Period: posledních {metrics.rolling_n} tradů",
            "",
            f"Paper: {metrics.rolling_n}t | WR {metrics.rolling_wr_paper:.1%} "
            f"| Avg PnL ${metrics.rolling_avg_pnl:.2f} | Sharpe {metrics.rolling_sharpe:.1f}",
        ]
        if metrics.rolling_wr_live > 0:
            lines.append(f"Live: WR {metrics.rolling_wr_live:.1%}")

        # Regime summary
        lines.append("")
        lines.append("Sigma Regime:")
        for name, stats in metrics.sigma_regime.items():
            if stats["n"] > 0:
                lines.append(f"  {name}: {stats['n']}t, {stats['wr']:.0%} WR, ${stats['pnl']:.0f}")

        # TA regime if available
        if metrics.adx_regime:
            lines.append("ADX Regime:")
            for name, stats in metrics.adx_regime.items():
                if stats["n"] > 0:
                    lines.append(f"  {name}: {stats['n']}t, {stats['wr']:.0%} WR")

        # PSI drift
        if metrics.psi_scores:
            drifts = [f"{k}: {v:.2f}" for k, v in metrics.psi_scores.items() if v > 0.10]
            if drifts:
                lines.append(f"\nPSI Drift: {' | '.join(drifts)}")

        # Active overrides
        overrides = self._config.get_all_overrides()
        if overrides:
            lines.append(f"\nActive overrides: {overrides}")

        lines.append(f"\nECE: {metrics.ece:.3f} | Status: {'OBSERVATION' if self._observation_mode else 'ACTIVE'}")

        if note:
            lines.append(f"\nNote: {note}")

        try:
            await self._slack.send_message("\n".join(lines))
        except Exception as e:
            logger.warning("b3_watchdog_slack_error", error=str(e))

    async def _send_action_report(
        self,
        metrics: B3MetricsSnapshot,
        decision: dict[str, Any],
        action_type: str,
    ) -> None:
        """Send autonomous action report to Slack."""
        if self._slack is None:
            return

        action = decision.get("action", {})
        param = action.get("param", "N/A")
        new_value = action.get("new_value", "?")

        # Prefer actual old_value from adaptive_config audit log
        # (Gemini sometimes omits old_value from its response)
        old_value = action.get("old_value")
        if old_value is None and param != "N/A":
            latest = self._config.get_active_change_for(param)
            if latest is not None:
                old_value = latest.old_value
            else:
                # Fall back to current default
                old_value = self._config._get_default(param)

        lines = [
            f"━━━ B3 Watchdog — AUTONOMOUS {action_type} ━━━",
            f"Trigger: {decision.get('root_cause', 'N/A')[:100]}",
            f"Action: {param}: {old_value} → {new_value}",
            f"Confidence: {decision.get('confidence', 'N/A')}",
            f"Auto-revert check: po {_AUTO_REVERT_WINDOW} tradech",
            f"\nCurrent WR: paper {metrics.rolling_wr_paper:.1%} | live {metrics.rolling_wr_live:.1%}",
        ]

        try:
            await self._slack.send_message("\n".join(lines))
        except Exception as e:
            logger.warning("b3_watchdog_slack_error", error=str(e))

    async def _send_anomaly_report(
        self,
        metrics: B3MetricsSnapshot,
        anomalies: list[dict],
    ) -> None:
        """Send anomaly report without Gemini analysis."""
        if self._slack is None:
            return

        lines = [
            f"━━━ B3 Watchdog — {len(anomalies)} ANOMALIES ━━━",
        ]
        for a in anomalies[:5]:
            lines.append(f"  {a['trigger']}: {a.get('severity', 'INFO')}")

        lines.append(f"\nWR: paper {metrics.rolling_wr_paper:.1%} | live {metrics.rolling_wr_live:.1%}")
        lines.append("(Gemini unavailable — manual review suggested)")

        try:
            await self._slack.send_message("\n".join(lines))
        except Exception as e:
            logger.warning("b3_watchdog_slack_error", error=str(e))

    async def _send_escalation(
        self,
        metrics: B3MetricsSnapshot,
        decision: dict[str, Any],
    ) -> None:
        """Send CRITICAL escalation to Slack."""
        if self._slack is None:
            return

        lines = [
            "━━━ 🔴 B3 Watchdog CRITICAL ━━━",
            f"Root cause: {decision.get('root_cause', 'N/A')}",
            "Verdict: ESCALATE — outside Watchdog bounds",
            f"WR: paper {metrics.rolling_wr_paper:.1%} | live {metrics.rolling_wr_live:.1%}",
            "",
            "Vyžaduje pozornost CEO",
        ]

        try:
            await self._slack.send_alert("\n".join(lines))
        except Exception as e:
            logger.warning("b3_watchdog_slack_error", error=str(e))

    async def _send_revert_report(
        self,
        change: Any,
        wr_drop: float,
        metrics: B3MetricsSnapshot,
    ) -> None:
        """Report auto-revert to Slack."""
        if self._slack is None:
            return

        lines = [
            "━━━ ↩️ B3 Watchdog AUTO-REVERT ━━━",
            f"Param: {change.param_name}: {change.new_value} → {change.old_value}",
            f"Reason: WR dropped {wr_drop:.1%} after {_AUTO_REVERT_WINDOW} trades",
            f"Current WR: {metrics.rolling_wr_paper:.1%}",
        ]

        try:
            await self._slack.send_message("\n".join(lines))
        except Exception as e:
            logger.warning("b3_watchdog_slack_error", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Status for health monitoring."""
        return {
            "running": self._running,
            "observation_mode": self._observation_mode,
            "last_eval_time": self._last_eval_time,
            "last_eval_trades": self._last_eval_trade_count,
            "baseline_set": self._baseline is not None,
            "config_status": self._config.get_status(),
        }


# ── Gemini System Prompt ─────────────────────────────────────────────

_GEMINI_SYSTEM_PROMPT = """Jsi quantitative strategy optimizer pro B3 (Binance Oracle Scalper na Polymarket).
B3 je momentum scalper na 5-min BTC Up/Down markets. Vstupuje v minutě 2-3,
drží do resolution (never-sell mode live). CDF model s Chainlink oracle.

TY ROZHODUJEŠ. Nejsi poradce — jsi autonomní decision engine.

Dostal jsi data o anomálii v B3 výkonu. Tvůj úkol:

1. IDENTIFIKUJ ROOT CAUSE.
2. ANALYZUJ REGIME-SPECIFIC DATA — který bucket degradoval a proč?
3. ROZHODNÍ O AKCI:
   - APPLY: Změnit parametr (musí být v available_params bounds)
   - REVERT: Vrátit předchozí změnu
   - ESCALATE: Problém mimo bounds
   - MONITOR: Nedostatek dat
4. PRO APPLY specifikuj: param, new_value, expected_impact
5. OPTIMALIZUJ PRO: vysoký WR, nízký DD, velkou obratku, stabilní edge.
6. POKUD DATA NESTAČÍ (< 30 tradů v bucketu), zvol MONITOR.
7. NIKDY nenavrhuj Tier 3 změny.

TA Context:
- ADX > 25 = silný trend → momentum edge větší
- ADX < 15 = ranging → slabý momentum
- RSI >80/<20 = mean-reversion risk
- Multi-TF aligned = vyšší confidence

Odpověz JSON:
{
  "verdict": "APPLY|REVERT|ESCALATE|MONITOR",
  "root_cause": "...",
  "evidence": ["..."],
  "action": {"param": "...", "old_value": ..., "new_value": ..., "expected_impact": "..."},
  "confidence": "HIGH|MEDIUM|LOW"
}"""
