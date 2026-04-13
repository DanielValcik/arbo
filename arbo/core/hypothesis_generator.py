"""HypothesisGenerator — Phase 2C.A.

Takes a PerformanceReport and asks Gemini Flash for ONE challenger
mutation that addresses the dominant failure mode. Output: a validated
VariantConfig ready for PoolManager.write().

Constraints enforced:
- Must change exactly 1-2 parameters from the champion
- Parameter must be in TIER_1_PARAMS (autonomous bounds) — no MAX_BET_SIZE,
  DAILY_LOSS_PCT, etc. (those are Tier 3, human-only)
- Rationale must reference a specific failure mode from the report

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2C.A
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from arbo.core.performance_analyzer import PerformanceReport
from arbo.core.variant_pool import VariantConfig, get_champion
from arbo.utils.logger import get_logger

logger = get_logger("hypothesis_generator")


# Tier 1: parameters the auto-generator may touch (within bounds).
# Tier 2: require CEO flag (not generated autonomously — sigma_scale, position pct).
# Tier 3: never touched by auto-gen (MAX_BET_SIZE, DAILY_LOSS_PCT, etc.)
TIER_1_PARAMS: dict[str, tuple[float, float]] = {
    # B3 / B3_15M
    "LIVE_MIN_EDGE":        (0.10, 0.60),
    "LIVE_MIN_BTC_MOVE":    (0.0, 100.0),
    "LIVE_MAX_BTC_MOVE":    (30.0, 200.0),
    "LIVE_MAX_VELOCITY":    (20.0, 120.0),
    "LIVE_MAX_DIR_DELTA":   (3.0, 30.0),
    "LIVE_MAX_MARKET_GAP":  (0.05, 0.50),
    "LIVE_MAX_FILL_PRICE":  (0.50, 1.05),
    "MIN_ENTRY_MIN":        (0, 10),
    "MAX_ENTRY_MIN":        (2, 14),
    "MIN_ENTRY_MKT_FV":     (0.10, 0.60),
    "MAX_ENTRY_MKT_FV":     (0.50, 0.99),
    # Project PARALLEL B2 + D additions (shared name MIN_EDGE etc. — bounds
    # are union-set wide enough for all strategies; per-strategy YAMLs
    # are still validated against per-strategy expectations)
    "MIN_EDGE":             (0.05, 0.30),
    "MAX_EDGE":             (0.20, 0.99),
    "MIN_PRICE":            (0.03, 0.40),
    "MAX_PRICE":            (0.30, 0.80),
    "MIN_HOLD_EDGE":        (0.00, 0.15),
    "MIN_TIME_TO_EXPIRY_H": (1.0, 48.0),
    "MAX_TIME_TO_EXPIRY_H": (48.0, 336.0),
    # D-specific
    "GREEN_BOOK_DELTA":     (0.05, 0.30),
    "STOP_LOSS_DELTA":      (0.05, 0.25),
    "MAX_HOLD_FRACTION":    (0.20, 1.00),
    "GAME_DURATION_HOURS":  (1.5, 4.0),
}


@dataclass
class Hypothesis:
    """A single proposed challenger from Gemini."""
    variant_id: str
    rationale: str           # English explanation referencing failure mode
    param_changes: dict[str, float]  # {"LIVE_MAX_VELOCITY": 50.0, ...}
    parent_variant_id: str   # always the current champion


def _cap_rationale(text: str, max_chars: int = 400) -> str:
    text = (text or "").strip()
    return text[:max_chars]


def _build_prompt(report: PerformanceReport) -> str:
    """Render the prompt sent to Gemini.

    Includes: strategy name, champion params, pool summary, failure modes,
    list of allowed Tier 1 parameters + bounds, response schema.
    """
    champ = get_champion(report.strategy)
    champ_params = champ.params if champ else {}

    pool_summary = []
    for v in report.variants:
        pool_summary.append({
            "variant_id": v.variant_id,
            "status": v.status,
            "live_n": v.live_n,
            "live_pnl": v.live_pnl,
            "live_wr_pct": v.live_wr,
            "shadow_pnl_per_share": v.shadow_pnl_per_share,
            "shadow_wr_pct": v.shadow_wr,
            # Phase 3.3: mid-trade signals (None if data not yet collected)
            "avg_mid_60s_drift": v.avg_mid_at_60s_drift,
            "avg_composite_reward": v.avg_composite_reward,
            "composite_n": v.composite_reward_n,
        })

    failure_modes = []
    for m in report.failure_modes:
        failure_modes.append({
            "feature": m.feature,
            "condition": m.condition,
            "n_total": m.n_total,
            "n_losses": m.n_losses,
            "loss_rate_pct": round(m.loss_rate * 100, 1),
            "avg_pnl": m.avg_pnl,
            "impact_score": m.impact,
        })

    tier1 = {k: list(v) for k, v in TIER_1_PARAMS.items()}

    system_ctx = (
        f"You are a quantitative strategy research assistant for Arbo — an "
        f"automated Polymarket trading system. Strategy {report.strategy} is "
        f"a scalping model on short-duration BTC price markets. The current "
        f"champion is '{report.champion_variant_id}' with {report.champion_live_n} "
        f"live trades and cumulative PnL ${report.champion_live_pnl}. Your job: "
        f"propose ONE new challenger variant that differs from the champion "
        f"by exactly ONE parameter, targeting the highest-impact failure mode "
        f"below. The challenger will run in shadow mode (no capital risk) for "
        f"paired-sample comparison vs champion."
    )

    rules = [
        "Change exactly ONE parameter (not two, not zero).",
        "Parameter must be in the allowed TIER_1 list.",
        "New value must stay within the bound range [lo, hi] for that param.",
        "If the top failure mode is 'feature X at threshold T', mutate the "
        "param that most directly gates feature X.",
        "Do NOT propose a parameter change that matches any existing variant "
        "in the pool.",
        "variant_id must be snake_case, start with 'auto_', end with a short "
        "descriptor of the change (e.g. 'auto_vel_45').",
        "rationale: 1-2 sentences max, reference the specific failure mode "
        "(feature + condition + loss_rate_pct).",
    ]

    response_schema = {
        "variant_id": "string (e.g. auto_vel_45)",
        "rationale": "string, 1-2 sentences",
        "param_changes": {
            "EXAMPLE_PARAM_NAME": "new_float_value",
        },
    }

    payload = {
        "context": system_ctx,
        "rules": rules,
        "strategy": report.strategy,
        "champion_params": champ_params,
        "variant_pool": pool_summary,
        "failure_modes_top5": failure_modes,
        "allowed_tier1_params_and_bounds": tier1,
        "response_schema_json": response_schema,
    }
    return (
        "Analyze the performance report and output ONE challenger proposal as "
        "valid JSON matching response_schema_json. No prose outside JSON.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def _validate(
    proposal: dict[str, Any], report: PerformanceReport
) -> tuple[Hypothesis | None, str | None]:
    """Return (Hypothesis, None) on success or (None, reason) on reject."""
    try:
        vid = str(proposal["variant_id"]).strip().lower()
        rationale = _cap_rationale(str(proposal.get("rationale", "")))
        changes_raw = proposal["param_changes"]
    except KeyError as e:
        return None, f"missing field {e}"

    if not vid.startswith("auto_"):
        return None, f"variant_id must start with 'auto_': {vid!r}"
    if not vid.replace("_", "").isalnum():
        return None, f"variant_id invalid chars: {vid!r}"

    if not isinstance(changes_raw, dict) or not changes_raw:
        return None, "param_changes must be non-empty dict"
    if len(changes_raw) > 2:
        return None, f"too many changes ({len(changes_raw)}); max 2"

    normalized: dict[str, float] = {}
    for k, v in changes_raw.items():
        kstr = str(k).strip()
        if kstr not in TIER_1_PARAMS:
            return None, f"param {kstr!r} not in TIER_1 list"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, f"param {kstr!r} value {v!r} not numeric"
        lo, hi = TIER_1_PARAMS[kstr]
        if not (lo <= fv <= hi):
            return None, f"param {kstr!r} value {fv} outside bounds [{lo}, {hi}]"
        normalized[kstr] = fv

    # Must actually differ from champion
    champ = get_champion(report.strategy)
    if champ is None:
        return None, "no champion in pool"
    actual_diff = {}
    for k, v in normalized.items():
        cur = champ.params.get(k)
        if cur is None:
            actual_diff[k] = v
            continue
        try:
            if abs(float(cur) - v) > 1e-9:
                actual_diff[k] = v
        except (TypeError, ValueError):
            actual_diff[k] = v
    if not actual_diff:
        return None, "proposal matches champion exactly"

    # Dedupe against existing pool
    from arbo.core.variant_pool import load_variants
    pool = load_variants(report.strategy)
    for existing in pool:
        match = True
        for k, v in actual_diff.items():
            try:
                if abs(float(existing.params.get(k, -1e18)) - v) > 1e-9:
                    match = False
                    break
            except (TypeError, ValueError):
                match = False
                break
        if match and existing.variant_id != champ.variant_id:
            return None, f"duplicate of existing variant {existing.variant_id}"

    return Hypothesis(
        variant_id=vid,
        rationale=rationale,
        param_changes=actual_diff,
        parent_variant_id=champ.variant_id,
    ), None


class HypothesisGenerator:
    """Wraps Gemini Flash for challenger proposal."""

    def __init__(self, llm_agent: Any | None = None) -> None:
        self._agent = llm_agent

    async def propose(self, report: PerformanceReport) -> Hypothesis | None:
        """Return a validated Hypothesis or None if generation/validation fails.

        Returns None (not raises) on any failure — caller schedules next
        attempt on next cycle.
        """
        if self._agent is None:
            logger.info("hypothesis_no_agent", strategy=report.strategy)
            return None
        if not report.failure_modes:
            logger.info("hypothesis_no_failure_modes", strategy=report.strategy)
            return None

        prompt = _build_prompt(report)
        try:
            result = await self._agent.raw_query(prompt)
        except Exception as e:
            logger.warning("hypothesis_llm_error", error=str(e))
            return None
        if not result or not isinstance(result, dict):
            logger.info("hypothesis_empty_response", strategy=report.strategy)
            return None

        hypo, reject_reason = _validate(result, report)
        if hypo is None:
            logger.info(
                "hypothesis_rejected",
                strategy=report.strategy,
                reason=reject_reason,
                raw=json.dumps(result, default=str)[:300],
            )
            return None

        logger.info(
            "hypothesis_accepted",
            strategy=report.strategy,
            variant_id=hypo.variant_id,
            param_changes=hypo.param_changes,
            rationale=hypo.rationale,
        )
        return hypo
