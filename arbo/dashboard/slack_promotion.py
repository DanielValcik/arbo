"""Slack promotion workflow — Phase 2C.C.

Block Kit message builders + action handlers for challenger promotion.
Registered on the existing SlackBot AsyncApp (Socket Mode).

Actions:
- promote_approve:<strategy>:<variant_id>   → PoolManager.promote()
- promote_reject:<strategy>:<variant_id>    → mark declined in DB
- promote_details:<strategy>:<variant_id>   → ephemeral deeper stats

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2C.C
"""
from __future__ import annotations

from typing import Any

from arbo.core.promotion_engine import PromotionCandidate
from arbo.utils.logger import get_logger

logger = get_logger("slack_promotion")

# Default channels per strategy (fallback if none set on SlackBot).
# B2 has no dedicated channel yet — uses bot default (CEO can assign
# a dedicated channel later by editing this dict).
STRATEGY_CHANNELS = {
    "B3":     "C0APX4K8Z2N",  # B3 live
    "B3_15M": "C0APX4K8Z2N",  # B3 live (shared with 5-min)
    "D":      "C0ARXNGKE1M",  # Strategy D (shared NBA/UFC/EPL)
    # "B2": TBD — using SlackBot default channel until assigned
}


def build_promotion_blocks(candidate: PromotionCandidate) -> list[dict[str, Any]]:
    """Build Slack Block Kit message for a promotion candidate."""
    strategy = candidate.strategy
    ch = candidate.challenger_id
    cp = candidate.champion_id
    tier = candidate.tier

    title_icon = {1: ":trophy:", 2: ":warning:", 3: ":no_entry:"}.get(tier, ":question:")
    auto_note = (
        "Auto-approve in 24h if no action (Tier 1)."
        if tier == 1 else
        "Tier 2 — CEO approval required. No auto-promote."
        if tier == 2 else
        "Tier 3 — rejected (silent log only)."
    )

    param_lines = []
    for name, (cp_val, ch_val) in candidate.param_diff.items():
        param_lines.append(f"  `{name}`:  {cp_val} → {ch_val}")

    stats_text = (
        f"*N paired:*  challenger {candidate.n_challenger}  vs  champion {candidate.n_champion}\n"
        f"*Sharpe:*  ch `{candidate.sharpe_challenger}`  vs  cp `{candidate.sharpe_champion}`  "
        f"(Δ deflation-adj `{candidate.dsr_delta}`)\n"
        f"*Mean PnL:*  ch `${candidate.mean_challenger:.4f}`  vs  cp `${candidate.mean_champion:.4f}`\n"
        f"*Win rate:*  ch `{candidate.wr_challenger}%`  vs  cp `{candidate.wr_champion}%`\n"
        f"*P(better):*  `{candidate.p_better}` (block-bootstrap, 1000 resamples)"
    )
    param_text = "\n".join(param_lines) if param_lines else "_(no param changes detected)_"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{title_icon} PROMOTION CANDIDATE — {strategy}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Challenger:* `{ch}`  →  *Champion:* `{cp}`   (Tier {tier})",
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": stats_text},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Parameter changes:*\n{param_text}"},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": auto_note}],
        },
    ]

    if tier == 1:
        blocks.append({
            "type": "actions",
            "block_id": f"promote_actions_{strategy}_{ch}",
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": ":white_check_mark: Approve & Promote"},
                    "action_id": f"promote_approve:{strategy}:{ch}",
                    "value": f"{strategy}:{ch}",
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": {"type": "plain_text", "text": ":x: Reject"},
                    "action_id": f"promote_reject:{strategy}:{ch}",
                    "value": f"{strategy}:{ch}",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": ":mag: Details"},
                    "action_id": f"promote_details:{strategy}:{ch}",
                    "value": f"{strategy}:{ch}",
                },
            ],
        })
    elif tier == 2:
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": "CEO only: reply with `/arbo promote " + ch + "` to confirm.",
            }],
        })

    return blocks


async def post_candidate(slack_bot: Any, candidate: PromotionCandidate) -> bool:
    """Post a promotion candidate message to the appropriate channel."""
    if slack_bot is None:
        return False
    if candidate.tier == 3:
        logger.info(
            "promotion_tier3_silent_reject",
            strategy=candidate.strategy,
            challenger=candidate.challenger_id,
            reason=candidate.reject_reason,
        )
        return False
    channel = STRATEGY_CHANNELS.get(
        candidate.strategy,
        getattr(slack_bot, "_channel_id", "") or "C0APX4K8Z2N",
    )
    blocks = build_promotion_blocks(candidate)
    fallback = (
        f"Promotion candidate: {candidate.challenger_id} "
        f"(Tier {candidate.tier}, P(better)={candidate.p_better})"
    )
    try:
        await slack_bot._post(channel, text=fallback, blocks=blocks)
        logger.info(
            "promotion_candidate_posted",
            strategy=candidate.strategy,
            challenger=candidate.challenger_id,
            tier=candidate.tier,
            channel=channel,
        )
        return True
    except Exception as e:
        logger.warning("promotion_candidate_post_error", error=str(e))
        return False


def register_action_handlers(slack_bot: Any) -> None:
    """Register approve/reject/details handlers on the SlackBot AsyncApp.

    Called from SlackBot._register_commands after app is initialized.
    """
    app = getattr(slack_bot, "_app", None)
    if app is None:
        logger.warning("promotion_handler_register_no_app")
        return

    import re

    _approve_re = re.compile(r"^promote_approve:(?P<s>[^:]+):(?P<v>.+)$")
    _reject_re  = re.compile(r"^promote_reject:(?P<s>[^:]+):(?P<v>.+)$")
    _details_re = re.compile(r"^promote_details:(?P<s>[^:]+):(?P<v>.+)$")

    @app.action(_approve_re)
    async def _approve(ack: Any, body: dict[str, Any], respond: Any) -> None:
        await ack()
        action = body.get("actions", [{}])[0]
        m = _approve_re.match(action.get("action_id", ""))
        if not m:
            return
        strategy = m.group("s")
        variant_id = m.group("v")
        user = body.get("user", {}).get("username", "?")
        logger.info(
            "promotion_approve_clicked",
            strategy=strategy, variant_id=variant_id, user=user,
        )
        try:
            from arbo.core import pool_manager
            ok, reason = await pool_manager.promote(variant_id, strategy, approved_by=user)
        except Exception as e:
            await respond(replace_original=False, text=f":warning: Error: {e}")
            logger.warning("promotion_approve_exec_error", error=str(e))
            return
        if ok:
            await respond(
                replace_original=False,
                text=f":white_check_mark: Promoted `{variant_id}` → champion of {strategy} (by {user})",
            )
        else:
            await respond(
                replace_original=False,
                text=f":warning: Promotion failed: {reason}",
            )

    @app.action(_reject_re)
    async def _reject(ack: Any, body: dict[str, Any], respond: Any) -> None:
        await ack()
        action = body.get("actions", [{}])[0]
        m = _reject_re.match(action.get("action_id", ""))
        if not m:
            return
        strategy = m.group("s")
        variant_id = m.group("v")
        user = body.get("user", {}).get("username", "?")
        logger.info(
            "promotion_reject_clicked",
            strategy=strategy, variant_id=variant_id, user=user,
        )
        await respond(
            replace_original=False,
            text=f":x: Rejected promotion of `{variant_id}` ({strategy}) by {user}. Challenger stays in shadow.",
        )

    @app.action(_details_re)
    async def _details(ack: Any, body: dict[str, Any], respond: Any) -> None:
        await ack()
        action = body.get("actions", [{}])[0]
        m = _details_re.match(action.get("action_id", ""))
        if not m:
            return
        strategy = m.group("s")
        variant_id = m.group("v")
        try:
            from arbo.core.performance_analyzer import PerformanceAnalyzer
            report = await PerformanceAnalyzer(strategy).analyze()
        except Exception as e:
            await respond(replace_original=False, text=f":warning: details error: {e}")
            return
        if report is None:
            await respond(replace_original=False, text="No report available")
            return
        snap = next(
            (v for v in report.variants if v.variant_id == variant_id),
            None,
        )
        if snap is None:
            await respond(replace_original=False, text=f"Variant {variant_id} not found")
            return
        txt = (
            f"*{variant_id}* detail:\n"
            f"• live N={snap.live_n}, WR={snap.live_wr}%, PnL=${snap.live_pnl}\n"
            f"• shadow qualified={snap.shadow_n_qualified}, resolved={snap.shadow_n_resolved}, "
            f"WR={snap.shadow_wr}%, Σ PnL/share=${snap.shadow_pnl_per_share}"
        )
        await respond(replace_original=False, text=txt)
