"""Daily PARALLEL digest — human-friendly morning briefing in Slack.

Runs once per day (09:00 UTC, systemd timer) and posts ONE readable
message per strategy that answers, in plain Czech:

  - How is the current strategy doing?
  - Are we testing a new version? How far along?
  - Is there a reason to worry (drift, stagnation, losses)?
  - Is there anything the user needs to decide?

Design goal: replace the stream of ad-hoc Slack alerts
(promotion candidates, drift alerts, challenger-created notices) with
a single coherent briefing that respects the user's attention.

Data sources:
  - `variant_pool.load_variants(strategy)` — variant statuses
  - `paper_trades` — live P&L per variant (24h and 7d windows)
  - `shadow_variant_signals` — challenger progress
  - `drift_monitor.evaluate_strategy_drift` — current drift state
  - `promotion_engine.PromotionEngine.evaluate` — pending candidates

LLM synthesis: Gemini 2.5 Flash turns the raw stats into a morning
briefing a non-engineer can read. Falls back to a plain template if
LLM is unavailable.

CLI:
    python scripts/parallel_digest.py            # posts all strategies
    python scripts/parallel_digest.py --strategy B2
    python scripts/parallel_digest.py --dry-run  # print, don't post

Systemd timer: see `scripts/systemd/arbo-parallel-digest.*`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GEMINI_MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """\
Napiš ranní Slack briefing o tradingové strategii {strategy}. Uživatel
je technicky zdatný ale ne engineer systému — píš česky, přirozeně,
bez zkratek jako "PH", "DSR", "P(better)". Vysvětli co znamenají
čísla krátce místo zkratek.

KONTEXT (syrová data, ty to přelož do lidštiny):

## Aktuální champion (produkční strategie)
- ID: {champion_id}
- Live obchody posledních 7 dní: {champion_n_7d}, WR {champion_wr_7d}, celkem {champion_pnl_7d}
- Live obchody posledních 24h: {champion_n_24h}, celkem {champion_pnl_24h}
- Otevřené pozice: {champion_open}

## Canary (incubating variant, pokud je)
{incubate_block}

## Drift (distribuce výsledků se mění?)
{drift_block}

## Čekající rozhodnutí nebo autonomní akce
{pending_block}

FORMÁT ODPOVĚDI — přesně tento markdown:

*🌅 Ranní briefing: {strategy} — {date_label}*

[Jedna krátká věta jak si strategie vede. Pokud vydělává, řekni to; pokud ztrácí, řekni proč.]

*Aktuální strategie*
[2-3 řádky o championovi — kolik obchodů, vítěznost, kolik peněz, jak se má proti minulému týdnu. Konkrétní čísla.]

*Testování nové verze*
[Pokud někdo incubuje: kolik reálných obchodů z 15 potřebných, zda zatím vypadá lépe/hůře/stejně než champion. Pokud ne: napiš "Aktuálně nic netestujeme."]

*Co systém hlídá*
[1-2 řádky — drift (pokud je), další kandidáti ve frontě, zjevná rizika.]

*Co musíš udělat*
[Buď "Nic — systém si to hlídá sám." nebo konkrétní akce typu "Zvaž kliknout Approve na ch_edge_tight — systém tam sám nešel kvůli slabší jistotě."]

Pravidla:
- Žádné emoji kromě prvního 🌅
- Žádná zkratka bez kontextu
- Krátké odstavce
- Pokud nejsou data, napiš "nemáme ještě dost dat"
- Když je vše OK, **ŘEKNI TO** — uživatel potřebuje klid
"""


# ─── Data collection ───────────────────────────────────────────────


@dataclass
class StrategyDigestData:
    strategy: str
    date_label: str
    # Champion live
    champion_id: str
    champion_n_7d: int
    champion_wr_7d: float | None
    champion_pnl_7d: float
    champion_n_24h: int
    champion_pnl_24h: float
    champion_open: int
    # Incubate (None = no canary)
    incubate_id: str | None
    incubate_n_live: int
    incubate_n_needed: int
    incubate_wr_live: float | None
    incubate_pnl_live: float
    incubate_champion_mean_live: float
    incubate_days_active: int | None
    # Drift
    drift_any: bool
    drift_details: list[str]
    # Pending
    pending_candidates: list[str]
    pending_auto_approved: int


async def collect_strategy_data(strategy: str) -> StrategyDigestData:
    """Gather all the inputs a single strategy's digest needs."""
    from arbo.core.variant_pool import load_variants, get_champion
    from arbo.utils.db import get_session_factory
    import sqlalchemy as sa

    variants = load_variants(strategy)
    champion = get_champion(strategy)
    incubate = next((v for v in variants if v.status == "incubate"), None)

    today = datetime.now(UTC).date()
    date_label = today.isoformat()

    factory = get_session_factory()
    async with factory() as session:
        # Champion live stats — 7 days and 24h windows
        res_7d = await session.execute(
            sa.text("""
                SELECT COUNT(*), COALESCE(SUM(actual_pnl),0)::numeric(10,2),
                  COUNT(*) FILTER (WHERE actual_pnl > 0)
                FROM paper_trades
                WHERE strategy=:s AND status='sold'
                  AND trade_details->>'variant_id' = :v
                  AND placed_at > NOW() - INTERVAL '7 days'
                  AND COALESCE(notes,'') NOT LIKE 'pre_reset%'
            """),
            {"s": strategy, "v": champion.variant_id if champion else "champion_v1"},
        )
        row = res_7d.fetchone()
        ch_n_7d = int(row[0] or 0)
        ch_pnl_7d = float(row[1] or 0)
        ch_wins_7d = int(row[2] or 0)
        ch_wr_7d = (ch_wins_7d / ch_n_7d) if ch_n_7d else None

        res_24h = await session.execute(
            sa.text("""
                SELECT COUNT(*), COALESCE(SUM(actual_pnl),0)::numeric(10,2)
                FROM paper_trades
                WHERE strategy=:s AND status='sold'
                  AND trade_details->>'variant_id' = :v
                  AND placed_at > NOW() - INTERVAL '24 hours'
                  AND COALESCE(notes,'') NOT LIKE 'pre_reset%'
            """),
            {"s": strategy, "v": champion.variant_id if champion else "champion_v1"},
        )
        row = res_24h.fetchone()
        ch_n_24h = int(row[0] or 0)
        ch_pnl_24h = float(row[1] or 0)

        res_open = await session.execute(
            sa.text("""
                SELECT COUNT(*) FROM paper_trades
                WHERE strategy=:s AND status='open'
                  AND COALESCE(notes,'') NOT LIKE 'pre_reset%'
            """),
            {"s": strategy},
        )
        ch_open = int(res_open.scalar_one() or 0)

        # Incubate live stats
        inc_id = incubate.variant_id if incubate else None
        inc_n = 0
        inc_wr = None
        inc_pnl = 0.0
        inc_days = None
        if incubate is not None:
            res_inc = await session.execute(
                sa.text("""
                    SELECT COUNT(*), COALESCE(SUM(actual_pnl),0)::numeric(10,2),
                      COUNT(*) FILTER (WHERE actual_pnl > 0)
                    FROM paper_trades
                    WHERE strategy=:s AND status='sold'
                      AND trade_details->>'variant_id' = :v
                      AND COALESCE(notes,'') NOT LIKE 'pre_reset%'
                """),
                {"s": strategy, "v": incubate.variant_id},
            )
            row = res_inc.fetchone()
            inc_n = int(row[0] or 0)
            inc_pnl = float(row[1] or 0)
            inc_wins = int(row[2] or 0)
            inc_wr = (inc_wins / inc_n) if inc_n else None
            # Days active — from YAML incubated_at timestamp if present
            try:
                # Re-read YAML raw — VariantConfig doesn't carry this
                import yaml as _yaml
                yaml_path = (
                    REPO_ROOT / "arbo" / "config" / "variants"
                    / strategy.lower() / f"{incubate.variant_id}.yaml"
                )
                if yaml_path.exists():
                    raw = _yaml.safe_load(yaml_path.read_text()) or {}
                    ts_str = raw.get("incubated_at", "")
                    # Expected format: "2026-04-19 12:04 UTC"
                    if ts_str:
                        inc_dt = datetime.strptime(
                            ts_str.replace(" UTC", ""), "%Y-%m-%d %H:%M",
                        ).replace(tzinfo=UTC)
                        inc_days = (datetime.now(UTC) - inc_dt).days
            except Exception:
                inc_days = None

    # Champion mean PnL for incubate comparison context
    inc_ch_mean = (ch_pnl_7d / ch_n_7d) if ch_n_7d else 0.0

    # Drift
    drift_any = False
    drift_details: list[str] = []
    try:
        from arbo.core.drift_monitor import evaluate_strategy_drift
        results = await evaluate_strategy_drift(strategy)
        firing = [r for r in results if r.firing]
        if firing:
            drift_any = True
            for r in firing:
                drift_details.append(
                    f"{r.variant_id}: signál driftu (PH={r.ph_stat}, N={r.n_samples})"
                )
    except Exception as e:
        drift_details.append(f"drift check selhal: {e}")

    # Pending candidates (would be emitted but aren't yet)
    pending_names: list[str] = []
    auto_approved = 0
    try:
        from arbo.core.promotion_engine import PromotionEngine, MIN_P_BETTER
        cands = await PromotionEngine(strategy).evaluate()
        for c in cands:
            if c.reject_reason or (c.tier == 1 and c.p_better < MIN_P_BETTER):
                continue
            if c.auto_approve:
                auto_approved += 1
            else:
                pending_names.append(
                    f"{c.challenger_id} (P(lepší)={c.p_better:.0%})"
                )
    except Exception:
        pass

    return StrategyDigestData(
        strategy=strategy,
        date_label=date_label,
        champion_id=(champion.variant_id if champion else "?"),
        champion_n_7d=ch_n_7d,
        champion_wr_7d=ch_wr_7d,
        champion_pnl_7d=ch_pnl_7d,
        champion_n_24h=ch_n_24h,
        champion_pnl_24h=ch_pnl_24h,
        champion_open=ch_open,
        incubate_id=inc_id,
        incubate_n_live=inc_n,
        incubate_n_needed=15,
        incubate_wr_live=inc_wr,
        incubate_pnl_live=inc_pnl,
        incubate_champion_mean_live=inc_ch_mean,
        incubate_days_active=inc_days,
        drift_any=drift_any,
        drift_details=drift_details,
        pending_candidates=pending_names,
        pending_auto_approved=auto_approved,
    )


# ─── Narrative ─────────────────────────────────────────────────────


def build_prompt(d: StrategyDigestData) -> str:
    def _pct(v: float | None) -> str:
        return f"{v * 100:.0f}%" if v is not None else "—"

    def _money(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:.2f}"

    incubate_block = "Aktuálně nic neinkubujeme."
    if d.incubate_id:
        age = f"{d.incubate_days_active} dní" if d.incubate_days_active else "právě začalo"
        incubate_block = (
            f"- ID: {d.incubate_id}\n"
            f"- Živé obchody: {d.incubate_n_live} z {d.incubate_n_needed} "
            f"potřebných pro rozhodnutí\n"
            f"- Vítěznost: {_pct(d.incubate_wr_live)}, PnL: {_money(d.incubate_pnl_live)}\n"
            f"- Running: {age}"
        )

    drift_block = "Bez driftu." if not d.drift_any else "\n".join(
        f"- {x}" for x in d.drift_details
    )

    pending_parts = []
    if d.pending_auto_approved:
        pending_parts.append(f"Systém sám schválil {d.pending_auto_approved} kanárek.")
    if d.pending_candidates:
        pending_parts.append("Čekají na tvé rozhodnutí: " + ", ".join(d.pending_candidates))
    if not pending_parts:
        pending_parts.append("Žádné pending rozhodnutí.")
    pending_block = "\n".join(pending_parts)

    return PROMPT_TEMPLATE.format(
        strategy=d.strategy,
        date_label=d.date_label,
        champion_id=d.champion_id,
        champion_n_7d=d.champion_n_7d,
        champion_wr_7d=_pct(d.champion_wr_7d),
        champion_pnl_7d=_money(d.champion_pnl_7d),
        champion_n_24h=d.champion_n_24h,
        champion_pnl_24h=_money(d.champion_pnl_24h),
        champion_open=d.champion_open,
        incubate_block=incubate_block,
        drift_block=drift_block,
        pending_block=pending_block,
    )


def llm_synthesis(prompt: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 1500},
            request_options={"timeout": 60},
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[llm] failed: {e}", file=sys.stderr)
        return None


def stats_fallback(d: StrategyDigestData) -> str:
    """If LLM fails, emit a plain-template version (still Czech)."""
    def _pct(v: float | None) -> str:
        return f"{v * 100:.0f}%" if v is not None else "—"
    def _money(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:.2f}"

    lines = [f"*🌅 Ranní briefing: {d.strategy} — {d.date_label}*", ""]
    if d.champion_n_7d == 0:
        lines.append("_Nemáme zatím dost dat za posledních 7 dní._")
    else:
        lines.append(f"*Aktuální strategie* ({d.champion_id})")
        lines.append(
            f"• Posledních 7 dní: {d.champion_n_7d} obchodů, "
            f"vítěznost {_pct(d.champion_wr_7d)}, celkem {_money(d.champion_pnl_7d)}"
        )
        lines.append(
            f"• Za 24h: {d.champion_n_24h} obchodů, {_money(d.champion_pnl_24h)}"
        )
        lines.append(f"• Otevřené pozice: {d.champion_open}")

    lines.append("")
    lines.append("*Testování nové verze*")
    if d.incubate_id:
        lines.append(
            f"• Testujeme `{d.incubate_id}` — {d.incubate_n_live} z "
            f"{d.incubate_n_needed} obchodů potřebných pro rozhodnutí"
        )
        if d.incubate_n_live > 0:
            lines.append(
                f"• Zatím {_pct(d.incubate_wr_live)} vítězných, "
                f"PnL {_money(d.incubate_pnl_live)}"
            )
    else:
        lines.append("• Aktuálně nic netestujeme.")

    lines.append("")
    lines.append("*Co systém hlídá*")
    if d.drift_any:
        lines.append("⚠️ Drift detekován u:")
        for x in d.drift_details:
            lines.append(f"• {x}")
    else:
        lines.append("• Žádný drift.")
    if d.pending_candidates:
        lines.append("• Ve frontě: " + ", ".join(d.pending_candidates))
    if d.pending_auto_approved:
        lines.append(
            f"• Systém sám schválil {d.pending_auto_approved} kanárek."
        )

    lines.append("")
    lines.append("*Co musíš udělat*")
    actions = []
    if d.pending_candidates and not d.incubate_id:
        actions.append(
            f"Zvaž `Approve` pro: {', '.join(d.pending_candidates)}"
        )
    if not actions:
        actions.append("Nic — systém si to hlídá sám.")
    for a in actions:
        lines.append(f"• {a}")

    return "\n".join(lines)


# ─── Dispatch ──────────────────────────────────────────────────────


async def post_digest(strategy: str, dry_run: bool) -> None:
    d = await collect_strategy_data(strategy)
    prompt = build_prompt(d)
    narrative = llm_synthesis(prompt) or stats_fallback(d)

    if dry_run:
        print(narrative)
        return

    # Post to Slack daily brief channel
    channel_id = os.environ.get("SLACK_DAILY_BRIEF_CHANNEL_ID") \
        or os.environ.get("SLACK_CHANNEL_ID")
    if not channel_id:
        print("[digest] no Slack channel configured; skipping post", file=sys.stderr)
        print(narrative)
        return

    try:
        from slack_sdk.web.async_client import AsyncWebClient
        token = os.environ["SLACK_BOT_TOKEN"]
        client = AsyncWebClient(token=token)
        await client.chat_postMessage(
            channel=channel_id, text=narrative, unfurl_links=False,
        )
        print(f"[digest] posted {strategy} to {channel_id}")
    except Exception as e:
        print(f"[digest] slack post failed: {e}", file=sys.stderr)
        print(narrative)


async def main_async(strategies: list[str], dry_run: bool) -> None:
    for s in strategies:
        try:
            await post_digest(s, dry_run)
        except Exception as e:
            print(f"[digest] {s} failed: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None,
                    help="Single strategy (B2, B3, B3_15M, D). Default: B2 only.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print to stdout instead of posting to Slack.")
    args = ap.parse_args()

    strategies = [args.strategy] if args.strategy else ["B2"]
    asyncio.run(main_async(strategies, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
