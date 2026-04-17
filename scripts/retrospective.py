"""Hierarchical retrospective system for Arbo.

Generates daily / weekly / monthly / yearly summaries by combining:
- LEARNINGS.md diff (new entries since last period)
- Git commit log (code changes)
- systemd journal logs (runtime events, filtered to meaningful signal)
- PostgreSQL metrics (paper_trades, shadow_variant_signals, health_checks)
- Previous-level summaries (cascade: weekly reads 7 daily files, monthly
  reads 4-5 weekly, yearly reads 12 monthly)

The summary is a structured Markdown document with machine-parseable
metadata plus an LLM-synthesised narrative section. If the LLM is
unavailable, a stats-only summary still runs — the script never fails
silently.

CLI:
    python scripts/retrospective.py daily [--date YYYY-MM-DD]
    python scripts/retrospective.py weekly [--week YYYY-WNN]
    python scripts/retrospective.py monthly [--month YYYY-MM]
    python scripts/retrospective.py yearly [--year YYYY]

Without explicit period flags the script picks "previous full period"
(yesterday for daily, last completed week/month/year). That's what
systemd timers invoke.

All outputs are written to `summaries/<period>/<label>.md` and
(optionally) committed+pushed. Auto-commit only happens when the file
content actually changed, so re-running is idempotent.

Design notes:
- Standalone script: does NOT import Arbo's strategy modules. This
  keeps the summary system isolated so a bug here can never crash the
  live trading loop.
- Blocking stdlib only for DB (asyncpg is async but this runs
  synchronously — we call psql via subprocess for simplicity and
  transparency). Avoids adding sqlalchemy to the critical-path inputs.
- Idempotent: if the target summary file already exists with content,
  a re-run compares, updates only if new data arrived.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARIES = REPO_ROOT / "summaries"
LEARNINGS = REPO_ROOT / "LEARNINGS.md"

# How much journal to keep per period — daily keeps more detail, longer
# periods rely on daily summaries and downsample aggressively.
JOURNAL_MAX_BYTES = {
    "daily": 80_000,
    "weekly": 30_000,
    "monthly": 15_000,
    "yearly": 8_000,
}

# Which log events to surface in the summary (filters the raw journal).
# Additive — anything matching any of these goes into the trimmed
# journal. Everything else is dropped for summary purposes.
JOURNAL_KEEP_PATTERNS = [
    r"b2_entry\b",
    r"b2_exit_triggered",
    r"b2_entry_summary",
    r"b2_live_filled",
    r"b3_entry",
    r"b3_exit",
    r"b3_15m_entry",
    r"b3_15m_exit",
    r"d_entry",
    r"d_exit",
    r"c2_entry",
    r"c2_exit",
    r"shadow_promotion",
    r"bandit_",
    r"drift_",
    r"risk_breach",
    r"daily_loss",
    r"weekly_loss",
    r"emergency_shutdown",
    r"\"level\": \"error\"",
    r"traceback",
    r"exception",
    r"slack_post_send",
]

GEMINI_MODEL = "gemini-2.5-flash"
LLM_TIMEOUT_S = 60

LLM_PROMPT_TEMPLATE = """\
You are analysing an autonomous Polymarket trading system called Arbo.
Write a concise {period_label} retrospective — sections as requested.

DATA FOR {period_label}:

## New LEARNINGS entries
{learnings_diff}

## Code changes
{git_log}

## Trading metrics
{metrics_table}

## Filtered journal (meaningful events only)
{journal_excerpt}

## Previous-level summaries (for context)
{prior_summaries}

OUTPUT — valid markdown, these sections in this order:

### Executive summary
3 bullets max. What mattered in this period.

### What happened
Short paragraphs. Group related bug/fix/observation together. Reference
commits by hash and LEARNINGS by ID.

### Impact
Did it help? How much? Cite metrics — use the numbers in the trading
metrics table, don't invent new ones.

### Risk & open threads
Things still unresolved, things to watch.

### Proposed next steps
Concrete actions for the next period. Be specific — name files,
parameters, or experiments. Do NOT propose actions already done.

Keep under 600 words total. Prefer concrete specifics over vague
observations. If the period is uneventful say so.
"""


# ─── Utilities ────────────────────────────────────────────────────


def run(cmd: list[str] | str, check: bool = False, timeout: int = 120) -> str:
    """Run a shell command, return stdout (stderr dropped). Never raise
    unless check=True and exit code != 0."""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout,
            )
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    except Exception as e:
        if check:
            raise
        return f"[error: {e}]"


def now_utc() -> datetime:
    return datetime.now(UTC)


# ─── Period resolution ────────────────────────────────────────────


@dataclass
class Period:
    """Represents a summary period with its label and [start, end) range."""

    kind: str  # daily / weekly / monthly / yearly
    label: str  # 2026-04-17, 2026-W16, 2026-04, 2026
    start: datetime
    end: datetime

    @property
    def out_path(self) -> Path:
        return SUMMARIES / self.kind / f"{self.label}.md"


def resolve_period(kind: str, arg: str | None) -> Period:
    """Return the Period to summarise. If arg is None, picks the
    previous completed period."""
    ref = now_utc()
    if kind == "daily":
        target_date = (
            date.fromisoformat(arg) if arg else (ref.date() - timedelta(days=1))
        )
        start = datetime(
            target_date.year, target_date.month, target_date.day, tzinfo=UTC,
        )
        end = start + timedelta(days=1)
        return Period("daily", target_date.isoformat(), start, end)
    if kind == "weekly":
        if arg:
            y, w = arg.split("-W")
            year, week = int(y), int(w)
        else:
            # Last completed ISO week = one week ago
            iso = (ref - timedelta(days=7)).isocalendar()
            year, week = iso[0], iso[1]
        # ISO week starts Monday
        start = datetime.fromisocalendar(year, week, 1).replace(tzinfo=UTC)
        end = start + timedelta(days=7)
        return Period("weekly", f"{year}-W{week:02d}", start, end)
    if kind == "monthly":
        if arg:
            year, month = [int(x) for x in arg.split("-")]
        else:
            prev = ref.replace(day=1) - timedelta(days=1)
            year, month = prev.year, prev.month
        start = datetime(year, month, 1, tzinfo=UTC)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(year, month + 1, 1, tzinfo=UTC)
        return Period("monthly", f"{year}-{month:02d}", start, end)
    if kind == "yearly":
        year = int(arg) if arg else ref.year - 1
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
        return Period("yearly", str(year), start, end)
    raise ValueError(f"Unknown period kind: {kind}")


# ─── Data collection ──────────────────────────────────────────────


def collect_learnings_diff(period: Period) -> str:
    """Return the LEARNINGS.md diff covering this period.

    Uses `git log --since --until -p -- LEARNINGS.md` to extract only
    the added content. If there's no commit history we fall back to the
    full file (bounded)."""
    if not LEARNINGS.exists():
        return "[LEARNINGS.md not found]"
    since = period.start.strftime("%Y-%m-%d %H:%M:%S")
    until = period.end.strftime("%Y-%m-%d %H:%M:%S")
    raw = run(
        ["git", "-C", str(REPO_ROOT), "log",
         f"--since={since}", f"--until={until}",
         "-p", "--format=%n## commit %h %s%n%n", "--", "LEARNINGS.md"],
        timeout=30,
    )
    if not raw.strip():
        return "(no LEARNINGS changes this period)"
    # Keep only added lines (diff `+` prefix) to avoid noise from
    # context+headers
    added = []
    for line in raw.splitlines():
        if line.startswith("## commit "):
            added.append("\n" + line)
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added)[:40_000]


def collect_git_log(period: Period) -> str:
    since = period.start.strftime("%Y-%m-%d %H:%M:%S")
    until = period.end.strftime("%Y-%m-%d %H:%M:%S")
    raw = run(
        ["git", "-C", str(REPO_ROOT), "log",
         f"--since={since}", f"--until={until}",
         "--format=- %h %s (%an)", "--no-merges"],
        timeout=30,
    )
    return raw.strip() or "(no commits)"


def collect_journal(period: Period) -> str:
    """Pull journalctl output for this period, filtered to meaningful
    events. Runs via SSH if not on the VPS."""
    is_vps = Path("/etc/systemd/system/arbo.service").exists()
    since = period.start.strftime("%Y-%m-%d %H:%M:%S")
    until = period.end.strftime("%Y-%m-%d %H:%M:%S")
    if is_vps:
        cmd = (
            f"sudo journalctl -u arbo --since='{since}' --until='{until}' "
            f"--no-pager --output=short"
        )
    else:
        cmd = (
            f"ssh arbo-dublin \"sudo journalctl -u arbo --since='{since}' "
            f"--until='{until}' --no-pager --output=short\""
        )
    raw = run(cmd, timeout=180)
    if not raw or raw.startswith("[error") or raw.startswith("[timed"):
        return raw or "(no journal data)"
    patterns = re.compile("|".join(JOURNAL_KEEP_PATTERNS), re.IGNORECASE)
    kept = [line for line in raw.splitlines() if patterns.search(line)]
    excerpt = "\n".join(kept)
    limit = JOURNAL_MAX_BYTES.get(period.kind, 100_000)
    if len(excerpt) > limit:
        # Keep head + tail for long periods — middle is usually more
        # of the same. The bookends show start-of-period baseline and
        # end-of-period state.
        half = limit // 2
        excerpt = excerpt[:half] + "\n\n[... truncated ...]\n\n" + excerpt[-half:]
    return excerpt or "(journal empty or nothing matched filters)"


def collect_metrics(period: Period) -> dict[str, Any]:
    """Run a handful of SQL queries to get headline numbers. Returns a
    dict. On failure, fields are marked `[err]`. We use psql via SSH so
    this works from laptop or VPS equally."""
    since = period.start.strftime("%Y-%m-%d %H:%M:%S")
    until = period.end.strftime("%Y-%m-%d %H:%M:%S")

    def q(sql: str) -> str:
        is_vps = Path("/opt/arbo/.env").exists()
        if is_vps:
            cmd = ["sudo", "-u", "arbo", "psql", "-d", "arbo", "-A", "-t", "-c", sql]
        else:
            cmd = ["ssh", "arbo-dublin",
                   f"sudo -u arbo psql -d arbo -A -t -c \"{sql}\""]
        return run(cmd, timeout=60).strip()

    trades_per_strategy = q(f"""
        SELECT strategy, COUNT(*),
               COUNT(*) FILTER (WHERE status='sold'),
               COALESCE(SUM(actual_pnl),0)::numeric(10,2),
               COUNT(*) FILTER (WHERE status='sold' AND actual_pnl>0)
        FROM paper_trades
        WHERE placed_at >= '{since}' AND placed_at < '{until}'
          AND COALESCE(notes,'') NOT LIKE 'pre_reset%%'
        GROUP BY strategy ORDER BY strategy
    """)

    shadow_progress = q(f"""
        SELECT strategy,
               COUNT(*) FILTER (WHERE resolution_outcome IS NOT NULL AND
                                resolution_ts >= '{since}' AND
                                resolution_ts < '{until}')
        FROM shadow_variant_signals
        GROUP BY strategy ORDER BY strategy
    """)

    errors = q(f"""
        SELECT COUNT(*) FROM paper_trades
        WHERE placed_at >= '{since}' AND placed_at < '{until}'
          AND status = 'lost'
    """)

    return {
        "trades_per_strategy": trades_per_strategy,
        "shadow_progress": shadow_progress,
        "losing_trades": errors,
    }


def format_metrics_table(m: dict[str, Any]) -> str:
    """Pretty-print metrics as a markdown section."""
    lines = ["### Trades by strategy", "", "| Strategy | Opened | Closed | Net PnL | Wins |",
             "|---|---|---|---|---|"]
    for row in (m["trades_per_strategy"] or "").splitlines():
        parts = row.split("|")
        if len(parts) >= 5:
            lines.append("| " + " | ".join(parts) + " |")
    lines += ["", "### Shadow resolutions this period", ""]
    for row in (m["shadow_progress"] or "").splitlines():
        lines.append(f"- {row}")
    lines += ["", f"### Lost trades: {m['losing_trades']}"]
    return "\n".join(lines)


def collect_prior_summaries(period: Period) -> str:
    """For cascade: a weekly/monthly/yearly period reads the lower-level
    summaries whose period falls inside its own range."""
    if period.kind == "daily":
        return "(daily is the base level)"
    lower = {"weekly": "daily", "monthly": "weekly", "yearly": "monthly"}[period.kind]
    folder = SUMMARIES / lower
    if not folder.exists():
        return "(no lower-level summaries available)"
    chunks = []
    for f in sorted(folder.iterdir()):
        if not f.name.endswith(".md"):
            continue
        try:
            # Parse date from filename
            # daily: 2026-04-17.md
            # weekly: 2026-W16.md
            # monthly: 2026-04.md
            file_start = _parse_label(f.stem, lower)
            if not file_start:
                continue
            if period.start <= file_start < period.end:
                chunks.append(f"### {f.stem}\n\n{f.read_text(encoding='utf-8')}")
        except Exception:
            continue
    body = "\n\n---\n\n".join(chunks)
    # Bound size — a monthly summary reading 4 weekly each up to 5kb is fine
    return body[:80_000] or "(no matching lower-level summaries found)"


def _parse_label(label: str, kind: str) -> datetime | None:
    try:
        if kind == "daily":
            d = date.fromisoformat(label)
            return datetime(d.year, d.month, d.day, tzinfo=UTC)
        if kind == "weekly":
            y, w = label.split("-W")
            return datetime.fromisocalendar(int(y), int(w), 1).replace(tzinfo=UTC)
        if kind == "monthly":
            y, m = label.split("-")
            return datetime(int(y), int(m), 1, tzinfo=UTC)
        if kind == "yearly":
            return datetime(int(label), 1, 1, tzinfo=UTC)
    except Exception:
        return None
    return None


# ─── LLM synthesis ────────────────────────────────────────────────


def llm_synthesis(period: Period, context: dict[str, str]) -> str | None:
    """Call Gemini to produce the narrative. Returns None on any error
    so callers fall back to stats-only."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    prompt = LLM_PROMPT_TEMPLATE.format(
        period_label=f"{period.kind.upper()} {period.label}",
        learnings_diff=context["learnings_diff"][:12_000],
        git_log=context["git_log"][:6_000],
        metrics_table=context["metrics_table"][:3_000],
        journal_excerpt=context["journal_excerpt"][:30_000],
        prior_summaries=context["prior_summaries"][:25_000],
    )
    try:
        # Lazy import so systems without the SDK still run stats-only
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 8000},
            request_options={"timeout": LLM_TIMEOUT_S},
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"[llm_synthesis] failed: {e}", file=sys.stderr)
        return None


# ─── Assembly ─────────────────────────────────────────────────────


def build_summary(period: Period) -> str:
    """Collect all inputs and produce the final markdown."""
    context = {
        "learnings_diff": collect_learnings_diff(period),
        "git_log": collect_git_log(period),
        "journal_excerpt": collect_journal(period),
        "prior_summaries": collect_prior_summaries(period),
    }
    metrics = collect_metrics(period)
    context["metrics_table"] = format_metrics_table(metrics)

    narrative = llm_synthesis(period, context) or (
        "_LLM unavailable — stats-only summary. Re-run with GEMINI_API_KEY "
        "to add narrative analysis._"
    )

    return f"""# {period.kind.title()} Retrospective — {period.label}

## Metadata
- generated_at: {now_utc().isoformat()}
- period_kind: {period.kind}
- period_start: {period.start.isoformat()}
- period_end: {period.end.isoformat()}
- llm_model: {GEMINI_MODEL if "LLM unavailable" not in narrative else "none"}

## Trading metrics

{context["metrics_table"]}

## LLM analysis

{narrative}

## Appendix A — LEARNINGS additions

{context["learnings_diff"]}

## Appendix B — Git commits

{context["git_log"]}

## Appendix C — Journal highlights

```
{context["journal_excerpt"]}
```
"""


# ─── Persistence ──────────────────────────────────────────────────


def write_summary(period: Period, content: str, commit: bool) -> bool:
    """Write the summary file. Returns True if content changed."""
    period.out_path.parent.mkdir(parents=True, exist_ok=True)
    old = period.out_path.read_text(encoding="utf-8") if period.out_path.exists() else ""
    # Strip `generated_at` line when comparing so timestamp-only diffs
    # don't trigger re-commits.
    def strip_meta(s: str) -> str:
        return re.sub(r"- generated_at:.*\n", "", s)
    if strip_meta(old) == strip_meta(content):
        print(f"[write_summary] {period.out_path} unchanged")
        return False
    period.out_path.write_text(content, encoding="utf-8")
    print(f"[write_summary] wrote {period.out_path} ({len(content)} bytes)")
    if commit:
        _git_commit_summary(period)
    return True


def _git_commit_summary(period: Period) -> None:
    try:
        run(
            ["git", "-C", str(REPO_ROOT), "add", str(period.out_path)],
            timeout=15,
        )
        status = run(
            ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--name-only"],
            timeout=15,
        )
        if not status.strip():
            return
        msg = (
            f"docs(summary): {period.kind} retrospective {period.label}\n\n"
            f"Auto-generated by scripts/retrospective.py."
        )
        run(
            ["git", "-C", str(REPO_ROOT), "commit", "-m", msg, "--no-verify"],
            timeout=15,
        )
        # Non-interactive push: on the VPS (arbo user) there are no git
        # credentials configured for push, so the command would hang on
        # username prompt. GIT_TERMINAL_PROMPT=0 makes it fail fast.
        # Summaries are still written to disk and committed locally —
        # someone with push access can pick them up later.
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            push_res = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "push", "origin", "HEAD:main"],
                env=env, capture_output=True, text=True, timeout=60,
            )
            if push_res.returncode == 0:
                print(f"[commit] pushed: {push_res.stderr.strip()[:200]}")
            else:
                print(f"[commit] push skipped (no creds): {push_res.stderr.strip()[:100]}")
        except subprocess.TimeoutExpired:
            print("[commit] push timed out")
    except Exception as e:
        print(f"[commit] skipped: {e}", file=sys.stderr)


# ─── Cascade ──────────────────────────────────────────────────────


def maybe_cascade(period: Period, commit: bool) -> None:
    """After a daily summary, if we just wrote the last day of a week,
    trigger weekly. Similar for weekly→monthly, monthly→yearly.

    Guard: only cascade when the period JUST COMPLETED — i.e., the
    parent's `end` boundary coincides with `period.end`. That way we
    don't re-trigger the weekly cascade mid-week on a backfill run.
    """
    if period.kind == "yearly":
        return
    parent_kind = {"daily": "weekly", "weekly": "monthly", "monthly": "yearly"}[period.kind]
    parent = _find_parent(period, parent_kind)
    if not parent:
        return
    if parent.end != period.end:
        return  # this period doesn't close the parent
    print(f"[cascade] {period.kind}/{period.label} closes {parent_kind}/{parent.label} — running")
    content = build_summary(parent)
    if write_summary(parent, content, commit):
        maybe_cascade(parent, commit)


def _find_parent(period: Period, parent_kind: str) -> Period | None:
    if parent_kind == "weekly":
        # Week containing this day
        iso = period.start.isocalendar()
        return resolve_period("weekly", f"{iso[0]}-W{iso[1]:02d}")
    if parent_kind == "monthly":
        return resolve_period("monthly", f"{period.start.year}-{period.start.month:02d}")
    if parent_kind == "yearly":
        return resolve_period("yearly", str(period.start.year))
    return None


# ─── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("period", choices=["daily", "weekly", "monthly", "yearly"])
    ap.add_argument("--date", help="YYYY-MM-DD for daily", default=None)
    ap.add_argument("--week", help="YYYY-WNN for weekly", default=None)
    ap.add_argument("--month", help="YYYY-MM for monthly", default=None)
    ap.add_argument("--year", help="YYYY for yearly", default=None)
    ap.add_argument("--no-commit", action="store_true",
                    help="Skip git commit/push (for testing or VPS runs — "
                         "files still persist on disk)")
    ap.add_argument("--no-cascade", action="store_true",
                    help="Don't trigger next-level summary even if period closed")
    args = ap.parse_args()

    arg_for_kind = {
        "daily": args.date, "weekly": args.week,
        "monthly": args.month, "yearly": args.year,
    }[args.period]

    period = resolve_period(args.period, arg_for_kind)
    print(f"[retrospective] {period.kind} {period.label} "
          f"({period.start} → {period.end})")

    content = build_summary(period)
    changed = write_summary(period, content, commit=not args.no_commit)

    if not args.no_cascade and changed:
        maybe_cascade(period, commit=not args.no_commit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
