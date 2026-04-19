"""PoolManager — Phase 2C.A.

Takes a Hypothesis, writes a new challenger YAML, enforces pool limits
(max 8 active variants per strategy), and appends a LEARNINGS.md entry.

YAML is written atomically (write to .tmp, fsync, rename).

Spec: docs/PROJECT_PARALLEL_ROADMAP.md §Phase 2C.A
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from arbo.core.hypothesis_generator import Hypothesis
from arbo.core.variant_pool import (
    VariantConfig,
    get_champion,
    get_active_variants,
    load_variants,
)
from arbo.utils.logger import get_logger

logger = get_logger("pool_manager")

MAX_ACTIVE_CHALLENGERS = 8
MIN_RETIRE_SHADOW_N = 30  # don't retire a challenger with < 30 shadow signals


def _pool_dir(strategy: str) -> Path:
    """Return the YAML directory for a strategy."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "arbo" / "config" / "variants" / strategy.lower()


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML atomically (write to .tmp, fsync, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _hypothesis_to_variant_dict(
    hypo: Hypothesis, strategy: str, champion_params: dict[str, Any]
) -> dict[str, Any]:
    """Build the YAML dict for the new challenger."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # New params = champion params with hypothesis mutations applied
    merged = dict(champion_params)
    for k, v in hypo.param_changes.items():
        merged[k] = v

    notes_lines = [
        f"Auto-generated {ts} by hypothesis_generator.",
        f"Rationale: {hypo.rationale}",
        "Parameter changes vs champion:",
    ]
    for k, v in hypo.param_changes.items():
        cur = champion_params.get(k)
        notes_lines.append(f"  - {k}: {cur} -> {v}")
    notes_lines.append("Shadow only, no capital risk.")

    return {
        "variant_id": hypo.variant_id,
        "strategy": strategy,
        "status": "challenger",
        "parent_variant": hypo.parent_variant_id,
        "auto_generated": True,
        "created_at": ts,
        "notes": "\n".join(notes_lines),
        "params": merged,
    }


async def _retire_worst_challenger(strategy: str) -> str | None:
    """If pool at capacity, retire worst-performing challenger.

    Worst = lowest shadow_pnl_per_share over at least MIN_RETIRE_SHADOW_N
    signals (exclude young variants from retirement). Writes their YAML
    with status='retired'. Returns retired variant_id or None if no
    eligible variant.
    """
    from arbo.utils.db import get_session_factory
    import sqlalchemy as sa

    active = get_active_variants(strategy)
    challengers = [v for v in active if v.status == "challenger"]
    if len(active) < MAX_ACTIVE_CHALLENGERS:
        return None
    if not challengers:
        logger.warning("pool_at_capacity_no_challengers", strategy=strategy)
        return None

    factory = get_session_factory()
    worst_id: str | None = None
    worst_pnl: float = float("inf")
    async with factory() as session:
        for ch in challengers:
            r = await session.execute(
                sa.text("""
                    SELECT
                        COUNT(*) FILTER (WHERE qualified) AS n_qual,
                        COALESCE(SUM(would_pnl_per_share) FILTER (
                            WHERE qualified AND would_pnl_per_share IS NOT NULL
                        ), 0) AS pnl
                    FROM shadow_variant_signals
                    WHERE strategy = :s AND variant_id = :v
                """),
                {"s": strategy, "v": ch.variant_id},
            )
            row = r.first()
            n_qual = int(row.n_qual) if row else 0
            pnl = float(row.pnl) if row else 0.0
            if n_qual < MIN_RETIRE_SHADOW_N:
                continue
            if pnl < worst_pnl:
                worst_pnl = pnl
                worst_id = ch.variant_id

    if worst_id is None:
        logger.info(
            "pool_no_retirable_challenger",
            strategy=strategy,
            reason=f"all challengers below {MIN_RETIRE_SHADOW_N} shadow signals",
        )
        return None

    # Rewrite that variant's YAML with status=retired
    src = _pool_dir(strategy) / f"{worst_id}.yaml"
    if not src.exists():
        logger.warning("pool_retire_yaml_missing", path=str(src))
        return None
    try:
        with open(src) as f:
            data = yaml.safe_load(f) or {}
        data["status"] = "retired"
        data["retired_at"] = datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        _atomic_write_yaml(src, data)
        logger.info(
            "pool_retired",
            strategy=strategy,
            variant_id=worst_id,
            shadow_pnl=round(worst_pnl, 4),
        )
        return worst_id
    except Exception as e:
        logger.warning(
            "pool_retire_error",
            strategy=strategy,
            variant_id=worst_id,
            error=str(e),
        )
        return None


def _append_learnings(hypo: Hypothesis, strategy: str, retired: str | None) -> None:
    """Append a single auto-gen entry to docs/STRATEGY_OPTIMIZATION_LEARNINGS.md."""
    repo_root = Path(__file__).resolve().parents[2]
    learnings = repo_root / "docs" / "STRATEGY_OPTIMIZATION_LEARNINGS.md"
    if not learnings.exists():
        logger.info("learnings_log_missing", path=str(learnings))
        return
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = [
        "",
        f"### {ts} — Auto-Challenger generated: `{hypo.variant_id}` ({strategy})",
        "",
        f"- **Rationale:** {hypo.rationale}",
        f"- **Parameter changes:** {hypo.param_changes}",
        f"- **Parent:** `{hypo.parent_variant_id}`",
    ]
    if retired:
        entry.append(f"- **Retired to make room:** `{retired}`")
    entry.append("- **Status:** shadow (no capital risk) pending 24h CEO veto window.")
    entry.append("")
    try:
        with open(learnings, "a", encoding="utf-8") as f:
            f.write("\n".join(entry) + "\n")
    except Exception as e:
        logger.warning("learnings_append_error", error=str(e))


async def promote(
    variant_id: str, strategy: str, *, approved_by: str = "system"
) -> tuple[bool, str | None]:
    """Atomically promote a challenger to champion (Phase 2C.D).

    Steps:
      1. Load challenger + current champion YAMLs
      2. Archive current champion YAML → _archive/<champion_id>_<ts>.yaml
      3. Rewrite current champion YAML with status='retired'
      4. Rewrite challenger YAML with status='champion'
      5. Append LEARNINGS.md entry

    Returns (success, reason).
    """
    pool = load_variants(strategy)
    champ = get_champion(strategy)
    if champ is None:
        return False, "no current champion"
    if variant_id == champ.variant_id:
        return False, f"{variant_id} is already champion"
    target = next((v for v in pool if v.variant_id == variant_id), None)
    if target is None:
        return False, f"variant {variant_id} not in pool"
    if target.status not in {"challenger", "incubate"}:
        return False, f"cannot promote from status={target.status}"

    pool_dir = _pool_dir(strategy)
    champ_yaml = pool_dir / f"{champ.variant_id}.yaml"
    target_yaml = pool_dir / f"{variant_id}.yaml"
    archive_dir = pool_dir / "_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if not champ_yaml.exists() or not target_yaml.exists():
        return False, "champion or challenger YAML missing"

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        with open(champ_yaml) as f:
            champ_data = yaml.safe_load(f) or {}
        with open(target_yaml) as f:
            tgt_data = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"yaml read error: {e}"

    archive_path = archive_dir / (
        f"{champ.variant_id}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    try:
        _atomic_write_yaml(archive_path, champ_data)
    except Exception as e:
        return False, f"archive write error: {e}"

    champ_data["status"] = "retired"
    champ_data["retired_at"] = ts
    champ_data["retired_reason"] = f"replaced by {variant_id} (approved_by={approved_by})"

    tgt_data["status"] = "champion"
    tgt_data["promoted_at"] = ts
    tgt_data["promoted_by"] = approved_by
    tgt_data["parent_variant"] = champ.variant_id

    try:
        _atomic_write_yaml(champ_yaml, champ_data)
        _atomic_write_yaml(target_yaml, tgt_data)
    except Exception as e:
        return False, f"yaml write error: {e}"

    # LEARNINGS.md entry
    try:
        repo_root = Path(__file__).resolve().parents[2]
        learnings = repo_root / "docs" / "STRATEGY_OPTIMIZATION_LEARNINGS.md"
        if learnings.exists():
            entry = [
                "",
                f"### {ts} — Promotion: `{variant_id}` becomes champion of {strategy}",
                "",
                f"- **Previous champion:** `{champ.variant_id}` (archived to `_archive/{archive_path.name}`)",
                f"- **Approved by:** {approved_by}",
                f"- **Param changes:** {_param_diff(champ.params, tgt_data.get('params', {}))}",
                "",
            ]
            with open(learnings, "a", encoding="utf-8") as f:
                f.write("\n".join(entry) + "\n")
    except Exception as e:
        logger.warning("promote_learnings_error", error=str(e))

    logger.info(
        "pool_promoted",
        strategy=strategy,
        new_champion=variant_id,
        ex_champion=champ.variant_id,
        approved_by=approved_by,
    )
    return True, None


async def promote_to_incubate(
    variant_id: str,
    strategy: str,
    *,
    capital_pct: float = 0.20,
    approved_by: str = "system",
) -> tuple[bool, str | None]:
    """Move a challenger into the `incubate` stage.

    Unlike `promote()` which does a full champion swap, this keeps the
    current champion in place and flips the challenger to `incubate`.
    Live strategy code (e.g. strategy_b2) reads active variants via
    `get_active_variants()` and routes `capital_pct` of candidate signals
    through the incubating variant's params. The remaining
    `1 - capital_pct` continues to use the champion.

    The incubate stage is a live canary: the variant sees real fills,
    real spread, real slippage — none of which shadow evaluation can
    capture. After sufficient live data accumulates, the watchdog
    (`b2_watchdog._eval_cycle`) either escalates incubate → champion
    via `promote()` or reverts via `revert_incubate_to_challenger()`.

    Args:
        variant_id: challenger variant YAML id.
        strategy: e.g. "B2".
        capital_pct: fraction of live capital/signals routed to the
            incubating variant. Clamped to [0.05, 0.50]. Default 0.20.
        approved_by: audit string.

    Returns:
        (success, reason_if_failed). No YAML changes if success=False.
    """
    if not 0.05 <= capital_pct <= 0.50:
        return False, f"capital_pct {capital_pct} outside [0.05, 0.50]"

    pool = load_variants(strategy)
    target = next((v for v in pool if v.variant_id == variant_id), None)
    if target is None:
        return False, f"variant {variant_id} not in pool"
    if target.status != "challenger":
        return False, (
            f"cannot move to incubate from status={target.status}; "
            f"promote_to_incubate only accepts challengers"
        )

    # Ensure no OTHER variant is already in incubate — we only
    # canary one at a time to keep attribution clean.
    existing_incubate = next((v for v in pool if v.status == "incubate"), None)
    if existing_incubate is not None:
        return False, (
            f"{existing_incubate.variant_id} is already in incubate; "
            f"resolve that one (escalate/revert) before starting a new canary"
        )

    target_yaml = _pool_dir(strategy) / f"{variant_id}.yaml"
    if not target_yaml.exists():
        return False, f"{variant_id}.yaml missing"

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        with open(target_yaml) as f:
            data = yaml.safe_load(f) or {}
        data["status"] = "incubate"
        data["incubated_at"] = ts
        data["incubated_by"] = approved_by
        data["incubate_capital_pct"] = round(capital_pct, 3)
        _atomic_write_yaml(target_yaml, data)
    except Exception as e:
        return False, f"yaml write error: {e}"

    # LEARNINGS log
    try:
        repo_root = Path(__file__).resolve().parents[2]
        learnings = repo_root / "docs" / "STRATEGY_OPTIMIZATION_LEARNINGS.md"
        if learnings.exists():
            entry = [
                "",
                f"### {ts} — Canary: `{variant_id}` → incubate ({strategy})",
                "",
                f"- **Capital share:** {capital_pct * 100:.0f}% of live signals",
                f"- **Approved by:** {approved_by}",
                "- **Escalation:** watchdog evaluates live P(better) after N≥15 paired trades.",
                "",
            ]
            with open(learnings, "a", encoding="utf-8") as f:
                f.write("\n".join(entry) + "\n")
    except Exception as e:
        logger.warning("incubate_learnings_error", error=str(e))

    logger.info(
        "pool_incubated",
        strategy=strategy,
        variant_id=variant_id,
        capital_pct=capital_pct,
        approved_by=approved_by,
    )
    return True, None


async def revert_incubate_to_challenger(
    variant_id: str,
    strategy: str,
    *,
    reason: str = "underperformed_live",
    decided_by: str = "watchdog",
) -> tuple[bool, str | None]:
    """Demote an incubating variant back to challenger (shadow-only).

    Used when live data shows the variant is NOT better than the
    current champion. Capital routing stops; the variant goes back to
    shadow evaluation.
    """
    pool = load_variants(strategy)
    target = next((v for v in pool if v.variant_id == variant_id), None)
    if target is None:
        return False, f"variant {variant_id} not in pool"
    if target.status != "incubate":
        return False, f"cannot revert from status={target.status}"

    target_yaml = _pool_dir(strategy) / f"{variant_id}.yaml"
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        with open(target_yaml) as f:
            data = yaml.safe_load(f) or {}
        data["status"] = "challenger"
        data["reverted_at"] = ts
        data["reverted_reason"] = reason
        data["reverted_by"] = decided_by
        # Strip incubate-specific keys
        data.pop("incubate_capital_pct", None)
        _atomic_write_yaml(target_yaml, data)
    except Exception as e:
        return False, f"yaml write error: {e}"

    logger.info(
        "pool_incubate_reverted",
        strategy=strategy,
        variant_id=variant_id,
        reason=reason,
        decided_by=decided_by,
    )
    return True, None


async def veto(
    variant_id: str, strategy: str, *, vetoed_by: str = "ceo"
) -> tuple[bool, str | None]:
    """Retire a variant immediately (Phase 2C.F).

    Cannot veto the current champion (would leave strategy headless).
    """
    champ = get_champion(strategy)
    if champ is not None and variant_id == champ.variant_id:
        return False, "cannot veto the current champion"
    path = _pool_dir(strategy) / f"{variant_id}.yaml"
    if not path.exists():
        return False, f"{variant_id}.yaml not found"
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data["status"] = "retired"
        data["retired_at"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        data["retired_reason"] = f"CEO veto by {vetoed_by}"
        _atomic_write_yaml(path, data)
        logger.info(
            "pool_vetoed",
            strategy=strategy, variant_id=variant_id, vetoed_by=vetoed_by,
        )
        return True, None
    except Exception as e:
        return False, f"error: {e}"


def _param_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return {key: (a_val, b_val)} for params that differ."""
    out: dict[str, tuple[Any, Any]] = {}
    for k, v in b.items():
        av = a.get(k)
        if av != v:
            out[k] = (av, v)
    return out


async def commit(hypo: Hypothesis, strategy: str) -> tuple[bool, str | None]:
    """Persist a Hypothesis as a new challenger YAML.

    Returns (success, reason). If pool at capacity, attempts to retire
    the worst challenger first.
    """
    # Retire if at cap
    active = get_active_variants(strategy)
    retired: str | None = None
    if len(active) >= MAX_ACTIVE_CHALLENGERS:
        retired = await _retire_worst_challenger(strategy)
        if retired is None:
            return False, "pool at capacity; no retirable challenger"

    champ = get_champion(strategy)
    if champ is None:
        return False, "no champion in pool"

    target = _pool_dir(strategy) / f"{hypo.variant_id}.yaml"
    if target.exists():
        return False, f"YAML already exists at {target}"

    try:
        yaml_data = _hypothesis_to_variant_dict(hypo, strategy, champ.params)
        _atomic_write_yaml(target, yaml_data)
    except Exception as e:
        logger.warning("pool_write_error", path=str(target), error=str(e))
        return False, f"write error: {e}"

    _append_learnings(hypo, strategy, retired)

    logger.info(
        "pool_challenger_committed",
        strategy=strategy,
        variant_id=hypo.variant_id,
        path=str(target),
        retired=retired,
    )
    return True, None
