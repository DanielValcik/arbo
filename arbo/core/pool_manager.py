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
