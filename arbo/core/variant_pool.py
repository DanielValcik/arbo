"""Declarative variant pool loader for Rapid Mode (Framework §11).

Loads YAML configs from `arbo/config/variants/<strategy>/*.yaml` into
VariantConfig dataclasses. Used by ShadowOrchestrator and the dashboard
Variant Leaderboard API.

Phase 1: static loading. YAML edits + service restart to change pool.
Phase 2+: AdaptiveConfig integration for runtime overrides.

Spec: docs/VARIANT_LEADERBOARD_SPEC.md
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML required for variant_pool. Install with: pip install pyyaml"
    ) from e


VARIANTS_ROOT = Path(__file__).resolve().parents[1] / "config" / "variants"

# Valid status values (FSM from Framework §11.10)
VALID_STATUS = {"shadow", "incubate", "live", "scaled", "champion", "challenger", "retired"}


@dataclass(frozen=True)
class VariantConfig:
    """Single variant of a strategy — a named parameter set with status.

    Fields:
        variant_id: unique ID within strategy (e.g. "champion_v1", "ch_edge_tight")
        strategy: parent strategy name ("B3", "B3_15M", "C", etc.)
        status: lifecycle stage (champion | challenger | shadow | retired | ...)
        params: parameter dict (mirrors quality_gate constants per strategy)
        notes: human-readable description
        parent_variant: lineage (for BO-derived configs, None for seeds)
    """

    variant_id: str
    strategy: str
    status: str
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    parent_variant: str | None = None
    # Canary promotion: set on variants with status=="incubate" by
    # pool_manager.promote_to_incubate. Fraction of live signals
    # routed to this variant's gate. None for non-incubating variants.
    incubate_capital_pct: float | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"Invalid status {self.status!r} for variant {self.variant_id}. "
                f"Must be one of {sorted(VALID_STATUS)}."
            )


def _strategy_dir(strategy: str, root: Path | None = None) -> Path:
    """Resolve directory for a strategy's variant YAMLs."""
    base = root if root is not None else VARIANTS_ROOT
    # Normalize strategy name to lowercase for filesystem
    return base / strategy.lower()


def load_variants(strategy: str, root: Path | None = None) -> list[VariantConfig]:
    """Load all YAML variants for a strategy.

    Args:
        strategy: strategy name (e.g. "B3_15M", "B3", "C").
        root: override variants root dir (for tests).

    Returns:
        List of VariantConfig, sorted by variant_id. Empty if dir missing.
    """
    sdir = _strategy_dir(strategy, root)
    if not sdir.exists():
        return []

    variants: list[VariantConfig] = []
    for yaml_path in sorted(sdir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Variant YAML {yaml_path} must be a dict at top level")
        try:
            v = VariantConfig(
                variant_id=raw["variant_id"],
                strategy=raw.get("strategy", strategy),
                status=raw["status"],
                params=raw.get("params", {}) or {},
                notes=raw.get("notes", ""),
                parent_variant=raw.get("parent_variant"),
                incubate_capital_pct=raw.get("incubate_capital_pct"),
            )
        except KeyError as e:
            raise ValueError(f"Variant YAML {yaml_path} missing required field: {e}")
        variants.append(v)

    return variants


def get_champion(strategy: str, root: Path | None = None) -> VariantConfig | None:
    """Return the single champion variant, or None if none / multiple found."""
    variants = load_variants(strategy, root)
    champions = [v for v in variants if v.status == "champion"]
    if len(champions) == 0:
        return None
    if len(champions) > 1:
        raise ValueError(
            f"Multiple champions for {strategy}: "
            f"{[v.variant_id for v in champions]}. Only one allowed."
        )
    return champions[0]


def get_active_variants(strategy: str, root: Path | None = None) -> list[VariantConfig]:
    """Return variants in any active state (NOT retired)."""
    return [v for v in load_variants(strategy, root) if v.status != "retired"]
