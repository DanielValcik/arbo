"""ShadowOrchestrator — multi-variant signal evaluation (Phase 1 skeleton).

Forks one market signal to multiple variant configs. Each variant evaluates
the same signal with its own quality-gate parameters; results logged with
variant_id for paired-sample champion-challenger statistics.

Phase 1 status: SKELETON ONLY — class defined, NOT wired into task loops.
The existing `b3_15m_shadow.py` continues to collect 144-signal dataset
unchanged (do NOT replace, see Framework §11.13 risk #1).

Phase 2 (next): wire orchestrator into main_rdh task loop, write to new
table `shadow_variant_signals` for paired-sample analytics.

Spec: docs/RAPID_MODEL_DISCOVERY.md §8 + docs/VARIANT_LEADERBOARD_SPEC.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from arbo.core.variant_pool import VariantConfig, get_active_variants
from arbo.utils.logger import get_logger

logger = get_logger("shadow_orchestrator")


class SignalProtocol(Protocol):
    """Common signal interface (B3Signal, B315MSignal, future strategies)."""
    direction: int
    edge: float
    btc_at_start: float
    btc_now: float
    sigma_per_min: float


@dataclass
class VariantDecision:
    """Result of one variant evaluating one signal."""
    variant_id: str
    qualified: bool                # passed all gates
    direction: int | None          # 1 (UP), -1 (DOWN), None if no signal
    entry_price: float | None
    edge: float
    skip_reason: str | None        # if not qualified, why
    signal_ts: float
    decision_ts: float


class ShadowOrchestrator:
    """Forks one signal to N variants, evaluates each independently.

    Usage (Phase 2 — not yet wired):
        orch = ShadowOrchestrator("B3_15M", evaluator_fn=my_eval)
        for sig in scanner.scan(...):
            decisions = orch.evaluate_all(sig)
            for d in decisions:
                paper_engine.simulate(d)  # log per variant_id
    """

    def __init__(
        self,
        strategy: str,
        evaluator: Callable[[SignalProtocol, dict[str, Any]], VariantDecision] | None = None,
    ) -> None:
        """
        Args:
            strategy: strategy name (loaded from arbo/config/variants/<strategy>/)
            evaluator: function (signal, params_dict) → VariantDecision.
                       Caller supplies the strategy-specific gate logic
                       (champion's existing scan() refactored to take params).
                       If None, only loads variants but cannot evaluate yet.
        """
        self.strategy = strategy
        self.variants: list[VariantConfig] = get_active_variants(strategy)
        self.evaluator = evaluator
        self._signal_count = 0
        self._decision_count = 0
        logger.info(
            "shadow_orchestrator_initialized",
            strategy=strategy,
            n_variants=len(self.variants),
            variant_ids=[v.variant_id for v in self.variants],
            evaluator_provided=evaluator is not None,
        )

    def evaluate_all(self, signal: SignalProtocol) -> list[VariantDecision]:
        """Evaluate one signal across all active variants.

        Returns one VariantDecision per variant. The orchestrator does NOT
        execute trades — caller is responsible for paper or live simulation
        per decision.

        Phase 1: this method requires self.evaluator to be wired. Returns
        empty list if no evaluator (skeleton mode).
        """
        if self.evaluator is None:
            logger.debug("shadow_orchestrator_no_evaluator", strategy=self.strategy)
            return []

        self._signal_count += 1
        signal_ts = time.time()
        decisions: list[VariantDecision] = []
        for v in self.variants:
            try:
                d = self.evaluator(signal, v.params)
                d.variant_id = v.variant_id
                d.signal_ts = signal_ts
                d.decision_ts = time.time()
                decisions.append(d)
                self._decision_count += 1
            except Exception as e:
                logger.warning(
                    "shadow_orchestrator_variant_eval_error",
                    strategy=self.strategy,
                    variant_id=v.variant_id,
                    error=str(e),
                )
        return decisions

    def reload_variants(self) -> int:
        """Re-read YAML pool from disk. Returns new variant count."""
        old_count = len(self.variants)
        self.variants = get_active_variants(self.strategy)
        new_count = len(self.variants)
        logger.info(
            "shadow_orchestrator_reload",
            strategy=self.strategy,
            old_count=old_count,
            new_count=new_count,
        )
        return new_count

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_variants": len(self.variants),
            "variant_ids": [v.variant_id for v in self.variants],
            "signals_evaluated": self._signal_count,
            "decisions_logged": self._decision_count,
        }
