"""Runtime Adaptive Config for B3 Watchdog autonomous parameter changes.

Watchdog reads anomaly data, decides on parameter adjustments, and writes
overrides here. strategy_b3.py reads parameters via get() which returns
the override if set, otherwise the default from b3_quality_gate.py.

Key safety features:
- 3-tier bounds: Tier 1 (autonomous), Tier 2 (autonomous+flag), Tier 3 (never)
- Every change logged to in-memory audit trail (persisted to DB by Watchdog)
- Auto-revert: Watchdog checks each change after 50 trades
- Restart resets all overrides to defaults (safe fallback)

See: docs/B3_WATCHDOG_SPEC.md (Autonomní Decision Engine)
     docs/TECHNICAL_DECISIONS.md (TD-017)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("adaptive_config")


@dataclass
class ParamBounds:
    """Safety bounds for a single parameter."""

    hard_min: float  # Absolute minimum (never go below)
    hard_max: float  # Absolute maximum (never go above)
    autonomous_min: float  # Watchdog can change within this range without flag
    autonomous_max: float  # (Tier 1 range)
    tier: int  # 1 = autonomous, 2 = autonomous+flag, 3 = never


@dataclass
class ConfigChange:
    """Audit record for a parameter change."""

    change_id: int
    timestamp: float
    param_name: str
    old_value: float
    new_value: float
    tier: int
    reason: str
    trigger_id: str | None = None
    trade_count_at_change: int = 0
    wr_at_change: float = 0.0
    status: str = "ACTIVE"  # ACTIVE | REVERTED | PROMOTED
    evaluation_result: str | None = None  # KEPT | REVERTED | None (pending)


# ── Parameter Definitions ────────────────────────────────────────────

# Tier 1: Watchdog can change autonomously (within bounds)
# Tier 2: Watchdog can change, but flags for CEO review
# Tier 3: Never autonomous (hardcoded, requires deploy)

PARAM_BOUNDS: dict[str, ParamBounds] = {
    # Tier 1 — Fully autonomous
    "LIVE_MAX_VELOCITY": ParamBounds(
        hard_min=30, hard_max=100, autonomous_min=40, autonomous_max=80, tier=1,
    ),
    "LIVE_MAX_DIR_DELTA": ParamBounds(
        hard_min=5, hard_max=40, autonomous_min=10, autonomous_max=25, tier=1,
    ),
    "LIVE_MIN_EDGE": ParamBounds(
        hard_min=0.20, hard_max=0.80, autonomous_min=0.30, autonomous_max=0.60, tier=1,
    ),
    "POSITION_PCT": ParamBounds(
        hard_min=0.005, hard_max=0.10, autonomous_min=0.013, autonomous_max=0.052, tier=1,
    ),
    "EDGE_SCALING": ParamBounds(
        hard_min=2.0, hard_max=25.0, autonomous_min=5.0, autonomous_max=20.0, tier=1,
    ),
    "LIVE_MIN_BTC_MOVE": ParamBounds(
        hard_min=20, hard_max=150, autonomous_min=30, autonomous_max=80, tier=1,
    ),
    # Tier 2 — Autonomous with escalation
    "SIGMA_SCALE": ParamBounds(
        hard_min=0.25, hard_max=0.50, autonomous_min=0.30, autonomous_max=0.40, tier=2,
    ),
    "ENTRY_THRESHOLD": ParamBounds(
        hard_min=0.01, hard_max=0.05, autonomous_min=0.015, autonomous_max=0.030, tier=2,
    ),
    "MIN_ENTRY_MIN": ParamBounds(
        hard_min=1, hard_max=3, autonomous_min=1, autonomous_max=2, tier=2,
    ),
    "MAX_ENTRY_MIN": ParamBounds(
        hard_min=2, hard_max=4, autonomous_min=2, autonomous_max=4, tier=2,
    ),
}

# Tier 3 parameters are NOT in PARAM_BOUNDS — they cannot be changed at runtime.
# MAX_BET_SIZE, MIN_ORDER_SIZE, DAILY_LOSS_LIMIT, LIVE_MAX_FILL_PRICE,
# REQUIRE_CHAINLINK, execution_mode, capital allocation.


# Per-strategy Tier 1 autonomous bounds for B3_15M (15-min variant).
# Calibrated from shadow autoresearch rank #1 (2026-04-12, 144 signals).
# Notes vs B3 (5-min):
#   - LIVE_MIN_EDGE: same range, different start (both 0.30)
#   - LIVE_MIN_BTC_MOVE: wider autonomous range (0-30) — edge filter does heavy lifting
#   - LIVE_MAX_BTC_MOVE: NEW — 15-min specific reversal cap
#   - LIVE_MAX_VELOCITY: higher range (80-120) — 15-min absorbs faster moves
#   - LIVE_MAX_DIR_DELTA: wider (15-35) — 15-min tolerates more CL lag
#   - LIVE_MAX_MARKET_GAP: NEW — 15-min specific gap filter
PARAM_BOUNDS_B3_15M: dict[str, ParamBounds] = {
    "LIVE_MIN_EDGE": ParamBounds(
        hard_min=0.15, hard_max=0.60, autonomous_min=0.20, autonomous_max=0.40, tier=1,
    ),
    "LIVE_MIN_BTC_MOVE": ParamBounds(
        hard_min=0, hard_max=100, autonomous_min=0, autonomous_max=30, tier=1,
    ),
    "LIVE_MAX_BTC_MOVE": ParamBounds(
        hard_min=50, hard_max=200, autonomous_min=70, autonomous_max=120, tier=1,
    ),
    "LIVE_MAX_VELOCITY": ParamBounds(
        hard_min=60, hard_max=200, autonomous_min=80, autonomous_max=120, tier=1,
    ),
    "LIVE_MAX_DIR_DELTA": ParamBounds(
        hard_min=10, hard_max=50, autonomous_min=15, autonomous_max=35, tier=1,
    ),
    "LIVE_MAX_MARKET_GAP": ParamBounds(
        hard_min=0.05, hard_max=1.0, autonomous_min=0.15, autonomous_max=0.50, tier=1,
    ),
    "POSITION_PCT": ParamBounds(
        hard_min=0.010, hard_max=0.10, autonomous_min=0.025, autonomous_max=0.070, tier=1,
    ),
    "EDGE_SCALING": ParamBounds(
        hard_min=2.0, hard_max=20.0, autonomous_min=4.0, autonomous_max=15.0, tier=1,
    ),
    # Tier 2
    "SIGMA_SCALE": ParamBounds(
        hard_min=0.35, hard_max=0.80, autonomous_min=0.45, autonomous_max=0.65, tier=2,
    ),
    "ENTRY_THRESHOLD": ParamBounds(
        hard_min=0.04, hard_max=0.15, autonomous_min=0.06, autonomous_max=0.12, tier=2,
    ),
    "MIN_ENTRY_MIN": ParamBounds(
        hard_min=3, hard_max=8, autonomous_min=4, autonomous_max=7, tier=2,
    ),
    "MAX_ENTRY_MIN": ParamBounds(
        hard_min=8, hard_max=13, autonomous_min=10, autonomous_max=12, tier=2,
    ),
}


class AdaptiveConfig:
    """Runtime parameter overrides with safety bounds.

    Usage::

        config = AdaptiveConfig()

        # Strategy reads:
        velocity_cap = config.get("LIVE_MAX_VELOCITY", default=60)

        # Watchdog writes:
        success = config.set("LIVE_MAX_VELOCITY", 55,
                            reason="High velocity trades losing",
                            trade_count=150, wr=0.65)

        # Auto-revert:
        config.revert("LIVE_MAX_VELOCITY", reason="WR dropped 6pp after change")
    """

    def __init__(
        self,
        quality_gate_module: Any = None,
        bounds: dict[str, ParamBounds] | None = None,
    ) -> None:
        """Create a per-strategy adaptive config.

        Args:
            quality_gate_module: Module to read default param values from.
                Defaults to b3_quality_gate (5-min). For B3_15M, pass
                b3_15m_quality_gate so _get_default picks the 15-min defaults.
            bounds: Per-strategy PARAM_BOUNDS override. Defaults to the
                module-level PARAM_BOUNDS (5-min). For B3_15M, pass
                PARAM_BOUNDS_B3_15M.
        """
        from collections import deque
        self._overrides: dict[str, float] = {}
        # Bounded: Watchdog logs every param change + revert pair. Over weeks
        # this reaches thousands of entries. 2000 keeps ~4 weeks of active
        # tuning history (DB audit log has authoritative long-term record).
        self._change_log: deque[ConfigChange] = deque(maxlen=2000)
        self._next_change_id: int = 1
        self._paused: bool = False  # Watchdog can pause live trading
        self._quality_gate_module = quality_gate_module
        self._bounds = bounds if bounds is not None else PARAM_BOUNDS

    def get(self, param: str, default: float) -> float:
        """Read parameter value. Returns override if set, otherwise default.

        This is the primary interface for strategy_b3.py — zero latency,
        just a dict lookup.
        """
        return self._overrides.get(param, default)

    def get_bool(self, param: str, default: bool) -> bool:
        """Read boolean parameter (for TA filter toggles)."""
        val = self._overrides.get(param)
        if val is None:
            return default
        return bool(val)

    @property
    def is_paused(self) -> bool:
        """Whether Watchdog has paused live trading."""
        return self._paused

    def set_pause(self, paused: bool, reason: str) -> None:
        """Pause or resume live trading."""
        old = self._paused
        self._paused = paused
        if old != paused:
            action = "PAUSED" if paused else "RESUMED"
            logger.info("adaptive_config_pause", action=action, reason=reason)
            self._log_change(
                param_name="_LIVE_PAUSED",
                old_value=1.0 if old else 0.0,
                new_value=1.0 if paused else 0.0,
                tier=1,
                reason=reason,
            )

    def set(
        self,
        param: str,
        value: float,
        reason: str,
        trigger_id: str | None = None,
        trade_count: int = 0,
        wr: float = 0.0,
    ) -> bool:
        """Set a parameter override. Returns True if applied, False if rejected.

        Validates against tier bounds. Tier 1 changes are applied silently.
        Tier 2 changes are applied but flagged. Tier 3 rejected outright.
        """
        bounds = self._bounds.get(param)
        if bounds is None:
            logger.warning(
                "adaptive_config_unknown_param",
                param=param,
                msg="Parameter not in PARAM_BOUNDS — rejected (may be Tier 3)",
            )
            return False

        # Hard bounds check
        if value < bounds.hard_min or value > bounds.hard_max:
            logger.warning(
                "adaptive_config_out_of_bounds",
                param=param,
                value=value,
                hard_min=bounds.hard_min,
                hard_max=bounds.hard_max,
            )
            return False

        old_value = self._overrides.get(param, self._get_default(param))

        # Skip if no change (same as default or existing override)
        if abs(old_value - value) < 1e-9:
            logger.info(
                "adaptive_config_noop",
                param=param,
                value=value,
                msg="No change — value equals current. Not logging as change.",
            )
            return False  # Signal to Watchdog: no actual change applied

        # Apply
        self._overrides[param] = value

        # Log
        change = self._log_change(
            param_name=param,
            old_value=old_value,
            new_value=value,
            tier=bounds.tier,
            reason=reason,
            trigger_id=trigger_id,
            trade_count=trade_count,
            wr=wr,
        )

        # Tier 2: flag for review
        needs_flag = bounds.tier == 2
        if needs_flag:
            logger.info(
                "adaptive_config_tier2_flag",
                param=param,
                old=old_value,
                new=value,
                reason=reason,
                msg="Tier 2 change — flagged for CEO review",
            )

        logger.info(
            "adaptive_config_set",
            param=param,
            old=old_value,
            new=value,
            tier=bounds.tier,
            reason=reason,
            change_id=change.change_id,
        )
        return True

    def revert(self, param: str, reason: str) -> bool:
        """Revert a parameter to its default value."""
        if param not in self._overrides:
            return False

        old_value = self._overrides[param]
        del self._overrides[param]

        # Mark the original change as REVERTED
        for change in reversed(self._change_log):
            if change.param_name == param and change.status == "ACTIVE":
                change.status = "REVERTED"
                change.evaluation_result = "REVERTED"
                break

        logger.info(
            "adaptive_config_reverted",
            param=param,
            old=old_value,
            reason=reason,
        )
        return True

    def promote(self, param: str) -> None:
        """Mark a change as promoted (validated, becomes new baseline)."""
        for change in reversed(self._change_log):
            if change.param_name == param and change.status == "ACTIVE":
                change.status = "PROMOTED"
                change.evaluation_result = "KEPT"
                logger.info(
                    "adaptive_config_promoted",
                    param=param,
                    value=change.new_value,
                )
                break

    def _get_default(self, param: str) -> float:
        """Get default value from quality_gate or built-in defaults.

        Some live params (LIVE_MAX_VELOCITY etc.) are defined inline in
        strategy_b3.py, not in quality_gate. We keep a fallback map here.
        """
        # First try the configured quality_gate module (may be 5-min or 15-min)
        if self._quality_gate_module is not None:
            val = getattr(self._quality_gate_module, param, None)
            if val is not None:
                return float(val)
        else:
            from arbo.strategies import b3_quality_gate as qg

            val = getattr(qg, param, None)
            if val is not None:
                return float(val)

        # Fallback defaults for live params (defined inline in strategy_b3.py)
        # Updated 2026-04-12: Data-driven relaxation based on 256 live resolved trades.
        #
        # Edge sensitivity analysis (256 resolved live trades):
        #   edge 0.25-0.30: 14t, 57% WR, avg +$3.54/trade (BEST per-trade)
        #   edge 0.30-0.35: 14t, 64% WR, avg +$4.25/trade (BEST per-trade)
        #   edge 0.35-0.40: 17t,  53% WR, avg +$2.00/trade
        #   edge 0.40+ (old): 148t, 73% WR, avg +$0.11/trade (payout asymmetry)
        # Decision: LIVE_MIN_EDGE 0.40 → 0.30 (captures +$4/trade bucket)
        #
        # BTC move sensitivity analysis:
        #   move 30-40: 16t, 75% WR, avg +$2.48/trade (EXCELLENT)
        #   move 40-50: 26t, 62% WR, avg -$2.18/trade (dead zone)
        #   move 50-70 (old): 70t, 74% WR, avg +$1.98/trade
        # Decision: LIVE_MIN_BTC_MOVE 50 → 35 (captures 30-40 winner bucket)
        #
        # Velocity/dir_delta: insufficient V6.0 data (19 trades) — keep defaults.
        #
        # Previous V6.0 qualified 1/114 paper signals (<1%) in 24h window.
        # New thresholds expected: ~15-25% qualification → 15-25 live trades/day.
        defaults: dict[str, float] = {
            "LIVE_MAX_VELOCITY": 60.0,
            "LIVE_MAX_DIR_DELTA": 15.0,
            "LIVE_MIN_EDGE": 0.30,        # Was 0.40 — data-driven relaxation
            "LIVE_MIN_BTC_MOVE": 35.0,    # Was 50.0 — data-driven relaxation
            # B3_15M additional params (defaults for safety; quality_gate overrides)
            "LIVE_MAX_BTC_MOVE": 80.0,
            "LIVE_MAX_MARKET_GAP": 0.30,
        }
        return defaults.get(param, 0.0)

    def _log_change(
        self,
        param_name: str,
        old_value: float,
        new_value: float,
        tier: int,
        reason: str,
        trigger_id: str | None = None,
        trade_count: int = 0,
        wr: float = 0.0,
    ) -> ConfigChange:
        """Record a change in the audit log."""
        change = ConfigChange(
            change_id=self._next_change_id,
            timestamp=time.time(),
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            tier=tier,
            reason=reason,
            trigger_id=trigger_id,
            trade_count_at_change=trade_count,
            wr_at_change=wr,
        )
        self._change_log.append(change)
        self._next_change_id += 1
        return change

    # ── Query Methods ────────────────────────────────────────────────

    def get_active_changes(self) -> list[ConfigChange]:
        """Get all currently active (non-reverted, non-promoted) changes."""
        return [c for c in self._change_log if c.status == "ACTIVE"]

    def get_active_change_for(self, param: str) -> ConfigChange | None:
        """Get the active change for a specific parameter (if any)."""
        for change in reversed(self._change_log):
            if change.param_name == param and change.status == "ACTIVE":
                return change
        return None

    def has_active_tier1_change(self) -> bool:
        """Check if there's an active Tier 1 change being evaluated.

        Used for cascade prevention: max 1 active Tier 1 change at a time.
        """
        return any(
            c.status == "ACTIVE" and c.tier == 1
            for c in self._change_log
        )

    def consecutive_reverts(self) -> int:
        """Count consecutive reverted changes (for hard reset trigger)."""
        count = 0
        for change in reversed(self._change_log):
            if change.status == "REVERTED":
                count += 1
            else:
                break
        return count

    def get_all_overrides(self) -> dict[str, float]:
        """Get current override values (for status display)."""
        return dict(self._overrides)

    def get_change_log(self, limit: int = 20) -> list[ConfigChange]:
        """Get recent change history (for Watchdog context / Slack reports)."""
        # deque has no slicing — materialize the tail via list()
        recent = list(self._change_log)[-limit:]
        return list(reversed(recent))

    def get_status(self) -> dict[str, Any]:
        """Status for health monitoring / Slack reports."""
        return {
            "overrides": dict(self._overrides),
            "active_changes": len(self.get_active_changes()),
            "total_changes": len(self._change_log),
            "consecutive_reverts": self.consecutive_reverts(),
            "paused": self._paused,
            "has_active_tier1": self.has_active_tier1_change(),
        }
