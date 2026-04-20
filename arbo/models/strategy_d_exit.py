"""Strategy D — ML-driven exit policy (production-clean).

Mirror of `research_d/exit_timing_features.py` feature math, plus a
lazy-loaded XGBoost model wrapper for live/paper exit decisions.

Keeps arbo/ independent of research_d/ — research code iterates freely
while production has a frozen copy of the feature logic that shipped
with model v1.

Related:
  docs/STRATEGY_D_ML_DESIGN.md
  research_d/exit_timing_features.py  (source of truth for feature math)
  research_d/train_exit_model.py      (trainer — model .ubj comes from here)
  research_d/eval_exit_policy.py      (A/B harness)

Only imports from stdlib + xgboost. No pandas/numpy required in the hot
path — features come out as plain dicts and we pack them once per call.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Feature math (identical semantics to research_d) ──────────────────


def _rolling_std(prices: list[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    mean = sum(prices) / n
    return math.sqrt(sum((x - mean) ** 2 for x in prices) / (n - 1))


def _rolling_slope(prices: list[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(prices) / n
    num = sum((xs[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 1e-12 else 0.0


def _pct_up(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.5
    ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
    return ups / (len(prices) - 1)


def _autocorr_1(returns: list[float]) -> float:
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    num = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    den = sum((r - mean) ** 2 for r in returns)
    return num / den if den > 1e-12 else 0.0


def _safe_div(num: float, den: float, cap: float = 100.0) -> float:
    if abs(den) < 1e-6:
        return math.copysign(cap, num) if num != 0 else 0.0
    val = num / den
    return max(-cap, min(cap, val))


def compute_exit_features(
    trajectory: list[tuple[float, float]],
    t: int,
    entry_price: float,
    target: float,
    stop_loss_price: float,
    side: str,
    model_prob: float,
    edge_at_entry: float,
    total_len_expected: int | None = None,
) -> dict[str, float]:
    """Compute exit-timing features at tick t. NO lookahead: uses only
    trajectory[0..t].
    """
    if t < 0 or t >= len(trajectory):
        raise ValueError(f"t={t} out of range [0, {len(trajectory)})")

    prices_so_far = [p for (_, p) in trajectory[: t + 1]]
    price_t = prices_so_far[-1]

    # Price-state
    price_return_since_entry = price_t - entry_price
    price_return_pct = (
        price_return_since_entry / entry_price if entry_price > 1e-9 else 0.0
    )
    max_so_far = max(prices_so_far)
    min_so_far = min(prices_so_far)
    max_minus_now = max_so_far - price_t
    drawdown_from_entry = entry_price - min_so_far

    if side == "yes":
        unrealized_edge = price_t - entry_price
        peak_profit = max_so_far - entry_price
    else:
        unrealized_edge = entry_price - price_t
        peak_profit = entry_price - min_so_far

    # Volatility
    vol_5 = _rolling_std(prices_so_far[-5:])
    vol_15 = _rolling_std(prices_so_far[-15:])
    vol_60 = _rolling_std(prices_so_far[-60:])
    vol_ratio_5_60 = vol_5 / vol_60 if vol_60 > 1e-9 else 1.0
    realized_range_15 = (
        max(prices_so_far[-15:]) - min(prices_so_far[-15:])
        if len(prices_so_far) >= 2 else 0.0
    )

    # Momentum
    slope_5 = _rolling_slope(prices_so_far[-5:])
    slope_15 = _rolling_slope(prices_so_far[-15:])
    pct_up_10 = _pct_up(prices_so_far[-10:])
    ticks = prices_so_far[-6:]
    rets = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    ret_autocorr = _autocorr_1(rets)

    # Time
    total_len = total_len_expected or max(len(trajectory), 1)
    elapsed_frac = t / max(total_len - 1, 1)
    remaining_frac = 1.0 - elapsed_frac

    # Distance to barriers
    if side == "yes":
        gb_distance = target - price_t
        gb_already_touched = 1 if max_so_far >= target else 0
        sl_distance = price_t - stop_loss_price if stop_loss_price > 0 else 10.0
    else:
        gb_distance = price_t - target
        gb_already_touched = 1 if min_so_far <= target else 0
        sl_distance = stop_loss_price - price_t if stop_loss_price < 2.0 else 10.0

    gb_distance_norm = _safe_div(gb_distance, vol_15)
    sl_distance_norm = _safe_div(sl_distance, vol_15)

    return {
        "price_return_since_entry": price_return_since_entry,
        "price_return_pct": price_return_pct,
        "max_since_entry": max_so_far,
        "min_since_entry": min_so_far,
        "max_minus_now": max_minus_now,
        "drawdown_from_entry": drawdown_from_entry,
        "unrealized_edge": unrealized_edge,
        "peak_profit": peak_profit,
        "vol_5": vol_5,
        "vol_15": vol_15,
        "vol_60": vol_60,
        "vol_ratio_5_60": vol_ratio_5_60,
        "realized_range_15": realized_range_15,
        "slope_5": slope_5,
        "slope_15": slope_15,
        "pct_up_10": pct_up_10,
        "ret_autocorr_5": ret_autocorr,
        "elapsed_frac": elapsed_frac,
        "remaining_frac": remaining_frac,
        "elapsed_ticks": float(t),
        "gb_distance": gb_distance,
        "gb_distance_norm": gb_distance_norm,
        "gb_already_touched": float(gb_already_touched),
        "sl_distance": sl_distance,
        "sl_distance_norm": sl_distance_norm,
        "model_prob_entry": model_prob,
        "edge_at_entry": edge_at_entry,
        "entry_price_level": entry_price,
    }


# Feature column order — must match training. Frozen at v1 ship time.
FEATURE_COLUMNS_V1: list[str] = [
    "price_return_since_entry",
    "price_return_pct",
    "max_since_entry",
    "min_since_entry",
    "max_minus_now",
    "drawdown_from_entry",
    "unrealized_edge",
    "peak_profit",
    "vol_5",
    "vol_15",
    "vol_60",
    "vol_ratio_5_60",
    "realized_range_15",
    "slope_5",
    "slope_15",
    "pct_up_10",
    "ret_autocorr_5",
    "elapsed_frac",
    "remaining_frac",
    "elapsed_ticks",
    "gb_distance",
    "gb_distance_norm",
    "gb_already_touched",
    "sl_distance",
    "sl_distance_norm",
    "model_prob_entry",
    "edge_at_entry",
    "entry_price_level",
]


# ── Model wrapper ────────────────────────────────────────────────────


@dataclass
class ExitPolicyDecision:
    """Result of querying the ML exit policy at a single tick."""
    should_exit: bool
    pred_log_t: float
    unrealized_profit: float
    reason: str  # "ml_hazard_v1", "insufficient_history", "not_profitable", ...


class ExitPolicyModel:
    """Lazy-loaded XGBoost survival model for exit timing.

    Thread-safety: single Booster instance; xgboost's predict is
    thread-safe for inference. We create one per StrategyDCore instance.
    """

    def __init__(
        self,
        model_path: Path | str,
        threshold_log_t: float = 6658.3,
        min_history_ticks: int = 5,
    ):
        self.model_path = Path(model_path)
        self.threshold_log_t = float(threshold_log_t)
        self.min_history_ticks = int(min_history_ticks)
        self._booster: Any | None = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        """Lazy-load the XGBoost booster. Returns True if model usable."""
        if self._booster is not None:
            return True
        if self._load_error is not None:
            # Previously failed; don't retry on every tick
            return False
        try:
            import xgboost as xgb
            if not self.model_path.exists():
                self._load_error = f"model file not found: {self.model_path}"
                log.warning("exit_policy_load_failed", extra={"err": self._load_error})
                return False
            booster = xgb.Booster()
            booster.load_model(str(self.model_path))
            self._booster = booster
            log.info(
                "exit_policy_loaded",
                extra={"path": str(self.model_path), "threshold": self.threshold_log_t},
            )
            return True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            log.warning(
                "exit_policy_load_failed",
                extra={"err": self._load_error, "path": str(self.model_path)},
            )
            return False

    def decide(
        self,
        trajectory: list[tuple[float, float]],
        entry_price: float,
        target: float,
        stop_loss_price: float,
        side: str,
        model_prob: float,
        edge_at_entry: float,
        total_len_expected: int | None = None,
    ) -> ExitPolicyDecision:
        """Query model: should we early-exit at the current tick (last
        entry in trajectory)?

        Returns ExitPolicyDecision with full diagnostic info.
        """
        if not trajectory or len(trajectory) < self.min_history_ticks:
            return ExitPolicyDecision(
                should_exit=False,
                pred_log_t=0.0,
                unrealized_profit=0.0,
                reason="insufficient_history",
            )

        current_price = trajectory[-1][1]
        unrealized = (
            (current_price - entry_price) if side == "yes"
            else (entry_price - current_price)
        )
        if unrealized <= 0:
            return ExitPolicyDecision(
                should_exit=False,
                pred_log_t=0.0,
                unrealized_profit=unrealized,
                reason="not_profitable",
            )

        if not self._ensure_loaded():
            return ExitPolicyDecision(
                should_exit=False,
                pred_log_t=0.0,
                unrealized_profit=unrealized,
                reason=f"model_unavailable:{self._load_error or 'unknown'}",
            )

        features = compute_exit_features(
            trajectory=trajectory,
            t=len(trajectory) - 1,
            entry_price=entry_price,
            target=target,
            stop_loss_price=stop_loss_price,
            side=side,
            model_prob=model_prob,
            edge_at_entry=edge_at_entry,
            total_len_expected=total_len_expected,
        )

        import xgboost as xgb
        x = [[features[f] for f in FEATURE_COLUMNS_V1]]
        dmat = xgb.DMatrix(x, feature_names=list(FEATURE_COLUMNS_V1))
        pred_log_t = float(self._booster.predict(dmat)[0])

        should = pred_log_t > self.threshold_log_t
        return ExitPolicyDecision(
            should_exit=should,
            pred_log_t=pred_log_t,
            unrealized_profit=unrealized,
            reason="ml_hazard_v1" if should else "below_threshold",
        )
