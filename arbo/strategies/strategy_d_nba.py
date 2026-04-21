"""Strategy D — NBA Variant.

NBA moneyline green booking. Strategy ID = "D" (legacy alias, original).

Parameters: winner #25 from sweep v4 (both-sides + always-close), 2026-04-05.
  - 344 experiments across 4 sweeps
  - Score 1,234, P&L +$1,665 on $1K over 20 months
  - Win Rate 50%, Green Book Rate 23%
  - Sharpe 7.03, Max DD 13%
  - ROI ~100%/year

Architecture: docs/STRATEGY_D_ARCHITECTURE.md
Backtest spec: docs/STRATEGY_D_SPEC.md

Poll cycle: 60s scan, 30s exit check.
"""

from __future__ import annotations

from arbo.strategies.strategy_d_core import StrategyDCore


class StrategyDNba(StrategyDCore):
    """NBA green book engine."""

    SPORT_NAME = "nba"
    STRATEGY_NAME = "D"
    STRATEGY_LABEL = "NBA Green Book"

    # Sweep v4 winner #25 params
    MIN_EDGE = 0.16
    MAX_EDGE = 0.25
    MIN_PRICE = 0.20
    MAX_PRICE = 0.65

    GREEN_BOOK_DELTA = 0.17
    STOP_LOSS_DELTA = 0.15
    MAX_HOLD_FRACTION = 0.50
    GAME_DURATION_HOURS = 2.5  # Avg NBA game

    BOTH_SIDES = True
    MAX_CONCURRENT = 8

    KELLY_FRACTION = 0.15
    KELLY_RAW_CAP = 0.10
    MAX_POSITION_PCT = 0.03

    ELO_WEIGHT = 0.40
    PINNACLE_WEIGHT = 0.60

    RISK_LAYER = 9

    # Shadow-exit telemetry — evaluates strategy_d_exit_v1 (NBA-trained)
    # on every NBA position without affecting exit behavior. Logs paired
    # "what would ML do" rows to shadow_exit_decisions table for later
    # P(better) analysis. Safe: never changes trading.
    # Disable anytime by setting to False (e.g. if model load errors
    # become noisy in logs).
    SHADOW_EXIT_LOG_ENABLED = True
    SHADOW_EXIT_MODEL_PATH = "arbo/data/models/strategy_d_exit_v1.ubj"
    SHADOW_EXIT_THRESHOLD = 6658.3


# Backward-compat alias (used by main_rdh.py)
StrategyD = StrategyDNba
