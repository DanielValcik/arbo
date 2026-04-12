"""Strategy D — UFC Variant.

UFC moneyline green booking. Strategy ID = "D_UFC".

Status: SWEEP-OPTIMIZED — params from UFC autoresearch (240 experiments).
Sweep winner #7: score=2.0, +$45 on $1K, 129 trades, DD 1%, Sharpe 13.21.

UFC sweep characteristics (2026-04-12):
  - 76% of 240 experiments profitable
  - Extremely low DD (1-2%) — stable strategy
  - High Sharpe (9-15) — consistent edge per trade
  - Small absolute P&L due to limited parseable markets (129 trades over 20 months)

Key findings from UFC sweep:
  - min_edge=0.05 best (lower than NBA — UFC has more subtle edges)
  - delta=0.20 best (avg $19 vs $9/$6/$1 for 0.15/0.25/0.30)
  - SL=0.25 best (wider SL needed for UFC volatility, avg $13.7)
  - mhf=0.40/0.50/0.60 all similar (avoid 1.0 = hold-to-resolution!)

Pinnacle-only model (UFC has no Elo ratings in DB).

Architecture: docs/STRATEGY_D_ARCHITECTURE.md
"""

from __future__ import annotations

from arbo.strategies.strategy_d_core import StrategyDCore


class StrategyDUfc(StrategyDCore):
    """UFC green book engine — UFC sweep winner #7."""

    SPORT_NAME = "ufc"
    STRATEGY_NAME = "D_UFC"
    STRATEGY_LABEL = "UFC Green Book"

    # Sweep winner #7 params
    MIN_EDGE = 0.05          # Lower than NBA (0.16) — UFC has subtle edges
    MAX_EDGE = 0.30          # Wider max (UFC market is less efficient)
    MIN_PRICE = 0.20
    MAX_PRICE = 0.70

    GREEN_BOOK_DELTA = 0.20  # Optimal from sweep (avg $19 vs $9/$6/$1)
    STOP_LOSS_DELTA = 0.25   # Wider SL needed for UFC volatility
    MAX_HOLD_FRACTION = 0.40 # Exit early — sweep showed 0.4/0.5/0.6 similar
    GAME_DURATION_HOURS = 1.5  # Avg UFC fight window

    BOTH_SIDES = True
    MAX_CONCURRENT = 4
    COOLDOWN_AFTER_TRADE_S = 60

    # Conservative sizing (small P&L in backtest justifies small positions)
    KELLY_FRACTION = 0.10
    KELLY_RAW_CAP = 0.08
    MAX_POSITION_PCT = 0.02

    # UFC has no Elo data — model is Pinnacle-only in effect
    ELO_WEIGHT = 0.00
    PINNACLE_WEIGHT = 1.00

    RISK_LAYER = 9
