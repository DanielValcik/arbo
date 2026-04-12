"""Strategy D — EPL Variant.

EPL (English Premier League + cup competitions) moneyline + draw green booking.
Strategy ID = "D_EPL".

Sweep winner #12 (288 experiments, 100% profitable!):
  - Score: 6.2, +$158 on $1K over 20 months
  - 91 trades, WR 64%, DD 3%, Sharpe 6.76
  - GB rate 64% — high green book triggering

Key characteristics:
  - 3-way outcomes: home_win / draw / away_win
  - Polymarket splits into binary markets per outcome
  - Question formats: "Will X beat Y?", "Will X win on DATE?",
    "Will X vs Y end in a draw?"
  - 1,940 Pinnacle odds from football-data.co.uk (free)
  - Includes EPL + FA Cup + League Cup + UCL + Community Shield
  - Pinnacle coverage limits us to ~280 unique EPL fixtures

Architecture: docs/STRATEGY_D_ARCHITECTURE.md
"""

from __future__ import annotations

from arbo.strategies.strategy_d_core import StrategyDCore


class StrategyDEpl(StrategyDCore):
    """EPL green book engine — sweep winner #12."""

    SPORT_NAME = "epl"
    STRATEGY_NAME = "D_EPL"
    STRATEGY_LABEL = "EPL Green Book"

    # Sweep winner #12 params
    MIN_EDGE = 0.03          # Lower edge — EPL has small but consistent edges
    MAX_EDGE = 0.30
    MIN_PRICE = 0.15
    MAX_PRICE = 0.70

    GREEN_BOOK_DELTA = 0.15  # Smaller than UFC (0.20) — EPL has gradual moves
    STOP_LOSS_DELTA = 0.25   # Wide SL (robust across 288 experiments)
    MAX_HOLD_FRACTION = 1.0  # Hold to resolution — EPL Pinnacle very accurate
    GAME_DURATION_HOURS = 2.0  # 90min match + stoppage + pre-game window

    BOTH_SIDES = True
    MAX_CONCURRENT = 6        # Multiple matches per gameweek
    COOLDOWN_AFTER_TRADE_S = 60

    # Conservative sizing (small P&L in backtest)
    KELLY_FRACTION = 0.12
    KELLY_RAW_CAP = 0.10
    MAX_POSITION_PCT = 0.03

    # Pinnacle-weighted (EPL Pinnacle very reliable)
    ELO_WEIGHT = 0.20
    PINNACLE_WEIGHT = 0.80

    RISK_LAYER = 9
