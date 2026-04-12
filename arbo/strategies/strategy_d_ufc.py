"""Strategy D — UFC Variant.

UFC moneyline green booking. Strategy ID = "D_UFC".

Status: BASELINE — uses NBA-sweep-inspired params as starting point.
       UFC-specific sweep pending. Params will be re-tuned after live data.

UFC characteristics:
  - ~40 events/year, year-round
  - 3-5 rounds × 5 min = ~25-45 min total fight, 1-2h event window
  - Binary outcome (winner), highly volatile round-to-round
  - Per spec: delta range ±20-40¢ per round, theta=0.2 (slow reversion)
  - Best fit for D2 overreaction fade (future) + D1 green book (current)

Pinnacle coverage: 7,817 UFC odds already in DB (best of any sport).

Baseline params (conservative — NBA-inspired, UFC-adjusted):
  - Larger GREEN_BOOK_DELTA (0.20) because per-round swings are ±20-40¢
  - Shorter GAME_DURATION_HOURS (1.5) — average fight is 15-25 min
  - Keep MAX_HOLD_FRACTION 0.50 (exit mid-fight, typically round 2-3)

Architecture: docs/STRATEGY_D_ARCHITECTURE.md
"""

from __future__ import annotations

from arbo.strategies.strategy_d_core import StrategyDCore


class StrategyDUfc(StrategyDCore):
    """UFC green book engine — baseline params pending UFC sweep."""

    SPORT_NAME = "ufc"
    STRATEGY_NAME = "D_UFC"
    STRATEGY_LABEL = "UFC Green Book"

    # Baseline params (conservative, to be tuned post-live)
    MIN_EDGE = 0.16          # Same as NBA — will sweep later
    MAX_EDGE = 0.30          # Slightly wider (UFC has bigger market errors)
    MIN_PRICE = 0.20
    MAX_PRICE = 0.70         # Slightly higher (UFC has clearer favorites)

    GREEN_BOOK_DELTA = 0.20  # Higher than NBA (0.17) — UFC round swings are bigger
    STOP_LOSS_DELTA = 0.20   # Wider SL for UFC volatility
    MAX_HOLD_FRACTION = 0.50 # Exit at ~half fight time
    GAME_DURATION_HOURS = 1.5  # Avg UFC fight window (prelim + main card)

    BOTH_SIDES = True
    MAX_CONCURRENT = 4        # Fewer concurrent UFC fights available
    COOLDOWN_AFTER_TRADE_S = 60

    # Conservative sizing — untested params
    KELLY_FRACTION = 0.10     # Half of NBA (0.15) until we have live data
    KELLY_RAW_CAP = 0.08
    MAX_POSITION_PCT = 0.02   # 2% vs NBA's 3%

    ELO_WEIGHT = 0.30         # Lower than NBA — fighter Elo less reliable
    PINNACLE_WEIGHT = 0.70    # Higher — Pinnacle is the sharp benchmark for UFC

    RISK_LAYER = 9
