"""Quality Gate for Strategy B3 — Binance Oracle Scalper.

Filters BTC 5-minute Up/Down scalping signals using autoresearch-optimized
parameters from 1,900 trials (89 days, 17,883 windows).

All thresholds from walk-forward Config #1 (best OOS score=6,140).
"""

from __future__ import annotations

from arbo.utils.logger import get_logger

logger = get_logger("b3_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from autoresearch sweep (2026-03-27)
# Config #1: OOS score=6,140, 3,329 trades, WR=52.1%, PnL=$20,285, Sharpe=22.0
# Dataset: 89,419 1-min Binance klines (89 days), walk-forward 70/30 split
# Fee model: maker 0% + 20% rebate (PostOnly)
# ═══════════════════════════════════════════════════════════════════════════════

# Volatility
SIGMA_WINDOW = 720                # 12h of 1-min klines for realized vol
SIGMA_METHOD = "realized"         # std(log_returns) — all 5 WF winners
SIGMA_SCALE = 0.644               # Signal amplification (< 1.0 = more aggressive)
SIGMA_FLOOR = 0.00005             # ~2.6% annualized minimum

# Entry
ENTRY_THRESHOLD = 0.095           # Min |signal_fv - 0.50| to trigger
MIN_ENTRY_MIN = 2                 # Earliest entry: minute 2
MAX_ENTRY_MIN = 2                 # Latest entry: minute 2 (ONLY minute 2!)
CONTRARIAN = False                # MOMENTUM: follow BTC direction

# Exit
PROFIT_TARGET = 0.207             # Take profit $0.207/share
STOP_LOSS = 0.038                 # FV-based stop (DISABLED when USE_BTC_STOP=True)
MAX_HOLD_MIN = 3                  # Max 3 minutes holding
EDGE_EXIT = 0.076                 # Exit if edge drops below 7.6%
EXIT_BEFORE_END = 0               # Don't force exit before resolution
ALLOW_RESOLUTION = True           # Can hold through market resolution

# BTC-price-based stop (replaces FV stop — linear, no CDF overshoot)
# Validated in b3_improvement_study: OOS score +47%, WR 52→66%, DD 33→23%
USE_BTC_STOP = True               # Use BTC % reversal stop instead of FV stop
BTC_STOP_PCT = 0.0015             # 0.15% BTC reversal = ~$100 at $66,500

# Market model
SPREAD = 0.01                     # $0.01 bid-ask spread
MAKER = True                      # PostOnly orders (0% fee + rebate)

# Price bounds
MIN_ENTRY_MKT_FV = 0.15          # Don't buy deep OTM (no liquidity)
MAX_ENTRY_MKT_FV = 0.85          # Don't buy deep ITM (too expensive)

# Sizing
POSITION_PCT = 0.067              # 6.7% of capital per trade
EDGE_SCALING = 4.838              # Size multiplier per unit of edge
MAX_BET_SIZE = 100.0              # Max $100 per trade (liquidity constraint)
MAX_SHARES = 500                  # Max shares per trade
MIN_ORDER_SIZE = 5.0              # Polymarket minimum $5

# Window
WINDOW_MIN = 5                    # 5-minute windows (all WF winners)

# Re-entry
REENTRY_COOLDOWN = 3              # 3 minutes before re-entry in same event

# Strategy
STRATEGY_NAME = "B3"
