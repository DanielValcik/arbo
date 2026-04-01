"""Quality Gate for Strategy B3 — Binance Oracle Scalper.

Filters BTC 5-minute Up/Down scalping signals using autoresearch-optimized
parameters from 2,400 Chainlink-calibrated trials (89 days, 17,883 windows).

All thresholds from walk-forward Config #5 (best OOS score=7,996).
"""

from __future__ import annotations

from arbo.utils.logger import get_logger

logger = get_logger("b3_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from Chainlink autoresearch sweep (2026-04-01)
# WF #5: OOS score=7,996, 4,423 trades, WR=70.7%, PnL=$28,215, Sharpe=18.0
# Dataset: 89,419 Binance klines + 20,300 Chainlink resolutions (79% coverage)
# Resolution: Polymarket Chainlink oracle truth (not Binance)
# Fee model: maker 0% + 20% rebate (PostOnly), never-sell (always resolve)
# ═══════════════════════════════════════════════════════════════════════════════

# Volatility
SIGMA_WINDOW = 1440               # 24h of 1-min klines for realized vol
SIGMA_METHOD = "realized"         # std(log_returns)
SIGMA_SCALE = 0.408               # Chainlink-calibrated (was 0.400 Binance)
SIGMA_FLOOR = 0.00005             # ~2.6% annualized minimum

# Entry
ENTRY_THRESHOLD = 0.121           # Min |signal_fv - 0.50| to trigger
MIN_ENTRY_MIN = 2                 # Earliest entry: minute 2
MAX_ENTRY_MIN = 3                 # Latest entry: minute 3 (was 2)
CONTRARIAN = False                # MOMENTUM: follow BTC direction

# Exit — never-sell mode: all positions held to Chainlink resolution
# Paper model still uses these for FV tracking, live ignores (holds to resolution)
PROFIT_TARGET = 0.207             # Paper-only: take profit threshold
STOP_LOSS = 99.0                  # Disabled (never-sell)
MAX_HOLD_MIN = 3                  # Paper-only: max hold
EDGE_EXIT = 0.076                 # Paper-only: edge exit
EXIT_BEFORE_END = 0               # Don't force exit before resolution
ALLOW_RESOLUTION = True           # Always resolve (never-sell)

# BTC-price-based stop (paper model only, live holds to resolution)
USE_BTC_STOP = True               # Paper-only
BTC_STOP_PCT = 0.0015             # Paper-only

# Market model
SPREAD = 0.005                    # $0.005 bid-ask spread (tighter, Chainlink-calibrated)
MAKER = True                      # PostOnly orders (0% fee + rebate)

# Price bounds
MIN_ENTRY_MKT_FV = 0.15          # Don't buy deep OTM (no liquidity)
MAX_ENTRY_MKT_FV = 0.85          # Don't buy deep ITM (too expensive)

# Sizing — smaller positions, more aggressive edge scaling
POSITION_PCT = 0.029              # 2.9% of capital per trade (was 6.7%)
EDGE_SCALING = 9.497              # Size multiplier per unit of edge (was 4.838)
MAX_BET_SIZE = 100.0              # Max $100 per trade (liquidity constraint)
MAX_SHARES = 500                  # Max shares per trade
MIN_ORDER_SIZE = 5.0              # Polymarket minimum $5

# Window
WINDOW_MIN = 5                    # 5-minute windows

# Re-entry
REENTRY_COOLDOWN = 0              # No cooldown (was 3) — more trades per event

# Strategy
STRATEGY_NAME = "B3"
