"""Quality Gate for Strategy B3 — Binance Oracle Scalper.

Filters BTC 5-minute Up/Down scalping signals using autoresearch-optimized
parameters from 2,400 Chainlink-calibrated trials (89 days, 17,883 windows).

All thresholds from REALISTIC autoresearch Config #3 (best OOS, 160 live fills).
"""

from __future__ import annotations

from arbo.utils.logger import get_logger

logger = get_logger("b3_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from Chainlink autoresearch sweep (2026-04-01)
# Realistic #3: OOS 2,874 trades, WR=64.6%, PnL=$305K (compound), Sharpe=69.8
# Avg fill price: 0.320, breakeven WR=32%, margin=32.6pp
# Key: max_entry_price=0.483 (never overpay)
# Dataset: 89,419 Binance klines + 20,300 Chainlink resolutions (79% coverage)
# Resolution: Polymarket Chainlink oracle truth (not Binance)
# Fee model: maker 0% + 20% rebate (PostOnly), never-sell (always resolve)
# ═══════════════════════════════════════════════════════════════════════════════

# Volatility
SIGMA_WINDOW = 1440               # 24h of 1-min klines for realized vol
SIGMA_METHOD = "realized"         # std(log_returns)
SIGMA_SCALE = 0.150               # Chainlink-calibrated (was 0.400 Binance)
SIGMA_FLOOR = 0.00005             # ~2.6% annualized minimum

# Entry
ENTRY_THRESHOLD = 0.030           # Min |signal_fv - 0.50| to trigger
MIN_ENTRY_MIN = 1                 # Earliest entry: minute 2
MAX_ENTRY_MIN = 4                 # Latest entry: minute 3 (was 2)
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
SPREAD = 0.060                    # $0.005 bid-ask spread (tighter, Chainlink-calibrated)
MAKER = True                      # PostOnly orders (0% fee + rebate)

# Price bounds
MIN_ENTRY_MKT_FV = 0.274          # Don't buy deep OTM (no liquidity)
MAX_ENTRY_MKT_FV = 0.483          # Don't buy deep ITM (too expensive)

# Sizing — smaller positions, more aggressive edge scaling
POSITION_PCT = 0.042              # 2.9% of capital per trade (was 6.7%)
EDGE_SCALING = 6.508              # Size multiplier per unit of edge (was 4.838)
MAX_BET_SIZE = 100.0              # Max $100 per trade (liquidity constraint)
MAX_SHARES = 500                  # Max shares per trade
MIN_ORDER_SIZE = 5.0              # Polymarket minimum $5

# Window
WINDOW_MIN = 5                    # 5-minute windows

# Re-entry
REENTRY_COOLDOWN = 3              # No cooldown (was 3) — more trades per event

# Strategy
STRATEGY_NAME = "B3"
