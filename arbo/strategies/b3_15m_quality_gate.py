"""Quality Gate for Strategy B3_15M — Binance Oracle Scalper (15-min variant).

Filters BTC 15-minute Up/Down scalping signals using parameters from
autoresearch grid sweep on 144 real shadow signals (2026-04-03 to 2026-04-12),
5-fold time-ordered cross validation.

Rank #1 robust config (all 5 folds positive, 94.4% avg WR, σ=0.103):
  - min_edge=0.30, max_btc_move=80, max_market_gap=0.30
  - entry minutes 4-11, no fill cap (OPPOSITE of 5-min)
  - N=49, total PnL=$9.89, avg $0.222/share

Key 15-min differences vs 5-min:
  - SIGMA_SCALE higher (15-min accumulates 3x more vol: σ√15 vs σ√5)
  - ENTRY_THRESHOLD much higher (0.089 vs 0.020) — larger window = more noise
  - LIVE_MAX_BTC_MOVE=80 (5-min has NO cap) — above $80, 15-min reverses
  - LIVE_MAX_FILL_PRICE=1.01 (OPPOSITE of 5-min 0.75) — high fill = best bucket (WR 86%)
  - MIN_ENTRY_MIN=4, MAX_ENTRY_MIN=11 (5-min: 1-3)
"""

from __future__ import annotations

from arbo.utils.logger import get_logger

logger = get_logger("b3_15m_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from shadow autoresearch (2026-04-12, 144 real resolutions)
# ═══════════════════════════════════════════════════════════════════════════════

# Volatility — 15-min window scaling
SIGMA_WINDOW = 1440               # 24h of 1-min klines (same as 5-min)
SIGMA_METHOD = "realized"         # std(log_returns)
SIGMA_SCALE = 0.526               # From sweep (15-min accumulation factor)
SIGMA_FLOOR = 0.00005             # ~2.6% annualized minimum

# Entry
ENTRY_THRESHOLD = 0.089           # Min |signal_fv - 0.50| (10x stricter than 5-min)
MIN_ENTRY_MIN = 4                 # Earliest entry: minute 4 (shadow data optimal)
MAX_ENTRY_MIN = 11                # Latest entry: minute 11
CONTRARIAN = False                # MOMENTUM: follow BTC direction

# Exit — never-sell mode: all positions held to Chainlink resolution
PAPER_MATCH_LIVE = True           # Paper holds to resolution like live
PROFIT_TARGET = 0.207             # Paper-only (ignored if PAPER_MATCH_LIVE)
STOP_LOSS = 99.0                  # Disabled (never-sell)
MAX_HOLD_MIN = 12                 # Paper-only (ignored if PAPER_MATCH_LIVE)
EDGE_EXIT = 0.076                 # Paper-only (ignored if PAPER_MATCH_LIVE)
EXIT_BEFORE_END = 0               # Don't force exit before resolution
ALLOW_RESOLUTION = True           # Always resolve (never-sell)

# BTC-price-based stop (paper model only, ignored if PAPER_MATCH_LIVE=True)
USE_BTC_STOP = True
BTC_STOP_PCT = 0.0025             # Wider stop for 15-min window

# Market model
SPREAD = 0.010                    # $0.010 bid-ask (slightly wider than 5-min on 15-min markets)
MAKER = True                      # PostOnly orders (0% fee + rebate)

# Price bounds — paper scanner
MIN_ENTRY_MKT_FV = 0.30           # Wider range for 15-min (more signal variance)
MAX_ENTRY_MKT_FV = 0.95

# Live filters — RANK #1 ROBUST CONFIG from shadow autoresearch
# All 5 time-folds positive, WR 94.4%, min fold N=4, stability σ=0.103
LIVE_MIN_EDGE = 0.30              # Strict edge gate (filters toxic 0.10-0.20 bucket)
LIVE_MIN_BTC_MOVE = 0.0           # NO lower cap — edge>=0.30 replaces move filter
LIVE_MAX_BTC_MOVE = 80.0          # NEW for 15-min: reversal risk above $80
LIVE_MAX_MARKET_GAP = 0.30        # Gap between model fv and market fv
LIVE_MAX_FILL_PRICE = 1.01        # NO fill cap (15-min high fill = best bucket)
LIVE_MAX_VELOCITY = 100.0         # $/min (wider than 5-min: longer window = higher velocity tolerated)
LIVE_MAX_DIR_DELTA = 25.0         # $ (Chainlink vs Binance divergence)

# Sizing — conservative for 15-min (higher per-trade PnL, fewer trades)
POSITION_PCT = 0.044              # 4.4% of capital per trade (sweep-optimal)
EDGE_SCALING = 6.757              # Size multiplier per unit of edge (sweep-optimal)
MAX_BET_SIZE = 20.0               # Max $20 per trade ($75 capital × 27% max)
MAX_SHARES = 500                  # Max shares per trade
MIN_ORDER_SIZE = 5.0              # Polymarket minimum $5

# Window
WINDOW_MIN = 15                   # 15-minute windows (vs 5 on B3)

# Re-entry
REENTRY_COOLDOWN = 3              # Slightly higher than 5-min (longer window absorbs reentries)

# Strategy
STRATEGY_NAME = "B3_15M"
