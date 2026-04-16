"""Quality Gate for Strategy B3 — Binance Oracle Scalper.

Filters BTC 5-minute Up/Down scalping signals using autoresearch-optimized
parameters from 2,400 Chainlink-calibrated trials (89 days, 17,883 windows).

All thresholds from PESSIMISTIC autoresearch (3 scenario stress test) (best OOS, 160 live fills).
"""

from __future__ import annotations

from arbo.utils.logger import get_logger

logger = get_logger("b3_quality_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS — from Chainlink autoresearch sweep (2026-04-01)
# Pessimistic #1: OOS 2,771 trades (pessimistic scenario), WR=64.6%, PnL=$305K (compound), Sharpe=69.8
# Avg fill price: 0.320, breakeven WR=32%, margin=32.6pp
# Key: max_entry_price=0.483 (never overpay)
# Dataset: 89,419 Binance klines + 20,300 Chainlink resolutions (79% coverage)
# Resolution: Polymarket Chainlink oracle truth (not Binance)
# Fee model: maker 0% + 20% rebate (PostOnly), never-sell (always resolve)
# ═══════════════════════════════════════════════════════════════════════════════

# Volatility
SIGMA_WINDOW = 1440               # 24h of 1-min klines for realized vol
SIGMA_METHOD = "realized"         # std(log_returns)
SIGMA_SCALE = 0.348               # Chainlink-calibrated (was 0.400 Binance)
SIGMA_FLOOR = 0.00005             # ~2.6% annualized minimum

# Entry
ENTRY_THRESHOLD = 0.020           # Min |signal_fv - 0.50| to trigger
MIN_ENTRY_MIN = 1                 # Earliest entry: minute 2
MAX_ENTRY_MIN = 3                 # Latest entry: minute 3 (min 4+ = 0W/2L)
CONTRARIAN = False                # MOMENTUM: follow BTC direction

# Exit — never-sell mode: all positions held to Chainlink resolution
# UPDATED 2026-04-12: PAPER_MATCH_LIVE=True makes paper behave exactly like live
# (hold to resolution, no early exits). Paper PnL then = simulated live PnL,
# making it directly comparable and usable for Watchdog analysis.
#
# Previous behavior: paper had early exits (profit/stop/time/edge_gone) which
# produced inflated losses (-$2000 paper vs -$41 live over 7 days). These early
# exits tested a DIFFERENT strategy than what live actually runs.
#
# Old early-exit constants kept for legacy support but ignored when PAPER_MATCH_LIVE=True.
PAPER_MATCH_LIVE = True           # Paper holds to resolution like live
PROFIT_TARGET = 0.207             # Paper-only (ignored if PAPER_MATCH_LIVE)
STOP_LOSS = 99.0                  # Disabled (never-sell)
MAX_HOLD_MIN = 3                  # Paper-only (ignored if PAPER_MATCH_LIVE)
EDGE_EXIT = 0.076                 # Paper-only (ignored if PAPER_MATCH_LIVE)
EXIT_BEFORE_END = 0               # Don't force exit before resolution
ALLOW_RESOLUTION = True           # Always resolve (never-sell)

# BTC-price-based stop (paper model only, ignored if PAPER_MATCH_LIVE=True)
USE_BTC_STOP = True               # Paper-only (ignored if PAPER_MATCH_LIVE)
BTC_STOP_PCT = 0.0015             # Paper-only

# Market model
SPREAD = 0.060                    # $0.005 bid-ask spread (tighter, Chainlink-calibrated)
MAKER = True                      # PostOnly orders (0% fee + rebate)

# Price bounds — paper scanner
MIN_ENTRY_MKT_FV = 0.413          # Don't buy deep OTM (no liquidity)
MAX_ENTRY_MKT_FV = 0.570          # Paper scanner ITM cap (collects data for autoresearch)

# LIVE fill price cap — from 278 live trade analysis (2026-04-10):
#
# The payout is asymmetric: buy at $0.75 → win $0.25/share, lose $0.75/share.
# Higher fill = higher WR but WORSE risk/reward. Data shows:
#
#   fill ≤0.45:  39% WR, breakeven 30%, avg +$2.54/trade  → VERY profitable
#   fill 0.46-0.57: 53% WR, breakeven 52%, avg -$0.26/trade  → breakeven
#   fill 0.58-0.70: 61% WR, breakeven 65%, avg +$0.02/trade  → breakeven
#   fill 0.71-0.85: 74% WR, breakeven 79%, avg -$0.22/trade  → LOSES money
#   fill >0.85:  90% WR, breakeven 91%, avg -$0.03/trade  → LOSES money
#
# Cumulative PnL peaks at cap=0.75 ($175.95, 178 trades) vs uncapped
# ($135.49, 278 trades). Fills >0.75 net lose $40 despite 78% WR.
#
# Why: model WR rises with fill price but NOT fast enough to overcome
# the worsening payout ratio. At fill=0.85, you need 85% WR just to
# break even — the model delivers 75%, a 10pp shortfall.
LIVE_MAX_FILL_PRICE = 0.75        # Max CLOB fill price (data-optimal)

# Sizing — smaller positions, more aggressive edge scaling
POSITION_PCT = 0.026              # 2.9% of capital per trade (was 6.7%)
EDGE_SCALING = 10.000              # Size multiplier per unit of edge (was 4.838)
MAX_BET_SIZE = 100.0              # Max $100 per trade (liquidity constraint)
MAX_SHARES = 500                  # Max shares per trade
MIN_ORDER_SIZE = 3.0              # Min trade size (was $5, but Polymarket's
                                   # actual CLOB min is $1). Lowered 2026-04-14
                                   # after 9h of B3 live qualifications yielded
                                   # 0 filled: wallet ~$190 × POSITION_PCT 0.026
                                   # = ~$4.9/order, blocked by old $5 floor.

# Window
WINDOW_MIN = 5                    # 5-minute windows

# Re-entry
REENTRY_COOLDOWN = 2              # No cooldown (was 3) — more trades per event

# Mirror-cancel debounce: after live fails to fill in mirror mode, block re-entry
# on the same token_id for this many seconds. Fixes observed cascade where same
# signal triggered 4 trades on same token in 16s (IDs 4905-4908, Apr 15 18:16).
# 120s covers ~24 poll cycles at 5s cadence — enough for stale signal to age out
# or for CLOB orderbook to refresh materially.
MIRROR_CANCEL_DEBOUNCE = 120

# Strategy
STRATEGY_NAME = "B3"
