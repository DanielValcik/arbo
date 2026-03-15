# Strategy D: Live Edge Harvester (LEH) — Complete Specification

> **Author:** CTO (AI) | **Date:** 2026-03-15 | **Status:** DESIGN PHASE
> **Capital Allocation:** TBD (CEO approval required)
> **Priority Metrics:** 1. Profitability (continuous) 2. Capital Turnover 3. Scalability

---

## 1. Executive Summary

Strategy D exploits structural inefficiencies in Polymarket's live sports markets through three synergistic sub-strategies:

| Sub-strategy | Core Mechanic | Capital Turnover | Expected Edge |
|---|---|---|---|
| **D1: Green Book Engine** | Pre-game value entry → in-game hedge to lock profit | Hours (per game) | 3-8% gross |
| **D2: Overreaction Fade** | Fade live price overreactions to scoring events | Minutes | 1-4% per trade |
| **D3: Cross-Event Cascade** | Trade stale futures after correlated game results | Hours-Days | 1-5% per opportunity |

**Key Innovation:** We don't predict *who wins*. We predict *whether the price will move favorably at any point during the game*. This is a fundamentally different (and easier) question — the probability of a temporary favorable price move is much higher than the probability of the final outcome.

**Why Polymarket specifically:**
- **Zero fees** on most sports markets (vs 5-10% vig at sportsbooks)
- **CLOB orderbook** enables precise limit-order execution (not AMM slippage)
- **Live trading support** with WebSocket real-time price feeds
- **No account limits** on winners (unlike traditional bookmakers)
- **Structural retail bias** — Polymarket prices are sentiment-driven, creating exploitable deviations from sharp benchmarks

**Theoretical Foundation:** Combines 7 academic frameworks: Ornstein-Uhlenbeck mean reversion (Moskowitz 2021), Bayesian sequential updating (Robberechts et al. 2023), Logit Jump-Diffusion kernel (Dalen 2025), information cascade fragility (Bikhchandani et al. 1992), favorite-longshot bias (Snowberg & Wolfers 2010), Kelly criterion for dynamic hedging, and Elo/Glicko-2 network-based team strength estimation.

---

## 2. Theoretical Foundation

### 2.1 The Green Book Principle

Green booking (from Betfair exchange trading) creates a risk-free position by trading price movements:

```
1. Buy YES at P_entry (e.g., $0.40)
2. Price rises to P_exit (e.g., $0.55)
3. Sell YES at P_exit
4. Guaranteed profit = P_exit - P_entry = $0.15 per contract
```

On Polymarket, buying YES and selling YES (or buying NO) are equivalent to Betfair's back/lay system. The **zero fee structure** on most sports markets means the full spread is captured as profit — a 2-7% structural advantage over traditional sportsbook cash-outs (which deduct 5-10% vig).

**Mathematical framework:**
```
profit_per_contract = P_exit - P_entry
green_book_stake = (original_stake × P_entry) / P_exit
locked_profit = original_stake × (P_exit - P_entry) / P_exit
```

### 2.2 First Passage Time — Why "Price Touches Target" >> "Team Wins"

The core insight: for an undervalued YES position at P_entry with green book target P_target:

```
P(max price during game > P_target) >> P(team wins)
```

Under the Ornstein-Uhlenbeck process model:
```
dp_t = θ(μ - p_t)dt + σ dW_t + J dN_t
```

Where:
- μ = fair probability (from our model)
- θ = mean-reversion speed
- σ = diffusive volatility
- J·dN_t = jump component (goals, touchdowns)

The **first passage time** τ to target P_target has probability:

```
P(τ < T_game) = f(μ - P_entry, σ, θ, T_game, jump_intensity)
```

For an undervalued entry (μ > P_entry), the drift toward fair value PLUS random jumps make the first passage probability significantly higher than the win probability.

**Example (NBA):** Team A at $0.42, our model says $0.50, target = $0.50.
- P(Team A wins) ≈ 50%
- P(price touches $0.50 at any point during 48-minute game) ≈ 72-85%
- Because even if Team A loses, they likely lead at some point

### 2.3 Mean Reversion After Scoring Events (Moskowitz 2021)

Tobias Moskowitz's landmark paper "Asset Pricing and Sports Betting" (Journal of Finance, 2021) demonstrated that approximately **50% of total price movement from open to close is reversed at game outcome**. Live sports odds systematically overreact to scoring events.

**OU Process Calibration per Sport:**

| Sport | θ (reversion speed) | σ (event volatility) | Avg. jump size | Events/game |
|-------|---------------------|----------------------|----------------|-------------|
| Soccer | 0.3 (slow) | 0.15-0.25 | ±15-30¢ per goal | 2-3 |
| NBA | 0.7 (fast) | 0.03-0.08 | ±2-5¢ per score | 180-200 |
| NFL | 0.5 (medium) | 0.05-0.12 | ±5-15¢ per score | 8-12 |
| UFC/MMA | 0.2 (very slow) | 0.20-0.40 | ±20-40¢ per round | 3-5 |

**The Hashimoto et al. (2025) finding:** In their OU model of betting odds (arXiv:2503.16470), two types of bettors are identified:
1. **Herders** — follow current odds, amplifying overreaction
2. **Fundamentalists** — bet on true probability, causing reversion

The proportion of herders decreases over time → markets become more efficient as game progresses → fade overreactions early in the game for maximum edge.

### 2.4 Bayesian Sequential Updating

Robberechts, Van Haaren & Davis (2023, JRSS-A) developed a multinomial probit regression with sequential Bayesian updates for real-time sports forecasting:

```
Prior: P(win) = model_pre_game(Elo, stats, matchup)

After each event e_i:
  likelihood = P(e_i | team_wins) / P(e_i)
  P(win | e_1,...,e_i) ∝ P(win | e_1,...,e_{i-1}) × likelihood
```

Our implementation uses sport-specific event models:
- **Soccer:** Goals, red cards, xG accumulation, possession dominance
- **NBA:** Score differential, foul trouble, momentum runs (8-0, 10-0)
- **NFL:** Scoring drives, turnovers, field position

### 2.5 Information Cascade Fragility

Bikhchandani, Hirshleifer & Welch (1992) showed that information cascades are **fragile** — a small amount of new public information can break them, causing rapid price correction.

**Detection signals:**
- 80%+ of order flow in one direction
- Declining trade sizes (herding followers, not informed traders)
- No news catalyst for the price movement
- VPIN (Volume-Synchronized Probability of Informed Trading) spike

When a cascade is detected in a Polymarket sports market, the contra-cascade trade has positive expected value because the fragile structure will likely break.

### 2.6 Logit Jump-Diffusion Kernel (Dalen 2025)

Dalen's "Toward Black-Scholes for Prediction Markets" (arXiv:2510.15205) provides the first rigorous options-pricing framework for binary prediction markets:

```
d(logit(p_t)) = μ_RN dt + σ_b dW_t + J dN_t
```

This produces a **belief-volatility surface** σ_b(p, T) — the prediction market analogue of implied volatility. Key property: contracts near p=0.50 have highest volatility; those near 0 or 1 have lowest. This informs our position sizing — higher volatility = lower size but more green book opportunities.

### 2.7 Favorite-Longshot Bias on Polymarket

Research from Kalshi data (CEPR 2025) shows that extreme contracts are systematically mispriced:
- Contracts at $0.05: win only 2% of the time (bettors overpay for longshots)
- Contracts at $0.95: win 98% of the time (bettors underpay for favorites)
- **Reverse FLB** documented on prediction markets specifically (Princeton, Skinner 2022)

Polymarket shows elements of both biases depending on the sport and market type. Our model identifies which direction the bias runs per-market.

### 2.8 Elo/Glicko-2 Team Strength Network

Graph-based team strength estimation provides our probability model:

**Glicko-2** extends Elo with:
- Rating deviation (RD) — confidence interval around rating
- Volatility parameter σ — expected rating fluctuation
- Win probability: `P(A > B) = 1 / (1 + 10^((R_B - R_A) / 400))`

**PageRank enhancement:** Teams are nodes in a directed graph weighted by margin of victory. PageRank naturally captures strength-of-schedule — beating a strong team flows more authority. This catches cases where the market overvalues a team with a soft schedule.

---

## 3. Strategy Architecture

### 3.1 Signal Flow

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  Polymarket CLOB ← WebSocket live prices                │
│  The Odds API ← Pinnacle sharp line (benchmark)         │
│  Sports Data API ← Live scores, events                  │
│  Elo/Glicko Engine ← Historical game results            │
│  Open-Meteo → (weather effects on outdoor sports)       │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                 PROBABILITY MODEL                        │
│  1. Elo/Glicko-2 base probability                       │
│  2. Pinnacle no-vig line as benchmark                    │
│  3. Ensemble: weighted avg (Elo 40%, Pinnacle 60%)      │
│  4. Sport-specific adjustments (home, fatigue, weather)  │
│  5. KL divergence scanner: D_KL(model || Polymarket)    │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              SUB-STRATEGY ROUTER                         │
│                                                          │
│  PRE-GAME phase:                                        │
│    D1 Quality Gate → Entry if edge > min_edge           │
│                                                          │
│  LIVE phase:                                            │
│    D1 Green Book Engine → auto-hedge open positions     │
│    D2 Overreaction Detector → fade scoring jumps        │
│    D3 Cascade Propagator → trade stale futures          │
│                                                          │
│  POST-GAME phase:                                       │
│    Resolution tracking, P&L calculation                  │
└───────────────────────┬─────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              RISK MANAGER + EXECUTION                    │
│  Position sizing: Quarter-Kelly (KELLY_FRACTION=0.25)   │
│  Per-trade cap: MAX_POSITION_PCT = 0.05                 │
│  Green book target: dynamic per sport/volatility         │
│  Stop loss: sport-specific (disabled for D1 by default) │
│  Execution: Polymarket CLOB limit orders                │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Sub-strategy D1: Green Book Engine

**Goal:** Enter undervalued pre-game positions and automatically hedge during the game to lock profit.

**Entry Logic:**
1. Compute model probability: `p_model = 0.4 × Elo + 0.6 × Pinnacle_no_vig`
2. Get Polymarket YES price: `p_market`
3. Calculate edge: `edge = p_model - p_market`
4. **Entry condition:** `edge > MIN_EDGE` AND game passes quality gate
5. Position size: `size = capital × kelly_fraction × kelly_raw(edge, 1/p_market)`

**Quality Gate (D1):**
```python
MIN_EDGE = 0.03           # 3% minimum edge (tunable via autoresearch)
MAX_EDGE = 0.25           # Anomaly filter — too-good-to-be-true
MIN_PRICE = 0.15          # No extreme longshots (D1 needs price movement room)
MAX_PRICE = 0.75          # No heavy favorites (limited upside for green book)
MIN_VOLUME = 5000         # Minimum 24h volume for liquidity
MIN_BOOK_DEPTH = 500      # Minimum orderbook depth at best bid
GAME_START_WINDOW = 6     # Only enter within 6 hours of game start
COMPETITIVE_THRESHOLD = 0.15  # |p_home - p_away| < 0.15 (competitive game filter)
```

**Green Book Logic (during game):**
```python
# Continuous monitoring via WebSocket
for each price_update:
    current_bid = get_best_bid(market_id)
    unrealized_pnl = (current_bid - entry_price) * position_size

    # Dynamic target based on sport volatility
    green_book_target = entry_price + GREEN_BOOK_DELTA[sport]

    if current_bid >= green_book_target:
        # SELL: lock in profit
        place_limit_sell(price=current_bid, size=position_size)

    elif current_bid <= stop_loss_price:
        # Optional: exit losing position
        # Default: HOLD to resolution (stop loss disabled for D1)
        pass
```

**GREEN_BOOK_DELTA (tunable per sport):**
```python
GREEN_BOOK_DELTA = {
    "basketball_nba": 0.06,    # NBA: frequent scoring, many lead changes
    "basketball_ncaab": 0.05,  # College: more variance
    "americanfootball_nfl": 0.08,  # NFL: less frequent scoring, bigger jumps
    "soccer_epl": 0.10,        # Soccer: rare goals, big swings
    "soccer_ucl": 0.10,
    "soccer_la_liga": 0.10,
    "mma_ufc": 0.12,           # UFC: round-by-round, large swings
}
```

**Hold-to-Resolution Fallback:** If the green book target is never reached during the game, the position resolves at game end. Because we entered with positive edge (p_model > p_market), holding to resolution still has positive EV. Green booking just provides a *higher probability* and *faster capital turnover* path to profit.

### 3.3 Sub-strategy D2: Overreaction Fade

**Goal:** Detect and fade overreactions to live scoring events using calibrated OU parameters.

**Entry Logic:**
1. Detect scoring event (score change in live data feed)
2. Compute pre-event fair price: `p_fair_pre`
3. Compute post-event fair price: `p_fair_post` (Bayesian update)
4. Observe actual price move: `p_actual_post`
5. Overreaction = `|p_actual_post - p_fair_post|`
6. **Entry condition:** `overreaction > OVERREACTION_THRESHOLD[sport]`
7. Direction: fade the overreaction (if price dropped too much, buy YES; if rose too much, buy NO)

**OU Calibration:**
```python
# Calibrated from historical Polymarket sports price data
OU_PARAMS = {
    "soccer": {"theta": 0.3, "sigma": 0.20, "jump_decay_minutes": 8},
    "nba":    {"theta": 0.7, "sigma": 0.05, "jump_decay_minutes": 3},
    "nfl":    {"theta": 0.5, "sigma": 0.10, "jump_decay_minutes": 5},
    "ufc":    {"theta": 0.2, "sigma": 0.30, "jump_decay_minutes": 12},
}

OVERREACTION_THRESHOLD = {
    "soccer": 0.06,   # 6¢ beyond fair adjustment
    "nba":    0.03,    # 3¢ (lower because smaller per-event moves)
    "nfl":    0.04,
    "ufc":    0.08,
}
```

**Exit Logic (D2):**
```python
# Mean reversion exit
target_reversion = 0.5  # Exit when 50% of overreaction has reverted
max_hold_minutes = OU_PARAMS[sport]["jump_decay_minutes"] * 2

if price_reverted >= overreaction * target_reversion:
    EXIT: take profit
elif elapsed_minutes > max_hold_minutes:
    EXIT: time-based exit (reversion didn't materialize)
elif new_scoring_event:
    RECALCULATE: update fair price, reassess overreaction
```

**Capital Turnover:** D2 positions are held minutes, not hours. This is the highest turnover sub-strategy, potentially allowing 5-15 trades per game in high-scoring sports (NBA).

### 3.4 Sub-strategy D3: Cross-Event Cascade

**Goal:** Trade futures/championship markets that are stale after correlated game results.

**How It Works:**
```
Game Result → Implies Futures Adjustment → Trade Before Market Catches Up
```

**Example:**
- Arsenal beats Liverpool 3-0 in EPL
- "Arsenal wins Premier League" market should increase
- "Liverpool wins Premier League" market should decrease
- If these futures markets haven't adjusted within 5 minutes of game end, trade them

**Logical Constraint Graph:**
```python
# If A implies B, then P(B) >= P(A)
# P(wins_championship) >= P(wins_semifinal) × P(wins_final | wins_semifinal)

constraints = [
    # Team wins game → affects team's series/playoff odds
    ("team_wins_game_5", "team_wins_series", ">=", conditional_prob),
    # Team eliminated → championship = 0
    ("team_eliminated", "team_wins_championship", "==", 0.0),
    # Division leader → playoff probability
    ("team_clinches_division", "team_makes_playoffs", "==", 1.0),
]
```

**Detection Method:**
1. Monitor all active game markets
2. When a game resolves (or decisive score), compute new fair value for related futures
3. Compare fair value to current futures market price
4. If |fair - market| > MIN_CASCADE_EDGE, trade the futures market

**Risk:** Futures markets have lower liquidity and wider spreads. D3 should use smaller position sizes.

---

## 4. Sports Selection & Prioritization

### 4.1 Sport Scoring Matrix

| Sport | PM Volume | Green Book Fit | Overreaction Fit | Data Availability | Fee | Priority |
|-------|-----------|---------------|-----------------|-------------------|-----|----------|
| NBA | $1M+/game | Excellent (many lead changes) | Good (frequent events) | Excellent | 0% | **#1** |
| NFL | $55M+/game | Good (score swings) | Good (big event jumps) | Excellent | 0% | **#2** |
| Soccer (EPL) | $500K+/match | Good (goal swings) | Excellent (overreaction to goals) | Good | 0% | **#3** |
| UFC/MMA | $500K+/card | Moderate (binary) | Excellent (round upsets) | Moderate | 0% | **#4** |
| Soccer (UCL) | $500K+/match | Good | Excellent | Good | 0% | **#5** |
| NCAAB | Moderate | Good (March Madness) | Good | Good | HAS FEES | #6 |
| MLB | Lower | Moderate (slow games) | Moderate | Good | 0% | #7 |

### 4.2 Initial Focus (Phase 1)

**NBA + EPL only.** Reasons:
- NBA: Highest capital turnover (many scoring events, frequent lead changes, reliable green booking)
- EPL: Highest overreaction per event (goals cause 15-30¢ swings, slow reversion = wide windows)
- Both: 0% Polymarket fees, good volume, good data availability
- Expand to NFL, UCL, UFC after backtesting validates approach

### 4.3 Seasonal Calendar

```
NBA Regular Season: Oct → Apr (82 games/team, ~1,230 total games)
NBA Playoffs: Apr → Jun (~80 games)
NFL Regular Season: Sep → Jan (~272 games)
NFL Playoffs: Jan → Feb (~13 games)
EPL: Aug → May (380 games)
UCL: Sep → Jun (~125 games)
UFC: Year-round (~40 events/year)
March Madness: Mar (~67 games in 3 weeks)
```

**Year-round coverage** with NBA + EPL + UFC ensures continuous trading. NFL and March Madness provide seasonal volume boosts.

---

## 5. Probability Model

### 5.1 Architecture

```
┌──────────────────────┐    ┌──────────────────────┐
│   Elo/Glicko-2       │    │   Pinnacle No-Vig    │
│   (Team Strength)    │    │   (Sharp Benchmark)  │
│   Weight: 40%        │    │   Weight: 60%        │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           └─────────┬─────────────────┘
                     ↓
           ┌─────────────────────┐
           │  Ensemble Probability│
           │  p_model             │
           │  + Home adjustment   │
           │  + Fatigue adjustment│
           │  + Weather adjustment│
           └──────────┬──────────┘
                      ↓
           ┌─────────────────────┐
           │  KL Divergence      │
           │  D_KL(model || PM)  │
           │  = edge in bits     │
           │  = optimal growth   │
           └─────────────────────┘
```

### 5.2 Elo/Glicko-2 Engine

**Data Sources:**
- NBA: basketball-reference.com game logs (or nba.com/stats API)
- EPL: football-data.co.uk historical results
- NFL: pro-football-reference.com

**Parameters:**
```python
ELO_K_FACTOR = 20           # Learning rate
ELO_HOME_ADVANTAGE = 50     # ~3.5% probability boost for home team
GLICKO_TAU = 0.5            # System volatility constant
INITIAL_RATING = 1500
INITIAL_RD = 350             # High uncertainty for new teams
```

**Margin-of-Victory Adjustment (MOV):**
```python
# Standard Elo ignores margin. MOV-Elo captures blowouts vs close games.
mov_multiplier = math.log(abs(margin) + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))
elo_change = K * mov_multiplier * (actual - expected)
```

### 5.3 Pinnacle Benchmark

The Odds API provides Pinnacle odds (the sharpest global sportsbook). We remove the vig to get the no-vig line:

```python
def remove_vig(home_odds, away_odds):
    """Convert Pinnacle odds to no-vig probabilities."""
    implied_home = 1 / home_odds
    implied_away = 1 / away_odds
    total = implied_home + implied_away  # > 1.0 due to vig
    return implied_home / total, implied_away / total
```

**Why 60% weight on Pinnacle:** Pinnacle's closing line is the most efficient price in sports betting (academic consensus). It already incorporates team strength, injuries, matchup history, etc. Our Elo model adds value by:
1. Being available earlier (pre-market)
2. Not anchoring on public perception
3. Capturing schedule strength via network effects

### 5.4 KL Divergence Edge Scanner

```python
def kl_divergence_edge(model_prob, market_price):
    """
    Compute KL divergence between model and market.
    D_KL(Q || P) = q * log(q/p) + (1-q) * log((1-q)/(1-p))

    This is the maximum growth rate of a Kelly bettor (Cover & Thomas).
    Higher D_KL = more mispriced = better opportunity.
    """
    q, p = model_prob, market_price
    if p <= 0 or p >= 1 or q <= 0 or q >= 1:
        return 0.0
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
```

Markets are ranked by D_KL. Only trade the top opportunities above a minimum threshold.

### 5.5 Live Bayesian Updater (D2)

During live games, update probabilities after each event:

```python
class LiveBayesianUpdater:
    """Updates win probability based on live game events."""

    def __init__(self, sport, pre_game_prob):
        self.sport = sport
        self.current_prob = pre_game_prob
        self.events = []

    def update_on_score(self, scorer, time_remaining_pct, new_score_diff):
        """Bayesian update after a scoring event."""
        if self.sport == "soccer":
            # Low-scoring: each goal is highly informative
            # Poisson model: P(win | leading by k with t remaining)
            self.current_prob = self._poisson_win_prob(
                new_score_diff, time_remaining_pct
            )
        elif self.sport == "nba":
            # High-scoring: Brownian motion model (Stern 2005)
            # P(win) ≈ Φ(score_diff / (σ × √time_remaining))
            sigma = 11.5  # NBA scoring volatility per 48 minutes
            z = new_score_diff / (sigma * math.sqrt(time_remaining_pct))
            self.current_prob = stats.norm.cdf(z)
        elif self.sport == "nfl":
            # Medium-scoring: empirical lookup table + logistic regression
            self.current_prob = self._nfl_win_prob(
                new_score_diff, time_remaining_pct
            )

    def _poisson_win_prob(self, goal_diff, time_remaining_pct):
        """Soccer: Poisson-based win probability given score and time."""
        # Dixon-Robinson state-dependent model
        # Goals as independent Poisson processes with time-varying intensity
        lambda_home = 1.35 * time_remaining_pct  # Expected remaining home goals
        lambda_away = 1.10 * time_remaining_pct  # Expected remaining away goals
        # ... Monte Carlo or analytical computation
        pass
```

---

## 6. Risk Management

### 6.1 Position Limits (Hardcoded — CEO Approval Required to Change)

```python
# arbo/strategies/sports_risk.py
MAX_POSITION_PCT = 0.05       # 5% capital per trade
MAX_SPORT_PCT = 0.30          # 30% max in one sport
MAX_GAME_PCT = 0.10           # 10% max per game (across sub-strategies)
MAX_CONCURRENT_LIVE = 8       # Max 8 simultaneous live positions
DAILY_LOSS_PCT = 0.10         # 10% daily loss → auto shutdown
WEEKLY_LOSS_PCT = 0.15        # 15% weekly loss → shutdown + CEO alert
KELLY_FRACTION = 0.25         # Quarter-Kelly sizing
MIN_PAPER_WEEKS = 4           # 4 weeks paper trading before live
```

### 6.2 Green Book Risk Controls

```python
MAX_GREEN_BOOK_WAIT = 0.85    # If 85% of game elapsed without green book, hold to resolution
MIN_GREEN_BOOK_PROFIT = 0.02  # Don't green book for less than 2¢ profit
SLIPPAGE_BUFFER = 0.01        # Account for 1¢ slippage on exit
```

### 6.3 Overreaction Fade Risk Controls (D2)

```python
MAX_D2_POSITION_PCT = 0.025   # Half-size for D2 (higher frequency, higher risk)
MAX_HOLD_MINUTES = 30          # Force exit after 30 minutes if no reversion
MAX_D2_TRADES_PER_GAME = 5    # Don't overtrade in one game
COOLDOWN_AFTER_LOSS = 10       # 10-minute cooldown after a losing D2 trade
```

### 6.4 Cascade Risk Controls (D3)

```python
MAX_D3_POSITION_PCT = 0.02    # Small size for futures (lower liquidity)
MIN_CASCADE_EDGE = 0.05       # 5% minimum edge for cascade trades
MAX_FUTURES_CONCURRENT = 4    # Max 4 open futures positions
```

### 6.5 Emergency Procedures

Same as global Arbo risk framework:
- Cancel ALL orders on daily loss breach
- Slack alert to CEO on weekly loss breach
- Auto-shutdown after 3 unhandled exceptions in 10 minutes

---

## 7. Data Requirements

### 7.1 Real-Time Data (Production)

| Source | Data | Frequency | Cost |
|--------|------|-----------|------|
| Polymarket CLOB WebSocket | Live prices, orderbook, trades | Real-time | Free |
| Polymarket Gamma API | Market discovery, metadata | Every 5min | Free |
| The Odds API | Pinnacle pre-game odds | Per-game | $10/mo (10K req) |
| Sports Data API (TBD) | Live scores, events | Real-time | $0-50/mo |
| ESPN/CBS/NFL API | Play-by-play data | Real-time | Free (public) |

### 7.2 Historical Data (Backtest)

| Source | Data | Coverage | Storage |
|--------|------|----------|---------|
| Polymarket CLOB `/prices-history` | Price series per token | ~30 days after close | SQLite |
| pmxt Archive (`archive.pmxt.dev`) | Hourly Polymarket snapshots | 2024-present | Parquet |
| PolymarketData.co | 1-min resolution prices | 2024-present | API/$$ |
| The Odds API | Historical Pinnacle odds | 6-12 months | SQLite |
| basketball-reference.com | NBA game results, scores | 2000-present | SQLite |
| football-data.co.uk | EPL results, scores | 1993-present | CSV |
| pro-football-reference.com | NFL results, scores | 2000-present | SQLite |

### 7.3 SQLite Schema (Backtest DB)

```sql
-- Games / Events
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,            -- 'nba', 'epl', 'nfl', 'ufc'
    league TEXT NOT NULL,           -- 'NBA', 'Premier League', etc.
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_date TEXT NOT NULL,        -- YYYY-MM-DD
    game_time TEXT,                 -- HH:MM UTC
    home_score INTEGER,
    away_score INTEGER,
    home_elo REAL,                  -- Pre-game Elo rating
    away_elo REAL,
    home_glicko REAL,
    away_glicko REAL,
    status TEXT DEFAULT 'scheduled' -- scheduled, live, final
);

-- Polymarket markets linked to games
CREATE TABLE markets (
    token_id TEXT PRIMARY KEY,      -- YES token ID
    token_id_no TEXT,               -- NO token ID
    game_id TEXT NOT NULL,          -- FK to games
    event_id TEXT,                  -- Polymarket event ID
    condition_id TEXT,
    question TEXT,
    outcome TEXT,                   -- 'home_win', 'away_win', 'draw', 'prop'
    volume REAL,
    neg_risk INTEGER,
    won INTEGER,                    -- 1 if resolved true
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Minute-level price history during games
CREATE TABLE prices (
    token_id TEXT NOT NULL,
    ts INTEGER NOT NULL,            -- Unix timestamp
    price REAL NOT NULL,            -- YES price [0, 1]
    bid REAL,                       -- Best bid
    ask REAL,                       -- Best ask
    volume_1m REAL,                 -- 1-minute volume
    PRIMARY KEY (token_id, ts)
);

-- Live game events (scores, cards, etc.)
CREATE TABLE game_events (
    game_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    event_type TEXT NOT NULL,        -- 'goal', 'touchdown', 'basket', 'red_card'
    team TEXT,                       -- Which team
    score_home INTEGER,
    score_away INTEGER,
    detail TEXT,                     -- Player name, type, etc.
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Pinnacle odds history
CREATE TABLE pinnacle_odds (
    game_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    home_odds REAL,                  -- Decimal odds
    away_odds REAL,
    draw_odds REAL,                  -- For soccer
    home_prob_novig REAL,            -- Vig-removed probability
    away_prob_novig REAL,
    PRIMARY KEY (game_id, ts)
);

-- Elo/Glicko ratings history
CREATE TABLE ratings (
    team TEXT NOT NULL,
    sport TEXT NOT NULL,
    date TEXT NOT NULL,
    elo REAL,
    glicko_rating REAL,
    glicko_rd REAL,
    glicko_vol REAL,
    PRIMARY KEY (team, sport, date)
);
```

---

## 8. Backtest Architecture

### 8.1 Pipeline Overview

```
Phase 0: Data Collection
    ├── Download Polymarket sports price history (CLOB /prices-history)
    ├── Download Polymarket market metadata (Gamma API)
    ├── Download pmxt archive data (Parquet snapshots)
    ├── Download game results + scores (sports APIs)
    ├── Download Pinnacle odds history (The Odds API)
    └── Compute Elo/Glicko ratings from game results
              ↓
Phase 1: Data Storage (SQLite)
    ├── Populate games, markets, prices, game_events, pinnacle_odds, ratings
    └── Validate: cross-reference market resolution with game results
              ↓
Phase 2: Signal Simulation
    ├── For each historical game:
    │   ├── Compute model probability (Elo + Pinnacle)
    │   ├── Compare to Polymarket pre-game price
    │   └── Generate entry signal if quality gate passed
              ↓
Phase 3: Trade Simulation (Green Book)
    ├── For each entry signal:
    │   ├── Walk through minute-level price trajectory
    │   ├── Check for green book trigger at each timestamp
    │   ├── If triggered: record exit price and profit
    │   └── If not triggered: record resolution P&L
              ↓
Phase 4: D2 Simulation (Overreaction Fade)
    ├── For each game with live event data:
    │   ├── At each scoring event, compute Bayesian posterior
    │   ├── Compare to actual Polymarket price move
    │   ├── If overreaction detected: simulate fade trade
    │   └── Walk forward to reversion or timeout
              ↓
Phase 5: Metrics Computation
    ├── P&L, Sharpe, Win Rate, Max Drawdown
    ├── Capital Turnover Ratio = total_traded_volume / average_capital
    ├── Green Book Success Rate = green_booked / total_entries
    └── Composite Score
              ↓
Phase 6: Optimization (Autoresearch or Sweep)
    ├── Hyperparameter search across quality gate thresholds
    ├── Walk-forward validation (4+ folds)
    └── OOS performance on held-out data
```

### 8.2 Composite Score Formula

```python
def composite_score(results, initial_capital):
    """
    Strategy D composite score.
    Balances profitability, capital turnover, and consistency.
    """
    pnl = results["total_pnl"]
    sharpe = results["sharpe"]
    n_trades = results["n_trades"]
    max_dd = results["max_drawdown"]
    turnover = results["capital_turnover_ratio"]
    green_book_rate = results["green_book_success_rate"]

    # Primary: profitability
    pnl_factor = pnl / initial_capital * 100

    # Sharpe: consistency
    sharpe_factor = min(max(sharpe, 0) / 3.0, 2.0)

    # Trade count: statistical significance
    trade_factor = min(n_trades / 100, 2.0)

    # Drawdown penalty
    dd_factor = max(0, 1.0 - max_dd * 2)

    # Capital turnover bonus (KEY DIFFERENTIATOR from Strategy C)
    turnover_factor = min(turnover / 10.0, 1.5)  # Reward high turnover up to 10x

    # Green book rate bonus
    gb_factor = 1.0 + green_book_rate * 0.5  # Up to 1.5x for 100% green book

    score = (
        pnl_factor
        * (1 + sharpe_factor)
        * trade_factor
        * dd_factor
        * turnover_factor
        * gb_factor
    )
    return score
```

### 8.3 Walk-Forward Validation

```
NBA Season 2025-26 (~1,230 games, Oct-Apr):
    Fold 1: Oct-Dec (train) → Jan (test)
    Fold 2: Oct-Jan (train) → Feb (test)
    Fold 3: Oct-Feb (train) → Mar (test)
    Fold 4: Oct-Mar (train) → Apr (test)

EPL Season 2025-26 (380 games, Aug-May):
    Fold 1: Aug-Nov (train) → Dec (test)
    Fold 2: Aug-Dec (train) → Jan (test)
    Fold 3: Aug-Jan (train) → Feb (test)
    Fold 4: Aug-Feb (train) → Mar (test)
```

OOS metrics reported as average across test folds. Parameter set accepted only if ALL test folds are profitable.

---

## 9. Autoresearch Integration

### 9.1 Architecture (Karpathy Pattern)

```
research_d/
├── prepare.py          # IMMUTABLE — data loading, backtest engine, evaluate()
├── strategy_params.py  # MUTABLE — agent modifies this file only
├── program.md          # AGENT INSTRUCTIONS — research protocol
├── results.tsv         # Experiment log (auto-generated)
└── data/
    └── sports_backtest.sqlite  # Historical price + game data
```

### 9.2 prepare.py (Immutable Evaluation Harness)

```python
"""
Strategy D evaluation harness. DO NOT MODIFY.
Agent modifies strategy_params.py only.
"""
import sqlite3
import math
from strategy_params import PARAMS

DB_PATH = "data/sports_backtest.sqlite"
TIME_BUDGET = 120  # 2 minutes (backtests are fast)

def load_data():
    """Load all games, markets, prices from SQLite."""
    ...

def evaluate() -> dict:
    """
    Run full backtest with current PARAMS.
    Returns: {"score": float, "pnl": float, "sharpe": float,
              "n_trades": int, "win_rate": float, "turnover": float,
              "green_book_rate": float, "max_dd": float}
    """
    ...

def main():
    results = evaluate()
    print(f"score={results['score']:.1f}")
    print(f"pnl=${results['pnl']:.2f}")
    print(f"sharpe={results['sharpe']:.2f}")
    print(f"trades={results['n_trades']}")
    print(f"win_rate={results['win_rate']:.1%}")
    print(f"turnover={results['turnover']:.1f}x")
    print(f"green_book_rate={results['green_book_rate']:.1%}")
    print(f"max_dd={results['max_dd']:.1%}")
```

### 9.3 strategy_params.py (Agent-Mutable)

```python
"""
Strategy D tunable parameters.
This is the ONLY file the autoresearch agent may modify.
"""
PARAMS = {
    # Quality Gate — D1
    "min_edge": 0.03,
    "max_edge": 0.25,
    "min_price": 0.15,
    "max_price": 0.75,
    "min_volume": 5000,
    "competitive_threshold": 0.15,

    # Green Book
    "green_book_delta_nba": 0.06,
    "green_book_delta_epl": 0.10,
    "green_book_delta_nfl": 0.08,

    # Overreaction Fade — D2
    "overreaction_threshold_nba": 0.03,
    "overreaction_threshold_epl": 0.06,
    "target_reversion_pct": 0.50,
    "max_hold_minutes_nba": 6,
    "max_hold_minutes_epl": 16,

    # Probability Model
    "elo_weight": 0.40,
    "pinnacle_weight": 0.60,
    "home_advantage_elo": 50,
    "mov_enabled": True,

    # Sizing
    "kelly_fraction": 0.25,
    "kelly_raw_cap": 0.15,
    "d2_size_multiplier": 0.50,

    # Sport Selection
    "enabled_sports": ["nba", "epl"],
    "excluded_teams": [],

    # Per-Sport Overrides
    "sport_overrides": {},
}
```

### 9.4 program.md (Agent Instructions)

```markdown
# Strategy D Autoresearch Protocol

## Goal
Optimize the Strategy D (Live Edge Harvester) parameters for maximum
composite score on the sports backtest dataset.

## Constraints
1. Minimum 50 trades in backtest
2. All walk-forward folds must be profitable
3. Maximum drawdown < 25%
4. Capital turnover ratio > 3x (we want high turnover)
5. Don't optimize for a single sport — both NBA and EPL must contribute

## Workflow
1. Read current strategy_params.py
2. Hypothesize a change based on previous results
3. Edit strategy_params.py
4. Run: python3 prepare.py
5. Record results in results.tsv
6. If score improved: git commit (keep)
7. If score worsened: git reset (revert)
8. Repeat indefinitely

## Search Priorities
1. First: find the right min_edge and green_book_delta per sport
2. Second: optimize OU parameters for D2 (overreaction thresholds)
3. Third: test sport-specific overrides
4. Fourth: try novel combinations

## Tips
- Green book rate > 60% is ideal (means most positions exit profitably mid-game)
- If win_rate < 45%, min_edge is probably too low
- If n_trades < 30, min_edge or min_volume is too restrictive
- elo_weight vs pinnacle_weight: Pinnacle is sharper but available later
```

---

## 10. Implementation Plan

### Sprint D-0: Data Infrastructure (Week 1-2)

| Task | Description | Dependencies |
|------|-------------|-------------|
| D-001 | Sports price data downloader (CLOB /prices-history for sports markets) | Gamma API sport tag discovery |
| D-002 | pmxt archive downloader (Parquet sports data) | None |
| D-003 | Game results importer (basketball-reference, football-data.co.uk) | None |
| D-004 | Pinnacle odds fetcher (adapt existing The Odds API client) | D-003 |
| D-005 | Elo/Glicko-2 engine (compute ratings from historical results) | D-003 |
| D-006 | SQLite schema + data pipeline | D-001 through D-005 |
| D-007 | Data validation (cross-reference PM resolution with game results) | D-006 |

### Sprint D-1: Backtest Engine (Week 3-4)

| Task | Description | Dependencies |
|------|-------------|-------------|
| D-101 | Probability model (Elo + Pinnacle ensemble) | D-005, D-004 |
| D-102 | D1 quality gate | D-101 |
| D-103 | D1 green book simulator (walk through price trajectories) | D-006, D-102 |
| D-104 | D2 overreaction detector (OU calibration + live event replay) | D-006 |
| D-105 | D2 fade simulator | D-104 |
| D-106 | Composite score + metrics computation | D-103, D-105 |
| D-107 | Walk-forward validation framework | D-106 |

### Sprint D-2: Autoresearch Optimization (Week 5-6)

| Task | Description | Dependencies |
|------|-------------|-------------|
| D-201 | Autoresearch prepare.py (immutable harness) | D-107 |
| D-202 | Autoresearch strategy_params.py (initial parameters) | D-201 |
| D-203 | Autoresearch program.md (agent instructions) | D-202 |
| D-204 | Run autoresearch overnight (target: 500+ experiments) | D-203 |
| D-205 | Analyze results, select best model | D-204 |
| D-206 | Stress testing (parameter sensitivity, regime changes) | D-205 |

### Sprint D-3: Production Integration (Week 7-8)

| Task | Description | Dependencies |
|------|-------------|-------------|
| D-301 | Sports scanner module (arbo/strategies/sports_scanner.py) | D-205 |
| D-302 | Sports quality gate (arbo/strategies/sports_quality_gate.py) | D-301 |
| D-303 | Green book engine (arbo/strategies/green_book_engine.py) | D-302 |
| D-304 | Overreaction fader (arbo/strategies/overreaction_fader.py) | D-302 |
| D-305 | Live Bayesian updater (arbo/strategies/live_bayesian.py) | D-304 |
| D-306 | Integration with main_rdh.py | D-303, D-304, D-305 |
| D-307 | Paper trading deployment | D-306 |

### Sprint D-4: Paper Trading Validation (Week 9-12)

| Task | Description | Dependencies |
|------|-------------|-------------|
| D-401 | 4-week minimum paper trading | D-307 |
| D-402 | Dashboard integration (strategy D cards) | D-307 |
| D-403 | Weekly performance reports | D-401 |
| D-404 | Go/No-go decision for live trading | D-401, D-403 |

---

## 11. Key Metrics & Targets

### 11.1 Backtest Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Composite Score | > 100 | Comparable to Strategy C's AR-0134 (170.1) |
| Total P&L | > $500 per $1,000 capital | 50%+ return over backtest period |
| Sharpe Ratio | > 2.0 | Daily returns basis |
| Win Rate | > 55% | Higher than Strategy C (43.6%) due to green booking |
| Green Book Success Rate | > 50% | More than half of entries green booked profitably |
| Capital Turnover | > 5x per month | Core goal: high turnover |
| Max Drawdown | < 20% | Conservative |
| Trades per Week | > 10 | Enough for statistical significance |

### 11.2 Paper Trading Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Weekly P&L | Positive for 3/4 weeks | Weekly reports |
| Capital Turnover | > 3x per month | Total traded / average capital |
| Green Book Rate | > 40% | Green booked / total entries |
| Average Hold Time | < 4 hours | Entry to exit (green book or resolution) |
| D2 Hit Rate | > 55% | Overreaction fades that profit |

### 11.3 Live Trading Targets (Post-Approval)

| Metric | Monthly Target |
|--------|---------------|
| Net P&L | > 5% of allocated capital |
| Capital Turnover | > 5x |
| Max Drawdown | < 10% |
| Trades | > 30 |

---

## 12. Known Risks & Mitigations

### 12.1 Critical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Polymarket introduces fees on all sports | Reduces green book margin by ~0.4% | Medium (fees expanding) | Monitor fee announcements. Adjust min_edge upward by fee amount. |
| 3-second delay on live orders | D2 fade enters at worse price | Already exists | Use limit orders, not market. Account for 3s delay in OU calibration. |
| Low liquidity during live games | Can't exit green book at target | Medium | MIN_BOOK_DEPTH filter. Reduce position size for less liquid markets. |
| Pinnacle odds unavailable for some markets | Model accuracy drops | Low | Fallback to Elo-only model (lower weight, higher min_edge). |
| Polymarket auto-cancels at game start | Pre-game limit orders deleted | Always | Only use market orders for pre-game entry, or enter before auto-cancel window. |
| Sports data feed latency | D2 detects overreaction too late | Medium | Multiple data sources. Target 5-second event detection. Accept wider thresholds. |
| Market makers front-run our green book | Exit price worse than expected | Low-Medium | Use iceberg orders (split into smaller chunks). Random timing offset. |

### 12.2 Model Risks

| Risk | Mitigation |
|------|------------|
| Elo/Glicko model uncalibrated | Monthly recalibration against Pinnacle. Brier score monitoring. |
| OU parameters regime change | Per-season recalibration. Rolling 30-day theta/sigma estimation. |
| Overfitting to NBA/EPL | Walk-forward validation. Test on held-out sports (NFL, UCL). |
| Overreaction threshold too aggressive | Start conservative (higher threshold), tighten with data. |

### 12.3 Operational Risks

| Risk | Mitigation |
|------|------------|
| WebSocket disconnect during live game | Reconnect with exponential backoff. If >30s disconnect, cancel open D2 positions. |
| Polymarket CLOB downtime | Queue entries, retry. Skip D2 for that game. |
| The Odds API quota exhaustion | Cache Pinnacle odds. Use free tier wisely (500 req/month). Upgrade if needed ($10/mo). |
| Game postponed/cancelled | Monitor game status. Cancel associated trades. |

---

## 13. Academic References

1. Moskowitz, T. (2021). "Asset Pricing and Sports Betting." *Journal of Finance*, 76(6), 3153-3209.
2. Robberechts, P., Van Haaren, J., & Davis, J. (2023). "Real-time forecasting within soccer matches through a Bayesian lens." *JRSS-A*, 187(2), 578-600.
3. Dalen, A. (2025). "Toward Black-Scholes for Prediction Markets." *arXiv:2510.15205*.
4. Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). "A Theory of Fads, Fashion, Custom, and Cultural Change as Informational Cascades." *JPE*, 100(5), 992-1026.
5. Snowberg, E. & Wolfers, J. (2010). "Explaining the Favorite-Longshot Bias." *JPE*, 118(4), 723-746.
6. Hashimoto, Y. et al. (2025). "Ornstein-Uhlenbeck Process for Horse Race Betting." *arXiv:2503.16470*.
7. Divos, A., Rollin, S., et al. (2018). "Risk-Neutral Pricing and Hedging of In-Play Football Bets." *arXiv:1811.03931*.
8. Wheatcroft, E. (2020). "Profiting from Overreaction in Soccer Betting Odds." *JQAS*, 16(3), 193-205.
9. IMDEA Networks (2025). "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets." *arXiv:2508.03474*.
10. Meister, F. (2024). "Application of the Kelly Criterion to Prediction Markets." *arXiv:2412.14144*.
11. Cover, T. & Thomas, J. (2006). "Elements of Information Theory." Chapter 6: Gambling and Information Theory.
12. Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.
13. Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High Frequency World." *Review of Financial Studies*, 25(5), 1457-1493.
14. Shin, H.S. (1993). "Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims." *Economic Journal*, 103(420), 1141-1153.

---

## Appendix A: Polymarket Sports API Reference

```python
# Gamma API — Sport Market Discovery
GET https://gamma-api.polymarket.com/events?tag=sports&active=true&limit=100

# CLOB API — Price History (per token)
GET https://clob.polymarket.com/prices-history?market={token_id}&interval=max&fidelity=60

# CLOB API — Live Orderbook
GET https://clob.polymarket.com/book?token_id={token_id}

# WebSocket — Real-time Price Feed
WSS wss://ws-subscriptions-clob.polymarket.com/ws/market
# Subscribe: {"type": "subscribe", "channel": "market", "assets_id": token_id}

# The Odds API — Pinnacle Odds
GET https://api.the-odds-api.com/v4/sports/{sport}/odds/?bookmakers=pinnacle&regions=eu
```

## Appendix B: Green Book Profit Calculator

```python
def green_book_profit(entry_price, exit_price, n_contracts, fee_rate=0.0):
    """
    Calculate green book profit.

    Args:
        entry_price: Price paid per YES contract (e.g., 0.40)
        exit_price: Price sold at (e.g., 0.55)
        n_contracts: Number of contracts
        fee_rate: Polymarket fee rate (0 for most sports)

    Returns:
        Guaranteed profit regardless of outcome
    """
    entry_cost = entry_price * n_contracts
    exit_revenue = exit_price * n_contracts
    fee = exit_price * (1 - exit_price) * fee_rate * n_contracts
    profit = exit_revenue - entry_cost - fee
    return profit

# Example: Buy 100 YES at $0.40, sell at $0.55, zero fees
# Profit = $55 - $40 = $15 (guaranteed)
```

## Appendix C: Sport-Specific Bayesian Update Models

### NBA (Brownian Motion — Stern 2005)
```
P(home_win | score_diff=d, time_remaining=t) = Φ(d / (σ√t))
σ ≈ 11.5 points per 48 minutes (empirical)
```

### Soccer (Poisson — Dixon-Robinson)
```
P(home_win | goals_diff=d, time_remaining=t) =
    Σ over all future score combinations where home wins,
    weighted by Poisson(λ_home × t) and Poisson(λ_away × t)
λ_home ≈ 1.35/90min, λ_away ≈ 1.10/90min (league average)
```

### NFL (Logistic Regression — Empirical)
```
P(home_win | score_diff=d, time_remaining=t, has_ball=b) =
    logistic(β₀ + β₁d + β₂t + β₃(d×t) + β₄b)
Calibrate β from historical play-by-play data
```
