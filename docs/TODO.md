# ARBO — Master TODO (Polymarket Pivot)

> Living document. Updated as work progresses.
> Status: [ ] pending, [~] in progress, [x] done, [-] blocked/waiting
> Pivoted from Matchbook to Polymarket: 2026-02-21

---

## Pre-Sprint: Project Restructuring

### Structure Migration
- [~] Create `arbo/` directory structure per v3 brief
- [~] Move `src/` to `_archive/` (preserve, exclude from imports)
- [ ] Migrate reusable utilities: logger.py, config pattern, db base, odds math
- [ ] Update pyproject.toml (packages, deps, tool configs)
- [ ] Update .env.example for Polymarket credentials
- [ ] Update config/settings.yaml for 9-layer architecture
- [ ] Update alembic config for new model location
- [ ] Verify `python3 -m pytest tests/ -v` passes after migration
- [ ] Verify `ruff check arbo/` and `black --check arbo/` clean

---

## Sprint 1: Foundation (Week 1–2)

**Goal:** Working connectivity to Polymarket + The Odds API, paper trading framework, opportunity scanning.
**Gate:** Connectivity OK, scanner runs, paper engine records trades.

### PM-001: Polymarket CLOB Client Wrapper
- [ ] `arbo/connectors/polymarket_client.py`
- [ ] Wrapper around py-clob-client with production-grade reliability
- [ ] Retry logic with exponential backoff (max 3 retries, base 1s)
- [ ] Rate limiting (respect Polymarket limits: POST 3500/10s, GET 1500/10s)
- [ ] Error handling: typed exceptions (RateLimitError, AuthError, NetworkError)
- [ ] Methods: get_markets(), get_orderbook(), get_price(), get_midpoint(), get_tick_size()
- [ ] L1 + L2 authentication (EIP-712 + HMAC-SHA256)
- [ ] Config via env vars: POLY_PRIVATE_KEY, POLY_FUNDER_ADDRESS, POLY_API_KEY, POLY_SECRET, POLY_PASSPHRASE
- [ ] Logging: every API call with timestamp, endpoint, response time, status
- [ ] `tests/test_polymarket_client.py`
- **Acceptance:** Connect to CLOB, get orderbook for 3 active markets, all return valid data (bids + asks non-empty, prices 0–1)

### PM-002: Market Discovery Module
- [ ] `arbo/connectors/market_discovery.py`
- [ ] Gamma API integration: list all active markets
- [ ] Filters: sport type, liquidity threshold (>$5K volume), active status, neg_risk flag, fee status
- [ ] Categorization: soccer, politics, crypto, entertainment, esports, attention_markets
- [ ] Persist to PostgreSQL: `markets` table with timestamp for historical tracking
- [ ] Refresh interval: every 15 minutes
- [ ] Extra fields: fee_enabled (boolean), maker_rebate_eligible (boolean)
- [ ] `tests/test_market_discovery.py`
- **Acceptance:** Returns ≥20 active soccer markets including EPL and La Liga. Results persisted in DB and re-loadable.

### PM-003: The Odds API Integration (ADAPT from Sprint 2)
- [ ] `arbo/connectors/odds_api_client.py`
- [ ] Adapt existing src/data/odds_api.py for Polymarket matching
- [ ] REST client for v4/sports/{sport}/odds endpoint
- [ ] Pinnacle odds extraction: moneyline, spread, totals for soccer leagues
- [ ] Mapping engine: The Odds API event ↔ Polymarket market (team names + date, fuzzy threshold 0.85)
- [ ] Rate limit management: track via x-requests-remaining header, NEVER exceed limit
- [ ] Cache: odds cache with 5min TTL (in-memory dict)
- [ ] `tests/test_odds_api_client.py`
- **Acceptance:** For 5 EPL matches, get Pinnacle odds and match with Polymarket markets. Handle team name variants.

### PM-004: Paper Trading Engine
- [ ] `arbo/core/paper_engine.py`
- [ ] PaperTradingEngine class — accepts OrderArgs, simulates fill at current midprice
- [ ] Simulated slippage: 0.5% default (configurable)
- [ ] P&L tracking: per-trade, per-strategy, per-day, per-week
- [ ] Position management: open positions, unrealized P&L, risk exposure per market type
- [ ] PostgreSQL tables: paper_trades, paper_positions, paper_snapshots
- [ ] Automatic resolution: polling resolved markets, update P&L
- [ ] Hourly snapshot: save portfolio state
- [ ] `tests/test_paper_engine.py`
- **Acceptance:** Simulated flow: open 3 positions, 1 WIN, 1 LOSE, 1 open. All P&L correct. Per-strategy breakdown works.

### PM-005: Opportunity Scanner (All Layers)
- [ ] `arbo/core/scanner.py`
- [ ] Layer 1 (MM): markets with spread >3%, volume >$1K/day
- [ ] Layer 2 (Value): Pinnacle vs Polymarket divergence >3% after fee
- [ ] Layer 3 (Arb placeholder): NegRisk markets where sum(YES prices) < $0.97 or > $1.03
- [ ] Layer 4 (Whale placeholder): log top wallet addresses from leaderboard
- [ ] Layer 6 (Crypto placeholder): 15min crypto market discovery
- [ ] Unified output format: Signal(layer, market_id, direction, edge, confidence, timestamp)
- [ ] All signals logged to DB (signals table) + console with timestamp
- [ ] `tests/test_scanner.py`
- **Acceptance:** Runs 30 minutes, detects ≥5 opportunities from Layer 1+2, all logged in DB and console.

### PM-006: Config & Secrets Management
- [ ] `arbo/config/settings.py` — Pydantic Settings class (adapt from existing)
- [ ] `config/settings.yaml` — risk params, strategy thresholds, API endpoints, scan intervals
- [ ] `.env.example` with Polymarket placeholders (NEVER real values)
- [ ] Pre-commit hook: detect-secrets scan
- [ ] NEVER copy private key to code, logs, or error messages
- [ ] `tests/test_config.py`
- **Acceptance:** `ruff check .` + `black --check .` clean. `grep -r "0x" arbo/ --include="*.py"` contains no private key.

### PM-007: Risk Manager (Core)
- [ ] `arbo/core/risk_manager.py` — SINGLETON
- [ ] `pre_trade_check(order)` → approve/reject with reason
- [ ] `post_trade_update(fill)` → update exposure, P&L, trigger alerts
- [ ] `emergency_shutdown()` → cancel all orders, log, notify CEO via Slack
- [ ] Checks: position size (5%), daily loss (10%), weekly loss (20%), market concentration (30%)
- [ ] Hardcoded limits — NOT overridable by config
- [ ] `tests/test_risk_manager.py`
- **Acceptance:**
  1. Order for 6% capital → REJECTED
  2. After 10% daily loss → automatic shutdown
  3. Order passes pre_trade_check → post_trade_update updates exposure
  4. Emergency shutdown cancels all orders (mock test)

### PM-104: Fee Model (moved from Sprint 2)
- [ ] `arbo/core/fee_model.py`
- [ ] `fee = p * (1-p) * FEE_RATE` for fee-enabled markets
- [ ] Dynamically detect fee-enabled markets (Gamma API feesEnabled field)
- [ ] Include in ALL edge calculations and P&L
- [ ] Maker rebate calculation for fee-enabled markets
- [ ] `tests/test_fee_model.py`
- [ ] **CEO note:** Approved moving to Sprint 1 as foundation dependency
- **Acceptance:** Unit tests for fee curve at 10 prices (0.05–0.95), match Polymarket docs. Fee-free markets return 0.

### PM-006b: Database Schema Migration (infrastructure subtask)
- [ ] New Alembic migration for Polymarket schema
- [ ] Tables: markets, signals, paper_trades, paper_positions, paper_snapshots, whale_wallets, order_flow
- [ ] Drop/archive Matchbook-specific tables (events, event_mappings, odds_snapshots)
- [ ] `tests/test_db.py`

### Sprint 1 — CEO Actions (Blockers)
- [-] MetaMask wallet setup — store recovery phrase offline
- [-] Polymarket account via MetaMask login — get proxy wallet address
- [-] CEX account (Kraken recommended — supports Polygon USDC withdrawal) — KYC 1-3 days
- [-] The Odds API registration (free tier to start) — API key
- [-] Gemini API key (Google AI Studio)
- [-] Alchemy account (free tier) — Polygon RPC key

---

## Sprint 2: Value Model + Market Making + Order Flow (Week 3–4)

**Goal:** XGBoost value model, functioning MM bot in shadow mode, order flow detection, temporal crypto scanner.
**Gate:** Model Brier < 0.22, MM shadow P&L positive.

### PM-101: XGBoost Value Model
- [ ] `arbo/models/xgboost_value.py`
- [ ] Features: Pinnacle implied prob, Polymarket mid, time to event, league, historical vol, volume trend
- [ ] Target: actual outcome (1/0)
- [ ] XGBoost >=3.0 (check breaking changes from 2.x)
- [ ] Training data: The Odds API historical + Polymarket resolved markets (min 200 samples)
- [ ] Calibration: Platt scaling → reliability diagram
- [ ] Train/test: 70/30, stratified by league
- [ ] Hyperparameter tuning: Optuna, 50 trials
- [ ] `arbo/models/feature_engineering.py`, `arbo/models/calibration.py`
- [ ] `tests/test_value_model.py`
- **Acceptance:** Backtest holdout: Brier < 0.22, simulated ROI > 2% on bets with > 3% edge.

### PM-102: Value Signal Generator
- [ ] `arbo/strategies/value_betting.py`
- [ ] Every 5 min: scan all matched Pinnacle-Polymarket markets
- [ ] Calculate `edge = model_prob - polymarket_prob - estimated_fee`
- [ ] If `edge > 0.03`: generate Signal(layer=2, ...)
- [ ] All signals to paper trading engine
- [ ] Half-Kelly position sizing
- [ ] `tests/test_value_betting.py`
- **Acceptance:** In 24h generates ≥3 valid signals with edge >3%, full audit trail.

### PM-103: Market Making Bot (Shadow Mode)
- [ ] `arbo/strategies/market_maker.py`
- [ ] Target: markets with spread >4%, volume $1K-$50K, low volatility
- [ ] Logic: symmetric limit orders on both sides (BUY YES + BUY NO)
- [ ] Spread management: adjust prices by orderbook depth
- [ ] Inventory management: max imbalance 60/40
- [ ] Heartbeat: maintain connection, auto-cancel on disconnect
- [ ] Priority: fee-enabled markets (maker rebates)
- [ ] SHADOW MODE: log actions, calculate simulated P&L, DO NOT EXECUTE
- [ ] `tests/test_market_maker.py`
- **Acceptance:** 24h shadow run — simulated P&L positive, max drawdown <3%, heartbeat stable.

### PM-104: LLM Probability Agent (Gemini)
- [ ] `arbo/agents/gemini_agent.py`
- [ ] Prompt template: market question + context → probability with reasoning
- [ ] `response_mime_type="application/json"`: `{probability, confidence, reasoning}`
- [ ] Rate limiting: max 60 calls/hour
- [ ] Fallback: Claude Haiku 4.5 on Gemini outage
- [ ] Output feeds ensemble model (Layer 2, weight 0.3)
- [ ] `tests/test_gemini_agent.py`
- **Acceptance:** For 5 active markets returns valid JSON with probability [0,1], non-empty reasoning, <5s response.

### PM-105: Order Flow Monitor (Layer 7 — Basics)
- [ ] `arbo/connectors/polygon_flow.py`
- [ ] WebSocket to Polygon RPC (Alchemy)
- [ ] Parse OrderFilled events from CTF Exchange contract
- [ ] Rolling metrics: volume Z-score (1h, 4h, 24h), cumulative delta, flow imbalance
- [ ] Store in DB: order_flow table
- [ ] Signal when 2+ metrics converge
- [ ] `tests/test_polygon_flow.py`
- **Acceptance:** 1h run — parses OrderFilled events, calculates Z-scores, data in DB. Min 100 events.

### PM-106: Temporal Crypto Scanner (Layer 6 — Basics)
- [ ] `arbo/strategies/temporal_crypto.py`
- [ ] Identify 15min crypto markets on Polymarket (BTC/ETH price targets)
- [ ] Monitor spot price via public Binance WebSocket (btcusdt@ticker)
- [ ] Compare spot vs Polymarket market price
- [ ] When divergence > threshold and time to resolution <15min → signal
- [ ] PostOnly order preference
- [ ] `tests/test_temporal_crypto.py`
- **Acceptance:** Identifies ≥3 active 15min crypto markets. Spot feed stable 30min. Logs price divergence.

---

## Sprint 3: Advanced Layers + Integration (Week 5–6)

**Goal:** Whale tracking, semantic market graph, Attention Markets, esports latency, complete 9-layer system in paper trading.
**Gate:** Whale detection <10s, all 9 layers integrated.

### PM-201: Whale Wallet Discovery
- [ ] `arbo/strategies/whale_discovery.py`
- [ ] Scrape Polymarket leaderboard: top 50 wallets by ROI and volume
- [ ] Data API supplementation (/positions endpoint)
- [ ] Filters: win rate >60%, ≥50 resolved positions, volume >$50K
- [ ] DB: whale_wallets table
- [ ] Weekly refresh
- [ ] `tests/test_whale_discovery.py`
- **Acceptance:** Identifies ≥15 whale wallets with verified profitability. Data persisted.

### PM-202: Whale Position Monitor
- [ ] `arbo/strategies/whale_monitor.py`
- [ ] Polling Data API every 4s for tracked wallets
- [ ] Diff detection: new/increased/closed position
- [ ] Signal: ≥2 whales in same market = STRONG signal → Confluence +1
- [ ] Latency target: <10s from whale tx to our signal
- [ ] `tests/test_whale_monitor.py`
- **Acceptance:** Monitors 10 wallets, detects new position within 10s, generates signal with full context.

### PM-203: NegRisk Arb Monitor
- [ ] `arbo/strategies/arb_monitor.py`
- [ ] MONITORING + ALERTING ONLY, no auto-execution
- [ ] Scan NegRisk events: sum of YES prices vs $1.00
- [ ] Alert if sum < $0.97 (long arb) or sum > $1.03 (short arb)
- [ ] Log: timestamp, event, sum, potential profit, window duration
- [ ] `tests/test_arb_monitor.py`
- **Acceptance:** 7-day run generates logs. Data collection, no P&L projections.

### PM-204: Gemini News Agent (Shadow Mode)
- [ ] `arbo/agents/news_agent.py`
- [ ] Input: RSS feeds (Reuters, AP, BBC) + Google News API
- [ ] Gemini: "Is this news relevant to active markets? Impact on probability?"
- [ ] Output: NewsSignal(market_id, direction, magnitude, confidence, source_url, reasoning)
- [ ] SHADOW MODE: log signals, manual CEO review weekly
- [ ] Max 40 LLM calls/hour for news analysis
- [ ] `tests/test_news_agent.py`
- **Acceptance:** In 48h generates ≥5 news signals with non-empty reasoning. CEO manually rates quality.

### PM-205: Semantic Market Graph (Layer 3)
- [ ] `arbo/models/market_graph.py`
- [ ] Download titles/descriptions of all active markets (Gamma API)
- [ ] e5-large-v2 embeddings per market
- [ ] Chroma DB: vectors + metadata (market_id, category, price, volume)
- [ ] Similarity search: top 10 most similar per market
- [ ] Relationship classification (Gemini): SUBSET, MUTEX, IMPLICATION, TEMPORAL
- [ ] Daily refresh
- [ ] `tests/test_market_graph.py`
- **Acceptance:** Processes ≥500 markets, identifies ≥20 relationships with confidence >0.7.

### PM-206: Logical Arb Scanner (Layer 5)
- [ ] `arbo/strategies/logical_arb.py`
- [ ] Input: relationships from Layer 3 + current prices
- [ ] Gemini prompt: pricing inconsistency detection
- [ ] `response_mime_type="application/json"`: {inconsistency, direction, estimated_edge, reasoning}
- [ ] Threshold: pricing violation >3% → SIGNAL
- [ ] Scan interval: every 15 minutes
- [ ] `tests/test_logical_arb.py`
- **Acceptance:** In 24h processes ≥20 market pairs, identifies ≥1 inconsistency (or logs "none found").

### PM-207: Attention Markets Scanner (Layer 8)
- [ ] `arbo/strategies/attention_markets.py`
- [ ] Identify active Attention Markets on Polymarket (category filter)
- [ ] Gemini sentiment analysis: X/Reddit for relevant topics
- [ ] Divergence >5% → signal
- [ ] Start with pilot markets (Polymarket mindshare, Crypto Twitter mindshare)
- [ ] `tests/test_attention_markets.py`
- **Acceptance:** Identifies ≥2 active Attention Markets. Generates sentiment report with Gemini analysis.

### PM-208: Live Sports Latency Module (Layer 9)
- [ ] `arbo/strategies/sports_latency.py`
- [ ] The Odds API live scores for soccer
- [ ] Esports: Riot Games API (LoL), Steam API (CS2)
- [ ] On outcome-determining event: compare with Polymarket price
- [ ] If price hasn't reacted and P >0.95 (or <0.05) → signal
- [ ] Fee check: trade ONLY at extreme probabilities where fee <0.3%
- [ ] `tests/test_sports_latency.py`
- **Acceptance:** WebSocket stable 30min. Detects ≥1 live event update. Logs Polymarket price at detection time.

### PM-209: Confluence Scoring Engine
- [ ] `arbo/core/confluence.py`
- [ ] Receives signals from all layers (unified Signal interface)
- [ ] Calculates confluence score per market with active signals
- [ ] Rules: 0-1 → no trade, 2 → standard size, 3+ → double size
- [ ] Logging: every trade decision with signal breakdown
- [ ] Feeds paper trading engine
- [ ] `tests/test_confluence.py`
- **Acceptance:** Simulated signals from 3 layers for same market → score correctly calculated. Trade size matches rules.

### PM-210: Dashboard + Reporting
- [ ] `arbo/dashboard/cli_dashboard.py` — terminal view of all 9 layers
- [ ] `arbo/dashboard/report_generator.py` — weekly/daily CSV + text reports
- [ ] `arbo/dashboard/slack_bot.py` — adapt existing for Polymarket data
  - [ ] High-priority alerts (confluence ≥3)
  - [ ] Daily P&L summary
  - [ ] Emergency shutdown notifications
- [ ] `tests/test_dashboard.py`
- **Acceptance:** Dashboard shows real-time data from all layers. CSV export works. Slack sends test message.

---

## Sprint 4: Paper Trading Validation (Week 7–10, 4 weeks)

**Goal:** All strategies running in paper mode. Minimum 4 weeks data for go-live decision.
**Gate:** 4 consecutive weeks paper P&L data, CEO personal approval.

> NON-NEGOTIABLE: 4 weeks paper trading before ANY live execution.
> No strategy goes live without 4 consecutive weeks of paper trading data.

### PM-301: Full System Paper Run
- [ ] `main.py` orchestrator: start all 9 layers as asyncio tasks
- [ ] Every trade logged: timestamp, strategy (layer), market_id, side, price, size, edge, confluence_score
- [ ] Automatic resolution tracking
- [ ] Health monitoring: process watchdog, auto-restart on crash
- **Acceptance:** System runs 7 days without manual intervention. Uptime >95%. Trades from ≥4 different layers.

### PM-302: Weekly Report Generator
- [ ] Automatic report format per brief Section 4 (PM-302)
- [ ] Portfolio: starting/ending balance, weekly P&L, cumulative P&L
- [ ] Per strategy: trades, win rate, P&L
- [ ] Risk metrics: max drawdown, Sharpe ratio, max single loss
- [ ] Top/Bottom 5 trades
- [ ] Confluence analysis: avg score of winning vs losing trades
- **Acceptance:** Report generated automatically, all sections present, numbers match raw DB data.

### PM-303: Bug Fixes & Optimization
- [ ] Model retraining if Brier score worsens >10%
- [ ] Fee model validation against real Polymarket data
- [ ] Strategy parameter tuning based on paper trading results
- **Acceptance:** All known issues fixed or documented. Model Brier < 0.22 on paper trading data.

### Weekly Review Metrics (Minimum for Go-Live)

| Metric | Minimum | Ideal |
|--------|---------|-------|
| Paper P&L | Positive 3 of 4 weeks | Positive all 4 |
| Sharpe (annualized) | > 1.0 | > 2.0 |
| Max drawdown | < 15% | < 8% |
| Win rate (value bets) | > 52% | > 55% |
| Confluence win rate | > 60% | > 70% |
| News agent quality | CEO approval > 60% | > 75% |
| System uptime | > 95% | > 99% |
| Whale detection latency | < 15s | < 5s |
| Active layers | ≥ 5 of 9 | All 9 |

---

## Cross-Cutting Concerns (Ongoing)

- [ ] All error handling per brief
- [ ] Secret masking in all logs (NEVER log private keys)
- [ ] Health monitoring (uptime, memory, disk)
- [ ] Database partitioning for high-volume tables (after first month)
- [ ] scripts/backup.sh — daily pg_dump
- [ ] WebSocket heartbeat stability (PING every 10s)
- [ ] Polymarket builder tier application (email builder@polymarket.com for Verified tier)
