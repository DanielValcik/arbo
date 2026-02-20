# ARBO — Master TODO

> Living document. Updated as work progresses.
> Status: [ ] pending, [~] in progress, [x] done, [-] blocked/waiting

---

## Sprint 1: Infrastructure + Matchbook Client (Week 1–2) ✓ COMPLETE

### Project Setup
- [x] Initialize git repository
- [x] Create full directory structure (Section 9)
- [x] Write pyproject.toml with all dependencies + tool configs
- [x] Write .gitignore
- [x] Write .env.example
- [x] Write .pre-commit-config.yaml (black, ruff, mypy)
- [x] VPS setup script (scripts/setup_vps.sh) — no Docker
- [x] Create all `__init__.py` files

### Config System
- [x] `src/utils/config.py` — Pydantic Settings with YAML loader
- [x] `config/settings.yaml` — full defaults
- [x] `config/settings.paper.yaml` — paper overrides
- [x] `config/settings.live.yaml` — live overrides

### Logging
- [x] `src/utils/logger.py` — structlog JSON config with secret masking

### Database
- [x] `src/utils/db.py` — async engine, session maker, all 7 SQLAlchemy models
- [x] `alembic/` setup — async env.py
- [x] First migration: all tables + indexes + constraints

### Matchbook Client
- [x] `src/exchanges/base.py` — BaseExchangeClient ABC + Pydantic DTOs
- [x] `src/exchanges/matchbook.py` — full REST client
  - [x] Session auth (POST /bpapi/rest/security/session)
  - [x] Session management (401 → Lock → re-auth → retry)
  - [x] Session token in Redis (TTL=18000s)
  - [x] GET /events with sport-ids filter
  - [x] GET /events/{eid}/markets
  - [x] GET /events/{eid}/markets/{mid}/runners?include-prices=true
  - [x] Token bucket rate limiter (10 req/s)
  - [x] Daily GET counter (for cost monitoring)
  - [x] Exponential backoff on failure (1s→2s→4s→...→5min)
  - [x] Error classes: MatchbookAuthError, MatchbookRateLimitError, MatchbookAPIError

### Data Pipeline
- [x] Odds snapshot writer — batch INSERT from price data
- [x] Event upsert — INSERT ON CONFLICT for events table

### Main Loop
- [x] `src/main.py` — asyncio event loop
  - [x] Startup sequence (config → logger → DB → Redis → Matchbook login → poll)
  - [x] Graceful shutdown (SIGINT/SIGTERM handlers)
  - [x] `--fetch-test` CLI flag
  - [x] Error handling per Section 8
  - [x] KILL_SWITCH global flag

### Testing
- [x] `tests/conftest.py` — shared fixtures
- [x] `tests/exchanges/test_matchbook.py` — 16 tests with aioresponses
- [x] ruff check passes
- [x] black --check passes

### Deployment
- [x] `arbo.service` systemd unit file
- [x] `scripts/deploy.sh`
- [x] `scripts/setup_vps.sh`

### CEO Manual Tasks
- [-] Provision Hetzner CX22 VPS
- [-] Register Matchbook account + KYC + Trader Plan
- [-] Register BetInAsia account + KYC + fund €100
- [-] Create Slack workspace + Slack app (Socket Mode, slash commands)

---

## Sprint 2: Arb Scanner + Slack Bot + Paper Trading (Week 3–4)

### Odds API Client
- [ ] `src/data/odds_api.py` — The Odds API v4 client
- [ ] Quota tracking via x-requests-remaining header
- [ ] Pydantic models for Odds API responses

### Odds Utilities
- [ ] `src/utils/odds.py` — all conversion functions
- [ ] `tests/utils/test_odds.py` — 100% coverage

### Event Matcher
- [ ] `src/data/event_matcher.py` — rapidfuzz fuzzy matching
- [ ] `config/team_aliases.yaml` — top 50 teams per league
- [ ] `tests/data/test_event_matcher.py`

### Arb Scanner
- [ ] `src/agents/arb_scanner.py` — arbitrage detection logic
- [ ] Cross-platform arb (Matchbook lay vs bookie back)
- [ ] Same-exchange arb (Matchbook back A vs lay B)
- [ ] `tests/agents/test_arb_scanner.py`

### Paper Trading
- [ ] `src/execution/position_tracker.py` — paper bet logging
- [ ] Paper bet settlement (after event finishes)
- [ ] Daily P&L aggregation (23:00 UTC cron)

### Slack Bot
- [ ] `src/alerts/slack_bot.py` — slack-bolt AsyncApp + Socket Mode
  - [ ] AsyncSocketModeHandler setup (runs alongside main asyncio loop)
  - [ ] Slash commands: /status, /pnl, /kill, /paper, /live
  - [ ] Proactive alerts via chat.postMessage + Block Kit formatting
  - [ ] Alert types: opportunities, daily P&L (23:00 UTC), errors, risk limit breaches
  - [ ] BetInAsia manual alert: `[EVENT] | [BOOKIE] | [SELECTION] | Odds | Stake | Edge`
- [ ] `tests/alerts/test_slack_bot.py`

### API Cost Optimization
- [ ] Tiered polling (>24h=60s, <6h=8s, live=5s)
- [ ] include-prices=true batching
- [ ] Redis caching for events/markets
- [ ] Daily GET count logging

---

## Sprint 3: Value Model + BetInAsia Monitor (Week 5–7)

### Historical Data
- [ ] `scripts/collect_historical.py` — football-data.org + api-football.com
- [ ] `historical_matches` table (or extend existing schema)

### Value Model
- [ ] `src/agents/value_model.py` — XGBoost feature engineering
- [ ] 8 features as specified in brief
- [ ] XGBoost training pipeline (2020–2024 train, 2024–2025 validate)
- [ ] CalibratedClassifierCV if needed
- [ ] Backtesting framework (ROI, drawdown, Sharpe, calibration plot)
- [ ] Must achieve: Brier < 0.05, ROI > 3% on 200+ bets

### BetInAsia Monitor
- [ ] `src/data/betinasia_monitor.py` — Playwright async scraper
- [ ] Login, navigate, extract odds table
- [ ] Session expiry handling (re-login)
- [ ] Max 1 event scrape per 60s
- [ ] Max 3 restarts/hour, then stop

### CEO Manual Task
- [-] Register AsianConnect account + KYC

---

## Sprint 4: Auto-Execution + LLM + Risk Management (Week 8–10)

### Matchbook Executor
- [ ] `src/execution/matchbook_executor.py`
- [ ] POST /edge/rest/v2/offers
- [ ] Poll status, handle partial fills (>60% accept, <60% cancel)
- [ ] Cancel after 30s unmatched
- [ ] cancel_all_open() for /kill

### Risk Manager
- [ ] `src/engine/risk.py` — HARDCODED constants
- [ ] check_can_execute() — all 6 limit checks
- [ ] `tests/engine/test_risk.py` — 100% coverage

### Kelly Calculator
- [ ] `src/engine/kelly.py` — HARDCODED constants
- [ ] calculate_stake() — half-Kelly with cap and floor
- [ ] `tests/engine/test_kelly.py` — 100% coverage

### News Aggregator
- [ ] `src/data/news_aggregator.py` — RSS + Reddit + GNews
- [ ] SHA-256 deduplication
- [ ] Configurable feeds in settings.yaml

### Situational Agent
- [ ] `src/agents/situational_agent.py`
- [ ] BaseLLMAgent ABC
- [ ] GeminiAgent implementation (google-genai SDK)
- [ ] ClaudeAgent implementation (anthropic SDK)
- [ ] Shadow mode (log only, W₃=0.0)
- [ ] Daily cap: 100 LLM calls

### Decision Engine
- [ ] `src/engine/decision.py` — weighted signal aggregation
- [ ] Risk check → Kelly → Execute pipeline

### 4-Week Paper Trading
- [ ] All agents active in paper mode
- [ ] LLM in shadow mode
- [ ] Daily /pnl review
- [ ] Must achieve: 4 consecutive weeks positive

---

## Sprint 5: Polymarket + Optimization + Live (Week 11–13)

### Polymarket
- [ ] `src/exchanges/polymarket.py` — py-clob-client
- [ ] Cross-domain arb scanner
- [ ] Polymarket-specific LLM prompt

### Analytics
- [ ] Performance analytics queries (ROI by strategy/sport/league)
- [ ] Weekly summary report

### Model Retraining
- [ ] Monthly XGBoost retraining pipeline
- [ ] Brier score comparison vs baseline

### Go Live
- [ ] Switch to live mode (if 4-week paper positive)
- [ ] Minimum stakes (€5–€10)
- [ ] CEO reviews every bet first week

---

## Cross-Cutting Concerns (Ongoing)

- [ ] All error handling per Section 8
- [ ] Secret masking in all logs
- [ ] Health monitoring (uptime, memory, disk)
- [ ] Database partitioning for odds_snapshots (after first month)
- [ ] scripts/backup.sh — daily pg_dump
