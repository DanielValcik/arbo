# ARBO — Development Brief v4.0 FINAL

**AI-Powered Sports Betting Intelligence System**

| | |
|---|---|
| **From** | CEO/CTO |
| **To** | Development Team |
| **Date** | 20 February 2026 |
| **Version** | 4.0 — incorporates dev team API review + CEO cost optimization |
| **Classification** | CONFIDENTIAL |

> This document is the single source of truth for building Arbo. Every technical claim has been verified from primary sources. Build exactly what is specified. Any deviation requires CEO approval.

### Changelog v3→v4

| Change | Detail |
|---|---|
| Matchbook auth endpoint | Fixed to `POST /bpapi/rest/security/session` |
| Matchbook offers endpoint | Fixed to `POST /edge/rest/v2/offers` (v2 everywhere) |
| Matchbook session TTL | Corrected: ~6 hours (not 15 min) |
| Matchbook API pricing | Corrected: £100 per 1M GET (not free) + added cost optimization task |
| LLM: Gemini 2.0 Flash | Replaced Claude Haiku as primary ($0.10/$0.40 vs $1/$5). 90%+ cost reduction. |
| LLM: EEA restriction | Google free tier unavailable in EEA — paid tier from day 1. Still ~€0.30/mo. |
| LLM: Shadow mode | Situational agent logs signals but does NOT auto-execute. CEO reviews first. |
| NewsAPI removed | Free tier unusable in production. Replaced with RSS + Reddit + Google News RSS. |
| Reddit API | Noted: requires OAuth app approval for non-commercial use |
| Polymarket fees | Corrected: 0–1.56% (not flat 2%) |
| pytest-asyncio | Updated: >=1.0 (breaking changes from 0.x) |
| XGBoost | Updated: >=3.0 (major version jump) |
| Dependencies | Added: `google-genai>=1.0` |

---

## Table of Contents

0. [Context: Why This Product](#0-context-why-this-product)
1. [What We Are Building](#1-what-we-are-building)
2. [Platform Access](#2-platform-access-all-verified-from-cz)
3. [System Architecture](#3-system-architecture)
4. [Database Schema](#4-database-schema)
5. [Environment & Configuration](#5-environment--configuration)
6. [Development Sprints](#6-development-sprints)
7. [Non-Negotiable Rules](#7-non-negotiable-rules-hardcoded-not-configurable)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Project Structure](#9-project-structure)
10. [Coding Standards](#10-coding-standards)
11. [Key API Reference](#11-key-api-reference)
12. [Dependencies](#12-dependencies)
13. [Monthly Budget](#13-monthly-budget)
14. [Communication Protocol](#14-communication-protocol)

---

## 0. Context: Why This Product

> You don't need this section to build the system. It's here so you understand the business logic behind technical decisions.

**The opportunity:** Sports exchange odds lag behind real-world information by 5–30 minutes. When a key player gets injured, when lineups are announced, when weather changes — there is a window where the odds haven't adjusted. We use AI to detect these windows and bet during them. When no informational edge exists, we detect pure mathematical arbitrage between exchanges and bookmakers.

**Why sports, not crypto or forex:** We evaluated Polymarket (crypto prediction markets), forex algo trading, crypto grid bots, and matched betting. Sports exchange arbitrage + AI gives the best risk-adjusted ROI for our capital size (€2–5K), location (Czech Republic), and tech skills. Full strategy analysis available in a separate document.

**Why Matchbook, not Betfair:** From Czech Republic, Betfair is blocked. Smarkets is blocked. Pinnacle is blocked. Our accessible exchanges are Matchbook (4% commission, full REST API) and BetInAsia (0% on bookies, web-only for now). This is actually an advantage — less competition from sophisticated bots that focus on Betfair.

---

## 1. What We Are Building

Arbo is an automated system that:

1. **Continuously monitors** odds across Matchbook exchange and 20+ bookmakers
2. **Detects** three types of opportunities:
   - **Pure arbitrage** — mathematical price discrepancy between exchange and bookmaker
   - **Value bets** — our ML model says the true probability differs from the market price
   - **Situational edge** — LLM detects news (injury, lineup, weather) that the market hasn't priced in yet
3. **Calculates** optimal stake sizes using Kelly criterion with hard risk limits
4. **Executes** bets automatically on Matchbook via API
5. **Alerts** the owner via Telegram for manual bets on BetInAsia bookmakers
6. **Reports** daily P&L, weekly performance, and system health

**What Arbo is NOT:**
- Not a consumer product, not a SaaS, not a mobile app
- Not a latency play — our edge is intelligence, not speed
- The only interfaces are Telegram bot + PostgreSQL for analytics
- Single-user system running on a VPS

---

## 2. Platform Access (All Verified from CZ)

| Platform | CZ Status | Commission | API | Phase | Role |
|---|---|---|---|---|---|
| Matchbook | ✅ accessible | 4% net win | REST (£100 per 1M GET) | P1 | Primary automated execution |
| BetInAsia BLACK | ✅ accessible | 0% on bookies | Web only (no API) | P1 | Best odds comparison |
| AsianConnect | ✅ accessible | 1.25% exchange | Limited | P2 | Lower-commission Matchbook access |
| Polymarket | ✅ accessible | 0–1.56% winner fee | py-clob-client SDK | P3 | Cross-domain arb supplement |
| BetInAsia API | ✅ accessible | 2% + €600 fee | Mollybet REST/WS | P4 | Full multi-bookie auto (future) |
| Betfair | ❌ blocked from CZ | — | — | — | — |
| Smarkets | ❌ blocked from CZ | — | — | — | — |
| Pinnacle | ❌ blocked from CZ | — | — | — | — |

> ⚠️ **BetInAsia API** requires €600 non-refundable connection fee + min £50K/month turnover. Do NOT integrate until Phase 4. Use their free web platform only in Phase 1–3.

**Verification sources:**
- Matchbook CZ commission 4%: tradematesports.com, goalprofits.com
- BetInAsia restricted list (CZ not on it): BookieBroker.com, T&Cs Lone Rock Holdings N.V.
- BetInAsia API: betinasia.zendesk.com — €600 + £50K/mo min turnover
- AsianConnect restricted USA/France/Singapore only: Punter2Pro, TheArbAcademy
- Pinnacle CZ blocked: arbusers.com, pinnacleoddsdropper.com
- Matchbook API pricing: developers.matchbook.com/docs/pricing
- Polymarket CZ accessible: docs.polymarket.com/FAQ/geoblocking

---

## 3. System Architecture

### 3.1 Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.12+ | Async ecosystem, ML libraries, rapid dev |
| Async HTTP | aiohttp | Non-blocking API calls, connection pooling |
| Database | PostgreSQL 16 | ACID, time-series queries, JSONB columns |
| Cache | Redis 7 | Rate limiting, session tokens, real-time odds cache |
| Data validation | Pydantic v2 | Strict type checking, JSON serialization |
| Logging | structlog | Structured JSON logs, context binding |
| Web scraping | Playwright | BetInAsia headless browser automation |
| ML | XGBoost + scikit-learn | Calibrated probability models |
| LLM | Gemini 2.0 Flash (primary) / Claude Haiku 4.5 (fallback) | Structured news extraction. Gemini = 10× cheaper. |
| Telegram | aiogram 3.x | Async Telegram bot framework |
| ORM | SQLAlchemy 2.0 (async) | Type-safe database access with asyncpg |
| Migrations | Alembic | Database schema versioning |
| Deployment | Docker Compose (dev), systemd (prod) | |

### 3.2 Hosting

Hetzner CX22, Falkenstein (Germany). 2 vCPU, 4GB RAM, 40GB SSD. €3.99/month. ~20ms to London (Matchbook servers).

### 3.3 Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│               LAYER 1: DATA INGESTION                │
│                                                      │
│  ┌───────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │
│  │ Matchbook │ │ Odds API │ │  News  │ │BetInAsia │ │
│  │  Poller   │ │  Poller  │ │ Aggreg │ │ Monitor  │ │
│  │  (5-10s)  │ │  (batch) │ │ (5min) │ │(60s/evt) │ │
│  └─────┬─────┘ └────┬─────┘ └───┬────┘ └────┬─────┘ │
│        │            │           │            │       │
│        ▼            ▼           ▼            ▼       │
│  ┌──────────────────────────────────────────────┐    │
│  │   Redis (live odds cache) + PostgreSQL       │    │
│  │   (odds_snapshots, events, news_items)       │    │
│  └───────────────────────┬──────────────────────┘    │
└──────────────────────────┼───────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────┐
│          LAYER 2: INTELLIGENCE AGENTS                │
│                           │                          │
│  ┌───────────┐  ┌────────┴────┐  ┌────────────────┐ │
│  │    Arb    │  │   Value     │  │  Situational   │ │
│  │  Scanner  │  │  Bet Model  │  │  Edge Agent    │ │
│  │   (P1)    │  │   (P2)      │  │   (P3)         │ │
│  └─────┬─────┘  └──────┬─────┘  └───────┬────────┘ │
│        │               │                │           │
│        ▼               ▼                ▼           │
│  ┌──────────────────────────────────────────────┐   │
│  │      Signal Aggregator (weighted ensemble)    │   │
│  └───────────────────────┬──────────────────────┘   │
└──────────────────────────┼──────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────┐
│            LAYER 3: DECISION ENGINE                  │
│                           │                          │
│  ┌───────────────────────▼─────────────────────┐    │
│  │  Risk Manager → Kelly Calculator → Execute?  │    │
│  └───────────────────────┬─────────────────────┘    │
└──────────────────────────┼──────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────┐
│              LAYER 4: EXECUTION                      │
│                     ┌────┴────┐                      │
│               ┌─────▼───┐ ┌──▼────────┐             │
│               │Matchbook│ │ Telegram  │             │
│               │Executor │ │ Alert     │             │
│               │ (auto)  │ │ (manual)  │             │
│               └─────────┘ └───────────┘             │
└─────────────────────────────────────────────────────┘
```

### 3.4 Layer Details

**Layer 1 — Data Ingestion (always running)**

| Poller | Source | Interval | Notes |
|---|---|---|---|
| Matchbook Poller | Matchbook REST API | 5–10s | Session-based auth. Store every price update to `odds_snapshots`. |
| Odds API Poller | The Odds API (20+ bookmakers) | Batch (budget quota) | Free: 500 req/mo. Track quota via `x-requests-remaining` header. |
| News Aggregator | RSS (BBC, ESPN, Sky, Google News) + Reddit | 5 min | Dedup by SHA-256. Reddit: requires OAuth (non-commercial). Google News RSS: free, no API key, no limits. **No NewsAPI.org in production.** |
| BetInAsia Monitor | Playwright headless browser | 60s per active event | Scrape BLACK dashboard. Extract best-odds per bookmaker → structured JSON. |

**Layer 2 — Intelligence Agents**

| Agent | Phase | Logic | Trigger |
|---|---|---|---|
| Arb Scanner | P1 | Compare Matchbook lay (adjusted for 4% commission) vs. best bookmaker back. Arb exists when combined implied probability < 1.0 | Edge > 2% after commission |
| Value Bet Model | P2 | XGBoost calibrated probability vs. best market odds | Model_prob − market_prob > 5% |
| Situational Edge | P3 | LLM (Gemini 2.0 Flash primary, Claude Haiku fallback) analyzes news for upcoming events. Structured JSON output: `{event_id, impact, magnitude: 1-10, direction, confidence: 0-1}` | Confidence > 0.7 AND magnitude > 5 |

**Layer 3 — Decision Engine**

1. **Signal Aggregator:** Weighted score = `(arb * W₁) + (value * W₂) + (situational * W₃)`. Phase 1: arb only (W₁=1.0, rest=0.0). Weights evolve based on live performance.
2. **Risk Manager:** Before every execution, checks all limits (see Section 7). Blocks if any breached.
3. **Kelly Calculator:** `stake = (edge / (odds − 1)) × bankroll × 0.5` (half-Kelly). Hard cap 5% bankroll. Floor €5.

**Layer 4 — Execution**

1. **Matchbook Executor:** `POST /edge/rest/v2/offers` → poll status → cancel unmatched after 30s → handle partial fills (accept >60%, cancel <60%) → log to `bets` table.
2. **BetInAsia Alert (manual):** Telegram message with exact instructions: `⚡ [EVENT] | [BOOKIE] | [SELECTION] | Odds [X.XX] | Stake €[XX] | Edge [X.X%] | Place within 2 min`

**Layer 5 — Monitoring**

| Channel | Content |
|---|---|
| Telegram (real-time) | Every bet placed, daily P&L at 23:00 UTC, system errors, risk limit breaches |
| Telegram commands | `/status`, `/pnl`, `/kill`, `/paper`, `/live` |
| Weekly report (P3+) | ROI by strategy/sport/league, model calibration, system uptime |

---

## 4. Database Schema

All timestamps UTC. All monetary values EUR. Use SQLAlchemy 2.0 async models with Alembic migrations.

```sql
-- ================================================================
-- EVENTS: Central event registry, normalized across all sources
-- ================================================================
CREATE TABLE events (
    id              SERIAL PRIMARY KEY,
    external_id     VARCHAR(64) NOT NULL,         -- Source platform's event ID
    source          VARCHAR(32) NOT NULL,          -- 'matchbook' | 'odds_api' | 'betinasia'
    sport           VARCHAR(32) NOT NULL,          -- 'football' | 'basketball' | 'tennis'
    league          VARCHAR(128),                  -- 'EPL' | 'Champions League' | 'NBA'
    home_team       VARCHAR(128) NOT NULL,
    away_team       VARCHAR(128) NOT NULL,
    start_time      TIMESTAMPTZ NOT NULL,
    status          VARCHAR(16) DEFAULT 'upcoming', -- upcoming | live | settled | cancelled
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, external_id)
);
CREATE INDEX idx_events_start ON events(start_time);
CREATE INDEX idx_events_sport_status ON events(sport, status);

-- ================================================================
-- EVENT_MAPPINGS: Links same real-world event across platforms
-- e.g., Matchbook event 123 = Odds API event "abc-def"
-- ================================================================
CREATE TABLE event_mappings (
    id              SERIAL PRIMARY KEY,
    canonical_id    INT NOT NULL,                  -- Internal "master" event ID (usually Matchbook)
    mapped_id       INT NOT NULL REFERENCES events(id),
    match_score     DECIMAL(4,3) NOT NULL,         -- Fuzzy match confidence 0.000–1.000
    matched_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(canonical_id, mapped_id)
);

-- ================================================================
-- ODDS_SNAPSHOTS: Time-series of every price update
-- Most write-heavy table. Expect millions of rows per month.
-- ================================================================
CREATE TABLE odds_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    event_id        INT NOT NULL REFERENCES events(id),
    source          VARCHAR(32) NOT NULL,          -- 'matchbook' | 'bet365' | 'unibet_via_bia'
    market_type     VARCHAR(32) NOT NULL,          -- 'h2h' | 'spreads' | 'totals'
    selection       VARCHAR(64) NOT NULL,          -- 'home' | 'away' | 'draw' | 'over_2.5'
    back_odds       DECIMAL(8,4),                  -- Best back price available
    lay_odds        DECIMAL(8,4),                  -- Best lay price (Matchbook only)
    back_stake      DECIMAL(12,2),                 -- Liquidity at back price
    lay_stake       DECIMAL(12,2),                 -- Liquidity at lay price
    bookmaker       VARCHAR(64),                   -- e.g. 'bet365', 'unibet' (NULL for exchange)
    captured_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_odds_event_time ON odds_snapshots(event_id, captured_at DESC);
CREATE INDEX idx_odds_source_market ON odds_snapshots(source, market_type);
-- Recommend: PARTITION BY RANGE (captured_at) monthly after first month

-- ================================================================
-- OPPORTUNITIES: Every detected arb/value/situational signal
-- ================================================================
CREATE TABLE opportunities (
    id              SERIAL PRIMARY KEY,
    event_id        INT NOT NULL REFERENCES events(id),
    strategy        VARCHAR(32) NOT NULL,          -- 'arb' | 'value' | 'situational'
    expected_edge   DECIMAL(6,4) NOT NULL,         -- e.g. 0.0350 = 3.5%
    details         JSONB NOT NULL,
    -- arb:         {"matchbook_lay": 2.10, "bookie_back": 2.15, "bookie": "bet365", "margin": 0.035}
    -- value:       {"model_prob": 0.58, "market_prob": 0.52, "best_odds": 1.92, "bookmaker": "unibet"}
    -- situational: {"impact": "injury", "detail": "Haaland out", "magnitude": 8, "confidence": 0.85}
    status          VARCHAR(16) DEFAULT 'detected',-- detected | executed | expired | skipped
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    expired_at      TIMESTAMPTZ
);
CREATE INDEX idx_opps_status ON opportunities(status, detected_at DESC);

-- ================================================================
-- BETS: Every bet placed (paper and live)
-- ================================================================
CREATE TABLE bets (
    id              SERIAL PRIMARY KEY,
    opportunity_id  INT REFERENCES opportunities(id),
    event_id        INT NOT NULL REFERENCES events(id),
    strategy        VARCHAR(32) NOT NULL,          -- 'arb' | 'value' | 'situational'
    platform        VARCHAR(32) NOT NULL,          -- 'matchbook' | 'betinasia_manual' | 'polymarket'
    external_bet_id VARCHAR(128),                  -- Platform's bet/offer ID (NULL for paper)
    side            VARCHAR(8) NOT NULL,           -- 'back' | 'lay'
    selection       VARCHAR(64) NOT NULL,          -- 'home' | 'away' | 'draw'
    odds            DECIMAL(8,4) NOT NULL,
    stake           DECIMAL(10,2) NOT NULL,        -- EUR
    potential_pnl   DECIMAL(10,2),                 -- Expected P&L if wins
    actual_pnl      DECIMAL(10,2),                 -- Actual P&L after settlement
    commission_paid DECIMAL(8,2) DEFAULT 0,
    edge_at_exec    DECIMAL(6,4),                  -- Edge at moment of execution
    fill_pct        DECIMAL(4,3) DEFAULT 1.000,    -- 1.000=fully matched, 0.600=60%
    status          VARCHAR(16) DEFAULT 'pending', -- pending | matched | partial | settled | cancelled
    is_paper        BOOLEAN NOT NULL DEFAULT TRUE,
    placed_at       TIMESTAMPTZ DEFAULT NOW(),
    matched_at      TIMESTAMPTZ,
    settled_at      TIMESTAMPTZ,
    notes           TEXT
);
CREATE INDEX idx_bets_status ON bets(status, placed_at DESC);
CREATE INDEX idx_bets_event ON bets(event_id);
CREATE INDEX idx_bets_paper ON bets(is_paper, status);

-- ================================================================
-- DAILY_PNL: Aggregated daily performance
-- ================================================================
CREATE TABLE daily_pnl (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    num_opportunities INT DEFAULT 0,
    num_bets        INT DEFAULT 0,
    num_wins        INT DEFAULT 0,
    total_staked    DECIMAL(12,2) DEFAULT 0,
    gross_pnl       DECIMAL(10,2) DEFAULT 0,
    total_commission DECIMAL(8,2) DEFAULT 0,
    net_pnl         DECIMAL(10,2) DEFAULT 0,
    bankroll_start  DECIMAL(12,2) NOT NULL,
    bankroll_end    DECIMAL(12,2) NOT NULL,
    roi_pct         DECIMAL(6,4),                  -- net_pnl / bankroll_start
    notes           TEXT
);

-- ================================================================
-- NEWS_ITEMS: Raw news and Reddit posts for LLM analysis
-- ================================================================
CREATE TABLE news_items (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(32) NOT NULL,          -- 'rss_bbc' | 'reddit_soccer' | 'gnews' | 'google_news'
    title           VARCHAR(512) NOT NULL,
    body            TEXT,
    url             VARCHAR(1024),
    content_hash    VARCHAR(64) NOT NULL UNIQUE,   -- SHA-256 for deduplication
    published_at    TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ DEFAULT NOW(),
    analyzed        BOOLEAN DEFAULT FALSE,
    analysis_result JSONB                          -- LLM structured output
);
CREATE INDEX idx_news_pending ON news_items(analyzed, fetched_at DESC);
```

---

## 5. Environment & Configuration

### 5.1 Environment Variables (.env)

```env
# ========== Platform Credentials ==========
MATCHBOOK_USERNAME=
MATCHBOOK_PASSWORD=
BETINASIA_USERNAME=
BETINASIA_PASSWORD=
ODDS_API_KEY=
# ========== LLM (Gemini primary, Claude fallback) ==========
GOOGLE_AI_API_KEY=
ANTHROPIC_API_KEY=
GNEWS_API_KEY=                    # Optional: gnews.io free tier backup
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=arbo/1.0

# ========== Telegram ==========
TELEGRAM_BOT_TOKEN=
TELEGRAM_OWNER_CHAT_ID=

# ========== Database ==========
DATABASE_URL=postgresql+asyncpg://arbo:password@localhost:5432/arbo
REDIS_URL=redis://localhost:6379/0

# ========== Runtime ==========
MODE=paper
LOG_LEVEL=INFO
```

> **CRITICAL:** `.env` is in `.gitignore`. Never committed. Provide `.env.example` with empty values.

### 5.2 YAML Config (config/settings.yaml)

```yaml
bankroll: 2000.00
currency: EUR

# --- Target sports/leagues (only fetch these) ---
sports:
  football:
    leagues: ["EPL", "Champions League", "La Liga", "Bundesliga", "Serie A"]
    markets: ["h2h", "spreads", "totals"]
  basketball:
    leagues: ["NBA", "Euroleague"]
    markets: ["h2h", "spreads", "totals"]

# --- Polling intervals (seconds) ---
polling:
  matchbook: 8
  odds_api_batch: 300          # 5 min (conserve free tier quota)
  news: 300                    # 5 min
  betinasia: 60                # per active event

# --- Agent weights (Phase 1: arb-only) ---
weights:
  arb: 1.0
  value: 0.0                  # Enable in Phase 2
  situational: 0.0            # Enable in Phase 3

# --- LLM provider ---
llm:
  provider: gemini                 # 'gemini' | 'claude'
  gemini_model: gemini-2.0-flash
  claude_model: claude-haiku-4-5-20251001
  max_calls_per_day: 100

# --- Thresholds ---
thresholds:
  min_edge: 0.02              # 2% minimum
  arb_margin: 0.04            # Matchbook 4% commission
  value_gap: 0.05             # 5% model vs market
  llm_confidence: 0.7
  llm_magnitude: 5

# --- Matchbook connection ---
matchbook:
  base_url: "https://api.matchbook.com/edge/rest"
  commission_pct: 0.04
  session_ttl_seconds: 18000    # ~6 hours — refresh proactively before expiry
  max_retries: 3
  timeout_seconds: 10
  unmatched_cancel_after: 30   # seconds
  min_fill_pct: 0.60          # Accept partial fill >60%

# --- Risk limits: DOCUMENTATION ONLY ---
# Actual limits hardcoded in src/engine/risk.py
# Cannot be overridden by config. See Section 7.
risk:
  max_bet_pct: 0.05
  daily_loss_pct: 0.10
  weekly_loss_pct: 0.20
  max_concurrent_bets: 3
  max_per_event: 1
  max_sport_exposure_pct: 0.40
```

### 5.3 Config Loading

```
Priority: .env → settings.yaml → settings.{MODE}.yaml (overlay)
```

Override files `settings.paper.yaml` and `settings.live.yaml` can override any value from the base. Implement with Pydantic Settings: load base YAML, merge mode-specific YAML on top.

---

## 6. Development Sprints

### Sprint 1 (Week 1–2): Infrastructure + Matchbook Client

> **Goal:** System connects to Matchbook, fetches live odds, stores them in PostgreSQL. Runs unattended on VPS.

**Tasks:**

1. Provision Hetzner CX22: Ubuntu 24.04, Python 3.12, PostgreSQL 16, Redis 7
2. Create repo `arbo/` with project structure (Section 9). Init git, `.gitignore`, `.env.example`, `pyproject.toml`
3. Set up Docker Compose: `postgres:16`, `redis:7`, app service with volume mounts
4. `src/utils/db.py`: async SQLAlchemy engine factory + session maker using asyncpg
5. Alembic init + first migration: create all tables from Section 4
6. `src/utils/logger.py`: structlog config — JSON output, bound context (event_id, strategy, module)
7. `src/utils/config.py`: Pydantic Settings model, load YAML + env overlay
8. `src/exchanges/base.py`: `BaseExchangeClient` ABC with methods:
   - `async login() → None`
   - `async get_events(sport, date_range) → list[Event]`
   - `async get_markets(event_id) → list[Market]`
   - `async get_prices(event_id, market_id) → list[Price]`
   - `async place_bet(params) → BetResult`
   - `async cancel_bet(bet_id) → bool`
   - `async get_open_bets() → list[Bet]`
9. `src/exchanges/matchbook.py`: full implementation:
   - `login()`: `POST /bpapi/rest/security/session` → store session-token in Redis with TTL=18000s (~6h)
   - **Session management:** on any HTTP 401 → re-authenticate → retry original request once. Use `asyncio.Lock` around refresh to prevent concurrent re-auth from multiple coroutines
   - `get_events()`: `GET /events?sport-ids={id}&after={date}&before={date}`
   - `get_markets()`: `GET /events/{id}/markets`
   - `get_prices()`: `GET /events/{id}/markets/{id}/runners` → parse back/lay prices + available liquidity
   - Token bucket rate limiter: max 10 req/s, configurable per client
10. Odds snapshot writer: every price fetch → batch INSERT into `odds_snapshots`
11. `src/main.py`: asyncio event loop running Matchbook poller at configured interval
12. systemd service file `arbo.service` for unattended VPS operation + auto-restart on failure
13. `scripts/deploy.sh`: SSH to VPS, git pull, docker-compose up, restart systemd
14. **Manual (CEO):** Register Matchbook account from CZ IP, verify KYC, upgrade to Trader Plan
15. **Manual (CEO):** Register BetInAsia account, complete KYC, fund €100

**Acceptance Tests:**

| # | Test | Pass criteria |
|---|---|---|
| 1 | `pytest tests/exchanges/test_matchbook.py` | All pass (mocked HTTP via `aioresponses`) |
| 2 | `mypy --strict src/` | Zero errors |
| 3 | `ruff check src/` | Zero warnings |
| 4 | Live VPS: `python -m arbo --fetch-test` | Returns ≥10 EPL events with back/lay prices to stdout |
| 5 | After 1 hour running: `SELECT COUNT(*) FROM odds_snapshots WHERE captured_at > NOW() - INTERVAL '1 hour'` | >100 rows |
| 6 | After 24 hours unattended | No crashes in `journalctl -u arbo --since "24 hours ago"`, no OOM |
| 7 | Docker Compose `docker compose up` on fresh machine | All services start, migrations run, system begins polling |

---

### Sprint 2 (Week 3–4): Arb Scanner + Telegram + Paper Trading

> **Goal:** System detects arbitrage opportunities, sends Telegram alerts, logs paper trades with theoretical P&L.

**Tasks:**

1. `src/data/odds_api.py`: fetch odds from The Odds API
   - `GET /sports/{sport}/odds?apiKey={key}&regions=eu&markets=h2h,spreads,totals&oddsFormat=decimal`
   - Parse response → normalize to internal Pydantic models
   - Track remaining API quota from response header `x-requests-remaining`
   - Log warning when remaining < 100
2. `src/utils/odds.py`: odds conversion functions
   - `decimal_to_implied(odds: float) → float`
   - `implied_to_decimal(prob: float) → float`
   - `american_to_decimal(odds: int) → float`
   - `fractional_to_decimal(num: int, den: int) → float`
   - `adjust_for_commission(implied_prob: float, commission: float) → float` — adjusts Matchbook lay for 4% net win commission
3. `src/data/event_matcher.py`:
   - Fuzzy match events between Matchbook and Odds API by team names + start time
   - Use `rapidfuzz` library (Levenshtein distance)
   - Maintain `config/team_aliases.yaml`:
     ```yaml
     "Man Utd": "Manchester United"
     "Man City": "Manchester City"
     "Spurs": "Tottenham Hotspur"
     # ... top 50 teams per configured league
     ```
   - Match threshold: fuzzy score ≥ 85 AND start_time within ±30 minutes
   - Write matches to `event_mappings` table with confidence score
   - Log any match scoring <90 for manual review
4. `src/agents/arb_scanner.py`:
   - For each event with both Matchbook and bookmaker prices:
     ```python
     matchbook_lay_implied = 1 / lay_odds  # adjusted for commission
     effective_lay_implied = matchbook_lay_implied + (commission_pct * matchbook_lay_implied)
     bookie_back_implied = 1 / back_odds
     combined = effective_lay_implied + bookie_back_implied
     if combined < 1.0:
         edge = 1.0 - combined
         if edge > config.thresholds.min_edge:
             # This is a profitable arb
     ```
   - Also scan: Matchbook back on selection A vs Matchbook lay on selection B (same-exchange arb)
   - Write to `opportunities` table with full details in JSONB
5. Paper trading engine (`src/execution/position_tracker.py`):
   - When opportunity detected → calculate Kelly stake → create `bets` record with `is_paper=TRUE`
   - Don't actually execute anything
   - Settle paper bets: after event finishes, fetch result via Odds API or Matchbook → calculate actual_pnl
   - Daily aggregate: cron task at 23:00 UTC writes to `daily_pnl`
6. `src/alerts/telegram_bot.py`:
   - aiogram 3.x async framework
   - Commands:
     - `/status` → uptime, mode, open bets count, today's P&L, last opportunity time
     - `/pnl` → today / this week / this month / all-time as text table
     - `/kill` → set `KILL_SWITCH` flag → main loop cancels open bets, stops pollers → confirms
     - `/paper` → switch to paper mode
     - `/live` → switch to live (only if 4+ weeks paper positive)
   - Auto-alerts:
     - Every opportunity: `🔍 ARB | Liverpool vs Arsenal | Matchbook lay 2.10 / bet365 back 2.15 | Edge 3.5% | Stake €32`
     - Daily P&L at 23:00 UTC
     - Any system error or risk limit breach
7. Configure target leagues in `settings.yaml`: EPL, Champions League, La Liga, Bundesliga, Serie A, NBA
8. **Matchbook API cost optimization** (critical — API charges £100/1M GET):
   - Implement tiered polling: events starting >24h → poll every 60s; events starting <6h → poll every 8s; live events → poll every 5s
   - Batch requests: use `include=prices` parameter to embed runner prices in market response (1 GET instead of 3)
   - Cache events and markets in Redis (they change rarely) — only poll runners/prices at high frequency
   - Log daily GET count to `daily_pnl.notes` — **target: <500K GET/month (<£50)**

**Acceptance Tests:**

| # | Test | Pass criteria |
|---|---|---|
| 1 | Unit: Matchbook lay 2.10 + bet365 back 2.15 with 4% commission | Scanner correctly computes edge, flags as arb |
| 2 | Unit: Matchbook lay 1.80 + bookie back 1.75 | Scanner correctly identifies NO arb |
| 3 | Unit: `adjust_for_commission(0.5, 0.04)` | Returns correct adjusted probability |
| 4 | Manually verify 20 matched events from event_matcher | ≥18 correct (90% accuracy) |
| 5 | After 7 days running: `SELECT COUNT(*) FROM opportunities WHERE strategy='arb'` | ≥5 rows |
| 6 | Telegram: send `/status` on VPS | Response within 2 seconds |
| 7 | Telegram: arb opportunity auto-alert | Arrives within 5 seconds of detection |
| 8 | After 14 days: `daily_pnl` table | 14 rows, all calculations manually verified correct |

---

### Sprint 3 (Week 5–7): Value Model + BetInAsia Monitor

> **Goal:** ML model predicts calibrated probabilities. BetInAsia web scraper adds multi-bookie odds coverage.

**Tasks:**

1. `scripts/collect_historical.py`: download historical match data:
   - football-data.org: free CSV files, ~20 years, top European leagues
   - api-football.com: structured JSON, 2+ seasons
   - Store in PostgreSQL table `historical_matches`
2. `src/agents/value_model.py` — feature engineering:
   | Feature | Calculation |
   |---|---|
   | `team_form_5g` | Points from last 5 games (W=3, D=1, L=0) / 15 |
   | `h2h_last5` | Head-to-head record last 5 meetings |
   | `home_win_pct` | Home team's home win rate this season |
   | `goals_scored_rolling` | Avg goals scored, last 5 games |
   | `goals_conceded_rolling` | Avg goals conceded, last 5 games |
   | `days_since_last_match` | Rest days for each team |
   | `league_position` | Normalized: 1st=1.0, last=0.0 |
   | `elo_rating` | Computed from historical results |
3. XGBoost model training:
   - Train on 2020–2024 seasons
   - Validation on 2024–2025 season
   - Target: home_win / draw / away_win (multi-class)
   - **Optimize for calibration (Brier score), NOT accuracy**
   - If raw XGBoost poorly calibrated → apply `CalibratedClassifierCV` (Platt scaling)
   - Save model to `models/value_model_v1.joblib` (track with git-lfs)
4. Backtesting framework:
   - Replay 2024–2025 season against historical closing odds
   - Apply Kelly staking where `model_prob − market_prob > 5%`
   - Calculate: cumulative ROI, max drawdown, Sharpe ratio
   - Generate calibration plot (predicted vs observed probability)
   - **Must show >3% ROI across full test set with ≥200 simulated bets**
5. `src/data/betinasia_monitor.py`:
   - Playwright async: launch headless Chromium
   - Login to BetInAsia BLACK with stored credentials
   - Navigate to configured events → extract odds table (bookmaker, selection, odds)
   - Parse → write to `odds_snapshots` with `source='betinasia_{bookmaker}'`
   - Handle: session expiry (re-login), page load timeout (retry), missing elements (log + skip)
   - Rate: max 1 event scrape per 60 seconds
6. Value bet detector: flag opportunity when `model_prob − best_market_prob > 5%` after commission adjustment
7. **Manual (CEO):** Register AsianConnect account, verify KYC, test Matchbook access at 1.25% commission

**Acceptance Tests:**

| # | Test | Pass criteria |
|---|---|---|
| 1 | Model Brier score on 2024–2025 test set | < 0.05 |
| 2 | Backtest ROI over full 2024–2025 season | > 3% (≥200 simulated bets) |
| 3 | Calibration plot visual check | Predicted probabilities track observed frequencies |
| 4 | No NaN values in feature output | All matches in test set produce valid features |
| 5 | BetInAsia monitor: live EPL match | Extracts odds from ≥5 bookmakers |
| 6 | BetInAsia monitor: 24-hour soak test | No Playwright crash (or auto-recovers per Section 8) |

---

### Sprint 4 (Week 8–10): Auto-Execution + LLM Agent + Risk Management

> **Goal:** System places real bets on Matchbook. LLM analyzes news in **shadow mode** (logs signals, does NOT auto-execute — CEO reviews via Telegram). Full risk management. 4-week paper validation begins.

**Tasks:**

1. `src/execution/matchbook_executor.py`:
   - `place_bet(side, event_id, market_id, runner_id, odds, stake)`:
     - `POST /edge/rest/v2/offers` with `{odds, stake, side, runner-id}`
     - Poll `GET /edge/rest/v2/offers/{id}` every 2s for up to 30s
     - Status `matched` → record in `bets`, Telegram confirmation
     - Status `unmatched` after 30s → `DELETE /edge/rest/v2/offers/{id}`, record as cancelled
     - Partial match >60% → accept, cancel remainder, record actual `fill_pct`
     - Partial match <60% → cancel all, record as cancelled
   - `cancel_all_open()`: `GET /bets?status=open` → `DELETE` each. Used by `/kill`
2. `src/engine/risk.py` — all limits as **Python constants, not config**:
   ```python
   # These are HARDCODED. Do NOT read from config or env.
   MAX_BET_PCT = 0.05
   DAILY_LOSS_PCT = 0.10
   WEEKLY_LOSS_PCT = 0.20
   MAX_CONCURRENT = 3
   MAX_PER_EVENT = 1
   MAX_SPORT_EXPOSURE_PCT = 0.40
   MIN_EDGE = 0.02
   MIN_PAPER_WEEKS = 4
   ```
   - `check_can_execute(bet_params, state) → (bool, reason_if_blocked)`
   - On daily limit breach → pause execution, Telegram alert, continue monitoring
   - On weekly limit breach → switch to paper mode, require `/live` to resume
3. `src/engine/kelly.py` — **hardcoded constants**:
   ```python
   KELLY_FRACTION = 0.5    # Half-Kelly
   MAX_BET_PCT = 0.05      # 5% cap
   MIN_BET_EUR = 5.0       # €5 floor
   ```
   - `calculate_stake(edge, odds, bankroll) → float`
   - Formula: `stake = (edge / (odds - 1)) * bankroll * KELLY_FRACTION`
   - Then: `min(stake, bankroll * MAX_BET_PCT)`, then `max(stake, MIN_BET_EUR)`
4. `src/data/news_aggregator.py`:
   - RSS: `feedparser` library. Feed URLs configured in `settings.yaml`:
     - BBC Sport, ESPN, Sky Sports, The Guardian Sport
     - Google News RSS: `https://news.google.com/rss/search?q={team}+{sport}&hl=en`
     - Team-specific RSS feeds (club official sites)
   - Reddit: `asyncpraw`. Subreddits: r/soccer, r/sportsbook + sport-specific subs. **Requires OAuth app approval (non-commercial use).**
   - GNews API (optional backup): `https://gnews.io/api/v4/search?q={query}&token={key}` — free 100 req/day
   - **No NewsAPI.org in production** — free tier is localhost-only with 24h delay
   - Dedup: SHA-256 of `lowercase(title + first_200_chars_body)`. Skip if hash exists in `news_items`
   - Store all in `news_items` with `analyzed=FALSE`
5. `src/agents/situational_agent.py`:
   - Implement `BaseLLMAgent` ABC with `analyze(news_items) → list[Signal]`
   - Implement `GeminiAgent(BaseLLMAgent)` using `google-genai` SDK:
     - Model: `gemini-2.0-flash` ($0.10/$0.40 per MTok paid tier)
     - **EEA restriction: free tier NOT available for EEA users. Use paid tier from day 1.**
     - Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`
     - Use `response_mime_type: "application/json"` for guaranteed valid JSON output
   - Implement `ClaudeAgent(BaseLLMAgent)` using `anthropic` SDK:
     - Model: `claude-haiku-4-5-20251001` ($1/$5 per MTok — 10× more expensive)
   - Config `llm.provider` in settings.yaml switches between them
   - **Both agents get identical system prompt and expected JSON schema**
   - For each upcoming event (starts within 3 hours):
     - Query unanalyzed `news_items` matching event team names
     - Batch max 5 items per LLM call
     - Call configured LLM (Gemini or Claude) with system prompt:
       ```
       System: You are a sports intelligence analyst. Analyze these news items
       about {home_team} vs {away_team}. Output ONLY valid JSON array.
       Each item: {event_id, impact, detail, magnitude (1-10), direction, confidence (0-1)}
       If no significant impact, return empty array [].
       ```
     - Parse response with Pydantic validation
     - If `confidence > 0.7 AND magnitude > 5` → write to `opportunities`
     - Mark items `analyzed=TRUE`
   - On invalid JSON from LLM → log raw response, mark items as `analyzed=FALSE` (will retry next cycle), skip batch
   - Daily cap: max 100 LLM calls (configurable in settings.yaml)
6. `src/engine/decision.py`:
   - Combine all agent signals for a given event
   - Apply configured weights
   - If weighted score passes threshold → Kelly → Risk check → Execute (or paper-log)
7. **4 weeks of paper trading with full system running:**
   - Arb + value agents active and feeding into paper execution
   - **LLM agent in SHADOW MODE:** logs signals to `opportunities` table + Telegram alert, but signal weight W₃ = 0.0 (does NOT influence execution). CEO manually reviews LLM signals for accuracy.
   - After 2–4 weeks shadow: if LLM signals correlate with real odds movements, CEO approves setting W₃ > 0 via config
   - Stakes calculated, opportunities logged, nothing executed on exchange
   - Daily review via `/pnl`
   - **Live execution ONLY after 4 consecutive weeks of positive paper P&L**

**Acceptance Tests:**

| # | Test | Pass criteria |
|---|---|---|
| 1 | Unit: Kelly(edge=0.05, odds=2.0, bankroll=2000) | Returns €50 (half-Kelly) |
| 2 | Unit: Kelly(edge=0.20, odds=1.5, bankroll=2000) | Returns €100 (capped at 5%) |
| 3 | Unit: Risk manager blocks when daily loss 10% exceeded | Returns (False, "daily_loss_limit") |
| 4 | Unit: Risk manager blocks 4th concurrent bet | Returns (False, "max_concurrent") |
| 5 | Unit: Risk manager switches to paper on weekly loss 20% | Mode changes, Telegram alert sent |
| 6 | Integration: executor places + cancels €5 test bet on Matchbook | Offer created + deleted, no funds lost |
| 7 | LLM agent: feed known injury headline | Produces valid JSON, confidence > 0.7 |
| 8 | LLM agent: feed irrelevant news | Produces empty array [] |
| 9 | Deliberately trigger all 5 risk limits in paper mode | Each triggers correct behavior per Section 8 |
| 10 | `/kill` command | All open paper bets cancelled within 3 seconds, system stopped |
| 11 | 4 weeks paper trading | Positive cumulative P&L |

---

### Sprint 5 (Week 11–13): Polymarket + Optimization + Live

> **Goal:** Polymarket integrated. Live execution begins (if paper validation passed). Performance analytics.

**Tasks:**

1. `src/exchanges/polymarket.py`:
   - `pip install py-clob-client`
   - Connect with Polygon wallet private key
   - Fetch sports-related markets, monitor prices
   - Paper order placement first
2. Cross-domain arb scanner: compare Polymarket contracts (e.g., "Will Liverpool win EPL?") vs. accumulated match odds from Matchbook/bookmakers
3. Polymarket-specific LLM prompt: detect mispriced contracts vs. breaking news
4. Performance analytics queries:
   - ROI by strategy / sport / league / day-of-week / time-of-day
   - Identify best and worst performing segments
   - Generate weekly summary (can be manual SQL for now, automated report in future)
5. Monthly model retraining pipeline:
   - Retrain XGBoost with new data including our own bet outcomes
   - Compare Brier score vs baseline → alert if degraded
6. **Go live on Matchbook** (if Sprint 4 paper validation passed):
   - Start with minimum stakes (€5–€10 per bet)
   - Gradually increase as confidence builds
   - First week: CEO reviews every bet before next cycle

**Acceptance Tests:**

| # | Test | Pass criteria |
|---|---|---|
| 1 | Polymarket: fetch ≥10 sports-related markets with prices | Data flows into system |
| 2 | Cross-domain scanner: find ≥1 discrepancy per week | Logged to opportunities |
| 3 | Performance analytics: ROI query matches manual `bets` table audit | Calculations verified |
| 4 | Live execution on Matchbook (if approved) | ≥5 live bets placed in first week, all logged correctly |
| 5 | System ROI (paper or live) | >5% monthly on bankroll |

---

## 7. Non-Negotiable Rules (Hardcoded, Not Configurable)

These are **Python constants in source code**. They CANNOT be overridden by environment variables, config files, or runtime flags. Changing any of these requires a code change, PR review, and explicit CEO approval.

| Rule | Constant | File | Value |
|---|---|---|---|
| Max bet size | `MAX_BET_PCT` | `src/engine/kelly.py` | 0.05 (5% of bankroll) |
| Kelly fraction | `KELLY_FRACTION` | `src/engine/kelly.py` | 0.5 (half-Kelly) |
| Min bet | `MIN_BET_EUR` | `src/engine/kelly.py` | 5.0 |
| Daily loss limit | `DAILY_LOSS_PCT` | `src/engine/risk.py` | 0.10 (10%) |
| Weekly loss limit | `WEEKLY_LOSS_PCT` | `src/engine/risk.py` | 0.20 (20%) |
| Max concurrent bets | `MAX_CONCURRENT` | `src/engine/risk.py` | 3 |
| Max bets per event | `MAX_PER_EVENT` | `src/engine/risk.py` | 1 |
| Max sport exposure | `MAX_SPORT_EXPOSURE_PCT` | `src/engine/risk.py` | 0.40 (40%) |
| Min edge to execute | `MIN_EDGE` | `src/engine/risk.py` | 0.02 (2%) |
| LLM confidence threshold | `LLM_CONF_THRESHOLD` | `src/agents/situational_agent.py` | 0.7 |
| LLM magnitude threshold | `LLM_MAG_THRESHOLD` | `src/agents/situational_agent.py` | 5 |
| Paper trading requirement | `MIN_PAPER_WEEKS` | `src/engine/risk.py` | 4 |

---

## 8. Error Handling & Recovery

Every failure mode has a defined behavior. Do not improvise — implement exactly as specified.

| Failure | Behavior | Recovery | Alert |
|---|---|---|---|
| Matchbook API unreachable | Exponential backoff: 1s → 2s → 4s → ... → max 5min | Auto-resume when reachable | Telegram after 3 consecutive failures |
| Matchbook 401 (session expired) | Re-authenticate (`POST /bpapi/rest/security/session`). Use `asyncio.Lock` — one re-auth at a time. Retry original request once. | Automatic | Log warning only. Alert if re-auth itself fails. |
| Matchbook partial fill <60% | Cancel entire order. Record as cancelled. | None needed (normal) | Log only, no alert |
| Matchbook partial fill ≥60% | Accept filled portion. Cancel remainder. Record actual `fill_pct`. | None needed | Log the partial fill details |
| Playwright crash (BetInAsia) | Kill browser process. Launch fresh Chromium instance. | Max 3 restarts/hour. After 3rd: stop BetInAsia monitoring until manual check. | Telegram after 3rd restart |
| PostgreSQL connection lost | Queue critical writes (bets, opportunities) in Redis list (max 1000 items). Non-critical data (odds_snapshots) silently dropped. Flush Redis queue on reconnect. | Auto-reconnect with backoff | Telegram immediately |
| Redis connection lost | Fall back to in-memory dict for odds cache. Rate limiter switches to conservative mode (half rate). | Auto-reconnect | Telegram immediately |
| LLM API failure (Gemini or Claude) | Skip situational analysis for this cycle. Arb + value agents continue independently. | Auto-retry next cycle | Telegram after 5 consecutive failures |
| LLM returns invalid JSON | Log raw response. Mark news items as `analyzed=FALSE` (will retry next cycle). | Automatic retry | Log warning. Alert if >10 failures/day. |
| Odds API quota exhausted | Stop polling Odds API. Continue with Matchbook + BetInAsia only. | Wait for monthly reset | Telegram with remaining quota info |
| VPS reboot | systemd auto-starts `arbo.service`. On startup: reconcile open bets with Matchbook API, recalculate today's P&L, log startup event. | Automatic | Telegram: "Arbo restarted at [time]. Open bets: [N]. Today P&L: [€X]" |
| `/kill` command | Set global `KILL_SWITCH=True`. Main loop exits before next cycle. Cancel all open Matchbook orders. Stop all pollers. | Stays stopped until manual `systemctl start arbo` | Telegram: "Arbo stopped by /kill at [time]" |
| Unhandled exception | Catch at `main.py` top level. Log full traceback with structlog. Sleep 60s. Resume main loop. If same exception type occurs 3× in 10 min → auto-trigger `/kill`. | Auto-resume (or auto-kill if repeated) | Telegram: error message + traceback summary |

---

## 9. Project Structure

```
arbo/
├── config/
│   ├── settings.yaml              # Primary config (all defaults)
│   ├── settings.paper.yaml        # Paper mode overrides
│   ├── settings.live.yaml         # Live mode overrides
│   └── team_aliases.yaml          # Team name mappings for fuzzy matching
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point: asyncio event loop, signal handlers
│   ├── exchanges/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseExchangeClient ABC
│   │   ├── matchbook.py           # Matchbook REST API client
│   │   └── polymarket.py          # Polymarket py-clob-client (Phase 3)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── arb_scanner.py         # Cross-platform arbitrage detection
│   │   ├── value_model.py         # XGBoost calibrated probability model
│   │   └── situational_agent.py   # LLM news analysis (Gemini primary, Claude fallback)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── decision.py            # Signal aggregation + weighted ensemble
│   │   ├── kelly.py               # Half-Kelly calculator (HARDCODED limits)
│   │   └── risk.py                # Risk manager (HARDCODED limits)
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── matchbook_executor.py  # Matchbook order placement + lifecycle
│   │   └── position_tracker.py    # Open positions, paper P&L, settlement
│   ├── data/
│   │   ├── __init__.py
│   │   ├── odds_api.py            # The Odds API client
│   │   ├── news_aggregator.py     # RSS + Reddit + Google News pipeline
│   │   ├── betinasia_monitor.py   # Playwright headless BetInAsia scraper
│   │   └── event_matcher.py       # Fuzzy match events across platforms
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── telegram_bot.py        # aiogram 3.x async bot
│   └── utils/
│       ├── __init__.py
│       ├── odds.py                # Odds format conversions
│       ├── logger.py              # structlog JSON config
│       ├── db.py                  # SQLAlchemy 2.0 async engine + models
│       └── config.py              # Pydantic Settings YAML loader
├── tests/
│   ├── conftest.py                # Shared fixtures: test db, mock HTTP, factories
│   ├── exchanges/
│   │   └── test_matchbook.py
│   ├── agents/
│   │   ├── test_arb_scanner.py
│   │   ├── test_value_model.py
│   │   └── test_situational.py
│   ├── engine/
│   │   ├── test_kelly.py          # 100% coverage REQUIRED
│   │   ├── test_risk.py           # 100% coverage REQUIRED
│   │   └── test_decision.py
│   ├── data/
│   │   ├── test_odds_api.py
│   │   └── test_event_matcher.py
│   └── utils/
│       └── test_odds.py           # 100% coverage REQUIRED
├── alembic/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/                  # Migration files
├── scripts/
│   ├── deploy.sh                  # VPS deployment
│   ├── backup.sh                  # Daily PostgreSQL pg_dump → remote storage
│   └── collect_historical.py      # One-time historical data import
├── models/                        # Trained ML models (git-lfs)
│   └── .gitkeep
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml        # black + ruff + mypy hooks
├── docker-compose.yml             # postgres:16, redis:7, app
├── Dockerfile
├── pyproject.toml
└── README.md                      # Setup instructions
```

---

## 10. Coding Standards

**Python:**
- Python 3.12+. Type hints on every function signature and return type.
- Pydantic v2 `BaseModel` for all data transfer objects.
- `async`/`await` for ALL I/O. Never use `requests` — always `aiohttp`.
- No `print()`. Use `structlog` for everything.

**Formatting & Linting:**
- `black` (line length 100)
- `ruff` for linting
- `mypy --strict`
- All three run via `pre-commit`. PR blocked if any fail.

**Testing:**
- `pytest` + `pytest-asyncio`
- 100% coverage on: `src/engine/kelly.py`, `src/engine/risk.py`, `src/utils/odds.py`
- Integration tests for exchange clients with `aioresponses`
- Use `factory_boy` for test data

**Git:**
- `main` is protected. Feature branches + PR + squash merge only.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`

**Logging:**
- Every bet, every decision, every error: structured JSON via structlog
- Bound context: `event_id`, `strategy`, `module`, `edge`, `stake`
- Secrets NEVER logged. Mask in log processors.

---

## 11. Key API Reference

### 11.1 Matchbook

**Docs:** https://developers.matchbook.com

**Base URL:** `https://api.matchbook.com/edge/rest`

**Auth flow:**
1. `POST /bpapi/rest/security/session` with `{"username": "...", "password": "..."}`
2. Response includes `session-token` value (also set as cookie)
3. Include `session-token` as HTTP header or cookie in all subsequent requests
4. Session lives for ~6 hours — only 1 login per 6h needed
5. On 401 → re-auth → retry once (with asyncio.Lock)

**Endpoints:**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/bpapi/rest/security/session` | Login | Returns session-token. ~6h lifetime. |
| GET | `/events` | List events | Params: `sport-ids`, `after`, `before` |
| GET | `/events/{eid}/markets` | Markets for event | |
| GET | `/events/{eid}/markets/{mid}/runners` | Prices | Back/lay odds + liquidity. **Use `include-prices=true` to embed prices in response (saves separate GET).** |
| POST | `/edge/rest/v2/offers` | Place bet | Body: `{odds, stake, side, runner-id}`. V2 = no live delay for unmatched. |
| GET | `/edge/rest/v2/offers/{oid}` | Check offer status | Poll for matching. V2 adds `delayed` status. |
| DELETE | `/edge/rest/v2/offers/{oid}` | Cancel unmatched | |
| GET | `/bets?status=open` | List open bets | Used by /kill |

**Rate limits:** £100 per 1M GET requests/month. WRITE requests free at reasonable frequency. Fee auto-deducted from Matchbook account at month end.

### 11.2 The Odds API

**Docs:** https://the-odds-api.com/liveapi/guides/v4/

**Base URL:** `https://api.the-odds-api.com/v4`

**Auth:** `apiKey` query parameter

**Key endpoint:**
```
GET /sports/{sport}/odds?apiKey={key}&regions=eu&markets=h2h,spreads,totals&oddsFormat=decimal
```

Returns all bookmaker odds for a sport in one call. Response header `x-requests-remaining` tracks quota.

**Quota:** Free 500 req/month. Paid $40/mo = 10K req/month.

### 11.3 Gemini 2.0 Flash (Primary LLM)

**Docs:** https://ai.google.dev/gemini-api/docs

**SDK:** `pip install google-genai`

**Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`

**Auth:** API key as query param or header `x-goog-api-key`

**Pricing:** $0.10/M input, $0.40/M output (paid tier). **EEA/UK/CH: free tier NOT available — paid tier required from day 1.** At our volume (~50K tok/day), cost is ~€0.30/month.

**Response format:** Request `response_mime_type: "application/json"` for guaranteed JSON output.

### 11.4 Claude Haiku 4.5 (Fallback LLM)

**Docs:** https://docs.anthropic.com/en/docs/about-claude/models

**Endpoint:** `POST https://api.anthropic.com/v1/messages`

**Headers:** `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`

**Body:**
```json
{
  "model": "claude-haiku-4-5-20251001",
  "max_tokens": 500,
  "system": "You are a sports intelligence analyst...",
  "messages": [{"role": "user", "content": "Analyze these news items..."}]
}
```

**Cost:** $1/M input tokens, $5/M output tokens. With prompt caching: reads $0.10/M. Budget ~50K tokens/day ≈ €0.50–0.75/day.

### 11.4 Polymarket (Phase 3)

**SDK:** `pip install py-clob-client`

**Auth:** Polygon wallet private key → `create_or_derive_api_creds()`

**Rate limits:** 100 req/min public, 60 orders/min trading

**Fee:** Most markets: 0% fee. Specific markets: 0.44–1.56% on winning outcome. ~$0.007 gas per tx.

---

## 12. Dependencies

```toml
[project]
name = "arbo"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    # Async core
    "aiohttp>=3.9",

    # Database
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "alembic>=1.13",

    # Cache
    "redis>=5.0",

    # Data validation & config
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "pyyaml>=6.0",

    # Logging
    "structlog>=24.1",

    # Telegram
    "aiogram>=3.3",

    # Web scraping
    "playwright>=1.40",

    # ML
    "xgboost>=3.0",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "joblib>=1.3",

    # LLM (Gemini primary, Claude fallback)
    "google-genai>=1.0",
    "anthropic>=0.40",

    # News & data
    "feedparser>=6.0",
    "asyncpraw>=7.7",

    # Fuzzy matching
    "rapidfuzz>=3.5",

    # Polymarket (Phase 3)
    # "py-clob-client>=0.12",

    # Utilities
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=1.0",
    "pytest-cov>=4.1",
    "aioresponses>=0.7",
    "factory-boy>=3.3",
    "black>=24.1",
    "ruff>=0.2",
    "mypy>=1.8",
    "pre-commit>=3.6",
]
```

---

## 13. Monthly Budget

| Item | Min/mo | Max/mo | Notes |
|---|---|---|---|
| Hetzner CX22 VPS | €3.99 | €7.49 | CX32 upgrade if memory needed |
| Matchbook API fees | €0 | €60 | £100/1M GET. Target <500K with tiered polling. |
| LLM (Gemini 2.0 Flash paid) | €0.30 | €2 | $0.10/$0.40 per MTok. EEA = paid tier required. Claude Haiku fallback $1/$5. |
| The Odds API | €0 | €35 | Free tier for Sprint 1–2 |
| News sources (RSS + Reddit + GNews) | €0 | €0 | All free. Reddit requires OAuth app approval (non-commercial). |
| **TOTAL INFRASTRUCTURE** | **€4** | **€105** | |
| Trading bankroll | €2,000 | €5,000 | Separate from infrastructure |

---

## 14. Communication Protocol

**Dev → CEO (Weekly, every Monday):**
- What was delivered last week (with evidence: test results, screenshots, metrics)
- What is blocked and what decision is needed from CEO
- What is planned for this week

**CEO → Owner (Bi-weekly):**
- Sprint completion % with evidence
- Paper/live trading P&L from actual data
- Budget burn vs plan
- Decisions requiring capital approval

**Automated (Daily):**
- Telegram: every detected opportunity, every placed bet, daily P&L at 23:00 UTC, system errors

---

> **This document is final. Build exactly what is specified. Any deviation requires CEO approval.**
