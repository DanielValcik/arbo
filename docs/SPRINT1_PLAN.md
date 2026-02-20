# Sprint 1 — Implementation Plan

**Goal:** System connects to Matchbook, fetches live odds, stores them in PostgreSQL. Runs unattended.
**Timeline:** Week 1–2 (10 working days)
**Status:** COMPLETE (2026-02-20)

---

## Phase 1A: Project Skeleton (Day 1)

### 1.1 Initialize Repository
- [x] Create directory structure per Section 9
- [ ] `git init` + `.gitignore` (Python, .env, __pycache__, .mypy_cache, models/*.joblib, *.egg-info)
- [ ] `.env.example` with all keys empty
- [ ] `pyproject.toml` with dependencies from Section 12 + tool configs
- [ ] `.pre-commit-config.yaml` (black, ruff, mypy)
- [ ] `README.md` — minimal setup instructions

### 1.2 VPS Deployment (No Docker)
- [x] `scripts/setup_vps.sh`: PostgreSQL 16, Redis 7, Python 3.12 on VPS
- [x] `scripts/deploy.sh`: rsync-based deploy to VPS
- [x] `arbo.service`: systemd unit file

### 1.3 Core Utilities (no external deps)
- [ ] `src/__init__.py` — empty
- [ ] `src/utils/__init__.py` — empty
- [ ] All other `__init__.py` files across src/ tree

---

## Phase 1B: Configuration System (Day 1–2)

### 1.4 Logger (`src/utils/logger.py`)
- [ ] structlog JSON config
- [ ] Console renderer for dev, JSON for prod
- [ ] Context binding: event_id, strategy, module, edge, stake
- [ ] Secret masking processor (password, token, api_key patterns)
- [ ] `get_logger(module_name)` factory function

### 1.5 Config (`src/utils/config.py`)
- [ ] Pydantic Settings model covering all settings.yaml sections
- [ ] Load .env first (via python-dotenv)
- [ ] Load settings.yaml as base
- [ ] Load settings.{MODE}.yaml as overlay (deep merge)
- [ ] Typed config: `MatchbookConfig`, `PollingConfig`, `ThresholdsConfig`, `RiskConfig`, `LLMConfig`
- [ ] `get_config() → ArboConfig` singleton factory
- [ ] Validation: bankroll > 0, polling intervals > 0, etc.

### 1.6 Config Files
- [ ] `config/settings.yaml` — full config from Section 5.2
- [ ] `config/settings.paper.yaml` — paper mode overrides (empty initially)
- [ ] `config/settings.live.yaml` — live mode overrides (empty initially)

---

## Phase 1C: Database Layer (Day 2–3)

### 1.7 Database Engine (`src/utils/db.py`)
- [ ] `create_async_engine()` with `DATABASE_URL` from config
- [ ] `async_sessionmaker` factory
- [ ] `get_session()` async context manager
- [ ] SQLAlchemy 2.0 declarative models for ALL 7 tables:
  - `Event`, `EventMapping`, `OddsSnapshot`, `Opportunity`, `Bet`, `DailyPnl`, `NewsItem`
- [ ] Model mixins: `TimestampMixin` (created_at, updated_at)
- [ ] All indexes as specified in Section 4
- [ ] UNIQUE constraints as specified

### 1.8 Alembic Setup
- [ ] `alembic init alembic/`
- [ ] Configure `alembic.ini` — point to DATABASE_URL
- [ ] Configure `alembic/env.py` — async support with asyncpg
- [ ] First migration: `001_create_all_tables.py`
  - All 7 tables with exact schema from Section 4
  - All indexes
  - All constraints
- [ ] Verify: `alembic upgrade head` creates all tables correctly

---

## Phase 1D: Matchbook Client (Day 3–6)

### 1.9 Base Exchange Client (`src/exchanges/base.py`)
- [ ] `BaseExchangeClient` ABC with methods:
  - `async login() → None`
  - `async get_events(sport: str, date_range: tuple) → list[Event]`
  - `async get_markets(event_id: int) → list[Market]`
  - `async get_prices(event_id: int, market_id: int) → list[Price]`
  - `async place_bet(params: BetParams) → BetResult`
  - `async cancel_bet(bet_id: str) → bool`
  - `async get_open_bets() → list[Bet]`
- [ ] Pydantic models: `Event`, `Market`, `Price`, `Runner`, `BetParams`, `BetResult`

### 1.10 Matchbook Client (`src/exchanges/matchbook.py`)
- [ ] `MatchbookClient(BaseExchangeClient)` implementation
- [ ] **Session management:**
  - `login()`: POST to `/bpapi/rest/security/session`
  - Store session-token in Redis with TTL=18000s
  - On 401 → asyncio.Lock → re-auth → retry once
  - Session check before each request
- [ ] **Token bucket rate limiter:**
  - Max 10 req/s (configurable)
  - Use `asyncio.Semaphore` or custom token bucket
  - Track daily GET count for cost monitoring
- [ ] **Event fetching:**
  - `GET /events?sport-ids={id}&after={date}&before={date}`
  - Parse to internal Event model
  - Map Matchbook sport IDs to our sport names
- [ ] **Market fetching:**
  - `GET /events/{eid}/markets`
  - Filter for configured market types (h2h, spreads, totals)
- [ ] **Price fetching:**
  - `GET /events/{eid}/markets/{mid}/runners`
  - Use `include-prices=true` to embed prices (saves GETs)
  - Parse back/lay odds + available liquidity
- [ ] **HTTP layer:**
  - aiohttp ClientSession with connection pooling
  - Configurable timeout (10s)
  - Max retries (3) with exponential backoff
  - Proper error classes: `MatchbookAuthError`, `MatchbookRateLimitError`, `MatchbookAPIError`
- [ ] **Matchbook sport ID mapping:**
  - Football → sport-id for Matchbook
  - Basketball → sport-id for Matchbook
  - (Need to discover these via `GET /lookups/sports` or docs)

### 1.11 Matchbook Tests (`tests/exchanges/test_matchbook.py`)
- [ ] Mock HTTP with `aioresponses`
- [ ] Test: successful login stores session token
- [ ] Test: 401 triggers re-auth and retry
- [ ] Test: concurrent 401s only trigger one re-auth (Lock)
- [ ] Test: rate limiter throttles requests
- [ ] Test: get_events returns parsed Event models
- [ ] Test: get_prices with include-prices returns Price models
- [ ] Test: connection timeout triggers retry
- [ ] Test: max retries exceeded raises MatchbookAPIError

---

## Phase 1E: Data Pipeline (Day 5–7)

### 1.12 Odds Snapshot Writer
- [ ] `src/data/snapshot_writer.py` (or in matchbook.py)
- [ ] Batch INSERT into odds_snapshots
- [ ] Use SQLAlchemy bulk_insert_mappings for performance
- [ ] Fields: event_id, source='matchbook', market_type, selection, back_odds, lay_odds, back_stake, lay_stake
- [ ] Handle: duplicate event creation (upsert on UNIQUE constraint)
- [ ] Event upsert: INSERT ... ON CONFLICT (source, external_id) DO UPDATE

### 1.13 Event Sync
- [ ] When Matchbook returns events → upsert into `events` table
- [ ] Map Matchbook event data to our schema:
  - external_id = Matchbook event ID
  - source = 'matchbook'
  - sport, league, home_team, away_team, start_time, status

---

## Phase 1F: Main Loop + Deployment (Day 7–10)

### 1.14 Main Entry Point (`src/main.py`)
- [ ] asyncio event loop setup
- [ ] Signal handlers (SIGINT, SIGTERM → graceful shutdown)
- [ ] Startup sequence:
  1. Load config
  2. Initialize logger
  3. Connect to PostgreSQL
  4. Connect to Redis
  5. Run Alembic migrations (or verify schema)
  6. Login to Matchbook
  7. Start polling loop
- [ ] Polling loop:
  - Fetch events for configured sports/leagues
  - For each event: fetch markets → fetch prices
  - Write odds snapshots to database
  - Sleep for configured interval (8s default)
- [ ] Graceful shutdown: close aiohttp session, close DB connections, close Redis
- [ ] `--fetch-test` CLI flag: fetch once, print to stdout, exit
- [ ] KILL_SWITCH global flag (for future /kill command)
- [ ] Error handling per Section 8:
  - Catch-all at top level
  - Log traceback
  - Sleep 60s
  - Resume
  - 3x same exception in 10min → auto-kill

### 1.15 Systemd Service
- [ ] `arbo.service` file:
  - User=arbo
  - WorkingDirectory=/opt/arbo
  - ExecStart=/opt/arbo/.venv/bin/python -m arbo
  - Restart=on-failure
  - RestartSec=10
- [ ] `scripts/deploy.sh`:
  - SSH to VPS
  - git pull
  - pip install
  - alembic upgrade head
  - systemctl restart arbo

### 1.16 Final Verification
- [ ] `mypy --strict src/` passes
- [ ] `ruff check src/` passes
- [ ] `black --check src/` passes
- [ ] `pytest tests/` all green
- [ ] `python -m src.main --fetch-test` returns EPL events with prices
- [ ] After 1 hour: >100 rows in odds_snapshots

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Matchbook sport IDs unknown | GET /lookups/sports on first login, cache in Redis |
| Matchbook returns unexpected data format | Pydantic strict validation, log and skip malformed data |
| Redis connection issues during auth | Fallback: store session token in memory if Redis unavailable |
| PostgreSQL slow on bulk inserts | Use batch inserts, consider COPY for large batches |
| Rate limit exceeded (10-min block) | Conservative token bucket, monitor GET count |

---

## Dependencies on CEO (Manual Tasks)

- [ ] Provision Hetzner CX22 VPS with Ubuntu 24.04
- [ ] Register Matchbook account from CZ IP, KYC, upgrade to Trader Plan
- [ ] Register BetInAsia account, complete KYC, fund €100
- [ ] Create Slack workspace + app (Socket Mode, slash commands)
- [ ] Set up Google Cloud billing for Gemini API (Sprint 4, but good to do early)
