# ARBO — Project Intelligence File

> This file is the authoritative reference for all development on Arbo.
> Updated: 2026-02-20

## What Is Arbo

AI-powered sports betting intelligence system. Detects arbitrage, value bets, and situational edge opportunities across Matchbook exchange and 20+ bookmakers. Single-user, runs on VPS, interfaces via Slack + PostgreSQL.

## Authoritative Documents

- `arbo_devbrief_v4_final.md` — THE source of truth. Every technical detail lives here.
- `arbo_cto_handoff_memo.md` — Explains reasoning behind v3→v4 changes.
- Any deviation from the brief requires CEO approval.

## Architecture Overview

```
Layer 1: Data Ingestion (Matchbook poller, Odds API, News RSS/Reddit, BetInAsia Playwright)
    ↓
Layer 2: Intelligence Agents (Arb Scanner P1, Value Model P2, Situational Edge P3)
    ↓
Layer 3: Decision Engine (Signal Aggregator → Risk Manager → Kelly Calculator)
    ↓
Layer 4: Execution (Matchbook auto-execute, Slack manual alerts)
    ↓
Layer 5: Monitoring (Slack bot commands, daily P&L, weekly reports)
```

## Critical Technical Facts

### Matchbook API (VERIFIED 2026-02-20)
- Auth: `POST https://api.matchbook.com/bpapi/rest/security/session` (NOT /edge/rest/)
- All data: `GET https://api.matchbook.com/edge/rest/events/...`
- Offers: `POST/GET/DELETE https://api.matchbook.com/edge/rest/v2/offers` (v2!)
- Session TTL: ~6 hours (Redis TTL=18000s)
- Commission: 4% net win from CZ
- API cost: £100 per 1M GET. Target <500K/month via tiered polling.
- Re-auth on 401 with asyncio.Lock, retry original request once

### LLM Strategy
- Primary: Gemini 2.0 Flash ($0.10/$0.40 per MTok) — `google-genai` SDK
- Fallback: Claude Haiku 4.5 ($1/$5 per MTok) — `anthropic` SDK
- EEA restriction: Google free tier NOT available. Paid from day 1. ~€0.30/month.
- Shadow mode in Sprint 4: logs signals, does NOT auto-execute until CEO validates

### News Sources (NO NewsAPI.org)
- RSS: BBC Sport, ESPN, Sky Sports, Google News RSS (all free, realtime)
- Reddit: asyncpraw (requires OAuth app approval, non-commercial)
- GNews API: optional backup (100 req/day free)

## Non-Negotiable Rules (HARDCODED in source)

These are Python constants, NOT configurable:
- MAX_BET_PCT = 0.05 (5% bankroll) — kelly.py
- KELLY_FRACTION = 0.5 (half-Kelly) — kelly.py
- MIN_BET_EUR = 5.0 — kelly.py
- DAILY_LOSS_PCT = 0.10 (10%) — risk.py
- WEEKLY_LOSS_PCT = 0.20 (20%) — risk.py
- MAX_CONCURRENT = 3 — risk.py
- MAX_PER_EVENT = 1 — risk.py
- MAX_SPORT_EXPOSURE_PCT = 0.40 — risk.py
- MIN_EDGE = 0.02 (2%) — risk.py
- MIN_PAPER_WEEKS = 4 — risk.py
- LLM_CONF_THRESHOLD = 0.7 — situational_agent.py
- LLM_MAG_THRESHOLD = 5 — situational_agent.py

## Tech Stack

- Python 3.12+, async everywhere (aiohttp, asyncpg, slack-bolt)
- PostgreSQL 16 + Redis 7
- SQLAlchemy 2.0 async + Alembic migrations
- Pydantic v2 for all DTOs
- structlog for JSON logging (no print())
- Slack: slack-bolt 1.27+ (AsyncApp + Socket Mode, no public URL needed)
- XGBoost 3.x + scikit-learn for ML
- Playwright for BetInAsia scraping
- No Docker — direct VPS (systemd + PostgreSQL + Redis)

### Slack Bot (REPLACES Telegram)
- Framework: slack-bolt AsyncApp + AsyncSocketModeHandler
- Connection: Socket Mode (outbound WebSocket, no public URL/webhook needed)
- Tokens: SLACK_BOT_TOKEN (xoxb-), SLACK_APP_TOKEN (xapp-)
- Scopes: commands, chat:write, chat:write.public, channels:read
- Formatting: Block Kit for rich alert messages
- Commands: /status, /pnl, /kill, /paper, /live (Slack Slash Commands)
- Runs alongside main asyncio loop via asyncio.create_task()
- Cost: free (Slack free plan, single workspace)

## Coding Standards

- Type hints on every function
- `black` (line length 100), `ruff`, `mypy --strict`
- `pre-commit` hooks for all three
- 100% test coverage: kelly.py, risk.py, odds.py
- `pytest` + `pytest-asyncio` (>=1.0, asyncio_mode="auto")
- `aioresponses` for HTTP mocking, `factory_boy` for test data
- Conventional commits: feat:, fix:, test:, docs:, refactor:
- Feature branches + PR + squash merge

## Sprint Timeline

| Sprint | Weeks | Focus |
|--------|-------|-------|
| 1 | 1–2 | Infrastructure + Matchbook Client |
| 2 | 3–4 | Arb Scanner + Slack Bot + Paper Trading |
| 3 | 5–7 | Value Model + BetInAsia Monitor |
| 4 | 8–10 | Auto-Execution + LLM Agent + Risk Management |
| 5 | 11–13 | Polymarket + Optimization + Live |

## Current Sprint: 2 (Arb Scanner + Slack Bot + Paper Trading)

Sprint 1 COMPLETE. Implemented:
- Project structure, config system, structured logging
- All 7 database tables + Alembic migration
- Matchbook REST client (auth, events, markets, prices, betting, rate limiter)
- Main polling loop with error handling + kill switch
- systemd service + VPS deploy scripts
- 16/16 tests passing, ruff + black clean

Sprint 2 Focus:
1. The Odds API v4 client
2. Fuzzy event matching (rapidfuzz)
3. Arb scanner
4. Slack bot (slash commands + alerts)
5. Paper trading

DO NOT START: LLM, BetInAsia, Value Model, Polymarket

## Config Loading Priority

```
.env → settings.yaml → settings.{MODE}.yaml (overlay)
```

## Error Handling Rules

See Section 8 of devbrief. Every failure mode has defined behavior:
- Matchbook unreachable: exponential backoff 1s→5min, Slack alert after 3 failures
- 401: re-auth with Lock, retry once
- Partial fills: >60% accept, <60% cancel all
- PostgreSQL down: queue critical writes in Redis (max 1000), drop odds_snapshots
- Redis down: fallback to in-memory dict, half rate limit
- Unhandled exception: log, sleep 60s, resume. 3x same in 10min → auto /kill
