# ARBO — Project Intelligence File

> This file is the authoritative reference for all development on Arbo.
> Updated: 2026-02-21 (POLYMARKET PIVOT)

## 🔴 MANDATORY: Read LEARNINGS.md before any change

`LEARNINGS.md` is the operational memory — every bug, fix, architectural
decision, and performance observation lives there. Before **any** of the
following, open LEARNINGS.md and scan for relevant entries:

- Strategy parameter change (even "small" ones)
- Autoresearch validation of existing strategy
- Deploying new code that touches live execution
- Investigation of production anomalies
- Optimization decisions

You MUST also **update LEARNINGS.md autonomously** whenever you:
- Find a bug (add entry before/while fixing)
- Ship a fix (link commit hash)
- Observe surprising behavior (WR, fill rate, slippage, latency)
- Make architectural decisions with trade-offs

Entries are grouped per strategy (B2, B3, B3_15M, D, ...) with a global
section for system-wide patterns. Keep it fresh — stale knowledge is
worse than no knowledge.

## What Is Arbo

Automated trading system for Polymarket — a decentralized prediction market on Polygon blockchain. The system combines 9 strategic layers for generating edge through information asymmetry, statistical models, and on-chain analytics. Single-user, runs on VPS, interfaces via Slack + PostgreSQL.

## Repository

- GitHub: https://github.com/DanielValcik/arbo.git
- Branch: `main`
- Remote: `origin`

## Authoritative Documents

- `ARBO_CTO_Development_Brief_v3.md` — THE source of truth for Polymarket system. Every technical detail lives here.
- `Arbo_CTO_Handoff_Memo.md` — Historical context (v3→v4 Matchbook changes, now superseded by pivot).
- Any deviation from the brief requires CEO approval.

## Architecture Overview — 9 Layers

```
Layer 1: Market Making + Rebate Harvesting (spread capture + maker rebates)
Layer 2: Value Betting — Multi-Source Ensemble (XGBoost + Gemini + historical)
Layer 3: Semantic Market Graph (e5-large-v2 embeddings + Chroma DB)
Layer 4: Whale Copy + Multi-Signal Confluence (on-chain wallet tracking)
Layer 5: Logical/Combinatorial Arbitrage (LLM semantic analysis)
Layer 6: Temporal Crypto Arbitrage (15min crypto markets vs spot)
Layer 7: Smart Money Order Flow (Polygon on-chain OrderFilled events)
Layer 8: Attention Markets (Kaito AI × Polymarket sentiment)
Layer 9: Live Sports Data Latency (sports data feed advantage)

Signal Flow:
  Strategy → Signal → Confluence Scorer → RiskManager.pre_trade_check() → CLOB
                                              ↓ (rejected)
                                          Log + Alert
```

## Critical Technical Facts

### Polymarket API (VERIFIED 2026-02-21)
- CLOB API: `https://clob.polymarket.com` — orders, orderbook, prices
- Gamma API: `https://gamma-api.polymarket.com` — market metadata, discovery (NO AUTH)
- Data API: `https://data-api.polymarket.com` — wallet positions, trade history (NO AUTH)
- WebSocket: `wss://ws-subscriptions-clob.polymarket.com/ws/market` — live orderbook
- User WS: `wss://ws-subscriptions-clob.polymarket.com/ws/user` — order/trade updates (AUTH)
- Network: Polygon (chain_id=137), USDC.e collateral, ~$0.007 gas/tx
- SDK: py-clob-client==0.34.6 (pin exact version)
- Auth: L1 (EIP-712 private key signing) + L2 (HMAC-SHA256 apiKey/secret/passphrase)
- Order types: GTC, GTD, FOK, FAK. PostOnly for maker-only orders.
- Batch: up to 15 orders per request
- Rate limits: POST /order 3500/10s, GET /book 1500/10s, Gamma 500/10s
- Heartbeat: PING every 10s on WebSocket, auto-cancel on disconnect
- NegRisk: multi-outcome events, pass `neg_risk=True` from Gamma API `enableNegRisk` field
- Tick sizes: "0.1", "0.01", "0.001", "0.0001" — market-specific

### Polymarket Fee Model
- Most sports markets: 0% fee (February 2026)
- Fee-enabled: 15min crypto, 5min crypto, NCAAB, Serie A
- Formula: `fee = price * (1 - price) * fee_rate` (max at p=0.50)
- Maker rebates: 20-25% of taker fees redistributed to liquidity providers
- PostOnly orders guarantee maker status

### Contract Addresses (Polygon)
- CTF Exchange: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- NegRisk CTF Exchange: `0xC5d563A36AE78145C45a50134d48A1215220f80a`
- NegRisk Adapter: `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`
- Conditional Tokens: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- USDC.e: `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`

### LLM Strategy
- Primary: Gemini 2.0 Flash ($0.10/$0.40 per MTok) — `google-generativeai` SDK
- Fallback: Claude Haiku 4.5 ($1/$5 per MTok) — `anthropic` SDK
- EEA restriction: Google free tier NOT available. Paid from day 1.
- `response_mime_type="application/json"` for structured output

### The Odds API
- Endpoint: `https://api.the-odds-api.com/v4`
- Pinnacle odds via `bookmakers=pinnacle&regions=eu`
- Free tier: 500 req/month. $10 tier: 10K req/month.
- Track quota via `x-requests-remaining` header — NEVER exceed limit

## Non-Negotiable Rules (HARDCODED in source)

These are Python constants, NOT configurable. Changes require CEO approval.

```python
# arbo/core/risk_manager.py
MAX_POSITION_PCT = 0.05          # 5% capital per trade
DAILY_LOSS_PCT = 0.10            # 10% daily loss → auto shutdown
WEEKLY_LOSS_PCT = 0.20           # 20% weekly loss → shutdown + CEO escalation
WHALE_COPY_MAX_PCT = 0.025       # 2.5% per copied whale position
MAX_MARKET_TYPE_PCT = 0.30       # 30% max in one market type
MAX_CONFLUENCE_DOUBLE_PCT = 0.05 # 5% hard cap even at confluence score 5
MIN_PAPER_WEEKS = 4              # 4 weeks paper trading before ANY live execution
KELLY_FRACTION = 0.5             # Half-Kelly sizing
```

## Tech Stack

- Python 3.12+, async everywhere (aiohttp, asyncpg, slack-bolt)
- PostgreSQL 16 (NO Redis — in-memory caches + asyncio queues instead)
- SQLAlchemy 2.0 async + Alembic migrations
- Pydantic v2 for all DTOs and config
- structlog for JSON logging (no print())
- Slack: slack-bolt 1.27+ (AsyncApp + Socket Mode, no public URL needed)
- XGBoost 3.x + scikit-learn for ML
- Chroma DB + sentence-transformers for semantic graph (Layer 3)
- web3.py for Polygon on-chain events (Layer 7)
- py-clob-client for Polymarket CLOB trading
- No Docker — direct VPS (systemd + PostgreSQL)

### Slack Bot
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
- `pytest` + `pytest-asyncio` (>=1.0, asyncio_mode="auto")
- `aioresponses` for HTTP mocking, `factory_boy` for test data
- Docstrings on all public functions (Google style)
- Conventional commits: feat:, fix:, test:, docs:, refactor:
- Feature branches: `feature/PM-XXX-description`

## Project Structure

```
arbo/
├── config/
│   ├── settings.yaml          # Risk params, thresholds, intervals
│   └── settings.py            # Pydantic Settings class
├── core/
│   ├── risk_manager.py        # SINGLETON risk manager
│   ├── paper_engine.py        # Paper trading engine
│   ├── portfolio.py           # Position tracking, P&L
│   ├── confluence.py          # Multi-signal confluence scorer
│   ├── scanner.py             # Unified opportunity scanner
│   └── fee_model.py           # Polymarket fee calculations
├── connectors/
│   ├── polymarket_client.py   # CLOB wrapper
│   ├── market_discovery.py    # Gamma API market catalog
│   ├── odds_api_client.py     # The Odds API (Pinnacle)
│   ├── polygon_flow.py        # On-chain order flow (Layer 7)
│   └── websocket_manager.py   # WS connection manager with reconnect
├── strategies/
│   ├── market_maker.py        # Layer 1
│   ├── value_betting.py       # Layer 2 (signal generator)
│   ├── whale_discovery.py     # Layer 4 (discovery)
│   ├── whale_monitor.py       # Layer 4 (real-time monitoring)
│   ├── logical_arb.py         # Layer 5
│   ├── temporal_crypto.py     # Layer 6
│   ├── arb_monitor.py         # Layer 3 (NegRisk monitoring)
│   ├── attention_markets.py   # Layer 8
│   └── sports_latency.py      # Layer 9
├── models/
│   ├── xgboost_value.py       # XGBoost value model
│   ├── feature_engineering.py # Feature extraction
│   ├── calibration.py         # Platt scaling, reliability
│   └── market_graph.py        # Semantic graph (Layer 3, Chroma)
├── agents/
│   ├── gemini_agent.py        # Gemini 2.0 Flash wrapper
│   └── news_agent.py          # News sentiment agent
├── dashboard/
│   ├── cli_dashboard.py       # Terminal dashboard
│   ├── report_generator.py    # Weekly/daily reports
│   └── slack_bot.py           # Slack alert notifications
├── utils/
│   ├── config.py              # Config loading (reused from Sprint 1-2)
│   ├── db.py                  # SQLAlchemy models + engine
│   ├── logger.py              # structlog config
│   └── odds.py                # Odds conversion + Kelly math
├── tests/                     # Per-module test files
├── main.py                    # Orchestrator — starts all layers
└── __init__.py
```

## Sprint Timeline

| Sprint | Weeks | Focus | Go/No-go Gate |
|--------|-------|-------|---------------|
| 1 | 1–2 | CLOB + Odds API + Paper engine | Connectivity OK, scanner runs |
| 2 | 3–4 | Value model + MM bot + Order flow | Model Brier < 0.22, MM shadow P&L+ |
| 3 | 5–6 | Whale + Semantic graph + Advanced layers | Whale detection < 10s, 9 layers integrated |
| 4 | 7–10 | Paper trading validation (4 weeks) | 4 weeks paper P&L data, CEO approval |

## Current Sprint: 1 (Foundation — Polymarket Pivot)

### Previous Work (Matchbook era — ARCHIVED to `_archive/`)
Sprint 1-2 of Matchbook version complete. 107/107 tests. Reusable components migrated:
- structlog config, Pydantic config loader, SQLAlchemy base, odds math, Slack bot pattern
- The Odds API client (adapted for Polymarket market matching)

### Sprint 1 Tasks (PM-001 through PM-007)
- PM-001: Polymarket CLOB client wrapper
- PM-002: Market discovery module (Gamma API)
- PM-003: The Odds API integration (adapt existing)
- PM-004: Paper trading engine
- PM-005: Opportunity scanner (all layers)
- PM-006: Config & secrets management
- PM-007: Risk manager (core)

## Config Loading Priority

```
.env → config/settings.yaml → config/settings.{MODE}.yaml (overlay)
```

## Error Handling Rules

- Polymarket CLOB unreachable: exponential backoff 1s→5min, Slack alert after 3 failures
- WebSocket disconnect: reconnect with exponential backoff, heartbeat every 10s
- Rate limit (throttled): back off, reduce request frequency
- PostgreSQL down: queue critical writes in asyncio.Queue (max 1000), drop non-critical
- LLM API down: skip LLM-dependent layers (5, 8), rest of system continues
- Unhandled exception: log, sleep 60s, resume. 3x same in 10min → auto /kill
- Emergency shutdown: cancel ALL orders, log reason, notify CEO via Slack

## Confluence Scoring (Central Decision Mechanism)

| Signal | Points |
|--------|--------|
| Whale buys position | +1 |
| Value model shows edge > 5% | +1 |
| News agent detects relevant event | +1 |
| Order flow spike in market (Layer 7) | +1 |
| Logical inconsistency with related market (Layer 5) | +1 |

- Score 0-1: NO TRADE
- Score 2: Standard size (2.5% capital)
- Score 3+: Double size (5% capital — hard cap)
