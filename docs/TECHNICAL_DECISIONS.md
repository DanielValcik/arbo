# ARBO — Technical Decisions Log

> Decisions made during project setup. Each entry explains WHAT was decided, WHY, and WHO approved it.

---

## TD-001: Dependency Version Pins (2026-02-20)

**Decision:** Update minimum version pins beyond what brief v4 specifies.

**Rationale:** Brief v4 uses conservative floors from original research. Since we're starting fresh, there's no backward compatibility concern. Using versions closer to current reduces the gap between our minimum and what actually installs.

**Update (decision):** Use brief's exact version pins for now. They will resolve to latest compatible versions anyway. No need to deviate from the brief unnecessarily.

---

## TD-002: pytest-asyncio Auto Mode (2026-02-20)

**Decision:** Use `asyncio_mode = "auto"` in pyproject.toml.

**Rationale:** pytest-asyncio 1.0+ removed the `event_loop` fixture. Auto mode eliminates need for `@pytest.mark.asyncio` on every test and is the recommended approach for new projects.

**Config:**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## TD-003: Matchbook Auth Path Separation (2026-02-20)

**Decision:** ~~Use separate base URLs for auth vs data endpoints.~~ **ARCHIVED** — Matchbook integration superseded by Polymarket pivot (TD-010).

---

## TD-004: Matchbook API Cost Optimization Strategy (2026-02-20)

**Decision:** ~~Implement 3-tier polling to keep GET requests under 500K/month.~~ **ARCHIVED** — Matchbook integration superseded by Polymarket pivot (TD-010).

---

## TD-005: SQLAlchemy Model Organization (2026-02-20)

**Decision:** All models in single DB module file.

**Rationale:** Manageable table count doesn't justify splitting into separate model files. Single file is easier to maintain, and Alembic autogenerate works better with all models imported from one place.

**Update (2026-02-21):** Schema will be rewritten for Polymarket data model. Same organizational principle applies — single `arbo/utils/db.py` file.

---

## TD-006: Redis Usage Strategy (2026-02-20)

**Decision:** ~~Redis serves 4 purposes: session cache, rate limiter, odds cache, write queue.~~

**Update (2026-02-21):** **Redis REMOVED.** CEO decision during Polymarket pivot. Rationale:
- Polymarket L2 API creds don't expire like Matchbook sessions
- Rate limiting: in-memory token bucket (already works)
- Odds cache: in-memory dict with TTL
- Write queue: asyncio.Queue
- One less dependency on VPS. Hetzner CX22 8GB RAM is sufficient for in-memory caches.

---

## TD-007: Structured Logging Strategy (2026-02-20)

**Decision:** All logs use structlog with JSON output in production, console renderer in development.

**Log levels:**
- DEBUG: Detailed API responses, raw data (only in dev)
- INFO: Markets fetched, signals generated, trades executed
- WARNING: Low API quota, WebSocket reconnect, fuzzy match <90
- ERROR: API failures, invalid data, risk limit breaches
- CRITICAL: Kill switch triggered, repeated failures, emergency shutdown

**Bound context fields (always present):**
- `module` — which component (polymarket_client, value_betting, etc.)
- `layer` — strategy layer number (1-9)
- `market_id` — when processing a specific market
- `mode` — paper or live

---

## TD-008: No NewsAPI.org (2026-02-20)

**Decision:** Removed from all phases. Free tier is localhost-only with 24h delay. Production tier $449/month.

**Replacement:** RSS feeds (BBC, ESPN, Sky, Google News RSS) + Reddit (asyncpraw) + GNews API (backup).

---

## TD-009: Slack Instead of Telegram (2026-02-20)

**Decision:** Replace Telegram (aiogram) with Slack (slack-bolt) for all bot commands and alerts.

**Approved by:** CEO

**Rationale:** CEO decision. Slack is the preferred communication platform.

**Update (2026-02-21):** Reconfirmed during Polymarket pivot. Slack stays despite v3 brief specifying Telegram.

---

## TD-010: Polymarket Pivot (2026-02-21)

**Decision:** Pivot entire system from Matchbook betting exchange to Polymarket prediction market.

**Approved by:** CEO

**Rationale:**
- Matchbook (original plan) cannot be used — cannot open account from Czech Republic
- Polymarket has no geo-block for CZ (blocked: US, FR, BE, CH, PL, SG, AU, RO, HU, PT, UA, UK)
- CLOB architecture on Polygon = transparent, programmable
- Most sports markets have 0% fee (February 2026). Maker rebates active.
- py-clob-client (official Python SDK) — REST + WebSocket, batch orders up to 15/call
- USDC collateral, gas ~$0.007/tx on Polygon

**What changes:**
1. **Exchange layer:** Matchbook REST → Polymarket CLOB (py-clob-client) + Gamma API + WebSocket
2. **Architecture:** 5-layer (ingestion → agents → decision → execution → monitoring) → 9-layer strategy system with confluence scoring
3. **Project structure:** `src/` → `arbo/` (per v3 brief specification)
4. **Database:** PostgreSQL stays, Redis removed. Schema rewritten for Polymarket data model.
5. **Risk limits:** Updated for Polymarket specifics (position sizing, market concentration, whale copy limits)
6. **Strategies:** Arb scanner → 9 specialized strategies (MM, Value, Semantic Graph, Whale, Logical Arb, Temporal Crypto, Order Flow, Attention Markets, Sports Latency)
7. **Sprint plan:** Reset to Sprint 1 with new PM-XXX task IDs
8. **Dependencies:** Add py-clob-client, web3.py, chromadb, sentence-transformers. Remove playwright, redis.

**What stays:**
- Slack bot (TD-009, already built)
- PostgreSQL + SQLAlchemy + Alembic (already proven)
- The Odds API client (Pinnacle odds still needed for value model)
- structlog, Pydantic config, black/ruff/mypy toolchain
- Odds math utilities (Kelly, implied prob)
- Config loading pattern (.env → YAML → overlay)
- VPS deployment (Hetzner CX22, systemd)

**Old code:** Moved to `_archive/` — available for reference, excluded from imports and tests.

---

## TD-011: PostgreSQL Over SQLite (2026-02-21)

**Decision:** Keep PostgreSQL + asyncpg + SQLAlchemy + Alembic for all phases (paper and live).

**Approved by:** CEO

**Rationale:** v3 brief specified SQLite for Sprint 1-4 assuming greenfield start. We already have:
- Working async PostgreSQL setup with asyncpg driver
- SQLAlchemy 2.0 async ORM patterns
- Alembic migration infrastructure
- Hetzner CX22 can easily run PostgreSQL

Switching to SQLite would be a step backward with zero value. Schema management via Alembic is proven.

---

## TD-012: No Redis (2026-02-21)

**Decision:** Remove Redis dependency entirely.

**Approved by:** CEO

**Rationale:** Polymarket doesn't need Redis for any of its original 4 purposes:
1. Session caching → L2 API creds don't expire (unlike Matchbook's 6h sessions)
2. Rate limiting → in-memory token bucket (Python dict + asyncio)
3. Odds cache → in-memory dict with TTL wrapper
4. Emergency write queue → asyncio.Queue (bounded, thread-safe)

One less process to manage on VPS. 8GB RAM on Hetzner CX22 is more than enough for in-memory caches.

---

## TD-013: Project Structure Migration (2026-02-21)

**Decision:** Migrate from `src/` to `arbo/` directory structure per v3 brief specification.

**Approved by:** CEO

**Implementation:**
1. Create `arbo/` with full v3 brief structure (connectors/, core/, strategies/, models/, agents/, dashboard/, utils/, tests/)
2. Move `src/` to `_archive/` (preserve for reference, exclude from imports)
3. Migrate reusable components: logger.py, config.py pattern, db.py base, odds.py math, slack_bot.py
4. Update pyproject.toml: package discovery, isort first-party, coverage source
5. Update all imports to `arbo.*`
6. Delete `_archive/` after Sprint 1 confirms nothing else is needed

---

## TD-014: B3 Watchdog — Plně Autonomní Lokální Daemon (2026-04-11)

**Decision:** B3 Watchdog bude plně autonomní lokální Python daemon, ne advisory systém s CEO approval a ne Anthropic Managed Agents.

**Approved by:** CEO

**Rationale:**
- **Autonomie vs Advisory**: Watchdog sám detekuje anomálie, analyzuje root cause přes Gemini Flash, rozhoduje o parametrických změnách a implementuje je za běhu. CEO dostává post-action report co se stalo a proč. Důvod: rychlost reakce (minuty ne hodiny), kontinuální optimalizace 24/7 bez čekání na lidský input.
- **Lokální vs Managed Agents**: $1/měsíc (Gemini Flash) vs $80-100/měsíc (Claude runtime). Přímý PostgreSQL přístup (localhost, ms latence). Plná kontrola, git-versioned. Žádný vendor lock-in.
- **Safety**: 3-tier autonomie (Tier 1: plně autonomní v bounds, Tier 2: autonomní s eskalací, Tier 3: nikdy autonomní). Auto-revert po 50 tradech pokud WR klesne o >5pp. Hardcoded bounds na každý parametr.

**Alternatives rejected:**
- Managed Agents: příliš drahé ($80/mo), latence (network), omezená kontrola
- Advisory-only (CEO approve): příliš pomalé, B3 má 5-min horizont, čekání na schválení = ztracené příležitosti
- No watchdog: alpha decay nedetekována dny/týdny

**Implementation:** `arbo/core/b3_watchdog.py`, `arbo/core/b3_metrics.py`, `arbo/core/b3_anomaly.py`, `arbo/core/adaptive_config.py`, `arbo/agents/watchdog_agent.py`

---

## TD-015: TA Integration — tradingview-ta + Background Cache (2026-04-11)

**Decision:** Technická analýza (RSI, ADX, MACD, BB) integrována přes `tradingview_ta` Python knihovnu jako background asyncio cache. MCP server pouze pro interaktivní research v Claude Code.

**Rationale:**
- MCP server přidává sekundy latence (Claude round-trip), B3 potřebuje <100ms
- tradingview-ta: 0ms read z cache, 60s background update, $0 (žádné API klíče)
- Produkce: background task + in-memory cache → strategies čtou s nulovou latencí
- Research: tradingview-mcp-server v Claude Code pro ad-hoc BTC analýzu
- Fallback: ccxt + pandas-ta pokud tradingview-ta endpoint selže

**What this enables:**
- B3: TA regime detection (ADX), mean-reversion risk (RSI), multi-TF alignment
- B2: denní probability adjustment (RSI, MACD, BB na daily timeframe)
- Watchdog: enriched context pro Gemini Flash analýzu

**Implementation:** `arbo/models/ta_feature_provider.py`

---

## TD-016: TA Features — Logging-First (2026-04-11)

**Decision:** TA features se nejdřív logují do trade_details JSONB bez filtrování. Aktivní filtering až po 100+ tradech s TA daty a data-driven validaci.

**Rationale:**
- V5.0 scoring model (10 features, 37 tradů) = overfit za hodiny → zjednodušen na V6.0 (2 pravidla)
- Lekce: nikdy nefiltrovat bez dat. Logging-first → analýza korelace → filtering pokud Cohen's d > 0.3 a N > 30 per bucket.
- Autonomní Watchdog sám rozhodne kdy aktivovat TA filtr na základě nasbíraných dat.

**Data fields logged:** `ta_rsi_5m`, `ta_adx_5m`, `ta_macd_hist_5m`, `ta_bb_width_5m`, `ta_recommend_5m`, `ta_rsi_1h`, `ta_adx_1h`, `ta_multi_tf_aligned`, `ta_adx_regime`, `ta_rsi_zone`

---

## TD-017: Runtime Adaptive Config (2026-04-11)

**Decision:** Watchdog mění B3 parametry za běhu přes runtime adaptive config systém (in-memory dict s audit logem v DB).

**Rationale:**
- B3 quality_gate.py obsahuje statické konstanty. Watchdog potřebuje měnit parametry bez redeploye.
- Řešení: `adaptive_config.py` — čte default z quality_gate.py, Watchdog přepisuje runtime overrides.
- `strategy_b3.py` čte z adaptive_config místo přímo z quality_gate.
- Každá změna se loguje do `watchdog_decisions` DB tabulky (audit trail).
- Restart systému → parametry se vrátí na default (safe fallback).
- Auto-revert: pokud WR klesne o >5pp za 50 tradů po změně → automatický revert.

**Safety bounds:**
- Tier 1 (autonomní): velocity [30-100], dir_delta [5-40], edge [0.20-0.80], sizing ±50%
- Tier 2 (autonomní+flag): sigma_scale [0.25-0.50], entry_threshold [0.01-0.05]
- Tier 3 (nikdy): MAX_BET_SIZE, DAILY_LOSS_LIMIT, LIVE_MAX_FILL_PRICE, REQUIRE_CHAINLINK

**Implementation:** `arbo/core/adaptive_config.py`, `watchdog_decisions` DB tabulka
