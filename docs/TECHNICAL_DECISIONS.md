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
