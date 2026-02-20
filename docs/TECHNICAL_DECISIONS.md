# ARBO — Technical Decisions Log

> Decisions made during project setup. Each entry explains WHAT was decided, WHY, and WHO approved it.

---

## TD-001: Dependency Version Pins (2026-02-20)

**Decision:** Update minimum version pins beyond what brief v4 specifies.

**Rationale:** Brief v4 uses conservative floors from original research. Since we're starting fresh, there's no backward compatibility concern. Using versions closer to current reduces the gap between our minimum and what actually installs.

**Approved versions for pyproject.toml:**

```
aiohttp>=3.11          # Brief: >=3.9, Current: 3.13.3
sqlalchemy[asyncio]>=2.0  # Brief: >=2.0, Current: 2.0.46 (2.1 is beta, stay on 2.0)
asyncpg>=0.30          # Brief: >=0.29, Current: 0.31.0
alembic>=1.16          # Brief: >=1.13, Current: 1.17.2
redis>=5.0             # Brief: >=5.0, Current: 7.1.1 (keep floor, auto-installs 7.x)
pydantic>=2.10         # Brief: >=2.5, Current: 2.12.5
pydantic-settings>=2.8 # Brief: >=2.1, Current: 2.13.1
structlog>=24.1        # Brief: >=24.1, Current: 25.5.0 (brief floor is fine)
slack-bolt>=1.27       # Replaced aiogram (TD-009)
playwright>=1.55       # Brief: >=1.40, Current: 1.58.0
xgboost>=3.0           # Brief: >=3.0 ✓
scikit-learn>=1.6      # Brief: >=1.4, Current: 1.8.0
google-genai>=1.0      # Brief: >=1.0 ✓
anthropic>=0.70        # Brief: >=0.40, Current: 0.81.0
rapidfuzz>=3.10        # Brief: >=3.5, Current: 3.14.3
pytest-asyncio>=1.0    # Brief: >=1.0 ✓
```

**Status:** Proposed — awaiting CEO review. Will use brief's exact pins if CEO prefers stability.

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

**Decision:** Use separate base URLs for auth vs data endpoints.

**Implementation:**
```python
AUTH_BASE_URL = "https://api.matchbook.com/bpapi/rest"
DATA_BASE_URL = "https://api.matchbook.com/edge/rest"
OFFERS_BASE_URL = "https://api.matchbook.com/edge/rest/v2"
```

**Rationale:** Matchbook uses different path prefixes:
- `/bpapi/rest/security/session` for auth
- `/edge/rest/events/...` for data
- `/edge/rest/v2/offers` for betting (v2 has no in-play delay)

---

## TD-004: Matchbook API Cost Optimization Strategy (2026-02-20)

**Decision:** Implement 3-tier polling to keep GET requests under 500K/month.

**Calculation:**
- Naive: 10 events × 3 markets × 1 GET/8s × 86400s/day = ~32K GET/day = ~1M/month = £100
- Optimized with tiers + batching:
  - Events >24h: 10 events × 1 GET/60s = ~14K/day
  - Events <6h: ~5 events × 1 GET/8s = ~54K/day (only a few events in this window at any time)
  - Live: ~2 events × 1 GET/5s = ~35K/day
  - With `include-prices=true`: each GET covers runners+prices (no separate price GET)
  - Total: ~100K/day = ~300K/month = ~£30

**Target: <500K GET/month (<£50/month)**

---

## TD-005: SQLAlchemy Model Organization (2026-02-20)

**Decision:** All models in `src/utils/db.py` as a single file.

**Rationale:** 7 tables is not enough to justify splitting into separate model files. Single file is easier to maintain, and Alembic autogenerate works better with all models imported from one place.

**If the model count grows beyond 12+, split into `src/models/` package.**

---

## TD-006: Redis Usage Strategy (2026-02-20)

**Decision:** Redis serves 4 purposes:
1. **Session token cache** — key: `matchbook:session_token`, TTL: 18000s
2. **Rate limiter state** — token bucket counters per client
3. **Live odds cache** — latest odds per event for fast reads
4. **Emergency write queue** — if PostgreSQL goes down, queue critical writes (max 1000)

**Key naming convention:** `arbo:{subsystem}:{identifier}`
```
arbo:session:matchbook → session token string
arbo:ratelimit:matchbook:tokens → float (remaining tokens)
arbo:odds:{event_id}:{market}:{selection} → JSON snapshot
arbo:writequeue:bets → list of serialized bet records
arbo:writequeue:opportunities → list of serialized opportunity records
arbo:metrics:daily_get_count → integer
```

---

## TD-007: Structured Logging Strategy (2026-02-20)

**Decision:** All logs use structlog with JSON output in production, console renderer in development.

**Log levels:**
- DEBUG: Detailed polling data, raw API responses (only in dev)
- INFO: Events fetched, odds stored, normal operations
- WARNING: Low API quota, partial fills, re-auth triggered, fuzzy match <90
- ERROR: API failures, invalid data, risk limit breaches
- CRITICAL: Kill switch triggered, repeated failures, data corruption

**Bound context fields (always present):**
- `module` — which component (matchbook_client, arb_scanner, etc.)
- `event_id` — when processing a specific event
- `strategy` — when in decision pipeline (arb, value, situational)
- `mode` — paper or live

**Secret masking:** structlog processor that redacts any value containing:
- `password`, `token`, `api_key`, `secret`, `session`

---

## TD-008: No NewsAPI.org (2026-02-20)

**Decision:** Removed from all phases. Free tier is localhost-only with 24h delay. Production tier $449/month.

**Replacement:** RSS feeds (BBC, ESPN, Sky, Google News RSS) + Reddit (asyncpraw) + GNews API (backup).

See CTO memo Section 4 for full details.

---

## TD-009: Slack Instead of Telegram (2026-02-20)

**Decision:** Replace Telegram (aiogram) with Slack (slack-bolt) for all bot commands and alerts.

**Approved by:** CEO

**Rationale:** CEO decision. Slack is the preferred communication platform.

**Implementation details:**

| Aspect | Old (Telegram) | New (Slack) |
|--------|---------------|-------------|
| Framework | aiogram 3.x | slack-bolt 1.27+ (AsyncApp) |
| Connection | Bot polling | Socket Mode (WebSocket, no public URL) |
| Commands | Telegram bot commands | Slack Slash Commands |
| Alerts | sendMessage | chat.postMessage + Block Kit |
| Formatting | Markdown | Block Kit (richer formatting) |
| File | `src/alerts/telegram_bot.py` | `src/alerts/slack_bot.py` |
| Dependency | `aiogram>=3.20` | `slack-bolt>=1.27` |
| Env vars | TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_CHAT_ID | SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_CHANNEL_ID |

**Slack setup requirements:**
1. Create Slack app at api.slack.com/apps
2. Enable Socket Mode
3. Create App-Level Token with `connections:write` scope → SLACK_APP_TOKEN (xapp-)
4. Add Bot Token Scopes: `commands`, `chat:write`, `chat:write.public`, `channels:read`
5. Install to workspace → SLACK_BOT_TOKEN (xoxb-)
6. Register slash commands: /status, /pnl, /kill, /paper, /live
7. Create #arbo-alerts channel → SLACK_CHANNEL_ID

**Architecture:**
- `AsyncSocketModeHandler` runs alongside main asyncio loop via `asyncio.create_task()`
- Proactive alerts sent via `app.client.chat_postMessage()` from polling loop
- Block Kit used for rich formatting (header + section fields + context)
- Acknowledge slash commands within 3s, then respond with data

**Cost:** Free (Slack free plan supports Socket Mode, custom apps, unlimited API calls).

**Impact on brief:**
- Section 3.1: aiogram → slack-bolt
- Section 3.4 Layer 5: Telegram → Slack everywhere
- Section 5.1: env vars change
- Section 6 Sprint 2: telegram_bot.py → slack_bot.py
- Section 8: all "Telegram alert" → "Slack alert"
- Section 9: file path change
- Section 12: dependency change
