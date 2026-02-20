# ARBO — Open Questions & Risks

> Items that need resolution before or during Sprint 1. Marked with priority.

---

## Questions for CEO (Blocking)

### Q1: Matchbook Account Status [HIGH]
**Status:** NEEDS ANSWER
Has the Matchbook account been registered from CZ IP? Is KYC complete? Is it upgraded to Trader Plan?
- Without this, we cannot do integration testing against live API
- We can develop and test with mocked responses, but need live access by end of Sprint 1

### Q2: Hetzner VPS Provisioned? [HIGH]
**Status:** NEEDS ANSWER
Is the Hetzner CX22 available? Need SSH access, IP address.
- Ubuntu 24.04, Python 3.12, PostgreSQL 16, Redis 7
- Or should dev team provision it?

### Q3: Slack App Setup [MEDIUM]
**Status:** NEEDS ANSWER
Has a Slack workspace + app been created? Need:
- Slack workspace (can be free plan)
- Slack app created at api.slack.com/apps with Socket Mode enabled
- Bot token (xoxb-...) — from OAuth & Permissions after install
- App-level token (xapp-...) — from Basic Info > App-Level Tokens (scope: connections:write)
- Channel ID for alerts (create #arbo-alerts channel)
- Slash commands registered: /status, /pnl, /kill, /paper, /live
- Bot scopes: commands, chat:write, chat:write.public, channels:read
- Not blocking for Sprint 1, but needed early Sprint 2

### Q4: Google Cloud Billing [LOW]
**Status:** NEEDS ANSWER for Sprint 4
Is Google Cloud billing enabled for Gemini API?
- EEA requires paid tier from day 1
- Not needed until Sprint 4, but setup takes time

---

## Technical Risks

### R1: Matchbook Sport IDs [MEDIUM]
**Risk:** We don't know Matchbook's internal sport IDs.
**Mitigation:** Call `GET /lookups/sports` on first successful login. Cache result. Log the full mapping.
**Status:** Will resolve during Sprint 1 implementation.

### R2: Matchbook Rate Limit Enforcement [MEDIUM]
**Risk:** If we exceed rate limits, we get a 10-minute block. Brief says max 10 req/s, but actual enforcement thresholds may differ per endpoint (events=700/min, betting=3000/min).
**Mitigation:** Start conservative (5 req/s), monitor, increase gradually. Log every 429 response.
**Status:** Will tune during Sprint 1.

### R3: odds_snapshots Table Growth [LOW]
**Risk:** Brief says millions of rows per month. After 3 months this could be 10M+ rows.
**Mitigation:** Plan table partitioning by month (captured_at). Don't implement now, but design for it.
**Status:** Monitor in Sprint 2, implement partitioning if needed.

### R4: BetInAsia Web Scraping Fragility [MEDIUM]
**Risk:** Playwright scraping is inherently fragile — any UI change breaks the scraper.
**Mitigation:** Sprint 3 task. Isolate scraper logic so selectors are configurable. Log every parse failure. Max 3 restarts/hour then stop.
**Status:** Sprint 3.

### R5: Reddit API Approval [LOW]
**Risk:** Reddit requires OAuth app approval for API access. May be delayed.
**Mitigation:** Apply early. RSS feeds + Google News RSS are the primary sources anyway. Reddit is supplementary.
**Status:** Apply during Sprint 1, expect to use in Sprint 4.

### R6: Matchbook API Cost Overrun [MEDIUM]
**Risk:** Without careful optimization, API costs could exceed £100/month.
**Mitigation:** Implement tiered polling (Sprint 2 task), daily GET count logging, alert if >15K GETs in a day.
**Status:** Sprint 2 explicit task.

---

## Decisions Needed (Not Blocking)

### D1: Python 3.12 vs 3.13
**Recommendation:** Target 3.12 as minimum, test on 3.13. All dependencies support both.
**Decision:** Use what's installed on VPS (likely 3.12 on Ubuntu 24.04).

### D2: Package Manager
**Options:** pip, uv, poetry
**Recommendation:** `uv` — fastest, modern, works with pyproject.toml natively. Falls back to pip if needed.
**Decision:** CEO preference?

### D3: Alembic Migration Strategy
**Options:**
  a) Auto-generate migrations from SQLAlchemy models
  b) Hand-write all migrations
**Recommendation:** Auto-generate with manual review. Alembic autogenerate catches schema drift.
**Decision:** Auto-generate.
