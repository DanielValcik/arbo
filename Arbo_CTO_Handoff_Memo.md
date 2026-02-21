# ARBO — CTO Handoff Memo for Development Team

**From:** CTO
**Date:** 20 February 2026
**Re:** Brief v4 changes, cost decisions, and LLM strategy

> Read this BEFORE opening the dev brief. It explains the reasoning behind key decisions so you don't waste time questioning things that have already been researched and resolved.

---

## 1. What Changed Since v3 and Why

The dev team reviewed the v3 brief against live API documentation and found 10 factual errors. All have been corrected in v4. Additionally, the CEO conducted a cost optimization pass that changed the LLM strategy and news data sources. Here's what you need to know.

---

## 2. Matchbook API — Corrected Endpoints

**What was wrong:** Brief had wrong auth endpoint, wrong offers endpoint, wrong session TTL, and wrong pricing claim.

**What's correct now:**

| Item | v3 (wrong) | v4 (correct) | Source |
|---|---|---|---|
| Auth | `POST /edge/rest/security/session` | `POST /bpapi/rest/security/session` | developers.matchbook.com/reference/login |
| Offers | `POST /offers` | `POST /edge/rest/v2/offers` | developers.matchbook.com/changelog — v2 has no in-play delay for unmatched |
| Session TTL | 15 min → Redis TTL 600s | ~6 hours → Redis TTL 18000s | developers.matchbook.com/reference/login: "approximately 6 hours" |
| API cost | "Free under 1M GET" | £100 per 1M GET, all tiers | developers.matchbook.com/docs/pricing |

**Impact on implementation:**

- **Session management is simpler** than originally thought. 6-hour sessions mean you re-auth ~4 times/day, not every 15 minutes. The `asyncio.Lock` around re-auth is still needed (concurrent requests share one session), but it fires much less often.

- **v2 offers endpoint** adds a `delayed` status for in-play offers. Handle this in the offer polling loop: if status is `delayed`, continue polling (it will transition to `open` or `matched` after the delay elapses).

- **API cost is real.** At naive polling rates we'd burn 2M+ GET/month = £200+. Sprint 2 now includes an explicit task for Matchbook API cost optimization:
  - **Tiered polling:** events >24h away → every 60s; events <6h → every 8s; live events → every 5s
  - **Batch with `include-prices=true`:** Confirmed parameter on runners endpoint. One GET returns runners WITH prices — saves 2 separate GETs per event per poll cycle.
  - **Cache events + markets in Redis:** These change rarely. Only poll runners/prices at high frequency.
  - **Target: <500K GET/month = <£50**

---

## 3. LLM Strategy — What, Why, and How

### What does the LLM do in Arbo?

One thing only: the **Situational Edge Agent** (Sprint 4). It takes 3–5 sports news headlines about an upcoming match and outputs a structured JSON classifying the impact on odds:

```json
{"magnitude": 8, "direction": "away_favored", "confidence": 0.85}
```

It does NOT do: creative writing, conversation, code generation, complex reasoning, or anything else. Pure structured extraction + classification from short text.

### Why Gemini 2.0 Flash, not Claude Haiku?

| | Gemini 2.0 Flash | Claude Haiku 4.5 |
|---|---|---|
| Input cost | $0.10/MTok | $1.00/MTok |
| Output cost | $0.40/MTok | $5.00/MTok |
| Cost at 50K tok/day | ~€0.30/month | ~€17/month |
| Guaranteed JSON | ✅ `response_mime_type: "application/json"` | ❌ sometimes wraps in markdown |
| Quality for this task | More than sufficient | Also sufficient, but 10× pricier |

For parsing "Haaland out with hamstring injury" into `{magnitude: 8}`, we don't need frontier intelligence. We need fast, reliable, cheap structured extraction — which is exactly what Gemini Flash is optimized for.

### EEA restriction — important

Google's Gemini API Additional Terms state: applications serving users in the EEA/UK/Switzerland **must use paid services only**. We're in Czech Republic (EEA), so the free tier is NOT available to us.

**Impact:** Enable Google Cloud billing from day 1. At $0.10/$0.40 per MTok and our volume of ~100 calls/day with ~500 tokens each, monthly cost is approximately **€0.30**. Not a concern.

**Implementation:** Set `GOOGLE_AI_API_KEY` in `.env`. The `google-genai` SDK handles paid tier automatically when a valid API key is provided.

### Shadow mode — critical for Sprint 4

The LLM agent runs in **shadow mode** initially:

1. It processes news items and writes signals to the `opportunities` table
2. It sends Telegram alerts for every signal: `🧠 LLM SIGNAL | Liverpool vs Arsenal | Injury: Salah doubtful | Magnitude: 7 | Conf: 0.82 | ⚠️ SHADOW — not executing`
3. Signal weight `W₃ = 0.0` in config — the signal is logged but does NOT influence the decision engine
4. CEO manually reviews signals vs. actual odds movements for 2–4 weeks
5. Only after CEO confirms signal quality does he set `W₃ > 0` in `settings.yaml`

**Why:** An LLM misinterpreting "Haaland returns to training" as negative (instead of positive) would cause Arbo to bet in the wrong direction. Shadow mode prevents this until we validate signal quality on live data.

### Claude Haiku as fallback

Claude stays in the codebase behind the `BaseLLMAgent` abstraction. If Gemini proves unreliable on structured JSON extraction (unlikely given the `response_mime_type` guarantee, but possible), switching is one config change:

```yaml
llm:
  provider: claude    # was: gemini
```

Both agents get identical system prompts and expected JSON schemas. The abstraction is in `src/agents/situational_agent.py`.

---

## 4. News Sources — NewsAPI Is Dead, Long Live RSS

### What happened

NewsAPI.org free tier has severe restrictions we missed:
- Localhost only (production requests blocked)
- 100 requests/day
- 24-hour delay on articles
- Production tier: **$449/month**

For a system that needs real-time injury news, a 24-hour delay is useless. And $449/month for news when our total infra budget is ~€100 is absurd.

### What replaces it

All free, all real-time:

| Source | Implementation | Delay | Cost |
|---|---|---|---|
| BBC Sport RSS | `feedparser`, poll every 5 min | ~realtime | €0 |
| ESPN RSS | `feedparser`, poll every 5 min | ~realtime | €0 |
| Sky Sports RSS | `feedparser`, poll every 5 min | ~realtime | €0 |
| Google News RSS | `https://news.google.com/rss/search?q={team}` — no API key needed | ~realtime | €0 |
| Team-specific RSS | Club official sites have RSS feeds | ~realtime | €0 |
| Reddit (asyncpraw) | r/soccer, r/sportsbook, sport-specific subs | Often BEFORE mainstream media | €0 |
| GNews API (backup) | Free tier: 100 req/day keyword search | ~realtime | €0 |

**Reddit is actually our best source.** Injury leaks, lineup confirmations, and insider info often hit Reddit 5–10 minutes before BBC or ESPN publish. This is exactly the window we're trying to exploit.

### Implementation note

Configure RSS feed URLs in `settings.yaml` under a new `news_feeds` section. Make it easy to add/remove feeds without code changes:

```yaml
news_feeds:
  rss:
    - url: "https://feeds.bbci.co.uk/sport/football/rss.xml"
      name: "BBC Sport Football"
    - url: "https://www.espn.com/espn/rss/soccer/news"
      name: "ESPN Soccer"
    - url: "https://news.google.com/rss/search?q=premier+league+injury&hl=en"
      name: "Google News EPL Injuries"
  reddit:
    subreddits: ["soccer", "sportsbook", "nba", "PremierLeague"]
  gnews:
    enabled: false    # Enable if RSS coverage insufficient
```

---

## 5. Updated Budget Summary

| Item | Min/mo | Max/mo | Notes |
|---|---|---|---|
| Hetzner CX22 | €3.99 | €7.49 | |
| Matchbook API | €0 | €60 | Target <500K GET via tiered polling |
| LLM (Gemini Flash) | €0.30 | €2 | Paid tier. Claude fallback: €15–20 |
| Odds API | €0 | €35 | Free 500/mo → paid $40/mo if needed |
| News (RSS+Reddit) | €0 | €0 | All free |
| **Total** | **€4** | **€105** | **Realistic: €20–40/mo** |

---

## 6. Dependency Version Changes

Three packages had major version jumps. Be aware:

**pytest-asyncio >=1.0** (was >=0.23)
- `event_loop` fixture is removed in 1.x
- Use `@pytest.mark.asyncio` decorator instead
- Set `asyncio_mode = "auto"` in `pyproject.toml` `[tool.pytest.ini_options]`

**XGBoost >=3.0** (was >=2.0)
- Major API changes from 2.x. Check migration guide.
- Key: `DMatrix` interface changed, some deprecated parameters removed

**google-genai >=1.0** (new dependency)
- Google's official Python SDK for Gemini API
- Do NOT confuse with `google-generativeai` (older package) — use `google-genai`

---

## 7. What Doesn't Change

Everything else in the brief is validated and correct:

- Database schema (Section 4) — all 7 tables, indexes, types
- Risk management constants (Section 7) — hardcoded, not configurable
- Kelly calculator (Section 7) — half-Kelly with 5% cap
- Sprint structure and timeline — 5 sprints, 13 weeks
- Coding standards (Section 10) — black, ruff, mypy, pytest
- Communication protocol (Section 14) — weekly dev→CEO, bi-weekly CEO→owner
- Architecture layers 1–5 — data ingestion, intelligence, decision, execution, monitoring

---

## 8. Priority Order for Sprint 1

Day 1 priorities (so you don't waste time):

1. ✅ Provision Hetzner VPS + install Python 3.12, PostgreSQL 16, Redis 7
2. ✅ Create `arbo/` repo with project structure from Section 9
3. ✅ Run Alembic migration to create all 7 tables
4. ✅ Implement Matchbook client with **correct** endpoints from this memo
5. ✅ Verify: single GET to Matchbook returns events with prices
6. ✅ Set up systemd service for unattended operation

Do NOT start on:
- ❌ LLM integration (Sprint 4)
- ❌ BetInAsia scraping (Sprint 3)
- ❌ Value model training (Sprint 3)
- ❌ Polymarket (Sprint 5)

---

## 9. Questions? Escalation Path

- **Technical questions about brief:** Ask CTO (me) directly
- **Questions about strategy, budget, or priorities:** Escalate to CEO
- **Matchbook API questions:** Email api-support@matchbook.com (they're responsive)
- **Blocked on account registration (Matchbook/BetInAsia KYC):** CEO handles this — not dev team's responsibility

---

> Build Arbo. Ship Sprint 1 in 2 weeks. Everything you need is in the brief and this memo.
