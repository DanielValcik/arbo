# CEO Decision: Strategy B Data Sources — FINAL

> Date: 2026-02-26
> Status: APPROVED — Supersedes CEO_DECISION_STRATEGY_B_DATA_SOURCE.md
> Decision by: CEO (Daniel Valcik)

## Executive Summary

Strategy B is split into two sub-strategies based on distinct data sources:

| Sub-Strategy | Data Source | What It Measures | Capital |
|---|---|---|---|
| **B1: Attention Markets Oracle** | Kaito Info Markets (platform-level) | Platform mindshare vs market price | €150 |
| **B2: Social Momentum Divergence** | Cookie.fun (per-project) | Project mindshare vs market price | €250 |

**Total B allocation remains €400** (20% of €2,000 bankroll).

## Decision Rationale

### Cookie.fun (B2 — PRIMARY)
- **14,000+ crypto projects** with daily mindshare %, sentiment scores, engagement metrics
- Data is **public and free** (no API key, no enterprise contract)
- Addressable market: ~500+ Polymarket crypto markets
- Signal: Project social momentum diverging from market price
- Example: If $SOL mindshare spikes 3x but Polymarket "SOL > $200" stays at 0.30, that's a buy signal

### Kaito Info Markets (B1 — SUPPLEMENTARY)
- **~10 platforms** tracked (Polymarket, Kalshi, Metaculus, etc.)
- Platform-level mindshare only (NOT per-market)
- Free to scrape from kaito.ai/info-markets
- Signal: Platform attention shifts predicting volume/price moves
- More limited but still useful as confluence signal

### Why This Split?
1. Cookie.fun covers the **larger market** (14K projects vs ~10 platforms)
2. Different signal types complement each other
3. Both are **zero additional monthly spend**
4. B2 can trade independently; B1 adds confluence

## Implementation Tasks

### Phase 1: Cookie.fun (B2) — Priority

**B2-01: Cookie.fun API Discovery** (Research)
- Reverse-engineer Cookie.fun's API endpoints
- Document: base URL, endpoints, response schemas, rate limits
- Identify: authentication (if any), pagination, available metrics
- Deliverable: `docs/COOKIE_FUN_API_REPORT.md`

**B2-02: Cookie.fun Client** (Code)
- `arbo/connectors/cookie_fun.py`
- Methods: `get_project_mindshare(symbol)`, `get_trending_projects()`, `get_historical_mindshare(symbol, days)`
- Rate limiting, caching, error handling
- Tests: `arbo/tests/test_cookie_fun.py`

**B2-03: Social Momentum Divergence Calculator** (Code)
- `arbo/strategies/social_divergence.py`
- Compare Cookie.fun mindshare trend vs Polymarket price trend
- Generate signal when divergence exceeds threshold
- Integrate with reflexivity_surfer.py (B2 signal source)
- Tests: `arbo/tests/test_social_divergence.py`

### Phase 2: Kaito Info Markets (B1)

**B1-01: Kaito Info Markets Scraper** (Code)
- `arbo/connectors/kaito_info_markets.py`
- Scrape kaito.ai/info-markets platform-level data
- Methods: `get_platform_mindshare()`, `get_platform_rankings()`
- Respect rate limits, handle anti-scraping
- Tests: `arbo/tests/test_kaito_info_markets.py`

**B1-02: Attention Markets Signal** (Code)
- Update `arbo/strategies/reflexivity_surfer.py`
- Add B1 signal source alongside B2
- Platform mindshare as confluence boost for B2 signals

### Phase 3: Integration

**B-INT: Integration Test** (Test)
- E2E test: Cookie.fun data → divergence signal → reflexivity gate → trade
- E2E test: Kaito platform data → confluence boost
- Verify B1+B2 capital stays within €400 allocation

## Budget Impact

| Item | Monthly Cost |
|---|---|
| Cookie.fun API | $0 (public) |
| Kaito Info Markets | $0 (public scraping) |
| **Total new spend** | **$0** |

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Cookie.fun blocks scraping | Respectful rate limiting, user-agent, fallback to Gemini LLM |
| Cookie.fun data quality | Cross-validate with CoinGecko social stats |
| Kaito anti-scraping | Minimal request rate, cache aggressively |
| Low correlation with PM prices | Paper trade 2 weeks before live, measure hit rate |

## Priority Order

1. **B2-01** (Cookie.fun API discovery) — START IMMEDIATELY
2. **B2-02** (Cookie.fun client)
3. **B2-03** (Social divergence calculator)
4. **B1-01** (Kaito scraper)
5. **B1-02** (Attention Markets signals)
6. **B-INT** (Integration tests)

## CTO Directive

CTO should begin with Task B2-01 (Cookie.fun API discovery) immediately. This is a research task — reverse-engineer the Cookie.fun API, document endpoints, and assess feasibility before writing any production code.
