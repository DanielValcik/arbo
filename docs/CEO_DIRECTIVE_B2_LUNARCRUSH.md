# CTO Directive: B2 Data Source Switch — Cookie.fun → LunarCrush

> Date: 26 February 2026
> Priority: HIGH — unblocks B2 pipeline
> From: CEO
> To: CTO
> Status: ACTIVE

## Decision

Cookie.fun API is gated ($499+/mo or 10K $COOKIE staking). B2 switches to LunarCrush API v4.
Owner approved potential $24/mo upgrade if free tier insufficient.

## LunarCrush API v4

- Base URL: `https://lunarcrush.com/api4`
- Auth: `Authorization: Bearer <API_KEY>`
- Docs: `https://github.com/lunarcrush/api`
- Pricing: Free Discover tier first. Individual $24/mo if rate-limited.

## Required Endpoints (3)

1. `GET /api4/public/coins/list/v2?sort=market_cap_rank&limit=100` — bulk snapshot, 4x/day
2. `GET /api4/public/coins/:coin/time-series/v2?bucket=hour&interval=1w` — historical, on-demand
3. `GET /api4/public/topic/:topic/v1` — topic detail, enrichment only

## Tasks

- B2-01R: LunarCrush client (`arbo/connectors/lunarcrush_client.py`)
- B2-02: Social momentum DB schema (Alembic migration)
- B2-03: Divergence calculator (`arbo/strategies/social_divergence.py`)
- B2-04: Scheduler integration

## Budget: ~1,500 calls/month (well within 5K Individual limit)

## Success Criteria (4-week paper trading)
- Minimum: ≥10 divergence signals generated
- Target: ≥55% hit rate on paper trades
