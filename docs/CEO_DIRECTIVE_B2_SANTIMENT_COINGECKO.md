# CTO Directive: B2 Data Source Refactor — LunarCrush → Santiment + CoinGecko

> Date: 26 February 2026
> Priority: HIGH — replaces CTO_DIRECTIVE_B2_LUNARCRUSH.md
> From: CEO
> Status: ACTIVE

## Decision

LunarCrush pricing increased to $90/mo (was $24). Not viable at €50-60/mo budget ceiling.
Switch to Santiment Free ($0) + CoinGecko Demo ($0).

## Trade-offs

- LOSE: Real-time sentiment (Santiment sentiment has 30d lag on Free)
- GAIN: On-chain activity (daily_active_addresses, transactions_count) + dev_activity
- Signal quality: Medium (proxy metrics) vs High (direct sentiment)
- Cost: $0/mo vs $90/mo

## Data Sources

### Santiment Free — GraphQL at https://api.santiment.net/graphql
- No auth for free metrics
- Free metrics: dev_activity, daily_active_addresses, active_addresses_24h, transactions_count, price_usd
- Rate limits: 1,000/month, 500/hour, 100/minute
- ~300 calls/month estimated

### CoinGecko Demo — REST at https://api.coingecko.com/api/v3
- Auth: x-cg-demo-api-key header (COINGECKO_API_KEY env var)
- Endpoints: /coins/markets (bulk), /coins/{id} (detail with community_data)
- Rate limits: 30/min, 10,000/month
- ~820 calls/month estimated

## Tasks

- B2-10: Santiment client (santiment_client.py)
- B2-11: CoinGecko client (coingecko_client.py)
- B2-12: DB schema update (migration 006, social_momentum_v2)
- B2-13: Divergence calculator rewrite (social_divergence.py)
- B2-14: Scheduler integration (main_rdh.py)
- B2-15: Cleanup (remove LunarCrush)
