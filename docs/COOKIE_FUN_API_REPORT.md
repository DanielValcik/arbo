# Cookie.fun API Discovery Report (B2-01)

> Date: 2026-02-26
> Status: COMPLETE — BLOCKER IDENTIFIED
> Task: B2-01 (Cookie.fun API Discovery)

## Executive Summary

Cookie.fun has a **private, API-key-gated REST API** at `api.cookie.fun`. Contrary to initial assumption, the data is **NOT freely accessible** — it requires either an enterprise contract ($499-$5K/mo in $COOKIE token) or staking 10,000 $COOKIE tokens (~$300-500 at current prices).

**This is a blocker for B2 as originally scoped.**

## API Infrastructure

| Property | Value |
|----------|-------|
| Base URL | `https://api.cookie.fun` |
| Server | Hetzner (37.27.117.104), NOT behind Cloudflare |
| Stack | ASP.NET Core (Kestrel) + Kong 3.6.1 API Gateway |
| Auth | `x-api-key` header required on all endpoints |
| Docs | Swagger exists at `/swagger/v2/swagger.json` but protected by HTTP Basic Auth |

## Authentication

All requests require `x-api-key` header. Without it:
```json
{"success":false,"error":{"errorType":"AuthorizationError","errorMessage":"API key is missing"}}
```

### How to Get an API Key

1. **Enterprise**: Contact `@St3cu` on Telegram. Pricing: $499-$5,000/month paid in $COOKIE token
2. **Token staking**: Lock 10,000 $COOKIE tokens in staking contract
3. **Sign up at cookie.fun**: Some sources mention requesting key after signup — unclear process

## Discovered Endpoints

### Only confirmed endpoint:
**`GET /v2/agents/agentsPaged`** — paginated list of tracked agents/projects with analytics data. Returns 401 without valid API key. Likely params: `page`, `pageSize`, `interval`, `category`, `chain`, `sort`, `search`.

### 500+ URL patterns tested, all returned 404:
- `/v2/agents/{search,trending,topAgents,mindshare,sentiment,...}`
- `/v2/tokens/*`, `/v2/kols/*`, `/v2/projects/*`, `/v2/trending/*`
- `/v1/influencers` (mentioned in old blog posts — deprecated)
- `/graphql` — not available
- `/health`, `/status`, `/docs` — not available

## Expected Response Schema (Inferred from Platform)

Successful responses likely contain:
```json
{
  "ok": { /* data */ },
  "success": true,
  "error": null
}
```

Per-agent data likely includes:
- Agent/project: name, symbol, twitter handle, contract address, chain
- Mindshare: percentage, rank, 3d/7d change
- Sentiment: bullish/bearish/neutral scores
- On-chain: market cap, holder count, volume, liquidity
- Social: tweet count, engagement rate, smart followers
- Token: price, price change

## Related Products (NOT the Same API)

| Product | URL | Purpose | Relevance |
|---------|-----|---------|-----------|
| Cookie3 Analytics | app.cookie3.co | Web3 marketing analytics | Different product, $299-749/mo |
| Moltbook | api.moltbook.com | AI agent social network | Different data, documented API |
| Cookie DAO Docs | docs.cookie.community | Token/governance docs | No API docs |

## Alternative Data Sources

If Cookie.fun API access is blocked, alternatives for crypto social mindshare:

| Source | API Status | Cost | Coverage |
|--------|-----------|------|----------|
| **LunarCrush** | Public documented API | Free tier available | Social intelligence for crypto |
| **Santiment** | Public documented API | Free tier (limited) | On-chain + social metrics |
| **CoinGecko** | Public API | Free (30 calls/min) | Social stats, developer activity |
| **Cookie.fun web scraping** | Possible but risky | Free | Full platform data |

## CEO Decision Required

**Options:**

1. **Stake $COOKIE tokens** (~$300-500 one-time) to unlock API access
   - Pro: Full API access, aligned with "zero monthly spend"
   - Con: Token price risk, staking lockup period unknown

2. **Switch to LunarCrush** as B2 data source
   - Pro: Documented API, free tier, similar data
   - Con: Requires re-scoping B2

3. **Scrape cookie.fun website** directly
   - Pro: Free, data is visible on the web UI
   - Con: Fragile, anti-scraping risk (Cloudflare on web frontend), TOS violation risk

4. **Contact @St3cu** for enterprise API key
   - Pro: Full documentation access (swagger)
   - Con: Monthly cost ($499+), paid in crypto

## Recommendation

**Option 2 (LunarCrush)** or **Option 1 ($COOKIE staking)** are the most viable. LunarCrush has a free tier with documented endpoints — I can do a quick API discovery to compare feasibility. $COOKIE staking is a one-time cost but carries token price risk.
