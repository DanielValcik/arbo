# ARBO — Open Questions & Risks (Polymarket Pivot)

> Items that need resolution before or during Sprint 1. Marked with priority.
> Updated: 2026-02-21

---

## Questions for CEO (Blocking)

### Q1: MetaMask Wallet Setup [HIGH — SPRINT 1 BLOCKER]
**Status:** NEEDS ACTION
CEO must:
1. Install MetaMask browser extension
2. Create new wallet (store recovery phrase OFFLINE, never digital)
3. Fund with small amount of POL for gas (~$1 worth)
4. Login to Polymarket.com via MetaMask
5. Note the proxy wallet address shown on Polymarket
6. Export private key from MetaMask (Settings → Security → Export)
Provide: `POLY_PRIVATE_KEY`, `POLY_FUNDER_ADDRESS` (proxy wallet address)

### Q2: CEX Account for USDC Funding [HIGH — SPRINT 1 BLOCKER]
**Status:** NEEDS ACTION
Recommended: Kraken (supports Polygon USDC.e withdrawal, CZ-friendly KYC)
Alternative: Coinbase (also supports Polygon)
KYC takes 1-3 business days. Fund with €500 for initial paper→live transition.
Not needed for paper trading, but KYC should start now.

### Q3: Polymarket API Credentials [HIGH — SPRINT 1 BLOCKER]
**Status:** NEEDS ACTION after Q1
Once MetaMask + Polymarket login is done:
1. Use py-clob-client to derive L2 API credentials: `client.create_or_derive_api_creds()`
2. Store in .env: `POLY_API_KEY`, `POLY_SECRET`, `POLY_PASSPHRASE`
CTO can assist with the derivation script once private key is available.

### Q4: The Odds API Registration [MEDIUM]
**Status:** NEEDS ACTION
Register at the-odds-api.com (free tier = 500 req/month to start).
Provide: `ODDS_API_KEY`
Upgrade to $10/month tier (10K requests) after Sprint 1 if needed.

### Q5: Alchemy Account [MEDIUM]
**Status:** NEEDS ACTION
Register at alchemy.com (free tier sufficient for Layer 7 order flow monitoring).
Create Polygon Mainnet app → get WebSocket URL.
Provide: `ALCHEMY_KEY`

### Q6: Google AI Studio (Gemini API) [MEDIUM]
**Status:** NEEDS ACTION for Sprint 2
EEA (European Economic Area) requires paid tier from day 1.
Register at aistudio.google.com → enable billing → get API key.
Provide: `GEMINI_API_KEY`
Budget: ~€10-20/month (80K tokens/day across 9 layers)

### Q7: Hetzner VPS Status [MEDIUM]
**Status:** NEEDS ANSWER
Is the Hetzner CX22 from Matchbook era still available? If yes:
- Need to update Python packages, remove Redis, install fresh PostgreSQL schema
- Same SSH access, same IP
If not provisioned yet: need Ubuntu 24.04, Python 3.12, PostgreSQL 16

### Q8: Slack App Credentials [LOW — Already have from Sprint 2?]
**Status:** VERIFY
Check if .env has valid SLACK_BOT_TOKEN and SLACK_APP_TOKEN from the Matchbook era.
If yes, they carry over. If not, follow TD-009 setup instructions.

---

## Technical Risks

### R1: Polymarket Account Geo-Restrictions [HIGH]
**Risk:** While CZ is not on the blocked list (US, FR, BE, CH, PL, SG, AU, RO, HU, PT, UA, UK), Polymarket may change policy.
**Mitigation:** VPS in Falkenstein (DE) serves as backup access point. Monitor Polymarket TOS updates.
**Status:** Monitor.

### R2: py-clob-client SDK Stability [MEDIUM]
**Risk:** py-clob-client updates frequently (v0.34.6 as of Feb 2026). Breaking changes possible.
**Mitigation:** Pin exact version `py-clob-client==0.34.6`. Test thoroughly before upgrading. Wrapper class isolates SDK from business logic.
**Status:** Pin in pyproject.toml.

### R3: Polymarket Builder Tier Limits [MEDIUM]
**Risk:** Unverified accounts limited to 100 relayer requests/day. This constrains live trading volume.
**Mitigation:** Apply for Verified tier (3000/day) at builder@polymarket.com early. Not needed for paper trading.
**Status:** Apply during Sprint 2.

### R4: WebSocket Heartbeat Reliability [MEDIUM]
**Risk:** Polymarket auto-cancels all open orders on WebSocket disconnect. If heartbeat fails, we lose all positions.
**Mitigation:** Robust reconnect logic with exponential backoff. Heartbeat PING every 10s (strict). Health monitoring with Slack alerts on disconnect.
**Status:** Core part of PM-001 implementation.

### R5: Polygon RPC Free Tier Limits [LOW]
**Risk:** Alchemy free tier may not handle high-volume OrderFilled event streaming for Layer 7.
**Mitigation:** Start with free tier, monitor. Alchemy allows 300M compute units/month free. If insufficient, upgrade ($0/month for Growth tier with 40M CU/day).
**Status:** Monitor during Sprint 2.

### R6: The Odds API Quota Exhaustion [MEDIUM]
**Risk:** Free tier = 500 requests/month. With 5-minute polling for value model, we consume ~8640 req/month.
**Mitigation:** Start on free tier for development. Upgrade to $10/month (10K req) for Sprint 2 value model. Implement request counter with hard stop.
**Status:** Budget approved in brief (€10-50/month).

### R7: XGBoost Training Data Availability [MEDIUM]
**Risk:** Value model needs 200+ resolved markets with both Pinnacle odds and Polymarket prices. Historical data may be limited.
**Mitigation:** Start collecting Pinnacle ↔ Polymarket matched data from Sprint 1 (scanner). Supplement with The Odds API historical endpoint ($10 per request, use sparingly).
**Status:** Data collection starts Sprint 1 via PM-005.

### R8: Whale Wallet Detection Accuracy [LOW]
**Risk:** Polymarket leaderboard data may not accurately represent current whale activity. Wallets may use multiple addresses.
**Mitigation:** Cross-reference with on-chain data (Layer 7). Require minimum 50 resolved positions and >$50K volume. Update weekly.
**Status:** Sprint 3.

---

## Decisions Needed (Not Blocking)

### D1: Python 3.12 vs 3.13
**Recommendation:** Target 3.12 as minimum, test on 3.13. All dependencies support both.
**Decision:** Use what's installed on VPS (likely 3.12 on Ubuntu 24.04).

### D2: Polymarket Builder Tier Application
**Recommendation:** Apply for Verified tier in Sprint 2 (when we start shadow mode MM bot).
**Decision:** Pending — depends on trading volume needs.

### D3: Chroma DB Deployment
**Options:**
  a) In-process (Python client, SQLite backend) — simplest
  b) Standalone Chroma server on VPS — more robust
**Recommendation:** In-process for Sprint 3. Migrate to standalone if performance issues.
**Decision:** In-process.
