







**ARBO**

Reflexive Decay Harvester

Automated Trading System for Polymarket


Strategy Reference Document v1.0

25 February 2026

**Classification: CONFIDENTIAL — Owner + CEO Only**






Target Capital: €2,000  |  Monthly Budget: €50–60  |  Projected Return: 8–18% monthly

1\. Executive Summary

Arbo is an automated trading system targeting Polymarket, a decentralized prediction market on the Polygon blockchain using USDC collateral. After extensive market research analyzing 72 million trades (Becker study on Kalshi) and 86 million bets (IMDEA study on Polymarket), we identified a novel strategic framework we call the **Reflexive Decay Harvester**.

The core innovation is treating prediction market mispricing not as a static snapshot, but as a **dynamic process unfolding over time**. We exploit three distinct temporal patterns: (1) the systematic decay of optimism premium on longshot contracts, (2) reflexive boom-bust cycles in the new Attention Markets category, and (3) compound resolution chaining for maximum capital turnover on weather markets.

This framework does not require sub-millisecond latency, massive capital, or 9 independent strategy layers. It requires intelligence, timing, and information asymmetry — which is where our edge lies.

**Parameter**

**Value**

**Notes**

Target Capital

€2,000 (~$2,100)

Minimum viable for this strategy

Monthly Infra Budget

€50–60

AWS + APIs + gas

Monthly Return Target

8–18%

Projection, NOT guarantee

First Possible Live Trade

May 2026

After 4-week paper validation

Risk Limits

5%/10%/20%

Position / daily loss / weekly loss

2\. Theoretical Foundations

2\.1 The Becker Study: Prediction Markets as Wealth Transfer Systems

Source: Becker et al. (2025), analysis of 72.1 million trades, $18.26 billion volume on Kalshi. The single most important finding for our strategy:

**Prediction markets are NOT efficient.** They are two-population wealth transfer systems where takers (retail) systematically lose money to makers (professional counterparties).

Key Data Points

`	`**•	Taker average excess return:** -1.12% (lose money systematically)

`	`**•	Maker average excess return:** +1.12% (profit from being counterparty to biased flow)

`	`**•	YES contracts at longshot prices:** systematically overpriced (-41% EV at 1¢ YES vs +23% EV at 1¢ NO)

`	`**•	Maker YES vs NO returns:** nearly identical (Cohen's d = 0.02–0.03) — makers don't profit from prediction, they profit from structure

Category Efficiency Gaps (Becker Data)

**Category**

**Efficiency Gap (pp)**

**Implication**

Finance

0\.17

Near-perfect efficiency — avoid

Sports

2\.23

Moderate opportunity

Crypto

2\.69

Good opportunity

Entertainment

4\.79

Strong opportunity

Media

7\.32

Strongest inefficiency

**Critical Insight:** The “Optimism Tax” — retail traders systematically overpay for affirmative (YES) outcomes, especially at longshot prices. This is not a temporary anomaly; it is a structural feature of prediction markets driven by behavioral bias (people prefer buying hope of gain over selling insurance against loss).

2\.2 Glosten-Milgrom Model and Information Thermodynamics

The Glosten-Milgrom (1985) model explains market maker spread setting: a MM sets bid/ask to compensate losses from informed traders with gains from uninformed traders. On Polymarket, this has a specific implication:

`	`**•	Volatility is measurable and predictable** (we can assess how uncertain an outcome is)

`	`**•	The actual outcome is NOT predictable** (nobody knows the future)

`	`**•	Edge comes from structure, not prediction** (being maker, not taker; selling premium, not buying hope)

A paper from the Max Planck Institute established an analogy between the Glosten-Milgrom model and Szilárd's information engine from thermodynamics. The key result: maximum profit of an informed trader is bounded by the amount of information they possess multiplied by “market temperature” (analogous to T × entropy). This gives us a quantitative framework for position sizing — we don't trade markets where our information edge is small (low profit per trade) or where market temperature is low (efficient markets).

2\.3 Soros Reflexivity Theory

George Soros's Theory of Reflexivity states that in social systems, participants' perceptions influence the reality they observe, creating feedback loops. In financial markets: prices affect fundamentals, and fundamentals affect prices. This creates boom-bust cycles that are systematically exploitable.

This theory becomes directly tradeable in Polymarket's new Attention Markets (launching March 2026), where:

`	`•	High price of a “mindshare” contract signals that a topic is important

`	`•	People see the signal and discuss the topic more (social media amplification)

`	`•	Increased discussion raises actual mindshare (Kaito AI metric)

`	`•	Rising mindshare validates the high price, attracting more buyers

`	`•	Cycle continues until mindshare peaks and price overshoots reality

`	`•	Bust: mindshare declines, price corrects violently

**Key:** Unlike traditional financial markets where reflexivity is hard to measure, in Attention Markets the fundamental (Kaito mindshare metric) is publicly observable in real time. We can detect the divergence between price and reality precisely.

3\. Strategies That Do NOT Work (and Why)

Before describing our approach, it is critical to document what we eliminated and why. This prevents revisiting dead ends.

**Strategy**

**Status**

**Reason for Elimination**

Binary Arb (YES+NO sum)

DEAD

Spreads collapsed from 4.5% (2023) to 1.2% (2025). Professional MM bots close gaps in milliseconds. Not viable at any capital level.

Temporal Crypto Arb (taker)

DEAD

500ms taker delay removed 18 Feb 2026 without warning. Taker fees (max 3.15% at p=0.50) now exceed exploitable spread. The $313→$414K/month bot strategy is no longer viable.

Large-Scale Market Making

LIMITED

€2K insufficient vs professional MM with $10K+. Maker rebates are discretionary and declining. Sources cite $10K minimum for meaningful MM returns.

Anti-Longshot NO Selling

RISKY

Profitable at scale (Hans323: $92K bet, 8% win rate → $1.1M) but requires large bankroll to survive variance. Not viable at €2K.

9-Layer Confluence Scoring

DEPRECATED

Original architecture. After 2.5 hours: 37,556 L2 signals, 0 confluence matches. Too many layers with too few signals = zero trades. Replaced by single-signal quality gates.

Sub-Second Latency Strategies

NOT US

Arbo's edge is intelligence + information speed, NOT network latency. 20ms to London/Polygon RPC is fine. HFT is not our game.

4\. Strategy Architecture: Three Pillars

The Reflexive Decay Harvester consists of three independent strategies, each targeting a different temporal pattern in Polymarket pricing. They share infrastructure but operate independently — failure of one does not affect the others.

**Strategy**

**Capital**

**Allocation**

**Monthly Target**

**Turnover**

**A: Theta Decay**

€400

20%

5–15%

3–5×

**B: Reflexivity Surfer**

€400

20%

10–30%

5–10×

**C: Compound Weather**

€1,000

50%

8–15%

20–30×

**Reserve**

€200

10%

0%

—

4\.1 Strategy A: Theta Decay on Longshot Markets

**Core Concept:** Systematically sell optimism premium on longshot YES contracts, timed to peak optimism moments. This is the prediction market equivalent of options theta decay.

4\.1.1 Theoretical Basis

Becker's data proves YES longshot contracts are systematically overpriced. But this overpricing is not constant — it is largest at market creation (when uncertainty is highest and the Optimism Tax is strongest) and decays toward resolution as information becomes concrete and prices converge to reality.

This is directly analogous to theta decay in options: options lose time value as expiration approaches. On Polymarket, longshot YES contracts lose their “optimism premium” as resolution approaches. Nobody frames it this way, and nobody systematically exploits it.

4\.1.2 Entry Signal: Peak Optimism Detection

We do NOT enter immediately when a longshot YES is overpriced. We wait for **peak optimism** — the moment when taker flow is most YES-biased. Detection method:

`	`•	Monitor on-chain OrderFilled events on CTF Exchange (0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E)

`	`•	Track taker YES vs taker NO flow ratio per market (rolling 4-hour window)

`	`•	Calculate z-score against 7-day historical average for that market

`	`•	When taker YES flow exceeds 3σ of historical average → PEAK OPTIMISM → entry signal

`	`•	Buy NO (equivalent to selling overpriced YES) at current market price

4\.1.3 Exit Rules

`	`**•	Primary exit:** Hold to resolution. Maximum profit = $1.00 minus entry price.

`	`**•	Partial exit:** If NO price rises >50% from entry before resolution, take profit on 50% of position.

`	`**•	Stop loss:** If NO price drops >30% from entry, exit entire position. This indicates our timing was wrong.

4\.1.4 Market Selection Criteria

`	`**•	Category:** Sports (2.23pp gap), Crypto (2.69pp), Entertainment (4.79pp), Media (7.32pp). AVOID Finance (0.17pp).

`	`**•	YES price:** < $0.15 (longshot territory where Optimism Tax is strongest)

`	`**•	Market age:** > 24 hours (avoid initial price discovery volatility)

`	`**•	Time to resolution:** 3–30 days (enough time for decay, not so long capital is locked)

`	`**•	Liquidity:** > $10K total traded volume (ensures we can exit if needed)

`	`**•	Fee status:** Fee-free markets ONLY (most Polymarket markets as of Feb 2026)

4\.1.5 Position Sizing

`	`**•	Per position:** $20–50 (1–2.5% of portfolio)

`	`**•	Max concurrent positions:** 10 ($500 max exposure = 25% of allocated capital)

`	`**•	Sizing method:** Quarter-Kelly based on historical win rate and average payoff

`	`**•	Expected win rate:** ~75–85% (based on Becker longshot data: NO at p>0.85 wins >80% of time)

4\.1.6 Projected Returns

**Metric**

**Conservative**

**Mid**

**Optimistic**

Monthly trades

15

30

50

Win rate

70%

80%

85%

Avg profit/win

$5

$8

$12

Avg loss/loss

$12

$10

$10

**Monthly P&L**

$-1.50

+$32

+$58.50

**% on €400**

-0.4%

+7.6%

+13.9%

**Note:** Conservative scenario is near break-even. This strategy requires backtest validation on IMDEA historical dataset (86M bets) before live deployment. If backtest shows <70% win rate on longshot NO, we reallocate capital to Strategy C.



4\.2 Strategy B: Reflexivity Surfer (Attention Markets)

**Core Concept:** Identify and trade reflexive feedback loops in Polymarket's new Attention Markets category (launching March 2026 in partnership with Kaito AI). This is a first-mover opportunity — the market category does not yet exist.

4\.2.1 What Are Attention Markets?

Announced February 10, 2026 by Polymarket and Kaito AI. Attention Markets allow users to wager on social media metrics — specifically:

`	`**•	Mindshare:** Volume and velocity of mentions across X, TikTok, Instagram, YouTube. Quantified by Kaito AI.

`	`**•	Sentiment:** Whether discussion tone is positive or negative. Also quantified by Kaito AI.

Example markets: “Will Anthropic's mindshare surpass OpenAI's next month?” or “Will sentiment toward Elon Musk improve this month?”

Pilot markets (Nov 2025) already attracted >$1.3M in volume. Rollout: dozens of markets in early March 2026, scaling to thousands by year-end. Initial focus: AI topics, then entertainment, global events.

4\.2.2 Why Reflexivity Applies Here

Attention Markets have a unique property that traditional prediction markets lack: **the act of trading can influence the outcome.**

If a trader buys YES on “Will Polymarket's mindshare exceed 80%?”, they have a financial incentive to promote Polymarket on social media — which directly increases the mindshare metric that determines resolution. This creates a textbook Sorosian reflexive feedback loop.

**The Four Phases of an Attention Market Cycle:**

**Phase**

**Name**

**What Happens**

**Our Action**

1

Start

New market opens. Price near fundamental (current Kaito mindshare).

Monitor. No action yet.

2

Boom

Traders buy YES → incentive to promote topic → mindshare grows → price rises → more buyers.

Buy YES. Ride the reflexive wave.

3

Peak / Overextension

Price outpaces real mindshare by >20%. Divergence = signal.

Sell YES. Buy NO.

4

Bust

Mindshare plateaus/declines. Price corrects to reality.

Hold NO to resolution.

4\.2.3 Phase Detection: Price vs Reality Divergence

The critical technical challenge is detecting phase transitions. Our approach:

`	`•	Fetch real-time Kaito mindshare metric via Kaito API (will be publicly integrated into their platform)

`	`•	Fetch current Polymarket contract price for the same metric

`	`•	Calculate divergence: (PM\_price − Kaito\_actual) / Kaito\_actual

`	`•	Divergence > +20% = Phase 3 entry (sell YES, buy NO)

`	`•	Divergence < −10% = Phase 1–2 (buy YES if positive momentum confirmed over 24h)

**Data integrity:** Kaito's metrics are validated by Brevis zero-knowledge proofs and EigenCloud verification. This means the resolution data is auditable and tamper-resistant, reducing oracle manipulation risk.

4\.2.4 Position Sizing and Risk

`	`**•	Phase 2 (Boom) positions:** $10–20, max 5 concurrent. These are momentum trades — higher risk.

`	`**•	Phase 3–4 (Bust) positions:** $20–50, max 5 concurrent. These are mean-reversion trades — higher conviction.

`	`**•	Total max exposure:** $350 (87.5% of allocated €400)

`	`**•	Stop loss:** 15% on Phase 2 positions (tight). 25% on Phase 3–4 positions.

4\.2.5 Risks Specific to This Strategy

`	`**•	Reflexive dynamics may be weak:** If after 4 weeks of observation, price-reality divergence never exceeds 10%, the reflexive loop isn't strong enough. Reallocate capital to C.

`	`**•	Low initial liquidity:** New market category. May have wide spreads initially. Start with smallest position sizes ($10) and scale up.

`	`**•	Kaito API access:** API must be publicly available for automated trading. If gated/paid, reassess cost-benefit.

`	`**•	Regulatory risk:** Attention Markets blur lines between gambling and derivatives. Czech law (Act 186/2016) allows online betting, but this is new territory.

4\.2.6 Projected Returns

**Metric**

**Conservative**

**Mid**

**Optimistic**

Monthly trades

8

20

35

Win rate

50%

60%

70%

Avg profit/win

$10

$15

$20

Avg loss/loss

$8

$10

$10

**Monthly P&L**

+$8

+$100

+$385

**% on €400**

+1.9%

+23.8%

+91.7%

**Note:** This is the highest-variance strategy. The optimistic scenario is deliberately extreme to illustrate upside. Realistic expectation is between conservative and mid. The strategy's value is asymmetric: limited downside (€400 max loss), potentially outsized upside if reflexive dynamics are strong.



4\.3 Strategy C: Compound Weather Resolution Chaining

**Core Concept:** Exploit the proven weather market mispricing using NOAA/Met Office forecast data, with a compound resolution chaining approach that maximizes capital turnover. This is our safest, most proven strategy and receives 50% of capital.

4\.3.1 Proven Track Record

Documented profitable weather traders on Polymarket:

**Trader**

**Profit**

**Deposit**

**Method**

gopfan2

$2,000,000+

Unknown

Buy YES <$0.15, NO >$0.45. NOAA data. Thousands of $1 micro-bets. 9,000%+ return.

meropi

$30,000

Small

Automated $1–3 bets. Some $0.01 shares → 500× payoff.

1pixel

$18,500

$2,300

NYC/London weather only. 800% return.

neobrother

$20,000+

Unknown

Temperature laddering: parallel bets across multiple temp ranges.

Anonymous A

$23,000

$1,000

London weather. 2,300% return since Apr 2025.

Anonymous B

$65,000

Unknown

NYC, London, Seoul.

**Critical:** Trader “1pixel” started with $2,300 (almost exactly our €2K) and turned it into $18,500. This is our closest benchmark.

4\.3.2 Why Weather Markets Work

`	`**•	Fee-free:** No taker fees on weather markets (unlike 5min/15min crypto, NCAAB, Serie A).

`	`**•	Deterministic resolution:** NOAA/Weather Underground/Met Office data. No subjectivity, no UMA oracle risk.

`	`**•	85–90% forecast accuracy:** NOAA 1–2 day temperature forecasts are 85–90% accurate. We are betting on science vs retail guessing.

`	`**•	Daily settlement:** Markets resolve every day. This is the key to high capital turnover.

`	`**•	Low competition:** Retail traders guess. We use meteorological data. Information asymmetry is massive.

`	`**•	Small capital viable:** $1–5 per trade, thousands of trades. No minimum size barrier.

4\.3.3 The Compound Chaining Innovation

Most weather bots simply bet on tomorrow's weather. Our innovation is **compound resolution chaining** — systematically routing capital through sequential daily resolutions across multiple cities:

**Day 1:**

`	`•	08:00 UTC: NOAA forecast for NYC tomorrow = 43°F

`	`•	NYC “40–45°F” bucket trading at $0.15 (15% implied probability)

`	`•	Buy 200 shares at $0.15 = $30 invested

`	`•	NYC resolves: actual temp 43°F → shares pay $1.00 each → $200 payout

`	`•	Profit: $170 (567% return on trade)

**Day 2:**

`	`•	$200 (original $30 + $170 profit) immediately deployed to London weather

`	`•	Met Office forecast: 8°C. “7–10°C” bucket at $0.20

`	`•	Buy 1,000 shares at $0.20 = $200 invested

`	`•	London resolves: 8°C → $1,000 payout

`	`•	Profit: $800 (400% return)

**Day 3:** 

`	`•	$1,000 deployed across Chicago, Seoul, Buenos Aires...

**Key insight:** With 5 cities and daily resolution, €1,000 allocated capital effectively becomes €5,000+ annualized turnover because capital is never idle for more than 24 hours. The 1pixel trader demonstrated this exact pattern.

4\.3.4 Cities and Data Sources

**City**

**Data Source**

**API**

**Cost**

New York City

NOAA (NWS)

api.weather.gov

Free

Chicago

NOAA (NWS)

api.weather.gov

Free

London

Met Office

datahub.metoffice.gov.uk

Free tier

Seoul

KMA / Open-Meteo

open-meteo.com

Free

Buenos Aires

SMN / Open-Meteo

open-meteo.com

Free

4\.3.5 Entry Rules

`	`•	Fetch forecast from primary data source for target city (1–2 day horizon)

`	`•	Identify the temperature bucket on Polymarket matching the forecast

`	`•	Check bucket price: entry ONLY if price < $0.20 (i.e., market assigns ≤20% probability to the correct outcome)

`	`•	Verify liquidity: minimum $500 total traded volume on the market

`	`•	Place order: $1–10 per trade depending on confidence and capital available

4\.3.6 Exit Rules

`	`**•	Primary:** Hold to resolution (24h). Daily settlement means we get our money back + profit every day.

`	`**•	Early exit:** If price rises above $0.50 before resolution, sell 50% to lock in guaranteed profit.

`	`**•	Forecast bust protection:** Max $10 per individual trade. Max 20 concurrent trades = $200 max total weather exposure (20% of C allocation).

4\.3.7 Temperature Laddering (neobrother Method)

For high-uncertainty days (forecast range spans 2+ buckets), deploy capital across adjacent buckets:

`	`•	Forecast: 42°F ±3°F → could be 39–45°F

`	`•	Buy: “35–40°F” bucket at $0.10 ($5), “40–45°F” bucket at $0.15 ($10), “45–50°F” bucket at $0.08 ($3)

`	`•	Total invested: $18. One bucket will pay $1.00 per share. Expected return: positive regardless of which bucket hits.

`	`•	This hedges forecast uncertainty while maintaining positive EV across the range.

4\.3.8 Projected Returns

**Metric**

**Conservative**

**Mid**

**Optimistic**

Daily trades

5

15

30

Win rate

75%

83%

88%

Avg profit/win

$3

$5

$8

Avg loss/loss

$3

$4

$5

**Monthly P&L**

+$56

+$135

+$354

**% on €1,000**

+5.6%

+13.5%

+35.4%



5\. Combined Portfolio Projections

**Metric**

**Conservative**

**Mid**

**Optimistic**

Strategy A (Theta Decay)

-$1.50

+$32

+$58.50

Strategy B (Reflexivity)

+$8

+$100

+$385

Strategy C (Weather)

+$56

+$135

+$354

**TOTAL MONTHLY**

**+$62.50**

**+$267**

**+$797.50**

**% ON €2,000**

**+3.0%**

**+12.7%**

**+38.0%**

**Realistic expectation:** Mid scenario (+12.7% monthly, ~$267/month) aligns with our 8–18% target range. Conservative scenario (3%) covers infrastructure costs. Optimistic scenario is possible but not plannable.

**These are projections, not promises.** Prediction markets have variance. We manage risk through position sizing, diversification across three independent strategies, and mandatory paper trading validation.

6\. Risk Management Framework

6\.1 Hardcoded Risk Limits (Non-Negotiable)

**Limit**

**Value**

**Enforcement**

Max single position

5% of portfolio

Hardcoded in risk manager. No override.

Max daily loss

10% of portfolio

Auto-shutdown. Resume next day.

Max weekly loss

20% of portfolio

Auto-shutdown. CEO review required.

Paper trading requirement

4 consecutive weeks

Positive P&L each week. No exceptions.

Max concurrent positions

30 total

10 per strategy maximum.

Reserve capital

10% (€200)

Never deployed. Emergency buffer only.

6\.2 Strategy-Specific Risks and Mitigations

Risk 1: Weather Market Efficiency Increases

As more bots enter weather markets, spreads will compress and mispricing will decrease.

`	`**•	Probability:** Medium (6–12 months timeline)

`	`**•	Impact:** Reduced returns on Strategy C

`	`**•	Mitigation:** Diversify across 5+ cities. Monitor spread compression monthly. If average entry price rises above $0.30, shift capital to Strategy A or new markets.

`	`**•	Pivot plan:** Expand to wind speed, precipitation, humidity markets if temperature becomes efficient.

Risk 2: Polymarket Rule Changes (Fees, Rewards)

Polymarket changed fee structure on 18 Feb 2026 without warning, killing temporal arb strategies overnight.

`	`**•	Probability:** High (has already happened)

`	`**•	Impact:** Could affect any strategy

`	`**•	Mitigation:** Modular architecture. Each strategy is independent. If one market type gets fees, we pivot capital within 24 hours.

`	`**•	Monitor:** Polymarket Discord, Twitter, and docs.polymarket.com daily for announcements.

Risk 3: NOAA / Met Office Forecast Busts

Weather forecasts are 85–90% accurate, not 100%. 10–15% of trades will lose due to forecast error.

`	`**•	Probability:** Certain (built into model)

`	`**•	Impact:** Individual trade losses

`	`**•	Mitigation:** Max $10 per position. Temperature laddering hedges across adjacent buckets. Portfolio-level risk: max 20 concurrent weather trades = $200 exposure.

Risk 4: Attention Markets Lack Reflexive Dynamics

Our core thesis on Strategy B may be wrong — reflexive feedback loops may not manifest strongly enough to trade.

`	`**•	Probability:** Medium

`	`**•	Impact:** Strategy B returns near zero

`	`**•	Mitigation:** Only 20% capital allocated. 4-week observation period before any live trades. If price-reality divergence never exceeds 10%, reallocate to Strategy C.

Risk 5: Heartbeat Disconnect (Market Making)

Polymarket auto-cancels all orders after 15 seconds without heartbeat ping.

`	`**•	Probability:** Low (AWS Lightsail 99.9%+ uptime)

`	`**•	Impact:** Open MM positions at risk if orders cancel during volatility

`	`**•	Mitigation:** 5-second heartbeat interval. Auto-reconnect logic. Conservative position sizing. This is mainly relevant if we add MM later.

Risk 6: UMA Oracle Manipulation

March 2025: UMA whale with 25% voting power manipulated $7M market resolution. Polymarket has since introduced MOOV2 (whitelisted proposers) but structural risk remains.

`	`**•	Probability:** Low for weather markets (objective resolution), Medium for political/subjective markets

`	`**•	Impact:** Complete loss on affected positions

`	`**•	Mitigation:** Focus on objectively resolvable markets (weather, crypto prices). Avoid politically subjective markets. No single position >5% of portfolio.



7\. Technical Implementation Requirements

7\.1 Infrastructure

`	`**•	VPS:** AWS Lightsail London, 99.9%+ uptime. 20ms to Polygon RPC.

`	`**•	Runtime:** Python 3.11+ with FastAPI

`	`**•	Blockchain:** Polygon RPC free tier. Polygon gas: ~$0.007/tx.

`	`**•	Auth:** Two-phase Polymarket auth — L1 (EIP-712 private key signing) → L2 (HMAC-SHA256 API key).

`	`**•	Client:** py-clob-client (post-only orders, batch up to 15 per call)

`	`**•	Dashboard:** arbo.click — real-time P&L, strategy breakdown, heartbeat monitoring

7\.2 Data Sources and APIs

**Source**

**URL**

**Cost**

**Purpose**

NOAA (NWS)

api.weather.gov

Free

NYC, Chicago forecasts

Met Office

datahub.metoffice.gov.uk

Free tier

London forecasts

Open-Meteo

open-meteo.com

Free

Seoul, Buenos Aires

Kaito AI

TBD (March 2026)

TBD

Mindshare/sentiment

Polymarket CLOB

clob.polymarket.com

Free

Order placement

Polymarket Gamma

gamma-api.polymarket.com

Free

Market discovery

Polygon RPC

polygon-rpc.com

Free tier

On-chain events

The Odds API

the-odds-api.com

$10–50/mo

Pinnacle odds (L2)

7\.3 Key Smart Contracts

`	`**•	CTF Exchange:** 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E — all OrderFilled events for binary markets

`	`**•	NegRisk CTF Exchange:** 0xC5d563A36AE78145C45a50134d48A1215220f80a — multi-outcome markets

`	`**•	CTF (Gnosis):** Conditional Tokens Framework — ERC-1155 token standard for outcome tokens

`	`**•	NegRiskAdapter:** Handles token conversion in multi-outcome markets

7\.4 Codebase Structure

Target file organization for CTO implementation:

**arbo/**

`  `strategies/weather\_specialist.py      # Strategy C: compound weather bot

`  `strategies/theta\_decay.py             # Strategy A: longshot NO selling

`  `strategies/reflexivity\_surfer.py      # Strategy B: Attention Markets

`  `connectors/noaa\_api.py                # NOAA forecast integration

`  `connectors/met\_office\_api.py          # Met Office UK data

`  `connectors/open\_meteo\_api.py          # Seoul, Buenos Aires data

`  `connectors/kaito\_api.py               # Kaito AI mindshare/sentiment

`  `connectors/polymarket\_clob.py         # CLOB order management

`  `connectors/polygon\_events.py          # On-chain event monitoring

`  `core/risk\_manager.py                  # Singleton risk manager

`  `core/heartbeat.py                     # WebSocket keepalive (5s)

`  `core/position\_tracker.py              # Cross-strategy position mgmt

`  `core/sizing\_engine.py                 # Kelly + info-theoretic sizing

`  `monitoring/dashboard\_feed.py          # arbo.click data feed

`  `monitoring/taker\_flow\_monitor.py      # Peak optimism detection

`  `config/settings.yaml                  # Strategy params, thresholds

7\.5 Monthly Infrastructure Cost

**Item**

**Cost/month**

**Notes**

AWS Lightsail London

$10–20

2GB RAM instance

The Odds API

$10–30

Pinnacle odds for L2 value signals

Polygon gas

$2–5

~$0.007/tx × ~500 tx/month

Domain (arbo.click)

~$1

Annual ÷12

Gemini 2.0 Flash (if used)

$2–5

$0.10/$0.40 per MTok

**TOTAL**

**$25–61**

Within €50–60 budget



8\. Implementation Timeline

Sprint 1 (Weeks 1–3): Foundation + Strategy C

`	`•	Polymarket auth integration (L1 + L2)

`	`•	py-clob-client setup with post-only orders

`	`•	NOAA + Met Office + Open-Meteo API connectors

`	`•	Weather market scanner (Gamma API filter)

`	`•	Weather bot core logic: forecast → bucket matching → entry/exit

`	`•	Risk manager singleton with hardcoded limits

`	`•	Paper trading mode for Strategy C

`	`•	Dashboard: weather P&L tracking

Sprint 2 (Weeks 4–6): Strategy A + On-Chain Monitoring

`	`•	Polygon event monitor (OrderFilled parsing)

`	`•	Taker flow analysis (YES/NO ratio, z-score calculation)

`	`•	Peak optimism detection engine

`	`•	Theta Decay strategy: market selection + entry/exit logic

`	`•	Paper trading mode for Strategy A

`	`•	Backtest engine on IMDEA historical data

Sprint 3 (Weeks 7–9): Strategy B + Attention Markets

`	`•	Kaito API integration (dependent on March 2026 launch)

`	`•	Price vs reality divergence calculator

`	`•	Reflexivity phase detection (4-phase model)

`	`•	Reflexivity Surfer strategy logic

`	`•	Paper trading mode for Strategy B

`	`•	Cross-strategy position manager

Sprint 4 (Weeks 10–12): Integration + Paper Validation

`	`•	All three strategies running concurrently in paper mode

`	`•	Capital allocation engine

`	`•	Compound resolution chaining (Strategy C optimization)

`	`•	Full dashboard: all strategies, risk metrics, P&L breakdown

`	`•	Performance analytics: win rate, Sharpe ratio, drawdown

Weeks 13–16: Mandatory Paper Trading Validation

`	`•	4 consecutive weeks of positive combined P&L required

`	`•	Each strategy validated independently

`	`•	CEO (me) reviews weekly metrics and approves/rejects go-live

`	`•	If any week is negative, clock resets

Week 17+: Live Trading (Earliest: May 2026)

`	`•	Start with Strategy C only (most proven)

`	`•	Add Strategy A after 1 week of live C performance

`	`•	Add Strategy B after 2 weeks if Attention Markets have sufficient liquidity

`	`•	Scale up gradually: 50% capital first week, 75% week 2, 100% week 3



9\. Polymarket Technical Reference

Quick reference for development team. All verified as of February 2026.

9\.1 API Endpoints

`	`**•	CLOB API:** https://clob.polymarket.com

`	`**•	Gamma API:** https://gamma-api.polymarket.com

`	`**•	WebSocket:** wss://ws-subscriptions-clob.polymarket.com

9\.2 Fee Structure (Post-18 Feb 2026)

`	`**•	Most markets:** 0% fee (sports, politics, weather, etc.)

`	`**•	Fee-enabled markets:** 5min crypto, 15min crypto, NCAAB, Serie A

`	`**•	Fee formula:** p × (1-p) × FEE\_RATE

`	`**•	Max effective rate:** 1.56% at midpoint (p=0.50), drops to ~0.06% at p=0.05

`	`**•	Maker rebates:** Funded by taker fees. Percentage is DISCRETIONARY (varies at Polymarket's discretion).

9\.3 Heartbeat Requirements

`	`**•	Ping interval:** Every 5 seconds

`	`**•	Timeout:** 10 seconds + 5 second buffer = 15 seconds max

`	`**•	Effect:** ALL open orders auto-cancelled on disconnect

`	`**•	Requirement:** VPS uptime 99.9%+ mandatory for market making

9\.4 Order Mechanics

`	`**•	Batch orders:** Up to 15 per call via py-clob-client

`	`**•	Post-only:** Supported (ensures maker status, avoids taker fees)

`	`**•	Order types:** Limit orders (FOK, GTC, GTD)

9\.5 Collateral and Settlement

`	`**•	Collateral:** USDC.e on Polygon

`	`**•	Resolution:** UMA Optimistic Oracle (2h challenge period) or Chainlink Data Streams (for crypto price markets)

`	`**•	Token standard:** ERC-1155 (Gnosis Conditional Token Framework)

`	`**•	YES + NO = $1.00 always** (fully collateralized)

9\.6 Regulatory Status (Czech Republic)

`	`**•	Polymarket access:** Czech Republic NOT on blocked countries list. Full access confirmed.

`	`**•	Czech gambling law:** Act 186/2016 Sb. allows online betting.

`	`**•	Tax status:** Player winnings are tax-free under Czech law.

`	`**•	Polymarket US vs International:** CZ users access International version (UMA oracle resolution).



10\. Decision Log

All major strategic decisions documented for reference.

**Date**

**Decision**

**Rationale**

25 Feb 2026

Pivot from 9-layer confluence to 3-strategy Reflexive Decay Harvester

Confluence scoring produced 0 trades in 2.5h. 37,556 signals, 0 matches. Fundamentally flawed for €2K capital.

25 Feb 2026

Abandon temporal crypto arbitrage

500ms taker delay removed 18 Feb 2026. Taker fees now exceed spread. Strategy is dead.

25 Feb 2026

Weather markets as primary motor (50% capital)

Proven track record ($1K→$24K, $2.3K→$18.5K). Fee-free. NOAA 85-90% accuracy. Daily settlement.

25 Feb 2026

Attention Markets as innovation bet (20% capital)

First-mover opportunity. Reflexive dynamics are tradeable. Limited downside (€400 max). Launches March 2026.

25 Feb 2026

Theta Decay as structural edge (20% capital)

Becker data proves longshot YES overpricing. On-chain taker flow as novel timing signal. Requires backtest validation.

11\. Glossary

`	`**•	Attention Markets:** New Polymarket category (March 2026) where contracts resolve based on Kaito AI mindshare/sentiment metrics.

`	`**•	CLOB:** Central Limit Order Book. Polymarket's hybrid on/off-chain order matching system.

`	`**•	Compound Resolution Chaining:** Arbo's technique of immediately redeploying capital from a resolved market into the next available market, maximizing turnover.

`	`**•	CTF:** Conditional Token Framework (Gnosis). ERC-1155 standard for prediction market outcome tokens.

`	`**•	Kaito AI:** Decentralized information engine that quantifies social media mindshare and sentiment. Partner for Polymarket's Attention Markets.

`	`**•	Maker:** A trader who adds liquidity (posts limit orders). On Polymarket, makers may earn rebates.

`	`**•	Mindshare:** Kaito AI metric measuring volume and velocity of social media mentions for a topic.

`	`**•	NegRisk:** Polymarket's system for multi-outcome markets. Allows converting NO tokens of one outcome to YES tokens of others.

`	`**•	Optimism Tax:** Becker study term for the systematic overpricing of YES longshot contracts by retail traders.

`	`**•	Peak Optimism:** Arbo-defined moment when taker YES flow reaches 3σ above historical average, indicating maximum mispricing.

`	`**•	Quarter-Kelly:** Position sizing at 25% of full Kelly criterion. Reduces variance at cost of lower expected return.

`	`**•	Reflexive Decay Harvester:** Arbo's overall strategic framework combining theta decay, reflexivity trading, and compound weather chaining.

`	`**•	Reflexivity:** Soros theory that prices and fundamentals form feedback loops in social systems. Directly applicable to Attention Markets.

`	`**•	Taker:** A trader who removes liquidity (executes against existing orders). On fee-enabled markets, takers pay fees.

`	`**•	Temperature Laddering:** Technique of placing parallel bets across adjacent weather buckets to hedge forecast uncertainty.

`	`**•	Theta Decay:** The systematic reduction of optimism premium on longshot contracts as resolution approaches. Analogous to options time decay.

`	`**•	UMA:** Universal Market Access. Optimistic oracle used by Polymarket for market resolution. 2-hour challenge period.

12\. Key Sources and References

`	`•	Becker et al. (2025): 72.1M trade analysis on Kalshi. Optimism Tax, maker/taker dynamics.

`	`•	IMDEA Networks (Saguillo et al.): 86M bets on Polymarket. $40M arbitrage profits documented Apr 2024–Apr 2025.

`	`•	Glosten & Milgrom (1985): Bid-ask spread setting under asymmetric information.

`	`•	George Soros: “The Alchemy of Finance” (1987). Theory of Reflexivity.

`	`•	Max Planck Institute: Information thermodynamics analogy to Glosten-Milgrom model.

`	`•	Polymarket Documentation: docs.polymarket.com (fee structure, heartbeat, CTF, API).

`	`•	PolyMaster: Reverse-engineering of maker rebate formula (Jan 2026).

`	`•	defiance\_cr: Weather bot profitability documentation.

`	`•	Kaito AI + Polymarket announcement: 10 Feb 2026 (Attention Markets).

`	`•	Orochi Network: UMA oracle manipulation analysis (March 2025 incident).


*END OF DOCUMENT*

Arbo — Reflexive Decay Harvester — v1.0 — 25 February 2026
