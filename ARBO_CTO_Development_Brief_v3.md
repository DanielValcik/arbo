# ARBO — Polymarket Trading System
## CTO Development Brief v3.0

**Od:** CEO  
**Pro:** CTO & Development Team  
**Datum:** 21. února 2026  
**Klasifikace:** INTERNAL — CTO & Dev Team Only  
**Status:** ZÁVAZNÉ ZADÁNÍ — odchylky vyžadují CEO approval

---

## 0. Jak číst tento dokument

Tento brief je jediný zdroj pravdy pro vývoj. Obsahuje:

1. **Strategický kontext** — proč stavíme co stavíme
2. **9-vrstvá architektura** — kompletní systém s prioritami
3. **Technický stack** — přesné technologie, verze, endpointy
4. **Sprint plán** — 4 sprinty + paper trading validace, každý task s ID a acceptance testem
5. **Risk management** — hardcoded limity, NEMĚNITELNÉ
6. **Coding standards** — kvalitativní požadavky na každý PR
7. **Projektová struktura** — adresářová struktura, moduly

Každý task má:
- **ID** (PM-XXX) pro trackování
- **Popis** co se má udělat
- **Technické detaily** jak to implementovat
- **Acceptance test** — MUSÍ projít před merge. Pokud neprojde, task není done.

---

## 1. Strategický kontext

### Co je Arbo

Automatizovaný trading systém pro Polymarket — decentralizovaný predikční trh na Polygon blockchainu. Systém kombinuje 9 strategických vrstev pro generování edge přes informační asymetrii, statistické modely a on-chain analytiku.

### Proč Polymarket

- Původní plán (Matchbook betting exchange) selhal — nelze zřídit účet z ČR
- Polymarket nemá geo-blok pro ČR (blokované: US, FR, BE, CH, PL, SG, AU, RO, HU, PT, UA, UK)
- CLOB architektura (Central Limit Order Book) na Polygon = transparentní, programovatelná
- Většina sportovních marketů má 0% fee (únor 2026). Maker rebates aktivní.
- py-clob-client (oficiální Python SDK) — REST + WebSocket, batch orders až 15/call
- USDC collateral, gas ~$0.007/tx na Polygon

### Kritické informace

- **Pinnacle API uzavřeno** od 23.7.2025 pro veřejnost. Řešení: The Odds API ($10–50/měs) zahrnuje Pinnacle odds.
- **Binární arb (YES+NO) je mrtvá** — spready klesly z 4.5% (2023) na 1.2% (2025). MM boty uzavírají v milisekundách. NESTAVÍME jako profit vrstvu.
- **Attention Markets** — nový typ marketu (Polymarket × Kaito AI, oznámeno 10.2.2026). Obchodování na "mindshare" a sentiment sociálních sítí. Desítky nových marketů od března 2026.
- **On-chain transparentnost** — všechny obchody na Polygon blockchainu jsou veřejně viditelné (OrderFilled events z CTF Exchange kontraktu). To je naše výhoda pro smart money tracking.

---

## 2. Architektura — 9 vrstev

### Přehled

| # | Vrstva | Typ | Priorita | Měs. ROI | Riziko | Sprint |
|---|--------|-----|----------|----------|--------|--------|
| 1 | Market Making + Rebates | Quasi-mechanická | VYSOKÁ | 4–10% | Adverse selection | S1–S2 |
| 2 | Value Betting (Ensemble) | Statistická edge | VYSOKÁ | 5–10% | Model chyba | S2 |
| 3 | Semantic Market Graph | Input vrstva | INPUT | N/A | Detection miss | S3 |
| 4 | Whale Copy + Confluence | Info arbitrage | STŘEDNÍ | 3–8% | Whale error | S3 |
| 5 | Logická/Kombinatorická Arb | LLM sémantická | STŘEDNÍ | 1–3% | Resolution rules | S3 |
| 6 | Temporal Crypto Arb | Mechanická | STŘEDNÍ | 2–5% | Fee compression | S2 |
| 7 | Smart Money Order Flow | On-chain analytika | STŘEDNÍ | 2–4% | False positives | S2–S3 |
| 8 | Attention Markets | LLM sentiment | NIŽŠÍ | 2–6% | Data manipulation | S3 |
| 9 | Live Sports Latency | Data arbitrage | NIŽŠÍ | 3–8% | Arms race | S3 |

**Kombinovaný realistický target:** 8–12% měsíčně na €2K = €160–240. NESLIBUJEME — je to projekce založená na backtestech a analýze. Reálný výsledek závisí na tržních podmínkách.

### Vrstva 1: Market Making + Rebate Harvesting

**Mechanismus:** Symetrické limit ordery na obě strany (BUY YES + BUY NO) v marketech s širokým spreadem. Profitujeme ze spreadu + maker rebates.

**Targeting:**
- Spread > 4%, volume $1K–$50K/den, nízká volatilita
- PRIORITNĚ fee-enabled markety (15min crypto, NCAAB, Serie A) kde maker rebates aktivní
- PostOnly ordery garantují maker status = zero taker fee + USDC rebates

**Inventory management:** Max imbalance 60/40 mezi YES a NO. Při překročení — adjustovat ceny na jedné straně.

**Heartbeat:** Polymarket automaticky cancelluje ordery při disconnect. Implementovat heartbeat loop s reconnect logikou.

**Klíčové parametry v YAML config:**
```yaml
market_maker:
  min_spread: 0.04        # 4% minimum spread
  min_volume_24h: 1000    # $1K denní volume
  max_volume_24h: 50000   # $50K — nad tím příliš kompetitivní
  max_inventory_imbalance: 0.6  # 60/40
  order_size_pct: 0.025   # 2.5% kapitálu per strana
  heartbeat_interval_s: 30
  prefer_fee_markets: true # Priorita na markety s maker rebates
```

### Vrstva 2: Value Betting — Multi-Source Ensemble

**Mechanismus:** Porovnání Polymarket ceny s naším odhadem skutečné pravděpodobnosti. Obchodujeme když divergence > edge_threshold po fee.

**Ensemble model:**
1. **XGBoost** (statistický) — trénovaný na Pinnacle implied prob vs actual outcomes
2. **Gemini 2.0 Flash** (LLM) — analýza news, kontextu, kvalitativních faktorů → pravděpodobnost
3. **Polymarket historical accuracy** — historická přesnost trhu pro danou kategorii

Vážený průměr → final probability estimate. Trade když `abs(model_prob - market_prob) > edge_threshold + fee`.

**Bayesian updating:** Model se adjustuje v reálném čase s novými informacemi (odds change, news, whale movement).

**Klíčové parametry:**
```yaml
value_model:
  edge_threshold: 0.03     # 3% minimum edge po fee
  scan_interval_s: 300     # Každých 5 minut
  min_training_samples: 200
  xgboost_weight: 0.5
  llm_weight: 0.3
  historical_weight: 0.2
  kelly_fraction: 0.5      # Half-Kelly sizing
  max_position_pct: 0.05   # 5% cap per trade
```

**XGBoost features:**
- Pinnacle implied probability
- Polymarket midpoint
- Time to event (hours)
- League / category
- Historical volatility (price std 24h)
- Volume trend (24h vs 7d average)
- Number of outcomes in event

### Vrstva 3: Semantic Market Graph (INPUT vrstva)

**Mechanismus:** Vektor-based sémantický graf všech aktivních marketů. Identifikuje logické závislosti: subset/superset, vzájemné vyloučení, implikace, temporální závislosti.

**Implementace:**
- `e5-large-v2` embeddings model pro sémantické matchování
- Chroma DB pro vektorové úložiště
- O(n²) porovnání všech aktivních marketů (denní refresh, ~10K+ marketů)
- Výstup: relationship graph `{market_a, market_b, relationship_type, confidence}`

**Relationship types:**
- `SUBSET`: "Democrats win presidency" ⊂ "Democrats win popular vote" (ne nutně, ale korelace)
- `MUTEX`: "Trump wins" ⊕ "Biden wins" (v rámci jednoho eventu)
- `IMPLICATION`: "Fed cuts rates in March" → higher P("S&P above 5000 by June")
- `TEMPORAL`: "Team wins semifinal" before "Team wins final"

**Výstup feeduje:** Vrstvu 5 (Logická Arb) a Vrstvu 4 (Confluence scoring)

### Vrstva 4: Whale Copy + Multi-Signal Confluence

**Mechanismus:** Sledování top wallet adres na Polymarket. Když whale otevře/zvětší pozici → signál.

**Whale discovery:**
- Scrape Polymarket leaderboard: top 50 walletů dle ROI a volume
- Doplnit data z Polygonscan / Data API
- Filtry: win rate > 60%, ≥ 50 resolved pozic, volume > $50K
- Kategorizace: sports / politics / crypto specialist

**Whale monitoring:**
- Polling Data API každé 4 sekundy pro tracked wallety
- Detekce: nová pozice / zvětšená pozice / uzavřená pozice
- ≥ 2 whales ve stejném marketu = STRONG signal

**Confluence scoring — CENTRÁLNÍ ROZHODOVACÍ MECHANISMUS:**

Každý potenciální trade dostane skóre 0–5:

| Signal | Body |
|--------|------|
| Whale kupuje pozici | +1 |
| Value model ukazuje edge > 5% | +1 |
| News agent detekuje relevantní event | +1 |
| Order flow spike v marketu (Vrstva 7) | +1 |
| Logická nekonzistence s related market (Vrstva 5) | +1 |

**Execution pravidla:**
- Skóre 0–1: NO TRADE
- Skóre 2: Standardní velikost (2.5% kapitálu)
- Skóre 3+: Dvojitá velikost (5% kapitálu — max limit)

```yaml
confluence:
  min_score: 2
  standard_size_pct: 0.025
  double_size_pct: 0.05
  whale_min_confidence: 0.6
  whale_poll_interval_s: 4
  whale_detection_target_s: 10  # Max latence od transakce do signálu
```

### Vrstva 5: Logická/Kombinatorická Arbitráž (LLM)

**Mechanismus:** Gemini 2.0 Flash analyzuje páry marketů z Vrstvy 3 (Semantic Graph) a hledá cenové nekonzistence.

**Příklad:**
- Market A: "Republicans win ≥ 3 swing states" = 45¢
- Market B: "Republicans win Pennsylvania" = 55¢
- Market C: "Republicans win Georgia" = 50¢
- Market D: "Republicans win Arizona" = 48¢
- Pokud B+C+D implikují A s P > 0.45, je A podhodnocený

**Implementace:**
- LLM prompt template: "Given markets X, Y, Z with prices P_x, P_y, P_z — is there a logical pricing inconsistency? If yes, which market is mispriced and in which direction?"
- Gemini response_mime_type="application/json" pro strukturovaný output
- Threshold: pricing violation > 3% → SIGNAL
- NegRisk multi-outcome events: suma YES cen vs $1.00 → alert pokud |sum - 1.0| > 0.03

```yaml
logical_arb:
  min_pricing_violation: 0.03
  scan_interval_s: 900     # Každých 15 minut
  llm_model: "gemini-2.0-flash"
  max_llm_calls_per_hour: 40
  negrisk_sum_threshold: 0.03
```

### Vrstva 6: Temporal Crypto Arbitráž (15min markety)

**Mechanismus:** 15-minutové crypto markety (BTC/ETH price above/below X) mají inherentní latenci — market price reaguje pomaleji než spot.

**Execution:**
- Monitor Binance/Coinbase spot price WebSocket
- Když spot výrazně breakne threshold ale 15min market ještě nereagoval → trade
- PostOnly ordery = maker rebates (tyto markety mají fees, ale maker rebates kompenzují)

**Klíčové:**
- Funguje POUZE na fee-enabled 15min crypto marketech
- Edge existuje v prvních 30–60s po cenovém pohybu
- Nutný real-time spot price feed

```yaml
temporal_crypto:
  spot_sources: ["binance_ws", "coinbase_ws"]
  price_deviation_threshold: 0.005  # 0.5% spot move
  market_lag_window_s: 60
  use_postonly: true
  max_trades_per_hour: 20
```

### Vrstva 7: Smart Money Order Flow Detection

**Mechanismus:** Monitorování Polygon blockchainu pro OrderFilled events z CTF Exchange kontraktu. Detekce anomálií v order flow.

**CTF Exchange contract:** `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`

**Detekční features:**
- **Volume Z-score:** Rolling mean/std přes 1h, 4h, 24h windows → flag když z > 2
- **Cumulative delta:** `cumsum(buy_vol - sell_vol)` — odhaluje směrový tlak
- **Flow imbalance ratio:** `buy_vol / (buy_vol + sell_vol)` → > 0.65 = silný buy signál
- **Large trade ratio:** `large_trades / total_trades` — institucionální aktivita
- **Off-hours trades:** Obchody v nízké likviditě = informované pozice
- **Wallet clustering:** Více walletů obchodující stejným směrem v krátkém okně = koordinace

**Signal:** Když 2+ features konvergují → follow flow direction, feeduje do Confluence (Vrstva 4)

**Implementace:** WebSocket na Polygon RPC (free tier) + event parsing

```yaml
order_flow:
  polygon_rpc_url: "wss://polygon-mainnet.g.alchemy.com/v2/{KEY}"
  ctf_exchange: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
  volume_zscore_threshold: 2.0
  flow_imbalance_threshold: 0.65
  rolling_windows: [3600, 14400, 86400]  # 1h, 4h, 24h v sekundách
  min_converging_signals: 2
```

### Vrstva 8: Attention Markets (Kaito AI × Polymarket)

**Mechanismus:** Nový typ marketů (od 10.2.2026) kde se obchoduje na "mindshare" a "sentiment" měřené Kaito AI napříč X, TikTok, Instagram, YouTube.

**Aktuální stav:**
- 2 pilotní markety live od listopadu 2025
- "Polymarket mindshare by March 31, 2026": $1.3M volume
- "Crypto Twitter mindshare ATH by March 31": $90K volume
- Desítky nových marketů od března 2026

**Implementace:**
- Gemini 2.0 Flash skenuje X/Reddit/TikTok sentiment pro relevantní témata
- Model odhaduje budoucí mindshare trend, porovnává s aktuální Polymarket cenou
- Pokud divergence > 5% → trade signál

**Rizika:**
- Bot traffic může zkreslit metriky
- Reflexivita: tradeři mohou aktivně promovat téma aby posunuli svou pozici
- Kaito data integrity — historie kritiky ohledně přesnosti

```yaml
attention_markets:
  sentiment_scan_interval_s: 1800  # Každých 30 minut
  min_divergence: 0.05             # 5% minimum
  max_position_pct: 0.05
  sources: ["twitter", "reddit", "tiktok"]
  llm_model: "gemini-2.0-flash"
```

### Vrstva 9: Live Sports Data Latency Arbitráž

**Mechanismus:** Informační tok má inherentní zpoždění:
- T+0s: Událost (gól, bod) se stane
- T+3s: Oficiální data feed aktualizuje (Opta, SportRadar, The Odds API)
- T+5s: Náš bot detekuje a obchoduje
- T+45–60s: Retail trader vidí na livestreamu
- T+60–90s: Polymarket cena se adjustuje

V okně 5–60s obchodujeme za ceny které neodráží aktuální stav hry.

**Focus:** Soccer/EPL + Esports (LoL, CS2, Dota 2)

**Esports výhoda:**
- Game servery jsou deterministické, API updatuje okamžitě (T+1–2s)
- Twitch stream laguje 15–45s
- Tournament broadcast přidává produkční delay → celkem 60s+

**Fee mitigation:** Obchodovat POUZE při extrémních pravděpodobnostech (p > 0.95 nebo p < 0.05) kde dynamické fees < 0.3%

```yaml
sports_latency:
  data_sources:
    - type: "the_odds_api"
      sports: ["soccer_epl", "soccer_laliga", "soccer_bundesliga"]
      poll_interval_s: 5
    - type: "riot_api"
      games: ["league_of_legends"]
    - type: "steam_api"
      games: ["cs2"]
  max_trade_size: 200              # €200 per trade
  max_trades_per_hour: 10
  min_probability_extreme: 0.95    # Nebo < 0.05
```

### Position Sizing: Half-Kelly Criterion

**Aplikováno na VŠECHNY vrstvy:**

```
full_kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
position_size = min(full_kelly * 0.5, max_position_pct) * capital
```

Pokud model říká 60% a market cena je 45¢ (decimal odds 2.22):
- full_kelly = (0.60 * 2.22 - 1) / (2.22 - 1) = 0.27 (27%)
- half_kelly = 13.5%
- S 5% capem → 5% kapitálu

Pro nižší edge (3–5%) Kelly automaticky redukuje sizing na 1–2%.

---

## 3. Technický stack

### Platform

| Komponenta | Technologie | Poznámka |
|-----------|------------|---------|
| Runtime | Python 3.12+ | asyncio pro všechny IO operace |
| Polymarket SDK | py-clob-client (latest) | Oficiální, rychle se mění — pinovat hash |
| Odds data | The Odds API v4 | Pinnacle + další bookmakeři, REST/JSON |
| On-chain data | Polygon RPC (Alchemy free) | WebSocket pro OrderFilled events |
| Orderbook WS | wss://ws-subscriptions-clob.polymarket.com | Real-time orderbook updates |
| LLM primární | Gemini 2.0 Flash | EEA = paid tier. $0.10/$0.40 per MTok |
| LLM fallback | Claude Haiku 4.5 | $1/$5 per MTok. Pouze při Gemini výpadku |
| Embeddings | e5-large-v2 | Semantic market graph (Vrstva 3) |
| Vector DB | Chroma | Market relationship storage |
| Databáze | SQLite (paper) → PostgreSQL (live) | SQLite pro sprint 1–4, PG pro produkci |
| VPS | Hetzner CX22 (Falkenstein) | 4 vCPU, 8GB RAM, 20ms do London |
| Blockchain | Polygon PoS | USDC.e collateral, ~$0.007 gas/tx |
| Notifikace | Telegram Bot API | Alerty, denní reporty |

### API endpointy

| Endpoint | URL | Účel |
|---------|-----|------|
| CLOB API | `https://clob.polymarket.com` | Orders, orderbook, prices |
| Gamma API | `https://gamma-api.polymarket.com` | Market metadata, discovery |
| Data API | `https://data-api.polymarket.com` | Pozice, trade history, whale data |
| WebSocket | `wss://ws-subscriptions-clob.polymarket.com` | Live orderbook, price updates |
| The Odds API | `https://api.the-odds-api.com/v4` | Pinnacle odds feed |
| Polygon RPC | `wss://polygon-mainnet.g.alchemy.com/v2/{KEY}` | On-chain events |
| Gemini | `https://generativelanguage.googleapis.com/v1beta` | LLM inference |

### Polymarket autentizace

Dvoufázová:
1. **L1 (Private Key):** EIP-712 podpis pro generování API credentials
2. **L2 (API Key):** HMAC-SHA256 podpis pro trading operace (`apiKey`, `secret`, `passphrase`)

Wallet typy: EOA (`signature_type=0`), Email/Magic (`type=1`), Browser proxy (`type=2`)

Allowances: USDC + conditional token allowance pro Exchange contract. Jednou za wallet.

**KRITICKÉ:** Heartbeat API — při disconnect se automaticky cancellují všechny otevřené ordery. Implementovat heartbeat loop s exponential backoff reconnect.

### Polymarket fee model

```python
def calculate_fee(price: float, fee_rate: float = 0.0315) -> float:
    """
    Polymarket dynamic fee formula.
    fee_rate = max taker fee rate (3.15% k únoru 2026)
    Aplikuje se POUZE na fee-enabled markety (check API field).
    Většina sportů = 0% fee.
    """
    if not market.fee_enabled:
        return 0.0
    return price * (1 - price) * fee_rate
    # Max fee je při p=0.50: 0.50 * 0.50 * 0.0315 = 0.79%
    # Při p=0.95: 0.95 * 0.05 * 0.0315 = 0.15%
```

### Dependency verze — DODRŽOVAT

| Package | Min verze | Důvod |
|---------|----------|-------|
| py-clob-client | latest (pin hash) | Polymarket official, breaking changes časté |
| xgboost | >=3.0 | Breaking API changes od 2.x — jiné importy |
| pytest-asyncio | >=1.0 | Breaking changes, vyžaduje `mode="auto"` |
| python-dotenv | >=1.0 | Secret management |
| aiohttp | >=3.9 | WebSocket klient |
| web3.py | >=7.0 | Polygon RPC pro Vrstvu 7 |
| chromadb | >=0.4 | Vector store pro Vrstvu 3 |
| sentence-transformers | >=2.2 | e5-large-v2 embeddings |
| google-generativeai | >=0.5 | Gemini SDK, response_mime_type support |

---

## 4. Sprint plán

### Timeline

| Sprint | Týdny | Klíčový deliverable | Go/No-go gate |
|--------|-------|---------------------|---------------|
| Sprint 1 | Týden 1–2 | CLOB + Odds API + Paper engine | Konektivita OK, scanner běží |
| Sprint 2 | Týden 3–4 | Value model + MM bot + Order flow | Model Brier < 0.22, MM shadow P&L+ |
| Sprint 3 | Týden 5–6 | Whale + Semantic graph + Advanced layers | Whale detection < 10s, 9 vrstev integrated |
| Sprint 4 | Týden 7–10 | Paper trading validace (4 týdny) | 4 týdny paper P&L data, CEO approval |
| Go-live | Týden 11+ | Live trading €500 | CEO osobně schvaluje |

**Celkový rámec:** ~10 týdnů do možného go-live. Realisticky 12 týdnů se safety marginem. První live trade: květen 2026.

---

### SPRINT 1: Foundation (2 týdny)

**Cíl:** Fungující konektivita k Polymarket + The Odds API, paper trading framework, opportunity scanning všech vrstev bez exekuce.

---

#### PM-001: Polymarket CLOB klient wrapper

**Co:** Wrapper kolem `py-clob-client` s production-grade reliability.

**Implementace:**
- Retry logic s exponential backoff (max 3 retries, base 1s)
- Rate limiting (respektovat Polymarket limits)
- Error handling s typed exceptions (`RateLimitError`, `AuthError`, `NetworkError`)
- Podpora: `get_markets()`, `get_orderbook()`, `get_price()`, `get_midpoint()`, `get_tick_size()`
- Konfigurace přes environment variables: `POLY_PRIVATE_KEY`, `POLY_FUNDER_ADDRESS`, `POLY_API_KEY`, `POLY_SECRET`, `POLY_PASSPHRASE`
- Logging: každý API call logován s timestamp, endpoint, response time, status

**Soubor:** `arbo/connectors/polymarket_client.py`

**Acceptance test:** `pytest tests/test_polymarket_client.py` — připojí na CLOB, získá orderbook pro 3 různé aktivní markety, všechny vrátí validní data (bids + asks neprázdné, prices v rozmezí 0–1). Test musí proběhnout bez manuálního zásahu.

---

#### PM-002: Market discovery module

**Co:** Automatické objevování a katalogizace aktivních marketů.

**Implementace:**
- Gamma API integrace: list všech aktivních marketů
- Filtry: sport type, liquidity threshold (> $5K volume), active status, `neg_risk` flag, fee status
- Kategorizace: soccer, politics, crypto, entertainment, esports, attention_markets
- Persist do SQLite: `markets` tabulka s timestamp pro historický tracking
- Refresh interval: každých 15 minut
- Extra field: `fee_enabled` (boolean), `maker_rebate_eligible` (boolean)

**Soubor:** `arbo/connectors/market_discovery.py`

**Acceptance test:** Modul vrátí ≥ 20 aktivních soccer marketů včetně EPL a La Liga. Výsledky persistovány v SQLite a znovu načitatelné.

---

#### PM-003: The Odds API integrace

**Co:** REST klient pro Pinnacle odds data.

**Implementace:**
- REST klient pro `v4/sports/{sport}/odds` endpoint
- Pinnacle odds extrakce: moneyline, spread, totals pro soccer ligy
- Mapping engine: The Odds API event ↔ Polymarket market (matching dle team names + date, fuzzy matching s threshold 0.85)
- Rate limit management: Free tier = 500 req/měsíc, $10 tier = 10K
- Request counter s persistent storage (SQLite) — NIKDY nepřekročit limit
- Cache: odds cache s 5min TTL

**Soubor:** `arbo/connectors/odds_api_client.py`

**Acceptance test:** Pro 5 EPL zápasů získá Pinnacle odds a úspěšně matchne s Polymarket markety. Mapping logika zvládne varianty názvů týmů (např. "Man Utd" vs "Manchester United").

---

#### PM-004: Paper trading engine

**Co:** Simulovaný trading engine pro validaci strategií bez reálného kapitálu.

**Implementace:**
- `PaperTradingEngine` class — přijímá `OrderArgs`, simuluje fill na current midprice
- Simulated slippage: 0.5% default (konfigurovatelné)
- P&L tracking: per-trade, per-strategy, per-day, per-week
- Position management: open positions, unrealized P&L, risk exposure per market type
- SQLite storage: tabulky `paper_trades`, `paper_positions`, `paper_snapshots`
- Automatic resolution: polling resolved markets, update P&L
- Snapshot: každou hodinu uložit stav portfolia

**Soubor:** `arbo/core/paper_engine.py`

**Acceptance test:** Simulovaný trade flow: open 3 pozice, 1 resolve WIN, 1 resolve LOSE, 1 zůstává open. Všechny P&L správně vypočítány. Per-strategy breakdown funguje.

---

#### PM-005: Opportunity scanner (všechny vrstvy)

**Co:** Unified scanner který detekuje příležitosti ze všech vrstev a loguje je.

**Implementace:**
- Layer 1 (MM): Identifikuj markety s spread > 3% a volume > $1K/den
- Layer 2 (Value): Pinnacle vs Polymarket divergence > 3% po fee
- Layer 3 (Arb placeholder): NegRisk markety kde sum(YES prices) < $0.97 nebo > $1.03
- Layer 4 (Whale placeholder): Logování top wallet adres z leaderboardu
- Layer 6 (Crypto placeholder): 15min crypto market discovery
- Unified output format: `Signal(layer, market_id, direction, edge, confidence, timestamp)`
- Vše logovat do DB (`signals` tabulka) + konzole s timestamp

**Soubor:** `arbo/core/scanner.py`

**Acceptance test:** Běží 30 minut, detekuje ≥ 5 příležitostí z Layer 1+2, vše čitelně zalogováno v DB i konzoli.

---

#### PM-006: Config & secrets management

**Co:** Bezpečná konfigurace a správa credentials.

**Implementace:**
- `.env` soubor pro secrets: `POLY_PRIVATE_KEY`, `ODDS_API_KEY`, `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `ALCHEMY_KEY`
- `.env.example` s placeholdery (NIKDY reálné hodnoty)
- Config YAML (`config/settings.yaml`): risk parametry, strategy thresholds, API endpoints, scan intervals
- `Settings` class (Pydantic) pro type-safe config loading
- Pre-commit hook: `detect-secrets` scan
- **NIKDY** nekopírovat private key do kódu, logů, nebo error messages

**Soubory:** `arbo/config/settings.py`, `config/settings.yaml`, `.env.example`

**Acceptance test:** `ruff check .` + `black --check .` clean. `grep -r "0x" arbo/ --include="*.py"` neobsahuje žádný private key. Config načten a validován bez chyb.

---

#### PM-007: Risk manager (core)

**Co:** Singleton risk manager — KAŽDÝ order MUSÍ projít přes něj.

**Implementace:**
- `RiskManager` singleton — sdílený všemi strategiemi
- `pre_trade_check(order)` → approve/reject s důvodem
- `post_trade_update(fill)` → update exposure, P&L, trigger alerts
- `emergency_shutdown()` → cancel all orders, log důvod, notify CEO via Telegram
- Checks: position size, daily loss, weekly loss, market concentration
- Hardcoded limity (viz sekce 5) — NEZMĚNITELNÉ v kódu, pouze v config s CEO approval

**Soubor:** `arbo/core/risk_manager.py`

**Acceptance test:**
1. Order na 6% kapitálu → REJECTED (limit 5%)
2. Po dosažení 10% denní ztráty → automatický shutdown všech strategií
3. Order projde pre_trade_check → post_trade_update správně aktualizuje expozici
4. Emergency shutdown cancelluje všechny otevřené ordery (mock test)

---

### Sprint 1 — CEO akce (Owner blockers)

> **BEZ TĚCHTO BODŮ NEPŮJDE SPRINT 1 DOKONČIT:**
> 1. MetaMask wallet setup — uschovat recovery phrase offline
> 2. Polymarket účet přes MetaMask login — získat proxy wallet address
> 3. CEX účet (Kraken doporučeno — podporuje Polygon USDC withdrawal) — KYC 1–3 dny
> 4. The Odds API registrace (free tier pro začátek) — API key
> 5. Gemini API key (Google AI Studio)
> 6. Alchemy account (free tier) — Polygon RPC key

---

### SPRINT 2: Value Model + Market Making + Order Flow (2 týdny)

**Cíl:** XGBoost value model, fungující MM bot v shadow mode, order flow detection, temporal crypto scanner.

---

#### PM-101: XGBoost value model

**Co:** Statistický model pro odhad skutečné pravděpodobnosti eventů.

**Implementace:**
- Features: Pinnacle implied prob, Polymarket mid, time to event (hours), league, historická vol, volume trend
- Target: actual outcome (1/0)
- XGBoost >=3.0 — **POZOR na breaking changes od major verze** (jiné importy, API)
- Training data: The Odds API historical + Polymarket resolved markets (min 200 samples)
- Kalibrace: Platt scaling → reliability diagram musí být vizuálně kalibrovaný
- Train/test split: 70/30, stratified by league
- Hyperparameter tuning: Optuna, 50 trials

**Soubory:** `arbo/models/xgboost_value.py`, `arbo/models/feature_engineering.py`, `arbo/models/calibration.py`

**Acceptance test:** Backtest na holdout set: Brier score < 0.22, simulated ROI > 2% na bets s > 3% edge threshold.

---

#### PM-102: Value signal generator

**Co:** Periodický scanner který generuje trade signály z value modelu.

**Implementace:**
- Každých 5 minut: scan všech matchovaných Pinnacle-Polymarket marketů
- Spočítá `edge = model_prob - polymarket_prob - estimated_fee`
- Pokud `edge > 0.03`: generuj `Signal(layer=2, market_id, side, confidence, edge)`
- Všechny signály do paper trading engine
- Half-Kelly position sizing

**Soubor:** `arbo/strategies/value_betting.py`

**Acceptance test:** Za 24h generuje ≥ 3 validní signály s edge > 3%, všechny zalogovány s plnou audit trail (timestamp, market, model_prob, market_prob, fee, edge, kelly_size).

---

#### PM-103: Market Making bot (shadow mode)

**Co:** MM bot který loguje co by dělal, ale neexekutuje.

**Implementace:**
- Target: markety s spread > 4%, volume $1K–$50K, nízká volatilita
- Logika: symetrické limit ordery na obě strany (BUY YES + BUY NO)
- Spread management: adjustovat ceny dle orderbook hloubky
- Inventory management: max imbalance 60/40
- Heartbeat implementace: udržovat connection, auto-cancel při disconnect
- Prioritně fee-enabled markety (maker rebates)
- **V SHADOW MODE:** loguje co by dělal, kalkuluje simulated P&L, NEEXEKUTUJE

**Soubor:** `arbo/strategies/market_maker.py`

**Acceptance test:** 24h shadow run — simulated P&L kladné, max drawdown < 3%, heartbeat stabilní bez disconnect > 60s.

---

#### PM-104: Fee model

**Co:** Přesná implementace Polymarket fee struktury.

**Implementace:**
- `fee = p * (1-p) * FEE_RATE` pro fee-enabled markety
- Dynamicky detekovat které markety mají fees (Gamma API field check)
- Zahrnout do VŠECH edge výpočtů a P&L kalkulací
- Maker rebate kalkulace pro fee-enabled markety

**Soubor:** `arbo/core/fee_model.py`

**Acceptance test:** Unit testy pro fee curve na 10 různých cenách (0.05, 0.10, 0.20, ..., 0.95), shoda s Polymarket docs. Test ověří že fee-free markety vrací 0.

---

#### PM-105: LLM probability agent (Gemini)

**Co:** Gemini 2.0 Flash agent pro kvalitativní odhad pravděpodobností.

**Implementace:**
- Prompt template: Předložit market otázku + kontext (odds, news headlines, timeline) → Gemini vrátí P(outcome) s reasoning
- `response_mime_type="application/json"` pro strukturovaný output: `{"probability": 0.65, "confidence": 0.8, "reasoning": "..."}`
- Rate limiting: max 60 calls/hodinu (budget control)
- Fallback: Claude Haiku 4.5 při Gemini výpadku
- Output feeduje do ensemble modelu (Vrstva 2, weight 0.3)

**Soubor:** `arbo/agents/gemini_agent.py`

**Acceptance test:** Pro 5 aktivních marketů vrátí validní JSON s probability v rozmezí [0,1] a neprázdný reasoning. Response time < 5s.

---

#### PM-106: Order flow monitor (Vrstva 7 — základy)

**Co:** WebSocket listener na Polygon pro OrderFilled events.

**Implementace:**
- WebSocket connection na Polygon RPC (Alchemy)
- Parse `OrderFilled` events z CTF Exchange kontraktu
- Kalkulace rolling metrics: volume Z-score (1h, 4h, 24h), cumulative delta, flow imbalance ratio
- Uložit do DB: `order_flow` tabulka (timestamp, market, buy_vol, sell_vol, zscore, imbalance)
- Signal: když 2+ metriky konvergují → `Signal(layer=7, ...)`

**Soubor:** `arbo/connectors/polygon_flow.py`

**Acceptance test:** 1h run — úspěšně parsuje OrderFilled events, kalkuluje volume Z-score, data v DB. Minimálně 100 eventů zachyceno.

---

#### PM-107: Temporal crypto scanner (Vrstva 6 — základy)

**Co:** Scanner pro 15min crypto market příležitosti.

**Implementace:**
- Identifikovat 15min crypto markety na Polymarket (BTC/ETH price targets)
- Monitor spot price via veřejný Binance WebSocket (`wss://stream.binance.com/ws/btcusdt@ticker`)
- Porovnat spot price vs Polymarket market price
- Když divergence > threshold a čas do resolution < 15 min → signál
- PostOnly order preference

**Soubor:** `arbo/strategies/temporal_crypto.py`

**Acceptance test:** Identifikuje ≥ 3 aktivní 15min crypto markety. Spot price feed stabilní 30 minut bez disconnect. Loguje price divergence.

---

### SPRINT 3: Advanced Layers + Integration (2 týdny)

**Cíl:** Whale tracking, semantic market graph, Attention Markets, esports latency, kompletní 9-vrstvý systém v paper trading.

---

#### PM-201: Whale wallet discovery

**Co:** Identifikace a katalogizace profitabilních Polymarket walletů.

**Implementace:**
- Scrape Polymarket leaderboard pro top 50 walletů dle ROI a volume
- Doplnit data z Data API (`/positions` endpoint)
- Filtry: win rate > 60%, ≥ 50 resolved pozic, volume > $50K
- Uložit do DB: `whale_wallets` tabulka — wallet address, historický win rate, specialization (sports/politics/crypto), last_seen
- Weekly refresh

**Soubor:** `arbo/strategies/whale_discovery.py`

**Acceptance test:** Identifikuje ≥ 15 whale walletů s verifikovanou profitabilitou. Data persistovány v DB.

---

#### PM-202: Whale position monitor

**Co:** Real-time monitoring pozic tracked whale walletů.

**Implementace:**
- Polling Data API: každé 4 sekundy check pozice tracked walletů
- Diff detection: nová pozice / zvětšená pozice / uzavřená pozice
- Signal: pokud ≥ 2 whales vejdou do stejného marketu = STRONG signal → Confluence +1
- Latence target: < 10 sekund od whale transakce do našeho signálu
- Feed do Confluence scoreru (Vrstva 4)

**Soubor:** `arbo/strategies/whale_monitor.py`

**Acceptance test:** Monitoruje 10 walletů, detekuje novou pozici do 10s (testovat na reálných whale transakcích), generuje signál s full context (wallet, market, side, size, timestamp).

---

#### PM-203: NegRisk arb monitor

**Co:** Monitoring NegRisk multi-outcome eventů pro pricing inconsistencies.

**Implementace:**
- POUZE monitoring + alerting, ŽÁDNÁ auto-exekuce
- Scan NegRisk eventů: suma YES cen všech outcomes vs $1.00
- Alert pokud: `sum < $0.97` (long arb) nebo `sum > $1.03` (short arb)
- Log: timestamp, event, sum, potenciální profit, trvání okna
- Data pro budoucí rozhodnutí o HFT investici

**Soubor:** `arbo/strategies/arb_monitor.py`

**Acceptance test:** 7-denní run, generuje logy. Neděláme P&L projekce — sbíráme data.

---

#### PM-204: Gemini news agent (shadow mode)

**Co:** LLM agent analyzující breaking news relevantní pro otevřené markety.

**Implementace:**
- Input: RSS feeds (Reuters, AP, BBC) + Google News API
- Gemini 2.0 Flash: "Is this news relevant to any of these active markets? If yes, what is the impact on probability?"
- Output: `NewsSignal(market_id, direction, magnitude, confidence, source_url, reasoning)`
- SHADOW MODE: logují se signály, manuální review týdně
- Max 40 LLM calls/hodinu pro news analysis
- Feed do Confluence scoreru (Vrstva 4, +1 bod)

**Soubor:** `arbo/agents/news_agent.py`

**Acceptance test:** Za 48h generuje ≥ 5 news signálů s neprázdným reasoning. CEO manuálně ohodnotí kvalitu.

---

#### PM-205: Semantic market graph (Vrstva 3)

**Co:** Vektorový graf logických vztahů mezi markety.

**Implementace:**
- Stáhnout titulky a popisy všech aktivních marketů (Gamma API)
- `e5-large-v2` embeddings pro každý market
- Chroma DB: uložit vektory, metadata (market_id, category, price, volume)
- Similarity search: pro každý market najdi top 10 nejpodobnějších
- Relationship classification (Gemini): "What is the logical relationship between Market A and Market B?"
- Output: `Relationship(market_a, market_b, type, confidence)` → uložit do DB
- Denní refresh

**Soubor:** `arbo/models/market_graph.py`

**Acceptance test:** Zpracuje ≥ 500 marketů, identifikuje ≥ 20 vztahů s confidence > 0.7. Relationship types jsou z definovaného setu (SUBSET, MUTEX, IMPLICATION, TEMPORAL).

---

#### PM-206: Logical arb scanner (Vrstva 5)

**Co:** LLM-based scanner pro cenové nekonzistence mezi logicky propojenými markety.

**Implementace:**
- Input: relationships z Vrstvy 3 + aktuální ceny
- Gemini prompt: "Given markets A (price X) and B (price Y) with relationship R, is there a pricing inconsistency?"
- `response_mime_type="application/json"`: `{"inconsistency": true, "direction": "buy_A", "estimated_edge": 0.05, "reasoning": "..."}`
- Threshold: pricing violation > 3% → SIGNAL
- Scan interval: každých 15 minut
- Feed do Confluence scoreru (+1 bod)

**Soubor:** `arbo/strategies/logical_arb.py`

**Acceptance test:** Za 24h zpracuje ≥ 20 market párů, identifikuje ≥ 1 pricing inconsistenci (nebo 0 s logem "no inconsistencies found" — to je validní výstup).

---

#### PM-207: Attention Markets scanner (Vrstva 8)

**Co:** Scanner pro Polymarket Attention Markets (Kaito AI).

**Implementace:**
- Identifikovat aktivní Attention Markets na Polymarket (category filter)
- Gemini sentiment analysis: skenovat X/Reddit pro relevantní témata
- Prompt: "Analyze current social media sentiment and volume for [topic]. Estimate future mindshare trend. Current Polymarket price is X."
- Pokud divergence > 5% → signál
- Začít s pilot markety (Polymarket mindshare, Crypto Twitter mindshare)

**Soubor:** `arbo/strategies/attention_markets.py`

**Acceptance test:** Identifikuje ≥ 2 aktivní Attention Markets. Generuje sentiment report pro každý s Gemini analysis.

---

#### PM-208: Live sports latency module (Vrstva 9)

**Co:** WebSocket feed na sportovní data pro latency arbitráž.

**Implementace:**
- The Odds API live scores endpoint pro soccer
- Esports: Riot Games API (LoL) — free, real-time match data
- Při detekci outcome-determining eventu: porovnat s Polymarket cenou
- Pokud cena ještě nereagovala a P > 0.95 (nebo < 0.05) → signál
- Fee check: obchodovat POUZE při extrémních pravděpodobnostech kde fee < 0.3%

**Soubor:** `arbo/strategies/sports_latency.py`

**Acceptance test:** WebSocket stabilní 30 minut. Detekuje ≥ 1 live event update (gól, game result). Loguje cenu na Polymarket v momentě detekce.

---

#### PM-209: Confluence scoring engine

**Co:** Centrální agregátor signálů ze všech vrstev.

**Implementace:**
- Přijímá signály z všech vrstev (unified `Signal` interface)
- Kalkuluje confluence skóre pro každý market s aktivními signály
- Pravidla: skóre 0–1 → no trade, 2 → standard size, 3+ → double size
- Logování: každý trade decision s breakdown které signály přispěly
- Feed do paper trading engine

**Soubor:** `arbo/core/confluence.py`

**Acceptance test:** Simulované signály z 3 vrstev pro stejný market → confluence skóre správně vypočítáno. Trade size odpovídá pravidlům. Log ukazuje breakdown.

---

#### PM-210: Unified dashboard + reporting

**Co:** Konzolový dashboard a automatické reporty.

**Implementace:**
- CLI dashboard: všech 9 vrstev, otevřené pozice, P&L, aktivní signály, risk utilization
- Denní CSV export pro CEO review
- Týdenní report generátor: P&L per strategy, cumulative P&L, risk metrics, top/bottom 5 trades
- Telegram bot: alerty pro high-priority signály (confluence ≥ 3), daily P&L summary, emergency shutdown notifikace

**Soubory:** `arbo/dashboard/cli_dashboard.py`, `arbo/dashboard/report_generator.py`, `arbo/dashboard/telegram_bot.py`

**Acceptance test:** Dashboard zobrazuje real-time data ze všech vrstev. CSV export funkční. Telegram bot pošle test message.

---

### SPRINT 4: Paper Trading Validace (4 týdny)

**Cíl:** Všechny strategie běží v paper mode. Minimum 4 týdny dat pro go-live rozhodnutí.

> ⚠️ **NON-NEGOTIABLE: 4 týdny paper trading před JAKOUKOLI live exekucí.**
> Toto není volitelné. Žádná strategie nepůjde live bez 4 konsekutivních týdnů paper trading dat. CEO musí osobně schválit go-live.

---

#### PM-301: Paper trading — full system run

**Co:** Všech 9 vrstev běží 24/7 v paper mode.

**Implementace:**
- `main.py` orchestrátor: spustí všechny vrstvy jako asyncio tasks
- Každý trade zalogován: timestamp, strategy (layer), market_id, side, price, size, edge, confluence_score
- Automatic resolution tracking: poll resolved markets, update P&L
- Health monitoring: process watchdog, auto-restart při crash

**Acceptance test:** Systém běží 7 dní bez manuálního zásahu. Uptime > 95%. Generuje trades ze ≥ 4 různých vrstev.

---

#### PM-302: Týdenní report generátor

**Co:** Automatický report pro CEO review.

**Formát:**
```
=== ARBO Weekly Report — Week X ===
Period: DD.MM.YYYY - DD.MM.YYYY

PORTFOLIO
  Starting balance: €XXXX (simulated)
  Ending balance: €XXXX
  Weekly P&L: +/- €XX (X.X%)
  Cumulative P&L: +/- €XX (X.X%)

PER STRATEGY
  Layer 1 (MM):      +€XX (XX trades, XX% win rate)
  Layer 2 (Value):   +€XX (XX trades, XX% win rate)
  Layer 4 (Whale):   +€XX (XX trades, XX% win rate)
  ...

RISK METRICS
  Max drawdown: X.X%
  Sharpe ratio (annualized): X.XX
  Max single loss: €XX
  Risk utilization: XX% of limits

TOP 5 TRADES: [details]
BOTTOM 5 TRADES: [details]
CONFLUENCE ANALYSIS: Average score of winning trades vs losing trades

ISSUES & FLAGS: [any anomalies detected]
```

**Acceptance test:** Report generován automaticky, obsahuje všechny sekce, čísla souhlasí s raw daty v DB.

---

#### PM-303: Bug fixes a optimalizace

**Co:** Kontinuální opravy issues nalezených během paper trading.

**Implementace:**
- Issue tracker: GitLab/GitHub issues tagované `paper-trading`
- Model retraining pokud data ukazují drift (Brier score zhoršení > 10%)
- Fee model validace proti reálným Polymarket datům
- Strategy parameter tuning na základě paper trading výsledků

**Acceptance test:** Všechny known issues opraveny nebo dokumentovány s workaroundem. Model Brier score na paper trading datech < 0.22.

---

### Týdenní review metriky (Sprint 4)

| Metrika | Minimum pro go-live | Ideál |
|---------|-------------------|-------|
| Paper P&L | Kladné ve 3 ze 4 týdnů | Kladné všechny 4 týdny |
| Sharpe ratio (annualizovaný) | > 1.0 | > 2.0 |
| Max drawdown | < 15% | < 8% |
| Win rate (value bets) | > 52% | > 55% |
| Confluence trades win rate | > 60% | > 70% |
| Signal quality (news agent) | CEO approval > 60% | > 75% |
| System uptime | > 95% | > 99% |
| Latence (whale detection) | < 15 sekund | < 5 sekund |
| Active layers generating signals | ≥ 5 z 9 | Všech 9 |

---

## 5. Risk Management — Hardcoded limity

**⛔ TYTO LIMITY JSOU NEZMĚNITELNÉ bez explicitního souhlasu Ownera a zdokumentovaného odůvodnění.**

| Parametr | Limit | Akce při překročení |
|---------|-------|-------------------|
| Max pozice (single trade) | 5% kapitálu | Order REJECTED |
| Max denní ztráta | 10% kapitálu | Automatický shutdown VŠECH strategií |
| Max týdenní ztráta | 20% kapitálu | Shutdown + eskalace na Ownera |
| Whale copy max | 2.5% kapitálu per kopírovanou pozici | Order capped |
| Max v jednom market type | 30% kapitálu | Nové ordery zablokovány |
| Paper trading minimum | 4 týdny před live | Go-live zablokován |
| USDC on Polymarket max | 30 dní | Automatický reminder na withdrawal |
| Max confluence double-size | 5% kapitálu | Hard cap i při score 5 |

### Implementace risk checks

```python
class RiskManager:
    """SINGLETON. Všechny strategie sdílí jednu instanci."""

    def pre_trade_check(self, order: Order) -> Tuple[bool, str]:
        """MUSÍ být volán před KAŽDÝM orderem. Vrací (approved, reason)."""
        # 1. Position size check
        # 2. Daily loss check
        # 3. Weekly loss check
        # 4. Market concentration check
        # 5. Strategy-specific limits

    def post_trade_update(self, fill: Fill) -> None:
        """Aktualizuje exposure, P&L, triggers alerts."""

    def emergency_shutdown(self, reason: str) -> None:
        """Cancel ALL orders, log, notify CEO via Telegram."""
```

**NIKDY neobcházet risk manager. Žádná strategie nemůže poslat order přímo na CLOB. Architektura:**

```
Strategy → Signal → Confluence → RiskManager.pre_trade_check() → CLOB
                                        ↓ (rejected)
                                    Log + Alert
```

---

## 6. Coding Standards & Quality

### Povinné pro každý PR

- `ruff check .` — zero warnings
- `black --check .` — zero changes
- `pytest -v` — all tests pass
- `pytest-asyncio >=1.0` — vyžaduje `mode="auto"` v `pyproject.toml`
- No secrets in code — pre-commit hook: `detect-secrets`
- Docstrings na všechny public funkce (Google style)
- Type hints na všechny funkce (mypy --strict doporučeno pro core/)

### pyproject.toml — povinná konfigurace

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.black]
line-length = 100
target-version = ["py312"]
```

### Git workflow

- `main` branch — always deployable
- Feature branches: `feature/PM-XXX-description`
- PR review povinné před merge
- Commit messages: `[PM-XXX] Short description`

### Logging standard

```python
import structlog
logger = structlog.get_logger()

# Vždy logovat s kontextem
logger.info("signal_generated",
    layer=2,
    market_id="0x...",
    edge=0.045,
    confluence_score=3,
    action="paper_trade"
)
```

---

## 7. Projektová struktura

```
arbo/
├── config/
│   ├── settings.yaml          # Risk params, thresholds, intervals
│   └── settings.py            # Pydantic Settings class
├── core/
│   ├── risk_manager.py        # SINGLETON risk manager
│   ├── paper_engine.py        # Paper trading engine
│   ├── portfolio.py           # Position tracking, P&L
│   ├── confluence.py          # Multi-signal confluence scorer
│   ├── scanner.py             # Unified opportunity scanner
│   └── fee_model.py           # Polymarket fee calculations
├── connectors/
│   ├── polymarket_client.py   # CLOB wrapper
│   ├── market_discovery.py    # Gamma API market catalog
│   ├── odds_api_client.py     # The Odds API (Pinnacle)
│   ├── polygon_flow.py        # On-chain order flow (Vrstva 7)
│   └── websocket_manager.py   # WS connection manager with reconnect
├── strategies/
│   ├── market_maker.py        # Vrstva 1
│   ├── value_betting.py       # Vrstva 2 (signal generator)
│   ├── whale_discovery.py     # Vrstva 4 (discovery)
│   ├── whale_monitor.py       # Vrstva 4 (real-time monitoring)
│   ├── logical_arb.py         # Vrstva 5
│   ├── temporal_crypto.py     # Vrstva 6
│   ├── arb_monitor.py         # Vrstva 3 (NegRisk monitoring)
│   ├── attention_markets.py   # Vrstva 8
│   └── sports_latency.py      # Vrstva 9
├── models/
│   ├── xgboost_value.py       # XGBoost value model
│   ├── feature_engineering.py # Feature extraction
│   ├── calibration.py         # Platt scaling, reliability
│   └── market_graph.py        # Semantic graph (Vrstva 3, Chroma)
├── agents/
│   ├── gemini_agent.py        # Gemini 2.0 Flash wrapper
│   └── news_agent.py          # News sentiment agent (Vrstva 8/204)
├── dashboard/
│   ├── cli_dashboard.py       # Terminal dashboard
│   ├── report_generator.py    # Weekly/daily reports
│   └── telegram_bot.py        # Alert notifications
├── tests/
│   ├── test_polymarket_client.py
│   ├── test_risk_manager.py
│   ├── test_paper_engine.py
│   ├── test_fee_model.py
│   ├── test_value_model.py
│   ├── test_confluence.py
│   └── ...                    # Per-module test files
├── scripts/
│   ├── setup_wallet.py        # Wallet setup helper
│   ├── backfill_data.py       # Historical data collection
│   └── run_backtest.py        # Offline backtesting
├── main.py                    # Orchestrátor — spouští vše
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 8. Rozpočet (interní — nesdílet externě)

| Položka | Měsíční náklad | Poznámka |
|---------|---------------|---------|
| VPS (Hetzner CX22) | €6–20 | Falkenstein, 4 vCPU, 8GB RAM |
| The Odds API | €10–50 | Free tier pro start, $10 tier = 10K req |
| Gemini 2.0 Flash | €10–20 | Paid tier pro EEA, ~80K tokens/den (9 vrstev) |
| Polygon gas | < €1 | ~$0.007/tx |
| Alchemy | €0 | Free tier dostatečný |
| Chroma DB | €0 | Self-hosted na VPS |
| Telegram Bot | €0 | Free |
| Domain + misc | €5 | Monitoring, backups |
| **CELKEM** | **€31–96** | **Target: €50–60/měsíc steady state** |

### Kapitálová alokace

| Fáze | Kapitál | Podmínka |
|------|---------|---------|
| Paper trading | €0 (simulace) | Sprint 1–4 |
| Live Phase 1 | €500 | 4 týdny paper P&L+, CEO approval |
| Live Phase 2 | €2,000 | 4 týdny live profitable |
| Scale-up | €5,000–10,000 | 3+ měsíců, Sharpe > 1.5, drawdown < 15% |

---

## 9. Následující kroky

1. **CTO:** Review tohoto briefu. Ptej se na nejasnosti. Konfirmuj timeline do 48h.
2. **Dev team:** Setup dev environment, naklonovat py-clob-client, získat The Odds API free key, Alchemy key.
3. **CEO:** Sprint 1 kickoff pondělí 24.2.2026 po potvrzení od CTO.

---

## 10. FAQ pro dev team

**Q: Můžu obejít risk manager pro testování?**
A: NE. Ani v testech. Mockni risk manager, ale flow musí jít přes něj.

**Q: Co když Gemini API nefunguje?**
A: Fallback na Claude Haiku 4.5. Implementovat v `gemini_agent.py` s automatic failover. Pokud obě API nefungují → LLM vrstvy (5, 8, 204) se pozastaví, zbytek systému pokračuje.

**Q: Můžu změnit risk limity v configu?**
A: NE. Hardcoded limity vyžadují CEO approval s dokumentovaným odůvodněním. V YAML jsou jako reference, ale enforced v kódu.

**Q: Jaký je budget na The Odds API?**
A: Začínáme na free tier (500 req/měsíc). Upgrade na $10 tier po Sprint 1 pokud potřeba. Tracking requestů je POVINNÝ — nikdy nepřekročit limit.

**Q: Jak často retrainovat XGBoost model?**
A: Měsíčně, nebo když Brier score na rolling 2-week window zhorší o > 10%. Automatický alert v dashboardu.

**Q: Co je Attention Market?**
A: Nový typ Polymarket marketu (od února 2026) kde se obchoduje na "mindshare" — jak moc se o tématu diskutuje na sociálních sítích. Měřeno Kaito AI. Pilotní markety už běží, desítky nových od března. Pro nás = nový, neeficientní trh kde LLM analýza dává edge.

**Q: Proč 9 vrstev místo jednodušších 3?**
A: Diverzifikace. Pokud jedna edge zmizí (a zmizí — trhy jsou adaptivní), ostatní pokračují. Nekorelovné zdroje edge = stabilnější výnosy a menší drawdown. Confluence scoring zajišťuje že obchodujeme jen když více signálů souhlasí.

---

*Tento dokument je závazným zadáním pro dev tým. Odchylky od specifikace vyžadují CEO approval.*

**Podpis CEO: Arbo — 21. února 2026**
