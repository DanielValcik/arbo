# CTO System Audit Report — Reflexive Decay Harvester Pivot

**Datum:** 25. února 2026
**Od:** CTO, Arbo
**Pro:** CEO
**Klasifikace:** INTERNAL

---

## 0. Executive Summary

Provedl jsem audit celého codebase (82 souborů, ~12,500 LOC production kódu, 581+ testů). Výsledek je pozitivní — **přibližně 60% stávajícího kódu je přepoužitelné** pro novou architekturu Reflexive Decay Harvester. Jádro systému (auth, order execution, risk manager, paper engine, dashboard, DB) je solidní a vyžaduje pouze adaptaci, ne přepsání.

Klíčové zjištění: Modulární task-per-layer design v orchestrátoru znamená, že přechod ze 9 vrstev na 3 strategie je architektonicky přirozený — stačí vyměnit layer tasky za strategy tasky.

**Odhad celkového effort:** ~148h (viz souhrnná tabulka na konci)

---

## 1. Per-Module Audit

---

### SHARED INFRASTRUCTURE — CONNECTORS

---

## Module: arbo/connectors/polymarket_client.py

**Category:** KEEP
**Current function:** CLOB API wrapper — L1+L2 auth, order placement (GTC/GTD/FOK/FAK), orderbook read, batch orders (15/call), rate limiting, retry logic
**New function:** Identická — sdílená infrastruktura pro všechny 3 strategie
**Changes needed:** Žádné
**Estimated effort:** 0h
**Dependencies:** py-clob-client==0.34.6, arbo.config.settings
**Risk:** Low
**LOC:** ~465

---

## Module: arbo/connectors/market_discovery.py

**Category:** ADAPT
**Current function:** Gamma API market catalog — paginated fetch, 15min refresh, DB sync, category filters (soccer, crypto, politics), sports series mapping, NegRisk detection
**New function:** Rozšířit o weather market discovery + longshot market filtering + Attention Markets category
**Changes needed:**
- Přidat weather market filter (`get_weather_markets()`) — filtr podle category/keywords ("temperature", "weather", city names)
- Přidat longshot market filter (`get_longshot_markets()`) — YES price < $0.15, volume > $10K, fee-free
- Rozšířit `CATEGORY_KEYWORDS` o weather + attention markets klíčová slova
- Přidat bucket parsing pro weather markets (extrakce temperature range z question)
- Přidat `get_attention_markets()` filter (already partially exists via category detection)
**Estimated effort:** 6h
**Dependencies:** aiohttp, sqlalchemy, arbo.utils.db
**Risk:** Low — rozšíření existujícího kódu, ne přepisování
**LOC:** ~782 → ~900

---

## Module: arbo/connectors/polygon_flow.py

**Category:** ADAPT → Strategy A (Theta Decay)
**Current function:** Layer 7 — Polygon OrderFilled event listener via HTTP polling (eth_getLogs), RollingWindow z-score, flow imbalance, cumulative delta, signal emission
**New function:** Základ pro **taker flow analysis** ve Strategy A — přesně to co potřebujeme
**Changes needed:**
- Přestavět signal emission z L7 formátu na Strategy A formát
- Přidat YES/NO taker flow ratio (aktuálně trackovány buy/sell, potřebujeme YES taker vs NO taker)
- Změnit z-score window z 1h na 4h rolling (per strategy doc: "rolling 4-hour window")
- Přidat 3σ threshold pro peak optimism detection (aktuálně 2.0σ)
- Přidat per-market tracking (aktuálně globální, potřebujeme per condition_id)
- Rozšířit RollingWindow o per-market instance management
- dRPC connection funguje (D3 fix implementován), incremental block tracking OK
**Estimated effort:** 12h
**Dependencies:** aiohttp, web3 (keccak), sqlalchemy, arbo.utils.db
**Risk:** Medium — z-score recalibrace může vyžadovat tuning na live datech
**LOC:** ~631 → ~750

**Poznámka k otázce 3.2:** Aktuální stav L7 je funkční. dRPC connection běží, incremental block tracking implementován (D3 fix). Alchemy CU problém vyřešen přechodem na dRPC. Event listener je přesně to co potřebujeme — jen potřebuje rozšířit o YES/NO taker ratio místo generic buy/sell.

---

## Module: arbo/connectors/binance_client.py

**Category:** REMOVE (archive)
**Current function:** Layer 6 — Binance REST client (klines, ticker, funding rates, RSI, momentum, volatility)
**New function:** Žádná — nová architektura nemá crypto arbitrage strategii
**Changes needed:** Žádné — archivovat
**Estimated effort:** 0h
**Dependencies:** aiohttp
**Risk:** None
**LOC:** ~443

---

## Module: arbo/connectors/odds_api_client.py

**Category:** REMOVE (archive)
**Current function:** The Odds API v4 — Pinnacle odds fetch, quota tracking, bookmaker fallback, vig removal
**New function:** Žádná — nová architektura nepoužívá sportovní odds jako signální zdroj
**Changes needed:** Žádné — archivovat
**Estimated effort:** 0h
**Dependencies:** aiohttp, pydantic
**Risk:** None — ušetříme $30/měsíc na API subscription
**LOC:** ~603

**Poznámka:** Pokud v budoucnu přidáme sports betting jako 4. strategii, kód je připravený k reactivaci.

---

## Module: arbo/connectors/event_matcher.py

**Category:** REMOVE (archive)
**Current function:** Fuzzy matching Polymarket markets ↔ The Odds API events (rapidfuzz, team aliases, league mapping)
**New function:** Žádná — závisí na odds_api_client který také rušíme
**Changes needed:** Žádné — archivovat
**Estimated effort:** 0h
**Dependencies:** rapidfuzz, odds_api_client, market_discovery
**Risk:** None
**LOC:** ~595

---

## Module: arbo/connectors/websocket_manager.py

**Category:** N/A — soubor neexistuje (zmíněn v CLAUDE.md ale nikdy implementován)
**Note:** WebSocket management je inline v polymarket_client.py a temporal_crypto.py

---

### SHARED INFRASTRUCTURE — CORE

---

## Module: arbo/core/risk_manager.py

**Category:** ADAPT
**Current function:** Singleton risk gatekeeper — 8 pre-trade checks, hardcoded limits (5%/10%/20%), per-category exposure, daily/weekly reset
**New function:** Rozšířit o **per-strategy alokaci** a per-strategy position limits
**Changes needed:**
- Přidat `StrategyAllocation` dataclass (strategy_id, allocated_capital, deployed_capital, available_capital)
- Přidat strategy-aware pre_trade_check() — validace proti strategy allocation, ne jen global capital
- Přidat per-strategy concurrent position limit (max 10 per strategy)
- Přidat reserve capital lock (€200 = 10%, nikdy deployed)
- Přidat cross-strategy total exposure tracking
- NEMĚNÍ SE: Hardcoded limits (MAX_POSITION_PCT, DAILY_LOSS_PCT, WEEKLY_LOSS_PCT)
- NEMĚNÍ SE: Emergency shutdown logic
**Estimated effort:** 8h
**Dependencies:** arbo.utils.logger
**Risk:** Medium — per-strategy tracking musí být bullet-proof, chyba = capital leak
**LOC:** ~334 → ~450

**Odpověď na otázku 3.5:** Ano, risk manager JE singleton pattern (`async get_instance()`). Přidání per-strategy limitů je technicky přímočaré — přidat dict `{strategy_id: StrategyAllocation}` a rozšířit `pre_trade_check()` o strategy_id parametr. Aktuální 8 checks zůstávají, přidá se 9. check (strategy allocation). Hardcoded limity se NEMĚNÍ.

---

## Module: arbo/core/paper_engine.py

**Category:** ADAPT
**Current function:** Paper trading simulator — place_trade(), resolve_market(), half-Kelly sizing, slippage, DB persistence, per-layer P&L tracking, portfolio snapshots
**New function:** Přestavět per-layer tracking na **per-strategy tracking** (A, B, C)
**Changes needed:**
- Změnit `_per_layer_realized_pnl: dict[int, Decimal]` na `_per_strategy_pnl: dict[str, Decimal]`
- PaperTrade: přidat `strategy` field (místo `layer`)
- Přidat `get_strategy_stats()` metodu
- Upravit `take_snapshot()` pro per-strategy breakdown
- Změnit Kelly z half-Kelly (0.5) na quarter-Kelly (0.25) — per strategy doc
- Aktualizovat DB persistence metody pro strategy field
**Estimated effort:** 6h
**Dependencies:** risk_manager, fee_model, odds, db
**Risk:** Low — refactoring tracking, ne core logic
**LOC:** ~612 → ~650

---

## Module: arbo/core/confluence.py

**Category:** REMOVE (archive)
**Current function:** Multi-signal confluence scorer — groups signals by market, computes 0-5 score, sizes positions (2.5% standard / 5% double)
**New function:** Žádná — nová architektura má per-strategy quality gates místo cross-layer confluence
**Changes needed:** Nahradit per-strategy quality gate modulem
**Estimated effort:** 0h (replacement is NEW module)
**Dependencies:** risk_manager, scanner
**Risk:** None
**LOC:** ~350

**Poznámka:** Tohle je CORE architektonická změna. Starý systém: 9 layers → confluence score ≥ 2 → trade. Nový systém: each strategy has its own entry signal → quality gate → trade. Confluence scorer nemá v nové architektuře smysl.

---

## Module: arbo/core/scanner.py

**Category:** ADAPT
**Current function:** Unified signal scanner — Signal dataclass, SignalDirection enum, scan_all() across 9 layers
**New function:** Zachovat Signal + SignalDirection jako universal DTOs, ale odstranit per-layer scan logic
**Changes needed:**
- KEEP: Signal dataclass, SignalDirection enum, to_db_dict() — tyto jsou universal
- REMOVE: scan_all(), scan_layer1_mm(), scan_layer2_value(), etc. — per-layer scanning
- RENAME: `layer` field → `strategy` field v Signal
- Přidat strategy-specific signal types (WeatherSignal, ThetaDecaySignal, ReflexivitySignal)
**Estimated effort:** 3h
**Dependencies:** fee_model, market_discovery
**Risk:** Low
**LOC:** ~386 → ~150 (menší po odebrání layer scans)

---

## Module: arbo/core/fee_model.py

**Category:** KEEP
**Current function:** Polymarket fee calculations — taker fee, maker rebate, fee favorability check
**New function:** Identická — fee model je nezávislý na strategii
**Changes needed:** Žádné
**Estimated effort:** 0h
**Dependencies:** stdlib only
**Risk:** None
**LOC:** ~122

---

### STRATEGIES — OLD LAYERS

---

## Module: arbo/strategies/market_maker.py (Layer 1)

**Category:** REMOVE (archive)
**Current function:** Shadow-mode market making — dynamic spread, inventory skew, simulated fills
**New function:** Žádná v MVP. Strategie brief říká "MOŽNÁ LATER" — není v Reflexive Decay Harvester
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** polymarket_client, market_discovery
**Risk:** None
**LOC:** ~370

---

## Module: arbo/strategies/value_signal.py (Layer 2)

**Category:** REMOVE (archive)
**Current function:** Multi-category value betting — Pinnacle odds comparison, XGBoost ensemble, Gemini politics, crypto Binance
**New function:** Žádná — nové strategie mají vlastní signální logiku (weather forecast, taker flow, Kaito divergence)
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** event_matcher, odds_api, market_discovery, xgboost_value, gemini_agent, binance_client
**Risk:** None
**LOC:** ~493

**Poznámka k CEO hodnocení L2 "ADAPT → XGBoost na weather data":** Po auditu NESOUHLASÍM s přetrénováním XGBoost na weather data. Weather strategy nepotřebuje ML model — potřebuje deterministický forecast comparison (NOAA forecast vs market bucket). ML přidá zbytečnou komplexitu. XGBoost dává smysl pro value betting kde probability estimation je inherentně nejistá. U weather máme 85-90% accurate NOAA forecast — to je náš edge, ne ML model.

---

## Module: arbo/strategies/whale_discovery.py (Layer 4A)

**Category:** REMOVE (archive)
**Current function:** Whale wallet discovery z Polymarket leaderboard
**New function:** Žádná — nová architektura nemá whale copy strategii
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** aiohttp
**Risk:** None
**LOC:** ~298

---

## Module: arbo/strategies/whale_monitor.py (Layer 4B)

**Category:** REMOVE (archive)
**Current function:** Real-time whale position polling + multi-whale confluence
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** whale_discovery, scanner
**Risk:** None
**LOC:** ~359

**Poznámka k CEO hodnocení L4 "ADAPT → Strategy A":** NESOUHLASÍM. L4 (whale wallet tracking) a Strategy A (taker flow analysis) řeší fundamentálně jiný problém. L4 trackuje specifické wallets a jejich pozice. Strategy A analyzuje anonymní YES/NO taker flow ratio na trhu. Kód z L4 není přepoužitelný. Základ pro Strategy A je polygon_flow.py (L7), ne whale_monitor.py (L4).

---

## Module: arbo/strategies/logical_arb.py (Layer 5)

**Category:** REMOVE (archive)
**Current function:** LLM-driven pricing inconsistency detection + NegRisk sum violations
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** gemini_agent, market_graph, market_discovery
**Risk:** None
**LOC:** ~299

---

## Module: arbo/strategies/temporal_crypto.py (Layer 6)

**Category:** REMOVE (archive)
**Current function:** 15min crypto arb — Binance WebSocket spot vs Polymarket contract divergence
**New function:** Žádná — 500ms taker delay odstraněn, strategie mrtvá
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** market_discovery, binance (ws)
**Risk:** None
**LOC:** ~363

---

## Module: arbo/strategies/arb_monitor.py (Layer 3)

**Category:** REMOVE (archive)
**Current function:** NegRisk monitoring (sum violations, logging only)
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** market_discovery
**Risk:** None
**LOC:** ~170

---

## Module: arbo/strategies/attention_markets.py (Layer 8)

**Category:** ADAPT → Strategy B (Reflexivity Surfer)
**Current function:** Gemini LLM sentiment estimation → divergence detection → signal emission
**New function:** Základ pro Reflexivity Surfer, ale potřebuje zásadní přestavbu
**Changes needed:**
- Nahradit Gemini sentiment za Kaito API mindshare/sentiment data
- Přidat 4-fázový state machine (Start → Boom → Peak → Bust)
- Přidat price vs reality divergence calculator (PM_price − Kaito_actual) / Kaito_actual
- Přidat phase transition detection (>20% divergence = Phase 3)
- Zachovat: market filtering pattern, divergence calculation pattern, signal emission
- Odstranit: hallucination filtering (nepotřebné s Kaito data), LLM cost control
**Estimated effort:** 16h (significant rewrite, ale architektonický pattern zůstává)
**Dependencies:** NEW Kaito API connector, market_discovery
**Risk:** High — Kaito API zatím neexistuje (launch březen 2026). Potřebujeme Plan B.
**LOC:** ~237 → ~400

**Odpověď na otázku 3.3:** L8 aktuálně volá Gemini LLM per market (max 10/scan, 30min interval), žádá o probability estimate, počítá divergenci proti market price, filtruje hallucinations (>25% divergence). Zdroj dat je čistě LLM — žádný external sentiment feed. Pro Kaito API integration je to **significant rewrite** — zachováme pattern (filter markets → fetch data → compute divergence → emit signal) ale data source a logika se mění kompletně. Doporučuji psát reflexivity_surfer.py jako nový modul s inspirací z attention_markets.py, ne přestavbu in-place.

---

## Module: arbo/strategies/sports_latency.py (Layer 9)

**Category:** REMOVE (archive)
**Current function:** Live sports latency arb (disabled — burns Odds API credits)
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** odds_api_client, market_discovery
**Risk:** None
**LOC:** ~268

---

### MODELS / ML

---

## Module: arbo/models/xgboost_value.py

**Category:** REMOVE (archive)
**Current function:** XGBoost value model (Brier 0.198, ROI 16% backtest) — soccer/crypto probability prediction
**New function:** Žádná — nové strategie nepoužívají ML probability estimation
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** joblib, numpy, pandas, xgboost, sklearn, optuna
**Risk:** None
**LOC:** ~422

---

## Module: arbo/models/feature_engineering.py

**Category:** REMOVE (archive)
**Current function:** 16-feature extraction pro value model (pinnacle_prob, odds_movement, etc.)
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**Dependencies:** pandas
**Risk:** None
**LOC:** ~138

---

## Module: arbo/models/xgboost_crypto.py

**Category:** REMOVE (archive)
**Current function:** Crypto XGBoost model (template, not deployed)
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**LOC:** ~318

---

## Module: arbo/models/crypto_features.py

**Category:** REMOVE (archive)
**Current function:** 12-feature extraction pro crypto model
**New function:** Žádná
**Changes needed:** Archivovat
**Estimated effort:** 0h
**LOC:** ~122

---

## Module: arbo/models/calibration.py

**Category:** REMOVE (archive)
**Current function:** Platt scaling, Brier score calculation
**New function:** Žádná — nové strategie nepoužívají ML calibration
**Changes needed:** Archivovat
**Estimated effort:** 0h
**LOC:** ~101

---

## Module: arbo/models/market_graph.py

**Category:** REMOVE (archive)
**Current function:** Semantic market graph — e5-large-v2 embeddings + Chroma DB + Gemini classification
**New function:** Žádná — nová architektura nemá semantic graph layer
**Changes needed:** Archivovat. Ušetříme ~1.3GB RAM (e5-large-v2 model)
**Estimated effort:** 0h
**Dependencies:** sentence-transformers, chromadb, gemini_agent
**Risk:** None — memory savings
**LOC:** ~302

---

### AGENTS

---

## Module: arbo/agents/gemini_agent.py

**Category:** ADAPT
**Current function:** LLM wrapper (Gemini 2.5 Flash primary, Claude Haiku fallback) — probability predictions, raw queries, rate limiting, JSON extraction
**New function:** Zachovat pro Strategy B (Reflexivity Surfer) a potenciální budoucí LLM use-cases
**Changes needed:**
- Zredukovat na raw_query() + rate limiting (predict() nebude potřeba pro nové strategie)
- Možnost přidat Kaito data enrichment via LLM jako fallback pro Strategy B
- Zachovat _extract_json() — robustní JSON parsing z LLM responses
**Estimated effort:** 2h
**Dependencies:** google-generativeai, anthropic
**Risk:** Low
**LOC:** ~394 → ~250

---

### DASHBOARD / REPORTING

---

## Module: arbo/dashboard/web.py

**Category:** ADAPT
**Current function:** FastAPI dashboard — 8 API routes (portfolio, positions, signals, layers, risk, infra, trades), HTTP Basic Auth
**New function:** Přestavět na per-strategy metriky
**Changes needed:**
- `/api/portfolio` → přidat per-strategy P&L breakdown (A, B, C)
- `/api/layers` → nahradit za `/api/strategies` (3 strategie místo 9 layers)
- Přidat `/api/weather` — forecast accuracy tracking, resolution chain status
- Přidat `/api/taker-flow` — taker flow visualization (Strategy A)
- Přidat `/api/divergence` — price vs reality divergence meter (Strategy B)
- Přidat `/api/capital` — capital utilization per strategy
- Zachovat: auth, infra, trades routes
**Estimated effort:** 10h
**Dependencies:** fastapi, sqlalchemy, psutil
**Risk:** Low — additive changes, ne breaking
**LOC:** ~610 → ~750

**Odpověď na otázku 3.4:** Dashboard je **FastAPI backend + single-page HTML frontend**. Architektura je API-first — frontend fetchuje JSON z 8 endpointů a renderuje v browser. To je modulární design — přidání nových metrik = přidání nových API endpointů + frontend sections. Odhadovaná náročnost přidání nových metrik: 10h (6h backend routes + 4h frontend HTML/JS).

---

## Module: arbo/dashboard/slack_bot.py

**Category:** KEEP
**Current function:** Slack Socket Mode bot — /status, /pnl, /kill, @mention, 3 channels (#daily-brief, #review-queue, #weekly-report)
**New function:** Identická infrastruktura, upravit formátování zpráv pro nové strategie
**Changes needed:**
- Upravit `_format_status_blocks()` — 3 strategies místo 9 layers
- Upravit `_format_pnl_blocks()` — per-strategy breakdown
- Dependency injection pattern zůstává — callbacks se změní v orchestrátoru, ne v slack_bot
**Estimated effort:** 2h
**Dependencies:** slack-bolt, slack-sdk
**Risk:** Low
**LOC:** ~311

---

## Module: arbo/dashboard/report_generator.py

**Category:** ADAPT
**Current function:** Daily/weekly reports — P&L, drawdown, per-layer breakdown, top/bottom trades, CSV export, Slack Block Kit formatting
**New function:** Přestavět per-layer na per-strategy reporting
**Changes needed:**
- DailyReport/WeeklyReport: `per_layer_pnl` → `per_strategy_pnl`
- WeeklyReport: `per_layer_trade_count`, `per_layer_win_rate` → per-strategy
- Přidat weather-specific metrics (forecast accuracy, resolution chain length)
- Přidat Strategy B metrics (divergence tracking, phase transitions)
- Zachovat: drawdown, top/bottom trades, CSV export, Slack formatting
**Estimated effort:** 4h
**Dependencies:** csv, logging
**Risk:** Low
**LOC:** ~631 → ~680

---

## Module: arbo/dashboard/cli_dashboard.py

**Category:** ADAPT
**Current function:** Terminal dashboard — 9-layer status grid, portfolio, risk, confluence
**New function:** 3-strategy status display
**Changes needed:**
- `format_layer_table()` → `format_strategy_table()` (3 rows místo 9)
- Odstranit confluence section
- Přidat weather forecast accuracy, resolution chain status
**Estimated effort:** 1h
**Dependencies:** logging
**Risk:** Low
**LOC:** ~175

---

### UTILITIES

---

## Module: arbo/utils/logger.py

**Category:** KEEP
**Current function:** structlog JSON logging, secret masking, noisy logger silencing
**New function:** Identická
**Changes needed:** Žádné
**Estimated effort:** 0h
**Dependencies:** structlog
**Risk:** None
**LOC:** ~109

---

## Module: arbo/utils/odds.py

**Category:** ADAPT
**Current function:** Odds conversion, half-Kelly sizing, Polymarket price utils
**New function:** Změnit half-Kelly na quarter-Kelly (per strategy doc)
**Changes needed:**
- Přidat `quarter_kelly()` funkci (KELLY_FRACTION = 0.25)
- Zachovat `half_kelly()` pro backward compatibility
- Přidat strategy-specific sizing helpers
**Estimated effort:** 1h
**Dependencies:** decimal
**Risk:** Low
**LOC:** ~89 → ~110

---

## Module: arbo/utils/db.py

**Category:** ADAPT
**Current function:** 12 SQLAlchemy models + async engine/session singleton
**New function:** Přidat 5 nových tabulek pro RDH strategie
**Changes needed:**
- NEW table: `weather_forecasts` (city, date, source, forecast_temp, actual_temp, bucket, accuracy)
- NEW table: `taker_flow_snapshots` (market_id, timestamp, yes_flow, no_flow, ratio, z_score)
- NEW table: `attention_market_state` (market_id, phase, kaito_mindshare, pm_price, divergence)
- NEW table: `resolution_chains` (chain_id, city_sequence, cumulative_pnl, status)
- NEW table: `strategy_allocation` (strategy, allocated, deployed, available, updated_at)
- MODIFY: PaperTrade — přidat `strategy` field (string: "A", "B", "C")
- MODIFY: PaperPosition — přidat `strategy` field
- MODIFY: PaperSnapshot — `per_layer_pnl` → `per_strategy_pnl`
- MODIFY: Signal — `layer` field → `strategy` field
- KEEP: Market, SystemState, DailyPnl, DailyTradeCounter
- REMOVE: WhaleWallet, OrderFlow (old format), NewsItem, RealMarketData — archivovat
**Estimated effort:** 6h
**Dependencies:** sqlalchemy
**Risk:** Medium — migration musí být clean, data loss = catastrophic
**LOC:** ~350 → ~400

**Odpověď na otázku 3.6:** Používáme **SQLAlchemy 2.0 async + Alembic** pro migrations. DB je **PostgreSQL 16** (ne SQLite). Aktuálně 3 migrations (001 = Matchbook, 002 = Polymarket pivot, 003 = system_state). Migrace na nové schéma = 1 nový Alembic migration file (~200 řádků). Odhad: 3h migration code + 3h testing = 6h total. Nejrizikovější část je modifikace existujících tabulek (paper_trades, signals) — potřebujeme `ALTER TABLE ADD COLUMN` s default hodnotami.

---

### ORCHESTRATOR

---

## Module: arbo/main.py

**Category:** ADAPT (major refactor)
**Current function:** 9-layer task-per-layer orchestrator — signal bus, confluence scoring, health monitor, graceful shutdown, 889 lines
**New function:** 3-strategy orchestrator — per-strategy quality gates místo confluence scoring
**Changes needed:**
- REMOVE: 9 layer task definitions (_run_market_maker, _run_value_signal, _run_semantic_graph, _run_whale_monitor, _run_logical_arb, _run_temporal_crypto, _run_attention, _run_sports_latency)
- REMOVE: confluence scorer integration (_process_signal_batch → get_tradeable)
- REMOVE: D2 signal buffering (cross-layer merge), correlation limiting
- ADD: 3 strategy task definitions (_run_theta_decay, _run_reflexivity_surfer, _run_weather_compound)
- ADD: per-strategy quality gates (replace confluence scorer)
- KEEP: signal queue + processor pattern (works for 3 strategies too)
- KEEP: health monitor (task crash detection, auto-restart)
- KEEP: graceful shutdown (SIGTERM handler)
- KEEP: DB init, Slack init, state restore
- KEEP: scheduled reports, snapshots, price updater, resolution checker
- ADAPT: _init_components() — initialize new strategy modules + connectors
- ADAPT: signal processing — route signals to per-strategy quality gates
**Estimated effort:** 20h
**Dependencies:** all strategy modules, all connectors, core
**Risk:** High — orchestrator je centrum systému, chyba = systém neběží
**LOC:** ~889 → ~700 (méně layers = méně kódu)

**Odpověď na otázku 3.1:** Polymarket integration (auth, CLOB client, market discovery, WebSocket, settlement detection, paper engine) je **dobře oddělená** od confluence logiky. Auth + order placement je v polymarket_client.py, discovery v market_discovery.py, paper trading v paper_engine.py. Žádný z těchto modulů neimportuje confluence.py. Jediné místo kde confluence zasahuje do execution flow je main.py (_process_signal_batch). Extrakce čistých PM integration modulů = straightforward.

---

### CONFIG

---

## Module: arbo/config/settings.py

**Category:** ADAPT
**Current function:** Pydantic Settings — 17 nested config models pro 9 layers + infra
**New function:** 3 strategy configs + shared infra
**Changes needed:**
- REMOVE: MarketMakerConfig, ValueModelConfig, ConfluenceConfig, LogicalArbConfig, BinanceConfig, TemporalCryptoConfig, SportsLatencyConfig, OddsApiConfig
- ADD: ThetaDecayConfig (z_score_threshold, rolling_window_hours, longshot_price_max, etc.)
- ADD: ReflexivitySurferConfig (divergence_threshold, phase_transition_pct, kaito_api_url, etc.)
- ADD: WeatherCompoundConfig (cities, data_sources, bucket_entry_max_price, ladder_enabled, etc.)
- ADD: StrategyAllocationConfig (strategy_a_pct, strategy_b_pct, strategy_c_pct, reserve_pct)
- KEEP: PolymarketConfig, OrderFlowConfig, AttentionMarketsConfig (adapted), LLMConfig, RiskConfig, OrchestratorConfig, PollingConfig
- KEEP: All credential fields (poly_*, slack_*, database_url, etc.)
**Estimated effort:** 4h
**Dependencies:** pydantic, yaml
**Risk:** Low
**LOC:** ~313 → ~300

---

## Module: config/settings.yaml

**Category:** ADAPT
**Current function:** Base config file
**New function:** Nové strategy sections
**Changes needed:** Přepsat strategy sections (remove 9-layer, add 3-strategy)
**Estimated effort:** 1h
**Risk:** Low
**LOC:** ~200

---

### SCRIPTS

---

## Module: scripts/backfill_data.py

**Category:** REMOVE (archive)
**Current function:** Football-data.org CSV + Odds API historical backfill for XGBoost training
**New function:** Žádná
**Estimated effort:** 0h
**LOC:** ~400

---

## Module: scripts/process_data.py

**Category:** REMOVE (archive)
**Current function:** XGBoost training data builder from backfilled data
**New function:** Žádná
**Estimated effort:** 0h
**LOC:** ~300

---

## Module: scripts/run_backtest.py

**Category:** REMOVE (archive)
**Current function:** Full backtest pipeline with CEO pass/fail gates
**New function:** Žádná — nové strategie mají jiný backtest model
**Estimated effort:** 0h
**LOC:** ~250

---

## Module: scripts/backfill_crypto.py

**Category:** REMOVE (archive)
**Current function:** Binance crypto data backfill
**New function:** Žádná
**Estimated effort:** 0h
**LOC:** ~200

---

### TESTS

---

## Module: arbo/tests/ (39 test files, 581+ tests)

**Category:** ADAPT
**Current function:** Comprehensive test suite for 9-layer architecture
**New function:** Tests pro 3-strategy architecture
**Changes needed:**
- KEEP: test_risk_manager.py, test_paper_engine.py, test_paper_engine_db.py, test_fee_model.py, test_slack_bot.py, test_web_dashboard.py (infrastruktura)
- ADAPT: test_polygon_flow.py (taker flow ratio tests), test_orchestrator.py (3 strategies), test_confluence.py → test_quality_gates.py
- REMOVE: test_value_signal.py, test_value_model.py, test_value_signal_router.py, test_market_maker.py, test_whale_monitor.py, test_whale_discovery.py, test_temporal_crypto.py, test_logical_arb.py, test_arb_monitor.py, test_sports_latency.py, test_xgboost_crypto.py, test_gemini_politics.py, test_event_matcher.py, test_odds_api_client.py, test_historical_odds.py, test_binance_client.py, test_backtest_pipeline.py, test_layer_fixes.py, test_market_graph.py
- NEW: test_weather_specialist.py, test_theta_decay.py, test_reflexivity_surfer.py, test_noaa_api.py, test_met_office_api.py, test_open_meteo_api.py, test_resolution_chain.py, test_quality_gates.py
**Estimated effort:** Included in per-module estimates (each NEW module includes tests)
**Risk:** Low
**LOC:** Tests follow source, not separate estimate

---

### NEW MODULES (neexistují, musíme vytvořit)

---

## Module: arbo/connectors/noaa_api.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** NOAA Weather API connector — fetch forecasts for NYC, Chicago
**Implementation:**
- REST client pro api.weather.gov (free, no auth)
- Fetch hourly + daily forecasts by city/station
- Parse temperature, precipitation, wind
- Cache 30min (forecasts don't change faster)
- Error handling: NOAA has occasional outages
**Estimated effort:** 6h
**Dependencies:** aiohttp
**Risk:** Low — well-documented free API
**Tests:** test_noaa_api.py (~3h included)

---

## Module: arbo/connectors/met_office_api.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Met Office UK API connector — fetch London forecasts
**Implementation:**
- REST client pro datahub.metoffice.gov.uk (free tier, API key)
- Fetch spot + 3-hourly forecasts
- Parse temperature (°C), convert to °F if needed for Polymarket
- Cache 30min
**Estimated effort:** 5h
**Dependencies:** aiohttp
**Risk:** Low — free tier, documented API
**Tests:** test_met_office_api.py (~2h included)

---

## Module: arbo/connectors/open_meteo_api.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Open-Meteo connector — Seoul, Buenos Aires forecasts
**Implementation:**
- REST client pro open-meteo.com (free, no auth, no API key)
- Fetch hourly forecasts by lat/lon
- Parse temperature, wind, precipitation
- Multi-city support (Seoul, Buenos Aires, extensible)
- Cache 30min
**Estimated effort:** 4h
**Dependencies:** aiohttp
**Risk:** Low — no auth, no rate limits (reasonable use)
**Tests:** test_open_meteo_api.py (~2h included)

---

## Module: arbo/connectors/kaito_api.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Kaito AI mindshare/sentiment connector pro Strategy B
**Implementation:**
- REST/WS client pro Kaito API (TBD — launch březen 2026)
- Fetch real-time mindshare metrics per topic
- Fetch sentiment scores
- Cache/polling interval configurable
- **BLOCKER:** API zatím neexistuje. Musíme implementovat mock/stub first.
**Estimated effort:** 8h (stub) + 4h (real integration po API launch)
**Dependencies:** aiohttp
**Risk:** HIGH — API neexistuje. Potřebujeme Plan B.
**Tests:** test_kaito_api.py (~3h included)

**Plan B (Kaito fallback):** Pokud Kaito API není dostupné do konce března 2026:
1. **Option 1:** Scrape Twitter/X mention counts via free API (nitter instances, socialblade)
2. **Option 2:** Use Google Trends API jako mindshare proxy
3. **Option 3:** Odložit Strategy B a přealokovat €400 na Strategy C
Doporučuji **Option 3** — je to nejbezpečnější. Strategy C je nejověřenější a extra €400 zvýší throughput.

---

## Module: arbo/strategies/weather_specialist.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Strategy C — Compound Weather Resolution Chaining
**Implementation:**
- Fetch forecasts z NOAA/Met Office/Open-Meteo
- Match forecast → Polymarket weather bucket
- Entry logic: forecast match + bucket price < $0.20
- Temperature laddering (adjacent buckets for high uncertainty)
- Resolution chaining engine (settlement → auto re-deploy to next city)
- Position sizing: $1-10 per trade
- Paper trading mode
**Estimated effort:** 20h
**Dependencies:** noaa_api, met_office_api, open_meteo_api, market_discovery, paper_engine, risk_manager
**Risk:** Medium — bucket matching logic musí být přesná
**Tests:** test_weather_specialist.py (~8h included)

---

## Module: arbo/strategies/theta_decay.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Strategy A — Theta Decay on Longshot Markets
**Implementation:**
- Monitor taker flow z polygon_flow.py (adapted)
- YES/NO taker flow ratio per market (4h rolling window)
- Peak optimism detection (3σ z-score)
- Market selection: p < 0.15, volume > $10K, fee-free, time-to-resolution 3-30d
- Entry: buy NO at peak optimism
- Exit: hold to resolution, partial exit at +50%, stop at -30%
- Quarter-Kelly sizing
- Paper trading mode
**Estimated effort:** 14h
**Dependencies:** polygon_flow (adapted), market_discovery, paper_engine, risk_manager
**Risk:** Medium — 3σ threshold needs calibration on live data
**Tests:** test_theta_decay.py (~6h included)

---

## Module: arbo/strategies/reflexivity_surfer.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Strategy B — Reflexivity Surfer (Attention Markets)
**Implementation:**
- 4-phase state machine (Start → Boom → Peak → Bust) per market
- Kaito API integration (mindshare + sentiment)
- Price vs reality divergence: (PM_price − Kaito_actual) / Kaito_actual
- Phase transition detection: >20% divergence = Phase 3
- Entry: Phase 2 = buy YES (momentum), Phase 3 = sell YES / buy NO (mean reversion)
- Exit: Phase 2 = stop at -15%, Phase 3-4 = hold to resolution / stop at -25%
- Paper trading mode
**Estimated effort:** 16h
**Dependencies:** kaito_api (NEW), market_discovery, paper_engine, risk_manager
**Risk:** HIGH — depends on Kaito API existence + reflexive dynamics being tradeable
**Tests:** test_reflexivity_surfer.py (~6h included)

---

## Module: arbo/core/quality_gates.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Per-strategy quality gates — nahrazuje confluence scorer
**Implementation:**
- `QualityGate` base class s `evaluate(signal) → TradeDecision`
- `ThetaDecayGate`: z-score ≥ 3σ + market criteria met + risk check
- `ReflexivityGate`: divergence threshold + phase confirmed + risk check
- `WeatherGate`: forecast confidence ≥ 85% + bucket price < $0.20 + risk check
- Každý gate je nezávislý — failure jednoho neblokuje ostatní
**Estimated effort:** 8h
**Dependencies:** risk_manager
**Risk:** Low — simpler than confluence (1 signal per strategy vs 5-way merge)
**Tests:** test_quality_gates.py (~3h included)

---

## Module: arbo/core/resolution_chain.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Resolution chaining engine pro Strategy C
**Implementation:**
- Track chain state (chain_id, city_sequence, cumulative P&L)
- Settlement detection → auto-trigger next city deployment
- Capital routing: resolved funds → next available weather market
- Multi-city scheduling (NYC → London → Chicago → Seoul → Buenos Aires)
- DB persistence (resolution_chains table)
**Estimated effort:** 8h
**Dependencies:** weather_specialist, paper_engine, db
**Risk:** Medium — timing logic must be precise (settlement detection → immediate re-deploy)
**Tests:** test_resolution_chain.py (~3h included)

---

### ALEMBIC MIGRATION

---

## Module: alembic/versions/004_rdh_schema.py (NEW)

**Category:** NEW
**Current function:** N/A
**New function:** Database migration pro Reflexive Decay Harvester
**Implementation:**
- CREATE TABLE: weather_forecasts, taker_flow_snapshots, attention_market_state, resolution_chains, strategy_allocation
- ALTER TABLE paper_trades: ADD COLUMN strategy VARCHAR DEFAULT 'legacy'
- ALTER TABLE paper_positions: ADD COLUMN strategy VARCHAR DEFAULT 'legacy'
- ALTER TABLE signals: ADD COLUMN strategy VARCHAR DEFAULT 'legacy'
- ALTER TABLE paper_snapshots: rename per_layer_pnl → per_strategy_pnl (JSON field, backward compatible)
**Estimated effort:** 3h
**Dependencies:** alembic, sqlalchemy
**Risk:** Medium — must be reversible (downgrade function)
**Tests:** Manual verification on staging DB

---

## 2. Odpovědi na specifické otázky (3.1–3.7)

### 3.1 Polymarket Integration
**Extrakce je straightforward.** Auth (L1+L2) je v polymarket_client.py, market discovery v market_discovery.py, paper trading v paper_engine.py, settlement detection v paper_engine + main.py. Žádný z těchto modulů neimportuje confluence.py nebo strategy-specific kód. Jediné coupling point je main.py kde confluence scorer filtruje signály — to je místo kde nasadíme quality gates.

### 3.2 On-Chain Monitoring (L7)
**Aktuální stav je funkční.** dRPC connection běží (D3 fix), incremental block tracking implementován. CU budget ~5.1K/hodinu. Event listener parsuje OrderFilled eventy z CTF Exchange kontraktu. Pro Strategy A potřebujeme rozšířit o:
- Per-market YES/NO taker flow ratio (aktuálně globální buy/sell)
- 4h rolling window (aktuálně 1h)
- 3σ threshold (aktuálně 2.0σ)
Základ je solidní, rozšíření = ~12h.

### 3.3 L8 Attention Layer
**L8 je Gemini LLM-based sentiment estimation.** Žádný external sentiment feed — čistě LLM probability prediction per market. Pro Strategy B potřebujeme kompletně jiný data source (Kaito API) a fundamentálně jinou logiku (4-phase state machine místo single-point divergence). Doporučuji psát reflexivity_surfer.py jako nový modul, ne přestavbu L8.

### 3.4 Dashboard
**FastAPI backend + single-page HTML frontend, API-first design.** 8 JSON endpoints, HTTP Basic Auth. Přidání nových metrik = nové API routes + frontend sections. Náročnost: ~10h (6h backend + 4h frontend).

### 3.5 Risk Manager
**Ano, singleton pattern** (`async get_instance()`). Přidání per-strategy limitů = nový `StrategyAllocation` dataclass + rozšíření `pre_trade_check()` o `strategy_id` parametr. Hardcoded limits se NEMĚNÍ. Odhadovaný effort: 8h.

### 3.6 Database
**PostgreSQL 16 + SQLAlchemy 2.0 async + Alembic migrations.** Aktuálně 3 verze migrací. Nová migrace = 5 nových tabulek + ALTER na 4 existujících. Effort: 6h (3h migration + 3h testing).

### 3.7 Layer Hodnocení — Moje protinávrhy

| Layer | CEO Hodnocení | CTO Hodnocení | Důvod odchylky |
|-------|--------------|---------------|----------------|
| L1 Market Maker | MOŽNÁ LATER | **Souhlasím** | Správně, ne MVP |
| L2 Value Signal | ADAPT → weather XGBoost | **REMOVE** | Weather nepotřebuje ML. NOAA 85-90% = deterministic edge, ne probabilistic |
| L3 Semantic Graph | REMOVE | **Souhlasím** | + bonus: ušetříme 1.3GB RAM |
| L4 Whale Monitor | ADAPT → Strategy A | **REMOVE** | L4 trackuje wallets, Strategy A analyzuje taker flow. Jsou to jiné problémy. Základ pro A je L7, ne L4 |
| L5 Logical Arb | REMOVE | **Souhlasím** | |
| L6 Temporal Crypto | REMOVE | **Souhlasím** | 500ms delay mrtvý, strategie mrtvá |
| L7 Order Flow | ADAPT → Strategy A | **Souhlasím** | Přesně. Event listener + RollingWindow = základ pro taker flow analysis |
| L8 Attention | ADAPT → Strategy B | **Partial agree** | Pattern je reusable ale data source se mění kompletně. Doporučuji nový modul s inspirací, ne in-place rewrite |
| L9 Sports Latency | REMOVE | **Souhlasím** | Už je disabled |

---

## 3. Souhrnná tabulka

| Category | Count | Modules | Total Hours |
|----------|-------|---------|-------------|
| **KEEP** | 4 | polymarket_client, fee_model, logger, slack_bot | 2h |
| **ADAPT** | 13 | market_discovery, polygon_flow, risk_manager, paper_engine, scanner, gemini_agent, web.py, report_generator, cli_dashboard, odds.py, db.py, main.py, settings.py+yaml | 84h |
| **REMOVE** | 22 | binance_client, odds_api_client, event_matcher, confluence, market_maker, value_signal, whale_discovery, whale_monitor, logical_arb, temporal_crypto, arb_monitor, sports_latency, xgboost_value, feature_engineering, xgboost_crypto, crypto_features, calibration, market_graph, 4× scripts | 0h |
| **NEW** | 8 | noaa_api, met_office_api, open_meteo_api, kaito_api, weather_specialist, theta_decay, reflexivity_surfer, quality_gates, resolution_chain, alembic migration | 62h |
| **TESTS** | — | Included in per-module estimates | (included) |
| **TOTAL** | **47** | | **~148h** |

---

## 4. Klíčové rizika identifikované auditorem

### Risk 1: Kaito API Neexistence (CRITICAL)
Strategy B (Reflexivity Surfer, 20% kapitálu) závisí na API které neexistuje. Launch je "březen 2026" — ale žádné konkrétní datum. Doporučuji:
- Sprint 3 (Strategy B) podmínit Kaito API availability
- Pokud API není do konce března, přealokovat €400 na Strategy C
- V mezičase implementovat stub/mock pro development

### Risk 2: Weather Market Efficiency (MEDIUM)
gopfan2 vydělal $2M, ale to přitáhne konkurenci. Spread compression je otázka měsíců. Mitigation: být mezi prvními automatizovanými boty, diversifikovat přes 5+ měst.

### Risk 3: Orchestrator Refactor (MEDIUM)
main.py je 889 řádků a centrum celého systému. Refactor na 3 strategie musí být atomic — částečný refactor = broken system. Doporučuji: psát nový orchestrator paralelně, ne refaktorovat stávající in-place.

### Risk 4: Database Migration (LOW-MEDIUM)
ALTER TABLE na production tabulky s daty. Mitigation: backup before migration, reversible Alembic downgrade, test on staging first.

---

## 5. Alternativní přístupy (sekce 5 z briefu)

### 5.1 Sprint struktura
Souhlasím s C → A → B pořadím (nejbezpečnější first). Navrhuji **3 sprinty místo 4** — Sprint 4 (integration + paper validation) je de facto continuous a nemusí být formální sprint.

### 5.2 Technology choices
- **SQLite vs PostgreSQL:** Zůstat u PostgreSQL. Máme funkční setup, async driver, Alembic migrations. SQLite by byla degradace.
- **Scheduling:** Zachovat asyncio-based scheduling v orchestrátoru (funguje, netřeba external scheduler)

### 5.3 Weather data sources
NOAA + Met Office + Open-Meteo je optimální kombinace. Všechny free, spolehlivé, dobře dokumentované. Žádný lepší alternativní zdroj neznám.

### 5.4 Dashboard
Zachovat FastAPI dashboard. Přestavba je ~10h, ne přepisování. Alternativa (Grafana) by vyžadovala nový stack — zbytečné.

### 5.5 Kaito API fallback
Viz Risk 1 výše. Doporučuji Option 3 (přealokovat na Strategy C) jako safest path.

---

*CTO, Arbo*
*25. února 2026*
