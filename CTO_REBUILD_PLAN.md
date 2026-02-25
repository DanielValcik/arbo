# CTO Rebuild Plan — Reflexive Decay Harvester

**Datum:** 25. února 2026
**Od:** CTO, Arbo
**Pro:** CEO ke schválení
**Klasifikace:** INTERNAL
**Ref:** CTO Audit Report (25.2.2026), CEO Audit Review (25.2.2026)

---

## 0. Kalendářní Assumptions

**Pracovní kapacita:** ~25h/týden (realistický odhad, ne optimistický)
**Celkový effort z auditu:** ~148h
**Kalendářní délka:** 6 týdnů development + 4 týdny paper validation = **10 týdnů celkem**

| Milestone | Datum | Poznámka |
|-----------|-------|----------|
| Sprint 1 kickoff | **2. března 2026** | Foundation + Strategy C |
| Sprint 1 complete | 20. března 2026 | Weather bot v paper mode |
| Sprint 2 complete | 10. dubna 2026 | Strategy A + archive cleanup |
| Sprint 3 complete | 1. května 2026 | Strategy B stub + new orchestrator |
| Paper trading start | **2. května 2026** | Všechny 3 strategie v paper mode |
| Paper validation end | 30. května 2026 | 4 consecutive weeks positive P&L |
| Earliest live trade | **1. června 2026** | Strategy C first, A+B phased in |

**Poznámka:** Strategy doc říká "First Possible Live Trade: May 2026" — můj odhad je **1. června 2026** (1 měsíc posun). Důvod: 148h při 25h/týden = 6 týdnů development + 4 týdny mandatory paper validation. Chci být realistický, ne optimistický.

---

## Sprint 0: Archive & Cleanup (Pre-Sprint)
**Délka:** 1 den (2. března 2026, před Sprint 1 kickoff)
**Cíl:** Vyčistit codebase — přesunout REMOVE moduly do archive, zrušit Odds API

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-001 | Archive removed modules | Přesunout 22 REMOVE modulů do `arbo/_archive/` s původní adresářovou strukturou. Zachovat git history. | `ls arbo/_archive/strategies/` obsahuje market_maker.py, value_signal.py, whale_discovery.py, whale_monitor.py, logical_arb.py, temporal_crypto.py, arb_monitor.py, sports_latency.py. `ls arbo/_archive/connectors/` obsahuje binance_client.py, odds_api_client.py, event_matcher.py. `ls arbo/_archive/models/` obsahuje xgboost_value.py, feature_engineering.py, xgboost_crypto.py, crypto_features.py, calibration.py, market_graph.py. `ls arbo/_archive/scripts/` obsahuje backfill_data.py, process_data.py, run_backtest.py, backfill_crypto.py. Archived tests v `arbo/_archive/tests/`. | 2h | None |
| RDH-002 | Archive confluence scorer | Přesunout `arbo/core/confluence.py` do `arbo/_archive/core/`. Odstranit importy z main.py (nahradit placeholder). | confluence.py v archive. `python3 -c "import arbo.core"` nechybí. | 1h | RDH-001 |
| RDH-003 | Cancel Odds API subscription | Zrušit The Odds API subscription ($30/mo). Odebrat ODDS_API_KEY z .env na serveru. | Subscription cancelled. Nový monthly budget ≤$31. | 0.5h | None |
| RDH-004 | Remove unused dependencies | Odebrat z requirements.txt / pyproject.toml: rapidfuzz, xgboost, optuna, scikit-learn, joblib, chromadb, sentence-transformers. Zachovat: web3, aiohttp, sqlalchemy, pydantic, structlog, slack-bolt, fastapi, py-clob-client. | `pip install -r requirements.txt` projde. Žádný import error. Memory usage na serveru klesne o ~1.3GB (e5-large-v2 gone). | 1h | RDH-001 |
| RDH-005 | Update imports & fix broken references | Projít main.py, settings.py a všechny KEEP/ADAPT moduly — odstranit importy archivovaných modulů. Dočasně stub orchestrator layer tasks (return None). | `python3 -m pytest arbo/tests/test_risk_manager.py arbo/tests/test_paper_engine.py arbo/tests/test_fee_model.py arbo/tests/test_slack_bot.py -v` — all pass. Žádné ImportError v KEEP/ADAPT modulech. | 2h | RDH-001, RDH-002 |

**Sprint 0 total:** 6.5h

---

## Sprint 1: Foundation + Strategy C (Weather)
**Délka:** 3 týdny (2.–20. března 2026)
**Cíl:** Funkční weather bot v paper trading mode pro 5 měst
**Budget:** ~75h (3 × 25h/týden)

### Week 1: Weather API Connectors + DB Migration

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-101 | NOAA API connector | `arbo/connectors/noaa_api.py` — REST client pro api.weather.gov. Fetch hourly + daily forecasts pro NYC a Chicago by station ID. Parse temperature (°F), precipitation, wind. 30min cache. Retry logic (NOAA má občasné výpadky). | `pytest test_noaa_api.py` — (1) fetch returns valid forecast for NYC (mock), (2) forecast contains temperature_f field, (3) cache returns same result within 30min, (4) handles NOAA 503 gracefully. | 6h | None |
| RDH-102 | Met Office API connector | `arbo/connectors/met_office_api.py` — REST client pro datahub.metoffice.gov.uk. API key auth (free tier). Fetch London spot + 3-hourly forecasts. Parse temperature (°C→°F conversion). 30min cache. | `pytest test_met_office_api.py` — (1) fetch returns London forecast (mock), (2) temperature converted to °F, (3) API key header present, (4) cache works. | 5h | None |
| RDH-103 | Open-Meteo connector | `arbo/connectors/open_meteo_api.py` — REST client pro open-meteo.com. No auth. Fetch hourly forecasts by lat/lon pro Seoul + Buenos Aires (extensible). Parse temperature. 30min cache. | `pytest test_open_meteo_api.py` — (1) fetch returns Seoul forecast (mock), (2) Buenos Aires forecast works, (3) no API key needed, (4) lat/lon configurable. | 4h | None |
| RDH-104 | Alembic migration 004 | `alembic/versions/004_rdh_schema.py` — CREATE TABLE: weather_forecasts, taker_flow_snapshots, attention_market_state, resolution_chains, strategy_allocation. ALTER TABLE: paper_trades ADD strategy, paper_positions ADD strategy, signals ADD strategy. Reversible downgrade. | `alembic upgrade head` succeeds. `alembic downgrade -1` succeeds. All new tables exist. ALTER columns have default 'legacy'. | 3h | None |
| RDH-105 | DB models for new tables | Přidat SQLAlchemy modely do `arbo/utils/db.py`: WeatherForecast, TakerFlowSnapshot, AttentionMarketState, ResolutionChain, StrategyAllocation. Přidat strategy field do PaperTrade, PaperPosition, Signal. | `pytest test_db_models.py` — (1) all models instantiable, (2) to_dict() works, (3) strategy field defaults to 'legacy'. | 3h | RDH-104 |

### Week 2: Weather Market Discovery + Weather Specialist

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-106 | Weather market discovery | Rozšířit `arbo/connectors/market_discovery.py`: přidat `get_weather_markets()` — filter by category keywords ("temperature", "weather", city names). Přidat bucket parsing (extrakce temperature range z question, e.g. "40-45°F" → {low: 40, high: 45}). Přidat `get_longshot_markets()` — YES price < $0.15, volume > $10K, fee-free, age > 24h. | `pytest test_market_discovery_weather.py` — (1) get_weather_markets() returns only weather markets (mock), (2) bucket parsing extracts correct range from "Will NYC temp be 40-45°F?", (3) get_longshot_markets() filters by price/volume/fee correctly. | 6h | None |
| RDH-107 | Weather specialist strategy | `arbo/strategies/weather_specialist.py` — Strategy C core logic. (1) Fetch forecasts z 3 connectors. (2) Match forecast → PM weather bucket. (3) Entry: forecast match + bucket price < $0.20 + volume > $500. (4) Position sizing $1-10. (5) Temperature laddering (adjacent buckets when forecast uncertainty spans 2+ buckets). (6) Hold to resolution (24h). (7) Early exit at price > $0.50. (8) Max $10 per trade, max 20 concurrent. | `pytest test_weather_specialist.py` — (1) forecast 43°F matches "40-45°F" bucket, (2) entry at price $0.15 → trade created, (3) entry rejected at price $0.30, (4) laddering places 3 trades across adjacent buckets, (5) max 20 concurrent respected, (6) paper trade recorded with strategy="C". | 14h | RDH-101, RDH-102, RDH-103, RDH-106 |
| RDH-108 | Weather quality gate | Přidat do `arbo/core/quality_gates.py` (nový modul): `QualityGate` base class + `WeatherGate`. WeatherGate: forecast confidence ≥ 85% + bucket price < $0.20 + risk check passed. | `pytest test_quality_gates.py` — (1) WeatherGate approves valid signal, (2) rejects low-confidence forecast, (3) rejects high-price bucket, (4) rejects when risk limit hit. | 4h | RDH-107 |

### Week 3: Resolution Chaining + Config + Integration

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-109 | Resolution chain engine | `arbo/core/resolution_chain.py` — Track chain state (chain_id, city_sequence, cumulative P&L). Settlement detection → auto-trigger next city deployment. Multi-city scheduling (NYC → London → Chicago → Seoul → Buenos Aires, timezone-aware). DB persistence (resolution_chains table). | `pytest test_resolution_chain.py` — (1) chain creates with NYC first, (2) after NYC settlement, London auto-triggered, (3) cumulative P&L tracks correctly across 3 cities, (4) chain persisted to DB, (5) chain resumes after restart. | 8h | RDH-105, RDH-107 |
| RDH-110 | Risk manager: per-strategy allocation | Rozšířit `arbo/core/risk_manager.py`: přidat StrategyAllocation dataclass. Přidat strategy-aware pre_trade_check() (validace proti strategy allocation). Per-strategy concurrent position limit (max 10). Reserve capital lock (€200 never deployed). Cross-strategy total exposure tracking. NEMĚNÍ SE: hardcoded limits. | `pytest test_risk_manager.py` — (1) strategy A allocation €400, (2) trade rejected when strategy A fully deployed, (3) reserve €200 untouchable, (4) max 10 positions per strategy enforced, (5) cross-strategy total ≤ 80%, (6) original 8 checks still pass. | 8h | None |
| RDH-111 | Paper engine: per-strategy tracking | Upravit `arbo/core/paper_engine.py`: strategy field na PaperTrade + PaperPosition. Per-strategy P&L tracking. Quarter-Kelly sizing (0.25 místo 0.5). get_strategy_stats(). Updated DB persistence. | `pytest test_paper_engine.py` — (1) trade records strategy="C", (2) per-strategy P&L correct, (3) quarter-Kelly sizes at 0.25, (4) snapshot contains per_strategy_pnl, (5) load_state_from_db restores strategy field. | 6h | RDH-105, RDH-110 |
| RDH-112 | Config: Strategy C settings | Přidat WeatherCompoundConfig do settings.py: cities (list), data_sources, bucket_entry_max_price ($0.20), max_trades_per_day (30), ladder_enabled (bool), chain_enabled (bool). Přidat StrategyAllocationConfig: a_pct (0.20), b_pct (0.20), c_pct (0.50), reserve_pct (0.10). Aktualizovat settings.yaml. | Config loads without error. `get_config().weather.bucket_entry_max_price == 0.20`. `get_config().allocation.c_pct == 0.50`. | 2h | None |
| RDH-113 | Sprint 1 integration test | End-to-end test: mock NOAA → forecast → bucket match → quality gate → paper trade → resolution → chain next city. | `pytest test_weather_e2e.py` — full pipeline produces ≥1 paper trade with strategy="C", resolution chains to next city. | 4h | RDH-107, RDH-108, RDH-109, RDH-111 |

**Sprint 1 total:** 73h (budgeted 75h — 2h buffer)

**Sprint 1 Go/No-Go Gate:**
- ✅ Weather bot places paper trades for 5 cities
- ✅ Resolution chaining works (settlement → next city)
- ✅ Temperature laddering works for uncertain forecasts
- ✅ Risk manager enforces per-strategy limits
- ✅ All new tests pass (`pytest arbo/tests/ -v`)

---

## Sprint 2: Strategy A (Theta Decay) + Scanner Cleanup
**Délka:** 3 týdny (21. března — 10. dubna 2026)
**Cíl:** Funkční theta decay bot v paper mode + polygon_flow adapted pro taker flow analysis
**Budget:** ~75h

### Week 4: Polygon Flow Adaptation

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-201 | Polygon flow: YES/NO taker ratio | Rozšířit `arbo/connectors/polygon_flow.py`: přidat per-market YES/NO taker flow tracking (ne globální buy/sell). Identifikovat YES vs NO z asset_id mapování (conditional token → outcome). Nový RollingWindow per condition_id. | `pytest test_polygon_flow.py` — (1) YES/NO ratio correctly computed from mock OrderFilled events, (2) per-market tracking isolates different markets, (3) ratio updates on new events. | 6h | None |
| RDH-202 | Polygon flow: 4h rolling + 3σ threshold | Změnit z-score window z 1h na 4h rolling (per strategy doc). Přidat 7-day historical average per market. 3σ peak optimism detection. DB persistence (taker_flow_snapshots table). | `pytest test_polygon_flow.py` — (1) z-score computed over 4h window, (2) 3σ threshold triggers peak_optimism flag, (3) 2.5σ does NOT trigger, (4) snapshots saved to DB. | 6h | RDH-201 |
| RDH-203 | Taker flow DB persistence | Implement save_taker_flow_snapshot() — periodic (every 5min) snapshots of per-market YES/NO flow, ratio, z-score to taker_flow_snapshots table. | `pytest test_taker_flow_db.py` — (1) snapshot saved with all fields, (2) query by market_id returns time series, (3) historical average calculable from stored data. | 3h | RDH-105, RDH-202 |

### Week 5: Theta Decay Strategy

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-204 | Theta decay strategy | `arbo/strategies/theta_decay.py` — Strategy A core logic. (1) Market selection: YES < $0.15, volume > $10K, fee-free, age > 24h, time-to-resolution 3-30d, category NOT finance. (2) Peak optimism detection: wait for 3σ YES taker flow. (3) Entry: buy NO at market price. (4) Exit: hold to resolution (primary), partial exit at NO +50% (sell 50%), stop loss at NO -30% (exit all). (5) Position sizing: $20-50, quarter-Kelly. (6) Max 10 concurrent positions. | `pytest test_theta_decay.py` — (1) market with YES=$0.10 passes filter, (2) YES=$0.20 rejected, (3) finance category rejected, (4) trade only placed when 3σ signal present, (5) no trade without signal, (6) partial exit at +50%, (7) full exit at -30%, (8) max 10 positions respected, (9) paper trade strategy="A". | 14h | RDH-202, RDH-106 |
| RDH-205 | Theta decay quality gate | Přidat `ThetaDecayGate` do quality_gates.py: z-score ≥ 3σ + market criteria (price, volume, fee, age, category) + risk check. | `pytest test_quality_gates.py` — (1) approves valid theta signal, (2) rejects sub-3σ, (3) rejects wrong category, (4) rejects when allocation exhausted. | 3h | RDH-204, RDH-108 |
| RDH-206 | Config: Strategy A settings | Přidat ThetaDecayConfig do settings.py: zscore_threshold (3.0), rolling_window_hours (4), longshot_price_max (0.15), min_volume (10000), min_age_hours (24), resolution_window_days_min (3), resolution_window_days_max (30), partial_exit_pct (0.50), stop_loss_pct (0.30). | Config loads. `get_config().theta_decay.zscore_threshold == 3.0`. | 1h | None |

### Week 6: Scanner Cleanup + Dashboard Adaptation

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-207 | Scanner refactor | Upravit `arbo/core/scanner.py`: zachovat Signal dataclass + SignalDirection enum jako universal DTOs. Přejmenovat `layer` field na `strategy`. Odstranit per-layer scan methods (scan_layer1_mm, scan_layer2_value, etc.). Přidat strategy-specific signal subtypes (WeatherSignal, ThetaDecaySignal, ReflexivitySignal) jako dataclass extensions. | `pytest test_scanner.py` — (1) Signal has strategy field, (2) WeatherSignal contains forecast_temp + bucket, (3) ThetaDecaySignal contains z_score + taker_ratio, (4) to_db_dict() includes strategy. | 3h | None |
| RDH-208 | Dashboard: per-strategy API routes | Upravit `arbo/dashboard/web.py`: `/api/layers` → `/api/strategies` (3 strategies). `/api/portfolio` → per-strategy P&L. Přidat `/api/weather` (forecast accuracy, chain status). Přidat `/api/taker-flow` (latest z-scores, flow charts). Přidat `/api/capital` (per-strategy utilization). | `pytest test_web_dashboard.py` — (1) /api/strategies returns 3 entries (A, B, C), (2) /api/portfolio contains per_strategy_pnl, (3) /api/weather returns forecast data, (4) /api/capital shows allocation vs deployed. | 10h | RDH-111 |
| RDH-209 | Report generator: per-strategy | Upravit report_generator.py: DailyReport/WeeklyReport per_strategy_pnl. Weather forecast accuracy in weekly report. Slack Block Kit formatting updated. | `pytest test_weekly_report.py` — (1) weekly report contains per_strategy breakdown, (2) weather accuracy metric present, (3) Slack formatting renders 3 strategies. | 4h | RDH-111 |
| RDH-210 | CLI dashboard + Slack bot update | cli_dashboard.py: 3-strategy status table. slack_bot.py: /status shows 3 strategies, /pnl shows per-strategy. | `pytest test_dashboard.py test_slack_bot.py` — (1) /status shows A/B/C, (2) /pnl breaks down by strategy. | 3h | RDH-208 |
| RDH-211 | Sprint 2 integration test | E2E: mock polygon events → taker flow spike 3σ → theta decay signal → quality gate → paper trade (buy NO) → hold. | `pytest test_theta_decay_e2e.py` — full pipeline produces paper trade with strategy="A" at 3σ signal. | 4h | RDH-204, RDH-205 |

**Sprint 2 total:** 57h (budgeted 75h — 18h buffer for unexpected issues)

**Sprint 2 Go/No-Go Gate:**
- ✅ Theta decay bot detects 3σ peak optimism and places paper NO trades
- ✅ Polygon flow correctly tracks per-market YES/NO taker ratio
- ✅ Dashboard shows 3 strategies with per-strategy P&L
- ✅ All tests pass

---

## Sprint 3: Strategy B (Reflexivity Stub) + New Orchestrator
**Délka:** 3 týdny (11. dubna — 1. května 2026)
**Cíl:** Strategy B architektonicky ready (stub), nový orchestrátor přepne systém
**Budget:** ~75h

### Week 7: Kaito API Stub + Reflexivity Surfer

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-301 | Kaito API stub connector | `arbo/connectors/kaito_api.py` — Stub implementace. Interface: `get_mindshare(topic) → MindshareData`, `get_sentiment(topic) → SentimentData`. Stub returns configurable mock data (pro development + paper testing). Real API integration placeholder (swap mock → HTTP client when API launches). Design contract tak aby přechod stub → live byl ≤4h práce. | `pytest test_kaito_api.py` — (1) get_mindshare returns MindshareData, (2) get_sentiment returns SentimentData, (3) stub mode flag, (4) mock data is configurable. | 5h | None |
| RDH-302 | Reflexivity surfer strategy | `arbo/strategies/reflexivity_surfer.py` — Strategy B core logic. (1) 4-phase state machine per market (Start → Boom → Peak → Bust). (2) Kaito data fetch (stub). (3) Divergence: (PM_price − Kaito_actual) / Kaito_actual. (4) Phase transitions: divergence > +20% = Phase 3 (sell YES, buy NO), divergence < -10% = Phase 1-2 (buy YES if momentum). (5) Phase 2 positions: $10-20, stop -15%. (6) Phase 3-4 positions: $20-50, stop -25%. (7) Max 5 concurrent per phase type. | `pytest test_reflexivity_surfer.py` — (1) state machine transitions correctly, (2) Phase 3 detected at +20% divergence, (3) Phase 2 trade placed (buy YES), (4) Phase 3 trade placed (buy NO), (5) stop loss at -15%/-25%, (6) max 5 per phase respected, (7) strategy="B" on paper trade. | 14h | RDH-301, RDH-106 |
| RDH-303 | Reflexivity quality gate | Přidat `ReflexivityGate` do quality_gates.py: divergence threshold met + phase confirmed (not Start) + risk check. | `pytest test_quality_gates.py` — (1) approves Phase 2 signal, (2) approves Phase 3 signal, (3) rejects Phase 1 (Start), (4) rejects insufficient divergence. | 3h | RDH-302, RDH-108 |
| RDH-304 | Config: Strategy B settings | Přidat ReflexivitySurferConfig: boom_divergence_threshold (-0.10), peak_divergence_threshold (0.20), phase2_max_position (20), phase3_max_position (50), phase2_stop_loss (0.15), phase3_stop_loss (0.25), max_concurrent_per_phase (5). | Config loads. Thresholds match strategy doc. | 1h | None |
| RDH-312 | Kaito API research (BLOCKING) | Aktivní research: existuje veřejné Kaito API? Dokumentace? Pricing? Rate limits? Přístupové podmínky? Kontaktovat Kaito tým pokud třeba. Deliverable: report pro CEO s URL, pricing, availability status. **Deadline: 18. dubna 2026 (Week 7).** CEO directive — žádná automatická realokace bez vlastníkova rozhodnutí. | CTO dodá written report: Kaito API status (available/unavailable/gated), dokumentace URL, pricing tier, rate limits, kontakt s Kaito týmem (pokud proběhl). | 3h | None |

### Week 8: New Orchestrator

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-305 | New orchestrator (main_rdh.py) | `arbo/main_rdh.py` — Nový orchestrátor pro 3 strategie. Task-per-strategy model (3 strategy tasks + discovery + shared infrastructure). Signal queue → per-strategy quality gates (NE confluence). Health monitor (crash detection, auto-restart). Graceful shutdown (SIGTERM). DB init, Slack init, state restore. Scheduled reports (daily 23:00, weekly Sun 20:00). Hourly snapshots. Position price updater (5min). Resolution checker (5min). Data collector (hourly). Capital allocation engine (A: €400, B: €400, C: €1000, Reserve: €200). | `pytest test_orchestrator_rdh.py` — (1) starts 3 strategy tasks, (2) signals routed to correct quality gate, (3) trade placed via paper engine with strategy field, (4) health monitor detects crashed task, (5) graceful shutdown cancels all tasks, (6) daily/weekly reports scheduled, (7) capital allocation enforced. | 20h | RDH-107, RDH-204, RDH-302, RDH-108, RDH-110, RDH-111 |
| RDH-306 | Orchestrator switchover plan | Implement `__main__.py` update: `--mode rdh` flag spouští main_rdh.py místo main.py. Starý orchestrátor zůstává jako `--mode legacy`. Paper trading data migration: strategy='legacy' pro existující trades. Rollback: `--mode legacy` vrátí starý systém. | `python3 -m arbo.main --mode rdh` starts new orchestrator. `--mode legacy` starts old. Paper data preserved. | 3h | RDH-305 |

### Week 9: Gemini Cleanup + Final Integration

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-307 | Gemini agent: slim down | Upravit gemini_agent.py: odstranit predict() (probability estimation pro L2/L5/L8). Zachovat raw_query() + rate limiting + _extract_json(). Přidat Kaito-fallback query method (LLM mindshare estimation jako Plan B pro Strategy B). | `pytest test_gemini_agent.py` — (1) raw_query works, (2) rate limiter works, (3) _extract_json works, (4) predict() removed. | 2h | None |
| RDH-308 | Odds utility: quarter-Kelly | Přidat `quarter_kelly()` do odds.py (KELLY_FRACTION=0.25). Zachovat half_kelly() pro backward compat. | `pytest test_odds.py` — (1) quarter_kelly(edge=0.10, odds=2.0) returns correct fraction, (2) max 5% cap, (3) half_kelly unchanged. | 1h | None |
| RDH-309 | Settings cleanup | Odstranit deprecated config models: MarketMakerConfig, ValueModelConfig, ConfluenceConfig, LogicalArbConfig, BinanceConfig, TemporalCryptoConfig, SportsLatencyConfig, OddsApiConfig. Aktualizovat settings.yaml. | Settings load without deprecated sections. No import errors. | 3h | RDH-112, RDH-206, RDH-304 |
| RDH-310 | Full system E2E test | End-to-end: new orchestrator starts → discovery fetches markets → Strategy C places weather trade → Strategy A detects taker flow → Strategy B in stub mode → reports generate → dashboard serves API. | `pytest test_rdh_e2e.py -v` — orchestrator runs 60s, places ≥1 weather trade, taker flow monitor active, stub B running, all APIs respond 200. | 6h | RDH-305, RDH-306 |
| RDH-311 | Deploy to VPS | Aktualizovat deploy.sh pro nový orchestrátor. arbo.service: ExecStart s `--mode rdh`. Deploy, verify all 3 strategies active. | SSH to VPS: `systemctl status arbo` shows active. Slack /status shows 3 strategies. Dashboard loads. | 3h | RDH-310 |

**Sprint 3 total:** 64h (budgeted 75h — 11h buffer)

**Sprint 3 Go/No-Go Gate:**
- ✅ New orchestrator runs all 3 strategies concurrently
- ✅ Strategy C places weather paper trades
- ✅ Strategy A detects taker flow spikes and places theta decay trades
- ✅ Strategy B runs in stub mode (no real Kaito data)
- ✅ Dashboard shows 3 strategies
- ✅ Switchover from legacy orchestrator is reversible
- ✅ VPS deployed and running

---

## Phase 4: Mandatory Paper Trading Validation
**Délka:** 4 týdny (2. — 30. května 2026)
**Cíl:** 4 consecutive weeks positive combined P&L

**Toto NENÍ development sprint.** Toto je monitoring + tuning phase.

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-401 | Week 1 paper validation | Monitor all 3 strategies. Daily check: trades placed, P&L tracking, no errors. Weekly review: per-strategy P&L, win rate, drawdown. **Per-strategy kill switch check:** pokud jakákoliv strategie má weekly drawdown > 15% své alokace → zastavit → eskalace CEO. **Weekly metrics report:** Sharpe ratio (annualizovaný), profit factor (gross wins / gross losses), average trade P&L — per strategie. Tune thresholds if needed. | Week 1 P&L > $0 combined. No strategy exceeds 15% weekly drawdown. All strategies active. Sharpe/profit factor/avg trade tracked. | 6h | RDH-311 |
| RDH-402 | Week 2 paper validation | Same as Week 1. Focus: Strategy C weather accuracy vs NOAA forecast accuracy. Resolution chain efficiency. **Kill switch + metrics tracking continues.** | Week 2 P&L > $0 combined. No 15% drawdown breach. Weather accuracy ≥ 80%. Chain completes ≥ 3 cities. | 6h | RDH-401 |
| RDH-403 | Week 3 paper validation | Same. Focus: Strategy A signal quality. Are 3σ signals actually identifying peak optimism? Win rate on NO positions. **Kill switch + metrics tracking continues.** | Week 3 P&L > $0 combined. No 15% drawdown breach. Strategy A win rate tracking. | 6h | RDH-402 |
| RDH-404 | Week 4 paper validation + CEO review | Final week. Compile 4-week report. **Full metrics dashboard:** per-strategy P&L, win rate, Sharpe ratio, profit factor, avg trade P&L, max drawdown, resolution chain stats. Present to CEO for go-live approval. | 4 consecutive weeks P&L > $0. No 15% drawdown breaches in any week. CEO approves go-live. | 6h | RDH-403 |
| RDH-405 | Kaito API integration (conditional) | Pokud Kaito API launched (per RDH-312 research) → implementovat real connector (nahradit stub, ≤4h). Pokud ne → eskalace CEO/vlastník pro rozhodnutí. **Žádná automatická realokace.** | Kaito real integration complete OR CEO decision documented. | 4h | RDH-301, RDH-312 |

**Phase 4 total:** 28h (6h/week monitoring with kill switch + metrics + 4h conditional Kaito)

---

## Phase 5: Go-Live (Earliest: 1. června 2026)

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-501 | Live mode: Strategy C only | Switch weather bot to live mode. Start with 50% capital (€500 of €1000 allocation). Monitor 24h. | First live trade placed. No errors. P&L tracking active. | 3h | RDH-404 |
| RDH-502 | Scale up + add Strategy A | After 1 week live C: scale to 100% C capital + add Strategy A live (50% of A allocation = €200). | Both strategies live. Capital allocation correct. | 2h | RDH-501 |
| RDH-503 | Add Strategy B (if Kaito ready) | After 2 weeks: add Strategy B if Kaito API available and Attention Markets have liquidity. Otherwise skip. | Strategy B live OR documented skip with capital reallocation. | 2h | RDH-502, RDH-405 |
| RDH-504 | Full scale (Week 3+) | All strategies at 100% capital allocation. 75% week 2, 100% week 3 per strategy doc. | €2000 fully deployed (minus €200 reserve). | 1h | RDH-503 |

**Phase 5 total:** 8h

---

## Dependency Graph

```
Sprint 0 (Archive)
  RDH-001 ──→ RDH-002 ──→ RDH-005
  RDH-001 ──→ RDH-004
  RDH-003 (independent)

Sprint 1 (Weather)
  RDH-101 ─┐
  RDH-102 ─┼──→ RDH-107 ──→ RDH-108 ──→ RDH-113
  RDH-103 ─┘                    │
  RDH-104 ──→ RDH-105 ──→ RDH-109 ──→ RDH-113
  RDH-106 ──→ RDH-107            │
  RDH-110 (independent) ──→ RDH-111 ──→ RDH-113
  RDH-112 (independent)

Sprint 2 (Theta Decay)
  RDH-201 ──→ RDH-202 ──→ RDH-203
  RDH-202 ──→ RDH-204 ──→ RDH-205 ──→ RDH-211
  RDH-206 (independent)
  RDH-207 (independent)
  RDH-208 ──→ RDH-210
  RDH-209 (independent)

Sprint 3 (Reflexivity + Orchestrator)
  RDH-301 ──→ RDH-302 ──→ RDH-303
  RDH-304 (independent)
  RDH-312 (independent, BLOCKING research — deadline 18.4.)
  RDH-107 + RDH-204 + RDH-302 ──→ RDH-305 ──→ RDH-306 ──→ RDH-310 ──→ RDH-311
  RDH-307, RDH-308, RDH-309 (independent)

Phase 4 (Validation)
  RDH-311 ──→ RDH-401 ──→ RDH-402 ──→ RDH-403 ──→ RDH-404
  RDH-301 + RDH-312 ──→ RDH-405 (conditional, CEO decision)

Phase 5 (Go-Live)
  RDH-404 ──→ RDH-501 ──→ RDH-502 ──→ RDH-503 ──→ RDH-504
```

---

## Kalendářní Timeline

```
BŘEZEN 2026
  1  Mon  ░░░░░░░░░░░░░░░░░░░░ Sprint 0 (Archive + Cleanup)
  2  Tue  ████████████████████ Sprint 1 START
  3-7      ████ Week 1: Weather API connectors + DB migration
  8-14     ████ Week 2: Weather specialist + quality gates
  15-20    ████ Week 3: Resolution chaining + risk manager + integration
  20 Fri  ████████████████████ Sprint 1 COMPLETE ✓ (weather bot paper)
  21 Mon  ████████████████████ Sprint 2 START
  22-28    ████ Week 4: Polygon flow adaptation

DUBEN 2026
  29-4     ████ Week 5: Theta decay strategy
  5-10     ████ Week 6: Dashboard + scanner cleanup + integration
  10 Fri  ████████████████████ Sprint 2 COMPLETE ✓ (theta decay paper)
  11 Mon  ████████████████████ Sprint 3 START
  12-18    ████ Week 7: Kaito stub + reflexivity surfer
  19-25    ████ Week 8: New orchestrator
  26-1     ████ Week 9: Gemini cleanup + E2E + VPS deploy

KVĚTEN 2026
  1  Thu  ████████████████████ Sprint 3 COMPLETE ✓ (all 3 strategies)
  2  Fri  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ PAPER TRADING START
  2-9      ▓▓▓▓ Paper Week 1
  10-16    ▓▓▓▓ Paper Week 2
  17-23    ▓▓▓▓ Paper Week 3
  24-30    ▓▓▓▓ Paper Week 4 + CEO Review

ČERVEN 2026
  1  Mon  ★★★★★★★★★★★★★★★★★★★★ LIVE TRADING START (Strategy C)
  8  Mon  ★★★★ Add Strategy A (50%)
  15 Mon  ★★★★ Add Strategy B (if Kaito ready)
  22 Mon  ★★★★ Full scale (100% capital deployed)
```

---

## Risk Register

| # | Risk | Probability | Impact | Mitigation | Owner |
|---|------|-------------|--------|------------|-------|
| R1 | Kaito API not available by May 2026 | HIGH | Medium (Strategy B delayed) | Stub-first approach. RDH-312: Kaito API research (deadline 18.4.). Fallback: CEO/vlastník rozhoduje — žádná automatická realokace. Architecture ready for instant integration (≤4h swap stub→live). | CTO+CEO |
| R2 | Weather market efficiency increases during development | Medium | Medium (reduced Strategy C returns) | Diversify across 5+ cities. Monitor spread compression. Expand to wind/precipitation if temperature becomes efficient. | CTO |
| R3 | Orchestrator refactor breaks system | Medium | HIGH (system down) | New orchestrator built in parallel (main_rdh.py). Legacy mode preserved (`--mode legacy`). Rollback ≤5 minutes. | CTO |
| R4 | NOAA API outages during paper trading | Low | Low (temporary data gap) | 3 independent weather sources (NOAA, Met Office, Open-Meteo). If one down, others continue. Retry with exponential backoff. | CTO |
| R5 | 3σ taker flow threshold too strict/loose | Medium | Medium (Strategy A under/over-trades) | Calibrate during paper trading. Start with 3σ, adjust based on signal quality. Track false positive rate. | CTO |
| R6 | Database migration data loss | Low | HIGH (history lost) | Backup before migration. Reversible Alembic downgrade. Test on staging DB first. ALTER TABLE with defaults (no data loss). | CTO |
| R7 | Polymarket fee changes (again) | Medium | Variable | Modular architecture. Per-strategy independence. If weather gets fees, reassess bucket pricing threshold. Pivot capital within 24h. | CEO+CTO |
| R8 | Paper trading shows negative P&L | Medium | HIGH (delays go-live) | 4-week validation catches this. Strategy-by-strategy evaluation. Disable underperforming strategy, reallocate capital. Clock resets on negative week. | CEO |
| R9 | VPS memory pressure with 3 strategies | Low | Medium | Removed e5-large-v2 (saves 1.3GB). New connectors are lightweight HTTP clients. Monitor with psutil. Pre-approved upgrade to 8GB if needed. | CTO |

---

## Go/No-Go Criteria for Live Trading

**ALL conditions must be met. No exceptions.**

| # | Criterion | Measurement | Threshold |
|---|-----------|-------------|-----------|
| 1 | Paper P&L | 4 consecutive weeks combined P&L | > $0 each week |
| 2 | Strategy C win rate | Weather trades resolved / total | ≥ 70% (CEO Ú1: sníženo z 75%) |
| 3 | Strategy A signal quality | Peak optimism → NO wins | ≥ 55% win rate (CEO Ú1: sníženo z 65% — asymetrický payoff ~3:1) |
| 4 | System stability | Uptime during paper phase | ≥ 99% (≤ 7h downtime / 4 weeks) |
| 5 | Risk limits | No hardcoded limit breaches | 0 breaches |
| 6 | Resolution chaining | Multi-city chains completed | ≥ 10 chains across 5 cities |
| 7 | Dashboard accuracy | Reported P&L matches DB trades | 100% match |
| 8 | CEO approval | Weekly review + final sign-off | Explicit "GO" from CEO |

---

## Total Effort Summary

| Phase | Hours | Calendar |
|-------|-------|----------|
| Sprint 0 (Archive) | 6.5h | 1 day |
| Sprint 1 (Weather) | 73h | 3 weeks |
| Sprint 2 (Theta Decay) | 57h | 3 weeks |
| Sprint 3 (Reflexivity + Orchestrator) | 64h | 3 weeks |
| Phase 4 (Paper Validation) | 28h | 4 weeks |
| Phase 5 (Go-Live) | 8h | 3+ weeks |
| **TOTAL** | **236.5h** | **~14 weeks** |

**Poznámka:** 236.5h vs 148h z auditu — rozdíl je integration testing (E2E testy per sprint), VPS deployment, paper validation monitoring (including Sharpe/profit factor/kill switch tracking per CEO Ú2/Ú3), Kaito API research (RDH-312), a go-live tasks které audit nezahrnoval. 148h bylo čistě development effort, 236.5h je celkový project effort.

---

## CEO Direktivy — Implementováno (25.2.2026)

Všechny CEO úpravy z Audit Review zapracovány:

| Direktiva | Status | Změna v plánu |
|-----------|--------|---------------|
| **Ú1:** Go/No-Go thresholdy snížit | ✅ | Strategy C: 75% → 70%. Strategy A: 65% → 55%. |
| **Ú2:** Sharpe ratio + profit factor + avg trade P&L tracking | ✅ | Přidáno do RDH-401 až RDH-404. Weekly metrics report. |
| **Ú3:** Per-strategy kill switch (15% weekly drawdown) | ✅ | Přidáno do RDH-401 až RDH-404. Breach → halt + CEO eskalace. |
| **Ú4:** Sprint 1 buffer → edge case testing | ✅ | Instrukce: ušetřený čas investovat do RDH-107 edge cases (bucket parsing, temperature laddering). |
| **RDH-312:** Kaito API research | ✅ | Nový task v Sprint 3 Week 7. Deadline 18.4.2026. BLOCKING. |
| **Strategy B fallback:** Žádná automatická realokace | ✅ | RDH-405 updatován. CEO/vlastník rozhoduje. |
| **Slack potvrzeno** | ✅ | Telegram zmínka v strategy doc ignorována. |
| **Reporting cadence** | ✅ | Pondělí 9:00 sprint status, pátek 17:00 weekly summary, sprint end go/no-go. |

**Sprint 1 Ú4 instrukce:** Pokud jakýkoliv task v Sprint 1 dokončen pod odhadem, investovat ušetřený čas do edge case testů pro RDH-107 (weather specialist):
- Bucket parsing: edge cases (°C vs °F, negative temps, ranges spanning zero)
- Temperature laddering: edge cases (exactly on boundary, single-degree buckets, 3+ bucket spans)
- City timezone handling (Buenos Aires = UTC-3, Seoul = UTC+9, London = GMT/BST)

---

*CTO, Arbo*
*25. února 2026*
*Plan schválen CEO 25.2.2026. Sprint 1 kickoff: 2. března 2026.*
