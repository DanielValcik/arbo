# CEO → CTO: Strategic Pivot — System Audit & Rebuild Plan

**Datum:** 25. února 2026
**Od:** CEO, Arbo
**Pro:** CTO & Development Team
**Priorita:** HIGH — Strategická změna architektury
**Klasifikace:** INTERNAL

---

## 0. Proč tento dokument

Na základě hloubkové analýzy trhu (72M tradů Becker study, 86M betů IMDEA study, on-chain analýza Polymarket) a vyhodnocení výsledků stávajícího systému provádíme **strategický pivot**. 9-vrstvá confluence architektura se ukázala jako nefunkční pro naši kapitálovou úroveň (€2K) — za 2.5 hodiny vygenerovala 37,556 L2 signálů a 0 confluence matches.

Nová strategie se jmenuje **Reflexive Decay Harvester** a skládá se ze 3 nezávislých strategických pilířů místo 9 vrstev. Detailní strategický dokument (`arbo_strategy_v1.docx`) je přílohou — přečti ho celý před tím, než začneš pracovat.

**Tvůj první úkol není kódovat. Je to audit.**

---

## 1. Co od tebe potřebuji (2 deliverables)

### Deliverable 1: System Audit Report

**Deadline:** 72 hodin od přijetí tohoto briefu

Projdi celý stávající codebase a pro **každý modul/soubor** rozhodni:

| Kategorie | Význam |
|-----------|--------|
| **KEEP** | Modul je použitelný as-is nebo s minimální úpravou (<2h práce) |
| **ADAPT** | Modul obsahuje užitečný kód, ale vyžaduje významnou přestavbu |
| **REMOVE** | Modul nemá v nové architektuře využití — archivovat, nemazat |
| **NEW** | Potřebujeme modul který neexistuje |

Pro každý modul chci:

```
Soubor: arbo/strategies/value_signal.py
Kategorie: ADAPT
Důvod: XGBoost ensemble model je základ pro weather forecast confidence scoring.
       Ale aktuální logika (Pinnacle odds srovnání, multi-category edge detection)
       je pro novou strategii irelevantní. Přestavět na weather-specific model.
Odhad práce: 8h
Závislosti: Nový NOAA connector musí existovat first.
```

### Deliverable 2: Rebuild Plan

**Deadline:** 5 dní od přijetí tohoto briefu (= 2 dny po auditu)

Na základě auditu navrhni:

1. **Sprint plán** — kolik sprintů, co v každém, task IDčka (RDH-XXX)
2. **Dependency graph** — co musí být hotové před čím
3. **Risk assessment** — kde vidíš technické riziko, co může zdržet
4. **Timeline estimate** — realistický, ne optimistický
5. **Alternativní přístupy** — pokud někde vidíš lepší řešení než co navrhuji, řekni

Rebuild Plan mi pošleš ke schválení. **Nekóduješ nic dokud plan neschválím.**

---

## 2. Nová architektura — přehled pro audit

Starý systém: 9 vrstev → confluence scoring → trade execution
Nový systém: 3 nezávislé strategie → per-strategy quality gates → trade execution

### Strategy A: Theta Decay (Longshot NO Selling)

**Co dělá:** Prodává optimism premium na longshot YES kontraktech (p < 0.15), timed na peak optimism momenty detekované z on-chain taker flow.

**Potřebuje:**
- Polygon event monitor (OrderFilled events z CTF Exchange)
- Taker flow analysis (YES/NO ratio, z-score, rolling window)
- Peak optimism detection (3σ threshold)
- Market scanner (Gamma API filter: p < 0.15, volume > $10K, fee-free)
- Quarter-Kelly sizing
- Paper trading mode

### Strategy B: Reflexivity Surfer (Attention Markets)

**Co dělá:** Traduje reflexivní boom-bust cykly v nové kategorii Attention Markets (Polymarket × Kaito AI, launch březen 2026).

**Potřebuje:**
- Kaito API connector (mindshare/sentiment data) — API zatím neexistuje, launch březen
- Price vs reality divergence calculator
- 4-fázový state machine (Start → Boom → Peak → Bust)
- Phase transition detection (>20% divergence trigger)
- Paper trading mode

### Strategy C: Compound Weather Resolution Chaining

**Co dělá:** Exploituje weather market mispricing pomocí NOAA/Met Office dat s compound resolution chainingem (kapitál se nikdy nezastaví — daily settlement → okamžitý re-deploy do dalšího města).

**Potřebuje:**
- NOAA API connector (NYC, Chicago) — api.weather.gov, free
- Met Office API connector (London) — datahub.metoffice.gov.uk, free tier
- Open-Meteo connector (Seoul, Buenos Aires) — open-meteo.com, free
- Weather market scanner (Gamma API: weather kategorie, bucket matching)
- Temperature laddering logic (paralelní bety přes sousední buckety)
- Resolution chaining engine (settlement detection → auto re-deploy)
- Paper trading mode

### Sdílená infrastruktura (všechny strategie)

- Risk manager singleton (hardcoded limity — BEZ ZMĚN: 5% pozice, 10% denní loss, 20% týdenní)
- Polymarket auth (L1 + L2) — py-clob-client
- Order execution engine (post-only orders, batch up to 15)
- Heartbeat manager (5s interval, auto-reconnect)
- Position tracker (cross-strategy)
- Paper trading engine
- Dashboard feed (arbo.click)
- Telegram notifications

---

## 3. Specifické otázky pro audit

Pro každou oblast potřebuji konkrétní odpověď:

### 3.1 Polymarket Integration

Stávající systém má funkční:
- Auth (L1 + L2) ✓
- py-clob-client order placement ✓
- Market discovery via Gamma API ✓
- WebSocket price feeds ✓
- Settlement detection ✓
- Paper trading engine ✓

**Otázka:** Jak moc je toto svázané s 9-layer confluence logikou? Můžeme extrahovat čisté Polymarket integration moduly bez závislosti na starém orchestrátoru?

### 3.2 On-Chain Monitoring (L7 Order Flow)

L7 měl problémy (restart, Alchemy CU limit, přechod na dRPC). Ale základní logika — Polygon event listener pro OrderFilled eventy — je přesně to co potřebujeme pro Strategy A (taker flow analysis).

**Otázka:** Jaký je aktuální stav L7 kódu? Funguje dRPC connection? Je incremental block tracking implementovaný (D3 fix)? Můžeme L7 event listener přepoužít jako základ pro taker flow monitor?

### 3.3 L8 Attention Layer

L8 generoval 25 signálů ale 0 tradů. Pro Strategy B (Reflexivity Surfer) potřebujeme sentiment/mindshare data, ale z Kaito AI, ne z aktuálního zdroje.

**Otázka:** Co přesně L8 dělá teď? Jaký je zdroj dat? Je tam něco použitelného pro Kaito API integration, nebo je to kompletní rewrite?

### 3.4 Dashboard

Dashboard na arbo.click funguje. Nová architektura má jiné metriky:
- Per-strategy P&L (A, B, C zvlášť)
- Weather forecast accuracy tracking
- Taker flow visualization
- Divergence meter (Strategy B)
- Capital utilization per strategy
- Resolution chain status (Strategy C)

**Otázka:** Jak je dashboard architektonicky postavený? Je to monolitický frontend, nebo modulární s API endpointy? Jak náročné je přidat nové metriky?

### 3.5 Risk Manager

Hardcoded limity se NEMĚNÍ. Ale risk manager potřebuje novou logiku:
- Per-strategy alokace (A: €400, B: €400, C: €1,000, Reserve: €200)
- Per-strategy position limits (max 10 concurrent per strategy)
- Cross-strategy total exposure tracking
- Reserve capital lock (€200 nikdy deployed)

**Otázka:** Je aktuální risk manager singleton pattern? Jak těžké je přidat per-strategy limity?

### 3.6 Database

Stávající schéma má `paper_trades`, system state, market cache. Nová architektura potřebuje:
- `weather_forecasts` (city, date, source, forecast_temp, actual_temp, accuracy)
- `taker_flow_snapshots` (market_id, timestamp, yes_flow, no_flow, ratio, z_score)
- `attention_market_state` (market_id, phase, kaito_mindshare, pm_price, divergence)
- `resolution_chains` (chain_id, city_sequence, cumulative_pnl, status)
- `strategy_allocation` (strategy, allocated, deployed, available)

**Otázka:** Jaké ORM/migration tool používáme? Alembic? Je DB PostgreSQL nebo SQLite? Jak velká je migrace?

### 3.7 Stávající strategie (L1–L9)

Explicitní rozhodnutí pro každou vrstvu:

| Layer | Nové využití | Poznámka |
|-------|-------------|----------|
| L1 Market Maker | MOŽNÁ LATER | Pokud přidáme fee-free MM jako doplněk. Není v MVP. |
| L2 Value Signal | ADAPT | XGBoost model → přetrénovat na weather data |
| L3 Semantic Graph | REMOVE | Nová architektura nepotřebuje market relationship graph |
| L4 Whale Monitor | ADAPT → Strategy A | On-chain wallet tracking → taker flow analysis |
| L5 Logical Arb | REMOVE | LLM-driven arbitrage není v nové strategii |
| L6 Temporal Crypto | REMOVE | 500ms delay odstraněn, taker fees. Strategie je mrtvá. |
| L7 Order Flow | ADAPT → Strategy A | Event listener → taker/maker ratio calculation |
| L8 Attention | ADAPT → Strategy B | Sentiment analysis → Kaito API integration |
| L9 Sports Latency | REMOVE | Live sports data latency není v nové strategii |

**Otázka:** Souhlasíš s tímto hodnocením? Pokud ne, argumentuj. Tohle je tvůj audit — pokud vidíš lepší využití nějakého modulu, řekni.

---

## 4. Co NESMÍŠ změnit

Následující jsou non-negotiable. Neměň je, nenavrhuj změny, nerelaxuj:

1. **Risk limity:** 5% max position, 10% daily loss, 20% weekly loss
2. **Paper trading requirement:** 4 consecutive weeks positive P&L BEFORE live
3. **Reserve capital:** €200 (10%) nikdy deployed
4. **Heartbeat:** 5s interval, auto-cancel on disconnect
5. **Post-only orders:** Vždy maker, nikdy taker (fee avoidance)
6. **Coding standards:** ruff + black clean, pytest coverage, type hints

---

## 5. Co MŮŽEŠ navrhnout jinak

Následující jsou moje doporučení, ale přijímám argumentovaný protinávrh:

1. **Sprint počet a délka** — navrhuji 4 sprinty po 2-3 týdnech, ale pokud vidíš efektivnější rozdělení, navrhni
2. **Priorita strategií** — navrhuji C → A → B (nejbezpečnější first), ale pokud A má lepší reuse stávajícího kódu, můžeš přehodit
3. **Technology choices** — SQLite vs PostgreSQL, FastAPI vs jiný framework, scheduling library
4. **Weather data sources** — pokud znáš lepší free API než NOAA/Met Office/Open-Meteo, navrhni
5. **Dashboard framework** — pokud je přestavba dashboardu příliš drahá, navrhni alternativu
6. **Kaito API fallback** — Strategy B závisí na Kaito API (launch březen 2026). Pokud API nebude dostupné, potřebujeme Plan B. Navrhni.

---

## 6. Formát audit reportu

Chci structured report, ne volný text. Pro každý soubor v codebase:

```markdown
## Module: arbo/connectors/polymarket_clob.py

**Category:** KEEP
**Current function:** CLOB order placement, auth, heartbeat
**New function:** Identical — shared infrastructure for all 3 strategies
**Changes needed:** None
**Estimated effort:** 0h
**Dependencies:** None
**Risk:** Low
```

Na konci audit reportu chci souhrnnou tabulku:

```markdown
| Category | Count | Total Hours |
|----------|-------|-------------|
| KEEP     | X     | Xh          |
| ADAPT    | X     | Xh          |
| REMOVE   | X     | 0h          |
| NEW      | X     | Xh          |
| **TOTAL**| **X** | **Xh**      |
```

---

## 7. Formát rebuild plánu

Pro každý sprint:

```markdown
### Sprint 1: Foundation + Strategy C (Weather)
**Délka:** 2-3 týdny
**Cíl:** Funkční weather bot v paper trading mode

| Task ID | Název | Popis | Acceptance Test | Odhad | Závislost |
|---------|-------|-------|-----------------|-------|-----------|
| RDH-001 | NOAA connector | ... | pytest: fetch returns valid forecast for NYC | 4h | None |
| RDH-002 | ... | ... | ... | ... | RDH-001 |
```

---

## 8. Timeline

| Deadline | Deliverable | Owner |
|----------|-------------|-------|
| 28 Feb 2026 | **Audit Report** — per-module KEEP/ADAPT/REMOVE/NEW | CTO |
| 2 Mar 2026 | **Rebuild Plan** — sprints, tasks, timeline, risks | CTO |
| 3 Mar 2026 | CEO review & feedback | CEO |
| 4 Mar 2026 | Plan finalizace + Sprint 1 kickoff | CTO + CEO |

---

## 9. Přílohy

- `arbo_strategy_v1.docx` — Kompletní strategický dokument (povinné čtení)
- Stávající CTO Development Brief v3 — referenční dokument pro task IDs z předchozích sprintů
- CEO → CTO Memo z 25.2.2026 (paper trading review) — kontext pro audit

---

## 10. Závěrem

Tohle je největší architektonická změna od založení projektu. Nespěchej na to. Důkladný audit a promyšlený plán ušetří týdny debuggingu později.

Pokud během auditu najdeš něco co mění moje předpoklady — třeba modul který jsem označil jako REMOVE ale má velkou hodnotu pro novou architekturu — řekni. Proto děláme audit a ne rovnou sprint kickoff.

**Potvrď přijetí tohoto briefu a odhadni kdy dodáš Audit Report.**

---

*CEO, Arbo*
*25. února 2026*
