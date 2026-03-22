# Session Handoff — 2026-03-22

> Zadání pro další session. Kompletní kontext co se dělo, co funguje, co je rozbité.

---

## Co jsme dělali (2026-03-15 → 2026-03-22)

### Strategy D: Live Edge Harvester — sportovní data
- Navrhli kompletní strategii (spec: `docs/STRATEGY_D_SPEC.md`)
- Postavili data infrastrukturu: `research_d/` (SQLite, Elo/Glicko-2, downloadery)
- **První backtest na 41 NBA tradech: +36.7% ROI, 82.9% WR, 73% green book rate**
- Stáhli 150K sportovních marketů v 10-min z PolymarketData.co Ultra ($360/mo)
- **Pass 2 (1-min game window) teď běží na download VPS** (18.134.241.60)
- Stáhli weather data: 19 měst, 10K marketů, 7.4M cen (`research/data/weather_pmd.sqlite`)

### Resolution Bug Fix (částečně)
Opravili 3 bugy v `arbo/main_rdh.py` ale resolution stále nefunguje úplně.

---

## KRITICKÝ NEVYŘEŠENÝ PROBLÉM: Trade Resolution

### Symptom
75+ paper tradů (strategie A, B, C) zůstává ve stavu "open" i po dnech/týdnech. Dashboard ukazuje jen 3 resolved trady za celou dobu provozu. Slack hlásí "Žádné obchody za 24h" — protože nic se neresolvuje.

### Root Cause Analýza (hotová)

Resolution checker (`main_rdh.py:1417 _resolution_checker`) běží každých 5 minut, kontroluje 30 open pozic. Ale `fetch_by_token_id()` (`market_discovery.py:831`) vrací `None` pro všechny naše pozice → resolution přeskočí.

**Proč `fetch_by_token_id` vrací None:**
Volá Gamma API: `GET /markets?clob_token_ids={token_id}` — ale buď:
1. Token ID formát neodpovídá tomu co Gamma očekává
2. Markety jsou closed/archived a Gamma je nevrací
3. API call tiše selhává (error je logován na `debug` úrovni, ne `warning`)

**Co jsme opravili (deployováno, funguje):**
- `self._last_refresh = -999_999` → discovery po rebootu vrací 6,018 marketů (vs 0 dříve)
- `self._slack` → `self._slack_bot` → Slack notifikace nechashnou
- Price-convergence resolution → A/B trady se resolvují když cena > 0.95 nebo < 0.05
- end_date skip odstraněn pro Strategy C → METAR resolution neblokovaná
- Per-position try/except → jeden chybný trade neblokuje ostatní
- GEFS gc.collect() + sleep → prevence OOM na 4GB VPS

**Co NEFUNGUJE:**
- `fetch_by_token_id(token_id)` vrací None pro naše pozice
- Bez marketu nemůže resolution checker zjistit question (pro METAR) ani cenu (pro price convergence)
- Žádný trade se neresolvuje

### Co potřebuje debug

1. **Ověřit formát token_id** — jsou v DB uloženy jako long int stringy? Gamma API je očekává jako `clob_token_ids` parametr — funguje to?
   ```bash
   # Test na VPS:
   curl "https://gamma-api.polymarket.com/markets?clob_token_ids=TOKEN_ID_Z_DB&limit=1"
   ```

2. **Přidat warning log** do `fetch_by_token_id` místo `debug` — abychom viděli proč vrací None

3. **Alternativní přístup**: pro Strategy C nepoužívat `fetch_by_token_id` vůbec — question a token info je v `trade_details` JSONB v PostgreSQL

4. **Pro Strategy A/B**: můžeme resolvovat přímo z CLOB API ceny (bez Gamma) — `get_price(token_id)` vrací aktuální cenu

### Relevantní soubory
- `arbo/main_rdh.py` řádky 1417-1590 — resolution checker
- `arbo/connectors/market_discovery.py` řádek 831 — fetch_by_token_id
- `arbo/strategies/weather_resolution.py` — METAR resolution pro Strategy C
- `arbo/core/paper_engine.py` řádek 352 — resolve_market()

### Jak testovat
```bash
# SSH na hlavní VPS
ssh -i ~/.ssh/lightsail-london.pem arbo@18.135.109.36

# Logy
sudo journalctl -u arbo --since '10 min ago' | grep -i resolution

# DB: open trades
cd /opt/arbo && .venv/bin/python3 -c "... asyncio query paper_trades ..."

# Test Gamma API pro konkrétní token
curl "https://gamma-api.polymarket.com/markets?clob_token_ids=TOKEN_ID&limit=1"
```

---

## STAV SPORT DATA DOWNLOAD

### Download VPS (18.134.241.60) — 4vCPU, 16GB, $84/mo
- **SSH**: `ssh -i ~/.ssh/lightsail-london.pem arbo@18.134.241.60`
- **Pass 1 (10-min)**: HOTOVO — 150K marketů, 338M cen, 102 GB
- **Pass 2 (1-min game window)**: BĚŽÍ
  - Spread: 18,693/28,490 (66%) — skoro hotový
  - Moneyline: 11,876/86,690 (14%) — běží
  - Worker logy: `tail /opt/arbo/research_d/data/pass2_ml.log`
  - Progress: `wc -l /opt/arbo/research_d/data/pmd_pass2_progress_*.txt`
- **DB**: `research_d/data/sports_backtest.sqlite` (~158 GB)
- **Disk**: 152 GB volné

### Po dokončení Pass 2:
1. SCP SQLite zpět na hlavní VPS nebo lokál
2. Smazat download VPS (úspora $84/mo)
3. Stornovat PolymarketData.co Ultra ($360/mo)
4. Spustit backtest na plném datasetu → sweep → walk-forward → autoresearch

---

## STAV VPS STABILITY

### Hlavní VPS (arbo-london, 18.135.109.36) — 2vCPU, 4GB, $20/mo
- Padá kvůli GEFS memory spike → OOM kill
- **Opraveno**: gc.collect() + sleep po každých 5 GEFS members (`gefs_downloader.py`)
- **Opraveno**: discovery `_last_refresh = -999_999` → funguje po rebootu
- **Potřebuje monitoring**: jestli se VPS stabilizoval po opravách

### Download VPS (arbo-download, 18.134.241.60) — 4vCPU, 16GB, $84/mo
- Stabilní, běží sport Pass 2 download
- **Smazat po dokončení** — jen dočasná instance

---

## WEATHER DATA — KOMPLETNÍ

- **DB**: `research/data/weather_pmd.sqlite` (lokálně + VPS kopie)
- **10,039 marketů**, 7.42M cen, 19 měst (chybí jen DC)
- **10-min rozlišení**, kompletní historie
- **Doplňkové**: `research/data/price_history.sqlite` (20 měst, hourly, 189K cen, CLOB+Goldsky)

---

## DALŠÍ KROKY (PRIORITY)

1. **FIX: `fetch_by_token_id` resolution** — nejkritičtější, blokuje celý paper trading
2. **Dokončit Pass 2 sport download** — čeká se (~1-2 dny)
3. **Backtest na plném datasetu** — jakmile máme data + resolution funguje
4. **Parameter sweep + walk-forward** — Sprint D-1
5. **Autoresearch** — Sprint D-2

---

## MEMORY: Aktuální stav memory souborů

Viz `/Users/dnl.vlck/.claude/projects/-Users-dnl-vlck-Arbo/memory/`:
- `strategy_d.md` — kompletní stav Strategy D
- `paper_trading_ar0134.md` — Strategy C model parametry
- `MEMORY.md` — index všech memory souborů

## COMMITY (tato session)

```
761d467  feat: Strategy D spec + Sprint D-0 infrastructure
ac3ba90  feat: Goldsky subgraph sports trade downloader
f8fdd01  fix: limit Goldsky sports discovery pagination
581de72  docs: add Strategy D sprint plan to master TODO
0339898  feat: historical Pinnacle odds (EPL + NBA)
0b53153  feat: Sprint D-1 backtest engine — first results positive
ebac06b  feat: pmxt archive downloader
f06d512  docs: add phased validation approach
e136d1a  fix: PolymarketData client usage parsing
8668572  feat: add game-window mode + market-type filter
5938822  feat: parallel PolymarketData downloader
5e4952b  fix: replace shared JSON progress with append-only log
d3b293d  perf: remove day-by-day chunking — single paginated request
e8043f2  feat: download progress card on dashboard
5afe1d5  fix: improve download progress card
f9a1e74  feat: weather download progress + PMD weather downloader
fb122fe  fix: weather card 10-min label
2167620  fix: weather dashboard shows HOTOVO
5be3f34  fix: replace weather download progress with static summary
d16b093  fix: critical resolution bugs — all strategies
93ef31e  fix: per-position error handling + debug logging
3a753df  fix: discovery 0-markets after reboot + GEFS memory
252f794  feat: split download card into Pass 1 + Pass 2
126e09c  fix: download card shows Pass 2 ETA
de3284f  fix: dashboard recognizes Pass 2 phase
d89d7ad  feat: add all leagues to downloader
```
