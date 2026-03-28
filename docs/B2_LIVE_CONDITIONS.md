# B2 Live Conditions — Jak Crypto Price Markets Fungují na Polymarketu

> Kritická dokumentace pro paper trading realism.
> Aktualizováno: 2026-03-28 na základě live pozorování.

## 1. Typy Crypto Price Markets

Na Polymarketu existují 3 typy crypto price markets:

### A) Denní "Above" (Hourly sessions)
```
Příklad: "Bitcoin above 67,400 on March 28, 11AM ET?"
Slug:    bitcoin-above-on-march-28-2026-11am-et
```

- **Životnost**: ~2-4 hodiny (vytvoří se krátce před resolution)
- **Resolution**: Binance BTC/USDT 1h candle CLOSE at specified time
- **Buckets**: ~10 strike thresholds (spacing ~$400-2000)
- **Fees**: Crypto fees (taker: price*(1-price)*0.25, maker: 0%)
- **NegRisk**: false (každý bucket je nezávislý binary market)
- **Likvidita**:
  - ATM (strike ±3% od current): bid/ask spread 1-5c, depth $50-500
  - ITM/OTM (>5% away): tenký book, spread 50%+, depth <$10
- **Sessions**: ~2AM, 8AM, 11AM, 12PM, 2PM ET (UTC: 6, 12, 15, 16, 18)
- **Kdy existují**: Objevují se ~2h před resolution, zmizí po close
- **Kde najít**: Gamma `/events?slug=bitcoin-above-on-{date}-{time}`
  - NEJSOU v `/markets` endpoint
  - NEJSOU pod tag `crypto-prices` ani `bitcoin`

### B) Týdenní "Price of"
```
Příklad: "Will the price of Bitcoin be above $70,000?"
```

- **Životnost**: 3-7 dní
- **Resolution**: Na konci určeného data (end_date)
- **CLOB**: Většinou PRÁZDNÝ book (bid=0.01, ask=0.99, spread=98%)
- **Gamma cena**: Ukazuje "férovou" cenu (0.05-0.50) ale NENÍ obchodovatelná
- **Likvidita**: Téměř nulová — nelze realisticky obchodovat
- **⚠️ PROBLÉM**: Tohle B2 teď obchoduje a je to špatně

### C) Monthly "Hit"
```
Příklad: "What price will Bitcoin hit in March?"
```

- **Fees**: 0% (oba strany)
- **Likvidita**: Vysoká ($81M volume)
- **Model**: Barrier option (first passage time)

## 2. CLOB Orderbook Realita

### Pozorování (28. března 2026, různé časy)

**Denní "above" ATM market (8AM ET session, strike ~$66K, BTC=$66.5K):**
```
bid=0.52   ask=0.56   spread=0.04 (7%)
bid_depth=$120   ask_depth=$85
→ OBCHODOVATELNÝ — reálné ceny, rozumný spread
```

**Denní "above" OTM market (strike $74K, BTC=$66.5K):**
```
bid=0.01   ask=0.99   spread=0.98 (98%)
bid_depth=$2   ask_depth=$0
→ NEOBCHODOVATELNÝ — žádná likvidita
```

**Týdenní "price of" market (strike $70K, end March 30):**
```
bid=0.01   ask=0.99   spread=0.98 (98%)
→ NEOBCHODOVATELNÝ — book je prázdný celý den
```

### Klíčové Pravidlo
```
Spread < 30%  → obchodovatelný (denní ATM)
Spread 30-50% → hraniční (denní near-money)
Spread > 50%  → neobchodovatelný (OTM, týdenní)
```

## 3. Gamma Price vs CLOB Price

**KRITICKÁ DISKREPANCE:**

| Zdroj | Co vrací | Realita |
|-------|---------|---------|
| Gamma API `outcomePrices` | "Férová" cena (e.g. 0.50) | Syntetická, neobchodovatelná |
| CLOB `/book` bid/ask | Reálné objednávky | Na tenkých bookech 0.01/0.99 |
| CLOB midpoint | Průměr bid/ask | Na tenkých bookech 0.50 (= Gamma) |

**Problém**: Gamma cena a CLOB midpoint na prázdných bookech jsou oba ~0.50
ale to NEZNAMENÁ že se dá za tu cenu koupit nebo prodat.

**Pravidlo pro B2:**
- NIKDY nekupovat za Gamma price
- NIKDY nekupovat na trhu kde spread > 30%
- Entry price = CLOB ask (co reálně zaplatíme)
- Exit price = CLOB bid (co reálně dostaneme)

## 4. Timing — Kdy Existují Obchodovatelné Markets

```
UTC     ET      Aktivita
─────────────────────────────────────────
05:00   1:00AM  Žádné denní markets
06:00   2:00AM  2AM session: markets se vytvářejí
07:00   3:00AM  2AM session: resolution → markets zmizí
...
10:00   6:00AM  8AM session: markets se vytvářejí
12:00   8:00AM  8AM session: resolution
...
13:00   9:00AM  11AM session: markets se vytvářejí
15:00   11:00AM 11AM session: resolution
16:00   12:00PM 12PM session: resolution
18:00   2:00PM  2PM session: resolution
```

**Trading window**: ~2h před resolution = doba kdy markets existují a mají likviditu.
**Mimo window**: žádné denní markets → B2 nemá co obchodovat (správné chování).

## 5. Důsledky pro Paper Trading

### Co autoresearch předpokládal (ŠPATNĚ):
- Cena z PMD = reálná fill price
- Spread = implicitně malý
- Všechny markets jsou likvidní
- 902 trades za 87 dní

### Co live ukazuje (REALITA):
- CLOB spread na týdenních = 98% (neobchodovatelné)
- Denní ATM spread = 5-15% (obchodovatelné v 2h window)
- Denní OTM spread = 50%+ (neobchodovatelné)
- Skutečný trading window = ~2h × 4-5 sessions/den
- Realistický počet trades = 3-5 per session × ATM only

### Co musí paper trading dělat:
1. **Kupovat JEN za CLOB ask** (ne Gamma, ne midpoint)
2. **Reject markets se spread > 30%**
3. **Prodávat JEN za CLOB bid** (ne Gamma)
4. **Current price = CLOB midpoint** (pokud spread < 30%, jinak "--")
5. **Obchodovat jen denní "above"** sessions (ne týdenní "price of")
6. **Filtrovat na ATM** (strike ±5% od current BTC price)

## 6. Jak Poznat Typ Marketu

| Pattern v question | Typ | Obchodovat? |
|-------------------|-----|-------------|
| "Bitcoin above X on DATE, TIME ET?" | Denní above | ✅ Pokud ATM + spread < 30% |
| "Will the price of Bitcoin be above X?" | Týdenní | ❌ Tenký book |
| "What price will Bitcoin hit?" | Monthly hit | ⚠️ Jiný model (barrier) |
| "Bitcoin Up or Down" | 5/15min | ⚠️ B3 strategie |
| "Will Bitcoin dip to X?" | Monthly dip | ❌ Jiný market type |

### Rozlišení v kódu:
```python
# Denní above (OBCHODOVAT):
"Bitcoin above 67,400 on March 28, 11AM ET?"  # má datum + čas
"Bitcoin above 65,000 on March 28, 8AM ET?"

# Týdenní (NEOBCHODOVAT):
"Will the price of Bitcoin be above $70,000?"  # nemá specifický čas
"Will the price of Ethereum be above $2,100?"
```

**Klíč**: denní markets mají v question **konkrétní čas** (8AM ET, 11AM ET).
Týdenní mají jen "above $X" bez času.

## 7. Entry/Exit Fill Price Realismus

### Entry (BUY):
```
Paper by měl: fill_price = CLOB ask + slippage(0.5%)
Reality:      fill_price = CLOB ask (maker PostOnly = exact price)
Slippage:     ~0 pro maker, ~1-2c pro taker na ATM
```

### Exit (SELL):
```
Paper by měl: fill_price = CLOB bid - slippage(0.5%)
Reality:      fill_price = CLOB bid (taker = instant)
Fee:          price*(1-price)*0.25 na exit (taker fee)
```

### Resolution (HOLD):
```
Paper by měl: fill_price = $1.00 (won) nebo $0.00 (lost)
Reality:      Shares settle automaticky, 0% fee
Toto je nejčistší — žádný spread, žádné fees
```

## 8. TODO pro Opravy

1. [ ] Scanner: filtrovat jen markety s konkrétním časem v question (denní)
2. [ ] Entry: kupovat POUZE za CLOB ask, reject pokud ask neexistuje
3. [ ] Current price: zobrazovat "--" místo 0.5000 na prázdných bookech
4. [ ] Exit monitor: používat CLOB bid, skip pokud bid < 0.02
5. [ ] ATM filtr: jen strike ±5% od Binance ceny
6. [ ] Autoresearch: přidat spread simulaci do backtestuexperiment
7. [ ] Dashboard: unrealized P&L only když current_price je reálný

---

*Tato dokumentace vychází z live pozorování 27-28. března 2026.
Aktualizovat po dalším live research v trading windows.*
