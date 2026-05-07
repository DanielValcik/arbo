# Arbo — jak to celé funguje

> Tato stránka je **pro tebe jako uživatele**, ne pro vývojáře. Vysvětluje v normální řeči jak systém pracuje, co dělá sám a kdy potřebuje tvoje rozhodnutí. Aktualizuje se automaticky při každé změně systému.
>
> Poslední úprava: 2026-04-19.

## O co tady jde

Arbo je **automatický trading systém** na Polymarketu. Běží nonstop na VPS v Dublinu. Sám nachází příležitosti, sám kupuje, sám prodává a tobě posílá do Slacku to co je důležité vědět.

## Strategie

Momentálně **aktivně obchoduje jen B2** (Crypto Price Edge). Ostatní strategie jsou buď zastavené, nebo v paper módu.

### B2 — Crypto Price Edge

**Co dělá:** Kupuje Polymarket pozice na "bude BTC / ETH nad cenou X v den Y?" když jeho model řekne že skutečná pravděpodobnost je vyšší než cena na trhu. Model staví na volatilitě (log-normal, Black-Scholes style).

**Proč funguje:** Polymarket uživatelé často platí moc za longshoty a podceňují near-the-money opce. Když model najde mispricing přes 8% edge, Arbo koupí a drží dokud:
- edge nespadne pod 3% (exit)
- cena nevyletí o +30 centů (profit take)
- market neresolvne (cash-out)

**Aktuální parametry championa:**
- Min edge pro vstup: 8%
- Min cena: $0.05 (5¢) — pod tím je spread moc velký
- Max cena: $0.60 — nad tím už není edge
- Min expirace: 8 hodin (kratší markety mají moc theta)
- Stop edge: 3% (pod tím prodáváme)

### B3 — Binance Oracle Scalper

**Momentálně STOPPED.** 5-minutové BTC up/down markety. Měla 3-denní drawdown -$595 (15.-17. dubna), takže jsme ji 18.4. vypnuli. Existující pozice resolvují normálně, ale žádné nové. Až budeme mít čas, autoresearch najde nové parametry a zapneme zpět.

### B3_15M, C2, D

**Všechny paper only.** Neobchodujeme je živě, jen sbíráme data.

## Jak systém rozhoduje co dělat

### Filtry (quality gate)

Než systém něco koupí, signál musí projít třemi filtry:

1. **Model filter** — model musí říct "edge ≥ 8%"
2. **CLOB filter** — po započtení skutečného spreadu (cena kterou opravdu zaplatíš) musí edge pořád ≥ 8%. Tohle většinu signálů zabije — spread je často 2-5 centů a to sežere edge.
3. **Risk filter** — není překročen wallet limit na asset, není drift alert, není emergency shutdown.

Takže když v Slacku vidíš že je `qualified=15` ale `executed=0`, znamená to že model našel 15 signálů ale realita trhu (spread) je všechny vyřadila. **Není to chyba — je to disciplína.**

### Vstup (buy)

Když signál projde všemi filtry:
1. Systém kupuje **maker order** (cena pro nás lepší, ale možná nevyjde)
2. Pokud se order nefillne do 20-30s, zruší se a spustí **taker fallback** (zaplatíme trochu víc, ale určitě koupíme)
3. Dostaneš Slack zprávu: `⚡ B2 LIVE BUY — BTC above $78000, 5 shares @ 0.14`

### Držení (holding)

Každých ~60s systém přepočte edge pro každou otevřenou pozici:
- Bere **zmraženou sigmu z entry** (změna sigmy za 3 minuty neznamená že se edge opravdu zhoršil — to byla jedna z největších bug lekcí, viz LEARNINGS B2-17)
- Bere **aktuální Binance cenu** (reálný exchange price)
- Bere **zbývající čas do expirace**

Pokud edge spadne pod 3% → prodá (reason: `edge_lost`).
Pokud cena vyletí o +30 centů → prodá (reason: `profit_take`).
Pokud market resolvne → cash-out.

### Prodej (sell)

Stejně jako buy — maker pokus, taker fallback. Slack zpráva:
- `✅ B2 LIVE SELL — BTC above $78000 (profit_take) — Trade: $+6.41`
- `❌ B2 LIVE SELL — ETH above $2400 (edge_lost) — Trade: $-0.53`

## Kanárek — jak testujeme novou verzi strategie

**Problém:** Kdybychom jen našli "lepší" parametry na papíře a hned je nasadili, mohlo by to selhat v reálu (spread, latence, slippage, které paper nevidí). A nevěděli bychom.

**Řešení — kanárek (canary):** Když shadow data ukáží že je nová verze potenciálně lepší, systém:

1. **Shadow fáze:** Nová verze (challenger) běží vedle championa a hodnotí by-virtual signály. Žádný kapitál. Sbíráme statistiky.

2. **Inkubace (canary):** Pokud shadow ukáže silný edge, klikneš v Slacku "Approve" — NEBO to systém udělá sám při velmi jasných případech. Nová verze pak dostane **20% živých signálů**, champion dál 80%. Sbíráme data z reálných fillů.

3. **Rozhodnutí:** Po ≥15 reálných obchodech nové verze watchdog porovná oba. Pokud:
   - Nová je jasně lepší (pravděpodobnost > 70%): **auto-povýší na championa**, starý se archivuje
   - Nová je jasně horší (pravděpodobnost < 30%): **auto-vrátí do shadow**, nic neriskujeme
   - Nejasné: drží dál a sbírá víc dat

**Jen jeden kanárek najednou.** Kdybychom měli 2 současně, těžko bychom rozlišili který je lepší.

## Kdy systém rozhoduje sám a kdy se ptá

### Rozhoduje sám (žádná akce od tebe):

- **Vstup a výstup** z každého obchodu (vždycky)
- **Stop-loss** když edge spadne
- **Profit take** když cena vyletí
- **Cash-out** při resolution
- **Emergency shutdown** při překročení daily/weekly loss limitu
- **Auto-promote kanárka** když má po N≥15 live tradech extrémně silné důkazy (P(lepší) > 0.85, N ≥ 1000 shadow, Sharpe Δ > 0.2)
- **Auto-revert kanárka** pokud ztrácí
- **Auto-challenger** — když Gemini navrhne nový challenger na základě stagnace, automaticky ho vytvoří (status=shadow, nulový kapitál)

### Ptá se tě (Slack zpráva s tlačítkem):

- **Kanárek "na hraně"** — P(lepší) mezi 0.75 a 0.85. Dost silný na zamyšlení, ne tak silný aby ho systém udělal sám.
- **Změny v Tier 2 parametrech** (sigma_scale, position sizing) — vždy chceme tvoje potvrzení
- **Emergency kill switch** — nikdy se nevypíná autonomně víc než daily/weekly loss

### Jak vypadá rozhodnutí ve Slacku

```
🏆 PROMOTION CANDIDATE — B2
Challenger: ch_short_fuse  →  Champion: champion_v1
Sharpe: ch 1.05 vs cp 0.64
P(better): 0.92

[✅ Approve → Incubate 20%]  [❌ Reject]  [🔍 Details]
```

- **Approve** → kanárek dostane 20% živých obchodů, watchdog pak rozhodne sám
- **Reject** → challenger se retiruje, už se nebude navrhovat
- **Details** → další čísla kolem shadow vs live PnL

## Slack kanály a zprávy — co znamenají

| Zpráva | Co to znamená | Co s tím |
|---|---|---|
| ⚡ BUY | Nakoupila se pozice | Nic — info |
| ✅ SELL (profit_take) | Prodali jsme se ziskem | Nic — 🎉 |
| ✅ SELL (edge_lost) s plusem | Edge zmizel při dobré ceně | Nic — info |
| ❌ SELL (edge_lost) se ztrátou | Edge zmizel, pozice proděl | Nic — systém to hlídal |
| ✅ SELL (resolved_yes) | Market resolved YES, výhra | Nic — 🎉 |
| ❌ SELL (resolved_no) | Market resolved NO, plná ztráta investice | Nic — u longshots (prob <15%) očekávané |
| 🏆 PROMOTION CANDIDATE | Shadow ukazuje lepšího challengera | Approve / Reject / Details |
| 🤖 AUTO-APPROVED CANARY | Systém sám spustil kanárek | Jen info — jak to dopadne uvidíš |
| 🌱 incubate on B2 | Kanárek rozběhnut | Nic — čekání na live data |
| 🏆 CANARY ESCALATED | Kanárek se stal championem | 🎉 nic |
| 🔄 CANARY REVERTED | Kanárek selhal v reálu | Nic — systém ho vrátil do shadow |
| 📉/⚠️/🚨 B2 drift | Strategie se v aktuálním režimu chová jinak než historicky. Ikona podle severity. Text vysvětluje konkrétně co se stalo a co systém dělá. | Pokud je v textu "Co musíš udělat: Nic" → nic. Pokud navrhuje pauzu (severe), zvaž `B2_EXECUTION_MODE=stopped` |
| 🤖 B2 Auto-Challenger generated | Gemini navrhl nový challenger | Nic — běží v shadow |
| 🔴 Úlohy přestaly reagovat (anomálie) | Task dostal 10 restartů v řadě, systém to vzdal | `sudo systemctl restart arbo` na VPS. Dedup 24h — nevrací se ihned. **Nezahrnuje strategie, které jsi sám zastavil přes `.env`** (`{S}_EXECUTION_MODE=stopped` nebo `DISABLE_{S}=1`). |
| 🌅 Ranní briefing | Denní souhrn v češtině | Přečíst si, pokud chceš |

## Denní briefing

Každý den **v 07:00 UTC** (09:00 místního času) dostaneš jednu zprávu v `#slack-daily-brief` která shrnuje:

- Jak si champion vede (posledních 7 dní + 24h)
- Jestli testujeme nějakou novou verzi a jak daleko jsme
- Co systém hlídá (drift, fronta kandidátů)
- **Co musíš udělat** — často "nic, systém si to hlídá sám"

Účelem je zbavit tě rozhodovací únavy z ad-hoc notifikací — **jedna zpráva = kompletní obraz**.

## Co když se něco rozbije

- **B2 přestane obchodovat:** zkontroluj Slack za emergency shutdown alertem. Pokud nic, ping mě (Claude), projdeme spolu logy.
- **Divná zpráva ve Slacku:** je to pravděpodobně popsaný scénář výše. Kdybys nerozuměl, zeptej se.
- **Chceš něco změnit:** neuprav to sám v YAML souborech — projdeme spolu co by se změnilo. Systém je propojený, jedna změna může ovlivnit víc věcí.
- **Okamžité vypnutí strategie:** v `/opt/arbo/.env` přidej buď `{S}_EXECUTION_MODE=stopped` (B3, B3_15M, C2, D, D_UFC) nebo `DISABLE_{S}=1` (B2, C, C2) + `sudo systemctl restart arbo`. Existující pozice se doresolvují normálně, nové vstupy se nezakládají, **anomaly check ani health digest tě o stopnuté strategii nebudou spamovat**.

## Metriky kde najdeš co

- **Dashboard** — `http://52.17.173.192:8000` — živé pozice, PnL, health
- **Slack** — live notifikace + daily briefing
- **GitHub summaries** — `summaries/daily/YYYY-MM-DD.md` — každý den retrospektiva (auto-pushed)
- **LEARNINGS.md** — historie všech bugů a oprav

## Slovník

- **Edge** — rozdíl mezi naším modelem a cenou trhu. `edge = prob − price`. Chceme > 8% na vstup.
- **Shadow** — paper simulace bez kapitálu. Sbírá data na všech aktivních variantách.
- **Challenger** — varianta s odlišnými parametry, běží jen v shadow.
- **Champion** — varianta která aktuálně obchoduje živě.
- **Incubate (canary)** — varianta na živé validaci s malým kapitálem (20%). Mezistupeň challenger → champion.
- **Sigma** — odhadovaná volatilita (jak moc se cena hýbe). Model staví na sigmě.
- **P(better)** — pravděpodobnost (bootstrap) že challenger je opravdu lepší než champion. Min 75% pro přezkum, 85% pro auto-approve.
- **DSR delta** — Deflation-adjusted Sharpe ratio difference. Čím vyšší, tím je challenger robustněji lepší.
- **Drift (Page-Hinkley)** — statistický test zda se distribuce výsledků v čase mění. Pokud ano, varianta přestává fungovat.

## Historické milníky

- **2026-02-21** — Pivot z Matchbook na Polymarket (geo-block CZ)
- **2026-03-14** — Strategy C paper trading AR-0134
- **2026-03-18** — EMOS ensemble C1f
- **2026-03-25** — Strategy C2 (EMOS + Edge Exit Fusion) deployed
- **2026-04-11** — B2 autoresearch complete, dual-mode config
- **2026-04-16** — B2 dual-mode LIVE, $100 wallet
- **2026-04-17** — Sigma freeze fix (B2-17), canary promotion framework designed
- **2026-04-18** — B3 stopped po 3-day drawdown, daily retrospectives auto-run
- **2026-04-19** — Canary promotion live, auto-approve logika, tento knowledge base

Když se bude něco měnit, přidej si tenhle řádek dolů nebo aktualizuj předešlé sekce. Systém bez uptodate KB je systém který sám ztratíš.
