# Kaito API Research Report (RDH-312)

**Datum:** 25. února 2026
**Od:** CTO, Arbo
**Pro:** CEO
**Deadline:** 18. dubna 2026 (delivered early)
**Status:** RESEARCH COMPLETE

---

## Executive Summary

Kaito API existuje jako enterprise produkt, ale **NENÍ dostupné pro self-serve single-user systém jako Arbo.** Custom pricing, no public docs, sales-gated access. Yaps (free tier) was sunset v lednu 2026 po X/Twitter API ban. Doporucuji stub-first approach s LLM fallback.

---

## 1. Existuje verejne Kaito API?

**Ano, ale je rozdeleno na 2 produkty:**

### A. Kaito API (Enterprise) — Mindshare & Sentiment
- URL: `https://www.kaito.ai/kaito-api`
- Provides: mindshare scores, sentiment, narrative tracking, 2000+ tokens
- Sources: X, TikTok, Instagram, YouTube, governance forums, news, podcasts
- **Access: CUSTOM PRICING ONLY. No self-serve signup. Must contact sales.**
- No public REST/GraphQL docs, no OpenAPI spec, no PyPI SDK
- Rate limit: 100 calls / 5 min (default)
- Target: funds, exchanges, project teams

### B. Yaps Open Protocol — DEAD
- Was free, public API for social attention scores
- Docs existed at `docs.kaito.ai/kaito-yaps-tokenized-attention`
- **SUNSET January 2026** — X/Twitter revoked API access for InfoFi apps
- 157K community banned from X, KAITO token -17%

## 2. Documentation

| Resource | URL | Status |
|----------|-----|--------|
| Kaito docs | docs.kaito.ai | Active (no API reference) |
| API product page | kaito.ai/kaito-api | Marketing only |
| Yaps docs | docs.kaito.ai/.../yaps-open-protocol | Dead |
| Pricing | kaito.ai/pricing | No public API prices |
| Kaito Pro | pro.kaito.ai/portal | Web dashboard |

**KRITICKE: Zadna verejna technicka API dokumentace (endpoints, auth, schemas).**

## 3. Pricing

- **Kaito Pro (web dashboard):** ~$833/month
- **Kaito API (programmatic):** Custom pricing, must contact sales
- **Odhad:** $500-$1000+/month minimum
- **Arbo budget impact:** SIGNIFICANT — presahuje rozumny budget pro single-user system

## 4. Pristupove podminky

- **Not open.** No self-serve API key generation.
- Sales-gated enterprise product.
- Must contact Kaito sales team.

## 5. Dostupna data

| Data | Popis |
|------|-------|
| Mindshare scores | % share of crypto conversation per token/topic |
| Sentiment | Positive/negative classification |
| Narrative tracking | Trending topics, narrative shifts |
| Coverage | 2000+ tokens, multi-platform |
| Historical | Backtesting available (Kaito Pro) |

## 6. Polymarket-specificky vyvoj (DULEZITE)

**Polymarket x Kaito "Attention Markets" Partnership (February 2026):**
- Polymarket a Kaito spustili "attention markets" — prediction markets kde outcome resolves na zaklade Kaito mindshare dat
- 2 pilot markets (Feb 2026): Polymarket mindshare + crypto Twitter mindshare ($1.3M volume)
- Planovano: desitky novych markets od brezna 2026, stovky do konce roku
- Kaito pouziva ZK proofs (Brevis) pro verifikovatelnou resolution

**Co to znamena pro Arbo:**
- Kaito data JE resolution source pro rostouci kategorii Polymarket marketu
- I BEZ Kaito API muzeme tradovat attention markets na CLOB — nase vlastni sentiment analyza (Gemini Flash)
- Attention markets jako sentiment signal pro jine trady

## 7. Riziko: X API Revocation (Leden 2026)

- X zakazal InfoFi apps (Kaito, Yaps) kvuli AI spam (7.75M bot postu/den)
- Kaito ztratil X API access — X byl primarni data source
- Kaito tvrdí, ze Kaito Pro a Kaito API NEJSOU ovlivneny
- **ALE:** kvalita dat bez X/Twitter je otazka — X byl dominantni crypto conversation platform

## 8. Doporuceni pro Arbo

**Kaito API NENI viable pro Arbo v soucasnem stavu.**

Duvody:
1. No self-serve access (enterprise sales)
2. Cenove neunosne ($500-1000+/mo vs Arbo budget)
3. Zadna verejna technicka dokumentace
4. X API revocation risk (degradovana kvalita dat)
5. Single-user system vs enterprise product

**Alternativni pristupy:**
1. **Trade attention markets primo** — nase vlastni sentiment analyza (Gemini Flash + free social APIs)
2. **Monitor Kaito public data** — web portal scraping jako signal
3. **Stub-first architecture** — hotova (RDH-301), swap na live <= 4h kdyz se situace zmeni
4. **Polymarket attention markets jako sentiment signal** — ceny na attention markets indikuji market sentiment

**CEO Decision Required:**
- [ ] Akceptovat stub + LLM fallback pro Strategy B
- [ ] Kontaktovat Kaito sales (zjistit real pricing)
- [ ] Realokovat Strategy B budget jinam
- [ ] Jina moznost: ___

---

*CTO, Arbo*
*25. unora 2026*
