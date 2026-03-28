# Price Oracle Edge Research — B3 Binance Oracle Scalper

> **Date**: 2026-03-28
> **Status**: VALIDATED, implementation in progress

---

## 1. Klicove zjisteni

Polymarket 5-min Up/Down markety resolvuji pomoci **Chainlink Data Streams** (NE
Chainlink on-chain Data Feeds). Toto je zcela jiny produkt:

| Vlastnost | On-chain Data Feed | Data Streams (pouzivane) |
|-----------|-------------------|------------------------|
| Update frekvence | 60s heartbeat | Sub-sekundovy |
| Architektura | Push (on-chain) | Pull (off-chain, overeni on-demand) |
| Polygon contract | `0xc907E116...` | N/A (off-chain) |
| Pouziti | DeFi lending/DEX | Polymarket resolution |

**Polymarket vystavuje tento feed ZDARMA pres WebSocket (RTDS):**
```
wss://ws-live-data.polymarket.com
Topic: crypto_prices_chainlink
Symbol: btc/usd
```

## 2. Porovnani data zdroju

| Zdroj | Update frekvence | Latence | Poznamka |
|-------|-----------------|---------|----------|
| **Binance miniTicker** (aktualne) | 1/sec (throttled) | 5-12ms | Pomale, 1 update/sec |
| **Binance trade stream** | Per-trade (100s/sec) | 5-12ms | 100-1000x rychlejsi |
| **Coinbase WS matches** | Per-trade | 8-15ms | |
| **Chainlink Data Streams** | Sub-sec (~1/sec) | Sub-sec | Resolution zdroj! |
| **RTDS crypto_prices_chainlink** | ~1/sec | WS latence | Presna resolution cena |
| **Pyth Network** | 400ms (2.5/sec) | <400ms | Pouzivane pro equity, ne crypto |
| **Chainlink on-chain** | 60s | N/A | Prilis pomale, NENI resolution zdroj |

## 3. Dual-Feed architektura

```
Layer 1: SIGNAL (Binance trade stream)
  └── Per-trade updates → nejrychlejsi detekce BTC pohybu
  └── Pouziti: entry trigger, early movement detection

Layer 2: RESOLUTION (Chainlink RTDS)
  └── Sub-sec updates → presna resolution cena
  └── Pouziti: fair value computation, exit monitoring
  └── Toto je PRESNE co Polymarket pouziva na resoluci

           Binance trades          Chainlink RTDS
           (rychly signal)         (resolution cena)
                |                        |
                v                        v
           [B3 Scanner]            [FV Computation]
                |                        |
                +--------+-------+-------+
                         |
                    [Entry/Exit Logic]
```

**Proc dual-feed:**
- Binance objem $50B+/den → cena se pohne PRVNI na Binance
- Chainlink agreguje z vice burz s malym zpozdenim
- Delta mezi nimi = nase edge window
- Pouzitim Chainlink pro FV = presna shoda s resolution

## 4. Implementace

### RTDS Connector (`arbo/connectors/rtds_chainlink.py`)
- WebSocket klient pro `wss://ws-live-data.polymarket.com`
- Subscribe na `crypto_prices_chainlink` pro `btc/usd`
- Zadna autentizace, zdarma
- Poskytuje `get_price("btc/usd")` pro aktualni Chainlink cenu

### Binance upgrade
- Zmena z `btcusdt@miniTicker` na `btcusdt@trade`
- Per-trade granularita misto 1/sec throttle
- Zpetne kompatibilni (`get_price()` vraci posledni trade price)

### B3 integrace
- `btc_at_start`: Chainlink cena v minute 0 (Binance kline jako fallback)
- Signal trigger: Binance trade stream (nejrychlejsi)
- FV computation: Chainlink cena (presna resolution)
- Exit monitoring: Chainlink cena
- Comparison logging: oba prices na kazdych trade pro mereni edge

## 5. Ocekavany dopad

### Pred (jen Binance miniTicker)
- Signal 1x/sec throttled
- FV based on Binance ≠ resolution price
- btc_at_start z Binance kline (blizko, ale ne presne)

### Po (dual-feed)
- Signal per-trade (100x rychlejsi detekce)
- FV based on Chainlink = PRESNE resolution price
- btc_at_start z Chainlink (presna resolution cena v minute 0)
- Meritelny: log Binance vs Chainlink delta na kazdem tradu

### Rizika
- RTDS Chainlink feed ma obcasne gaps (7+ sec) — GitHub issues #31, #32
- Fallback na Binance kdyz RTDS neni dostupny
- Polymarket od 30.3.2026 rozsiruje dynamic taker fees na crypto (az 1.8%)
  PostOnly maker zustava 0% + rebate (nas pristup)

## 6. Zdroje

- [Polymarket RTDS docs](https://docs.polymarket.com/market-data/websocket/rtds)
- [Chainlink Data Streams](https://docs.chain.link/data-streams)
- [BTC/USD Data Stream](https://data.chain.link/streams/btc-usd-cexprice-streams)
- [Chainlink sponsored access for Polymarket](https://pm-ds-request.streams.chain.link/)
- [Polymarket x Chainlink partnership](https://www.prnewswire.com/news-releases/polymarket-partners-with-chainlink-to-enhance-accuracy-of-prediction-market-resolutions-302555123.html)
- [RTDS missing ticks issue #31](https://github.com/Polymarket/real-time-data-client/issues/31)
- [RTDS missing ticks issue #32](https://github.com/Polymarket/real-time-data-client/issues/32)

---

*Autoresearch: sweep_b3_scalper.py (1,900 trials, 89 days). Tato vylepseni
by mela zlepsit presnost FV modelu a snizit avg loss z $11.46 na ~$3.80
(backtest uroven).*
