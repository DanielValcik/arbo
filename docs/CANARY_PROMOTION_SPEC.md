# Canary Promotion — specification

> Framework: Project PARALLEL — Phase 2C extension.
> Built 2026-04-19 after discovery that shadow stats alone cannot
> justify live promotion (shadow paper mode has spread-earning bias,
> stale prices, and no real fill semantics — see `LEARNINGS.md` G1).
>
> Status: **B2 only** at time of writing; pattern can be copied to
> other strategies.

## Why

Project PARALLEL's shadow evaluation answers: *"would this variant
qualify these signals and what's the theoretical PnL if they
resolved?"* That's a useful filter, but it's not a promotion
guarantee. Shadow can't observe:

- **Spread cost on entry/exit** — shadow fills at midpoint or inferred
  price, live pays the actual ask (for buys) or bid (for sells)
- **Slippage / partial fills** on thin orderbooks
- **Cancel→refill latency** when posting maker orders
- **Order-placement races** with other traders

A variant that beats the champion on shadow stats can still lose
money live. The canary stage exists to **validate on real fills
before committing full capital**.

## Lifecycle

```
  challenger        (shadow only, ≥ MIN_PAIRED_N ≥ 100 shadow signals)
       │
       │  Slack "Approve → Incubate 20%" click
       ▼
  incubate          (live: 20% of signals routed to this variant's gate,
                     80% continue to champion; both attribute trades
                     via trade_details.variant_id)
       │
       │  watchdog _incubate_cycle every 6h
       │
       ├── P(live better) ≥ 0.70 after N_live ≥ 15 each ──►  champion
       │                                                       (old retired,
       │                                                        see pool_manager.promote)
       │
       ├── P(live better) ≤ 0.30 after N_live ≥ 20 each ──►  challenger
       │                                                       (back to shadow only,
       │                                                        see pool_manager.revert_incubate_to_challenger)
       │
       └── otherwise: hold, accumulate
```

## Contracts

### `pool_manager.promote_to_incubate(variant_id, strategy, capital_pct=0.20, approved_by)`

- Pre: variant exists, status=`challenger`, no other variant is
  currently `incubate` (only one canary per strategy at a time).
- Post: variant YAML updated with `status: incubate`,
  `incubate_capital_pct`, `incubated_at`, `incubated_by`.
- Idempotent on YAML location (atomic write).

### `pool_manager.revert_incubate_to_challenger(variant_id, strategy, reason, decided_by)`

- Pre: variant exists, status=`incubate`.
- Post: status → `challenger`, `incubate_capital_pct` stripped,
  `reverted_at`/`reverted_reason` recorded.

### `pool_manager.promote(variant_id, strategy, approved_by)` — unchanged

Still does the atomic `challenger|incubate → champion` swap. Watchdog
calls this when canary escalates.

## Routing

`strategy_b2._get_live_routing()` returns `(champion, incubate_or_None,
incubate_pct)`. For each candidate signal, `_route_signal(sig, pct)`
hashes `(token_id, minute_bucket)` and returns a deterministic bool —
the *same* signal would always be routed the same way within a minute,
which means attribution is consistent across retries and restarts.

Each qualified signal passes through ONE variant's `check_signal_quality`
(not both). The variant that admitted the signal is recorded on the
resulting `paper_trades.trade_details.variant_id` so the watchdog can
later compute per-variant live PnL.

## Why 20% default

- Small enough to cap downside if the canary is worse (max 20% of
  B2's ~$100 wallet at risk)
- Large enough that with B2's ~3-5 trades/day, we reach N=15 paired
  trades in 2-3 weeks
- Adjustable via `capital_pct` arg — promotion engine may raise it to
  30-40% for very high-confidence candidates

## Statistical decision

Watchdog uses the same block-bootstrap as `PromotionEngine`, but
against LIVE `paper_trades.actual_pnl` series (not shadow
`would_pnl_per_share`). Block-bootstrap handles serial correlation in
the time-ordered PnL stream.

Thresholds (conservative, in `b2_watchdog._incubate_cycle`):
- `MIN_PAIRED_N_LIVE = 15` — need at least this many LIVE trades on
  each side before any decision
- `P_ESCALATE = 0.70` — fraction of bootstrap resamples that favor
  incubate
- `P_REVERT = 0.30` — symmetric threshold
- `MIN_N_BEFORE_REVERT = 20` — extra safety: only kill the canary
  after it has had a fair chance

If neither threshold is crossed, watchdog holds. Incubate stays.

## Failure modes

| Mode | Detection | Action |
|------|-----------|--------|
| Incubate steadily loses | `_incubate_cycle` sees p_better < 0.30 | auto-revert to challenger |
| Incubate mostly wins | p_better > 0.70 | auto-escalate to champion |
| Ambiguous | 0.30 < p < 0.70 | hold; accumulate; watchdog re-evaluates every 6h |
| No live data | N_live < 15 | skip; log `b2_incubate_insufficient_live_data` |
| Champion missing | `get_champion` returns None | log warning, skip |
| Pool manager write fails | exception in YAML write | revert not applied; logged |

## Operator controls

- **Slack "Reject" button** on promotion candidate → calls `veto()` → challenger → retired
- **Manual escalate/revert from watchdog** — kill the watchdog, edit
  YAML manually, restart
- **Emergency pause** — set `B2_EXECUTION_MODE=stopped` (see G-2
  pattern) — freezes both champion and incubate entries

## Per-variant live stats query

```sql
SELECT actual_pnl
FROM paper_trades
WHERE strategy = 'B2'
  AND status = 'sold'
  AND actual_pnl IS NOT NULL
  AND COALESCE(notes, '') NOT LIKE 'pre_reset%'
  AND trade_details ->> 'variant_id' = :v
ORDER BY resolved_at ASC NULLS LAST;
```

This query is the foundation of the canary decision. Make sure
`trade_details.variant_id` is actually written at trade-placement
time (see `strategy_b2.py` around line 620 — the
`signal_variant.get(token_id, "champion_v1")` lookup).

## Ports to other strategies

To enable canary on B3 / B3_15M / D:

1. Change strategy's live filter from hardcoded params to per-variant
   (same pattern as B2's `_get_live_routing` + `_route_signal`)
2. Make strategy write the actual variant_id to
   `trade_details['variant_id']` at `place_trade` time
3. Wire `_incubate_cycle` into that strategy's watchdog (copy from
   `b2_watchdog.py`)
4. Update Slack approve handler to call `promote_to_incubate` for
   that strategy (the existing handler in `slack_promotion.py` is
   generic, should just work)

The abstraction ceiling here is per-strategy. Each strategy's
`filter_signals` / quality gate pattern is slightly different
(B2 uses kwargs params, B3 uses class-level constants) so it's
not a 1-line port. Start with the closest existing pattern and
refactor minimally.
