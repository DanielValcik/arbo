# Variant Leaderboard Dashboard Card — Specification

> MANDATORY dashboard card for any strategy running ≥ 2 variants in parallel (champion-challenger, shadow, bandit-allocated).
> Without this card, variants become invisible failures. User must always see the pool.

**Status**: spec
**Date**: 2026-04-13
**Triggered by**: Framework §11.11 requirement (Rapid Mode dashboard visibility)

---

## 1. Card Placement

One leaderboard card **per strategy tab** on the dashboard (B3 tab, C tab, D tab, etc.). Position: below the main strategy overview card, above open positions / closed trades tables.

Example for B3 tab (merged 5m + 15m):
```
┌─────────────────────────────────────────────────┐
│ B3 5-min Overview │ B3 15m Overview │
├─────────────────────────────────────────────────┤
│ Polymarket Wallet (shared)                       │
├─────────────────────────────────────────────────┤
│ 🆕 Variant Leaderboard — B3 5-min                │  ← new card
├─────────────────────────────────────────────────┤
│ 🆕 Variant Leaderboard — B3 15-min               │  ← new card
├─────────────────────────────────────────────────┤
│ Kumulativni P&L (merged)                         │
│ Porovnani 5m vs 15m                              │
│ Model & Expectation vs Reality (×2)              │
│ Open Positions / Closed Trades / Watchdog        │
└─────────────────────────────────────────────────┘
```

**Rule**: leaderboard card appears ONLY if strategy has ≥ 2 active variants. Single-variant strategies skip this card (it would be empty/confusing).

---

## 2. Card Structure

### 2a. Header
```
Variant Leaderboard — <Strategy Name>  [active: 3/8]
Phase: shadow | incubate | live | scaled
Last promotion: <variant_id> at <date>  |  Last retirement: <variant_id> at <date>
```

- `active: 3/8` = currently running / max pool size
- Phase shows lifecycle stage (FSM from Framework §11.10)

### 2b. Ranking Table (primary content)

| Rank | Variant | Status | N | WR (CI) | DSR | Capital | Reward | Drift |
|:---:|---|---|---:|---:|---:|---:|---:|:---:|
| 🏆 | `ch_edge_tight` (champion) | live | 142 | 71.2% (67-75%) | 1.34 | $52 (70%) | 0.42 | 🟢 |
| 2 | `ch_sigma_06` | live | 138 | 68.5% (65-72%) | 1.05 | $15 (20%) | 0.35 | 🟢 |
| 3 | `ch_move_wide` | live | 140 | 66.8% (63-71%) | 0.87 | $8 (10%) | 0.28 | 🟡 |
| 4 | `ch_threshold_tight` | shadow | 85 | 62.4% (56-68%) | 0.42 | — | 0.19 | 🟢 |
| 5 | `ch_no_filter` | shadow | 92 | 58.2% (52-64%) | 0.15 | — | 0.12 | 🟢 |
| 💀 | `ch_aggressive` (retired 2026-04-10) | — | 67 | 48.2% | -0.3 | — | — | — |

**Column definitions:**
- **Rank**: 🏆 champion, 💀 retired, numeric for active challengers
- **Variant**: human-readable name (YAML config ID)
- **Status**: `shadow` (no capital), `incubate` ($5-25), `live` ($25-100), `scaled` ($100+), `retired`
- **N**: paired observations (same for all — one per signal across variants)
- **WR (CI)**: rolling 50-trade win rate + 95% block-bootstrap CI (Politis-Romano stationary)
- **DSR**: Deflated Sharpe Ratio. Highlight >0.95 green, 0.5-0.95 yellow, <0.5 red.
- **Capital**: current $ allocation (via MAB/Thompson Sampling if live)
- **Reward**: composite `0.4×dir_60s + 0.6×norm_pnl` trailing mean (§11.7)
- **Drift**: 🟢 OK, 🟡 warning (Page-Hinkley approaching threshold), 🔴 firing (paused)

### 2c. Capital Allocation Pie/Bar Chart (visual aid)

Small bar chart showing current capital split per variant. When MAB rebalances (daily), chart animates/updates.

### 2d. Promotion Pipeline Gauge

Linear progress bar showing the lifecycle of each variant:
```
shadow ═════════▶ incubate ─────▶ live ──────▶ scaled
           N=85       N=50         N=30       N=200
          ch_4        ch_1,2,3                champion
```

Shows at a glance: which variants are where, how much data each has, what's next.

### 2e. Recent Activity Feed (last 5 events)

```
2026-04-13 09:42 — ch_sigma_06 passed DSR>0.95, promoted shadow→incubate
2026-04-12 14:18 — ch_aggressive drift detector fired, paused
2026-04-12 06:55 — MAB rebalance: champion 70→52%, ch_sigma_06 15→20%
2026-04-11 22:03 — new challenger ch_no_filter enrolled (shadow)
2026-04-10 16:30 — ch_4 retired (DSR<0.3 after N=67)
```

Tied to actions from `STRATEGY_OPTIMIZATION_LEARNINGS.md`.

### 2f. Drill-Down Link

Clickable variant names expand to show:
- Full config (YAML)
- Equity curve vs champion (paired differences)
- DSR evolution over time
- Regime-conditional performance

---

## 3. Data Sources

### 3a. Backend API endpoints (new)

| Endpoint | Returns |
|---|---|
| `GET /api/variants?strategy=B3` | List of active + recent variants with all fields |
| `GET /api/variants/<variant_id>/history` | Per-variant trade history, for drill-down |
| `GET /api/variants/<variant_id>/equity` | Equity curve data for chart |
| `GET /api/bandit-state?strategy=B3` | Current MAB posterior means, allocations, last rebalance time |
| `GET /api/drift-status?strategy=B3` | Per-variant drift detector state |

### 3b. DB schema additions

`paper_trades` table needs new column:
```sql
ALTER TABLE paper_trades ADD COLUMN variant_id TEXT;
CREATE INDEX idx_paper_trades_variant ON paper_trades (strategy, variant_id, placed_at);
```

New tables:
```sql
CREATE TABLE variant_configs (
    variant_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    retired_at TIMESTAMPTZ,
    status TEXT NOT NULL,  -- shadow | incubate | live | scaled | retired
    parent_variant TEXT    -- lineage (for BO-derived configs)
);

CREATE TABLE bandit_state (
    strategy TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    alpha FLOAT NOT NULL,  -- Beta posterior α
    beta FLOAT NOT NULL,   -- Beta posterior β
    last_reward TIMESTAMPTZ,
    capital_pct FLOAT NOT NULL,
    PRIMARY KEY (strategy, variant_id)
);

CREATE TABLE drift_state (
    strategy TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    detector TEXT NOT NULL,   -- 'page_hinkley' | 'adwin'
    status TEXT NOT NULL,      -- 'ok' | 'warning' | 'firing'
    fire_count_24h INT DEFAULT 0,
    last_fire_at TIMESTAMPTZ,
    PRIMARY KEY (strategy, variant_id, detector)
);
```

---

## 4. Frontend Implementation

### 4a. Component location
```
arbo/dashboard/variant_leaderboard.py   # backend API
arbo/dashboard/templates/dashboard.html  # HTML partial + JS fetch
```

### 4b. Update cadence
- Auto-refresh every 30s (same as other cards)
- Manual refresh button for immediate update

### 4c. Alerting overlay

If any variant has:
- Drift firing → show red banner at top of card
- Status="live" but N<30 → show yellow "early exploration" badge
- DSR < 0 on ≥ 100 paired obs → show red "retirement candidate" flag

---

## 5. Interaction with Watchdog

Watchdog (in Rapid Mode) becomes the **Variant Orchestrator**:
- Schedules daily MAB rebalance (writes to `bandit_state` table)
- Runs drift detectors every 30 min, updates `drift_state`
- Auto-promotes / auto-retires variants per rules (logs to LEARNINGS.md)
- Escalates to CEO via Slack if:
  - DSR of champion drops below 0.50 for 3 consecutive days
  - New BO-proposed config exceeds champion DSR by ≥ τ (variant admission gate)
  - Drift detector fires on champion (not just challenger)

---

## 6. Implementation Phases

### Phase A (Week 1): Read-only leaderboard on existing single-variant strategies
- Show current strategy with `variant_id = "champion_v1"` synthetic
- DSR computed from existing trade history
- No MAB yet (all capital still goes to single champion)
- Validates the card layout and data pipeline

### Phase B (Week 2): Multi-variant shadow testing
- `ShadowOrchestrator` spawns 3-5 shadow challengers
- Leaderboard shows shadow rows (Capital column empty)
- User can inspect variants without financial risk

### Phase C (Week 3): Bandit-allocated live
- MABWiser integration, daily rebalance
- Leaderboard Capital column populated
- Pie chart animates on rebalance

### Phase D (Week 4): Drift + automatic lifecycle
- Page-Hinkley detectors per variant
- Auto-promote / auto-retire rules
- Full visual pipeline gauge

---

## 7. Acceptance Criteria

**Phase A complete when:**
- [ ] Leaderboard card renders on B3 tab
- [ ] Shows current champion with DSR, WR+CI, capital
- [ ] API endpoints return correct data
- [ ] Block-bootstrap CI displayed

**Phase B complete when:**
- [ ] 3+ shadow variants visible
- [ ] Paired observations tracked per variant
- [ ] Status lifecycle displayed

**Phase C complete when:**
- [ ] MAB rebalance visible in Activity Feed
- [ ] Capital column updates daily
- [ ] Pie chart animates

**Phase D complete when:**
- [ ] Drift status 🟢🟡🔴 per variant
- [ ] Auto-promotion logged in Activity Feed
- [ ] Alerts fire on DSR drops

---

## 8. Mockup (ASCII, for reference)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🏆 Variant Leaderboard — B3 5-min              [active: 3/8]  LIVE      │
│ Phase: live (champion scaled)                                            │
│ Last promotion: ch_sigma_06 (2026-04-13)  Last retire: ch_aggressive    │
├─────────────────────────────────────────────────────────────────────────┤
│ Rank │ Variant          │ Status  │ N   │ WR(CI)      │ DSR   │ Cap%  │
├──────┼──────────────────┼─────────┼─────┼─────────────┼───────┼───────┤
│  🏆  │ champion         │ live    │ 142 │ 71(67-75)   │ 1.34  │ 70%   │
│   2  │ ch_sigma_06      │ live    │ 138 │ 68(65-72)   │ 1.05  │ 20%   │
│   3  │ ch_move_wide     │ live    │ 140 │ 67(63-71)   │ 0.87  │ 10%   │
│   4  │ ch_threshold     │ shadow  │  85 │ 62(56-68)   │ 0.42  │ —     │
│   5  │ ch_no_filter     │ shadow  │  92 │ 58(52-64)   │ 0.15  │ —     │
│  💀  │ ch_aggressive    │ retired │  67 │ 48          │ -0.30 │ —     │
├─────────────────────────────────────────────────────────────────────────┤
│ Capital: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░ $75 total                                 │
│         champ 70% ch_sigma 20% ch_move 10%                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Pipeline:                                                                │
│  shadow ═════ ▶ incubate ──────▶ live ───▶ scaled                       │
│   ch_4,5      (none)           ch_1,2,3    (none)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Recent Activity:                                                         │
│  09:42 ch_sigma_06 promoted shadow→incubate (DSR>0.95)                  │
│  06:55 MAB rebalance: champion 70→52%                                   │
│  04-12 14:18 ch_aggressive drift fired, paused                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Non-Goals (out of scope)

- Full MLOps experiment tracker (we have this simplified — no MLflow needed)
- User ability to create variants via UI (CLI / YAML file for now)
- Real-time streaming (30s refresh is plenty)
- Historical replay of retired variants (archived in DB, not displayed)

---

## 10. References

- Framework §11.11: dashboard requirement
- RAPID_MODEL_DISCOVERY.md §8: shadow deployment A/B infrastructure
- Two Sigma "head-to-head" internal tooling inspiration
- AWS ML Lens MLREL-11 deployment patterns
