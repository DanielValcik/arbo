"""
B3 Pessimistic Autoresearch — Gap offset stress test
=====================================================

Runs same autoresearch 3x with progressively worse fill conditions:
1. Realistic: actual gap distribution (avg fill ~$0.38)
2. Moderate:  gaps + $0.05 offset (avg fill ~$0.43)
3. Pessimistic: gaps + $0.10 offset (avg fill ~$0.48)

Parameters that are profitable across ALL THREE = truly robust.
Parameters that only work on Realistic = fragile (current problem).

Usage:
    python3 research/innovations/sweep_b3_pessimistic.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crypto_price_pmd.sqlite"
RESULTS_PATH = DATA_DIR / "experiments" / "b3_pessimistic_sweep.json"
LOG_PATH = Path(__file__).parent.parent / "b3_pessimistic_sweep_log.txt"

INITIAL_CAPITAL = 300.0
MIN_ORDER_SIZE = 5.0
MAX_BET_SIZE = 100.0
MAX_SHARES = 500
SIGMA_FLOOR = 0.00005
WINDOW_MIN = 5
WF_IS_PCT = 0.70

GEN0_TRIALS = 1200
GEN1_TRIALS = 800
GEN2_TRIALS = 400

_SQRT2 = math.sqrt(2.0)
_log_fh = None

_FILL_DIST = json.loads(Path("/tmp/b3_live_fill_distribution.json").read_text())
_REAL_GAPS = _FILL_DIST["gaps"]
_FILL_RATE = _FILL_DIST["fill_rate"]
_PARTIAL_RATE = _FILL_DIST["partial_rate"]

# THREE gap offset scenarios
SCENARIOS = [
    ("realistic", 0.00),    # Original gaps
    ("moderate", 0.05),     # +5 cents worse
    ("pessimistic", 0.10),  # +10 cents worse (matches live reality)
]


def log(msg=""):
    global _log_fh
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else ""
    print(line)
    if _log_fh is None:
        _log_fh = open(LOG_PATH, "w")
    _log_fh.write(line + "\n")
    _log_fh.flush()


def _norm_cdf(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))
def _maker_rebate(p):
    pp = min(p, 1.0 - p)
    return 0.20 * 0.10 * pp * pp


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT ts, close FROM binance_klines WHERE symbol='BTCUSDT' ORDER BY ts").fetchall()
    cl = conn.execute("SELECT ts, up_won FROM chainlink_resolutions").fetchall()
    conn.close()
    closes = [r[1] for r in rows]
    timestamps = [r[0] for r in rows]
    chainlink = {r[0]: bool(r[1]) for r in cl}
    lr = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i-1] > 0:
            lr.append(math.log(closes[i] / closes[i-1]))
        else: lr.append(0.0)
    windows = []
    i = 0
    while i + WINDOW_MIN < len(closes):
        if timestamps[i+1] - timestamps[i] <= 120: windows.append(i)
        i += WINDOW_MIN
    return closes, timestamps, lr, windows, chainlink


def compute_sigma(lr, idx, win):
    s = max(1, idx - win)
    r = lr[s:idx]
    if len(r) < 10: return SIGMA_FLOOR
    n = len(r); m = sum(r)/n
    return max(math.sqrt(sum((x-m)**2 for x in r)/(n-1)), SIGMA_FLOOR)


@dataclass
class Cfg:
    sigma_window: int = 720
    sigma_scale: float = 0.5
    entry_threshold: float = 0.10
    min_entry_min: int = 1
    max_entry_min: int = 4
    max_entry_price: float = 0.55
    min_entry_fv: float = 0.30
    position_pct: float = 0.03
    edge_scaling: float = 5.0
    reentry_cooldown: int = 0


def simulate(closes, ts, lr, windows, cl, cfg, gap_offset=0.0, seed=42):
    rng = random.Random(seed)
    capital = INITIAL_CAPITAL
    peak = capital; max_dd = 0
    pnls = []; daily = defaultdict(float); total_fill = 0

    for si in windows:
        if capital < MIN_ORDER_SIZE: break
        sigma = compute_sigma(lr, si, cfg.sigma_window)
        S_start = closes[si]
        if S_start <= 0: continue

        for t in range(cfg.min_entry_min, min(cfg.max_entry_min+1, WINDOW_MIN)):
            S_now = closes[si+t]
            if S_now <= 0: continue
            t_rem = WINDOW_MIN - t
            if t_rem <= 0: continue

            log_ratio = math.log(S_now/S_start)
            sqrt_t = math.sqrt(t_rem)
            sr = sigma * sqrt_t
            mu = _norm_cdf(log_ratio/sr) if sr > 1e-12 else 0.5
            mu = max(0.02, min(0.98, mu))
            sm = sigma * cfg.sigma_scale * sqrt_t
            su = _norm_cdf(log_ratio/sm) if sm > 1e-12 else 0.5
            sd = su - 0.50
            if abs(sd) < cfg.entry_threshold: continue

            d = 1 if sd > 0 else -1
            efv = mu if d == 1 else (1-mu)
            if efv < cfg.min_entry_fv: continue

            if rng.random() > _FILL_RATE: continue
            gap = rng.choice(_REAL_GAPS) + gap_offset  # APPLY OFFSET
            fp = max(0.03, min(0.97, efv + gap))
            if fp > cfg.max_entry_price: continue

            sr2 = 1.0
            if rng.random() < _PARTIAL_RATE:
                sr2 = max(0.2, min(0.9, 0.6 + rng.uniform(-0.15, 0.15)))

            edge = abs(sd)
            rp = min(cfg.position_pct, edge * cfg.edge_scaling)
            bet = min(capital * rp, MAX_BET_SIZE)
            if bet < MIN_ORDER_SIZE: continue
            sh = min(bet/fp, MAX_SHARES) * sr2
            if sh < 1: continue

            wts = (ts[si]//300)*300
            rup = cl[wts] if wts in cl else closes[si+WINDOW_MIN] >= S_start
            won = (d==1 and rup) or (d==-1 and not rup)
            xp = 1.0 if won else 0.0
            net = (xp - fp) * sh + _maker_rebate(fp) * sh

            capital += net
            pnls.append(net)
            total_fill += fp
            daily[datetime.utcfromtimestamp(ts[si]).strftime("%Y-%m-%d")] += net
            if capital > peak: peak = capital
            dd = (peak-capital)/peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            break

    n = len(pnls)
    if n < 30: return None
    wins = sum(1 for p in pnls if p > 0)
    tot = sum(pnls)
    dv = list(daily.values())
    sharpe = 0
    if len(dv) >= 5:
        m = sum(dv)/len(dv)
        v = sum((x-m)**2 for x in dv)/(len(dv)-1)
        s = math.sqrt(max(v, 1e-10))
        if s > 1e-8: sharpe = min((m/s)*math.sqrt(365), 100)

    return {
        "trades": n, "wins": wins, "wr": round(wins/n*100, 1),
        "pnl": round(tot, 2), "avg_pnl": round(tot/n, 2),
        "avg_fill": round(total_fill/n, 3),
        "dd": round(max_dd*100, 1), "sharpe": round(sharpe, 1),
    }


def score(r):
    if r is None or r["trades"] < 30 or r["pnl"] <= 0: return 0
    roi = r["pnl"]/INITIAL_CAPITAL*100
    tf = 1 + math.log10(max(r["trades"], 1))
    dp = max(0, 1 - (r["dd"]/100)**2)
    sf = min(max(r["sharpe"], 0), 15)/15
    return roi * tf * dp * sf


def rand_cfg():
    return Cfg(
        sigma_window=random.choice([240, 360, 720, 1440]),
        sigma_scale=round(random.uniform(0.1, 2.0), 3),
        entry_threshold=round(random.uniform(0.02, 0.35), 3),
        min_entry_min=random.choice([1, 2]),
        max_entry_min=random.choice([2, 3, 4]),
        max_entry_price=round(random.uniform(0.35, 0.65), 2),
        min_entry_fv=round(random.uniform(0.20, 0.50), 2),
        position_pct=round(random.uniform(0.02, 0.10), 3),
        edge_scaling=round(random.uniform(0.5, 10.0), 2),
        reentry_cooldown=random.randint(0, 3),
    )


def mutate(base, intensity=0.25):
    c = Cfg(**asdict(base))
    def jf(v, lo, hi):
        return round(max(lo, min(hi, v+(hi-lo)*intensity*random.gauss(0,1))), 3)
    c.sigma_window = random.choice([240, 360, 720, 1440])
    c.sigma_scale = jf(c.sigma_scale, 0.1, 2.0)
    c.entry_threshold = jf(c.entry_threshold, 0.02, 0.35)
    c.min_entry_min = random.choice([1, 2])
    c.max_entry_min = random.choice([2, 3, 4])
    c.max_entry_price = jf(c.max_entry_price, 0.35, 0.65)
    c.min_entry_fv = jf(c.min_entry_fv, 0.20, 0.50)
    c.position_pct = jf(c.position_pct, 0.02, 0.10)
    c.edge_scaling = jf(c.edge_scaling, 0.5, 10.0)
    c.reentry_cooldown = random.choice([0, 1, 2, 3])
    return c


def multi_scenario_score(closes, ts, lr, windows, cl, cfg):
    """Score = MINIMUM score across all 3 scenarios. Only robust configs win."""
    scores = []
    for name, offset in SCENARIOS:
        results = [simulate(closes, ts, lr, windows, cl, cfg, gap_offset=offset, seed=s) for s in [42, 43, 44]]
        valid = [r for r in results if r is not None]
        if not valid:
            return 0, {}
        avg = {k: sum(r[k] for r in valid)/len(valid) for k in ["trades","wr","pnl","avg_pnl","avg_fill","dd","sharpe"]}
        avg["trades"] = valid[0]["trades"]
        s = score(avg)
        scores.append((name, s, avg))

    if not scores or any(s[1] == 0 for s in scores):
        return 0, {}

    # MINIMUM score = robustness metric
    min_score = min(s[1] for s in scores)
    detail = {name: {"score": round(sc, 0), **avg} for name, sc, avg in scores}
    return min_score, detail


# Global data for multiprocessing workers
_G_CLOSES = None
_G_TS = None
_G_LR = None
_G_WINDOWS = None
_G_CL = None


def _init_worker(closes, ts, lr, windows, cl):
    global _G_CLOSES, _G_TS, _G_LR, _G_WINDOWS, _G_CL
    _G_CLOSES, _G_TS, _G_LR, _G_WINDOWS, _G_CL = closes, ts, lr, windows, cl


def _eval_cfg(cfg_dict):
    """Evaluate a single config across all scenarios. For multiprocessing."""
    cfg = Cfg(**cfg_dict)
    ms, detail = multi_scenario_score(_G_CLOSES, _G_TS, _G_LR, _G_WINDOWS, _G_CL, cfg)
    return cfg_dict, ms, detail


def _eval_cfg_oos(args):
    """Evaluate OOS for a single config + scenario. For multiprocessing."""
    cfg_dict, name, offset, seeds = args
    cfg = Cfg(**cfg_dict)
    split = int(len(_G_WINDOWS) * WF_IS_PCT)
    oos = _G_WINDOWS[split:]
    results = [simulate(_G_CLOSES, _G_TS, _G_LR, oos, _G_CL, cfg, gap_offset=offset, seed=s) for s in seeds]
    valid = [r for r in results if r]
    if not valid:
        return name, None
    avg = {k: sum(r[k] for r in valid)/len(valid) for k in ["trades","wr","pnl","avg_pnl","avg_fill","dd","sharpe"]}
    avg["trades"] = valid[0]["trades"]
    return name, avg


def main():
    import multiprocessing as mp

    log("B3 Pessimistic Autoresearch — PARALLEL (multiprocessing)")
    log("Score = MIN across realistic/moderate/pessimistic scenarios")
    log("=" * 60)

    closes, timestamps, lr, windows, cl = load_data()
    split = int(len(windows) * WF_IS_PCT)
    n_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
    log(f"Data: {len(windows)} windows, {len(cl)} chainlink")
    log(f"Workers: {n_workers}, Scenarios: {[(n, f'+${o:.2f}') for n, o in SCENARIOS]}")

    all_results = []
    best_score = 0
    best_cfg = None

    # Gen 0 — parallel
    log(f"\nGEN 0: RANDOM ({GEN0_TRIALS}) — {n_workers} workers")
    t0 = time.time()
    random.seed(42)
    gen0_cfgs = [asdict(rand_cfg()) for _ in range(GEN0_TRIALS)]

    with mp.Pool(n_workers, initializer=_init_worker, initargs=(closes, timestamps, lr, windows, cl)) as pool:
        for i, (cfg_dict, ms, detail) in enumerate(pool.imap_unordered(_eval_cfg, gen0_cfgs, chunksize=20)):
            all_results.append({"gen": 0, "config": cfg_dict, "min_score": ms, "scenarios": detail})
            if ms > best_score: best_score = ms; best_cfg = Cfg(**cfg_dict)
            if (i+1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                log(f"  [{i+1}/{GEN0_TRIALS}] best={best_score:.0f} ({rate:.1f}/s)")

    log(f"Gen 0: {time.time()-t0:.0f}s, best={best_score:.0f}")

    all_results.sort(key=lambda x: x["min_score"], reverse=True)
    top15 = [Cfg(**r["config"]) for r in all_results[:15]]
    for i, r in enumerate(all_results[:3]):
        c = r["config"]
        log(f"  {i+1}. min_score={r['min_score']:.0f} maxP={c['max_entry_price']} σs={c['sigma_scale']} thr={c['entry_threshold']}")
        for name, data in r["scenarios"].items():
            log(f"     {name}: WR={data['wr']}% fill=${data['avg_fill']} PnL=${data['pnl']:.0f} DD={data['dd']}%")

    # Gen 1 — parallel
    log(f"\nGEN 1: MUTATIONS ({GEN1_TRIALS}) — {n_workers} workers")
    t1 = time.time()
    random.seed(100)
    gen1_cfgs = [asdict(mutate(random.choice(top15))) for _ in range(GEN1_TRIALS)]
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(closes, timestamps, lr, windows, cl)) as pool:
        for i, (cfg_dict, ms, detail) in enumerate(pool.imap_unordered(_eval_cfg, gen1_cfgs, chunksize=20)):
            all_results.append({"gen": 1, "config": cfg_dict, "min_score": ms, "scenarios": detail})
            if ms > best_score: best_score = ms; best_cfg = Cfg(**cfg_dict)
            if (i+1) % 200 == 0: log(f"  [{i+1}/{GEN1_TRIALS}] best={best_score:.0f}")
    log(f"Gen 1: {time.time()-t1:.0f}s, best={best_score:.0f}")

    all_results.sort(key=lambda x: x["min_score"], reverse=True)
    top5 = [Cfg(**r["config"]) for r in all_results[:5]]

    # Gen 2 — parallel
    log(f"\nGEN 2: FINE-TUNE ({GEN2_TRIALS}) — {n_workers} workers")
    t2 = time.time()
    random.seed(200)
    gen2_cfgs = [asdict(mutate(random.choice(top5), intensity=0.15)) for _ in range(GEN2_TRIALS)]
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(closes, timestamps, lr, windows, cl)) as pool:
        for i, (cfg_dict, ms, detail) in enumerate(pool.imap_unordered(_eval_cfg, gen2_cfgs, chunksize=10)):
            all_results.append({"gen": 2, "config": cfg_dict, "min_score": ms, "scenarios": detail})
            if ms > best_score: best_score = ms; best_cfg = Cfg(**cfg_dict)
    log(f"Gen 2: {time.time()-t2:.0f}s, best={best_score:.0f}")

    # Walk-forward — parallel per scenario
    log(f"\n{'='*60}\nWALK-FORWARD (top 5, OOS, all scenarios)\n{'='*60}")
    all_results.sort(key=lambda x: x["min_score"], reverse=True)
    wf = []

    for i, r in enumerate(all_results[:5]):
        cfg = Cfg(**r["config"])
        oos_windows = windows[split:]
        oos_results = {}
        for name, offset in SCENARIOS:
            res = [simulate(closes, timestamps, lr, oos_windows, cl, cfg, gap_offset=offset, seed=s) for s in [42,43,44]]
            valid = [x for x in res if x]
            if valid:
                avg = {k: sum(x[k] for x in valid)/len(valid) for k in ["trades","wr","pnl","avg_pnl","avg_fill","dd","sharpe"]}
                avg["trades"] = valid[0]["trades"]
                oos_results[name] = avg

        c = r["config"]
        log(f"\n  #{i+1} maxP={c['max_entry_price']} σs={c['sigma_scale']} thr={c['entry_threshold']} minFV={c['min_entry_fv']}")
        for name, data in oos_results.items():
            margin = data["wr"] - data["avg_fill"] * 100
            log(f"    {name:>12}: n={data['trades']} WR={data['wr']}% fill=${data['avg_fill']} "
                f"PnL=${data['pnl']:.0f} $/trade=${data['avg_pnl']:.2f} DD={data['dd']}% margin={margin:.0f}pp")

        wf.append({"rank": i+1, "config": c, "oos": oos_results})

    # Best
    best_wf = wf[0] if wf else None
    if best_wf:
        c = best_wf["config"]
        pess = best_wf["oos"].get("pessimistic", {})
        log(f"\n{'='*60}")
        log(f"BEST ROBUST CONFIG:")
        for k, v in c.items(): log(f"  {k}: {v}")
        if pess:
            log(f"\n  PESSIMISTIC OOS: n={pess.get('trades',0)} WR={pess.get('wr',0)}% "
                f"fill=${pess.get('avg_fill',0)} PnL=${pess.get('pnl',0):.0f} "
                f"margin={pess.get('wr',0) - pess.get('avg_fill',0)*100:.0f}pp")

    output = {"meta": {"sweep": "b3_pessimistic", "timestamp": datetime.now().isoformat(),
        "scenarios": SCENARIOS, "trials": len(all_results)},
        "walk_forward": wf, "all_trials": all_results[:30]}
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    log(f"\nResults: {RESULTS_PATH}")
    log(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    random.seed(42)
    main()
