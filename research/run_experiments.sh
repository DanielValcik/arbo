#!/bin/bash
# Autonomous experiment runner for Strategy C Weather optimization
# Runs experiments overnight, logging results to results.tsv
# Kill with: kill $(cat research/experiment.pid)

cd /Users/dnl.vlck/Arbo
echo $$ > research/experiment.pid

BEST_SCORE=24.090813
LOG_FILE="research/experiment_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_experiment() {
    local DESC="$1"
    log "=== EXPERIMENT: $DESC ==="

    # Commit the change
    git add research/strategy_experiment.py
    git commit -m "experiment: $DESC" --no-gpg-sign -q 2>/dev/null
    local COMMIT=$(git rev-parse --short HEAD)

    # Run backtest (no timeout on macOS — backtests take <1s)
    python3 research/backtest_harness.py > research/run.log 2>&1
    local EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        log "CRASH (exit=$EXIT_CODE)"
        local ERROR=$(tail -5 research/run.log)
        log "Error: $ERROR"
        echo -e "${COMMIT}\t0.000000\t0.00\t0.00\t0.00\t0.0\t0\tcrash\t${DESC}" >> research/results.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi

    # Extract metrics
    local SCORE=$(grep "^composite_score:" research/run.log | awk '{print $2}')
    local SHARPE=$(grep "^avg_sharpe:" research/run.log | awk '{print $2}')
    local PNL=$(grep "^avg_pnl_pct:" research/run.log | awk '{print $2}')
    local DD=$(grep "^max_drawdown_pct:" research/run.log | awk '{print $2}')
    local WR=$(grep "^avg_win_rate:" research/run.log | awk '{print $2}')
    local NT=$(grep "^num_trades:" research/run.log | awk '{print $2}')

    if [ -z "$SCORE" ]; then
        log "CRASH: no score in output"
        echo -e "${COMMIT}\t0.000000\t0.00\t0.00\t0.00\t0.0\t0\tcrash\t${DESC}" >> research/results.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi

    log "Score: $SCORE (best: $BEST_SCORE) | Sharpe: $SHARPE | Trades: $NT | DD: $DD | WR: $WR"

    # Compare (using bc for float comparison)
    local BETTER=$(echo "$SCORE > $BEST_SCORE" | bc -l 2>/dev/null)

    if [ "$BETTER" = "1" ]; then
        log "KEEP - improved from $BEST_SCORE to $SCORE"
        echo -e "${COMMIT}\t${SCORE}\t${SHARPE}\t${PNL}\t${DD}\t${WR}\t${NT}\tkeep\t${DESC}" >> research/results.tsv
        BEST_SCORE=$SCORE
        return 0
    else
        log "DISCARD - $SCORE <= $BEST_SCORE"
        echo -e "${COMMIT}\t${SCORE}\t${SHARPE}\t${PNL}\t${DD}\t${WR}\t${NT}\tdiscard\t${DESC}" >> research/results.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi
}

# Save original file for restoring after discards
cp research/strategy_experiment.py research/strategy_experiment.py.baseline

log "Starting autonomous experiment loop. Best score: $BEST_SCORE"
log "PID: $$"

# ============================================================================
# EXPERIMENT 1: Lower MIN_EDGE to 0.08
# ============================================================================
cat > /tmp/patch_exp.py << 'PYEOF'
import re, sys
f = sys.argv[1]
with open(f) as fh: c = fh.read()
c = c.replace('MIN_EDGE = 0.10', 'MIN_EDGE = 0.08')
with open(f, 'w') as fh: fh.write(c)
PYEOF
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 /tmp/patch_exp.py research/strategy_experiment.py
run_experiment "MIN_EDGE 0.10->0.08"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 2: Lower MIN_EDGE to 0.06
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_EDGE = 0.10', 'MIN_EDGE = 0.08').replace('MIN_EDGE = 0.08', 'MIN_EDGE = 0.06') if 'MIN_EDGE = 0.08' in c else c.replace('MIN_EDGE = 0.10', 'MIN_EDGE = 0.06')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.06"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 3: Increase KELLY_FRACTION to 0.30
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('KELLY_FRACTION = 0.25', 'KELLY_FRACTION = 0.30')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION 0.25->0.30"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 4: Increase KELLY_FRACTION to 0.35
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('KELLY_FRACTION = 0.25', 'KELLY_FRACTION = 0.35').replace('KELLY_FRACTION = 0.30', 'KELLY_FRACTION = 0.35')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION->0.35"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 5: Remove CONVICTION_RATIO (set to 0)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CONVICTION_RATIO = 1.5', 'CONVICTION_RATIO = 0.0')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CONVICTION_RATIO 1.5->0 (disabled)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 6: Lower CONVICTION_RATIO to 1.2
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CONVICTION_RATIO = 1.5', 'CONVICTION_RATIO = 1.2').replace('CONVICTION_RATIO = 0.0', 'CONVICTION_RATIO = 1.2')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CONVICTION_RATIO->1.2"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 7: MAX_PRICE 0.85->0.90
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MAX_PRICE = 0.85', 'MAX_PRICE = 0.90')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MAX_PRICE 0.85->0.90"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 8: Switch to normal distribution
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('DISTRIBUTION = \"student_t\"', 'DISTRIBUTION = \"normal\"')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DISTRIBUTION student_t->normal"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 9: Days out [0, 1, 2] (add day 0, remove day 3)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('DAYS_OUT_TO_TRADE = [1, 2, 3]', 'DAYS_OUT_TO_TRADE = [0, 1, 2]')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT [1,2,3]->[0,1,2]"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 10: Days out [0, 1] only
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('DAYS_OUT_TO_TRADE = [1, 2, 3]', 'DAYS_OUT_TO_TRADE = [0, 1]').replace('DAYS_OUT_TO_TRADE = [0, 1, 2]', 'DAYS_OUT_TO_TRADE = [0, 1]')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT->[0,1] only"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 11: Days out [1, 2] only
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('DAYS_OUT_TO_TRADE = [1, 2, 3]', 'DAYS_OUT_TO_TRADE = [1, 2]').replace('DAYS_OUT_TO_TRADE = [0, 1, 2]', 'DAYS_OUT_TO_TRADE = [1, 2]').replace('DAYS_OUT_TO_TRADE = [0, 1]', 'DAYS_OUT_TO_TRADE = [1, 2]')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT->[1,2] only"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 12: Tighter sigmas for NOAA cities (nyc, chicago)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"nyc\":     {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
    \"chicago\": {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
}''')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CITY_SIGMA: tighter for nyc/chicago NOAA"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 13: Wider sigmas for weak cities (seoul, buenos_aires)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"seoul\":        {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
    \"buenos_aires\": {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
}''')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CITY_SIGMA: wider for seoul/buenos_aires"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 14: Full per-city sigma differentiation
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"nyc\":          {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
    \"chicago\":      {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
    \"london\":       {0: 1.3, 1: 1.8, 2: 2.7, 3: 3.2, 4: 3.8, 5: 4.2, 6: 4.8},
    \"seoul\":        {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
    \"buenos_aires\": {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
}''')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CITY_SIGMA: full differentiation all 5 cities"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 15: Per-city min_edge overrides (lower for NOAA cities)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_OVERRIDES = {}', '''CITY_OVERRIDES = {
    \"nyc\":     {\"min_edge\": 0.07},
    \"chicago\": {\"min_edge\": 0.07},
    \"seoul\":        {\"min_edge\": 0.13},
    \"buenos_aires\": {\"min_edge\": 0.13},
}''')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CITY_OVERRIDES: min_edge 0.07 NOAA, 0.13 weak"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 16: Tighter global sigmas (all -0.5)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new = '''FORECAST_SIGMA = {
    0: 1.0,
    1: 1.5,
    2: 2.5,
    3: 3.0,
    4: 3.5,
    5: 4.0,
    6: 4.5,
}'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "FORECAST_SIGMA: all -0.5 (tighter)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 17: Much tighter global sigmas (all -1.0)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new = '''FORECAST_SIGMA = {
    0: 0.8,
    1: 1.2,
    2: 2.0,
    3: 2.5,
    4: 3.0,
    5: 3.5,
    6: 4.0,
}'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "FORECAST_SIGMA: all -1.0 (much tighter)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 18: STUDENT_T_DF 5->3 (heavier tails)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('STUDENT_T_DF = 5', 'STUDENT_T_DF = 3')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF 5->3 (heavier tails)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 19: STUDENT_T_DF 5->8 (lighter tails)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('STUDENT_T_DF = 5', 'STUDENT_T_DF = 8').replace('STUDENT_T_DF = 3', 'STUDENT_T_DF = 8')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF->8 (lighter tails)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 20: STUDENT_T_DF 5->15 (near-normal)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('STUDENT_T_DF = 5', 'STUDENT_T_DF = 15')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF->15 (near-normal)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 21: MIN_FORECAST_PROB 0.20->0.15
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_FORECAST_PROB = 0.20', 'MIN_FORECAST_PROB = 0.15')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_FORECAST_PROB 0.20->0.15"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 22: MIN_FORECAST_PROB 0.20->0.10
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_FORECAST_PROB = 0.20', 'MIN_FORECAST_PROB = 0.10').replace('MIN_FORECAST_PROB = 0.15', 'MIN_FORECAST_PROB = 0.10')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_FORECAST_PROB->0.10"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 23: MAX_POSITION_PCT 0.05->0.08
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MAX_POSITION_PCT = 0.05', 'MAX_POSITION_PCT = 0.08')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MAX_POSITION_PCT 0.05->0.08"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 24: MAX_POSITION_PCT 0.05->0.10
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MAX_POSITION_PCT = 0.05', 'MAX_POSITION_PCT = 0.10').replace('MAX_POSITION_PCT = 0.08', 'MAX_POSITION_PCT = 0.10')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MAX_POSITION_PCT->0.10"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 25: MIN_VOLUME 2000->1000 (more markets)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_VOLUME = 2000.0', 'MIN_VOLUME = 1000.0')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_VOLUME 2000->1000"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 26: MIN_LIQUIDITY 1000->500
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_LIQUIDITY = 1000.0', 'MIN_LIQUIDITY = 500.0')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_LIQUIDITY 1000->500"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 27: Days out [0, 1, 2, 3] (add day 0)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
import re
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1, 2, 3]', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT [0,1,2,3] (add day 0)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 28: Edge-weighted Kelly sizing
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
# Replace the position_size function with edge-weighted Kelly
old_fn = '''    kelly_adjusted = kelly_raw * KELLY_FRACTION'''
new_fn = '''    # Edge-weighted: scale Kelly by confidence (edge/0.30 capped at 1.0)
    confidence = min(1.0, edge / 0.30)
    kelly_adjusted = kelly_raw * KELLY_FRACTION * (0.5 + 0.5 * confidence)'''
c = c.replace(old_fn, new_fn)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "edge-weighted Kelly (confidence scaling)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 29: Ensemble probability (avg of normal + student-t)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    if DISTRIBUTION == \"student_t\":
        cdf = lambda x: _student_t_cdf(x, forecast_temp_c, sigma, STUDENT_T_DF)
    else:
        cdf = lambda x: _normal_cdf(x, forecast_temp_c, sigma)'''
new = '''    # Ensemble: average normal and Student-t CDFs for robustness
    cdf_normal = lambda x: _normal_cdf(x, forecast_temp_c, sigma)
    cdf_student = lambda x: _student_t_cdf(x, forecast_temp_c, sigma, STUDENT_T_DF)
    cdf = lambda x: 0.5 * cdf_normal(x) + 0.5 * cdf_student(x)'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "ensemble probability (0.5*normal + 0.5*student-t)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 30: Ensemble with 70% normal, 30% student-t
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    if DISTRIBUTION == \"student_t\":
        cdf = lambda x: _student_t_cdf(x, forecast_temp_c, sigma, STUDENT_T_DF)
    else:
        cdf = lambda x: _normal_cdf(x, forecast_temp_c, sigma)'''
new = '''    # Ensemble: 70% normal + 30% Student-t
    cdf_normal = lambda x: _normal_cdf(x, forecast_temp_c, sigma)
    cdf_student = lambda x: _student_t_cdf(x, forecast_temp_c, sigma, STUDENT_T_DF)
    cdf = lambda x: 0.7 * cdf_normal(x) + 0.3 * cdf_student(x)'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "ensemble probability (0.7*normal + 0.3*student-t)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 31: MIN_PRICE 0.05->0.03
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MIN_PRICE = 0.05', 'MIN_PRICE = 0.03')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_PRICE 0.05->0.03"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 32: MAX_TOTAL_EXPOSURE_PCT 0.80->0.95
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('MAX_TOTAL_EXPOSURE_PCT = 0.80', 'MAX_TOTAL_EXPOSURE_PCT = 0.95')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MAX_TOTAL_EXPOSURE_PCT 0.80->0.95"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 33: CONVICTION_RATIO 1.5->1.3
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CONVICTION_RATIO = 1.5', 'CONVICTION_RATIO = 1.3').replace('CONVICTION_RATIO = 0.0', 'CONVICTION_RATIO = 1.3').replace('CONVICTION_RATIO = 1.2', 'CONVICTION_RATIO = 1.3')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CONVICTION_RATIO->1.3"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 34: KELLY_FRACTION 0.25->0.20 (more conservative)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
import re
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.20', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION->0.20 (more conservative)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 35: Adaptive sigma (linear model: base + slope*days_out)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))'''
new = '''SIGMA_BASE = 1.2
SIGMA_SLOPE = 0.6

def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    # Adaptive linear sigma model
    return SIGMA_BASE + SIGMA_SLOPE * days_out'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "adaptive sigma (1.2 + 0.6*days_out)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 36: Adaptive sigma (tighter: 0.8 + 0.5*days_out)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))'''
new = '''SIGMA_BASE = 0.8
SIGMA_SLOPE = 0.5

def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    # Adaptive linear sigma model
    return SIGMA_BASE + SIGMA_SLOPE * days_out'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "adaptive sigma (0.8 + 0.5*days_out)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 37: Quadratic sigma model
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))'''
new = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    # Quadratic sigma: captures non-linear forecast degradation
    return 1.0 + 0.3 * days_out + 0.1 * days_out ** 2'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "quadratic sigma (1.0 + 0.3d + 0.1d^2)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 38: sqrt sigma model
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import math
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))'''
new = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    # sqrt model: sigma grows as sqrt of days_out (physics-inspired diffusion)
    import math
    return 1.0 + 1.2 * math.sqrt(max(0, days_out))'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "sqrt sigma (1.0 + 1.2*sqrt(d))"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 39: MIN_EDGE 0.10->0.12 (higher selectivity)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.12', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.12 (higher selectivity)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 40: MIN_EDGE 0.10->0.15
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.15', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.15 (very selective)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# PHASE 2: COMBINATIONS OF WINNERS
# At this point we've tried many individual changes. Now combine the ones
# that worked. Read the results.tsv to see what improved.
# ============================================================================

log "Phase 1 complete. Starting Phase 2: combinations of winners."
log "Current best score: $BEST_SCORE"

# ============================================================================
# EXPERIMENT 41: Combine: tighter sigmas (-0.5) + lower MIN_EDGE (0.08)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 1.0,
    1: 1.5,
    2: 2.5,
    3: 3.0,
    4: 3.5,
    5: 4.0,
    6: 4.5,
}'''
c = c.replace(old_sig, new_sig)
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.08', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "combo: tighter sigma(-0.5) + MIN_EDGE=0.08"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 42: Combine: normal dist + lower MIN_EDGE 0.08
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('DISTRIBUTION = \"student_t\"', 'DISTRIBUTION = \"normal\"')
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.08', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "combo: normal dist + MIN_EDGE=0.08"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 43: Combine: city sigma + city overrides + lower edge
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"nyc\":          {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
    \"chicago\":      {0: 1.0, 1: 1.5, 2: 2.2, 3: 2.8, 4: 3.5, 5: 4.0, 6: 4.5},
    \"london\":       {0: 1.3, 1: 1.8, 2: 2.7, 3: 3.2, 4: 3.8, 5: 4.2, 6: 4.8},
    \"seoul\":        {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
    \"buenos_aires\": {0: 2.0, 1: 2.5, 2: 3.5, 3: 4.5, 4: 5.0, 5: 5.5, 6: 6.0},
}''')
c = c.replace('CITY_OVERRIDES = {}', '''CITY_OVERRIDES = {
    \"nyc\":     {\"min_edge\": 0.07},
    \"chicago\": {\"min_edge\": 0.07},
    \"seoul\":        {\"min_edge\": 0.13},
    \"buenos_aires\": {\"min_edge\": 0.13},
}''')
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.08', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "combo: city_sigma + city_overrides + MIN_EDGE=0.08"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 44: Combine: tighter sigma + KELLY=0.30 + DAYS_OUT=[0,1,2]
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 1.0,
    1: 1.5,
    2: 2.5,
    3: 3.0,
    4: 3.5,
    5: 4.0,
    6: 4.5,
}'''
c = c.replace(old_sig, new_sig)
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.30', c)
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1, 2]', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "combo: tighter sigma + KELLY=0.30 + DAYS=[0,1,2]"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 45: All best single improvements combined
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
# Tighter sigma
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 1.0,
    1: 1.5,
    2: 2.5,
    3: 3.0,
    4: 3.5,
    5: 4.0,
    6: 4.5,
}'''
c = c.replace(old_sig, new_sig)
# Lower min_edge
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.08', c)
# Higher Kelly
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.30', c)
# Remove conviction ratio
c = c.replace('CONVICTION_RATIO = 1.5', 'CONVICTION_RATIO = 0.0')
# Normal dist
c = c.replace('DISTRIBUTION = \"student_t\"', 'DISTRIBUTION = \"normal\"')
# More days
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1, 2]', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "all-best-combined: sigma-0.5,edge0.08,kelly0.3,noconv,normal,days012"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 46: KELLY_FRACTION 0.25->0.40 (aggressive)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.40', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION->0.40 (aggressive)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 47: DAYS_OUT_TO_TRADE = [0, 1, 2, 3, 4]
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1, 2, 3, 4]', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT [0,1,2,3,4]"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 48: Very tight day0+1 sigma, normal dist, no conviction
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 0.8,
    1: 1.2,
    2: 2.0,
    3: 2.5,
    4: 3.0,
    5: 3.5,
    6: 4.0,
}'''
c = c.replace(old_sig, new_sig)
c = c.replace('DISTRIBUTION = \"student_t\"', 'DISTRIBUTION = \"normal\"')
c = c.replace('CONVICTION_RATIO = 1.5', 'CONVICTION_RATIO = 0.0')
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1]', c)
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.06', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "tight d0d1: sigma-1.0,normal,noconv,days01,edge0.06"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 49: Aggressive sizing combo
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.35', c)
c = re.sub(r'MAX_POSITION_PCT = [0-9.]+', 'MAX_POSITION_PCT = 0.08', c)
c = re.sub(r'MAX_TOTAL_EXPOSURE_PCT = [0-9.]+', 'MAX_TOTAL_EXPOSURE_PCT = 0.95', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "aggressive sizing: kelly0.35,maxpos0.08,exposure0.95"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 50: Min edge 0.05 (very low threshold)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.05', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.05 (very low)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# PHASE 3: Fine-tuning around best parameters
# ============================================================================

log "Phase 2 complete. Starting Phase 3: fine-tuning."
log "Current best score: $BEST_SCORE"

# ============================================================================
# EXPERIMENT 51: CONVICTION_RATIO 1.5->1.1
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'CONVICTION_RATIO = [0-9.]+', 'CONVICTION_RATIO = 1.1', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CONVICTION_RATIO->1.1"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 52: CONVICTION_RATIO 1.5->1.4
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'CONVICTION_RATIO = [0-9.]+', 'CONVICTION_RATIO = 1.4', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CONVICTION_RATIO->1.4"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 53: MIN_EDGE 0.10->0.09
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.09', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.09"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 54: MIN_EDGE 0.10->0.11
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.11', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "MIN_EDGE->0.11"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 55: KELLY_FRACTION 0.25->0.28
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.28', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION->0.28"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 56: KELLY_FRACTION 0.25->0.32
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.32', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "KELLY_FRACTION->0.32"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 57: STUDENT_T_DF 5->4
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'STUDENT_T_DF = [0-9]+', 'STUDENT_T_DF = 4', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF->4"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 58: STUDENT_T_DF 5->6
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'STUDENT_T_DF = [0-9]+', 'STUDENT_T_DF = 6', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF->6"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 59: STUDENT_T_DF 5->10
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'STUDENT_T_DF = [0-9]+', 'STUDENT_T_DF = 10', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "STUDENT_T_DF->10"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 60: Sigma tighter by 0.3 (smaller step)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 1.2,
    1: 1.7,
    2: 2.7,
    3: 3.2,
    4: 3.7,
    5: 4.2,
    6: 4.7,
}'''
c = c.replace(old_sig, new_sig)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "FORECAST_SIGMA: all -0.3 (modest tighter)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 61: Days [0, 1, 2, 3]
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = [0, 1, 2, 3]', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "DAYS_OUT [0,1,2,3]"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 62: City-specific Kelly (higher for NOAA cities)
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
# Add city Kelly multiplier to position sizing
old = '''    kelly_adjusted = kelly_raw * KELLY_FRACTION'''
new = '''    # City-specific Kelly: NOAA cities get full fraction, weak cities get 70%
    city_kelly_mult = {\"nyc\": 1.0, \"chicago\": 1.0, \"london\": 0.9, \"seoul\": 0.7, \"buenos_aires\": 0.7}
    mult = city_kelly_mult.get(city, 1.0) if city else 1.0
    kelly_adjusted = kelly_raw * KELLY_FRACTION * mult'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "city-specific Kelly (NOAA 1.0, weak 0.7)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 63: Exclude weak cities entirely
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    if edge < _get_threshold(\"min_edge\", city):
        return False'''
new = '''    # Skip weak data cities
    if city in (\"seoul\", \"buenos_aires\"):
        return False
    if edge < _get_threshold(\"min_edge\", city):
        return False'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "exclude seoul and buenos_aires"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 64: Higher MIN_EDGE for day3 trades
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    if edge < _get_threshold(\"min_edge\", city):
        return False'''
new = '''    min_e = _get_threshold(\"min_edge\", city)
    # Day 3+ needs higher edge threshold
    if days_out >= 3:
        min_e = max(min_e, 0.15)
    if edge < min_e:
        return False'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "higher MIN_EDGE (0.15) for days_out>=3"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 65: Days-out adaptive edge
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    if edge < _get_threshold(\"min_edge\", city):
        return False'''
new = '''    base_edge = _get_threshold(\"min_edge\", city)
    # Adaptive edge: tighter for near-term, wider for far-out
    edge_by_days = {0: base_edge * 0.7, 1: base_edge * 0.8, 2: base_edge, 3: base_edge * 1.3}
    required_edge = edge_by_days.get(days_out, base_edge * 1.5)
    if edge < required_edge:
        return False'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "adaptive edge by days_out (0.7x-1.5x multiplier)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 66: Days-out adaptive Kelly
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''    kelly_adjusted = kelly_raw * KELLY_FRACTION'''
new = '''    # Days-out factor: bet more on near-term (d0=1.2x, d3=0.7x)
    # We don't have days_out here, so use edge as proxy — higher edge = likely closer day
    days_proxy = min(1.0, edge / 0.20)  # normalized edge
    kelly_adjusted = kelly_raw * KELLY_FRACTION * (0.8 + 0.4 * days_proxy)'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "edge-proxy days Kelly (0.8-1.2x)"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 67: NOAA city sigma even tighter
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"nyc\":     {0: 0.8, 1: 1.2, 2: 1.8, 3: 2.2, 4: 2.8, 5: 3.5, 6: 4.0},
    \"chicago\": {0: 0.8, 1: 1.2, 2: 1.8, 3: 2.2, 4: 2.8, 5: 3.5, 6: 4.0},
}''')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "CITY_SIGMA: very tight for nyc/chicago NOAA"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 68: Small position size min threshold
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('if size < 1.0:', 'if size < 0.5:')
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "min position size 1.0->0.5"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 69: Combo - best sigma + normal + lower edge + higher kelly
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
old_sig = '''FORECAST_SIGMA = {
    0: 1.5,
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}'''
new_sig = '''FORECAST_SIGMA = {
    0: 1.2,
    1: 1.7,
    2: 2.7,
    3: 3.2,
    4: 3.7,
    5: 4.2,
    6: 4.7,
}'''
c = c.replace(old_sig, new_sig)
c = c.replace('DISTRIBUTION = \"student_t\"', 'DISTRIBUTION = \"normal\"')
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = 0.08', c)
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.30', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "combo: sigma-0.3, normal, edge0.08, kelly0.30"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 70: Full city optimization with sigma+overrides+kelly
# ============================================================================
cp research/strategy_experiment.py.baseline research/strategy_experiment.py
python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = c.replace('CITY_SIGMA = {}', '''CITY_SIGMA = {
    \"nyc\":          {0: 0.8, 1: 1.2, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0},
    \"chicago\":      {0: 0.8, 1: 1.2, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0},
    \"london\":       {0: 1.2, 1: 1.7, 2: 2.5, 3: 3.0, 4: 3.5, 5: 4.0, 6: 4.5},
    \"seoul\":        {0: 1.8, 1: 2.3, 2: 3.2, 3: 4.0, 4: 4.5, 5: 5.0, 6: 5.5},
    \"buenos_aires\": {0: 1.5, 1: 2.0, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0},
}''')
c = c.replace('CITY_OVERRIDES = {}', '''CITY_OVERRIDES = {
    \"nyc\":          {\"min_edge\": 0.06, \"conviction_ratio\": 1.3},
    \"chicago\":      {\"min_edge\": 0.06, \"conviction_ratio\": 1.3},
    \"london\":       {\"min_edge\": 0.08, \"conviction_ratio\": 1.4},
    \"seoul\":        {\"min_edge\": 0.12, \"conviction_ratio\": 1.6},
    \"buenos_aires\": {\"min_edge\": 0.10, \"conviction_ratio\": 1.5},
}''')
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = 0.30', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
run_experiment "full city optimization: sigma+overrides+kelly0.30"
if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi

# ============================================================================
# EXPERIMENT 71-80: Grid search on sigma base (using adaptive linear model)
# ============================================================================
for BASE in "0.5" "0.6" "0.7" "0.9" "1.1" "1.3" "1.5"; do
    for SLOPE in "0.4" "0.5" "0.6" "0.7" "0.8"; do
        cp research/strategy_experiment.py.baseline research/strategy_experiment.py
        python3 -c "
with open('research/strategy_experiment.py') as f: c = f.read()
old = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    return FORECAST_SIGMA.get(days_out, FORECAST_SIGMA.get(6, 5.0))'''
new = '''def _get_sigma(days_out, city=None):
    \"\"\"Get forecast sigma for a given days_out and optional city.\"\"\"
    if city and city in CITY_SIGMA:
        return CITY_SIGMA[city].get(days_out, CITY_SIGMA[city].get(6, 5.0))
    # Adaptive linear sigma
    return ${BASE} + ${SLOPE} * days_out'''
c = c.replace(old, new)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
        run_experiment "grid sigma: base=${BASE} slope=${SLOPE}"
        if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi
    done
done

# ============================================================================
# EXPERIMENT: Grid search MIN_EDGE x KELLY_FRACTION
# ============================================================================
for EDGE in "0.06" "0.07" "0.08" "0.09" "0.10" "0.12"; do
    for KELLY in "0.20" "0.25" "0.28" "0.30" "0.35"; do
        cp research/strategy_experiment.py.baseline research/strategy_experiment.py
        python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'MIN_EDGE = [0-9.]+', 'MIN_EDGE = ${EDGE}', c)
c = re.sub(r'KELLY_FRACTION = [0-9.]+', 'KELLY_FRACTION = ${KELLY}', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
        run_experiment "grid: MIN_EDGE=${EDGE} KELLY=${KELLY}"
        if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi
    done
done

# ============================================================================
# EXPERIMENT: Grid search CONVICTION_RATIO x DAYS_OUT
# ============================================================================
for CONV in "0.0" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6"; do
    for DAYS in "[0, 1]" "[0, 1, 2]" "[1, 2]" "[1, 2, 3]" "[0, 1, 2, 3]"; do
        cp research/strategy_experiment.py.baseline research/strategy_experiment.py
        python3 -c "
import re
with open('research/strategy_experiment.py') as f: c = f.read()
c = re.sub(r'CONVICTION_RATIO = [0-9.]+', 'CONVICTION_RATIO = ${CONV}', c)
c = re.sub(r'DAYS_OUT_TO_TRADE = \[.*?\]', 'DAYS_OUT_TO_TRADE = ${DAYS}', c)
with open('research/strategy_experiment.py', 'w') as f: f.write(c)
"
        DESC="grid: CONV=${CONV} DAYS=${DAYS}"
        run_experiment "$DESC"
        if [ $? -eq 0 ]; then cp research/strategy_experiment.py research/strategy_experiment.py.baseline; fi
    done
done

# ============================================================================
# FINAL REPORT
# ============================================================================
log "============================================"
log "EXPERIMENT RUN COMPLETE"
log "Final best score: $BEST_SCORE"
log "Results saved to research/results.tsv"
log "============================================"

# Clean up
rm -f research/strategy_experiment.py.baseline /tmp/patch_exp.py research/experiment.pid

log "Done."
