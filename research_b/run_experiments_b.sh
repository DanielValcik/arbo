#!/bin/bash
# Autonomous experiment runner for Strategy B Reflexivity Surfer optimization
# Runs experiments, logging results to results_b.tsv
# Kill with: kill $(cat research_b/experiment_b.pid)

cd /Users/dnl.vlck/Arbo
echo $$ > research_b/experiment_b.pid

BEST_SCORE=0.000000
LOG_FILE="research_b/experiment_b_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_experiment() {
    local DESC="$1"
    log "=== EXPERIMENT: $DESC ==="

    # Commit the change
    git add research_b/strategy_b_experiment.py
    git commit -m "experiment: $DESC" --no-gpg-sign -q 2>/dev/null
    local COMMIT=$(git rev-parse --short HEAD)

    # Run backtest
    timeout 120 python3 research_b/backtest_b_harness.py > research_b/run.log 2>&1
    local EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        log "CRASH (exit=$EXIT_CODE)"
        local ERROR=$(tail -5 research_b/run.log)
        log "Error: $ERROR"
        echo -e "${COMMIT}\t0.000000\t0.00\t0.00\t0.00\t0.0\t0\tcrash\t${DESC}" >> research_b/results_b.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi

    # Extract metrics
    local SCORE=$(grep "^composite_score:" research_b/run.log | awk '{print $2}')
    local SHARPE=$(grep "^avg_sharpe:" research_b/run.log | awk '{print $2}')
    local PNL=$(grep "^avg_pnl_pct:" research_b/run.log | awk '{print $2}')
    local DD=$(grep "^max_drawdown_pct:" research_b/run.log | awk '{print $2}')
    local WR=$(grep "^avg_win_rate:" research_b/run.log | awk '{print $2}')
    local NT=$(grep "^num_trades:" research_b/run.log | awk '{print $2}')

    if [ -z "$SCORE" ]; then
        log "CRASH: no score in output"
        echo -e "${COMMIT}\t0.000000\t0.00\t0.00\t0.00\t0.0\t0\tcrash\t${DESC}" >> research_b/results_b.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi

    log "Score: $SCORE (best: $BEST_SCORE) | Sharpe: $SHARPE | Trades: $NT | DD: $DD | WR: $WR"

    # Compare (using bc for float comparison)
    local BETTER=$(echo "$SCORE > $BEST_SCORE" | bc -l 2>/dev/null)

    if [ "$BETTER" = "1" ]; then
        log "KEEP - improved from $BEST_SCORE to $SCORE"
        echo -e "${COMMIT}\t${SCORE}\t${SHARPE}\t${PNL}\t${DD}\t${WR}\t${NT}\tkeep\t${DESC}" >> research_b/results_b.tsv
        BEST_SCORE=$SCORE
        return 0
    else
        log "DISCARD - $SCORE <= $BEST_SCORE"
        echo -e "${COMMIT}\t${SCORE}\t${SHARPE}\t${PNL}\t${DD}\t${WR}\t${NT}\tdiscard\t${DESC}" >> research_b/results_b.tsv
        git reset --hard HEAD~1 -q
        return 1
    fi
}

log "Starting Strategy B autonomous experiment loop. Best score: $BEST_SCORE"
log "PID: $$"

# NOTE: This script provides the run_experiment() function.
# The AI agent drives the loop by editing strategy_b_experiment.py
# and calling run_experiment with a description.
#
# To run manually (for a single experiment):
#   source research_b/run_experiments_b.sh
#   # edit strategy_b_experiment.py
#   run_experiment "description of change"
#
# Or the AI agent runs the full loop via program_b.md instructions.

# Baseline run (first experiment with default parameters)
run_experiment "BASELINE"
if [ $? -eq 0 ]; then
    cp research_b/strategy_b_experiment.py research_b/strategy_b_experiment.py.baseline
fi

log "Baseline complete. Score: $BEST_SCORE"
log "Agent should now take over the loop."
