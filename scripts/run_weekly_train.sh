#!/bin/bash
# QuantPilot Weekly Training Pipeline (native venv)
# Schedule: cron Saturday 10:00
#
# Steps:
# 1. Sync latest Qlib data from NAS
# 2. Run model training + backtest + email report (in venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Step 1: Sync data
log "Step 1: Syncing data..."
"$SCRIPT_DIR/sync_data.sh"

# Step 2: Run trainer natively
log "Step 2: Running weekly training..."
cd "$PROJECT_DIR"
source .venv/bin/activate

QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
STRATEGY_DIR="$PROJECT_DIR" \
MODELS_DIR="$DATA_DIR/models" \
OUTPUT_DIR="$DATA_DIR/output" \
TRADE_PRED_PATH="$DATA_DIR/models/pred_sh.pkl" \
    python -m trainer.weekly_train

log "Weekly training pipeline finished!"
