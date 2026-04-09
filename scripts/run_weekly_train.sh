#!/bin/bash
# QuantPilot Weekly Training Pipeline (native venv)
# Schedule: cron Saturday 10:00
#
# Steps:
# 1. Sync latest Qlib data from NAS
# 2. Run model training + backtest + signal promotion + email report (in venv)

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

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
if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    log "  ERROR: venv activate script missing: $PROJECT_DIR/.venv/bin/activate"
    exit 1
fi

log "  Working directory: $PROJECT_DIR"
log "  DATA_DIR: $DATA_DIR"
log "  MODELS_DIR: $DATA_DIR/models"
log "  OUTPUT_DIR: $DATA_DIR/output"
log "  SIGNAL_DIR: $DATA_DIR/signals"
log "  SMTP_HOST: ${SMTP_HOST:-smtp.gmail.com}"
log "  SMTP_PORT: ${SMTP_PORT:-587}"
log "  SMTP_USER: $( [ -n "${SMTP_USER:-}" ] && printf set || printf missing )"
log "  SMTP_PASSWORD: $( [ -n "${SMTP_PASSWORD:-}" ] && printf set || printf missing )"
log "  EMAIL_TO: $( [ -n "${EMAIL_TO:-}" ] && printf set || printf missing )"
log "  REPORT_TO: $( [ -n "${REPORT_TO:-}" ] && printf set || printf missing )"

source .venv/bin/activate

QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
STRATEGY_DIR="$PROJECT_DIR" \
MODELS_DIR="$DATA_DIR/models" \
OUTPUT_DIR="$DATA_DIR/output" \
SIGNAL_DIR="$DATA_DIR/signals" \
TRADE_PRED_PATH="$DATA_DIR/models/pred_sh.pkl" \
    python -m trainer.weekly_train

log "Weekly training pipeline finished!"
