#!/bin/bash
# QuantPilot Trading — run trader natively in venv
# Schedule: cron 14:50 Mon-Fri

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

DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    log "run_trade: skip (weekend)"
    exit 0
fi

log "run_trade: start"
cd "$PROJECT_DIR"
source .venv/bin/activate

FUTU_HOST="${FUTU_HOST:-192.168.100.248}" \
FUTU_PORT="${FUTU_PORT:-11111}" \
PRED_PATH="$DATA_DIR/signals/pred_sh_latest.pkl" \
QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
SIGNAL_DIR="$DATA_DIR/signals" \
TOP_N="${TOP_N:-5}" \
HOLD_BONUS="${HOLD_BONUS:-0.05}" \
STOP_LOSS_PCT="${STOP_LOSS_PCT:--0.08}" \
DRY_RUN="${DRY_RUN:-false}" \
    python -m trader.trade_daily

log "run_trade: done"
