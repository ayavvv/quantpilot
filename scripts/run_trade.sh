#!/bin/bash
# QuantPilot Trading — run trader natively in venv
# Schedule: cron 14:50 Mon-Fri

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Preserve caller-provided overrides before sourcing .env defaults.
EXTERNAL_DATA_DIR="${DATA_DIR-}"
EXTERNAL_FUTU_HOST="${FUTU_HOST-}"
EXTERNAL_FUTU_PORT="${FUTU_PORT-}"
EXTERNAL_FUTU_SIM_ACC_ID="${FUTU_SIM_ACC_ID-}"
EXTERNAL_FUTU_RSA_KEY="${FUTU_RSA_KEY-}"
EXTERNAL_TOP_N="${TOP_N-}"
EXTERNAL_HOLD_BONUS="${HOLD_BONUS-}"
EXTERNAL_STOP_LOSS_PCT="${STOP_LOSS_PCT-}"
EXTERNAL_DRY_RUN="${DRY_RUN-}"

# Load .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

[ -n "$EXTERNAL_DATA_DIR" ] && DATA_DIR="$EXTERNAL_DATA_DIR"
[ -n "$EXTERNAL_FUTU_HOST" ] && FUTU_HOST="$EXTERNAL_FUTU_HOST"
[ -n "$EXTERNAL_FUTU_PORT" ] && FUTU_PORT="$EXTERNAL_FUTU_PORT"
[ -n "$EXTERNAL_FUTU_SIM_ACC_ID" ] && FUTU_SIM_ACC_ID="$EXTERNAL_FUTU_SIM_ACC_ID"
[ -n "$EXTERNAL_FUTU_RSA_KEY" ] && FUTU_RSA_KEY="$EXTERNAL_FUTU_RSA_KEY"
[ -n "$EXTERNAL_TOP_N" ] && TOP_N="$EXTERNAL_TOP_N"
[ -n "$EXTERNAL_HOLD_BONUS" ] && HOLD_BONUS="$EXTERNAL_HOLD_BONUS"
[ -n "$EXTERNAL_STOP_LOSS_PCT" ] && STOP_LOSS_PCT="$EXTERNAL_STOP_LOSS_PCT"
[ -n "$EXTERNAL_DRY_RUN" ] && DRY_RUN="$EXTERNAL_DRY_RUN"

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
FUTU_SIM_ACC_ID="${FUTU_SIM_ACC_ID:-0}" \
FUTU_RSA_KEY="${FUTU_RSA_KEY:-}" \
PRED_PATH="$DATA_DIR/signals/pred_sh_latest.pkl" \
QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
SIGNAL_DIR="$DATA_DIR/signals" \
TOP_N="${TOP_N:-5}" \
HOLD_BONUS="${HOLD_BONUS:-0.05}" \
STOP_LOSS_PCT="${STOP_LOSS_PCT:--0.08}" \
DRY_RUN="${DRY_RUN:-false}" \
    python -m trader.trade_daily

log "run_trade: done"
