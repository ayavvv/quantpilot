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
EXTERNAL_ENABLE_PRETRADE_CAPITAL_FLOW_CHECK="${ENABLE_PRETRADE_CAPITAL_FLOW_CHECK-}"
EXTERNAL_ENABLE_CAPITAL_FLOW_ADVISORY="${ENABLE_CAPITAL_FLOW_ADVISORY-}"

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
[ -n "$EXTERNAL_ENABLE_PRETRADE_CAPITAL_FLOW_CHECK" ] && ENABLE_PRETRADE_CAPITAL_FLOW_CHECK="$EXTERNAL_ENABLE_PRETRADE_CAPITAL_FLOW_CHECK"
[ -n "$EXTERNAL_ENABLE_CAPITAL_FLOW_ADVISORY" ] && ENABLE_CAPITAL_FLOW_ADVISORY="$EXTERNAL_ENABLE_CAPITAL_FLOW_ADVISORY"

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
ENABLE_PRETRADE_CAPITAL_FLOW_CHECK="${ENABLE_PRETRADE_CAPITAL_FLOW_CHECK:-true}"
ENABLE_CAPITAL_FLOW_ADVISORY="${ENABLE_CAPITAL_FLOW_ADVISORY:-true}"
PRETRADE_CAPITAL_FLOW_TOP_N="${PRETRADE_CAPITAL_FLOW_TOP_N:-10}"
PRETRADE_CAPITAL_FLOW_DAYS="${PRETRADE_CAPITAL_FLOW_DAYS:-30}"
PRETRADE_CAPITAL_FLOW_CONNECT_TIMEOUT="${PRETRADE_CAPITAL_FLOW_CONNECT_TIMEOUT:-5}"
PRETRADE_CAPITAL_FLOW_SIGNAL_CSV="${PRETRADE_CAPITAL_FLOW_SIGNAL_CSV:-$DATA_DIR/signals/signal_latest.csv}"
PRETRADE_CAPITAL_FLOW_OVERLAY_CSV="${PRETRADE_CAPITAL_FLOW_OVERLAY_CSV:-$DATA_DIR/output/pretrade_futu_capital_flow_signal_overlay_latest.csv}"
PRETRADE_CAPITAL_FLOW_CSV="${PRETRADE_CAPITAL_FLOW_CSV:-$DATA_DIR/output/pretrade_futu_capital_flow_latest.csv}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_healthcheck() {
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.daily_healthcheck \
        --phase trade \
        --alert-on error || true
}

DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    log "run_trade: skip (weekend)"
    exit 0
fi

log "run_trade: start"
cd "$PROJECT_DIR"
source .venv/bin/activate

if [ "$ENABLE_PRETRADE_CAPITAL_FLOW_CHECK" = "true" ]; then
    if [ -f "$PRETRADE_CAPITAL_FLOW_SIGNAL_CSV" ]; then
        log "Pre-trade Futu capital-flow advisory..."
        if PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" -m scripts.build_futu_capital_flow_overlay \
                --signal-csv "$PRETRADE_CAPITAL_FLOW_SIGNAL_CSV" \
                --fetch-latest \
                --host "${FUTU_HOST:-192.168.100.248}" \
                --port "${FUTU_PORT:-11111}" \
                --days "$PRETRADE_CAPITAL_FLOW_DAYS" \
                --include-distribution \
                --connect-timeout "$PRETRADE_CAPITAL_FLOW_CONNECT_TIMEOUT" \
                --signal-top-n "$PRETRADE_CAPITAL_FLOW_TOP_N" \
                --output "$PRETRADE_CAPITAL_FLOW_OVERLAY_CSV" \
                --flow-output "$PRETRADE_CAPITAL_FLOW_CSV"; then
            log "  Pre-trade capital-flow advisory ready: $PRETRADE_CAPITAL_FLOW_OVERLAY_CSV"
        else
            log "  WARNING: Pre-trade capital-flow advisory failed; continuing without capital-flow advisory"
        fi
    else
        log "Pre-trade Futu capital-flow advisory skipped; signal csv missing: $PRETRADE_CAPITAL_FLOW_SIGNAL_CSV"
    fi
fi

TRADE_RC=0
if FUTU_HOST="${FUTU_HOST:-192.168.100.248}" \
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
ENABLE_CAPITAL_FLOW_ADVISORY="$ENABLE_CAPITAL_FLOW_ADVISORY" \
CAPITAL_FLOW_OVERLAY_CSV="$PRETRADE_CAPITAL_FLOW_OVERLAY_CSV" \
    python -m trader.trade_daily; then
    TRADE_RC=0
else
    TRADE_RC=$?
fi

log "run_trade: done"
run_healthcheck
exit "$TRADE_RC"
