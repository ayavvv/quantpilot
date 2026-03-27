#!/bin/bash
# QuantPilot Daily Pipeline
# Schedule: cron 17:30 Mon-Fri (after market close + data collection)
#
# Steps:
# 0. Wait for NAS collector to finish today's data
# 1. Sync Qlib bin data from NAS (or skip if single-machine)
# 2. Run inference natively in venv (validate data -> LightGBM predict)
# 3. Run reporter via Docker (generate + send daily report)

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
NAS_HOST="${NAS_HOST:-}"
NAS_USER="${NAS_USER:-}"
NAS_QLIB_PATH="${NAS_QLIB_PATH:-/volume1/docker/quantpilot/qlib_data}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DOCKER="${DOCKER:-docker}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Step 0: Wait for NAS collector to finish today's data
if [ -n "$NAS_HOST" ] && [ -n "$NAS_USER" ]; then
    log "Step 0: Waiting for NAS data to be ready..."
    TODAY=$(date +%Y-%m-%d)
    WAITED=0
    NAS_LAST=""
    while [ $WAITED -lt $MAX_WAIT_SECONDS ]; do
        NAS_LAST=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
            "${NAS_USER}@${NAS_HOST}" \
            "tail -1 ${NAS_QLIB_PATH}/calendars/day.txt" 2>/dev/null | tr -d '[:space:]')
        if [ "$NAS_LAST" = "$TODAY" ]; then
            log "  NAS data ready (last_date=$NAS_LAST)"
            break
        fi
        log "  NAS last_date=$NAS_LAST, waiting for $TODAY... (${WAITED}s/${MAX_WAIT_SECONDS}s)"
        sleep $WAIT_INTERVAL_SECONDS
        WAITED=$((WAITED + WAIT_INTERVAL_SECONDS))
    done
    if [ "$NAS_LAST" != "$TODAY" ]; then
        log "  WARNING: NAS data not ready after ${MAX_WAIT_SECONDS}s, proceeding with available data ($NAS_LAST)"
    fi
fi

# Step 1: Sync Qlib data from NAS (if NAS_HOST is configured)
if [ -n "$NAS_HOST" ]; then
    log "Step 1: Syncing Qlib data from NAS..."
    "$SCRIPT_DIR/sync_data.sh"
    log "  Sync complete"
else
    log "Step 1: Skipped (NAS_HOST not configured, single-machine mode)"
fi

# Step 2: Run inference natively (no Docker, avoids Rosetta OOM)
log "Step 2: Running inference..."
cd "$PROJECT_DIR"
source .venv/bin/activate
QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
MODEL_DIR="$DATA_DIR/models" \
SIGNAL_DIR="$DATA_DIR/signals" \
    python -m inference.run_daily
log "  Inference complete"

# Step 3: Run reporter via Docker
log "Step 3: Running reporter..."
$DOCKER compose -f docker-compose.mac.yml run --rm reporter
log "  Report complete"

log "Daily pipeline finished!"
