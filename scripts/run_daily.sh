#!/bin/bash
# QuantPilot Daily Pipeline
# Schedule: cron 19:00 Mon-Fri (after NAS daily collection flush)
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
ALLOW_STALE_SYNC="${ALLOW_STALE_SYNC:-false}"
NAS_COLLECTOR_CONTAINER="${NAS_COLLECTOR_CONTAINER:-quantpilot-collector}"
TARGET_DATE_LOOKBACK_DAYS="${TARGET_DATE_LOOKBACK_DAYS:-31}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Step 0: Wait for NAS collector to finish today's data
if [ -n "$NAS_HOST" ] && [ -n "$NAS_USER" ]; then
    log "Step 0: Waiting for NAS data to be ready..."
    TODAY=$(date +%Y-%m-%d)
    TARGET_A_SHARE_DATE=$(
        PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-target-date \
            --nas-host "$NAS_HOST" \
            --nas-user "$NAS_USER" \
            --ssh-key "$SSH_KEY" \
            --today "$TODAY" \
            --collector-container "$NAS_COLLECTOR_CONTAINER" \
            --lookback-days "$TARGET_DATE_LOOKBACK_DAYS"
    )
    if [ -z "$TARGET_A_SHARE_DATE" ]; then
        log "  ERROR: failed to resolve target A-share trading date"
        exit 1
    fi
    log "  Target A-share trading date: $TARGET_A_SHARE_DATE"
    WAITED=0
    NAS_LAST=""
    while [ $WAITED -lt $MAX_WAIT_SECONDS ]; do
        NAS_LAST=$(
            PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-completed-date \
                --nas-host "$NAS_HOST" \
                --nas-user "$NAS_USER" \
                --ssh-key "$SSH_KEY" \
                --nas-qlib-path "$NAS_QLIB_PATH"
        )
        if [ -n "$NAS_LAST" ] && [ "$NAS_LAST" \> "$TARGET_A_SHARE_DATE" -o "$NAS_LAST" = "$TARGET_A_SHARE_DATE" ]; then
            log "  NAS A-share data ready (completed_a_share=$NAS_LAST)"
            break
        fi
        log "  NAS completed_a_share=$NAS_LAST, waiting for $TARGET_A_SHARE_DATE... (${WAITED}s/${MAX_WAIT_SECONDS}s)"
        sleep $WAIT_INTERVAL_SECONDS
        WAITED=$((WAITED + WAIT_INTERVAL_SECONDS))
    done
    if [ -z "$NAS_LAST" ] || [ "$NAS_LAST" \< "$TARGET_A_SHARE_DATE" ]; then
        if [ "$ALLOW_STALE_SYNC" = "true" ]; then
            log "  WARNING: NAS A-share data not ready after ${MAX_WAIT_SECONDS}s, proceeding with available data ($NAS_LAST)"
        else
            log "  ERROR: NAS A-share data not ready after ${MAX_WAIT_SECONDS}s, aborting to avoid stale/inconsistent sync ($NAS_LAST)"
            exit 1
        fi
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
