#!/bin/bash
# QuantPilot Daily Pipeline
# Schedule: cron 17:00 Mon-Fri (after market close + data collection)
#
# Steps:
# 1. Sync Qlib bin data from NAS (or skip if single-machine)
# 2. Run inference (validate data -> LightGBM predict)
# 3. Run reporter (generate + send daily report)

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
DOCKER="${DOCKER:-docker}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Step 1: Sync Qlib data from NAS (if NAS_HOST is configured)
if [ -n "$NAS_HOST" ]; then
    log "Step 1: Syncing Qlib data from NAS..."
    "$SCRIPT_DIR/sync_data.sh"
    log "  Sync complete"
else
    log "Step 1: Skipped (NAS_HOST not configured, single-machine mode)"
fi

# Step 2: Run inference
log "Step 2: Running inference..."
cd "$PROJECT_DIR"
$DOCKER compose run --rm inference
log "  Inference complete"

# Step 3: Run reporter
log "Step 3: Running reporter..."
$DOCKER compose run --rm reporter
log "  Report complete"

log "Daily pipeline finished!"
