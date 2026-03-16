#!/bin/bash
# QuantPilot Weekly Training Pipeline
# Schedule: cron Saturday 10:00
#
# Steps:
# 1. Sync latest kline data
# 2. Run model training + backtest
# 3. Deploy new model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

DOCKER="${DOCKER:-docker}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Step 1: Sync data
log "Step 1: Syncing data..."
"$SCRIPT_DIR/sync_data.sh"

# Step 2: Run trainer
log "Step 2: Running weekly training..."
cd "$PROJECT_DIR"
$DOCKER compose run --rm trainer
log "  Training complete"

log "Weekly training pipeline finished!"
