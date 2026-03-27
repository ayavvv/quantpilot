#!/bin/bash
# QuantPilot Data Sync - Sync Qlib bin data from NAS to local
#
# Uses tar+SSH (not rsync) because Synology NAS rsync has permission issues.
# Syncs the Qlib binary directory (~30MB) which is much faster than parquet.

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
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
REPAIR_QLIB_METADATA="${REPAIR_QLIB_METADATA:-true}"

if [ -z "$NAS_HOST" ] || [ -z "$NAS_USER" ]; then
    echo "Error: NAS_HOST and NAS_USER must be set in .env or environment"
    exit 1
fi

QLIB_DIR="${QLIB_DATA_DIR:-$DATA_DIR/qlib_data}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing Qlib data from ${NAS_USER}@${NAS_HOST}:${NAS_QLIB_PATH}..."

mkdir -p "$QLIB_DIR"

# 原子同步: 先写临时目录，成功后再替换，避免中断导致数据损坏
SYNC_TMP="${QLIB_DIR}.sync_tmp"
rm -rf "$SYNC_TMP"
mkdir -p "$SYNC_TMP"

if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "${NAS_USER}@${NAS_HOST}" \
    "cd ${NAS_QLIB_PATH} && tar cf - calendars instruments features" | \
    tar xf - -C "$SYNC_TMP/"; then
    # 同步成功，原子替换各子目录
    for subdir in calendars instruments features; do
        rm -rf "${QLIB_DIR:?}/$subdir"
        mv "$SYNC_TMP/$subdir" "$QLIB_DIR/$subdir"
    done
    rm -rf "$SYNC_TMP"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Sync failed, keeping existing data"
    rm -rf "$SYNC_TMP"
    exit 1
fi

# Stats
if [ "$REPAIR_QLIB_METADATA" != "false" ]; then
    if [ ! -x "$PYTHON_BIN" ]; then
        PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repairing Qlib instruments metadata..."
    "$PYTHON_BIN" "$SCRIPT_DIR/repair_qlib_metadata.py" --qlib-dir "$QLIB_DIR"
fi

N_DAYS=$(wc -l < "$QLIB_DIR/calendars/day.txt" 2>/dev/null | tr -d ' ' || echo 0)
N_STOCKS=$(ls -d "$QLIB_DIR/features/"* 2>/dev/null | wc -l | tr -d ' ')
SIZE=$(du -sh "$QLIB_DIR" 2>/dev/null | cut -f1)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync complete: ${N_DAYS} days, ${N_STOCKS} stocks, ${SIZE}"
