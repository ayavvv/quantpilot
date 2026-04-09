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
EXPECTED_TARGET_A_SHARE_DATE="${EXPECTED_TARGET_A_SHARE_DATE:-}"

validate_staged_snapshot() {
    if [ -z "$EXPECTED_TARGET_A_SHARE_DATE" ]; then
        return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Validating staged Qlib snapshot for target A-share date ${EXPECTED_TARGET_A_SHARE_DATE}..."
    "$PYTHON_BIN" - <<'PY' "$SYNC_TMP" "$EXPECTED_TARGET_A_SHARE_DATE" "$PROJECT_DIR"
import sys
from pathlib import Path

qlib_dir = Path(sys.argv[1])
expected = sys.argv[2]
project_dir = Path(sys.argv[3])
sys.path.insert(0, str(project_dir))

from scripts.a_share_readiness import validate_staged_qlib_snapshot

completed, latest = validate_staged_qlib_snapshot(
    qlib_dir=qlib_dir,
    expected_target_date=expected,
)
print(f"validated staged snapshot: completed_a_share={completed}, latest_a_share={latest}")
PY
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing Qlib data from ${NAS_USER}@${NAS_HOST}:${NAS_QLIB_PATH}..."

mkdir -p "$QLIB_DIR"

# 原子同步: 先写临时目录，成功后再替换，避免中断导致数据损坏
SYNC_TMP="${QLIB_DIR}.sync_tmp"
rm -rf "$SYNC_TMP"
mkdir -p "$SYNC_TMP"
cleanup() {
    rm -rf "$SYNC_TMP"
}
trap cleanup EXIT

if [ "$REPAIR_QLIB_METADATA" != "false" ] && [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "${NAS_USER}@${NAS_HOST}" \
    "cd ${NAS_QLIB_PATH} && tar cf - calendars instruments features metadata" | \
    tar xf - -C "$SYNC_TMP/"; then
    if [ "$REPAIR_QLIB_METADATA" != "false" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repairing Qlib instruments metadata..."
        "$PYTHON_BIN" "$SCRIPT_DIR/repair_qlib_metadata.py" --qlib-dir "$SYNC_TMP"
    fi

    validate_staged_snapshot

    # 同步成功，原子替换各子目录
    for subdir in calendars instruments features metadata; do
        rm -rf "${QLIB_DIR:?}/$subdir"
        mv "$SYNC_TMP/$subdir" "$QLIB_DIR/$subdir"
    done
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Sync failed, keeping existing data"
    exit 1
fi

trap - EXIT
cleanup

N_DAYS=$(wc -l < "$QLIB_DIR/calendars/day.txt" 2>/dev/null | tr -d ' ' || echo 0)
N_STOCKS=$(ls -d "$QLIB_DIR/features/"* 2>/dev/null | wc -l | tr -d ' ')
SIZE=$(du -sh "$QLIB_DIR" 2>/dev/null | cut -f1)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync complete: ${N_DAYS} days, ${N_STOCKS} stocks, ${SIZE}"
