#!/bin/bash
# Retry the nightly pipeline once NAS A-share data is finally ready.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TARGET_A_SHARE_DATE="${1:-${TARGET_A_SHARE_DATE_OVERRIDE:-}}"
if [ -z "$TARGET_A_SHARE_DATE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: TARGET_A_SHARE_DATE is required"
    exit 1
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_DIR/.env"
    set +a
fi

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
NAS_HOST="${NAS_HOST:-}"
NAS_USER="${NAS_USER:-}"
NAS_QLIB_PATH="${NAS_QLIB_PATH:-/volume1/docker/quantpilot/qlib_data}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
NAS_COLLECTOR_CONTAINER="${NAS_COLLECTOR_CONTAINER:-quantpilot-collector}"
PRED_PATH="${PRED_PATH:-$DATA_DIR/signals/pred_sh_latest.pkl}"
AUTO_RETRY_MAX_WAIT_SECONDS="${AUTO_RETRY_MAX_WAIT_SECONDS:-54000}"
AUTO_RETRY_POLL_INTERVAL_SECONDS="${AUTO_RETRY_POLL_INTERVAL_SECONDS:-300}"
AUTO_RETRY_TRIGGER_MAX_WAIT_SECONDS="${AUTO_RETRY_TRIGGER_MAX_WAIT_SECONDS:-300}"
AUTO_RETRY_LOG_PATH="${AUTO_RETRY_LOG_PATH:-$PROJECT_DIR/logs/daily_retry.log}"
LOCK_ROOT="${AUTO_RETRY_LOCK_ROOT:-$PROJECT_DIR/logs/nightly_retry_locks}"

mkdir -p "$LOCK_ROOT" "$(dirname "$AUTO_RETRY_LOG_PATH")"
LOCK_DIR="$LOCK_ROOT/$TARGET_A_SHARE_DATE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry:$TARGET_A_SHARE_DATE] $*"
}

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    if [ -f "$LOCK_DIR/pid" ]; then
        local existing_pid
        existing_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
        if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
            log "Retry watcher already running with pid=$existing_pid, exiting"
            return 1
        fi
    fi

    rm -rf "$LOCK_DIR"
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s\n' "$$" > "$LOCK_DIR/pid"
        return 0
    fi

    log "Failed to acquire retry lock at $LOCK_DIR"
    return 1
}

latest_signal_date() {
    if [ ! -x "$PYTHON_BIN" ]; then
        return 0
    fi
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness pred-latest-signal-date \
        --pred-path "$PRED_PATH" 2>/dev/null || true
}

nas_completed_date() {
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-completed-date \
        --nas-host "$NAS_HOST" \
        --nas-user "$NAS_USER" \
        --ssh-key "$SSH_KEY" \
        --nas-qlib-path "$NAS_QLIB_PATH"
}

nas_latest_date() {
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-latest-date \
        --nas-host "$NAS_HOST" \
        --nas-user "$NAS_USER" \
        --ssh-key "$SSH_KEY" \
        --nas-qlib-path "$NAS_QLIB_PATH"
}

nightly_running() {
    pgrep -f "$SCRIPT_DIR/run_daily.sh|python -m inference.run_daily" >/dev/null 2>&1
}

if [ -z "$NAS_HOST" ] || [ -z "$NAS_USER" ]; then
    log "NAS_HOST/NAS_USER missing, cannot watch NAS readiness"
    exit 1
fi

if ! acquire_lock; then
    exit 0
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

deadline_epoch=$(( $(date +%s) + AUTO_RETRY_MAX_WAIT_SECONDS ))
log "Watching NAS readiness for target=$TARGET_A_SHARE_DATE (deadline in ${AUTO_RETRY_MAX_WAIT_SECONDS}s)"

while [ "$(date +%s)" -lt "$deadline_epoch" ]; do
    local_signal_date="$(latest_signal_date)"
    if [ -n "$local_signal_date" ] && [ "$local_signal_date" \> "$TARGET_A_SHARE_DATE" -o "$local_signal_date" = "$TARGET_A_SHARE_DATE" ]; then
        log "Local signal already up to date (signal_date=$local_signal_date), exiting"
        exit 0
    fi

    if nightly_running; then
        log "Nightly pipeline already running, sleeping"
        sleep "$AUTO_RETRY_POLL_INTERVAL_SECONDS"
        continue
    fi

    nas_last="$(nas_completed_date)"
    nas_latest="$(nas_latest_date)"
    effective_nas_date="$nas_last"
    if [ -z "$effective_nas_date" ] || { [ -n "$nas_latest" ] && [ "$nas_latest" \> "$effective_nas_date" ]; }; then
        effective_nas_date="$nas_latest"
    fi
    if [ -n "$effective_nas_date" ] && [ "$effective_nas_date" \> "$TARGET_A_SHARE_DATE" -o "$effective_nas_date" = "$TARGET_A_SHARE_DATE" ]; then
        log "NAS ready (completed_a_share=${nas_last:-N/A}, latest_a_share=${nas_latest:-N/A}), triggering retry run_daily.sh"
        TARGET_A_SHARE_DATE_OVERRIDE="$TARGET_A_SHARE_DATE" \
        AUTO_RETRY_ON_NAS_READY="false" \
        MAX_WAIT_SECONDS="$AUTO_RETRY_TRIGGER_MAX_WAIT_SECONDS" \
        WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}" \
            "$SCRIPT_DIR/run_daily.sh" >> "$PROJECT_DIR/logs/daily.log" 2>&1
        rc=$?
        log "Retry run_daily.sh finished with rc=$rc"
        exit "$rc"
    fi

    log "NAS not ready yet (completed_a_share=${nas_last:-N/A}, latest_a_share=${nas_latest:-N/A}), sleeping ${AUTO_RETRY_POLL_INTERVAL_SECONDS}s"
    sleep "$AUTO_RETRY_POLL_INTERVAL_SECONDS"
done

log "Deadline reached without NAS readiness; giving up"
exit 1
