#!/bin/bash
# Keep the Polymarket scheduler alive when started from cron @reboot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="$LOG_DIR/polymarket_scheduler_guard.lock"
PID_FILE="$LOG_DIR/polymarket_scheduler_guard.pid"
RESTART_DELAY_SECONDS="${POLY_SCHEDULER_RESTART_DELAY_SECONDS:-10}"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -f "$PID_FILE" ]; then
        existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
            exit 0
        fi
    fi
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi

cleanup() {
    rm -f "$PID_FILE"
    rm -rf "$LOCK_DIR"
}

stop() {
    cleanup
    exit 0
}

trap cleanup EXIT
trap stop INT TERM

echo "$$" > "$PID_FILE"

while true; do
    if pgrep -f "[p]ython.*-m polymarket.scheduler" >/dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S %z') polymarket scheduler already running; guard exiting"
        exit 0
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S %z') starting polymarket scheduler"
    set +e
    "$PROJECT_DIR/scripts/run_polymarket_scheduler.sh"
    status=$?
    set -e
    echo "$(date '+%Y-%m-%d %H:%M:%S %z') polymarket scheduler exited status=$status; restarting in ${RESTART_DELAY_SECONDS}s"
    sleep "$RESTART_DELAY_SECONDS"
done
