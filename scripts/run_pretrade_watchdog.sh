#!/bin/bash
# Ensure signal freshness before the 14:50 trade window.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

run_healthcheck() {
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.daily_healthcheck \
        --phase pretrade \
        --alert-on error || true
}

WATCHDOG_RC=0
if PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.pretrade_watchdog; then
    WATCHDOG_RC=0
else
    WATCHDOG_RC=$?
fi

run_healthcheck
exit "$WATCHDOG_RC"
