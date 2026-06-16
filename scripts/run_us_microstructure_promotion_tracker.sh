#!/bin/bash
# Daily US-microstructure promotion tracker.
# Refreshes the forward-validation ledger locally (no NAS sync, so it is fast and
# immune to the main report job being wedged on a stale lock), then prints a concise
# promotion-progress status. Exit code 10 means "alert-worthy" (a side validated,
# went off-track, or is approaching the gate). stdout is meant to be relayed to the user.
set -uo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || exit 1

PY="$PROJECT_DIR/.venv/bin/python"

# Refresh the ledger (best-effort; the tracker still reports the last-known state if this fails).
"$PY" -m scripts.validate_us_microstructure_flow --no-nas-sync >/tmp/us_microstructure_promotion_validate.log 2>&1 || \
    echo "[tracker] validate refresh failed (using last-known ledger); see /tmp/us_microstructure_promotion_validate.log"

"$PY" -m scripts.track_us_microstructure_promotion
exit $?
