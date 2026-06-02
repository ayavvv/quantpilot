#!/bin/bash
# Morning/evening watchdog for US microstructure automation.

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXTERNAL_DATA_DIR="${DATA_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_DIR="${US_MICROSTRUCTURE_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_DATE="${US_MICROSTRUCTURE_DATE-}"
EXTERNAL_US_MICROSTRUCTURE_NAS_HOST="${US_MICROSTRUCTURE_NAS_HOST-}"
EXTERNAL_US_MICROSTRUCTURE_NAS_DIR="${US_MICROSTRUCTURE_NAS_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR="${US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR-}"
EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_DRY_RUN="${US_MICROSTRUCTURE_WATCHDOG_DRY_RUN-}"
EXTERNAL_QLIB_DATA_DIR="${QLIB_DATA_DIR-}"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

[ -n "$EXTERNAL_DATA_DIR" ] && DATA_DIR="$EXTERNAL_DATA_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_DIR" ] && US_MICROSTRUCTURE_DIR="$EXTERNAL_US_MICROSTRUCTURE_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_DATE" ] && US_MICROSTRUCTURE_DATE="$EXTERNAL_US_MICROSTRUCTURE_DATE"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_NAS_HOST" ] && US_MICROSTRUCTURE_NAS_HOST="$EXTERNAL_US_MICROSTRUCTURE_NAS_HOST"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_NAS_DIR" ] && US_MICROSTRUCTURE_NAS_DIR="$EXTERNAL_US_MICROSTRUCTURE_NAS_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR" ] && US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR="$EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_DRY_RUN" ] && US_MICROSTRUCTURE_WATCHDOG_DRY_RUN="$EXTERNAL_US_MICROSTRUCTURE_WATCHDOG_DRY_RUN"
[ -n "$EXTERNAL_QLIB_DATA_DIR" ] && QLIB_DATA_DIR="$EXTERNAL_QLIB_DATA_DIR"

MODE="${1:-auto}"
case "$MODE" in
    auto|morning|evening|manual)
        ;;
    *)
        echo "usage: $0 [auto|morning|evening|manual]" >&2
        exit 2
        ;;
esac

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
US_MICROSTRUCTURE_DIR="${US_MICROSTRUCTURE_DIR:-$DATA_DIR/us_microstructure}"
US_MICROSTRUCTURE_DATE="${US_MICROSTRUCTURE_DATE:-}"
US_MICROSTRUCTURE_NAS_HOST="${US_MICROSTRUCTURE_NAS_HOST:-nas}"
US_MICROSTRUCTURE_NAS_DIR="${US_MICROSTRUCTURE_NAS_DIR:-/volume1/docker/quantpilot/us_microstructure}"
US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR="${US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR:-true}"
US_MICROSTRUCTURE_WATCHDOG_DRY_RUN="${US_MICROSTRUCTURE_WATCHDOG_DRY_RUN:-false}"
QLIB_DATA_DIR="${QLIB_DATA_DIR:-$DATA_DIR/qlib_data}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/logs/us_microstructure_watchdog.lock}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_us_microstructure_watchdog: $*"
}

mkdir -p "$PROJECT_DIR/logs"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "skip (lock exists: $LOCK_DIR)"
    exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
    log "python not found"
    exit 2
fi

if [ -z "$US_MICROSTRUCTURE_DATE" ]; then
    US_MICROSTRUCTURE_DATE="$(PYTHONPATH="$PYTHONPATH" DATA_DIR="$DATA_DIR" "$PYTHON_BIN" -m scripts.us_microstructure_dates default-report-date --base-dir "$US_MICROSTRUCTURE_DIR")"
fi

READINESS_PATH="$PROJECT_DIR/logs/us_microstructure_watchdog_readiness_latest.json"

run_recover() {
    log "recover mode=$MODE"
    if [ "$US_MICROSTRUCTURE_WATCHDOG_DRY_RUN" = "true" ]; then
        US_MICROSTRUCTURE_RECOVER_DRY_RUN=true /bin/bash "$PROJECT_DIR/scripts/run_us_microstructure_recover.sh"
    else
        /bin/bash "$PROJECT_DIR/scripts/run_us_microstructure_recover.sh"
    fi
}

run_readiness() {
    local output_path="$1"
    local exit_code=0
    set +e
    PYTHONPATH="$PYTHONPATH" DATA_DIR="$DATA_DIR" QLIB_DATA_DIR="$QLIB_DATA_DIR" "$PYTHON_BIN" -m scripts.us_microstructure_readiness \
        --base-dir "$US_MICROSTRUCTURE_DIR" \
        --date "$US_MICROSTRUCTURE_DATE" \
        --nas-host "$US_MICROSTRUCTURE_NAS_HOST" \
        --nas-dir "$US_MICROSTRUCTURE_NAS_DIR" \
        --all-manifests > "$output_path"
    exit_code=$?
    set -e
    return "$exit_code"
}

print_readiness_summary() {
    local input_path="$1"
    "$PYTHON_BIN" - <<'PY' "$input_path"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = payload.get("checks", {})
summary = {
    "ok": payload.get("ok"),
    "date": payload.get("date"),
    "high_confidence_ready": payload.get("high_confidence_ready"),
    "issues": payload.get("issues", []),
    "manifest_ok": checks.get("manifest_full_session", {}).get("ok"),
    "report_ok": checks.get("report", {}).get("ok"),
    "launchd_ok": checks.get("launchd", {}).get("ok"),
}
print(json.dumps(summary, ensure_ascii=False))
PY
}

write_readiness_archive() {
    if [ "$US_MICROSTRUCTURE_WATCHDOG_DRY_RUN" = "true" ]; then
        return 0
    fi
    PYTHONPATH="$PYTHONPATH" DATA_DIR="$DATA_DIR" QLIB_DATA_DIR="$QLIB_DATA_DIR" "$PYTHON_BIN" -m scripts.us_microstructure_readiness \
        --base-dir "$US_MICROSTRUCTURE_DIR" \
        --date "$US_MICROSTRUCTURE_DATE" \
        --nas-host "$US_MICROSTRUCTURE_NAS_HOST" \
        --nas-dir "$US_MICROSTRUCTURE_NAS_DIR" \
        --all-manifests \
        --write-json >/dev/null
}

repair_known_issues() {
    if [ "$US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR" != "true" ]; then
        log "auto repair disabled"
        return 0
    fi
    if [ "$US_MICROSTRUCTURE_WATCHDOG_DRY_RUN" = "true" ]; then
        log "dry-run repair: recover + repair uploads"
        return 0
    fi

    log "repair: recover"
    /bin/bash "$PROJECT_DIR/scripts/run_us_microstructure_recover.sh" || true

    log "repair: NAS uploads date=$US_MICROSTRUCTURE_DATE"
    PYTHONPATH="$PYTHONPATH" DATA_DIR="$DATA_DIR" "$PYTHON_BIN" -m scripts.repair_us_microstructure_nas_uploads \
        --base-dir "$US_MICROSTRUCTURE_DIR" \
        --date "$US_MICROSTRUCTURE_DATE" \
        --nas-host "$US_MICROSTRUCTURE_NAS_HOST" \
        --nas-dir "$US_MICROSTRUCTURE_NAS_DIR" || true
}

log "start mode=$MODE date=$US_MICROSTRUCTURE_DATE"
run_recover

readiness_exit=0
run_readiness "$READINESS_PATH" || readiness_exit=$?
log "readiness $(print_readiness_summary "$READINESS_PATH") exit=$readiness_exit"

if [ "$readiness_exit" -eq 0 ]; then
    write_readiness_archive || true
    log "done ok"
    exit 0
fi

repair_known_issues

second_exit=0
run_readiness "$READINESS_PATH" || second_exit=$?
log "readiness_after_repair $(print_readiness_summary "$READINESS_PATH") exit=$second_exit"
write_readiness_archive || true

if [ "$second_exit" -ne 0 ]; then
    log "done with unresolved issues"
    exit "$second_exit"
fi

log "done repaired"
