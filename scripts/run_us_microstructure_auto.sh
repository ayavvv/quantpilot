#!/bin/bash
# QuantPilot US microstructure auto entrypoint for Mac/NAS schedulers.

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST="${US_MICROSTRUCTURE_REMOTE_HOST-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS="${US_MICROSTRUCTURE_REMOTE_HOSTS-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_PROJECT_DIR="${US_MICROSTRUCTURE_REMOTE_PROJECT_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS="${US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS-}"
EXTERNAL_US_MICROSTRUCTURE_FORCE_LOCAL="${US_MICROSTRUCTURE_FORCE_LOCAL-}"
EXTERNAL_US_MICROSTRUCTURE_FORCE_REMOTE="${US_MICROSTRUCTURE_FORCE_REMOTE-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_DRY_RUN="${US_MICROSTRUCTURE_REMOTE_DRY_RUN-}"

REMOTE_ENV_NAMES=(
    DATA_DIR
    PYTHONPATH
    REPORT_DELIVERY_METHOD
    REPORT_TO
    REPORT_FROM
    SMTP_HOST
    SMTP_PORT
    SMTP_USER
    SMTP_PASSWORD
    FUTU_HOST
    FUTU_PORT
    FUTU_RSA_KEY
    QLIB_DATA_DIR
    US_MICROSTRUCTURE_DIR
    US_MICROSTRUCTURE_SYMBOLS
    US_MICROSTRUCTURE_UNIVERSE_FILE
    US_MICROSTRUCTURE_BUILD_UNIVERSE
    US_MICROSTRUCTURE_DYNAMIC_UNIVERSE_FILE
    US_MICROSTRUCTURE_UNIVERSE_TARGET_SIZE
    US_MICROSTRUCTURE_UNIVERSE_HISTORY_POOL_SIZE
    US_MICROSTRUCTURE_UNIVERSE_MINUTE_POOL_SIZE
    US_MICROSTRUCTURE_UNIVERSE_SKIP_DAILY_KLINE
    US_MICROSTRUCTURE_UNIVERSE_SKIP_MINUTE_KLINE
    US_MICROSTRUCTURE_NAS_HOST
    US_MICROSTRUCTURE_NAS_DIR
    US_MICROSTRUCTURE_COLLECT_DURATION_SECONDS
    US_MICROSTRUCTURE_POLL_INTERVAL_SECONDS
    US_MICROSTRUCTURE_BOOK_INTERVAL_SECONDS
    US_MICROSTRUCTURE_QUOTE_INTERVAL_SECONDS
    US_MICROSTRUCTURE_BATCH_SECONDS
    US_MICROSTRUCTURE_BOOK_LEVELS
    US_MICROSTRUCTURE_NO_NAS_SYNC
    US_MICROSTRUCTURE_DATE
    US_MICROSTRUCTURE_REPORT_SYMBOLS
    US_MICROSTRUCTURE_SEND_EMAIL
    US_MICROSTRUCTURE_RUN_VALIDATION
    US_MICROSTRUCTURE_VALIDATION_END
    US_MICROSTRUCTURE_POST_REPORT_VALIDATION
    US_MICROSTRUCTURE_UPDATE_PRICES
    US_MICROSTRUCTURE_REPAIR_UPLOADS
    US_MICROSTRUCTURE_RUN_READINESS
    US_MICROSTRUCTURE_RUN_INTRADAY_REPLAY
    US_MICROSTRUCTURE_PRICE_CSV
    US_MICROSTRUCTURE_PRICE_LOOKBACK_DAYS
)
EXTERNAL_REMOTE_ENV_NAMES=""
for name in "${REMOTE_ENV_NAMES[@]}"; do
    if [ "${!name+x}" = "x" ]; then
        EXTERNAL_REMOTE_ENV_NAMES="$EXTERNAL_REMOTE_ENV_NAMES $name"
    fi
done

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST" ] && US_MICROSTRUCTURE_REMOTE_HOST="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS" ] && US_MICROSTRUCTURE_REMOTE_HOSTS="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_PROJECT_DIR" ] && US_MICROSTRUCTURE_REMOTE_PROJECT_DIR="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_PROJECT_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS" ] && US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_FORCE_LOCAL" ] && US_MICROSTRUCTURE_FORCE_LOCAL="$EXTERNAL_US_MICROSTRUCTURE_FORCE_LOCAL"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_FORCE_REMOTE" ] && US_MICROSTRUCTURE_FORCE_REMOTE="$EXTERNAL_US_MICROSTRUCTURE_FORCE_REMOTE"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_DRY_RUN" ] && US_MICROSTRUCTURE_REMOTE_DRY_RUN="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_DRY_RUN"

TARGET="${1:-report}"
case "$TARGET" in
    collect|report)
        ;;
    *)
        echo "usage: $0 collect|report" >&2
        exit 2
        ;;
esac

LOCAL_SCRIPT="$PROJECT_DIR/scripts/run_us_microstructure_${TARGET}.sh"
REMOTE_SCRIPT_NAME="run_us_microstructure_${TARGET}.sh"
US_MICROSTRUCTURE_REMOTE_PROJECT_DIR="${US_MICROSTRUCTURE_REMOTE_PROJECT_DIR:-/Users/theo/quantpilot}"
US_MICROSTRUCTURE_FORCE_LOCAL="${US_MICROSTRUCTURE_FORCE_LOCAL:-false}"
US_MICROSTRUCTURE_FORCE_REMOTE="${US_MICROSTRUCTURE_FORCE_REMOTE:-false}"
US_MICROSTRUCTURE_REMOTE_DRY_RUN="${US_MICROSTRUCTURE_REMOTE_DRY_RUN:-false}"
US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS="${US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS:-}"

if [ -n "${US_MICROSTRUCTURE_REMOTE_HOSTS-}" ]; then
    REMOTE_HOSTS="$US_MICROSTRUCTURE_REMOTE_HOSTS"
elif [ -n "${US_MICROSTRUCTURE_REMOTE_HOST-}" ]; then
    REMOTE_HOSTS="$US_MICROSTRUCTURE_REMOTE_HOST"
else
    REMOTE_HOSTS="theomac-mini theodeMac-mini-2.local"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_us_microstructure_auto: $*"
}

quote() {
    printf "%q" "$1"
}

is_local_mac() {
    [ "$(uname -s)" = "Darwin" ]
}

run_local() {
    if [ ! -f "$LOCAL_SCRIPT" ]; then
        log "missing local script: $LOCAL_SCRIPT"
        exit 2
    fi
    log "local target=$TARGET script=$LOCAL_SCRIPT"
    if /bin/bash "$LOCAL_SCRIPT"; then
        log "local target=$TARGET done"
        exit 0
    else
        local exit_code=$?
        log "local target=$TARGET failed exit=$exit_code"
        exit "$exit_code"
    fi
}

is_external_remote_env() {
    case " $EXTERNAL_REMOTE_ENV_NAMES " in
        *" $1 "*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

build_remote_command() {
    local remote_script="$US_MICROSTRUCTURE_REMOTE_PROJECT_DIR/scripts/$REMOTE_SCRIPT_NAME"
    local remote_env="US_MICROSTRUCTURE_REMOTE_DISPATCHED=1 US_MICROSTRUCTURE_FORCE_LOCAL=true"
    local name
    for name in "${REMOTE_ENV_NAMES[@]}"; do
        if is_external_remote_env "$name" && [ "${!name+x}" = "x" ]; then
            remote_env="$remote_env $name=$(quote "${!name}")"
        fi
    done
    printf "cd %s && %s /bin/bash %s" \
        "$(quote "$US_MICROSTRUCTURE_REMOTE_PROJECT_DIR")" \
        "$remote_env" \
        "$(quote "$remote_script")"
}

run_remote() {
    local remote_command
    remote_command="$(build_remote_command)"
    local ssh_options=(
        -o BatchMode=yes
        -o ConnectTimeout=10
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=2
    )
    if [ -n "$US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS" ]; then
        read -r -a extra_ssh_options <<< "$US_MICROSTRUCTURE_REMOTE_SSH_OPTIONS"
        ssh_options+=("${extra_ssh_options[@]}")
    fi

    local host
    local last_exit=255
    for host in $REMOTE_HOSTS; do
        log "remote target=$TARGET host=$host project=$US_MICROSTRUCTURE_REMOTE_PROJECT_DIR"
        if [ "$US_MICROSTRUCTURE_REMOTE_DRY_RUN" = "true" ]; then
            printf '%s\n' "ssh ${ssh_options[*]} $host $remote_command"
            return 0
        fi
        if ssh "${ssh_options[@]}" "$host" "$remote_command"; then
            log "remote target=$TARGET host=$host done"
            return 0
        fi
        last_exit=$?
        log "remote target=$TARGET host=$host failed exit=$last_exit"
    done
    return "$last_exit"
}

if [ "$US_MICROSTRUCTURE_FORCE_LOCAL" = "true" ] || [ "${US_MICROSTRUCTURE_REMOTE_DISPATCHED-}" = "1" ]; then
    run_local
fi

if [ "$US_MICROSTRUCTURE_FORCE_REMOTE" != "true" ] && is_local_mac; then
    run_local
fi

remote_exit=0
if run_remote; then
    exit 0
else
    remote_exit=$?
fi

log "all remote attempts failed target=$TARGET exit=$remote_exit"
exit "$remote_exit"
