#!/bin/bash
# QuantPilot market-wide Futu capital-flow scanner.
#
# Intended cron usage:
#   HK: after Hong Kong close, before the 19:00 daily report.
#   US: after US close in Asia/Shanghai time.
#
# This is separate from run_daily.sh because Futu capital-flow scanning is
# per-symbol and can run for hours on broad HK/US universes.

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
FUTU_MARKET_FLOW_MARKETS="${FUTU_MARKET_FLOW_MARKETS:-HK,US}"
FUTU_MARKET_FLOW_HOST="${FUTU_MARKET_FLOW_HOST:-${FUTU_HOST:-127.0.0.1}}"
FUTU_MARKET_FLOW_PORT="${FUTU_MARKET_FLOW_PORT:-${FUTU_PORT:-11111}}"
FUTU_MARKET_FLOW_CONNECT_TIMEOUT="${FUTU_MARKET_FLOW_CONNECT_TIMEOUT:-8}"
FUTU_MARKET_FLOW_DAYS="${FUTU_MARKET_FLOW_DAYS:-30}"
FUTU_MARKET_FLOW_OUTPUT_DIR="${FUTU_MARKET_FLOW_OUTPUT_DIR:-$DATA_DIR/capital_flow/futu_market}"
FUTU_MARKET_FLOW_PAUSE_SECONDS="${FUTU_MARKET_FLOW_PAUSE_SECONDS:-1.1}"
FUTU_MARKET_FLOW_RATE_LIMIT_DELAY="${FUTU_MARKET_FLOW_RATE_LIMIT_DELAY:-0}"
FUTU_MARKET_FLOW_MIN_REQUEST_INTERVAL="${FUTU_MARKET_FLOW_MIN_REQUEST_INTERVAL:-1.05}"
FUTU_MARKET_FLOW_BATCH_FLUSH="${FUTU_MARKET_FLOW_BATCH_FLUSH:-50}"
FUTU_MARKET_FLOW_MIN_OK_RATIO="${FUTU_MARKET_FLOW_MIN_OK_RATIO:-0}"
FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES="${FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES:-US_PINK,N/A}"
FUTU_MARKET_FLOW_INCLUDE_DISTRIBUTION="${FUTU_MARKET_FLOW_INCLUDE_DISTRIBUTION:-false}"
FUTU_MARKET_FLOW_MAX_CODES="${FUTU_MARKET_FLOW_MAX_CODES:-0}"
FUTU_MARKET_FLOW_CODES="${FUTU_MARKET_FLOW_CODES:-}"
RUN_MAJOR_MONEY_DIGEST_AFTER_SCAN="${RUN_MAJOR_MONEY_DIGEST_AFTER_SCAN:-true}"
MAJOR_MONEY_DIGEST_SOURCES="${MAJOR_MONEY_DIGEST_SOURCES:-auto}"
MAJOR_MONEY_EXPECTED_MARKETS="${MAJOR_MONEY_EXPECTED_MARKETS:-A,HK,US,US_OTC}"
MAJOR_MONEY_DIGEST_JSON="${MAJOR_MONEY_DIGEST_JSON:-$DATA_DIR/output/major_money_digest_latest.json}"
MAJOR_MONEY_DIGEST_CSV="${MAJOR_MONEY_DIGEST_CSV:-$DATA_DIR/output/major_money_digest_latest.csv}"
MAJOR_MONEY_DIGEST_ARCHIVE_DIR="${MAJOR_MONEY_DIGEST_ARCHIVE_DIR:-$DATA_DIR/output/major_money_digest}"
ENABLE_US_OTC_PROXY_FLOW="${ENABLE_US_OTC_PROXY_FLOW:-false}"
US_OTC_PROXY_FLOW_PROVIDER="${US_OTC_PROXY_FLOW_PROVIDER:-polygon}"
US_OTC_PROXY_FLOW_OUTPUT_DIR="${US_OTC_PROXY_FLOW_OUTPUT_DIR:-$DATA_DIR/capital_flow/us_otc_proxy}"
US_OTC_PROXY_FLOW_UNIVERSE_CSV="${US_OTC_PROXY_FLOW_UNIVERSE_CSV:-$DATA_DIR/capital_flow/futu_market/US_latest_source_universe.csv}"
US_OTC_PROXY_FLOW_EXCHANGE_TYPES="${US_OTC_PROXY_FLOW_EXCHANGE_TYPES:-US_PINK}"
US_OTC_PROXY_FLOW_MAX_CODES="${US_OTC_PROXY_FLOW_MAX_CODES:-0}"
US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME="${US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME:-0}"
US_OTC_PROXY_FLOW_DATE="${US_OTC_PROXY_FLOW_DATE:-}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/logs/market_capital_flow_${FUTU_MARKET_FLOW_MARKETS//,/}.lock}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

market_list_contains() {
    local list
    local value
    list=",$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]'),"
    value="$(printf '%s' "$2" | tr '[:lower:]' '[:upper:]')"
    [[ "$list" == *",$value,"* ]]
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "market_capital_flow: skip (lock exists: $LOCK_DIR)"
    exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

SCAN_ARGS=(
    --markets "$FUTU_MARKET_FLOW_MARKETS"
    --host "$FUTU_MARKET_FLOW_HOST"
    --port "$FUTU_MARKET_FLOW_PORT"
    --connect-timeout "$FUTU_MARKET_FLOW_CONNECT_TIMEOUT"
    --days "$FUTU_MARKET_FLOW_DAYS"
    --output-dir "$FUTU_MARKET_FLOW_OUTPUT_DIR"
    --pause-seconds "$FUTU_MARKET_FLOW_PAUSE_SECONDS"
    --rate-limit-delay "$FUTU_MARKET_FLOW_RATE_LIMIT_DELAY"
    --min-request-interval "$FUTU_MARKET_FLOW_MIN_REQUEST_INTERVAL"
    --batch-flush "$FUTU_MARKET_FLOW_BATCH_FLUSH"
    --min-ok-ratio "$FUTU_MARKET_FLOW_MIN_OK_RATIO"
    --exclude-exchange-types "$FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES"
)

if [ "$FUTU_MARKET_FLOW_INCLUDE_DISTRIBUTION" = "true" ]; then
    SCAN_ARGS+=(--include-distribution)
fi
if [ "$FUTU_MARKET_FLOW_MAX_CODES" != "0" ]; then
    SCAN_ARGS+=(--max-codes "$FUTU_MARKET_FLOW_MAX_CODES")
fi
if [ -n "$FUTU_MARKET_FLOW_CODES" ]; then
    SCAN_ARGS+=(--codes "$FUTU_MARKET_FLOW_CODES")
fi

log "market_capital_flow: start markets=$FUTU_MARKET_FLOW_MARKETS output=$FUTU_MARKET_FLOW_OUTPUT_DIR"
source .venv/bin/activate
PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.scan_futu_market_capital_flow "${SCAN_ARGS[@]}"
log "market_capital_flow: scan complete"

US_OTC_PROXY_FLOW_AVAILABLE=false
if [ "$ENABLE_US_OTC_PROXY_FLOW" = "true" ] && market_list_contains "$FUTU_MARKET_FLOW_MARKETS" "US"; then
    log "market_capital_flow: building US OTC/Pink proxy flow"
    if [ "$US_OTC_PROXY_FLOW_PROVIDER" = "polygon" ] && [ -z "${POLYGON_API_KEY:-}" ] && [ -z "${POLYGON_API_KEY_FILE:-}" ]; then
        log "market_capital_flow: WARNING: POLYGON_API_KEY or POLYGON_API_KEY_FILE is not set; US_OTC will remain missing"
    else
        US_OTC_PROXY_ARGS=(
            --provider "$US_OTC_PROXY_FLOW_PROVIDER"
            --universe-csv "$US_OTC_PROXY_FLOW_UNIVERSE_CSV"
            --exchange-types "$US_OTC_PROXY_FLOW_EXCHANGE_TYPES"
            --output-dir "$US_OTC_PROXY_FLOW_OUTPUT_DIR"
            --min-dollar-volume "$US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME"
        )
        if [ -n "$US_OTC_PROXY_FLOW_DATE" ]; then
            US_OTC_PROXY_ARGS+=(--date "$US_OTC_PROXY_FLOW_DATE")
        fi
        if [ "$US_OTC_PROXY_FLOW_MAX_CODES" != "0" ]; then
            US_OTC_PROXY_ARGS+=(--max-codes "$US_OTC_PROXY_FLOW_MAX_CODES")
        fi
        if PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.scan_us_otc_proxy_flow "${US_OTC_PROXY_ARGS[@]}"; then
            US_OTC_PROXY_FLOW_AVAILABLE=true
            log "market_capital_flow: US OTC/Pink proxy flow complete"
        else
            log "market_capital_flow: WARNING: US OTC/Pink proxy flow failed; digest will show US_OTC coverage as missing"
        fi
    fi
fi

if [ "$RUN_MAJOR_MONEY_DIGEST_AFTER_SCAN" = "true" ]; then
    log "market_capital_flow: rebuilding major-money digest"
    MAJOR_MONEY_SOURCE_ARGS=()
    if [ "$MAJOR_MONEY_DIGEST_SOURCES" != "auto" ]; then
        IFS=';' read -r -a MAJOR_MONEY_SOURCE_SPECS <<< "$MAJOR_MONEY_DIGEST_SOURCES"
        for spec in "${MAJOR_MONEY_SOURCE_SPECS[@]}"; do
            if [ -n "$spec" ]; then
                MAJOR_MONEY_SOURCE_ARGS+=(--source "$spec")
            fi
        done
    else
        A_SHARE_FLOW="$DATA_DIR/output/eastmoney_fund_flow_rank_latest.csv"
        if [ -f "$A_SHARE_FLOW" ]; then
            MAJOR_MONEY_SOURCE_ARGS+=(--source "A:$A_SHARE_FLOW:eastmoney")
        fi
        for market in HK US; do
            latest_flow="$FUTU_MARKET_FLOW_OUTPUT_DIR/${market}_latest_flow.csv"
            if [ -f "$latest_flow" ]; then
                MAJOR_MONEY_SOURCE_ARGS+=(--source "$market:$latest_flow:futu")
            fi
        done
        otc_latest_flow="$US_OTC_PROXY_FLOW_OUTPUT_DIR/US_OTC_latest_flow.csv"
        if [ "$US_OTC_PROXY_FLOW_AVAILABLE" = "true" ] && [ -f "$otc_latest_flow" ]; then
            MAJOR_MONEY_SOURCE_ARGS+=(--source "US_OTC:$otc_latest_flow:${US_OTC_PROXY_FLOW_PROVIDER}_otc_proxy")
        fi
    fi
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.build_major_money_digest \
        "${MAJOR_MONEY_SOURCE_ARGS[@]}" \
        --expected-markets "$MAJOR_MONEY_EXPECTED_MARKETS" \
        --output-json "$MAJOR_MONEY_DIGEST_JSON" \
        --output-csv "$MAJOR_MONEY_DIGEST_CSV" \
        --archive-dir "$MAJOR_MONEY_DIGEST_ARCHIVE_DIR"
    log "market_capital_flow: digest complete"
fi

log "market_capital_flow: done"
