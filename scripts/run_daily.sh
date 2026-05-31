#!/bin/bash
# QuantPilot Daily Pipeline
# Schedule: cron 19:00 Mon-Fri (after NAS daily collection flush)
#
# Steps:
# 0. Wait for NAS collector to finish today's data
# 1. Sync Qlib bin data from NAS (or skip if single-machine)
# 2. Run inference natively in venv (validate data -> LightGBM predict)
# 3. Run reporter natively (generate + send daily report)

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

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
DOCKER="${DOCKER:-docker}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
ALLOW_STALE_SYNC="${ALLOW_STALE_SYNC:-false}"
AUTO_RETRY_ON_NAS_READY="${AUTO_RETRY_ON_NAS_READY:-true}"
AUTO_RETRY_LOG_PATH="${AUTO_RETRY_LOG_PATH:-$PROJECT_DIR/logs/daily_retry.log}"
NAS_COLLECTOR_CONTAINER="${NAS_COLLECTOR_CONTAINER:-quantpilot-collector}"
TARGET_DATE_LOOKBACK_DAYS="${TARGET_DATE_LOOKBACK_DAYS:-31}"
TARGET_A_SHARE_DATE_OVERRIDE="${TARGET_A_SHARE_DATE_OVERRIDE:-}"
LOCAL_A_SHARE_RESCUE="${LOCAL_A_SHARE_RESCUE:-true}"
LOCAL_A_SHARE_RESCUE_RATE_LIMIT="${LOCAL_A_SHARE_RESCUE_RATE_LIMIT:-0.03}"
LOCAL_A_SHARE_RESCUE_SOCKET_TIMEOUT="${LOCAL_A_SHARE_RESCUE_SOCKET_TIMEOUT:-20}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
SKIP_NAS_SYNC="false"
ENABLE_A_SHARE_CAPITAL_FLOW_OVERLAY="${ENABLE_A_SHARE_CAPITAL_FLOW_OVERLAY:-true}"
A_SHARE_CAPITAL_FLOW_TOP_N="${A_SHARE_CAPITAL_FLOW_TOP_N:-30}"
A_SHARE_CAPITAL_FLOW_DAYS="${A_SHARE_CAPITAL_FLOW_DAYS:-30}"
A_SHARE_CAPITAL_FLOW_HOST="${A_SHARE_CAPITAL_FLOW_HOST:-${FUTU_HOST:-127.0.0.1}}"
A_SHARE_CAPITAL_FLOW_PORT="${A_SHARE_CAPITAL_FLOW_PORT:-${FUTU_PORT:-11111}}"
A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR="${A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR:-$DATA_DIR/capital_flow/futu}"
A_SHARE_CAPITAL_FLOW_CONNECT_TIMEOUT="${A_SHARE_CAPITAL_FLOW_CONNECT_TIMEOUT:-8}"
ENABLE_A_SHARE_CAPITAL_FLOW_EVAL="${ENABLE_A_SHARE_CAPITAL_FLOW_EVAL:-true}"
A_SHARE_CAPITAL_FLOW_EVAL_HORIZONS="${A_SHARE_CAPITAL_FLOW_EVAL_HORIZONS:-1,3,5}"
A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR="${A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR:-$DATA_DIR/output/futu_capital_flow_eval_latest}"
A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT="${A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT:-20}"
A_SHARE_CAPITAL_FLOW_GATE_MIN_CONFIRMING_HORIZONS="${A_SHARE_CAPITAL_FLOW_GATE_MIN_CONFIRMING_HORIZONS:-2}"
A_SHARE_CAPITAL_FLOW_GATE_RISK_ALPHA_THRESHOLD="${A_SHARE_CAPITAL_FLOW_GATE_RISK_ALPHA_THRESHOLD:--0.005}"
A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_ALPHA_THRESHOLD="${A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_ALPHA_THRESHOLD:-0.005}"
A_SHARE_CAPITAL_FLOW_GATE_RISK_MAX_HIT_RATE="${A_SHARE_CAPITAL_FLOW_GATE_RISK_MAX_HIT_RATE:-0.45}"
A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_MIN_HIT_RATE="${A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_MIN_HIT_RATE:-0.55}"
ENABLE_MAJOR_MONEY_DIGEST="${ENABLE_MAJOR_MONEY_DIGEST:-true}"
MAJOR_MONEY_DIGEST_SOURCES="${MAJOR_MONEY_DIGEST_SOURCES:-auto}"
MAJOR_MONEY_EXPECTED_MARKETS="${MAJOR_MONEY_EXPECTED_MARKETS:-A,HK,US,US_OTC}"
MAJOR_MONEY_DIGEST_JSON="${MAJOR_MONEY_DIGEST_JSON:-$DATA_DIR/output/major_money_digest_latest.json}"
MAJOR_MONEY_DIGEST_CSV="${MAJOR_MONEY_DIGEST_CSV:-$DATA_DIR/output/major_money_digest_latest.csv}"
ENABLE_US_OTC_PROXY_FLOW="${ENABLE_US_OTC_PROXY_FLOW:-false}"
US_OTC_PROXY_FLOW_PROVIDER="${US_OTC_PROXY_FLOW_PROVIDER:-polygon}"
US_OTC_PROXY_FLOW_OUTPUT_DIR="${US_OTC_PROXY_FLOW_OUTPUT_DIR:-$DATA_DIR/capital_flow/us_otc_proxy}"
US_OTC_PROXY_FLOW_UNIVERSE_CSV="${US_OTC_PROXY_FLOW_UNIVERSE_CSV:-$DATA_DIR/capital_flow/futu_market/US_latest_source_universe.csv}"
US_OTC_PROXY_FLOW_EXCHANGE_TYPES="${US_OTC_PROXY_FLOW_EXCHANGE_TYPES:-US_PINK}"
US_OTC_PROXY_FLOW_MAX_CODES="${US_OTC_PROXY_FLOW_MAX_CODES:-0}"
US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME="${US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME:-0}"
US_OTC_PROXY_FLOW_DATE="${US_OTC_PROXY_FLOW_DATE:-}"
ENABLE_EASTMONEY_FUND_FLOW_REFRESH="${ENABLE_EASTMONEY_FUND_FLOW_REFRESH:-true}"
EASTMONEY_FUND_FLOW_RANK_OUTPUT="${EASTMONEY_FUND_FLOW_RANK_OUTPUT:-$DATA_DIR/output/eastmoney_fund_flow_rank_latest.csv}"
EASTMONEY_FUND_FLOW_ARCHIVE_DIR="${EASTMONEY_FUND_FLOW_ARCHIVE_DIR:-$DATA_DIR/fund_flow/eastmoney}"
EASTMONEY_FUND_FLOW_LIMIT="${EASTMONEY_FUND_FLOW_LIMIT:-6000}"
EASTMONEY_FUND_FLOW_MIN_ROWS="${EASTMONEY_FUND_FLOW_MIN_ROWS:-1000}"
EASTMONEY_FUND_FLOW_SOURCE="${EASTMONEY_FUND_FLOW_SOURCE:-auto}"
EASTMONEY_FUND_FLOW_TIMEOUT="${EASTMONEY_FUND_FLOW_TIMEOUT:-10}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_healthcheck() {
    local phase="${1:-nightly}"
    local alert_on="${2:-error}"
    local target_args=()
    if [ -n "${TARGET_A_SHARE_DATE:-}" ]; then
        target_args+=(--target-a-share-date "$TARGET_A_SHARE_DATE")
    fi
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.daily_healthcheck \
        --phase "$phase" \
        --alert-on "$alert_on" \
        "${target_args[@]}" || true
}

resolve_signal_output_tag() {
    local date_value="${1:-}"
    if [ -z "$date_value" ]; then
        return 0
    fi
    echo "${date_value//-/}"
}

spawn_ready_retry() {
    local target_date="$1"
    if [ "$AUTO_RETRY_ON_NAS_READY" != "true" ] || [ -z "$target_date" ]; then
        return 0
    fi

    if [ ! -x "$SCRIPT_DIR/run_daily_when_ready.sh" ]; then
        log "  WARNING: retry watcher script missing, skip auto retry"
        return 0
    fi

    mkdir -p "$(dirname "$AUTO_RETRY_LOG_PATH")"
    nohup "$SCRIPT_DIR/run_daily_when_ready.sh" "$target_date" >> "$AUTO_RETRY_LOG_PATH" 2>&1 </dev/null &
    log "  Auto retry watcher started for target=$target_date (pid=$!)"
}

# Step 0: Wait for NAS collector to finish today's data
if [ -n "$NAS_HOST" ] && [ -n "$NAS_USER" ]; then
    log "Step 0: Waiting for NAS data to be ready..."
    if [ -n "$TARGET_A_SHARE_DATE_OVERRIDE" ]; then
        TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE_OVERRIDE"
        log "  Target A-share trading date override: $TARGET_A_SHARE_DATE"
    else
        TODAY=$(date +%Y-%m-%d)
        TARGET_A_SHARE_DATE=$(
            PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-target-date \
                --nas-host "$NAS_HOST" \
                --nas-user "$NAS_USER" \
                --ssh-key "$SSH_KEY" \
                --today "$TODAY" \
                --collector-container "$NAS_COLLECTOR_CONTAINER" \
                --lookback-days "$TARGET_DATE_LOOKBACK_DAYS"
        )
    fi
    if [ -z "$TARGET_A_SHARE_DATE" ]; then
        log "  ERROR: failed to resolve target A-share trading date"
        exit 1
    fi
    log "  Target A-share trading date: $TARGET_A_SHARE_DATE"
    WAITED=0
    NAS_LAST=""
    NAS_LATEST=""
    SYNC_TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE"
    while [ $WAITED -lt $MAX_WAIT_SECONDS ]; do
        NAS_LAST=$(
            PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-completed-date \
                --nas-host "$NAS_HOST" \
                --nas-user "$NAS_USER" \
                --ssh-key "$SSH_KEY" \
                --nas-qlib-path "$NAS_QLIB_PATH"
        )
        NAS_LATEST=$(
            PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.a_share_readiness nas-latest-date \
                --nas-host "$NAS_HOST" \
                --nas-user "$NAS_USER" \
                --ssh-key "$SSH_KEY" \
                --nas-qlib-path "$NAS_QLIB_PATH"
        )
        if [ -n "$NAS_LAST" ] && [ "$NAS_LAST" \> "$TARGET_A_SHARE_DATE" -o "$NAS_LAST" = "$TARGET_A_SHARE_DATE" ]; then
            log "  NAS A-share data ready via completion metadata (completed_a_share=$NAS_LAST)"
            break
        fi
        if [ -n "$NAS_LATEST" ] && [ "$NAS_LATEST" \> "$TARGET_A_SHARE_DATE" -o "$NAS_LATEST" = "$TARGET_A_SHARE_DATE" ]; then
            log "  NAS A-share data ready via instruments snapshot (latest_a_share=$NAS_LATEST, completed_a_share=${NAS_LAST:-N/A})"
            break
        fi
        log "  NAS completed_a_share=${NAS_LAST:-N/A}, latest_a_share=${NAS_LATEST:-N/A}, waiting for $TARGET_A_SHARE_DATE... (${WAITED}s/${MAX_WAIT_SECONDS}s)"
        sleep $WAIT_INTERVAL_SECONDS
        WAITED=$((WAITED + WAIT_INTERVAL_SECONDS))
    done
    EFFECTIVE_NAS_DATE="$NAS_LAST"
    if [ -z "$EFFECTIVE_NAS_DATE" ] || { [ -n "$NAS_LATEST" ] && [ "$NAS_LATEST" \> "$EFFECTIVE_NAS_DATE" ]; }; then
        EFFECTIVE_NAS_DATE="$NAS_LATEST"
    fi
    if [ -z "$EFFECTIVE_NAS_DATE" ] || [ "$EFFECTIVE_NAS_DATE" \< "$TARGET_A_SHARE_DATE" ]; then
        if [ "$ALLOW_STALE_SYNC" = "true" ]; then
            log "  WARNING: NAS A-share data not ready after ${MAX_WAIT_SECONDS}s, proceeding with available data (${EFFECTIVE_NAS_DATE:-N/A})"
            if [ -n "$EFFECTIVE_NAS_DATE" ]; then
                SYNC_TARGET_A_SHARE_DATE="$EFFECTIVE_NAS_DATE"
            else
                SYNC_TARGET_A_SHARE_DATE=""
            fi
        else
            if [ "$LOCAL_A_SHARE_RESCUE" = "true" ]; then
                log "  WARNING: NAS A-share data not ready after ${MAX_WAIT_SECONDS}s; running local Baostock rescue for $TARGET_A_SHARE_DATE"
                PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.backfill_a_share_baostock \
                    --qlib-dir "$DATA_DIR/qlib_data" \
                    --target-date "$TARGET_A_SHARE_DATE" \
                    --rate-limit "$LOCAL_A_SHARE_RESCUE_RATE_LIMIT" \
                    --socket-timeout "$LOCAL_A_SHARE_RESCUE_SOCKET_TIMEOUT"
                SYNC_TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE"
                SKIP_NAS_SYNC="true"
            else
                spawn_ready_retry "$TARGET_A_SHARE_DATE"
                run_healthcheck nightly error
                log "  ERROR: NAS A-share data not ready after ${MAX_WAIT_SECONDS}s, aborting to avoid stale/inconsistent sync (completed=${NAS_LAST:-N/A}, latest=${NAS_LATEST:-N/A})"
                exit 1
            fi
        fi
    fi
fi

# Step 1: Sync Qlib data from NAS (if NAS_HOST is configured)
if [ -n "$NAS_HOST" ] && [ "$SKIP_NAS_SYNC" != "true" ]; then
    log "Step 1: Syncing Qlib data from NAS..."
    EXPECTED_TARGET_A_SHARE_DATE="${SYNC_TARGET_A_SHARE_DATE:-}" "$SCRIPT_DIR/sync_data.sh"
    log "  Sync complete"
elif [ "$SKIP_NAS_SYNC" = "true" ]; then
    log "Step 1: Skipped NAS sync (local Baostock rescue already updated local Qlib)"
else
    log "Step 1: Skipped (NAS_HOST not configured, single-machine mode)"
fi

# Step 2: Run inference natively (no Docker, avoids Rosetta OOM)
log "Step 2: Running inference..."
source .venv/bin/activate
INFERENCE_RC=0
SIGNAL_OUTPUT_TAG_VALUE="${SIGNAL_OUTPUT_TAG_OVERRIDE:-}"
if [ -z "$SIGNAL_OUTPUT_TAG_VALUE" ]; then
    SIGNAL_OUTPUT_TAG_VALUE="$(resolve_signal_output_tag "${SYNC_TARGET_A_SHARE_DATE:-${TARGET_A_SHARE_DATE:-}}")"
fi
if QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
    MODEL_DIR="$DATA_DIR/models" \
    SIGNAL_DIR="$DATA_DIR/signals" \
    SIGNAL_OUTPUT_TAG="$SIGNAL_OUTPUT_TAG_VALUE" \
    python -m inference.run_daily; then
    INFERENCE_RC=0
else
    INFERENCE_RC=$?
fi
if [ "$INFERENCE_RC" -ne 0 ]; then
    run_healthcheck nightly error
    exit "$INFERENCE_RC"
fi
log "  Inference complete"

# Step 2b: Persist Futu capital-flow overlay for forward validation.
if [ "$ENABLE_A_SHARE_CAPITAL_FLOW_OVERLAY" = "true" ]; then
    log "Step 2b: Building Futu capital-flow overlay..."
    if PYTHONPATH="$PYTHONPATH" \
        "$PYTHON_BIN" -m scripts.build_futu_capital_flow_overlay \
            --signal-csv "$DATA_DIR/signals/signal_latest.csv" \
            --fetch-latest \
            --host "$A_SHARE_CAPITAL_FLOW_HOST" \
            --port "$A_SHARE_CAPITAL_FLOW_PORT" \
            --days "$A_SHARE_CAPITAL_FLOW_DAYS" \
            --include-distribution \
            --connect-timeout "$A_SHARE_CAPITAL_FLOW_CONNECT_TIMEOUT" \
            --signal-top-n "$A_SHARE_CAPITAL_FLOW_TOP_N" \
            --archive \
            --archive-dir "$A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR" \
            --output "$DATA_DIR/output/futu_capital_flow_signal_overlay_latest.csv" \
            --flow-output "$DATA_DIR/output/futu_capital_flow_latest.csv"; then
        log "  Futu capital-flow overlay complete"
    else
        log "  WARNING: Futu capital-flow overlay failed; continuing daily pipeline"
    fi
else
    log "Step 2b: Skipped Futu capital-flow overlay"
fi

# Step 2c: Re-evaluate archived Futu capital-flow overlays as future closes arrive.
if [ "$ENABLE_A_SHARE_CAPITAL_FLOW_EVAL" = "true" ]; then
    log "Step 2c: Evaluating Futu capital-flow forward returns..."
    if PYTHONPATH="$PYTHONPATH" \
        "$PYTHON_BIN" -m scripts.evaluate_futu_capital_flow_overlay \
            --qlib-dir "$DATA_DIR/qlib_data" \
            --archive-dir "$A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR" \
            --horizons "$A_SHARE_CAPITAL_FLOW_EVAL_HORIZONS" \
            --output-dir "$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR" \
            --gate-min-date-count "$A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT" \
            --gate-min-confirming-horizons "$A_SHARE_CAPITAL_FLOW_GATE_MIN_CONFIRMING_HORIZONS" \
            --gate-risk-alpha-threshold "$A_SHARE_CAPITAL_FLOW_GATE_RISK_ALPHA_THRESHOLD" \
            --gate-confirm-alpha-threshold "$A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_ALPHA_THRESHOLD" \
            --gate-risk-max-hit-rate "$A_SHARE_CAPITAL_FLOW_GATE_RISK_MAX_HIT_RATE" \
            --gate-confirm-min-hit-rate "$A_SHARE_CAPITAL_FLOW_GATE_CONFIRM_MIN_HIT_RATE"; then
        log "  Futu capital-flow evaluation complete"
    else
        log "  WARNING: Futu capital-flow evaluation failed; continuing daily pipeline"
    fi
else
    log "Step 2c: Skipped Futu capital-flow evaluation"
fi

# Step 2d: Build market-wide major-money digest for the email report.
if [ "$ENABLE_MAJOR_MONEY_DIGEST" = "true" ]; then
    log "Step 2d: Building market-wide major-money digest..."
    if [ "$ENABLE_EASTMONEY_FUND_FLOW_REFRESH" = "true" ]; then
        log "  Refreshing A-share Eastmoney fund-flow rank..."
        if PYTHONPATH="$PYTHONPATH" \
            "$PYTHON_BIN" -m scripts.refresh_eastmoney_fund_flow_rank \
                --output "$EASTMONEY_FUND_FLOW_RANK_OUTPUT" \
                --archive-dir "$EASTMONEY_FUND_FLOW_ARCHIVE_DIR" \
                --limit "$EASTMONEY_FUND_FLOW_LIMIT" \
                --min-rows "$EASTMONEY_FUND_FLOW_MIN_ROWS" \
                --source "$EASTMONEY_FUND_FLOW_SOURCE" \
                --timeout "$EASTMONEY_FUND_FLOW_TIMEOUT"; then
            log "  Eastmoney fund-flow refresh complete"
        else
            log "  WARNING: Eastmoney fund-flow refresh failed; digest will use any existing rank artifact"
        fi
    fi
    US_OTC_PROXY_FLOW_AVAILABLE=false
    if [ "$ENABLE_US_OTC_PROXY_FLOW" = "true" ]; then
        log "  Building US OTC/Pink proxy flow..."
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
            log "  US OTC/Pink proxy flow complete"
        else
            log "  WARNING: US OTC/Pink proxy flow failed; digest will show US_OTC coverage as missing"
        fi
    fi
    MAJOR_MONEY_SOURCE_ARGS=()
    if [ "$MAJOR_MONEY_DIGEST_SOURCES" != "auto" ]; then
        IFS=';' read -r -a MAJOR_MONEY_SOURCE_SPECS <<< "$MAJOR_MONEY_DIGEST_SOURCES"
        for spec in "${MAJOR_MONEY_SOURCE_SPECS[@]}"; do
            if [ -n "$spec" ]; then
                MAJOR_MONEY_SOURCE_ARGS+=(--source "$spec")
            fi
        done
    else
        if [ -f "$EASTMONEY_FUND_FLOW_RANK_OUTPUT" ]; then
            MAJOR_MONEY_SOURCE_ARGS+=(--source "A:$EASTMONEY_FUND_FLOW_RANK_OUTPUT:eastmoney")
        fi
        for market in HK US; do
            latest_flow="$DATA_DIR/capital_flow/futu_market/${market}_latest_flow.csv"
            if [ -f "$latest_flow" ]; then
                MAJOR_MONEY_SOURCE_ARGS+=(--source "$market:$latest_flow:futu")
            fi
        done
        otc_latest_flow="$US_OTC_PROXY_FLOW_OUTPUT_DIR/US_OTC_latest_flow.csv"
        if [ "$US_OTC_PROXY_FLOW_AVAILABLE" = "true" ] && [ -f "$otc_latest_flow" ]; then
            MAJOR_MONEY_SOURCE_ARGS+=(--source "US_OTC:$otc_latest_flow:${US_OTC_PROXY_FLOW_PROVIDER}_otc_proxy")
        fi
    fi
    if PYTHONPATH="$PYTHONPATH" \
        "$PYTHON_BIN" -m scripts.build_major_money_digest \
            "${MAJOR_MONEY_SOURCE_ARGS[@]}" \
            --expected-markets "$MAJOR_MONEY_EXPECTED_MARKETS" \
            --output-json "$MAJOR_MONEY_DIGEST_JSON" \
            --output-csv "$MAJOR_MONEY_DIGEST_CSV"; then
        log "  Market-wide major-money digest complete"
    else
        log "  WARNING: market-wide major-money digest failed; continuing daily pipeline"
    fi
else
    log "Step 2d: Skipped market-wide major-money digest"
fi

# Step 3: Run reporter natively so Mail.app fallback is available on macOS host
log "Step 3: Running reporter..."
log "  Working directory: $PROJECT_DIR"
log "  Reporter env file: $PROJECT_DIR/reporter/.env"
REPORTER_RC=0
if REPORTER_ENV_FILE="$PROJECT_DIR/reporter/.env" \
    REPORT_DIR="$DATA_DIR/reports" \
    SIGNAL_DIR="$DATA_DIR/signals" \
    QLIB_DATA_DIR="$DATA_DIR/qlib_data" \
    MAJOR_MONEY_DIGEST_JSON="$MAJOR_MONEY_DIGEST_JSON" \
    CAPITAL_FLOW_EVAL_SUMMARY_CSV="$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR/summary.csv" \
    CAPITAL_FLOW_GATE_JSON="$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR/gate.json" \
    TRADE_LOG="$PROJECT_DIR/logs/trade.log" \
    PYTHONPATH="$PYTHONPATH" \
    "$PYTHON_BIN" -m reporter.send_report; then
    REPORTER_RC=0
else
    REPORTER_RC=$?
fi
run_healthcheck nightly error
if [ "$REPORTER_RC" -ne 0 ]; then
    exit "$REPORTER_RC"
fi
log "  Report complete"

log "Daily pipeline finished!"
