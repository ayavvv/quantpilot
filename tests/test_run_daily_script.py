from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DAILY = REPO_ROOT / "scripts" / "run_daily.sh"
SYNC_DATA = REPO_ROOT / "scripts" / "sync_data.sh"
RUN_DAILY_WHEN_READY = REPO_ROOT / "scripts" / "run_daily_when_ready.sh"
RUN_PRETRADE_WATCHDOG = REPO_ROOT / "scripts" / "run_pretrade_watchdog.sh"
RUN_TRADE = REPO_ROOT / "scripts" / "run_trade.sh"


def test_run_daily_passes_target_date_to_sync_script():
    content = RUN_DAILY.read_text()
    assert 'SKIP_NAS_SYNC="${SKIP_NAS_SYNC:-false}"' in content
    assert 'if [ -n "$NAS_HOST" ] && [ -n "$NAS_USER" ] && [ "$SKIP_NAS_SYNC" != "true" ]; then' in content
    assert 'SYNC_TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE"' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE="${SYNC_TARGET_A_SHARE_DATE:-}" "$SCRIPT_DIR/sync_data.sh"' in content


def test_run_daily_stale_sync_uses_nas_last_when_available():
    content = RUN_DAILY.read_text()
    assert 'EFFECTIVE_NAS_DATE="$NAS_LAST"' in content
    assert 'EFFECTIVE_NAS_DATE="$NAS_LATEST"' in content
    assert 'SYNC_TARGET_A_SHARE_DATE="$EFFECTIVE_NAS_DATE"' in content
    assert 'SYNC_TARGET_A_SHARE_DATE=""' in content


def test_run_daily_supports_target_date_override():
    content = RUN_DAILY.read_text()
    assert 'TARGET_A_SHARE_DATE_OVERRIDE="${TARGET_A_SHARE_DATE_OVERRIDE:-}"' in content
    assert 'Target A-share trading date override: $TARGET_A_SHARE_DATE' in content


def test_run_daily_schedules_ready_retry_on_timeout():
    content = RUN_DAILY.read_text()
    assert 'AUTO_RETRY_ON_NAS_READY="${AUTO_RETRY_ON_NAS_READY:-true}"' in content
    assert 'spawn_ready_retry "$TARGET_A_SHARE_DATE"' in content
    assert 'nohup "$SCRIPT_DIR/run_daily_when_ready.sh" "$target_date"' in content
    assert 'scripts.a_share_readiness nas-latest-date' in content


def test_run_daily_when_ready_respects_skip_nas_sync():
    content = RUN_DAILY_WHEN_READY.read_text()
    assert 'SKIP_NAS_SYNC="${SKIP_NAS_SYNC:-false}"' in content
    assert 'SKIP_NAS_SYNC=true, not watching NAS readiness' in content


def test_run_daily_runs_healthcheck_on_failures_and_completion():
    content = RUN_DAILY.read_text()
    assert 'run_healthcheck() {' in content
    assert 'run_healthcheck nightly error' in content
    assert 'target_args+=(--target-a-share-date "$TARGET_A_SHARE_DATE")' in content
    assert '"$PYTHON_BIN" -m scripts.daily_healthcheck' in content


def test_run_daily_runs_reporter_natively_with_reporter_env():
    content = RUN_DAILY.read_text()
    assert 'REPORTER_ENV_FILE="$PROJECT_DIR/reporter/.env"' in content
    assert '"$PYTHON_BIN" -m reporter.send_report' in content
    assert 'MAJOR_MONEY_DIGEST_JSON="$MAJOR_MONEY_DIGEST_JSON"' in content
    assert 'CAPITAL_FLOW_EVAL_SUMMARY_CSV="$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR/summary.csv"' in content
    assert 'CAPITAL_FLOW_GATE_JSON="$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR/gate.json"' in content
    assert 'TRADE_LOG="$PROJECT_DIR/logs/trade.log"' in content


def test_run_daily_preserves_actual_exit_codes():
    content = RUN_DAILY.read_text()
    assert 'if QLIB_DATA_DIR="$DATA_DIR/qlib_data" \\' in content
    assert 'if REPORTER_ENV_FILE="$PROJECT_DIR/reporter/.env" \\' in content
    assert 'if ! QLIB_DATA_DIR="$DATA_DIR/qlib_data" \\' not in content
    assert 'if ! REPORTER_ENV_FILE="$PROJECT_DIR/reporter/.env" \\' not in content


def test_run_daily_sets_signal_output_tag_from_target_date():
    content = RUN_DAILY.read_text()
    assert 'resolve_signal_output_tag() {' in content
    assert 'SIGNAL_OUTPUT_TAG_VALUE="${SIGNAL_OUTPUT_TAG_OVERRIDE:-}"' in content
    assert 'SIGNAL_OUTPUT_TAG_VALUE="$(resolve_signal_output_tag "${SYNC_TARGET_A_SHARE_DATE:-${TARGET_A_SHARE_DATE:-}}")"' in content
    assert 'SIGNAL_OUTPUT_TAG="$SIGNAL_OUTPUT_TAG_VALUE"' in content


def test_run_daily_evaluates_archived_capital_flow_overlays():
    content = RUN_DAILY.read_text()
    assert 'ENABLE_A_SHARE_CAPITAL_FLOW_EVAL="${ENABLE_A_SHARE_CAPITAL_FLOW_EVAL:-true}"' in content
    assert 'A_SHARE_CAPITAL_FLOW_EVAL_HORIZONS="${A_SHARE_CAPITAL_FLOW_EVAL_HORIZONS:-1,3,5}"' in content
    assert 'A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT="${A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT:-20}"' in content
    assert '"$PYTHON_BIN" -m scripts.evaluate_futu_capital_flow_overlay' in content
    assert '--archive-dir "$A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR"' in content
    assert '--output-dir "$A_SHARE_CAPITAL_FLOW_EVAL_OUTPUT_DIR"' in content
    assert '--gate-min-date-count "$A_SHARE_CAPITAL_FLOW_GATE_MIN_DATE_COUNT"' in content
    assert '--gate-risk-alpha-threshold "$A_SHARE_CAPITAL_FLOW_GATE_RISK_ALPHA_THRESHOLD"' in content


def test_run_daily_builds_market_wide_major_money_digest():
    content = RUN_DAILY.read_text()
    assert 'ENABLE_MAJOR_MONEY_DIGEST="${ENABLE_MAJOR_MONEY_DIGEST:-true}"' in content
    assert 'MAJOR_MONEY_EXPECTED_MARKETS="${MAJOR_MONEY_EXPECTED_MARKETS:-A,HK,US,US_OTC}"' in content
    assert 'MAJOR_MONEY_DIGEST_ARCHIVE_DIR="${MAJOR_MONEY_DIGEST_ARCHIVE_DIR:-$DATA_DIR/output/major_money_digest}"' in content
    assert 'ENABLE_EASTMONEY_FUND_FLOW_REFRESH="${ENABLE_EASTMONEY_FUND_FLOW_REFRESH:-true}"' in content
    assert 'ENABLE_US_OTC_PROXY_FLOW="${ENABLE_US_OTC_PROXY_FLOW:-false}"' in content
    assert 'US_OTC_PROXY_FLOW_EXCHANGE_TYPES="${US_OTC_PROXY_FLOW_EXCHANGE_TYPES:-US_PINK}"' in content
    assert 'US_OTC_PROXY_FLOW_REQUEST_DELAY="${US_OTC_PROXY_FLOW_REQUEST_DELAY:-0.2}"' in content
    assert 'US_OTC_PROXY_FLOW_MAX_RETRIES="${US_OTC_PROXY_FLOW_MAX_RETRIES:-2}"' in content
    assert 'US_OTC_PROXY_FLOW_TIMEOUT="${US_OTC_PROXY_FLOW_TIMEOUT:-15}"' in content
    assert 'US_OTC_PROXY_FLOW_BATCH_FLUSH="${US_OTC_PROXY_FLOW_BATCH_FLUSH:-100}"' in content
    assert 'US_OTC_PROXY_FLOW_CONCURRENCY="${US_OTC_PROXY_FLOW_CONCURRENCY:-1}"' in content
    assert 'US_OTC_PROXY_FLOW_OVERWRITE="${US_OTC_PROXY_FLOW_OVERWRITE:-false}"' in content
    assert 'US_OTC_PROXY_FLOW_AVAILABLE=false' in content
    assert '"$PYTHON_BIN" -m scripts.refresh_eastmoney_fund_flow_rank' in content
    assert '"$PYTHON_BIN" -m scripts.scan_us_otc_proxy_flow' in content
    assert '--exchange-types "$US_OTC_PROXY_FLOW_EXCHANGE_TYPES"' in content
    assert '--request-delay "$US_OTC_PROXY_FLOW_REQUEST_DELAY"' in content
    assert '--max-retries "$US_OTC_PROXY_FLOW_MAX_RETRIES"' in content
    assert '--timeout "$US_OTC_PROXY_FLOW_TIMEOUT"' in content
    assert '--batch-flush "$US_OTC_PROXY_FLOW_BATCH_FLUSH"' in content
    assert '--concurrency "$US_OTC_PROXY_FLOW_CONCURRENCY"' in content
    assert 'US_OTC_PROXY_ARGS+=(--overwrite)' in content
    assert 'US_OTC_PROXY_ARGS+=(--date "$US_OTC_PROXY_FLOW_DATE")' in content
    assert 'US_OTC_PROXY_FLOW_AVAILABLE=true' in content
    assert '--output "$EASTMONEY_FUND_FLOW_RANK_OUTPUT"' in content
    assert 'MAJOR_MONEY_SOURCE_ARGS+=(--source "A:$EASTMONEY_FUND_FLOW_RANK_OUTPUT:eastmoney")' in content
    assert 'latest_flow="$DATA_DIR/capital_flow/futu_market/${market}_latest_flow.csv"' in content
    assert 'if [ "$US_OTC_PROXY_FLOW_AVAILABLE" = "true" ] && [ -f "$otc_latest_flow" ]; then' in content
    assert 'MAJOR_MONEY_DIGEST_SOURCES="${MAJOR_MONEY_DIGEST_SOURCES:-auto}"' in content
    assert '"$PYTHON_BIN" -m scripts.build_major_money_digest' in content
    assert '--expected-markets "$MAJOR_MONEY_EXPECTED_MARKETS"' in content
    assert '--output-json "$MAJOR_MONEY_DIGEST_JSON"' in content
    assert '--archive-dir "$MAJOR_MONEY_DIGEST_ARCHIVE_DIR"' in content


def test_run_daily_when_ready_replays_target_date():
    content = RUN_DAILY_WHEN_READY.read_text()
    assert 'TARGET_A_SHARE_DATE_OVERRIDE="$TARGET_A_SHARE_DATE"' in content
    assert 'AUTO_RETRY_ON_NAS_READY="false"' in content
    assert 'pred-latest-signal-date' in content


def test_run_pretrade_watchdog_calls_python_watchdog_and_healthcheck():
    content = RUN_PRETRADE_WATCHDOG.read_text()
    assert '"$PYTHON_BIN" -m scripts.pretrade_watchdog' in content
    assert '--phase pretrade' in content
    assert '--alert-on error' in content
    assert 'if PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m scripts.pretrade_watchdog; then' in content


def test_run_trade_runs_healthcheck_after_trade():
    content = RUN_TRADE.read_text()
    assert '--phase trade' in content
    assert '"$PYTHON_BIN" -m scripts.daily_healthcheck' in content
    assert 'if FUTU_HOST="${FUTU_HOST:-192.168.100.248}" \\' in content


def test_run_trade_builds_pretrade_capital_flow_advisory():
    content = RUN_TRADE.read_text()
    assert 'ENABLE_PRETRADE_CAPITAL_FLOW_CHECK="${ENABLE_PRETRADE_CAPITAL_FLOW_CHECK:-true}"' in content
    assert '"$PYTHON_BIN" -m scripts.build_futu_capital_flow_overlay' in content
    assert '--signal-top-n "$PRETRADE_CAPITAL_FLOW_TOP_N"' in content
    assert 'CAPITAL_FLOW_OVERLAY_CSV="$PRETRADE_CAPITAL_FLOW_OVERLAY_CSV"' in content
    assert 'ENABLE_CAPITAL_FLOW_ADVISORY="$ENABLE_CAPITAL_FLOW_ADVISORY"' in content


def test_sync_data_syncs_and_promotes_metadata():
    content = SYNC_DATA.read_text()
    assert 'SKIP_NAS_SYNC="${SKIP_NAS_SYNC:-false}"' in content
    assert 'NAS sync disabled by SKIP_NAS_SYNC=true' in content
    assert 'NAS_HOST/NAS_USER not configured; skipping NAS sync' in content
    assert 'tar cf - calendars instruments features metadata' in content
    assert 'for subdir in calendars instruments features metadata; do' in content


def test_sync_data_validates_staged_snapshot_against_expected_target():
    content = SYNC_DATA.read_text()
    assert 'validate_staged_snapshot()' in content
    assert 'validate_staged_qlib_snapshot' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE' in content
    assert 'allow_metadata_lag=True' in content
