from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DAILY = REPO_ROOT / "scripts" / "run_daily.sh"
SYNC_DATA = REPO_ROOT / "scripts" / "sync_data.sh"
RUN_DAILY_WHEN_READY = REPO_ROOT / "scripts" / "run_daily_when_ready.sh"
RUN_PRETRADE_WATCHDOG = REPO_ROOT / "scripts" / "run_pretrade_watchdog.sh"
RUN_TRADE = REPO_ROOT / "scripts" / "run_trade.sh"


def test_run_daily_passes_target_date_to_sync_script():
    content = RUN_DAILY.read_text()
    assert 'SYNC_TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE"' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE="${SYNC_TARGET_A_SHARE_DATE:-}" "$SCRIPT_DIR/sync_data.sh"' in content


def test_run_daily_stale_sync_uses_nas_last_when_available():
    content = RUN_DAILY.read_text()
    assert 'SYNC_TARGET_A_SHARE_DATE="$NAS_LAST"' in content
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


def test_run_daily_runs_healthcheck_on_failures_and_completion():
    content = RUN_DAILY.read_text()
    assert 'run_healthcheck() {' in content
    assert 'run_healthcheck nightly error' in content
    assert 'target_args+=(--target-a-share-date "$TARGET_A_SHARE_DATE")' in content
    assert '"$PYTHON_BIN" -m scripts.daily_healthcheck' in content


def test_run_daily_sets_signal_output_tag_from_target_date():
    content = RUN_DAILY.read_text()
    assert 'resolve_signal_output_tag() {' in content
    assert 'SIGNAL_OUTPUT_TAG_VALUE="${SIGNAL_OUTPUT_TAG_OVERRIDE:-}"' in content
    assert 'SIGNAL_OUTPUT_TAG_VALUE="$(resolve_signal_output_tag "${SYNC_TARGET_A_SHARE_DATE:-${TARGET_A_SHARE_DATE:-}}")"' in content
    assert 'SIGNAL_OUTPUT_TAG="$SIGNAL_OUTPUT_TAG_VALUE"' in content


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


def test_run_trade_runs_healthcheck_after_trade():
    content = RUN_TRADE.read_text()
    assert '--phase trade' in content
    assert '"$PYTHON_BIN" -m scripts.daily_healthcheck' in content


def test_sync_data_syncs_and_promotes_metadata():
    content = SYNC_DATA.read_text()
    assert 'tar cf - calendars instruments features metadata' in content
    assert 'for subdir in calendars instruments features metadata; do' in content


def test_sync_data_validates_staged_snapshot_against_expected_target():
    content = SYNC_DATA.read_text()
    assert 'validate_staged_snapshot()' in content
    assert 'validate_staged_qlib_snapshot' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE' in content
