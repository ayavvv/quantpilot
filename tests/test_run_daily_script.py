from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DAILY = REPO_ROOT / "scripts" / "run_daily.sh"
SYNC_DATA = REPO_ROOT / "scripts" / "sync_data.sh"


def test_run_daily_passes_target_date_to_sync_script():
    content = RUN_DAILY.read_text()
    assert 'SYNC_TARGET_A_SHARE_DATE="$TARGET_A_SHARE_DATE"' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE="${SYNC_TARGET_A_SHARE_DATE:-}" "$SCRIPT_DIR/sync_data.sh"' in content


def test_run_daily_stale_sync_uses_nas_last_when_available():
    content = RUN_DAILY.read_text()
    assert 'SYNC_TARGET_A_SHARE_DATE="$NAS_LAST"' in content
    assert 'SYNC_TARGET_A_SHARE_DATE=""' in content


def test_sync_data_syncs_and_promotes_metadata():
    content = SYNC_DATA.read_text()
    assert 'tar cf - calendars instruments features metadata' in content
    assert 'for subdir in calendars instruments features metadata; do' in content


def test_sync_data_validates_staged_snapshot_against_expected_target():
    content = SYNC_DATA.read_text()
    assert 'validate_staged_snapshot()' in content
    assert 'validate_staged_qlib_snapshot' in content
    assert 'EXPECTED_TARGET_A_SHARE_DATE' in content
