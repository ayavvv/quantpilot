from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLIST_PATH = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.polymarket.scheduler.plist"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_polymarket_launchd.sh"
GUARD_SCRIPT = REPO_ROOT / "scripts" / "run_polymarket_scheduler_guard.sh"


def test_polymarket_launchd_plist_contains_required_keys():
    content = PLIST_PATH.read_text()
    assert '<string>com.quantpilot.polymarket.scheduler</string>' in content
    assert '<key>RunAtLoad</key>' in content
    assert '<key>KeepAlive</key>' in content
    assert '<key>UserName</key>' in content
    assert '<string>__USER__</string>' in content
    assert '<key>WorkingDirectory</key>' in content
    assert '<string>__PROJECT_DIR__</string>' in content
    assert '<key>StandardOutPath</key>' in content
    assert '<string>__PROJECT_DIR__/logs/polymarket_scheduler.out.log</string>' in content
    assert '<key>StandardErrorPath</key>' in content
    assert '<string>__PROJECT_DIR__/logs/polymarket_scheduler.err.log</string>' in content


def test_polymarket_launchd_plist_executes_scheduler_wrapper():
    content = PLIST_PATH.read_text()
    assert '<string>/bin/bash</string>' in content
    assert '<string>__PROJECT_DIR__/scripts/run_polymarket_scheduler.sh</string>' in content
    assert '<key>EnvironmentVariables</key>' in content
    assert '<key>PYTHONUNBUFFERED</key>' in content
    assert '<string>__HOME__</string>' in content


def test_install_script_renders_and_bootstraps_launch_daemon():
    content = INSTALL_SCRIPT.read_text()
    assert 'TARGET_DIR="/Library/LaunchDaemons"' in content
    assert 'INSTALL_UID="$(id -u "$INSTALL_USER")"' in content
    assert 'mkdir -p "$LOG_DIR"' in content
    assert "replace('__PROJECT_DIR__', project_dir)" in content
    assert "replace('__HOME__', home_dir)" in content
    assert "replace('__USER__', user_name)" in content
    assert 'launchctl bootout "gui/$INSTALL_UID/$LABEL"' in content
    assert 'launchctl bootout "user/$INSTALL_UID/$LABEL"' in content
    assert 'sudo install -m 644 -o root -g wheel "$TMP_PATH" "$TARGET_PATH"' in content
    assert 'sudo launchctl bootstrap system "$TARGET_PATH"' in content
    assert 'sudo launchctl kickstart -k "system/$LABEL"' in content


def test_polymarket_scheduler_guard_supports_reboot_cron():
    content = GUARD_SCRIPT.read_text()
    assert "polymarket_scheduler_guard.lock" in content
    assert "polymarket_scheduler_guard.pid" in content
    assert 'pgrep -f "[p]ython.*-m polymarket.scheduler"' in content
    assert '"$PROJECT_DIR/scripts/run_polymarket_scheduler.sh"' in content
    assert "POLY_SCHEDULER_RESTART_DELAY_SECONDS" in content
