from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLIST_PATH = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.polymarket.scheduler.plist"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_polymarket_launchd.sh"


def test_polymarket_launchd_plist_contains_required_keys():
    content = PLIST_PATH.read_text()
    assert '<string>com.quantpilot.polymarket.scheduler</string>' in content
    assert '<key>RunAtLoad</key>' in content
    assert '<key>KeepAlive</key>' in content
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


def test_install_script_renders_and_bootstraps_launch_agent():
    content = INSTALL_SCRIPT.read_text()
    assert 'mkdir -p "$TARGET_DIR" "$LOG_DIR"' in content
    assert "replace('__PROJECT_DIR__', project_dir)" in content
    assert "replace('__HOME__', home_dir)" in content
    assert 'launchctl bootstrap "gui/$(id -u)" "$TARGET_PATH"' in content
    assert 'launchctl kickstart -k "gui/$(id -u)/com.quantpilot.polymarket.scheduler"' in content
