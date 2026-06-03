import plistlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_SCRIPT = REPO_ROOT / "scripts" / "run_us_microstructure_auto.sh"
RECOVER_SCRIPT = REPO_ROOT / "scripts" / "run_us_microstructure_recover.sh"
WATCHDOG_SCRIPT = REPO_ROOT / "scripts" / "run_us_microstructure_watchdog.sh"
COLLECT_SCRIPT = REPO_ROOT / "scripts" / "run_us_microstructure_collect.sh"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_us_microstructure_report.sh"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_us_microstructure_launchd.sh"
COLLECT_PLIST = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.us_microstructure.collect.plist"
REPORT_PLIST = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.us_microstructure.report.plist"
RECOVER_PLIST = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.us_microstructure.recover.plist"
WATCHDOG_PLIST = REPO_ROOT / "deploy" / "launchd" / "com.quantpilot.us_microstructure.watchdog.plist"
CORE_SYMBOLS = REPO_ROOT / "config" / "us_microstructure_core_symbols.txt"


def _load_plist(path: Path) -> dict:
    with path.open("rb") as handle:
        return plistlib.load(handle)


def test_core_symbol_file_contains_liquid_watchlist():
    symbols = [line.strip() for line in CORE_SYMBOLS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert symbols[:4] == ["US.SPY", "US.QQQ", "US.IWM", "US.DIA"]
    assert "US.LI" in symbols
    assert "US.NVDA" in symbols
    assert "US.AAPL" in symbols
    assert len(symbols) == len(set(symbols))


def test_collect_script_runs_futu_collector_with_nas_and_lock():
    content = COLLECT_SCRIPT.read_text(encoding="utf-8")
    assert "us_microstructure_collect.lock" in content
    assert "US_MICROSTRUCTURE_COLLECT_DURATION_SECONDS" in content
    assert "US_MICROSTRUCTURE_BUILD_UNIVERSE" in content
    assert "US_MICROSTRUCTURE_DYNAMIC_UNIVERSE_FILE" in content
    assert "US_MICROSTRUCTURE_UNIVERSE_TARGET_SIZE" in content
    assert "US_MICROSTRUCTURE_CORE_SOURCE" in content
    assert "US_MICROSTRUCTURE_CORE_WATCHLIST_GROUPS" in content
    assert "US_MICROSTRUCTURE_CORE_WATCHLIST_GROUP_TYPE" in content
    assert "US_MICROSTRUCTURE_UNIVERSE_FILE" in content
    assert "config/us_microstructure_core_symbols.txt" in content
    assert "--core-source" in content
    assert "--core-watchlist-groups" in content
    assert "--core-watchlist-group-type" in content
    assert '"$PYTHON_BIN" -m scripts.build_us_microstructure_universe' in content
    assert "dynamic universe build failed" in content
    assert "--nas-host" in content
    assert "--nas-dir" in content
    assert '"$PYTHON_BIN" -m scripts.collect_us_microstructure' in content


def test_auto_script_runs_locally_on_mac_and_dispatches_from_nas():
    content = AUTO_SCRIPT.read_text(encoding="utf-8")
    assert "TARGET=\"${1:-report}\"" in content
    assert "collect|report" in content
    assert "run_us_microstructure_${TARGET}.sh" in content
    assert "US_MICROSTRUCTURE_FORCE_LOCAL" in content
    assert "US_MICROSTRUCTURE_FORCE_REMOTE" in content
    assert "US_MICROSTRUCTURE_REMOTE_DISPATCHED=1" in content
    assert "US_MICROSTRUCTURE_REMOTE_PROJECT_DIR" in content
    assert "theomac-mini theodeMac-mini-2.local" in content
    assert "[ \"$(uname -s)\" = \"Darwin\" ]" in content
    assert "REMOTE_ENV_NAMES=(" in content
    assert "EXTERNAL_REMOTE_ENV_NAMES" in content
    assert "is_external_remote_env" in content
    assert "US_MICROSTRUCTURE_POST_REPORT_VALIDATION" in content
    assert "BatchMode=yes" in content
    assert "ConnectTimeout=10" in content
    assert 'ssh "${ssh_options[@]}" "$host" "$remote_command"' in content
    assert "US_MICROSTRUCTURE_REMOTE_DRY_RUN" in content


def test_recover_script_guards_reboot_catchup_windows():
    content = RECOVER_SCRIPT.read_text(encoding="utf-8")
    assert "US_MICROSTRUCTURE_RECOVER_COLLECT_START" in content
    assert "US_MICROSTRUCTURE_RECOVER_COLLECT_END" in content
    assert "US_MICROSTRUCTURE_RECOVER_REPORT_START" in content
    assert "US_MICROSTRUCTURE_RECOVER_REPORT_END" in content
    assert "US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS" in content
    assert "US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL" in content
    assert "US_MICROSTRUCTURE_RECOVER_DRY_RUN" in content
    assert "Asia/Shanghai" in content
    assert "run_collect" in content
    assert "collect_seconds" in content
    assert "report_needed" in content
    assert 'US_MICROSTRUCTURE_COLLECT_DURATION_SECONDS="$collect_seconds"' in content
    assert 'US_MICROSTRUCTURE_SEND_EMAIL="$US_MICROSTRUCTURE_RECOVER_SEND_EMAIL"' in content
    assert 'run_us_microstructure_auto.sh" collect' in content
    assert 'run_us_microstructure_auto.sh" report' in content
    assert "us_microstructure_recover.lock" in content


def test_watchdog_script_runs_recovery_readiness_and_safe_repairs():
    content = WATCHDOG_SCRIPT.read_text(encoding="utf-8")
    assert "US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR" in content
    assert "US_MICROSTRUCTURE_WATCHDOG_DRY_RUN" in content
    assert "us_microstructure_watchdog.lock" in content
    assert "run_us_microstructure_recover.sh" in content
    assert "scripts.us_microstructure_readiness" in content
    assert "--all-manifests" in content
    assert "repair_known_issues()" in content
    assert "scripts.repair_us_microstructure_nas_uploads" in content
    assert "readiness_after_repair" in content
    assert "high_confidence_ready" in content


def test_report_script_updates_prices_before_validation_by_default():
    content = REPORT_SCRIPT.read_text(encoding="utf-8")
    assert "US_MICROSTRUCTURE_UPDATE_PRICES" in content
    assert "US_MICROSTRUCTURE_POST_REPORT_VALIDATION" in content
    assert "scripts.us_microstructure_dates default-report-date" in content
    assert "scripts.us_microstructure_dates validation-end-date" in content
    assert '"$PYTHON_BIN" -m scripts.update_us_microstructure_prices' in content
    assert '"$PYTHON_BIN" -m scripts.repair_us_microstructure_nas_uploads' in content
    assert '"$PYTHON_BIN" -m scripts.validate_us_microstructure_flow' in content
    assert '"$PYTHON_BIN" -m scripts.report_us_microstructure_flow' in content
    assert '"$PYTHON_BIN" -m scripts.replay_us_microstructure_intraday' in content
    assert "--rebuild-features" in content
    assert "report_exit=0" in content
    assert "|| report_exit=$?" in content
    assert "run_validation()" in content
    assert "report_args=(" in content
    assert "delivery_report_args" in content
    assert "stage report before same-day validation" in content
    assert "post-report validation end=$US_MICROSTRUCTURE_DATE" in content
    assert "final report" in content
    assert "readiness_exit=0" in content
    assert '"$PYTHON_BIN" -m scripts.us_microstructure_readiness' in content
    assert '--nas-host "$US_MICROSTRUCTURE_NAS_HOST"' in content
    assert '--nas-dir "$US_MICROSTRUCTURE_NAS_DIR"' in content
    assert '--end-date "$validation_end"' in content
    assert content.index("scripts.update_us_microstructure_prices") < content.index('run_validation "$US_MICROSTRUCTURE_VALIDATION_END"')
    assert content.index("scripts.repair_us_microstructure_nas_uploads") < content.index("scripts.replay_us_microstructure_intraday")
    assert content.index('run_validation "$US_MICROSTRUCTURE_VALIDATION_END"') < content.index("scripts.replay_us_microstructure_intraday")
    assert content.index("scripts.replay_us_microstructure_intraday") < content.index("stage report before same-day validation")
    assert content.index("stage report before same-day validation") < content.index("post-report validation end=$US_MICROSTRUCTURE_DATE")
    assert content.index("post-report validation end=$US_MICROSTRUCTURE_DATE") < content.index("final report")
    assert content.rindex("scripts.report_us_microstructure_flow") < content.index("scripts.us_microstructure_readiness")


def test_us_microstructure_collect_launchd_plist_is_scheduled_weekday_evenings():
    payload = _load_plist(COLLECT_PLIST)
    assert payload["Label"] == "com.quantpilot.us_microstructure.collect"
    assert payload["UserName"] == "__USER__"
    assert payload["WorkingDirectory"] == "__PROJECT_DIR__"
    assert payload["ProgramArguments"] == ["/bin/bash", "__PROJECT_DIR__/scripts/run_us_microstructure_auto.sh", "collect"]
    assert payload["StandardOutPath"] == "__PROJECT_DIR__/logs/us_microstructure_collect.out.log"
    assert payload["StandardErrorPath"] == "__PROJECT_DIR__/logs/us_microstructure_collect.err.log"
    intervals = payload["StartCalendarInterval"]
    assert len(intervals) == 5
    assert {item["Weekday"] for item in intervals} == {1, 2, 3, 4, 5}
    assert {item["Hour"] for item in intervals} == {21}
    assert {item["Minute"] for item in intervals} == {25}


def test_us_microstructure_report_launchd_plist_is_scheduled_china_mornings():
    payload = _load_plist(REPORT_PLIST)
    assert payload["Label"] == "com.quantpilot.us_microstructure.report"
    assert payload["ProgramArguments"] == ["/bin/bash", "__PROJECT_DIR__/scripts/run_us_microstructure_auto.sh", "report"]
    assert payload["StandardOutPath"] == "__PROJECT_DIR__/logs/us_microstructure_report.out.log"
    assert payload["StandardErrorPath"] == "__PROJECT_DIR__/logs/us_microstructure_report.err.log"
    intervals = payload["StartCalendarInterval"]
    assert len(intervals) == 5
    assert {item["Weekday"] for item in intervals} == {2, 3, 4, 5, 6}
    assert {item["Hour"] for item in intervals} == {8}
    assert {item["Minute"] for item in intervals} == {30}
    assert payload["EnvironmentVariables"]["US_MICROSTRUCTURE_SEND_EMAIL"] == "true"


def test_us_microstructure_recover_launchd_plist_runs_at_load_and_periodically():
    payload = _load_plist(RECOVER_PLIST)
    assert payload["Label"] == "com.quantpilot.us_microstructure.recover"
    assert payload["UserName"] == "__USER__"
    assert payload["WorkingDirectory"] == "__PROJECT_DIR__"
    assert payload["ProgramArguments"] == ["/bin/bash", "__PROJECT_DIR__/scripts/run_us_microstructure_recover.sh"]
    assert payload["RunAtLoad"] is True
    assert payload["StartInterval"] == 900
    assert payload["StandardOutPath"] == "__PROJECT_DIR__/logs/us_microstructure_recover.out.log"
    assert payload["StandardErrorPath"] == "__PROJECT_DIR__/logs/us_microstructure_recover.err.log"
    assert payload["EnvironmentVariables"]["US_MICROSTRUCTURE_RECOVER_SEND_EMAIL"] == "true"


def test_us_microstructure_watchdog_launchd_plist_checks_morning_and_evening():
    payload = _load_plist(WATCHDOG_PLIST)
    assert payload["Label"] == "com.quantpilot.us_microstructure.watchdog"
    assert payload["UserName"] == "__USER__"
    assert payload["WorkingDirectory"] == "__PROJECT_DIR__"
    assert payload["ProgramArguments"] == ["/bin/bash", "__PROJECT_DIR__/scripts/run_us_microstructure_watchdog.sh"]
    assert payload["RunAtLoad"] is True
    assert payload["StandardOutPath"] == "__PROJECT_DIR__/logs/us_microstructure_watchdog.out.log"
    assert payload["StandardErrorPath"] == "__PROJECT_DIR__/logs/us_microstructure_watchdog.err.log"
    assert payload["EnvironmentVariables"]["US_MICROSTRUCTURE_WATCHDOG_AUTO_REPAIR"] == "true"
    intervals = payload["StartCalendarInterval"]
    assert len(intervals) == 10
    assert sum(1 for item in intervals if item["Hour"] == 9 and item["Minute"] == 15) == 5
    assert sum(1 for item in intervals if item["Hour"] == 21 and item["Minute"] == 10) == 5
    assert {item["Weekday"] for item in intervals if item["Hour"] == 9} == {2, 3, 4, 5, 6}
    assert {item["Weekday"] for item in intervals if item["Hour"] == 21} == {1, 2, 3, 4, 5}


def test_us_microstructure_install_script_installs_launch_jobs():
    content = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert 'TARGET_DIR="/Library/LaunchDaemons"' in content
    assert 'AGENT_TARGET_DIR="$HOME_DIR/Library/LaunchAgents"' in content
    assert 'mktemp "/tmp/${label}.plist.XXXXXX"' in content
    assert 'US_MICROSTRUCTURE_LAUNCHD_MODE' in content
    assert 'sudo -n true' in content
    assert 'render_and_install "com.quantpilot.us_microstructure.collect"' in content
    assert 'render_and_install "com.quantpilot.us_microstructure.report"' in content
    assert 'render_and_install "com.quantpilot.us_microstructure.recover"' in content
    assert 'render_and_install "com.quantpilot.us_microstructure.watchdog"' in content
    assert 'payload.pop("UserName", None)' in content
    assert 'sudo launchctl bootstrap system "$target_path"' in content
    assert 'sudo launchctl enable "system/$label"' in content
    assert 'launchctl bootstrap "gui/$INSTALL_UID" "$target_path"' in content
    assert 'launchctl enable "gui/$INSTALL_UID/$label"' in content
