"""Pre-trade watchdog that ensures the latest aligned signal exists before the trade window."""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts import a_share_readiness


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
QLIB_DIR = DATA_DIR / "qlib_data"
SIGNAL_DIR = DATA_DIR / "signals"
MODEL_DIR = DATA_DIR / "models"
PRED_PATH = SIGNAL_DIR / "pred_sh_latest.pkl"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", str(PROJECT_DIR / ".venv" / "bin" / "python")))
SYNC_SCRIPT = PROJECT_DIR / "scripts" / "sync_data.sh"
LOCK_ROOT = Path(os.environ.get("PRETRADE_WATCHDOG_LOCK_ROOT", str(PROJECT_DIR / "logs" / "pretrade_watchdog_locks")))
LOCK_DIR = LOCK_ROOT / "active"

NAS_HOST = os.environ.get("NAS_HOST", "")
NAS_USER = os.environ.get("NAS_USER", "")
NAS_QLIB_PATH = os.environ.get("NAS_QLIB_PATH", "/volume1/docker/quantpilot/qlib_data")
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def latest_local_completed_date() -> str:
    return a_share_readiness.latest_completed_a_share_date_from_status(
        QLIB_DIR / "metadata" / "a_share_sync_status.json"
    )


def latest_local_a_share_date() -> str:
    inst_path = QLIB_DIR / "instruments" / "all.txt"
    if not inst_path.exists():
        return ""
    return a_share_readiness.latest_a_share_date_from_instruments(inst_path)


def latest_signal_date() -> str:
    return a_share_readiness.latest_signal_date_from_prediction(PRED_PATH)


def latest_nas_completed_date() -> str:
    if not (NAS_HOST and NAS_USER):
        return ""
    return a_share_readiness.latest_nas_a_share_completed_date(
        nas_host=NAS_HOST,
        nas_user=NAS_USER,
        ssh_key=SSH_KEY,
        nas_qlib_path=NAS_QLIB_PATH,
    )


def _env_with_project_path() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_DIR}{':' + current_pythonpath if current_pythonpath else ''}"
    return env


def process_running(patterns: list[str]) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "|".join(patterns)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def sync_if_needed(target_date: str) -> None:
    if not target_date:
        return
    env = os.environ.copy()
    env["EXPECTED_TARGET_A_SHARE_DATE"] = target_date
    log(f"Local snapshot behind NAS, syncing target={target_date}")
    subprocess.run([str(SYNC_SCRIPT)], cwd=PROJECT_DIR, env=env, check=True)


def inference_output_tag(target_date: str) -> str:
    return target_date.replace("-", "")


def run_inference(target_date: str) -> None:
    env = _env_with_project_path()
    env.update(
        {
            "QLIB_DATA_DIR": str(QLIB_DIR),
            "MODEL_DIR": str(MODEL_DIR),
            "SIGNAL_DIR": str(SIGNAL_DIR),
            "PROMOTE_LATEST": "true",
            "SIGNAL_OUTPUT_TAG": inference_output_tag(target_date),
        }
    )
    log(f"Signal stale, running inference for target={target_date}")
    subprocess.run(
        [str(PYTHON_BIN), "-m", "inference.run_daily"],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
    )


@contextmanager
def watchdog_lock():
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError:
        pid_path = LOCK_DIR / "pid"
        existing_pid = pid_path.read_text().strip() if pid_path.exists() else ""
        if existing_pid:
            log(f"Watchdog already running with pid={existing_pid}, exiting")
        else:
            log("Watchdog lock already held, exiting")
        raise SystemExit(0)

    try:
        (LOCK_DIR / "pid").write_text(str(os.getpid()))
        yield
    finally:
        shutil.rmtree(LOCK_DIR, ignore_errors=True)


@dataclass
class WatchdogState:
    local_completed: str
    local_latest: str
    signal_date: str
    nas_completed: str


def collect_state() -> WatchdogState:
    nas_completed = ""
    try:
        nas_completed = latest_nas_completed_date()
    except Exception as exc:
        log(f"WARNING: failed to query NAS completion metadata: {exc}")

    return WatchdogState(
        local_completed=latest_local_completed_date(),
        local_latest=latest_local_a_share_date(),
        signal_date=latest_signal_date(),
        nas_completed=nas_completed,
    )


def ensure_signal_ready() -> int:
    state = collect_state()
    log(
        "Pre-trade state: "
        f"local_completed={state.local_completed or 'N/A'} "
        f"local_latest={state.local_latest or 'N/A'} "
        f"signal={state.signal_date or 'N/A'} "
        f"nas_completed={state.nas_completed or 'N/A'}"
    )

    target_sync_date = state.nas_completed or state.local_completed
    if target_sync_date and (
        (state.local_completed and state.local_completed < target_sync_date)
        or (state.local_latest and state.local_latest < target_sync_date)
    ):
        sync_if_needed(target_sync_date)
        state = collect_state()
        log(
            "After sync: "
            f"local_completed={state.local_completed or 'N/A'} "
            f"local_latest={state.local_latest or 'N/A'} "
            f"signal={state.signal_date or 'N/A'}"
        )

    if not state.local_latest:
        log("ERROR: local latest A-share date is unavailable")
        return 1

    if state.signal_date == state.local_latest:
        log(f"Signal already aligned with local A-share data ({state.local_latest})")
        return 0

    if process_running(["python -m inference.run_daily", "run_daily.sh"]):
        log("Nightly or inference process already running; skip duplicate watchdog rerun")
        return 0

    run_inference(state.local_latest)
    refreshed_signal_date = latest_signal_date()
    if refreshed_signal_date != state.local_latest:
        log(
            "ERROR: signal still stale after watchdog inference: "
            f"signal={refreshed_signal_date or 'N/A'} latest_a_share={state.local_latest}"
        )
        return 1

    log(f"Watchdog inference complete: signal={refreshed_signal_date}")
    return 0


def main() -> int:
    with watchdog_lock():
        return ensure_signal_ready()


if __name__ == "__main__":
    raise SystemExit(main())
