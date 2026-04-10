"""Daily health snapshot + alerting for collector, signal generation, and trading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("REPORT_DIR", str(PROJECT_DIR / "logs" / "reports"))

from reporter.send_report import send_email
from scripts import a_share_readiness


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
QLIB_DIR = DATA_DIR / "qlib_data"
SIGNAL_DIR = DATA_DIR / "signals"
LOGS_DIR = PROJECT_DIR / "logs"
HEALTH_DIR = Path(os.environ.get("HEALTHCHECK_DIR", str(LOGS_DIR / "health")))
TRADE_LOG = Path(os.environ.get("TRADE_LOG", str(LOGS_DIR / "trade.log")))
DAILY_LOG = Path(os.environ.get("DAILY_LOG", str(LOGS_DIR / "daily.log")))
RETRY_LOG = Path(os.environ.get("DAILY_RETRY_LOG", str(LOGS_DIR / "daily_retry.log")))
PRED_PATH = SIGNAL_DIR / "pred_sh_latest.pkl"
NAS_HOST = os.environ.get("NAS_HOST", "")
NAS_USER = os.environ.get("NAS_USER", "")
NAS_QLIB_PATH = os.environ.get("NAS_QLIB_PATH", "/volume1/docker/quantpilot/qlib_data")
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))

LEVEL_ORDER = {"ok": 0, "warn": 1, "error": 2}


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


def latest_nas_completed_date() -> tuple[str, str]:
    if not (NAS_HOST and NAS_USER):
        return "", ""
    try:
        value = a_share_readiness.latest_nas_a_share_completed_date(
            nas_host=NAS_HOST,
            nas_user=NAS_USER,
            ssh_key=SSH_KEY,
            nas_qlib_path=NAS_QLIB_PATH,
        )
        return value, ""
    except Exception as exc:
        return "", str(exc)


def _bump_level(current: str, new_level: str) -> str:
    return new_level if LEVEL_ORDER[new_level] > LEVEL_ORDER[current] else current


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


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def analyze_trade_log(today: str) -> dict[str, Any]:
    lines = [line for line in _read_lines(TRADE_LOG) if today in line]
    starts = [line for line in lines if "run_trade: start" in line]
    dones = [line for line in lines if "run_trade: done" in line]
    order_failures = [line for line in lines if "  FAIL " in line]
    order_fills = [line for line in lines if "  OK " in line]
    errors = [line for line in lines if "[ERROR]" in line]
    stale_signal_errors = [
        line for line in lines if "信号日期与本地 A 股最新数据不一致" in line
    ]
    return {
        "starts": len(starts),
        "done": len(dones),
        "order_failures": len(order_failures),
        "order_fills": len(order_fills),
        "errors": len(errors),
        "stale_signal_errors": stale_signal_errors,
        "latest_line": lines[-1] if lines else "",
    }


def analyze_daily_logs(today: str) -> dict[str, Any]:
    daily_lines = [line for line in _read_lines(DAILY_LOG) if today in line]
    retry_lines = [line for line in _read_lines(RETRY_LOG) if today in line]
    timeout_lines = [line for line in daily_lines if "not ready after" in line]
    inference_failures = [line for line in daily_lines if "推理失败" in line]
    return {
        "timeouts": len(timeout_lines),
        "inference_failures": len(inference_failures),
        "retry_activity": len(retry_lines),
        "latest_daily_line": daily_lines[-1] if daily_lines else "",
        "latest_retry_line": retry_lines[-1] if retry_lines else "",
    }


def build_snapshot(phase: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")

    local_completed = latest_local_completed_date()
    local_latest = latest_local_a_share_date()
    signal_date = latest_signal_date()
    nas_completed, nas_error = latest_nas_completed_date()
    trade = analyze_trade_log(today)
    nightly_logs = analyze_daily_logs(today)
    processes = {
        "nightly_running": process_running(["python -m inference.run_daily", "run_daily.sh"]),
        "retry_watcher_running": process_running(["run_daily_when_ready.sh"]),
        "pretrade_watchdog_running": process_running(["pretrade_watchdog.py"]),
    }

    overall = "ok"
    issues: list[str] = []

    if not local_latest:
        overall = _bump_level(overall, "error")
        issues.append("Local A-share snapshot missing latest instruments date")

    if local_latest and signal_date != local_latest:
        overall = _bump_level(overall, "error")
        issues.append(f"Signal stale: signal={signal_date or 'N/A'} latest_a_share={local_latest}")

    if nas_completed:
        if not local_completed or local_completed < nas_completed:
            level = "error" if phase in {"pretrade", "trade"} else "warn"
            overall = _bump_level(overall, level)
            issues.append(
                f"Local completed snapshot lags NAS: local_completed={local_completed or 'N/A'} nas_completed={nas_completed}"
            )
    elif nas_error:
        overall = _bump_level(overall, "warn")
        issues.append(f"Failed to query NAS completion metadata: {nas_error}")

    if phase == "nightly":
        if local_latest and signal_date == local_latest:
            pass
        elif processes["nightly_running"] or processes["retry_watcher_running"]:
            overall = _bump_level(overall, "warn")
            issues.append("Nightly pipeline still running or waiting on retry watcher")
        elif nightly_logs["timeouts"] or nightly_logs["inference_failures"]:
            overall = _bump_level(overall, "error")
            issues.append(
                f"Nightly logs show timeout={nightly_logs['timeouts']} inference_failures={nightly_logs['inference_failures']}"
            )

    if phase == "pretrade" and processes["pretrade_watchdog_running"]:
        overall = _bump_level(overall, "warn")
        issues.append("Pre-trade watchdog is currently running")

    if phase == "trade":
        if trade["starts"] == 0 and trade["done"] == 0:
            overall = _bump_level(overall, "error")
            issues.append("No run_trade execution found today")
        if trade["stale_signal_errors"]:
            overall = _bump_level(overall, "error")
            issues.append("Trading saw stale-signal protection")
        elif trade["errors"]:
            overall = _bump_level(overall, "warn")
            issues.append(f"Trading log has {trade['errors']} error line(s)")
        elif trade["order_failures"]:
            overall = _bump_level(overall, "warn")
            issues.append(f"Trading log has {trade['order_failures']} failed order(s)")

    if not issues:
        issues.append("All monitored checks passed")

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": today,
        "phase": phase,
        "overall_status": overall,
        "issues": issues,
        "local": {
            "completed_a_share_date": local_completed,
            "latest_a_share_date": local_latest,
            "latest_signal_date": signal_date,
            "signal_aligned": bool(local_latest) and signal_date == local_latest,
        },
        "nas": {
            "completed_a_share_date": nas_completed,
            "query_error": nas_error,
        },
        "processes": processes,
        "nightly": nightly_logs,
        "trade": trade,
    }


def write_snapshot(snapshot: dict[str, Any]) -> Path:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = HEALTH_DIR / f"health_{snapshot['date'].replace('-', '')}_{snapshot['phase']}.json"
    latest_path = HEALTH_DIR / f"health_latest_{snapshot['phase']}.json"
    dated_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return dated_path


def alert_threshold_met(level: str, threshold: str) -> bool:
    return LEVEL_ORDER[level] >= LEVEL_ORDER[threshold]


def render_snapshot_html(snapshot: dict[str, Any]) -> str:
    issue_items = "".join(f"<li>{issue}</li>" for issue in snapshot["issues"])
    local = snapshot["local"]
    nas = snapshot["nas"]
    trade = snapshot["trade"]
    return f"""
    <html>
    <body style="font-family: -apple-system, sans-serif; max-width: 760px; margin: 0 auto; padding: 20px;">
      <h1>QuantPilot Health Alert</h1>
      <p><strong>Phase:</strong> {snapshot['phase']}</p>
      <p><strong>Status:</strong> {snapshot['overall_status'].upper()}</p>
      <p><strong>Timestamp:</strong> {snapshot['timestamp']}</p>
      <h2>Issues</h2>
      <ul>{issue_items}</ul>
      <h2>Local State</h2>
      <p>completed_a_share={local['completed_a_share_date'] or 'N/A'}<br>
      latest_a_share={local['latest_a_share_date'] or 'N/A'}<br>
      latest_signal={local['latest_signal_date'] or 'N/A'}<br>
      signal_aligned={local['signal_aligned']}</p>
      <h2>NAS State</h2>
      <p>completed_a_share={nas['completed_a_share_date'] or 'N/A'}<br>
      query_error={nas['query_error'] or 'N/A'}</p>
      <h2>Trade Log</h2>
      <p>starts={trade['starts']} done={trade['done']} fills={trade['order_fills']} failures={trade['order_failures']} errors={trade['errors']}</p>
    </body>
    </html>
    """


def maybe_send_alert(snapshot: dict[str, Any], threshold: str) -> bool:
    if not alert_threshold_met(snapshot["overall_status"], threshold):
        return False

    alerts_dir = HEALTH_DIR / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        json.dumps(
            {
                "phase": snapshot["phase"],
                "status": snapshot["overall_status"],
                "issues": snapshot["issues"],
                "date": snapshot["date"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    stamp = alerts_dir / f"{snapshot['date'].replace('-', '')}_{snapshot['phase']}_{digest}.sent"
    if stamp.exists():
        return False

    subject = f"QuantPilot Alert [{snapshot['overall_status'].upper()}] {snapshot['phase']} {snapshot['date']}"
    send_email(render_snapshot_html(snapshot), subject)
    stamp.write_text(snapshot["timestamp"], encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantPilot daily health snapshot")
    parser.add_argument("--phase", default="manual", choices=["manual", "nightly", "pretrade", "trade"])
    parser.add_argument("--alert-on", default="error", choices=["ok", "warn", "error"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = build_snapshot(args.phase)
    output_path = write_snapshot(snapshot)
    alerted = maybe_send_alert(snapshot, args.alert_on)
    print(
        f"healthcheck phase={snapshot['phase']} status={snapshot['overall_status']} "
        f"alerted={'yes' if alerted else 'no'} output={output_path}"
    )
    for issue in snapshot["issues"]:
        print(f"- {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
