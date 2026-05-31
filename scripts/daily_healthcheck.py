"""Daily health snapshot + alerting for collector, signal generation, and trading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]

from reporter.send_report import send_email
from scripts import a_share_readiness


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
QLIB_DIR = DATA_DIR / "qlib_data"
SIGNAL_DIR = DATA_DIR / "signals"
LOGS_DIR = PROJECT_DIR / "logs"
HEALTH_REPORT_DIR = Path(os.environ.get("HEALTHCHECK_REPORT_DIR", os.environ.get("REPORT_DIR", str(LOGS_DIR / "reports"))))
HEALTH_DIR = Path(os.environ.get("HEALTHCHECK_DIR", str(LOGS_DIR / "health")))
TRADE_LOG = Path(os.environ.get("TRADE_LOG", str(LOGS_DIR / "trade.log")))
DAILY_LOG = Path(os.environ.get("DAILY_LOG", str(LOGS_DIR / "daily.log")))
RETRY_LOG = Path(os.environ.get("DAILY_RETRY_LOG", str(LOGS_DIR / "daily_retry.log")))
PRED_PATH = SIGNAL_DIR / "pred_sh_latest.pkl"
NAS_HOST = os.environ.get("NAS_HOST", "")
NAS_USER = os.environ.get("NAS_USER", "")
NAS_QLIB_PATH = os.environ.get("NAS_QLIB_PATH", "/volume1/docker/quantpilot/qlib_data")
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
NAS_COLLECTOR_CONTAINER = os.environ.get("NAS_COLLECTOR_CONTAINER", "quantpilot-collector")
TARGET_DATE_LOOKBACK_DAYS = int(os.environ.get("TARGET_DATE_LOOKBACK_DAYS", "31"))
DISK_WARN_THRESHOLD = float(os.environ.get("DISK_USAGE_WARN_THRESHOLD", "0.80"))
DISK_ERROR_THRESHOLD = float(os.environ.get("DISK_USAGE_ERROR_THRESHOLD", "0.90"))
HEALTHCHECK_CAPITAL_FLOW_ENABLED = os.environ.get("HEALTHCHECK_CAPITAL_FLOW_ENABLED", "true").lower() == "true"
CAPITAL_FLOW_OVERLAY_PATH = Path(
    os.environ.get("CAPITAL_FLOW_OVERLAY_CSV", str(DATA_DIR / "output" / "futu_capital_flow_signal_overlay_latest.csv"))
)
PRETRADE_CAPITAL_FLOW_OVERLAY_PATH = Path(
    os.environ.get(
        "PRETRADE_CAPITAL_FLOW_OVERLAY_CSV",
        str(DATA_DIR / "output" / "pretrade_futu_capital_flow_signal_overlay_latest.csv"),
    )
)
CAPITAL_FLOW_ARCHIVE_DIR = Path(
    os.environ.get("A_SHARE_CAPITAL_FLOW_ARCHIVE_DIR", str(DATA_DIR / "capital_flow" / "futu"))
)
CAPITAL_FLOW_GATE_PATH = Path(
    os.environ.get("CAPITAL_FLOW_GATE_JSON", str(DATA_DIR / "output" / "futu_capital_flow_eval_latest" / "gate.json"))
)
HEALTHCHECK_MARKET_MONEY_ENABLED = os.environ.get("HEALTHCHECK_MARKET_MONEY_ENABLED", "true").lower() == "true"
MAJOR_MONEY_DIGEST_PATH = Path(
    os.environ.get("MAJOR_MONEY_DIGEST_JSON", str(DATA_DIR / "output" / "major_money_digest_latest.json"))
)
EASTMONEY_FUND_FLOW_RANK_PATH = Path(
    os.environ.get("EASTMONEY_FUND_FLOW_RANK_OUTPUT", str(DATA_DIR / "output" / "eastmoney_fund_flow_rank_latest.csv"))
)
EASTMONEY_FUND_FLOW_MIN_ROWS = int(os.environ.get("EASTMONEY_FUND_FLOW_MIN_ROWS", "1000"))
MARKET_CAPITAL_FLOW_DIR = Path(
    os.environ.get("FUTU_MARKET_FLOW_OUTPUT_DIR", str(DATA_DIR / "capital_flow" / "futu_market"))
)
MARKET_CAPITAL_FLOW_MARKETS = [
    item.strip().upper()
    for item in os.environ.get("HEALTHCHECK_MARKET_FLOW_MARKETS", "HK,US").split(",")
    if item.strip()
]
HEALTHCHECK_MARKET_FLOW_MIN_OK_RATIO = float(os.environ.get("HEALTHCHECK_MARKET_FLOW_MIN_OK_RATIO", "0.5"))

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


def latest_nas_a_share_date() -> tuple[str, str]:
    if not (NAS_HOST and NAS_USER):
        return "", ""
    try:
        value = a_share_readiness.latest_nas_a_share_date(
            nas_host=NAS_HOST,
            nas_user=NAS_USER,
            ssh_key=SSH_KEY,
            nas_qlib_path=NAS_QLIB_PATH,
        )
        return value, ""
    except Exception as exc:
        return "", str(exc)


def expected_pretrade_signal_date(now: datetime | None = None) -> tuple[str, str]:
    if not (NAS_HOST and NAS_USER):
        return "", ""
    now = now or datetime.now()
    try:
        value = a_share_readiness.previous_trade_date_via_collector(
            nas_host=NAS_HOST,
            nas_user=NAS_USER,
            ssh_key=SSH_KEY,
            today=now.strftime("%Y-%m-%d"),
            collector_container=NAS_COLLECTOR_CONTAINER,
            lookback_days=TARGET_DATE_LOOKBACK_DAYS,
        )
        return value, ""
    except Exception as exc:
        return "", str(exc)


def local_disk_status() -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(DATA_DIR)
    except FileNotFoundError:
        usage = shutil.disk_usage(DATA_DIR.parent)

    used_ratio = usage.used / usage.total if usage.total else 0.0
    return {
        "path": str(DATA_DIR),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_ratio": used_ratio,
    }


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


def _latest_date(values: Any) -> str:
    try:
        series = values.dropna().astype(str).str[:10]
    except AttributeError:
        return ""
    cleaned = sorted(value for value in series.unique().tolist() if value and value.lower() != "nan")
    return cleaned[-1] if cleaned else ""


def _read_capital_flow_overlay_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "row_count": 0,
        "signal_date": "",
        "capital_flow_latest_date": "",
        "labels": {},
        "error": "",
    }
    if not path.exists():
        return status
    try:
        import pandas as pd

        df = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return status

    status["row_count"] = int(len(df))
    status["ok"] = not df.empty and "capital_flow_label" in df.columns
    if "signal_date" in df.columns:
        status["signal_date"] = _latest_date(df["signal_date"])
    if "capital_flow_latest_date" in df.columns:
        status["capital_flow_latest_date"] = _latest_date(df["capital_flow_latest_date"])
    if "capital_flow_label" in df.columns:
        status["labels"] = {
            str(label): int(count)
            for label, count in df["capital_flow_label"].fillna("unknown").astype(str).value_counts().items()
        }
    return status


def _latest_archive_overlay_status(archive_dir: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(archive_dir),
        "exists": archive_dir.exists(),
        "latest_file": "",
        "latest_archive_date": "",
        "overlay": {},
    }
    if not archive_dir.exists():
        return status
    paths = sorted(archive_dir.glob("*_overlay.csv"))
    if not paths:
        return status
    latest = paths[-1]
    status["latest_file"] = str(latest)
    status["latest_archive_date"] = latest.name.split("_", 1)[0]
    status["overlay"] = _read_capital_flow_overlay_status(latest)
    return status


def _read_capital_flow_gate_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "overall_action": "",
        "message": "",
        "criteria": {},
        "error": "",
    }
    if not path.exists():
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return status
    status["ok"] = True
    status["overall_action"] = str(payload.get("overall_action", ""))
    status["message"] = str(payload.get("message", ""))
    criteria = payload.get("criteria", {})
    status["criteria"] = criteria if isinstance(criteria, dict) else {}
    return status


def _file_mtime_date(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _read_csv_artifact_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "row_count": 0,
        "mtime_date": _file_mtime_date(path),
        "error": "",
    }
    if not path.exists():
        return status
    try:
        import pandas as pd

        df = pd.read_csv(path)
    except Exception as exc:
        status["error"] = str(exc)
        return status
    status["row_count"] = int(len(df))
    status["ok"] = True
    return status


def _read_major_money_digest_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "flow_date": "",
        "available_market_count": 0,
        "market_count": 0,
        "markets": {},
        "mtime_date": _file_mtime_date(path),
        "error": "",
    }
    if not path.exists():
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return status

    markets = payload.get("markets", [])
    market_status: dict[str, Any] = {}
    if isinstance(markets, list):
        for item in markets:
            if isinstance(item, dict) and item.get("market"):
                market_status[str(item.get("market"))] = {
                    "available": bool(item.get("available")),
                    "ok_rows": int(item.get("ok_rows") or 0),
                    "total_rows": int(item.get("total_rows") or 0),
                    "flow_date": str(item.get("flow_date") or ""),
                    "source": str(item.get("source") or ""),
                }

    status.update(
        {
            "ok": True,
            "flow_date": str(payload.get("flow_date") or ""),
            "available_market_count": int(payload.get("available_market_count") or 0),
            "market_count": int(payload.get("market_count") or len(market_status)),
            "markets": market_status,
        }
    )
    return status


def _read_market_scan_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "ok": False,
        "status": "",
        "market": "",
        "attempted_count": 0,
        "ok_count": 0,
        "error_count": 0,
        "empty_count": 0,
        "ok_ratio": 0.0,
        "finished_at": "",
        "finished_date": "",
        "mtime_date": _file_mtime_date(path),
        "error": "",
    }
    if not path.exists():
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["error"] = str(exc)
        return status

    status_value = str(payload.get("status") or "")
    ok_count = int(payload.get("ok_count") or 0)
    attempted_count = int(payload.get("attempted_count") or 0)
    status.update(
        {
            "ok": status_value == "ok" and ok_count > 0,
            "status": status_value,
            "market": str(payload.get("market") or ""),
            "attempted_count": attempted_count,
            "ok_count": ok_count,
            "error_count": int(payload.get("error_count") or 0),
            "empty_count": int(payload.get("empty_count") or 0),
            "ok_ratio": float(payload.get("ok_ratio") or 0.0),
            "finished_at": str(payload.get("finished_at") or ""),
            "finished_date": str(payload.get("finished_at") or "")[:10],
            "message": str(payload.get("message") or ""),
            "output": str(payload.get("output") or ""),
            "latest": str(payload.get("latest") or ""),
        }
    )
    return status


def analyze_market_money_artifacts(reference_date: str = "") -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": HEALTHCHECK_MARKET_MONEY_ENABLED,
        "reference_date": reference_date,
        "a_share_rank": {},
        "digest": {},
        "market_scans": {},
        "issues": [],
    }
    if not HEALTHCHECK_MARKET_MONEY_ENABLED:
        return status

    issues: list[str] = []
    a_share_rank = _read_csv_artifact_status(EASTMONEY_FUND_FLOW_RANK_PATH)
    digest = _read_major_money_digest_status(MAJOR_MONEY_DIGEST_PATH)
    status.update({"a_share_rank": a_share_rank, "digest": digest})

    if not a_share_rank["exists"]:
        issues.append(f"Eastmoney A-share fund-flow rank missing: {a_share_rank['path']}")
    elif not a_share_rank["ok"]:
        issues.append(
            "Eastmoney A-share fund-flow rank unreadable: "
            f"path={a_share_rank['path']} error={a_share_rank.get('error') or 'unknown'}"
        )
    elif a_share_rank["row_count"] < EASTMONEY_FUND_FLOW_MIN_ROWS:
        issues.append(
            "Eastmoney A-share fund-flow rank too small: "
            f"rows={a_share_rank['row_count']} min={EASTMONEY_FUND_FLOW_MIN_ROWS}"
        )
    elif reference_date and a_share_rank.get("mtime_date") and a_share_rank["mtime_date"] < reference_date:
        issues.append(
            "Eastmoney A-share fund-flow rank stale: "
            f"mtime_date={a_share_rank['mtime_date']} reference={reference_date}"
        )

    if not digest["exists"]:
        issues.append(f"Major-money digest missing: {digest['path']}")
    elif not digest["ok"]:
        issues.append(
            "Major-money digest unreadable: "
            f"path={digest['path']} error={digest.get('error') or 'unknown'}"
        )
    else:
        a_market = digest.get("markets", {}).get("A", {})
        if not a_market.get("available"):
            issues.append("Major-money digest missing available A-share market coverage")
        digest_date = str(digest.get("flow_date") or "")
        if reference_date and digest_date and digest_date < reference_date:
            issues.append(
                "Major-money digest stale: "
                f"flow_date={digest_date} reference={reference_date}"
            )
        elif reference_date and digest.get("mtime_date") and digest["mtime_date"] < reference_date:
            issues.append(
                "Major-money digest file stale: "
                f"mtime_date={digest['mtime_date']} reference={reference_date}"
            )

    market_scans: dict[str, Any] = {}
    for market in MARKET_CAPITAL_FLOW_MARKETS:
        scan_status = _read_market_scan_status(MARKET_CAPITAL_FLOW_DIR / f"{market}_latest_status.json")
        market_scans[market] = scan_status
        if not scan_status["exists"]:
            issues.append(f"Futu market-wide capital-flow status missing for {market}: {scan_status['path']}")
        elif not scan_status["ok"]:
            issues.append(
                "Futu market-wide capital-flow scan not healthy: "
                f"market={market} status={scan_status.get('status') or 'N/A'} "
                f"ok={scan_status.get('ok_count', 0)}/{scan_status.get('attempted_count', 0)} "
                f"message={scan_status.get('message') or scan_status.get('error') or 'N/A'}"
            )
        elif reference_date and (scan_status.get("finished_date") or scan_status.get("mtime_date")) < reference_date:
            issues.append(
                "Futu market-wide capital-flow scan stale: "
                f"market={market} finished_date={scan_status.get('finished_date') or 'N/A'} "
                f"mtime_date={scan_status.get('mtime_date') or 'N/A'} reference={reference_date}"
            )
        elif scan_status.get("ok_ratio", 0.0) < HEALTHCHECK_MARKET_FLOW_MIN_OK_RATIO:
            issues.append(
                "Futu market-wide capital-flow scan coverage too low: "
                f"market={market} ok_ratio={scan_status.get('ok_ratio', 0.0):.1%} "
                f"min={HEALTHCHECK_MARKET_FLOW_MIN_OK_RATIO:.1%}"
            )
    status["market_scans"] = market_scans
    status["issues"] = issues
    return status


def analyze_capital_flow_artifacts(phase: str, reference_date: str = "") -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": HEALTHCHECK_CAPITAL_FLOW_ENABLED,
        "reference_date": reference_date,
        "daily_overlay": {},
        "pretrade_overlay": {},
        "archive": {},
        "gate": {},
        "issues": [],
    }
    if not HEALTHCHECK_CAPITAL_FLOW_ENABLED:
        return status

    issues: list[str] = []
    if phase == "nightly":
        daily_overlay = _read_capital_flow_overlay_status(CAPITAL_FLOW_OVERLAY_PATH)
        archive = _latest_archive_overlay_status(CAPITAL_FLOW_ARCHIVE_DIR)
        gate = _read_capital_flow_gate_status(CAPITAL_FLOW_GATE_PATH)
        status.update({"daily_overlay": daily_overlay, "archive": archive, "gate": gate})

        if not daily_overlay["exists"]:
            issues.append(f"Futu capital-flow latest overlay missing: {daily_overlay['path']}")
        elif not daily_overlay["ok"]:
            issues.append(
                "Futu capital-flow latest overlay unreadable or empty: "
                f"path={daily_overlay['path']} error={daily_overlay.get('error') or 'empty/missing labels'}"
            )
        elif reference_date and daily_overlay.get("signal_date") and daily_overlay["signal_date"] < reference_date:
            issues.append(
                "Futu capital-flow latest overlay stale: "
                f"signal_date={daily_overlay['signal_date']} reference={reference_date}"
            )

        if not archive["exists"] or not archive.get("latest_file"):
            issues.append(f"Futu capital-flow archive missing: {archive['path']}")
        elif reference_date and archive.get("latest_archive_date") and archive["latest_archive_date"] < reference_date.replace("-", ""):
            issues.append(
                "Futu capital-flow archive stale: "
                f"latest_archive={archive['latest_archive_date']} reference={reference_date}"
            )

        if not gate["exists"]:
            issues.append(f"Futu capital-flow promotion gate missing: {gate['path']}")
        elif not gate["ok"]:
            issues.append(
                "Futu capital-flow promotion gate unreadable: "
                f"path={gate['path']} error={gate.get('error') or 'unknown'}"
            )
        elif gate.get("overall_action") in {"review_filter", "review_boost"}:
            issues.append(f"Futu capital-flow gate requests manual review: {gate.get('overall_action')}")

    elif phase == "trade":
        pretrade_overlay = _read_capital_flow_overlay_status(PRETRADE_CAPITAL_FLOW_OVERLAY_PATH)
        status["pretrade_overlay"] = pretrade_overlay
        if not pretrade_overlay["exists"]:
            issues.append(f"Pre-trade Futu capital-flow overlay missing: {pretrade_overlay['path']}")
        elif not pretrade_overlay["ok"]:
            issues.append(
                "Pre-trade Futu capital-flow overlay unreadable or empty: "
                f"path={pretrade_overlay['path']} error={pretrade_overlay.get('error') or 'empty/missing labels'}"
            )
        elif reference_date and pretrade_overlay.get("signal_date") and pretrade_overlay["signal_date"] < reference_date:
            issues.append(
                "Pre-trade Futu capital-flow overlay stale: "
                f"signal_date={pretrade_overlay['signal_date']} reference={reference_date}"
            )

    status["issues"] = issues
    return status


def build_snapshot(
    phase: str,
    now: datetime | None = None,
    target_a_share_date: str = "",
) -> dict[str, Any]:
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")

    local_completed = latest_local_completed_date()
    local_latest = latest_local_a_share_date()
    signal_date = latest_signal_date()
    nas_completed, nas_error = latest_nas_completed_date()
    nas_latest, nas_latest_error = latest_nas_a_share_date()
    disk = local_disk_status()
    expected_signal_date = ""
    expected_signal_error = ""
    if phase in {"pretrade", "trade"}:
        expected_signal_date, expected_signal_error = expected_pretrade_signal_date(now)
    trade = analyze_trade_log(today)
    nightly_logs = analyze_daily_logs(today)
    processes = {
        "nightly_running": process_running(["python -m inference.run_daily", "run_daily.sh"]),
        "retry_watcher_running": process_running(["run_daily_when_ready.sh"]),
        "pretrade_watchdog_running": process_running(["pretrade_watchdog.py"]),
    }
    capital_flow = analyze_capital_flow_artifacts(
        phase,
        reference_date=target_a_share_date or expected_signal_date or signal_date or local_latest,
    )
    market_money = analyze_market_money_artifacts(
        reference_date=target_a_share_date or expected_signal_date or signal_date or local_latest,
    )

    overall = "ok"
    issues: list[str] = []

    if not local_latest:
        overall = _bump_level(overall, "error")
        issues.append("Local A-share snapshot missing latest instruments date")

    if disk["used_ratio"] >= DISK_ERROR_THRESHOLD:
        overall = _bump_level(overall, "error")
        issues.append(
            "Local data disk usage above error threshold: "
            f"used_ratio={disk['used_ratio']:.1%} threshold={DISK_ERROR_THRESHOLD:.0%}"
        )
    elif disk["used_ratio"] >= DISK_WARN_THRESHOLD:
        overall = _bump_level(overall, "warn")
        issues.append(
            "Local data disk usage above warning threshold: "
            f"used_ratio={disk['used_ratio']:.1%} threshold={DISK_WARN_THRESHOLD:.0%}"
        )

    if local_latest and signal_date != local_latest:
        overall = _bump_level(overall, "error")
        issues.append(f"Signal stale: signal={signal_date or 'N/A'} latest_a_share={local_latest}")

    if target_a_share_date:
        if not local_completed or local_completed < target_a_share_date:
            overall = _bump_level(overall, "error")
            issues.append(
                "Local completed snapshot below nightly target: "
                f"local_completed={local_completed or 'N/A'} target={target_a_share_date}"
            )
        if not local_latest or local_latest < target_a_share_date:
            overall = _bump_level(overall, "error")
            issues.append(
                "Local A-share snapshot below nightly target: "
                f"latest_a_share={local_latest or 'N/A'} target={target_a_share_date}"
            )
        if not signal_date or signal_date < target_a_share_date:
            overall = _bump_level(overall, "error")
            issues.append(
                "Signal output below nightly target: "
                f"signal={signal_date or 'N/A'} target={target_a_share_date}"
            )

    if expected_signal_date:
        if not local_latest or local_latest < expected_signal_date:
            overall = _bump_level(overall, "error")
            issues.append(
                "Local A-share snapshot below expected pre-trade target: "
                f"latest_a_share={local_latest or 'N/A'} expected={expected_signal_date}"
            )
        if not signal_date or signal_date < expected_signal_date:
            overall = _bump_level(overall, "error")
            issues.append(
                "Signal output below expected pre-trade target: "
                f"signal={signal_date or 'N/A'} expected={expected_signal_date}"
            )
        effective_nas_date = max(nas_completed or "", nas_latest or "")
        if effective_nas_date and effective_nas_date < expected_signal_date:
            overall = _bump_level(overall, "warn")
            issues.append(
                "NAS snapshot below expected pre-trade target: "
                f"nas_completed={nas_completed or 'N/A'} nas_latest={nas_latest or 'N/A'} expected={expected_signal_date}"
            )
    elif expected_signal_error:
        overall = _bump_level(overall, "warn")
        issues.append(f"Failed to resolve expected pre-trade signal date: {expected_signal_error}")

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
    elif nas_latest_error:
        overall = _bump_level(overall, "warn")
        issues.append(f"Failed to query NAS latest A-share snapshot: {nas_latest_error}")

    if phase == "nightly":
        target_satisfied = (
            bool(target_a_share_date)
            and bool(local_completed)
            and bool(local_latest)
            and bool(signal_date)
            and local_completed >= target_a_share_date
            and local_latest >= target_a_share_date
            and signal_date >= target_a_share_date
        )
        if target_satisfied or (not target_a_share_date and local_latest and signal_date == local_latest):
            pass
        elif processes["nightly_running"] or processes["retry_watcher_running"]:
            overall = _bump_level(overall, "warn")
            issues.append(
                "Nightly pipeline still running or waiting on retry watcher"
                + (f" (target={target_a_share_date})" if target_a_share_date else "")
            )
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

    for issue in capital_flow.get("issues", []):
        overall = _bump_level(overall, "warn")
        issues.append(issue)

    for issue in market_money.get("issues", []):
        overall = _bump_level(overall, "warn")
        issues.append(issue)

    if not issues:
        issues.append("All monitored checks passed")

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": today,
        "phase": phase,
        "target_a_share_date": target_a_share_date,
        "expected_signal_date": expected_signal_date,
        "overall_status": overall,
        "issues": issues,
        "local": {
            "completed_a_share_date": local_completed,
            "latest_a_share_date": local_latest,
            "latest_signal_date": signal_date,
            "signal_aligned": bool(local_latest) and signal_date == local_latest,
            "disk": disk,
        },
        "nas": {
            "completed_a_share_date": nas_completed,
            "latest_a_share_date": nas_latest,
            "query_error": nas_error,
            "latest_query_error": nas_latest_error,
        },
        "processes": processes,
        "nightly": nightly_logs,
        "trade": trade,
        "capital_flow": capital_flow,
        "market_money": market_money,
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
    capital_flow = snapshot.get("capital_flow", {})
    market_money = snapshot.get("market_money", {})
    gate = capital_flow.get("gate", {}) if isinstance(capital_flow, dict) else {}
    daily_overlay = capital_flow.get("daily_overlay", {}) if isinstance(capital_flow, dict) else {}
    pretrade_overlay = capital_flow.get("pretrade_overlay", {}) if isinstance(capital_flow, dict) else {}
    digest = market_money.get("digest", {}) if isinstance(market_money, dict) else {}
    a_share_rank = market_money.get("a_share_rank", {}) if isinstance(market_money, dict) else {}
    disk = local.get(
        "disk",
        {"path": str(DATA_DIR), "used_ratio": 0.0},
    )
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
      expected_signal={snapshot.get('expected_signal_date') or 'N/A'}<br>
      signal_aligned={local['signal_aligned']}<br>
      data_dir={disk['path']}<br>
      disk_used={disk['used_ratio']:.1%}</p>
      <h2>NAS State</h2>
      <p>completed_a_share={nas['completed_a_share_date'] or 'N/A'}<br>
      latest_a_share={nas.get('latest_a_share_date') or 'N/A'}<br>
      query_error={nas['query_error'] or 'N/A'}<br>
      latest_query_error={nas.get('latest_query_error') or 'N/A'}</p>
      <h2>Trade Log</h2>
      <p>starts={trade['starts']} done={trade['done']} fills={trade['order_fills']} failures={trade['order_failures']} errors={trade['errors']}</p>
      <h2>Capital Flow</h2>
      <p>enabled={capital_flow.get('enabled', False)}<br>
      reference_date={capital_flow.get('reference_date') or 'N/A'}<br>
      daily_overlay_date={daily_overlay.get('signal_date') or 'N/A'} rows={daily_overlay.get('row_count', 0)}<br>
      pretrade_overlay_date={pretrade_overlay.get('signal_date') or 'N/A'} rows={pretrade_overlay.get('row_count', 0)}<br>
      gate_action={gate.get('overall_action') or 'N/A'}</p>
      <h2>Market-Wide Major Money</h2>
      <p>enabled={market_money.get('enabled', False)}<br>
      digest_date={digest.get('flow_date') or 'N/A'} available_markets={digest.get('available_market_count', 0)}/{digest.get('market_count', 0)}<br>
      eastmoney_rows={a_share_rank.get('row_count', 0)} mtime={a_share_rank.get('mtime_date') or 'N/A'}</p>
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
    send_email(render_snapshot_html(snapshot), subject, report_dir=HEALTH_REPORT_DIR)
    stamp.write_text(snapshot["timestamp"], encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantPilot daily health snapshot")
    parser.add_argument("--phase", default="manual", choices=["manual", "nightly", "pretrade", "trade"])
    parser.add_argument("--alert-on", default="error", choices=["ok", "warn", "error"])
    parser.add_argument("--target-a-share-date", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = build_snapshot(args.phase, target_a_share_date=args.target_a_share_date)
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
