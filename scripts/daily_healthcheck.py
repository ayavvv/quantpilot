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
from scripts import a_share_readiness, major_money_readiness


def _secret_present(value: str = "", file_path: str = "") -> bool:
    if str(value or "").strip():
        return True
    if not file_path:
        return False
    target = Path(file_path).expanduser()
    if not target.exists():
        return False
    return bool(target.read_text(encoding="utf-8", errors="replace").strip())


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
HEALTHCHECK_MAJOR_MONEY_READINESS_ENABLED = (
    os.environ.get("HEALTHCHECK_MAJOR_MONEY_READINESS_ENABLED", "true").lower() == "true"
)
MAJOR_MONEY_DIGEST_PATH = Path(
    os.environ.get("MAJOR_MONEY_DIGEST_JSON", str(DATA_DIR / "output" / "major_money_digest_latest.json"))
)
MAJOR_MONEY_DIGEST_ARCHIVE_DIR_VALUE = os.environ.get(
    "MAJOR_MONEY_DIGEST_ARCHIVE_DIR",
    str(DATA_DIR / "output" / "major_money_digest"),
).strip()
MAJOR_MONEY_DIGEST_ARCHIVE_DIR = (
    Path(MAJOR_MONEY_DIGEST_ARCHIVE_DIR_VALUE).expanduser()
    if MAJOR_MONEY_DIGEST_ARCHIVE_DIR_VALUE
    else None
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
HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO = float(os.environ.get("HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO", "0.05"))
MAJOR_MONEY_EXPECTED_MARKETS = [
    item.strip().upper()
    for item in os.environ.get("MAJOR_MONEY_EXPECTED_MARKETS", "A,HK,US,US_OTC").split(",")
    if item.strip()
]
US_OTC_PROXY_FLOW_ENABLED = os.environ.get("ENABLE_US_OTC_PROXY_FLOW", "false").lower() == "true"
US_OTC_PROXY_FLOW_PROVIDER = os.environ.get("US_OTC_PROXY_FLOW_PROVIDER", "polygon")
US_OTC_PROXY_FLOW_OUTPUT_DIR = Path(
    os.environ.get("US_OTC_PROXY_FLOW_OUTPUT_DIR", str(DATA_DIR / "capital_flow" / "us_otc_proxy"))
)
US_OTC_PROXY_FLOW_UNIVERSE_CSV = Path(
    os.environ.get(
        "US_OTC_PROXY_FLOW_UNIVERSE_CSV",
        str(DATA_DIR / "capital_flow" / "futu_market" / "US_latest_source_universe.csv"),
    )
)
POLYGON_API_KEY_FILE = os.environ.get("POLYGON_API_KEY_FILE", "")
POLYGON_API_KEY_PRESENT = _secret_present(os.environ.get("POLYGON_API_KEY", ""), POLYGON_API_KEY_FILE)

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


def _readiness_issue_already_covered(issue: str, existing_issues: set[str]) -> bool:
    if issue in existing_issues:
        return True
    if issue.startswith("Major-money digest expected market unavailable:"):
        market = issue.rsplit(":", 1)[-1].strip()
        return any(
            "Major-money digest missing available expected market coverage" in existing and market in existing
            for existing in existing_issues
        )
    if issue.startswith("US OTC/Pink proxy disabled:"):
        return any("US OTC/Pink proxy flow disabled:" in existing for existing in existing_issues)
    if issue.startswith("US OTC/Pink proxy missing POLYGON_API_KEY"):
        return any("US OTC/Pink" in existing and "POLYGON_API_KEY" in existing for existing in existing_issues)
    return False


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


def _date_tag(value: str) -> str:
    text = str(value or "")[:10]
    if len(text) == 10 and text[4] == "-" and text[7] == "-" and text.replace("-", "").isdigit():
        return text.replace("-", "")
    return ""


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
                    "empty_rows": int(item.get("empty_rows") or 0),
                    "error_rows": int(item.get("error_rows") or 0),
                    "non_ok_rows": int(item.get("non_ok_rows") or 0),
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


def _read_major_money_digest_archive_status(archive_dir: Path | None, *, flow_date: str) -> dict[str, Any]:
    enabled = archive_dir is not None
    date_tag = _date_tag(flow_date)
    json_path = archive_dir / f"{date_tag}_major_money_digest.json" if enabled and date_tag else Path("")
    csv_path = archive_dir / f"{date_tag}_major_money_digest.csv" if enabled and date_tag else Path("")
    status: dict[str, Any] = {
        "enabled": enabled,
        "path": str(archive_dir) if archive_dir is not None else "",
        "exists": archive_dir.exists() if archive_dir is not None else False,
        "ok": False,
        "flow_date": flow_date,
        "date_tag": date_tag,
        "json_path": str(json_path) if date_tag else "",
        "json_exists": json_path.exists() if date_tag else False,
        "json_flow_date": "",
        "csv_path": str(csv_path) if date_tag else "",
        "csv_exists": csv_path.exists() if date_tag else False,
        "error": "",
        "issues": [],
    }
    if not enabled:
        status["ok"] = True
        return status

    issues: list[str] = []
    if not date_tag:
        issues.append("Major-money digest archive date unavailable: digest flow_date missing")
    elif not archive_dir.exists():
        issues.append(f"Major-money digest archive directory missing: {archive_dir}")
    else:
        if not json_path.exists():
            issues.append(f"Major-money digest archive JSON missing: {json_path}")
        else:
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                status["error"] = str(exc)
                issues.append(f"Major-money digest archive JSON unreadable: {exc}")
            else:
                if not isinstance(payload, dict):
                    status["error"] = "not a JSON object"
                    issues.append("Major-money digest archive JSON unreadable: not a JSON object")
                else:
                    status["json_flow_date"] = str(payload.get("flow_date") or "")
                    if status["json_flow_date"] != flow_date:
                        issues.append(
                            "Major-money digest archive JSON flow_date mismatch: "
                            f"archive={status['json_flow_date'] or 'N/A'} digest={flow_date}"
                        )
        if not csv_path.exists():
            issues.append(f"Major-money digest archive CSV missing: {csv_path}")
        elif csv_path.stat().st_size <= 0:
            issues.append(f"Major-money digest archive CSV empty: {csv_path}")

    status["ok"] = not issues
    status["issues"] = issues
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
        "source_exchange_types": {},
        "selected_exchange_types": {},
        "excluded_exchange_types": {},
        "unsupported_exchange_types": {},
        "status_by_exchange_type": {},
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
            "source_exchange_types": (
                payload.get("source_exchange_types") if isinstance(payload.get("source_exchange_types"), dict) else {}
            ),
            "selected_exchange_types": (
                payload.get("selected_exchange_types") if isinstance(payload.get("selected_exchange_types"), dict) else {}
            ),
            "excluded_exchange_types": (
                payload.get("excluded_exchange_types") if isinstance(payload.get("excluded_exchange_types"), dict) else {}
            ),
            "unsupported_exchange_types": (
                payload.get("unsupported_exchange_types")
                if isinstance(payload.get("unsupported_exchange_types"), dict)
                else {}
            ),
            "status_by_exchange_type": (
                payload.get("status_by_exchange_type") if isinstance(payload.get("status_by_exchange_type"), dict) else {}
            ),
            "finished_at": str(payload.get("finished_at") or ""),
            "finished_date": str(payload.get("finished_at") or "")[:10],
            "message": str(payload.get("message") or ""),
            "output": str(payload.get("output") or ""),
            "latest": str(payload.get("latest") or ""),
        }
    )
    return status


def _read_us_otc_proxy_status() -> dict[str, Any]:
    flow_path = US_OTC_PROXY_FLOW_OUTPUT_DIR / "US_OTC_latest_flow.csv"
    status_path = US_OTC_PROXY_FLOW_OUTPUT_DIR / "US_OTC_latest_status.json"
    status: dict[str, Any] = {
        "enabled": US_OTC_PROXY_FLOW_ENABLED,
        "provider": US_OTC_PROXY_FLOW_PROVIDER,
        "api_key_present": POLYGON_API_KEY_PRESENT,
        "universe_path": str(US_OTC_PROXY_FLOW_UNIVERSE_CSV),
        "universe_exists": US_OTC_PROXY_FLOW_UNIVERSE_CSV.exists(),
        "flow_path": str(flow_path),
        "flow_exists": flow_path.exists(),
        "status_path": str(status_path),
        "status_exists": status_path.exists(),
        "ok": False,
        "status": "",
        "date": "",
        "finished_at": "",
        "finished_date": "",
        "attempted_count": 0,
        "ok_count": 0,
        "error_count": 0,
        "empty_count": 0,
        "ok_ratio": 0.0,
        "message": "",
        "error": "",
    }
    if not status_path.exists():
        return status
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
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
            "date": str(payload.get("date") or ""),
            "finished_at": str(payload.get("finished_at") or ""),
            "finished_date": str(payload.get("finished_at") or "")[:10],
            "attempted_count": attempted_count,
            "ok_count": ok_count,
            "error_count": int(payload.get("error_count") or 0),
            "empty_count": int(payload.get("empty_count") or 0),
            "ok_ratio": float(payload.get("ok_ratio") or 0.0),
            "message": str(payload.get("message") or ""),
        }
    )
    return status


def analyze_market_money_artifacts(reference_date: str = "") -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": HEALTHCHECK_MARKET_MONEY_ENABLED,
        "reference_date": reference_date,
        "expected_markets": MAJOR_MONEY_EXPECTED_MARKETS,
        "a_share_rank": {},
        "digest": {},
        "digest_archive": {},
        "market_scans": {},
        "us_otc_proxy": {},
        "issues": [],
    }
    if not HEALTHCHECK_MARKET_MONEY_ENABLED:
        return status

    issues: list[str] = []
    a_share_rank = _read_csv_artifact_status(EASTMONEY_FUND_FLOW_RANK_PATH)
    digest = _read_major_money_digest_status(MAJOR_MONEY_DIGEST_PATH)
    digest_archive = _read_major_money_digest_archive_status(
        MAJOR_MONEY_DIGEST_ARCHIVE_DIR,
        flow_date=str(digest.get("flow_date") or ""),
    )
    us_otc_proxy = _read_us_otc_proxy_status() if "US_OTC" in MAJOR_MONEY_EXPECTED_MARKETS else {}
    status.update(
        {
            "a_share_rank": a_share_rank,
            "digest": digest,
            "digest_archive": digest_archive,
            "us_otc_proxy": us_otc_proxy,
        }
    )

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
        digest_markets = digest.get("markets", {})
        missing_expected_markets = sorted(
            market for market in MAJOR_MONEY_EXPECTED_MARKETS if market not in digest_markets
        )
        if missing_expected_markets:
            issues.append(
                "Major-money digest missing expected market rows: "
                + ",".join(missing_expected_markets)
            )
        a_market = digest.get("markets", {}).get("A", {})
        if not a_market.get("available"):
            issues.append("Major-money digest missing available A-share market coverage")
        unavailable_markets = sorted(
            market
            for market, market_status in digest.get("markets", {}).items()
            if isinstance(market_status, dict) and not market_status.get("available")
        )
        if unavailable_markets:
            issues.append(
                "Major-money digest missing available expected market coverage: "
                + ",".join(unavailable_markets)
            )
        for market, market_status in sorted(digest.get("markets", {}).items()):
            if not isinstance(market_status, dict) or not market_status.get("available"):
                continue
            total_rows = int(market_status.get("total_rows") or 0)
            non_ok_rows = int(market_status.get("non_ok_rows") or 0)
            if total_rows <= 0 or non_ok_rows <= 0:
                continue
            non_ok_ratio = non_ok_rows / total_rows
            if non_ok_ratio > HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO:
                issues.append(
                    "Major-money digest partial source coverage: "
                    f"market={market} non_ok={non_ok_rows}/{total_rows} ({non_ok_ratio:.1%}) "
                    f"empty={int(market_status.get('empty_rows') or 0)} "
                    f"error={int(market_status.get('error_rows') or 0)} "
                    f"max={HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO:.1%}"
                )
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
        issues.extend(digest_archive.get("issues") or [])

    if "US_OTC" in MAJOR_MONEY_EXPECTED_MARKETS:
        if not us_otc_proxy.get("enabled"):
            issues.append(
                "US OTC/Pink proxy flow disabled: set ENABLE_US_OTC_PROXY_FLOW=true and configure "
                "POLYGON_API_KEY or POLYGON_API_KEY_FILE"
            )
        elif us_otc_proxy.get("provider") == "polygon" and not us_otc_proxy.get("api_key_present"):
            issues.append("US OTC/Pink proxy flow missing POLYGON_API_KEY or POLYGON_API_KEY_FILE")
        elif not us_otc_proxy.get("universe_exists"):
            issues.append(
                "US OTC/Pink proxy universe missing: "
                f"{us_otc_proxy.get('universe_path')}"
            )
        elif not us_otc_proxy.get("status_exists"):
            issues.append(
                "US OTC/Pink proxy status missing: "
                f"{us_otc_proxy.get('status_path')}"
            )
        elif not us_otc_proxy.get("ok"):
            issues.append(
                "US OTC/Pink proxy scan not healthy: "
                f"status={us_otc_proxy.get('status') or 'N/A'} "
                f"ok={us_otc_proxy.get('ok_count', 0)}/{us_otc_proxy.get('attempted_count', 0)} "
                f"message={us_otc_proxy.get('message') or us_otc_proxy.get('error') or 'N/A'}"
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


def analyze_major_money_readiness() -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": HEALTHCHECK_MAJOR_MONEY_READINESS_ENABLED,
        "ok": False,
        "expected_markets": [],
        "checks": {},
        "issues": [],
        "error": "",
    }
    if not HEALTHCHECK_MAJOR_MONEY_READINESS_ENABLED:
        return status
    try:
        snapshot = major_money_readiness.build_readiness_snapshot(project_dir=PROJECT_DIR)
    except Exception as exc:
        status["issues"] = [f"Major-money readiness check failed: {exc}"]
        status["error"] = str(exc)
        return status
    return {
        "enabled": True,
        "ok": bool(snapshot.get("ok")),
        "expected_markets": snapshot.get("expected_markets") or [],
        "checks": snapshot.get("checks") or {},
        "issues": snapshot.get("issues") or [],
        "error": "",
    }


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
    major_money_readiness_status = analyze_major_money_readiness() if phase == "nightly" else {
        "enabled": HEALTHCHECK_MAJOR_MONEY_READINESS_ENABLED,
        "ok": False,
        "expected_markets": [],
        "checks": {},
        "issues": [],
        "error": "",
    }

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

    existing_issues = set(issues)
    for issue in major_money_readiness_status.get("issues", []):
        if _readiness_issue_already_covered(str(issue), existing_issues):
            continue
        overall = _bump_level(overall, "warn")
        issues.append(f"Major-money readiness: {issue}")
        existing_issues.add(str(issue))

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
        "major_money_readiness": major_money_readiness_status,
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
    readiness = snapshot.get("major_money_readiness", {})
    gate = capital_flow.get("gate", {}) if isinstance(capital_flow, dict) else {}
    daily_overlay = capital_flow.get("daily_overlay", {}) if isinstance(capital_flow, dict) else {}
    pretrade_overlay = capital_flow.get("pretrade_overlay", {}) if isinstance(capital_flow, dict) else {}
    digest = market_money.get("digest", {}) if isinstance(market_money, dict) else {}
    digest_archive = market_money.get("digest_archive", {}) if isinstance(market_money, dict) else {}
    a_share_rank = market_money.get("a_share_rank", {}) if isinstance(market_money, dict) else {}
    readiness_checks = readiness.get("checks", {}) if isinstance(readiness, dict) else {}
    cron_readiness = readiness_checks.get("cron", {}) if isinstance(readiness_checks, dict) else {}
    email_readiness = readiness_checks.get("email", {}) if isinstance(readiness_checks, dict) else {}
    archive_readiness = readiness_checks.get("digest_archive", {}) if isinstance(readiness_checks, dict) else {}
    otc_readiness = readiness_checks.get("us_otc_proxy", {}) if isinstance(readiness_checks, dict) else {}
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
      digest_archive_enabled={digest_archive.get('enabled', False)} digest_archive_ok={digest_archive.get('ok', False)} archive_date={digest_archive.get('date_tag') or 'N/A'}<br>
      eastmoney_rows={a_share_rank.get('row_count', 0)} mtime={a_share_rank.get('mtime_date') or 'N/A'}</p>
      <h2>Major-Money Notification Readiness</h2>
      <p>enabled={readiness.get('enabled', False)} ok={readiness.get('ok', False)}<br>
      expected_markets={','.join(readiness.get('expected_markets') or []) or 'N/A'}<br>
      cron_ok={cron_readiness.get('ok', False)} email_ok={email_readiness.get('ok', False)} archive_ok={archive_readiness.get('ok', False)}<br>
      us_otc_enabled={otc_readiness.get('enabled', False)} us_otc_ok={otc_readiness.get('ok', False)}<br>
      us_otc_status={otc_readiness.get('status') or 'N/A'} us_otc_ok_count={otc_readiness.get('ok_count', 0)}</p>
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
