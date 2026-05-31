"""Readiness checks for the market-wide major-money email notification system."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "quantpilot_data"


def _strip_env_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def load_env_file(path: str | Path) -> dict[str, str]:
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            result[key] = _strip_env_value(value)
    return result


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def secret_present(env: dict[str, str], *, value_key: str, file_key: str) -> bool:
    if str(env.get(value_key) or "").strip():
        return True
    path_value = str(env.get(file_key) or "").strip()
    if not path_value:
        return False
    target = Path(path_value).expanduser()
    if not target.exists():
        return False
    return bool(target.read_text(encoding="utf-8", errors="replace").strip())


def _split_csv(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _active_cron_lines(crontab_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in crontab_text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def read_crontab() -> tuple[str, str]:
    try:
        result = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    except Exception as exc:
        return "", str(exc)
    if result.returncode != 0:
        return result.stdout, result.stderr.strip() or f"crontab exited {result.returncode}"
    return result.stdout, ""


def check_cron(crontab_text: str, *, project_dir: Path) -> dict[str, Any]:
    active_lines = _active_cron_lines(crontab_text)

    def has_line(*needles: str) -> bool:
        return any(all(needle in line for needle in needles) for line in active_lines)

    checks = {
        "daily_report": has_line(str(project_dir / "scripts" / "run_daily.sh")) or has_line("run_daily.sh"),
        "hk_market_scan": has_line("FUTU_MARKET_FLOW_MARKETS=HK", "run_market_capital_flow.sh"),
        "us_market_scan": has_line("FUTU_MARKET_FLOW_MARKETS=US", "run_market_capital_flow.sh"),
    }
    issues = []
    if not checks["daily_report"]:
        issues.append("Cron missing daily report job: scripts/run_daily.sh")
    if not checks["hk_market_scan"]:
        issues.append("Cron missing HK market-wide major-money scan: FUTU_MARKET_FLOW_MARKETS=HK")
    if not checks["us_market_scan"]:
        issues.append("Cron missing US market-wide major-money scan: FUTU_MARKET_FLOW_MARKETS=US")
    return {
        "ok": not issues,
        "checks": checks,
        "active_line_count": len(active_lines),
        "issues": issues,
    }


def check_email_config(env: dict[str, str], *, reporter_env_path: Path) -> dict[str, Any]:
    method = str(env.get("REPORT_DELIVERY_METHOD") or "auto").lower()
    report_to = env.get("REPORT_TO", "")
    smtp_ready = all(
        env.get(key, "")
        for key in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "REPORT_TO"]
    )
    mail_app_allowed = method in {"auto", "mailapp"} and _truthy(env.get("MAIL_APP_FALLBACK", "true"))
    sendmail_allowed = method in {"auto", "sendmail", "smtp"} and _truthy(env.get("SENDMAIL_FALLBACK", "true"))
    sendmail_path = shutil.which("sendmail")
    delivery_ready = bool(report_to) and (
        smtp_ready
        or (mail_app_allowed and sys.platform == "darwin")
        or (sendmail_allowed and bool(sendmail_path))
    )
    issues = []
    if not reporter_env_path.exists():
        issues.append(f"Reporter env file missing: {reporter_env_path}")
    if not report_to:
        issues.append("Email recipient missing: REPORT_TO")
    if method == "smtp" and not smtp_ready:
        issues.append("SMTP delivery selected but SMTP_HOST/PORT/USER/PASSWORD/REPORT_TO is incomplete")
    if not delivery_ready:
        issues.append("No usable email delivery path detected for the daily report")
    return {
        "ok": not issues,
        "reporter_env_path": str(reporter_env_path),
        "reporter_env_exists": reporter_env_path.exists(),
        "delivery_method": method,
        "report_to_present": bool(report_to),
        "smtp_ready": smtp_ready,
        "mail_app_allowed": mail_app_allowed,
        "sendmail_allowed": sendmail_allowed,
        "sendmail_path": sendmail_path or "",
        "issues": issues,
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, f"missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "not a JSON object"
    return payload, ""


def check_digest(path: Path, *, expected_markets: list[str]) -> dict[str, Any]:
    payload, error = _read_json(path)
    markets = payload.get("markets") if isinstance(payload, dict) else []
    by_market: dict[str, dict[str, Any]] = {}
    if isinstance(markets, list):
        for item in markets:
            if isinstance(item, dict) and item.get("market"):
                by_market[str(item["market"]).upper()] = item

    issues = []
    if error:
        issues.append(f"Major-money digest unreadable: {error}")
    for market in expected_markets:
        row = by_market.get(market)
        if not row:
            issues.append(f"Major-money digest missing expected market: {market}")
        elif not row.get("available"):
            issues.append(f"Major-money digest expected market unavailable: {market}")

    return {
        "ok": not issues,
        "path": str(path),
        "exists": path.exists(),
        "flow_date": str(payload.get("flow_date") or "") if payload else "",
        "market_count": int(payload.get("market_count") or len(by_market)) if payload else 0,
        "available_market_count": int(payload.get("available_market_count") or 0) if payload else 0,
        "markets": {
            market: {
                "available": bool(row.get("available")),
                "source": str(row.get("source") or ""),
                "ok_rows": int(row.get("ok_rows") or 0),
                "total_rows": int(row.get("total_rows") or 0),
                "entry_count": int(row.get("entry_count") or 0),
                "exit_count": int(row.get("exit_count") or 0),
            }
            for market, row in by_market.items()
        },
        "issues": issues,
    }


def check_us_otc_proxy(env: dict[str, str], *, data_dir: Path, expected_markets: list[str]) -> dict[str, Any]:
    output_dir = Path(env.get("US_OTC_PROXY_FLOW_OUTPUT_DIR", str(data_dir / "capital_flow" / "us_otc_proxy"))).expanduser()
    universe_path = Path(
        env.get(
            "US_OTC_PROXY_FLOW_UNIVERSE_CSV",
            str(data_dir / "capital_flow" / "futu_market" / "US_latest_source_universe.csv"),
        )
    ).expanduser()
    status_path = output_dir / "US_OTC_latest_status.json"
    flow_path = output_dir / "US_OTC_latest_flow.csv"
    enabled = _truthy(env.get("ENABLE_US_OTC_PROXY_FLOW", "false"))
    provider = str(env.get("US_OTC_PROXY_FLOW_PROVIDER", "polygon")).lower()
    api_key_file = str(env.get("POLYGON_API_KEY_FILE", "")).strip()
    api_key_present = secret_present(env, value_key="POLYGON_API_KEY", file_key="POLYGON_API_KEY_FILE")

    issues = []
    if "US_OTC" in expected_markets:
        if not enabled:
            issues.append("US OTC/Pink proxy disabled: set ENABLE_US_OTC_PROXY_FLOW=true")
        elif provider == "polygon" and not api_key_present:
            issues.append("US OTC/Pink proxy missing POLYGON_API_KEY or POLYGON_API_KEY_FILE")
        elif not universe_path.exists():
            issues.append(f"US OTC/Pink proxy universe missing: {universe_path}")
        elif not status_path.exists():
            issues.append(f"US OTC/Pink proxy status missing: {status_path}")
        elif not flow_path.exists():
            issues.append(f"US OTC/Pink proxy flow missing: {flow_path}")

    status_payload, status_error = _read_json(status_path) if status_path.exists() else ({}, "")
    status_value = str(status_payload.get("status") or "")
    ok_count = int(status_payload.get("ok_count") or 0) if status_payload else 0
    attempted_count = int(status_payload.get("attempted_count") or 0) if status_payload else 0
    if "US_OTC" in expected_markets and enabled and status_payload:
        if status_error:
            issues.append(f"US OTC/Pink proxy status unreadable: {status_error}")
        elif status_value != "ok" or ok_count <= 0:
            issues.append(
                "US OTC/Pink proxy scan not healthy: "
                f"status={status_value or 'N/A'} ok={ok_count}/{attempted_count}"
            )

    return {
        "ok": not issues,
        "expected": "US_OTC" in expected_markets,
        "enabled": enabled,
        "provider": provider,
        "api_key_present": api_key_present,
        "api_key_file": api_key_file,
        "universe_path": str(universe_path),
        "universe_exists": universe_path.exists(),
        "status_path": str(status_path),
        "status_exists": status_path.exists(),
        "flow_path": str(flow_path),
        "flow_exists": flow_path.exists(),
        "status": status_value,
        "attempted_count": attempted_count,
        "ok_count": ok_count,
        "issues": issues,
    }


def build_readiness_snapshot(
    *,
    project_dir: Path = PROJECT_DIR,
    crontab_text: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    project_env = load_env_file(project_dir / ".env")
    merged_env = {**project_env, **os.environ}
    if env:
        merged_env.update(env)

    data_dir = Path(merged_env.get("DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    reporter_env_path = Path(
        merged_env.get("REPORTER_ENV_FILE", str(project_dir / "reporter" / ".env"))
    ).expanduser()
    reporter_env = load_env_file(reporter_env_path)
    email_env = {**reporter_env, **merged_env}
    expected_markets = _split_csv(merged_env.get("MAJOR_MONEY_EXPECTED_MARKETS", "A,HK,US,US_OTC"))
    digest_path = Path(
        merged_env.get("MAJOR_MONEY_DIGEST_JSON", str(data_dir / "output" / "major_money_digest_latest.json"))
    ).expanduser()

    cron_error = ""
    if crontab_text is None:
        crontab_text, cron_error = read_crontab()

    checks = {
        "cron": check_cron(crontab_text or "", project_dir=project_dir),
        "email": check_email_config(email_env, reporter_env_path=reporter_env_path),
        "digest": check_digest(digest_path, expected_markets=expected_markets),
        "us_otc_proxy": check_us_otc_proxy(merged_env, data_dir=data_dir, expected_markets=expected_markets),
    }
    issues = []
    if cron_error:
        issues.append(f"Could not read crontab: {cron_error}")
    for section in checks.values():
        issues.extend(section.get("issues") or [])

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ok": not issues,
        "project_dir": str(project_dir),
        "data_dir": str(data_dir),
        "expected_markets": expected_markets,
        "checks": checks,
        "issues": issues,
    }


def _print_human(snapshot: dict[str, Any]) -> None:
    print(f"major-money readiness: {'ok' if snapshot['ok'] else 'not-ready'}")
    print(f"project_dir={snapshot['project_dir']}")
    print(f"data_dir={snapshot['data_dir']}")
    print(f"expected_markets={','.join(snapshot['expected_markets'])}")
    if snapshot["issues"]:
        print("issues:")
        for issue in snapshot["issues"]:
            print(f"- {issue}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check market-wide major-money notification readiness.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact human summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = build_readiness_snapshot()
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        _print_human(snapshot)
    return 0 if snapshot["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
