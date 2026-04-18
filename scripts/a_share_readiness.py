"""Helpers for checking NAS A-share readiness from the host pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from market_scope import a_share_model_prefixes, code_matches_prefixes


def _run_ssh_command(
    *,
    nas_host: str,
    nas_user: str,
    ssh_key: str,
    remote_command: str,
) -> str:
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"{nas_user}@{nas_host}",
        remote_command,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ssh failed ({result.returncode}): {stderr}")
    return result.stdout.strip()


def _last_non_empty_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _last_date_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", line.strip())]
    return lines[-1] if lines else _last_non_empty_line(output)


def latest_nas_a_share_date(
    *,
    nas_host: str,
    nas_user: str,
    ssh_key: str,
    nas_qlib_path: str,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    prefixes = prefixes or a_share_model_prefixes()
    prefix_arg = ",".join(prefixes)
    script = """
from pathlib import Path
import sys

latest = ""
prefixes = tuple(part for part in sys.argv[2].split(",") if part)
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = line.strip().split("\\t")
    if len(parts) < 3:
        continue
    code, _, end_date = parts[:3]
    if code.startswith(prefixes) and end_date > latest:
        latest = end_date
print(latest)
""".strip()
    remote_command = (
        f"python3 -c {shlex.quote(script)} "
        f"{shlex.quote(f'{nas_qlib_path}/instruments/all.txt')} "
        f"{shlex.quote(prefix_arg)}"
    )
    return _last_date_line(
        _run_ssh_command(
            nas_host=nas_host,
            nas_user=nas_user,
            ssh_key=ssh_key,
            remote_command=remote_command,
        )
    )


def latest_nas_a_share_completed_date(
    *,
    nas_host: str,
    nas_user: str,
    ssh_key: str,
    nas_qlib_path: str,
) -> str:
    script = """
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    data = json.loads(path.read_text())
    print(data.get("last_completed_trade_date", ""))
""".strip()
    remote_command = (
        f"python3 -c {shlex.quote(script)} "
        f"{shlex.quote(f'{nas_qlib_path}/metadata/a_share_sync_status.json')}"
    )
    return _last_date_line(
        _run_ssh_command(
            nas_host=nas_host,
            nas_user=nas_user,
            ssh_key=ssh_key,
            remote_command=remote_command,
        )
    )


def latest_trade_date_via_collector(
    *,
    nas_host: str,
    nas_user: str,
    ssh_key: str,
    today: str,
    collector_container: str = "quantpilot-collector",
    lookback_days: int = 31,
) -> str:
    script = """
import sys
from datetime import datetime, timedelta

import baostock as bs

today = sys.argv[1]
lookback_days = int(sys.argv[2])
start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
lg = bs.login()
if lg.error_code != "0":
    raise SystemExit(f"baostock login failed: {lg.error_msg}")
try:
    rs = bs.query_trade_dates(start_date=start, end_date=today)
    if rs.error_code != "0":
        raise SystemExit(f"query_trade_dates error: {rs.error_msg}")
    field_map = {name: idx for idx, name in enumerate(rs.fields)}
    cal_idx = field_map["calendar_date"]
    trade_idx = field_map["is_trading_day"]
    dates = []
    while rs.next():
        row = rs.get_row_data()
        if row[trade_idx] == "1":
            dates.append(row[cal_idx])
    print(dates[-1] if dates else "")
finally:
    bs.logout()
""".strip()
    remote_command = (
        f"sudo /usr/local/bin/docker exec {shlex.quote(collector_container)} "
        f"python -c {shlex.quote(script)} {shlex.quote(today)} {lookback_days}"
    )
    return _last_date_line(
        _run_ssh_command(
            nas_host=nas_host,
            nas_user=nas_user,
            ssh_key=ssh_key,
            remote_command=remote_command,
        )
    )


def previous_trade_date_via_collector(
    *,
    nas_host: str,
    nas_user: str,
    ssh_key: str,
    today: str,
    collector_container: str = "quantpilot-collector",
    lookback_days: int = 31,
) -> str:
    script = """
import sys
from datetime import datetime, timedelta

import baostock as bs

today = datetime.strptime(sys.argv[1], "%Y-%m-%d")
lookback_days = int(sys.argv[2])
end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
start_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
lg = bs.login()
if lg.error_code != "0":
    raise SystemExit(f"baostock login failed: {lg.error_msg}")
try:
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        raise SystemExit(f"query_trade_dates error: {rs.error_msg}")
    field_map = {name: idx for idx, name in enumerate(rs.fields)}
    cal_idx = field_map["calendar_date"]
    trade_idx = field_map["is_trading_day"]
    dates = []
    while rs.next():
        row = rs.get_row_data()
        if row[trade_idx] == "1":
            dates.append(row[cal_idx])
    print(dates[-1] if dates else "")
finally:
    bs.logout()
""".strip()
    remote_command = (
        f"sudo /usr/local/bin/docker exec {shlex.quote(collector_container)} "
        f"python -c {shlex.quote(script)} {shlex.quote(today)} {lookback_days}"
    )
    return _last_date_line(
        _run_ssh_command(
            nas_host=nas_host,
            nas_user=nas_user,
            ssh_key=ssh_key,
            remote_command=remote_command,
        )
    )


def is_a_share_ready(latest_date: str, target_date: str) -> bool:
    return bool(latest_date) and latest_date >= target_date


def latest_a_share_date_from_instruments(
    instruments_path: str | Path,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    prefixes = prefixes or a_share_model_prefixes()
    latest = ""
    for line in Path(instruments_path).read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, _, end_date = parts[:3]
        if code_matches_prefixes(code, prefixes) and end_date > latest:
            latest = end_date
    return latest


def latest_completed_a_share_date_from_status(status_path: str | Path) -> str:
    path = Path(status_path)
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    completed = data.get("last_completed_trade_date", "")
    return completed if isinstance(completed, str) else ""


def latest_signal_date_from_prediction(pred_path: str | Path) -> str:
    path = Path(pred_path)
    if not path.exists():
        return ""

    with path.open("rb") as handle:
        pred = pickle.load(handle)

    if not hasattr(pred, "index"):
        return ""

    try:
        dates = sorted(pred.index.get_level_values("datetime").unique())
    except (KeyError, AttributeError, TypeError, ValueError):
        return ""

    if not dates:
        return ""

    latest = dates[-1]
    if hasattr(latest, "strftime"):
        return latest.strftime("%Y-%m-%d")
    return str(latest)


def validate_staged_qlib_snapshot(
    *,
    qlib_dir: str | Path,
    expected_target_date: str,
    prefixes: tuple[str, ...] | None = None,
    allow_metadata_lag: bool = False,
) -> tuple[str, str]:
    qlib_path = Path(qlib_dir)
    prefixes = prefixes or a_share_model_prefixes()
    completed_date = latest_completed_a_share_date_from_status(
        qlib_path / "metadata" / "a_share_sync_status.json"
    )
    if not allow_metadata_lag and completed_date != expected_target_date:
        raise RuntimeError(
            "Staged NAS completion metadata mismatch: "
            f"completed_a_share={completed_date or 'N/A'}, expected={expected_target_date}"
        )

    latest_instruments_date = latest_a_share_date_from_instruments(
        qlib_path / "instruments" / "all.txt",
        prefixes=prefixes,
    )
    if latest_instruments_date != expected_target_date:
        raise RuntimeError(
            "Staged NAS instruments mismatch: "
            f"latest_a_share={latest_instruments_date or 'N/A'}, expected={expected_target_date}"
        )

    if allow_metadata_lag and (not completed_date or completed_date < expected_target_date):
        completed_date = latest_instruments_date

    return completed_date, latest_instruments_date


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAS A-share readiness helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--nas-host", required=True)
    common.add_argument("--nas-user", required=True)
    common.add_argument("--ssh-key", required=True)

    latest_parser = subparsers.add_parser("nas-latest-date", parents=[common])
    latest_parser.add_argument("--nas-qlib-path", required=True)

    completed_parser = subparsers.add_parser("nas-completed-date", parents=[common])
    completed_parser.add_argument("--nas-qlib-path", required=True)

    target_parser = subparsers.add_parser("nas-target-date", parents=[common])
    target_parser.add_argument(
        "--today",
        default=datetime.now().strftime("%Y-%m-%d"),
    )
    target_parser.add_argument(
        "--collector-container",
        default="quantpilot-collector",
    )
    target_parser.add_argument(
        "--lookback-days",
        type=int,
        default=31,
    )

    signal_parser = subparsers.add_parser("pred-latest-signal-date")
    signal_parser.add_argument("--pred-path", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.command == "nas-latest-date":
        print(
            latest_nas_a_share_date(
                nas_host=args.nas_host,
                nas_user=args.nas_user,
                ssh_key=args.ssh_key,
                nas_qlib_path=args.nas_qlib_path,
            )
        )
        return 0

    if args.command == "nas-completed-date":
        print(
            latest_nas_a_share_completed_date(
                nas_host=args.nas_host,
                nas_user=args.nas_user,
                ssh_key=args.ssh_key,
                nas_qlib_path=args.nas_qlib_path,
            )
        )
        return 0

    if args.command == "nas-target-date":
        print(
            latest_trade_date_via_collector(
                nas_host=args.nas_host,
                nas_user=args.nas_user,
                ssh_key=args.ssh_key,
                today=args.today,
                collector_container=args.collector_container,
                lookback_days=args.lookback_days,
            )
        )
        return 0

    if args.command == "pred-latest-signal-date":
        print(latest_signal_date_from_prediction(args.pred_path))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
