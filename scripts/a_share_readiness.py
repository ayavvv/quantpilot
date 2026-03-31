"""Helpers for checking NAS A-share readiness from the host pipeline."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from datetime import datetime


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
) -> str:
    script = """
from pathlib import Path
import sys

latest = ""
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = line.strip().split("\\t")
    if len(parts) < 3:
        continue
    code, _, end_date = parts[:3]
    if code.startswith(("SH.", "SZ.")) and end_date > latest:
        latest = end_date
print(latest)
""".strip()
    remote_command = (
        f"python3 -c {shlex.quote(script)} "
        f"{shlex.quote(f'{nas_qlib_path}/instruments/all.txt')}"
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


def is_a_share_ready(latest_date: str, target_date: str) -> bool:
    return bool(latest_date) and latest_date >= target_date


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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
