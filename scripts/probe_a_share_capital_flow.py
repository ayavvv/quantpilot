"""Probe Futu A-share capital-flow availability for selected symbols."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from collector.config import settings
from collector.futu_client import FutuClient


DEFAULT_CODES = ("SH.600000", "SZ.000001")


def _parse_codes(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _codes_from_major_force(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "code" not in df.columns:
        return []
    return [str(code).upper() for code in df["code"].dropna().head(limit).tolist()]


def _default_start_end(days: int) -> tuple[str, str]:
    end = datetime.now().date()
    start = end - timedelta(days=max(days, 1))
    return start.isoformat(), end.isoformat()


def probe_capital_flow(
    codes: list[str],
    *,
    host: str,
    port: int,
    period: str,
    start: str,
    end: str,
    include_distribution: bool,
    connect_timeout: float = 3.0,
) -> dict:
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": host,
        "port": port,
        "period": period,
        "start": start,
        "end": end,
        "codes": [],
    }
    try:
        with socket.create_connection((host, port), timeout=connect_timeout):
            pass
    except OSError as exc:
        result["connection_status"] = "error"
        result["connection_error"] = str(exc)
        result["codes"] = [
            {"code": code, "flow_status": "not_run", "distribution_status": "not_run"}
            for code in codes
        ]
        return result

    client = FutuClient(host, port)
    client.connect_timeout = connect_timeout
    if not client.connect():
        result["connection_status"] = "error"
        result["connection_error"] = f"failed to connect Futu OpenD at {host}:{port}"
        return result

    result["connection_status"] = "ok"
    try:
        for code in codes:
            code_result = {"code": code, "flow_status": "not_run", "distribution_status": "not_run"}
            try:
                flow = client.get_capital_flow(code, period_type=period, start=start, end=end)
                code_result["flow_status"] = "ok"
                code_result["flow_count"] = len(flow)
                code_result["flow_latest"] = flow[-1] if flow else {}
                code_result["flow_columns"] = sorted(flow[-1].keys()) if flow else []
            except Exception as exc:
                code_result["flow_status"] = "error"
                code_result["flow_error"] = str(exc)

            if include_distribution:
                try:
                    distribution = client.get_capital_distribution(code)
                    code_result["distribution_status"] = "ok" if distribution else "empty"
                    code_result["distribution"] = distribution
                except Exception as exc:
                    code_result["distribution_status"] = "error"
                    code_result["distribution_error"] = str(exc)

            result["codes"].append(code_result)
    finally:
        client.disconnect()
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Futu A-share capital-flow fields and account permission.")
    parser.add_argument("--codes", default="", help="Comma-separated stock codes. Defaults to latest major-force top picks.")
    parser.add_argument("--major-force-csv", default="~/quantpilot_data/output/major_force_latest.csv")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", settings.futu_host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", settings.futu_port)))
    parser.add_argument("--period", default="DAY", choices=["INTRADAY", "DAY", "WEEK", "MONTH"])
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--include-distribution", action="store_true")
    parser.add_argument("--output", default="~/quantpilot_data/output/a_share_capital_flow_probe.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.codes:
        codes = _parse_codes(args.codes)
    else:
        codes = _codes_from_major_force(Path(args.major_force_csv).expanduser(), args.limit)
    if not codes:
        codes = list(DEFAULT_CODES)[: max(args.limit, 1)]

    start, end = (args.start, args.end)
    if not start or not end:
        start, end = _default_start_end(args.days)

    result = probe_capital_flow(
        codes[: max(args.limit, 1)],
        host=args.host,
        port=args.port,
        period=args.period,
        start=start,
        end=end,
        include_distribution=args.include_distribution,
        connect_timeout=args.connect_timeout,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if result.get("connection_status") == "error":
        print(f"connection_error: {result.get('connection_error')}")
    for item in result["codes"]:
        latest = item.get("flow_latest") or {}
        main_flow = latest.get("main_in_flow", "N/A")
        print(
            f"{item['code']}: flow={item['flow_status']} count={item.get('flow_count', 0)} "
            f"latest_date={latest.get('date') or latest.get('time') or 'N/A'} main_in_flow={main_flow} "
            f"distribution={item.get('distribution_status')}"
        )
        if item.get("flow_error"):
            print(f"  flow_error: {item['flow_error']}")
        if item.get("distribution_error"):
            print(f"  distribution_error: {item['distribution_error']}")
    print(f"Wrote probe result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
