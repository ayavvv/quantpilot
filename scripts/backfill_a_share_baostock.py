"""Local Baostock A-share backfill into Qlib bin data.

This is intended as a host-side rescue path when the NAS collector is stale.
It writes directly to the local Qlib directory and refreshes the completion
metadata used by the daily readiness checks.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from collector.baostock_client import BaostockClient
from converter.incremental import QlibDirectWriter
from market_scope import a_share_model_prefixes, code_matches_prefixes


def _default_qlib_dir() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", Path.home() / "quantpilot_data"))
    return Path(os.environ.get("QLIB_DATA_DIR", data_dir / "qlib_data")).expanduser()


def _next_day(date_value: str) -> str:
    return (datetime.strptime(date_value, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _load_stale_codes(qlib_dir: Path, target_date: str, prefixes: tuple[str, ...]) -> list[tuple[str, str]]:
    inst_path = qlib_dir / "instruments" / "all.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"instruments file missing: {inst_path}")

    stale: list[tuple[str, str]] = []
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, _, end_date = parts[:3]
        if code_matches_prefixes(code, prefixes) and end_date < target_date:
            stale.append((code, end_date))
    return stale


def _latest_a_share_date(qlib_dir: Path, prefixes: tuple[str, ...]) -> str:
    latest = ""
    inst_path = qlib_dir / "instruments" / "all.txt"
    if not inst_path.exists():
        return latest
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, _, end_date = parts[:3]
        if code_matches_prefixes(code, prefixes) and end_date > latest:
            latest = end_date
    return latest


def _write_completion_metadata(
    qlib_dir: Path,
    *,
    target_date: str,
    total_codes: int,
    written_codes: int,
    written_rows: int,
    empty_codes: list[str],
    failed_codes: list[tuple[str, str]],
    started_at: str,
) -> None:
    meta_dir = qlib_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = {
        "last_completed_trade_date": target_date,
        "completed_at": completed_at,
        "started_at": started_at,
        "total_codes": total_codes,
        "source": "local_baostock_rescue",
    }
    summary = {
        **status,
        "written_codes": written_codes,
        "written_rows": written_rows,
        "empty_codes": empty_codes,
        "failed_codes": [{"code": code, "error": error} for code, error in failed_codes],
    }
    (meta_dir / "a_share_sync_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta_dir / "a_share_sync_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill local A-share Qlib data via Baostock")
    parser.add_argument("--qlib-dir", default=str(_default_qlib_dir()))
    parser.add_argument("--target-date", required=True, help="Target trade date, YYYY-MM-DD")
    parser.add_argument("--start-date", default="", help="Optional global start date, YYYY-MM-DD")
    parser.add_argument("--rate-limit", type=float, default=0.03)
    parser.add_argument("--socket-timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-codes", type=int, default=0, help="Debug cap; 0 means no cap")
    parser.add_argument("--allow-query-failures", action="store_true")
    parser.add_argument("--prefix", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    target_date = args.target_date
    datetime.strptime(target_date, "%Y-%m-%d")
    if args.start_date:
        datetime.strptime(args.start_date, "%Y-%m-%d")

    prefixes = tuple(args.prefix) if args.prefix else a_share_model_prefixes()
    stale_codes = _load_stale_codes(qlib_dir, target_date, prefixes)
    if args.max_codes > 0:
        stale_codes = stale_codes[: args.max_codes]

    print(
        f"local baostock backfill: qlib={qlib_dir} target={target_date} "
        f"codes={len(stale_codes)} prefixes={','.join(prefixes)}",
        flush=True,
    )
    if not stale_codes:
        latest = _latest_a_share_date(qlib_dir, prefixes)
        print(f"local A-share already up to date: latest={latest}", flush=True)
        return 0

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    client = BaostockClient(
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
        socket_timeout=args.socket_timeout,
    )
    writer = QlibDirectWriter(qlib_dir)
    written_codes = 0
    written_rows = 0
    empty_codes: list[str] = []
    failed_codes: list[tuple[str, str]] = []

    try:
        for idx, (code, current_end) in enumerate(stale_codes, 1):
            start_date = args.start_date or _next_day(current_end)
            if start_date > target_date:
                start_date = target_date
            rows = client.get_history_kline(code, start=start_date, end=target_date, ktype="K_DAY")
            status = client.get_last_history_kline_status() or {}
            if rows:
                n = writer.write_stock_records(code, rows)
                if n:
                    written_codes += 1
                    written_rows += n
            else:
                if status.get("status") == "query_failed":
                    failed_codes.append((code, str(status.get("error", "query_failed"))))
                else:
                    empty_codes.append(code)

            if idx % 100 == 0 or idx == len(stale_codes):
                print(
                    f"progress {idx}/{len(stale_codes)} written_codes={written_codes} "
                    f"rows={written_rows} empty={len(empty_codes)} failed={len(failed_codes)} last={code}",
                    flush=True,
                )
    finally:
        writer.flush()
        client.close()

    latest = _latest_a_share_date(qlib_dir, prefixes)
    print(
        f"backfill done latest={latest} written_codes={written_codes} rows={written_rows} "
        f"empty={len(empty_codes)} failed={len(failed_codes)}",
        flush=True,
    )
    if empty_codes:
        print(f"sample empty={empty_codes[:20]}", flush=True)
    if failed_codes:
        print(f"sample failed={failed_codes[:10]}", flush=True)

    if latest < target_date:
        raise RuntimeError(f"local latest A-share date stayed below target: latest={latest}, target={target_date}")
    if failed_codes and not args.allow_query_failures:
        raise RuntimeError(f"baostock query failures remain: {len(failed_codes)}")

    _write_completion_metadata(
        qlib_dir,
        target_date=target_date,
        total_codes=len(stale_codes),
        written_codes=written_codes,
        written_rows=written_rows,
        empty_codes=empty_codes,
        failed_codes=failed_codes,
        started_at=started_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
