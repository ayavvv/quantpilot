"""Backfill point-in-time A-share ST flags into Qlib ``is_st`` features."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from collector.baostock_client import BaostockClient
from converter.incremental import QlibDirectWriter
from market_scope import a_share_model_prefixes, code_matches_prefixes


def _default_qlib_dir() -> Path:
    data_dir = Path(os.environ.get("DATA_DIR", Path.home() / "quantpilot_data"))
    return Path(os.environ.get("QLIB_DATA_DIR", data_dir / "qlib_data")).expanduser()


def _load_a_share_instruments(qlib_dir: Path, prefixes: tuple[str, ...]) -> list[tuple[str, str, str]]:
    inst_path = qlib_dir / "instruments" / "all.txt"
    if not inst_path.exists():
        raise FileNotFoundError(f"instruments file missing: {inst_path}")

    instruments: list[tuple[str, str, str]] = []
    for line in inst_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, start_date, end_date = parts[:3]
        if code_matches_prefixes(code, prefixes):
            instruments.append((code, start_date, end_date))
    return sorted(instruments)


def _to_bs_code(futu_code: str) -> str:
    return futu_code.lower()


def _fetch_is_st(client: BaostockClient, code: str, start_date: str, end_date: str) -> list[dict]:
    rs = client._run_query(
        lambda *query_args, **query_kwargs: client._bs.query_history_k_data_plus(*query_args, **query_kwargs),
        _to_bs_code(code),
        "date,code,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )

    rows: list[dict] = []
    while rs.next():
        raw = rs.get_row_data()
        try:
            date_value = raw[0]
            is_st = float(raw[2]) if raw[2] else 0.0
        except (IndexError, TypeError, ValueError):
            continue
        rows.append({"date": date_value, "is_st": is_st})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical A-share ST flags into Qlib")
    parser.add_argument("--qlib-dir", default=str(_default_qlib_dir()))
    parser.add_argument("--start-date", default="", help="Optional global start date, YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="Optional global end date, YYYY-MM-DD")
    parser.add_argument("--rate-limit", type=float, default=0.03)
    parser.add_argument("--socket-timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-codes", type=int, default=0, help="Debug cap; 0 means no cap")
    parser.add_argument("--prefix", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    prefixes = tuple(args.prefix) if args.prefix else a_share_model_prefixes()
    instruments = _load_a_share_instruments(qlib_dir, prefixes)
    if args.max_codes > 0:
        instruments = instruments[: args.max_codes]
    if args.start_date:
        datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        datetime.strptime(args.end_date, "%Y-%m-%d")

    print(
        f"backfill is_st: qlib={qlib_dir} codes={len(instruments)} "
        f"prefixes={','.join(prefixes)}",
        flush=True,
    )

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
        for idx, (code, inst_start, inst_end) in enumerate(instruments, 1):
            start_date = max(args.start_date, inst_start) if args.start_date else inst_start
            end_date = min(args.end_date, inst_end) if args.end_date else inst_end
            if start_date > end_date:
                empty_codes.append(code)
                continue
            try:
                rows = _fetch_is_st(client, code, start_date, end_date)
            except Exception as exc:
                failed_codes.append((code, str(exc)))
                rows = []

            if rows:
                n = writer.write_feature_records(code, rows, ["is_st"])
                if n:
                    written_codes += 1
                    written_rows += n
            else:
                empty_codes.append(code)

            if idx % 100 == 0 or idx == len(instruments):
                print(
                    f"progress {idx}/{len(instruments)} written_codes={written_codes} "
                    f"rows={written_rows} empty={len(empty_codes)} failed={len(failed_codes)} last={code}",
                    flush=True,
                )
    finally:
        writer.flush()
        client.close()

    print(
        f"backfill is_st done written_codes={written_codes} rows={written_rows} "
        f"empty={len(empty_codes)} failed={len(failed_codes)}",
        flush=True,
    )
    if empty_codes:
        print(f"sample empty={empty_codes[:20]}", flush=True)
    if failed_codes:
        print(f"sample failed={failed_codes[:10]}", flush=True)
        raise RuntimeError(f"is_st backfill failed for {len(failed_codes)} codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
