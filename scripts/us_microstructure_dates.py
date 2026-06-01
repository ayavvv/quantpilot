"""Date helpers for US microstructure collection and reports."""

from __future__ import annotations

import argparse
from datetime import date as date_type
from datetime import datetime, timedelta
from pathlib import Path


COLLECTION_DATE_DIRS = ("manifests", "trades", "order_book", "quotes")


def _is_yyyy_mm_dd(value: str) -> bool:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def collection_dates(base_dir: str | Path, *, kinds: tuple[str, ...] = COLLECTION_DATE_DIRS) -> list[str]:
    base = Path(base_dir).expanduser()
    dates: set[str] = set()
    for kind in kinds:
        root = base / kind
        if not root.exists():
            continue
        for path in root.glob("date=*"):
            value = path.name.split("=", 1)[1]
            if _is_yyyy_mm_dd(value):
                dates.add(value)
    return sorted(dates)


def _parse_today(value: str | None) -> date_type:
    if not value:
        return date_type.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def default_report_date(base_dir: str | Path, *, today: str | None = None) -> str:
    dates = collection_dates(base_dir)
    if dates:
        return dates[-1]
    return (_parse_today(today) - timedelta(days=1)).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve US microstructure dates.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("default-report-date")
    report.add_argument("--base-dir", required=True)
    report.add_argument("--today", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "default-report-date":
        print(default_report_date(args.base_dir, today=args.today))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
