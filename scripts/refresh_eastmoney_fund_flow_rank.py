"""Refresh A-share market-wide Eastmoney fund-flow rank artifacts."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from collector.eastmoney_fund_flow import fetch_fund_flow_rank


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))


def refresh_rank(
    *,
    output: Path,
    archive_dir: Path | None,
    limit: int,
    timeout: float,
    source: str,
    min_rows: int,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    df = fetch_fund_flow_rank(limit=limit, timeout=timeout, source=source)
    if len(df) < min_rows:
        raise RuntimeError(f"Eastmoney fund-flow rows below minimum: {len(df)} < {min_rows}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp")
    df.to_csv(tmp_output, index=False)
    tmp_output.replace(output)

    paths: dict[str, Path] = {"latest": output}
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d")
        archive_path = archive_dir / f"{date_tag}_rank.csv"
        df.to_csv(archive_path, index=False)
        paths["archive"] = archive_path
    return df, paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh A-share Eastmoney market-wide fund-flow rank.")
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "EASTMONEY_FUND_FLOW_RANK_OUTPUT",
            str(DATA_DIR / "output" / "eastmoney_fund_flow_rank_latest.csv"),
        ),
    )
    parser.add_argument(
        "--archive-dir",
        default=os.environ.get("EASTMONEY_FUND_FLOW_ARCHIVE_DIR", str(DATA_DIR / "fund_flow" / "eastmoney")),
    )
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("EASTMONEY_FUND_FLOW_LIMIT", "6000")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("EASTMONEY_FUND_FLOW_TIMEOUT", "10")))
    parser.add_argument(
        "--source",
        choices=["auto", "push2", "datacenter"],
        default=os.environ.get("EASTMONEY_FUND_FLOW_SOURCE", "auto"),
    )
    parser.add_argument("--min-rows", type=int, default=int(os.environ.get("EASTMONEY_FUND_FLOW_MIN_ROWS", "1000")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_dir = None if args.no_archive else Path(args.archive_dir).expanduser()
    df, paths = refresh_rank(
        output=Path(args.output).expanduser(),
        archive_dir=archive_dir,
        limit=args.limit,
        timeout=args.timeout,
        source=args.source,
        min_rows=args.min_rows,
    )
    print(f"Wrote Eastmoney fund-flow rank: {paths['latest']} rows={len(df)}")
    if "archive" in paths:
        print(f"Wrote Eastmoney fund-flow archive: {paths['archive']}")
    if not df.empty:
        print(df.head(min(len(df), 10)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
