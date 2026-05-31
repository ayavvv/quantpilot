"""Build the daily market-wide major-money digest."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from strategy.major_money_digest import (
    build_digest,
    build_market_summary,
    digest_rows,
    load_source_csv,
    normalize_market,
)


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))


def _status_for_source(path: Path, market: str) -> dict:
    candidates = [
        path.with_name(f"{market}_latest_status.json"),
        path.with_name(path.name.replace("_flow.csv", "_status.json")),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _healthy_otc_proxy_source(path: Path) -> bool:
    status = _status_for_source(path, "US_OTC")
    if not status:
        return False
    return str(status.get("status") or "").lower() == "ok" and int(status.get("ok_count") or 0) > 0


def _default_sources(data_dir: Path) -> list[tuple[str, Path, str]]:
    candidates = [
        ("A", data_dir / "output" / "eastmoney_fund_flow_rank_latest.csv", "eastmoney"),
        ("HK", data_dir / "capital_flow" / "futu_market" / "HK_latest_flow.csv", "futu"),
        ("US", data_dir / "capital_flow" / "futu_market" / "US_latest_flow.csv", "futu"),
    ]
    sources = [(market, path, source) for market, path, source in candidates if path.exists()]
    otc_path = data_dir / "capital_flow" / "us_otc_proxy" / "US_OTC_latest_flow.csv"
    if otc_path.exists() and _healthy_otc_proxy_source(otc_path):
        sources.append(("US_OTC", otc_path, "polygon_otc_proxy"))
    return sources


def _parse_source(value: str) -> tuple[str, Path, str]:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("source must be MARKET:/path/to/file.csv[:source_name]")
    market = normalize_market(parts[0])
    path = Path(parts[1]).expanduser()
    source = parts[2].strip() if len(parts) == 3 and parts[2].strip() else "csv"
    return market, path, source


def _parse_markets(value: str) -> list[str]:
    return [normalize_market(item) for item in value.split(",") if item.strip()]


def _snapshot_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market-wide major-money digest JSON/CSV.")
    parser.add_argument(
        "--source",
        action="append",
        type=_parse_source,
        default=[],
        help="MARKET:/path/to/file.csv[:source_name]. Defaults to known QuantPilot artifacts.",
    )
    parser.add_argument("--expected-markets", default=os.environ.get("MAJOR_MONEY_EXPECTED_MARKETS", "A,HK,US,US_OTC"))
    parser.add_argument("--top-n", type=int, default=int(os.environ.get("MAJOR_MONEY_TOP_N", "10")))
    parser.add_argument(
        "--output-json",
        default=os.environ.get(
            "MAJOR_MONEY_DIGEST_JSON",
            str(DATA_DIR / "output" / "major_money_digest_latest.json"),
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=os.environ.get(
            "MAJOR_MONEY_DIGEST_CSV",
            str(DATA_DIR / "output" / "major_money_digest_latest.csv"),
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = args.source or _default_sources(DATA_DIR)
    summaries = []
    for market, path, source in sources:
        if not path.exists():
            raise FileNotFoundError(f"major-money source not found: {path}")
        df = load_source_csv(path)
        summaries.append(
            build_market_summary(
                df,
                market=market,
                source=source,
                top_n=args.top_n,
                snapshot_date=_snapshot_date(path),
                source_status=_status_for_source(path, market),
            )
        )

    digest = build_digest(summaries, expected_markets=_parse_markets(args.expected_markets))

    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(digest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    digest_rows(digest).to_csv(output_csv, index=False)

    print(f"Wrote major-money digest JSON: {output_json}")
    print(f"Wrote major-money digest CSV: {output_csv}")
    for market in digest["markets"]:
        print(
            "{market}: available={available} rows={ok_rows}/{total_rows} "
            "entry={entry_count} exit={exit_count} net={net_amount:.0f} {currency} source={source}".format(
                market=market.get("market"),
                available=market.get("available"),
                ok_rows=market.get("ok_rows", 0),
                total_rows=market.get("total_rows", 0),
                entry_count=market.get("entry_count", 0),
                exit_count=market.get("exit_count", 0),
                net_amount=float(market.get("net_amount") or 0.0),
                currency=market.get("currency", ""),
                source=market.get("source", ""),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
