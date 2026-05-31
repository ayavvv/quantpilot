"""Scan Futu capital-flow data for broad market universes.

This is intentionally resumable and conservative because Futu's capital-flow
API is per symbol.  A full US/HK scan can take hours; use --max-codes for
smoke tests and keep the output coverage fields visible in downstream reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from futu import Market, RET_OK, SecurityType

from collector.config import settings
from collector.futu_client import FutuClient
from strategy.futu_capital_flow_overlay import summarize_capital_flow
from strategy.major_money_digest import normalize_market


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
MARKET_MAP = {
    "HK": Market.HK,
    "US": Market.US,
    "SH": Market.SH,
    "SZ": Market.SZ,
}
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 1.05


def _default_start(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _date_tag(value: str) -> str:
    return value[:10].replace("-", "")


def _markets(value: str) -> list[str]:
    result = []
    for item in value.split(","):
        market = normalize_market(item)
        if market == "A":
            result.extend(["SH", "SZ"])
        elif market:
            result.append(market)
    return result


def _split_csv(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _exchange_type_counts(df: pd.DataFrame) -> dict[str, int]:
    if "exchange_type" not in df.columns:
        return {}
    values = df["exchange_type"].fillna("").astype(str).str.strip()
    counts = values[values != ""].value_counts().sort_index()
    return {str(exchange): int(count) for exchange, count in counts.items()}


def _exchange_type_delta(source: dict[str, int], selected: dict[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for exchange, source_count in source.items():
        missing = int(source_count) - int(selected.get(exchange, 0))
        if missing > 0:
            delta[exchange] = missing
    return delta


def _status_counts_by_exchange_type(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "exchange_type" not in df.columns or "capital_flow_status" not in df.columns:
        return {}
    result: dict[str, dict[str, int]] = {}
    work = df[["exchange_type", "capital_flow_status"]].copy()
    work["exchange_type"] = work["exchange_type"].fillna("").astype(str).str.strip()
    work["capital_flow_status"] = work["capital_flow_status"].fillna("").astype(str).str.strip()
    work = work[(work["exchange_type"] != "") & (work["capital_flow_status"] != "")]
    if work.empty:
        return {}
    grouped = work.groupby(["exchange_type", "capital_flow_status"]).size()
    for (exchange, status), count in grouped.items():
        result.setdefault(str(exchange), {})[str(status)] = int(count)
    return result


def _unsupported_exchange_types(df: pd.DataFrame) -> dict[str, int]:
    if "exchange_type" not in df.columns or "capital_flow_error" not in df.columns:
        return {}
    work = df.copy()
    work["exchange_type"] = work["exchange_type"].fillna("").astype(str).str.strip()
    errors = work["capital_flow_error"].fillna("").astype(str).str.lower()
    unsupported = work[(work["exchange_type"] != "") & errors.str.contains("do not support otc market data", regex=False)]
    return _exchange_type_counts(unsupported)


def _codes_by_market(value: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in value.split(","):
        code = item.strip().upper()
        if not code:
            continue
        if "." not in code:
            raise ValueError(f"code must include market prefix: {code}")
        market = normalize_market(code.split(".", 1)[0])
        grouped.setdefault(market, []).append(code)
    return grouped


def fetch_futu_universe(
    client: FutuClient,
    market: str,
    *,
    include_exchange_types: set[str] | None = None,
    exclude_exchange_types: set[str] | None = None,
) -> pd.DataFrame:
    if market not in MARKET_MAP:
        raise ValueError(f"unsupported Futu market: {market}")
    ret, data = client.ctx.get_stock_basicinfo(MARKET_MAP[market], SecurityType.STOCK)
    if ret != RET_OK:
        raise RuntimeError(f"get_stock_basicinfo failed for {market}: {data}")
    df = data.copy()
    df["market"] = market
    if "delisting" in df.columns:
        df = df[~df["delisting"].fillna(False).astype(bool)].copy()
    source_exchange_types = _exchange_type_counts(df)
    if "exchange_type" in df.columns:
        exchange = df["exchange_type"].fillna("").astype(str).str.upper()
        if include_exchange_types:
            df = df[exchange.isin(include_exchange_types)].copy()
            exchange = df["exchange_type"].fillna("").astype(str).str.upper()
        if exclude_exchange_types:
            df = df[~exchange.isin(exclude_exchange_types)].copy()
    if "code" not in df.columns:
        raise RuntimeError(f"Futu stock basic info missing code column for {market}")
    result = df.reset_index(drop=True)
    selected_exchange_types = _exchange_type_counts(result)
    result.attrs["source_exchange_types"] = source_exchange_types
    result.attrs["selected_exchange_types"] = selected_exchange_types
    result.attrs["excluded_exchange_types"] = _exchange_type_delta(source_exchange_types, selected_exchange_types)
    result.attrs["include_exchange_types"] = sorted(include_exchange_types or [])
    result.attrs["exclude_exchange_types"] = sorted(exclude_exchange_types or [])
    return result


def _load_resume(path: Path, overwrite: bool) -> tuple[pd.DataFrame, set[str]]:
    if overwrite or not path.exists():
        return pd.DataFrame(), set()
    existing = pd.read_csv(path)
    codes = set(existing["code"].dropna().astype(str).tolist()) if "code" in existing.columns else set()
    return existing, codes


def _write_outputs(df: pd.DataFrame, output_path: Path, latest_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df.to_csv(latest_path, index=False)


def _write_status(status: dict[str, Any], output_dir: Path, market: str, date_tag: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_path = output_dir / f"{market}_{date_tag}_status.json"
    latest_path = output_dir / f"{market}_latest_status.json"
    content = json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    dated_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")
    return {"status": dated_path, "latest_status": latest_path}


def _effective_pause_seconds(
    *,
    client_rate_limit_delay: float,
    pause_seconds: float,
    min_request_interval: float,
) -> float:
    if min_request_interval <= 0:
        return pause_seconds
    effective_interval = max(client_rate_limit_delay, 0.0) + max(pause_seconds, 0.0)
    if effective_interval >= min_request_interval:
        return pause_seconds
    return pause_seconds + (min_request_interval - effective_interval)


def scan_market(
    client: FutuClient,
    universe: pd.DataFrame,
    *,
    market: str,
    output_dir: Path,
    start: str,
    end: str,
    period: str,
    include_distribution: bool,
    max_codes: int,
    batch_flush: int,
    overwrite: bool,
    pause_seconds: float,
    min_ok_ratio: float,
    include_exchange_types: set[str] | None = None,
    exclude_exchange_types: set[str] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    date_tag = _date_tag(end)
    output_path = output_dir / f"{market}_{date_tag}_flow.csv"
    latest_path = output_dir / f"{market}_latest_flow.csv"
    existing, done_codes = _load_resume(output_path, overwrite=overwrite)
    records = existing.to_dict("records") if not existing.empty else []

    work = universe.copy()
    if max_codes > 0:
        work = work.head(max_codes).copy()
    total = len(work)

    for idx, row in work.iterrows():
        code = str(row.get("code", "")).strip()
        if not code or code in done_codes:
            continue
        base = {
            "market": market,
            "code": code,
            "name": row.get("name", ""),
            "exchange_type": row.get("exchange_type", ""),
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "source": "futu",
        }
        try:
            flow_records = client.get_capital_flow(code, period_type=period, start=start, end=end)
            distribution = client.get_capital_distribution(code) if include_distribution else {}
            summary = summarize_capital_flow(code, flow_records, distribution)
            record = {**base, **summary}
        except Exception as exc:
            record = {
                **base,
                "capital_flow_status": "error",
                "capital_flow_error": str(exc),
                "capital_flow_count": 0,
                "capital_flow_latest_date": "",
            }
        records.append(record)
        done_codes.add(code)

        if len(records) % max(batch_flush, 1) == 0:
            _write_outputs(pd.DataFrame(records), output_path, latest_path)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    result = pd.DataFrame(records)
    statuses = result.get("capital_flow_status", pd.Series(dtype=str)).fillna("").astype(str)
    ok_count = int((statuses == "ok").sum())
    error_count = int((statuses == "error").sum())
    empty_count = int((statuses == "empty").sum())
    attempted = len(result)
    ok_ratio = ok_count / attempted if attempted else 0.0
    status_value = "ok"
    status_message = "ok"
    if not attempted:
        status_value = "empty"
        status_message = "No symbols were scanned."
    elif ok_ratio < min_ok_ratio:
        status_value = "failed"
        status_message = f"ok_ratio too low: {ok_ratio:.1%} < {min_ok_ratio:.1%}"

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_outputs(result, output_path, latest_path)
    universe_path = output_dir / f"{market}_{date_tag}_universe.csv"
    universe.to_csv(universe_path, index=False)
    source_exchange_types = universe.attrs.get("source_exchange_types") or _exchange_type_counts(universe)
    selected_exchange_types = universe.attrs.get("selected_exchange_types") or _exchange_type_counts(work)
    excluded_exchange_types = universe.attrs.get("excluded_exchange_types") or _exchange_type_delta(
        source_exchange_types,
        selected_exchange_types,
    )

    status_payload = {
        "status": status_value,
        "message": status_message,
        "market": market,
        "period": period,
        "start": start,
        "end": end,
        "date_tag": date_tag,
        "started_at": started_at,
        "finished_at": finished_at,
        "include_distribution": include_distribution,
        "max_codes": max_codes,
        "universe_count": int(len(universe)),
        "selected_count": int(total),
        "attempted_count": int(attempted),
        "ok_count": ok_count,
        "error_count": error_count,
        "empty_count": empty_count,
        "ok_ratio": ok_ratio,
        "min_ok_ratio": min_ok_ratio,
        "source_exchange_types": source_exchange_types,
        "selected_exchange_types": selected_exchange_types,
        "excluded_exchange_types": excluded_exchange_types,
        "status_by_exchange_type": _status_counts_by_exchange_type(result),
        "unsupported_exchange_types": _unsupported_exchange_types(result),
        "include_exchange_types": sorted(include_exchange_types or universe.attrs.get("include_exchange_types") or []),
        "exclude_exchange_types": sorted(exclude_exchange_types or universe.attrs.get("exclude_exchange_types") or []),
        "output": str(output_path),
        "latest": str(latest_path),
        "universe": str(universe_path),
    }

    status_paths = _write_status(status_payload, output_dir, market, date_tag)
    if attempted and ok_ratio < min_ok_ratio:
        raise RuntimeError(f"{market} Futu capital-flow ok_ratio too low: {ok_ratio:.1%} < {min_ok_ratio:.1%}")

    return {
        "market": market,
        "universe_count": len(universe),
        "selected_count": total,
        "attempted_count": attempted,
        "ok_count": ok_count,
        "output": str(output_path),
        "latest": str(latest_path),
        "universe": str(universe_path),
        "status": str(status_paths["status"]),
        "latest_status": str(status_paths["latest_status"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Futu capital flow for HK/US/A-share market universes.")
    parser.add_argument("--markets", default=os.environ.get("FUTU_MARKET_FLOW_MARKETS", "HK,US"))
    parser.add_argument("--codes", default=os.environ.get("FUTU_MARKET_FLOW_CODES", ""))
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", settings.futu_host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", settings.futu_port)))
    parser.add_argument("--connect-timeout", type=float, default=float(os.environ.get("FUTU_CONNECT_TIMEOUT", "8")))
    parser.add_argument("--period", default=os.environ.get("FUTU_MARKET_FLOW_PERIOD", "DAY"))
    parser.add_argument("--days", type=int, default=int(os.environ.get("FUTU_MARKET_FLOW_DAYS", "30")))
    parser.add_argument("--start", default=os.environ.get("FUTU_MARKET_FLOW_START", ""))
    parser.add_argument("--end", default=os.environ.get("FUTU_MARKET_FLOW_END", datetime.now().strftime("%Y-%m-%d")))
    parser.add_argument("--include-distribution", action="store_true")
    parser.add_argument("--max-codes", type=int, default=int(os.environ.get("FUTU_MARKET_FLOW_MAX_CODES", "0")))
    parser.add_argument("--batch-flush", type=int, default=int(os.environ.get("FUTU_MARKET_FLOW_BATCH_FLUSH", "50")))
    parser.add_argument("--pause-seconds", type=float, default=float(os.environ.get("FUTU_MARKET_FLOW_PAUSE_SECONDS", "1.1")))
    parser.add_argument("--rate-limit-delay", type=float, default=float(os.environ.get("FUTU_MARKET_FLOW_RATE_LIMIT_DELAY", "0.0")))
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=float(os.environ.get("FUTU_MARKET_FLOW_MIN_REQUEST_INTERVAL", str(DEFAULT_MIN_REQUEST_INTERVAL_SECONDS))),
        help="Minimum seconds between Futu capital-flow requests. Use 0 to disable automatic pacing.",
    )
    parser.add_argument("--min-ok-ratio", type=float, default=float(os.environ.get("FUTU_MARKET_FLOW_MIN_OK_RATIO", "0.5")))
    parser.add_argument("--include-exchange-types", default=os.environ.get("FUTU_MARKET_FLOW_INCLUDE_EXCHANGE_TYPES", ""))
    parser.add_argument(
        "--exclude-exchange-types",
        default=os.environ.get("FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES", "US_PINK,N/A"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default=os.environ.get("FUTU_MARKET_FLOW_OUTPUT_DIR", str(DATA_DIR / "capital_flow" / "futu_market")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = args.start or _default_start(args.days)
    output_dir = Path(args.output_dir).expanduser()

    client = FutuClient(args.host, args.port)
    client.connect_timeout = args.connect_timeout
    if args.rate_limit_delay > 0:
        client.rate_limit_delay = args.rate_limit_delay
    args.pause_seconds = _effective_pause_seconds(
        client_rate_limit_delay=float(client.rate_limit_delay),
        pause_seconds=float(args.pause_seconds),
        min_request_interval=float(args.min_request_interval),
    )
    if not client.connect():
        raise RuntimeError(f"failed to connect Futu OpenD at {args.host}:{args.port}")

    try:
        include_exchange_types = _split_csv(args.include_exchange_types)
        exclude_exchange_types = _split_csv(args.exclude_exchange_types)
        explicit_codes = _codes_by_market(args.codes) if args.codes else {}
        for market in _markets(args.markets):
            if explicit_codes:
                codes = explicit_codes.get(market, [])
                if not codes:
                    continue
                universe = pd.DataFrame({"market": market, "code": codes, "name": "", "exchange_type": ""})
            else:
                universe = fetch_futu_universe(
                    client,
                    market,
                    include_exchange_types=include_exchange_types,
                    exclude_exchange_types=exclude_exchange_types,
                )
            stats = scan_market(
                client,
                universe,
                market=market,
                output_dir=output_dir,
                start=start,
                end=args.end,
                period=args.period,
                include_distribution=args.include_distribution,
                max_codes=args.max_codes,
                batch_flush=args.batch_flush,
                overwrite=args.overwrite,
                pause_seconds=args.pause_seconds,
                min_ok_ratio=args.min_ok_ratio,
                include_exchange_types=include_exchange_types,
                exclude_exchange_types=exclude_exchange_types,
            )
            print(
                "{market}: universe={universe_count} selected={selected_count} ok={ok_count}/{attempted_count} "
                "latest={latest}".format(**stats)
            )
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
