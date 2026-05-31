"""Build a US OTC/Pink proxy flow artifact.

This does not claim true institutional capital flow. It uses provider daily
OHLCV aggregates to produce a directional dollar-volume proxy for venues where
Futu's capital-flow API does not support OTC/Pink market data.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
POLYGON_GROUPED_DAILY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
US_EASTERN = ZoneInfo("America/New_York")
US_SESSION_READY_TIME = time(16, 30)


def latest_completed_us_session_date(now: datetime | None = None) -> str:
    current = now or datetime.now(US_EASTERN)
    if current.tzinfo is None:
        current = current.replace(tzinfo=US_EASTERN)
    current_et = current.astimezone(US_EASTERN)
    candidate = current_et.date()
    if candidate.weekday() >= 5 or current_et.time() < US_SESSION_READY_TIME:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.isoformat()


def _default_date() -> str:
    return latest_completed_us_session_date()


def _date_tag(value: str) -> str:
    return value[:10].replace("-", "")


def _split_csv(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." not in text:
        return f"US.{text}"
    return text


def _ticker_from_code(value: Any) -> str:
    code = _normalize_code(value)
    if "." not in code:
        return code
    return code.split(".", 1)[1]


def _exchange_type_counts(df: pd.DataFrame) -> dict[str, int]:
    if "exchange_type" not in df.columns:
        return {}
    values = df["exchange_type"].fillna("").astype(str).str.strip()
    counts = values[values != ""].value_counts().sort_index()
    return {str(exchange): int(count) for exchange, count in counts.items()}


def load_otc_universe(path: str | Path, *, exchange_types: set[str], max_codes: int = 0) -> pd.DataFrame:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"US OTC universe CSV not found: {target}")
    df = pd.read_csv(target)
    if "code" not in df.columns:
        raise ValueError(f"US OTC universe CSV missing code column: {target}")
    if "exchange_type" in df.columns and exchange_types:
        exchange = df["exchange_type"].fillna("").astype(str).str.upper()
        df = df[exchange.isin(exchange_types)].copy()
    else:
        df = df.copy()
    df["code"] = df["code"].apply(_normalize_code)
    df["ticker"] = df["code"].apply(_ticker_from_code)
    if "name" not in df.columns:
        df["name"] = ""
    if "exchange_type" not in df.columns:
        df["exchange_type"] = "US_PINK"
    df = df[df["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)
    if max_codes > 0:
        df = df.head(max_codes).copy()
    return df


def fetch_polygon_grouped_daily(
    *,
    api_key: str,
    date: str,
    include_otc: bool = True,
    adjusted: bool = True,
    base_url: str = POLYGON_GROUPED_DAILY_URL,
) -> list[dict[str, Any]]:
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is required for provider=polygon")
    params = {
        "adjusted": str(adjusted).lower(),
        "include_otc": str(include_otc).lower(),
        "apiKey": api_key,
    }
    url = f"{base_url.format(date=date)}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "QuantPilot/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") not in {"OK", "DELAYED"}:
        raise RuntimeError(f"Polygon grouped daily request failed: {payload.get('status')} {payload.get('error')}")
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError("Polygon grouped daily response did not contain a result list")
    return [row for row in results if isinstance(row, dict)]


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def build_proxy_records(
    universe: pd.DataFrame,
    aggregates: list[dict[str, Any]],
    *,
    date: str,
    provider: str,
    min_dollar_volume: float = 0.0,
) -> pd.DataFrame:
    by_ticker = {str(row.get("T") or row.get("ticker") or "").upper(): row for row in aggregates}
    records: list[dict[str, Any]] = []
    for _, item in universe.iterrows():
        ticker = str(item.get("ticker") or _ticker_from_code(item.get("code"))).upper()
        aggregate = by_ticker.get(ticker)
        base = {
            "market": "US_OTC",
            "code": _normalize_code(item.get("code")),
            "name": str(item.get("name") or ""),
            "exchange_type": str(item.get("exchange_type") or "US_PINK"),
            "capital_flow_latest_date": date,
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "source": f"{provider}_otc_proxy",
            "proxy_method": "directional_dollar_volume",
        }
        if not aggregate:
            records.append(
                {
                    **base,
                    "capital_flow_status": "empty",
                    "capital_flow_error": "No provider aggregate for ticker.",
                    "capital_flow_count": 0,
                }
            )
            continue
        open_price = _num(aggregate, "o")
        close_price = _num(aggregate, "c")
        high_price = _num(aggregate, "h")
        low_price = _num(aggregate, "l")
        volume = _num(aggregate, "v") or 0.0
        if close_price is None or open_price is None or volume <= 0:
            records.append(
                {
                    **base,
                    "capital_flow_status": "empty",
                    "capital_flow_error": "Provider aggregate missing price or volume.",
                    "capital_flow_count": 0,
                    "latest_price": close_price,
                    "volume": volume,
                }
            )
            continue
        dollar_volume = close_price * volume
        if dollar_volume < min_dollar_volume:
            records.append(
                {
                    **base,
                    "capital_flow_status": "empty",
                    "capital_flow_error": f"Dollar volume below threshold: {dollar_volume:.2f} < {min_dollar_volume:.2f}.",
                    "capital_flow_count": 0,
                    "latest_price": close_price,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                }
            )
            continue
        sign = 1.0 if close_price >= open_price else -1.0
        records.append(
            {
                **base,
                "capital_flow_status": "ok",
                "capital_flow_error": "",
                "capital_flow_count": 1,
                "open": open_price,
                "close": close_price,
                "high": high_price,
                "low": low_price,
                "volume": volume,
                "latest_price": close_price,
                "dollar_volume": dollar_volume,
                "change_pct": ((close_price / open_price) - 1.0) * 100.0 if open_price else None,
                "latest_main_in_flow": sign * dollar_volume,
            }
        )
    return pd.DataFrame(records)


def _write_status(status: dict[str, Any], output_dir: Path, date_tag: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_path = output_dir / f"US_OTC_{date_tag}_status.json"
    latest_path = output_dir / "US_OTC_latest_status.json"
    content = json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    dated_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")
    return {"status": dated_path, "latest_status": latest_path}


def write_outputs(
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    date: str,
    provider: str,
    universe: pd.DataFrame,
    min_dollar_volume: float,
) -> dict[str, Any]:
    date_tag = _date_tag(date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"US_OTC_{date_tag}_flow.csv"
    latest_path = output_dir / "US_OTC_latest_flow.csv"
    universe_path = output_dir / f"US_OTC_{date_tag}_universe.csv"
    latest_universe_path = output_dir / "US_OTC_latest_universe.csv"
    rows.to_csv(output_path, index=False)
    rows.to_csv(latest_path, index=False)
    universe.to_csv(universe_path, index=False)
    universe.to_csv(latest_universe_path, index=False)

    statuses = rows.get("capital_flow_status", pd.Series(dtype=str)).fillna("").astype(str)
    ok_count = int((statuses == "ok").sum())
    empty_count = int((statuses == "empty").sum())
    error_count = int((statuses == "error").sum())
    attempted_count = int(len(rows))
    ok_ratio = ok_count / attempted_count if attempted_count else 0.0
    status_payload = {
        "status": "ok" if attempted_count else "empty",
        "message": "ok" if attempted_count else "No OTC/Pink symbols were processed.",
        "market": "US_OTC",
        "provider": provider,
        "source": f"{provider}_otc_proxy",
        "proxy_method": "directional_dollar_volume",
        "date": date,
        "date_tag": date_tag,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "universe_count": int(len(universe)),
        "selected_count": attempted_count,
        "attempted_count": attempted_count,
        "ok_count": ok_count,
        "error_count": error_count,
        "empty_count": empty_count,
        "ok_ratio": ok_ratio,
        "min_dollar_volume": min_dollar_volume,
        "source_exchange_types": _exchange_type_counts(universe),
        "selected_exchange_types": _exchange_type_counts(universe),
        "excluded_exchange_types": {},
        "output": str(output_path),
        "latest": str(latest_path),
        "universe": str(universe_path),
        "latest_universe": str(latest_universe_path),
    }
    status_paths = _write_status(status_payload, output_dir, date_tag)
    status_payload.update({key: str(path) for key, path in status_paths.items()})
    return status_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a US OTC/Pink proxy-flow artifact.")
    parser.add_argument("--provider", default=os.environ.get("US_OTC_PROXY_FLOW_PROVIDER", "polygon"))
    parser.add_argument("--api-key", default=os.environ.get("POLYGON_API_KEY", ""))
    parser.add_argument("--date", default=os.environ.get("US_OTC_PROXY_FLOW_DATE", _default_date()))
    parser.add_argument(
        "--universe-csv",
        default=os.environ.get(
            "US_OTC_PROXY_FLOW_UNIVERSE_CSV",
            str(DATA_DIR / "capital_flow" / "futu_market" / "US_latest_source_universe.csv"),
        ),
    )
    parser.add_argument("--exchange-types", default=os.environ.get("US_OTC_PROXY_FLOW_EXCHANGE_TYPES", "US_PINK"))
    parser.add_argument("--max-codes", type=int, default=int(os.environ.get("US_OTC_PROXY_FLOW_MAX_CODES", "0")))
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=float(os.environ.get("US_OTC_PROXY_FLOW_MIN_DOLLAR_VOLUME", "0")),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "US_OTC_PROXY_FLOW_OUTPUT_DIR",
            str(DATA_DIR / "capital_flow" / "us_otc_proxy"),
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provider = str(args.provider).strip().lower()
    if provider != "polygon":
        raise ValueError(f"unsupported US OTC proxy provider: {args.provider}")
    universe = load_otc_universe(
        args.universe_csv,
        exchange_types=_split_csv(args.exchange_types),
        max_codes=args.max_codes,
    )
    aggregates = fetch_polygon_grouped_daily(api_key=args.api_key, date=args.date, include_otc=True)
    rows = build_proxy_records(
        universe,
        aggregates,
        date=args.date,
        provider=provider,
        min_dollar_volume=args.min_dollar_volume,
    )
    status = write_outputs(
        rows,
        output_dir=Path(args.output_dir).expanduser(),
        date=args.date,
        provider=provider,
        universe=universe,
        min_dollar_volume=args.min_dollar_volume,
    )
    print(
        "US_OTC: universe={universe_count} selected={selected_count} ok={ok_count}/{attempted_count} "
        "latest={latest}".format(**status)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
