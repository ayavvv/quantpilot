"""Build a US OTC/Pink proxy flow artifact.

This does not claim true institutional capital flow. It uses provider daily
OHLCV aggregates to produce a directional dollar-volume proxy for venues where
Futu's capital-flow API does not support OTC/Pink market data.
"""

from __future__ import annotations

import argparse
import json
import os
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
POLYGON_GROUPED_DAILY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
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


def resolve_api_key(api_key: str = "", api_key_file: str | Path = "") -> str:
    direct = str(api_key or "").strip()
    if direct:
        return direct
    if not api_key_file:
        return ""
    target = Path(api_key_file).expanduser()
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8", errors="replace").strip()


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
        raise RuntimeError("POLYGON_API_KEY or POLYGON_API_KEY_FILE is required for provider=polygon")
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


def _yahoo_period_range(date: str) -> tuple[int, int]:
    session = datetime.strptime(date[:10], "%Y-%m-%d").replace(tzinfo=US_EASTERN)
    return int(session.timestamp()), int((session + timedelta(days=2)).timestamp())


def fetch_yahoo_chart_daily_bar(
    ticker: str,
    *,
    date: str,
    timeout: float = 15.0,
    base_url: str = YAHOO_CHART_URL,
) -> dict[str, Any] | None:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    period1, period2 = _yahoo_period_range(date)
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "includePrePost": "false",
        "events": "history",
    }
    url = f"{base_url.format(ticker=symbol)}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "QuantPilot/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        if isinstance(error, dict):
            code = error.get("code") or "error"
            description = error.get("description") or ""
            raise RuntimeError(f"Yahoo chart request failed for {symbol}: {code} {description}".strip())
        raise RuntimeError(f"Yahoo chart request failed for {symbol}: {error}")
    results = chart.get("result") or []
    if not results:
        return None
    result = results[0] if isinstance(results[0], dict) else {}
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or []
    quote = quotes[0] if quotes and isinstance(quotes[0], dict) else {}
    if not timestamps or not quote:
        return None

    target_date = date[:10]
    for idx, timestamp in enumerate(timestamps):
        try:
            session_date = datetime.fromtimestamp(int(timestamp), US_EASTERN).date().isoformat()
        except (TypeError, ValueError, OSError):
            continue
        if session_date != target_date:
            continue
        return {
            "T": symbol,
            "o": _list_value(quote.get("open"), idx),
            "c": _list_value(quote.get("close"), idx),
            "h": _list_value(quote.get("high"), idx),
            "l": _list_value(quote.get("low"), idx),
            "v": _list_value(quote.get("volume"), idx),
        }
    return None


def _list_value(values: Any, idx: int) -> Any:
    if not isinstance(values, list) or idx >= len(values):
        return None
    return values[idx]


def fetch_yahoo_chart_daily(
    universe: pd.DataFrame,
    *,
    date: str,
    request_delay: float = 0.2,
    max_retries: int = 2,
    timeout: float = 15.0,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    aggregates: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    if universe.empty:
        return aggregates, errors
    tickers = universe["ticker"].fillna("").astype(str).str.upper()
    tickers = [ticker for ticker in dict.fromkeys(tickers.tolist()) if ticker]
    attempts = max(int(max_retries), 0) + 1
    for index, ticker in enumerate(tickers):
        for attempt in range(attempts):
            try:
                row = fetch_yahoo_chart_daily_bar(ticker, date=date, timeout=timeout)
                if row:
                    aggregates.append(row)
                break
            except HTTPError as exc:
                errors[ticker] = f"Yahoo chart HTTP {exc.code}"
                if exc.code == 404 or attempt >= attempts - 1:
                    break
                time_module.sleep(min(2 ** attempt, 8))
            except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                errors[ticker] = f"Yahoo chart fetch failed: {exc}"
                if attempt >= attempts - 1:
                    break
                time_module.sleep(min(2 ** attempt, 8))
        if request_delay > 0 and index < len(tickers) - 1:
            time_module.sleep(request_delay)
    return aggregates, errors


def _fetch_yahoo_chart_daily_bar_with_retries(
    ticker: str,
    *,
    date: str,
    max_retries: int,
    timeout: float,
) -> tuple[dict[str, Any] | None, str]:
    attempts = max(int(max_retries), 0) + 1
    for attempt in range(attempts):
        try:
            return fetch_yahoo_chart_daily_bar(ticker, date=date, timeout=timeout), ""
        except HTTPError as exc:
            message = f"Yahoo chart HTTP {exc.code}"
            if exc.code == 404 or attempt >= attempts - 1:
                return None, message
            time_module.sleep(min(2 ** attempt, 8))
        except (URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            message = f"Yahoo chart fetch failed: {exc}"
            if attempt >= attempts - 1:
                return None, message
            time_module.sleep(min(2 ** attempt, 8))
    return None, "Yahoo chart fetch failed."


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
    aggregate_errors: dict[str, str] | None = None,
) -> pd.DataFrame:
    by_ticker = {str(row.get("T") or row.get("ticker") or "").upper(): row for row in aggregates}
    aggregate_errors = aggregate_errors or {}
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
                    "capital_flow_error": aggregate_errors.get(ticker, "No provider aggregate for ticker."),
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


def _flow_paths(output_dir: Path, date: str) -> dict[str, Path]:
    date_tag = _date_tag(date)
    return {
        "output": output_dir / f"US_OTC_{date_tag}_flow.csv",
        "latest": output_dir / "US_OTC_latest_flow.csv",
        "universe": output_dir / f"US_OTC_{date_tag}_universe.csv",
        "latest_universe": output_dir / "US_OTC_latest_universe.csv",
    }


def _write_flow_files(rows: pd.DataFrame, universe: pd.DataFrame, *, output_dir: Path, date: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _flow_paths(output_dir, date)
    rows.to_csv(paths["output"], index=False)
    rows.to_csv(paths["latest"], index=False)
    universe.to_csv(paths["universe"], index=False)
    universe.to_csv(paths["latest_universe"], index=False)
    return paths


def _load_resume(path: Path, *, overwrite: bool) -> tuple[pd.DataFrame, set[str]]:
    if overwrite or not path.exists():
        return pd.DataFrame(), set()
    existing = pd.read_csv(path)
    if "ticker" in existing.columns:
        tickers = set(existing["ticker"].dropna().astype(str).str.upper().tolist())
    elif "code" in existing.columns:
        tickers = {_ticker_from_code(code) for code in existing["code"].dropna().astype(str).tolist()}
    else:
        tickers = set()
    return existing, {ticker for ticker in tickers if ticker}


def scan_yahoo_chart_proxy_records(
    universe: pd.DataFrame,
    *,
    output_dir: Path,
    date: str,
    min_dollar_volume: float,
    request_delay: float,
    max_retries: int,
    timeout: float,
    batch_flush: int,
    overwrite: bool,
) -> pd.DataFrame:
    paths = _flow_paths(output_dir, date)
    existing, done_tickers = _load_resume(paths["output"], overwrite=overwrite)
    records = existing.to_dict("records") if not existing.empty else []
    batch_count = 0
    provider = "yahoo_chart"

    for _, item in universe.iterrows():
        ticker = str(item.get("ticker") or _ticker_from_code(item.get("code"))).upper()
        if not ticker or ticker in done_tickers:
            continue
        aggregate, error = _fetch_yahoo_chart_daily_bar_with_retries(
            ticker,
            date=date,
            max_retries=max_retries,
            timeout=timeout,
        )
        row = build_proxy_records(
            pd.DataFrame([item]),
            [aggregate] if aggregate else [],
            date=date,
            provider=provider,
            min_dollar_volume=min_dollar_volume,
            aggregate_errors={ticker: error} if error else {},
        )
        records.extend(row.to_dict("records"))
        done_tickers.add(ticker)
        batch_count += 1
        if batch_count % max(batch_flush, 1) == 0:
            partial = pd.DataFrame(records)
            paths = _write_flow_files(partial, universe, output_dir=output_dir, date=date)
            statuses = partial.get("capital_flow_status", pd.Series(dtype=str)).fillna("").astype(str)
            print(
                "US_OTC yahoo_chart progress: "
                f"rows={len(partial)}/{len(universe)} "
                f"ok={int((statuses == 'ok').sum())} "
                f"empty={int((statuses == 'empty').sum())} "
                f"latest={paths['latest']}",
                flush=True,
            )
        if request_delay > 0:
            time_module.sleep(request_delay)

    result = pd.DataFrame(records)
    _write_flow_files(result, universe, output_dir=output_dir, date=date)
    return result


def write_failure_status(
    *,
    output_dir: Path,
    date: str,
    provider: str,
    message: str,
    universe: pd.DataFrame | None = None,
    min_dollar_volume: float = 0.0,
) -> dict[str, Any]:
    date_tag = _date_tag(date)
    universe_count = int(len(universe)) if universe is not None else 0
    status_payload = {
        "status": "failed",
        "message": message,
        "market": "US_OTC",
        "provider": provider,
        "source": f"{provider}_otc_proxy",
        "proxy_method": "directional_dollar_volume",
        "date": date,
        "date_tag": date_tag,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "universe_count": universe_count,
        "selected_count": universe_count,
        "attempted_count": universe_count,
        "ok_count": 0,
        "error_count": universe_count,
        "empty_count": 0,
        "ok_ratio": 0.0,
        "min_dollar_volume": min_dollar_volume,
        "source_exchange_types": _exchange_type_counts(universe) if universe is not None else {},
        "selected_exchange_types": _exchange_type_counts(universe) if universe is not None else {},
        "excluded_exchange_types": {},
        "output": "",
        "latest": str(output_dir / "US_OTC_latest_flow.csv"),
        "universe": "",
        "latest_universe": str(output_dir / "US_OTC_latest_universe.csv"),
    }
    status_paths = _write_status(status_payload, output_dir, date_tag)
    status_payload.update({key: str(path) for key, path in status_paths.items()})
    return status_payload


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
    paths = _write_flow_files(rows, universe, output_dir=output_dir, date=date)

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
        "output": str(paths["output"]),
        "latest": str(paths["latest"]),
        "universe": str(paths["universe"]),
        "latest_universe": str(paths["latest_universe"]),
    }
    status_paths = _write_status(status_payload, output_dir, date_tag)
    status_payload.update({key: str(path) for key, path in status_paths.items()})
    return status_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a US OTC/Pink proxy-flow artifact.")
    parser.add_argument("--provider", default=os.environ.get("US_OTC_PROXY_FLOW_PROVIDER", "polygon"))
    parser.add_argument("--api-key", default=os.environ.get("POLYGON_API_KEY", ""))
    parser.add_argument("--api-key-file", default=os.environ.get("POLYGON_API_KEY_FILE", ""))
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
        "--request-delay",
        type=float,
        default=float(os.environ.get("US_OTC_PROXY_FLOW_REQUEST_DELAY", "0.2")),
        help="Seconds to sleep between per-symbol provider requests when the provider needs it.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("US_OTC_PROXY_FLOW_MAX_RETRIES", "2")),
        help="Retries per symbol for per-symbol provider requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("US_OTC_PROXY_FLOW_TIMEOUT", "15")),
        help="HTTP timeout in seconds for provider requests.",
    )
    parser.add_argument(
        "--batch-flush",
        type=int,
        default=int(os.environ.get("US_OTC_PROXY_FLOW_BATCH_FLUSH", "100")),
        help="Rows between partial CSV flushes for per-symbol provider scans.",
    )
    parser.add_argument("--overwrite", action="store_true")
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
    if provider == "yahoo":
        provider = "yahoo_chart"
    if provider not in {"polygon", "yahoo_chart"}:
        raise ValueError(f"unsupported US OTC proxy provider: {args.provider}")
    universe = load_otc_universe(
        args.universe_csv,
        exchange_types=_split_csv(args.exchange_types),
        max_codes=args.max_codes,
    )
    aggregate_errors: dict[str, str] = {}
    try:
        if provider == "polygon":
            api_key = resolve_api_key(args.api_key, args.api_key_file)
            aggregates = fetch_polygon_grouped_daily(api_key=api_key, date=args.date, include_otc=True)
            rows = build_proxy_records(
                universe,
                aggregates,
                date=args.date,
                provider=provider,
                min_dollar_volume=args.min_dollar_volume,
            )
        else:
            rows = scan_yahoo_chart_proxy_records(
                universe,
                output_dir=Path(args.output_dir).expanduser(),
                date=args.date,
                min_dollar_volume=args.min_dollar_volume,
                request_delay=args.request_delay,
                max_retries=args.max_retries,
                timeout=args.timeout,
                batch_flush=args.batch_flush,
                overwrite=args.overwrite,
            )
    except Exception as exc:
        write_failure_status(
            output_dir=Path(args.output_dir).expanduser(),
            date=args.date,
            provider=provider,
            message=str(exc),
            universe=universe,
            min_dollar_volume=args.min_dollar_volume,
        )
        raise
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
