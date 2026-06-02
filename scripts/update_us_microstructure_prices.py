"""Update daily close prices used by US microstructure forward validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from scripts.collect_us_microstructure import DEFAULT_RSA_KEY, DEFAULT_SYMBOLS, _copy_to_nas
from strategy.us_microstructure_features import normalize_us_symbol, normalize_us_symbols
from strategy.us_microstructure_validation import discover_signal_files


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
DEFAULT_NAS_DIR = "/volume1/docker/quantpilot/us_microstructure"
DEFAULT_UNIVERSE_FILE = "universe/us_microstructure_candidates_latest.txt"


def _default_end_date() -> str:
    return date.today().isoformat()


def _date_days_before(end_date: str, days: int) -> str:
    end = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
    return (end - timedelta(days=max(1, int(days)))).isoformat()


def _parse_symbols(raw: str | Iterable[object]) -> list[str]:
    if isinstance(raw, str):
        values = [item for item in raw.split(",") if item.strip()]
    else:
        values = list(raw)
    return normalize_us_symbols(values)


def _symbols_from_signal_files(base_dir: Path, *, start_date: str = "", end_date: str = "") -> list[str]:
    symbols: list[str] = []
    for path in discover_signal_files(base_dir, start_date=start_date, end_date=end_date):
        try:
            df = pd.read_csv(path, usecols=lambda column: column in {"symbol", "code"})
        except Exception:
            continue
        column = "symbol" if "symbol" in df.columns else "code" if "code" in df.columns else ""
        if column:
            symbols.extend(df[column].dropna().tolist())
    return normalize_us_symbols(symbols)


def _symbols_from_universe_file(base_dir: Path, universe_file: str = "") -> list[str]:
    raw_path = str(universe_file or "").strip()
    path = Path(raw_path).expanduser() if raw_path else base_dir / DEFAULT_UNIVERSE_FILE
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return []
    try:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []
    return normalize_us_symbols(values)


def build_price_symbol_universe(
    base_dir: str | Path,
    *,
    explicit_symbols: Iterable[object] | None = None,
    benchmark: str = "US.SPY",
    include_signal_symbols: bool = True,
    include_default_symbols: bool = True,
    include_universe_symbols: bool = True,
    universe_file: str = "",
    start_date: str = "",
    end_date: str = "",
) -> list[str]:
    base_path = Path(base_dir).expanduser()
    values: list[object] = []
    values.append(benchmark)
    if include_default_symbols:
        values.extend(DEFAULT_SYMBOLS)
    if explicit_symbols:
        values.extend(explicit_symbols)
    if include_signal_symbols:
        values.extend(_symbols_from_signal_files(base_path, start_date=start_date, end_date=end_date))
    if include_universe_symbols:
        values.extend(_symbols_from_universe_file(base_path, universe_file=universe_file))
    return normalize_us_symbols(values)


def normalize_kline_rows(rows: list[dict], *, symbol: str, source: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df
    if "time_key" in df.columns:
        df["date"] = df["time_key"].astype(str).str[:10]
    elif "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    else:
        raise ValueError("Futu kline rows missing time_key/date")
    df["symbol"] = normalize_us_symbol(symbol)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["amount"] = df["turnover"]
    df["source"] = source
    df["updated_at"] = datetime.now().isoformat(timespec="seconds")
    columns = ["date", "symbol", "open", "high", "low", "close", "volume", "turnover", "amount", "source", "updated_at"]
    return df[columns].dropna(subset=["date", "symbol", "close"]).copy()


def merge_price_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for frame in (existing, incoming):
        if frame is None or frame.empty:
            continue
        part = frame.copy()
        if "symbol" not in part.columns and "code" in part.columns:
            part["symbol"] = part["code"]
        if "date" not in part.columns or "symbol" not in part.columns or "close" not in part.columns:
            continue
        part["date"] = part["date"].astype(str).str[:10]
        part["symbol"] = part["symbol"].map(normalize_us_symbol)
        for column in ("open", "high", "low", "close", "volume", "turnover", "amount"):
            if column not in part.columns:
                part[column] = pd.NA
            part[column] = pd.to_numeric(part[column], errors="coerce")
        if "source" not in part.columns:
            part["source"] = ""
        if "updated_at" not in part.columns:
            part["updated_at"] = ""
        frames.append(part[["date", "symbol", "open", "high", "low", "close", "volume", "turnover", "amount", "source", "updated_at"]])
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "turnover", "amount", "source", "updated_at"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["date", "symbol", "close"])
    merged = merged[merged["symbol"] != ""].copy()
    merged = merged.sort_values(["symbol", "date", "updated_at"])
    merged = merged.drop_duplicates(["date", "symbol"], keep="last")
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def _read_existing_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fetch_futu_daily_prices(
    ctx,
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
    autype,
    ktype,
    sleep_seconds: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, str]]:
    from futu import RET_OK

    frames = []
    errors: dict[str, str] = {}
    for symbol in symbols:
        page_req_key = None
        symbol_rows: list[dict] = []
        while True:
            ret, data, page_req_key = ctx.request_history_kline(
                code=symbol,
                start=start_date,
                end=end_date,
                ktype=ktype,
                autype=autype,
                max_count=1000,
                page_req_key=page_req_key,
            )
            if ret != RET_OK:
                errors[symbol] = str(data)
                break
            if data is not None and len(data) > 0:
                symbol_rows.extend(data.to_dict("records"))
            if page_req_key is None:
                break
        if symbol_rows:
            frames.append(normalize_kline_rows(symbol_rows, symbol=symbol, source="futu_kday"))
        elif symbol not in errors:
            errors[symbol] = "empty"
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    incoming = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return incoming, errors


def write_price_outputs(
    base_dir: str | Path,
    *,
    prices: pd.DataFrame,
    errors: dict[str, str],
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, Path]:
    output_dir = Path(base_dir).expanduser() / "validation" / "prices"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "us_daily_prices.csv"
    parquet_path = output_dir / "us_daily_prices.parquet"
    status_path = output_dir / "us_daily_prices_status.json"
    prices.to_csv(csv_path, index=False)
    prices.to_parquet(parquet_path, index=False)
    status = {
        "status": "ok" if not errors else "partial",
        "symbol_count": len(symbols),
        "price_row_count": int(len(prices)),
        "start_date": start_date,
        "end_date": end_date,
        "errors": errors,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"csv": csv_path, "parquet": parquet_path, "status": status_path}


def update_price_history(
    base_dir: str | Path,
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    fetcher: Callable[[list[str], str, str], tuple[pd.DataFrame, dict[str, str]]],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Path]]:
    output_dir = Path(base_dir).expanduser() / "validation" / "prices"
    csv_path = output_dir / "us_daily_prices.csv"
    existing = _read_existing_prices(csv_path)
    incoming, errors = fetcher(symbols, start_date, end_date)
    merged = merge_price_frames(existing, incoming)
    outputs = write_price_outputs(
        base_dir,
        prices=merged,
        errors=errors,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    return merged, errors, outputs


def _configure_futu_encryption(rsa_key: str):
    from futu import SysConfig

    key = Path(rsa_key).expanduser() if rsa_key else None
    if key and key.exists():
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(str(key))
    else:
        SysConfig.enable_proto_encrypt(False)


def _autype_from_text(value: str):
    from futu import AuType

    text = str(value or "").strip().lower()
    if text == "none":
        return AuType.NONE
    if text == "hfq":
        return AuType.HFQ
    return AuType.QFQ


def _sync_outputs(paths: Iterable[Path], *, base_dir: Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    results = []
    if not nas_host or not nas_dir:
        return results
    for path in paths:
        status, remote_path, error = _copy_to_nas(path, base_dir, nas_host, nas_dir)
        results.append({"local_path": str(path), "nas_path": remote_path, "status": status, "error": error})
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update daily US prices for microstructure validation.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--symbols", default=os.environ.get("US_MICROSTRUCTURE_PRICE_SYMBOLS", ""))
    parser.add_argument("--benchmark", default=os.environ.get("US_MICROSTRUCTURE_BENCHMARK", "US.SPY"))
    parser.add_argument("--start-date", default=os.environ.get("US_MICROSTRUCTURE_PRICE_START", ""))
    parser.add_argument("--end-date", default=os.environ.get("US_MICROSTRUCTURE_PRICE_END", _default_end_date()))
    parser.add_argument("--lookback-days", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_PRICE_LOOKBACK_DAYS", "45")))
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", "11111")))
    parser.add_argument("--rsa-key", default=os.environ.get("FUTU_RSA_KEY", str(DEFAULT_RSA_KEY)))
    parser.add_argument("--autype", default=os.environ.get("US_MICROSTRUCTURE_PRICE_AUTYPE", "qfq"), choices=["qfq", "hfq", "none"])
    parser.add_argument("--sleep-seconds", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_PRICE_SLEEP_SECONDS", "0.2")))
    parser.add_argument("--universe-file", default=os.environ.get("US_MICROSTRUCTURE_PRICE_UNIVERSE_FILE", ""))
    parser.add_argument("--no-default-symbols", action="store_true")
    parser.add_argument("--no-signal-symbols", action="store_true")
    parser.add_argument("--no-universe-symbols", action="store_true")
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from futu import KLType, OpenQuoteContext

    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    end_date = str(args.end_date)[:10]
    start_date = str(args.start_date or _date_days_before(end_date, args.lookback_days))[:10]
    symbols = build_price_symbol_universe(
        base_dir,
        explicit_symbols=_parse_symbols(args.symbols),
        benchmark=args.benchmark,
        include_signal_symbols=not args.no_signal_symbols,
        include_default_symbols=not args.no_default_symbols,
        include_universe_symbols=not args.no_universe_symbols,
        universe_file=args.universe_file,
        start_date=start_date,
        end_date=end_date,
    )
    if not symbols:
        raise ValueError("no symbols configured for price update")

    _configure_futu_encryption(args.rsa_key)
    ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        autype = _autype_from_text(args.autype)

        def fetcher(fetch_symbols: list[str], fetch_start: str, fetch_end: str):
            return fetch_futu_daily_prices(
                ctx,
                fetch_symbols,
                start_date=fetch_start,
                end_date=fetch_end,
                autype=autype,
                ktype=KLType.K_DAY,
                sleep_seconds=args.sleep_seconds,
            )

        prices, errors, outputs = update_price_history(
            base_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fetcher=fetcher,
        )
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    sync_results = []
    if not args.no_nas_sync:
        sync_results = _sync_outputs(outputs.values(), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        if sync_results:
            status_path = outputs["status"]
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["nas_sync"] = sync_results
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if args.nas_host and args.nas_dir:
                _copy_to_nas(status_path, base_dir, args.nas_host, args.nas_dir)

    print(f"Updated US microstructure prices: rows={len(prices)} symbols={len(symbols)} errors={len(errors)}")
    print(f"Wrote price CSV: {outputs['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
