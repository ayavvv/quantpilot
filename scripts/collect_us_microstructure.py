"""Collect US tick/order-book/quote data from Futu OpenD.

This is the phase-1 collector for the US microstructure major-flow project. It
writes local parquet batches first, then optionally copies completed files to a
NAS archive. The collector is intentionally small and polling-based so a short
smoke run can verify data quality before building a long-running service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd


DEFAULT_LOCAL_DIR = "~/quantpilot_data/us_microstructure"
DEFAULT_NAS_DIR = "/volume1/docker/quantpilot/us_microstructure"
DEFAULT_RSA_KEY = Path(__file__).resolve().parents[1] / "keys" / "futu_rsa_1024.pem"
DEFAULT_SYMBOLS = (
    "US.SPY",
    "US.QQQ",
    "US.LI",
    "US.YINN",
    "US.CQQQ",
    "US.KWEB",
    "US.FXI",
    "US.AAPL",
    "US.MSFT",
    "US.NVDA",
    "US.TSLA",
    "US.AMD",
    "US.AMZN",
    "US.META",
    "US.GOOGL",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalise_symbol(value: str) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        return ""
    if "." not in symbol:
        symbol = f"US.{symbol}"
    return symbol


def _normalise_symbols(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        symbol = _normalise_symbol(value)
        if symbol and symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    return result


def _symbols_from_args(raw_symbols: str, universe_file: str | None) -> list[str]:
    values: list[str] = []
    if raw_symbols:
        values.extend(item.strip() for item in raw_symbols.split(","))
    if universe_file:
        path = Path(universe_file).expanduser()
        if path.exists():
            values.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not values:
        values.extend(DEFAULT_SYMBOLS)
    return _normalise_symbols(values)


def _safe_partition_value(value: str) -> str:
    return str(value).replace("/", "_").replace(":", "_").replace(" ", "_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_range(df: pd.DataFrame) -> tuple[str, str]:
    for column in ("event_time", "time", "data_time", "recv_time"):
        if column in df.columns:
            values = df[column].dropna().astype(str)
            if not values.empty:
                return values.min(), values.max()
    return "", ""


def _partition_path(base_dir: Path, kind: str, date: str, symbol: str, run_id: str, batch_index: int) -> Path:
    return (
        base_dir
        / kind
        / f"date={date}"
        / f"symbol={_safe_partition_value(symbol)}"
        / f"part-{run_id}-{batch_index:05d}.parquet"
    )


def _write_partition(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    base_dir: Path,
    date: str,
    run_id: str,
    batch_index: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    symbol_col = "symbol" if "symbol" in df.columns else "code"
    if symbol_col not in df.columns:
        df["symbol"] = "UNKNOWN"
        symbol_col = "symbol"

    manifests: list[dict[str, Any]] = []
    for symbol, part in df.groupby(symbol_col, dropna=False, sort=True):
        symbol_text = str(symbol or "UNKNOWN")
        output = _partition_path(base_dir, kind, date, symbol_text, run_id, batch_index)
        output.parent.mkdir(parents=True, exist_ok=True)
        part = part.reset_index(drop=True)
        part.to_parquet(output, index=False)
        min_event_time, max_event_time = _event_range(part)
        manifests.append(
            {
                "kind": kind,
                "symbol": symbol_text,
                "date": date,
                "run_id": run_id,
                "batch_index": batch_index,
                "local_path": str(output),
                "row_count": int(len(part)),
                "sha256": _sha256_file(output),
                "min_event_time": min_event_time,
                "max_event_time": max_event_time,
                "created_at": _utc_now_iso(),
            }
        )
    return manifests


def _append_manifest(base_dir: Path, date: str, run_id: str, records: list[dict[str, Any]]) -> Path | None:
    if not records:
        return None
    manifest_dir = base_dir / "manifests" / f"date={date}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"manifest-{run_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _copy_to_nas(local_path: Path, local_base: Path, nas_host: str, nas_dir: str) -> tuple[str, str, str]:
    relative = local_path.relative_to(local_base)
    remote_path = f"{nas_dir.rstrip('/')}/{relative.as_posix()}"
    remote_parent = str(PurePosixPath(remote_path).parent)
    remote_command = f"mkdir -p {shlex.quote(remote_parent)} && tar -xf - -C {shlex.quote(remote_parent)}"

    tar_proc = subprocess.Popen(
        ["tar", "-cf", "-", "-C", str(local_path.parent), local_path.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tar_proc.stdout is None:
        return "failed", remote_path, "failed to open tar stdout pipe"

    ssh_proc = subprocess.Popen(
        ["ssh", nas_host, remote_command],
        stdin=tar_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_proc.stdout.close()
    ssh_stdout, ssh_stderr = ssh_proc.communicate()
    tar_stderr = tar_proc.stderr.read() if tar_proc.stderr else b""
    tar_code = tar_proc.wait()

    if tar_code != 0:
        return "failed", remote_path, tar_stderr.decode("utf-8", errors="replace").strip()
    if ssh_proc.returncode != 0:
        message = (ssh_stderr or ssh_stdout or b"").decode("utf-8", errors="replace").strip()
        return "failed", remote_path, message
    return "ok", remote_path, ""


def _sync_manifests_to_nas(
    manifests: list[dict[str, Any]],
    *,
    local_base: Path,
    nas_host: str,
    nas_dir: str,
) -> list[dict[str, Any]]:
    if not nas_host or not nas_dir:
        for record in manifests:
            record["nas_upload_status"] = "skipped"
            record["nas_path"] = ""
            record["nas_error"] = ""
        return manifests

    updated: list[dict[str, Any]] = []
    for record in manifests:
        local_path = Path(str(record["local_path"]))
        status, remote_path, error = _copy_to_nas(local_path, local_base, nas_host, nas_dir)
        item = dict(record)
        item["nas_upload_status"] = status
        item["nas_path"] = remote_path
        item["nas_error"] = error
        updated.append(item)
    return updated


def _flatten_order_book(raw: dict[str, Any], *, symbol: str, recv_time: str, levels: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "code": raw.get("code", symbol),
        "name": raw.get("name", ""),
        "recv_time": recv_time,
        "svr_recv_time_bid": raw.get("svr_recv_time_bid", ""),
        "svr_recv_time_ask": raw.get("svr_recv_time_ask", ""),
    }
    bids = list(raw.get("Bid") or [])[:levels]
    asks = list(raw.get("Ask") or [])[:levels]
    for index in range(levels):
        bid = bids[index] if index < len(bids) else (None, None, None, {})
        ask = asks[index] if index < len(asks) else (None, None, None, {})
        level = index + 1
        row[f"bid_px_{level}"] = bid[0]
        row[f"bid_sz_{level}"] = bid[1]
        row[f"bid_order_count_{level}"] = bid[2] if len(bid) > 2 else None
        row[f"ask_px_{level}"] = ask[0]
        row[f"ask_sz_{level}"] = ask[1]
        row[f"ask_order_count_{level}"] = ask[2] if len(ask) > 2 else None

    best_bid = row.get("bid_px_1")
    best_ask = row.get("ask_px_1")
    if best_bid and best_ask:
        mid = (float(best_bid) + float(best_ask)) / 2.0
        row["mid"] = mid
        row["spread_bps"] = (float(best_ask) - float(best_bid)) / mid * 10_000 if mid else None
    else:
        row["mid"] = None
        row["spread_bps"] = None
    return row


def _prepare_trade_rows(df: pd.DataFrame, *, symbol: str, recv_time: str, seen_sequences: set[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in df.to_dict("records"):
        sequence = str(raw.get("sequence") or "")
        if sequence and sequence in seen_sequences:
            continue
        if sequence:
            seen_sequences.add(sequence)
        item = dict(raw)
        item["symbol"] = symbol
        item["event_time"] = item.get("time", "")
        item["recv_time"] = recv_time
        rows.append(item)
    return rows


def _prepare_quote_rows(df: pd.DataFrame, *, recv_time: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in df.to_dict("records"):
        item = dict(raw)
        item["symbol"] = item.get("code", "")
        item["event_time"] = " ".join(str(item.get(key, "")).strip() for key in ("data_date", "data_time")).strip()
        item["recv_time"] = recv_time
        rows.append(item)
    return rows


def _flush_batch(
    buffers: dict[str, list[dict[str, Any]]],
    *,
    local_dir: Path,
    nas_host: str,
    nas_dir: str,
    date: str,
    run_id: str,
    batch_index: int,
) -> int:
    manifest_records: list[dict[str, Any]] = []
    for kind, rows in buffers.items():
        written = _write_partition(
            rows,
            kind=kind,
            base_dir=local_dir,
            date=date,
            run_id=run_id,
            batch_index=batch_index,
        )
        manifest_records.extend(
            _sync_manifests_to_nas(written, local_base=local_dir, nas_host=nas_host, nas_dir=nas_dir)
        )
        rows.clear()
    manifest_path = _append_manifest(local_dir, date, run_id, manifest_records)
    if manifest_path and nas_host and nas_dir:
        status, _, error = _copy_to_nas(manifest_path, local_dir, nas_host, nas_dir)
        if status != "ok":
            print(f"WARNING: manifest NAS sync failed: {error}")
    return len(manifest_records)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect US microstructure data from Futu OpenD.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols, e.g. US.AAPL,NVDA,SPY.")
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--book-interval-seconds", type=float, default=1.0)
    parser.add_argument("--quote-interval-seconds", type=float, default=5.0)
    parser.add_argument("--batch-seconds", type=float, default=60.0)
    parser.add_argument("--book-levels", type=int, default=10)
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", "11111")))
    parser.add_argument("--rsa-key", default=os.environ.get("FUTU_RSA_KEY", str(DEFAULT_RSA_KEY)))
    parser.add_argument("--local-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", DEFAULT_LOCAL_DIR))
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    parser.add_argument("--extended-time", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from futu import OpenQuoteContext, RET_OK, SubType, SysConfig

    args = _parse_args(argv)
    symbols = _symbols_from_args(args.symbols, args.universe_file)
    if not symbols:
        raise ValueError("no symbols configured")

    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    nas_host = "" if args.no_nas_sync else str(args.nas_host or "")
    nas_dir = "" if args.no_nas_sync else str(args.nas_dir or "")

    rsa_key = Path(args.rsa_key).expanduser() if args.rsa_key else None
    if rsa_key and rsa_key.exists():
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(str(rsa_key))
    else:
        SysConfig.enable_proto_encrypt(False)
        print(f"WARNING: Futu RSA key not found, connecting without encryption: {rsa_key}")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d")
    buffers: dict[str, list[dict[str, Any]]] = {
        "trades": [],
        "order_book": [],
        "quotes": [],
    }
    seen_sequences: dict[str, set[str]] = defaultdict(set)
    start = time.monotonic()
    end = start + max(1, args.duration_seconds)
    next_book = start
    next_quote = start
    next_flush = start + max(1.0, args.batch_seconds)
    batch_index = 0

    ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        ret, data = ctx.subscribe(
            symbols,
            [SubType.TICKER, SubType.ORDER_BOOK, SubType.QUOTE],
            subscribe_push=False,
            extended_time=bool(args.extended_time),
        )
        if ret != RET_OK:
            raise RuntimeError(f"Futu subscribe failed: {data}")
        print(
            f"collecting symbols={len(symbols)} duration={args.duration_seconds}s "
            f"local_dir={local_dir} nas={nas_host}:{nas_dir if nas_host else ''}"
        )

        while time.monotonic() < end:
            loop_time = time.monotonic()
            recv_time = _utc_now_iso()
            for symbol in symbols:
                ret, data = ctx.get_rt_ticker(symbol, num=1000)
                if ret == RET_OK:
                    buffers["trades"].extend(
                        _prepare_trade_rows(data, symbol=symbol, recv_time=recv_time, seen_sequences=seen_sequences[symbol])
                    )
                else:
                    print(f"WARNING: get_rt_ticker failed symbol={symbol}: {data}")

            if loop_time >= next_book:
                for symbol in symbols:
                    ret, data = ctx.get_order_book(symbol, num=max(1, args.book_levels))
                    if ret == RET_OK:
                        buffers["order_book"].append(
                            _flatten_order_book(data, symbol=symbol, recv_time=recv_time, levels=max(1, args.book_levels))
                        )
                    else:
                        print(f"WARNING: get_order_book failed symbol={symbol}: {data}")
                next_book = loop_time + max(0.2, args.book_interval_seconds)

            if loop_time >= next_quote:
                ret, data = ctx.get_stock_quote(symbols)
                if ret == RET_OK:
                    buffers["quotes"].extend(_prepare_quote_rows(data, recv_time=recv_time))
                else:
                    print(f"WARNING: get_stock_quote failed: {data}")
                next_quote = loop_time + max(0.5, args.quote_interval_seconds)

            if loop_time >= next_flush:
                batch_index += 1
                count = _flush_batch(
                    buffers,
                    local_dir=local_dir,
                    nas_host=nas_host,
                    nas_dir=nas_dir,
                    date=date,
                    run_id=run_id,
                    batch_index=batch_index,
                )
                print(f"flushed batch={batch_index} manifest_records={count}")
                next_flush = loop_time + max(1.0, args.batch_seconds)

            sleep_for = max(0.0, args.poll_interval_seconds - (time.monotonic() - loop_time))
            time.sleep(sleep_for)

        batch_index += 1
        count = _flush_batch(
            buffers,
            local_dir=local_dir,
            nas_host=nas_host,
            nas_dir=nas_dir,
            date=date,
            run_id=run_id,
            batch_index=batch_index,
        )
        print(f"flushed final batch={batch_index} manifest_records={count}")
        return 0
    finally:
        try:
            ctx.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
