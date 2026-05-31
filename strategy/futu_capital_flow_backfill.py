"""Backfill archived Futu capital-flow overlays from historical signal CSVs."""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from collector.config import settings
from strategy.futu_capital_flow_overlay import (
    archive_capital_flow_outputs,
    build_capital_flow_overlay,
    fetch_capital_flow_summaries,
)


SIGNAL_FILE_RE = re.compile(r"^signal_(\d{8})\.csv$")


def _date_from_signal_path(path: Path) -> str:
    match = SIGNAL_FILE_RE.match(path.name)
    if not match:
        return ""
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def discover_signal_files(
    signal_dir: str | Path,
    *,
    start_date: str = "",
    end_date: str = "",
) -> list[Path]:
    """Return date-stamped signal CSVs sorted by signal date."""

    directory = Path(signal_dir).expanduser()
    if not directory.exists():
        return []

    paths = []
    for path in directory.glob("signal_*.csv"):
        signal_date = _date_from_signal_path(path)
        if not signal_date:
            continue
        if start_date and signal_date < start_date:
            continue
        if end_date and signal_date > end_date:
            continue
        paths.append(path)
    return sorted(paths, key=_date_from_signal_path)


def select_signal_files(
    signal_dir: str | Path,
    *,
    start_date: str = "",
    end_date: str = "",
    max_dates: int = 0,
) -> list[Path]:
    paths = discover_signal_files(signal_dir, start_date=start_date, end_date=end_date)
    if max_dates > 0:
        paths = paths[-max_dates:]
    return paths


def _default_start(signal_date: str, days: int) -> str:
    parsed = datetime.strptime(signal_date, "%Y-%m-%d")
    return (parsed - timedelta(days=days)).strftime("%Y-%m-%d")


def _read_signal_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path).expanduser())
    if df.empty:
        return df
    if "rank" not in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    if "signal_date" not in df.columns:
        signal_date = _date_from_signal_path(Path(path))
        if signal_date:
            df["signal_date"] = signal_date
    return df


def backfill_one_signal_file(
    signal_csv: str | Path,
    *,
    archive_dir: str | Path,
    signal_top_n: int = 30,
    flow_days: int = 30,
    host: str = settings.futu_host,
    port: int = settings.futu_port,
    period: str = "DAY",
    include_distribution: bool = False,
    connect_timeout: float = 8.0,
    min_ok_ratio: float = 0.5,
    overwrite: bool = False,
    fetcher: Callable[..., pd.DataFrame] = fetch_capital_flow_summaries,
) -> dict[str, object]:
    signal_path = Path(signal_csv).expanduser()
    signal_df = _read_signal_csv(signal_path)
    if signal_df.empty:
        raise ValueError(f"empty signal csv: {signal_path}")
    if "code" not in signal_df.columns:
        raise ValueError(f"signal csv missing code column: {signal_path}")

    if "signal_date" in signal_df.columns:
        signal_date = str(signal_df["signal_date"].dropna().iloc[0])[:10]
    else:
        signal_date = _date_from_signal_path(signal_path)
    if not signal_date:
        raise ValueError(f"could not resolve signal date from {signal_path}")

    archive_path = Path(archive_dir).expanduser()
    overlay_path = archive_path / f"{signal_date.replace('-', '')}_overlay.csv"
    flow_path = archive_path / f"{signal_date.replace('-', '')}_flow.csv"
    if overlay_path.exists() and flow_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "signal_date": signal_date,
            "signal_csv": str(signal_path),
            "overlay": overlay_path,
            "flow": flow_path,
            "row_count": 0,
        }

    signals = signal_df.copy()
    signals = signals.sort_values("rank").head(signal_top_n)
    codes = signals["code"].astype(str).tolist()
    start = _default_start(signal_date, flow_days)
    flow = fetcher(
        codes,
        host=host,
        port=port,
        start=start,
        end=signal_date,
        period=period,
        include_distribution=include_distribution,
        connect_timeout=connect_timeout,
    )
    ok_count = int((flow.get("capital_flow_status") == "ok").sum()) if not flow.empty else 0
    ok_ratio = ok_count / max(len(codes), 1)
    if ok_ratio < min_ok_ratio:
        return {
            "status": "failed",
            "signal_date": signal_date,
            "signal_csv": str(signal_path),
            "overlay": overlay_path,
            "flow": flow_path,
            "row_count": 0,
            "ok_count": ok_count,
            "ok_ratio": ok_ratio,
            "error": f"capital-flow ok ratio below threshold: {ok_ratio:.1%} < {min_ok_ratio:.1%}",
        }

    overlay = build_capital_flow_overlay(signal_df, flow, signal_top_n=signal_top_n)
    paths = archive_capital_flow_outputs(flow, overlay, archive_path, archive_date=signal_date)
    return {
        "status": "written",
        "signal_date": signal_date,
        "signal_csv": str(signal_path),
        "overlay": paths["overlay"],
        "flow": paths["flow"],
        "row_count": int(len(overlay)),
        "ok_count": ok_count,
        "ok_ratio": ok_ratio,
    }


def backfill_capital_flow_archives(
    signal_dir: str | Path,
    *,
    archive_dir: str | Path,
    start_date: str = "",
    end_date: str = "",
    max_dates: int = 0,
    signal_top_n: int = 30,
    flow_days: int = 30,
    host: str = settings.futu_host,
    port: int = settings.futu_port,
    period: str = "DAY",
    include_distribution: bool = False,
    connect_timeout: float = 8.0,
    date_pause_seconds: float = 0.0,
    min_ok_ratio: float = 0.5,
    overwrite: bool = False,
    fetcher: Callable[..., pd.DataFrame] = fetch_capital_flow_summaries,
) -> list[dict[str, object]]:
    paths = select_signal_files(
        signal_dir,
        start_date=start_date,
        end_date=end_date,
        max_dates=max_dates,
    )
    results = []
    for idx, path in enumerate(paths):
        results.append(
            backfill_one_signal_file(
                path,
                archive_dir=archive_dir,
                signal_top_n=signal_top_n,
                flow_days=flow_days,
                host=host,
                port=port,
                period=period,
                include_distribution=include_distribution,
                connect_timeout=connect_timeout,
                min_ok_ratio=min_ok_ratio,
                overwrite=overwrite,
                fetcher=fetcher,
            )
        )
        if date_pause_seconds > 0 and idx < len(paths) - 1:
            time.sleep(date_pause_seconds)
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill archived Futu capital-flow overlays.")
    parser.add_argument("--signal-dir", default=os.environ.get("SIGNAL_DIR", "~/quantpilot_data/signals"))
    parser.add_argument("--archive-dir", default="~/quantpilot_data/capital_flow/futu")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--signal-top-n", type=int, default=30)
    parser.add_argument("--flow-days", type=int, default=30)
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", settings.futu_host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", settings.futu_port)))
    parser.add_argument("--period", default="DAY")
    parser.add_argument("--include-distribution", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument("--date-pause-seconds", type=float, default=5.0)
    parser.add_argument("--min-ok-ratio", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results = backfill_capital_flow_archives(
        args.signal_dir,
        archive_dir=args.archive_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=max(0, args.max_dates),
        signal_top_n=max(1, args.signal_top_n),
        flow_days=max(1, args.flow_days),
        host=args.host,
        port=args.port,
        period=args.period,
        include_distribution=args.include_distribution,
        connect_timeout=args.connect_timeout,
        date_pause_seconds=max(0.0, args.date_pause_seconds),
        min_ok_ratio=max(0.0, min(1.0, args.min_ok_ratio)),
        overwrite=args.overwrite,
    )
    if not results:
        print("No signal files selected.")
        return 0
    for result in results:
        print(
            f"{result['signal_date']} {result['status']} rows={result['row_count']} "
            f"ok={result.get('ok_count', 0)} ratio={result.get('ok_ratio', 0):.1%} "
            f"overlay={result['overlay']}"
            + (f" error={result['error']}" if result.get("error") else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
