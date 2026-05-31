"""Forward-return validation for archived Futu capital-flow overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from converter.incremental import QlibBinReader


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return tuple(values)


def discover_overlay_files(archive_dir: str | Path) -> list[Path]:
    path = Path(archive_dir).expanduser()
    if not path.exists():
        return []
    return sorted(path.glob("*_overlay.csv"))


def load_archived_overlays(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["overlay_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "signal_date" not in result.columns:
        raise ValueError("archived overlay is missing signal_date")
    if "capital_flow_label" not in result.columns:
        raise ValueError("archived overlay is missing capital_flow_label")
    return result


def _code_to_fname(code: str) -> str:
    replace_names = ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(10)] + [f"LPT{i}" for i in range(10)]
    if str(code).upper() in replace_names:
        return "_qlib_" + str(code)
    return str(code)


def _read_close(reader: QlibBinReader, qlib_dir: Path, code: str) -> pd.Series:
    feat_dir = qlib_dir / "features" / _code_to_fname(code).lower()
    bin_path = feat_dir / "close.day.bin"
    if not bin_path.exists():
        return pd.Series(dtype="float64")
    data = np.fromfile(str(bin_path), dtype="<f4")
    if len(data) == 0:
        return pd.Series(dtype="float64")
    start_idx = int(data[0])
    values = data[1:]
    end_idx = start_idx + len(values)
    dates = reader.calendar[start_idx:end_idx]
    return pd.Series(values.astype("float64"), index=dates, name="close")


def _forward_return(close: pd.Series, as_of_date: str, horizon: int, entry_lag_days: int) -> float:
    valid = close[pd.to_numeric(close, errors="coerce").notna()]
    if as_of_date not in valid.index:
        return np.nan
    loc = valid.index.get_loc(as_of_date)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return np.nan
    entry_idx = int(loc) + entry_lag_days
    exit_idx = entry_idx + horizon
    if entry_idx < 0 or exit_idx >= len(valid):
        return np.nan
    entry = float(valid.iloc[entry_idx])
    exit_ = float(valid.iloc[exit_idx])
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0:
        return np.nan
    return exit_ / entry - 1.0


def evaluate_archived_capital_flow_overlays(
    qlib_dir: str | Path,
    overlays: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 5),
    entry_lag_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate archived Futu overlay labels against future close returns.

    Returns ``(summary, rows)``. The row-level output contains one row per
    overlay pick and horizon; the summary groups by label and compares each
    label against the same-date archived overlay universe.
    """

    if overlays.empty:
        return pd.DataFrame(), pd.DataFrame()

    qlib_path = Path(qlib_dir).expanduser()
    reader = QlibBinReader(qlib_path)
    close_cache: dict[str, pd.Series] = {}
    row_records = []

    for _, row in overlays.iterrows():
        code = str(row.get("code", ""))
        signal_date = str(row.get("signal_date", ""))[:10]
        if not code or not signal_date:
            continue
        close = close_cache.get(code)
        if close is None:
            close = _read_close(reader, qlib_path, code)
            close_cache[code] = close
        if close.empty:
            continue
        base = row.to_dict()
        for horizon in horizons:
            value = _forward_return(close, signal_date, horizon, entry_lag_days)
            if np.isfinite(value):
                out = dict(base)
                out["horizon"] = horizon
                out["fwd_return"] = float(value)
                row_records.append(out)

    rows = pd.DataFrame(row_records)
    if rows.empty:
        return pd.DataFrame(), rows

    universe = (
        rows.groupby(["signal_date", "horizon"], as_index=False)
        .agg(universe_return=("fwd_return", "mean"), universe_count=("code", "count"))
    )
    daily = (
        rows.groupby(["signal_date", "horizon", "capital_flow_label"], as_index=False)
        .agg(
            selected_count=("code", "count"),
            avg_model_rank=("model_rank", "mean"),
            avg_return=("fwd_return", "mean"),
            median_return=("fwd_return", "median"),
            hit_rate=("fwd_return", lambda s: float((s > 0).mean())),
        )
        .merge(universe, on=["signal_date", "horizon"], how="left")
    )
    daily["alpha"] = daily["avg_return"] - daily["universe_return"]
    summary = (
        daily.groupby(["capital_flow_label", "horizon"], as_index=False)
        .agg(
            date_count=("signal_date", "nunique"),
            avg_selected_count=("selected_count", "mean"),
            avg_model_rank=("avg_model_rank", "mean"),
            avg_return=("avg_return", "mean"),
            median_return=("median_return", "median"),
            avg_universe_return=("universe_return", "mean"),
            avg_alpha=("alpha", "mean"),
            win_rate_days=("avg_return", lambda s: float((s > 0).mean())),
            avg_hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["horizon", "capital_flow_label"])
        .reset_index(drop=True)
    )
    return summary, rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate archived Futu capital-flow overlays.")
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--archive-dir", default="~/quantpilot_data/capital_flow/futu")
    parser.add_argument("--overlay-csv", action="append", default=[])
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--entry-lag-days", type=int, default=1)
    parser.add_argument("--output-dir", default="~/quantpilot_data/output/futu_capital_flow_eval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = [Path(item).expanduser() for item in args.overlay_csv]
    if not paths:
        paths = discover_overlay_files(args.archive_dir)
    overlays = load_archived_overlays(paths)
    summary, rows = evaluate_archived_capital_flow_overlays(
        args.qlib_dir,
        overlays,
        horizons=_parse_int_tuple(args.horizons),
        entry_lag_days=max(0, args.entry_lag_days),
    )

    if summary.empty:
        print("No forward-return rows produced.")
    else:
        display = summary.copy()
        for col in ["avg_return", "median_return", "avg_universe_return", "avg_alpha", "win_rate_days", "avg_hit_rate"]:
            if col in display.columns:
                display[col] = display[col].map(lambda v: f"{v:.2%}")
        print(display.to_string(index=False))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows_path = output_dir / "rows.csv"
    summary.to_csv(summary_path, index=False)
    rows.to_csv(rows_path, index=False)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote rows: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
