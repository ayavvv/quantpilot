"""Historical validation for the A-share major-flow proxy scanner."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from converter.incremental import QlibBinReader
from strategy.major_force import (
    A_SHARE_PREFIXES,
    DEFAULT_FIELDS,
    MajorForceConfig,
    compute_major_force_metrics,
    score_major_force_frame,
)


@dataclass(frozen=True)
class MajorForceEvalConfig:
    """Configuration for forward-return validation."""

    top_ns: tuple[int, ...] = (10, 30, 50)
    horizons: tuple[int, ...] = (5, 10, 20)
    entry_lag_days: int = 1
    date_step: int = 1
    max_dates: int | None = None
    min_active_stocks: int = 100
    min_score: float | None = None
    stages: tuple[str, ...] | None = None


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return tuple(values)


def _window_bound(calendar: list[str], date_value: str, offset: int) -> str:
    if not calendar:
        return date_value
    try:
        pos = calendar.index(date_value)
    except ValueError:
        pos = max(0, min(len(calendar) - 1, np.searchsorted(calendar, date_value)))
    target = max(0, min(len(calendar) - 1, pos + offset))
    return calendar[target]


def _evaluation_dates(
    calendar: list[str],
    start_date: str,
    end_date: str,
    *,
    date_step: int = 1,
) -> list[str]:
    dates = [date for date in calendar if start_date <= date <= end_date]
    step = max(1, int(date_step))
    if step > 1:
        dates = dates[::step]
    return dates


def _load_stock_frames(
    reader: QlibBinReader,
    instruments: dict[str, tuple[str, str]],
    *,
    prefixes: tuple[str, ...],
    min_load_date: str,
    max_load_date: str,
    min_rows: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for code, (_, end_date) in sorted(instruments.items()):
        if not code.startswith(prefixes) or end_date < min_load_date:
            continue
        df = reader.read_stock(code, DEFAULT_FIELDS)
        if df.empty:
            continue
        df = df.loc[min_load_date:max_load_date].sort_index()
        if len(df) >= min_rows:
            frames[code] = df
    return frames


def _forward_return(
    df: pd.DataFrame,
    as_of_date: str,
    horizon: int,
    entry_lag_days: int,
) -> float:
    if df.empty or "close" not in df.columns:
        return np.nan
    valid = df[pd.to_numeric(df["close"], errors="coerce").notna()]
    if as_of_date not in valid.index:
        return np.nan
    loc = valid.index.get_loc(as_of_date)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return np.nan
    entry_idx = int(loc) + entry_lag_days
    exit_idx = entry_idx + horizon
    if entry_idx < 0 or exit_idx >= len(valid):
        return np.nan
    entry = float(valid.iloc[entry_idx]["close"])
    exit_ = float(valid.iloc[exit_idx]["close"])
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0:
        return np.nan
    return exit_ / entry - 1.0


def _returns_for_ranked(
    ranked: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    as_of_date: str,
    horizon: int,
    entry_lag_days: int,
) -> pd.DataFrame:
    rows = []
    for code in ranked["code"].astype(str):
        df = frames.get(code)
        if df is None:
            continue
        value = _forward_return(df, as_of_date, horizon, entry_lag_days)
        if np.isfinite(value):
            rows.append({"code": code, "fwd_return": float(value)})
    if not rows:
        return pd.DataFrame(columns=["code", "fwd_return"])
    return pd.DataFrame(rows)


def evaluate_major_force_forward_returns(
    qlib_dir: str | Path,
    start_date: str,
    end_date: str,
    *,
    scan_config: MajorForceConfig | None = None,
    eval_config: MajorForceEvalConfig | None = None,
    prefixes: Iterable[str] = A_SHARE_PREFIXES,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate ranked major-flow candidates against future close returns.

    Returns ``(summary, daily, picks)``:
    - summary: aggregate by top_n and horizon
    - daily: daily equal-weight return and universe comparison
    - picks: per-date top max(top_ns) rows with forward return columns
    """

    scan_cfg = scan_config or MajorForceConfig()
    eval_cfg = eval_config or MajorForceEvalConfig()
    prefix_tuple = tuple(prefixes)
    reader = QlibBinReader(qlib_dir)
    calendar = reader.calendar
    dates = _evaluation_dates(
        calendar,
        start_date,
        end_date,
        date_step=eval_cfg.date_step,
    )
    if not dates:
        empty = pd.DataFrame()
        return empty, empty, empty

    max_horizon = max(eval_cfg.horizons)
    min_load_date = _window_bound(calendar, start_date, -scan_cfg.lookback_days * 2 - 10)
    max_load_date = _window_bound(calendar, end_date, (eval_cfg.entry_lag_days + max_horizon) * 2 + 10)
    instruments = reader.list_instruments("all")
    frames = _load_stock_frames(
        reader,
        instruments,
        prefixes=prefix_tuple,
        min_load_date=min_load_date,
        max_load_date=max_load_date,
        min_rows=scan_cfg.min_history,
    )
    active_date_counts: dict[str, int] = {}
    for df in frames.values():
        valid_close_dates = df.index[pd.to_numeric(df["close"], errors="coerce").notna()]
        for value in valid_close_dates:
            date = str(value)[:10]
            active_date_counts[date] = active_date_counts.get(date, 0) + 1
    dates = [date for date in dates if active_date_counts.get(date, 0) >= eval_cfg.min_active_stocks]
    if eval_cfg.max_dates is not None and eval_cfg.max_dates > 0:
        dates = dates[-eval_cfg.max_dates:]
    if not dates:
        empty = pd.DataFrame()
        return empty, empty, empty
    if progress:
        print(
            f"[major_force_eval] loaded {len(frames)} stock frames; "
            f"evaluating {len(dates)} A-share dates from {dates[0]} to {dates[-1]}",
            flush=True,
        )

    max_top_n = max(eval_cfg.top_ns)
    daily_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []

    for date_idx, date in enumerate(dates, start=1):
        metric_rows = []
        for code, df in frames.items():
            if date not in df.index:
                continue
            metrics = compute_major_force_metrics(code, df, as_of_date=date, config=scan_cfg)
            if metrics is not None:
                metric_rows.append(metrics)
        if not metric_rows:
            if progress:
                print(f"[major_force_eval] {date_idx}/{len(dates)} {date}: no candidates", flush=True)
            continue

        all_ranked = score_major_force_frame(pd.DataFrame(metric_rows))
        if all_ranked.empty:
            if progress:
                print(f"[major_force_eval] {date_idx}/{len(dates)} {date}: empty universe ranking", flush=True)
            continue

        ranked = all_ranked.copy()
        if eval_cfg.stages:
            ranked = ranked[ranked["stage"].isin(eval_cfg.stages)]
        if eval_cfg.min_score is not None:
            ranked = ranked[ranked["score"] >= eval_cfg.min_score]
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        if ranked.empty:
            if progress:
                print(f"[major_force_eval] {date_idx}/{len(dates)} {date}: empty ranking", flush=True)
            continue
        if progress:
            print(
                f"[major_force_eval] {date_idx}/{len(dates)} {date}: "
                f"candidates={len(ranked)} universe={len(all_ranked)}",
                flush=True,
            )

        top_picks = ranked.head(max_top_n).copy()
        top_picks["eval_date"] = date
        for horizon in eval_cfg.horizons:
            returns_df = _returns_for_ranked(all_ranked, frames, date, horizon, eval_cfg.entry_lag_days)
            if returns_df.empty:
                continue
            selected_returns = ranked.merge(returns_df, on="code", how="inner")
            universe_ret = float(returns_df["fwd_return"].mean())
            universe_count = int(len(returns_df))

            pick_ret_map = dict(zip(returns_df["code"], returns_df["fwd_return"]))
            top_picks[f"fwd_return_{horizon}d"] = top_picks["code"].map(pick_ret_map)

            for top_n in eval_cfg.top_ns:
                selected = selected_returns.head(top_n)
                if selected.empty:
                    continue
                selected_ret = float(selected["fwd_return"].mean())
                daily_rows.append(
                    {
                        "date": date,
                        "horizon": horizon,
                        "top_n": top_n,
                        "selected_count": int(len(selected)),
                        "universe_count": universe_count,
                        "avg_score": float(selected["score"].mean()),
                        "selected_return": selected_ret,
                        "universe_return": universe_ret,
                        "alpha": selected_ret - universe_ret,
                        "hit_rate": float((selected["fwd_return"] > 0).mean()),
                    }
                )

        pick_cols = [
            "eval_date",
            "rank",
            "code",
            "date",
            "score",
            "stage",
            "close",
            "today_chg_pct",
            "amount_ratio_5_20",
            "cmf_20",
            "close_location_10",
            "breakout_20",
            "reason",
        ] + [f"fwd_return_{horizon}d" for horizon in eval_cfg.horizons]
        pick_rows.extend(top_picks[[col for col in pick_cols if col in top_picks.columns]].to_dict("records"))

    daily = pd.DataFrame(daily_rows)
    picks = pd.DataFrame(pick_rows)
    if daily.empty:
        return pd.DataFrame(), daily, picks

    summary = (
        daily.groupby(["top_n", "horizon"], as_index=False)
        .agg(
            date_count=("date", "nunique"),
            avg_selected_count=("selected_count", "mean"),
            avg_score=("avg_score", "mean"),
            avg_return=("selected_return", "mean"),
            median_return=("selected_return", "median"),
            avg_universe_return=("universe_return", "mean"),
            avg_alpha=("alpha", "mean"),
            win_rate_days=("selected_return", lambda s: float((s > 0).mean())),
            avg_hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["horizon", "top_n"])
        .reset_index(drop=True)
    )
    return summary, daily, picks


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-share major-flow proxy forward returns.")
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default=os.environ.get("MAJOR_FORCE_EVAL_DIR", "~/quantpilot_data/output/major_force_eval"))
    parser.add_argument("--top-ns", default="10,30,50")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--entry-lag-days", type=int, default=1)
    parser.add_argument("--date-step", type=int, default=1, help="Evaluate every Nth calendar date in the range.")
    parser.add_argument("--max-dates", type=int, default=0, help="Keep only the last N evaluation dates after stepping.")
    parser.add_argument("--min-active-stocks", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--stages", default="", help="Comma-separated stage filter, e.g. accumulation_candidate,watch.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min-amount", type=float, default=50_000_000.0)
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--include-limit-up", action="store_true")
    parser.add_argument("--include-limit-down", action="store_true")
    parser.add_argument("--include-st", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scan_cfg = MajorForceConfig(
        min_amount=args.min_amount,
        min_history=args.min_history,
        exclude_limit_up=not args.include_limit_up,
        exclude_limit_down=not args.include_limit_down,
        exclude_st=not args.include_st,
    )
    eval_cfg = MajorForceEvalConfig(
        top_ns=_parse_int_tuple(args.top_ns),
        horizons=_parse_int_tuple(args.horizons),
        entry_lag_days=max(0, args.entry_lag_days),
        date_step=max(1, args.date_step),
        max_dates=args.max_dates if args.max_dates > 0 else None,
        min_active_stocks=max(1, args.min_active_stocks),
        min_score=args.min_score,
        stages=tuple(item.strip() for item in args.stages.split(",") if item.strip()) or None,
    )
    summary, daily, picks = evaluate_major_force_forward_returns(
        Path(args.qlib_dir).expanduser(),
        args.start_date,
        args.end_date,
        scan_config=scan_cfg,
        eval_config=eval_cfg,
        progress=not args.quiet,
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
    daily_path = output_dir / "daily.csv"
    picks_path = output_dir / "picks.csv"
    summary.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)
    picks.to_csv(picks_path, index=False)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote daily: {daily_path}")
    print(f"Wrote picks: {picks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
