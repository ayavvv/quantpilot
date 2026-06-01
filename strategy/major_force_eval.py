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
    MARKET_CONTEXT_FIELDS,
    MajorForceConfig,
    score_major_force_frame,
)


@dataclass(frozen=True)
class MajorForceEvalConfig:
    """Configuration for forward-return validation."""

    sides: tuple[str, ...] = ("buy",)
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


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one value")
    invalid = [value for value in values if value not in {"buy", "sell"}]
    if invalid:
        raise ValueError(f"unsupported side(s): {', '.join(invalid)}")
    return values


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


def _date_from_lookback(calendar: list[str], end_date: str, lookback_days: int) -> str:
    if not calendar:
        return end_date
    try:
        end_pos = calendar.index(end_date)
    except ValueError:
        end_pos = max(0, min(len(calendar) - 1, np.searchsorted(calendar, end_date) - 1))
    start_pos = max(0, end_pos - max(1, int(lookback_days)) + 1)
    return calendar[start_pos]


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


def _safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _limit_up_threshold_pct(code: str) -> float:
    if code.startswith(("SZ.300", "SZ.301", "SH.688")):
        return 19.5
    return 9.5


def _precompute_market_context_by_date(
    frames: dict[str, pd.DataFrame],
    dates: list[str],
) -> dict[str, dict[str, float]]:
    if not frames or not dates:
        return {}

    close_series: list[pd.Series] = []
    for code, df in frames.items():
        if df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if close.empty:
            continue
        close.index = close.index.astype(str).str[:10]
        close = close[~close.index.duplicated(keep="last")].sort_index()
        if len(close) < 21:
            continue
        close.name = code
        close_series.append(close)
    if not close_series:
        return {date: {field: np.nan for field in MARKET_CONTEXT_FIELDS} for date in dates}

    closes = pd.concat(close_series, axis=1).sort_index()
    returns = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    market_daily_return = returns.mean(axis=1, skipna=True)
    market_index = (1.0 + market_daily_return.fillna(0.0)).cumprod()
    breadth_count = returns.notna().sum(axis=1)
    positive_rate = _rate((returns > 0).sum(axis=1), breadth_count)
    ma20 = closes.rolling(20, min_periods=10).mean()
    above_ma20_rate = _rate(((closes > ma20) & ma20.notna()).sum(axis=1), ma20.notna().sum(axis=1))

    context = pd.DataFrame(index=closes.index)
    context["market_return_20"] = market_index / market_index.shift(20) - 1.0
    context["market_return_60"] = market_index / market_index.shift(60) - 1.0
    context["market_positive_rate_20"] = positive_rate.rolling(20, min_periods=10).mean()
    context["market_above_ma20_rate"] = above_ma20_rate
    context["market_volatility_20"] = market_daily_return.rolling(20, min_periods=10).std()
    context["market_drawdown_20"] = market_index / market_index.rolling(20, min_periods=10).max() - 1.0
    context = context.replace([np.inf, -np.inf], np.nan)

    result: dict[str, dict[str, float]] = {}
    for date in dates:
        if date in context.index:
            row = context.loc[date]
        else:
            prior = context.loc[:date]
            row = prior.iloc[-1] if not prior.empty else pd.Series(dtype=float)
        result[date] = {
            field: float(row[field]) if field in row and pd.notna(row[field]) else np.nan
            for field in MARKET_CONTEXT_FIELDS
        }
    return result


def _precompute_metric_rows_for_dates(
    code: str,
    df: pd.DataFrame,
    eval_dates: set[str],
    cfg: MajorForceConfig,
) -> list[dict[str, object]]:
    if df.empty or not eval_dates:
        return []

    stock = df.copy().sort_index()
    stock.index = stock.index.astype(str).str[:10]
    for field in DEFAULT_FIELDS:
        if field not in stock.columns:
            stock[field] = np.nan
        stock[field] = pd.to_numeric(stock[field], errors="coerce")
    stock = stock.replace([np.inf, -np.inf], np.nan)
    stock = stock.dropna(subset=["open", "high", "low", "close", "amount"])
    if len(stock) < cfg.min_history:
        return []

    high = stock["high"]
    low = stock["low"]
    close = stock["close"]
    amount = stock["amount"].clip(lower=0)
    turnover = stock["turnover_rate"]
    spread = (high - low).replace(0, np.nan)
    close_location = ((close - low) / spread).clip(0, 1).fillna(0.5)
    money_flow_amount = (2.0 * close_location - 1.0).clip(-1, 1) * amount

    fw = cfg.fast_window
    mw = cfg.medium_window
    flow_w = cfg.flow_window
    base_w = cfg.baseline_window
    returns = close.pct_change()
    today_chg_pct = (close / close.shift(1) - 1.0) * 100.0
    today_chg_pct = today_chg_pct.fillna(stock["change_rate"])
    threshold = _limit_up_threshold_pct(code)
    is_limit_up = today_chg_pct >= threshold
    is_limit_down = today_chg_pct <= -threshold

    amount_fast = amount.rolling(fw, min_periods=fw).mean()
    amount_base = amount.rolling(base_w, min_periods=base_w).mean()
    amount_prev_base = amount.shift(fw).rolling(base_w, min_periods=base_w).mean()
    turnover_fast = turnover.rolling(fw, min_periods=fw).mean()
    turnover_base = turnover.rolling(base_w, min_periods=base_w).mean()
    rolling_high = high.shift(1).rolling(flow_w, min_periods=flow_w).max()
    rolling_peak_10 = close.rolling(mw, min_periods=mw).max()

    metrics = pd.DataFrame(
        {
            "code": code,
            "date": stock.index,
            "close": close.round(4),
            "amount": amount.round(2),
            "turnover_rate": stock["turnover_rate"].round(4),
            "today_chg_pct": today_chg_pct.round(4),
            "cmf_5": _safe_div_series(
                money_flow_amount.rolling(fw, min_periods=fw).sum(),
                amount.rolling(fw, min_periods=fw).sum(),
            ),
            "cmf_10": _safe_div_series(
                money_flow_amount.rolling(mw, min_periods=mw).sum(),
                amount.rolling(mw, min_periods=mw).sum(),
            ),
            "cmf_20": _safe_div_series(
                money_flow_amount.rolling(flow_w, min_periods=flow_w).sum(),
                amount.rolling(flow_w, min_periods=flow_w).sum(),
            ),
            "amount_ratio_5_20": _safe_div_series(amount_fast, amount_base),
            "amount_ratio_5_prev20": _safe_div_series(amount_fast, amount_prev_base),
            "turnover_ratio_5_20": _safe_div_series(turnover_fast, turnover_base),
            "close_location_today": close_location,
            "close_location_10": close_location.rolling(mw, min_periods=mw).mean(),
            "positive_flow_days_20": (money_flow_amount > 0).astype(float).rolling(flow_w, min_periods=flow_w).mean(),
            "up_days_10": (returns > 0).astype(float).rolling(mw, min_periods=mw).mean(),
            "price_change_10": close / close.shift(mw) - 1.0,
            "price_change_20": close / close.shift(flow_w) - 1.0,
            "breakout_20": _safe_div_series(close, rolling_high) - 1.0,
            "drawdown_10": _safe_div_series(close, rolling_peak_10) - 1.0,
            "volatility_20": returns.rolling(flow_w, min_periods=flow_w).std(),
            "avg_range_10": ((high - low) / close.replace(0, np.nan)).rolling(mw, min_periods=mw).mean(),
            "is_limit_up": is_limit_up.astype(bool),
            "is_limit_down": is_limit_down.astype(bool),
            "_is_st": stock["is_st"],
            "_history_count": np.arange(1, len(stock) + 1),
        },
        index=stock.index,
    )
    metrics["cmf_accel_5_20"] = metrics["cmf_5"] - metrics["cmf_20"]

    mask = metrics.index.isin(eval_dates)
    mask &= metrics["_history_count"] >= cfg.min_history
    mask &= pd.to_numeric(metrics["close"], errors="coerce") >= cfg.min_close
    mask &= pd.to_numeric(metrics["amount"], errors="coerce") >= cfg.min_amount
    if cfg.exclude_st:
        mask &= pd.to_numeric(metrics["_is_st"], errors="coerce").fillna(0.0) < 0.5
    if cfg.exclude_limit_up:
        mask &= ~metrics["is_limit_up"]
    if cfg.exclude_limit_down:
        mask &= ~metrics["is_limit_down"]

    metrics = metrics.loc[mask].drop(columns=["_is_st", "_history_count"])
    if metrics.empty:
        return []
    return metrics.to_dict("records")


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


def _rank_for_side(
    ranked: pd.DataFrame,
    *,
    side: str,
    min_score: float | None,
    stages: tuple[str, ...] | None,
) -> pd.DataFrame:
    result = ranked.copy()
    side_name = str(side or "buy").lower()
    if side_name == "sell":
        score_col = "distribution_score"
        rank_col = "distribution_rank"
        default_stages = {"distribution_risk", "washout_or_risk"}
    else:
        score_col = "score"
        rank_col = "rank"
        default_stages = {"stealth_accumulation", "accumulation_candidate", "watch"}

    if score_col not in result.columns:
        return pd.DataFrame(columns=result.columns)
    if "stage" not in result.columns:
        result["stage"] = ""
    if stages:
        result = result[result["stage"].isin(stages)]
    elif side_name == "sell":
        result = result[(result["stage"].isin(default_stages)) | (pd.to_numeric(result[score_col], errors="coerce") >= 70)]
    else:
        result = result[result["stage"].isin(default_stages)]
    if min_score is not None:
        result = result[pd.to_numeric(result[score_col], errors="coerce") >= min_score]
    result = result.sort_values([score_col, "amount"], ascending=[False, False], kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["signal_side"] = side_name
    result["side_score"] = pd.to_numeric(result[score_col], errors="coerce")
    if rank_col in result.columns:
        result["source_rank"] = pd.to_numeric(result[rank_col], errors="coerce")
    else:
        result["source_rank"] = np.nan
    return result


def _side_hit_rate(returns: pd.Series, side: str) -> float:
    if side == "sell":
        return float((returns < 0).mean())
    return float((returns > 0).mean())


def _side_alpha(selected_return: float, universe_return: float, side: str) -> float:
    if side == "sell":
        return universe_return - selected_return
    return selected_return - universe_return


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
    eval_date_set = set(dates)
    market_context_by_date = _precompute_market_context_by_date(frames, dates)
    metrics_by_date: dict[str, list[dict[str, object]]] = {date: [] for date in dates}
    for code, df in frames.items():
        for row in _precompute_metric_rows_for_dates(code, df, eval_date_set, scan_cfg):
            metrics_by_date.setdefault(str(row.get("date", ""))[:10], []).append(row)

    daily_rows: list[dict[str, object]] = []
    pick_rows: list[dict[str, object]] = []

    for date_idx, date in enumerate(dates, start=1):
        metric_rows = metrics_by_date.get(date, [])
        if not metric_rows:
            if progress:
                print(f"[major_force_eval] {date_idx}/{len(dates)} {date}: no candidates", flush=True)
            continue

        all_ranked = score_major_force_frame(pd.DataFrame(metric_rows))
        if all_ranked.empty:
            if progress:
                print(f"[major_force_eval] {date_idx}/{len(dates)} {date}: empty universe ranking", flush=True)
            continue
        market_context = market_context_by_date.get(date, {})
        for key, value in market_context.items():
            all_ranked[key] = value

        for horizon in eval_cfg.horizons:
            returns_df = _returns_for_ranked(all_ranked, frames, date, horizon, eval_cfg.entry_lag_days)
            if returns_df.empty:
                continue
            universe_ret = float(returns_df["fwd_return"].mean())
            universe_count = int(len(returns_df))

            for side in eval_cfg.sides:
                ranked = _rank_for_side(
                    all_ranked,
                    side=side,
                    min_score=eval_cfg.min_score,
                    stages=eval_cfg.stages,
                )
                if ranked.empty:
                    if progress:
                        print(
                            f"[major_force_eval] {date_idx}/{len(dates)} {date}: "
                            f"empty {side} ranking",
                            flush=True,
                        )
                    continue
                if progress and horizon == eval_cfg.horizons[0]:
                    print(
                        f"[major_force_eval] {date_idx}/{len(dates)} {date}: "
                        f"{side}_candidates={len(ranked)} universe={len(all_ranked)}",
                        flush=True,
                    )
                selected_returns = ranked.merge(returns_df, on="code", how="inner")
                pick_ret_map = dict(zip(returns_df["code"], returns_df["fwd_return"]))
                top_picks = ranked.head(max_top_n).copy()
                top_picks["eval_date"] = date
                top_picks["signal_side"] = side
                top_picks[f"fwd_return_{horizon}d"] = top_picks["code"].map(pick_ret_map)

                for top_n in eval_cfg.top_ns:
                    selected = selected_returns.head(top_n)
                    if selected.empty:
                        continue
                    selected_ret = float(selected["fwd_return"].mean())
                    daily_rows.append(
                        {
                            "date": date,
                            "signal_side": side,
                            "horizon": horizon,
                            "top_n": top_n,
                            "selected_count": int(len(selected)),
                            "universe_count": universe_count,
                            "avg_score": float(selected["side_score"].mean()),
                            "selected_return": selected_ret,
                            "universe_return": universe_ret,
                            "alpha": _side_alpha(selected_ret, universe_ret, side),
                            "win_day": _side_hit_rate(pd.Series([selected_ret]), side),
                            "hit_rate": _side_hit_rate(selected["fwd_return"], side),
                            **market_context,
                        }
                    )
                pick_cols = [
                    "eval_date",
                    "signal_side",
                    "rank",
                    "source_rank",
                    "code",
                    "date",
                    "side_score",
                    "stealth_score",
                    "score",
                    "distribution_score",
                    "stage",
                    "close",
                    "today_chg_pct",
                    "cmf_5",
                    "cmf_10",
                    "amount_ratio_5_20",
                    "amount_ratio_5_prev20",
                    "turnover_ratio_5_20",
                    "cmf_accel_5_20",
                    "cmf_20",
                    "close_location_today",
                    "close_location_10",
                    "positive_flow_days_20",
                    "up_days_10",
                    "price_change_10",
                    "price_change_20",
                    "breakout_20",
                    "drawdown_10",
                    "volatility_20",
                    "avg_range_10",
                    *MARKET_CONTEXT_FIELDS,
                    "reason",
                ] + [f"fwd_return_{horizon}d"]
                pick_rows.extend(top_picks[[col for col in pick_cols if col in top_picks.columns]].to_dict("records"))

    daily = pd.DataFrame(daily_rows)
    picks = pd.DataFrame(pick_rows)
    if daily.empty:
        return pd.DataFrame(), daily, picks

    summary = (
        daily.groupby(["signal_side", "top_n", "horizon"], as_index=False)
        .agg(
            date_count=("date", "nunique"),
            avg_selected_count=("selected_count", "mean"),
            avg_score=("avg_score", "mean"),
            avg_return=("selected_return", "mean"),
            median_return=("selected_return", "median"),
            avg_universe_return=("universe_return", "mean"),
            avg_alpha=("alpha", "mean"),
            win_rate_days=("win_day", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["signal_side", "horizon", "top_n"])
        .reset_index(drop=True)
    )
    return summary, daily, picks


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-share major-flow proxy forward returns.")
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--start-date", default=os.environ.get("MAJOR_FORCE_EVAL_START_DATE", ""))
    parser.add_argument("--end-date", default=os.environ.get("MAJOR_FORCE_EVAL_END_DATE", ""))
    parser.add_argument("--lookback-days", type=int, default=int(os.environ.get("MAJOR_FORCE_EVAL_LOOKBACK_DAYS", "252")))
    parser.add_argument("--output-dir", default=os.environ.get("MAJOR_FORCE_EVAL_DIR", "~/quantpilot_data/output/major_force_eval"))
    parser.add_argument("--sides", default=os.environ.get("MAJOR_FORCE_EVAL_SIDES", "buy"))
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
    qlib_dir = Path(args.qlib_dir).expanduser()
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        reader = QlibBinReader(qlib_dir)
        calendar = reader.calendar
        end_date = args.end_date or (calendar[-1] if calendar else "")
        start_date = args.start_date or _date_from_lookback(calendar, end_date, args.lookback_days)
    if not start_date or not end_date:
        raise ValueError("start/end date could not be resolved from qlib calendar")
    scan_cfg = MajorForceConfig(
        min_amount=args.min_amount,
        min_history=args.min_history,
        exclude_limit_up=not args.include_limit_up,
        exclude_limit_down=not args.include_limit_down,
        exclude_st=not args.include_st,
    )
    eval_cfg = MajorForceEvalConfig(
        sides=_parse_str_tuple(args.sides),
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
        qlib_dir,
        start_date,
        end_date,
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
