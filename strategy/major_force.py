"""A-share major-flow proxy scanner.

This module does not try to identify the actual buyer behind each print. That
requires tick/order-book data. Instead it scores daily-bar footprints that are
often consistent with accumulation: positive close location, volume/turnover
expansion, positive Chaikin money flow, and controlled price extension.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from converter.incremental import QlibBinReader


A_SHARE_PREFIXES = ("SH.", "SZ.")
DEFAULT_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "change_rate",
    "is_st",
]


@dataclass(frozen=True)
class MajorForceConfig:
    """Configuration for the daily-bar major-flow proxy."""

    lookback_days: int = 90
    min_history: int = 60
    fast_window: int = 5
    medium_window: int = 10
    flow_window: int = 20
    baseline_window: int = 20
    min_amount: float = 50_000_000.0
    min_close: float = 2.0
    exclude_st: bool = True
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return default
    return num / den


def _limit_up_threshold_pct(code: str) -> float:
    if code.startswith("SZ.300") or code.startswith("SH.688"):
        return 19.5
    return 9.5


def _pct_rank(series: pd.Series, neutral: float = 0.5) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    ranked = cleaned.rank(pct=True, method="average")
    return ranked.fillna(neutral)


def _prepare_stock_frame(df: pd.DataFrame, as_of_date: str | None, cfg: MajorForceConfig) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy().sort_index()
    for field in DEFAULT_FIELDS:
        if field not in result.columns:
            result[field] = np.nan
        result[field] = pd.to_numeric(result[field], errors="coerce")

    result = result.replace([np.inf, -np.inf], np.nan)
    if as_of_date:
        result = result.loc[:as_of_date]
    result = result.tail(cfg.lookback_days)
    return result.dropna(subset=["open", "high", "low", "close", "amount"])


def compute_major_force_metrics(
    code: str,
    df: pd.DataFrame,
    as_of_date: str | None = None,
    config: MajorForceConfig | None = None,
) -> dict[str, object] | None:
    """Compute per-stock daily-bar accumulation metrics.

    Returns ``None`` when the stock does not have enough usable history or fails
    the basic liquidity/ST/limit-up filters.
    """

    cfg = config or MajorForceConfig()
    stock = _prepare_stock_frame(df, as_of_date, cfg)
    if len(stock) < cfg.min_history:
        return None

    latest = stock.iloc[-1]
    latest_date = str(stock.index[-1])[:10]
    if as_of_date and latest_date < as_of_date:
        return None

    close = _finite_float(latest["close"])
    amount = _finite_float(latest["amount"])
    if close < cfg.min_close or amount < cfg.min_amount:
        return None

    is_st = _finite_float(latest.get("is_st", 0.0), default=0.0)
    if cfg.exclude_st and is_st >= 0.5:
        return None

    high = stock["high"]
    low = stock["low"]
    close_s = stock["close"]
    amount_s = stock["amount"].clip(lower=0)
    turnover_s = stock["turnover_rate"]

    if len(close_s) >= 2 and close_s.iloc[-2] != 0 and math.isfinite(float(close_s.iloc[-2])):
        today_chg_pct = (float(close_s.iloc[-1]) / float(close_s.iloc[-2]) - 1.0) * 100.0
    else:
        # Baostock/Futu change_rate is stored in percentage points in this repo.
        today_chg_pct = _finite_float(latest.get("change_rate", np.nan))

    limit_up_threshold = _limit_up_threshold_pct(code)
    is_limit_up = math.isfinite(today_chg_pct) and today_chg_pct >= limit_up_threshold
    is_limit_down = math.isfinite(today_chg_pct) and today_chg_pct <= -limit_up_threshold
    if cfg.exclude_limit_up and is_limit_up:
        return None
    if cfg.exclude_limit_down and is_limit_down:
        return None

    spread = (high - low).replace(0, np.nan)
    close_location = ((close_s - low) / spread).clip(0, 1).fillna(0.5)
    money_flow_multiplier = (2.0 * close_location - 1.0).clip(-1, 1)
    money_flow_amount = money_flow_multiplier * amount_s

    fw = cfg.fast_window
    mw = cfg.medium_window
    flow_w = cfg.flow_window
    base_w = cfg.baseline_window

    cmf_fast = _safe_ratio(float(money_flow_amount.tail(fw).sum()), float(amount_s.tail(fw).sum()))
    cmf_medium = _safe_ratio(float(money_flow_amount.tail(mw).sum()), float(amount_s.tail(mw).sum()))
    cmf_flow = _safe_ratio(float(money_flow_amount.tail(flow_w).sum()), float(amount_s.tail(flow_w).sum()))

    fast_amount = float(amount_s.tail(fw).mean())
    base_amount = float(amount_s.tail(base_w).mean())
    prev_base_amount = float(amount_s.iloc[-(fw + base_w):-fw].mean()) if len(amount_s) >= fw + base_w else base_amount
    amount_ratio_5_20 = _safe_ratio(fast_amount, base_amount)
    amount_ratio_5_prev20 = _safe_ratio(fast_amount, prev_base_amount)

    fast_turnover = float(turnover_s.tail(fw).mean())
    base_turnover = float(turnover_s.tail(base_w).mean())
    turnover_ratio_5_20 = _safe_ratio(fast_turnover, base_turnover)

    returns = close_s.pct_change()
    price_change_10 = _safe_ratio(float(close_s.iloc[-1]), float(close_s.iloc[-mw - 1]), default=np.nan) - 1.0
    price_change_20 = _safe_ratio(float(close_s.iloc[-1]), float(close_s.iloc[-flow_w - 1]), default=np.nan) - 1.0
    rolling_high_20 = float(high.shift(1).tail(flow_w).max())
    breakout_20 = _safe_ratio(float(close_s.iloc[-1]), rolling_high_20, default=np.nan) - 1.0

    recent_close = close_s.tail(mw)
    drawdown_10 = float((recent_close / recent_close.cummax() - 1.0).min())
    volatility_20 = float(returns.tail(flow_w).std())
    avg_range_10 = float(((high - low) / close_s.replace(0, np.nan)).tail(mw).mean())
    close_location_10 = float(close_location.tail(mw).mean())
    close_location_today = float(close_location.iloc[-1])
    positive_flow_days_20 = float((money_flow_amount.tail(flow_w) > 0).mean())
    up_days_10 = float((returns.tail(mw) > 0).mean())

    return {
        "code": code,
        "date": latest_date,
        "close": round(close, 4),
        "amount": round(amount, 2),
        "turnover_rate": round(_finite_float(latest.get("turnover_rate", np.nan)), 4),
        "today_chg_pct": round(today_chg_pct, 4) if math.isfinite(today_chg_pct) else np.nan,
        "cmf_5": cmf_fast,
        "cmf_10": cmf_medium,
        "cmf_20": cmf_flow,
        "cmf_accel_5_20": cmf_fast - cmf_flow if math.isfinite(cmf_fast) and math.isfinite(cmf_flow) else np.nan,
        "amount_ratio_5_20": amount_ratio_5_20,
        "amount_ratio_5_prev20": amount_ratio_5_prev20,
        "turnover_ratio_5_20": turnover_ratio_5_20,
        "close_location_today": close_location_today,
        "close_location_10": close_location_10,
        "positive_flow_days_20": positive_flow_days_20,
        "up_days_10": up_days_10,
        "price_change_10": price_change_10,
        "price_change_20": price_change_20,
        "breakout_20": breakout_20,
        "drawdown_10": drawdown_10,
        "volatility_20": volatility_20,
        "avg_range_10": avg_range_10,
        "is_limit_up": bool(is_limit_up),
        "is_limit_down": bool(is_limit_down),
    }


def _overheat_penalty(df: pd.DataFrame) -> pd.Series:
    price_ext = pd.to_numeric(df["price_change_10"], errors="coerce").fillna(0.0)
    day_ext = pd.to_numeric(df["today_chg_pct"], errors="coerce").fillna(0.0) / 100.0
    penalty = ((price_ext - 0.18) / 0.18).clip(lower=0)
    penalty += ((day_ext - 0.075) / 0.075).clip(lower=0)
    penalty += df["is_limit_up"].astype(float) * 0.5
    if "is_limit_down" in df.columns:
        penalty += df["is_limit_down"].astype(float) * 0.5
    return penalty.clip(lower=0, upper=1)


def _build_reason(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("cmf_20", 0) >= 0.12:
        parts.append("20d_positive_flow")
    if row.get("cmf_accel_5_20", 0) > 0:
        parts.append("flow_accelerating")
    if row.get("amount_ratio_5_20", 0) >= 1.35:
        parts.append("volume_expansion")
    if row.get("close_location_10", 0) >= 0.65:
        parts.append("closes_near_high")
    if row.get("breakout_20", 0) >= 0:
        parts.append("near_20d_breakout")
    if row.get("price_change_10", 0) > 0.18:
        parts.append("price_extended")
    if row.get("today_chg_pct", 0) <= -6.0:
        parts.append("sharp_down_day")
    if row.get("is_limit_up", False):
        parts.append("limit_up_risk")
    if row.get("is_limit_down", False):
        parts.append("limit_down_risk")
    return ",".join(parts) if parts else "mixed"


def _classify_stage(row: pd.Series) -> str:
    if row.get("cmf_20", 0) <= -0.10 and row.get("amount_ratio_5_20", 0) >= 1.2:
        return "distribution_risk"
    if row.get("score", 0) >= 80 and row.get("today_chg_pct", 0) <= -6.0:
        return "washout_or_risk"
    if row.get("score", 0) >= 80 and row.get("price_change_10", 0) <= 0.18:
        return "accumulation_candidate"
    if row.get("score", 0) >= 75 and row.get("price_change_10", 0) > 0.18:
        return "markup_or_overheated"
    if row.get("score", 0) >= 65:
        return "watch"
    return "weak"


def score_major_force_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional score/rank/stage columns to metric rows."""

    if metrics.empty:
        return metrics.copy()

    df = metrics.copy()
    df["amount_ratio_log"] = np.log(pd.to_numeric(df["amount_ratio_5_20"], errors="coerce").clip(lower=1e-9))
    df["turnover_ratio_log"] = np.log(pd.to_numeric(df["turnover_ratio_5_20"], errors="coerce").clip(lower=1e-9))
    df["overheat_penalty"] = _overheat_penalty(df)

    score_unit = (
        0.28 * _pct_rank(df["cmf_20"])
        + 0.12 * _pct_rank(df["cmf_accel_5_20"])
        + 0.14 * _pct_rank(df["amount_ratio_log"])
        + 0.08 * _pct_rank(df["turnover_ratio_log"])
        + 0.14 * _pct_rank(df["close_location_10"])
        + 0.10 * _pct_rank(df["breakout_20"])
        + 0.08 * _pct_rank(df["positive_flow_days_20"])
        + 0.06 * (1.0 - _pct_rank(df["volatility_20"]))
        - 0.18 * df["overheat_penalty"]
    )
    df["score"] = (score_unit.clip(lower=0, upper=1) * 100).round(2)
    df = df.sort_values(["score", "amount"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["stage"] = df.apply(_classify_stage, axis=1)
    df["reason"] = df.apply(_build_reason, axis=1)

    drop_cols = ["amount_ratio_log", "turnover_ratio_log"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def _resolve_as_of_date(reader: QlibBinReader, instruments: dict[str, tuple[str, str]], prefixes: tuple[str, ...]) -> str | None:
    latest = None
    for code, (_, end_date) in instruments.items():
        if code.startswith(prefixes) and (latest is None or end_date > latest):
            latest = end_date
    return latest or reader.latest_date


def scan_major_force(
    qlib_dir: str | Path,
    as_of_date: str | None = None,
    config: MajorForceConfig | None = None,
    prefixes: Iterable[str] = A_SHARE_PREFIXES,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """Scan A-share instruments and return ranked major-flow candidates."""

    cfg = config or MajorForceConfig()
    prefix_tuple = tuple(prefixes)
    reader = QlibBinReader(qlib_dir)
    instruments = reader.list_instruments("all")
    scan_date = as_of_date or _resolve_as_of_date(reader, instruments, prefix_tuple)
    if not scan_date:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for code, (_, end_date) in sorted(instruments.items()):
        if not code.startswith(prefix_tuple) or end_date < scan_date:
            continue
        stock_df = reader.read_stock(code, DEFAULT_FIELDS)
        metrics = compute_major_force_metrics(code, stock_df, as_of_date=scan_date, config=cfg)
        if metrics is not None:
            rows.append(metrics)

    scored = score_major_force_frame(pd.DataFrame(rows))
    if top_n is not None and top_n > 0:
        return scored.head(top_n).reset_index(drop=True)
    return scored


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan A-share major-flow proxy candidates.")
    parser.add_argument("--qlib-dir", default=os.environ.get("QLIB_DATA_DIR", "~/quantpilot_data/qlib_data"))
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-amount", type=float, default=50_000_000.0)
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--include-limit-up", action="store_true")
    parser.add_argument("--include-limit-down", action="store_true")
    parser.add_argument("--include-st", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = MajorForceConfig(
        min_amount=args.min_amount,
        min_history=args.min_history,
        exclude_limit_up=not args.include_limit_up,
        exclude_limit_down=not args.include_limit_down,
        exclude_st=not args.include_st,
    )
    result = scan_major_force(
        Path(args.qlib_dir).expanduser(),
        as_of_date=args.as_of_date,
        config=cfg,
        top_n=args.top_n,
    )

    display_cols = [
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
    ]
    if result.empty:
        print("No candidates found.")
    else:
        print(result[[col for col in display_cols if col in result.columns]].to_string(index=False))

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"Wrote {len(result)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
