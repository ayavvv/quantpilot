"""US microstructure feature aggregation.

This module turns Futu OpenD tick/order-book/quote parquet batches into
one-minute features used by the US major-flow report. It intentionally keeps
the raw evidence separate from validation: these features can support a warmup
diagnostic report immediately, while calibrated confidence must come from a
forward validation gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DATA_KINDS = ("trades", "order_book", "quotes")
US_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MicrostructureFeatureConfig:
    """Configuration for one-minute US microstructure features."""

    book_levels: int = 5
    expected_regular_minutes: int = 390
    rolling_window_minutes: int = 20


def normalize_us_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." not in text:
        return f"US.{text}"
    return text


def normalize_us_symbols(values: Iterable[object] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = normalize_us_symbol(value)
        if symbol and symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    return result


def safe_partition_value(value: str) -> str:
    return str(value).replace("/", "_").replace(":", "_").replace(" ", "_")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _read_kind(base_dir: Path, kind: str, date: str, symbols: list[str]) -> pd.DataFrame:
    root = base_dir / kind / f"date={date}"
    if not root.exists():
        return _empty_frame()
    files: list[Path] = []
    if symbols:
        for symbol in symbols:
            symbol_dir = root / f"symbol={safe_partition_value(symbol)}"
            files.extend(sorted(symbol_dir.glob("*.parquet")))
    else:
        files.extend(sorted(root.glob("symbol=*/part-*.parquet")))
        files.extend(sorted(root.glob("*.parquet")))
    if not files:
        return _empty_frame()
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if kind == "trades":
        frame = _filter_trade_partition_date(frame, date)
    return frame


def _date_prefix(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    has_date = text.str.len().ge(10) & text.str[4:5].eq("-") & text.str[7:8].eq("-")
    return text.str[:10].where(has_date, "")


def _filter_trade_partition_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    if df.empty or not date:
        return df
    source = None
    for column in ("event_time", "time"):
        if column in df.columns:
            source = df[column]
            break
    if source is None:
        return df
    event_dates = _date_prefix(source)
    return df[(event_dates == "") | (event_dates == date)].copy()


def read_microstructure_inputs(
    base_dir: str | Path,
    *,
    date: str,
    symbols: Iterable[object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read raw microstructure parquet inputs for one collection date."""

    base = Path(base_dir).expanduser()
    normalized_symbols = normalize_us_symbols(symbols)
    return {kind: _read_kind(base, kind, date, normalized_symbols) for kind in DATA_KINDS}


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _symbol_series(df: pd.DataFrame) -> pd.Series:
    if "symbol" in df.columns:
        return df["symbol"].map(normalize_us_symbol)
    if "code" in df.columns:
        return df["code"].map(normalize_us_symbol)
    return pd.Series("", index=df.index)


def _coerce_event_ts(df: pd.DataFrame, preferred: tuple[str, ...]) -> pd.Series:
    for column in preferred:
        if column not in df.columns:
            continue
        parsed = df[column].map(_parse_timestamp_to_utc)
        if parsed.notna().any():
            return parsed
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def _parse_timestamp_to_utc(value: object) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(US_EASTERN)
    return timestamp.tz_convert("UTC")


def _minute(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.floor("min")


def _regular_session_mask(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    eastern = timestamps.dt.tz_convert(US_EASTERN)
    minute_of_day = eastern.dt.hour * 60 + eastern.dt.minute
    return (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)


def _safe_div(num: pd.Series | float, den: pd.Series | float, default: float = 0.0):
    if isinstance(num, pd.Series) or isinstance(den, pd.Series):
        if isinstance(num, pd.Series):
            numerator = num
            denominator = den if isinstance(den, pd.Series) else pd.Series(float(den), index=numerator.index)
        else:
            denominator = den
            numerator = pd.Series(float(num), index=denominator.index)
        result = numerator / denominator.replace(0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)
    if float(den) == 0:
        return default
    result = float(num) / float(den)
    if not math.isfinite(result):
        return default
    return result


def _trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return _empty_frame()
    df = trades.copy()
    df["symbol"] = _symbol_series(df)
    df = df[df["symbol"] != ""].copy()
    if df.empty:
        return _empty_frame()

    df["event_ts"] = _coerce_event_ts(df, ("event_time", "time", "recv_time"))
    df = df[df["event_ts"].notna()].copy()
    if df.empty:
        return _empty_frame()
    df["minute"] = _minute(df["event_ts"])

    df["price"] = _to_numeric(df.get("price", pd.Series(index=df.index, dtype=float)), np.nan)
    df["volume"] = _to_numeric(df.get("volume", pd.Series(index=df.index, dtype=float)), 0.0)
    df["turnover"] = _to_numeric(df.get("turnover", pd.Series(index=df.index, dtype=float)), np.nan)
    df["turnover"] = df["turnover"].fillna(df["price"] * df["volume"])
    df = df[df["price"].notna()].copy()
    if df.empty:
        return _empty_frame()

    direction = df.get("ticker_direction", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    side = np.select([direction.str.contains("BUY"), direction.str.contains("SELL")], [1, -1], default=0)
    trade_type = df.get("type", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    df["active_buy_dollar"] = np.where(side > 0, df["turnover"], 0.0)
    df["active_sell_dollar"] = np.where(side < 0, df["turnover"], 0.0)
    df["neutral_dollar"] = np.where(side == 0, df["turnover"], 0.0)
    df["odd_lot_dollar"] = np.where(trade_type.str.contains("ODD") | (df["volume"] < 100), df["turnover"], 0.0)
    df["buy_trade_count"] = np.where(side > 0, 1, 0)
    df["sell_trade_count"] = np.where(side < 0, 1, 0)
    df["neutral_trade_count"] = np.where(side == 0, 1, 0)

    if "sequence" in df.columns:
        df["sequence_text"] = df["sequence"].fillna("").astype(str)
    else:
        df["sequence_text"] = ""
    has_sequence = df["sequence_text"] != ""
    duplicate_sequence_mask = has_sequence & df.duplicated(["symbol", "sequence_text"], keep="first")
    duplicate_counts = (
        df.assign(is_duplicate_sequence=duplicate_sequence_mask)
        .groupby(["symbol", "minute"], as_index=False)
        .agg(
            raw_trade_count=("price", "size"),
            duplicate_sequence_count=("is_duplicate_sequence", "sum"),
        )
    )
    df = df[~duplicate_sequence_mask].copy()

    grouped = (
        df.sort_values(["symbol", "event_ts"])
        .groupby(["symbol", "minute"], as_index=False)
        .agg(
            open_price=("price", "first"),
            high_price=("price", "max"),
            low_price=("price", "min"),
            last_price=("price", "last"),
            share_volume=("volume", "sum"),
            dollar_volume=("turnover", "sum"),
            trade_count=("price", "size"),
            unique_sequence_count=("sequence_text", "nunique"),
            active_buy_dollar=("active_buy_dollar", "sum"),
            active_sell_dollar=("active_sell_dollar", "sum"),
            neutral_dollar=("neutral_dollar", "sum"),
            odd_lot_dollar=("odd_lot_dollar", "sum"),
            buy_trade_count=("buy_trade_count", "sum"),
            sell_trade_count=("sell_trade_count", "sum"),
            neutral_trade_count=("neutral_trade_count", "sum"),
        )
    )
    grouped = grouped.merge(duplicate_counts, on=["symbol", "minute"], how="left")
    grouped["raw_trade_count"] = _to_numeric(grouped.get("raw_trade_count", grouped["trade_count"]), 0.0)
    grouped["duplicate_sequence_count"] = _to_numeric(grouped.get("duplicate_sequence_count", pd.Series(index=grouped.index)), 0.0)
    grouped["minute_vwap"] = _safe_div(grouped["dollar_volume"], grouped["share_volume"], np.nan)
    grouped["avg_trade_size"] = _safe_div(grouped["dollar_volume"], grouped["trade_count"], 0.0)
    grouped["active_buy_ratio"] = _safe_div(
        grouped["active_buy_dollar"],
        grouped["active_buy_dollar"] + grouped["active_sell_dollar"],
        0.5,
    )
    grouped["net_active_dollar"] = grouped["active_buy_dollar"] - grouped["active_sell_dollar"]
    grouped["net_active_ratio"] = _safe_div(
        grouped["net_active_dollar"],
        grouped["active_buy_dollar"] + grouped["active_sell_dollar"],
        0.0,
    )
    grouped["neutral_dollar_ratio"] = _safe_div(grouped["neutral_dollar"], grouped["dollar_volume"], 0.0)
    grouped["odd_lot_ratio"] = _safe_div(grouped["odd_lot_dollar"], grouped["dollar_volume"], 0.0)
    grouped["range_bps"] = _safe_div(grouped["high_price"] - grouped["low_price"], grouped["minute_vwap"], 0.0) * 10_000
    grouped["minute_return_bps"] = _safe_div(grouped["last_price"], grouped["open_price"], 1.0).sub(1.0) * 10_000
    grouped["duplicate_sequence_rate"] = _safe_div(
        grouped["duplicate_sequence_count"],
        grouped["raw_trade_count"],
        0.0,
    ).clip(lower=0.0)
    return grouped


def _depth_imbalance(df: pd.DataFrame, levels: int) -> pd.Series:
    bid_cols = [f"bid_sz_{idx}" for idx in range(1, levels + 1) if f"bid_sz_{idx}" in df.columns]
    ask_cols = [f"ask_sz_{idx}" for idx in range(1, levels + 1) if f"ask_sz_{idx}" in df.columns]
    if not bid_cols or not ask_cols:
        return pd.Series(np.nan, index=df.index)
    bid = df[bid_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    ask = df[ask_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return _safe_div(bid - ask, bid + ask, np.nan)


def _book_features(order_book: pd.DataFrame, config: MicrostructureFeatureConfig) -> pd.DataFrame:
    if order_book.empty:
        return _empty_frame()
    df = order_book.copy()
    df["symbol"] = _symbol_series(df)
    df = df[df["symbol"] != ""].copy()
    if df.empty:
        return _empty_frame()
    df["event_ts"] = _coerce_event_ts(df, ("recv_time", "svr_recv_time_bid", "svr_recv_time_ask"))
    df = df[df["event_ts"].notna()].copy()
    if df.empty:
        return _empty_frame()
    df["minute"] = _minute(df["event_ts"])

    for column in ("mid", "spread_bps", "bid_sz_1", "ask_sz_1", "bid_px_1", "ask_px_1"):
        if column in df.columns:
            df[column] = _to_numeric(df[column], np.nan)
        else:
            df[column] = np.nan
    levels = max(1, int(config.book_levels))
    df["depth_imbalance_1"] = _depth_imbalance(df, 1)
    df["depth_imbalance_5"] = _depth_imbalance(df, min(levels, 5))
    df["top_depth"] = df["bid_sz_1"].fillna(0.0) + df["ask_sz_1"].fillna(0.0)

    grouped = (
        df.sort_values(["symbol", "event_ts"])
        .groupby(["symbol", "minute"], as_index=False)
        .agg(
            book_snapshot_count=("mid", "size"),
            mid=("mid", "mean"),
            spread_bps=("spread_bps", "mean"),
            depth_imbalance_1=("depth_imbalance_1", "mean"),
            depth_imbalance_5=("depth_imbalance_5", "mean"),
            bid_sz_1_mean=("bid_sz_1", "mean"),
            ask_sz_1_mean=("ask_sz_1", "mean"),
            top_depth_mean=("top_depth", "mean"),
            bid_sz_1_first=("bid_sz_1", "first"),
            bid_sz_1_last=("bid_sz_1", "last"),
            ask_sz_1_first=("ask_sz_1", "first"),
            ask_sz_1_last=("ask_sz_1", "last"),
        )
    )
    grouped["bid_replenish_1"] = grouped["bid_sz_1_last"] - grouped["bid_sz_1_first"]
    grouped["ask_replenish_1"] = grouped["ask_sz_1_last"] - grouped["ask_sz_1_first"]
    return grouped


def _quote_features(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return _empty_frame()
    df = quotes.copy()
    df["symbol"] = _symbol_series(df)
    df = df[df["symbol"] != ""].copy()
    if df.empty:
        return _empty_frame()
    df["event_ts"] = _coerce_event_ts(df, ("recv_time", "event_time", "data_time"))
    df = df[df["event_ts"].notna()].copy()
    if df.empty:
        return _empty_frame()
    df["minute"] = _minute(df["event_ts"])
    for column in ("last_price", "volume", "turnover", "pre_price", "after_price", "overnight_price"):
        if column in df.columns:
            df[column] = _to_numeric(df[column], np.nan)
        else:
            df[column] = np.nan
    return (
        df.sort_values(["symbol", "event_ts"])
        .groupby(["symbol", "minute"], as_index=False)
        .agg(
            quote_count=("last_price", "size"),
            quote_last_price=("last_price", "last"),
            quote_day_volume=("volume", "max"),
            quote_day_turnover=("turnover", "max"),
            pre_price=("pre_price", "last"),
            after_price=("after_price", "last"),
            overnight_price=("overnight_price", "last"),
        )
    )


def _merge_features(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_frame()
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["symbol", "minute"], how="outer")
    return result.sort_values(["symbol", "minute"]).reset_index(drop=True)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=min(3, window))
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_microstructure_features(
    trades: pd.DataFrame,
    order_book: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    config: MicrostructureFeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute one-minute feature rows from raw Futu microstructure tables."""

    cfg = config or MicrostructureFeatureConfig()
    features = _merge_features([_trade_features(trades), _book_features(order_book, cfg), _quote_features(quotes)])
    if features.empty:
        return features

    zero_columns = [
        "share_volume",
        "dollar_volume",
        "trade_count",
        "active_buy_dollar",
        "active_sell_dollar",
        "neutral_dollar",
        "odd_lot_dollar",
        "buy_trade_count",
        "sell_trade_count",
        "neutral_trade_count",
        "book_snapshot_count",
        "quote_count",
    ]
    for column in zero_columns:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = _to_numeric(features[column], 0.0)

    for column in (
        "open_price",
        "high_price",
        "low_price",
        "last_price",
        "minute_vwap",
        "mid",
        "quote_last_price",
        "spread_bps",
        "depth_imbalance_1",
        "depth_imbalance_5",
        "bid_replenish_1",
        "ask_replenish_1",
        "range_bps",
        "minute_return_bps",
        "duplicate_sequence_rate",
        "active_buy_ratio",
        "net_active_ratio",
        "neutral_dollar_ratio",
        "odd_lot_ratio",
        "avg_trade_size",
    ):
        if column not in features.columns:
            features[column] = np.nan
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    features["reference_price"] = features["last_price"].combine_first(features["quote_last_price"]).combine_first(features["mid"])
    features["has_trade_data"] = features["trade_count"] > 0
    features["has_book_data"] = features["book_snapshot_count"] > 0
    features["has_quote_data"] = features["quote_count"] > 0
    features["minute"] = pd.to_datetime(features["minute"], errors="coerce", utc=True)
    features = features[features["minute"].notna()].sort_values(["symbol", "minute"]).reset_index(drop=True)
    features["is_regular_session"] = _regular_session_mask(features["minute"])

    enriched: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=True):
        part = group.sort_values("minute").copy()
        regular_mask = part["is_regular_session"].fillna(False)
        part["session_dollar_volume"] = part["dollar_volume"].where(regular_mask, 0.0).cumsum()
        part["session_share_volume"] = part["share_volume"].where(regular_mask, 0.0).cumsum()
        part["session_net_active_dollar"] = (
            part["active_buy_dollar"].where(regular_mask, 0.0) - part["active_sell_dollar"].where(regular_mask, 0.0)
        ).cumsum()
        part["session_active_buy_dollar"] = part["active_buy_dollar"].where(regular_mask, 0.0).cumsum()
        part["session_active_sell_dollar"] = part["active_sell_dollar"].where(regular_mask, 0.0).cumsum()
        part["session_trade_count"] = part["trade_count"].where(regular_mask, 0.0).cumsum()
        part["session_vwap"] = _safe_div(part["session_dollar_volume"], part["session_share_volume"], np.nan)
        active_total = part["session_active_buy_dollar"] + part["session_active_sell_dollar"]
        part["session_net_active_ratio"] = _safe_div(part["session_net_active_dollar"], active_total, 0.0)
        part["session_active_buy_ratio"] = _safe_div(part["session_active_buy_dollar"], active_total, 0.5)
        part["vwap_deviation_bps"] = _safe_div(part["reference_price"], part["session_vwap"], 1.0).sub(1.0) * 10_000
        part["price_impact_bps_per_musd"] = _safe_div(
            part["minute_return_bps"].abs(),
            part["dollar_volume"] / 1_000_000.0,
            0.0,
        )
        window = max(3, int(cfg.rolling_window_minutes))
        part["dollar_volume_z"] = _rolling_z(part["dollar_volume"], window)
        part["trade_count_z"] = _rolling_z(part["trade_count"], window)
        regular = part[regular_mask]
        trade_minutes = int(regular["has_trade_data"].sum())
        book_minutes = int(regular["has_book_data"].sum())
        quote_minutes = int(regular["has_quote_data"].sum())
        coverage_minutes = int((regular["has_trade_data"] | regular["has_book_data"]).sum())
        expected_minutes = float(max(1, cfg.expected_regular_minutes))
        part["regular_session_minutes_seen"] = int(len(regular))
        part["trade_coverage_minutes"] = trade_minutes
        part["book_coverage_minutes"] = book_minutes
        part["quote_coverage_minutes"] = quote_minutes
        part["coverage_minutes"] = coverage_minutes
        part["trade_coverage_ratio_regular"] = min(1.0, float(trade_minutes) / expected_minutes)
        part["book_coverage_ratio_regular"] = min(1.0, float(book_minutes) / expected_minutes)
        part["quote_coverage_ratio_regular"] = min(1.0, float(quote_minutes) / expected_minutes)
        part["coverage_ratio_regular"] = min(
            1.0,
            float(coverage_minutes) / expected_minutes,
        )
        enriched.append(part)

    return pd.concat(enriched, ignore_index=True) if enriched else _empty_frame()


def build_feature_table(
    base_dir: str | Path,
    *,
    date: str,
    symbols: Iterable[object] | None = None,
    config: MicrostructureFeatureConfig | None = None,
) -> pd.DataFrame:
    inputs = read_microstructure_inputs(base_dir, date=date, symbols=symbols)
    return compute_microstructure_features(
        inputs["trades"],
        inputs["order_book"],
        inputs["quotes"],
        config=config,
    )


def write_feature_table(features: pd.DataFrame, base_dir: str | Path, *, date: str) -> Path:
    output_dir = Path(base_dir).expanduser() / "features_1m" / f"date={date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "part-us-microstructure-features.parquet"
    features.to_parquet(output, index=False)
    return output
