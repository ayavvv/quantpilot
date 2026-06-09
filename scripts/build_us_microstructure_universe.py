"""Build the dynamic candidate universe for US microstructure collection.

Layer 1 of the US major-flow system is intentionally cheaper than tick/order
book collection: scan the broad US stock universe with snapshots, then enrich a
smaller pool with daily and one-minute bars.  The collector consumes the text
candidate file produced here and only subscribes the selected symbols.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.collect_us_microstructure import DEFAULT_RSA_KEY, _copy_to_nas
from scripts.scan_futu_market_capital_flow import (
    DEFAULT_EXCLUDE_SECURITY_CLASSES,
    fetch_futu_universe,
)
from strategy.us_microstructure_features import normalize_us_symbol, normalize_us_symbols


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
DEFAULT_NAS_DIR = "/volume1/docker/quantpilot/us_microstructure"
DEFAULT_CORE_SYMBOLS_FILE = Path(__file__).resolve().parents[1] / "config" / "us_microstructure_core_symbols.txt"
DEFAULT_FLOW_RANKING_FILE = DATA_DIR / "capital_flow" / "futu_market" / "US_latest_flow.csv"
DEFAULT_CORE_SOURCE = "futu_watchlist"
DEFAULT_CORE_WATCHLIST_GROUP_TYPE = "ALL"
DEFAULT_CORE_WATCHLIST_MAX_GROUPS = 8
DEFAULT_HISTORY_SLEEP_SECONDS = 0.55
FUTU_WATCHLIST_RATE_LIMIT_SLEEP_SECONDS = 31.0
US_EASTERN = ZoneInfo("America/New_York")
STATUS_SCHEMA_VERSION = 1


def _screen_date_from_utc(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(US_EASTERN).strftime("%Y-%m-%d")


def _date_days_before(end_date: str, days: int) -> str:
    end = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
    return (end - timedelta(days=max(1, int(days)))).isoformat()


def _split_csv(value: str) -> set[str]:
    return {item.strip().upper() for item in str(value or "").split(",") if item.strip()}


def _split_classes(value: str) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _read_symbol_file(path: str | Path) -> list[str]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return []
    values = [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    return normalize_us_symbols(values)


def _normalize_core_source(value: object) -> str:
    source = str(value or DEFAULT_CORE_SOURCE).strip().lower().replace("-", "_")
    aliases = {
        "futu": "futu_watchlist",
        "watchlist": "futu_watchlist",
        "futu_watchlist": "futu_watchlist",
        "file": "file",
        "static": "file",
        "static_file": "file",
        "core_file": "file",
        "none": "none",
        "disabled": "none",
    }
    if source not in aliases:
        raise ValueError(f"unsupported core source: {value}")
    return aliases[source]


def _split_group_names(value: object) -> list[str]:
    if value is None:
        raw_values: Iterable[object] = []
    elif isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        name = str(item or "").strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _normalize_watchlist_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith("US."):
        return normalize_us_symbol(text)
    if "." in text:
        return ""
    if not any(char.isalpha() for char in text):
        return ""
    return normalize_us_symbol(text)


def _extract_us_watchlist_symbols(data: Any) -> tuple[list[str], int]:
    frame = pd.DataFrame(data).copy() if data is not None else pd.DataFrame()
    if frame.empty or "code" not in frame.columns:
        return [], int(len(frame))

    if "stock_type" in frame.columns:
        stock_type = frame["stock_type"].fillna("").astype(str).str.upper()
        frame = frame[stock_type.isin({"", "N/A", "STOCK", "ETF"})].copy()
    if "option_type" in frame.columns:
        option_type = frame["option_type"].fillna("").astype(str).str.upper()
        frame = frame[option_type.isin({"", "N/A"})].copy()

    return normalize_us_symbols(_normalize_watchlist_code(value) for value in frame["code"].tolist()), int(len(frame))


def _select_default_watchlist_groups(group_frame: pd.DataFrame, *, max_groups: int = DEFAULT_CORE_WATCHLIST_MAX_GROUPS) -> list[str]:
    if group_frame.empty or "group_name" not in group_frame.columns:
        return []
    names = _split_group_names(group_frame["group_name"].dropna().tolist())
    name_by_lower = {name.lower(): name for name in names}
    for preferred in ("All", "全部", "Favorites", "自选股"):
        if preferred.lower() in name_by_lower:
            return [name_by_lower[preferred.lower()]]

    custom_names: list[str] = []
    if "group_type" in group_frame.columns:
        frame = group_frame.copy()
        frame["group_type"] = frame["group_type"].fillna("").astype(str).str.upper()
        custom_names = _split_group_names(frame.loc[frame["group_type"] == "CUSTOM", "group_name"].dropna().tolist())
    selected = custom_names or names
    return selected[: max(1, int(max_groups))]


def _is_futu_watchlist_rate_limit_error(value: object) -> bool:
    text = str(value or "").lower()
    return "too frequent" in text or "频繁" in text or "频率" in text


def _fetch_watchlist_groups_once(ctx: Any, groups: list[str]) -> tuple[list[str], dict[str, dict[str, int]], dict[str, str]]:
    from futu import RET_OK

    all_symbols: list[str] = []
    group_status: dict[str, dict[str, int]] = {}
    errors: dict[str, str] = {}
    for group in groups:
        try:
            ret, data = ctx.get_user_security(group)
        except Exception as exc:
            errors[group] = str(exc)
            continue
        if ret != RET_OK:
            errors[group] = str(data)
            continue
        symbols, row_count = _extract_us_watchlist_symbols(data)
        group_status[group] = {
            "row_count": int(row_count),
            "us_symbol_count": int(len(symbols)),
        }
        all_symbols.extend(symbols)
    return normalize_us_symbols(all_symbols), group_status, errors


def fetch_futu_watchlist_core_symbols(
    ctx: Any,
    *,
    group_names: object = "",
    group_type: str = DEFAULT_CORE_WATCHLIST_GROUP_TYPE,
) -> tuple[list[str], dict[str, object]]:
    from futu import RET_OK

    requested_groups = _split_group_names(group_names)
    resolved_group_type = str(group_type or DEFAULT_CORE_WATCHLIST_GROUP_TYPE).strip().upper()
    meta: dict[str, object] = {
        "core_watchlist_group_type": resolved_group_type,
        "core_watchlist_groups_requested": requested_groups,
        "core_watchlist_groups": [],
        "core_watchlist_group_count": 0,
        "core_watchlist_error_count": 0,
        "core_watchlist_errors": {},
        "core_watchlist_us_symbol_count": 0,
    }
    if ctx is None:
        meta["core_watchlist_error"] = "futu quote context unavailable"
        return [], meta

    groups = list(requested_groups)
    if not groups:
        try:
            ret, data = ctx.get_user_security_group(group_type=resolved_group_type)
        except Exception as exc:
            meta["core_watchlist_error"] = f"get_user_security_group: {exc}"
            return [], meta
        if ret != RET_OK:
            meta["core_watchlist_error"] = f"get_user_security_group: {data}"
            return [], meta
        group_frame = pd.DataFrame(data).copy() if data is not None else pd.DataFrame()
        if group_frame.empty or "group_name" not in group_frame.columns:
            meta["core_watchlist_error"] = "get_user_security_group: empty"
            return [], meta
        meta["core_watchlist_groups_discovered"] = _split_group_names(group_frame["group_name"].dropna().tolist())
        groups = _select_default_watchlist_groups(group_frame)

    symbols, group_status, errors = _fetch_watchlist_groups_once(ctx, groups)
    if not symbols and errors and any(_is_futu_watchlist_rate_limit_error(error) for error in errors.values()):
        time.sleep(FUTU_WATCHLIST_RATE_LIMIT_SLEEP_SECONDS)
        retry_symbols, retry_group_status, retry_errors = _fetch_watchlist_groups_once(ctx, groups)
        if retry_symbols:
            symbols = retry_symbols
            group_status = retry_group_status
            errors = retry_errors
            meta["core_watchlist_rate_limit_retry"] = True

    meta.update(
        {
            "core_watchlist_groups": groups,
            "core_watchlist_group_count": int(len(groups)),
            "core_watchlist_group_status": group_status,
            "core_watchlist_error_count": int(len(errors)),
            "core_watchlist_errors": dict(list(errors.items())[:20]),
            "core_watchlist_us_symbol_count": int(len(symbols)),
        }
    )
    if errors and not symbols:
        meta["core_watchlist_error"] = "; ".join(f"{group}: {error}" for group, error in list(errors.items())[:3])
    elif not symbols:
        meta["core_watchlist_error"] = "no US symbols in Futu watchlist"
    return symbols, meta


def resolve_core_symbols(
    ctx: Any,
    *,
    core_symbols_file: str | Path,
    core_source: object = DEFAULT_CORE_SOURCE,
    core_watchlist_groups: object = "",
    core_watchlist_group_type: str = DEFAULT_CORE_WATCHLIST_GROUP_TYPE,
) -> tuple[list[str], dict[str, object]]:
    source = _normalize_core_source(core_source)
    file_symbols = _read_symbol_file(core_symbols_file)
    meta: dict[str, object] = {
        "core_symbol_source_requested": source,
        "core_symbols_file": str(Path(core_symbols_file).expanduser()),
        "core_file_symbol_count": int(len(file_symbols)),
    }
    if source == "none":
        meta.update({"core_symbol_source": "none", "core_symbol_fallback_used": False})
        return [], meta
    if source == "file":
        meta.update({"core_symbol_source": "file", "core_symbol_fallback_used": False})
        return file_symbols, meta

    watchlist_symbols, watchlist_meta = fetch_futu_watchlist_core_symbols(
        ctx,
        group_names=core_watchlist_groups,
        group_type=core_watchlist_group_type,
    )
    meta.update(watchlist_meta)
    if watchlist_symbols:
        meta.update({"core_symbol_source": "futu_watchlist", "core_symbol_fallback_used": False})
        return watchlist_symbols, meta

    meta.update(
        {
            "core_symbol_source": "file_fallback",
            "core_symbol_fallback_used": True,
            "core_symbol_fallback_reason": str(watchlist_meta.get("core_watchlist_error") or "no US watchlist symbols"),
        }
    )
    return file_symbols, meta


def _number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return float(parsed)


def _first_existing(row: pd.Series, columns: Iterable[str], default: float = 0.0) -> float:
    for column in columns:
        if column in row:
            value = _number(row.get(column), default=float("nan"))
            if pd.notna(value):
                return value
    return default


def _series_number(frame: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    result = pd.Series(default, index=frame.index, dtype="float64")
    pending = pd.Series(True, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = pending & values.notna()
        result.loc[mask] = values.loc[mask].astype(float)
        pending = pending & ~mask
    return result


def _percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    values = values.fillna(0.0).clip(lower=0.0)
    if values.empty or float(values.max()) <= float(values.min()):
        return pd.Series(0.0, index=series.index, dtype="float64")
    return values.rank(pct=True).fillna(0.0)


def _symbol_first_letter(symbol: object) -> str:
    text = normalize_us_symbol(symbol)
    ticker = text.split(".", 1)[1] if "." in text else text
    for char in ticker:
        if char.isalpha():
            return char.upper()
    return "0"


def _first_letter_counts(symbols: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        letter = _symbol_first_letter(symbol)
        counts[letter] = counts.get(letter, 0) + 1
    return dict(sorted(counts.items()))


def _unique_symbols(symbols: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for symbol in normalize_us_symbols(symbols):
        if symbol and symbol not in seen:
            result.append(symbol)
            seen.add(symbol)
    return result


def _stratify_symbols_by_first_letter(symbols: Iterable[object]) -> list[str]:
    buckets: OrderedDict[str, list[str]] = OrderedDict()
    for symbol in _unique_symbols(symbols):
        buckets.setdefault(_symbol_first_letter(symbol), []).append(symbol)

    result: list[str] = []
    while buckets:
        for letter in list(buckets.keys()):
            bucket = buckets[letter]
            if bucket:
                result.append(bucket.pop(0))
            if not bucket:
                del buckets[letter]
    return result


def _take_rows_stratified_by_first_letter(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    limit = max(0, int(limit))
    if limit <= 0 or frame.empty or "symbol" not in frame.columns:
        return frame.head(0).copy()

    work = frame.copy()
    work["_first_letter"] = work["symbol"].map(_symbol_first_letter)
    buckets: OrderedDict[str, list[int]] = OrderedDict()
    for index, letter in work["_first_letter"].items():
        buckets.setdefault(letter, []).append(index)

    selected_indexes: list[int] = []
    while buckets and len(selected_indexes) < limit:
        for letter in list(buckets.keys()):
            bucket = buckets[letter]
            if bucket:
                selected_indexes.append(bucket.pop(0))
                if len(selected_indexes) >= limit:
                    break
            if not bucket:
                del buckets[letter]

    return work.loc[selected_indexes].drop(columns="_first_letter").copy()


def _snapshot_positive_liquidity_count(snapshot: pd.DataFrame) -> int:
    if snapshot.empty or "snapshot_turnover" not in snapshot.columns or "snapshot_volume" not in snapshot.columns:
        return 0
    turnover = pd.to_numeric(snapshot["snapshot_turnover"], errors="coerce").fillna(0.0)
    volume = pd.to_numeric(snapshot["snapshot_volume"], errors="coerce").fillna(0.0)
    return int(((turnover > 0) & (volume > 0)).sum())


def _normalize_snapshot_frame(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "snapshot_price",
                "snapshot_open",
                "snapshot_prev_close",
                "snapshot_volume",
                "snapshot_turnover",
                "snapshot_change_pct",
                "snapshot_gap_pct",
                "snapshot_turn_rate",
            ]
        )
    frame = snapshot.copy()
    if "symbol" not in frame.columns:
        frame["symbol"] = frame["code"] if "code" in frame.columns else ""
    frame["symbol"] = frame["symbol"].map(normalize_us_symbol)
    frame = frame[frame["symbol"] != ""].copy()
    frame["snapshot_price"] = _series_number(frame, ["last_price", "cur_price", "nominal_price", "price"])
    frame["snapshot_open"] = _series_number(frame, ["open_price", "open"])
    frame["snapshot_prev_close"] = _series_number(frame, ["prev_close_price", "prev_close", "pre_close"])
    frame["snapshot_volume"] = _series_number(frame, ["volume", "vol"])
    frame["snapshot_turnover"] = _series_number(frame, ["turnover", "amount"])
    missing_turnover = frame["snapshot_turnover"] <= 0
    frame.loc[missing_turnover, "snapshot_turnover"] = (
        frame.loc[missing_turnover, "snapshot_volume"] * frame.loc[missing_turnover, "snapshot_price"]
    )
    change = _series_number(frame, ["change_rate", "change_pct", "change_ratio"], default=float("nan"))
    computed_change = (
        (frame["snapshot_price"] / frame["snapshot_prev_close"] - 1.0) * 100.0
    ).where(frame["snapshot_prev_close"] > 0)
    frame["snapshot_change_pct"] = change.where(change.notna(), computed_change).fillna(0.0)
    valid_open = (frame["snapshot_open"] > 0) & (frame["snapshot_prev_close"] > 0)
    frame["snapshot_gap_pct"] = (
        (frame["snapshot_open"] / frame["snapshot_prev_close"] - 1.0) * 100.0
    ).where(valid_open).fillna(0.0)
    frame["snapshot_turn_rate"] = _series_number(frame, ["turn_rate", "turnover_rate"], default=0.0)
    keep = [
        "symbol",
        "snapshot_price",
        "snapshot_open",
        "snapshot_prev_close",
        "snapshot_volume",
        "snapshot_turnover",
        "snapshot_change_pct",
        "snapshot_gap_pct",
        "snapshot_turn_rate",
    ]
    for column in ["name", "exchange_type", "security_class"]:
        if column in frame.columns:
            keep.append(column)
    return frame[keep].drop_duplicates("symbol", keep="first").reset_index(drop=True)


def fetch_market_snapshots(
    ctx: Any,
    symbols: list[str],
    *,
    batch_size: int = 200,
    sleep_seconds: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, str]]:
    from futu import RET_OK

    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    batch_size = max(1, int(batch_size))
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        ret, data = ctx.get_market_snapshot(batch)
        if ret != RET_OK:
            for symbol in batch:
                errors[symbol] = str(data)
        elif data is not None and len(data) > 0:
            frames.append(pd.DataFrame(data).copy())
        else:
            for symbol in batch:
                errors[symbol] = "empty"
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    snapshot = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _normalize_snapshot_frame(snapshot), errors


def load_flow_ranking(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(resolved)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()

    result = frame.copy()
    if "symbol" not in result.columns:
        result["symbol"] = result["code"] if "code" in result.columns else ""
    result["symbol"] = result["symbol"].map(normalize_us_symbol)
    result = result[result["symbol"] != ""].drop_duplicates("symbol", keep="first").reset_index(drop=True)
    if result.empty:
        return pd.DataFrame()

    flow_columns = [
        "latest_main_in_flow",
        "latest_super_in_flow",
        "latest_big_in_flow",
        "main_3d_sum",
        "super_3d_sum",
        "big_3d_sum",
        "main_5d_sum",
        "super_5d_sum",
        "big_5d_sum",
        "main_10d_sum",
        "super_10d_sum",
        "big_10d_sum",
    ]
    available = [column for column in flow_columns if column in result.columns]
    if available:
        numeric = result[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
        result["flow_activity"] = numeric.max(axis=1)
    else:
        result["flow_activity"] = 0.0

    result["flow_activity_component"] = _percentile_rank(result["flow_activity"].map(lambda value: max(value, 0.0)) ** 0.5)
    rename_map = {
        "capital_flow_status": "flow_status",
        "capital_flow_latest_date": "flow_latest_date",
        "latest_main_in_flow": "flow_latest_main_in_flow",
        "main_3d_sum": "flow_main_3d_sum",
        "main_5d_sum": "flow_main_5d_sum",
        "main_10d_sum": "flow_main_10d_sum",
    }
    result = result.rename(columns={old: new for old, new in rename_map.items() if old in result.columns})
    keep = [
        "symbol",
        "flow_activity",
        "flow_activity_component",
        "flow_status",
        "flow_latest_date",
        "flow_latest_main_in_flow",
        "flow_main_3d_sum",
        "flow_main_5d_sum",
        "flow_main_10d_sum",
    ]
    for column in keep:
        if column not in result.columns:
            result[column] = "" if column in {"flow_status", "flow_latest_date"} else 0.0
    return result[keep].reset_index(drop=True)


def select_enrichment_symbols(
    universe_symbols: list[str],
    snapshot: pd.DataFrame,
    flow_ranking: pd.DataFrame,
    *,
    core_symbols: list[str],
    limit: int,
) -> tuple[list[str], str]:
    limit = max(0, int(limit))
    if limit <= 0:
        return [], "disabled"

    selected: list[str] = []
    selected.extend(normalize_us_symbols(core_symbols))
    positive_snapshot_count = _snapshot_positive_liquidity_count(snapshot)
    source = "stratified_universe"

    if positive_snapshot_count >= max(20, min(limit, len(snapshot)) // 5):
        ranked = (
            snapshot.sort_values(["snapshot_turnover", "snapshot_volume"], ascending=[False, False])["symbol"].tolist()
            if "symbol" in snapshot.columns
            else []
        )
        selected.extend(ranked)
        source = "snapshot_liquidity"
    elif not flow_ranking.empty and "flow_activity" in flow_ranking.columns:
        ranked_flow = flow_ranking[pd.to_numeric(flow_ranking["flow_activity"], errors="coerce").fillna(0.0) > 0].copy()
        if not ranked_flow.empty:
            universe_set = set(normalize_us_symbols(universe_symbols))
            ranked_flow = ranked_flow[ranked_flow["symbol"].isin(universe_set)].copy()
        if not ranked_flow.empty:
            ranked_flow = ranked_flow.sort_values(["flow_activity", "symbol"], ascending=[False, True])
            selected.extend(ranked_flow["symbol"].tolist())
            source = "capital_flow_activity"

    selected.extend(_stratify_symbols_by_first_letter(universe_symbols))
    return _unique_symbols(selected)[:limit], source


def _kline_date_column(frame: pd.DataFrame) -> pd.Series:
    if "time_key" in frame.columns:
        return frame["time_key"].astype(str).str[:10]
    if "date" in frame.columns:
        return frame["date"].astype(str).str[:10]
    if "time" in frame.columns:
        return frame["time"].astype(str).str[:10]
    return pd.Series("", index=frame.index)


def summarize_daily_klines(rows: list[dict[str, Any]], *, symbol: str) -> dict[str, object]:
    if not rows:
        return {
            "symbol": normalize_us_symbol(symbol),
            "daily_status": "empty",
        }
    frame = pd.DataFrame(rows).copy()
    frame["date"] = _kline_date_column(frame)
    frame["close"] = _series_number(frame, ["close", "last_price"])
    frame["volume"] = _series_number(frame, ["volume", "vol"])
    frame["turnover"] = _series_number(frame, ["turnover", "amount"])
    missing_turnover = frame["turnover"] <= 0
    frame.loc[missing_turnover, "turnover"] = frame.loc[missing_turnover, "volume"] * frame.loc[missing_turnover, "close"]
    frame = frame[frame["date"] != ""].sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    if frame.empty:
        return {
            "symbol": normalize_us_symbol(symbol),
            "daily_status": "empty",
        }
    latest = frame.iloc[-1]
    previous = frame.iloc[:-1].tail(20)
    if previous.empty:
        previous = frame.tail(20)
    avg_turnover = float(previous["turnover"].mean()) if not previous.empty else 0.0
    avg_volume = float(previous["volume"].mean()) if not previous.empty else 0.0
    latest_turnover = _number(latest.get("turnover"))
    latest_volume = _number(latest.get("volume"))
    return {
        "symbol": normalize_us_symbol(symbol),
        "daily_status": "ok",
        "daily_latest_date": str(latest.get("date") or ""),
        "daily_close": _number(latest.get("close")),
        "daily_turnover": latest_turnover,
        "daily_volume": latest_volume,
        "daily_avg_turnover_20d": avg_turnover,
        "daily_avg_volume_20d": avg_volume,
        "daily_turnover_ratio_20d": latest_turnover / avg_turnover if avg_turnover > 0 else 0.0,
        "daily_volume_ratio_20d": latest_volume / avg_volume if avg_volume > 0 else 0.0,
    }


def fetch_daily_metrics(
    ctx: Any,
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
    ktype: Any,
    autype: Any,
    sleep_seconds: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, str]]:
    from futu import RET_OK

    rows: list[dict[str, object]] = []
    errors: dict[str, str] = {}
    for symbol in symbols:
        page_req_key = None
        symbol_rows: list[dict[str, Any]] = []
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
                symbol_rows.extend(pd.DataFrame(data).to_dict("records"))
            if page_req_key is None:
                break
        rows.append(summarize_daily_klines(symbol_rows, symbol=symbol))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows), errors


def summarize_minute_klines(rows: list[dict[str, Any]], *, symbol: str) -> dict[str, object]:
    if not rows:
        return {
            "symbol": normalize_us_symbol(symbol),
            "minute_status": "empty",
        }
    frame = pd.DataFrame(rows).copy()
    frame["time_key"] = frame["time_key"].astype(str) if "time_key" in frame.columns else _kline_date_column(frame)
    frame["close"] = _series_number(frame, ["close", "last_price"])
    frame["volume"] = _series_number(frame, ["volume", "vol"])
    frame["turnover"] = _series_number(frame, ["turnover", "amount"])
    missing_turnover = frame["turnover"] <= 0
    frame.loc[missing_turnover, "turnover"] = frame.loc[missing_turnover, "volume"] * frame.loc[missing_turnover, "close"]
    frame = frame.sort_values("time_key").reset_index(drop=True)
    if frame.empty:
        return {
            "symbol": normalize_us_symbol(symbol),
            "minute_status": "empty",
        }
    last5 = frame.tail(5)
    baseline = frame.iloc[:-5]
    if baseline.empty:
        baseline = frame
    baseline_median = float(baseline["turnover"].median()) if not baseline.empty else 0.0
    last5_avg = float(last5["turnover"].mean()) if not last5.empty else 0.0
    return {
        "symbol": normalize_us_symbol(symbol),
        "minute_status": "ok",
        "minute_last_time": str(frame.iloc[-1].get("time_key") or ""),
        "minute_count": int(len(frame)),
        "minute_turnover_last5": float(last5["turnover"].sum()) if not last5.empty else 0.0,
        "minute_avg_turnover_last5": last5_avg,
        "minute_median_turnover_baseline": baseline_median,
        "minute_turnover_burst_ratio": last5_avg / baseline_median if baseline_median > 0 else 0.0,
    }


def fetch_minute_metrics(
    ctx: Any,
    symbols: list[str],
    *,
    lookback: int,
    ktype: Any,
    autype: Any,
    sleep_seconds: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, str]]:
    from futu import RET_OK

    rows: list[dict[str, object]] = []
    errors: dict[str, str] = {}
    for symbol in symbols:
        try:
            ret, data = ctx.get_cur_kline(code=symbol, num=max(5, int(lookback)), ktype=ktype, autype=autype)
        except TypeError:
            ret, data = ctx.get_cur_kline(symbol, max(5, int(lookback)), ktype)
        if ret != RET_OK:
            errors[symbol] = str(data)
            rows.append({"symbol": symbol, "minute_status": "error"})
        else:
            symbol_rows = pd.DataFrame(data).to_dict("records") if data is not None and len(data) > 0 else []
            rows.append(summarize_minute_klines(symbol_rows, symbol=symbol))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows), errors


def _score_candidates(
    universe: pd.DataFrame,
    snapshot: pd.DataFrame,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    flow_ranking: pd.DataFrame | None = None,
    *,
    core_symbols: list[str],
    min_price: float,
    min_snapshot_turnover: float,
    min_snapshot_volume: float,
) -> pd.DataFrame:
    base = universe.copy()
    if "code" in base.columns:
        base["symbol"] = base["code"]
    elif "symbol" not in base.columns:
        base["symbol"] = ""
    base["symbol"] = base["symbol"].map(normalize_us_symbol)
    base = base[base["symbol"] != ""].drop_duplicates("symbol", keep="first").reset_index(drop=True)
    for frame in (snapshot, daily, minute):
        if not frame.empty and "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].map(normalize_us_symbol)

    result = base.merge(snapshot.drop_duplicates("symbol"), on="symbol", how="left", suffixes=("", "_snapshot"))
    if not daily.empty:
        result = result.merge(daily.drop_duplicates("symbol"), on="symbol", how="left")
    if not minute.empty:
        result = result.merge(minute.drop_duplicates("symbol"), on="symbol", how="left")
    if flow_ranking is not None and not flow_ranking.empty:
        result = result.merge(flow_ranking.drop_duplicates("symbol"), on="symbol", how="left")

    numeric_defaults = [
        "snapshot_price",
        "snapshot_volume",
        "snapshot_turnover",
        "snapshot_change_pct",
        "snapshot_gap_pct",
        "snapshot_turn_rate",
        "daily_turnover",
        "daily_close",
        "daily_volume",
        "daily_avg_turnover_20d",
        "daily_avg_volume_20d",
        "daily_turnover_ratio_20d",
        "daily_volume_ratio_20d",
        "minute_turnover_last5",
        "minute_turnover_burst_ratio",
        "flow_activity",
        "flow_activity_component",
    ]
    for column in numeric_defaults:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    core_set = set(normalize_us_symbols(core_symbols))
    result["core_symbol"] = result["symbol"].isin(core_set)
    fallback_price = result["daily_close"] if "daily_close" in result.columns else 0.0
    result["active_price_for_liquidity"] = result["snapshot_price"].where(result["snapshot_price"] > 0, fallback_price)
    result["active_turnover_for_liquidity"] = result["snapshot_turnover"].where(result["snapshot_turnover"] > 0, result["daily_turnover"])
    result["active_volume_for_liquidity"] = result["snapshot_volume"].where(result["snapshot_volume"] > 0, result["daily_volume"])
    result["liquidity_pass"] = (
        result["core_symbol"]
        | (
            (result["active_price_for_liquidity"] >= float(min_price))
            & (result["active_turnover_for_liquidity"] >= float(min_snapshot_turnover))
            & (result["active_volume_for_liquidity"] >= float(min_snapshot_volume))
        )
    )
    avg_turnover = result["daily_avg_turnover_20d"].where(result["daily_avg_turnover_20d"] > 0, result["active_turnover_for_liquidity"])
    result["liquidity_score_component"] = _percentile_rank(result["active_turnover_for_liquidity"].map(lambda value: max(value, 0.0)) ** 0.5)
    result["daily_liquidity_component"] = _percentile_rank(avg_turnover.map(lambda value: max(value, 0.0)) ** 0.5)
    result["move_component"] = _percentile_rank(result["snapshot_change_pct"].abs())
    result["gap_component"] = _percentile_rank(result["snapshot_gap_pct"].abs())
    result["abnormal_volume_component"] = _percentile_rank(
        result[["daily_turnover_ratio_20d", "daily_volume_ratio_20d"]].max(axis=1)
    )
    result["minute_component"] = _percentile_rank(result["minute_turnover_burst_ratio"])
    result["coarse_score"] = (
        30.0 * result["liquidity_score_component"]
        + 20.0 * result["daily_liquidity_component"]
        + 18.0 * result["abnormal_volume_component"]
        + 15.0 * result["move_component"]
        + 10.0 * result["gap_component"]
        + 7.0 * result["minute_component"]
        + 10.0 * result["flow_activity_component"]
    )
    result.loc[result["core_symbol"], "coarse_score"] = result.loc[result["core_symbol"], "coarse_score"] + 5.0
    result["screen_reason"] = result.apply(_screen_reason, axis=1)
    return result.sort_values(["coarse_score", "active_turnover_for_liquidity"], ascending=[False, False]).reset_index(drop=True)


def _screen_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("core_symbol")):
        reasons.append("core")
    if _number(row.get("snapshot_turnover")) > 0:
        reasons.append("turnover")
    elif _number(row.get("daily_turnover")) > 0:
        reasons.append("daily_liquidity")
    if _number(row.get("flow_activity")) > 0:
        reasons.append("flow_activity")
    if abs(_number(row.get("snapshot_change_pct"))) >= 3.0:
        reasons.append("move")
    if abs(_number(row.get("snapshot_gap_pct"))) >= 2.0:
        reasons.append("gap")
    if max(_number(row.get("daily_turnover_ratio_20d")), _number(row.get("daily_volume_ratio_20d"))) >= 1.5:
        reasons.append("abnormal_volume")
    if _number(row.get("minute_turnover_burst_ratio")) >= 1.5:
        reasons.append("minute_burst")
    return ",".join(reasons) if reasons else "ranked"


def select_candidates(scored: pd.DataFrame, *, target_size: int, core_symbols: list[str]) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    target_size = max(1, int(target_size))
    core_set = set(normalize_us_symbols(core_symbols))
    core = scored[scored["symbol"].isin(core_set)].copy()
    if not core.empty:
        core["selection_source"] = "core"
    ranked = scored[(scored["liquidity_pass"]) & (~scored["symbol"].isin(core_set))].copy()
    remaining_slots = max(0, target_size - len(core))
    selected_ranked = ranked.head(remaining_slots).copy()
    if not selected_ranked.empty:
        selected_ranked["selection_source"] = "liquidity_ranked"
    selected_frames = [selected_ranked, core]
    selected_symbols = set(selected_ranked["symbol"].dropna().astype(str))
    selected_symbols.update(core["symbol"].dropna().astype(str))
    fallback_slots = max(0, target_size - len(selected_ranked) - len(core))
    if fallback_slots > 0:
        fallback_pool = scored[
            (~scored["symbol"].isin(core_set))
            & (~scored["symbol"].isin(selected_symbols))
        ].copy()
        fallback = _take_rows_stratified_by_first_letter(fallback_pool, limit=fallback_slots)
        if not fallback.empty:
            fallback["selection_source"] = "fallback_ranked"
            selected_frames.insert(1, fallback)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.drop_duplicates("symbol", keep="first")
    tie_breaker = "active_turnover_for_liquidity" if "active_turnover_for_liquidity" in selected.columns else "snapshot_turnover"
    selected = selected.sort_values(["coarse_score", tie_breaker], ascending=[False, False])
    selected = selected.reset_index(drop=True)
    selected["rank"] = range(1, len(selected) + 1)
    return selected


def write_universe_outputs(
    base_dir: str | Path,
    *,
    date_value: str,
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    status: dict[str, object],
    write_latest: bool = True,
) -> dict[str, Path]:
    output_dir = Path(base_dir).expanduser() / "universe" / f"date={date_value}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = output_dir / "us_microstructure_candidates.csv"
    scored_csv = output_dir / "us_microstructure_screened_universe.csv"
    candidates_txt = output_dir / "us_microstructure_candidates.txt"
    status_path = output_dir / "status.json"
    candidates.to_csv(candidates_csv, index=False)
    scored.to_csv(scored_csv, index=False)
    symbols = candidates["symbol"].dropna().astype(str).tolist() if "symbol" in candidates.columns else []
    candidates_txt.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs = {
        "candidates_csv": candidates_csv,
        "scored_csv": scored_csv,
        "candidates_txt": candidates_txt,
        "status": status_path,
    }
    if write_latest:
        latest_dir = Path(base_dir).expanduser() / "universe"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_csv = latest_dir / "us_microstructure_candidates_latest.csv"
        latest_scored = latest_dir / "us_microstructure_screened_universe_latest.csv"
        latest_txt = latest_dir / "us_microstructure_candidates_latest.txt"
        latest_status = latest_dir / "us_microstructure_universe_status_latest.json"
        candidates.to_csv(latest_csv, index=False)
        scored.to_csv(latest_scored, index=False)
        latest_txt.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
        latest_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.update(
            {
                "candidates_latest_csv": latest_csv,
                "scored_latest_csv": latest_scored,
                "candidates_latest_txt": latest_txt,
                "status_latest": latest_status,
            }
        )
    return outputs


def _sync_outputs(paths: Iterable[Path], *, base_dir: Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    results = []
    if not nas_host or not nas_dir:
        return results
    for path in paths:
        status, remote_path, error = _copy_to_nas(path, base_dir, nas_host, nas_dir)
        results.append({"local_path": str(path), "nas_path": remote_path, "status": status, "error": error})
    return results


def build_universe(
    *,
    ctx: Any,
    base_dir: str | Path,
    date_value: str,
    target_size: int,
    core_symbols: list[str],
    include_exchange_types: set[str],
    exclude_exchange_types: set[str],
    exclude_security_classes: set[str],
    max_universe_codes: int,
    min_price: float,
    min_snapshot_turnover: float,
    min_snapshot_volume: float,
    history_pool_size: int,
    minute_pool_size: int,
    daily_lookback_days: int,
    minute_lookback: int,
    snapshot_batch_size: int,
    snapshot_sleep_seconds: float,
    history_sleep_seconds: float,
    minute_sleep_seconds: float,
    skip_daily_kline: bool,
    skip_minute_kline: bool,
    flow_ranking_file: str | Path | None = None,
    core_metadata: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    from futu import AuType, KLType

    universe = fetch_futu_universe(
        type("Client", (), {"ctx": ctx})(),
        "US",
        include_exchange_types=include_exchange_types,
        exclude_exchange_types=exclude_exchange_types,
        exclude_security_classes=exclude_security_classes,
        reference_date=date_value,
    )
    if max_universe_codes > 0:
        universe = universe.head(max_universe_codes).copy()
    universe_symbols = normalize_us_symbols(universe["code"].dropna().astype(str).tolist()) if "code" in universe.columns else []
    for symbol in normalize_us_symbols(core_symbols):
        if symbol not in universe_symbols:
            universe = pd.concat([universe, pd.DataFrame([{"code": symbol, "symbol": symbol, "name": "", "exchange_type": "CORE"}])], ignore_index=True)
            universe_symbols.append(symbol)

    snapshot, snapshot_errors = fetch_market_snapshots(
        ctx,
        universe_symbols,
        batch_size=snapshot_batch_size,
        sleep_seconds=snapshot_sleep_seconds,
    )
    flow_ranking = load_flow_ranking(flow_ranking_file)
    flow_positive_count = (
        int((pd.to_numeric(flow_ranking["flow_activity"], errors="coerce").fillna(0.0) > 0).sum())
        if not flow_ranking.empty and "flow_activity" in flow_ranking.columns
        else 0
    )
    enrichment_symbols, enrichment_ranking_source = select_enrichment_symbols(
        universe_symbols,
        snapshot,
        flow_ranking,
        core_symbols=core_symbols,
        limit=max(int(history_pool_size), int(minute_pool_size), int(target_size)),
    )

    daily = pd.DataFrame()
    daily_errors: dict[str, str] = {}
    daily_symbols: list[str] = []
    if not skip_daily_kline and enrichment_symbols:
        daily_symbols = enrichment_symbols[: max(0, int(history_pool_size))]
        daily, daily_errors = fetch_daily_metrics(
            ctx,
            daily_symbols,
            start_date=_date_days_before(date_value, daily_lookback_days),
            end_date=date_value,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            sleep_seconds=history_sleep_seconds,
        )

    minute = pd.DataFrame()
    minute_errors: dict[str, str] = {}
    minute_symbols: list[str] = []
    if not skip_minute_kline and enrichment_symbols:
        minute_symbols = enrichment_symbols[: max(0, int(minute_pool_size))]
        minute, minute_errors = fetch_minute_metrics(
            ctx,
            minute_symbols,
            lookback=minute_lookback,
            ktype=KLType.K_1M,
            autype=AuType.QFQ,
            sleep_seconds=minute_sleep_seconds,
        )

    scored = _score_candidates(
        universe,
        snapshot,
        daily,
        minute,
        flow_ranking,
        core_symbols=core_symbols,
        min_price=min_price,
        min_snapshot_turnover=min_snapshot_turnover,
        min_snapshot_volume=min_snapshot_volume,
    )
    candidates = select_candidates(scored, target_size=target_size, core_symbols=core_symbols)
    selection_source = (
        candidates["selection_source"].value_counts().to_dict()
        if not candidates.empty and "selection_source" in candidates.columns
        else {}
    )
    status = {
        "status_schema_version": STATUS_SCHEMA_VERSION,
        "status": "ok" if not candidates.empty else "empty",
        "date": date_value,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(Path(base_dir).expanduser()),
        "market": "US",
        "layer": "coarse_screen",
        "target_size": int(target_size),
        "universe_count": int(len(universe)),
        "snapshot_symbol_count": int(snapshot["symbol"].nunique()) if not snapshot.empty else 0,
        "snapshot_positive_liquidity_count": _snapshot_positive_liquidity_count(snapshot),
        "flow_ranking_file": str(Path(flow_ranking_file).expanduser()) if flow_ranking_file else "",
        "flow_ranking_symbol_count": int(flow_ranking["symbol"].nunique()) if not flow_ranking.empty and "symbol" in flow_ranking.columns else 0,
        "flow_ranking_positive_count": int(flow_positive_count),
        "enrichment_ranking_source": enrichment_ranking_source,
        "enrichment_symbol_count": int(len(enrichment_symbols)),
        "enrichment_first_letter_counts": _first_letter_counts(enrichment_symbols),
        "daily_symbol_requested_count": int(len(daily_symbols)),
        "minute_symbol_requested_count": int(len(minute_symbols)),
        "daily_symbol_count": int(daily["symbol"].nunique()) if not daily.empty and "symbol" in daily.columns else 0,
        "minute_symbol_count": int(minute["symbol"].nunique()) if not minute.empty and "symbol" in minute.columns else 0,
        "candidate_count": int(len(candidates)),
        "candidate_first_letter_counts": _first_letter_counts(candidates["symbol"].tolist()) if not candidates.empty and "symbol" in candidates.columns else {},
        "core_symbol_count": int(len(normalize_us_symbols(core_symbols))),
        "core_symbol_source": "explicit",
        "candidate_core_count": int(candidates["core_symbol"].sum()) if not candidates.empty and "core_symbol" in candidates.columns else 0,
        "candidate_liquidity_ranked_count": int(selection_source.get("liquidity_ranked", 0)),
        "candidate_fallback_ranked_count": int(selection_source.get("fallback_ranked", 0)),
        "candidate_target_shortfall": max(0, int(target_size) - int(len(candidates))),
        "min_price": float(min_price),
        "min_snapshot_turnover": float(min_snapshot_turnover),
        "min_snapshot_volume": float(min_snapshot_volume),
        "history_pool_size": int(history_pool_size),
        "minute_pool_size": int(minute_pool_size),
        "daily_lookback_days": int(daily_lookback_days),
        "minute_lookback": int(minute_lookback),
        "include_exchange_types": sorted(include_exchange_types),
        "exclude_exchange_types": sorted(exclude_exchange_types),
        "exclude_security_classes": sorted(exclude_security_classes),
        "snapshot_error_count": int(len(snapshot_errors)),
        "daily_error_count": int(len(daily_errors)),
        "minute_error_count": int(len(minute_errors)),
        "snapshot_errors": dict(list(snapshot_errors.items())[:20]),
        "daily_errors": dict(list(daily_errors.items())[:20]),
        "minute_errors": dict(list(minute_errors.items())[:20]),
        "candidates": candidates[
            [
                column
                for column in ("rank", "symbol", "coarse_score", "screen_reason", "selection_source")
                if column in candidates.columns
            ]
        ].to_dict("records") if not candidates.empty else [],
    }
    if core_metadata:
        status.update(core_metadata)
        status["core_symbol_count"] = int(len(normalize_us_symbols(core_symbols)))
    return candidates, scored, status


def _configure_futu_encryption(rsa_key: str) -> None:
    from futu import SysConfig

    key = Path(rsa_key).expanduser() if rsa_key else None
    if key and key.exists():
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(str(key))
    else:
        SysConfig.enable_proto_encrypt(False)
        print(f"WARNING: Futu RSA key not found, connecting without encryption: {key}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic US microstructure collection universe.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_UNIVERSE_DATE", _screen_date_from_utc()))
    parser.add_argument("--target-size", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_TARGET_SIZE", "124")))
    parser.add_argument("--core-symbols-file", default=os.environ.get("US_MICROSTRUCTURE_CORE_SYMBOLS_FILE", str(DEFAULT_CORE_SYMBOLS_FILE)))
    parser.add_argument("--core-source", default=os.environ.get("US_MICROSTRUCTURE_CORE_SOURCE", DEFAULT_CORE_SOURCE))
    parser.add_argument("--core-watchlist-groups", default=os.environ.get("US_MICROSTRUCTURE_CORE_WATCHLIST_GROUPS", ""))
    parser.add_argument("--core-watchlist-group-type", default=os.environ.get("US_MICROSTRUCTURE_CORE_WATCHLIST_GROUP_TYPE", DEFAULT_CORE_WATCHLIST_GROUP_TYPE))
    parser.add_argument("--host", default=os.environ.get("FUTU_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FUTU_PORT", "11111")))
    parser.add_argument("--rsa-key", default=os.environ.get("FUTU_RSA_KEY", str(DEFAULT_RSA_KEY)))
    parser.add_argument("--flow-ranking-file", default=os.environ.get("US_MICROSTRUCTURE_FLOW_RANKING_FILE", str(DEFAULT_FLOW_RANKING_FILE)))
    parser.add_argument("--include-exchange-types", default=os.environ.get("US_MICROSTRUCTURE_UNIVERSE_INCLUDE_EXCHANGE_TYPES", ""))
    parser.add_argument("--exclude-exchange-types", default=os.environ.get("US_MICROSTRUCTURE_UNIVERSE_EXCLUDE_EXCHANGE_TYPES", "US_PINK,N/A"))
    parser.add_argument("--exclude-security-classes", default=os.environ.get("US_MICROSTRUCTURE_UNIVERSE_EXCLUDE_SECURITY_CLASSES", DEFAULT_EXCLUDE_SECURITY_CLASSES))
    parser.add_argument("--max-universe-codes", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MAX_CODES", "0")))
    parser.add_argument("--min-price", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MIN_PRICE", "2")))
    parser.add_argument("--min-snapshot-turnover", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MIN_TURNOVER", "1000000")))
    parser.add_argument("--min-snapshot-volume", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MIN_VOLUME", "50000")))
    parser.add_argument("--history-pool-size", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_HISTORY_POOL_SIZE", "500")))
    parser.add_argument("--minute-pool-size", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MINUTE_POOL_SIZE", "300")))
    parser.add_argument("--daily-lookback-days", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_DAILY_LOOKBACK_DAYS", "30")))
    parser.add_argument("--minute-lookback", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MINUTE_LOOKBACK", "30")))
    parser.add_argument("--snapshot-batch-size", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_SNAPSHOT_BATCH_SIZE", "200")))
    parser.add_argument("--snapshot-sleep-seconds", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_SNAPSHOT_SLEEP_SECONDS", "0.05")))
    parser.add_argument("--history-sleep-seconds", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_HISTORY_SLEEP_SECONDS", str(DEFAULT_HISTORY_SLEEP_SECONDS))))
    parser.add_argument("--minute-sleep-seconds", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_UNIVERSE_MINUTE_SLEEP_SECONDS", "0.02")))
    parser.add_argument("--skip-daily-kline", action="store_true")
    parser.add_argument("--skip-minute-kline", action="store_true")
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from futu import OpenQuoteContext

    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    _configure_futu_encryption(args.rsa_key)
    ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        core_symbols, core_metadata = resolve_core_symbols(
            ctx,
            core_symbols_file=args.core_symbols_file,
            core_source=args.core_source,
            core_watchlist_groups=args.core_watchlist_groups,
            core_watchlist_group_type=args.core_watchlist_group_type,
        )
        candidates, scored, status = build_universe(
            ctx=ctx,
            base_dir=base_dir,
            date_value=args.date[:10],
            target_size=args.target_size,
            core_symbols=core_symbols,
            include_exchange_types=_split_csv(args.include_exchange_types),
            exclude_exchange_types=_split_csv(args.exclude_exchange_types),
            exclude_security_classes=_split_classes(args.exclude_security_classes),
            max_universe_codes=args.max_universe_codes,
            min_price=args.min_price,
            min_snapshot_turnover=args.min_snapshot_turnover,
            min_snapshot_volume=args.min_snapshot_volume,
            history_pool_size=args.history_pool_size,
            minute_pool_size=args.minute_pool_size,
            daily_lookback_days=args.daily_lookback_days,
            minute_lookback=args.minute_lookback,
            snapshot_batch_size=args.snapshot_batch_size,
            snapshot_sleep_seconds=args.snapshot_sleep_seconds,
            history_sleep_seconds=args.history_sleep_seconds,
            minute_sleep_seconds=args.minute_sleep_seconds,
            skip_daily_kline=bool(args.skip_daily_kline),
            skip_minute_kline=bool(args.skip_minute_kline),
            flow_ranking_file=args.flow_ranking_file,
            core_metadata=core_metadata,
        )
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    outputs = write_universe_outputs(base_dir, date_value=args.date[:10], candidates=candidates, scored=scored, status=status)
    if not args.no_nas_sync:
        nas_results = _sync_outputs(outputs.values(), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        if nas_results:
            status["nas_sync"] = nas_results
            outputs = write_universe_outputs(base_dir, date_value=args.date[:10], candidates=candidates, scored=scored, status=status)
            _sync_outputs([outputs["status"], outputs.get("status_latest", outputs["status"])], base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)

    print(
        "Built US microstructure universe: candidates={candidates} universe={universe} "
        "snapshot={snapshot} daily={daily} minute={minute}".format(
            candidates=len(candidates),
            universe=int(status.get("universe_count") or 0),
            snapshot=int(status.get("snapshot_symbol_count") or 0),
            daily=int(status.get("daily_symbol_count") or 0),
            minute=int(status.get("minute_symbol_count") or 0),
        )
    )
    print(f"Wrote candidate file: {outputs['candidates_latest_txt']}")
    return 0 if not candidates.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
