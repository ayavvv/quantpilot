"""
A股股票池过滤器。

过滤规则：
1. ST / *ST：最新股票名称包含 "ST" 的剔除
2. 次新股：交易日数不足 252 天（约1年）的剔除
3. 低流动性：近60个交易日日均成交额 < 5000万 的剔除

数据源：K线 Parquet（含 name, time_key, turnover 字段）
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MIN_TURNOVER = 50_000_000  # 5000万
DEFAULT_MIN_TRADING_DAYS = 252  # ~1年交易日
LOOKBACK_DAYS = 60  # 成交额计算窗口
A_SHARE_ST_METADATA = "a_share_stock_basic"
ST_NAME_PREFIXES = ("*ST", "ST", "S*ST", "*SST", "SST")


def is_st_stock_name(name: object) -> bool:
    """Return True for current A-share ST/*ST display names."""
    if name is None:
        return False
    normalized = unicodedata.normalize("NFKC", str(name)).strip().upper()
    normalized = normalized.replace(" ", "")
    return normalized.startswith(ST_NAME_PREFIXES)


def a_share_st_filter_enabled() -> bool:
    return os.environ.get("A_SHARE_EXCLUDE_ST", "true").lower() not in {"0", "false", "no"}


def build_a_share_stock_basic_metadata(rows: Iterable[dict], source: str = "baostock") -> dict:
    """Build compact JSON metadata for current A-share names and ST status."""
    stocks = []
    for row in rows:
        code = str(row.get("code", "")).strip().upper()
        if not code.startswith(("SH.", "SZ.")):
            continue
        name = str(row.get("name", row.get("code_name", ""))).strip()
        is_st = is_st_stock_name(name)
        stocks.append(
            {
                "code": code,
                "name": name,
                "is_st": is_st,
                "ipoDate": row.get("ipoDate", ""),
                "outDate": row.get("outDate", ""),
                "type": row.get("type", ""),
                "status": row.get("status", ""),
            }
        )

    st_codes = sorted(stock["code"] for stock in stocks if stock["is_st"])
    return {
        "source": source,
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(stocks),
        "st_count": len(st_codes),
        "st_codes": st_codes,
        "stocks": sorted(stocks, key=lambda item: item["code"]),
    }


def load_a_share_st_codes(provider_uri: str | Path) -> set[str]:
    """Load current ST/*ST A-share codes from Qlib metadata."""
    if not a_share_st_filter_enabled():
        return set()

    meta_path = Path(provider_uri).expanduser().resolve() / "metadata" / f"{A_SHARE_ST_METADATA}.json"
    if not meta_path.exists():
        log.warning("A-share ST metadata missing: %s", meta_path)
        return set()

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read A-share ST metadata %s: %s", meta_path, exc)
        return set()

    st_codes = payload.get("st_codes")
    if isinstance(st_codes, list):
        return {str(code).strip().upper() for code in st_codes if str(code).strip()}

    stocks = payload.get("stocks")
    if isinstance(stocks, list):
        return {
            str(stock.get("code", "")).strip().upper()
            for stock in stocks
            if stock.get("is_st") or is_st_stock_name(stock.get("name"))
        }

    return set()


def load_a_share_st_codes_for_date(
    provider_uri: str | Path,
    codes: Iterable[str],
    as_of_date: str,
) -> set[str] | None:
    """Load ST flags from the point-in-time Qlib ``is_st`` feature.

    Returns None when the historical field is not available or too sparse,
    allowing callers to fall back to the current stock-basic metadata.
    """
    if not a_share_st_filter_enabled():
        return set()

    code_list = list(codes)
    if not code_list:
        return set()

    try:
        from converter.incremental import QlibBinReader

        reader = QlibBinReader(provider_uri)
        st_df = reader.read_field_matrix(
            code_list,
            "is_st",
            start_date=as_of_date,
            end_date=as_of_date,
        )
    except Exception as exc:
        log.warning("Failed to read historical A-share ST flags for %s: %s", as_of_date, exc)
        return None

    if st_df.empty or as_of_date not in st_df.index:
        return None

    # Treat very sparse historical coverage as unavailable. This avoids mixing a
    # partially backfilled is_st feature with an otherwise current-ST fallback.
    if len(st_df.columns) < max(1, int(len(code_list) * 0.9)):
        return None

    row = st_df.loc[as_of_date].fillna(0)
    return {str(code) for code, value in row.items() if float(value) >= 0.5}


def filter_st_codes(
    provider_uri: str | Path,
    codes: Iterable[str],
    *,
    context: str = "A-share universe",
    as_of_date: str | None = None,
) -> list[str]:
    """Remove ST/*ST names from an A-share instrument list.

    When ``as_of_date`` is provided, prefer the historical ``is_st`` feature so
    filtering is point-in-time. If that feature is unavailable, fall back to the
    current metadata snapshot.
    """
    code_list = list(codes)
    st_codes = (
        load_a_share_st_codes_for_date(provider_uri, code_list, as_of_date)
        if as_of_date
        else None
    )
    if st_codes is None:
        st_codes = load_a_share_st_codes(provider_uri)
    if not st_codes:
        return code_list

    filtered = [code for code in code_list if str(code).strip().upper() not in st_codes]
    removed = len(code_list) - len(filtered)
    if removed:
        log.info("%s: excluded %d ST/*ST stocks", context, removed)
        print(f"[INFO] {context}: 排除 ST/*ST {removed} 只")
    return filtered


def filter_stock_universe(
    data_source: Path,
    min_avg_turnover: float = DEFAULT_MIN_TURNOVER,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
) -> tuple[list[str], dict[str, str]]:
    """
    扫描 data_source 下所有 A 股 parquet，返回通过过滤的股票代码列表。

    Args:
        data_source: K_DAY 数据目录（每只股票一个子目录，内含 data.parquet）
        min_avg_turnover: 近 N 日日均成交额阈值（元）
        min_trading_days: 最少交易日数（近似上市时长）

    Returns:
        (passed_codes, rejected: {code: reason})
    """
    passed: list[str] = []
    rejected: dict[str, str] = {}

    dirs = sorted(d for d in data_source.iterdir() if d.is_dir())
    for code_dir in dirs:
        code = code_dir.name

        # 只处理 A 股（沪 SH / 深 SZ）
        if not (code.startswith("SH.") or code.startswith("SZ.")):
            continue

        parquet_path = code_dir / "data.parquet"
        if not parquet_path.exists():
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            rejected[code] = "parquet_read_error"
            continue

        if df.empty:
            rejected[code] = "empty_data"
            continue

        # --- 1. ST / *ST 检测（最近一条记录的 name）---
        if "name" in df.columns:
            latest_name = str(df.iloc[-1]["name"])
            if is_st_stock_name(latest_name):
                rejected[code] = f"ST: {latest_name}"
                continue

        # --- 2. 次新股检测 ---
        n_trading_days = len(df)
        if n_trading_days < min_trading_days:
            rejected[code] = f"次新股: {n_trading_days} 天 < {min_trading_days}"
            continue

        # --- 3. 日均成交额检测 ---
        turnover_col = None
        for col in ("turnover", "amount", "turn_over"):
            if col in df.columns:
                turnover_col = col
                break
        if turnover_col:
            recent = df.tail(LOOKBACK_DAYS)
            avg_turnover = recent[turnover_col].astype(float).mean()
            if pd.isna(avg_turnover) or avg_turnover < min_avg_turnover:
                val = avg_turnover / 1e6 if pd.notna(avg_turnover) else 0
                rejected[code] = f"低流动性: 日均{val:.0f}M < {min_avg_turnover / 1e6:.0f}M"
                continue

        passed.append(code)

    log.info(
        f"股票过滤: {len(passed)} 只通过, {len(rejected)} 只剔除 "
        f"(ST={sum(1 for r in rejected.values() if r.startswith('ST'))} "
        f"次新={sum(1 for r in rejected.values() if r.startswith('次新'))} "
        f"低流动={sum(1 for r in rejected.values() if r.startswith('低流动'))})"
    )
    return passed, rejected
