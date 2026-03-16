"""
A股股票池过滤器。

过滤规则：
1. ST / *ST：最新股票名称包含 "ST" 的剔除
2. 次新股：交易日数不足 252 天（约1年）的剔除
3. 低流动性：近60个交易日日均成交额 < 5000万 的剔除

数据源：K线 Parquet（含 name, time_key, turnover 字段）
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MIN_TURNOVER = 50_000_000  # 5000万
DEFAULT_MIN_TRADING_DAYS = 252  # ~1年交易日
LOOKBACK_DAYS = 60  # 成交额计算窗口


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
            if "ST" in latest_name.upper():
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
