"""
NAS Docker 每日自动交易 — 严格匹配回测时序。

每个交易日 14:50 由 cron 触发:
1. 从 pred_sh.pkl 提取最近一个交易日的信号（信号日 t）
2. 从 Qlib bin 格式读取信号日涨跌幅
3. 查当前持仓，应用持仓惯性
4. 获取实时行情，双重涨停过滤（信号日 + 买入日）
5. Top-N 选股，止损检查
6. 先卖后买，等权分配

数据源:
    pred_sh.pkl  → /models/pred_sh.pkl (mount)
    Qlib 数据    → /qlib_data/         (mount, features + calendars)
    futu OpenD   → futu-opend:11111    (Docker network)
"""

from __future__ import annotations

import bisect
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from futu import (
    OpenSecTradeContext,
    OpenQuoteContext,
    TrdMarket,
    TrdEnv,
    TrdSide,
    OrderType,
    RET_OK,
)

# ─── 安全锁 ────────────────────────────────────────────
SAFE_TRD_ENV = TrdEnv.SIMULATE
assert SAFE_TRD_ENV == TrdEnv.SIMULATE, "禁止使用真实交易环境！"

# ─── 配置（环境变量覆盖）────────────────────────────────
FUTU_HOST = os.environ.get("FUTU_HOST", "futu-opend")
FUTU_PORT = int(os.environ.get("FUTU_PORT", "11111"))
PRED_PATH = Path(os.environ.get("PRED_PATH", "/models/pred_sh.pkl"))
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/signals"))

TOP_N = int(os.environ.get("TOP_N", "5"))
HOLD_BONUS = float(os.environ.get("HOLD_BONUS", "0.05"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "-0.08"))
POSITION_RATIO = 0.95
MIN_LOT_SH = 100
BUY_PRICE_SLIPPAGE = 1.01    # 买入价上浮 1% 确保成交
SELL_PRICE_SLIPPAGE = 0.99   # 卖出价下浮 1% 确保成交
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trader")


# ─── 工具函数 ───────────────────────────────────────────

def _round_lot(qty: float) -> int:
    return max(int(qty // MIN_LOT_SH) * MIN_LOT_SH, 0)


def _get_limit_up_pct(code: str) -> float:
    if code.startswith("SH.300") or code.startswith("SH.688"):
        return 19.5
    return 9.5


def _is_limit_up(code: str, change_rate: float) -> bool:
    if not code.startswith("SH."):
        return False
    return change_rate >= _get_limit_up_pct(code)


# ─── 信号提取（从 pred_sh.pkl）────────────────────────

def extract_signals(pred_path: Path, signal_date: str | None = None) -> tuple[pd.DataFrame, str]:
    """从 pred_sh.pkl 提取指定日期（默认最新）的信号。

    返回 (DataFrame[code, score], signal_date_str)
    """
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)

    dates = sorted(pred.index.get_level_values("datetime").unique())

    if signal_date:
        target = pd.Timestamp(signal_date)
        if target not in dates:
            # 找最近的 <= target 的日期
            valid = [d for d in dates if d <= target]
            if not valid:
                raise ValueError(f"pred_sh.pkl 无 <= {signal_date} 的数据")
            target = valid[-1]
    else:
        target = dates[-1]

    day_pred = pred.xs(target, level="datetime")
    if isinstance(day_pred, pd.DataFrame):
        day_pred = day_pred.iloc[:, 0]
    day_pred = day_pred.dropna()

    df = pd.DataFrame({
        "code": day_pred.index.astype(str),
        "score": day_pred.values,
    })
    # 只取 A 股
    df = df[df["code"].str.startswith("SH.")].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    date_str = target.strftime("%Y-%m-%d")
    log.info(f"信号: {date_str}, A股 {len(df)} 只, "
             f"Top-3 score: {df['score'].head(3).tolist()}")
    return df, date_str


# ─── Qlib bin 读取工具 ─────────────────────────────────

def _code_to_fname(code: str) -> str:
    """Convert stock code to Qlib feature directory name."""
    replace_names = ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(10)] + [f"LPT{i}" for i in range(10)]
    if str(code).upper() in replace_names:
        return "_qlib_" + str(code)
    return str(code)


def _load_calendar() -> list[str]:
    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    if not cal_path.exists():
        return []
    return [l.strip() for l in cal_path.read_text().splitlines() if l.strip()]


def _read_qlib_field(code: str, field: str, calendar: list[str]) -> pd.Series:
    """Read a single field from Qlib bin file."""
    feat_dir = QLIB_DATA_DIR / "features" / _code_to_fname(code).lower()
    bin_path = feat_dir / f"{field}.day.bin"
    if not bin_path.exists():
        return pd.Series(dtype="float64")
    data = np.fromfile(str(bin_path), dtype="<f4")
    if len(data) == 0:
        return pd.Series(dtype="float64")
    start_idx = int(data[0])
    values = data[1:]
    end_idx = start_idx + len(values)
    dates = calendar[start_idx:end_idx]
    return pd.Series(values.astype("float64"), index=dates, name=field)


# ─── 信号日涨跌幅（从 Qlib bin）──────────────────────

def load_signal_day_changes(signal_date: str, codes: list[str]) -> dict[str, float]:
    """从 Qlib bin 读取信号日涨跌幅。"""
    calendar = _load_calendar()
    if not calendar:
        log.warning("Qlib calendar not found")
        return {}
    changes = {}
    for code in codes:
        try:
            s = _read_qlib_field(code, "change_rate", calendar)
            if signal_date in s.index:
                v = float(s[signal_date])
                if pd.notna(v):
                    changes[code] = v
        except Exception:
            continue
    return changes


# ─── 交易执行 ───────────────────────────────────────────

def get_positions(trd_ctx) -> dict[str, dict]:
    ret, data = trd_ctx.position_list_query(trd_env=SAFE_TRD_ENV)
    if ret != RET_OK:
        log.error(f"查询持仓失败: {data}")
        return {}
    positions = {}
    for _, row in data.iterrows():
        if row["qty"] > 0:
            positions[row["code"]] = {
                "qty": int(row["qty"]),
                "market_val": float(row["market_val"]),
                "cost_price": float(row["cost_price"]),
                "pl_ratio": float(row.get("pl_ratio", 0)),
            }
    return positions


def get_account_info(trd_ctx) -> dict:
    ret, data = trd_ctx.accinfo_query(trd_env=SAFE_TRD_ENV)
    if ret != RET_OK:
        log.error(f"查询账户失败: {data}")
        return {}
    row = data.iloc[0]
    info = {
        "total_assets": float(row.get("total_assets", 0)),
        "cash": float(row.get("cash", 0)),
        "market_val": float(row.get("market_val", 0)),
    }
    log.info(f"账户: 总资产={info['total_assets']:,.0f}  "
             f"现金={info['cash']:,.0f}  持仓市值={info['market_val']:,.0f}")
    return info


def get_latest_prices(quote_ctx, codes: list[str]) -> tuple[dict, dict]:
    prices, changes = {}, {}
    for code in codes:
        ret, data = quote_ctx.get_market_snapshot([code])
        if ret == RET_OK and not data.empty:
            row = data.iloc[0]
            prices[code] = float(row["last_price"])
            changes[code] = float(row.get("change_rate", 0))
        else:
            log.warning(f"行情失败: {code}")
        time.sleep(0.3)
    return prices, changes


def run_trade(
    trd_ctx, quote_ctx,
    signals_df: pd.DataFrame,
    signal_day_changes: dict[str, float],
    dry_run: bool = False,
):
    """执行换仓 — 与 backtest.py 完全一致。"""

    account = get_account_info(trd_ctx)
    if not account:
        return
    positions = get_positions(trd_ctx)
    current_codes = set(positions.keys())

    # 持仓惯性（backtest.py 第 88-91 行）
    candidates = signals_df.copy()
    if HOLD_BONUS > 0 and current_codes:
        held = candidates["code"].isin(current_codes)
        candidates.loc[held, "score"] += HOLD_BONUS
        n = held.sum()
        if n > 0:
            log.info(f"持仓惯性: {n} 只 score += {HOLD_BONUS}")
        candidates = candidates.sort_values("score", ascending=False)

    codes_ordered = candidates["code"].tolist()

    # 获取实时行情
    query_codes = list(set(codes_ordered[:TOP_N * 3]) | current_codes)
    prices, buy_day_chg = get_latest_prices(quote_ctx, query_codes)

    # 候选筛选（backtest.py 第 96-115 行）
    filtered = []
    for code in codes_ordered:
        if code not in prices:
            continue

        if code.startswith("SH."):
            limit_pct = _get_limit_up_pct(code)

            # 信号日涨停（backtest.py 第 108 行）
            chg_t = signal_day_changes.get(code, float("nan"))
            if pd.notna(chg_t) and chg_t >= limit_pct:
                log.warning(f"涨停过滤(信号日): {code} {chg_t:.1f}%")
                continue

            # 买入日涨停（backtest.py 第 109 行）
            chg_t1 = buy_day_chg.get(code, 0)
            if chg_t1 >= limit_pct:
                log.warning(f"涨停过滤(买入日): {code} {chg_t1:.1f}%")
                continue

        filtered.append(code)
        if len(filtered) >= TOP_N:
            break

    target_set = set(filtered)
    log.info(f"目标持仓 Top-{TOP_N}: {filtered}")

    # 止损
    stop_loss_sells = set()
    for code, pos in positions.items():
        pl_ratio = pos.get("pl_ratio", 0) / 100.0
        if pl_ratio <= STOP_LOSS_PCT:
            log.warning(f"止损: {code} 浮亏 {pl_ratio:.1%}")
            stop_loss_sells.add(code)
            target_set.discard(code)

    # 买卖计算（backtest.py 第 131-133 行）
    sells = (current_codes - target_set) | stop_loss_sells
    buys = target_set - current_codes
    holds = current_codes & target_set

    log.info(f"当前: {sorted(current_codes) or '空仓'}")
    log.info(f"卖出: {sorted(sells) or '无'}  买入: {sorted(buys) or '无'}  "
             f"持有: {sorted(holds) or '无'}")

    if not sells and not buys:
        log.info("无需调仓")
        return

    # 先卖（价格下浮确保成交）
    for code in sells:
        qty = positions[code]["qty"]
        price = prices.get(code)
        if not price:
            continue
        sell_price = round(price * SELL_PRICE_SLIPPAGE, 2)
        log.info(f"卖出 {code}: {qty}股 @ {sell_price:.2f} (市价{price:.2f})")
        if not dry_run:
            ret, data = trd_ctx.place_order(
                price=sell_price, qty=qty, code=code,
                trd_side=TrdSide.SELL, order_type=OrderType.NORMAL,
                trd_env=SAFE_TRD_ENV,
            )
            log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
            time.sleep(1)

    if sells and not dry_run:
        time.sleep(3)
        account = get_account_info(trd_ctx)

    # 后买
    if not buys:
        return

    budget = account.get("cash", 0) * POSITION_RATIO / max(len(target_set), 1)
    log.info(f"每只预算: {budget:,.0f}")

    for code in buys:
        price = prices.get(code)
        if not price or price <= 0:
            continue
        buy_price = round(price * BUY_PRICE_SLIPPAGE, 2)
        qty = _round_lot(budget / buy_price)
        if qty <= 0:
            log.warning(f"跳过 {code}: 资金不足")
            continue
        log.info(f"买入 {code}: {qty}股 @ {buy_price:.2f} (市价{price:.2f}) = {qty * buy_price:,.0f}")
        if not dry_run:
            ret, data = trd_ctx.place_order(
                price=buy_price, qty=qty, code=code,
                trd_side=TrdSide.BUY, order_type=OrderType.NORMAL,
                trd_env=SAFE_TRD_ENV,
            )
            log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
            time.sleep(1)


# ─── 主流程 ──────────────────────────────────────────

def main():
    dry_run = DRY_RUN or "--dry-run" in sys.argv
    signal_date = None
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            signal_date = arg.split("=")[1]

    if dry_run:
        log.info("=== DRY RUN ===")

    log.info(f"配置: TOP_N={TOP_N} HOLD_BONUS={HOLD_BONUS} "
             f"STOP_LOSS={STOP_LOSS_PCT} FUTU={FUTU_HOST}:{FUTU_PORT}")

    # 1. 提取信号
    if not PRED_PATH.exists():
        log.error(f"预测文件不存在: {PRED_PATH}")
        return
    signals_df, sig_date = extract_signals(PRED_PATH, signal_date)

    # 2. 读取信号日涨跌幅
    signal_changes = {}
    if KLINE_DIR.exists():
        sh_codes = signals_df["code"].tolist()
        signal_changes = load_signal_day_changes(sig_date, sh_codes[:50])
        n_limit = sum(1 for c in signal_changes if _is_limit_up(c, signal_changes[c]))
        log.info(f"信号日涨跌幅: {len(signal_changes)} 只, 涨停 {n_limit} 只")
    else:
        log.warning(f"kline 数据不可用: {KLINE_DIR}")

    # 保存信号快照
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    sig_file = SIGNAL_DIR / f"signal_{sig_date}.csv"
    signals_df["signal_change_rate"] = signals_df["code"].map(signal_changes)
    signals_df.to_csv(sig_file, index=False)
    log.info(f"信号快照: {sig_file}")

    # 3. 连接 futu
    log.info(f"连接 OpenD {FUTU_HOST}:{FUTU_PORT} ...")
    trd_ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.CN,
        host=FUTU_HOST, port=FUTU_PORT,
    )
    quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)

    try:
        ret, acc_list = trd_ctx.get_acc_list()
        if ret != RET_OK:
            log.error(f"账户列表失败: {acc_list}")
            return

        sim = acc_list[acc_list["trd_env"] == "SIMULATE"]
        real = acc_list[acc_list["trd_env"] == "REAL"]
        if sim.empty:
            log.error("无模拟账户")
            return
        log.info(f"模拟账户: {sim['acc_id'].tolist()}")
        if not real.empty:
            log.warning(f"真实账户 {real['acc_id'].tolist()} — 不触碰")
        assert SAFE_TRD_ENV == TrdEnv.SIMULATE

        # 4. 执行交易
        run_trade(trd_ctx, quote_ctx, signals_df, signal_changes, dry_run=dry_run)
        log.info("完成")

    finally:
        trd_ctx.close()
        quote_ctx.close()


if __name__ == "__main__":
    main()
