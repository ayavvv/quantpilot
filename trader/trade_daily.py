"""
每日自动交易 -- 严格匹配回测时序。

每个交易日 14:50 由 cron 触发:
1. 从 pred_sh.pkl 提取最近一个交易日的信号(信号日 t)
2. 从 Qlib bin 读取信号日涨跌幅
3. 查当前持仓, 应用持仓惯性
4. 获取实时行情, 双重涨停过滤(信号日 + 买入日)
5. Top-N 选股, 止损检查
6. 先卖后买, 等权分配

数据源:
    pred_sh.pkl  -> /models/pred_sh.pkl (mount)
    Qlib bin     -> /qlib_data/         (mount, 与 collector 共享)
    futu OpenD   -> FUTU_HOST:FUTU_PORT (Docker network)
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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

# --- Safety Lock ---
SAFE_TRD_ENV = TrdEnv.SIMULATE
assert SAFE_TRD_ENV == TrdEnv.SIMULATE, "Real trading environment is disabled!"

# --- Configuration (env var overrides) ---
FUTU_HOST = os.environ.get("FUTU_HOST", "localhost")
FUTU_PORT = int(os.environ.get("FUTU_PORT", "11111"))
PRED_PATH = Path(os.environ.get("PRED_PATH", "/models/pred_sh.pkl"))
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/signals"))

TOP_N = int(os.environ.get("TOP_N", "5"))
HOLD_BONUS = float(os.environ.get("HOLD_BONUS", "0.05"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "-0.08"))
POSITION_RATIO = 0.95
MIN_LOT_SH = 100
BUY_PRICE_SLIPPAGE = 1.01    # buy price +1% to ensure execution
SELL_PRICE_SLIPPAGE = 0.99   # sell price -1% to ensure execution
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trader")


# --- Utility Functions ---

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


# --- Signal Extraction (from pred_sh.pkl) ---

def extract_signals(pred_path: Path, signal_date: str | None = None) -> tuple[pd.DataFrame, str]:
    """Extract signals for a given date (default: latest) from pred_sh.pkl.

    Returns (DataFrame[code, score], signal_date_str)
    """
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)

    dates = sorted(pred.index.get_level_values("datetime").unique())

    if signal_date:
        target = pd.Timestamp(signal_date)
        if target not in dates:
            valid = [d for d in dates if d <= target]
            if not valid:
                raise ValueError(f"pred_sh.pkl has no data <= {signal_date}")
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
    # A-shares only
    df = df[df["code"].str.startswith("SH.")].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    date_str = target.strftime("%Y-%m-%d")
    log.info(f"Signal: {date_str}, A-shares {len(df)}, "
             f"Top-3 score: {df['score'].head(3).tolist()}")
    return df, date_str


# --- Signal Day Change Rates (from Qlib bin) ---

def load_signal_day_changes(signal_date: str, codes: list[str]) -> dict[str, float]:
    """Load signal day change rates from Qlib bin data."""
    try:
        from converter.incremental import QlibBinReader
        reader = QlibBinReader(QLIB_DATA_DIR)
    except ImportError:
        log.warning("QlibBinReader not available")
        return {}

    changes = {}
    for code in codes:
        s = reader.read_field(code, "change_rate")
        if signal_date in s.index:
            v = float(s[signal_date])
            if pd.notna(v):
                changes[code] = v
    return changes


# --- Trade Execution ---

def get_positions(trd_ctx) -> dict[str, dict]:
    ret, data = trd_ctx.position_list_query(trd_env=SAFE_TRD_ENV)
    if ret != RET_OK:
        log.error(f"Failed to query positions: {data}")
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
        log.error(f"Failed to query account: {data}")
        return {}
    row = data.iloc[0]
    info = {
        "total_assets": float(row.get("total_assets", 0)),
        "cash": float(row.get("cash", 0)),
        "market_val": float(row.get("market_val", 0)),
    }
    log.info(f"Account: total={info['total_assets']:,.0f}  "
             f"cash={info['cash']:,.0f}  market_val={info['market_val']:,.0f}")
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
            log.warning(f"Quote failed: {code}")
        time.sleep(0.3)
    return prices, changes


def run_trade(
    trd_ctx, quote_ctx,
    signals_df: pd.DataFrame,
    signal_day_changes: dict[str, float],
    dry_run: bool = False,
):
    """Execute portfolio rebalancing -- consistent with backtest.py logic."""

    account = get_account_info(trd_ctx)
    if not account:
        return
    positions = get_positions(trd_ctx)
    current_codes = set(positions.keys())

    # Hold inertia (backtest.py line 88-91)
    candidates = signals_df.copy()
    if HOLD_BONUS > 0 and current_codes:
        held = candidates["code"].isin(current_codes)
        candidates.loc[held, "score"] += HOLD_BONUS
        n = held.sum()
        if n > 0:
            log.info(f"Hold inertia: {n} stocks score += {HOLD_BONUS}")
        candidates = candidates.sort_values("score", ascending=False)

    codes_ordered = candidates["code"].tolist()

    # Get real-time quotes
    query_codes = list(set(codes_ordered[:TOP_N * 3]) | current_codes)
    prices, buy_day_chg = get_latest_prices(quote_ctx, query_codes)

    # Candidate filtering (backtest.py line 96-115)
    filtered = []
    for code in codes_ordered:
        if code not in prices:
            continue

        if code.startswith("SH."):
            limit_pct = _get_limit_up_pct(code)

            # Signal day limit-up filter (backtest.py line 108)
            chg_t = signal_day_changes.get(code, float("nan"))
            if pd.notna(chg_t) and chg_t >= limit_pct:
                log.warning(f"Limit-up filter (signal day): {code} {chg_t:.1f}%")
                continue

            # Buy day limit-up filter (backtest.py line 109)
            chg_t1 = buy_day_chg.get(code, 0)
            if chg_t1 >= limit_pct:
                log.warning(f"Limit-up filter (buy day): {code} {chg_t1:.1f}%")
                continue

        filtered.append(code)
        if len(filtered) >= TOP_N:
            break

    target_set = set(filtered)
    log.info(f"Target portfolio Top-{TOP_N}: {filtered}")

    # Stop-loss
    stop_loss_sells = set()
    for code, pos in positions.items():
        pl_ratio = pos.get("pl_ratio", 0) / 100.0
        if pl_ratio <= STOP_LOSS_PCT:
            log.warning(f"Stop-loss: {code} P/L {pl_ratio:.1%}")
            stop_loss_sells.add(code)
            target_set.discard(code)

    # Buy/sell calculation (backtest.py line 131-133)
    sells = (current_codes - target_set) | stop_loss_sells
    buys = target_set - current_codes
    holds = current_codes & target_set

    log.info(f"Current: {sorted(current_codes) or 'empty'}")
    log.info(f"Sell: {sorted(sells) or 'none'}  Buy: {sorted(buys) or 'none'}  "
             f"Hold: {sorted(holds) or 'none'}")

    if not sells and not buys:
        log.info("No rebalancing needed")
        return

    # Sell first (price discount to ensure execution)
    for code in sells:
        qty = positions[code]["qty"]
        price = prices.get(code)
        if not price:
            continue
        sell_price = round(price * SELL_PRICE_SLIPPAGE, 2)
        log.info(f"Sell {code}: {qty} shares @ {sell_price:.2f} (market {price:.2f})")
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

    # Then buy
    if not buys:
        return

    budget = account.get("cash", 0) * POSITION_RATIO / max(len(target_set), 1)
    log.info(f"Budget per stock: {budget:,.0f}")

    for code in buys:
        price = prices.get(code)
        if not price or price <= 0:
            continue
        buy_price = round(price * BUY_PRICE_SLIPPAGE, 2)
        qty = _round_lot(budget / buy_price)
        if qty <= 0:
            log.warning(f"Skip {code}: insufficient funds")
            continue
        log.info(f"Buy {code}: {qty} shares @ {buy_price:.2f} (market {price:.2f}) = {qty * buy_price:,.0f}")
        if not dry_run:
            ret, data = trd_ctx.place_order(
                price=buy_price, qty=qty, code=code,
                trd_side=TrdSide.BUY, order_type=OrderType.NORMAL,
                trd_env=SAFE_TRD_ENV,
            )
            log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
            time.sleep(1)


# --- Main ---

def main():
    dry_run = DRY_RUN or "--dry-run" in sys.argv
    signal_date = None
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            signal_date = arg.split("=")[1]

    if dry_run:
        log.info("=== DRY RUN ===")

    log.info(f"Config: TOP_N={TOP_N} HOLD_BONUS={HOLD_BONUS} "
             f"STOP_LOSS={STOP_LOSS_PCT} FUTU={FUTU_HOST}:{FUTU_PORT}")

    # 1. Extract signals
    if not PRED_PATH.exists():
        log.error(f"Prediction file not found: {PRED_PATH}")
        return
    signals_df, sig_date = extract_signals(PRED_PATH, signal_date)

    # 2. Load signal day change rates
    signal_changes = {}
    if QLIB_DATA_DIR.exists():
        sh_codes = signals_df["code"].tolist()
        signal_changes = load_signal_day_changes(sig_date, sh_codes[:50])
        n_limit = sum(1 for c in signal_changes if _is_limit_up(c, signal_changes[c]))
        log.info(f"Signal day changes: {len(signal_changes)} stocks, limit-up {n_limit}")
    else:
        log.warning(f"Qlib data unavailable: {QLIB_DATA_DIR}")

    # Save signal snapshot
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    sig_file = SIGNAL_DIR / f"signal_{sig_date}.csv"
    signals_df["signal_change_rate"] = signals_df["code"].map(signal_changes)
    signals_df.to_csv(sig_file, index=False)
    log.info(f"Signal snapshot: {sig_file}")

    # 3. Connect to futu
    log.info(f"Connecting to OpenD {FUTU_HOST}:{FUTU_PORT} ...")
    trd_ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.CN,
        host=FUTU_HOST, port=FUTU_PORT,
    )
    quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)

    try:
        ret, acc_list = trd_ctx.get_acc_list()
        if ret != RET_OK:
            log.error(f"Failed to get account list: {acc_list}")
            return

        sim = acc_list[acc_list["trd_env"] == "SIMULATE"]
        real = acc_list[acc_list["trd_env"] == "REAL"]
        if sim.empty:
            log.error("No simulation account found")
            return
        log.info(f"Simulation accounts: {sim['acc_id'].tolist()}")
        if not real.empty:
            log.warning(f"Real accounts {real['acc_id'].tolist()} -- not touching")
        assert SAFE_TRD_ENV == TrdEnv.SIMULATE

        # 4. Execute trade
        run_trade(trd_ctx, quote_ctx, signals_df, signal_changes, dry_run=dry_run)
        log.info("Done")

    finally:
        trd_ctx.close()
        quote_ctx.close()


if __name__ == "__main__":
    main()
