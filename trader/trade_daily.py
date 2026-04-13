"""
Mac mini 宿主机每日自动交易 — 严格匹配回测时序。

每个交易日 14:50 由宿主机 cron / `scripts/run_trade.sh` 触发:
1. 从 pred_sh.pkl 提取最近一个交易日的信号（信号日 t）
2. 从 Qlib bin 格式读取信号日涨跌幅
3. 查当前持仓，剔除失真快照后应用持仓惯性
4. 获取实时行情，双重涨停过滤（信号日 + 买入日）
5. Top-N 选股，止损检查
6. 先卖后买，等权分配

执行保护:
    - 仅使用模拟盘账户 (`FUTU_SIM_ACC_ID`)
    - 卖出前逐只复核实时持仓
    - 若 OpenD 判断沪深休市，则默认自动切换为 dry-run

数据源:
    pred_sh.pkl  → $DATA_DIR/signals/pred_sh_latest.pkl
    Qlib 数据    → $DATA_DIR/qlib_data
    futu OpenD   → $FUTU_HOST:$FUTU_PORT
"""

from __future__ import annotations

import bisect
import logging
import os
import pickle
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
    SysConfig,
)

# ─── 安全锁 ────────────────────────────────────────────
SAFE_TRD_ENV = TrdEnv.SIMULATE
assert SAFE_TRD_ENV == TrdEnv.SIMULATE, "禁止使用真实交易环境！"

# ─── 配置（环境变量覆盖）────────────────────────────────
FUTU_HOST = os.environ.get("FUTU_HOST", "futu-opend")
FUTU_PORT = int(os.environ.get("FUTU_PORT", "11111"))
FUTU_RSA_KEY = os.environ.get("FUTU_RSA_KEY", "")
FUTU_SIM_ACC_ID = int(os.environ.get("FUTU_SIM_ACC_ID", "0") or "0")
ALLOW_OFF_HOURS_TRADING = os.environ.get("ALLOW_OFF_HOURS_TRADING", "false").lower() == "true"

# ─── RSA 加密（跨网络交易需要）────────────────────────────
if FUTU_RSA_KEY and Path(FUTU_RSA_KEY).is_file():
    SysConfig.enable_proto_encrypt(True)
    SysConfig.set_init_rsa_file(FUTU_RSA_KEY)
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
DEFAULT_PRICE_TICK = 0.01
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
ALLOW_STALE_SIGNAL = os.environ.get("ALLOW_STALE_SIGNAL", "false").lower() == "true"
CN_TZ = ZoneInfo("Asia/Shanghai")
A_SHARE_TRADING_SESSIONS = (
    (dt_time(9, 30), dt_time(11, 30)),
    (dt_time(13, 0), dt_time(15, 0)),
)
A_SHARE_LIVE_MARKET_STATES = {"MORNING", "AFTERNOON", "AUCTION", "TRADE_AT_LAST", "TRADE_AUCTION"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trader")


# ─── 工具函数 ───────────────────────────────────────────

def _round_lot(qty: float, lot_size: int = MIN_LOT_SH) -> int:
    return max(int(qty // max(lot_size, 1)) * max(lot_size, 1), 0)


def _get_price_tick(code: str) -> float:
    return DEFAULT_PRICE_TICK


def _is_st_stock(snapshot: dict[str, object] | None = None) -> bool:
    if not snapshot:
        return False
    for key in ("name", "stock_name"):
        value = snapshot.get(key)
        if not value:
            continue
        normalized = str(value).strip().upper()
        if normalized.startswith(("ST", "*ST", "SST", "*SST")):
            return True
    return False


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num) or num <= 0:
        return None
    return num


def _snapshot_value(snapshot: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        if key in snapshot:
            value = _coerce_float(snapshot[key])
            if value is not None:
                return value
    return None


def _round_to_tick(price: float, tick: float, side: TrdSide) -> float:
    if tick <= 0:
        return round(price, 2)
    scaled = price / tick
    if side == TrdSide.SELL:
        adjusted = np.floor(scaled + 1e-9)
    else:
        adjusted = np.ceil(scaled - 1e-9)
    return round(float(adjusted * tick), 4)


def _extract_price_limits(snapshot: dict[str, object]) -> tuple[float | None, float | None]:
    lower = _snapshot_value(
        snapshot,
        "lower_limit_price",
        "lower_limit",
    )
    upper = _snapshot_value(
        snapshot,
        "upper_limit_price",
        "upper_limit",
    )
    return lower, upper


def _derive_price_limits(code: str, snapshot: dict[str, object]) -> tuple[float | None, float | None]:
    prev_close = _snapshot_value(snapshot, "prev_close_price", "prev_close")
    if prev_close is None:
        return None, None

    limit_pct = _get_limit_up_pct(code, snapshot=snapshot) / 100.0
    lower = round(prev_close * (1 - limit_pct), 2)
    upper = round(prev_close * (1 + limit_pct), 2)
    return lower, upper


def _resolve_price_limits(code: str, snapshot: dict[str, object]) -> tuple[float | None, float | None]:
    lower, upper = _extract_price_limits(snapshot)
    if lower is not None and upper is not None:
        return lower, upper
    return _derive_price_limits(code, snapshot)


def _clamp_price_to_limits(
    price: float,
    lower: float | None,
    upper: float | None,
    tick: float,
    side: TrdSide,
) -> float:
    clamped = price
    if lower is not None:
        clamped = max(clamped, lower)
    if upper is not None:
        clamped = min(clamped, upper)
    clamped = _round_to_tick(clamped, tick, side)
    if lower is not None and clamped < lower:
        clamped = _round_to_tick(lower, tick, TrdSide.BUY)
    if upper is not None and clamped > upper:
        clamped = _round_to_tick(upper, tick, TrdSide.SELL)
    return round(clamped, 2)


def build_order_price(
    code: str,
    side: TrdSide,
    snapshot: dict[str, object],
    slippage: float,
) -> float | None:
    reference_price = _snapshot_value(
        snapshot,
        "last_price",
        "nominal_price",
        "open_price",
        "prev_close_price",
    )
    if reference_price is None:
        return None

    tick = _get_price_tick(code)
    lower_limit, upper_limit = _resolve_price_limits(code, snapshot)
    if lower_limit is None or upper_limit is None:
        log.warning(f"{code}: snapshot 缺少合法价格区间，跳过真实下单")
        return None
    target = reference_price * slippage
    target = _round_to_tick(target, tick, side)
    return _clamp_price_to_limits(target, lower_limit, upper_limit, tick, side)


def _get_limit_up_pct(code: str, snapshot: dict[str, object] | None = None) -> float:
    if _is_st_stock(snapshot):
        return 5.0
    if code.startswith("SZ.300") or code.startswith("SH.688"):
        return 19.5
    return 9.5


def _order_adjust_limit(side: TrdSide) -> float:
    return -0.01 if side == TrdSide.BUY else 0.01


def _is_limit_up(code: str, change_rate: float) -> bool:
    if not code.startswith("SH."):
        return False
    return change_rate >= _get_limit_up_pct(code)


def _current_cn_datetime() -> datetime:
    return datetime.now(CN_TZ)


def is_a_share_trading_time(now: datetime | None = None) -> tuple[bool, str]:
    now = now or _current_cn_datetime()
    now = now.astimezone(CN_TZ)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")

    if now.weekday() >= 5:
        return False, f"{ts} 是周末"

    current_time = now.timetz().replace(tzinfo=None)
    for start, end in A_SHARE_TRADING_SESSIONS:
        if start <= current_time < end:
            return True, f"{ts} 在 A 股交易时段"

    return False, f"{ts} 不在 A 股交易时段(09:30-11:30, 13:00-15:00)"


def is_a_share_market_live(global_state: dict[str, object]) -> tuple[bool, str]:
    sh_state = str(global_state.get("market_sh", "N/A")).upper()
    sz_state = str(global_state.get("market_sz", "N/A")).upper()
    live = sh_state in A_SHARE_LIVE_MARKET_STATES or sz_state in A_SHARE_LIVE_MARKET_STATES
    reason = f"OpenD 市场状态: SH={sh_state}, SZ={sz_state}"
    return live, reason


def resolve_dry_run_mode(
    requested_dry_run: bool,
    now: datetime | None = None,
    global_state: dict[str, object] | None = None,
) -> tuple[bool, str | None]:
    if requested_dry_run:
        return True, None

    if global_state is not None:
        live_allowed, reason = is_a_share_market_live(global_state)
    else:
        live_allowed, reason = is_a_share_trading_time(now=now)
    if live_allowed:
        return False, None

    if ALLOW_OFF_HOURS_TRADING:
        return False, reason

    return True, reason


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


def _latest_a_share_date() -> str | None:
    inst_path = QLIB_DATA_DIR / "instruments" / "all.txt"
    if not inst_path.exists():
        return None

    latest = None
    for line in inst_path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        code, _, end_date = parts[:3]
        if not code.startswith(("SH.", "SZ.")):
            continue
        if latest is None or end_date > latest:
            latest = end_date
    return latest


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

def _positions_from_frame(data: pd.DataFrame) -> dict[str, dict]:
    positions = {}
    if data is None or data.empty:
        return positions

    for _, row in data.iterrows():
        qty_raw = row.get("qty", 0)
        qty = int(qty_raw) if pd.notna(qty_raw) else 0
        if qty <= 0:
            continue

        code = str(row["code"]).upper()
        can_sell_qty_raw = row.get("can_sell_qty", qty)
        can_sell_qty = int(can_sell_qty_raw) if pd.notna(can_sell_qty_raw) else qty

        positions[code] = {
            "qty": qty,
            "can_sell_qty": can_sell_qty,
            "market_val": float(row.get("market_val", 0) or 0),
            "cost_price": float(row.get("cost_price", 0) or 0),
            "pl_ratio": float(row.get("pl_ratio", 0) or 0),
        }

    return positions


def get_positions(trd_ctx, acc_id: int, code: str = "", refresh_cache: bool = False) -> dict[str, dict]:
    ret, data = trd_ctx.position_list_query(
        code=code,
        trd_env=SAFE_TRD_ENV,
        acc_id=acc_id,
        refresh_cache=refresh_cache,
    )
    if ret != RET_OK:
        log.error(f"查询持仓失败: {data}")
        return {}
    return _positions_from_frame(data)


def get_account_info(trd_ctx, acc_id: int, refresh_cache: bool = False) -> dict:
    ret, data = trd_ctx.accinfo_query(
        trd_env=SAFE_TRD_ENV,
        acc_id=acc_id,
        refresh_cache=refresh_cache,
    )
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


def get_market_snapshots(quote_ctx, codes: list[str]) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    if not codes:
        return snapshots

    batch_size = 200
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ret, data = quote_ctx.get_market_snapshot(batch)
        if ret != RET_OK or data is None or data.empty:
            log.warning(f"行情失败: batch={batch} | ret={ret} | data={data}")
            time.sleep(0.3)
            continue

        for _, row in data.iterrows():
            code = str(row.get("code", "")).upper()
            if not code:
                continue
            snapshots[code] = row.to_dict()
        time.sleep(0.3)

    if len(snapshots) < len(codes):
        log.warning(f"行情汇总: {len(snapshots)}/{len(codes)} 成功")
    return snapshots


def run_trade(
    trd_ctx, quote_ctx,
    acc_id: int,
    signals_df: pd.DataFrame,
    signal_day_changes: dict[str, float],
    dry_run: bool = False,
):
    """执行换仓 — 与 backtest.py 完全一致。"""

    account = get_account_info(trd_ctx, acc_id=acc_id, refresh_cache=True)
    if not account:
        return
    positions = get_positions(trd_ctx, acc_id=acc_id, refresh_cache=True)
    current_codes = set(positions.keys())

    candidates = signals_df.copy()
    if HOLD_BONUS > 0 and current_codes:
        held = candidates["code"].isin(current_codes)
        candidates.loc[held, "score"] += HOLD_BONUS
        n = held.sum()
        if n > 0:
            log.info(f"持仓惯性: {n} 只 score += {HOLD_BONUS}")
        candidates = candidates.sort_values("score", ascending=False)

    codes_ordered = candidates["code"].tolist()
    query_codes = list(set(codes_ordered[:TOP_N * 3]) | current_codes)
    snapshots = get_market_snapshots(quote_ctx, query_codes)

    eligible = []
    for code in codes_ordered:
        snapshot = snapshots.get(code)
        if not snapshot:
            continue

        if code.startswith("SH."):
            limit_pct = _get_limit_up_pct(code, snapshot=snapshot)
            chg_t = signal_day_changes.get(code, float("nan"))
            if pd.notna(chg_t) and chg_t >= limit_pct:
                log.warning(f"涨停过滤(信号日): {code} {chg_t:.1f}%")
                continue

            chg_t1 = _snapshot_value(snapshot, "change_rate") or 0
            if chg_t1 >= limit_pct:
                log.warning(f"涨停过滤(买入日): {code} {chg_t1:.1f}%")
                continue

        eligible.append(code)

    target_set = set(eligible[:TOP_N])
    log.info(f"目标持仓 Top-{TOP_N}: {eligible[:TOP_N]}")

    if not target_set and current_codes and len(snapshots) < len(query_codes) * 0.5:
        log.warning(f"行情异常保护: 仅 {len(snapshots)}/{len(query_codes)} 行情成功，保持现有持仓不动")
        return

    stop_loss_sells = set()
    for code, pos in positions.items():
        pl_ratio = pos.get("pl_ratio", 0) / 100.0
        if pl_ratio <= STOP_LOSS_PCT:
            log.warning(f"止损: {code} 浮亏 {pl_ratio:.1%}")
            stop_loss_sells.add(code)
            target_set.discard(code)

    sells = (current_codes - target_set) | stop_loss_sells
    planned_buys = target_set - current_codes
    holds = current_codes & target_set

    log.info(f"当前: {sorted(current_codes) or '空仓'}")
    log.info(f"卖出: {sorted(sells) or '无'}  买入: {sorted(planned_buys) or '无'}  持有: {sorted(holds) or '无'}")

    if not sells and not planned_buys:
        log.info("无需调仓")
        return

    sell_failures = []
    sell_results = []
    for code in sorted(sells):
        pos = positions.get(code)
        if pos is None:
            log.warning(f"卖出跳过 {code}: 当前账户无持仓")
            sell_results.append((code, "missing_position"))
            continue

        qty = pos.get("can_sell_qty", pos["qty"])
        if qty <= 0:
            log.warning(f"卖出跳过 {code}: 当前账户无可卖仓位")
            sell_failures.append(code)
            sell_results.append((code, "no_sell_qty"))
            continue

        snapshot = snapshots.get(code)
        if not snapshot:
            log.warning(f"卖出跳过 {code}: 行情缺失")
            sell_failures.append(code)
            sell_results.append((code, "missing_snapshot"))
            continue

        sell_price = build_order_price(code, TrdSide.SELL, snapshot, SELL_PRICE_SLIPPAGE)
        market_price = _snapshot_value(snapshot, "last_price")
        if sell_price is None or market_price is None:
            log.warning(f"卖出跳过 {code}: 无法生成合法卖价")
            sell_failures.append(code)
            sell_results.append((code, "invalid_sell_price"))
            continue

        log.info(f"卖出 {code}: {qty}股 @ {sell_price:.2f} (市价{market_price:.2f})")
        if dry_run:
            sell_results.append((code, "dry_run"))
            continue

        ret, data = trd_ctx.place_order(
            price=sell_price, qty=qty, code=code,
            trd_side=TrdSide.SELL, order_type=OrderType.NORMAL,
            adjust_limit=_order_adjust_limit(TrdSide.SELL),
            trd_env=SAFE_TRD_ENV,
            acc_id=acc_id,
        )
        log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
        if ret != RET_OK:
            sell_failures.append(code)
            sell_results.append((code, "failed"))
        else:
            sell_results.append((code, "ok"))
        time.sleep(1)

    if sells:
        log.info(f"卖出结果汇总: {sell_results}")

    attempted_live_sells = any(status in {"ok", "failed"} for _, status in sell_results)
    if attempted_live_sells and not dry_run:
        time.sleep(3)
        account = get_account_info(trd_ctx, acc_id=acc_id, refresh_cache=True)
        positions = get_positions(trd_ctx, acc_id=acc_id, refresh_cache=True)
        current_codes = set(positions.keys())
        log.info(f"卖后刷新持仓: {sorted(current_codes) or '空仓'}")

    if sell_failures:
        log.warning(f"卖出失败但继续保守买入: {sorted(set(sell_failures))}")

    available_slots = max(TOP_N - len(current_codes), 0)
    buys = [
        code for code in eligible
        if code not in current_codes and code not in stop_loss_sells
    ][:available_slots]
    if not buys:
        return

    budget = account.get("cash", 0) * POSITION_RATIO / max(len(target_set), 1)
    log.info(f"每只预算: {budget:,.0f}")

    for code in buys:
        snapshot = snapshots.get(code)
        if not snapshot:
            continue
        buy_price = build_order_price(code, TrdSide.BUY, snapshot, BUY_PRICE_SLIPPAGE)
        market_price = _snapshot_value(snapshot, "last_price")
        if buy_price is None or market_price is None:
            log.warning(f"跳过 {code}: 无法生成合法买价")
            continue
        lot_size = int(_snapshot_value(snapshot, "lot_size") or MIN_LOT_SH)
        qty = _round_lot(budget / buy_price, lot_size=lot_size)
        if qty <= 0:
            log.warning(f"跳过 {code}: 资金不足")
            continue
        log.info(f"买入 {code}: {qty}股 @ {buy_price:.2f} (市价{market_price:.2f}) = {qty * buy_price:,.0f}")
        if not dry_run:
            ret, data = trd_ctx.place_order(
                price=buy_price, qty=qty, code=code,
                trd_side=TrdSide.BUY, order_type=OrderType.NORMAL,
                adjust_limit=_order_adjust_limit(TrdSide.BUY),
                trd_env=SAFE_TRD_ENV,
                acc_id=acc_id,
            )
            log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
            time.sleep(1)


def select_sim_acc_id(acc_list: pd.DataFrame, preferred_acc_id: int = 0) -> int:
    sim = acc_list[acc_list["trd_env"] == "SIMULATE"]
    if sim.empty:
        raise ValueError("无模拟账户")

    sim_ids = [int(acc_id) for acc_id in sim["acc_id"].tolist()]
    if preferred_acc_id:
        if preferred_acc_id not in sim_ids:
            raise ValueError(
                f"指定模拟账户不存在: preferred={preferred_acc_id}, available={sim_ids}"
            )
        return preferred_acc_id
    return sim_ids[0]


# ─── 主流程 ──────────────────────────────────────────

def main():
    requested_dry_run = DRY_RUN or "--dry-run" in sys.argv
    dry_run, dry_run_reason = resolve_dry_run_mode(requested_dry_run)
    signal_date = None
    for arg in sys.argv[1:]:
        if arg.startswith("--date="):
            signal_date = arg.split("=")[1]

    log.info(f"配置: TOP_N={TOP_N} HOLD_BONUS={HOLD_BONUS} "
             f"STOP_LOSS={STOP_LOSS_PCT} FUTU={FUTU_HOST}:{FUTU_PORT}")

    # 1. 提取信号
    if not PRED_PATH.exists():
        log.error(f"预测文件不存在: {PRED_PATH}")
        return
    signals_df, sig_date = extract_signals(PRED_PATH, signal_date)
    latest_a_share_date = _latest_a_share_date()
    if latest_a_share_date and sig_date != latest_a_share_date:
        level = log.warning if ALLOW_STALE_SIGNAL else log.error
        level(
            f"信号日期与本地 A 股最新数据不一致: signal={sig_date}, latest_a_share={latest_a_share_date}"
        )
        if not ALLOW_STALE_SIGNAL:
            return

    # 信号新鲜度检查: 信号日期不应超过 3 个交易日前
    from datetime import datetime as _dt
    sig_age = (_dt.now() - _dt.strptime(sig_date, "%Y-%m-%d")).days
    if sig_age > 5:
        log.warning(f"信号过旧: {sig_date} ({sig_age} 天前)，可能推理管线异常")
    elif sig_age > 3:
        log.warning(f"信号较旧: {sig_date} ({sig_age} 天前)")

    # 2. 读取信号日涨跌幅
    signal_changes = {}
    if QLIB_DATA_DIR.exists():
        sh_codes = signals_df["code"].tolist()
        signal_changes = load_signal_day_changes(sig_date, sh_codes[:50])
        n_limit = sum(1 for c in signal_changes if _is_limit_up(c, signal_changes[c]))
        log.info(f"信号日涨跌幅: {len(signal_changes)} 只, 涨停 {n_limit} 只")
    else:
        log.warning(f"Qlib 数据不可用: {QLIB_DATA_DIR}")

    # 保存信号快照
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    sig_file = SIGNAL_DIR / f"signal_{sig_date}.csv"
    signals_df["signal_change_rate"] = signals_df["code"].map(signal_changes)
    signals_df.to_csv(sig_file, index=False)
    log.info(f"信号快照: {sig_file}")

    # 3. 连接 futu
    log.info(f"连接 OpenD {FUTU_HOST}:{FUTU_PORT} ...")

    # 设置连接超时 (默认 Futu SDK 无限重试)
    import signal as _signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"OpenD 连接超时 ({FUTU_HOST}:{FUTU_PORT})")

    _signal.signal(_signal.SIGALRM, _timeout_handler)
    _signal.alarm(30)
    try:
        trd_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.CN,
            host=FUTU_HOST, port=FUTU_PORT,
        )
        quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
    except TimeoutError as e:
        log.error(str(e))
        return
    finally:
        _signal.alarm(0)

    try:
        ret, global_state = quote_ctx.get_global_state()
        if ret == RET_OK:
            dry_run, dry_run_reason = resolve_dry_run_mode(
                requested_dry_run,
                global_state=global_state,
            )
        elif not requested_dry_run:
            log.warning(f"获取 OpenD 市场状态失败，回退到本地时段判断: {global_state}")

        if requested_dry_run:
            log.info("=== DRY RUN ===")
        elif dry_run:
            log.warning(f"非交易时段，强制切换为 DRY RUN: {dry_run_reason}")
            log.info("=== AUTO DRY RUN ===")
        elif dry_run_reason:
            log.warning(f"非交易时段，但 ALLOW_OFF_HOURS_TRADING=true，继续运行: {dry_run_reason}")

        ret, acc_list = trd_ctx.get_acc_list()
        if ret != RET_OK:
            log.error(f"账户列表失败: {acc_list}")
            return

        sim = acc_list[acc_list["trd_env"] == "SIMULATE"]
        real = acc_list[acc_list["trd_env"] == "REAL"]
        try:
            sim_acc_id = select_sim_acc_id(acc_list, preferred_acc_id=FUTU_SIM_ACC_ID)
        except ValueError as exc:
            log.error(str(exc))
            return

        log.info(f"模拟账户: {sim['acc_id'].tolist()}")
        log.info(f"使用模拟账户: {sim_acc_id}")
        if not real.empty:
            log.warning(f"真实账户 {real['acc_id'].tolist()} — 不触碰")
        assert SAFE_TRD_ENV == TrdEnv.SIMULATE

        # 4. 执行交易
        run_trade(
            trd_ctx,
            quote_ctx,
            sim_acc_id,
            signals_df,
            signal_changes,
            dry_run=dry_run,
        )
        log.info("完成")

    finally:
        trd_ctx.close()
        quote_ctx.close()


if __name__ == "__main__":
    main()
