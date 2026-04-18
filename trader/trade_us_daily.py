from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from futu import OpenQuoteContext, OpenSecTradeContext, TrdEnv, TrdMarket, RET_OK, SysConfig

from trader import trade_daily as base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trader_us")

SAFE_TRD_ENV = TrdEnv.SIMULATE
assert SAFE_TRD_ENV == TrdEnv.SIMULATE, "禁止使用真实交易环境！"

FUTU_HOST = os.environ.get("FUTU_HOST", base.FUTU_HOST)
FUTU_PORT = int(os.environ.get("FUTU_PORT", str(base.FUTU_PORT)))
FUTU_SIM_ACC_ID = int(os.environ.get("FUTU_SIM_ACC_ID", str(base.FUTU_SIM_ACC_ID)) or "0")
FUTU_RSA_KEY = os.environ.get("FUTU_RSA_KEY", "")
ALLOW_OFF_HOURS_TRADING = os.environ.get("ALLOW_OFF_HOURS_TRADING", "false").lower() == "true"
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"
POSITION_RATIO = float(os.environ.get("POSITION_RATIO", "0.95"))
FUTU_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("US_FUTU_CONNECT_TIMEOUT_SECONDS", "30"))
US_SIGNAL_DIR = Path(
    os.environ.get(
        "US_SIGNAL_DIR",
        str(Path(os.environ.get("SIGNAL_DIR", str(Path.home() / "quantpilot_data" / "signals"))) / "us"),
    )
)
US_TRADE_PLAN_PATH = Path(
    os.environ.get("US_TRADE_PLAN_PATH", str(US_SIGNAL_DIR / "us_trade_plan_latest.json"))
)
US_TZ = ZoneInfo(os.environ.get("US_MARKET_TIMEZONE", "America/New_York"))
US_REGULAR_SESSIONS = ((dt_time(9, 30), dt_time(16, 0)),)
US_LIVE_MARKET_STATES = {"AFTERNOON", "MORNING", "OPEN", "REST"}


def is_us_trading_time(now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now(US_TZ)
    now = now.astimezone(US_TZ)
    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    if now.weekday() >= 5:
        return False, f"{ts} 是周末"
    current_time = now.timetz().replace(tzinfo=None)
    for start, end in US_REGULAR_SESSIONS:
        if start <= current_time < end:
            return True, f"{ts} 在美股常规交易时段"
    return False, f"{ts} 不在美股常规交易时段(09:30-16:00 ET)"


def is_us_market_live(global_state: dict[str, object]) -> tuple[bool, str]:
    us_state = str(global_state.get("market_us", "N/A")).upper()
    live = us_state in US_LIVE_MARKET_STATES
    return live, f"OpenD 市场状态: US={us_state}"


def resolve_dry_run_mode(
    requested_dry_run: bool,
    now: datetime | None = None,
    global_state: dict[str, object] | None = None,
) -> tuple[bool, str | None]:
    if requested_dry_run:
        return True, None
    if global_state is not None:
        live_allowed, reason = is_us_market_live(global_state)
    else:
        live_allowed, reason = is_us_trading_time(now=now)
    if live_allowed:
        return False, None
    if ALLOW_OFF_HOURS_TRADING:
        return False, reason
    return True, reason


def load_trade_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("trade plan must be an object")
    orders = payload.get("orders")
    if not isinstance(orders, list) or not orders:
        raise ValueError("trade plan orders must be a non-empty list")
    return payload


def build_signals_df(plan: dict) -> tuple[base.pd.DataFrame, str]:
    rows = []
    for order in plan["orders"]:
        code = str(order.get("code", "")).upper()
        action = str(order.get("action", "HOLD")).upper()
        if not code or action not in {"BUY", "HOLD"}:
            continue
        weight = float(order.get("target_weight", 0.0) or 0.0)
        rows.append(
            {
                "code": code,
                "score": weight,
                "target_weight": weight,
                "action": action,
            }
        )
    if rows:
        df = base.pd.DataFrame(rows).sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)
    else:
        df = base.pd.DataFrame(columns=["code", "score", "target_weight", "action"])
    generated_at = str(plan.get("generated_at", ""))
    signal_date = generated_at[:10] if len(generated_at) >= 10 else datetime.now().strftime("%Y-%m-%d")
    return df, signal_date


def run_trade_us(trd_ctx, quote_ctx, acc_id: int, signals_df: base.pd.DataFrame, dry_run: bool = False) -> None:
    account = base.get_account_info(trd_ctx, acc_id=acc_id, refresh_cache=True)
    if not account:
        return
    positions = base.get_positions(trd_ctx, acc_id=acc_id, refresh_cache=True)
    current_codes = set(positions.keys())
    target_codes = set(signals_df["code"].tolist())
    query_codes = sorted(current_codes | target_codes)
    snapshots = base.get_market_snapshots(quote_ctx, query_codes)

    sells = sorted(current_codes - target_codes)
    buys = [code for code in signals_df["code"].tolist() if code not in current_codes]
    holds = sorted(current_codes & target_codes)
    log.info(f"当前: {sorted(current_codes) or '空仓'}")
    log.info(f"卖出: {sells or '无'}  买入: {buys or '无'}  持有: {holds or '无'}")

    sell_results = []
    for code in sells:
        pos = positions.get(code)
        if pos is None:
            continue
        qty = pos.get("can_sell_qty", pos["qty"])
        if qty <= 0:
            log.warning(f"卖出跳过 {code}: 当前账户无可卖仓位")
            continue
        snapshot = snapshots.get(code)
        if not snapshot:
            log.warning(f"卖出跳过 {code}: 行情缺失")
            continue
        sell_price = base.build_order_price(code, base.TrdSide.SELL, snapshot, base.SELL_PRICE_SLIPPAGE)
        market_price = base._snapshot_value(snapshot, "last_price")
        if sell_price is None or market_price is None:
            log.warning(f"卖出跳过 {code}: 无法生成合法卖价")
            continue
        log.info(f"卖出 {code}: {qty}股 @ {sell_price:.2f} (市价{market_price:.2f})")
        if dry_run:
            sell_results.append((code, "dry_run"))
            continue
        ret, data = trd_ctx.place_order(
            price=sell_price,
            qty=qty,
            code=code,
            trd_side=base.TrdSide.SELL,
            order_type=base.OrderType.NORMAL,
            adjust_limit=base._order_adjust_limit(base.TrdSide.SELL),
            trd_env=SAFE_TRD_ENV,
            acc_id=acc_id,
        )
        log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
        sell_results.append((code, "ok" if ret == RET_OK else "failed"))
        base.time.sleep(1)

    attempted_live_sells = any(status in {"ok", "failed"} for _, status in sell_results)
    if attempted_live_sells and not dry_run:
        base.time.sleep(3)
        account = base.get_account_info(trd_ctx, acc_id=acc_id, refresh_cache=True)
        positions = base.get_positions(trd_ctx, acc_id=acc_id, refresh_cache=True)
        current_codes = set(positions.keys())
        log.info(f"卖后刷新持仓: {sorted(current_codes) or '空仓'}")

    target_weights = {str(row.code): float(row.target_weight) for row in signals_df.itertuples(index=False)}
    total_assets = float(account.get("total_assets", 0))
    cash = float(account.get("cash", 0))
    for code in signals_df["code"].tolist():
        if code in current_codes:
            continue
        snapshot = snapshots.get(code)
        if not snapshot:
            log.warning(f"买入跳过 {code}: 行情缺失")
            continue
        buy_price = base.build_order_price(code, base.TrdSide.BUY, snapshot, base.BUY_PRICE_SLIPPAGE)
        market_price = base._snapshot_value(snapshot, "last_price")
        if buy_price is None or market_price is None:
            log.warning(f"买入跳过 {code}: 无法生成合法买价")
            continue
        target_cash = total_assets * POSITION_RATIO * target_weights.get(code, 0.0)
        budget = min(target_cash, cash)
        lot_size = int(base._snapshot_value(snapshot, "lot_size") or 1)
        qty = base._round_lot(budget / buy_price, lot_size=lot_size)
        if qty <= 0:
            log.warning(f"买入跳过 {code}: 资金不足")
            continue
        log.info(f"买入 {code}: {qty}股 @ {buy_price:.2f} (市价{market_price:.2f}) = {qty * buy_price:,.0f}")
        if dry_run:
            continue
        ret, data = trd_ctx.place_order(
            price=buy_price,
            qty=qty,
            code=code,
            trd_side=base.TrdSide.BUY,
            order_type=base.OrderType.NORMAL,
            adjust_limit=base._order_adjust_limit(base.TrdSide.BUY),
            trd_env=SAFE_TRD_ENV,
            acc_id=acc_id,
        )
        log.info(f"  {'OK' if ret == RET_OK else 'FAIL'} {data}")
        if ret == RET_OK:
            cash = max(cash - qty * buy_price, 0.0)
        base.time.sleep(1)


def main() -> None:
    requested_dry_run = DRY_RUN or "--dry-run" in sys.argv
    dry_run, dry_run_reason = resolve_dry_run_mode(requested_dry_run)
    if not US_TRADE_PLAN_PATH.exists():
        log.error(f"US trade plan not found: {US_TRADE_PLAN_PATH}")
        sys.exit(1)

    plan = load_trade_plan(US_TRADE_PLAN_PATH)
    signals_df, signal_date = build_signals_df(plan)
    log.info(f"配置: FUTU={FUTU_HOST}:{FUTU_PORT} plan={US_TRADE_PLAN_PATH} signal_date={signal_date}")
    log.info(f"US target codes: {signals_df['code'].tolist()}")
    log.info(f"连接 OpenD {FUTU_HOST}:{FUTU_PORT} ...")

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"OpenD 连接超时 ({FUTU_HOST}:{FUTU_PORT})")

    if FUTU_RSA_KEY and Path(FUTU_RSA_KEY).is_file():
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(FUTU_RSA_KEY)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FUTU_CONNECT_TIMEOUT_SECONDS)
    try:
        trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=FUTU_HOST, port=FUTU_PORT)
        quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
    finally:
        signal.alarm(0)

    try:
        ret, global_state = quote_ctx.get_global_state()
        if ret == RET_OK:
            dry_run, dry_run_reason = resolve_dry_run_mode(requested_dry_run, global_state=global_state)
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
            sys.exit(1)
        sim_acc_id = base.select_sim_acc_id(acc_list, preferred_acc_id=FUTU_SIM_ACC_ID)
        sim = acc_list[acc_list["trd_env"] == "SIMULATE"]
        real = acc_list[acc_list["trd_env"] == "REAL"]
        log.info(f"模拟账户: {sim['acc_id'].tolist()}")
        log.info(f"使用模拟账户: {sim_acc_id}")
        if not real.empty:
            log.warning(f"真实账户 {real['acc_id'].tolist()} — 不触碰")
        assert SAFE_TRD_ENV == TrdEnv.SIMULATE

        run_trade_us(trd_ctx, quote_ctx, sim_acc_id, signals_df, dry_run=dry_run)
        log.info("完成")
    finally:
        trd_ctx.close()
        quote_ctx.close()


if __name__ == "__main__":
    main()
