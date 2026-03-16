"""
模拟交易 (Paper Trading / Dry Run) 模块。

基于 TopkDropout 策略，每日：
1. 加载已训练模型，预测所有股票下一日得分
2. 根据得分排名决定买卖（持仓 topk 只，每日最多换 n_drop 只）
3. 以最近收盘价模拟成交，更新虚拟组合
4. 持久化组合状态到 JSON 文件

用法：
  python main.py dryrun          # 执行一次模拟交易
  python main.py dryrun-status   # 查看组合状态
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_STATE_DIR = _project_root() / "dryrun"
DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/my_quant_data"
HK_PROVIDER_URI = "~/.qlib/qlib_data/hk_quant_data"


class PaperTrader:
    """TopkDropout 模拟交易器。"""

    def __init__(
        self,
        state_dir: Path | str | None = None,
        topk: int = 10,
        n_drop: int = 2,
        initial_cash: float = 1_000_000.0,
        provider_uri: str | None = None,
        hk_mode: bool = False,
    ) -> None:
        self.state_dir = Path(state_dir or DEFAULT_STATE_DIR)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "portfolio_state.json"

        self.hk_mode = hk_mode
        self.provider_uri = provider_uri or (HK_PROVIDER_URI if hk_mode else DEFAULT_PROVIDER_URI)

        # 初始化 Qlib
        import qlib
        from qlib.constant import REG_CN
        uri = str(Path(self.provider_uri).expanduser().resolve())
        qlib.init(provider_uri=uri, region=REG_CN)

        # 加载或初始化状态
        self.state = self._load_state()
        if not self.state.get("created_at"):
            # 首次运行，初始化状态
            self.state = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "initial_cash": initial_cash,
                "cash": initial_cash,
                "positions": {},
                "config": {
                    "topk": topk,
                    "n_drop": n_drop,
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5.0,
                },
                "trade_history": [],
                "daily_snapshots": [],
            }
            self._save_state()
        else:
            # 已有状态，使用保存的配置
            topk = self.state["config"]["topk"]
            n_drop = self.state["config"]["n_drop"]

        self.topk = topk
        self.n_drop = n_drop

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _get_latest_trade_date(self) -> str:
        """获取 Qlib 日历中最近的交易日。"""
        from qlib.data import D
        from datetime import timedelta
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        cal = D.calendar(start_time="2020-01-01", end_time=end, freq="day")
        if cal is None or len(cal) == 0:
            raise RuntimeError("Qlib 日历为空，请先运行数据转换")
        return pd.Timestamp(cal[-1]).strftime("%Y-%m-%d")

    def _get_prices(self, codes: list[str], date: str) -> dict[str, float]:
        """从 Qlib 获取指定日期的收盘价。"""
        from qlib.data import D
        if not codes:
            return {}
        try:
            df = D.features(codes, ["$close"], start_time=date, end_time=date, freq="day")
        except Exception:
            return {}
        if df is None or df.empty:
            return {}
        prices = {}
        for code in codes:
            try:
                val = df.xs(code, level="instrument")["$close"].iloc[0]
                if pd.notna(val) and np.isfinite(val):
                    prices[code] = float(val)
            except (KeyError, IndexError):
                continue
        return prices

    def _get_signals(self) -> pd.DataFrame:
        """运行模型预测，返回 DataFrame[code, score, rank]。"""
        from strategy.engine import StrategyEngine
        engine = StrategyEngine(provider_uri=self.provider_uri)
        df = engine.predict_next_day(hk_mode=self.hk_mode)
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "score", "rank"])
        return df

    def _topk_dropout_decide(
        self, signals: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """
        TopkDropout 换仓决策。

        返回 (sell_codes, buy_codes)。
        """
        if signals.empty:
            return [], []

        scores = signals.set_index("code")["score"]
        current_holdings = set(self.state["positions"].keys())

        # 当前持仓在新信号中的得分
        held_scores = scores.reindex(list(current_holdings)).dropna()
        n_held = len(held_scores)

        if n_held == 0 and len(current_holdings) > 0:
            # 当前持仓全部不在信号列表中（数据问题），不操作
            return [], []

        # 非持仓股票中得分最高的候选
        candidates = scores[~scores.index.isin(current_holdings)].sort_values(ascending=False)

        # 需要多少个新买入名额
        slots_to_fill = self.topk - n_held  # 如果持仓不足 topk
        n_new = min(self.n_drop + max(slots_to_fill, 0), len(candidates))
        new_codes = candidates.head(n_new).index.tolist()

        # 合并持仓 + 新候选，统一排名
        combined_codes = list(held_scores.index) + new_codes
        combined_scores = scores.reindex(combined_codes).dropna().sort_values(ascending=False)

        # 卖出：当前持仓中排在 combined 底部的 n_drop 只
        bottom = combined_scores.tail(self.n_drop).index.tolist()
        sell_codes = [c for c in bottom if c in current_holdings]

        # 买入：新候选中排在 combined 顶部的，数量 = 卖出数 + 空缺
        free_slots = len(sell_codes) + max(self.topk - n_held, 0)
        buy_codes = [c for c in combined_scores.index if c not in current_holdings][:free_slots]

        return sell_codes, buy_codes

    def _calc_commission(self, amount: float, is_buy: bool) -> float:
        """计算交易佣金。"""
        cfg = self.state["config"]
        rate = cfg["open_cost"] if is_buy else cfg["close_cost"]
        return max(amount * rate, cfg["min_cost"])

    def execute_daily(self) -> dict:
        """
        执行一次每日模拟交易。

        Returns:
            交易结果摘要 dict。
        """
        trade_date = self._get_latest_trade_date()

        # 幂等检查：今天是否已经跑过
        for snap in self.state.get("daily_snapshots", []):
            if snap["date"] == trade_date:
                print(f"[跳过] {trade_date} 已执行过模拟交易")
                return {"date": trade_date, "skipped": True, **snap}

        print(f"交易日: {trade_date}")
        print(f"当前持仓: {len(self.state['positions'])} 只, 现金: {self.state['cash']:,.2f}")

        # 1. 获取信号
        print("正在运行模型预测...")
        signals = self._get_signals()
        if signals.empty:
            print("[警告] 模型预测结果为空，跳过本日交易")
            return {"date": trade_date, "skipped": True, "reason": "empty_signals"}

        print(f"预测信号: {len(signals)} 只股票")

        # 2. 决定买卖
        sell_codes, buy_codes = self._topk_dropout_decide(signals)

        # 3. 获取价格
        all_codes = list(set(
            list(self.state["positions"].keys()) + sell_codes + buy_codes
        ))
        prices = self._get_prices(all_codes, trade_date)

        # 过滤掉没有价格的标的
        sell_codes = [c for c in sell_codes if c in prices]
        buy_codes = [c for c in buy_codes if c in prices]

        trades = []

        # 4. 执行卖出
        for code in sell_codes:
            pos = self.state["positions"].get(code)
            if not pos:
                continue
            price = prices[code]
            shares = pos["shares"]
            amount = shares * price
            commission = self._calc_commission(amount, is_buy=False)
            proceeds = amount - commission
            self.state["cash"] += proceeds
            del self.state["positions"][code]

            score = signals.set_index("code")["score"].get(code, 0)
            rank = int(signals.set_index("code")["rank"].get(code, 0))
            trade = {
                "date": trade_date,
                "action": "SELL",
                "code": code,
                "shares": shares,
                "price": price,
                "amount": amount,
                "commission": round(commission, 2),
                "reason": f"排名跌至 #{rank} (score={score:.4f})",
            }
            trades.append(trade)
            self.state["trade_history"].append(trade)
            print(f"  SELL {code}  {shares} 股 @ {price:.2f}  金额: {amount:,.2f}")

        # 5. 执行买入 — 等额分配现金
        if buy_codes:
            available_cash = self.state["cash"]
            per_stock = available_cash / len(buy_codes) * 0.95  # 留 5% 余量

            for code in buy_codes:
                price = prices[code]
                if price <= 0:
                    continue
                shares = int(per_stock / price)
                if shares <= 0:
                    continue
                amount = shares * price
                commission = self._calc_commission(amount, is_buy=True)
                total_cost = amount + commission

                if total_cost > self.state["cash"]:
                    shares = int((self.state["cash"] - self.state["config"]["min_cost"]) / price)
                    if shares <= 0:
                        continue
                    amount = shares * price
                    commission = self._calc_commission(amount, is_buy=True)
                    total_cost = amount + commission

                self.state["cash"] -= total_cost
                self.state["positions"][code] = {
                    "shares": shares,
                    "avg_cost": price,
                    "entry_date": trade_date,
                }

                score = signals.set_index("code")["score"].get(code, 0)
                rank = int(signals.set_index("code")["rank"].get(code, 0))
                trade = {
                    "date": trade_date,
                    "action": "BUY",
                    "code": code,
                    "shares": shares,
                    "price": price,
                    "amount": amount,
                    "commission": round(commission, 2),
                    "reason": f"排名 #{rank} (score={score:.4f})",
                }
                trades.append(trade)
                self.state["trade_history"].append(trade)
                print(f"  BUY  {code}  {shares} 股 @ {price:.2f}  金额: {amount:,.2f}")

        # 6. 计算组合快照
        position_value = 0.0
        positions_detail = {}
        for code, pos in self.state["positions"].items():
            p = prices.get(code, pos["avg_cost"])
            val = pos["shares"] * p
            position_value += val
            positions_detail[code] = round(val, 2)

        total_value = self.state["cash"] + position_value
        initial = self.state["initial_cash"]
        cum_return = (total_value - initial) / initial

        prev_total = initial
        if self.state["daily_snapshots"]:
            prev_total = self.state["daily_snapshots"][-1]["total_value"]
        daily_return = (total_value - prev_total) / prev_total if prev_total > 0 else 0.0

        snapshot = {
            "date": trade_date,
            "total_value": round(total_value, 2),
            "cash": round(self.state["cash"], 2),
            "position_value": round(position_value, 2),
            "daily_return": round(daily_return, 6),
            "cumulative_return": round(cum_return, 6),
            "n_positions": len(self.state["positions"]),
            "n_sells": len(sell_codes),
            "n_buys": len(buy_codes),
        }
        self.state["daily_snapshots"].append(snapshot)
        self._save_state()

        return {
            "date": trade_date,
            "skipped": False,
            "trades": trades,
            "snapshot": snapshot,
            "signals_top5": signals.head(5)[["code", "score"]].to_dict("records"),
        }

    def get_daily_report(self, result: dict) -> str:
        """生成每日报告文本。"""
        if result.get("skipped"):
            return f"[{result['date']}] 已执行过或无信号，跳过。"

        lines = []
        lines.append("=" * 50)
        lines.append(f"模拟交易日报 - {result['date']}")
        lines.append("=" * 50)

        # 交易摘要
        trades = result.get("trades", [])
        sells = [t for t in trades if t["action"] == "SELL"]
        buys = [t for t in trades if t["action"] == "BUY"]

        lines.append(f"\n--- 交易摘要 ---")
        if sells:
            lines.append(f"卖出: {len(sells)} 只")
            for t in sells:
                lines.append(f"  SELL {t['code']}  {t['shares']} 股 @ {t['price']:.2f}  "
                             f"金额: {t['amount']:,.2f}  ({t['reason']})")
        else:
            lines.append("卖出: 无")

        if buys:
            lines.append(f"买入: {len(buys)} 只")
            for t in buys:
                lines.append(f"  BUY  {t['code']}  {t['shares']} 股 @ {t['price']:.2f}  "
                             f"金额: {t['amount']:,.2f}  ({t['reason']})")
        else:
            lines.append("买入: 无")

        # 组合概况
        snap = result["snapshot"]
        n_days = len(self.state["daily_snapshots"])
        lines.append(f"\n--- 组合概况 ---")
        lines.append(f"总资产:   {snap['total_value']:>14,.2f}")
        lines.append(f"现金:     {snap['cash']:>14,.2f}  "
                     f"({snap['cash']/snap['total_value']*100:.1f}%)")
        lines.append(f"持仓:     {snap['position_value']:>14,.2f}  "
                     f"({snap['n_positions']} 只)")
        lines.append(f"日收益:   {snap['daily_return']:>+14.4%}")
        lines.append(f"累计收益: {snap['cumulative_return']:>+14.4%}  "
                     f"(运行 {n_days} 天)")

        # 当前持仓
        lines.append(f"\n--- 当前持仓 ({len(self.state['positions'])} 只) ---")
        if self.state["positions"]:
            lines.append(f"{'代码':<12} {'持仓':>6} {'成本':>10} {'入场日期':>12}")
            for code, pos in sorted(self.state["positions"].items()):
                lines.append(f"{code:<12} {pos['shares']:>6} "
                             f"{pos['avg_cost']:>10.2f} {pos['entry_date']:>12}")

        # 信号 Top 5
        top5 = result.get("signals_top5", [])
        if top5:
            lines.append(f"\n--- 今日信号 Top 5 ---")
            for i, s in enumerate(top5, 1):
                lines.append(f"  #{i} {s['code']}  score={s['score']:.4f}")

        lines.append("")
        return "\n".join(lines)

    def get_portfolio_summary(self) -> str:
        """查看当前组合状态。"""
        if not self.state.get("created_at"):
            return "尚未初始化模拟交易组合。请先运行: python main.py dryrun"

        lines = []
        lines.append("=" * 50)
        lines.append("模拟交易组合状态")
        lines.append("=" * 50)

        initial = self.state["initial_cash"]
        cash = self.state["cash"]
        positions = self.state["positions"]

        # 尝试获取最新价格
        trade_date = None
        try:
            trade_date = self._get_latest_trade_date()
        except Exception:
            pass

        position_value = 0.0
        pos_lines = []
        if positions and trade_date:
            prices = self._get_prices(list(positions.keys()), trade_date)
            for code, pos in sorted(positions.items()):
                cur_price = prices.get(code, pos["avg_cost"])
                val = pos["shares"] * cur_price
                pnl = (cur_price - pos["avg_cost"]) / pos["avg_cost"]
                position_value += val
                pos_lines.append(
                    f"  {code:<12} {pos['shares']:>6} 股  "
                    f"成本 {pos['avg_cost']:>8.2f}  现价 {cur_price:>8.2f}  "
                    f"盈亏 {pnl:>+7.2%}  市值 {val:>12,.2f}"
                )
        elif positions:
            for code, pos in sorted(positions.items()):
                val = pos["shares"] * pos["avg_cost"]
                position_value += val
                pos_lines.append(
                    f"  {code:<12} {pos['shares']:>6} 股  成本 {pos['avg_cost']:>8.2f}  "
                    f"市值(成本) {val:>12,.2f}"
                )

        total = cash + position_value
        cum_ret = (total - initial) / initial

        lines.append(f"创建时间: {self.state['created_at'][:10]}")
        lines.append(f"最后更新: {self.state['updated_at'][:16]}")
        lines.append(f"参数: Top-{self.state['config']['topk']}, "
                     f"每日换 {self.state['config']['n_drop']} 只")
        lines.append(f"初始资金: {initial:>14,.2f}")
        lines.append(f"当前现金: {cash:>14,.2f}")
        lines.append(f"持仓市值: {position_value:>14,.2f}")
        lines.append(f"总资产:   {total:>14,.2f}")
        lines.append(f"累计收益: {cum_ret:>+14.4%}")
        lines.append(f"持仓数:   {len(positions)} 只")
        lines.append(f"交易次数: {len(self.state.get('trade_history', []))} 笔")
        lines.append(f"运行天数: {len(self.state.get('daily_snapshots', []))} 天")

        if pos_lines:
            lines.append(f"\n--- 持仓明细 ---")
            lines.extend(pos_lines)

        # 最近 5 笔交易
        history = self.state.get("trade_history", [])
        if history:
            lines.append(f"\n--- 最近交易 (最多 5 笔) ---")
            for t in history[-5:]:
                lines.append(
                    f"  {t['date']} {t['action']:>4} {t['code']:<12} "
                    f"{t['shares']:>6} 股 @ {t['price']:.2f}  {t['reason']}"
                )

        lines.append("")
        return "\n".join(lines)
