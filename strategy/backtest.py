"""
回测模块：基于 Qlib TopkDropoutStrategy，对训练好的模型做历史组合回测。
支持 A 股（沪+深）不同板块的涨跌停限制。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def _models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _get_board_limit(code: str) -> float:
    """按板块返回涨停阈值（收益率）。创业板(300)/科创板(688) 20%，主板 10%。"""
    if code.startswith("SZ.300") or code.startswith("SZ300"):
        return 0.195
    if code.startswith("SH.688") or code.startswith("SH688"):
        return 0.195
    return 0.095


def _get_limit_threshold(provider_uri: str) -> float:
    """
    根据数据目录推断涨停阈值（用于 Qlib exchange 兜底）。
    A 股：取 0.195（兼容创业板/科创板 20% 限制，信号层已按板块精确过滤）。
    港股：无涨停。
    """
    if "hk_quant_data" in provider_uri:
        return None  # 港股无涨停
    return 0.195  # 兜底阈值，信号层已按板块精确过滤


def _filter_limit_up_pred(pred: pd.DataFrame, provider_uri: str) -> pd.DataFrame:
    """从预测信号中排除涨停股（信号日涨停 → 无法买入）。

    按板块分别判断：主板 >= 9.5%，创业板/科创板 >= 19.5%。
    被排除的股票 score 设为 -inf，排名自然沉底不会被选入 TopK。
    """
    if "hk_quant_data" in provider_uri:
        return pred

    try:
        from qlib.data import D

        # 找 instrument / datetime level
        inst_level, dt_level = None, None
        for i, name in enumerate(pred.index.names):
            vals = pred.index.get_level_values(i)
            if pd.api.types.is_datetime64_any_dtype(vals):
                dt_level = i
            else:
                inst_level = i
        if inst_level is None or dt_level is None:
            return pred

        instruments = list(pred.index.get_level_values(inst_level).unique())
        dt_vals = pred.index.get_level_values(dt_level)
        start_d, end_d = str(dt_vals.min())[:10], str(dt_vals.max())[:10]

        # 信号日涨跌幅
        day_ret = D.features(instruments, ["$close/Ref($close,1)-1"],
                             start_time=start_d, end_time=end_d)
        day_ret.columns = ["day_ret"]

        # 对齐 index level 顺序
        if isinstance(day_ret.index, pd.MultiIndex):
            ret_dt_level = None
            for i in range(day_ret.index.nlevels):
                if pd.api.types.is_datetime64_any_dtype(day_ret.index.get_level_values(i)):
                    ret_dt_level = i
                    break
            if ret_dt_level is not None and ret_dt_level != dt_level:
                day_ret = day_ret.swaplevel()
            day_ret.index.names = pred.index.names
            day_ret = day_ret.sort_index()

        matched_ret = day_ret["day_ret"].reindex(pred.index).fillna(0)

        # 按板块设定涨停阈值
        inst_str = pred.index.get_level_values(inst_level).astype(str)
        thresholds = inst_str.map(_get_board_limit)
        thresholds = pd.Series(thresholds.values, index=pred.index, dtype=float)

        is_limit_up = matched_ret >= thresholds
        n_filtered = int(is_limit_up.sum())

        pred_out = pred.copy()
        if n_filtered > 0:
            print(f"[回测] 排除涨停信号 {n_filtered} 条")
            pred_out.loc[is_limit_up, pred_out.columns[0]] = float("-inf")

        return pred_out
    except Exception as e:
        print(f"[WARNING] 涨停过滤失败: {e}，使用原始信号")
        return pred


def run_backtest(
    pred_pkl: Path | None = None,
    provider_uri: str = "~/.qlib/qlib_data/my_quant_data",
    topk: int = 10,
    n_drop: int = 2,
    start_time: str | None = None,
    end_time: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    对 pred.pkl 中的预测分数执行 TopkDropout 组合回测。
    """
    import qlib
    from qlib.constant import REG_CN
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import risk_analysis

    uri = str(Path(provider_uri).expanduser().resolve())
    qlib.init(provider_uri=uri, region=REG_CN)

    # 加载预测
    pred_path = pred_pkl or (_models_dir() / "pred_a.pkl")
    if not pred_path.exists():
        # 兼容旧文件名
        pred_path = _models_dir() / "pred.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"未找到预测文件: {pred_path}")

    with open(pred_path, "rb") as f:
        pred = pickle.load(f)

    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif isinstance(pred, pd.DataFrame) and pred.columns[0] != "score":
        pred = pred.rename(columns={pred.columns[0]: "score"})

    # 排除涨停股（信号日涨停 → 无法买入，不应进入候选池）
    pred = _filter_limit_up_pred(pred, provider_uri)

    # 找到 datetime level
    if "datetime" in pred.index.names:
        dates = pred.index.get_level_values("datetime")
    else:
        # 尝试找 Timestamp 类型的 level
        for i, name in enumerate(pred.index.names):
            vals = pred.index.get_level_values(i)
            if pd.api.types.is_datetime64_any_dtype(vals):
                dates = vals
                break
        else:
            dates = pred.index.get_level_values(0)
    bt_start = start_time or str(dates.min())[:10]
    # 回测结束日不能等于日历最后一天（Qlib 边界 bug），取倒数第二天
    bt_end_raw = end_time or str(dates.max())[:10]
    unique_dates = sorted(dates.unique())
    if len(unique_dates) >= 2 and str(unique_dates[-1])[:10] == bt_end_raw:
        bt_end = str(unique_dates[-2])[:10]
    else:
        bt_end = bt_end_raw

    if verbose:
        print(f"=== 回测参数 ===")
        print(f"  时间范围: {bt_start} ~ {bt_end}")
        print(f"  持仓 Top-{topk}，每日最多换 {n_drop} 只")
        print(f"  股票池: {pred.index.get_level_values(0).nunique()} 只")

    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=topk,
        n_drop=n_drop,
    )

    limit_threshold = _get_limit_threshold(provider_uri)
    exchange_kwargs = {
        "deal_price": "close",
        "open_cost": 0.0005,  # 万五佣金
        "close_cost": 0.0015,  # 千一.五（含印花税）
        "min_cost": 5,
    }
    if limit_threshold is not None:
        exchange_kwargs["limit_threshold"] = limit_threshold

    portfolio_metric_dict, indicator_dict = backtest(
        start_time=bt_start,
        end_time=bt_end,
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        benchmark="SH.600000",
        account=1_000_000,
        exchange_kwargs=exchange_kwargs,
    )

    # 解析 Qlib backtest 返回值
    # 格式: (portfolio_metric_dict, indicator_dict)
    # portfolio_metric_dict["1day"] = (DataFrame, order_dict)
    # indicator_dict["1day"] = (DataFrame, Indicator)
    pmd_day = portfolio_metric_dict.get("1day", (pd.DataFrame(),))
    report_normal = pmd_day[0] if isinstance(pmd_day, tuple) else pmd_day

    if isinstance(report_normal, pd.DataFrame) and not report_normal.empty and "return" in report_normal.columns:
        if "bench" in report_normal.columns:
            returns = report_normal["return"] - report_normal["bench"]
        else:
            returns = report_normal["return"]
        analysis = risk_analysis(returns)
        result = {
            "annual_return": float(analysis.loc["mean", "risk"] * 252),
            "sharpe":        float(analysis.loc["mean", "risk"] / (analysis.loc["std", "risk"] + 1e-9) * (252 ** 0.5)),
            "max_drawdown":  float(analysis.loc["max_drawdown", "risk"]),
        }
    else:
        result = {"annual_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan")}

    # IC/ICIR from indicator
    result["IC"] = float("nan")
    result["ICIR"] = float("nan")
    try:
        ind_day = indicator_dict.get("1day", (None, None))
        if isinstance(ind_day, tuple) and len(ind_day) >= 2:
            indicator_obj = ind_day[1]
            if hasattr(indicator_obj, "get_metric"):
                ic_val = indicator_obj.get_metric("IC")
                icir_val = indicator_obj.get_metric("ICIR")
                if ic_val is not None:
                    result["IC"] = float(ic_val)
                if icir_val is not None:
                    result["ICIR"] = float(icir_val)
    except Exception:
        pass

    if verbose:
        print("\n=== 回测结果 ===")
        print(f"  年化超额收益: {result['annual_return']:.2%}")
        print(f"  夏普比率: {result['sharpe']:.3f}")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")
        print(f"  IC:       {result['IC']:.4f}")
        print(f"  ICIR:     {result['ICIR']:.4f}")

    return result


def run_hk_backtest(
    topk: int = 10,
    n_drop: int = 2,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """港股组合回测。"""
    pred_path = _models_dir() / "hk_pred.pkl"
    return run_backtest(
        pred_pkl=pred_path,
        provider_uri="~/.qlib/qlib_data/hk_quant_data",
        topk=topk,
        n_drop=n_drop,
        start_time=start_time,
        end_time=end_time,
    )
