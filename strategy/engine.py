"""
策略引擎：基于 Qlib 的模型训练与推理预测。

支持两种模式：
- train(): 完整训练流程，生成模型文件
- predict(): 加载已训练模型，预测下一日股票排名分数（含 300 天 lookback 优化）
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
except ImportError as e:
    raise ImportError("请安装 pyqlib: pip install pyqlib") from e


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_models_dir() -> Path:
    """默认模型目录（与 strategy 同级）。"""
    return _this_dir().parent / "models"


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (_this_dir() / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _calendar_range(provider_uri: str, freq: str = "day") -> tuple[str, str]:
    """从 Qlib 数据目录读取 calendar，返回 (首日, 末日) 字符串。"""
    cal_path = Path(provider_uri).expanduser().resolve() / "calendars" / f"{freq}.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"日历文件不存在: {cal_path}，请先运行数据转换")
    lines = cal_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError("日历文件为空")
    return lines[0].strip(), lines[-1].strip()


def _get_provider_uri() -> str:
    """
    获取 Qlib 数据目录。
    优先使用环境变量 QLIB_DATA_DIR，默认 ~/.qlib/qlib_data/my_quant_data
    """
    env_path = os.environ.get("QLIB_DATA_DIR")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str(Path("~/.qlib/qlib_data/my_quant_data").expanduser().resolve())


# SRE Config Audit：实际数据起始日，早于此日的 segment 会触发红色警告
DATA_START_DATE = "2006-01-03"

# 预热期：给 Alpha158 的 60 日均线等特征留足窗口
WARMUP_START = "2014-10-01"
WARMUP_END = "2014-12-31"
FIXED_SEGMENTS = {
    "train": ["2015-01-01", "2024-06-30"],   # 9.5 年
    "valid": ["2024-07-01", "2025-06-30"],   # 1 年
    "test": ["2025-07-01", os.environ.get("TEST_END_DATE", "2026-03-07")],
}
# Handler 需要从预热期开始，以便 Train 首日能算均线
HANDLER_START = WARMUP_START

# ANSI 红色，用于终端警告
_RED = "\033[91m"
_RESET = "\033[0m"


def _sre_config_audit(segments: dict[str, list[str]]) -> None:
    """
    SRE Config Audit：打印 dataset.segments 日期范围、Train 交易日数，
    并检查是否早于实际数据起始日（DATA_START_DATE）。
    """
    print("\n=== SRE Config Audit ===")
    print("dataset -> segments 日期范围:")
    for name, (start, end) in segments.items():
        print(f"  {name}: [{start}, {end}]")

    train_start, train_end = segments["train"][0], segments["train"][1]
    cal = D.calendar(start_time=train_start, end_time=train_end, freq="day")
    n_train = len(cal) if cal is not None else 0
    print(f"Train 阶段交易日数（D.calendar）: {n_train}")

    if train_start < DATA_START_DATE:
        print(
            f"{_RED}[警告] 配置的 Train 开始时间 ({train_start}) 早于实际数据起始日 ({DATA_START_DATE})，"
            f"该区间内无真实行情，模型可能使用空/填充数据。{_RESET}"
        )
    print("=== SRE Config Audit 结束 ===\n")


def _segments_from_calendar(first_date: str, last_date: str) -> dict[str, list[str]]:
    """
    根据日历首尾日期生成 train/valid/test 分段（约 60% / 20% / 20%）。
    保证训练集至少约 120 个交易日，否则按 80% train、20% test 划分。
    """
    from datetime import datetime, timedelta
    t0 = datetime.strptime(first_date, "%Y-%m-%d")
    t1 = datetime.strptime(last_date, "%Y-%m-%d")
    n_days = (t1 - t0).days
    if n_days < 180:
        # 数据不足半年：80% 训练，20% 测试
        split1 = int(n_days * 0.8)
        train_end = (t0 + timedelta(days=split1)).strftime("%Y-%m-%d")
        return {
            "train": [first_date, train_end],
            "valid": [train_end, train_end],
            "test": [train_end, last_date],
        }
    # 60% train, 20% valid, 20% test
    s1 = int(n_days * 0.6)
    s2 = int(n_days * 0.8)
    valid_start = (t0 + timedelta(days=s1)).strftime("%Y-%m-%d")
    test_start = (t0 + timedelta(days=s2)).strftime("%Y-%m-%d")
    train_end = valid_start
    valid_end = test_start
    return {
        "train": [first_date, train_end],
        "valid": [valid_start, valid_end],
        "test": [test_start, last_date],
    }


class StrategyEngine:
    """基于 Qlib 的策略引擎：训练 LightGBM 模型并支持下一日预测。"""

    def __init__(
        self,
        provider_uri: str | Path | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        self.provider_uri = provider_uri or _get_provider_uri()
        self.provider_uri = str(Path(self.provider_uri).expanduser().resolve())
        self._models_dir = Path(models_dir).resolve() if models_dir else _default_models_dir()
        qlib.init(provider_uri=self.provider_uri, region=REG_CN)

    def train(
        self,
        config_path: Path | None = None,
        experiment_name: str = "lgb_alpha158fund",
        save_dir: Path | None = None,
        market: str = "all",
    ) -> dict[str, float]:
        """
        加载 config.yaml，使用 qlib.workflow.R 启动实验，训练 LightGBM 并保存到 models/。
        返回测试集上的 IC、ICIR。
        """
        config = _load_config(config_path)
        task = config.get("task", {})
        if not task:
            raise ValueError("config.yaml 中缺少 task 段")

        model_cfg = task["model"]
        dataset_cfg = task["dataset"].copy()
        # 使用固定时间切分
        first_date, last_date = _calendar_range(self.provider_uri)
        segments = FIXED_SEGMENTS.copy()
        handler_end = last_date  # 不超出实际日历
        print(f"数据日历: {first_date} ~ {last_date} | 固定切分 train: {segments['train']} valid: {segments['valid']} test: {segments['test']}")
        _sre_config_audit(segments)
        if dataset_cfg.get("kwargs") is None:
            dataset_cfg["kwargs"] = {}
        dataset_cfg["kwargs"] = dict(dataset_cfg["kwargs"])
        dataset_cfg["kwargs"]["segments"] = segments
        handler_cfg = (dataset_cfg["kwargs"].get("handler") or {}).copy()
        if isinstance(handler_cfg, dict):
            h_kwargs = dict(handler_cfg.get("kwargs") or {})
            h_kwargs["start_time"] = HANDLER_START
            h_kwargs["end_time"] = handler_end
            h_kwargs["fit_start_time"] = HANDLER_START
            h_kwargs["fit_end_time"] = handler_end
            h_kwargs["instruments"] = market
            handler_cfg["kwargs"] = h_kwargs
            dataset_cfg["kwargs"]["handler"] = handler_cfg

        models_dir = Path(save_dir) if save_dir else self._models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            from qlib.contrib.eva.alpha import calc_ic
            from qlib.utils import flatten_dict
        except ImportError:
            calc_ic = None
            flatten_dict = None

        model = init_instance_by_config(model_cfg)
        dataset = init_instance_by_config(dataset_cfg)

        # 训练前检查：Train 集不能为空
        try:
            train_prep = dataset.prepare("train", col_set=["feature"])
        except Exception:
            train_prep = dataset.prepare("train")
        train_df = train_prep[0] if isinstance(train_prep, (list, tuple)) and len(train_prep) > 0 else train_prep
        train_shape = getattr(train_df, "shape", (0,))
        train_rows = train_shape[0] if len(train_shape) >= 1 else 0
        print(f"dataset.prepare('train').shape: {train_shape}")
        if train_rows == 0:
            raise RuntimeError(
                f"{_RED}[错误] 训练集为空 (shape[0]=0)，请检查数据范围与 segments 是否在 calendar 内。{_RESET}"
            )

        with R.start(experiment_name=experiment_name):
            if flatten_dict:
                R.log_params(**flatten_dict(task))
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})

            # 测试集预测与 IC/ICIR
            pred = model.predict(dataset, segment="test")
            prep = dataset.prepare("test", col_set=["label"], data_key=type(dataset.handler).DK_L)
            df_label = prep[0] if isinstance(prep, (list, tuple)) and len(prep) > 0 else prep
            if pred is None or df_label is None or pred.size == 0:
                ic_val, icir_val = float("nan"), float("nan")
            else:
                # pred/df_label 可能为 DataFrame 或 Series，统一取一维
                pred_flat = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
                label_flat = df_label.iloc[:, 0] if isinstance(df_label, pd.DataFrame) else df_label
                if calc_ic is not None:
                    ic_series, ric_series = calc_ic(pred_flat, label_flat)
                    ic_val = float(ic_series.mean()) if hasattr(ic_series, "mean") else float(ic_series)
                    ic_std = float(ic_series.std()) if hasattr(ic_series, "std") else float("nan")
                    icir_val = ic_val / ic_std if ic_std and ic_std != 0 else float("nan")
                else:
                    ic_val, icir_val = float("nan"), float("nan")
            R.log_metrics(IC=ic_val, ICIR=icir_val)

            # 保存 pred.pkl
            suffix = f"_{market}" if market != "all" else ""
            pred_path = models_dir / f"pred{suffix}.pkl"
            with open(pred_path, "wb") as f:
                pickle.dump(pred, f)
            print(f"已保存 Test 集预测: {pred_path}")

            # 保存模型供 predict() 使用
            latest_path = models_dir / f"lightgbm{suffix}_latest.pkl"
            with open(latest_path, "wb") as f:
                pickle.dump(model, f)

        print(f"Test 集 IC: {ic_val:.6f}  ICIR: {icir_val:.6f}")
        return {"IC": ic_val, "ICIR": icir_val}

    def predict(
        self,
        model_path: str | Path | None = None,
        market: str = "all",
        lookback_days: int = 300,
    ) -> pd.DataFrame:
        """
        加载已训练模型，用最新一天的数据预测全部股票分数。

        使用 lookback 优化：仅加载最近 N 天数据（默认 300 天），
        避免在低内存环境上加载全量历史数据导致 OOM。

        Args:
            model_path: 模型文件路径，默认自动查找 models/lightgbm_{market}_latest.pkl
            market: 市场代码（如 "all", "sh", "hk"）
            lookback_days: 回看天数，Alpha158 最大需 60 日窗口，默认 300 天留足余量

        Returns:
            DataFrame [code, score, rank, top5]，按 score 降序
        """
        # 在 Streamlit/子进程环境下禁用 Qlib 多进程
        try:
            from qlib.config import C
            _old_backend = getattr(C, "joblib_backend", None)
            C.joblib_backend = "sequential"
            C["kwargs"] = {"max_workers": 2}
        except Exception:
            _old_backend = None

        try:
            return self._predict_impl(model_path, market, lookback_days)
        finally:
            if _old_backend is not None:
                try:
                    C.joblib_backend = _old_backend
                except Exception:
                    pass

    def _predict_impl(
        self,
        model_path: str | Path | None,
        market: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """推理实现：加载模型 -> 构建单日 dataset -> 预测 -> 输出信号。"""
        # 加载模型
        if model_path is None:
            suffix = f"_{market}" if market != "all" else ""
            model_path = self._models_dir / f"lightgbm{suffix}_latest.pkl"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        config = _load_config()
        task = config.get("task", {})
        if not task:
            raise ValueError("config.yaml 中缺少 task 段")

        # 获取最新交易日
        from datetime import datetime
        end_time = datetime.now().strftime("%Y-%m-%d")
        cal = D.calendar(start_time="2020-01-01", end_time=end_time, freq="day")
        if cal is None or len(cal) == 0:
            raise RuntimeError("Qlib 日历为空，请检查数据转换是否完成")
        last_date = pd.Timestamp(cal[-1]).strftime("%Y-%m-%d")

        # 限制 start_time：Alpha158 最大需 60 日回看，用 lookback_days 留足余量
        # 避免在低内存环境上加载全量历史数据 OOM
        if len(cal) > lookback_days:
            start_time = pd.Timestamp(cal[-lookback_days]).strftime("%Y-%m-%d")
        else:
            start_time = pd.Timestamp(cal[0]).strftime("%Y-%m-%d")

        # 构建单日推理 dataset
        dataset_cfg = task["dataset"].copy()
        kwargs = dataset_cfg.get("kwargs", {}).copy()
        handler_cfg = kwargs.get("handler", {}).copy()
        h_kwargs = handler_cfg.get("kwargs", {}).copy()
        h_kwargs["start_time"] = start_time
        h_kwargs["end_time"] = last_date
        h_kwargs["fit_start_time"] = start_time
        h_kwargs["fit_end_time"] = last_date
        h_kwargs["instruments"] = market
        handler_cfg["kwargs"] = h_kwargs
        kwargs["handler"] = handler_cfg
        kwargs["segments"] = {"infer": [last_date, last_date]}
        dataset_cfg["kwargs"] = kwargs

        dataset = init_instance_by_config(dataset_cfg)
        dataset.prepare("infer", col_set=["feature"])

        pred = model.predict(dataset, segment="infer")
        if pred is None or (hasattr(pred, "empty") and pred.empty):
            return pd.DataFrame(columns=["code", "score", "rank", "top5"])

        # 解析 MultiIndex 预测结果
        if hasattr(pred, "index") and isinstance(pred.index, pd.MultiIndex):
            level = "instrument" if "instrument" in pred.index.names else pred.index.names[1]
            codes = pred.index.get_level_values(level).astype(str).tolist()
        elif hasattr(pred, "index"):
            codes = pred.index.astype(str).tolist()
        else:
            codes = list(range(len(pred)))

        if isinstance(pred, pd.Series):
            scores = pred.values.flatten()
        elif hasattr(pred, "iloc") and pred.ndim >= 2:
            scores = pred.iloc[:, 0].values
        else:
            scores = np.asarray(pred).flatten()

        df = pd.DataFrame({"code": codes, "score": scores})
        df = df.dropna(subset=["score"])
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        df["top5"] = df["rank"] <= 5
        return df[["code", "score", "rank", "top5"]]


# Alias for inference module compatibility
InferenceEngine = StrategyEngine
