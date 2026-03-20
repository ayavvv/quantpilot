"""
策略引擎：基于 Qlib 的模型训练与下一日预测。
"""

from __future__ import annotations

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


def _get_limit_threshold(code: str) -> float:
    """按板块返回涨停阈值（收益率）。创业板(300)/科创板(688) 20%，主板 10%。"""
    if code.startswith("SZ.300") or code.startswith("SZ300"):
        return 0.195
    if code.startswith("SH.688") or code.startswith("SH688"):
        return 0.195
    return 0.095


def _filter_limit_up_for_ic(
    pred_flat: pd.Series, label_flat: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """排除涨停股后返回过滤后的 pred/label，用于 IC/ICIR 计算。

    涨停股无法买入，纳入 IC 会虚增选股信号质量。
    使用信号日涨跌幅 ($close/Ref($close,1)-1) 判断是否涨停。
    """
    try:
        common_idx = pred_flat.index.intersection(label_flat.index)
        if len(common_idx) == 0:
            return pred_flat, label_flat
        pred_flat = pred_flat.loc[common_idx]
        label_flat = label_flat.loc[common_idx]

        if not isinstance(common_idx, pd.MultiIndex):
            return pred_flat, label_flat

        # 找 instrument / datetime level
        inst_level, dt_level = None, None
        for i, name in enumerate(common_idx.names):
            vals = common_idx.get_level_values(i)
            if pd.api.types.is_datetime64_any_dtype(vals):
                dt_level = i
            else:
                inst_level = i
        if inst_level is None or dt_level is None:
            return pred_flat, label_flat

        instruments = list(common_idx.get_level_values(inst_level).unique())
        dt_vals = common_idx.get_level_values(dt_level)
        start_d, end_d = str(dt_vals.min())[:10], str(dt_vals.max())[:10]

        # 信号日涨跌幅
        day_ret = D.features(instruments, ["$close/Ref($close,1)-1"],
                             start_time=start_d, end_time=end_d)
        day_ret.columns = ["day_ret"]

        # 对齐 index level 顺序（D.features 返回 instrument,datetime）
        if isinstance(day_ret.index, pd.MultiIndex):
            ret_dt_level = None
            for i in range(day_ret.index.nlevels):
                if pd.api.types.is_datetime64_any_dtype(day_ret.index.get_level_values(i)):
                    ret_dt_level = i
                    break
            if ret_dt_level is not None and ret_dt_level != dt_level:
                day_ret = day_ret.swaplevel()
            day_ret.index.names = common_idx.names
            day_ret = day_ret.sort_index()

        matched_ret = day_ret["day_ret"].reindex(common_idx).fillna(0)

        # 按板块设定涨停阈值
        inst_str = common_idx.get_level_values(inst_level).astype(str)
        thresholds = inst_str.map(_get_limit_threshold)
        thresholds = pd.Series(thresholds.values, index=common_idx, dtype=float)

        not_limit_up = matched_ret < thresholds
        n_excluded = int((~not_limit_up).sum())
        if n_excluded > 0:
            print(f"[IC/ICIR] 排除涨停股 {n_excluded}/{len(common_idx)} 条")

        return pred_flat[not_limit_up], label_flat[not_limit_up]
    except Exception as e:
        print(f"[WARNING] 涨停过滤失败: {e}，使用全量数据")
        return pred_flat, label_flat


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _models_dir() -> Path:
    return _this_dir().parent / "models"


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (_this_dir() / "config_a.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _calendar_range(provider_uri: str, freq: str = "day") -> tuple[str, str]:
    cal_path = Path(provider_uri).expanduser().resolve() / "calendars" / f"{freq}.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"日历文件不存在: {cal_path}，请先运行数据转换 (python main.py convert)")
    lines = cal_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError("日历文件为空")
    return lines[0].strip(), lines[-1].strip()


# ── A 股配置 ──────────────────────────────────────────

A_CONFIG_PATH = _this_dir() / "config_a.yaml"

# 固定时间段（全 A 股，6年训练 + 1年验证 + 测试）
A_FIXED_SEGMENTS = {
    "train": ["2018-01-01", "2024-06-30"],
    "valid": ["2024-07-01", "2025-06-30"],
    "test":  ["2025-07-01", "2026-12-31"],
}
A_WARMUP_START = "2017-07-01"  # 给 Alpha158 均线预热 6 个月

# ── 港股配置 ──────────────────────────────────────────

HK_PROVIDER_URI = "~/.qlib/qlib_data/hk_quant_data"
HK_CONFIG_PATH = _this_dir() / "config_hk.yaml"

HK_FIXED_SEGMENTS = {
    "train": ["2016-01-04", "2023-12-31"],
    "valid": ["2024-01-01", "2024-06-30"],
    "test":  ["2024-07-01", "2026-03-03"],
}
HK_WARMUP_START = "2016-01-04"


def _segments_from_calendar(first_date: str, last_date: str) -> dict[str, list[str]]:
    """根据日历首尾日期自适应生成 train/valid/test 分段（约 60/20/20）。"""
    from datetime import datetime, timedelta
    t0 = datetime.strptime(first_date, "%Y-%m-%d")
    t1 = datetime.strptime(last_date, "%Y-%m-%d")
    n_days = (t1 - t0).days
    if n_days < 180:
        split1 = int(n_days * 0.8)
        train_end = (t0 + timedelta(days=split1)).strftime("%Y-%m-%d")
        return {
            "train": [first_date, train_end],
            "valid": [train_end, train_end],
            "test": [train_end, last_date],
        }
    s1 = int(n_days * 0.6)
    s2 = int(n_days * 0.8)
    valid_start = (t0 + timedelta(days=s1)).strftime("%Y-%m-%d")
    test_start = (t0 + timedelta(days=s2)).strftime("%Y-%m-%d")
    return {
        "train": [first_date, valid_start],
        "valid": [valid_start, test_start],
        "test": [test_start, last_date],
    }


class StrategyEngine:
    """基于 Qlib 的策略引擎：训练 LightGBM 模型并支持下一日预测。"""

    def __init__(
        self,
        provider_uri: str | Path = "~/.qlib/qlib_data/my_quant_data",
    ) -> None:
        self.provider_uri = str(Path(provider_uri).expanduser().resolve())
        qlib.init(provider_uri=self.provider_uri, region=REG_CN)

    @classmethod
    def for_hk(cls) -> "StrategyEngine":
        return cls(provider_uri=HK_PROVIDER_URI)

    def train_a_model(
        self,
        experiment_name: str = "a_share_lgbm",
        save_dir: Path | None = None,
    ) -> dict[str, float]:
        """训练全 A 股选股模型。"""
        return self.train_model(
            config_path=A_CONFIG_PATH,
            experiment_name=experiment_name,
            save_dir=save_dir,
            use_dynamic_segments=True,
            fixed_segments_override=A_FIXED_SEGMENTS,
            warmup_start_override=A_WARMUP_START,
            hk_mode=False,
        )

    def train_hk_model(
        self,
        experiment_name: str = "hk_lgbm",
        save_dir: Path | None = None,
    ) -> dict[str, float]:
        return self.train_model(
            config_path=HK_CONFIG_PATH,
            experiment_name=experiment_name,
            save_dir=save_dir,
            use_dynamic_segments=True,
            fixed_segments_override=HK_FIXED_SEGMENTS,
            warmup_start_override=HK_WARMUP_START,
            hk_mode=True,
        )

    def train_model(
        self,
        config_path: Path | None = None,
        experiment_name: str = "a_share_lgbm",
        save_dir: Path | None = None,
        use_dynamic_segments: bool = False,
        fixed_segments_override: dict | None = None,
        warmup_start_override: str | None = None,
        hk_mode: bool = False,
    ) -> dict[str, float]:
        config = _load_config(config_path)
        task = config.get("task", {})
        if not task:
            raise ValueError("config 中缺少 task 段")

        model_cfg = task["model"]
        dataset_cfg = task["dataset"].copy()
        first_date, last_date = _calendar_range(self.provider_uri)
        handler_end = last_date

        _fixed = fixed_segments_override or A_FIXED_SEGMENTS
        if use_dynamic_segments:
            try:
                train_start = _fixed["train"][0]
                if train_start < first_date:
                    segments = _segments_from_calendar(first_date, last_date)
                    print(f"[INFO] 数据起始({first_date})晚于固定段，自适应切分")
                else:
                    segments = {k: list(v) for k, v in _fixed.items()}
                    segments["test"][1] = min(segments["test"][1], last_date)
            except Exception:
                segments = _segments_from_calendar(first_date, last_date)
        else:
            segments = {k: list(v) for k, v in _fixed.items()}

        _warmup = warmup_start_override or A_WARMUP_START
        print(f"数据日历: {first_date} ~ {last_date}")
        print(f"Train: {segments['train']}  Valid: {segments['valid']}  Test: {segments['test']}")

        if dataset_cfg.get("kwargs") is None:
            dataset_cfg["kwargs"] = {}
        dataset_cfg["kwargs"] = dict(dataset_cfg["kwargs"])
        dataset_cfg["kwargs"]["segments"] = segments
        handler_cfg = (dataset_cfg["kwargs"].get("handler") or {}).copy()
        if isinstance(handler_cfg, dict):
            h_kwargs = dict(handler_cfg.get("kwargs") or {})
            h_kwargs["start_time"] = _warmup
            h_kwargs["end_time"] = handler_end
            h_kwargs["fit_start_time"] = _warmup
            h_kwargs["fit_end_time"] = handler_end
            handler_cfg["kwargs"] = h_kwargs
            dataset_cfg["kwargs"]["handler"] = handler_cfg

        models_dir = Path(save_dir) if save_dir else _models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            from qlib.contrib.eva.alpha import calc_ic
        except ImportError:
            calc_ic = None

        model = init_instance_by_config(model_cfg)
        dataset = init_instance_by_config(dataset_cfg)

        # 训练前检查
        try:
            train_prep = dataset.prepare("train", col_set=["feature"])
        except Exception:
            train_prep = dataset.prepare("train")
        train_df = train_prep[0] if isinstance(train_prep, (list, tuple)) and len(train_prep) > 0 else train_prep
        train_shape = getattr(train_df, "shape", (0,))
        train_rows = train_shape[0] if len(train_shape) >= 1 else 0
        print(f"Train shape: {train_shape}")
        if train_rows == 0:
            raise RuntimeError("训练集为空，请检查数据范围与 segments")

        with R.start(experiment_name=experiment_name):
            model.fit(dataset)

            pred = model.predict(dataset, segment="test")
            prep = dataset.prepare("test", col_set=["label"], data_key=type(dataset.handler).DK_L)
            df_label = prep[0] if isinstance(prep, (list, tuple)) and len(prep) > 0 else prep
            if pred is None or df_label is None or pred.size == 0:
                ic_val, icir_val = float("nan"), float("nan")
            else:
                pred_flat = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
                label_flat = df_label.iloc[:, 0] if isinstance(df_label, pd.DataFrame) else df_label
                # A 股：排除涨停股计算 IC/ICIR（涨停无法买入，纳入会虚增信号质量）
                if not hk_mode:
                    pred_flat, label_flat = _filter_limit_up_for_ic(pred_flat, label_flat)
                if calc_ic is not None:
                    ic_series, ric_series = calc_ic(pred_flat, label_flat)
                    ic_val = float(ic_series.mean()) if hasattr(ic_series, "mean") else float(ic_series)
                    ic_std = float(ic_series.std()) if hasattr(ic_series, "std") else float("nan")
                    icir_val = ic_val / ic_std if ic_std and ic_std != 0 else float("nan")
                else:
                    ic_val, icir_val = float("nan"), float("nan")
            R.log_metrics(IC=ic_val, ICIR=icir_val)

            # 保存预测
            pred_suffix = "hk_pred.pkl" if hk_mode else "pred_a.pkl"
            pred_path = models_dir / pred_suffix
            with open(pred_path, "wb") as f:
                pickle.dump(pred, f)
            print(f"已保存 Test 集预测: {pred_path}")

            # 同时保存一份 pred.pkl 兼容旧接口
            if not hk_mode:
                compat_path = models_dir / "pred.pkl"
                with open(compat_path, "wb") as f:
                    pickle.dump(pred, f)

            model_suffix = "lightgbm_hk_latest.pkl" if hk_mode else "lightgbm_a_latest.pkl"
            latest_path = models_dir / model_suffix
            with open(latest_path, "wb") as f:
                pickle.dump(model, f)

            # 兼容旧接口
            if not hk_mode:
                compat_model = models_dir / "lightgbm_latest.pkl"
                with open(compat_model, "wb") as f:
                    pickle.dump(model, f)

        print(f"Test 集 IC: {ic_val:.6f}  ICIR: {icir_val:.6f}")
        return {"IC": ic_val, "ICIR": icir_val}

    def predict_next_day(self, hk_mode: bool | None = None) -> pd.DataFrame:
        if hk_mode is None:
            hk_mode = "hk_quant_data" in self.provider_uri
        try:
            from qlib.config import C
            _old_backend = getattr(C, "joblib_backend", None)
            C.joblib_backend = "sequential"
        except Exception:
            _old_backend = None
        try:
            return self._predict_next_day_impl(hk_mode=hk_mode)
        finally:
            if _old_backend is not None:
                try:
                    C.joblib_backend = _old_backend
                except Exception:
                    pass

    def _predict_next_day_impl(self, hk_mode: bool = False) -> pd.DataFrame:
        models_dir = _models_dir()
        if hk_mode:
            model_file = "lightgbm_hk_latest.pkl"
        else:
            # 优先用新文件名，兼容旧的
            model_file = "lightgbm_a_latest.pkl"
            if not (models_dir / model_file).exists():
                model_file = "lightgbm_latest.pkl"
        latest_path = models_dir / model_file
        if not latest_path.exists():
            raise FileNotFoundError(f"未找到模型文件 {latest_path}")

        with open(latest_path, "rb") as f:
            model = pickle.load(f)

        config_path = HK_CONFIG_PATH if hk_mode else A_CONFIG_PATH
        if not config_path.exists():
            config_path = _this_dir() / "config.yaml"
        config = _load_config(config_path)
        task = config.get("task", {})
        if not task:
            raise ValueError("config 中缺少 task 段")

        from datetime import datetime, timedelta
        end_time = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            cal = D.calendar(start_time="2020-01-01", end_time=end_time, freq="day")
            if cal is None or (hasattr(cal, "__len__") and len(cal) == 0):
                raise ValueError("日历为空")
            last_date = pd.Timestamp(cal[-1]).strftime("%Y-%m-%d")
        except Exception as e:
            raise RuntimeError(f"获取最近交易日失败: {e}") from e

        # 防止日历被 US/MACRO 数据推进超过 A 股数据范围：
        # 取 A 股 instruments 的最大 end_time，限制 infer_date
        if not hk_mode:
            inst_path = Path(self.provider_uri) / "instruments" / "all.txt"
            if inst_path.exists():
                max_a_end = None
                for line in inst_path.read_text().splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 3 and (parts[0].startswith("SH") or parts[0].startswith("SZ")):
                        if max_a_end is None or parts[2] > max_a_end:
                            max_a_end = parts[2]
                if max_a_end and max_a_end < last_date:
                    print(f"[WARN] 日历最后一天 {last_date} 超过 A 股数据范围 {max_a_end}，回退到 {max_a_end}")
                    last_date = max_a_end

        dataset_cfg = task["dataset"].copy()
        kwargs = dataset_cfg.get("kwargs", {}).copy()
        handler_cfg = kwargs.get("handler", {}).copy()
        h_kwargs = handler_cfg.get("kwargs", {}).copy()
        h_kwargs["end_time"] = last_date
        h_kwargs["fit_end_time"] = last_date
        handler_cfg["kwargs"] = h_kwargs
        kwargs["handler"] = handler_cfg
        segs = kwargs.get("segments", {}).copy()
        segs["infer"] = [last_date, last_date]
        kwargs["segments"] = segs
        dataset_cfg["kwargs"] = kwargs

        dataset = init_instance_by_config(dataset_cfg)

        try:
            dataset.prepare("infer", col_set=["feature"])
        except Exception as e:
            raise RuntimeError(f"准备推理数据失败（{last_date}）: {e}") from e

        pred = model.predict(dataset, segment="infer")
        if pred is None or (hasattr(pred, "empty") and pred.empty):
            return pd.DataFrame(columns=["code", "score", "rank", "top5"])

        if hasattr(pred, "index") and isinstance(pred.index, pd.MultiIndex):
            level_instrument = "instrument" if "instrument" in pred.index.names else pred.index.names[1]
            codes = pred.index.get_level_values(level_instrument).astype(str).tolist()
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
