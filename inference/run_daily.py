"""
每日推理流水线：验证数据 → 模型预测 → 输出信号。

流程:
1. 验证 Qlib bin 数据 (/qlib_data/) 是否就绪
2. 加载 LightGBM 模型预测最新一天全部股票分数
3. 输出信号文件到 /data/signals/

数据源: NAS collector 直写 Qlib bin → scheduler sync 到本地
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily_inference")

# 路径配置
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))


def step1_validate() -> str:
    """验证 Qlib bin 数据是否就绪。"""
    log.info("Step 1: 验证 Qlib bin 数据 ...")

    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    if not cal_path.exists():
        raise RuntimeError(f"Qlib 日历文件不存在: {cal_path}")
    lines = cal_path.read_text().strip().splitlines()
    if not lines:
        raise RuntimeError("Qlib 日历为空")
    last_date = lines[-1].strip()
    log.info(f"  日历范围: {lines[0]} ~ {last_date}, 共 {len(lines)} 天")

    inst_path = QLIB_DATA_DIR / "instruments" / "all.txt"
    if inst_path.exists():
        n = len(inst_path.read_text().strip().splitlines())
        log.info(f"  股票总数: {n} 只")

    feat_dir = QLIB_DATA_DIR / "features"
    if not feat_dir.exists() or not any(feat_dir.iterdir()):
        raise RuntimeError(f"Qlib features 目录为空: {feat_dir}")

    return last_date


def step2_predict(last_date: str):
    """加载模型，预测最新一天。"""
    log.info(f"Step 2: 模型推理 (最新日期: {last_date}) ...")

    from strategy.engine import StrategyEngine
    engine = StrategyEngine(provider_uri=str(QLIB_DATA_DIR))
    df = engine.predict_next_day(hk_mode=False)

    log.info(f"  预测完成: {len(df)} 只股票")
    if len(df) > 0:
        top5 = df[df["top5"]]["code"].tolist()
        log.info(f"  Top-5: {top5}")

    return df


def step3_output(df, last_date: str):
    """输出信号文件。"""
    log.info("Step 3: 输出信号文件 ...")

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    import pandas as pd

    # CSV 格式（人可读）
    csv_path = SIGNAL_DIR / f"signal_{today}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  CSV: {csv_path}")

    # pkl 格式（程序可读，兼容 trade_daily.py）
    ts = pd.Timestamp(last_date)
    idx = pd.MultiIndex.from_arrays(
        [[ts] * len(df), df["code"].tolist()],
        names=["datetime", "instrument"],
    )
    pred_series = pd.Series(df["score"].values, index=idx, name="score")

    pkl_path = SIGNAL_DIR / f"pred_sh_daily_{today}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pred_series, f)
    log.info(f"  PKL: {pkl_path}")

    # 同时更新 latest 软链接
    latest_pkl = SIGNAL_DIR / "pred_sh_latest.pkl"
    latest_csv = SIGNAL_DIR / "signal_latest.csv"
    for link, target in [(latest_pkl, pkl_path), (latest_csv, csv_path)]:
        link.unlink(missing_ok=True)
        link.symlink_to(target.name)
    log.info(f"  Latest 链接已更新")


def main():
    log.info("=" * 50)
    log.info("QuantPilot 每日推理")
    log.info("=" * 50)
    start = datetime.now()

    try:
        last_date = step1_validate()
        df = step2_predict(last_date)

        if df.empty:
            log.warning("预测结果为空，跳过输出")
        else:
            step3_output(df, last_date)

        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"全部完成! 耗时 {elapsed:.0f} 秒")

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error(f"推理失败: {e}", exc_info=True)
        log.error(f"耗时 {elapsed:.0f} 秒")
        sys.exit(1)


if __name__ == "__main__":
    main()
