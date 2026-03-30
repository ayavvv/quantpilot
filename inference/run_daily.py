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
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily_inference")

# 路径配置
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", os.environ.get("MODELS_DIR", "/app/models")))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))
PROMOTE_LATEST = os.environ.get("PROMOTE_LATEST", "true").lower() == "true"
SIGNAL_OUTPUT_TAG = os.environ.get("SIGNAL_OUTPUT_TAG", "").strip()


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


def latest_a_share_date() -> str | None:
    """返回 instruments 中最新的 A 股日期。"""
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


def step2_predict(last_date: str):
    """加载模型，预测最新一天。"""
    log.info(f"Step 2: 模型推理 (最新日期: {last_date}) ...")

    from strategy.engine import StrategyEngine
    engine = StrategyEngine(provider_uri=str(QLIB_DATA_DIR), models_dir=str(MODEL_DIR))
    df = engine.predict_next_day(hk_mode=False)

    log.info(f"  预测完成: {len(df)} 只股票")
    if len(df) > 0:
        top5 = df[df["top5"]]["code"].tolist()
        log.info(f"  Top-5: {top5}")

    return df


def validate_signal_alignment(validated_date: str, signal_date: str, latest_a_date: str | None) -> None:
    """校验信号日期是否与本地可交易 A 股数据对齐。"""
    if latest_a_date:
        log.info(f"  Latest A-share date: {latest_a_date}")
        if validated_date != latest_a_date:
            log.warning(
                f"  日历最后日期 {validated_date} 与最新 A 股日期 {latest_a_date} 不一致，"
                "以 A 股日期作为信号新鲜度基准"
            )
        if signal_date != latest_a_date:
            raise RuntimeError(
                f"信号日期与最新 A 股日期不一致: signal={signal_date}, latest_a_share={latest_a_date}"
            )
        return

    if signal_date != validated_date:
        raise RuntimeError(
            f"信号日期与日历最后日期不一致: signal={signal_date}, validated={validated_date}"
        )


def _output_tag() -> str:
    return SIGNAL_OUTPUT_TAG or datetime.now().strftime("%Y%m%d")


def _replace_symlink(link: Path, target: Path) -> None:
    tmp_link = link.with_name(f".{link.name}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target.name)
    tmp_link.replace(link)


def _build_pred_series(df: pd.DataFrame, signal_date: str) -> pd.Series:
    ts = pd.Timestamp(signal_date)
    idx = pd.MultiIndex.from_arrays(
        [[ts] * len(df), df["code"].tolist()],
        names=["datetime", "instrument"],
    )
    return pd.Series(df["score"].values, index=idx, name="score")


def step3_output(df: pd.DataFrame, signal_date: str, promote_latest: bool = True) -> dict[str, Path]:
    """输出信号文件；仅在校验通过后才允许提升 latest。"""
    log.info("Step 3: 输出信号文件 ...")

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = _output_tag()
    paths: dict[str, Path] = {}

    csv_path = SIGNAL_DIR / f"signal_{tag}.csv"
    tmp_csv_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    out_df = df.copy()
    out_df["signal_date"] = signal_date
    out_df.to_csv(tmp_csv_path, index=False)
    tmp_csv_path.replace(csv_path)
    log.info(f"  CSV: {csv_path}")
    paths["csv"] = csv_path

    pkl_path = SIGNAL_DIR / f"pred_sh_daily_{tag}.pkl"
    tmp_pkl_path = pkl_path.with_suffix(f"{pkl_path.suffix}.tmp")
    pred_series = _build_pred_series(out_df, signal_date)
    with open(tmp_pkl_path, "wb") as f:
        pickle.dump(pred_series, f)
    tmp_pkl_path.replace(pkl_path)
    log.info(f"  PKL: {pkl_path}")
    paths["pkl"] = pkl_path

    if promote_latest:
        latest_pkl = SIGNAL_DIR / "pred_sh_latest.pkl"
        latest_csv = SIGNAL_DIR / "signal_latest.csv"
        for link, target in [(latest_pkl, pkl_path), (latest_csv, csv_path)]:
            _replace_symlink(link, target)
        log.info("  Latest 链接已更新")
        paths["latest_pkl"] = latest_pkl
        paths["latest_csv"] = latest_csv
    else:
        log.info("  Latest 链接未更新")

    return paths


def run_inference(promote_latest: bool = PROMOTE_LATEST) -> dict[str, Any]:
    """执行一次日推理，并在信号日期校验通过后更新输出。"""
    validated_date = step1_validate()
    df = step2_predict(validated_date)
    signal_date = pd.Timestamp(df.attrs.get("infer_date", validated_date)).strftime("%Y-%m-%d")
    latest_a_date = latest_a_share_date()

    log.info(f"  信号日期: {signal_date}")
    validate_signal_alignment(validated_date, signal_date, latest_a_date)

    if df.empty:
        raise RuntimeError("预测结果为空，跳过输出")

    paths = step3_output(df, signal_date, promote_latest=promote_latest)
    return {
        "validated_date": validated_date,
        "latest_a_share_date": latest_a_date,
        "signal_date": signal_date,
        "signal_count": len(df),
        "paths": paths,
    }


def main():
    log.info("=" * 50)
    log.info("QuantPilot 每日推理")
    log.info("=" * 50)
    start = datetime.now()

    try:
        result = run_inference(promote_latest=PROMOTE_LATEST)
        log.info(
            "  校验通过: validated=%s latest_a_share=%s signal=%s count=%s",
            result["validated_date"],
            result["latest_a_share_date"] or "N/A",
            result["signal_date"],
            result["signal_count"],
        )

        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"全部完成! 耗时 {elapsed:.0f} 秒")

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        log.error(f"推理失败: {e}", exc_info=True)
        log.error(f"耗时 {elapsed:.0f} 秒")
        sys.exit(1)


if __name__ == "__main__":
    main()
