"""
将 Parquet K 线数据转换为 Qlib 所需的 .bin 格式。

数据源：通过环境变量 DATA_SOURCE 指定，或使用项目根目录下的默认路径
输出：通过环境变量 QLIB_DATA_DIR 指定，默认 ~/.qlib/qlib_data/my_quant_data
"""

import os
import tempfile
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

# Qlib 依赖（可选，用于 dump_bin）
try:
    from qlib.utils import fname_to_code, code_to_fname
except ImportError:

    def fname_to_code(fname: str) -> str:
        """文件名转股票代码"""
        prefix = "_qlib_"
        if str(fname).startswith(prefix):
            fname = fname[len(prefix) :]
        return str(fname).upper()

    def code_to_fname(code: str) -> str:
        """股票代码转文件名"""
        replace_names = ["CON", "PRN", "AUX", "NUL"] + [
            f"COM{i}" for i in range(10)
        ] + [f"LPT{i}" for i in range(10)]
        if str(code).upper() in replace_names:
            return "_qlib_" + str(code)
        return str(code)


# 必需的列（Qlib 格式）
REQUIRED_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount"]
# 可选列，用于复权因子
OPTIONAL_FACTOR = "factor"

# 列映射：nas-quant-collector / Futu 可能使用的列名 -> 目标列名
COLUMN_ALIAS = {
    "time_key": "date",
    "turn_over": "amount",
    "turnover": "amount",
}


def _get_root_dir() -> Path:
    """获取项目根目录（脚本父目录的父目录）"""
    return Path(__file__).resolve().parent.parent


def _get_data_source() -> Path:
    """
    获取数据源路径。
    优先：环境变量 DATA_SOURCE > ROOT_DIR/data/kline/1d > ROOT_DIR/data/kline/K_DAY
    """
    if env_path := os.environ.get("DATA_SOURCE"):
        return Path(env_path).expanduser().resolve()
    root = _get_root_dir()
    for sub in ["data/kline/1d", "data/kline/K_DAY"]:
        p = root / sub
        if p.exists():
            return p
    return root / "data" / "kline" / "1d"


def _get_output_dir() -> Path:
    """
    Qlib 数据输出路径。
    优先使用环境变量 QLIB_DATA_DIR，默认 ~/.qlib/qlib_data/my_quant_data
    """
    env_path = os.environ.get("QLIB_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path("~/.qlib/qlib_data/my_quant_data").expanduser().resolve()


def _collect_parquet_files(data_source: Path) -> list[tuple[Path, str]]:
    """
    递归收集所有 .parquet 文件，返回 [(文件路径, 股票代码), ...]
    股票代码从父目录名推导（如 K_DAY/HK00700/data.parquet -> HK00700）
    """
    result = []
    for parquet_path in data_source.rglob("*.parquet"):
        # 父目录名即股票代码（如 HK00700、SH600000）
        code = parquet_path.parent.name
        if code and not code.startswith("."):
            result.append((parquet_path, code))
    return result


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名，确保包含 date, open, close, high, low, volume, amount"""
    df = df.copy()
    # 应用列别名
    for old, new in COLUMN_ALIAS.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    # 确保有 date
    if "date" not in df.columns and "time_key" in df.columns:
        df["date"] = df["time_key"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保必需列存在，缺失的用 NaN 填充；添加 factor 若不存在"""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if OPTIONAL_FACTOR not in df.columns:
        df[OPTIONAL_FACTOR] = 1.0
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数据：处理 NaN（Qlib 用 NaN 表示停牌等）"""
    df = df.copy()
    # 按 date 去重，保留最后一条
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="last")
    # 数值列中的 inf 转为 nan
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _fill_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    SRE 数据补全：确保 vwap 列存在且无 inf/NaN，供 Alpha158 等因子使用。
    - 若不存在 vwap 但有 amount 和 volume：vwap = amount / volume
    - 若无 amount 或 amount/volume 不可用：vwap = (open + high + low + close) / 4
    - 补全后：inf/NaN 填为 close，无 close 则填 0
    """
    df = df.copy()
    has_amount = "amount" in df.columns
    has_volume = "volume" in df.columns
    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
    need_vwap = "vwap" not in df.columns or df["vwap"].isna().all()

    if need_vwap:
        if has_amount and has_volume:
            # 避免除零：volume 为 0 或 NaN 时该行留 nan，后续用 close 填充
            mask = (pd.Series(df["volume"]).fillna(0) != 0) & (df["amount"].notna())
            vwap = np.where(mask, df["amount"].astype(float) / df["volume"].astype(float), np.nan)
        else:
            vwap = np.nan
        if (not np.any(np.isfinite(vwap))) and has_ohlc:
            vwap = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        df["vwap"] = vwap

    # 清洗：inf -> nan，再 nan 用 close 填，无 close 用 0
    v = df["vwap"].replace([np.inf, -np.inf], np.nan)
    if "close" in df.columns and df["close"].notna().any():
        v = v.fillna(df["close"])
    df["vwap"] = v.fillna(0.0)
    return df


def _prepare_flat_parquet_dir(
    parquet_items: list[tuple[Path, str]], temp_dir: Path
) -> int:
    """
    将嵌套的 parquet 转为平铺结构 temp_dir/{code}.parquet，供 Qlib dump_bin 使用。
    返回成功处理的文件数。
    """
    dump_fields = ["open", "close", "high", "low", "volume", "amount", "factor", "vwap", "pe_ratio", "turnover_rate"]
    count = 0
    for parquet_path, code in parquet_items:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  [跳过] {code}: 读取失败 - {e}")
            continue
        df = _normalize_columns(df)
        df = _ensure_required_columns(df)
        df = _clean_data(df)
        df = _fill_vwap(df)
        # 清洗 pe_ratio: 0 或负值（亏损公司）转为 NaN
        if "pe_ratio" in df.columns:
            df.loc[df["pe_ratio"] <= 0, "pe_ratio"] = np.nan
        # 只保留需要的列
        cols = [c for c in ["date"] + dump_fields if c in df.columns]
        df = df[cols]
        if df.empty or df["date"].isna().all():
            print(f"  [跳过] {code}: 无有效数据")
            continue
        out_path = temp_dir / f"{code}.parquet"
        df.to_parquet(out_path, index=False)
        count += 1
    return count


def convert_data(qlib_dir: str = None) -> None:
    """
    将数据源目录下的 Parquet 转换为 Qlib .bin 格式。
    """
    data_source = _get_data_source()
    output_dir = Path(qlib_dir).resolve() if qlib_dir else _get_output_dir()

    print(f"数据源: {data_source}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    if not data_source.exists():
        print(f"错误: 数据源目录不存在: {data_source}")
        print("提示: 请设置环境变量 DATA_SOURCE 指向行情数据目录")
        return

    parquet_items = _collect_parquet_files(data_source)
    if not parquet_items:
        print("未找到任何 .parquet 文件")
        return

    print(f"找到 {len(parquet_items)} 个 Parquet 文件")
    print("正在准备数据并转换为 Qlib 格式...")

    with tempfile.TemporaryDirectory(prefix="qlib_loader_") as temp_dir:
        temp_path = Path(temp_dir)
        n_prepared = _prepare_flat_parquet_dir(parquet_items, temp_path)
        if n_prepared == 0:
            print("无有效数据可转换")
            return

        print(f"已准备 {n_prepared} 个股票数据，开始 dump 到 Qlib...")

        use_standalone = True
        try:
            try:
                from scripts.dump_bin import DumpDataAll
            except ImportError:
                from qlib.scripts.dump_bin import DumpDataAll
            use_standalone = False
            dumper = DumpDataAll(
                data_path=str(temp_path),
                qlib_dir=str(output_dir),
                include_fields="open,close,high,low,volume,amount,factor,vwap,pe_ratio,turnover_rate",
                date_field_name="date",
                file_suffix=".parquet",
            )
            dumper.dump()
        except ImportError:
            # Qlib 未安装或 dump_bin 不可用，使用 subprocess
            import subprocess
            import sys

            dump_script = None
            for candidate in [
                Path(__file__).resolve().parent.parent / "scripts" / "dump_bin.py",
            ]:
                if candidate.exists():
                    dump_script = candidate
                    break
            if dump_script is None:
                try:
                    import qlib
                    qlib_path = Path(qlib.__file__).parent
                    found = list(qlib_path.rglob("dump_bin.py"))
                    if found:
                        dump_script = found[0]
                except Exception:
                    pass
            if dump_script and dump_script.exists():
                cmd = [
                    sys.executable,
                    str(dump_script),
                    "dump_all",
                    "--data_path",
                    str(temp_path),
                    "--qlib_dir",
                    str(output_dir),
                    "--include_fields",
                    "open,close,high,low,volume,amount,factor,vwap,pe_ratio,turnover_rate",
                    "--date_field_name",
                    "date",
                    "--file_suffix",
                    ".parquet",
                ]
                subprocess.run(cmd, check=True)
                use_standalone = False
            else:
                use_standalone = True
        except Exception as e:
            print(f"使用 Qlib dump_bin 失败: {e}")
            print("使用内置转换逻辑...")
            use_standalone = True

        if use_standalone:
            _dump_bin_standalone(temp_path, output_dir, parquet_items)

    print("-" * 50)
    print("转换完成!")
    print(f"Qlib 数据已写入: {output_dir}")


def _dump_bin_standalone(
    data_path: Path,
    qlib_dir: Path,
    parquet_items: list[tuple[Path, str]],
) -> None:
    """
    不依赖 Qlib 脚本，自行实现 dump 到 .bin 格式（兼容 Qlib 数据结构）。
    """
    freq = "day"
    date_field = "date"
    dump_fields = ["open", "close", "high", "low", "volume", "amount", "factor", "vwap", "pe_ratio", "turnover_rate"]
    bin_suffix = ".bin"
    daily_fmt = "%Y-%m-%d"
    calendars_dir = qlib_dir / "calendars"
    features_dir = qlib_dir / "features"
    instruments_dir = qlib_dir / "instruments"
    instruments_sep = "\t"

    # 1. 收集所有数据，构建 calendar 和 instruments
    all_dates = set()
    instruments_rows = []

    flat_files = list(data_path.glob("*.parquet"))
    for i, fp in enumerate(flat_files):
        code = fname_to_code(fp.stem.lower()).upper()
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if date_field not in df.columns:
            continue
        df[date_field] = pd.to_datetime(df[date_field])
        dates = df[date_field].dropna().unique()
        if len(dates) == 0:
            continue
        all_dates.update(pd.Timestamp(d) for d in dates)
        start_d = df[date_field].min().strftime(daily_fmt)
        end_d = df[date_field].max().strftime(daily_fmt)
        instruments_rows.append((code, start_d, end_d))
        if (i + 1) % 50 == 0:
            print(f"  已扫描 {i + 1}/{len(flat_files)} 个文件...")

    calendars_list = sorted(all_dates)
    if not calendars_list:
        print("无有效交易日历")
        return

    # 2. 保存 calendars
    calendars_dir.mkdir(parents=True, exist_ok=True)
    cal_path = calendars_dir / f"{freq}.txt"
    with open(cal_path, "w", encoding="utf-8") as f:
        for d in calendars_list:
            f.write(pd.Timestamp(d).strftime(daily_fmt) + "\n")
    print(f"  已生成 calendars: {cal_path}")

    # 3. 保存 instruments（all + 按市场拆分）
    instruments_dir.mkdir(parents=True, exist_ok=True)
    inst_path = instruments_dir / "all.txt"
    with open(inst_path, "w", encoding="utf-8") as f:
        for row in instruments_rows:
            f.write(instruments_sep.join(row) + "\n")
    print(f"  已生成 instruments: {inst_path} ({len(instruments_rows)} 只)")

    for market_tag, prefix in [("sh", "SH."), ("sz", "SZ."), ("hk", "HK.")]:
        market_rows = [r for r in instruments_rows if r[0].startswith(prefix)]
        mp = instruments_dir / f"{market_tag}.txt"
        with open(mp, "w", encoding="utf-8") as f:
            for row in market_rows:
                f.write(instruments_sep.join(row) + "\n")
        print(f"  已生成 instruments: {mp} ({len(market_rows)} 只)")

    # 4. 对每个股票 dump bin
    for i, fp in enumerate(flat_files):
        code = fname_to_code(fp.stem.lower()).upper()
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if date_field not in df.columns:
            continue
        df[date_field] = pd.to_datetime(df[date_field])
        df = df.drop_duplicates(subset=[date_field], keep="last")
        df = df.set_index(date_field).sort_index()

        # 对齐到 calendar
        start_idx = bisect.bisect_left(calendars_list, df.index.min())
        end_idx = bisect.bisect_right(calendars_list, df.index.max())
        if start_idx >= end_idx:
            continue
        subset_cal = calendars_list[start_idx:end_idx]
        aligned = df.reindex(subset_cal)
        date_index = start_idx

        feat_dir = features_dir / code_to_fname(code).lower()
        feat_dir.mkdir(parents=True, exist_ok=True)

        for field in dump_fields:
            if field not in aligned.columns:
                continue
            arr = np.array(aligned[field], dtype=np.float32)
            arr = np.where(np.isfinite(arr), arr, np.nan)
            bin_path = feat_dir / f"{field.lower()}.{freq}{bin_suffix}"
            # Qlib bin 格式: [date_index, v1, v2, ...]，date_index 为该股票在 calendar 中的起始索引
            data = np.hstack([np.array([date_index], dtype="<f4"), arr]).astype("<f4")
            data.tofile(str(bin_path))

        if (i + 1) % 50 == 0 or i == len(flat_files) - 1:
            print(f"  已 dump {i + 1}/{len(flat_files)} 个股票")
