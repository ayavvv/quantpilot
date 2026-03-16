"""
将 Parquet K 线数据转换为 Qlib 所需的 .bin 格式。

数据源：nas-quant-collector 采集的数据
- 本地：ROOT_DIR/data/kline/1d 或 data/kline/K_DAY
- NAS/SMB：通过环境变量 DATA_SOURCE 指定挂载路径

输出：~/.qlib/qlib_data/my_quant_data
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
        prefix = "_qlib_"
        if str(fname).startswith(prefix):
            fname = fname[len(prefix):]
        return str(fname).upper()

    def code_to_fname(code: str) -> str:
        replace_names = ["CON", "PRN", "AUX", "NUL"] + [
            f"COM{i}" for i in range(10)
        ] + [f"LPT{i}" for i in range(10)]
        if str(code).upper() in replace_names:
            return "_qlib_" + str(code)
        return str(code)


# 必需的列（Qlib 格式）
REQUIRED_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount"]
OPTIONAL_FACTOR = "factor"

# 列映射
COLUMN_ALIAS = {
    "time_key": "date",
    "turn_over": "amount",
    "turnover": "amount",
}


def _get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _get_data_source() -> Path:
    root = _get_root_dir()
    if env_path := os.environ.get("DATA_SOURCE"):
        return Path(env_path).expanduser().resolve()
    for sub in ["data/kline/1d", "data/kline/K_DAY"]:
        p = root / sub
        if p.exists():
            return p
    return root / "data" / "kline" / "1d"


def _get_output_dir() -> Path:
    return Path("~/.qlib/qlib_data/my_quant_data").expanduser().resolve()


def _collect_parquet_files(data_source: Path) -> list[tuple[Path, str]]:
    result = []
    for parquet_path in data_source.rglob("*.parquet"):
        code = parquet_path.parent.name
        if code and not code.startswith("."):
            result.append((parquet_path, code))
    return result


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for old, new in COLUMN_ALIAS.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "date" not in df.columns and "time_key" in df.columns:
        df["date"] = df["time_key"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if OPTIONAL_FACTOR not in df.columns:
        df[OPTIONAL_FACTOR] = 1.0
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="last")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _load_fundamentals_snapshot(data_source: Path) -> dict[str, pd.DataFrame]:
    candidates = [
        data_source.parent.parent / "fundamentals" / "daily_snapshot.parquet",
        data_source.parent / "fundamentals" / "daily_snapshot.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "code" in df.columns and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    return {code: grp for code, grp in df.groupby("code")}
            except Exception as e:
                print(f"  [警告] 读取基本面快照失败: {e}")
    return {}


def _fill_fundamentals(df: pd.DataFrame, fund_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    if fund_df is not None and "pb_ratio" in fund_df.columns and "date" in df.columns:
        fund_pb = fund_df[["date", "pb_ratio"]].copy()
        fund_pb["date"] = pd.to_datetime(fund_pb["date"])
        df = df.merge(fund_pb, on="date", how="left", suffixes=("", "_fund"))
        if "pb_ratio_fund" in df.columns:
            if "pb_ratio" not in df.columns:
                df["pb_ratio"] = df["pb_ratio_fund"]
            else:
                df["pb_ratio"] = df["pb_ratio"].fillna(df["pb_ratio_fund"])
            df.drop(columns=["pb_ratio_fund"], inplace=True)
    for col in ["pe_ratio", "turnover_rate", "pb_ratio"]:
        if col in df.columns:
            df[col] = df[col].astype(float).replace(0, np.nan)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        else:
            df[col] = np.nan
    return df


def _load_macro_data(data_source: Path) -> pd.DataFrame | None:
    macro_map = {
        "MACRO.VIX": "vix",
        "MACRO.DXY": "dxy",
        "MACRO.TNX": "tnx",
    }
    etf_map = {
        "US.SPY": "spy",
        "US.QQQ": "qqq",
    }
    all_map = {**macro_map, **etf_map}
    kday_dir = data_source
    merged = None
    for code, col_name in all_map.items():
        fp = kday_dir / code / "data.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
            df = _normalize_columns(df)
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "close"]].drop_duplicates(subset=["date"], keep="last")
            df = df.rename(columns={"close": col_name})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="date", how="outer")
        except Exception as e:
            print(f"  [警告] 加载宏观数据 {code} 失败: {e}")
    if merged is not None:
        merged = merged.sort_values("date").reset_index(drop=True)
        for col in merged.columns:
            if col != "date":
                merged[col] = merged[col].ffill()
        print(f"  已加载宏观数据: {[c for c in merged.columns if c != 'date']}, {len(merged)} 天")
    return merged


def _merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    if macro_df is None:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(macro_df, on="date", how="left")
    macro_cols = [c for c in macro_df.columns if c != "date"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def _fill_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_amount = "amount" in df.columns
    has_volume = "volume" in df.columns
    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
    need_vwap = "vwap" not in df.columns or df["vwap"].isna().all()

    if need_vwap:
        if has_amount and has_volume:
            mask = (pd.Series(df["volume"]).fillna(0) != 0) & (df["amount"].notna())
            vwap = np.where(mask, df["amount"].astype(float) / df["volume"].astype(float), np.nan)
        else:
            vwap = np.nan
        if (not np.any(np.isfinite(vwap))) and has_ohlc:
            vwap = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        df["vwap"] = vwap

    v = df["vwap"].replace([np.inf, -np.inf], np.nan)
    if "close" in df.columns and df["close"].notna().any():
        v = v.fillna(df["close"])
    df["vwap"] = v.fillna(0.0)
    return df


def _prepare_flat_parquet_dir(
    parquet_items: list[tuple[Path, str]], temp_dir: Path,
    fund_map: dict[str, pd.DataFrame] = None,
    macro_df: pd.DataFrame = None,
    whitelist: set[str] | None = None,
) -> int:
    """
    将嵌套的 parquet 转为平铺结构。
    whitelist: 如果提供，只转换在白名单中的股票代码。
    """
    macro_cols = [c for c in (macro_df.columns if macro_df is not None else []) if c != "date"]
    dump_fields = ["open", "close", "high", "low", "volume", "amount", "factor", "vwap",
                   "pe_ratio", "turnover_rate", "pb_ratio"] + macro_cols
    if fund_map is None:
        fund_map = {}
    count = 0
    for parquet_path, code in parquet_items:
        # 跳过宏观/ETF 数据本身
        if code.startswith("MACRO.") or code.startswith("US."):
            continue
        # 白名单过滤
        if whitelist is not None and code not in whitelist:
            continue
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  [跳过] {code}: 读取失败 - {e}")
            continue
        df = _normalize_columns(df)
        df = _ensure_required_columns(df)
        df = _clean_data(df)
        df = _fill_vwap(df)
        df = _fill_fundamentals(df, fund_df=fund_map.get(code))
        df = _merge_macro(df, macro_df)
        cols = [c for c in ["date"] + dump_fields if c in df.columns]
        df = df[cols]
        if df.empty or df["date"].isna().all():
            print(f"  [跳过] {code}: 无有效数据")
            continue
        out_path = temp_dir / f"{code}.parquet"
        df.to_parquet(out_path, index=False)
        count += 1
    return count


def convert_data(whitelist: set[str] | None = None) -> None:
    """
    将数据源目录下的 Parquet 转换为 Qlib .bin 格式。

    Args:
        whitelist: 可选股票白名单（通过 stock_filter 生成），None 表示不过滤。
    """
    data_source = _get_data_source()
    output_dir = _get_output_dir()

    print(f"数据源: {data_source}")
    print(f"输出目录: {output_dir}")
    if whitelist is not None:
        print(f"股票白名单: {len(whitelist)} 只")
    print("-" * 50)

    if not data_source.exists():
        print(f"错误: 数据源目录不存在: {data_source}")
        return

    parquet_items = _collect_parquet_files(data_source)
    if not parquet_items:
        print("未找到任何 .parquet 文件")
        return

    print(f"找到 {len(parquet_items)} 个 Parquet 文件")

    fund_map = _load_fundamentals_snapshot(data_source)
    if fund_map:
        print(f"已加载 {len(fund_map)} 只股票的基本面快照数据")

    macro_df = _load_macro_data(data_source)

    print("正在准备数据并转换为 Qlib 格式...")

    with tempfile.TemporaryDirectory(prefix="qlib_loader_") as temp_dir:
        temp_path = Path(temp_dir)
        n_prepared = _prepare_flat_parquet_dir(
            parquet_items, temp_path,
            fund_map=fund_map, macro_df=macro_df,
            whitelist=whitelist,
        )
        if n_prepared == 0:
            print("无有效数据可转换")
            return

        base_fields = ["open", "close", "high", "low", "volume", "amount", "factor", "vwap",
                       "pe_ratio", "turnover_rate", "pb_ratio"]
        macro_cols = [c for c in (macro_df.columns if macro_df is not None else []) if c != "date"]
        all_fields = base_fields + macro_cols
        include_fields_str = ",".join(all_fields)

        print(f"已准备 {n_prepared} 个股票数据，开始 dump 到 Qlib...")
        if macro_cols:
            print(f"  宏观因子列: {macro_cols}")

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
                include_fields=include_fields_str,
                date_field_name="date",
                file_suffix=".parquet",
            )
            dumper.dump()
        except ImportError:
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
                    "--data_path", str(temp_path),
                    "--qlib_dir", str(output_dir),
                    "--include_fields", include_fields_str,
                    "--date_field_name", "date",
                    "--file_suffix", ".parquet",
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
    freq = "day"
    date_field = "date"
    base_dump_fields = ["open", "close", "high", "low", "volume", "amount", "factor", "vwap",
                        "pe_ratio", "turnover_rate", "pb_ratio"]
    extra_cols = set()
    for fp in data_path.glob("*.parquet"):
        try:
            cols = pd.read_parquet(fp, columns=None).columns.tolist()
            extra_cols.update(c for c in cols if c not in base_dump_fields and c != date_field)
        except Exception:
            pass
        break
    dump_fields = base_dump_fields + sorted(extra_cols)
    bin_suffix = ".bin"
    daily_fmt = "%Y-%m-%d"
    calendars_dir = qlib_dir / "calendars"
    features_dir = qlib_dir / "features"
    instruments_dir = qlib_dir / "instruments"
    instruments_sep = "\t"

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
        if (i + 1) % 100 == 0:
            print(f"  已扫描 {i + 1}/{len(flat_files)} 个文件...")

    calendars_list = sorted(all_dates)
    if not calendars_list:
        print("无有效交易日历")
        return

    calendars_dir.mkdir(parents=True, exist_ok=True)
    cal_path = calendars_dir / f"{freq}.txt"
    with open(cal_path, "w", encoding="utf-8") as f:
        for d in calendars_list:
            f.write(pd.Timestamp(d).strftime(daily_fmt) + "\n")
    print(f"  已生成 calendars: {cal_path}")

    instruments_dir.mkdir(parents=True, exist_ok=True)
    inst_path = instruments_dir / "all.txt"
    with open(inst_path, "w", encoding="utf-8") as f:
        for row in instruments_rows:
            f.write(instruments_sep.join(row) + "\n")
    print(f"  已生成 instruments: {inst_path}")

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
            data = np.hstack([np.array([date_index], dtype="<f4"), arr]).astype("<f4")
            data.tofile(str(bin_path))

        if (i + 1) % 100 == 0 or i == len(flat_files) - 1:
            print(f"  已 dump {i + 1}/{len(flat_files)} 个股票")
