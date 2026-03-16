"""
量化控制台 - Streamlit Dashboard
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# set_page_config 必须是第一个 Streamlit 命令
st.set_page_config(page_title="量化控制台", page_icon="📈", layout="wide")

# 添加项目根目录到 sys.path，以便能 import strategy 和 converter
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 导入兄弟目录 strategy 中的模块
from strategy.loader import convert_data
from strategy.engine import StrategyEngine


def _get_data_dir() -> Path:
    """获取数据目录：优先 DATA_SOURCE 环境变量，否则 data/kline/1d 或 data/kline/K_DAY"""
    root = _project_root
    if env_path := os.environ.get("DATA_SOURCE"):
        return Path(env_path).expanduser().resolve()
    for sub in ["data/kline/1d", "data/kline/K_DAY"]:
        p = root / sub
        if p.exists():
            return p
    return root / "data" / "kline" / "1d"


@st.cache_data
def _get_predictions() -> tuple:
    """
    加载 StrategyEngine 并调用 predict_next_day()，获取预测结果及 Top 5 的收盘价。
    返回 (DataFrame, last_date_str) 供展示使用。
    """
    engine = StrategyEngine()
    pred_df = engine.predict_next_day()

    if pred_df.empty:
        return pred_df, None

    # 获取 Top 5 的股票代码
    top5_df = pred_df[pred_df["top5"]].head(5)
    if top5_df.empty:
        return pred_df, None

    # 获取最近交易日
    from datetime import datetime, timedelta
    from qlib.data import D

    end_time = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    cal = D.calendar(start_time="2020-01-01", end_time=end_time, freq="day")
    if not cal:
        return pred_df, None

    last_date = str(cal[-1])[:10]
    codes = top5_df["code"].tolist()

    # 获取最新收盘价
    pred_df = pred_df.copy()
    try:
        close_df = D.features(codes, ["$close"], start_time=last_date, end_time=last_date)
        if close_df is not None and not close_df.empty:
            if hasattr(close_df.index, "get_level_values") and hasattr(close_df.index, "names"):
                level = "instrument" if "instrument" in close_df.index.names else close_df.index.names[0]
                close_series = close_df.groupby(level=level)["$close"].first()
                pred_df["close"] = pred_df["code"].map(close_series)
            else:
                pred_df["close"] = float("nan")
        else:
            pred_df["close"] = float("nan")
    except Exception:
        pred_df["close"] = float("nan")

    return pred_df, last_date


def _run_convert_data_with_progress():
    """在进度条下执行数据同步"""
    progress_bar = st.progress(0, text="正在同步数据...")
    status_text = st.empty()

    try:
        status_text.info("正在收集 Parquet 文件并转换为 Qlib 格式，请稍候...")
        progress_bar.progress(20, text="正在准备数据...")

        # convert_data 是同步的，无进度回调，用线程 + 模拟进度
        import threading
        done = threading.Event()
        error_holder = [None]  # 用于捕获异常

        def run():
            try:
                convert_data()
            except Exception as e:
                error_holder[0] = e
            finally:
                done.set()

        t = threading.Thread(target=run)
        t.start()

        # 模拟进度更新
        pct = 20
        while not done.wait(0.5) and pct < 90:
            pct = min(pct + 5, 90)
            progress_bar.progress(pct / 100, text=f"同步中... {pct}%")

        t.join(timeout=0.1)
        progress_bar.progress(1.0, text="完成")
        status_text.empty()

        if error_holder[0]:
            raise error_holder[0]

        st.success("数据同步完成！")
    except Exception as e:
        st.error(f"数据同步失败: {e}")
    finally:
        progress_bar.empty()
        if "status_text" in dir():
            try:
                status_text.empty()
            except Exception:
                pass


def _run_train_model():
    """执行模型训练"""
    try:
        with st.spinner("正在训练模型，请耐心等待..."):
            engine = StrategyEngine()
            metrics = engine.train_model()
        st.success("模型训练完成！")
        st.json(metrics)
        # 训练完成后清除预测缓存，下次加载会重新预测
        st.cache_data.clear()
    except Exception as e:
        st.error(f"模型训练失败: {e}")


def _render_data_overview():
    """渲染数据概览 Tab"""
    data_dir = _get_data_dir()

    if not data_dir.exists():
        st.warning(f"数据目录不存在: {data_dir}")
        st.info("请先进行数据同步，或检查 DATA_SOURCE 环境变量。")
        return

    parquet_files = list(data_dir.rglob("*.parquet"))
    codes = set()
    latest_date = None
    date_columns = ["date", "time_key"]

    for fp in parquet_files:
        code = fp.parent.name
        if code and not code.startswith("."):
            codes.add(code)
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        cols = [c for c in date_columns if c in df.columns]
        if cols:
            df = df[[cols[0]]].copy()
            df[cols[0]] = pd.to_datetime(df[cols[0]], errors="coerce")
            max_d = df[cols[0]].max()
            if pd.notna(max_d):
                if latest_date is None or max_d > latest_date:
                    latest_date = max_d

    st.metric("股票数量", len(codes))
    if latest_date is not None:
        st.metric("最新数据日期", pd.Timestamp(latest_date).strftime("%Y-%m-%d"))
    else:
        st.metric("最新数据日期", "—")

    st.caption(f"数据目录: `{data_dir}`")


st.title("📈 量化控制台")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("操作")
    if st.button("🔄 数据同步"):
        _run_convert_data_with_progress()

    if st.button("🎯 重新训练模型"):
        _run_train_model()

# ========== 主界面 Tabs ==========
tab1, tab2 = st.tabs(["🔮 明日预测 (Signals)", "📊 数据概览"])

with tab1:
    try:
        pred_df, last_date = _get_predictions()
    except FileNotFoundError as e:
        st.warning(str(e))
        st.info("请先在侧边栏点击「重新训练模型」完成模型训练。")
        pred_df = pd.DataFrame()
        last_date = None

    if not pred_df.empty:
        top5 = pred_df[pred_df["top5"]].head(5)

        if last_date:
            st.caption(f"预测基准日: {last_date} | 共 {len(pred_df)} 只股票")

        # Top 5 高亮展示
        st.subheader("🏆 Top 5 推荐")
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                close_val = row.get("close", float("nan"))
                if pd.isna(close_val):
                    close_str = "—"
                else:
                    close_str = f"¥{close_val:.2f}"
                st.metric(
                    label=row["code"],
                    value=close_str,
                    delta=f"得分 {row['score']:.4f}",
                    delta_color="off",
                )

        # 完整表格
        st.subheader("📋 完整预测列表")
        display_df = pred_df.drop(columns=["top5"], errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    elif pred_df.empty and "pred_df" in dir() and not str(pred_df).startswith("error"):
        st.info("暂无预测数据。请先完成数据同步和模型训练。")

with tab2:
    _render_data_overview()
