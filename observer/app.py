"""
QuantPilot Observer - System monitoring dashboard (Streamlit).

Reads all data from Qlib bin format + signal/model/report files.
Self-contained: includes backtest metric computation for pred_sh.pkl.
"""

from __future__ import annotations

import os
import pickle
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/data/reports"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/data/models"))

FREQ = "day"
DEFAULT_FIELDS = ["open", "close", "high", "low", "volume", "amount", "change_rate"]

st.set_page_config(page_title="QuantPilot Observer", layout="wide")


# ---------------------------------------------------------------------------
# Qlib bin reader (lightweight, self-contained — no external import needed)
# ---------------------------------------------------------------------------
class _QlibReader:
    """Minimal Qlib bin reader embedded in observer (avoids converter dep)."""

    def __init__(self, qlib_dir: Path):
        self.qlib_dir = qlib_dir
        self._calendar: list[str] | None = None

    @property
    def calendar(self) -> list[str]:
        if self._calendar is None:
            cal_path = self.qlib_dir / "calendars" / "day.txt"
            if cal_path.exists():
                self._calendar = cal_path.read_text().strip().splitlines()
            else:
                self._calendar = []
        return self._calendar

    @property
    def latest_date(self) -> str | None:
        cal = self.calendar
        return cal[-1] if cal else None

    def list_instruments(self, market: str = "all") -> dict[str, tuple[str, str]]:
        inst_path = self.qlib_dir / "instruments" / f"{market}.txt"
        if not inst_path.exists():
            return {}
        result = {}
        for line in inst_path.read_text().strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                result[parts[0]] = (parts[1], parts[2])
        return result

    def _read_bin(self, bin_path: Path):
        if not bin_path.exists():
            return None
        data = bin_path.read_bytes()
        if len(data) < 4:
            return None
        n_floats = len(data) // 4
        values = np.array(struct.unpack(f"<{n_floats}f", data), dtype=np.float32)
        start_idx = int(values[0])
        return start_idx, values[1:]

    def read_field(self, code: str, field: str) -> pd.Series:
        fname = code.lower()
        bin_path = self.qlib_dir / "features" / fname / f"{field}.{FREQ}.bin"
        result = self._read_bin(bin_path)
        if result is None:
            return pd.Series(dtype="float64")
        start_idx, values = result
        cal = self.calendar
        end_idx = start_idx + len(values)
        if end_idx > len(cal):
            end_idx = len(cal)
            values = values[: end_idx - start_idx]
        dates = cal[start_idx:end_idx]
        return pd.Series(values.astype("float64"), index=dates, name=field)

    def read_stock(self, code: str, fields: list[str] | None = None) -> pd.DataFrame:
        fields = fields or DEFAULT_FIELDS
        series = {}
        for f in fields:
            s = self.read_field(code, f)
            if not s.empty:
                series[f] = s
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index()

    def read_field_matrix(self, codes: list[str], field: str,
                          start_date: str | None = None,
                          end_date: str | None = None) -> pd.DataFrame:
        frames = {}
        for code in codes:
            s = self.read_field(code, field)
            if not s.empty:
                frames[code] = s
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames).sort_index()
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]
        return df


@st.cache_resource(ttl=300)
def get_reader() -> _QlibReader | None:
    if not (QLIB_DATA_DIR / "calendars" / "day.txt").exists():
        return None
    return _QlibReader(QLIB_DATA_DIR)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_data_status() -> dict:
    reader = get_reader()
    if reader is None or not reader.calendar:
        return {"total": 0, "sh": 0, "sz": 0, "hk": 0,
                "latest_date": "N/A", "calendar_days": 0,
                "cal_start": "N/A"}

    instruments = reader.list_instruments("all")
    sh = [c for c in instruments if c.startswith("SH")]
    sz = [c for c in instruments if c.startswith("SZ")]
    hk = [c for c in instruments if c.startswith("HK")]

    return {
        "total": len(instruments),
        "sh": len(sh),
        "sz": len(sz),
        "hk": len(hk),
        "latest_date": reader.latest_date or "N/A",
        "cal_start": reader.calendar[0] if reader.calendar else "N/A",
        "calendar_days": len(reader.calendar),
    }


@st.cache_data(ttl=60)
def load_signal_list() -> list[dict]:
    """List all signal CSV files with metadata."""
    if not SIGNAL_DIR.exists():
        return []
    results = []
    for f in sorted(SIGNAL_DIR.glob("signal_*.csv"), reverse=True):
        if f.name == "signal_latest.csv":
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        try:
            df = pd.read_csv(f, nrows=1)
            n_rows = sum(1 for _ in open(f)) - 1
        except Exception:
            n_rows = 0
        results.append({
            "file": f.name,
            "date": f.stem.replace("signal_", ""),
            "stocks": n_rows,
            "updated": mtime.strftime("%Y-%m-%d %H:%M"),
            "path": str(f),
        })
    return results


@st.cache_data(ttl=60)
def load_latest_signal() -> pd.DataFrame | None:
    """Load latest signal CSV."""
    if not SIGNAL_DIR.exists():
        return None
    latest = SIGNAL_DIR / "signal_latest.csv"
    if not latest.exists():
        csvs = sorted(SIGNAL_DIR.glob("signal_*.csv"))
        if not csvs:
            return None
        latest = csvs[-1]
    try:
        return pd.read_csv(latest)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_model_list() -> list[dict]:
    """List model files."""
    if not MODEL_DIR.exists():
        return []
    results = []
    for f in sorted(MODEL_DIR.glob("*.pkl")):
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / (1024 * 1024)
        results.append({
            "name": f.name,
            "size": f"{size_mb:.1f} MB" if size_mb >= 1 else f"{stat.st_size / 1024:.0f} KB",
            "updated": mtime.strftime("%Y-%m-%d %H:%M"),
            "path": str(f),
        })
    return results


@st.cache_data(ttl=300)
def load_pred_pkl(path: str) -> dict | None:
    """Load pred_sh.pkl and compute backtest-style metrics."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            pred = pickle.load(f)

        if not isinstance(pred, pd.Series):
            return None

        dates = sorted(pred.index.get_level_values("datetime").unique())
        instruments = sorted(pred.index.get_level_values("instrument").unique())

        return {
            "n_dates": len(dates),
            "n_instruments": len(instruments),
            "date_start": dates[0].strftime("%Y-%m-%d"),
            "date_end": dates[-1].strftime("%Y-%m-%d"),
            "pred": pred,
        }
    except Exception:
        return None


def compute_backtest_from_pred(pred: pd.Series, reader: _QlibReader,
                               top_n: int = 5) -> dict | None:
    """Run lightweight backtest from pred Series + Qlib close prices."""
    signal_dates = sorted(pred.index.get_level_values("datetime").unique())
    instruments = sorted(pred.index.get_level_values("instrument").unique())

    # Load close prices
    start_str = signal_dates[0].strftime("%Y-%m-%d")
    end_str = signal_dates[-1].strftime("%Y-%m-%d")
    close_df = reader.read_field_matrix(instruments, "close", start_str)
    if close_df.empty:
        return None

    price_dates = sorted(close_df.index)
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    records = []
    for t in signal_dates:
        t_str = t.strftime("%Y-%m-%d")
        if t_str not in date_to_idx:
            continue
        idx = date_to_idx[t_str]
        if idx + 2 >= len(price_dates):
            continue
        t1 = price_dates[idx + 1]
        t2 = price_dates[idx + 2]

        day_scores = pred.xs(t, level="datetime")
        if isinstance(day_scores, pd.DataFrame):
            day_scores = day_scores.iloc[:, 0]
        day_scores = day_scores.dropna().sort_values(ascending=False)

        # Select top-N with valid close prices
        candidates = []
        for code in day_scores.index:
            if code not in close_df.columns:
                continue
            c1 = close_df.at[t1, code] if t1 in close_df.index else np.nan
            c2 = close_df.at[t2, code] if t2 in close_df.index else np.nan
            if pd.notna(c1) and pd.notna(c2) and c1 > 0:
                candidates.append((code, c2 / c1 - 1))
            if len(candidates) >= top_n:
                break

        if not candidates:
            continue
        gross_ret = np.mean([r for _, r in candidates])
        # Simple fee estimate: 0.15% per side for turnover
        fee = 0.003  # ~0.3% round trip as rough estimate
        net_ret = gross_ret - fee
        records.append({"date": t_str, "gross": gross_ret, "net": net_ret,
                        "n": len(candidates)})

    if not records:
        return None

    df = pd.DataFrame(records)
    nav_net = np.cumprod(1 + df["net"].values)
    nav_gross = np.cumprod(1 + df["gross"].values)
    total_return = nav_net[-1] - 1
    n_days = len(df)
    ann_factor = 252 / n_days
    ann_return = (1 + total_return) ** ann_factor - 1
    ann_vol = np.std(df["net"].values, ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    peak = np.maximum.accumulate(nav_net)
    drawdown = (nav_net - peak) / peak
    max_dd = abs(drawdown.min())
    calmar = ann_return / max_dd if max_dd > 0 else 0
    wins = df["net"].values[df["net"].values > 0]
    win_rate = len(wins) / n_days if n_days > 0 else 0

    return {
        "n_days": n_days,
        "date_range": f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}",
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "nav_dates": df["date"].tolist(),
        "nav_net": nav_net.tolist(),
        "nav_gross": nav_gross.tolist(),
        "drawdown": drawdown.tolist(),
        "daily_returns": df["net"].tolist(),
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.title("QuantPilot")
    st.caption("A-share Quantitative Trading System")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Paths")
    st.code(f"QLIB_DATA: {QLIB_DATA_DIR}\n"
            f"SIGNALS:   {SIGNAL_DIR}\n"
            f"MODELS:    {MODEL_DIR}\n"
            f"REPORTS:   {REPORT_DIR}", language=None)


# Load data
data_status = load_data_status()

# Title
st.header("QuantPilot Observer")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Stocks", f"{data_status['total']:,}")
with c2:
    st.metric("SH / SZ / HK",
              f"{data_status['sh']} / {data_status['sz']} / {data_status['hk']}")
with c3:
    st.metric("Calendar Days", f"{data_status['calendar_days']:,}")
with c4:
    st.metric("Data Range",
              f"{data_status['cal_start'][:7]} ~ {data_status['latest_date'][:7]}"
              if data_status['latest_date'] != 'N/A' else "N/A")
with c5:
    latest = data_status["latest_date"]
    today = datetime.now().strftime("%Y-%m-%d")
    if latest == today:
        st.metric("Freshness", "Today")
        st.success("Up to date")
    elif latest != "N/A":
        st.metric("Latest Date", latest)
        st.warning("May be stale")
    else:
        st.metric("Freshness", "N/A")
        st.error("No data")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_backtest, tab_signals, tab_models, tab_reports = st.tabs(
    ["Data", "Backtest", "Signals", "Models", "Reports"]
)

# --- Tab: Data ---
with tab_data:
    reader = get_reader()
    if reader is None or data_status["total"] == 0:
        st.warning("No Qlib bin data found. Run collector first.")
    else:
        st.subheader("Market Breakdown")
        market_df = pd.DataFrame({
            "Market": ["Shanghai (SH)", "Shenzhen (SZ)", "Hong Kong (HK)"],
            "Stocks": [data_status["sh"], data_status["sz"], data_status["hk"]],
        })
        st.dataframe(market_df, use_container_width=True, hide_index=True)

        # Calendar stats
        st.subheader("Calendar")
        cal = reader.calendar
        if cal:
            recent_dates = cal[-20:]
            st.write(f"Total: **{len(cal):,}** trading days "
                     f"({cal[0]} ~ {cal[-1]})")
            st.write("Last 20 trading days:")
            col_dates = st.columns(5)
            for i, d in enumerate(reversed(recent_dates)):
                col_dates[i % 5].code(d)

        # Stock data viewer
        st.subheader("Stock Data Viewer")
        all_instruments = reader.list_instruments("all")
        # Show SH stocks first (more commonly used)
        sh_codes = sorted(c for c in all_instruments if c.startswith("SH"))
        sz_codes = sorted(c for c in all_instruments if c.startswith("SZ"))
        hk_codes = sorted(c for c in all_instruments if c.startswith("HK"))

        market_choice = st.radio("Market", ["SH", "SZ", "HK"], horizontal=True)
        code_list = {"SH": sh_codes, "SZ": sz_codes, "HK": hk_codes}[market_choice]

        if code_list:
            selected_code = st.selectbox(
                "Select stock",
                code_list,
                index=min(0, len(code_list) - 1),
            )
            if selected_code:
                stock_df = reader.read_stock(selected_code)
                if not stock_df.empty:
                    date_range = all_instruments.get(selected_code, ("?", "?"))
                    st.write(f"**{selected_code}** | "
                             f"{len(stock_df)} records | "
                             f"{stock_df.index[0]} ~ {stock_df.index[-1]} | "
                             f"Instrument range: {date_range[0]} ~ {date_range[1]}")

                    # Price chart (last 120 trading days)
                    chart_data = stock_df[["close"]].tail(120).copy()
                    chart_data.index = pd.to_datetime(chart_data.index)
                    st.line_chart(chart_data, y="close", use_container_width=True)

                    # Recent data table
                    st.write("Recent 20 trading days:")
                    display_df = stock_df.tail(20).copy()
                    # Format numbers
                    for col in ["open", "close", "high", "low"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].round(2)
                    if "volume" in display_df.columns:
                        display_df["volume"] = display_df["volume"].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                    if "amount" in display_df.columns:
                        display_df["amount"] = display_df["amount"].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                    if "change_rate" in display_df.columns:
                        display_df["change_rate"] = display_df["change_rate"].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) else "")
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning(f"No bin data for {selected_code}")
        else:
            st.info(f"No {market_choice} instruments found.")


# --- Tab: Backtest ---
with tab_backtest:
    st.subheader("Model Prediction & Backtest")

    # Find pred_sh.pkl
    pred_path = None
    for candidate in [
        MODEL_DIR / "pred_sh.pkl",
        SIGNAL_DIR / "pred_latest.pkl",
    ]:
        if candidate.exists():
            pred_path = candidate
            break

    if pred_path is None:
        st.info("No prediction file (pred_sh.pkl) found. "
                "Run training or inference first to generate predictions.")
        st.caption(f"Looked in: {MODEL_DIR}, {SIGNAL_DIR}")
    else:
        pred_info = load_pred_pkl(str(pred_path))
        if pred_info is None:
            st.error(f"Failed to load {pred_path}")
        else:
            st.success(f"Loaded: **{pred_path.name}** | "
                       f"{pred_info['n_dates']} dates | "
                       f"{pred_info['n_instruments']} instruments | "
                       f"{pred_info['date_start']} ~ {pred_info['date_end']}")

            # Backtest parameters
            col_a, col_b = st.columns(2)
            with col_a:
                bt_top_n = st.number_input("Top N", min_value=1, max_value=50,
                                           value=5, step=1)
            with col_b:
                st.write("")  # spacer

            reader = get_reader()
            if reader is None:
                st.warning("Qlib data not available for backtest.")
            else:
                with st.spinner("Running backtest ..."):
                    bt = compute_backtest_from_pred(
                        pred_info["pred"], reader, top_n=bt_top_n)

                if bt is None:
                    st.warning("Backtest produced no results. "
                               "Check if Qlib close prices cover the prediction period.")
                else:
                    # Key metrics
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Total Return", f"{bt['total_return']:.2%}")
                    m2.metric("Ann. Return", f"{bt['ann_return']:.2%}")
                    m3.metric("Sharpe", f"{bt['sharpe']:.2f}")
                    m4.metric("Max Drawdown", f"{bt['max_drawdown']:.2%}")
                    m5.metric("Calmar", f"{bt['calmar']:.2f}")
                    m6.metric("Win Rate", f"{bt['win_rate']:.1%}")

                    st.caption(f"Backtest: {bt['n_days']} trading days | "
                               f"{bt['date_range']} | "
                               f"Ann. Vol: {bt['ann_volatility']:.2%}")

                    # NAV curve
                    st.subheader("NAV Curve")
                    nav_df = pd.DataFrame({
                        "Net NAV": bt["nav_net"],
                        "Gross NAV": bt["nav_gross"],
                    }, index=pd.to_datetime(bt["nav_dates"]))
                    st.line_chart(nav_df, use_container_width=True)

                    # Drawdown
                    st.subheader("Drawdown")
                    dd_df = pd.DataFrame({
                        "Drawdown": [d * 100 for d in bt["drawdown"]],
                    }, index=pd.to_datetime(bt["nav_dates"]))
                    st.area_chart(dd_df, use_container_width=True, color="#d62728")

                    # Daily returns distribution
                    st.subheader("Daily Returns Distribution")
                    ret_df = pd.DataFrame({
                        "Daily Return (%)": [r * 100 for r in bt["daily_returns"]],
                    })
                    st.bar_chart(
                        pd.DataFrame({
                            "Return (%)": [r * 100 for r in bt["daily_returns"]],
                        }, index=pd.to_datetime(bt["nav_dates"])),
                        use_container_width=True,
                    )


# --- Tab: Signals ---
with tab_signals:
    st.subheader("Daily Signals")

    signal_df = load_latest_signal()
    if signal_df is not None and not signal_df.empty:
        # Latest signal info
        latest_csv = SIGNAL_DIR / "signal_latest.csv"
        if latest_csv.exists():
            mtime = datetime.fromtimestamp(latest_csv.stat().st_mtime)
            st.success(f"Latest signal updated: {mtime.strftime('%Y-%m-%d %H:%M')}")

        # Top stocks
        if "score" in signal_df.columns:
            st.subheader("Top Predictions")
            top_df = signal_df.head(10).copy()
            if "rank" in top_df.columns:
                display_cols = ["rank", "code", "score"]
                if "top5" in top_df.columns:
                    display_cols.append("top5")
                st.dataframe(top_df[display_cols], use_container_width=True,
                             hide_index=True)
            else:
                st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Score distribution
        if "score" in signal_df.columns and len(signal_df) > 10:
            st.subheader("Score Distribution")
            hist_data = signal_df["score"].dropna()
            st.bar_chart(hist_data.value_counts(bins=30).sort_index(),
                         use_container_width=True)

        # Full table (expandable)
        with st.expander(f"Full signal table ({len(signal_df)} stocks)"):
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
    else:
        st.info("No signal files found. Run inference to generate signals.")

    # Signal history
    st.subheader("Signal History")
    signal_list = load_signal_list()
    if signal_list:
        st.dataframe(
            pd.DataFrame(signal_list)[["date", "stocks", "updated"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No historical signal files.")


# --- Tab: Models ---
with tab_models:
    st.subheader("Model Files")

    model_list = load_model_list()
    if model_list:
        st.dataframe(
            pd.DataFrame(model_list),
            use_container_width=True, hide_index=True,
        )

        # pred_sh.pkl detail
        for m in model_list:
            if "pred" in m["name"]:
                pred_info = load_pred_pkl(m["path"])
                if pred_info:
                    st.write(f"**{m['name']}**: "
                             f"{pred_info['n_dates']} dates, "
                             f"{pred_info['n_instruments']} instruments, "
                             f"{pred_info['date_start']} ~ {pred_info['date_end']}")
    else:
        st.info("No model files found. Run training to generate models.")
        st.caption(f"Expected path: {MODEL_DIR}")


# --- Tab: Reports ---
with tab_reports:
    st.subheader("Report Files")

    if REPORT_DIR.exists():
        reports = sorted(REPORT_DIR.glob("*"), reverse=True)
        if reports:
            report_data = []
            for f in reports[:20]:
                stat = f.stat()
                report_data.append({
                    "name": f.name,
                    "size": f"{stat.st_size / 1024:.0f} KB",
                    "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(
                pd.DataFrame(report_data),
                use_container_width=True, hide_index=True,
            )

            # Preview HTML reports
            html_reports = [f for f in reports if f.suffix == ".html"]
            if html_reports:
                selected_report = st.selectbox(
                    "Preview report",
                    [f.name for f in html_reports[:10]],
                )
                if selected_report:
                    report_path = REPORT_DIR / selected_report
                    if report_path.exists():
                        html_content = report_path.read_text(encoding="utf-8",
                                                              errors="replace")
                        st.components.v1.html(html_content, height=600,
                                              scrolling=True)
        else:
            st.info("No reports generated yet.")
    else:
        st.info("Report directory not found. Run reporter to generate reports.")
        st.caption(f"Expected path: {REPORT_DIR}")
