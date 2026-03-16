"""Performance metrics calculation + chart generation"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from daily net return series."""
    rets = df["net_return"].values
    n_days = len(rets)
    if n_days == 0:
        return {}

    # Cumulative NAV
    nav = np.cumprod(1 + rets)
    total_return = nav[-1] - 1

    # Annualized
    ann_factor = 252 / n_days
    ann_return = (1 + total_return) ** ann_factor - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    max_dd = abs(drawdown.min())
    calmar = ann_return / max_dd if max_dd > 0 else np.nan

    # Win rate & profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / n_days if n_days > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    # Average turnover
    avg_turnover = df["turnover"].mean()

    # Total fee ratio
    total_fee = df["fee_cost"].sum()
    gross_total = df["gross_return"].sum()

    return {
        "backtest_days": n_days,
        "date_range": f"{df['signal_date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['signal_date'].iloc[-1].strftime('%Y-%m-%d')}",
        "total_return": f"{total_return:.2%}",
        "ann_return": f"{ann_return:.2%}",
        "ann_volatility": f"{ann_vol:.2%}",
        "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{max_dd:.2%}",
        "calmar": f"{calmar:.2f}",
        "win_rate": f"{win_rate:.2%}",
        "profit_factor": f"{profit_factor:.2f}",
        "avg_turnover": f"{avg_turnover:.2%}",
        "total_fee": f"{total_fee:.4f}",
        "gross_total": f"{gross_total:.4f}",
    }


def generate_charts(df: pd.DataFrame, metrics: dict, output_dir: Path,
                    top_n: int = 20, slippage: float = 0.001) -> Path:
    """Generate backtest report charts, return image path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nav = np.cumprod(1 + df["net_return"].values)
    nav_gross = np.cumprod(1 + df["gross_return"].values)
    dates = df["signal_date"].values

    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak * 100

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.suptitle(f"Backtest Report  |  Top-{top_n} Equal Weight  |  Slippage {slippage:.1%}/side",
                 fontsize=14, fontweight="bold")

    # --- NAV curve ---
    ax1.plot(dates, nav_gross, color="#bbbbbb", linewidth=1, label="Gross NAV")
    ax1.plot(dates, nav, color="#1f77b4", linewidth=1.8, label="Net NAV")
    ax1.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Key metrics annotation
    info_text = (f"Ann. Return {metrics['ann_return']}  |  Sharpe {metrics['sharpe']}  |  "
                 f"Max DD {metrics['max_drawdown']}  |  Calmar {metrics['calmar']}  |  "
                 f"Win Rate {metrics['win_rate']}")
    ax1.set_title(info_text, fontsize=10, color="#555555", pad=8)

    # --- Drawdown ---
    ax2.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # --- Daily turnover ---
    ax3.bar(dates, df["turnover"].values * 100, color="#2ca02c", alpha=0.6, width=1)
    ax3.set_ylabel("Turnover (%)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    chart_path = output_dir / "backtest_report.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {chart_path}")
    return chart_path
