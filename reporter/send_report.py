"""
Daily quant report: data collection status + model signals + trade status.
Supports SMTP, local sendmail, and Mail.app delivery, and always saves HTML locally.
"""

import os
import pickle
import shutil
import smtplib
import ssl
import subprocess
import sys
import mimetypes
import json
from datetime import datetime, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

import pandas as pd
from jinja2 import Template

SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/data/reports"))
REPORTER_ENV_PATH = Path(os.environ.get("REPORTER_ENV_FILE", Path(__file__).with_name(".env")))
CAPITAL_FLOW_OVERLAY_CSV = Path(
    os.environ.get(
        "CAPITAL_FLOW_OVERLAY_CSV",
        str(SIGNAL_DIR.parent / "output" / "futu_capital_flow_signal_overlay_latest.csv"),
    )
)
CAPITAL_FLOW_EVAL_SUMMARY_CSV = Path(
    os.environ.get(
        "CAPITAL_FLOW_EVAL_SUMMARY_CSV",
        str(SIGNAL_DIR.parent / "output" / "futu_capital_flow_eval_latest" / "summary.csv"),
    )
)
CAPITAL_FLOW_GATE_JSON = Path(
    os.environ.get(
        "CAPITAL_FLOW_GATE_JSON",
        str(SIGNAL_DIR.parent / "output" / "futu_capital_flow_eval_latest" / "gate.json"),
    )
)
MAJOR_MONEY_DIGEST_JSON = Path(
    os.environ.get(
        "MAJOR_MONEY_DIGEST_JSON",
        str(SIGNAL_DIR.parent / "output" / "major_money_digest_latest.json"),
    )
)
TRADE_START_TIME = time(14, 50)

REPORT_TEMPLATE = """
<html>
<head>
<style>
body { font-family: -apple-system, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; }
h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }
h2 { color: #16213e; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #16213e; color: white; }
tr:nth-child(even) { background-color: #f8f9fa; }
.top5 { background-color: #d4edda !important; font-weight: bold; }
.metric { display: inline-block; margin: 8px 16px 8px 0; }
.metric-value { font-size: 24px; font-weight: bold; color: #16213e; }
.metric-label { font-size: 12px; color: #666; }
.ok { color: #28a745; }
.warn { color: #ffc107; }
.error { color: #dc3545; }
.muted { color: #666; }
.flow-confirm { background-color: #e6f4ea !important; }
.flow-risk { background-color: #fdecea !important; }
.flow-watch { background-color: #fff8e1 !important; }
.gate-box { border-left: 4px solid #ccc; padding: 8px 12px; margin: 12px 0; background: #f8f9fa; }
.gate-review { border-left-color: #dc3545; }
.gate-advisory { border-left-color: #ffc107; }
.coverage-missing { background-color: #fdecea !important; }
</style>
</head>
<body>
<h1>QuantPilot Daily Report - {{ date }}</h1>

<h2>1. Data Collection</h2>
<div class="metric">
    <div class="metric-value {{ 'ok' if data_ok else 'error' }}">{{ data_status }}</div>
    <div class="metric-label">Collection Status</div>
</div>
<div class="metric">
    <div class="metric-value">{{ stock_count }}</div>
    <div class="metric-label">Stocks</div>
</div>
<div class="metric">
    <div class="metric-value">{{ data_date }}</div>
    <div class="metric-label">Latest Data Date</div>
</div>

<h2>2. Model Signals ({{ signal_date }})</h2>
{% if signal_count > 0 %}
<div class="metric">
    <div class="metric-value">{{ signal_count }}</div>
    <div class="metric-label">Predicted Stocks</div>
</div>
<p><strong>Top 10:</strong></p>
<table>
<tr><th>Rank</th><th>Code</th><th>Score</th></tr>
{% for row in top10 %}
<tr class="{{ 'top5' if row.rank <= 5 else '' }}">
    <td>{{ row.rank }}</td><td>{{ row.code }}</td><td>{{ row.score_fmt }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p class="warn">No signal data today</p>
{% endif %}

<h2>3. Market-Wide Major Money ({{ major_money_date }})</h2>
{% if major_money_available %}
<p class="muted">{{ major_money_message }}</p>
<table>
<tr><th>Market</th><th>Source</th><th>Coverage</th><th>Entry</th><th>Entry Amount</th><th>Exit</th><th>Exit Amount</th><th>Net</th></tr>
{% for row in major_money_markets %}
<tr class="{{ row.row_class }}">
    <td>{{ row.market }}</td>
    <td>{{ row.source }}</td>
    <td>{{ row.coverage }}</td>
    <td>{{ row.entry_count }}</td>
    <td>{{ row.entry_amount }}</td>
    <td>{{ row.exit_count }}</td>
    <td>{{ row.exit_amount }}</td>
    <td>{{ row.net_amount }}</td>
</tr>
{% endfor %}
</table>
<p><strong>Top Entries:</strong></p>
<table>
<tr><th>Market</th><th>Code</th><th>Name</th><th>Main Flow</th></tr>
{% for row in major_money_top_entries %}
<tr class="flow-confirm"><td>{{ row.market }}</td><td>{{ row.code }}</td><td>{{ row.name }}</td><td>{{ row.main_flow }}</td></tr>
{% endfor %}
</table>
<p><strong>Top Exits:</strong></p>
<table>
<tr><th>Market</th><th>Code</th><th>Name</th><th>Main Flow</th></tr>
{% for row in major_money_top_exits %}
<tr class="flow-risk"><td>{{ row.market }}</td><td>{{ row.code }}</td><td>{{ row.name }}</td><td>{{ row.main_flow }}</td></tr>
{% endfor %}
</table>
{% else %}
<p class="warn">{{ major_money_message }}</p>
{% endif %}

<h2>4. Futu Capital Flow ({{ capital_flow_date }})</h2>
{% if capital_flow_available %}
<p class="muted">{{ capital_flow_message }}</p>
<table>
<tr><th>Label</th><th>Count</th></tr>
{% for row in capital_flow_counts %}
<tr class="{{ row.row_class }}"><td>{{ row.label }}</td><td>{{ row.count }}</td></tr>
{% endfor %}
</table>
<p><strong>Top Model Candidates:</strong></p>
<table>
<tr><th>Rank</th><th>Code</th><th>Label</th><th>Latest Main</th><th>5D Main</th></tr>
{% for row in capital_flow_top %}
<tr class="{{ row.row_class }}">
    <td>{{ row.rank }}</td>
    <td>{{ row.code }}</td>
    <td>{{ row.label }}</td>
    <td>{{ row.latest_main }}</td>
    <td>{{ row.main_5d }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p class="warn">{{ capital_flow_message }}</p>
{% endif %}

<h2>5. Capital Flow Validation</h2>
<div class="gate-box {{ capital_flow_gate_class }}">
    <strong>Rule Gate:</strong> {{ capital_flow_gate_message }}
</div>
{% if capital_flow_eval_available %}
<p class="muted">{{ capital_flow_eval_message }}</p>
<table>
<tr><th>Label</th><th>Horizon</th><th>Dates</th><th>Avg Return</th><th>Alpha</th><th>Hit Rate</th></tr>
{% for row in capital_flow_eval_rows %}
<tr class="{{ row.row_class }}">
    <td>{{ row.label }}</td>
    <td>{{ row.horizon }}</td>
    <td>{{ row.date_count }}</td>
    <td>{{ row.avg_return }}</td>
    <td>{{ row.avg_alpha }}</td>
    <td>{{ row.avg_hit_rate }}</td>
</tr>
{% endfor %}
</table>
{% else %}
<p class="warn">{{ capital_flow_eval_message }}</p>
{% endif %}

<h2>6. Trading Status</h2>
<p>{{ trade_status }}</p>

<hr>
<p style="color: #999; font-size: 12px;">
QuantPilot Auto Report | Generated: {{ gen_time }}
</p>
</body>
</html>
"""


def check_data_status():
    """Check Qlib bin data collection status."""
    qlib_dir = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        return {"data_ok": False, "data_status": "Qlib data missing", "stock_count": 0, "data_date": "N/A"}

    lines = cal_path.read_text().strip().splitlines()
    calendar_date = lines[-1].strip() if lines else "N/A"

    inst_path = qlib_dir / "instruments" / "all.txt"
    stock_count = 0
    a_share_date = None
    if inst_path.exists():
        for line in inst_path.read_text().strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            stock_count += 1
            code, _, end_date = parts[:3]
            if code.startswith(("SH.", "SZ.")) and (a_share_date is None or end_date > a_share_date):
                a_share_date = end_date

    return {
        "data_ok": stock_count > 1000,
        "data_status": "OK" if stock_count > 1000 else "Warning",
        "stock_count": stock_count,
        "data_date": a_share_date or calendar_date,
    }


def check_signal_status():
    """Check signal status."""
    today = datetime.now().strftime("%Y%m%d")
    signal_file = SIGNAL_DIR / f"signal_{today}.csv"
    latest_file = SIGNAL_DIR / "signal_latest.csv"
    latest_pred = SIGNAL_DIR / "pred_sh_latest.pkl"

    target = signal_file if signal_file.exists() else latest_file
    actual_signal_date = None
    if latest_pred.exists():
        try:
            with open(latest_pred, "rb") as f:
                pred = pickle.load(f)
            dates = sorted(pred.index.get_level_values("datetime").unique())
            if dates:
                actual_signal_date = dates[-1].strftime("%Y-%m-%d")
        except Exception:
            pass

    if not target.exists():
        return {"signal_count": 0, "signal_date": today, "top10": []}

    df = pd.read_csv(target)
    signal_date = actual_signal_date or (
        str(df["signal_date"].iloc[0]) if "signal_date" in df.columns and not df.empty else today
    )
    top10_df = df.head(10)
    top10 = []
    for _, row in top10_df.iterrows():
        top10.append({
            "rank": int(row["rank"]),
            "code": row["code"],
            "score_fmt": f"{row['score']:.4f}",
        })

    return {
        "signal_count": len(df),
        "signal_date": signal_date,
        "top10": top10,
    }


def _format_money(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"

    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}bn"
    if abs_amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f}m"
    if abs_amount >= 1_000:
        return f"{amount / 1_000:.1f}k"
    return f"{amount:.0f}"


def _format_money_with_currency(value, currency: str) -> str:
    formatted = _format_money(value)
    if formatted == "N/A" or not currency:
        return formatted
    return f"{formatted} {currency}"


def _flow_row_class(label: str) -> str:
    normalized = label.lower()
    if "risk" in normalized or "outflow" in normalized:
        return "flow-risk"
    if "confirm" in normalized:
        return "flow-confirm"
    if "watch" in normalized:
        return "flow-watch"
    return ""


def _safe_int(value, fallback: int) -> int:
    try:
        if pd.isna(value):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def check_major_money_digest_status(
    digest_json: Path | None = None,
    top_n: int = 8,
):
    """Summarise market-wide major-money digest for the daily report."""
    target = digest_json or Path(os.environ.get("MAJOR_MONEY_DIGEST_JSON", str(MAJOR_MONEY_DIGEST_JSON)))
    if not target.exists():
        return {
            "major_money_available": False,
            "major_money_date": "N/A",
            "major_money_message": f"No market-wide major-money digest found at {target}.",
            "major_money_markets": [],
            "major_money_top_entries": [],
            "major_money_top_exits": [],
        }

    try:
        digest = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "major_money_available": False,
            "major_money_date": "N/A",
            "major_money_message": f"Could not read market-wide major-money digest: {exc}",
            "major_money_markets": [],
            "major_money_top_entries": [],
            "major_money_top_exits": [],
        }

    markets = digest.get("markets", [])
    if not isinstance(markets, list) or not markets:
        return {
            "major_money_available": False,
            "major_money_date": "N/A",
            "major_money_message": "Market-wide major-money digest is empty.",
            "major_money_markets": [],
            "major_money_top_entries": [],
            "major_money_top_exits": [],
        }

    market_rows = []
    entry_rows = []
    exit_rows = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        currency = str(market.get("currency") or "")
        available = bool(market.get("available"))
        ok_rows = _safe_int(market.get("ok_rows"), 0)
        total_rows = _safe_int(market.get("total_rows"), 0)
        market_rows.append(
            {
                "market": market.get("market", "N/A"),
                "source": market.get("source") or "missing",
                "coverage": f"{ok_rows}/{total_rows}" if total_rows else "missing",
                "entry_count": _safe_int(market.get("entry_count"), 0),
                "entry_amount": _format_money_with_currency(market.get("entry_amount"), currency),
                "exit_count": _safe_int(market.get("exit_count"), 0),
                "exit_amount": _format_money_with_currency(market.get("exit_amount"), currency),
                "net_amount": _format_money_with_currency(market.get("net_amount"), currency),
                "row_class": "" if available else "coverage-missing",
            }
        )
        top_entries = market.get("top_entries")
        if not isinstance(top_entries, list):
            top_entries = []
        for row in top_entries:
            entry_rows.append(
                {
                    "market": market.get("market", "N/A"),
                    "code": row.get("code", "N/A"),
                    "name": row.get("name", ""),
                    "main_flow_raw": row.get("main_flow"),
                    "main_flow": _format_money_with_currency(row.get("main_flow"), currency),
                }
            )
        top_exits = market.get("top_exits")
        if not isinstance(top_exits, list):
            top_exits = []
        for row in top_exits:
            exit_rows.append(
                {
                    "market": market.get("market", "N/A"),
                    "code": row.get("code", "N/A"),
                    "name": row.get("name", ""),
                    "main_flow_raw": row.get("main_flow"),
                    "main_flow": _format_money_with_currency(row.get("main_flow"), currency),
                }
            )

    entry_rows = sorted(entry_rows, key=lambda row: float(row.get("main_flow_raw") or 0.0), reverse=True)[:top_n]
    exit_rows = sorted(exit_rows, key=lambda row: float(row.get("main_flow_raw") or 0.0))[:top_n]
    available_count = _safe_int(digest.get("available_market_count"), 0)
    market_count = _safe_int(digest.get("market_count"), len(markets))
    return {
        "major_money_available": available_count > 0,
        "major_money_date": digest.get("flow_date") or "N/A",
        "major_money_message": (
            f"Loaded {available_count}/{market_count} market-wide flow source(s). "
            "Counts use vendor/proxy major-money fields and remain advisory."
        ),
        "major_money_markets": market_rows,
        "major_money_top_entries": entry_rows,
        "major_money_top_exits": exit_rows,
    }


def check_capital_flow_status(
    overlay_csv: Path | None = None,
    top_n: int = 10,
):
    """Summarise Futu capital-flow overlay for the daily report."""
    target = overlay_csv or Path(os.environ.get("CAPITAL_FLOW_OVERLAY_CSV", str(CAPITAL_FLOW_OVERLAY_CSV)))
    if not target.exists():
        return {
            "capital_flow_available": False,
            "capital_flow_date": "N/A",
            "capital_flow_message": f"No Futu capital-flow overlay found at {target}.",
            "capital_flow_counts": [],
            "capital_flow_top": [],
        }

    try:
        df = pd.read_csv(target)
    except Exception as exc:
        return {
            "capital_flow_available": False,
            "capital_flow_date": "N/A",
            "capital_flow_message": f"Could not read Futu capital-flow overlay: {exc}",
            "capital_flow_counts": [],
            "capital_flow_top": [],
        }

    if df.empty or "capital_flow_label" not in df.columns:
        return {
            "capital_flow_available": False,
            "capital_flow_date": "N/A",
            "capital_flow_message": "Futu capital-flow overlay is empty or missing labels.",
            "capital_flow_counts": [],
            "capital_flow_top": [],
        }

    date_col = "capital_flow_latest_date" if "capital_flow_latest_date" in df.columns else "signal_date"
    if date_col in df.columns:
        dates = sorted(str(value) for value in df[date_col].dropna().unique())
        flow_date = dates[-1] if dates else "N/A"
    else:
        flow_date = "N/A"

    labels = df["capital_flow_label"].fillna("unknown").astype(str)
    counts = [
        {
            "label": label,
            "count": int(count),
            "row_class": _flow_row_class(label),
        }
        for label, count in labels.value_counts().items()
    ]

    ranked = df.copy()
    if "model_rank" in ranked.columns:
        ranked = ranked.sort_values("model_rank", kind="stable")

    top_rows = []
    for idx, (_, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
        label = str(row.get("capital_flow_label", "unknown"))
        top_rows.append(
            {
                "rank": _safe_int(row.get("model_rank"), idx),
                "code": row.get("code", "N/A"),
                "label": label,
                "latest_main": _format_money(row.get("latest_main_in_flow")),
                "main_5d": _format_money(row.get("main_5d_sum")),
                "row_class": _flow_row_class(label),
            }
        )

    return {
        "capital_flow_available": True,
        "capital_flow_date": flow_date,
        "capital_flow_message": f"Loaded {len(df)} model candidate(s); capital flow is advisory and not an auto-trade rule.",
        "capital_flow_counts": counts,
        "capital_flow_top": top_rows,
    }


def check_capital_flow_eval_status(
    summary_csv: Path | None = None,
    top_n: int = 12,
):
    """Summarise archived Futu capital-flow forward-return validation."""
    target = summary_csv or Path(
        os.environ.get("CAPITAL_FLOW_EVAL_SUMMARY_CSV", str(CAPITAL_FLOW_EVAL_SUMMARY_CSV))
    )
    if not target.exists():
        return {
            "capital_flow_eval_available": False,
            "capital_flow_eval_message": f"No capital-flow validation summary found at {target}.",
            "capital_flow_eval_rows": [],
        }

    try:
        df = pd.read_csv(target)
    except pd.errors.EmptyDataError:
        return {
            "capital_flow_eval_available": False,
            "capital_flow_eval_message": "Capital-flow validation has no forward-return samples yet.",
            "capital_flow_eval_rows": [],
        }
    except Exception as exc:
        return {
            "capital_flow_eval_available": False,
            "capital_flow_eval_message": f"Could not read capital-flow validation summary: {exc}",
            "capital_flow_eval_rows": [],
        }

    required = {"capital_flow_label", "horizon", "date_count", "avg_return", "avg_alpha", "avg_hit_rate"}
    if df.empty or not required.issubset(df.columns):
        return {
            "capital_flow_eval_available": False,
            "capital_flow_eval_message": "Capital-flow validation has no forward-return samples yet.",
            "capital_flow_eval_rows": [],
        }

    rows = []
    ranked = df.sort_values(["horizon", "capital_flow_label"], kind="stable").head(top_n)
    for _, row in ranked.iterrows():
        label = str(row.get("capital_flow_label", "unknown"))
        rows.append(
            {
                "label": label,
                "horizon": _safe_int(row.get("horizon"), 0),
                "date_count": _safe_int(row.get("date_count"), 0),
                "avg_return": _format_percent(row.get("avg_return")),
                "avg_alpha": _format_percent(row.get("avg_alpha")),
                "avg_hit_rate": _format_percent(row.get("avg_hit_rate")),
                "row_class": _flow_row_class(label),
            }
        )

    return {
        "capital_flow_eval_available": True,
        "capital_flow_eval_message": (
            f"Loaded {len(df)} validation row(s); use this before promoting advisory labels to trade rules."
        ),
        "capital_flow_eval_rows": rows,
    }


def check_capital_flow_gate_status(gate_json: Path | None = None):
    """Summarise the evidence gate for capital-flow rule promotion."""
    target = gate_json or Path(os.environ.get("CAPITAL_FLOW_GATE_JSON", str(CAPITAL_FLOW_GATE_JSON)))
    if not target.exists():
        return {
            "capital_flow_gate_available": False,
            "capital_flow_gate_class": "gate-advisory",
            "capital_flow_gate_message": f"No capital-flow promotion gate found at {target}; keep labels advisory.",
        }

    try:
        gate = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "capital_flow_gate_available": False,
            "capital_flow_gate_class": "gate-advisory",
            "capital_flow_gate_message": f"Could not read capital-flow promotion gate: {exc}; keep labels advisory.",
        }

    action = str(gate.get("overall_action", "keep_advisory"))
    message = str(gate.get("message", "Keep capital-flow labels advisory."))
    criteria = gate.get("criteria", {}) if isinstance(gate.get("criteria"), dict) else {}
    min_dates = criteria.get("min_date_count", "N/A")
    decisions = gate.get("decisions", [])
    candidates = []
    if isinstance(decisions, list):
        candidates = [
            str(item.get("label"))
            for item in decisions
            if isinstance(item, dict) and str(item.get("status", "")).startswith("candidate_")
        ]

    if action in {"review_filter", "review_boost"}:
        gate_class = "gate-review"
    else:
        gate_class = "gate-advisory"

    suffix = f" Min dates={min_dates}."
    if candidates:
        suffix += f" Candidate label(s): {', '.join(candidates)}."
    return {
        "capital_flow_gate_available": True,
        "capital_flow_gate_class": gate_class,
        "capital_flow_gate_message": f"{message}{suffix}",
    }


def check_trade_status(now: datetime | None = None, trade_log: Path | None = None) -> str:
    """Summarise today's automatic trade status from trade.log."""
    current = now or datetime.now()
    today = current.strftime("%Y-%m-%d")
    log_path = trade_log or Path.home() / "quantpilot/logs/trade.log"
    default_status = "Trading module active (simulation mode)."
    trade_day = current.weekday() < 5

    if not log_path.exists():
        if not trade_day:
            return "No automatic trade scheduled today."
        if current.time() < TRADE_START_TIME:
            return "Today's 14:50 automatic trade has not started yet."
        return default_status

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return default_status

    filled = [line for line in lines if today in line and "  OK " in line]
    failed = [line for line in lines if today in line and "  FAIL " in line]
    today_errors = [line for line in lines if today in line and ("行情失败" in line or "ERROR" in line)]
    run_starts = [line for line in lines if today in line and "run_trade: start" in line]
    run_done = [line for line in lines if today in line and "run_trade: done" in line]

    if failed:
        return f"WARNING: Today filled {len(filled)} order(s), failed {len(failed)} order(s)."
    if filled:
        return f"Today: {len(filled)} order(s) filled (simulation)."
    if today_errors:
        return f"WARNING: Trading ran but had {len(today_errors)} error(s). Check trade.log."
    if run_done:
        return "Today: automatic trade run completed with no orders filled."
    if run_starts:
        return "WARNING: Automatic trade started but no completion record was found."
    if not trade_day:
        return "No automatic trade scheduled today."
    if current.time() < TRADE_START_TIME:
        return "Today's 14:50 automatic trade has not started yet."
    return "WARNING: No trading execution found today."


def load_env_defaults():
    """Load reporter .env defaults without overriding existing env vars."""
    if not REPORTER_ENV_PATH.exists():
        return
    for raw_line in REPORTER_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def save_report_locally(
    html_content: str,
    filename: str | None = None,
    report_dir: str | os.PathLike[str] | Path | None = None,
) -> Path:
    target_dir = Path(report_dir) if report_dir is not None else Path(os.environ.get("REPORT_DIR", str(REPORT_DIR)))
    target_dir.mkdir(parents=True, exist_ok=True)
    report_name = filename or f"report_{datetime.now().strftime('%Y%m%d')}.html"
    report_path = target_dir / report_name
    report_path.write_text(html_content, encoding="utf-8")
    print(f"Report saved: {report_path}")
    return report_path


def email_config():
    load_env_defaults()
    smtp_user = os.environ.get("SMTP_USER", "")
    return {
        "report_delivery_method": os.environ.get("REPORT_DELIVERY_METHOD", "auto").lower(),
        "smtp_host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", "465")),
        "smtp_user": smtp_user,
        "smtp_password": os.environ.get("SMTP_PASSWORD", ""),
        "report_to": os.environ.get("REPORT_TO", ""),
        "report_from": os.environ.get("REPORT_FROM", smtp_user),
        "mail_app_from": os.environ.get("MAIL_APP_FROM", os.environ.get("REPORT_FROM", smtp_user)),
        "smtp_timeout": int(os.environ.get("SMTP_TIMEOUT_SECONDS", "10")),
        "smtp_retries": int(os.environ.get("SMTP_RETRIES", "1")),
        "sendmail_fallback": os.environ.get("SENDMAIL_FALLBACK", "true").lower() == "true",
        "mail_app_fallback": os.environ.get("MAIL_APP_FALLBACK", "true").lower() == "true",
    }


def log_config_status(config):
    print(
        "SMTP config status: "
        f"method={config['report_delivery_method']} "
        f"host={config['smtp_host']} port={config['smtp_port']} "
        f"user={'set' if config['smtp_user'] else 'missing'} "
        f"password={'set' if config['smtp_password'] else 'missing'} "
        f"report_to={'set' if config['report_to'] else 'missing'} "
        f"report_from={'set' if config['report_from'] else 'missing'} "
        f"mail_app_from={'set' if config['mail_app_from'] else 'missing'} "
        f"sendmail_fallback={'on' if config['sendmail_fallback'] else 'off'} "
        f"mail_app_fallback={'on' if config['mail_app_fallback'] else 'off'}"
    )
    return [
        name
        for name, value in {
            "SMTP_USER": config["smtp_user"],
            "SMTP_PASSWORD": config["smtp_password"],
            "REPORT_TO": config["report_to"],
        }.items()
        if not value
    ]


def _normalize_attachment_paths(attachment_paths):
    if not attachment_paths:
        return []
    normalized = []
    for path in attachment_paths:
        candidate = Path(path)
        if candidate.exists():
            normalized.append(candidate)
    return normalized


def _attach_files(msg, attachment_paths):
    for path in _normalize_attachment_paths(attachment_paths):
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type:
            maintype, subtype = mime_type.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"

        with open(path, "rb") as f:
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", f"attachment; filename={path.name}")
        msg.attach(attachment)


def build_message(html_content, subject, report_from, report_to, attachment_paths=None):
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = report_from
    msg["To"] = report_to
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(html_content, "html", "utf-8"))
    msg.attach(body)
    _attach_files(msg, attachment_paths)
    return msg


def send_via_smtp(config, msg):
    context = ssl.create_default_context()
    attempts = max(1, config["smtp_retries"])
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            if config["smtp_port"] == 465:
                with smtplib.SMTP_SSL(
                    config["smtp_host"],
                    config["smtp_port"],
                    timeout=config["smtp_timeout"],
                    context=context,
                ) as server:
                    server.login(config["smtp_user"], config["smtp_password"])
                    server.send_message(msg)
            else:
                with smtplib.SMTP(
                    config["smtp_host"],
                    config["smtp_port"],
                    timeout=config["smtp_timeout"],
                ) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(config["smtp_user"], config["smtp_password"])
                    server.send_message(msg)
            print(f"Email sent to {config['report_to']} via SMTP")
            return True, ""
        except Exception as exc:
            last_error = exc
            print(f"SMTP attempt {attempt}/{attempts} failed: {exc}")
    return False, str(last_error) if last_error else "unknown SMTP error"


def send_via_mail_app(subject, report_to, report_from, report_path, attachment_paths=None):
    if sys.platform != "darwin":
        return False, "Mail.app fallback only available on macOS"
    if not report_to:
        return False, "REPORT_TO missing"

    apple_script = r'''
on run argv
    set subjectLine to item 1 of argv
    set recipientAddress to item 2 of argv
    set preferredSender to item 3 of argv
    set plainBody to item 4 of argv

    tell application "Mail"
        set accountList to every account
        if (count of accountList) is 0 then error "No Mail accounts configured"

        set selectedAccount to item 1 of accountList
        if preferredSender is not "" then
            repeat with acct in accountList
                try
                    if preferredSender is in (email addresses of acct) then
                        set selectedAccount to acct
                        exit repeat
                    end if
                end try
            end repeat
        end if

        set outgoingMessage to make new outgoing message with properties {subject:subjectLine, content:plainBody & return & return, visible:false}
        tell outgoingMessage
            make new to recipient at end of to recipients with properties {address:recipientAddress}
            try
                set sender to item 1 of (email addresses of selectedAccount)
            end try
            repeat with idx from 5 to count of argv
                set attachmentPath to POSIX file (item idx of argv)
                make new attachment with properties {file name:attachmentPath} at after the last paragraph
            end repeat
        end tell
        ignoring application responses
            send outgoingMessage
        end ignoring
    end tell
end run
'''
    fallback_body = (
        "QuantPilot daily report attached.\n\n"
        "Queued via Mail.app on this Mac."
    )
    attachments = [str(report_path), *[str(path) for path in _normalize_attachment_paths(attachment_paths)]]
    try:
        proc = subprocess.Popen(
            [
                "osascript",
                "-e",
                apple_script,
                subject,
                report_to,
                report_from,
                fallback_body,
                *attachments,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"Email queued to {report_to} via Mail.app (pid={proc.pid})")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_sendmail_message(subject, report_to, report_from, report_path, attachment_paths=None):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = report_from
    msg["To"] = report_to
    body = (
        "QuantPilot daily report attached.\n\n"
        "SMTP delivery failed on this host, so this message was relayed via local sendmail."
    )
    msg.attach(MIMEText(body, "plain", "utf-8"))
    with open(report_path, "rb") as f:
        attachment = MIMEBase("text", "html")
        attachment.set_payload(f.read())
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", f"attachment; filename={report_path.name}")
    msg.attach(attachment)
    _attach_files(msg, attachment_paths)
    return msg


def send_via_sendmail(subject, report_to, report_from, report_path, attachment_paths=None):
    sendmail_bin = shutil.which("sendmail")
    if not sendmail_bin:
        return False, "sendmail not found"
    msg = build_sendmail_message(subject, report_to, report_from, report_path, attachment_paths=attachment_paths)
    try:
        subprocess.run(
            [sendmail_bin, "-t", "-oi"],
            input=msg.as_bytes(),
            check=True,
            capture_output=True,
        )
        print(f"Email queued to {report_to} via sendmail")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_delivery_plan(config):
    method = config["report_delivery_method"]
    if method == "mailapp":
        return ["mailapp", "smtp", "sendmail"]
    if method == "sendmail":
        return ["sendmail", "mailapp"]
    if method == "smtp":
        return ["smtp", "sendmail", "mailapp"]
    return ["smtp", "sendmail", "mailapp"]


def send_email(
    html_content,
    subject,
    report_filename: str | None = None,
    report_dir: str | os.PathLike[str] | Path | None = None,
    attachment_paths=None,
):
    """Send email using configured delivery method(s)."""
    config = email_config()
    missing = log_config_status(config)
    report_path = save_report_locally(html_content, filename=report_filename, report_dir=report_dir)
    if missing and not config["sendmail_fallback"] and not config["mail_app_fallback"]:
        print(f"Email not configured, missing: {', '.join(missing)}.")
        return False

    for channel in build_delivery_plan(config):
        if channel == "smtp":
            if missing:
                print(f"SMTP not fully configured, missing: {', '.join(missing)}")
                continue
            msg = build_message(
                html_content,
                subject,
                config["report_from"],
                config["report_to"],
                attachment_paths=attachment_paths,
            )
            sent, error = send_via_smtp(config, msg)
            if sent:
                return True
            print(f"Email failed via SMTP: {error}")
        elif channel == "sendmail":
            if not config["sendmail_fallback"]:
                continue
            sent, error = send_via_sendmail(
                subject,
                config["report_to"],
                config["report_from"],
                report_path,
                attachment_paths=attachment_paths,
            )
            if sent:
                return True
            print(f"sendmail fallback failed: {error}")
        elif channel == "mailapp":
            if not config["mail_app_fallback"] and config["report_delivery_method"] != "mailapp":
                continue
            sent, error = send_via_mail_app(
                subject,
                config["report_to"],
                config["mail_app_from"],
                report_path,
                attachment_paths=attachment_paths,
            )
            if sent:
                return True
            print(f"Mail.app fallback failed: {error}")

    return False


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Generating daily report: {today}")

    data_info = check_data_status()
    signal_info = check_signal_status()
    major_money_info = check_major_money_digest_status()
    capital_flow_info = check_capital_flow_status()
    capital_flow_eval_info = check_capital_flow_eval_status()
    capital_flow_gate_info = check_capital_flow_gate_status()

    trade_log_env = os.environ.get("TRADE_LOG", "").strip()
    trade_log = Path(trade_log_env) if trade_log_env else Path.home() / "quantpilot/logs/trade.log"
    trade_status = check_trade_status(trade_log=trade_log)

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        date=today,
        gen_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trade_status=trade_status,
        **data_info,
        **signal_info,
        **major_money_info,
        **capital_flow_info,
        **capital_flow_eval_info,
        **capital_flow_gate_info,
    )

    subject = f"QuantPilot Daily Report - {today}"
    if not send_email(html, subject):
        raise SystemExit(1)
    print("Report generation complete")


if __name__ == "__main__":
    main()
