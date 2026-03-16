"""
Daily quant report: data collection status + model signals + trade status.
Sends via SMTP email or saves HTML locally.
"""

import os
import smtplib
import ssl
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import pandas as pd
from jinja2 import Template

SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/data/reports"))

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

<h2>3. Trading Status</h2>
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
    data_date = lines[-1].strip() if lines else "N/A"

    inst_path = qlib_dir / "instruments" / "all.txt"
    stock_count = 0
    if inst_path.exists():
        stock_count = len(inst_path.read_text().strip().splitlines())

    return {
        "data_ok": stock_count > 1000,
        "data_status": "OK" if stock_count > 1000 else "Warning",
        "stock_count": stock_count,
        "data_date": data_date,
    }


def check_signal_status():
    """Check signal status."""
    today = datetime.now().strftime("%Y%m%d")
    signal_file = SIGNAL_DIR / f"signal_{today}.csv"
    latest_file = SIGNAL_DIR / "signal_latest.csv"

    target = signal_file if signal_file.exists() else latest_file
    if not target.exists():
        return {"signal_count": 0, "signal_date": today, "top10": []}

    df = pd.read_csv(target)
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
        "signal_date": today,
        "top10": top10,
    }


def send_email(html_content, subject):
    """Send email via SMTP."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    report_to = os.environ.get("REPORT_TO", "")
    report_from = os.environ.get("REPORT_FROM", smtp_user)

    if not all([smtp_user, smtp_pass, report_to]):
        print("Email not configured, saving report locally.")
        report_path = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.html"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_content, encoding="utf-8")
        print(f"Report saved: {report_path}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = report_from
    msg["To"] = report_to
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(report_from, [report_to], msg.as_string())
        print(f"Email sent to {report_to}")
    except Exception as e:
        print(f"Email failed: {e}")
        report_path = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.html"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_content, encoding="utf-8")
        print(f"Report saved: {report_path}")


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Generating daily report: {today}")

    data_info = check_data_status()
    signal_info = check_signal_status()

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        date=today,
        gen_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trade_status="Trading module active (simulation mode).",
        **data_info,
        **signal_info,
    )

    subject = f"QuantPilot Daily Report - {today}"
    send_email(html, subject)
    print("Report generation complete")


if __name__ == "__main__":
    main()
