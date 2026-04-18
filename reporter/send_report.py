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
    )

    subject = f"QuantPilot Daily Report - {today}"
    if not send_email(html, subject):
        raise SystemExit(1)
    print("Report generation complete")


if __name__ == "__main__":
    main()
