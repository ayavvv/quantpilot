"""
Weekly auto retrain + backtest + trade-signal promotion + email report.

Flow:
1. Check/sync Qlib bin data (skip sync if data already available)
2. Train LightGBM model (test end date via env var, not file modification)
3. Run backtest with new pred_sh.pkl, generate report
4. Deploy model + pred to shared volume
5. Promote latest trade signal with the new model
6. Send backtest report via email

Environment variables:
    QLIB_DATA_DIR   - Qlib data directory (default: /qlib_data)
    STRATEGY_DIR    - Strategy code root (default: /app)
    MODELS_DIR      - Model output directory (default: /data/models)
    OUTPUT_DIR      - Report output directory (default: /data/output)
    SIGNAL_DIR      - Shared signal directory (default: /data/signals)
    TRADE_PRED_PATH - Deployment path for pred_sh.pkl (default: /data/models/pred_sh.pkl)
    NAS_HOST        - NAS hostname/IP for data sync (empty = skip sync)
    NAS_USER        - NAS SSH username
    NAS_QLIB_PATH   - Remote Qlib data path on NAS
    SMTP_HOST/PORT/USER/PASSWORD - Email config
    EMAIL_FROM/TO   - Email addresses
"""

from __future__ import annotations

import logging
import os
import pickle
import smtplib
import subprocess
import sys
import shutil
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

# --- Configuration (all via env vars, Docker-friendly defaults) ---
NAS_HOST = os.environ.get("NAS_HOST", "")
NAS_USER = os.environ.get("NAS_USER", "")
NAS_QLIB_PATH = os.environ.get("NAS_QLIB_PATH", "/qlib_data")

STRATEGY_DIR = Path(os.environ.get("STRATEGY_DIR", str(Path(__file__).resolve().parents[1])))
QLIB_DATA_DIR = Path(os.environ.get("QLIB_DATA_DIR", "/qlib_data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", "/data/signals"))
TRADE_PRED_PATH = Path(os.environ.get("TRADE_PRED_PATH", "/data/models/pred_sh.pkl"))

# Email configuration (SMTP)
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")
EMAIL_TO = os.environ.get("EMAIL_TO", "")
REPORT_FROM = os.environ.get("REPORT_FROM", "")
REPORT_TO = os.environ.get("REPORT_TO", "")
DEFAULT_WEEKLY_TIMEOUT_SECONDS = 12 * 60 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("weekly_train")


def resolve_email_config() -> dict[str, str]:
    fallback = _load_env_file(STRATEGY_DIR / "reporter" / ".env")

    smtp_host = os.environ.get("SMTP_HOST") or fallback.get("SMTP_HOST") or SMTP_HOST
    smtp_port = os.environ.get("SMTP_PORT") or fallback.get("SMTP_PORT") or str(SMTP_PORT)
    smtp_user = os.environ.get("SMTP_USER") or fallback.get("SMTP_USER") or SMTP_USER
    smtp_password = os.environ.get("SMTP_PASSWORD") or fallback.get("SMTP_PASSWORD") or SMTP_PASSWORD
    report_from = (
        os.environ.get("EMAIL_FROM")
        or os.environ.get("REPORT_FROM")
        or fallback.get("EMAIL_FROM")
        or fallback.get("REPORT_FROM")
        or EMAIL_FROM
        or REPORT_FROM
        or smtp_user
    )
    report_to = (
        os.environ.get("EMAIL_TO")
        or os.environ.get("REPORT_TO")
        or fallback.get("EMAIL_TO")
        or fallback.get("REPORT_TO")
        or EMAIL_TO
        or REPORT_TO
    )
    return {
        "smtp_host": smtp_host,
        "smtp_port": str(smtp_port),
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
        "report_from": report_from,
        "report_to": report_to,
    }


def log_email_config_status(config: dict[str, str]):
    missing = [
        name
        for name, value in {
            "SMTP_USER": config["smtp_user"],
            "SMTP_PASSWORD": config["smtp_password"],
            "REPORT_TO": config["report_to"],
        }.items()
        if not value
    ]
    log.info(
        "  SMTP config status: "
        f"host={config['smtp_host']} port={config['smtp_port']} "
        f"user={'set' if config['smtp_user'] else 'missing'} "
        f"password={'set' if config['smtp_password'] else 'missing'} "
        f"report_to={'set' if config['report_to'] else 'missing'} "
        f"report_from={'set' if config['report_from'] else 'missing'}"
    )
    return missing


def save_report_locally(filename: str, body_text: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / filename
    report_path.write_text(body_text, encoding="utf-8")
    log.info(f"  Report saved locally: {report_path}")
    return report_path


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _resolve_timeout_seconds(name: str, default: int = DEFAULT_WEEKLY_TIMEOUT_SECONDS) -> int:
    raw_value = os.environ.get(name) or os.environ.get("WEEKLY_TIMEOUT_SECONDS")
    if not raw_value:
        return default

    try:
        timeout_seconds = int(raw_value)
    except ValueError:
        log.warning(f"  Invalid {name}={raw_value!r}, fallback to {default}s")
        return default

    if timeout_seconds <= 0:
        log.warning(f"  Non-positive {name}={raw_value!r}, fallback to {default}s")
        return default

    return timeout_seconds


# --- Step 1: Check/Sync Qlib data ---

def sync_qlib_data():
    """Check Qlib bin data availability; sync from NAS only if missing."""
    log.info("Step 1: Checking Qlib data...")

    cal_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    if cal_path.exists():
        lines = cal_path.read_text().strip().splitlines()
        if lines:
            inst_path = QLIB_DATA_DIR / "instruments" / "all.txt"
            n_stocks = len(inst_path.read_text().strip().splitlines()) if inst_path.exists() else 0
            log.info(f"  Qlib data available: {len(lines)} days, "
                     f"latest: {lines[-1]}, {n_stocks} stocks")
            return

    # Data not available locally — sync from NAS
    if not NAS_HOST or not NAS_USER:
        raise RuntimeError(
            f"Qlib data not found at {QLIB_DATA_DIR} and NAS_HOST not configured. "
            "Run collector first or configure NAS sync."
        )

    log.info(f"  Syncing from NAS {NAS_USER}@{NAS_HOST}:{NAS_QLIB_PATH} ...")
    QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'ssh -o StrictHostKeyChecking=no {NAS_USER}@{NAS_HOST} '
        f'"cd {NAS_QLIB_PATH} && tar cf - calendars instruments features" | '
        f'(cd {QLIB_DATA_DIR} && tar xf -)'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("Qlib data sync failed")

    if cal_path.exists():
        lines = cal_path.read_text().strip().splitlines()
        log.info(f"  Sync complete: {len(lines)} days, latest: {lines[-1]}")
    else:
        raise RuntimeError("Sync failed: calendar file missing")


# --- Step 2: Train model ---

def get_latest_date() -> str:
    """Read latest date from Qlib calendar."""
    qlib_cal = QLIB_DATA_DIR / "calendars" / "day.txt"
    if qlib_cal.exists():
        lines = qlib_cal.read_text().strip().splitlines()
        if lines:
            return lines[-1].strip()
    return datetime.now().strftime("%Y-%m-%d")


def train_model():
    """Train SH market LightGBM model."""
    log.info("Step 2: Training model...")
    last_date = get_latest_date()
    log.info(f"  Test segment end date: {last_date}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pass TEST_END_DATE and MODELS_DIR via environment
    env = os.environ.copy()
    env["TEST_END_DATE"] = last_date
    env["MODELS_DIR"] = str(MODELS_DIR)
    env["QLIB_DATA_DIR"] = str(QLIB_DATA_DIR)

    main_py = STRATEGY_DIR / "main.py"
    if not main_py.exists():
        raise RuntimeError(f"Training entry point not found: {main_py}")

    timeout_seconds = _resolve_timeout_seconds("WEEKLY_TRAIN_TIMEOUT_SECONDS")
    log.info(f"  Train timeout: {timeout_seconds}s")

    result = subprocess.run(
        [sys.executable, str(main_py), "train", "--market", "sh"],
        cwd=str(STRATEGY_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        log.error(f"Training stdout:\n{result.stdout[-2000:]}")
        log.error(f"Training stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError("Model training failed")

    # Parse IC/ICIR from training output
    ic_val, icir_val = "N/A", "N/A"
    for line in result.stdout.split("\n"):
        if "IC:" in line and "ICIR:" in line:
            log.info(f"  {line.strip()}")
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "IC:":
                    ic_val = parts[i + 1] if i + 1 < len(parts) else "N/A"
                if p == "ICIR:":
                    icir_val = parts[i + 1] if i + 1 < len(parts) else "N/A"

    pred_path = MODELS_DIR / "pred_sh.pkl"
    if not pred_path.exists():
        raise RuntimeError(f"Training did not produce pred_sh.pkl: {pred_path}")

    # Check prediction coverage
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)
    dates = sorted(pred.index.get_level_values("datetime").unique())
    n_stocks = len(pred.index.get_level_values("instrument").unique())
    log.info(f"  pred_sh.pkl: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}, "
             f"{len(dates)} days, {n_stocks} stocks")

    # Verify model file also exists
    model_path = MODELS_DIR / "lightgbm_sh_latest.pkl"
    if model_path.exists():
        log.info(f"  Model: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    else:
        log.warning(f"  Model file not found: {model_path}")

    return {
        "ic": ic_val,
        "icir": icir_val,
        "pred_start": dates[0].strftime("%Y-%m-%d"),
        "pred_end": dates[-1].strftime("%Y-%m-%d"),
        "n_days": len(dates),
        "n_stocks": n_stocks,
    }


# --- Step 3: Backtest ---

def run_backtest():
    """Run backtest with new pred_sh.pkl."""
    log.info("Step 3: Running backtest...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = MODELS_DIR / "pred_sh.pkl"

    # Use -m to handle relative imports correctly
    env = os.environ.copy()
    env["QLIB_DATA_DIR"] = str(QLIB_DATA_DIR)

    timeout_seconds = _resolve_timeout_seconds("WEEKLY_BACKTEST_TIMEOUT_SECONDS")
    log.info(f"  Backtest timeout: {timeout_seconds}s")

    result = subprocess.run(
        [
            sys.executable, "-m", "trainer.backtest.run",
            "--pred", str(pred_path),
            "--price-dir", str(QLIB_DATA_DIR),
            "--top-n", "5",
            "--slippage", "0.001",
            "--output", str(OUTPUT_DIR),
        ],
        cwd=str(STRATEGY_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        log.error(f"Backtest stdout:\n{result.stdout[-2000:]}")
        log.error(f"Backtest stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError("Backtest failed")

    # Parse metrics
    metrics = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("="):
            parts = line.split(":", 1)
            if len(parts) == 2:
                k, v = parts[0].strip(), parts[1].strip()
                if k and v and k not in ("Prediction file", "Price directory"):
                    metrics[k] = v

    report_path = OUTPUT_DIR / "backtest_report.png"
    metrics_path = OUTPUT_DIR / "metrics.txt"

    log.info(f"  Report chart: {report_path}")
    log.info(f"  Metrics: {metrics_path}")

    return metrics, report_path, metrics_path


# --- Step 4: Deploy model files ---

def deploy_pred():
    """Deploy model + pred files to shared volume (if not already there)."""
    log.info("Step 4: Deploying model files...")

    # Deploy pred_sh.pkl
    src_pred = MODELS_DIR / "pred_sh.pkl"
    dst_pred = TRADE_PRED_PATH
    if src_pred.resolve() != dst_pred.resolve():
        dst_pred.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pred, dst_pred)
        log.info(f"  pred: {src_pred} -> {dst_pred}")
    else:
        log.info(f"  pred: already at {dst_pred}")

    # Deploy lightgbm model (inference needs it)
    src_model = MODELS_DIR / "lightgbm_sh_latest.pkl"
    dst_model = dst_pred.parent / "lightgbm_sh_latest.pkl"
    if src_model.exists() and src_model.resolve() != dst_model.resolve():
        shutil.copy2(src_model, dst_model)
        log.info(f"  model: {src_model} -> {dst_model}")
    elif src_model.exists():
        log.info(f"  model: already at {dst_model}")


# --- Step 5: Promote latest trade signal ---

def promote_trade_signal():
    """Run post-train inference and atomically promote fresh latest trade signals."""
    log.info("Step 5: Promoting latest trade signal...")

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QLIB_DATA_DIR"] = str(QLIB_DATA_DIR)
    env["MODEL_DIR"] = str(MODELS_DIR)
    env["SIGNAL_DIR"] = str(SIGNAL_DIR)
    env["PROMOTE_LATEST"] = "true"

    timeout_seconds = _resolve_timeout_seconds("WEEKLY_SIGNAL_PROMOTION_TIMEOUT_SECONDS")
    log.info(f"  Signal promotion timeout: {timeout_seconds}s")

    result = subprocess.run(
        [sys.executable, "-m", "inference.run_daily"],
        cwd=str(STRATEGY_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        log.error(f"Signal promotion stdout:\n{result.stdout[-2000:]}")
        log.error(f"Signal promotion stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError("Post-train signal promotion failed")

    latest_pred = SIGNAL_DIR / "pred_sh_latest.pkl"
    signal_date = "N/A"
    if latest_pred.exists():
        with open(latest_pred, "rb") as f:
            pred = pickle.load(f)
        dates = sorted(pred.index.get_level_values("datetime").unique())
        if dates:
            signal_date = dates[-1].strftime("%Y-%m-%d")
    log.info(f"  Latest trade signal promoted: {latest_pred} (signal_date={signal_date})")


# --- Step 6: Send email report ---

def send_report_email(train_info: dict, metrics: dict, report_path: Path, metrics_path: Path):
    """Send backtest report via SMTP email."""
    log.info("Step 6: Sending email report...")

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"Quant Weekly Report {today} | IC={train_info.get('ic', 'N/A')} | {metrics.get('ann_return', 'N/A')}"

    body_lines = [
        f"Quant Model Weekly Report -- {today}",
        "",
        "[Model Training]",
        f"  Prediction coverage: {train_info.get('pred_start', '?')} ~ {train_info.get('pred_end', '?')}",
        f"  Trading days: {train_info.get('n_days', '?')} days, {train_info.get('n_stocks', '?')} stocks",
        f"  IC: {train_info.get('ic', 'N/A')}  ICIR: {train_info.get('icir', 'N/A')}",
        "",
        "[Backtest Results] Top-5 equal weight, 0.1% slippage",
    ]
    for k, v in metrics.items():
        body_lines.append(f"  {k}: {v}")

    attachment_paths = [filepath for filepath in [report_path, metrics_path] if filepath.exists()]
    if attachment_paths:
        body_lines.extend([
            "",
            "[Artifacts]",
            *[f"  {filepath}" for filepath in attachment_paths],
        ])

    body_text = "\n".join(body_lines)
    config = resolve_email_config()
    missing = log_email_config_status(config)
    if missing:
        log.warning(f"  Email not configured, missing: {', '.join(missing)}. Saving report locally.")
        save_report_locally(f"weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = config["report_from"]
        msg["To"] = config["report_to"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        for filepath in attachment_paths:
            with open(filepath, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={filepath.name}")
            msg.attach(part)

        with smtplib.SMTP(config["smtp_host"], int(config["smtp_port"])) as server:
            server.starttls()
            server.login(config["smtp_user"], config["smtp_password"])
            server.send_message(msg)

        log.info(f"  Email sent to {config['report_to']}")
        return True

    except Exception as e:
        log.error(f"  Email sending failed: {e}")
        save_report_locally(f"weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)
        return False


def send_failure_email(error: Exception):
    """Send failure notification via SMTP email."""
    today = datetime.now().strftime("%Y-%m-%d")
    body_text = f"Weekly training pipeline failed:\n\n{error}"
    config = resolve_email_config()
    missing = log_email_config_status(config)
    if missing:
        log.warning(f"  Failure email not configured, missing: {', '.join(missing)}. Saving notice locally.")
        save_report_locally(f"weekly_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = config["report_from"]
        msg["To"] = config["report_to"]
        msg["Subject"] = f"[FAILED] Quant Weekly Report {today}"
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        with smtplib.SMTP(config["smtp_host"], int(config["smtp_port"])) as server:
            server.starttls()
            server.login(config["smtp_user"], config["smtp_password"])
            server.send_message(msg)
    except Exception as exc:
        log.error(f"  Failure email sending failed: {exc}")
        save_report_locally(f"weekly_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)


# --- Main ---

def main():
    log.info("=" * 60)
    log.info("Quant Weekly Train + Backtest + Report")
    log.info("=" * 60)
    start_time = datetime.now()

    try:
        # 1. Check/sync Qlib data
        sync_qlib_data()

        # 2. Train model
        train_info = train_model()

        # 3. Backtest
        metrics, report_path, metrics_path = run_backtest()

        # 4. Deploy
        deploy_pred()

        # 5. Promote latest trade signal
        promote_trade_signal()

        # 6. Email
        send_report_email(train_info, metrics, report_path, metrics_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"All done! Elapsed {elapsed:.0f} seconds")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        log.error(f"Failed, elapsed {elapsed:.0f} seconds")

        send_failure_email(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
