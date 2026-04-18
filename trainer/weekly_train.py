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
import subprocess
import sys
import shutil
from datetime import datetime
from html import escape
from pathlib import Path

from reporter.send_report import send_email

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
DEFAULT_WEEKLY_STAGE_ROOT_NAME = "weekly_runs"
_WEEKLY_STAGE_TAG: str | None = None

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


def _safe_send_email(
    html_content: str,
    subject: str,
    *,
    report_filename: str,
    error_prefix: str,
) -> bool:
    try:
        return send_email(
            html_content,
            subject,
            report_filename=report_filename,
        )
    except Exception as exc:
        log.error(f"  {error_prefix}: {exc}")
        return False


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


def _stage_timestamp() -> str:
    global _WEEKLY_STAGE_TAG
    if _WEEKLY_STAGE_TAG:
        return _WEEKLY_STAGE_TAG
    override = os.environ.get("WEEKLY_STAGE_TAG", "").strip()
    if override:
        _WEEKLY_STAGE_TAG = override
        return _WEEKLY_STAGE_TAG
    _WEEKLY_STAGE_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _WEEKLY_STAGE_TAG


def _stage_models_dir() -> Path:
    root = Path(
        os.environ.get(
            "WEEKLY_STAGE_MODELS_ROOT",
            str(MODELS_DIR / DEFAULT_WEEKLY_STAGE_ROOT_NAME),
        )
    )
    return root / _stage_timestamp()


def _stage_output_dir() -> Path:
    root = Path(
        os.environ.get(
            "WEEKLY_STAGE_OUTPUT_ROOT",
            str(OUTPUT_DIR / DEFAULT_WEEKLY_STAGE_ROOT_NAME),
        )
    )
    return root / _stage_timestamp()


def _parse_metric_value(raw: str, *, percent: bool) -> float:
    value = raw.strip()
    if not value:
        raise ValueError("empty metric value")
    if percent:
        value = value.rstrip("%")
        return float(value) / 100.0
    return float(value)


def _promotion_threshold(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        log.warning(f"  Invalid {name}={raw_value!r}, fallback to {default}")
        return default


def evaluate_promotion_gate(candidate_metrics: dict, baseline_metrics: dict | None) -> tuple[bool, list[str]]:
    if not baseline_metrics:
        return True, ["No baseline metrics available; allow first promotion"]

    reasons: list[str] = []
    min_ann_return_diff = _promotion_threshold("WEEKLY_PROMOTION_MIN_ANN_RETURN_DIFF", 0.0)
    min_sharpe_diff = _promotion_threshold("WEEKLY_PROMOTION_MIN_SHARPE_DIFF", 0.0)
    max_drawdown_delta = _promotion_threshold("WEEKLY_PROMOTION_MAX_DRAWDOWN_DELTA", 0.01)

    try:
        candidate_ann = _parse_metric_value(candidate_metrics["ann_return"], percent=True)
        baseline_ann = _parse_metric_value(baseline_metrics["ann_return"], percent=True)
        candidate_sharpe = _parse_metric_value(candidate_metrics["sharpe"], percent=False)
        baseline_sharpe = _parse_metric_value(baseline_metrics["sharpe"], percent=False)
        candidate_mdd = abs(_parse_metric_value(candidate_metrics["max_drawdown"], percent=True))
        baseline_mdd = abs(_parse_metric_value(baseline_metrics["max_drawdown"], percent=True))
    except KeyError as exc:
        return False, [f"Missing promotion metric: {exc}"]
    except ValueError as exc:
        return False, [f"Invalid promotion metric: {exc}"]

    ann_diff = candidate_ann - baseline_ann
    sharpe_diff = candidate_sharpe - baseline_sharpe
    mdd_delta = candidate_mdd - baseline_mdd

    if ann_diff < min_ann_return_diff:
        reasons.append(
            "ann_return below gate: "
            f"candidate={candidate_metrics['ann_return']} baseline={baseline_metrics['ann_return']} "
            f"required_diff>={min_ann_return_diff:.4f}"
        )
    if sharpe_diff < min_sharpe_diff:
        reasons.append(
            "sharpe below gate: "
            f"candidate={candidate_metrics['sharpe']} baseline={baseline_metrics['sharpe']} "
            f"required_diff>={min_sharpe_diff:.4f}"
        )
    if mdd_delta > max_drawdown_delta:
        reasons.append(
            "max_drawdown above gate: "
            f"candidate={candidate_metrics['max_drawdown']} baseline={baseline_metrics['max_drawdown']} "
            f"allowed_delta<={max_drawdown_delta:.4f}"
        )

    if reasons:
        return False, reasons
    return True, [
        "Promotion gate passed",
        f"ann_return {candidate_metrics['ann_return']} vs {baseline_metrics['ann_return']}",
        f"sharpe {candidate_metrics['sharpe']} vs {baseline_metrics['sharpe']}",
        f"max_drawdown {candidate_metrics['max_drawdown']} vs {baseline_metrics['max_drawdown']}",
    ]


def _render_report_html(title: str, sections: list[tuple[str, list[str]]]) -> str:
    parts = [
        "<html><body style=\"font-family:-apple-system,sans-serif;max-width:760px;margin:0 auto;padding:20px;\">",
        f"<h1>{escape(title)}</h1>",
    ]
    for section_title, lines in sections:
        parts.append(f"<h2>{escape(section_title)}</h2><ul>")
        for line in lines:
            parts.append(f"<li>{escape(line)}</li>")
        parts.append("</ul>")
    parts.append("</body></html>")
    return "".join(parts)


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


def train_model(models_dir: Path):
    """Train SH market LightGBM model."""
    log.info("Step 2: Training model...")
    last_date = get_latest_date()
    log.info(f"  Test segment end date: {last_date}")

    models_dir.mkdir(parents=True, exist_ok=True)

    # Pass TEST_END_DATE and MODELS_DIR via environment
    env = os.environ.copy()
    env["TEST_END_DATE"] = last_date
    env["MODELS_DIR"] = str(models_dir)
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

    pred_path = models_dir / "pred_sh.pkl"
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
    model_path = models_dir / "lightgbm_sh_latest.pkl"
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

def run_backtest(pred_path: Path, output_dir: Path):
    """Run backtest for a specific prediction file."""
    log.info("Step 3: Running backtest...")
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "--hold-bonus", "0.05",
            "--stop-loss-pct", "-0.08",
            "--position-ratio", "0.95",
            "--allowed-prefix", "SH.",
            "--filter-limit-up",
            "--slippage", "0.001",
            "--output", str(output_dir),
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

    report_path = output_dir / "backtest_report.png"
    metrics_path = output_dir / "metrics.txt"

    log.info(f"  Report chart: {report_path}")
    log.info(f"  Metrics: {metrics_path}")

    return metrics, report_path, metrics_path


# --- Step 4: Deploy model files ---

def deploy_pred(models_dir: Path):
    """Deploy staged model + pred files to shared production paths."""
    log.info("Step 4: Deploying model files...")

    # Deploy pred_sh.pkl
    src_pred = models_dir / "pred_sh.pkl"
    dst_pred = TRADE_PRED_PATH
    if src_pred.resolve() != dst_pred.resolve():
        dst_pred.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pred, dst_pred)
        log.info(f"  pred: {src_pred} -> {dst_pred}")
    else:
        log.info(f"  pred: already at {dst_pred}")

    # Deploy lightgbm model (inference needs it)
    src_model = models_dir / "lightgbm_sh_latest.pkl"
    dst_model = dst_pred.parent / "lightgbm_sh_latest.pkl"
    if src_model.exists() and src_model.resolve() != dst_model.resolve():
        shutil.copy2(src_model, dst_model)
        log.info(f"  model: {src_model} -> {dst_model}")
    elif src_model.exists():
        log.info(f"  model: already at {dst_model}")


# --- Step 5: Promote latest trade signal ---

def promote_trade_signal(models_dir: Path):
    """Run post-train inference and atomically promote fresh latest trade signals."""
    log.info("Step 5: Promoting latest trade signal...")

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QLIB_DATA_DIR"] = str(QLIB_DATA_DIR)
    env["MODEL_DIR"] = str(models_dir)
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
    """Send weekly summary via the shared reporter delivery chain."""
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

    html_content = _render_report_html(
        subject,
        [
            (
                "Model Training",
                [
                    f"Prediction coverage: {train_info.get('pred_start', '?')} ~ {train_info.get('pred_end', '?')}",
                    f"Trading days: {train_info.get('n_days', '?')} days, {train_info.get('n_stocks', '?')} stocks",
                    f"IC: {train_info.get('ic', 'N/A')}  ICIR: {train_info.get('icir', 'N/A')}",
                ],
            ),
            (
                "Backtest Results",
                [f"{k}: {v}" for k, v in metrics.items()],
            ),
            (
                "Artifacts",
                [str(filepath) for filepath in attachment_paths] or ["No local artifacts"],
            ),
        ],
    )
    sent = _safe_send_email(
        html_content,
        subject,
        report_filename=f"weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        error_prefix="Weekly report email sending failed",
    )
    if not sent:
        save_report_locally(f"weekly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)
        return False
    log.info(f"  Email sent to {config['report_to']}")
    return True


def send_failure_email(error: Exception):
    """Send failure notification via the shared reporter delivery chain."""
    today = datetime.now().strftime("%Y-%m-%d")
    body_text = f"Weekly training pipeline failed:\n\n{error}"
    config = resolve_email_config()
    missing = log_email_config_status(config)
    if missing:
        log.warning(f"  Failure email not configured, missing: {', '.join(missing)}. Saving notice locally.")
        save_report_locally(f"weekly_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)
        return

    subject = f"[FAILED] Quant Weekly Report {today}"
    html_content = _render_report_html(subject, [("Failure", [body_text])])
    if not _safe_send_email(
        html_content,
        subject,
        report_filename=f"weekly_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        error_prefix="Failure email sending failed",
    ):
        save_report_locally(f"weekly_report_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", body_text)


# --- Main ---

def main():
    log.info("=" * 60)
    log.info("Quant Weekly Train + Backtest + Report")
    log.info("=" * 60)
    start_time = datetime.now()

    try:
        stage_models_dir = _stage_models_dir()
        stage_output_dir = _stage_output_dir()
        candidate_output_dir = stage_output_dir / "candidate"
        baseline_output_dir = stage_output_dir / "baseline"
        log.info(f"  Stage models dir: {stage_models_dir}")
        log.info(f"  Stage output dir: {stage_output_dir}")

        # 1. Check/sync Qlib data
        sync_qlib_data()

        # 2. Train model
        train_info = train_model(stage_models_dir)

        # 3. Backtest
        metrics, report_path, metrics_path = run_backtest(
            stage_models_dir / "pred_sh.pkl",
            candidate_output_dir,
        )

        baseline_metrics = None
        if TRADE_PRED_PATH.exists():
            log.info("Step 3b: Running baseline backtest on current production signal...")
            baseline_metrics, _, _ = run_backtest(TRADE_PRED_PATH, baseline_output_dir)
        else:
            log.info("Step 3b: Skipped baseline backtest (no current production pred_sh.pkl)")

        gate_ok, gate_reasons = evaluate_promotion_gate(metrics, baseline_metrics)
        for reason in gate_reasons:
            log.info(f"  Promotion gate: {reason}")
        if not gate_ok:
            raise RuntimeError("Promotion gate rejected staged weekly model: " + "; ".join(gate_reasons))

        # 4. Deploy
        deploy_pred(stage_models_dir)

        # 5. Promote latest trade signal
        promote_trade_signal(stage_models_dir)

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
