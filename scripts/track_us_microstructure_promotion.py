"""Daily promotion tracker for US microstructure major-flow validation.

Reads the forward-validation ledger (active_gate.json + rule_metrics.csv +
signal_events.parquet) and emits a concise, human-facing status of how close each
side is to promotion ("high confidence"), an ETA based on the recent official-event
accrual rate, and an on-track / off-track / validated verdict. Appends one row to a
history CSV so the trend is visible over time.

This is a read-only reporter; it does not modify the ledger. Run it AFTER the daily
`validate_us_microstructure_flow` refresh (the wrapper does that) so the numbers are
current even when the main report job is wedged.

Designed to run unattended via launchd: every parse is defensive, and no
delivery/IO failure may crash the job.
"""

from __future__ import annotations

import json
import math
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd


def _safe_int(value: object, default: int = 0) -> int:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return int(result) if math.isfinite(result) else default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
VALIDATION_DIR = BASE_DIR / "validation"
PROMOTION_HORIZON = _safe_int(os.environ.get("US_MICROSTRUCTURE_PROMOTION_HORIZON"), 5)
HIGH_SCORE = _safe_float(os.environ.get("US_MICROSTRUCTURE_HIGH_SCORE"), 85.0)
# Reuse the daily reporter's SMTP config (reporter/.env: SMTP_HOST/PORT/USER/PASSWORD, REPORT_TO/FROM).
REPORTER_ENV_PATH = Path(os.environ.get("REPORTER_ENV_FILE", str(REPO_ROOT / "reporter" / ".env")))


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def _send_email(subject: str, body: str) -> str:
    """Best-effort SMTP send reusing reporter/.env. Returns a status string; never raises."""
    try:
        cfg = _load_env_file(REPORTER_ENV_PATH)
        host = os.environ.get("SMTP_HOST") or cfg.get("SMTP_HOST")
        port = _safe_int(os.environ.get("SMTP_PORT") or cfg.get("SMTP_PORT"), 465)
        user = os.environ.get("SMTP_USER") or cfg.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD") or cfg.get("SMTP_PASSWORD")
        to_addr = os.environ.get("REPORT_TO") or cfg.get("REPORT_TO")
        from_addr = os.environ.get("REPORT_FROM") or cfg.get("REPORT_FROM") or user
        if not (host and user and password and to_addr):
            return f"email skipped: incomplete SMTP config in {REPORTER_ENV_PATH}"
        recipients = [a.strip() for a in str(to_addr).split(",") if a.strip()]
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as s:
                s.login(user, password)
                s.sendmail(from_addr, recipients, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls(context=context)
                s.login(user, password)
                s.sendmail(from_addr, recipients, msg.as_string())
        return f"email sent to {to_addr}"
    except Exception as exc:  # pragma: no cover - delivery must not crash the job
        return f"email FAILED: {exc}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_pct(value: object) -> str:
    f = _safe_float(value, default=float("nan"))
    return "n/a" if not math.isfinite(f) else f"{f * 100:.1f}%"


def _side_metrics(rule_metrics: pd.DataFrame, side: str) -> dict:
    if rule_metrics.empty or not {"side", "horizon"}.issubset(rule_metrics.columns):
        return {}
    try:
        row = rule_metrics[(rule_metrics["side"] == side) & (rule_metrics["horizon"] == PROMOTION_HORIZON)]
    except Exception:
        return {}
    if row.empty:
        return {}
    return row.iloc[-1].to_dict()


def _accrual_rate(events: pd.DataFrame, side: str) -> tuple[float, int, int]:
    """events/trading-day, total official events, distinct signal-days for `side`.

    The denominator is the number of *trading days elapsed* between the first and
    last official signal (business days), not just days that produced an event, so
    the resulting rate (and ETA) is not optimistically inflated.
    """
    if events.empty or not {"side", "signal_date"}.issubset(events.columns):
        return 0.0, 0, 0
    dates = pd.to_datetime(events["signal_date"], errors="coerce").dropna()
    if dates.empty:
        return 0.0, 0, 0
    span = max(1, len(pd.bdate_range(dates.min(), dates.max())))
    sub = events[events["side"] == side]
    n = int(len(sub))
    days = int(sub["signal_date"].nunique())
    return n / span, n, days


def _verdict(crit: dict, m: dict) -> tuple[str, bool]:
    """Return (human verdict, alert?) for a side from its matured 5d metrics."""
    if not m:
        return "⏳ 预热中：还没有成熟的 5 日样本", False
    obs = _safe_int(m.get("observation_count"))
    days = _safe_int(m.get("signal_day_count"))
    alpha = _safe_float(m.get("avg_alpha"))
    hit = _safe_float(m.get("hit_rate"))
    recent_hit = _safe_float(m.get("recent_hit_rate"))
    wilson = _safe_float(m.get("wilson_lower"))
    share = _safe_float(m.get("max_symbol_sample_share"), 1.0)

    need_obs = _safe_int(crit.get("min_observations_per_side"), 40)
    need_days = _safe_int(crit.get("min_signal_days_per_side"), 10)
    min_alpha = _safe_float(crit.get("min_alpha"), 0.0075)
    min_hit = _safe_float(crit.get("min_hit_rate"), 0.58)
    min_recent_hit = _safe_float(crit.get("min_recent_hit_rate"), 0.55)
    min_wilson = _safe_float(crit.get("min_wilson_lower"), 0.50)
    max_share = _safe_float(crit.get("max_symbol_sample_share"), 0.20)

    checks = {
        "样本数": obs >= need_obs,
        "信号日": days >= need_days,
        "alpha": alpha >= min_alpha,
        "命中率": hit >= min_hit,
        "近期命中": recent_hit >= min_recent_hit,
        "Wilson": wilson > min_wilson,
        "集中度": share <= max_share,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if not failed:
        return "✅ 已满足全部提升条件（应已 validated）", True
    if obs >= need_obs:
        return f"❌ 样本已满但统计未达标：差 {('、'.join(failed))} —— 信号预测力不足", True
    if obs >= 15 and alpha < 0:
        return f"⚠️ 跑偏：{obs} 个成熟样本上 alpha 为负（{_fmt_pct(alpha)}），暂无预测力迹象", True
    if obs >= max(30, int(need_obs * 0.75)):
        return f"🔜 逼近：样本将满（{obs}/{need_obs}），紧盯统计护栏", True
    if obs >= 5:
        return f"📈 累积中：{obs}/{need_obs} 样本，alpha={_fmt_pct(alpha)}、命中={_fmt_pct(hit)}", False
    return f"⏳ 预热中：成熟样本太少（{obs} 个），无法判断", False


def _eta_text(rate: float, matured_obs: int, matured_days: int, need_obs: int, need_days: int) -> str:
    if matured_obs >= need_obs and matured_days >= need_days:
        return "ETA: 样本量已满足，等统计护栏（命中率/alpha/Wilson）达标即可提升"
    if rate <= 0:
        return "ETA: 当前无官方事件累积，无法估算"
    obs_gate_days = math.ceil(max(0, need_obs - matured_obs) / rate)
    gate_days = max(obs_gate_days, max(0, need_days - matured_days))
    total = gate_days + PROMOTION_HORIZON + 1  # +maturation lag for the last needed signal
    weeks = round(total / 5, 1)
    return f"ETA(乐观): 按 ~{rate:.2f} 事件/交易日 → 约 {total} 交易日（~{weeks} 周）攒满样本"


def main() -> int:
    gate = _load_json(VALIDATION_DIR / "active_gate.json")
    crit = gate.get("criteria") if isinstance(gate.get("criteria"), dict) else {}
    try:
        rule_metrics = pd.read_csv(VALIDATION_DIR / "rule_metrics.csv")
    except Exception:
        rule_metrics = pd.DataFrame()
    try:
        events = pd.read_parquet(VALIDATION_DIR / "signal_events.parquet")
    except Exception:
        events = pd.DataFrame()

    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    state = str(gate.get("state") or "unknown")
    validated_sides = gate.get("validated_sides") if isinstance(gate.get("validated_sides"), dict) else {}

    lines = [f"📊 抓主力·验证进度  {today}", f"门控状态: {state}  (event_count={gate.get('event_count', '?')})"]
    alert = False
    history_row = {"date": today, "state": state, "event_count": gate.get("event_count")}

    for side, cn in (("distribution", "派发"), ("accumulation", "吸筹")):
        m = _side_metrics(rule_metrics, side)
        rate, n_events, n_days = _accrual_rate(events, side)
        matured_obs = _safe_int(m.get("observation_count")) if m else 0
        matured_days = _safe_int(m.get("signal_day_count")) if m else 0
        need_obs = _safe_int(crit.get("min_observations_per_side"), 40)
        need_days = _safe_int(crit.get("min_signal_days_per_side"), 10)
        verdict, side_alert = _verdict(crit, m)
        alert = alert or side_alert
        sv = "✅validated" if validated_sides.get(side) else "warmup"
        if m:
            detail = (
                f"  {cn}侧[{sv}]: 成熟样本 {matured_obs}/{need_obs} · 信号日 {matured_days}/{need_days} · "
                f"命中 {_fmt_pct(m.get('hit_rate'))} · 近期命中 {_fmt_pct(m.get('recent_hit_rate'))} · "
                f"alpha {_fmt_pct(m.get('avg_alpha'))} · Wilson下界 {_safe_float(m.get('wilson_lower')):.2f} · "
                f"集中度 {_fmt_pct(m.get('max_symbol_sample_share'))}"
            )
        else:
            detail = f"  {cn}侧[{sv}]: 暂无成熟 5 日样本（官方事件累计 {n_events} 个 / {n_days} 天）"
        lines.append(detail)
        lines.append(f"     判定: {verdict}")
        lines.append(f"     {_eta_text(rate, matured_obs, matured_days, need_obs, need_days)}")
        history_row[f"{side}_matured_obs"] = matured_obs
        history_row[f"{side}_hit_rate"] = m.get("hit_rate") if m else None
        history_row[f"{side}_recent_hit_rate"] = m.get("recent_hit_rate") if m else None
        history_row[f"{side}_avg_alpha"] = m.get("avg_alpha") if m else None
        history_row[f"{side}_signal_days"] = matured_days if m else None

    if HIGH_SCORE and not validated_sides.get("accumulation"):
        lines.append(f"  注: 吸筹分历史上限 < {HIGH_SCORE:.0f}，即便验证通过也难排出 high（除非市场真现强吸筹）")

    lines.append(f"  提醒触发: {'⚠️ 是 — 值得看一眼' if alert else '无（正常累积中）'}")
    message = "\n".join(lines)
    print(message)

    # Append trend history (best-effort; must never crash the job or clobber history).
    try:
        hist_path = Path(os.environ.get("US_MICROSTRUCTURE_PROMOTION_HISTORY", str(VALIDATION_DIR / "promotion_tracker_history.csv")))
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        row_df = pd.DataFrame([history_row])
        if hist_path.exists():
            try:
                prev = pd.read_csv(hist_path)
                prev = prev[prev["date"].astype(str) != str(today)]  # one row per date, last run wins
                _atomic_write_csv(pd.concat([prev, row_df], ignore_index=True), hist_path)
            except Exception as exc:
                # Don't clobber an unreadable history — append today's row instead.
                print(f"[tracker] history dedupe failed, appending instead: {exc}")
                row_df.to_csv(hist_path, mode="a", header=False, index=False)
        else:
            _atomic_write_csv(row_df, hist_path)
        _atomic_write_text(VALIDATION_DIR / "promotion_tracker_latest.txt", message + "\n")
    except Exception as exc:  # pragma: no cover - reporting must not crash the job
        print(f"[tracker] history write skipped: {exc}")

    # Email push: only when it matters. Send on alert days, plus a weekly heartbeat
    # (default Tuesday, weekday=1) so the inbox isn't flooded during the multi-week warmup.
    email_enabled = os.environ.get("US_MICROSTRUCTURE_PROMOTION_EMAIL", "").strip().lower() in {"1", "true", "yes"}
    heartbeat_weekday = _safe_int(os.environ.get("US_MICROSTRUCTURE_PROMOTION_HEARTBEAT_WEEKDAY"), 1)
    is_heartbeat = datetime.now().astimezone().weekday() == heartbeat_weekday
    if email_enabled and (alert or is_heartbeat):
        prefix = "⚠️ 抓主力验证·提醒" if alert else "抓主力验证·周度进度"
        status = _send_email(f"{prefix} {today}", message)
        print(f"[tracker] {status}")

    # Exit code 10 signals "alert-worthy" so a wrapper can choose to push only on alerts.
    return 10 if alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
