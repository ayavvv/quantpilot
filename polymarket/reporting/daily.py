"""Daily reporting for isolated Polymarket paper trading."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from polymarket.config import PolySettings, settings
from polymarket.storage import PolyStorage


def default_report_date(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now.date() - timedelta(days=1)).isoformat()


def build_daily_report(cfg: PolySettings | None = None, target_date: str | None = None) -> dict[str, Any]:
    cfg = cfg or settings
    report_date = target_date or default_report_date()
    snapshot = PolyStorage.load_report_snapshot(cfg.duckdb_path, report_date)
    generated_at = datetime.now(timezone.utc).isoformat()

    payload: dict[str, Any] = {
        "status": "ok" if snapshot is not None else "no_data",
        "report_date": report_date,
        "generated_at": generated_at,
        "db_path": str(cfg.duckdb_path),
        "reports_path": str(cfg.reports_path),
        "summary": snapshot,
    }
    return payload


def write_daily_report_artifacts(payload: dict[str, Any], cfg: PolySettings | None = None) -> dict[str, Path]:
    cfg = cfg or settings
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    report_date = payload["report_date"]
    latest_path = cfg.reports_path / "daily_summary_latest.json"
    dated_path = cfg.reports_path / f"daily_summary_{report_date}.json"
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_path.write_text(content, encoding="utf-8")
    dated_path.write_text(content, encoding="utf-8")
    return {"latest": latest_path, "dated": dated_path}


def generate_daily_report(cfg: PolySettings | None = None, target_date: str | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = cfg or settings
    payload = build_daily_report(cfg=cfg, target_date=target_date)
    paths = write_daily_report_artifacts(payload, cfg=cfg)
    return payload, paths
