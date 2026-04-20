"""Daily reporting for isolated Polymarket paper trading."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from polymarket.config import PolySettings, settings
from polymarket.storage import PolyStorage
from polymarket.traders.storage import MirrorStorage


def default_report_date(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now.date() - timedelta(days=1)).isoformat()


def build_daily_report(cfg: PolySettings | None = None, target_date: str | None = None) -> dict[str, Any]:
    cfg = cfg or settings
    report_date = target_date or default_report_date()
    snapshot = PolyStorage.load_report_snapshot(cfg.duckdb_path, report_date)
    mirror_snapshot = MirrorStorage.load_daily_summary(cfg.mirror_duckdb_path, report_date) if cfg.enable_top_trader_mirror else None
    generated_at = datetime.now(timezone.utc).isoformat()

    payload: dict[str, Any] = {
        "status": "ok" if snapshot is not None else "no_data",
        "report_date": report_date,
        "generated_at": generated_at,
        "db_path": str(cfg.duckdb_path),
        "reports_path": str(cfg.reports_path),
        "summary": snapshot,
        "mirror_enabled": cfg.enable_top_trader_mirror,
        "mirror_reports_path": str(cfg.mirror_reports_path),
        "mirror_summary": mirror_snapshot,
    }
    return payload


def write_daily_report_artifacts(payload: dict[str, Any], cfg: PolySettings | None = None) -> dict[str, Path]:
    cfg = cfg or settings
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    cfg.mirror_reports_path.mkdir(parents=True, exist_ok=True)
    report_date = payload["report_date"]
    latest_path = cfg.reports_path / "daily_summary_latest.json"
    dated_path = cfg.reports_path / f"daily_summary_{report_date}.json"
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_path.write_text(content, encoding="utf-8")
    dated_path.write_text(content, encoding="utf-8")
    if payload.get("mirror_summary") is not None:
        mirror_latest = cfg.mirror_reports_path / "mirror_daily_summary_latest.json"
        mirror_dated = cfg.mirror_reports_path / f"mirror_daily_summary_{report_date}.json"
        mirror_content = json.dumps(payload["mirror_summary"], ensure_ascii=False, indent=2) + "\n"
        mirror_latest.write_text(mirror_content, encoding="utf-8")
        mirror_dated.write_text(mirror_content, encoding="utf-8")
    return {"latest": latest_path, "dated": dated_path}


def generate_daily_report(cfg: PolySettings | None = None, target_date: str | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = cfg or settings
    payload = build_daily_report(cfg=cfg, target_date=target_date)
    paths = write_daily_report_artifacts(payload, cfg=cfg)
    return payload, paths
