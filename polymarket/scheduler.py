"""Scheduler for isolated Polymarket paper-trading jobs."""
from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from polymarket.config import PolySettings, settings
from polymarket.pipeline import PolymarketPipeline
from datetime import datetime, timezone

from polymarket.reporting.daily import default_report_date, generate_daily_report


class PolymarketScheduler:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.pipeline = PolymarketPipeline(self.cfg)
        self.scheduler = BlockingScheduler(timezone="UTC")

    def register_jobs(self) -> None:
        self.scheduler.add_job(
            self.run_scan,
            "interval",
            seconds=self.cfg.scan_interval_seconds,
            id="polymarket_scan",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_daily_report,
            trigger=CronTrigger(hour=0, minute=5, timezone="UTC"),
            id="polymarket_daily_report",
            replace_existing=True,
        )

    def run_scan(self):
        result = self.pipeline.run_once()
        logger.info(
            "polymarket scan complete: markets=%s opps=%s trades=%s",
            result.markets_seen,
            result.opportunities_found,
            result.trades_simulated,
        )
        return result

    def run_daily_report(self, target_date: str | None = None):
        target_date = target_date or datetime.now(timezone.utc).date().isoformat()
        payload, paths = generate_daily_report(cfg=self.cfg, target_date=target_date)
        logger.info(
            "polymarket daily report complete: status=%s date=%s latest=%s",
            payload["status"],
            payload["report_date"],
            paths["latest"],
        )
        return payload, paths

    def start(self) -> None:
        self.register_jobs()
        self.scheduler.start()
