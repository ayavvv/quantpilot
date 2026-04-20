"""Scheduler for isolated Polymarket paper-trading jobs."""
from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from polymarket.config import PolySettings, settings
from polymarket.pipeline import PolymarketPipeline
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
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(self.cfg.scan_interval_seconds, 30),
        )
        self.scheduler.add_job(
            self.run_daily_report,
            trigger=CronTrigger(hour=0, minute=5, timezone="UTC"),
            id="polymarket_daily_report",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=43200,
        )

    def run_scan(self):
        result = self.pipeline.run_once()
        logger.info(
            f"polymarket scan complete: markets={result.markets_seen} "
            f"opps={result.opportunities_found} trades={result.trades_simulated}"
        )
        return result

    def run_daily_report(self, target_date: str | None = None):
        target_date = target_date or default_report_date()
        payload, paths = generate_daily_report(cfg=self.cfg, target_date=target_date)
        logger.info(
            f"polymarket daily report complete: status={payload['status']} "
            f"date={payload['report_date']} latest={paths['latest']}"
        )
        return payload, paths

    def start(self) -> None:
        self.register_jobs()
        logger.info(
            f"starting polymarket scheduler: scan_interval_seconds={self.cfg.scan_interval_seconds} "
            f"top_trader_mirror={self.cfg.enable_top_trader_mirror} data_dir={self.cfg.root_data_path}"
        )
        self.run_scan()
        self.scheduler.start()


def main() -> None:
    PolymarketScheduler().start()


if __name__ == "__main__":
    main()
