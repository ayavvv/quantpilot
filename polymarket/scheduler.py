"""Scheduler for isolated Polymarket paper-trading jobs."""
from __future__ import annotations

from threading import Lock
from time import perf_counter

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from polymarket.config import PolySettings, settings
from polymarket.pipeline import PolymarketPipeline
from polymarket.reporting.daily import default_report_date, generate_daily_report
from polymarket.reporting.email import send_daily_report_email


class PolymarketScheduler:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.pipeline = PolymarketPipeline(self.cfg)
        self.scheduler = BlockingScheduler(timezone="UTC")
        self._job_lock = Lock()

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
        self.scheduler.add_job(
            self.run_catalog_refresh,
            "interval",
            seconds=self.cfg.catalog_refresh_job_seconds,
            id="polymarket_catalog_refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(self.cfg.catalog_refresh_job_seconds, 60),
        )
        self.scheduler.add_job(
            self.run_book_top_retention,
            "interval",
            seconds=self.cfg.book_top_retention_job_seconds,
            id="polymarket_book_top_retention",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(self.cfg.book_top_retention_job_seconds, 60),
        )

    def run_scan(self):
        with self._job_lock:
            started = perf_counter()
            result = self.pipeline.run_once()
            duration_seconds = perf_counter() - started
            stage_timings = result.stage_timings or {}
            logger.info(
                f"polymarket scan complete: markets={result.markets_seen} "
                f"opps={result.opportunities_found} trades={result.trades_simulated} "
                f"duration_seconds={duration_seconds:.2f} "
                f"catalog_load_seconds={stage_timings.get('catalog_load_seconds', 0.0):.2f} "
                f"catalog_snapshot_seconds={stage_timings.get('catalog_snapshot_seconds', 0.0):.2f} "
                f"book_fetch_seconds={stage_timings.get('book_fetch_seconds', 0.0):.2f} "
                f"scan_compute_seconds={stage_timings.get('scan_compute_seconds', 0.0):.2f} "
                f"storage_write_seconds={stage_timings.get('storage_write_seconds', 0.0):.2f}"
            )
            return result

    def run_catalog_refresh(self):
        with self._job_lock:
            try:
                markets_seen, catalog_load_seconds, catalog_snapshot_seconds = self.pipeline.refresh_catalog()
            except Exception as exc:
                logger.warning(f"polymarket catalog refresh failed: {exc}")
                return 0, 0.0, 0.0
            logger.info(
                f"polymarket catalog refresh complete: markets={markets_seen} "
                f"catalog_load_seconds={catalog_load_seconds:.2f} catalog_snapshot_seconds={catalog_snapshot_seconds:.2f}"
            )
            return markets_seen, catalog_load_seconds, catalog_snapshot_seconds

    def run_daily_report(self, target_date: str | None = None):
        target_date = target_date or default_report_date()
        payload, paths = generate_daily_report(cfg=self.cfg, target_date=target_date)
        if self.cfg.email_report_enabled:
            email_sent = send_daily_report_email(payload, paths, cfg=self.cfg)
            logger.info(
                f"polymarket daily email complete: date={payload['report_date']} sent={email_sent}"
            )
        logger.info(
            f"polymarket daily report complete: status={payload['status']} "
            f"date={payload['report_date']} latest={paths['latest']}"
        )
        return payload, paths

    def run_book_top_retention(self):
        with self._job_lock:
            try:
                deleted_rows, deleted_snapshot_files = self.pipeline.prune_book_data()
            except Exception as exc:
                logger.warning(f"polymarket book_top retention failed: {exc}")
                return 0
            logger.info(
                f"polymarket book_top retention complete: retention_hours={self.cfg.book_top_retention_hours} "
                f"deleted_rows={deleted_rows} deleted_snapshot_files={deleted_snapshot_files}"
            )
            return deleted_rows, deleted_snapshot_files

    def start(self) -> None:
        self.register_jobs()
        logger.info(
            f"starting polymarket scheduler: scan_interval_seconds={self.cfg.scan_interval_seconds} "
            f"catalog_refresh_job_seconds={self.cfg.catalog_refresh_job_seconds} "
            f"book_top_retention_hours={self.cfg.book_top_retention_hours} "
            f"top_trader_mirror={self.cfg.enable_top_trader_mirror} data_dir={self.cfg.root_data_path}"
        )
        self.run_book_top_retention()
        self.run_scan()
        try:
            self.scheduler.start()
        finally:
            self.pipeline.close()


def main() -> None:
    PolymarketScheduler().start()


if __name__ == "__main__":
    main()
