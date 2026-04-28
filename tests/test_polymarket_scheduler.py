import sys
import types
from datetime import datetime, timedelta, timezone

_apscheduler = types.ModuleType("apscheduler")
_schedulers = types.ModuleType("apscheduler.schedulers")
_blocking = types.ModuleType("apscheduler.schedulers.blocking")
_triggers = types.ModuleType("apscheduler.triggers")
_cron = types.ModuleType("apscheduler.triggers.cron")


class _FakeBlockingScheduler:
    def __init__(self, timezone=None):
        self.timezone = timezone
        self.jobs = []
        self.started = False

    def add_job(self, func, *args, **kwargs):
        self.jobs.append((func, args, kwargs))

    def start(self):
        self.started = True
        return None


class _FakeCronTrigger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_blocking.BlockingScheduler = _FakeBlockingScheduler
_cron.CronTrigger = _FakeCronTrigger
sys.modules.setdefault("apscheduler", _apscheduler)
sys.modules.setdefault("apscheduler.schedulers", _schedulers)
sys.modules.setdefault("apscheduler.schedulers.blocking", _blocking)
sys.modules.setdefault("apscheduler.triggers", _triggers)
sys.modules.setdefault("apscheduler.triggers.cron", _cron)

from polymarket.config import PolySettings
from polymarket.models import Opportunity, MarketInfo
from polymarket.paper.simulator import PaperSimulator
from polymarket.reporting.daily import default_report_date
from polymarket.pipeline import PipelineResult
from polymarket.scheduler import PolymarketScheduler, main
from polymarket.storage import PolyStorage


def _market() -> MarketInfo:
    return MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Will X happen?",
        slug="will-x-happen",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )


def test_scheduler_registers_isolated_jobs(tmp_path):
    scheduler = PolymarketScheduler(PolySettings(data_dir=str(tmp_path)))
    scheduler.register_jobs()

    job_ids = [job[2]["id"] for job in scheduler.scheduler.jobs]
    assert job_ids == [
        "polymarket_scan",
        "polymarket_daily_report",
        "polymarket_catalog_refresh",
        "polymarket_book_top_retention",
    ]
    assert scheduler.scheduler.jobs[0][2]["max_instances"] == 1
    assert scheduler.scheduler.jobs[0][2]["coalesce"] is True


def test_scheduler_daily_report_generates_previous_day_artifact(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)
    now = datetime.now(timezone.utc)
    report_date = default_report_date(now)
    simulator.consume(
        _market(),
        [
            Opportunity(
                market_id="m1",
                question="Will X happen?",
                direction="buy_both_merge",
                gross_cost=0.98,
                fee_cost=0.0,
                yes_fee_cost=0.0,
                no_fee_cost=0.0,
                gas_cost=0.0,
                slippage_buffer=0.0,
                net_cost=0.98,
                net_edge=0.02,
                capacity=1.0,
                mergeable_qty=1.0,
                yes_qty=1.0,
                no_qty=1.0,
                yes_price=0.49,
                no_price=0.49,
                yes_book_timestamp_ms=1000,
                no_book_timestamp_ms=1000,
                ts=now - timedelta(days=1),
            )
        ],
    )
    storage.upsert_daily_summary(
        {
            "date": report_date,
            "strategy_type": "full_set_arb",
            "signals": 1,
            "accepted_signals": 1,
            "simulated_trades": 1,
            "gross_edge_sum": 0.02,
            "net_edge_sum": 0.02,
            "realized_pnl": 0.0,
            "max_inventory_used": 0.98,
            "updated_at": now,
        }
    )

    scheduler = PolymarketScheduler(cfg)
    payload, paths = scheduler.run_daily_report()

    assert payload["report_date"] == report_date
    assert payload["status"] == "ok"
    assert paths["latest"].exists()


def test_scheduler_sends_email_when_enabled(monkeypatch, tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), email_report_enabled=True)
    scheduler = PolymarketScheduler(cfg)
    calls = {"sent": False}

    def fake_generate_daily_report(cfg=None, target_date=None):
        return {
            "status": "no_data",
            "report_date": target_date,
            "generated_at": "2026-04-20T00:05:00+00:00",
            "summary": None,
            "mirror_summary": None,
            "db_path": str(cfg.duckdb_path),
            "reports_path": str(cfg.reports_path),
            "mirror_enabled": False,
            "mirror_reports_path": str(cfg.mirror_reports_path),
        }, {"latest": cfg.reports_path / "daily_summary_latest.json", "dated": cfg.reports_path / f"daily_summary_{target_date}.json"}

    def fake_send_daily_report_email(payload, paths, cfg=None):
        calls["sent"] = True
        return True

    monkeypatch.setattr("polymarket.scheduler.generate_daily_report", fake_generate_daily_report)
    monkeypatch.setattr("polymarket.scheduler.send_daily_report_email", fake_send_daily_report_email)

    payload, _ = scheduler.run_daily_report("2026-04-19")

    assert payload["report_date"] == "2026-04-19"
    assert calls["sent"] is True


def test_scheduler_start_runs_scan_before_blocking(monkeypatch, tmp_path):
    scheduler = PolymarketScheduler(PolySettings(data_dir=str(tmp_path)))
    calls = []

    def fake_retention():
        calls.append('retention')
        return None

    def fake_scan():
        calls.append('scan')
        return None

    def fake_start():
        calls.append('start')
        return None

    monkeypatch.setattr(scheduler, 'run_book_top_retention', fake_retention)
    monkeypatch.setattr(scheduler, 'run_scan', fake_scan)
    monkeypatch.setattr(scheduler.scheduler, 'start', fake_start)

    scheduler.start()

    assert calls == ['retention', 'scan', 'start']


def test_scheduler_run_scan_logs_duration(monkeypatch, tmp_path):
    scheduler = PolymarketScheduler(PolySettings(data_dir=str(tmp_path)))
    result = PipelineResult(markets_seen=1, opportunities_found=0, trades_simulated=0)
    messages = []

    monkeypatch.setattr(scheduler.pipeline, 'run_once', lambda: result)
    monkeypatch.setattr('polymarket.scheduler.logger.info', lambda message: messages.append(message))

    returned = scheduler.run_scan()

    assert returned == result
    assert any('duration_seconds=' in message for message in messages)
    assert any('catalog_load_seconds=' in message for message in messages)
    assert any('book_fetch_seconds=' in message for message in messages)


def test_scheduler_run_book_top_retention_logs_deleted_rows(monkeypatch, tmp_path):
    scheduler = PolymarketScheduler(PolySettings(data_dir=str(tmp_path), book_top_retention_hours=72))
    messages = []

    monkeypatch.setattr(scheduler.pipeline.storage, "prune_book_tops", lambda retention_hours: 123)
    monkeypatch.setattr(scheduler.pipeline.storage, "prune_book_snapshots", lambda retention_hours: 4)
    monkeypatch.setattr('polymarket.scheduler.logger.info', lambda message: messages.append(message))

    deleted_rows, deleted_snapshot_files = scheduler.run_book_top_retention()

    assert deleted_rows == 123
    assert deleted_snapshot_files == 4
    assert any('retention_hours=72' in message for message in messages)
    assert any('deleted_rows=123' in message for message in messages)
    assert any('deleted_snapshot_files=4' in message for message in messages)


def test_scheduler_main_starts_blocking_scheduler(monkeypatch):
    started = {"value": False}

    def fake_start(self):
        started["value"] = True

    monkeypatch.setattr(PolymarketScheduler, "start", fake_start)

    main()

    assert started["value"] is True
