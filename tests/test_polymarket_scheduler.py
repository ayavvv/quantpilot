import sys
import types
from datetime import datetime, timezone

_apscheduler = types.ModuleType("apscheduler")
_schedulers = types.ModuleType("apscheduler.schedulers")
_blocking = types.ModuleType("apscheduler.schedulers.blocking")
_triggers = types.ModuleType("apscheduler.triggers")
_cron = types.ModuleType("apscheduler.triggers.cron")


class _FakeBlockingScheduler:
    def __init__(self, timezone=None):
        self.timezone = timezone
        self.jobs = []

    def add_job(self, func, *args, **kwargs):
        self.jobs.append((func, args, kwargs))

    def start(self):
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
from polymarket.scheduler import PolymarketScheduler
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

    assert len(scheduler.scheduler.jobs) == 2
    assert scheduler.scheduler.jobs[0][2]["id"] == "polymarket_scan"
    assert scheduler.scheduler.jobs[1][2]["id"] == "polymarket_daily_report"


def test_scheduler_daily_report_generates_artifact(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)
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
                ts=datetime.now(timezone.utc),
            )
        ],
    )

    scheduler = PolymarketScheduler(cfg)
    payload, paths = scheduler.run_daily_report()

    assert payload["report_date"] == datetime.now(timezone.utc).date().isoformat()
    assert payload["status"] == "ok"
    assert paths["latest"].exists()
