import sys
import types
from datetime import datetime


_apscheduler = types.ModuleType("apscheduler")
_schedulers = types.ModuleType("apscheduler.schedulers")
_blocking = types.ModuleType("apscheduler.schedulers.blocking")
_triggers = types.ModuleType("apscheduler.triggers")
_cron = types.ModuleType("apscheduler.triggers.cron")


class _FakeBlockingScheduler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeCronTrigger:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


_blocking.BlockingScheduler = _FakeBlockingScheduler
_cron.CronTrigger = _FakeCronTrigger

_duckdb = types.ModuleType("duckdb")

sys.modules.setdefault("apscheduler", _apscheduler)
sys.modules.setdefault("apscheduler.schedulers", _schedulers)
sys.modules.setdefault("apscheduler.schedulers.blocking", _blocking)
sys.modules.setdefault("apscheduler.triggers", _triggers)
sys.modules.setdefault("apscheduler.triggers.cron", _cron)
sys.modules.setdefault("duckdb", _duckdb)

from collector.scheduler import DataCollectorScheduler


class _FakeQlibWriter:
    def __init__(self, loaded=None):
        self.loaded = loaded or {}
        self.saved = {}

    def load_metadata(self, name):
        return self.loaded.get(name)

    def save_metadata(self, name, data):
        self.saved[name] = data


def test_latest_completed_a_share_date_reads_completion_metadata():
    scheduler = DataCollectorScheduler()
    scheduler.qlib_writer = _FakeQlibWriter(
        {
            scheduler.A_SHARE_SYNC_STATUS_METADATA: {
                "last_completed_trade_date": "2026-03-30",
            }
        }
    )

    assert scheduler._latest_completed_a_share_date() == "2026-03-30"


def test_mark_a_share_sync_completed_saves_completion_metadata():
    scheduler = DataCollectorScheduler()
    scheduler.qlib_writer = _FakeQlibWriter()

    scheduler._mark_a_share_sync_completed(
        "2026-03-30",
        total_codes=5191,
        started_at=datetime(2026, 3, 31, 18, 0, 0),
    )

    saved = scheduler.qlib_writer.saved[scheduler.A_SHARE_SYNC_STATUS_METADATA]
    assert saved["last_completed_trade_date"] == "2026-03-30"
    assert saved["total_codes"] == 5191
    assert saved["started_at"] == "2026-03-31 18:00:00"
    assert saved["completed_at"]
