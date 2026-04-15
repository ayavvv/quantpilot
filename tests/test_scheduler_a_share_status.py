import os
import sys
import types
from datetime import datetime

import pytest


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
    def __init__(self, loaded=None, instruments=None):
        self.loaded = loaded or {}
        self.saved = {}
        self.instruments = instruments or {}

    def load_metadata(self, name):
        return self.loaded.get(name)

    def save_metadata(self, name, data):
        self.saved[name] = data

    def flush(self):
        return None


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


def test_latest_a_share_date_prefers_in_memory_writer_over_stale_disk(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib_data"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text(
        "SH.600000\t2026-01-01\t2026-04-13\n"
        "SZ.000001\t2026-01-01\t2026-04-13\n"
    )

    scheduler = DataCollectorScheduler()
    scheduler.qlib_writer = _FakeQlibWriter(
        instruments={
            "SH.600000": ("2026-01-01", "2026-04-14"),
            "SZ.000001": ("2026-01-01", "2026-04-13"),
        }
    )
    monkeypatch.setenv("QLIB_DATA_DIR", str(qlib_dir))

    assert scheduler._latest_a_share_date() == "2026-04-14"


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


def test_mark_a_share_sync_completed_requires_qlib_writer(monkeypatch):
    scheduler = DataCollectorScheduler()
    monkeypatch.delenv("QLIB_DATA_DIR", raising=False)
    monkeypatch.setattr(scheduler, "_init_qlib_writer", lambda: None)

    with pytest.raises(RuntimeError, match="QLIB_DATA_DIR is required"):
        scheduler._mark_a_share_sync_completed(
            "2026-03-30",
            total_codes=5191,
            started_at=datetime(2026, 3, 31, 18, 0, 0),
        )


def test_sync_a_share_kline_reports_empty_data_reason():
    scheduler = DataCollectorScheduler()

    class _FakeDBEngine:
        def get_kline_max_date(self, code, ktype):
            return None

        def log_job(self, *args, **kwargs):
            return None

    class _FakeBaostockClient:
        def get_history_kline(self, code, start, end, ktype):
            return []

        def get_last_history_kline_status(self):
            return {"status": "empty_data", "code": "SH.600000"}

    scheduler.db_engine = _FakeDBEngine()
    scheduler.bs_client = _FakeBaostockClient()
    scheduler.qlib_writer = None

    result = scheduler.sync_a_share_kline("SH.600000", target_end_date="2026-03-30")
    assert result["ok"] is False
    assert result["reason"] == "empty_data"
    assert result["code"] == "SH.600000"


def test_sync_a_share_kline_normalizes_empty_ok_status():
    scheduler = DataCollectorScheduler()

    class _FakeDBEngine:
        def get_kline_max_date(self, code, ktype):
            return "2026-04-03"

        def log_job(self, *args, **kwargs):
            return None

    class _FakeBaostockClient:
        def get_history_kline(self, code, start, end, ktype):
            return []

        def get_last_history_kline_status(self):
            return {"status": "ok", "rows": 0, "raw_rows": 1}

    scheduler.db_engine = _FakeDBEngine()
    scheduler.bs_client = _FakeBaostockClient()
    scheduler.qlib_writer = None

    result = scheduler.sync_a_share_kline("SH.600355", target_end_date="2026-04-09")
    assert result["ok"] is False
    assert result["reason"] == "empty_data"
    assert result["latest_date"] == "2026-04-03"


def test_sync_a_share_kline_reports_target_not_reached():
    scheduler = DataCollectorScheduler()

    class _FakeDBEngine:
        def __init__(self):
            self.max_dates = {}

        def get_kline_max_date(self, code, ktype):
            return self.max_dates.get((code, ktype))

        def append_kline(self, df, code, ktype):
            self.max_dates[(code, ktype)] = "2026-03-28"

        def log_job(self, *args, **kwargs):
            return None

    class _FakeBaostockClient:
        def get_history_kline(self, code, start, end, ktype):
            return [{"date": "2026-03-28", "close": 1}]

        def get_last_history_kline_status(self):
            return {"status": "ok", "rows": 1}

    scheduler.db_engine = _FakeDBEngine()
    scheduler.bs_client = _FakeBaostockClient()
    scheduler.qlib_writer = None

    result = scheduler.sync_a_share_kline("SH.600000", target_end_date="2026-03-30")
    assert result["ok"] is False
    assert result["reason"] == "target_not_reached"
    assert result["latest_date"] == "2026-03-28"


def test_save_a_share_sync_summary_persists_metadata():
    scheduler = DataCollectorScheduler()
    scheduler.qlib_writer = _FakeQlibWriter()

    scheduler._save_a_share_sync_summary(
        {
            "status": "incomplete",
            "target_trade_date": "2026-03-30",
            "failed_count": 2,
            "failed_codes_preview": ["SH.600000", "SZ.000001"],
        }
    )

    saved = scheduler.qlib_writer.saved[scheduler.A_SHARE_SYNC_SUMMARY_METADATA]
    assert saved["status"] == "incomplete"
    assert saved["failed_count"] == 2
    assert saved["failed_codes_preview"] == ["SH.600000", "SZ.000001"]


def test_allowed_a_share_failures_reads_env(monkeypatch):
    scheduler = DataCollectorScheduler()
    monkeypatch.setenv("ALLOWED_A_SHARE_FAILURES", "3")
    assert scheduler._allowed_a_share_failures() == 3


def test_allowed_a_share_failures_falls_back_on_invalid_env(monkeypatch):
    scheduler = DataCollectorScheduler()
    monkeypatch.setenv("ALLOWED_A_SHARE_FAILURES", "abc")
    assert scheduler._allowed_a_share_failures() == 0


def test_tolerable_a_share_gap_accepts_non_trading_symbol():
    scheduler = DataCollectorScheduler()

    result = {
        "ok": False,
        "reason": "target_not_reached",
        "code": "SH.600355",
        "latest_date": "2026-04-03",
        "query_status": {"status": "ok", "rows": 2},
    }

    assert scheduler._is_tolerable_a_share_gap(result, "2026-04-08") is True


def test_tolerable_a_share_gap_accepts_converted_empty_symbol():
    scheduler = DataCollectorScheduler()

    result = {
        "ok": False,
        "reason": "converted_empty",
        "code": "SH.600355",
        "latest_date": "2026-04-03",
        "query_status": {"status": "converted_empty", "rows": 0, "raw_rows": 1},
    }

    assert scheduler._is_tolerable_a_share_gap(result, "2026-04-08") is True


def test_tolerable_a_share_gap_rejects_query_failures():
    scheduler = DataCollectorScheduler()

    result = {
        "ok": False,
        "reason": "target_not_reached",
        "code": "SH.600355",
        "latest_date": "2026-04-03",
        "query_status": {"status": "query_failed", "error": "socket timeout"},
    }

    assert scheduler._is_tolerable_a_share_gap(result, "2026-04-08") is False


def test_a_share_completion_allows_non_blocking_gaps():
    scheduler = DataCollectorScheduler()

    failed_runs = [
        {
            "ok": False,
            "reason": "target_not_reached",
            "code": "SH.600355",
            "latest_date": "2026-04-03",
            "query_status": {"status": "ok", "rows": 2},
        },
        {
            "ok": False,
            "reason": "empty_data",
            "code": "SZ.300067",
            "latest_date": "2026-04-07",
            "query_status": {"status": "converted_empty", "rows": 0, "raw_rows": 1},
        },
    ]

    result = scheduler._evaluate_a_share_sync_completion(
        total_codes=5195,
        failed_a_share_runs=failed_runs,
        latest_a_share_date="2026-04-09",
        target_a_share_date="2026-04-09",
        allowed_failures=0,
    )

    assert result["status"] == "complete_with_non_blocking_gaps"
    assert result["completed_a_share_target"] == "2026-04-09"
    assert result["market_ready"] is True
    assert len(result["blocking_failures"]) == 0
    assert len(result["non_blocking_gap_runs"]) == 2
    assert result["target_hit_count"] == 5193
    assert result["target_hit_ratio"] == pytest.approx(5193 / 5195)


def test_a_share_completion_blocks_when_market_not_ready():
    scheduler = DataCollectorScheduler()

    failed_runs = [
        {
            "ok": False,
            "reason": "target_not_reached",
            "code": "SH.600355",
            "latest_date": "2026-04-03",
            "query_status": {"status": "ok", "rows": 2},
        }
    ]

    result = scheduler._evaluate_a_share_sync_completion(
        total_codes=5195,
        failed_a_share_runs=failed_runs,
        latest_a_share_date="2026-04-08",
        target_a_share_date="2026-04-09",
        allowed_failures=0,
    )

    assert result["status"] == "incomplete"
    assert result["completed_a_share_target"] is None
    assert result["market_ready"] is False


def test_a_share_completion_blocks_hard_failures():
    scheduler = DataCollectorScheduler()

    failed_runs = [
        {
            "ok": False,
            "reason": "query_failed",
            "code": "SH.600355",
            "latest_date": "2026-04-08",
            "query_status": {"status": "query_failed", "error": "socket timeout"},
            "error": "socket timeout",
        }
    ]

    result = scheduler._evaluate_a_share_sync_completion(
        total_codes=5195,
        failed_a_share_runs=failed_runs,
        latest_a_share_date="2026-04-09",
        target_a_share_date="2026-04-09",
        allowed_failures=0,
    )

    assert result["status"] == "incomplete"
    assert result["completed_a_share_target"] is None
    assert result["market_ready"] is True
    assert len(result["blocking_failures"]) == 1
