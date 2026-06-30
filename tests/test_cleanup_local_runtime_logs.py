import os
from datetime import datetime, timedelta
from pathlib import Path

from scripts import cleanup_local_runtime_logs as cleanup


def _write_file(path: Path, content: str = "log") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _set_mtime(path: Path, modified: datetime) -> None:
    timestamp = modified.timestamp()
    os.utime(path, (timestamp, timestamp))


def _dated_file(path: Path, modified: datetime, content: str = "log") -> Path:
    written = _write_file(path, content)
    _set_mtime(written, modified)
    return written


def test_cleanup_deletes_only_old_runtime_logs(tmp_path):
    now = datetime(2026, 7, 1, 12, 0, 0)
    old = now - timedelta(days=45)
    recent = now - timedelta(days=2)
    futu_dir = tmp_path / "futu" / "Log"
    quantpilot_dir = tmp_path / "quantpilot" / "logs"
    old_futu_files = [
        _dated_file(futu_dir / "py_2026_06_01.log", old),
        _dated_file(futu_dir / "py_2026_06_01.log.1", old),
        _dated_file(futu_dir / "api_2026_06_01.logs", old),
        _dated_file(futu_dir / "futu_2026_06_01.ftlog", old),
    ]
    recent_futu = _dated_file(futu_dir / "py_2026_06_30.log", recent)
    non_log = _dated_file(futu_dir / "raw_snapshot.parquet", old)
    old_quantpilot_files = [
        _dated_file(quantpilot_dir / "market_capital_flow_us.log", old),
        _dated_file(quantpilot_dir / "nested" / "polymarket_scheduler.err.log", old),
    ]
    recent_quantpilot = _dated_file(quantpilot_dir / "us_microstructure_cleanup.out.log", recent)
    report = _dated_file(quantpilot_dir / "report.html", old)

    result = cleanup.cleanup_local_runtime_logs(
        futu_log_dir=futu_dir,
        futu_retention_days=14,
        quantpilot_log_dir=quantpilot_dir,
        quantpilot_retention_days=30,
        now=now.isoformat(),
        execute=True,
    )

    assert result["ok"] is True
    assert result["futu_candidate_count"] == 4
    assert result["quantpilot_candidate_count"] == 2
    assert result["deleted_file_count"] == 6
    assert all(not path.exists() for path in old_futu_files + old_quantpilot_files)
    assert recent_futu.exists()
    assert non_log.exists()
    assert recent_quantpilot.exists()
    assert report.exists()


def test_cleanup_dry_run_plans_without_deleting(tmp_path):
    now = datetime(2026, 7, 1, 12, 0, 0)
    old = now - timedelta(days=45)
    futu_dir = tmp_path / "futu" / "Log"
    quantpilot_dir = tmp_path / "quantpilot" / "logs"
    futu_log = _dated_file(futu_dir / "old.log", old, "futu")
    quantpilot_log = _dated_file(quantpilot_dir / "old.log", old, "quantpilot")

    result = cleanup.cleanup_local_runtime_logs(
        futu_log_dir=futu_dir,
        futu_retention_days=14,
        quantpilot_log_dir=quantpilot_dir,
        quantpilot_retention_days=30,
        now=now.isoformat(),
        execute=False,
    )

    assert result["execute"] is False
    assert result["planned_file_count"] == 2
    assert result["planned_bytes"] == len("futu") + len("quantpilot")
    assert result["deleted_file_count"] == 0
    assert futu_log.exists()
    assert quantpilot_log.exists()
