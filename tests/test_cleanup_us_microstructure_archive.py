import json
from pathlib import Path

from scripts import cleanup_us_microstructure_archive as cleanup


def _raw_path(base: Path, day: str, kind: str, symbol: str = "US.AAPL", name: str = "part-1.parquet") -> Path:
    path = base / kind / f"date={day}" / f"symbol={symbol}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{kind}-{day}", encoding="utf-8")
    return path


def _write_manifest(base: Path, day: str, records: list[dict]) -> None:
    manifest = base / "manifests" / f"date={day}" / "manifest-run.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _ok_record(path: Path, *, kind: str, day: str) -> dict:
    return {
        "kind": kind,
        "date": day,
        "local_path": str(path),
        "nas_path": f"/volume1/docker/quantpilot/us_microstructure/{path.name}",
        "nas_upload_status": "ok",
    }


def test_cleanup_deletes_only_old_fully_archived_raw_files(tmp_path):
    old_day = "2026-06-20"
    recent_day = "2026-06-30"
    old_files = [_raw_path(tmp_path, old_day, kind) for kind in cleanup.RAW_KINDS]
    recent_file = _raw_path(tmp_path, recent_day, "trades")
    feature = tmp_path / "features_1m" / f"date={old_day}" / "part-us-microstructure-features.parquet"
    feature.parent.mkdir(parents=True)
    feature.write_text("features stay local", encoding="utf-8")
    _write_manifest(tmp_path, old_day, [_ok_record(path, kind=path.parts[-4], day=old_day) for path in old_files])
    _write_manifest(tmp_path, recent_day, [_ok_record(recent_file, kind="trades", day=recent_day)])

    result = cleanup.cleanup_us_microstructure_archive(
        base_dir=tmp_path,
        retention_days=7,
        today="2026-07-01",
        execute=True,
    )

    assert result["eligible_dates"] == [old_day]
    assert result["deleted_file_count"] == 3
    assert all(not path.exists() for path in old_files)
    assert recent_file.exists()
    assert feature.exists()


def test_cleanup_skips_date_with_unmanifested_local_raw_file(tmp_path):
    day = "2026-06-20"
    archived = _raw_path(tmp_path, day, "trades", name="part-ok.parquet")
    unarchived = _raw_path(tmp_path, day, "trades", name="part-missing-manifest.parquet")
    _write_manifest(tmp_path, day, [_ok_record(archived, kind="trades", day=day)])

    result = cleanup.cleanup_us_microstructure_archive(
        base_dir=tmp_path,
        retention_days=7,
        today="2026-07-01",
        execute=True,
    )

    assert result["eligible_dates"] == []
    assert result["deleted_file_count"] == 0
    assert archived.exists()
    assert unarchived.exists()
    assert "without ok NAS manifest" in result["skipped_dates"][0]["issues"][-1]


def test_cleanup_skips_non_ok_manifest_records_by_default(tmp_path):
    day = "2026-06-20"
    raw_file = _raw_path(tmp_path, day, "quotes")
    record = _ok_record(raw_file, kind="quotes", day=day)
    record["nas_upload_status"] = "failed"
    record["nas_error"] = "network"
    _write_manifest(tmp_path, day, [record])

    result = cleanup.cleanup_us_microstructure_archive(
        base_dir=tmp_path,
        retention_days=7,
        today="2026-07-01",
        execute=True,
    )

    assert result["eligible_dates"] == []
    assert result["deleted_file_count"] == 0
    assert raw_file.exists()
    assert "non-ok" in result["skipped_dates"][0]["issues"][0]


def test_cleanup_can_require_smb_mount_copy_to_exist(tmp_path):
    day = "2026-06-20"
    raw_file = _raw_path(tmp_path, day, "order_book")
    _write_manifest(tmp_path, day, [_ok_record(raw_file, kind="order_book", day=day)])
    mount = tmp_path / "smb_mount"
    mount.mkdir()

    missing = cleanup.plan_date_cleanup(
        tmp_path,
        day,
        nas_mount_dir=mount,
        verify_nas_mount=True,
    )

    assert missing["eligible"] is False
    assert missing["missing_mount_file_count"] == 1

    remote_copy = mount / raw_file.relative_to(tmp_path)
    remote_copy.parent.mkdir(parents=True)
    remote_copy.write_text("remote", encoding="utf-8")

    present = cleanup.plan_date_cleanup(
        tmp_path,
        day,
        nas_mount_dir=mount,
        verify_nas_mount=True,
    )

    assert present["eligible"] is True
