import json
from pathlib import Path

from scripts import repair_us_microstructure_nas_uploads as repair_script


def _write_manifest(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_repair_manifest_uploads_retries_failed_rows_and_syncs_manifest(tmp_path, monkeypatch):
    data_file = tmp_path / "trades" / "date=2026-06-01" / "symbol=US.AAPL" / "part-test.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("parquet", encoding="utf-8")
    manifest_path = tmp_path / "manifests" / "date=2026-06-01" / "manifest-run.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "trades",
                "symbol": "US.AAPL",
                "local_path": str(data_file),
                "nas_upload_status": "failed",
                "nas_error": "old error",
            }
        ],
    )
    copied = []

    def fake_copy_to_nas(local_path, local_base, nas_host, nas_dir):
        copied.append(Path(local_path))
        return "ok", f"{nas_dir}/{Path(local_path).relative_to(local_base).as_posix()}", ""

    monkeypatch.setattr(repair_script, "_copy_to_nas", fake_copy_to_nas)

    result = repair_script.repair_manifest_uploads(
        base_dir=tmp_path,
        date="2026-06-01",
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )
    repaired = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])

    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["repaired"] == 1
    assert result["changed_manifest_paths"] == [str(manifest_path)]
    assert copied == [data_file, manifest_path]
    assert repaired["nas_upload_status"] == "ok"
    assert repaired["previous_nas_upload_status"] == "failed"
    assert repaired["previous_nas_error"] == "old error"
    assert repaired["nas_error"] == ""
    assert "repaired_at" in repaired


def test_repair_manifest_uploads_keeps_missing_local_file_failed(tmp_path):
    missing_file = tmp_path / "trades" / "date=2026-06-01" / "symbol=US.AAPL" / "missing.parquet"
    manifest_path = tmp_path / "manifests" / "date=2026-06-01" / "manifest-run.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "trades",
                "symbol": "US.AAPL",
                "local_path": str(missing_file),
                "nas_upload_status": "failed",
            }
        ],
    )

    result = repair_script.repair_manifest_uploads(
        base_dir=tmp_path,
        date="2026-06-01",
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )
    repaired = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])

    assert result["ok"] is False
    assert result["checked"] == 1
    assert result["missing_local"] == 1
    assert repaired["nas_upload_status"] == "failed"
    assert "local file missing" in repaired["nas_error"]
