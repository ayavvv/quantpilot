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
    second_data_file = tmp_path / "quotes" / "date=2026-06-01" / "symbol=US.NVDA" / "part-test.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("parquet", encoding="utf-8")
    second_data_file.parent.mkdir(parents=True)
    second_data_file.write_text("parquet", encoding="utf-8")
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
            },
            {
                "kind": "quotes",
                "symbol": "US.NVDA",
                "local_path": str(second_data_file),
                "nas_upload_status": "failed",
                "nas_error": "second old error",
            }
        ],
    )
    batches = []

    def fake_sync_paths_to_nas(local_paths, *, local_base, nas_host, nas_dir):
        paths = [Path(path) for path in local_paths]
        batches.append(paths)
        return [
            {
                "local_path": str(path),
                "nas_path": f"{nas_dir}/{path.relative_to(local_base).as_posix()}",
                "status": "ok",
                "error": "",
            }
            for path in paths
        ]

    monkeypatch.setattr(repair_script, "_nas_sync_enabled", lambda nas_host, nas_dir: True)
    monkeypatch.setattr(repair_script, "_sync_paths_to_nas", fake_sync_paths_to_nas)

    result = repair_script.repair_manifest_uploads(
        base_dir=tmp_path,
        date="2026-06-01",
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )
    repaired_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert result["ok"] is True
    assert result["checked"] == 2
    assert result["repaired"] == 2
    assert result["changed_manifest_paths"] == [str(manifest_path)]
    assert batches == [[data_file, second_data_file], [manifest_path]]
    assert [row["nas_upload_status"] for row in repaired_rows] == ["ok", "ok"]
    assert [row["previous_nas_upload_status"] for row in repaired_rows] == ["failed", "failed"]
    assert [row["previous_nas_error"] for row in repaired_rows] == ["old error", "second old error"]
    assert [row["nas_error"] for row in repaired_rows] == ["", ""]
    assert all("repaired_at" in row for row in repaired_rows)


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
