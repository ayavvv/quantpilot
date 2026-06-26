from pathlib import Path

from scripts import collect_us_microstructure as collect


def test_sync_manifests_to_nas_uploads_partition_files_in_one_batch(tmp_path, monkeypatch):
    first = tmp_path / "trades" / "date=2026-06-26" / "symbol=US.AAPL" / "part-1.parquet"
    second = tmp_path / "quotes" / "date=2026-06-26" / "symbol=US.NVDA" / "part-1.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    calls = []

    def fake_copy_many_to_nas(local_paths, local_base, nas_host, nas_dir):
        paths = [Path(path) for path in local_paths]
        calls.append((paths, local_base, nas_host, nas_dir))
        return (
            "ok",
            {path: f"{nas_dir}/{path.relative_to(local_base).as_posix()}" for path in paths},
            "",
        )

    monkeypatch.setattr(collect, "_copy_many_to_nas", fake_copy_many_to_nas)

    updated = collect._sync_manifests_to_nas(
        [
            {"kind": "trades", "local_path": str(first), "row_count": 10},
            {"kind": "quotes", "local_path": str(second), "row_count": 2},
        ],
        local_base=tmp_path,
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )

    assert len(calls) == 1
    assert calls[0][0] == [first, second]
    assert calls[0][1] == tmp_path
    assert calls[0][2] == "nas"
    assert [record["nas_upload_status"] for record in updated] == ["ok", "ok"]
    assert updated[0]["nas_path"].endswith("/trades/date=2026-06-26/symbol=US.AAPL/part-1.parquet")
    assert updated[1]["nas_path"].endswith("/quotes/date=2026-06-26/symbol=US.NVDA/part-1.parquet")
    assert [record["nas_error"] for record in updated] == ["", ""]


def test_sync_manifests_to_nas_marks_whole_batch_failed(tmp_path, monkeypatch):
    first = tmp_path / "trades" / "date=2026-06-26" / "symbol=US.AAPL" / "part-1.parquet"
    second = tmp_path / "order_book" / "date=2026-06-26" / "symbol=US.AAPL" / "part-1.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    def fake_copy_many_to_nas(local_paths, local_base, nas_host, nas_dir):
        paths = [Path(path) for path in local_paths]
        return (
            "failed",
            {path: f"{nas_dir}/{path.relative_to(local_base).as_posix()}" for path in paths},
            "ssh failed",
        )

    monkeypatch.setattr(collect, "_copy_many_to_nas", fake_copy_many_to_nas)

    updated = collect._sync_manifests_to_nas(
        [
            {"kind": "trades", "local_path": str(first), "row_count": 10},
            {"kind": "order_book", "local_path": str(second), "row_count": 1},
        ],
        local_base=tmp_path,
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )

    assert [record["nas_upload_status"] for record in updated] == ["failed", "failed"]
    assert [record["nas_error"] for record in updated] == ["ssh failed", "ssh failed"]
    assert all(record["nas_path"].startswith("/volume1/docker/quantpilot/us_microstructure/") for record in updated)


def test_flush_batch_uploads_all_written_partitions_together(tmp_path, monkeypatch):
    data_uploads = []
    manifest_uploads = []

    def fake_write_partition(rows, *, kind, base_dir, date, run_id, batch_index):
        if not rows:
            return []
        return [
            {
                "kind": kind,
                "symbol": rows[0]["symbol"],
                "date": date,
                "run_id": run_id,
                "batch_index": batch_index,
                "local_path": str(base_dir / kind / f"{rows[0]['symbol']}.parquet"),
                "row_count": len(rows),
            }
        ]

    def fake_copy_many_to_nas(local_paths, local_base, nas_host, nas_dir):
        paths = [Path(path) for path in local_paths]
        data_uploads.append(paths)
        return (
            "ok",
            {path: f"{nas_dir}/{path.relative_to(local_base).as_posix()}" for path in paths},
            "",
        )

    def fake_copy_to_nas(local_path, local_base, nas_host, nas_dir):
        path = Path(local_path)
        manifest_uploads.append(path)
        return "ok", f"{nas_dir}/{path.relative_to(local_base).as_posix()}", ""

    monkeypatch.setattr(collect, "_write_partition", fake_write_partition)
    monkeypatch.setattr(collect, "_copy_many_to_nas", fake_copy_many_to_nas)
    monkeypatch.setattr(collect, "_copy_to_nas", fake_copy_to_nas)

    buffers = {
        "trades": [{"symbol": "US.AAPL"}],
        "order_book": [{"symbol": "US.NVDA"}],
        "quotes": [],
    }

    count = collect._flush_batch(
        buffers,
        local_dir=tmp_path,
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
        date="2026-06-26",
        run_id="run",
        batch_index=1,
    )

    assert count == 2
    assert len(data_uploads) == 1
    assert data_uploads[0] == [
        tmp_path / "trades" / "US.AAPL.parquet",
        tmp_path / "order_book" / "US.NVDA.parquet",
    ]
    assert len(manifest_uploads) == 1
    assert manifest_uploads[0] == tmp_path / "manifests" / "date=2026-06-26" / "manifest-run.jsonl"
    assert all(rows == [] for rows in buffers.values())
