import json

from scripts import us_microstructure_readiness as readiness


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_readiness_snapshot_accepts_ready_warmup_system(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-run.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "trades", "row_count": 100, "nas_upload_status": "ok"}),
                json.dumps({"kind": "order_book", "row_count": 5, "nas_upload_status": "ok"}),
                json.dumps({"kind": "quotes", "row_count": 2, "nas_upload_status": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "validation" / "prices" / "us_daily_prices_status.json",
        {"status": "ok", "symbol_count": 2, "price_row_count": 10, "errors": {}},
    )
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {
            "state": "warmup",
            "validated": False,
            "validated_sides": {"accumulation": False, "distribution": False},
            "event_count": 0,
            "forward_return_count": 0,
            "reason": "collecting samples",
        },
    )
    _write_json(
        tmp_path / "reports" / "date=2026-06-01" / "status.json",
        {"signal_count": 2, "high_count": 0, "watch_count": 0},
    )
    (tmp_path / "reports" / "date=2026-06-01" / "us_microstructure_flow_report.html").write_text(
        "<html></html>",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "us_microstructure_flow_report_latest.html").write_text("<html></html>", encoding="utf-8")

    def fake_launchd(label):
        return 0, "state = not running\nruns = 0\n"

    snapshot = readiness.build_readiness_snapshot(
        base_dir=tmp_path,
        date="2026-06-01",
        launchd_runner=fake_launchd,
    )

    assert snapshot["ok"] is True
    assert snapshot["checks"]["manifest"]["row_counts"]["trades"] == 100
    assert snapshot["checks"]["validation_gate"]["state"] == "warmup"


def test_readiness_snapshot_flags_failed_upload_and_missing_launchd(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-run.jsonl").write_text(
        json.dumps({"kind": "trades", "row_count": 1, "nas_upload_status": "failed"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "validation" / "prices" / "us_daily_prices_status.json",
        {"status": "ok", "symbol_count": 1, "price_row_count": 1, "errors": {}},
    )
    _write_json(tmp_path / "validation" / "active_gate.json", {"state": "warmup", "validated": False})

    def missing_launchd(label):
        return 113, "Could not find service"

    snapshot = readiness.build_readiness_snapshot(
        base_dir=tmp_path,
        date="2026-06-01",
        launchd_runner=missing_launchd,
    )

    assert snapshot["ok"] is False
    assert any("failed NAS uploads" in issue for issue in snapshot["issues"])
    assert any("launchd service not loaded" in issue for issue in snapshot["issues"])


def test_manifest_check_defaults_to_latest_run(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-20260601T010000.jsonl").write_text(
        json.dumps({"kind": "trades", "row_count": 1, "nas_upload_status": "failed"}) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "manifest-20260601T020000.jsonl").write_text(
        json.dumps({"kind": "trades", "row_count": 2, "nas_upload_status": "ok"}) + "\n",
        encoding="utf-8",
    )

    latest = readiness.check_manifest(tmp_path, date="2026-06-01")
    all_runs = readiness.check_manifest(tmp_path, date="2026-06-01", latest_only=False)

    assert latest["ok"] is True
    assert latest["row_counts"]["trades"] == 2
    assert all_runs["ok"] is False
    assert all_runs["status_counts"]["failed"] == 1


def test_write_readiness_snapshot_writes_latest_and_dated_file(tmp_path):
    snapshot = {
        "ok": True,
        "date": "2026-06-01",
        "checks": {},
        "issues": [],
    }

    path = readiness.write_readiness_snapshot(tmp_path, snapshot)

    assert path.name == "us_microstructure_readiness_20260601.json"
    assert path.exists()
    assert (tmp_path / "readiness" / "us_microstructure_readiness_latest.json").exists()


def test_report_check_flags_high_signals_without_data_quality_gate(tmp_path):
    report_dir = tmp_path / "reports" / "date=2026-06-01"
    _write_json(
        report_dir / "status.json",
        {"signal_count": 1, "high_count": 1, "watch_count": 0, "data_quality": {"high_confidence_data_quality_ok": False}},
    )
    (report_dir / "us_microstructure_flow_report.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "reports" / "us_microstructure_flow_report_latest.html").write_text("<html></html>", encoding="utf-8")

    result = readiness.check_report(tmp_path, date="2026-06-01")

    assert result["ok"] is False
    assert any("data-quality gate" in issue for issue in result["issues"])


def test_sync_readiness_outputs_copies_snapshots_to_nas(tmp_path, monkeypatch):
    dated_path = tmp_path / "readiness" / "us_microstructure_readiness_20260601.json"
    latest_path = tmp_path / "readiness" / "us_microstructure_readiness_latest.json"
    dated_path.parent.mkdir(parents=True)
    dated_path.write_text("{}", encoding="utf-8")
    latest_path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_copy_to_nas(local_path, local_base, nas_host, nas_dir):
        calls.append((local_path, local_base, nas_host, nas_dir))
        return "ok", f"{nas_dir}/readiness/{local_path.name}", ""

    monkeypatch.setattr(readiness, "_copy_to_nas", fake_copy_to_nas)

    results = readiness.sync_readiness_outputs(
        [dated_path, latest_path],
        base_dir=tmp_path,
        nas_host="nas",
        nas_dir="/volume1/docker/quantpilot/us_microstructure",
    )

    assert [item["status"] for item in results] == ["ok", "ok"]
    assert len(calls) == 2
    assert calls[0][2] == "nas"
