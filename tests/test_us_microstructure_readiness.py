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
            "side_reasons": {"accumulation": "missing 5d validation metrics"},
            "criteria": {"promotion_horizon": 5, "min_observations_per_side": 100},
            "signal_file_count": 2,
            "event_count": 0,
            "forward_return_count": 0,
            "shadow_min_event_score": 65,
            "shadow_event_count": 3,
            "shadow_forward_return_count": 6,
            "price_symbol_count": 15,
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
    assert snapshot["high_confidence_ready"] is False
    assert snapshot["high_confidence_requirements"]["validation_gate_validated"] is False
    assert snapshot["checks"]["manifest"]["row_counts"]["trades"] == 100
    assert snapshot["checks"]["manifest"]["symbol_count"] == 0
    assert snapshot["checks"]["validation_gate"]["state"] == "warmup"
    assert snapshot["checks"]["validation_gate"]["side_reasons"]["accumulation"] == "missing 5d validation metrics"
    assert snapshot["checks"]["validation_gate"]["criteria"]["promotion_horizon"] == 5
    assert snapshot["checks"]["validation_gate"]["signal_file_count"] == 2
    assert snapshot["checks"]["validation_gate"]["shadow_min_event_score"] == 65
    assert snapshot["checks"]["validation_gate"]["shadow_event_count"] == 3
    assert snapshot["checks"]["validation_gate"]["shadow_forward_return_count"] == 6
    assert snapshot["checks"]["validation_gate"]["price_symbol_count"] == 15


def test_readiness_snapshot_marks_high_confidence_ready_when_gates_pass(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-run.jsonl").write_text(
        json.dumps({"kind": "trades", "row_count": 100, "nas_upload_status": "ok"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "validation" / "prices" / "us_daily_prices_status.json",
        {"status": "ok", "symbol_count": 2, "price_row_count": 10, "errors": {}},
    )
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {"state": "validated", "validated": True, "validated_sides": {"accumulation": True}},
    )
    _write_json(
        tmp_path / "reports" / "date=2026-06-01" / "status.json",
        {
            "signal_count": 1,
            "high_count": 1,
            "watch_count": 0,
            "data_quality": {"high_confidence_data_quality_ok": True, "eligible_symbol_count": 1},
        },
    )
    (tmp_path / "reports" / "date=2026-06-01" / "us_microstructure_flow_report.html").write_text(
        "<html></html>",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "us_microstructure_flow_report_latest.html").write_text("<html></html>", encoding="utf-8")

    def fake_launchd(label):
        return 0, "state = not running\nruns = 1\n"

    snapshot = readiness.build_readiness_snapshot(
        base_dir=tmp_path,
        date="2026-06-01",
        launchd_runner=fake_launchd,
    )

    assert snapshot["ok"] is True
    assert snapshot["high_confidence_ready"] is True
    assert snapshot["high_confidence_requirements"]["validation_gate_validated"] is True
    assert snapshot["high_confidence_requirements"]["data_quality_gate_ready"] is True
    assert snapshot["high_confidence_requirements"]["nas_uploads_complete"] is True


def test_readiness_snapshot_requires_full_session_manifest_for_high_confidence(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-20260601T010000.jsonl").write_text(
        json.dumps({"kind": "trades", "symbol": "US.AAPL", "batch_index": 1, "row_count": 1, "nas_upload_status": "failed"})
        + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "manifest-20260601T020000.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "trades", "symbol": "US.AAPL", "batch_index": 2, "row_count": 100, "nas_upload_status": "ok"}),
                json.dumps({"kind": "order_book", "symbol": "US.AAPL", "batch_index": 2, "row_count": 5, "nas_upload_status": "ok"}),
                json.dumps({"kind": "quotes", "symbol": "US.AAPL", "batch_index": 2, "row_count": 5, "nas_upload_status": "ok"}),
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
        {"state": "validated", "validated": True, "validated_sides": {"accumulation": True}},
    )
    _write_json(
        tmp_path / "reports" / "date=2026-06-01" / "status.json",
        {
            "signal_count": 1,
            "high_count": 1,
            "watch_count": 0,
            "data_quality": {"high_confidence_data_quality_ok": True, "eligible_symbol_count": 1},
        },
    )
    (tmp_path / "reports" / "date=2026-06-01" / "us_microstructure_flow_report.html").write_text(
        "<html></html>",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "us_microstructure_flow_report_latest.html").write_text("<html></html>", encoding="utf-8")

    def fake_launchd(label):
        return 0, "state = not running\nruns = 1\n"

    snapshot = readiness.build_readiness_snapshot(
        base_dir=tmp_path,
        date="2026-06-01",
        launchd_runner=fake_launchd,
    )

    assert snapshot["checks"]["manifest"]["ok"] is True
    assert snapshot["checks"]["manifest_full_session"]["ok"] is False
    assert snapshot["high_confidence_ready"] is False
    assert snapshot["high_confidence_requirements"]["nas_uploads_complete"] is False
    assert any("manifest_full_session" in issue and "failed NAS uploads" in issue for issue in snapshot["issues"])


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


def test_manifest_check_flags_missing_kind_coverage_by_symbol(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-run.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "trades", "symbol": "US.AAPL", "batch_index": 1, "row_count": 10, "nas_upload_status": "ok"}),
                json.dumps({"kind": "order_book", "symbol": "US.AAPL", "batch_index": 1, "row_count": 1, "nas_upload_status": "ok"}),
                json.dumps({"kind": "quotes", "symbol": "US.AAPL", "batch_index": 1, "row_count": 1, "nas_upload_status": "ok"}),
                json.dumps({"kind": "trades", "symbol": "US.NVDA", "batch_index": 1, "row_count": 10, "nas_upload_status": "ok"}),
                json.dumps({"kind": "quotes", "symbol": "US.NVDA", "batch_index": 1, "row_count": 1, "nas_upload_status": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = readiness.check_manifest(tmp_path, date="2026-06-01")

    assert result["ok"] is False
    assert result["symbol_count"] == 2
    assert result["complete_symbol_count"] == 1
    assert result["batch_count"] == 1
    assert result["missing_kind_symbols"] == {"order_book": ["US.NVDA"]}
    assert any("missing kind coverage" in issue for issue in result["issues"])


def test_manifest_check_flags_skipped_nas_uploads(tmp_path):
    manifest_dir = tmp_path / "manifests" / "date=2026-06-01"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest-run.jsonl").write_text(
        json.dumps({"kind": "trades", "symbol": "US.AAPL", "row_count": 10, "nas_upload_status": "skipped"}) + "\n",
        encoding="utf-8",
    )

    result = readiness.check_manifest(tmp_path, date="2026-06-01")

    assert result["ok"] is False
    assert result["non_ok_upload_count"] == 1
    assert any("non-ok NAS uploads" in issue for issue in result["issues"])


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
        {
            "signal_count": 1,
            "high_count": 1,
            "watch_count": 0,
            "data_quality": {"high_confidence_data_quality_ok": False},
            "validation_eligibility": {"validation_eligible_count": 0, "score_pass_count": 1},
        },
    )
    (report_dir / "us_microstructure_flow_report.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "reports" / "us_microstructure_flow_report_latest.html").write_text("<html></html>", encoding="utf-8")

    result = readiness.check_report(tmp_path, date="2026-06-01")

    assert result["ok"] is False
    assert result["validation_eligibility"]["score_pass_count"] == 1
    assert any("data-quality gate" in issue for issue in result["issues"])


def test_report_check_does_not_require_latest_alias_for_non_final_report(tmp_path):
    report_dir = tmp_path / "reports" / "date=2026-06-01"
    _write_json(
        report_dir / "status.json",
        {
            "is_final_report": False,
            "latest_alias_updated": False,
            "signal_count": 1,
            "high_count": 0,
            "watch_count": 0,
            "data_quality": {"high_confidence_data_quality_ok": False},
        },
    )
    (report_dir / "us_microstructure_flow_report.html").write_text("<html></html>", encoding="utf-8")

    result = readiness.check_report(tmp_path, date="2026-06-01")

    assert result["ok"] is True
    assert result["is_final_report"] is False
    assert result["latest_required"] is False
    assert result["latest_html_exists"] is False


def test_intraday_replay_check_reports_calibration_counts(tmp_path):
    _write_json(
        tmp_path / "validation" / "intraday_replay" / "date=2026-06-01" / "status.json",
        {
            "event_count": 4,
            "quality_event_count": 3,
            "return_count": 8,
            "quality_return_count": 6,
            "cutoff_count": 2,
            "metric_count": 2,
        },
    )
    _write_json(tmp_path / "validation" / "intraday_replay" / "latest_status.json", {"event_count": 4})

    result = readiness.check_intraday_replay(tmp_path, date="2026-06-01")

    assert result["ok"] is True
    assert result["exists"] is True
    assert result["latest_exists"] is True
    assert result["event_count"] == 4
    assert result["quality_return_count"] == 6
    assert result["cutoff_count"] == 2


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
