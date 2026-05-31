from datetime import datetime

from scripts import daily_healthcheck


def test_build_snapshot_pretrade_flags_stale_signal_and_nas_lag(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 50, "free_bytes": 50, "used_ratio": 0.5},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-08")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-08")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(daily_healthcheck, "expected_pretrade_signal_date", lambda now=None: ("2026-04-09", ""))
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_trade_log",
        lambda today: {
            "starts": 0,
            "done": 0,
            "order_failures": 0,
            "order_fills": 0,
            "errors": 0,
            "stale_signal_errors": [],
            "latest_line": "",
        },
    )
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_daily_logs",
        lambda today: {
            "timeouts": 0,
            "inference_failures": 0,
            "retry_activity": 0,
            "latest_daily_line": "",
            "latest_retry_line": "",
        },
    )
    monkeypatch.setattr(daily_healthcheck, "process_running", lambda patterns: False)

    snapshot = daily_healthcheck.build_snapshot("pretrade", now=datetime(2026, 4, 10, 10, 0, 0))

    assert snapshot["overall_status"] == "error"
    assert any("Signal stale" in issue for issue in snapshot["issues"])
    assert any("lags NAS" in issue for issue in snapshot["issues"])


def test_build_snapshot_trade_warns_on_failed_orders(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 50, "free_bytes": 50, "used_ratio": 0.5},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(daily_healthcheck, "expected_pretrade_signal_date", lambda now=None: ("2026-04-09", ""))
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_trade_log",
        lambda today: {
            "starts": 1,
            "done": 1,
            "order_failures": 1,
            "order_fills": 3,
            "errors": 0,
            "stale_signal_errors": [],
            "latest_line": "done",
        },
    )
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_daily_logs",
        lambda today: {
            "timeouts": 0,
            "inference_failures": 0,
            "retry_activity": 0,
            "latest_daily_line": "",
            "latest_retry_line": "",
        },
    )
    monkeypatch.setattr(daily_healthcheck, "process_running", lambda patterns: False)

    snapshot = daily_healthcheck.build_snapshot("trade", now=datetime(2026, 4, 10, 15, 0, 0))

    assert snapshot["overall_status"] == "warn"
    assert any("failed order" in issue for issue in snapshot["issues"])


def test_build_snapshot_nightly_flags_target_date_not_reached(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 50, "free_bytes": 50, "used_ratio": 0.5},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("2026-04-09", ""))
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_trade_log",
        lambda today: {
            "starts": 0,
            "done": 0,
            "order_failures": 0,
            "order_fills": 0,
            "errors": 0,
            "stale_signal_errors": [],
            "latest_line": "",
        },
    )
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_daily_logs",
        lambda today: {
            "timeouts": 1,
            "inference_failures": 0,
            "retry_activity": 1,
            "latest_daily_line": "",
            "latest_retry_line": "",
        },
    )
    monkeypatch.setattr(daily_healthcheck, "process_running", lambda patterns: False)

    snapshot = daily_healthcheck.build_snapshot(
        "nightly",
        now=datetime(2026, 4, 11, 11, 0, 0),
        target_a_share_date="2026-04-10",
    )

    assert snapshot["overall_status"] == "error"
    assert any("below nightly target" in issue for issue in snapshot["issues"])
    assert snapshot["target_a_share_date"] == "2026-04-10"


def test_maybe_send_alert_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_healthcheck, "HEALTH_DIR", tmp_path)
    sent_subjects: list[str] = []
    monkeypatch.setattr(
        daily_healthcheck,
        "send_email",
        lambda html, subject: sent_subjects.append(subject),
    )
    snapshot = {
        "phase": "pretrade",
        "overall_status": "error",
        "issues": ["Signal stale"],
        "date": "2026-04-10",
        "timestamp": "2026-04-10 10:00:00",
        "local": {
            "completed_a_share_date": "2026-04-09",
            "latest_a_share_date": "2026-04-09",
            "latest_signal_date": "2026-04-08",
            "signal_aligned": False,
        },
        "nas": {
            "completed_a_share_date": "2026-04-09",
            "latest_a_share_date": "2026-04-09",
            "query_error": "",
            "latest_query_error": "",
        },
        "trade": {
            "starts": 0,
            "done": 0,
            "order_fills": 0,
            "order_failures": 0,
            "errors": 0,
        },
    }

    assert daily_healthcheck.maybe_send_alert(snapshot, "error") is True
    assert daily_healthcheck.maybe_send_alert(snapshot, "error") is False
    assert len(sent_subjects) == 1


def test_build_snapshot_pretrade_errors_when_expected_target_not_reached(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 50, "free_bytes": 50, "used_ratio": 0.5},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-15")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-15")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-15")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("2026-04-15", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("2026-04-15", ""))
    monkeypatch.setattr(daily_healthcheck, "expected_pretrade_signal_date", lambda now=None: ("2026-04-16", ""))
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_trade_log",
        lambda today: {
            "starts": 0,
            "done": 0,
            "order_failures": 0,
            "order_fills": 0,
            "errors": 0,
            "stale_signal_errors": [],
            "latest_line": "",
        },
    )
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_daily_logs",
        lambda today: {
            "timeouts": 0,
            "inference_failures": 0,
            "retry_activity": 0,
            "latest_daily_line": "",
            "latest_retry_line": "",
        },
    )
    monkeypatch.setattr(daily_healthcheck, "process_running", lambda patterns: False)

    snapshot = daily_healthcheck.build_snapshot("pretrade", now=datetime(2026, 4, 17, 10, 0, 0))

    assert snapshot["overall_status"] == "error"
    assert snapshot["expected_signal_date"] == "2026-04-16"
    assert any("expected pre-trade target" in issue for issue in snapshot["issues"])


def test_build_snapshot_warns_when_disk_usage_high(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 85, "free_bytes": 15, "used_ratio": 0.85},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-16")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-16")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-16")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("2026-04-16", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("2026-04-16", ""))
    monkeypatch.setattr(daily_healthcheck, "expected_pretrade_signal_date", lambda now=None: ("2026-04-16", ""))
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_trade_log",
        lambda today: {
            "starts": 0,
            "done": 0,
            "order_failures": 0,
            "order_fills": 0,
            "errors": 0,
            "stale_signal_errors": [],
            "latest_line": "",
        },
    )
    monkeypatch.setattr(
        daily_healthcheck,
        "analyze_daily_logs",
        lambda today: {
            "timeouts": 0,
            "inference_failures": 0,
            "retry_activity": 0,
            "latest_daily_line": "",
            "latest_retry_line": "",
        },
    )
    monkeypatch.setattr(daily_healthcheck, "process_running", lambda patterns: False)

    snapshot = daily_healthcheck.build_snapshot("pretrade", now=datetime(2026, 4, 17, 10, 0, 0))

    assert snapshot["overall_status"] == "warn"
    assert any("disk usage above warning threshold" in issue for issue in snapshot["issues"])


def test_analyze_capital_flow_artifacts_nightly_flags_stale_and_review(monkeypatch, tmp_path):
    overlay = tmp_path / "futu_capital_flow_signal_overlay_latest.csv"
    overlay.write_text(
        "code,signal_date,capital_flow_label,capital_flow_latest_date\n"
        "SH.600000,2026-04-08,risk_flag_main_outflow,2026-04-08\n",
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "20260408_overlay.csv").write_text(
        "code,signal_date,capital_flow_label\n"
        "SH.600000,2026-04-08,risk_flag_main_outflow\n",
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        '{"overall_action":"review_filter","message":"review","criteria":{"min_date_count":20}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_CAPITAL_FLOW_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "CAPITAL_FLOW_OVERLAY_PATH", overlay)
    monkeypatch.setattr(daily_healthcheck, "CAPITAL_FLOW_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(daily_healthcheck, "CAPITAL_FLOW_GATE_PATH", gate)

    status = daily_healthcheck.analyze_capital_flow_artifacts("nightly", reference_date="2026-04-09")

    assert status["daily_overlay"]["signal_date"] == "2026-04-08"
    assert any("latest overlay stale" in issue for issue in status["issues"])
    assert any("archive stale" in issue for issue in status["issues"])
    assert any("manual review" in issue for issue in status["issues"])


def test_analyze_capital_flow_artifacts_trade_accepts_current_pretrade_overlay(monkeypatch, tmp_path):
    overlay = tmp_path / "pretrade_futu_capital_flow_signal_overlay_latest.csv"
    overlay.write_text(
        "code,signal_date,capital_flow_label,capital_flow_latest_date\n"
        "SH.600000,2026-04-09,capital_flow_confirm,2026-04-09\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_CAPITAL_FLOW_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "PRETRADE_CAPITAL_FLOW_OVERLAY_PATH", overlay)

    status = daily_healthcheck.analyze_capital_flow_artifacts("trade", reference_date="2026-04-09")

    assert status["issues"] == []
    assert status["pretrade_overlay"]["row_count"] == 1
    assert status["pretrade_overlay"]["labels"] == {"capital_flow_confirm": 1}
