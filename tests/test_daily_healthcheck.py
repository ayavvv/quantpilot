from datetime import datetime

from scripts import daily_healthcheck


def _write_major_money_digest_archive(archive_dir, *, flow_date: str = "2026-04-09"):
    date_tag = flow_date.replace("-", "")
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"{date_tag}_major_money_digest.json").write_text(
        f'{{"flow_date":"{flow_date}"}}',
        encoding="utf-8",
    )
    (archive_dir / f"{date_tag}_major_money_digest.csv").write_text(
        "market,available,total_rows,ok_rows\nA,true,1,1\n",
        encoding="utf-8",
    )


def test_secret_present_accepts_secret_file(tmp_path):
    key_file = tmp_path / "polygon.key"
    key_file.write_text(" file-secret \n", encoding="utf-8")

    assert daily_healthcheck._secret_present("", str(key_file)) is True
    assert daily_healthcheck._secret_present("direct-secret", "") is True
    assert daily_healthcheck._secret_present("", str(tmp_path / "missing.key")) is False


def test_env_value_uses_project_env_defaults(monkeypatch):
    monkeypatch.delenv("ENABLE_US_OTC_PROXY_FLOW", raising=False)
    monkeypatch.setattr(daily_healthcheck, "PROJECT_ENV_DEFAULTS", {"ENABLE_US_OTC_PROXY_FLOW": "true"})

    assert daily_healthcheck._env_value("ENABLE_US_OTC_PROXY_FLOW", "false") == "true"


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


def test_build_snapshot_nightly_includes_major_money_readiness(monkeypatch):
    monkeypatch.setattr(
        daily_healthcheck,
        "local_disk_status",
        lambda: {"path": "/tmp/quantpilot_data", "total_bytes": 100, "used_bytes": 50, "free_bytes": 50, "used_ratio": 0.5},
    )
    monkeypatch.setattr(daily_healthcheck, "latest_local_completed_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_local_a_share_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_signal_date", lambda: "2026-04-09")
    monkeypatch.setattr(daily_healthcheck, "latest_nas_completed_date", lambda: ("", ""))
    monkeypatch.setattr(daily_healthcheck, "latest_nas_a_share_date", lambda: ("", ""))
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
    monkeypatch.setattr(daily_healthcheck, "analyze_capital_flow_artifacts", lambda phase, reference_date="": {"issues": []})
    monkeypatch.setattr(daily_healthcheck, "analyze_market_money_artifacts", lambda reference_date="": {"issues": []})
    monkeypatch.setattr(
        daily_healthcheck.major_money_readiness,
        "build_readiness_snapshot",
        lambda project_dir: {
            "ok": False,
            "expected_markets": ["A", "HK", "US", "US_OTC"],
            "checks": {"cron": {"ok": True}, "email": {"ok": True}, "us_otc_proxy": {"ok": False}},
            "issues": ["US OTC/Pink proxy disabled: set ENABLE_US_OTC_PROXY_FLOW=true"],
        },
    )

    snapshot = daily_healthcheck.build_snapshot("nightly", now=datetime(2026, 4, 10, 19, 0, 0))

    assert snapshot["overall_status"] == "warn"
    assert snapshot["major_money_readiness"]["ok"] is False
    assert snapshot["major_money_readiness"]["expected_markets"] == ["A", "HK", "US", "US_OTC"]
    assert any("Major-money readiness: US OTC/Pink proxy disabled" in issue for issue in snapshot["issues"])


def test_maybe_send_alert_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_healthcheck, "HEALTH_DIR", tmp_path)
    sent_subjects: list[str] = []
    sent_report_dirs: list[object] = []
    monkeypatch.setattr(
        daily_healthcheck,
        "send_email",
        lambda html, subject, report_dir=None: (
            sent_subjects.append(subject),
            sent_report_dirs.append(report_dir),
        ),
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
    assert sent_report_dirs == [daily_healthcheck.HEALTH_REPORT_DIR]


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


def test_analyze_market_money_artifacts_accepts_healthy_sources(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\nSZ.000001,-50\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":3,"market_count":3,'
        '"markets":[{"market":"A","available":true,"ok_rows":2,"total_rows":2},'
        '{"market":"HK","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"US","available":true,"ok_rows":2,"total_rows":2}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    archive_dir = tmp_path / "major_money_digest_archive"
    _write_major_money_digest_archive(archive_dir)
    for market, ok_count in {"HK": 1, "US": 2}.items():
        (flow_dir / f"{market}_latest_status.json").write_text(
            '{"scanner_schema_version":2,"status":"ok","market":"%s","attempted_count":%s,"ok_count":%s,'
            '"error_count":0,"empty_count":0,"ok_ratio":1.0,"finished_at":"2026-04-09T18:00:00+08:00"}'
            % (market, ok_count, ok_count),
            encoding="utf-8",
        )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 2)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", ["HK", "US"])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A", "HK", "US"])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert status["issues"] == []
    assert status["a_share_rank"]["row_count"] == 2
    assert status["digest"]["available_market_count"] == 3
    assert status["digest_archive"]["ok"] is True
    assert status["market_scans"]["US"]["ok_count"] == 2
    assert status["market_scans"]["US"]["scanner_schema_version"] == 2


def test_analyze_market_money_artifacts_flags_missing_digest_archive(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":1,"market_count":1,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1}]}',
        encoding="utf-8",
    )
    archive_dir = tmp_path / "major_money_digest_archive"
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", [])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A"])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert status["digest_archive"]["date_tag"] == "20260409"
    assert any("Major-money digest archive directory missing" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_missing_market_scan(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":1,"market_count":3,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", ["HK"])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert any("status missing for HK" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_old_market_scan_schema(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":2,"market_count":2,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"US","available":true,"ok_rows":1,"total_rows":1}]}',
        encoding="utf-8",
    )
    archive_dir = tmp_path / "major_money_digest_archive"
    _write_major_money_digest_archive(archive_dir)
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    (flow_dir / "US_latest_status.json").write_text(
        '{"status":"ok","market":"US","attempted_count":1,"ok_count":1,'
        '"error_count":0,"empty_count":0,"ok_ratio":1.0,"finished_at":"2026-04-09T18:00:00+08:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", ["US"])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A", "US"])
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_FLOW_MIN_SCHEMA_VERSION", 2)

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert status["market_scans"]["US"]["scanner_schema_version"] == 0
    assert any("scan needs refresh with current scanner: market=US schema=0 min=2" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_unavailable_expected_market(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":1,"market_count":2,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"US_OTC","available":false,"message":"missing"}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", [])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert any("US_OTC" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_partial_digest_coverage(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":2,"market_count":2,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"HK","available":true,"ok_rows":8,"total_rows":10,'
        '"empty_rows":2,"error_rows":0,"non_ok_rows":2}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO", 0.05)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", [])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A", "HK"])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert any(
        "Major-money digest partial source coverage: market=HK non_ok=2/10 (20.0%) empty=2 error=0 max=5.0%"
        in issue
        for issue in status["issues"]
    )


def test_analyze_market_money_artifacts_flags_missing_expected_market_row(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":3,"market_count":3,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"HK","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"US","available":true,"ok_rows":1,"total_rows":1}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    for market in ["HK", "US"]:
        (flow_dir / f"{market}_latest_status.json").write_text(
            '{"scanner_schema_version":2,"status":"ok","market":"%s","attempted_count":1,"ok_count":1,'
            '"error_count":0,"empty_count":0,"ok_ratio":1.0,"finished_at":"2026-04-09T18:00:00+08:00"}'
            % market,
            encoding="utf-8",
        )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", ["HK", "US"])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A", "HK", "US", "US_OTC"])
    monkeypatch.setattr(daily_healthcheck, "US_OTC_PROXY_FLOW_ENABLED", False)

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert any("missing expected market rows: US_OTC" in issue for issue in status["issues"])
    assert any("US OTC/Pink proxy flow disabled" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_us_otc_proxy_missing_key(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-09","available_market_count":1,"market_count":2,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1},'
        '{"market":"US_OTC","available":false,"message":"missing"}]}',
        encoding="utf-8",
    )
    universe = tmp_path / "US_latest_source_universe.csv"
    universe.write_text("code,exchange_type\nUS.AABB,US_PINK\n", encoding="utf-8")
    proxy_dir = tmp_path / "us_otc_proxy"
    proxy_dir.mkdir()
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", [])
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_EXPECTED_MARKETS", ["A", "US_OTC"])
    monkeypatch.setattr(daily_healthcheck, "US_OTC_PROXY_FLOW_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "US_OTC_PROXY_FLOW_PROVIDER", "polygon")
    monkeypatch.setattr(daily_healthcheck, "POLYGON_API_KEY_PRESENT", False)
    monkeypatch.setattr(daily_healthcheck, "US_OTC_PROXY_FLOW_UNIVERSE_CSV", universe)
    monkeypatch.setattr(daily_healthcheck, "US_OTC_PROXY_FLOW_OUTPUT_DIR", proxy_dir)

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert status["us_otc_proxy"]["enabled"] is True
    assert status["us_otc_proxy"]["api_key_present"] is False
    assert any("missing POLYGON_API_KEY" in issue for issue in status["issues"])


def test_analyze_market_money_artifacts_flags_stale_sources(monkeypatch, tmp_path):
    rank = tmp_path / "eastmoney_rank.csv"
    rank.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    digest = tmp_path / "major_money_digest.json"
    digest.write_text(
        '{"flow_date":"2026-04-08","available_market_count":1,"market_count":3,'
        '"markets":[{"market":"A","available":true,"ok_rows":1,"total_rows":1}]}',
        encoding="utf-8",
    )
    flow_dir = tmp_path / "futu_market"
    flow_dir.mkdir()
    (flow_dir / "HK_latest_status.json").write_text(
        '{"status":"ok","market":"HK","attempted_count":1,"ok_count":1,'
        '"error_count":0,"empty_count":0,"ok_ratio":1.0,"finished_at":"2026-04-08T18:00:00+08:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_healthcheck, "HEALTHCHECK_MARKET_MONEY_ENABLED", True)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_RANK_PATH", rank)
    monkeypatch.setattr(daily_healthcheck, "EASTMONEY_FUND_FLOW_MIN_ROWS", 1)
    monkeypatch.setattr(daily_healthcheck, "MAJOR_MONEY_DIGEST_PATH", digest)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_DIR", flow_dir)
    monkeypatch.setattr(daily_healthcheck, "MARKET_CAPITAL_FLOW_MARKETS", ["HK"])

    status = daily_healthcheck.analyze_market_money_artifacts(reference_date="2026-04-09")

    assert any("Major-money digest stale" in issue for issue in status["issues"])
    assert any("capital-flow scan stale" in issue for issue in status["issues"])
