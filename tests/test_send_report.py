from pathlib import Path

from reporter import send_report
from datetime import datetime


def test_load_env_defaults_uses_reporter_env_without_overriding(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "SMTP_HOST=smtp.gmail.com",
                "SMTP_PORT=587",
                "SMTP_USER=fallback@example.com",
                "SMTP_PASSWORD=fallback-secret",
                "REPORT_TO=owner@example.com",
                "REPORT_FROM=sender@example.com",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(send_report, "REPORTER_ENV_PATH", env_path)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.setenv("SMTP_USER", "env@example.com")

    config = send_report.email_config()

    assert config["smtp_host"] == "smtp.gmail.com"
    assert config["smtp_port"] == 587
    assert config["smtp_user"] == "env@example.com"
    assert config["smtp_password"] == "fallback-secret"
    assert config["report_to"] == "owner@example.com"
    assert config["report_from"] == "sender@example.com"
    assert config["mail_app_from"] == "sender@example.com"
    assert config["report_delivery_method"] == "auto"


def test_send_email_falls_back_to_mail_app(monkeypatch, tmp_path):
    monkeypatch.setattr(send_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        send_report,
        "email_config",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "report_to": "owner@example.com",
            "report_from": "sender@example.com",
            "mail_app_from": "icloud@example.com",
            "report_delivery_method": "auto",
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": True,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args, **kwargs: (False, "sendmail down"))
    fallback_calls = []
    monkeypatch.setattr(
        send_report,
        "send_via_mail_app",
        lambda subject, report_to, report_from, report_path, attachment_paths=None: (
            fallback_calls.append((subject, report_to, report_from, report_path, attachment_paths)) or True,
            "",
        ),
    )

    assert send_report.send_email("<html>hi</html>", "subject") is True
    assert len(fallback_calls) == 1
    assert fallback_calls[0][2] == "icloud@example.com"
    assert fallback_calls[0][3] == tmp_path / f"report_{send_report.datetime.now().strftime('%Y%m%d')}.html"
    assert fallback_calls[0][4] is None


def test_send_email_returns_false_when_all_delivery_paths_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(send_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        send_report,
        "email_config",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "report_to": "owner@example.com",
            "report_from": "sender@example.com",
            "mail_app_from": "icloud@example.com",
            "report_delivery_method": "auto",
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": True,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args, **kwargs: (False, "sendmail down"))
    monkeypatch.setattr(send_report, "send_via_mail_app", lambda *args, **kwargs: (False, "mail down"))

    assert send_report.send_email("<html>hi</html>", "subject") is False
    assert (tmp_path / f"report_{send_report.datetime.now().strftime('%Y%m%d')}.html").exists()


def test_send_email_returns_true_when_sendmail_fallback_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(send_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        send_report,
        "email_config",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "report_to": "owner@example.com",
            "report_from": "sender@example.com",
            "mail_app_from": "icloud@example.com",
            "report_delivery_method": "auto",
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": False,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args, **kwargs: (True, ""))

    assert send_report.send_email("<html>hi</html>", "subject") is True


def test_send_email_uses_explicit_report_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(send_report, "REPORT_DIR", Path("/data/reports"))
    monkeypatch.setattr(
        send_report,
        "email_config",
        lambda: {
            "smtp_host": "smtp.mail.me.com",
            "smtp_port": 465,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "report_to": "owner@example.com",
            "report_from": "sender@example.com",
            "mail_app_from": "icloud@example.com",
            "report_delivery_method": "smtp",
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": False,
            "mail_app_fallback": False,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda *args: (True, ""))

    explicit_dir = tmp_path / "weekly_output"
    assert send_report.send_email("<html>hi</html>", "subject", report_dir=explicit_dir) is True
    assert (explicit_dir / f"report_{send_report.datetime.now().strftime('%Y%m%d')}.html").exists()


def test_build_message_attaches_files(tmp_path):
    attachment = tmp_path / "metrics.txt"
    attachment.write_text("ann_return: 10.00%\n", encoding="utf-8")

    msg = send_report.build_message(
        "<html>hi</html>",
        "subject",
        "sender@example.com",
        "owner@example.com",
        attachment_paths=[attachment],
    )

    filenames = [
        part.get_filename()
        for part in msg.walk()
        if part.get_filename()
    ]
    assert "metrics.txt" in filenames


def test_check_major_money_digest_status_summarises_markets(tmp_path):
    digest = {
        "generated_at": "2026-05-31T00:00:00+08:00",
        "flow_date": "2026-05-29",
        "market_count": 3,
        "available_market_count": 1,
        "entry_count": 1,
        "exit_count": 1,
        "amount_by_currency": {
            "CNY": {"entry_amount": 80_000_000, "exit_amount": 70_000_000, "net_amount": 10_000_000}
        },
        "markets": [
            {
                "market": "A",
                "source": "eastmoney",
                "currency": "CNY",
                "available": True,
                "flow_date": "2026-05-29",
                "total_rows": 2,
                "ok_rows": 2,
                "non_ok_rows": 0,
                "exchange_types": {"SSE": 1, "SZSE": 1},
                "coverage_notes": ["source coverage note"],
                "entry_count": 1,
                "entry_amount": 80_000_000,
                "exit_count": 1,
                "exit_amount": 70_000_000,
                "net_amount": 10_000_000,
                "top_entries": [{"code": "SH.600000", "name": "Entry", "main_flow": 80_000_000}],
                "top_exits": [{"code": "SZ.000001", "name": "Exit", "main_flow": -70_000_000}],
            },
            {"market": "HK", "currency": "HKD", "available": False, "message": "missing"},
        ],
    }
    path = tmp_path / "major_money_digest.json"
    path.write_text(send_report.json.dumps(digest), encoding="utf-8")

    status = send_report.check_major_money_digest_status(digest_json=path)

    assert status["major_money_available"] is True
    assert status["major_money_date"] == "2026-05-29"
    assert status["major_money_markets"][0]["coverage"] == "2/2 (SSE=1, SZSE=1)"
    assert status["major_money_markets"][0]["coverage_note"] == "source coverage note"
    assert status["major_money_markets"][0]["entry_amount"] == "80.0m CNY"
    assert status["major_money_markets"][1]["row_class"] == "coverage-missing"
    assert status["major_money_markets"][1]["coverage_note"] == "missing"
    assert "Missing coverage: HK" in status["major_money_message"]
    assert status["major_money_summary"] == (
        "Major entries: 1; major exits: 1. "
        "Amounts by currency: CNY: entry 80.0m, exit 70.0m, net 10.0m."
    )
    assert status["major_money_subject_summary"] == "MM 1 in/1 out; 1/3 src; missing HK"
    assert status["major_money_top_entries"][0]["code"] == "SH.600000"
    assert status["major_money_top_exits"][0]["main_flow"] == "-70.0m CNY"


def test_check_major_money_digest_status_keeps_top_rows_per_market(tmp_path):
    digest = {
        "flow_date": "2026-05-29",
        "market_count": 2,
        "available_market_count": 2,
        "markets": [
            {
                "market": "A",
                "source": "eastmoney",
                "currency": "CNY",
                "available": True,
                "total_rows": 2,
                "ok_rows": 2,
                "top_entries": [
                    {"code": "SH.A1", "name": "A Entry 1", "main_flow": 100},
                    {"code": "SH.A2", "name": "A Entry 2", "main_flow": 90},
                ],
                "top_exits": [
                    {"code": "SH.X1", "name": "A Exit 1", "main_flow": -100},
                    {"code": "SH.X2", "name": "A Exit 2", "main_flow": -90},
                ],
            },
            {
                "market": "US",
                "source": "futu",
                "currency": "USD",
                "available": True,
                "total_rows": 2,
                "ok_rows": 2,
                "top_entries": [
                    {"code": "US.U1", "name": "US Entry 1", "main_flow": 10},
                    {"code": "US.U2", "name": "US Entry 2", "main_flow": 9},
                ],
                "top_exits": [
                    {"code": "US.Y1", "name": "US Exit 1", "main_flow": -10},
                    {"code": "US.Y2", "name": "US Exit 2", "main_flow": -9},
                ],
            },
        ],
    }
    path = tmp_path / "major_money_digest.json"
    path.write_text(send_report.json.dumps(digest), encoding="utf-8")

    status = send_report.check_major_money_digest_status(digest_json=path, top_n=1)

    assert [row["code"] for row in status["major_money_top_entries"]] == ["SH.A1", "US.U1"]
    assert [row["code"] for row in status["major_money_top_exits"]] == ["SH.X1", "US.Y1"]


def test_check_major_money_digest_status_missing_file_degrades(tmp_path):
    status = send_report.check_major_money_digest_status(digest_json=tmp_path / "missing.json")

    assert status["major_money_available"] is False
    assert status["major_money_markets"] == []
    assert status["major_money_subject_summary"] == ""
    assert "No market-wide major-money digest found" in status["major_money_message"]


def test_build_report_subject_includes_major_money_summary():
    subject = send_report.build_report_subject(
        "2026-05-31",
        {
            "major_money_available": True,
            "major_money_subject_summary": "MM 368 in/783 out; 3/4 src; missing US_OTC",
        },
    )

    assert subject == "QuantPilot Daily Report - 2026-05-31 | MM 368 in/783 out; 3/4 src; missing US_OTC"


def test_check_stealth_money_status_summarises_candidates_and_accuracy(tmp_path):
    major_csv = tmp_path / "major_force.csv"
    major_csv.write_text(
        "\n".join(
            [
                "code,date,score,rank,distribution_score,distribution_rank,stage,amount,market_positive_rate_20,reason",
                "SH.600000,2026-05-29,91,1,20,300,accumulation_candidate,80000000,0.55,volume_expansion",
                "SH.600001,2026-05-29,93,2,18,320,accumulation_candidate,70000000,0.40,volume_expansion",
                "SZ.000001,2026-05-29,45,300,88,2,distribution_risk,90000000,0.45,20d_negative_flow",
                "SZ.000002,2026-05-29,44,320,90,3,distribution_risk,85000000,0.60,20d_negative_flow",
            ]
        ),
        encoding="utf-8",
    )
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "signal_side,top_n,horizon,date_count,avg_hit_rate,avg_alpha",
                "buy,30,10,80,0.6125,0.013",
                "sell,30,10,80,0.55,0.009",
            ]
        ),
        encoding="utf-8",
    )
    validation_json = tmp_path / "validation.json"
    validation_json.write_text(
        send_report.json.dumps(
            {
                "validated": True,
                "rules": [
                    {
                        "side": "buy",
                        "horizon": 10,
                        "rank_n": 10,
                        "min_score": 80,
                        "min_market_positive_rate_20": 0.50,
                        "test": {"avg_hit_rate": 0.6125, "avg_alpha": 0.013, "date_count": 80},
                    },
                    {
                        "side": "sell",
                        "horizon": 10,
                        "rank_n": 10,
                        "min_score": 80,
                        "max_market_positive_rate_20": 0.50,
                        "test": {"avg_hit_rate": 0.55, "avg_alpha": 0.009, "date_count": 80},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    status = send_report.check_stealth_money_status(
        major_csv=major_csv,
        eval_summary_csv=summary_csv,
        validation_json=validation_json,
    )

    assert status["stealth_money_available"] is True
    assert status["stealth_money_date"] == "2026-05-29"
    assert status["stealth_money_subject_summary"] == "Stealth 1 buy/1 sell"
    assert status["stealth_money_buys"][0]["code"] == "SH.600000"
    assert status["stealth_money_sells"][0]["code"] == "SZ.000001"
    assert "buy 10d: 61.25% hit" in status["stealth_money_validation"]
    assert "sell 10d: 55.00% hit" in status["stealth_money_validation"]


def test_check_stealth_money_status_hides_unvalidated_candidates(tmp_path):
    major_csv = tmp_path / "major_force.csv"
    major_csv.write_text(
        "code,date,score,rank,distribution_score,distribution_rank,stage,amount,reason\n"
        "SH.600000,2026-05-29,91,1,20,300,accumulation_candidate,80000000,volume_expansion\n",
        encoding="utf-8",
    )
    validation_json = tmp_path / "validation.json"
    validation_json.write_text('{"validated":false,"rules":[],"message":"failed gate"}', encoding="utf-8")

    status = send_report.check_stealth_money_status(
        major_csv=major_csv,
        eval_summary_csv=tmp_path / "missing_summary.csv",
        validation_json=validation_json,
    )

    assert status["stealth_money_available"] is False
    assert status["stealth_money_subject_summary"] == ""
    assert status["stealth_money_buys"] == []
    assert "No validated stealth rule" in status["stealth_money_message"]


def test_check_stealth_money_status_explains_missing_buy_rule(tmp_path):
    major_csv = tmp_path / "major_force.csv"
    major_csv.write_text(
        "\n".join(
            [
                "code,date,score,rank,distribution_score,distribution_rank,stage,amount,close_location_10,reason",
                "SH.600000,2026-05-29,91,1,20,300,accumulation_candidate,80000000,0.70,volume_expansion",
                "SZ.000001,2026-05-29,45,300,95,1,distribution_risk,90000000,0.30,20d_negative_flow",
            ]
        ),
        encoding="utf-8",
    )
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        "signal_side,top_n,horizon,date_count,avg_hit_rate,avg_alpha\nsell,10,5,40,0.6,0.011\n",
        encoding="utf-8",
    )
    validation_json = tmp_path / "validation.json"
    validation_json.write_text(
        send_report.json.dumps(
            {
                "validated": True,
                "candidate_rule_count_by_side": {"buy": 3348, "sell": 5832},
                "train_passed_count_by_side": {"buy": 28, "sell": 142},
                "rules": [
                    {
                        "side": "sell",
                        "horizon": 5,
                        "rank_n": 10,
                        "min_score": 92,
                        "max_close_location_10": 0.35,
                        "test": {"avg_hit_rate": 0.6, "avg_alpha": 0.011, "date_count": 40},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = send_report.check_stealth_money_status(
        major_csv=major_csv,
        eval_summary_csv=summary_csv,
        validation_json=validation_json,
    )

    assert status["stealth_money_subject_summary"] == "Stealth 0 buy/1 sell"
    assert status["stealth_money_buys"] == []
    assert status["stealth_money_sells"][0]["code"] == "SZ.000001"
    assert "sell 5d: 60.00% hit" in status["stealth_money_validation"]
    assert "buy: no validated rule (28 train-passed/3348 searched)" in status["stealth_money_validation"]


def test_build_report_subject_includes_stealth_summary():
    subject = send_report.build_report_subject(
        "2026-05-31",
        {
            "major_money_available": True,
            "major_money_subject_summary": "MM 10 in/20 out; 4/4 src",
        },
        {
            "stealth_money_available": True,
            "stealth_money_subject_summary": "Stealth 8 buy/8 sell",
        },
    )

    assert subject == "QuantPilot Daily Report - 2026-05-31 | MM 10 in/20 out; 4/4 src | Stealth 8 buy/8 sell"


def test_check_major_money_digest_status_subject_includes_partial_markets(monkeypatch, tmp_path):
    monkeypatch.setenv("HEALTHCHECK_MAJOR_MONEY_MAX_ERROR_RATIO", "0.05")
    digest = {
        "flow_date": "2026-05-29",
        "market_count": 2,
        "available_market_count": 2,
        "entry_count": 2,
        "exit_count": 1,
        "markets": [
            {"market": "A", "currency": "CNY", "available": True, "total_rows": 100, "ok_rows": 99, "non_ok_rows": 1},
            {
                "market": "US",
                "currency": "USD",
                "available": True,
                "total_rows": 100,
                "ok_rows": 90,
                "error_rows": 10,
                "non_ok_rows": 10,
            },
        ],
    }
    path = tmp_path / "major_money_digest.json"
    path.write_text(send_report.json.dumps(digest), encoding="utf-8")

    status = send_report.check_major_money_digest_status(digest_json=path)

    assert status["major_money_subject_summary"] == "MM 2 in/1 out; 2/2 src; partial US"


def test_build_report_subject_degrades_without_major_money_summary():
    assert send_report.build_report_subject("2026-05-31", {"major_money_available": False}) == (
        "QuantPilot Daily Report - 2026-05-31"
    )


def test_send_email_prefers_mailapp_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(send_report, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        send_report,
        "email_config",
        lambda: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "user@example.com",
            "smtp_password": "secret",
            "report_to": "owner@example.com",
            "report_from": "sender@example.com",
            "mail_app_from": "icloud@example.com",
            "report_delivery_method": "mailapp",
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": True,
        },
    )
    called = {"smtp": 0, "sendmail": 0, "mailapp": 0}
    monkeypatch.setattr(send_report, "send_via_smtp", lambda *args, **kwargs: (called.__setitem__("smtp", called["smtp"] + 1) or False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args, **kwargs: (called.__setitem__("sendmail", called["sendmail"] + 1) or False, "sendmail down"))
    monkeypatch.setattr(
        send_report,
        "send_via_mail_app",
        lambda *args, **kwargs: (called.__setitem__("mailapp", called["mailapp"] + 1) or True, ""),
    )

    assert send_report.send_email("<html>hi</html>", "subject") is True
    assert called == {"smtp": 0, "sendmail": 0, "mailapp": 1}


def test_check_trade_status_before_trade_window_reports_not_started(tmp_path):
    trade_log = tmp_path / "trade.log"
    trade_log.write_text("", encoding="utf-8")

    status = send_report.check_trade_status(
        now=datetime(2026, 4, 17, 10, 47, 28),
        trade_log=trade_log,
    )

    assert status == "Today's 14:50 automatic trade has not started yet."


def test_check_trade_status_non_trading_day_reports_no_schedule(tmp_path):
    trade_log = tmp_path / "trade.log"
    trade_log.write_text("", encoding="utf-8")

    status = send_report.check_trade_status(
        now=datetime(2026, 4, 18, 10, 47, 28),
        trade_log=trade_log,
    )

    assert status == "No automatic trade scheduled today."


def test_check_trade_status_after_trade_window_without_runs_warns(tmp_path):
    trade_log = tmp_path / "trade.log"
    trade_log.write_text("", encoding="utf-8")

    status = send_report.check_trade_status(
        now=datetime(2026, 4, 17, 15, 1, 0),
        trade_log=trade_log,
    )

    assert status == "WARNING: No trading execution found today."


def test_check_trade_status_completed_without_fills_reports_completed(tmp_path):
    trade_log = tmp_path / "trade.log"
    trade_log.write_text(
        "[2026-04-18 14:50:00] run_trade: start\n"
        "[2026-04-18 14:50:12] run_trade: done\n",
        encoding="utf-8",
    )

    status = send_report.check_trade_status(
        now=datetime(2026, 4, 18, 15, 1, 0),
        trade_log=trade_log,
    )

    assert status == "Today: automatic trade run completed with no orders filled."


def test_check_trade_status_prefers_fill_summary(tmp_path):
    trade_log = tmp_path / "trade.log"
    trade_log.write_text(
        "[2026-04-18 14:50:00] run_trade: start\n"
        "2026-04-18 14:50:03 [INFO]   OK         code SH.600000\n"
        "[2026-04-18 14:50:12] run_trade: done\n",
        encoding="utf-8",
    )

    status = send_report.check_trade_status(
        now=datetime(2026, 4, 18, 15, 1, 0),
        trade_log=trade_log,
    )

    assert status == "Today: 1 order(s) filled (simulation)."


def test_check_capital_flow_status_summarises_overlay(tmp_path):
    overlay = tmp_path / "futu_capital_flow_signal_overlay_latest.csv"
    overlay.write_text(
        "\n".join(
            [
                "code,signal_date,model_rank,capital_flow_label,capital_flow_latest_date,latest_main_in_flow,main_5d_sum",
                "SH.600000,2026-05-29,2,capital_flow_confirm,2026-05-29,125000000,345000000",
                "SZ.000001,2026-05-29,1,risk_flag_main_outflow,2026-05-29,-50000000,-120000000",
                "SH.600519,2026-05-29,3,fund_flow_watch,2026-05-29,3000000,5000000",
            ]
        ),
        encoding="utf-8",
    )

    status = send_report.check_capital_flow_status(overlay_csv=overlay, top_n=2)

    assert status["capital_flow_available"] is True
    assert status["capital_flow_date"] == "2026-05-29"
    assert status["capital_flow_message"] == (
        "Loaded 3 model candidate(s); capital flow is advisory and not an auto-trade rule."
    )
    assert {row["label"]: row["count"] for row in status["capital_flow_counts"]} == {
        "capital_flow_confirm": 1,
        "risk_flag_main_outflow": 1,
        "fund_flow_watch": 1,
    }
    assert [row["code"] for row in status["capital_flow_top"]] == ["SZ.000001", "SH.600000"]
    assert status["capital_flow_top"][0]["latest_main"] == "-50.0m"
    assert status["capital_flow_top"][0]["row_class"] == "flow-risk"
    assert status["capital_flow_top"][1]["latest_main"] == "125.0m"
    assert status["capital_flow_top"][1]["row_class"] == "flow-confirm"


def test_check_capital_flow_status_missing_file_degrades(tmp_path):
    status = send_report.check_capital_flow_status(overlay_csv=tmp_path / "missing.csv")

    assert status["capital_flow_available"] is False
    assert status["capital_flow_date"] == "N/A"
    assert status["capital_flow_counts"] == []
    assert status["capital_flow_top"] == []
    assert "No Futu capital-flow overlay found" in status["capital_flow_message"]


def test_check_capital_flow_eval_status_summarises_summary(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "\n".join(
            [
                "capital_flow_label,horizon,date_count,avg_return,avg_alpha,avg_hit_rate",
                "capital_flow_confirm,1,3,0.0123,0.0045,0.6667",
                "risk_flag_main_outflow,1,3,-0.021,-0.0301,0.3333",
            ]
        ),
        encoding="utf-8",
    )

    status = send_report.check_capital_flow_eval_status(summary_csv=summary, top_n=2)

    assert status["capital_flow_eval_available"] is True
    assert status["capital_flow_eval_message"] == (
        "Loaded 2 validation row(s); use this before promoting advisory labels to trade rules."
    )
    assert status["capital_flow_eval_rows"][0] == {
        "label": "capital_flow_confirm",
        "horizon": 1,
        "date_count": 3,
        "avg_return": "1.23%",
        "avg_alpha": "0.45%",
        "avg_hit_rate": "66.67%",
        "row_class": "flow-confirm",
    }
    assert status["capital_flow_eval_rows"][1]["row_class"] == "flow-risk"


def test_check_capital_flow_eval_status_empty_file_degrades(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text("", encoding="utf-8")

    status = send_report.check_capital_flow_eval_status(summary_csv=summary)

    assert status["capital_flow_eval_available"] is False
    assert status["capital_flow_eval_rows"] == []
    assert status["capital_flow_eval_message"] == "Capital-flow validation has no forward-return samples yet."


def test_check_capital_flow_gate_status_summarises_gate(tmp_path):
    gate_json = tmp_path / "gate.json"
    gate_json.write_text(
        """
{
  "overall_action": "review_filter",
  "message": "Capital-flow risk labels have enough evidence for manual filter review.",
  "criteria": {"min_date_count": 20},
  "decisions": [
    {"label": "risk_flag_main_outflow", "status": "candidate_filter_review"},
    {"label": "capital_flow_confirm", "status": "keep_advisory"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    status = send_report.check_capital_flow_gate_status(gate_json=gate_json)

    assert status["capital_flow_gate_available"] is True
    assert status["capital_flow_gate_class"] == "gate-review"
    assert status["capital_flow_gate_message"] == (
        "Capital-flow risk labels have enough evidence for manual filter review. "
        "Min dates=20. Candidate label(s): risk_flag_main_outflow."
    )


def test_check_capital_flow_gate_status_missing_file_degrades(tmp_path):
    status = send_report.check_capital_flow_gate_status(gate_json=tmp_path / "missing.json")

    assert status["capital_flow_gate_available"] is False
    assert status["capital_flow_gate_class"] == "gate-advisory"
    assert "keep labels advisory" in status["capital_flow_gate_message"]
