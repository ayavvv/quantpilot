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
