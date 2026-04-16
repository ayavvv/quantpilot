from pathlib import Path

from reporter import send_report


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
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": True,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args: (False, "sendmail down"))
    fallback_calls = []
    monkeypatch.setattr(
        send_report,
        "send_via_mail_app",
        lambda subject, report_to, report_from, report_path: (
            fallback_calls.append((subject, report_to, report_from, report_path)) or True,
            "",
        ),
    )

    assert send_report.send_email("<html>hi</html>", "subject") is True
    assert len(fallback_calls) == 1
    assert fallback_calls[0][3] == tmp_path / f"report_{send_report.datetime.now().strftime('%Y%m%d')}.html"


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
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": True,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args: (False, "sendmail down"))
    monkeypatch.setattr(send_report, "send_via_mail_app", lambda *args: (False, "mail down"))

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
            "smtp_timeout": 5,
            "smtp_retries": 1,
            "sendmail_fallback": True,
            "mail_app_fallback": False,
        },
    )
    monkeypatch.setattr(send_report, "send_via_smtp", lambda config, msg: (False, "smtp down"))
    monkeypatch.setattr(send_report, "send_via_sendmail", lambda *args: (True, ""))

    assert send_report.send_email("<html>hi</html>", "subject") is True
