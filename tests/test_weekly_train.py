from pathlib import Path

from trainer import weekly_train


def test_resolve_email_config_falls_back_to_reporter_env(tmp_path, monkeypatch):
    reporter_env = tmp_path / "reporter" / ".env"
    reporter_env.parent.mkdir(parents=True, exist_ok=True)
    reporter_env.write_text(
        "\n".join(
            [
                "SMTP_HOST=smtp.gmail.com",
                "SMTP_PORT=587",
                "SMTP_USER=bot@example.com",
                "SMTP_PASSWORD=secret",
                "REPORT_TO=owner@example.com",
                "REPORT_FROM=QuantPilot <bot@example.com>",
            ]
        ),
        encoding="utf-8",
    )

    for key in [
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
        "EMAIL_TO",
        "EMAIL_FROM",
        "REPORT_TO",
        "REPORT_FROM",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(weekly_train, "STRATEGY_DIR", tmp_path)

    config = weekly_train.resolve_email_config()

    assert config["smtp_host"] == "smtp.gmail.com"
    assert config["smtp_port"] == "587"
    assert config["smtp_user"] == "bot@example.com"
    assert config["smtp_password"] == "secret"
    assert config["report_to"] == "owner@example.com"
    assert config["report_from"] == "QuantPilot <bot@example.com>"


def test_resolve_email_config_prefers_process_env(tmp_path, monkeypatch):
    reporter_env = tmp_path / "reporter" / ".env"
    reporter_env.parent.mkdir(parents=True, exist_ok=True)
    reporter_env.write_text(
        "\n".join(
            [
                "SMTP_USER=fallback@example.com",
                "SMTP_PASSWORD=fallback-secret",
                "REPORT_TO=fallback-owner@example.com",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(weekly_train, "STRATEGY_DIR", tmp_path)
    monkeypatch.setenv("SMTP_USER", "env@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "env-secret")
    monkeypatch.setenv("EMAIL_TO", "env-owner@example.com")

    config = weekly_train.resolve_email_config()

    assert config["smtp_user"] == "env@example.com"
    assert config["smtp_password"] == "env-secret"
    assert config["report_to"] == "env-owner@example.com"


def test_resolve_timeout_seconds_default_and_validation(monkeypatch):
    monkeypatch.delenv("WEEKLY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("WEEKLY_TRAIN_TIMEOUT_SECONDS", raising=False)
    assert weekly_train._resolve_timeout_seconds("WEEKLY_TRAIN_TIMEOUT_SECONDS") == 43200

    monkeypatch.setenv("WEEKLY_TRAIN_TIMEOUT_SECONDS", "7200")
    assert weekly_train._resolve_timeout_seconds("WEEKLY_TRAIN_TIMEOUT_SECONDS") == 7200

    monkeypatch.setenv("WEEKLY_TRAIN_TIMEOUT_SECONDS", "bad-value")
    assert weekly_train._resolve_timeout_seconds("WEEKLY_TRAIN_TIMEOUT_SECONDS") == 43200

    monkeypatch.delenv("WEEKLY_TRAIN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("WEEKLY_TIMEOUT_SECONDS", "14400")
    assert weekly_train._resolve_timeout_seconds("WEEKLY_TRAIN_TIMEOUT_SECONDS") == 14400


def test_weekly_run_backtest_uses_live_trade_params(tmp_path, monkeypatch):
    calls = {}

    class Result:
        returncode = 0
        stdout = "ann_return: 10.00%\nsharpe: 1.23\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(weekly_train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(weekly_train, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(weekly_train, "QLIB_DATA_DIR", tmp_path / "qlib")
    monkeypatch.setattr(weekly_train, "STRATEGY_DIR", tmp_path / "repo")
    monkeypatch.setattr(weekly_train.subprocess, "run", fake_run)
    monkeypatch.setattr(weekly_train, "_resolve_timeout_seconds", lambda _: 123)

    weekly_train.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (weekly_train.MODELS_DIR / "pred_sh.pkl").write_text("stub", encoding="utf-8")

    weekly_train.run_backtest()

    assert calls["cmd"] == [
        weekly_train.sys.executable, "-m", "trainer.backtest.run",
        "--pred", str(weekly_train.MODELS_DIR / "pred_sh.pkl"),
        "--price-dir", str(weekly_train.QLIB_DATA_DIR),
        "--top-n", "5",
        "--hold-bonus", "0.05",
        "--stop-loss-pct", "-0.08",
        "--position-ratio", "0.95",
        "--allowed-prefix", "SH.",
        "--filter-limit-up",
        "--slippage", "0.001",
        "--output", str(weekly_train.OUTPUT_DIR),
    ]
