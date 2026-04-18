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

    monkeypatch.setattr(weekly_train, "QLIB_DATA_DIR", tmp_path / "qlib")
    monkeypatch.setattr(weekly_train, "STRATEGY_DIR", tmp_path / "repo")
    monkeypatch.setattr(weekly_train.subprocess, "run", fake_run)
    monkeypatch.setattr(weekly_train, "_resolve_timeout_seconds", lambda _: 123)

    pred_path = tmp_path / "models" / "pred_sh.pkl"
    output_dir = tmp_path / "output"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text("stub", encoding="utf-8")

    weekly_train.run_backtest(pred_path, output_dir)

    assert calls["cmd"] == [
        weekly_train.sys.executable, "-m", "trainer.backtest.run",
        "--pred", str(pred_path),
        "--price-dir", str(weekly_train.QLIB_DATA_DIR),
        "--top-n", "5",
        "--hold-bonus", "0.05",
        "--stop-loss-pct", "-0.08",
        "--position-ratio", "0.95",
        "--allowed-prefix", "SH.",
        "--filter-limit-up",
        "--slippage", "0.001",
        "--output", str(output_dir),
    ]


def test_stage_dirs_use_stable_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly_train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(weekly_train, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setenv("WEEKLY_STAGE_TAG", "20260418_100000")
    monkeypatch.setattr(weekly_train, "_WEEKLY_STAGE_TAG", None)

    models_dir = weekly_train._stage_models_dir()
    output_dir = weekly_train._stage_output_dir()

    assert models_dir == tmp_path / "models" / "weekly_runs" / "20260418_100000"
    assert output_dir == tmp_path / "output" / "weekly_runs" / "20260418_100000"


def test_evaluate_promotion_gate_rejects_weaker_candidate():
    ok, reasons = weekly_train.evaluate_promotion_gate(
        {"ann_return": "61.00%", "sharpe": "2.00", "max_drawdown": "15.50%"},
        {"ann_return": "62.00%", "sharpe": "2.10", "max_drawdown": "14.90%"},
    )

    assert ok is False
    assert any("ann_return below gate" in reason for reason in reasons)
    assert any("sharpe below gate" in reason for reason in reasons)


def test_send_report_email_uses_shared_reporter_delivery(tmp_path, monkeypatch):
    saved = []
    sent = []
    monkeypatch.setattr(weekly_train, "save_report_locally", lambda filename, body: saved.append((filename, body)))
    monkeypatch.setattr(weekly_train, "resolve_email_config", lambda: {
        "smtp_host": "smtp.mail.me.com",
        "smtp_port": "465",
        "smtp_user": "bot@example.com",
        "smtp_password": "secret",
        "report_to": "owner@example.com",
        "report_from": "bot@example.com",
    })
    monkeypatch.setattr(weekly_train, "log_email_config_status", lambda config: [])
    monkeypatch.setattr(weekly_train, "send_email", lambda html, subject, report_filename=None: sent.append((subject, report_filename)) or True)

    report_path = tmp_path / "backtest_report.png"
    metrics_path = tmp_path / "metrics.txt"
    report_path.write_text("png", encoding="utf-8")
    metrics_path.write_text("metrics", encoding="utf-8")

    result = weekly_train.send_report_email(
        {"ic": "0.1", "icir": "0.2", "pred_start": "2026-01-01", "pred_end": "2026-04-01", "n_days": 50, "n_stocks": 100},
        {"ann_return": "10.00%", "sharpe": "1.23", "max_drawdown": "5.00%"},
        report_path,
        metrics_path,
    )

    assert result is True
    assert sent
    assert sent[0][1].startswith("weekly_report_")
    assert saved == []


def test_send_report_email_saves_local_copy_when_shared_delivery_raises(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(weekly_train, "save_report_locally", lambda filename, body: saved.append((filename, body)))
    monkeypatch.setattr(weekly_train, "resolve_email_config", lambda: {
        "smtp_host": "smtp.mail.me.com",
        "smtp_port": "465",
        "smtp_user": "bot@example.com",
        "smtp_password": "secret",
        "report_to": "owner@example.com",
        "report_from": "bot@example.com",
    })
    monkeypatch.setattr(weekly_train, "log_email_config_status", lambda config: [])
    monkeypatch.setattr(weekly_train, "send_email", lambda *args, **kwargs: (_ for _ in ()).throw(NameError("MIMEMultipart")))

    report_path = tmp_path / "backtest_report.png"
    metrics_path = tmp_path / "metrics.txt"
    report_path.write_text("png", encoding="utf-8")
    metrics_path.write_text("metrics", encoding="utf-8")

    result = weekly_train.send_report_email(
        {"ic": "0.1", "icir": "0.2", "pred_start": "2026-01-01", "pred_end": "2026-04-01", "n_days": 50, "n_stocks": 100},
        {"ann_return": "10.00%", "sharpe": "1.23", "max_drawdown": "5.00%"},
        report_path,
        metrics_path,
    )

    assert result is False
    assert saved
    assert saved[0][0].startswith("weekly_report_")


def test_send_failure_email_saves_local_copy_when_shared_delivery_raises(monkeypatch):
    saved = []
    monkeypatch.setattr(weekly_train, "save_report_locally", lambda filename, body: saved.append((filename, body)))
    monkeypatch.setattr(weekly_train, "resolve_email_config", lambda: {
        "smtp_host": "smtp.mail.me.com",
        "smtp_port": "465",
        "smtp_user": "bot@example.com",
        "smtp_password": "secret",
        "report_to": "owner@example.com",
        "report_from": "bot@example.com",
    })
    monkeypatch.setattr(weekly_train, "log_email_config_status", lambda config: [])
    monkeypatch.setattr(weekly_train, "send_email", lambda *args, **kwargs: (_ for _ in ()).throw(NameError("MIMEMultipart")))

    weekly_train.send_failure_email(RuntimeError("boom"))

    assert saved
    assert saved[0][0].startswith("weekly_report_failed_")
    assert "boom" in saved[0][1]


def test_main_passes_stage_models_dir_to_train_model(tmp_path, monkeypatch):
    stage_models_dir = tmp_path / "models" / "weekly_runs" / "20260418_100000"
    stage_output_dir = tmp_path / "output" / "weekly_runs" / "20260418_100000"
    candidate_output_dir = stage_output_dir / "candidate"
    captured = {}

    monkeypatch.setattr(weekly_train, "_stage_models_dir", lambda: stage_models_dir)
    monkeypatch.setattr(weekly_train, "_stage_output_dir", lambda: stage_output_dir)
    monkeypatch.setattr(weekly_train, "sync_qlib_data", lambda: captured.setdefault("synced", True))

    def fake_train_model(models_dir):
        captured["models_dir"] = models_dir
        return {
            "ic": "0.1",
            "icir": "0.2",
            "pred_start": "2026-01-01",
            "pred_end": "2026-04-17",
            "n_days": 50,
            "n_stocks": 100,
        }

    monkeypatch.setattr(weekly_train, "train_model", fake_train_model)
    monkeypatch.setattr(
        weekly_train,
        "run_backtest",
        lambda pred_path, output_dir: (
            {"ann_return": "10.00%", "sharpe": "1.23", "max_drawdown": "5.00%"},
            output_dir / "backtest_report.png",
            output_dir / "metrics.txt",
        ),
    )
    monkeypatch.setattr(weekly_train, "TRADE_PRED_PATH", tmp_path / "missing_pred.pkl")
    monkeypatch.setattr(weekly_train, "deploy_pred", lambda models_dir: captured.setdefault("deployed", models_dir))
    monkeypatch.setattr(weekly_train, "promote_trade_signal", lambda models_dir: captured.setdefault("promoted", models_dir))
    monkeypatch.setattr(
        weekly_train,
        "send_report_email",
        lambda train_info, metrics, report_path, metrics_path: captured.setdefault(
            "reported",
            (train_info, metrics, report_path, metrics_path),
        ),
    )

    weekly_train.main()

    assert captured["synced"] is True
    assert captured["models_dir"] == stage_models_dir
    assert captured["deployed"] == stage_models_dir
    assert captured["promoted"] == stage_models_dir
    assert captured["reported"][2] == candidate_output_dir / "backtest_report.png"
