import json
from pathlib import Path

from scripts import major_money_readiness as readiness


def _write_reporter_env(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "REPORT_DELIVERY_METHOD=smtp",
                "SMTP_HOST=smtp.example.com",
                "SMTP_PORT=465",
                "SMTP_USER=sender@example.com",
                "SMTP_PASSWORD=secret",
                "REPORT_TO=to@example.com",
                "REPORT_FROM=sender@example.com",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_digest(path: Path, *, us_otc_available: bool, archive_dir: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    markets = [
        {"market": "A", "source": "eastmoney", "available": True, "ok_rows": 2, "total_rows": 2},
        {"market": "HK", "source": "futu", "available": True, "ok_rows": 1, "total_rows": 1},
        {"market": "US", "source": "futu", "available": True, "ok_rows": 1, "total_rows": 1},
        {
            "market": "US_OTC",
            "source": "polygon_otc_proxy" if us_otc_available else "",
            "available": us_otc_available,
            "ok_rows": 1 if us_otc_available else 0,
            "total_rows": 1 if us_otc_available else 0,
        },
    ]
    payload = {
        "flow_date": "2026-05-29",
        "market_count": 4,
        "available_market_count": 4 if us_otc_available else 3,
        "markets": markets,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "20260529_major_money_digest.json").write_text(json.dumps(payload), encoding="utf-8")
        (archive_dir / "20260529_major_money_digest.csv").write_text(
            "market,available,total_rows,ok_rows\nA,true,2,2\n",
            encoding="utf-8",
        )


def _cron(project_dir: Path) -> str:
    return "\n".join(
        [
            f"0 19 * * 1-5 {project_dir}/scripts/run_daily.sh >> {project_dir}/logs/daily.log 2>&1",
            f"40 16 * * 1-5 FUTU_MARKET_FLOW_MARKETS=HK {project_dir}/scripts/run_market_capital_flow.sh",
            f"10 5 * * 2-6 FUTU_MARKET_FLOW_MARKETS=US {project_dir}/scripts/run_market_capital_flow.sh",
        ]
    )


def test_build_readiness_snapshot_accepts_ready_system(tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    reporter_env = project_dir / "reporter" / ".env"
    digest = data_dir / "output" / "major_money_digest_latest.json"
    archive_dir = data_dir / "output" / "major_money_digest"
    otc_dir = data_dir / "capital_flow" / "us_otc_proxy"
    universe = data_dir / "capital_flow" / "futu_market" / "US_latest_source_universe.csv"
    universe.parent.mkdir(parents=True)
    universe.write_text("code,exchange_type\nUS.AABB,US_PINK\n", encoding="utf-8")
    otc_dir.mkdir(parents=True)
    (otc_dir / "US_OTC_latest_flow.csv").write_text("code,capital_flow_status\nUS.AABB,ok\n", encoding="utf-8")
    (otc_dir / "US_OTC_latest_status.json").write_text(
        '{"status":"ok","attempted_count":1,"ok_count":1}',
        encoding="utf-8",
    )
    _write_reporter_env(reporter_env)
    _write_digest(digest, us_otc_available=True, archive_dir=archive_dir)

    snapshot = readiness.build_readiness_snapshot(
        project_dir=project_dir,
        crontab_text=_cron(project_dir),
        env={
            "DATA_DIR": str(data_dir),
            "REPORTER_ENV_FILE": str(reporter_env),
            "MAJOR_MONEY_DIGEST_JSON": str(digest),
            "MAJOR_MONEY_DIGEST_ARCHIVE_DIR": str(archive_dir),
            "MAJOR_MONEY_EXPECTED_MARKETS": "A,HK,US,US_OTC",
            "ENABLE_US_OTC_PROXY_FLOW": "true",
            "POLYGON_API_KEY": "secret",
        },
    )

    assert snapshot["ok"] is True
    assert snapshot["issues"] == []
    assert snapshot["checks"]["digest"]["markets"]["US_OTC"]["available"] is True
    assert snapshot["checks"]["digest_archive"]["ok"] is True


def test_build_readiness_snapshot_accepts_polygon_key_file(tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    reporter_env = project_dir / "reporter" / ".env"
    digest = data_dir / "output" / "major_money_digest_latest.json"
    archive_dir = data_dir / "output" / "major_money_digest"
    otc_dir = data_dir / "capital_flow" / "us_otc_proxy"
    universe = data_dir / "capital_flow" / "futu_market" / "US_latest_source_universe.csv"
    key_file = tmp_path / "polygon.key"
    key_file.write_text("secret\n", encoding="utf-8")
    universe.parent.mkdir(parents=True)
    universe.write_text("code,exchange_type\nUS.AABB,US_PINK\n", encoding="utf-8")
    otc_dir.mkdir(parents=True)
    (otc_dir / "US_OTC_latest_flow.csv").write_text("code,capital_flow_status\nUS.AABB,ok\n", encoding="utf-8")
    (otc_dir / "US_OTC_latest_status.json").write_text(
        '{"status":"ok","attempted_count":1,"ok_count":1}',
        encoding="utf-8",
    )
    _write_reporter_env(reporter_env)
    _write_digest(digest, us_otc_available=True, archive_dir=archive_dir)

    snapshot = readiness.build_readiness_snapshot(
        project_dir=project_dir,
        crontab_text=_cron(project_dir),
        env={
            "DATA_DIR": str(data_dir),
            "REPORTER_ENV_FILE": str(reporter_env),
            "MAJOR_MONEY_DIGEST_JSON": str(digest),
            "MAJOR_MONEY_DIGEST_ARCHIVE_DIR": str(archive_dir),
            "MAJOR_MONEY_EXPECTED_MARKETS": "A,HK,US,US_OTC",
            "ENABLE_US_OTC_PROXY_FLOW": "true",
            "POLYGON_API_KEY_FILE": str(key_file),
        },
    )

    assert snapshot["ok"] is True
    assert snapshot["checks"]["us_otc_proxy"]["api_key_present"] is True
    assert snapshot["checks"]["us_otc_proxy"]["api_key_file"] == str(key_file)


def test_build_readiness_snapshot_flags_missing_digest_archive(tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    reporter_env = project_dir / "reporter" / ".env"
    digest = data_dir / "output" / "major_money_digest_latest.json"
    archive_dir = data_dir / "output" / "major_money_digest"
    _write_reporter_env(reporter_env)
    _write_digest(digest, us_otc_available=True)

    snapshot = readiness.build_readiness_snapshot(
        project_dir=project_dir,
        crontab_text=_cron(project_dir),
        env={
            "DATA_DIR": str(data_dir),
            "REPORTER_ENV_FILE": str(reporter_env),
            "MAJOR_MONEY_DIGEST_JSON": str(digest),
            "MAJOR_MONEY_DIGEST_ARCHIVE_DIR": str(archive_dir),
            "MAJOR_MONEY_EXPECTED_MARKETS": "A",
            "ENABLE_US_OTC_PROXY_FLOW": "false",
        },
    )

    assert snapshot["ok"] is False
    assert any("Major-money digest archive directory missing" in issue for issue in snapshot["issues"])
    assert snapshot["checks"]["digest_archive"]["date_tag"] == "20260529"


def test_build_readiness_snapshot_flags_missing_us_otc_proxy(tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    reporter_env = project_dir / "reporter" / ".env"
    digest = data_dir / "output" / "major_money_digest_latest.json"
    _write_reporter_env(reporter_env)
    _write_digest(digest, us_otc_available=False)

    snapshot = readiness.build_readiness_snapshot(
        project_dir=project_dir,
        crontab_text=_cron(project_dir),
        env={
            "DATA_DIR": str(data_dir),
            "REPORTER_ENV_FILE": str(reporter_env),
            "MAJOR_MONEY_DIGEST_JSON": str(digest),
            "MAJOR_MONEY_EXPECTED_MARKETS": "A,HK,US,US_OTC",
            "ENABLE_US_OTC_PROXY_FLOW": "false",
        },
    )

    assert snapshot["ok"] is False
    assert any("expected market unavailable: US_OTC" in issue for issue in snapshot["issues"])
    assert any("US OTC/Pink proxy disabled" in issue for issue in snapshot["issues"])


def test_build_readiness_snapshot_flags_partial_digest_coverage(tmp_path):
    project_dir = tmp_path / "project"
    data_dir = tmp_path / "data"
    reporter_env = project_dir / "reporter" / ".env"
    digest = data_dir / "output" / "major_money_digest_latest.json"
    _write_reporter_env(reporter_env)
    digest.parent.mkdir(parents=True)
    digest.write_text(
        json.dumps(
            {
                "flow_date": "2026-05-29",
                "market_count": 2,
                "available_market_count": 2,
                "markets": [
                    {"market": "A", "available": True, "ok_rows": 1, "total_rows": 1},
                    {
                        "market": "HK",
                        "available": True,
                        "ok_rows": 8,
                        "total_rows": 10,
                        "empty_rows": 2,
                        "error_rows": 0,
                        "non_ok_rows": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    snapshot = readiness.build_readiness_snapshot(
        project_dir=project_dir,
        crontab_text=_cron(project_dir),
        env={
            "DATA_DIR": str(data_dir),
            "REPORTER_ENV_FILE": str(reporter_env),
            "MAJOR_MONEY_DIGEST_JSON": str(digest),
            "MAJOR_MONEY_EXPECTED_MARKETS": "A,HK",
            "HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO": "0.05",
        },
    )

    assert snapshot["ok"] is False
    assert any(
        "Major-money digest partial source coverage: market=HK non_ok=2/10 (20.0%) empty=2 error=0 max=5.0%"
        in issue
        for issue in snapshot["issues"]
    )
    assert snapshot["checks"]["digest"]["markets"]["HK"]["non_ok_rows"] == 2


def test_check_cron_ignores_disabled_lines(tmp_path):
    project_dir = tmp_path / "project"
    status = readiness.check_cron(
        "\n".join(
            [
                f"# 0 19 * * 1-5 {project_dir}/scripts/run_daily.sh",
                f"10 5 * * 2-6 FUTU_MARKET_FLOW_MARKETS=US {project_dir}/scripts/run_market_capital_flow.sh",
            ]
        ),
        project_dir=project_dir,
    )

    assert status["ok"] is False
    assert status["checks"]["daily_report"] is False
    assert status["checks"]["us_market_scan"] is True
