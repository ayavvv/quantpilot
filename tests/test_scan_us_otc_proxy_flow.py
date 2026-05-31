import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts import scan_us_otc_proxy_flow as scanner


def test_load_otc_universe_filters_pink_codes(tmp_path):
    path = tmp_path / "source_universe.csv"
    pd.DataFrame(
        [
            {"code": "US.AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"},
            {"code": "US.AAPL", "name": "Apple", "exchange_type": "US_NASDAQ"},
        ]
    ).to_csv(path, index=False)

    universe = scanner.load_otc_universe(path, exchange_types={"US_PINK"})

    assert universe["code"].tolist() == ["US.AABB"]
    assert universe["ticker"].tolist() == ["AABB"]


def test_build_proxy_records_uses_directional_dollar_volume():
    universe = pd.DataFrame(
        [
            {"code": "US.AABB", "ticker": "AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"},
            {"code": "US.AACAY", "ticker": "AACAY", "name": "AAC", "exchange_type": "US_PINK"},
        ]
    )
    aggregates = [
        {"T": "AABB", "o": 0.01, "c": 0.02, "h": 0.02, "l": 0.01, "v": 1_000_000},
        {"T": "AACAY", "o": 10.0, "c": 9.0, "h": 10.5, "l": 8.5, "v": 10_000},
    ]

    rows = scanner.build_proxy_records(universe, aggregates, date="2026-05-29", provider="polygon")
    by_code = {row["code"]: row for row in rows.to_dict(orient="records")}

    assert by_code["US.AABB"]["capital_flow_status"] == "ok"
    assert by_code["US.AABB"]["latest_main_in_flow"] == 20_000
    assert by_code["US.AACAY"]["latest_main_in_flow"] == -90_000


def test_build_proxy_records_uses_provider_error_for_missing_aggregate():
    universe = pd.DataFrame(
        [{"code": "US.MISSING", "ticker": "MISSING", "name": "Missing", "exchange_type": "US_PINK"}]
    )

    rows = scanner.build_proxy_records(
        universe,
        [],
        date="2026-05-29",
        provider="yahoo_chart",
        aggregate_errors={"MISSING": "Yahoo chart HTTP 404"},
    )

    row = rows.iloc[0].to_dict()
    assert row["capital_flow_status"] == "empty"
    assert row["capital_flow_error"] == "Yahoo chart HTTP 404"


def test_fetch_yahoo_chart_daily_bar_parses_chart_response(monkeypatch):
    timestamp = int(datetime(2026, 5, 29, 9, 30, tzinfo=ZoneInfo("America/New_York")).timestamp())
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [timestamp],
                    "indicators": {
                        "quote": [
                            {
                                "open": [0.0146],
                                "close": [0.01632],
                                "high": [0.0166],
                                "low": [0.0142],
                                "volume": [12868065],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert "AABB" in request.full_url
        assert timeout == 7
        return FakeResponse()

    monkeypatch.setattr(scanner, "urlopen", fake_urlopen)

    row = scanner.fetch_yahoo_chart_daily_bar("aabb", date="2026-05-29", timeout=7)

    assert row == {"T": "AABB", "o": 0.0146, "c": 0.01632, "h": 0.0166, "l": 0.0142, "v": 12868065}


def test_write_outputs_creates_digest_compatible_artifacts(tmp_path):
    universe = pd.DataFrame(
        [{"code": "US.AABB", "ticker": "AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"}]
    )
    rows = scanner.build_proxy_records(
        universe,
        [{"T": "AABB", "o": 0.01, "c": 0.02, "h": 0.02, "l": 0.01, "v": 1_000_000}],
        date="2026-05-29",
        provider="polygon",
    )

    status = scanner.write_outputs(
        rows,
        output_dir=tmp_path,
        date="2026-05-29",
        provider="polygon",
        universe=universe,
        min_dollar_volume=0.0,
    )

    assert status["ok_count"] == 1
    assert (tmp_path / "US_OTC_latest_flow.csv").exists()
    payload = json.loads((tmp_path / "US_OTC_latest_status.json").read_text(encoding="utf-8"))
    assert payload["market"] == "US_OTC"
    assert payload["source_exchange_types"] == {"US_PINK": 1}


def test_main_supports_yahoo_chart_without_api_key(monkeypatch, tmp_path):
    universe = tmp_path / "source_universe.csv"
    pd.DataFrame([{"code": "US.AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"}]).to_csv(
        universe,
        index=False,
    )

    def fake_fetch(ticker, **kwargs):
        assert ticker == "AABB"
        assert kwargs["date"] == "2026-05-29"
        return {"T": "AABB", "o": 0.01, "c": 0.02, "h": 0.02, "l": 0.01, "v": 1_000_000}, ""

    monkeypatch.setattr(scanner, "_fetch_yahoo_chart_daily_bar_with_retries", fake_fetch)

    scanner.main(
        [
            "--provider",
            "yahoo_chart",
            "--date",
            "2026-05-29",
            "--universe-csv",
            str(universe),
            "--output-dir",
            str(tmp_path),
            "--request-delay",
            "0",
            "--concurrency",
            "2",
        ]
    )

    output = pd.read_csv(tmp_path / "US_OTC_latest_flow.csv")
    payload = json.loads((tmp_path / "US_OTC_latest_status.json").read_text(encoding="utf-8"))
    assert output["source"].tolist() == ["yahoo_chart_otc_proxy"]
    assert output["capital_flow_status"].tolist() == ["ok"]
    assert payload["provider"] == "yahoo_chart"
    assert payload["ok_count"] == 1


def test_yahoo_chart_scan_concurrency_processes_pending_rows(monkeypatch, tmp_path):
    universe = pd.DataFrame(
        [
            {"code": "US.AABB", "ticker": "AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"},
            {"code": "US.AACAY", "ticker": "AACAY", "name": "AAC", "exchange_type": "US_PINK"},
            {"code": "US.AAGC", "ticker": "AAGC", "name": "All American", "exchange_type": "US_PINK"},
        ]
    )
    calls = []

    def fake_fetch(ticker, **kwargs):
        calls.append(ticker)
        return {"T": ticker, "o": 1.0, "c": 2.0, "h": 2.0, "l": 1.0, "v": 1_000}, ""

    monkeypatch.setattr(scanner, "_fetch_yahoo_chart_daily_bar_with_retries", fake_fetch)

    rows = scanner.scan_yahoo_chart_proxy_records(
        universe,
        output_dir=tmp_path,
        date="2026-05-29",
        min_dollar_volume=0.0,
        request_delay=0.0,
        max_retries=0,
        timeout=1.0,
        batch_flush=2,
        overwrite=False,
        concurrency=2,
    )

    assert set(calls) == {"AABB", "AACAY", "AAGC"}
    assert set(rows["code"]) == {"US.AABB", "US.AACAY", "US.AAGC"}
    assert (tmp_path / "US_OTC_latest_flow.csv").exists()


def test_yahoo_chart_scan_resumes_existing_dated_output(monkeypatch, tmp_path):
    universe = pd.DataFrame(
        [
            {"code": "US.AABB", "ticker": "AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"},
            {"code": "US.AACAY", "ticker": "AACAY", "name": "AAC", "exchange_type": "US_PINK"},
        ]
    )
    existing = scanner.build_proxy_records(
        universe.head(1),
        [{"T": "AABB", "o": 0.01, "c": 0.02, "h": 0.02, "l": 0.01, "v": 1_000_000}],
        date="2026-05-29",
        provider="yahoo_chart",
    )
    scanner.write_outputs(
        existing,
        output_dir=tmp_path,
        date="2026-05-29",
        provider="yahoo_chart",
        universe=universe,
        min_dollar_volume=0.0,
    )
    calls = []

    def fake_fetch(ticker, **kwargs):
        calls.append(ticker)
        return {"T": ticker, "o": 10.0, "c": 9.0, "h": 10.5, "l": 8.5, "v": 10_000}, ""

    monkeypatch.setattr(scanner, "_fetch_yahoo_chart_daily_bar_with_retries", fake_fetch)

    rows = scanner.scan_yahoo_chart_proxy_records(
        universe,
        output_dir=tmp_path,
        date="2026-05-29",
        min_dollar_volume=0.0,
        request_delay=0.0,
        max_retries=0,
        timeout=1.0,
        batch_flush=1,
        overwrite=False,
    )

    output = pd.read_csv(tmp_path / "US_OTC_latest_flow.csv")
    assert calls == ["AACAY"]
    assert rows["code"].tolist() == ["US.AABB", "US.AACAY"]
    assert output["code"].tolist() == ["US.AABB", "US.AACAY"]


def test_latest_completed_us_session_date_skips_weekend_before_monday_close():
    now = datetime(2026, 6, 1, 7, 0, tzinfo=ZoneInfo("America/New_York"))

    assert scanner.latest_completed_us_session_date(now) == "2026-05-29"


def test_latest_completed_us_session_date_accepts_china_time_after_us_close():
    now = datetime(2026, 5, 30, 5, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert scanner.latest_completed_us_session_date(now) == "2026-05-29"


def test_resolve_api_key_accepts_secret_file(tmp_path):
    key_file = tmp_path / "polygon.key"
    key_file.write_text(" file-secret \n", encoding="utf-8")

    assert scanner.resolve_api_key("", key_file) == "file-secret"
    assert scanner.resolve_api_key("direct-secret", key_file) == "direct-secret"


def test_main_writes_failed_status_when_provider_fetch_fails(monkeypatch, tmp_path):
    universe = tmp_path / "source_universe.csv"
    pd.DataFrame([{"code": "US.AABB", "name": "Asia Broadband", "exchange_type": "US_PINK"}]).to_csv(
        universe,
        index=False,
    )

    def fail_fetch(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(scanner, "fetch_polygon_grouped_daily", fail_fetch)

    with pytest.raises(RuntimeError, match="provider down"):
        scanner.main(
            [
                "--api-key",
                "secret",
                "--date",
                "2026-05-29",
                "--universe-csv",
                str(universe),
                "--output-dir",
                str(tmp_path),
            ]
        )

    payload = json.loads((tmp_path / "US_OTC_latest_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["message"] == "provider down"
    assert payload["ok_count"] == 0
    assert payload["error_count"] == 1
    assert payload["source_exchange_types"] == {"US_PINK": 1}
    assert not (tmp_path / "US_OTC_latest_flow.csv").exists()
