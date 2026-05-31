import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

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


def test_latest_completed_us_session_date_skips_weekend_before_monday_close():
    now = datetime(2026, 6, 1, 7, 0, tzinfo=ZoneInfo("America/New_York"))

    assert scanner.latest_completed_us_session_date(now) == "2026-05-29"


def test_latest_completed_us_session_date_accepts_china_time_after_us_close():
    now = datetime(2026, 5, 30, 5, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert scanner.latest_completed_us_session_date(now) == "2026-05-29"
