import json

import pandas as pd
import pytest

from scripts.scan_futu_market_capital_flow import _effective_pause_seconds, scan_market


class FakeFutuClient:
    def get_capital_flow(self, code, period_type="DAY", start=None, end=None):
        if code == "US.BAD":
            raise RuntimeError("no permission")
        return [
            {"code": code, "date": "2026-05-29", "main_in_flow": 10.0, "super_in_flow": 6.0, "big_in_flow": 4.0}
        ]

    def get_capital_distribution(self, code):
        return {}


def test_scan_market_writes_latest_status(tmp_path):
    universe = pd.DataFrame(
        [
            {"code": "US.AAPL", "name": "Apple", "exchange_type": "US_NASDAQ"},
            {"code": "US.BAD", "name": "Bad", "exchange_type": "US_NYSE"},
        ]
    )

    stats = scan_market(
        FakeFutuClient(),
        universe,
        market="US",
        output_dir=tmp_path,
        start="2026-05-01",
        end="2026-05-31",
        period="DAY",
        include_distribution=False,
        max_codes=0,
        batch_flush=50,
        overwrite=True,
        pause_seconds=0,
        min_ok_ratio=0.0,
    )

    assert stats["ok_count"] == 1
    assert (tmp_path / "US_latest_flow.csv").exists()
    latest_status = tmp_path / "US_latest_status.json"
    assert latest_status.exists()
    payload = json.loads(latest_status.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["attempted_count"] == 2
    assert payload["ok_count"] == 1
    assert payload["error_count"] == 1
    assert payload["source_exchange_types"] == {"US_NASDAQ": 1, "US_NYSE": 1}
    assert payload["selected_exchange_types"] == {"US_NASDAQ": 1, "US_NYSE": 1}
    assert payload["excluded_exchange_types"] == {}
    assert payload["status_by_exchange_type"] == {"US_NASDAQ": {"ok": 1}, "US_NYSE": {"error": 1}}
    assert payload["unsupported_exchange_types"] == {}


def test_scan_market_writes_failed_status_before_raise(tmp_path):
    universe = pd.DataFrame(
        [
            {"code": "US.AAPL", "name": "Apple", "exchange_type": "US_NASDAQ"},
            {"code": "US.BAD", "name": "Bad", "exchange_type": "US_NYSE"},
        ]
    )

    with pytest.raises(RuntimeError, match="ok_ratio too low"):
        scan_market(
            FakeFutuClient(),
            universe,
            market="US",
            output_dir=tmp_path,
            start="2026-05-01",
            end="2026-05-31",
            period="DAY",
            include_distribution=False,
            max_codes=0,
            batch_flush=50,
            overwrite=True,
            pause_seconds=0,
            min_ok_ratio=0.75,
        )

    payload = json.loads((tmp_path / "US_latest_status.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["attempted_count"] == 2
    assert payload["ok_count"] == 1
    assert payload["error_count"] == 1


def test_effective_pause_seconds_enforces_minimum_interval():
    assert _effective_pause_seconds(
        client_rate_limit_delay=0.35,
        pause_seconds=0.0,
        min_request_interval=1.05,
    ) == pytest.approx(0.70)
    assert _effective_pause_seconds(
        client_rate_limit_delay=0.6,
        pause_seconds=1.1,
        min_request_interval=1.05,
    ) == pytest.approx(1.1)
    assert _effective_pause_seconds(
        client_rate_limit_delay=0.1,
        pause_seconds=0.0,
        min_request_interval=0,
    ) == pytest.approx(0.0)
