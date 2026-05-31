import json

import pandas as pd
import pytest

from scripts.scan_futu_market_capital_flow import (
    _effective_pause_seconds,
    classify_security_class,
    fetch_futu_universe,
    scan_market,
)


class FakeFutuClient:
    def get_capital_flow(self, code, period_type="DAY", start=None, end=None):
        if code == "US.BAD":
            raise RuntimeError("no permission")
        return [
            {"code": code, "date": "2026-05-29", "main_in_flow": 10.0, "super_in_flow": 6.0, "big_in_flow": 4.0}
        ]

    def get_capital_distribution(self, code):
        return {}


class FakeUniverseClient:
    def __init__(self, rows):
        self.ctx = self
        self.rows = rows

    def get_stock_basicinfo(self, market, security_type):
        return 0, pd.DataFrame(self.rows)


def test_classify_security_class_labels_non_common_instruments():
    assert (
        classify_security_class(
            {"code": "US.AAIC.PRB", "name": "ARLINGTON ASSET 7% CUM PRF", "listing_date": "1970-01-01"},
            reference_date="2026-05-31",
        )
        == "preferred"
    )
    assert (
        classify_security_class(
            {"code": "US.AAIN", "name": "SENIOR NOTES DUE 01/08/2026", "listing_date": "1970-01-01"},
            reference_date="2026-05-31",
        )
        == "note_debt"
    )
    assert classify_security_class({"code": "US.AAPL", "name": "Apple", "listing_date": "1980-12-12"}) == "common_or_unknown"
    assert (
        classify_security_class({"code": "US.NEW", "name": "New Co", "listing_date": "2026-06-03"}, reference_date="2026-05-31")
        == "future_listing"
    )


def test_fetch_futu_universe_filters_excluded_security_classes():
    rows = [
        {"code": "US.AAPL", "name": "Apple", "exchange_type": "US_NASDAQ", "delisting": False},
        {"code": "US.PREF.PRA", "name": "Issuer Preferred Series A", "exchange_type": "US_NYSE", "delisting": False},
        {"code": "US.NOTE", "name": "Issuer Notes Due 2029", "exchange_type": "US_NYSE", "delisting": False},
        {"code": "US.PINKY", "name": "Pink Common", "exchange_type": "US_PINK", "delisting": False},
    ]

    universe = fetch_futu_universe(
        FakeUniverseClient(rows),
        "US",
        exclude_exchange_types={"US_PINK"},
        exclude_security_classes={"preferred", "note_debt"},
        reference_date="2026-05-31",
    )

    assert universe["code"].tolist() == ["US.AAPL"]
    assert universe["security_class"].tolist() == ["common_or_unknown"]
    assert universe.attrs["source_exchange_types"] == {"US_NASDAQ": 1, "US_NYSE": 2, "US_PINK": 1}
    assert universe.attrs["selected_exchange_types"] == {"US_NASDAQ": 1}
    assert universe.attrs["excluded_exchange_types"] == {"US_PINK": 1}
    assert universe.attrs["source_security_classes"] == {"common_or_unknown": 1, "note_debt": 1, "preferred": 1}
    assert universe.attrs["selected_security_classes"] == {"common_or_unknown": 1}
    assert universe.attrs["excluded_security_classes"] == {"note_debt": 1, "preferred": 1}


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
    assert payload["selected_security_classes"] == {"common_or_unknown": 2}
    assert payload["status_by_security_class"] == {"common_or_unknown": {"error": 1, "ok": 1}}
    assert payload["status_by_exchange_type"] == {"US_NASDAQ": {"ok": 1}, "US_NYSE": {"error": 1}}
    assert payload["unsupported_exchange_types"] == {}
    assert (tmp_path / "US_latest_universe.csv").exists()


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
