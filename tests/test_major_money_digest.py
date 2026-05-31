import json

import pandas as pd

from strategy.major_money_digest import build_digest, build_market_summary, digest_rows


def test_build_market_summary_classifies_entry_and_exit():
    df = pd.DataFrame(
        [
            {"code": "SH.600000", "name": "Entry", "main_net_inflow": 80_000_000, "update_time": "2026-05-29"},
            {"code": "SZ.000001", "name": "Exit", "main_net_inflow": -70_000_000, "update_time": "2026-05-29"},
            {"code": "SZ.000002", "name": "Neutral", "main_net_inflow": 3_000_000, "update_time": "2026-05-29"},
        ]
    )

    summary = build_market_summary(df, market="A", source="eastmoney", top_n=3)

    assert summary["available"] is True
    assert summary["currency"] == "CNY"
    assert summary["ok_rows"] == 3
    assert summary["entry_count"] == 1
    assert summary["entry_amount"] == 80_000_000
    assert summary["exit_count"] == 1
    assert summary["exit_amount"] == 70_000_000
    assert summary["top_entries"][0]["code"] == "SH.600000"
    assert summary["top_exits"][0]["code"] == "SZ.000001"


def test_build_digest_keeps_missing_expected_markets_visible():
    summary = build_market_summary(
        pd.DataFrame([{"code": "US.AAPL", "latest_main_in_flow": 30_000_000, "capital_flow_latest_date": "2026-05-29"}]),
        market="US",
        source="futu",
    )

    digest = build_digest([summary], expected_markets=["A", "HK", "US"], generated_at="2026-05-31T00:00:00+08:00")

    assert digest["available_market_count"] == 1
    assert digest["entry_count"] == 1
    markets = {item["market"]: item for item in digest["markets"]}
    assert markets["US"]["available"] is True
    assert markets["A"]["available"] is False
    assert markets["HK"]["available"] is False
    assert digest["amount_by_currency"]["USD"]["entry_amount"] == 30_000_000


def test_digest_rows_writes_flat_summary_shape(tmp_path):
    summary = build_market_summary(
        pd.DataFrame([{"code": "HK.00700", "latest_main_in_flow": 60_000_000, "capital_flow_latest_date": "2026-05-29"}]),
        market="HK",
        source="futu",
    )
    digest = build_digest([summary], expected_markets=["HK"])
    path = tmp_path / "digest.json"
    path.write_text(json.dumps(digest), encoding="utf-8")

    rows = digest_rows(digest)

    assert rows.iloc[0]["market"] == "HK"
    assert rows.iloc[0]["entry_count"] == 1
