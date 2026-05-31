import pandas as pd

from collector import eastmoney_fund_flow as emf
from strategy.fund_flow_overlay import build_fund_flow_overlay


def test_fetch_fund_flow_rank_normalizes_eastmoney_fields(monkeypatch):
    payload = {
        "data": {
            "diff": [
                {
                    "f12": "600000",
                    "f14": "浦发银行",
                    "f2": 9.37,
                    "f3": 1.74,
                    "f62": 100.0,
                    "f184": 4.5,
                    "f66": 60.0,
                    "f69": 2.0,
                    "f72": 40.0,
                    "f75": 1.5,
                    "f78": -30.0,
                    "f81": -1.0,
                    "f84": -70.0,
                    "f87": -2.5,
                    "f124": 1780042315,
                }
            ]
        }
    }

    monkeypatch.setattr(emf, "_http_json", lambda *args, **kwargs: payload)

    df = emf.fetch_fund_flow_rank(limit=10)

    row = df.iloc[0]
    assert row["code"] == "SH.600000"
    assert row["fund_flow_rank"] == 1
    assert row["main_net_inflow"] == 100.0
    assert row["main_net_inflow_pct"] == 4.5
    assert isinstance(row["update_time"], str)


def test_fetch_fund_flow_rank_paginates(monkeypatch):
    calls = []

    def fake_http_json(_url, params, **_kwargs):
        calls.append(params)
        page = int(params["pn"])
        if page == 1:
            return {"data": {"total": 3, "diff": [{"f12": "600000", "f14": "浦发银行", "f62": 3}]}}
        return {"data": {"total": 3, "diff": [{"f12": "000001", "f14": "平安银行", "f62": 2}]}}

    monkeypatch.setattr(emf, "_http_json", fake_http_json)

    df = emf.fetch_fund_flow_rank(limit=2, page_size=1)

    assert list(df["code"]) == ["SH.600000", "SZ.000001"]
    assert list(df["fund_flow_rank"]) == [1, 2]
    assert [call["pn"] for call in calls] == ["1", "2"]


def test_fetch_fund_flow_rank_falls_back_to_datacenter(monkeypatch):
    def fake_http_json(url, _params, **_kwargs):
        if url == emf.RANK_ENDPOINT:
            raise ConnectionError("push2 unavailable")
        return {
            "result": {
                "count": 1,
                "data": [
                    {
                        "SECUCODE": "600487.SH",
                        "SECURITY_CODE": "600487",
                        "SECURITY_NAME_ABBR": "亨通光电",
                        "NEW_PRICE": 77.12,
                        "CHANGE_RATE": 9.42,
                        "MAIN_NETINFLOW": 3976260608.0,
                    }
                ],
            }
        }

    monkeypatch.setattr(emf, "_http_json", fake_http_json)

    df = emf.fetch_fund_flow_rank(limit=1)

    row = df.iloc[0]
    assert row["code"] == "SH.600487"
    assert row["fund_flow_source"] == "eastmoney_datacenter"
    assert row["main_net_inflow"] == 3976260608.0
    assert pd.isna(row["main_net_inflow_pct"])


def test_fetch_individual_fund_flow_parses_daily_history(monkeypatch):
    payload = {
        "data": {
            "code": "600000",
            "market": 1,
            "name": "浦发银行",
            "klines": [
                "2026-05-29,-8205550.0,10864248.0,-2658688.0,-15674682.0,7469132.0,-0.94,1.25,-0.31,-1.80,0.86,9.37,1.74"
            ],
        }
    }
    monkeypatch.setattr(emf, "_http_json", lambda *args, **kwargs: payload)

    df = emf.fetch_individual_fund_flow("SH.600000", limit=5)

    row = df.iloc[0]
    assert row["code"] == "SH.600000"
    assert row["date"] == "2026-05-29"
    assert row["main_net_inflow"] == -8205550.0
    assert row["close"] == 9.37


def test_fund_flow_overlay_flags_confirm_and_outflow():
    signals = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 0.9, "rank": 1, "signal_date": "2026-05-29"},
            {"code": "SZ.000001", "score": 0.8, "rank": 2, "signal_date": "2026-05-29"},
            {"code": "SZ.000002", "score": 0.7, "rank": 3, "signal_date": "2026-05-29"},
        ]
    )
    fund = pd.DataFrame(
        [
            {"code": "SH.600000", "fund_flow_rank": 10, "main_net_inflow": 100.0, "main_net_inflow_pct": 5.0},
            {"code": "SZ.000001", "fund_flow_rank": 20, "main_net_inflow": -100.0, "main_net_inflow_pct": -5.0},
        ]
    )

    overlay = build_fund_flow_overlay(signals, fund, signal_top_n=10, confirm_pct=3.0, risk_pct=-3.0)

    labels = dict(zip(overlay["code"], overlay["fund_flow_label"]))
    assert labels["SH.600000"] == "fund_flow_confirm"
    assert labels["SZ.000001"] == "risk_flag_main_outflow"
    assert labels["SZ.000002"] == "model_only_no_fund_flow"


def test_fund_flow_overlay_rank_confirms_when_pct_missing():
    signals = pd.DataFrame(
        [{"code": "SH.600000", "score": 0.9, "rank": 1, "signal_date": "2026-05-29"}]
    )
    fund = pd.DataFrame(
        [{"code": "SH.600000", "fund_flow_rank": 10, "main_net_inflow": 100.0, "main_net_inflow_pct": None}]
    )

    overlay = build_fund_flow_overlay(signals, fund, signal_top_n=10, confirm_rank=500)

    assert overlay.iloc[0]["fund_flow_label"] == "fund_flow_rank_confirm"
