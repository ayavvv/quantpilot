import json

from polymarket.catalog import load_binary_markets, normalize_market
from polymarket.config import PolySettings


def test_normalize_market_extracts_yes_no_pair():
    market = normalize_market(
        {
            "id": "m1",
            "conditionId": "c1",
            "question": "Will it rain?",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["yes-token", "no-token"]),
            "outcomes": json.dumps(["Yes", "No"]),
            "minimumOrderSize": 5,
            "minimumTickSize": 0.01,
            "takerBaseFee": 35,
        }
    )

    assert market is not None
    assert market.yes_token_id == "yes-token"
    assert market.no_token_id == "no-token"
    assert market.taker_base_fee_bps == 35.0
    assert market.tick_size == 0.01


def test_normalize_market_rejects_neg_risk_and_non_binary():
    neg_risk = normalize_market(
        {
            "id": "m1",
            "question": "X",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": True,
            "clobTokenIds": json.dumps(["a", "b"]),
            "outcomes": json.dumps(["Yes", "No"]),
        }
    )
    not_binary = normalize_market(
        {
            "id": "m2",
            "question": "Y",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["a", "b", "c"]),
            "outcomes": json.dumps(["Yes", "No", "Maybe"]),
        }
    )

    assert neg_risk is None
    assert not_binary is None


def test_load_binary_markets_skips_fee_fetch_when_market_fee_present(monkeypatch):
    cfg = PolySettings()
    rows = [
        {
            "id": "m1",
            "conditionId": "c1",
            "question": "Will it rain?",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["yes-token", "no-token"]),
            "outcomes": json.dumps(["Yes", "No"]),
            "minimumOrderSize": 5,
            "minimumTickSize": 0.01,
            "takerBaseFee": 12,
        }
    ]

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_markets", lambda self: rows)

    def fail_fee_fetch(self, token_id):
        raise AssertionError("fee fetch should not be called")

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_fee_rate_bps", fail_fee_fetch)

    markets = load_binary_markets(cfg)

    assert len(markets) == 1
    assert markets[0].taker_base_fee_bps == 12.0


def test_load_binary_markets_skips_fee_fetch_for_rejected_markets(monkeypatch):
    cfg = PolySettings()
    rows = [
        {
            "id": "neg",
            "conditionId": "neg",
            "question": "Neg risk",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": True,
            "clobTokenIds": json.dumps(["yes-neg", "no-neg"]),
            "outcomes": json.dumps(["Yes", "No"]),
        },
        {
            "id": "multi",
            "conditionId": "multi",
            "question": "Multi",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["a", "b", "c"]),
            "outcomes": json.dumps(["Yes", "No", "Maybe"]),
        },
        {
            "id": "ok",
            "conditionId": "ok",
            "question": "OK",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["yes-ok", "no-ok"]),
            "outcomes": json.dumps(["Yes", "No"]),
        },
    ]
    calls = []

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_markets", lambda self: rows)

    def fake_fee_fetch(self, token_id):
        calls.append(token_id)
        return 9

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_fee_rate_bps", fake_fee_fetch)

    markets = load_binary_markets(cfg)

    assert len(markets) == 1
    assert markets[0].market_id == "ok"
    assert markets[0].taker_base_fee_bps == 9.0
    assert calls == ["yes-ok"]


def test_load_binary_markets_keeps_trying_fee_lookup_per_token(monkeypatch):
    cfg = PolySettings()
    rows = [
        {
            "id": "m1",
            "conditionId": "c1",
            "question": "One",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["yes-token-1", "no-token-1"]),
            "outcomes": json.dumps(["Yes", "No"]),
        },
        {
            "id": "m2",
            "conditionId": "c2",
            "question": "Two",
            "active": True,
            "closed": False,
            "enableOrderBook": True,
            "negRisk": False,
            "clobTokenIds": json.dumps(["yes-token-2", "no-token-2"]),
            "outcomes": json.dumps(["Yes", "No"]),
        },
    ]
    calls = {"count": 0}

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_markets", lambda self: rows)

    def fake_fee_fetch(self, token_id):
        calls["count"] += 1
        return None

    monkeypatch.setattr("polymarket.catalog.GammaClient.fetch_fee_rate_bps", fake_fee_fetch)

    markets = load_binary_markets(cfg)

    assert len(markets) == 2
    assert calls["count"] == 2
