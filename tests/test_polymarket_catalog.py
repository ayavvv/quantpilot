import json

from polymarket.catalog import normalize_market


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
