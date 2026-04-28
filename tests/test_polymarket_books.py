from polymarket.books import ClobClient, _parse_levels, _parse_optional_float
from polymarket.config import PolySettings
from polymarket.models import BookLevel, OrderBook


def test_parse_levels_skips_invalid_rows():
    levels = _parse_levels([
        {"price": "0.5", "size": "10"},
        {"price": "", "size": "3"},
        {"size": "2"},
        {"price": "0.4", "size": "bad"},
    ])

    assert len(levels) == 1
    assert levels[0].price == 0.5
    assert levels[0].size == 10.0


def test_parse_optional_float_handles_blank_values():
    assert _parse_optional_float("") is None
    assert _parse_optional_float(None) is None
    assert _parse_optional_float("0.42") == 0.42


def test_order_book_best_prices_do_not_depend_on_input_order():
    book = OrderBook(
        token_id="t1",
        market_id="m1",
        timestamp_ms=1,
        bids=[BookLevel(price=0.10, size=1), BookLevel(price=0.40, size=1), BookLevel(price=0.20, size=1)],
        asks=[BookLevel(price=0.90, size=1), BookLevel(price=0.60, size=1), BookLevel(price=0.80, size=1)],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
    )

    assert book.best_bid.price == 0.40
    assert book.best_ask.price == 0.60


def test_fetch_books_posts_batch_payload(monkeypatch):
    cfg = PolySettings(clob_base_url="https://example.test")
    client = ClobClient(cfg)
    calls = {}

    def fake_post_json(path, payload):
        calls["path"] = path
        calls["payload"] = payload
        return [
            {
                "asset_id": "yes",
                "market": "m1",
                "timestamp": "123",
                "bids": [{"price": "0.4", "size": "10"}],
                "asks": [{"price": "0.5", "size": "11"}],
                "min_order_size": "5",
                "tick_size": "0.01",
                "neg_risk": False,
                "last_trade_price": "",
            },
            {
                "asset_id": "no",
                "market": "m1",
                "timestamp": "124",
                "bids": [{"price": "0.45", "size": "12"}],
                "asks": [{"price": "0.55", "size": "13"}],
                "min_order_size": "5",
                "tick_size": "0.01",
                "neg_risk": False,
                "last_trade_price": "0.49",
            },
        ]

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    books = client.fetch_books(["yes", "no"])

    assert calls["path"] == "books"
    assert calls["payload"] == [{"token_id": "yes"}, {"token_id": "no"}]
    assert set(books) == {"yes", "no"}
    assert books["yes"].best_ask.price == 0.5
    assert books["yes"].last_trade_price is None
    assert books["no"].last_trade_price == 0.49
