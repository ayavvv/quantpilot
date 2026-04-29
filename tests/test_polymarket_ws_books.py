from polymarket.models import BookLevel, MarketInfo, OrderBook
from polymarket.ws_books import PolymarketBookCache


def _market() -> MarketInfo:
    return MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )


def test_ws_book_cache_applies_snapshot_and_price_changes():
    cache = PolymarketBookCache()
    cache.set_connected(True)
    cache.apply_payload(
        [
            {
                "event_type": "book",
                "asset_id": "yes",
                "market": "m1",
                "timestamp": "1777442824176",
                "bids": [{"price": "0.42", "size": "10"}, {"price": "0.43", "size": "12"}],
                "asks": [{"price": "0.45", "size": "7"}, {"price": "0.46", "size": "8"}],
            },
            {
                "event_type": "book",
                "asset_id": "no",
                "market": "m1",
                "timestamp": "1777442824176",
                "bids": [{"price": "0.53", "size": "9"}],
                "asks": [{"price": "0.56", "size": "11"}],
            },
        ]
    )
    cache.apply_payload(
        {
            "market": "m1",
            "timestamp": "1777442870151",
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "yes", "price": "0.44", "size": "20", "side": "BUY"},
                {"asset_id": "yes", "price": "0.45", "size": "0", "side": "SELL"},
                {"asset_id": "no", "price": "0.55", "size": "15", "side": "SELL"},
            ],
        }
    )

    books, errors = cache.get_market_books([_market()], connection_stale_seconds=30)

    assert errors == {}
    yes_book = books["m1"]["yes"]
    no_book = books["m1"]["no"]
    assert yes_book.best_bid.price == 0.44
    assert yes_book.best_bid.size == 20
    assert yes_book.best_ask.price == 0.46
    assert no_book.best_ask.price == 0.55
    assert no_book.best_ask.size == 15
    assert cache.pop_dirty_market_ids() == {"m1"}


def test_ws_book_cache_requires_initial_snapshot():
    cache = PolymarketBookCache()
    cache.set_connected(True)
    cache.apply_payload(
        {
            "market": "m1",
            "timestamp": "1777442870151",
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "yes", "price": "0.44", "size": "20", "side": "BUY"},
                {"asset_id": "no", "price": "0.55", "size": "15", "side": "SELL"},
            ],
        }
    )

    books, errors = cache.get_market_books([_market()], connection_stale_seconds=30)

    assert books == {}
    assert "m1" in errors


def test_ws_book_cache_reconcile_replaces_snapshot_and_marks_dirty_on_top_drift():
    cache = PolymarketBookCache()
    cache.set_connected(True)
    cache.apply_book(
        {
            "asset_id": "yes",
            "market": "m1",
            "timestamp": "1777442824176",
            "bids": [{"price": "0.42", "size": "10"}],
            "asks": [{"price": "0.45", "size": "7"}],
        }
    )
    cache.pop_dirty_market_ids()

    changed = cache.reconcile_order_book(
        OrderBook(
            token_id="yes",
            market_id="m1",
            timestamp_ms=1777442825000,
            bids=[BookLevel(price=0.43, size=11)],
            asks=[BookLevel(price=0.46, size=8)],
            tick_size=0.01,
            min_order_size=1,
            neg_risk=False,
        )
    )

    assert changed is True
    assert cache.pop_dirty_market_ids() == {"m1"}
