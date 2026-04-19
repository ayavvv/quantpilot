import pytest

from polymarket.models import BookLevel, MarketInfo, OrderBook
from polymarket.scanner.full_set import scan_market
from polymarket.config import PolySettings


def _market() -> MarketInfo:
    return MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Will X happen?",
        slug="will-x-happen",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )


def _book(token_id: str, bid: float, ask: float, ts: int = 1000, size: float = 1) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_id="m1",
        timestamp_ms=ts,
        bids=[BookLevel(price=bid, size=size)],
        asks=[BookLevel(price=ask, size=size)],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
        last_trade_price=ask,
    )


def test_scan_market_finds_buy_both_merge_opportunity():
    cfg = PolySettings(
        data_dir="/tmp/poly-test",
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
    )
    opportunities = scan_market(_market(), _book("yes", bid=0.40, ask=0.49), _book("no", bid=0.40, ask=0.49), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].direction == "buy_both_merge"
    assert opportunities[0].mergeable_qty == pytest.approx(1.0)
    assert opportunities[0].net_edge == pytest.approx(0.02)


def test_scan_market_rejects_stale_books():
    cfg = PolySettings(data_dir="/tmp/poly-test", max_book_staleness_ms=50)
    opportunities = scan_market(_market(), _book("yes", bid=0.4, ask=0.49, ts=1000), _book("no", bid=0.4, ask=0.49, ts=1200), cfg)

    assert opportunities == []


def test_scan_market_fee_can_remove_thin_opportunity():
    market = _market()
    market.taker_base_fee_bps = 100
    cfg = PolySettings(data_dir="/tmp/poly-test", min_net_edge=0.0, slippage_buffer=0.0, default_gas_cost=0.0)
    opportunities = scan_market(market, _book("yes", bid=0.4, ask=0.49), _book("no", bid=0.4, ask=0.49), cfg)

    assert opportunities == []


def test_scan_market_applies_max_notional_cap():
    cfg = PolySettings(data_dir="/tmp/poly-test", min_net_edge=-1, slippage_buffer=0.0, default_gas_cost=0.0, max_notional_per_opp=0.98)
    opportunities = scan_market(_market(), _book("yes", bid=0.4, ask=0.49, size=10), _book("no", bid=0.4, ask=0.49, size=10), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].net_cost <= 0.980001
    assert opportunities[0].yes_qty == pytest.approx(opportunities[0].no_qty)


def test_scan_market_equalizes_buy_qty_with_fee():
    market = _market()
    market.taker_base_fee_bps = 100
    cfg = PolySettings(data_dir="/tmp/poly-test", min_net_edge=-1, slippage_buffer=0.0, default_gas_cost=0.0, max_notional_per_opp=10)
    opportunities = scan_market(market, _book("yes", bid=0.4, ask=0.2, size=10), _book("no", bid=0.4, ask=0.7, size=10), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].mergeable_qty <= opportunities[0].yes_qty
    assert opportunities[0].mergeable_qty <= opportunities[0].no_qty
    assert opportunities[0].yes_qty != pytest.approx(opportunities[0].no_qty)
