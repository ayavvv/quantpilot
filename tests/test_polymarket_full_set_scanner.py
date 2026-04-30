import pytest

from polymarket.models import BookLevel, MarketInfo, OrderBook
from polymarket.scanner.full_set import rejection_reason, scan_market
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


def _depth_book(token_id: str, asks: list[tuple[float, float]], ts: int = 1000) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_id="m1",
        timestamp_ms=ts,
        bids=[BookLevel(price=0.1, size=100)],
        asks=[BookLevel(price=price, size=size) for price, size in asks],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
        last_trade_price=asks[0][0],
    )


def _bid_depth_book(token_id: str, bids: list[tuple[float, float]], ts: int = 1000) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_id="m1",
        timestamp_ms=ts,
        bids=[BookLevel(price=price, size=size) for price, size in bids],
        asks=[],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
        last_trade_price=bids[0][0],
    )


def test_scan_market_finds_buy_both_merge_opportunity():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test", enable_split_sell=False,
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
    cfg = PolySettings(_env_file=None, data_dir="/tmp/poly-test", enable_split_sell=False, max_book_staleness_ms=50)
    opportunities = scan_market(_market(), _book("yes", bid=0.4, ask=0.49, ts=1000), _book("no", bid=0.4, ask=0.49, ts=1200), cfg)

    assert opportunities == []
    assert rejection_reason(_market(), _book("yes", bid=0.4, ask=0.49, ts=1000), _book("no", bid=0.4, ask=0.49, ts=1200), cfg) == "stale_books"


def test_scan_market_fee_can_remove_thin_opportunity():
    market = _market()
    market.taker_base_fee_bps = 100
    cfg = PolySettings(_env_file=None, data_dir="/tmp/poly-test", enable_split_sell=False, min_net_edge=0.0, slippage_buffer=0.0, default_gas_cost=0.0)
    opportunities = scan_market(market, _book("yes", bid=0.4, ask=0.49), _book("no", bid=0.4, ask=0.49), cfg)

    assert opportunities == []
    assert rejection_reason(market, _book("yes", bid=0.4, ask=0.49), _book("no", bid=0.4, ask=0.49), cfg) in {"edge_below_threshold", "capacity_below_min_order"}


def test_scan_market_applies_max_notional_cap():
    cfg = PolySettings(_env_file=None, data_dir="/tmp/poly-test", enable_split_sell=False, min_net_edge=-1, slippage_buffer=0.0, default_gas_cost=0.0, max_notional_per_opp=0.98)
    opportunities = scan_market(_market(), _book("yes", bid=0.4, ask=0.49, size=10), _book("no", bid=0.4, ask=0.49, size=10), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].net_cost <= 0.980001
    assert opportunities[0].yes_qty == pytest.approx(opportunities[0].no_qty)


def test_scan_market_uses_order_book_depth_to_reach_target_notional():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test", enable_split_sell=False,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
        target_notional_per_opp=25,
        max_notional_per_opp=250,
    )
    opportunities = scan_market(
        _market(),
        _depth_book("yes", [(0.45, 5), (0.48, 20)]),
        _depth_book("no", [(0.45, 5), (0.48, 20)]),
        cfg,
    )

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.mergeable_qty == pytest.approx(25.0)
    assert opportunity.net_cost == pytest.approx(23.7)
    assert opportunity.net_edge == pytest.approx(1.3)
    assert opportunity.yes_price == pytest.approx(0.474)


def test_scan_market_stops_before_negative_marginal_depth():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test", enable_split_sell=False,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
        target_notional_per_opp=25,
        min_depth_edge_per_share=0.0,
    )
    opportunities = scan_market(
        _market(),
        _depth_book("yes", [(0.45, 5), (0.51, 20)]),
        _depth_book("no", [(0.45, 5), (0.51, 20)]),
        cfg,
    )

    assert len(opportunities) == 1
    assert opportunities[0].mergeable_qty == pytest.approx(5.0)
    assert opportunities[0].net_cost == pytest.approx(4.5)


def test_scan_market_equalizes_buy_qty_with_fee():
    market = _market()
    market.taker_base_fee_bps = 100
    cfg = PolySettings(_env_file=None, data_dir="/tmp/poly-test", enable_split_sell=False, min_net_edge=-1, slippage_buffer=0.0, default_gas_cost=0.0, max_notional_per_opp=10)
    opportunities = scan_market(market, _book("yes", bid=0.4, ask=0.2, size=10), _book("no", bid=0.4, ask=0.7, size=10), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].mergeable_qty <= opportunities[0].yes_qty
    assert opportunities[0].mergeable_qty <= opportunities[0].no_qty
    assert opportunities[0].yes_qty != pytest.approx(opportunities[0].no_qty)


def test_scan_market_finds_split_sell_when_buy_side_is_not_profitable():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test",
        enable_split_sell=True,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
        target_notional_per_opp=25,
    )
    opportunities = scan_market(_market(), _book("yes", bid=0.53, ask=0.60, size=10), _book("no", bid=0.53, ask=0.60, size=10), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].direction == "split_sell_both"
    assert opportunities[0].mergeable_qty == pytest.approx(10.0)
    assert opportunities[0].gross_cost == pytest.approx(10.0)
    assert opportunities[0].net_cost == pytest.approx(10.6)
    assert opportunities[0].net_edge == pytest.approx(0.6)


def test_scan_market_finds_split_sell_when_best_asks_are_missing():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test",
        enable_split_sell=True,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
    )
    opportunities = scan_market(_market(), _bid_depth_book("yes", [(0.53, 10)]), _bid_depth_book("no", [(0.53, 10)]), cfg)

    assert len(opportunities) == 1
    assert opportunities[0].direction == "split_sell_both"
    assert rejection_reason(_market(), _bid_depth_book("yes", [(0.53, 10)]), _bid_depth_book("no", [(0.53, 10)]), cfg) is None


def test_scan_market_uses_bid_depth_for_split_sell_target_notional():
    cfg = PolySettings(_env_file=None,
        data_dir="/tmp/poly-test",
        enable_split_sell=True,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
        target_notional_per_opp=25,
        max_notional_per_opp=250,
    )
    opportunities = scan_market(
        _market(),
        _bid_depth_book("yes", [(0.53, 5), (0.52, 20)]),
        _bid_depth_book("no", [(0.53, 5), (0.52, 20)]),
        cfg,
    )

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.direction == "split_sell_both"
    assert opportunity.mergeable_qty == pytest.approx(25.0)
    assert opportunity.gross_cost == pytest.approx(25.0)
    assert opportunity.net_cost == pytest.approx(26.1)
    assert opportunity.net_edge == pytest.approx(1.1)
    assert opportunity.yes_price == pytest.approx(0.522)
