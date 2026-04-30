"""Full-set scanner for binary Polymarket markets."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook


EPOCH_MS_CUTOFF = 10**11


class _BuyLegPlan(NamedTuple):
    qty: float
    cost: float
    fee: float
    vwap: float


class _BuyBothPlan(NamedTuple):
    mergeable_qty: float
    yes_qty: float
    no_qty: float
    yes_cost: float
    no_cost: float
    yes_fee: float
    no_fee: float
    yes_vwap: float
    no_vwap: float


class _SellLegPlan(NamedTuple):
    qty: float
    revenue: float
    fee: float
    vwap: float


class _SellBothPlan(NamedTuple):
    split_qty: float
    yes_qty: float
    no_qty: float
    yes_revenue: float
    no_revenue: float
    yes_fee: float
    no_fee: float
    yes_vwap: float
    no_vwap: float


def fee_rate_from_bps(fee_bps: float) -> float:
    return max(fee_bps, 0.0) / 10_000.0


def taker_fee(price: float, fee_rate: float, quantity: float = 1.0) -> float:
    fee = quantity * fee_rate * price * max(1.0 - price, 0.0)
    return round(fee, 5)


def buy_net_share_factor(price: float, fee_rate: float) -> float:
    return max(1.0 - fee_rate * max(1.0 - price, 0.0), 0.0)


def buy_fee_in_shares(price: float, fee_rate: float, quantity: float = 1.0) -> float:
    factor = buy_net_share_factor(price, fee_rate)
    return max(quantity * (1.0 - factor), 0.0)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _fresh_enough(yes_book: OrderBook, no_book: OrderBook, cfg: PolySettings) -> bool:
    if yes_book.timestamp_ms <= 0 or no_book.timestamp_ms <= 0:
        return False
    if yes_book.timestamp_ms < EPOCH_MS_CUTOFF or no_book.timestamp_ms < EPOCH_MS_CUTOFF:
        return abs(yes_book.timestamp_ms - no_book.timestamp_ms) <= cfg.max_book_staleness_ms
    now_ms = _now_ms()
    yes_age = now_ms - yes_book.timestamp_ms
    no_age = now_ms - no_book.timestamp_ms
    if yes_age < 0 or no_age < 0:
        return False
    if yes_age > cfg.max_book_staleness_ms or no_age > cfg.max_book_staleness_ms:
        return False
    return abs(yes_book.timestamp_ms - no_book.timestamp_ms) <= cfg.max_book_staleness_ms


def _sorted_asks(book: OrderBook) -> list:
    return sorted((level for level in book.asks if level.size > 0), key=lambda level: level.price)


def _sorted_bids(book: OrderBook) -> list:
    return sorted((level for level in book.bids if level.size > 0), key=lambda level: level.price, reverse=True)


def _plan_buy_leg(book: OrderBook, mergeable_qty: float, fee_rate: float) -> _BuyLegPlan | None:
    remaining_net_qty = mergeable_qty
    raw_qty = 0.0
    cost = 0.0
    fee = 0.0
    for level in _sorted_asks(book):
        factor = buy_net_share_factor(level.price, fee_rate)
        if factor <= 0:
            continue
        level_net_qty = level.size * factor
        if level_net_qty <= 0:
            continue
        take_net_qty = min(remaining_net_qty, level_net_qty)
        take_raw_qty = take_net_qty / factor
        raw_qty += take_raw_qty
        cost += take_raw_qty * level.price
        fee += taker_fee(level.price, fee_rate, take_raw_qty)
        remaining_net_qty -= take_net_qty
        if remaining_net_qty <= 1e-9:
            vwap = cost / raw_qty if raw_qty > 0 else 0.0
            return _BuyLegPlan(round(raw_qty, 8), cost, fee, vwap)
    return None


def _plan_sell_leg(book: OrderBook, split_qty: float, fee_rate: float) -> _SellLegPlan | None:
    remaining_qty = split_qty
    raw_qty = 0.0
    revenue = 0.0
    fee = 0.0
    for level in _sorted_bids(book):
        take_qty = min(remaining_qty, level.size)
        if take_qty <= 0:
            continue
        raw_qty += take_qty
        revenue += take_qty * level.price
        fee += taker_fee(level.price, fee_rate, take_qty)
        remaining_qty -= take_qty
        if remaining_qty <= 1e-9:
            vwap = revenue / raw_qty if raw_qty > 0 else 0.0
            return _SellLegPlan(round(raw_qty, 8), revenue, fee, vwap)
    return None


def _plan_buy_both_depth(
    yes_book: OrderBook,
    no_book: OrderBook,
    fee_rate: float,
    min_order_size: float,
    cfg: PolySettings,
) -> _BuyBothPlan | None:
    lot_size = min_order_size if min_order_size > 0 else 1.0
    max_notional = cfg.max_notional_per_opp if cfg.max_notional_per_opp > 0 else float("inf")
    target_notional = cfg.target_notional_per_opp if cfg.target_notional_per_opp > 0 else max_notional
    budget = min(max_notional, target_notional)
    if budget <= cfg.default_gas_cost:
        return None

    best_plan: _BuyBothPlan | None = None
    previous_qty = 0.0
    previous_edge = -cfg.default_gas_cost
    lots = 1
    while True:
        mergeable_qty = round(lots * lot_size, 8)
        yes_plan = _plan_buy_leg(yes_book, mergeable_qty, fee_rate)
        no_plan = _plan_buy_leg(no_book, mergeable_qty, fee_rate)
        if yes_plan is None or no_plan is None:
            break
        gross_cost = yes_plan.cost + no_plan.cost
        net_cost = gross_cost + (mergeable_qty * cfg.slippage_buffer) + cfg.default_gas_cost
        if net_cost > budget + 1e-9:
            break
        net_edge = mergeable_qty - net_cost
        if mergeable_qty > previous_qty:
            marginal_edge = (net_edge - previous_edge) / (mergeable_qty - previous_qty)
            if marginal_edge < cfg.min_depth_edge_per_share:
                break
        if net_edge >= cfg.min_net_edge:
            best_plan = _BuyBothPlan(
                mergeable_qty=mergeable_qty,
                yes_qty=yes_plan.qty,
                no_qty=no_plan.qty,
                yes_cost=yes_plan.cost,
                no_cost=no_plan.cost,
                yes_fee=yes_plan.fee,
                no_fee=no_plan.fee,
                yes_vwap=yes_plan.vwap,
                no_vwap=no_plan.vwap,
            )
        previous_qty = mergeable_qty
        previous_edge = net_edge
        lots += 1
        if lots > 10000:
            break
    return best_plan


def _plan_split_sell_depth(
    yes_book: OrderBook,
    no_book: OrderBook,
    fee_rate: float,
    min_order_size: float,
    cfg: PolySettings,
) -> _SellBothPlan | None:
    lot_size = min_order_size if min_order_size > 0 else 1.0
    max_notional = cfg.max_notional_per_opp if cfg.max_notional_per_opp > 0 else float("inf")
    target_notional = cfg.target_notional_per_opp if cfg.target_notional_per_opp > 0 else max_notional
    budget = min(max_notional, target_notional)
    if budget <= cfg.default_gas_cost:
        return None

    best_plan: _SellBothPlan | None = None
    previous_qty = 0.0
    previous_edge = -cfg.default_gas_cost
    lots = 1
    while True:
        split_qty = round(lots * lot_size, 8)
        if split_qty > budget + 1e-9:
            break
        yes_plan = _plan_sell_leg(yes_book, split_qty, fee_rate)
        no_plan = _plan_sell_leg(no_book, split_qty, fee_rate)
        if yes_plan is None or no_plan is None:
            break
        gross_cost = split_qty
        gross_revenue = yes_plan.revenue + no_plan.revenue
        fee_cost = yes_plan.fee + no_plan.fee
        net_revenue = gross_revenue - fee_cost - (split_qty * cfg.slippage_buffer) - cfg.default_gas_cost
        net_edge = net_revenue - gross_cost
        if split_qty > previous_qty:
            marginal_edge = (net_edge - previous_edge) / (split_qty - previous_qty)
            if marginal_edge < cfg.min_depth_edge_per_share:
                break
        if net_edge >= cfg.min_net_edge:
            best_plan = _SellBothPlan(
                split_qty=split_qty,
                yes_qty=yes_plan.qty,
                no_qty=no_plan.qty,
                yes_revenue=yes_plan.revenue,
                no_revenue=no_plan.revenue,
                yes_fee=yes_plan.fee,
                no_fee=no_plan.fee,
                yes_vwap=yes_plan.vwap,
                no_vwap=no_plan.vwap,
            )
        previous_qty = split_qty
        previous_edge = net_edge
        lots += 1
        if lots > 10000:
            break
    return best_plan


def _build_buy_both_opportunity(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings,
) -> tuple[Opportunity | None, str | None]:
    if market.neg_risk or yes_book.neg_risk or no_book.neg_risk:
        return None, "neg_risk_filtered"
    if not _fresh_enough(yes_book, no_book, cfg):
        return None, "stale_books"
    if yes_book.best_ask is None or no_book.best_ask is None:
        return None, "missing_best_ask"

    fee_rate = fee_rate_from_bps(market.taker_base_fee_bps)
    min_order_size = max(market.min_order_size, yes_book.min_order_size, no_book.min_order_size)
    plan = _plan_buy_both_depth(yes_book, no_book, fee_rate, min_order_size, cfg)
    if plan is None:
        return None, "capacity_below_min_order"
    if plan.yes_qty < min_order_size or plan.no_qty < min_order_size:
        return None, "qty_below_min_order"

    gross_cost = plan.yes_cost + plan.no_cost
    fee_cost = plan.yes_fee + plan.no_fee
    slippage_cost = plan.mergeable_qty * cfg.slippage_buffer
    gas_cost = cfg.default_gas_cost
    net_cost = gross_cost + slippage_cost + gas_cost
    net_edge = plan.mergeable_qty - net_cost
    if net_edge < cfg.min_net_edge:
        return None, "edge_below_threshold"

    return (
        Opportunity(
            market_id=market.market_id,
            question=market.question,
            direction="buy_both_merge",
            gross_cost=gross_cost,
            fee_cost=fee_cost,
            yes_fee_cost=plan.yes_fee,
            no_fee_cost=plan.no_fee,
            gas_cost=gas_cost,
            slippage_buffer=slippage_cost,
            net_cost=net_cost,
            net_edge=net_edge,
            capacity=plan.mergeable_qty,
            mergeable_qty=plan.mergeable_qty,
            yes_qty=plan.yes_qty,
            no_qty=plan.no_qty,
            yes_price=plan.yes_vwap,
            no_price=plan.no_vwap,
            yes_book_timestamp_ms=yes_book.timestamp_ms,
            no_book_timestamp_ms=no_book.timestamp_ms,
            ts=datetime.now(timezone.utc),
        ),
        None,
    )


def _build_split_sell_opportunity(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings,
) -> tuple[Opportunity | None, str | None]:
    if market.neg_risk or yes_book.neg_risk or no_book.neg_risk:
        return None, "neg_risk_filtered"
    if not _fresh_enough(yes_book, no_book, cfg):
        return None, "stale_books"
    if yes_book.best_bid is None or no_book.best_bid is None:
        return None, "missing_best_bid"

    fee_rate = fee_rate_from_bps(market.taker_base_fee_bps)
    min_order_size = max(market.min_order_size, yes_book.min_order_size, no_book.min_order_size)
    plan = _plan_split_sell_depth(yes_book, no_book, fee_rate, min_order_size, cfg)
    if plan is None:
        return None, "capacity_below_min_order"
    if plan.yes_qty < min_order_size or plan.no_qty < min_order_size:
        return None, "qty_below_min_order"

    gross_cost = plan.split_qty
    gross_revenue = plan.yes_revenue + plan.no_revenue
    fee_cost = plan.yes_fee + plan.no_fee
    slippage_cost = plan.split_qty * cfg.slippage_buffer
    gas_cost = cfg.default_gas_cost
    net_revenue = gross_revenue - fee_cost - slippage_cost - gas_cost
    net_edge = net_revenue - gross_cost
    if net_edge < cfg.min_net_edge:
        return None, "edge_below_threshold"

    return (
        Opportunity(
            market_id=market.market_id,
            question=market.question,
            direction="split_sell_both",
            gross_cost=gross_cost,
            fee_cost=fee_cost,
            yes_fee_cost=plan.yes_fee,
            no_fee_cost=plan.no_fee,
            gas_cost=gas_cost,
            slippage_buffer=slippage_cost,
            net_cost=net_revenue,
            net_edge=net_edge,
            capacity=plan.split_qty,
            mergeable_qty=plan.split_qty,
            yes_qty=plan.yes_qty,
            no_qty=plan.no_qty,
            yes_price=plan.yes_vwap,
            no_price=plan.no_vwap,
            yes_book_timestamp_ms=yes_book.timestamp_ms,
            no_book_timestamp_ms=no_book.timestamp_ms,
            ts=datetime.now(timezone.utc),
        ),
        None,
    )


def rejection_reason(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings | None = None,
) -> str | None:
    cfg = cfg or settings
    buy_both, buy_reason = _build_buy_both_opportunity(market, yes_book, no_book, cfg)
    if buy_both is not None:
        return None
    if cfg.enable_split_sell:
        split_sell, split_reason = _build_split_sell_opportunity(market, yes_book, no_book, cfg)
        if split_sell is not None:
            return None
        if buy_reason in {"neg_risk_filtered", "stale_books"}:
            return buy_reason
        return split_reason or buy_reason
    return buy_reason


def scan_market(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings | None = None,
) -> list[Opportunity]:
    cfg = cfg or settings
    opportunities: list[Opportunity] = []
    buy_both, reason = _build_buy_both_opportunity(market, yes_book, no_book, cfg)
    if buy_both is not None:
        opportunities.append(buy_both)
    elif reason in {"neg_risk_filtered", "stale_books"}:
        return opportunities

    if cfg.enable_split_sell:
        split_sell, _ = _build_split_sell_opportunity(market, yes_book, no_book, cfg)
        if split_sell is not None:
            opportunities.append(split_sell)

    return opportunities
