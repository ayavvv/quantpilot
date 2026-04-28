"""Full-set scanner for binary Polymarket markets."""
from __future__ import annotations

from datetime import datetime, timezone
from math import floor

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook


EPOCH_MS_CUTOFF = 10**11


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


def _best_pair_capacity(yes_book: OrderBook, no_book: OrderBook, side: str, min_order_size: float) -> float:
    if side == "buy":
        if yes_book.best_ask is None or no_book.best_ask is None:
            return 0.0
        capacity = min(yes_book.best_ask.size, no_book.best_ask.size)
    else:
        if yes_book.best_bid is None or no_book.best_bid is None:
            return 0.0
        capacity = min(yes_book.best_bid.size, no_book.best_bid.size)
    if min_order_size > 0:
        lots = floor(capacity / min_order_size)
        return round(lots * min_order_size, 8)
    return round(capacity, 8)


def _clip_capacity(capacity: float, min_order_size: float, max_capacity: float | None) -> float:
    if max_capacity is not None:
        capacity = min(capacity, max_capacity)
    if capacity <= 0:
        return 0.0
    if min_order_size > 0:
        lots = floor(capacity / min_order_size)
        return round(lots * min_order_size, 8)
    return round(capacity, 8)


def rejection_reason(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings | None = None,
) -> str | None:
    cfg = cfg or settings
    if market.neg_risk or yes_book.neg_risk or no_book.neg_risk:
        return "neg_risk_filtered"
    if not _fresh_enough(yes_book, no_book, cfg):
        return "stale_books"
    if yes_book.best_ask is None or no_book.best_ask is None:
        return "missing_best_ask"

    fee_rate = fee_rate_from_bps(market.taker_base_fee_bps)
    min_order_size = max(market.min_order_size, yes_book.min_order_size, no_book.min_order_size)
    yes_price = yes_book.best_ask.price
    no_price = no_book.best_ask.price
    yes_factor = buy_net_share_factor(yes_price, fee_rate)
    no_factor = buy_net_share_factor(no_price, fee_rate)
    if yes_factor <= 0 or no_factor <= 0:
        return "invalid_fee_adjustment"
    raw_mergeable_capacity = min(
        yes_book.best_ask.size * yes_factor,
        no_book.best_ask.size * no_factor,
    )
    max_capacity = None
    per_mergeable_cost = (yes_price / yes_factor) + (no_price / no_factor) + cfg.slippage_buffer
    if cfg.max_notional_per_opp > 0 and per_mergeable_cost > 0 and cfg.max_notional_per_opp > cfg.default_gas_cost:
        max_capacity = (cfg.max_notional_per_opp - cfg.default_gas_cost) / per_mergeable_cost
    mergeable_qty = _clip_capacity(raw_mergeable_capacity, min_order_size, max_capacity)
    if mergeable_qty <= 0:
        return "capacity_below_min_order"
    yes_qty = round(mergeable_qty / yes_factor, 8)
    no_qty = round(mergeable_qty / no_factor, 8)
    if yes_qty < min_order_size or no_qty < min_order_size:
        return "qty_below_min_order"
    gross_cost = (yes_qty * yes_price) + (no_qty * no_price)
    slippage_cost = mergeable_qty * cfg.slippage_buffer
    gas_cost = cfg.default_gas_cost
    net_cost = gross_cost + slippage_cost + gas_cost
    net_edge = mergeable_qty - net_cost
    if net_edge < cfg.min_net_edge:
        return "edge_below_threshold"
    return None


def scan_market(
    market: MarketInfo,
    yes_book: OrderBook,
    no_book: OrderBook,
    cfg: PolySettings | None = None,
) -> list[Opportunity]:
    cfg = cfg or settings
    opportunities: list[Opportunity] = []
    reason = rejection_reason(market, yes_book, no_book, cfg)
    if reason is not None:
        return opportunities

    fee_rate = fee_rate_from_bps(market.taker_base_fee_bps)
    min_order_size = max(market.min_order_size, yes_book.min_order_size, no_book.min_order_size)

    if yes_book.best_ask is not None and no_book.best_ask is not None:
        yes_price = yes_book.best_ask.price
        no_price = no_book.best_ask.price
        yes_factor = buy_net_share_factor(yes_price, fee_rate)
        no_factor = buy_net_share_factor(no_price, fee_rate)
        if yes_factor > 0 and no_factor > 0:
            raw_mergeable_capacity = min(
                yes_book.best_ask.size * yes_factor,
                no_book.best_ask.size * no_factor,
            )
            max_capacity = None
            per_mergeable_cost = (yes_price / yes_factor) + (no_price / no_factor) + cfg.slippage_buffer
            if cfg.max_notional_per_opp > 0 and per_mergeable_cost > 0 and cfg.max_notional_per_opp > cfg.default_gas_cost:
                max_capacity = (cfg.max_notional_per_opp - cfg.default_gas_cost) / per_mergeable_cost
            mergeable_qty = _clip_capacity(raw_mergeable_capacity, min_order_size, max_capacity)
            if mergeable_qty > 0:
                yes_qty = round(mergeable_qty / yes_factor, 8)
                no_qty = round(mergeable_qty / no_factor, 8)
                if yes_qty >= min_order_size and no_qty >= min_order_size:
                    gross_cost = (yes_qty * yes_price) + (no_qty * no_price)
                    yes_fee_cost = taker_fee(yes_price, fee_rate, yes_qty)
                    no_fee_cost = taker_fee(no_price, fee_rate, no_qty)
                    fee_cost = yes_fee_cost + no_fee_cost
                    slippage_cost = mergeable_qty * cfg.slippage_buffer
                    gas_cost = cfg.default_gas_cost
                    net_cost = gross_cost + slippage_cost + gas_cost
                    net_edge = mergeable_qty - net_cost
                    if net_edge >= cfg.min_net_edge:
                        opportunities.append(
                            Opportunity(
                                market_id=market.market_id,
                                question=market.question,
                                direction="buy_both_merge",
                                gross_cost=gross_cost,
                                fee_cost=fee_cost,
                                yes_fee_cost=yes_fee_cost,
                                no_fee_cost=no_fee_cost,
                                gas_cost=gas_cost,
                                slippage_buffer=slippage_cost,
                                net_cost=net_cost,
                                net_edge=net_edge,
                                capacity=mergeable_qty,
                                mergeable_qty=mergeable_qty,
                                yes_qty=yes_qty,
                                no_qty=no_qty,
                                yes_price=yes_price,
                                no_price=no_price,
                                yes_book_timestamp_ms=yes_book.timestamp_ms,
                                no_book_timestamp_ms=no_book.timestamp_ms,
                                ts=datetime.now(timezone.utc),
                            )
                        )

    if cfg.enable_split_sell and yes_book.best_bid is not None and no_book.best_bid is not None:
        yes_price = yes_book.best_bid.price
        no_price = no_book.best_bid.price
        raw_capacity = _best_pair_capacity(yes_book, no_book, side="sell", min_order_size=min_order_size)
        if raw_capacity > 0 and cfg.max_notional_per_opp > 0:
            raw_capacity = _clip_capacity(raw_capacity, min_order_size, cfg.max_notional_per_opp)
        if raw_capacity > 0:
            gross_revenue = raw_capacity * (yes_price + no_price)
            yes_fee_cost = taker_fee(yes_price, fee_rate, raw_capacity)
            no_fee_cost = taker_fee(no_price, fee_rate, raw_capacity)
            fee_cost = yes_fee_cost + no_fee_cost
            slippage_cost = raw_capacity * cfg.slippage_buffer
            gas_cost = cfg.default_gas_cost
            net_revenue = gross_revenue - fee_cost - slippage_cost - gas_cost
            net_edge = net_revenue - raw_capacity
            if net_edge >= cfg.min_net_edge:
                opportunities.append(
                    Opportunity(
                        market_id=market.market_id,
                        question=market.question,
                        direction="split_sell_both",
                        gross_cost=raw_capacity,
                        fee_cost=fee_cost,
                        yes_fee_cost=yes_fee_cost,
                        no_fee_cost=no_fee_cost,
                        gas_cost=gas_cost,
                        slippage_buffer=slippage_cost,
                        net_cost=net_revenue,
                        net_edge=net_edge,
                        capacity=raw_capacity,
                        mergeable_qty=raw_capacity,
                        yes_qty=raw_capacity,
                        no_qty=raw_capacity,
                        yes_price=yes_price,
                        no_price=no_price,
                        yes_book_timestamp_ms=yes_book.timestamp_ms,
                        no_book_timestamp_ms=no_book.timestamp_ms,
                        ts=datetime.now(timezone.utc),
                    )
                )

    return opportunities
