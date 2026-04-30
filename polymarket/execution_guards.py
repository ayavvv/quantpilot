"""In-process execution guards for Polymarket paper scanning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from polymarket.models import BookLevel, MarketInfo, Opportunity, OrderBook


@dataclass(slots=True)
class _Depletion:
    qty: float
    expires_at: datetime


class LocalBookDepletion:
    """Subtract recently simulated fills from fresh books.

    The websocket cache can replay the same top-of-book after a paper fill because
    the simulator does not touch the real CLOB. This guard makes subsequent scans
    behave as if our simulated order consumed local liquidity for a short TTL.
    """

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = max(float(ttl_seconds), 0.0)
        self._depletions: dict[tuple[str, str, float], _Depletion] = {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup(self, now: datetime | None = None) -> None:
        now = now or self._now()
        expired = [key for key, depletion in self._depletions.items() if depletion.expires_at <= now or depletion.qty <= 0]
        for key in expired:
            self._depletions.pop(key, None)

    def apply(self, book: OrderBook) -> OrderBook:
        if self.ttl_seconds <= 0:
            return book
        self._cleanup()
        asks = self._apply_side(book.token_id, "ask", book.asks)
        bids = self._apply_side(book.token_id, "bid", book.bids)
        return OrderBook(
            token_id=book.token_id,
            market_id=book.market_id,
            timestamp_ms=book.timestamp_ms,
            bids=bids,
            asks=asks,
            tick_size=book.tick_size,
            min_order_size=book.min_order_size,
            neg_risk=book.neg_risk,
            last_trade_price=book.last_trade_price,
        )

    def _apply_side(self, token_id: str, side: str, levels: list[BookLevel]) -> list[BookLevel]:
        adjusted: list[BookLevel] = []
        for level in levels:
            depletion = self._depletions.get((token_id, side, round(level.price, 8)))
            depleted_qty = depletion.qty if depletion is not None else 0.0
            remaining = round(max(level.size - depleted_qty, 0.0), 8)
            if remaining > 0:
                adjusted.append(BookLevel(price=level.price, size=remaining))
        return adjusted

    def record(self, market: MarketInfo, yes_book: OrderBook, no_book: OrderBook, opportunity: Opportunity) -> None:
        if self.ttl_seconds <= 0:
            return
        if opportunity.direction == "buy_both_merge":
            self._record_side(yes_book.token_id, "ask", yes_book.asks, opportunity.yes_qty)
            self._record_side(no_book.token_id, "ask", no_book.asks, opportunity.no_qty)
        elif opportunity.direction == "split_sell_both":
            self._record_side(yes_book.token_id, "bid", yes_book.bids, opportunity.yes_qty)
            self._record_side(no_book.token_id, "bid", no_book.bids, opportunity.no_qty)

    def _record_side(self, token_id: str, side: str, levels: list[BookLevel], qty: float) -> None:
        self._cleanup()
        remaining = max(qty, 0.0)
        if side == "ask":
            ordered = sorted(levels, key=lambda level: level.price)
        else:
            ordered = sorted(levels, key=lambda level: level.price, reverse=True)
        expires_at = self._now() + timedelta(seconds=self.ttl_seconds)
        for level in ordered:
            if remaining <= 1e-9:
                break
            take = min(level.size, remaining)
            key = (token_id, side, round(level.price, 8))
            existing = self._depletions.get(key)
            if existing is None:
                self._depletions[key] = _Depletion(qty=take, expires_at=expires_at)
            else:
                existing.qty += take
                existing.expires_at = max(existing.expires_at, expires_at)
            remaining -= take
