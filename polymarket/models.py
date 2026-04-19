"""Domain models for isolated Polymarket paper trading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class TokenInfo:
    token_id: str
    outcome: str


@dataclass(slots=True)
class MarketInfo:
    market_id: str
    condition_id: str
    question: str
    slug: str | None
    end_date_iso: str | None
    min_order_size: float
    tick_size: float
    neg_risk: bool
    enable_order_book: bool
    taker_base_fee_bps: float
    yes_token_id: str
    no_token_id: str
    collateral_symbol: str = "USDC.e"
    fee_source: str = "taker_base_fee"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BookLevel:
    price: float
    size: float


@dataclass(slots=True)
class OrderBook:
    token_id: str
    market_id: str
    timestamp_ms: int
    bids: list[BookLevel]
    asks: list[BookLevel]
    tick_size: float
    min_order_size: float
    neg_risk: bool
    last_trade_price: float | None = None

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "market_id": self.market_id,
            "timestamp_ms": self.timestamp_ms,
            "tick_size": self.tick_size,
            "min_order_size": self.min_order_size,
            "neg_risk": self.neg_risk,
            "last_trade_price": self.last_trade_price,
            "bids": [asdict(level) for level in self.bids],
            "asks": [asdict(level) for level in self.asks],
        }


@dataclass(slots=True)
class Opportunity:
    market_id: str
    question: str
    direction: str
    gross_cost: float
    fee_cost: float
    yes_fee_cost: float
    no_fee_cost: float
    gas_cost: float
    slippage_buffer: float
    net_cost: float
    net_edge: float
    capacity: float
    mergeable_qty: float
    yes_qty: float
    no_qty: float
    yes_price: float
    no_price: float
    yes_book_timestamp_ms: int = 0
    no_book_timestamp_ms: int = 0
    ts: datetime | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.ts is None:
            self.ts = datetime.now(timezone.utc)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts"] = self.ts.astimezone(timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class PaperFill:
    opportunity_id: str
    market_id: str
    token_id: str
    side: str
    qty: float
    price: float
    fee: float
    filled_at: datetime

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filled_at"] = self.filled_at.astimezone(timezone.utc).isoformat()
        return payload
