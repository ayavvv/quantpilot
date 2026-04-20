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
    strategy_type: str = "full_set_arb"
    source_trader_wallet: str | None = None
    source_trader_name: str | None = None
    mirror_lag_seconds: int | None = None
    source_event_fingerprint: str | None = None
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
    strategy_type: str = "full_set_arb"
    source_trader_wallet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filled_at"] = self.filled_at.astimezone(timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class TraderProfile:
    wallet: str
    user_name: str | None = None
    pseudonym: str | None = None
    verified_badge: bool = False
    profile_image: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraderScore:
    wallet: str
    score: float
    rank: int | None = None
    pnl: float = 0.0
    volume: float = 0.0
    trade_count: int = 0
    diversity_count: int = 0
    realized_pnl: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraderEvent:
    wallet: str
    event_type: str
    market_id: str
    asset: str
    side: str | None
    size: float
    price: float | None
    timestamp: int
    transaction_hash: str | None
    title: str | None = None
    outcome: str | None = None
    user_name: str | None = None

    def fingerprint(self) -> str:
        return f"{self.wallet}:{self.event_type}:{self.transaction_hash or ''}:{self.asset}:{self.side or ''}:{self.timestamp}:{self.size}"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint()
        return payload


@dataclass(slots=True)
class MirrorSignal:
    wallet: str
    market_id: str
    asset: str
    title: str
    outcome: str | None
    side: str
    source_size: float
    source_price: float | None
    signal_size: float
    lag_seconds: int
    timestamp: int
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
