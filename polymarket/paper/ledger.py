"""Paper ledger for Polymarket full-set simulations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

import pandas as pd

from polymarket.models import Opportunity, PaperFill


@dataclass(slots=True)
class LedgerState:
    cash: float
    realized_pnl: float = 0.0


class PaperLedger:
    def __init__(self, initial_cash: float, realized_pnl: float = 0.0):
        self.state = LedgerState(cash=initial_cash, realized_pnl=realized_pnl)
        self.positions = pd.DataFrame(columns=["token_id", "qty", "avg_price", "realized_pnl", "updated_at"])

    def can_fill(self, opportunity: Opportunity) -> bool:
        required_cash = opportunity.net_cost if opportunity.direction == "buy_both_merge" else opportunity.gross_cost
        return self.state.cash >= required_cash

    def build_opportunity_id(self, opportunity: Opportunity) -> str:
        fingerprint = (
            f"{opportunity.market_id}:{opportunity.direction}:"
            f"{opportunity.yes_book_timestamp_ms}:{opportunity.no_book_timestamp_ms}:"
            f"{opportunity.yes_price:.8f}:{opportunity.no_price:.8f}:"
            f"{opportunity.mergeable_qty:.8f}:{opportunity.yes_qty:.8f}:{opportunity.no_qty:.8f}"
        )
        return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()

    def apply_opportunity(self, market, opportunity: Opportunity) -> list[PaperFill]:
        if not self.can_fill(opportunity):
            return []

        now = datetime.now(timezone.utc)
        opportunity_id = self.build_opportunity_id(opportunity)

        side = "buy" if opportunity.direction == "buy_both_merge" else "sell"
        fills = [
            PaperFill(opportunity_id, market.market_id, market.yes_token_id, side, opportunity.yes_qty, opportunity.yes_price, opportunity.yes_fee_cost, now),
            PaperFill(opportunity_id, market.market_id, market.no_token_id, side, opportunity.no_qty, opportunity.no_price, opportunity.no_fee_cost, now),
        ]

        if opportunity.direction == "buy_both_merge":
            self.state.cash -= opportunity.net_cost
            self.state.cash += opportunity.mergeable_qty
        else:
            self.state.cash -= opportunity.gross_cost
            self.state.cash += opportunity.net_cost
        self.state.realized_pnl += opportunity.net_edge

        self.positions = pd.DataFrame(
            [
                {
                    "token_id": market.yes_token_id,
                    "qty": 0.0,
                    "avg_price": 0.0,
                    "realized_pnl": self.state.realized_pnl,
                    "updated_at": now,
                },
                {
                    "token_id": market.no_token_id,
                    "qty": 0.0,
                    "avg_price": 0.0,
                    "realized_pnl": self.state.realized_pnl,
                    "updated_at": now,
                },
            ]
        )
        return fills
