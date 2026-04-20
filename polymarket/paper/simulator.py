"""Paper-trading simulator for isolated Polymarket opportunities."""
from __future__ import annotations

from datetime import datetime, timezone

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity
from polymarket.paper.ledger import PaperLedger
from polymarket.storage import PolyStorage


class PaperSimulator:
    def __init__(self, storage: PolyStorage, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.storage = storage
        if not self.cfg.paper_only:
            raise ValueError("Polymarket simulator only supports paper_only=true")
        state = self.storage.load_state()
        initial_cash = self.cfg.paper_initial_cash if state is None else state["cash"]
        realized_pnl = 0.0 if state is None else state["realized_pnl"]
        self.ledger = PaperLedger(initial_cash=initial_cash, realized_pnl=realized_pnl)

    def consume(self, market: MarketInfo, opportunities: list[Opportunity]) -> int:
        accepted = 0
        gross_edge_sum = 0.0
        net_edge_sum = 0.0
        max_inventory_used = 0.0
        for opportunity in opportunities:
            opportunity_id = self.ledger.build_opportunity_id(opportunity)
            fills = self.ledger.apply_opportunity(market, opportunity)
            if not fills:
                continue
            if not self.storage.claim_opportunity(opportunity_id):
                continue
            accepted += 1
            gross_edge_sum += opportunity.mergeable_qty - opportunity.gross_cost
            net_edge_sum += opportunity.net_edge
            max_inventory_used = max(max_inventory_used, opportunity.net_cost if opportunity.direction == "buy_both_merge" else opportunity.gross_cost)
            self.storage.save_fills(fills)
        self.storage.upsert_positions(self.ledger.positions)
        self.storage.save_state(self.ledger.state.cash, self.ledger.state.realized_pnl)
        strategy_type = opportunities[0].strategy_type if opportunities else "full_set_arb"
        self.storage.upsert_daily_summary(
            {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "strategy_type": strategy_type,
                "signals": len(opportunities),
                "accepted_signals": accepted,
                "simulated_trades": accepted,
                "gross_edge_sum": gross_edge_sum,
                "net_edge_sum": net_edge_sum,
                "realized_pnl": self.ledger.state.realized_pnl,
                "max_inventory_used": max_inventory_used,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        return accepted
