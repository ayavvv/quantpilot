"""Paper-trading simulator for isolated Polymarket opportunities."""
from __future__ import annotations

import json
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
        self.last_accepted_opportunities: list[Opportunity] = []
        self.last_rejection_counts: dict[str, int] = {}
        self._last_fill_by_market = self.storage.load_latest_fill_times_by_market()
        self._daily_notional_by_market: dict[str, float] = {}
        self._daily_notional = 0.0
        self._day_key = ""
        self._day_start_realized_pnl = self.ledger.state.realized_pnl
        self._reset_day_if_needed()

    def _reset_day_if_needed(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        day_key = now.date().isoformat()
        if day_key == self._day_key:
            return
        self._day_key = day_key
        self._daily_notional_by_market = self.storage.load_fill_notional_by_market(day_key)
        self._daily_notional = sum(self._daily_notional_by_market.values())
        self._day_start_realized_pnl = self.ledger.state.realized_pnl

    def _opportunity_notional(self, opportunity: Opportunity) -> float:
        if opportunity.direction == "buy_both_merge":
            return max(opportunity.net_cost, 0.0)
        return max(opportunity.gross_cost, 0.0)

    def _risk_rejection(self, market: MarketInfo, opportunity: Opportunity, now: datetime) -> str | None:
        self._reset_day_if_needed(now)
        cooldown = max(float(self.cfg.market_cooldown_seconds), 0.0)
        last_fill = self._last_fill_by_market.get(market.market_id)
        if cooldown > 0 and last_fill is not None:
            if last_fill.tzinfo is None:
                last_fill = last_fill.replace(tzinfo=timezone.utc)
            if (now - last_fill).total_seconds() < cooldown:
                return "market_cooldown"

        daily_pnl = self.ledger.state.realized_pnl - self._day_start_realized_pnl
        if self.cfg.max_daily_loss > 0 and daily_pnl <= -self.cfg.max_daily_loss:
            return "daily_loss_limit"

        notional = self._opportunity_notional(opportunity)
        market_notional = self._daily_notional_by_market.get(market.market_id, 0.0)
        if self.cfg.max_market_notional_per_day > 0 and market_notional + notional > self.cfg.max_market_notional_per_day:
            return "market_notional_limit"
        if self.cfg.max_daily_notional > 0 and self._daily_notional + notional > self.cfg.max_daily_notional:
            return "daily_notional_limit"
        if not self.ledger.can_fill(opportunity):
            return "insufficient_cash"
        return None

    def record_scan_heartbeat(self, strategy_type: str = "full_set_arb", rejection_counts: dict[str, int] | None = None) -> None:
        self.storage.upsert_daily_summary(
            {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "strategy_type": strategy_type,
                "signals": 0,
                "accepted_signals": 0,
                "simulated_trades": 0,
                "gross_edge_sum": 0.0,
                "net_edge_sum": 0.0,
                "realized_pnl": self.ledger.state.realized_pnl,
                "max_inventory_used": 0.0,
                "rejection_counts_json": json.dumps(rejection_counts or {}, ensure_ascii=False, sort_keys=True),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def consume(self, market: MarketInfo, opportunities: list[Opportunity]) -> int:
        self.last_accepted_opportunities = []
        self.last_rejection_counts = {}
        if not opportunities:
            return 0
        accepted = 0
        gross_edge_sum = 0.0
        net_edge_sum = 0.0
        max_inventory_used = 0.0
        for opportunity in opportunities:
            now = datetime.now(timezone.utc)
            opportunity_id = self.ledger.build_opportunity_id(opportunity)
            rejection = self._risk_rejection(market, opportunity, now)
            if rejection is not None:
                self.last_rejection_counts[rejection] = int(self.last_rejection_counts.get(rejection, 0)) + 1
                continue
            if not self.storage.claim_opportunity(opportunity_id):
                self.last_rejection_counts["duplicate_opportunity"] = int(self.last_rejection_counts.get("duplicate_opportunity", 0)) + 1
                continue
            fills = self.ledger.apply_opportunity(market, opportunity)
            if not fills:
                self.last_rejection_counts["unfilled_after_claim"] = int(self.last_rejection_counts.get("unfilled_after_claim", 0)) + 1
                continue
            accepted += 1
            gross_edge_sum += opportunity.mergeable_qty - opportunity.gross_cost
            net_edge_sum += opportunity.net_edge
            max_inventory_used = max(max_inventory_used, opportunity.net_cost if opportunity.direction == "buy_both_merge" else opportunity.gross_cost)
            self.storage.save_fills(fills)
            notional = self._opportunity_notional(opportunity)
            self._daily_notional += notional
            self._daily_notional_by_market[market.market_id] = self._daily_notional_by_market.get(market.market_id, 0.0) + notional
            self._last_fill_by_market[market.market_id] = now
            self.last_accepted_opportunities.append(opportunity)
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
