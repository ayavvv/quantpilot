"""Paper bookkeeping for the isolated top-trader mirror strategy."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from polymarket.models import MirrorSignal
from polymarket.scanner.full_set import buy_fee_in_shares, fee_rate_from_bps, taker_fee
from polymarket.traders.storage import MirrorStorage


@dataclass(slots=True)
class MirrorState:
    cash: float
    realized_pnl: float = 0.0


class MirrorBookkeeper:
    def __init__(self, storage: MirrorStorage, initial_cash: float):
        self.storage = storage
        state = self.storage.load_state()
        if state is None:
            self.state = MirrorState(cash=initial_cash)
        else:
            self.state = MirrorState(cash=state["cash"], realized_pnl=state["realized_pnl"])
        self.positions = self._load_positions()

    def _load_positions(self) -> pd.DataFrame:
        if not self.storage.db_path.exists():
            return pd.DataFrame(columns=["asset", "market_id", "title", "outcome", "qty", "avg_price", "realized_pnl", "updated_at"])
        try:
            import duckdb
            conn = duckdb.connect(str(self.storage.db_path), read_only=True)
            try:
                return conn.execute("SELECT asset, market_id, title, outcome, qty, avg_price, realized_pnl, updated_at FROM mirror_positions").df()
            finally:
                conn.close()
        except Exception:
            return pd.DataFrame(columns=["asset", "market_id", "title", "outcome", "qty", "avg_price", "realized_pnl", "updated_at"])

    def _position_for(self, asset: str) -> dict | None:
        if self.positions.empty:
            return None
        rows = self.positions[self.positions["asset"] == asset]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def apply_signal(self, signal: MirrorSignal, market_id: str, title: str, outcome: str | None, execution_price: float, fee_bps: float) -> bool:
        fee_rate = fee_rate_from_bps(fee_bps)
        now = datetime.now(timezone.utc)
        signal_size = signal.signal_size
        existing = self._position_for(signal.asset)

        if signal.side == "BUY":
            gross_cost = signal_size * execution_price
            if self.state.cash < gross_cost:
                return False
            fee_cash = taker_fee(execution_price, fee_rate, signal_size)
            net_qty = signal_size - buy_fee_in_shares(execution_price, fee_rate, signal_size)
            if existing is None:
                new_qty = net_qty
                avg_price = gross_cost / max(net_qty, 1e-9)
                realized_pnl = self.state.realized_pnl
            else:
                new_qty = float(existing["qty"]) + net_qty
                avg_price = ((float(existing["qty"]) * float(existing["avg_price"])) + gross_cost) / max(new_qty, 1e-9)
                realized_pnl = float(existing["realized_pnl"])
            self.state.cash -= gross_cost
            fill = {
                "signal_fingerprint": signal.fingerprint,
                "wallet": signal.wallet,
                "market_id": market_id,
                "asset": signal.asset,
                "side": signal.side,
                "qty": signal_size,
                "gross_price": execution_price,
                "fee_cash": fee_cash,
                "net_qty": net_qty,
                "proceeds": -gross_cost,
                "filled_at": now,
            }
        else:
            if existing is None or float(existing["qty"]) <= 0:
                return False
            sell_qty = min(signal_size, float(existing["qty"]))
            gross_proceeds = sell_qty * execution_price
            fee_cash = taker_fee(execution_price, fee_rate, sell_qty)
            proceeds = gross_proceeds - fee_cash
            avg_price = float(existing["avg_price"])
            pnl_delta = proceeds - sell_qty * avg_price
            new_qty = float(existing["qty"]) - sell_qty
            realized_pnl = float(existing["realized_pnl"]) + pnl_delta
            self.state.cash += proceeds
            self.state.realized_pnl += pnl_delta
            fill = {
                "signal_fingerprint": signal.fingerprint,
                "wallet": signal.wallet,
                "market_id": market_id,
                "asset": signal.asset,
                "side": signal.side,
                "qty": sell_qty,
                "gross_price": execution_price,
                "fee_cash": fee_cash,
                "net_qty": sell_qty,
                "proceeds": proceeds,
                "filled_at": now,
            }

        if not self.storage.claim_signal(signal.fingerprint):
            return False
        self.storage.save_mirror_fill(fill)
        row = {
            "asset": signal.asset,
            "market_id": market_id,
            "title": title,
            "outcome": outcome,
            "qty": new_qty,
            "avg_price": avg_price,
            "realized_pnl": realized_pnl,
            "updated_at": now,
        }
        if self.positions.empty:
            self.positions = pd.DataFrame([row])
        else:
            self.positions = self.positions[self.positions["asset"] != signal.asset]
            self.positions = pd.concat([self.positions, pd.DataFrame([row])], ignore_index=True)
        self.storage.upsert_positions(self.positions)
        self.storage.save_state(self.state.cash, self.state.realized_pnl)
        return True
