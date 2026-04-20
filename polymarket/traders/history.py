"""Public trader history ingestion for Polymarket mirror strategy."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polymarket.config import PolySettings, settings
from polymarket.models import TraderEvent


DATA_API_BASE = "https://data-api.polymarket.com"


class TraderHistoryClient:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{DATA_API_BASE.rstrip('/')}/{path.lstrip('/')}" + query
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.cfg.http_timeout_seconds) as response:
            return json.load(response)

    def fetch_positions(self, wallet: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._get_json("positions", params={"user": wallet, "limit": limit, "offset": 0}))

    def fetch_closed_positions(self, wallet: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._get_json("closed-positions", params={"user": wallet, "limit": limit, "offset": 0}))

    def fetch_activity(self, wallet: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._get_json("activity", params={"user": wallet, "limit": limit, "offset": 0}))

    def fetch_trades(self, wallet: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._get_json("trades", params={"user": wallet, "limit": limit, "offset": 0, "takerOnly": "false"}))


def normalize_activity_events(wallet: str, rows: list[dict[str, Any]]) -> list[TraderEvent]:
    events: list[TraderEvent] = []
    for row in rows:
        events.append(
            TraderEvent(
                wallet=wallet,
                event_type=str(row.get("type") or "UNKNOWN"),
                market_id=str(row.get("conditionId") or ""),
                asset=str(row.get("asset") or ""),
                side=(row.get("side") or None),
                size=float(row.get("size") or 0.0),
                price=float(row["price"]) if row.get("price") is not None else None,
                timestamp=int(row.get("timestamp") or 0),
                transaction_hash=(row.get("transactionHash") or None),
                title=(row.get("title") or None),
                outcome=(row.get("outcome") or None),
                user_name=(row.get("name") or None),
            )
        )
    return events


def summarize_trader_history(positions: list[dict[str, Any]], closed_positions: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, float | int]:
    diversity = {str(item.get("conditionId") or item.get("eventId") or "") for item in positions + closed_positions + trades}
    diversity.discard("")
    realized_pnl = sum(float(item.get("realizedPnl") or 0.0) for item in closed_positions)
    volume = sum(float(item.get("totalBought") or item.get("initialValue") or item.get("currentValue") or 0.0) for item in positions + closed_positions)
    return {
        "trade_count": len(trades),
        "diversity_count": len(diversity),
        "realized_pnl": realized_pnl,
        "volume": volume,
    }
