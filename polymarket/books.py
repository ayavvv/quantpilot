"""Order book fetchers for Polymarket."""
from __future__ import annotations

import json
from time import sleep
from typing import Any
from urllib.request import Request, urlopen

from polymarket.config import PolySettings, settings
from polymarket.models import BookLevel, OrderBook


def _parse_levels(levels: list[dict[str, Any]]) -> list[BookLevel]:
    parsed: list[BookLevel] = []
    for level in levels:
        try:
            parsed.append(BookLevel(price=float(level["price"]), size=float(level["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


class ClobClient:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.cfg.clob_base_url.rstrip('/')}/{path.lstrip('/')}"
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.cfg.http_timeout_seconds) as response:
            return json.load(response)

    def fetch_book(self, token_id: str) -> OrderBook:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                data = self._get_json(f"book?token_id={token_id}")
                bids = _parse_levels(data.get("bids", []))
                asks = _parse_levels(data.get("asks", []))
                return OrderBook(
                    token_id=str(data.get("asset_id") or token_id),
                    market_id=str(data.get("market") or ""),
                    timestamp_ms=int(data.get("timestamp") or 0),
                    bids=bids,
                    asks=asks,
                    tick_size=float(data.get("tick_size") or 0.01),
                    min_order_size=float(data.get("min_order_size") or 0),
                    neg_risk=bool(data.get("neg_risk")),
                    last_trade_price=float(data["last_trade_price"]) if data.get("last_trade_price") is not None else None,
                )
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                if attempt == 2:
                    raise
                sleep(0.2 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"failed to fetch book for {token_id}")
