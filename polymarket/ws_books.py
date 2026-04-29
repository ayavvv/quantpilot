"""WebSocket order book cache for Polymarket market data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from time import monotonic, sleep
from typing import Any

from loguru import logger

from polymarket.config import PolySettings, settings
from polymarket.models import BookLevel, MarketInfo, OrderBook


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class _CachedOrderBook:
    token_id: str
    market_id: str = ""
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    timestamp_ms: int = 0
    tick_size: float = 0.01
    min_order_size: float = 0.0
    neg_risk: bool = False
    last_trade_price: float | None = None
    snapshot_ready: bool = False
    updated_monotonic: float = 0.0
    best_bid_price: float | None = None
    best_bid_size: float | None = None
    best_ask_price: float | None = None
    best_ask_size: float | None = None

    @staticmethod
    def _best_level(levels: dict[float, float], reverse: bool) -> tuple[float | None, float | None]:
        usable = [(price, size) for price, size in levels.items() if size > 0]
        if not usable:
            return None, None
        price, size = max(usable) if reverse else min(usable)
        return price, size

    def recompute_tops(self) -> None:
        self.best_bid_price, self.best_bid_size = self._best_level(self.bids, reverse=True)
        self.best_ask_price, self.best_ask_size = self._best_level(self.asks, reverse=False)

    def _top_levels(self, side: str) -> list[BookLevel]:
        if side == "bid":
            price, size = self.best_bid_price, self.best_bid_size
        else:
            price, size = self.best_ask_price, self.best_ask_size
        if price is None or size is None or size <= 0:
            return []
        return [BookLevel(price=price, size=size)]

    def _all_levels(self, levels: dict[float, float], reverse: bool) -> list[BookLevel]:
        usable = [(price, size) for price, size in levels.items() if size > 0]
        return [BookLevel(price=price, size=size) for price, size in sorted(usable, reverse=reverse)]

    def to_order_book(self, timestamp_ms: int, top_only: bool = True) -> OrderBook:
        if top_only:
            bids = self._top_levels("bid")
            asks = self._top_levels("ask")
        else:
            bids = self._all_levels(self.bids, reverse=True)
            asks = self._all_levels(self.asks, reverse=False)
        return OrderBook(
            token_id=self.token_id,
            market_id=self.market_id,
            timestamp_ms=timestamp_ms,
            bids=bids,
            asks=asks,
            tick_size=self.tick_size,
            min_order_size=self.min_order_size,
            neg_risk=self.neg_risk,
            last_trade_price=self.last_trade_price,
        )


class PolymarketBookCache:
    """Thread-safe in-memory order book cache fed by CLOB WebSocket events."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._books: dict[str, _CachedOrderBook] = {}
        self._connected = False
        self._last_message_monotonic = 0.0
        self._last_error: str | None = None
        self._dirty_market_ids: set[str] = set()

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self._connected = connected
            if connected:
                self._last_message_monotonic = monotonic()

    def mark_message(self) -> None:
        with self._lock:
            self._last_message_monotonic = monotonic()

    def set_error(self, error: Exception | str | None) -> None:
        with self._lock:
            self._last_error = None if error is None else str(error)

    def is_healthy(self, max_silence_seconds: float) -> bool:
        with self._lock:
            if not self._connected:
                return False
            if self._last_message_monotonic <= 0:
                return False
            return monotonic() - self._last_message_monotonic <= max_silence_seconds

    def ready_count(self, token_ids: list[str]) -> int:
        with self._lock:
            return sum(1 for token_id in token_ids if self._books.get(token_id, None) and self._books[token_id].snapshot_ready)

    def wait_until_ready(self, token_ids: list[str], timeout_seconds: float, min_ready_ratio: float = 1.0) -> bool:
        if not token_ids:
            return True
        deadline = monotonic() + max(timeout_seconds, 0.0)
        required = max(1, int(len(token_ids) * min(max(min_ready_ratio, 0.0), 1.0)))
        while True:
            if self.ready_count(token_ids) >= required:
                return True
            if monotonic() >= deadline:
                return False
            sleep(0.05)

    def stats(self, token_ids: list[str] | None = None) -> dict[str, object]:
        with self._lock:
            total = len(self._books) if token_ids is None else len(token_ids)
            ready = (
                sum(1 for book in self._books.values() if book.snapshot_ready)
                if token_ids is None
                else self.ready_count(token_ids)
            )
            return {
                "connected": self._connected,
                "ready_tokens": ready,
                "total_tokens": total,
                "last_message_age_seconds": (
                    monotonic() - self._last_message_monotonic if self._last_message_monotonic else None
                ),
                "last_error": self._last_error,
                "dirty_markets": len(self._dirty_market_ids),
            }

    def mark_market_dirty(self, market_id: str) -> None:
        if not market_id:
            return
        with self._lock:
            self._dirty_market_ids.add(market_id)

    def pop_dirty_market_ids(self, limit: int | None = None) -> set[str]:
        with self._lock:
            if not self._dirty_market_ids:
                return set()
            if limit is None or limit <= 0 or limit >= len(self._dirty_market_ids):
                market_ids = set(self._dirty_market_ids)
                self._dirty_market_ids.clear()
                return market_ids
            market_ids = set(list(self._dirty_market_ids)[:limit])
            self._dirty_market_ids.difference_update(market_ids)
            return market_ids

    def apply_message_text(self, message: str) -> None:
        payload = json.loads(message)
        self.apply_payload(payload)

    def apply_payload(self, payload: Any) -> None:
        self.mark_message()
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    self._apply_event(item)
            return
        if isinstance(payload, dict):
            self._apply_event(payload)

    def _apply_event(self, data: dict[str, Any]) -> None:
        event_type = data.get("event_type")
        if event_type == "book" or ("bids" in data and "asks" in data and "asset_id" in data):
            self.apply_book(data)
        elif event_type == "price_change":
            self.apply_price_change(data)
        elif event_type == "last_trade_price":
            self.apply_last_trade(data)
        elif event_type == "best_bid_ask":
            self.apply_best_bid_ask(data)

    def _state_for(self, token_id: str, market_id: str = "") -> _CachedOrderBook:
        state = self._books.get(token_id)
        if state is None:
            state = _CachedOrderBook(token_id=token_id, market_id=market_id)
            self._books[token_id] = state
        elif market_id and not state.market_id:
            state.market_id = market_id
        return state

    def apply_book(self, data: dict[str, Any]) -> None:
        token_id = str(data.get("asset_id") or "")
        if not token_id:
            return
        market_id = str(data.get("market") or "")
        timestamp_ms = _parse_int(data.get("timestamp"))
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        for side, target in (("bids", bids), ("asks", asks)):
            levels = data.get(side, [])
            if not isinstance(levels, list):
                continue
            for level in levels:
                if not isinstance(level, dict):
                    continue
                price = _parse_float(level.get("price"))
                size = _parse_float(level.get("size"))
                if price is None or size is None or size <= 0:
                    continue
                target[price] = size
        with self._lock:
            state = self._state_for(token_id, market_id)
            state.bids = bids
            state.asks = asks
            state.timestamp_ms = timestamp_ms
            state.tick_size = float(data.get("tick_size") or state.tick_size or 0.01)
            state.min_order_size = float(data.get("min_order_size") or state.min_order_size or 0.0)
            state.neg_risk = bool(data.get("neg_risk", state.neg_risk))
            state.last_trade_price = _parse_float(data.get("last_trade_price")) or state.last_trade_price
            state.snapshot_ready = True
            state.updated_monotonic = monotonic()
            state.recompute_tops()
            if market_id:
                self._dirty_market_ids.add(market_id)

    def apply_price_change(self, data: dict[str, Any]) -> None:
        timestamp_ms = _parse_int(data.get("timestamp"))
        market_id = str(data.get("market") or "")
        changes = data.get("price_changes", [])
        if not isinstance(changes, list):
            return
        with self._lock:
            for change in changes:
                if not isinstance(change, dict):
                    continue
                token_id = str(change.get("asset_id") or "")
                price = _parse_float(change.get("price"))
                size = _parse_float(change.get("size"))
                side = str(change.get("side") or "").upper()
                if not token_id or price is None or size is None:
                    continue
                state = self._state_for(token_id, market_id)
                levels = state.bids if side == "BUY" else state.asks if side == "SELL" else None
                if levels is None:
                    continue
                if size <= 0:
                    levels.pop(price, None)
                else:
                    levels[price] = size
                if side == "BUY":
                    if (
                        state.best_bid_price is None
                        or price > state.best_bid_price
                        or price == state.best_bid_price
                        or size <= 0
                    ):
                        state.best_bid_price, state.best_bid_size = state._best_level(state.bids, reverse=True)
                else:
                    if (
                        state.best_ask_price is None
                        or price < state.best_ask_price
                        or price == state.best_ask_price
                        or size <= 0
                    ):
                        state.best_ask_price, state.best_ask_size = state._best_level(state.asks, reverse=False)
                state.timestamp_ms = timestamp_ms or state.timestamp_ms
                state.updated_monotonic = monotonic()
                if market_id:
                    self._dirty_market_ids.add(market_id)

    def apply_last_trade(self, data: dict[str, Any]) -> None:
        token_id = str(data.get("asset_id") or "")
        if not token_id:
            return
        market_id = str(data.get("market") or "")
        price = _parse_float(data.get("price"))
        timestamp_ms = _parse_int(data.get("timestamp"))
        with self._lock:
            state = self._state_for(token_id, market_id)
            state.last_trade_price = price
            state.timestamp_ms = timestamp_ms or state.timestamp_ms
            state.updated_monotonic = monotonic()
            if market_id:
                self._dirty_market_ids.add(market_id)

    def apply_best_bid_ask(self, data: dict[str, Any]) -> None:
        token_id = str(data.get("asset_id") or "")
        if not token_id:
            return
        market_id = str(data.get("market") or "")
        timestamp_ms = _parse_int(data.get("timestamp"))
        with self._lock:
            state = self._state_for(token_id, market_id)
            state.timestamp_ms = timestamp_ms or state.timestamp_ms
            state.updated_monotonic = monotonic()
            if market_id:
                self._dirty_market_ids.add(market_id)

    def reconcile_order_book(self, book: OrderBook) -> bool:
        """Replace one cached book with an HTTP snapshot and return whether top-of-book changed."""
        with self._lock:
            state = self._state_for(book.token_id, book.market_id)
            before_top = (
                state.best_bid_price,
                state.best_bid_size,
                state.best_ask_price,
                state.best_ask_size,
            )
            state.bids = {level.price: level.size for level in book.bids if level.size > 0}
            state.asks = {level.price: level.size for level in book.asks if level.size > 0}
            state.timestamp_ms = book.timestamp_ms
            state.tick_size = book.tick_size
            state.min_order_size = book.min_order_size
            state.neg_risk = book.neg_risk
            state.last_trade_price = book.last_trade_price
            state.snapshot_ready = True
            state.updated_monotonic = monotonic()
            state.recompute_tops()
            after_top = (
                state.best_bid_price,
                state.best_bid_size,
                state.best_ask_price,
                state.best_ask_size,
            )
            top_changed = before_top != after_top
            if top_changed and book.market_id:
                self._dirty_market_ids.add(book.market_id)
            return top_changed

    def get_market_books(
        self,
        markets: list[MarketInfo],
        connection_stale_seconds: float,
        top_only: bool = True,
    ) -> tuple[dict[str, dict[str, OrderBook]], dict[str, Exception]]:
        books_by_market: dict[str, dict[str, OrderBook]] = {}
        errors: dict[str, Exception] = {}
        healthy = self.is_healthy(connection_stale_seconds)
        now_ms = int(__import__("time").time() * 1000)
        with self._lock:
            for market in markets:
                market_books: dict[str, OrderBook] = {}
                for side, token_id in (("yes", market.yes_token_id), ("no", market.no_token_id)):
                    state = self._books.get(token_id)
                    if state is None or not state.snapshot_ready:
                        errors[market.market_id] = RuntimeError(f"missing cached book for {token_id}")
                        break
                    timestamp_ms = now_ms if healthy else state.timestamp_ms
                    market_books[side] = state.to_order_book(timestamp_ms=timestamp_ms, top_only=top_only)
                if len(market_books) == 2:
                    books_by_market[market.market_id] = market_books
        return books_by_market, errors


class PolymarketBookStream:
    """Background WebSocket client that feeds a PolymarketBookCache."""

    def __init__(self, cfg: PolySettings | None = None, cache: PolymarketBookCache | None = None):
        self.cfg = cfg or settings
        self.cache = cache or PolymarketBookCache()
        self._asset_ids: list[str] = []
        self._asset_lock = RLock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._ws = None

    def update_assets(self, asset_ids: list[str]) -> None:
        deduped = list(dict.fromkeys(str(asset_id) for asset_id in asset_ids if asset_id))
        with self._asset_lock:
            if deduped == self._asset_ids:
                return
            self._asset_ids = deduped
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def start(self, asset_ids: list[str] | None = None) -> None:
        if asset_ids is not None:
            self.update_assets(asset_ids)
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="polymarket-book-ws", daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _asset_snapshot(self) -> list[str]:
        with self._asset_lock:
            return list(self._asset_ids)

    def _send_subscription(self, ws, asset_ids: list[str]) -> None:
        batch_size = max(1, min(self.cfg.ws_subscribe_batch_size, 500))
        for idx, start in enumerate(range(0, len(asset_ids), batch_size)):
            chunk = asset_ids[start : start + batch_size]
            if idx == 0:
                payload = {
                    "assets_ids": chunk,
                    "type": "market",
                    "custom_feature_enabled": True,
                }
            else:
                payload = {
                    "assets_ids": chunk,
                    "operation": "subscribe",
                    "custom_feature_enabled": True,
                }
            ws.send(json.dumps(payload))

    def _run(self) -> None:  # pragma: no cover - integration tested with live smoke checks
        try:
            import websocket
        except Exception as exc:
            self.cache.set_error(exc)
            logger.warning(f"polymarket websocket unavailable: {exc}")
            return

        reconnect_sleep = max(self.cfg.ws_reconnect_min_seconds, 0.1)
        while not self._stop.is_set():
            asset_ids = self._asset_snapshot()
            if not asset_ids:
                sleep(0.5)
                continue
            try:
                ws = websocket.create_connection(
                    self.cfg.ws_market_url,
                    timeout=self.cfg.http_timeout_seconds,
                    header=[f"User-Agent: {self.cfg.user_agent}"],
                )
                ws.settimeout(1.0)
                self._ws = ws
                self.cache.set_connected(True)
                self.cache.set_error(None)
                self._send_subscription(ws, asset_ids)
                logger.info(f"polymarket websocket subscribed: assets={len(asset_ids)}")
                reconnect_sleep = max(self.cfg.ws_reconnect_min_seconds, 0.1)
                last_ping = 0.0
                while not self._stop.is_set():
                    if self._asset_snapshot() != asset_ids:
                        break
                    now = monotonic()
                    if now - last_ping >= self.cfg.ws_heartbeat_seconds:
                        ws.send("PING")
                        last_ping = now
                    try:
                        message = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if message == "PONG":
                        self.cache.mark_message()
                        continue
                    if message == "PING":
                        ws.send("PONG")
                        self.cache.mark_message()
                        continue
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")
                    self.cache.apply_message_text(message)
            except Exception as exc:
                if not self._stop.is_set():
                    self.cache.set_error(exc)
                    logger.warning(f"polymarket websocket disconnected: {exc}")
            finally:
                self.cache.set_connected(False)
                ws = self._ws
                self._ws = None
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
            if not self._stop.is_set():
                sleep(reconnect_sleep)
                reconnect_sleep = min(
                    max(reconnect_sleep * 2, self.cfg.ws_reconnect_min_seconds),
                    self.cfg.ws_reconnect_max_seconds,
                )
