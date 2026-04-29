"""Market catalog ingestion for Polymarket."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from time import sleep
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo


class GammaClient:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.cfg.gamma_base_url.rstrip('/')}/{path.lstrip('/')}" + query
        request = Request(url, headers={"User-Agent": self.cfg.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.cfg.http_timeout_seconds) as response:
            return json.load(response)

    def fetch_markets_page(self, limit: int, offset: int = 0) -> list[dict[str, Any]]:
        params = {"active": "true", "closed": "false", "limit": limit, "offset": offset}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                data = self._get_json("markets", params=params)
                return list(data)
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                if attempt == 2:
                    raise
                sleep(0.5 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        return []

    def fetch_markets(self, limit: int | None = None) -> list[dict[str, Any]]:
        total_limit = self.cfg.max_active_markets if limit is None else limit
        page_size = max(1, min(self.cfg.catalog_page_size, 1000))
        if total_limit > page_size and self.cfg.catalog_fetch_workers > 1:
            return self._fetch_markets_concurrent(total_limit=total_limit, page_size=page_size)
        return self._fetch_markets_sequential(total_limit=total_limit, page_size=page_size)

    def _fetch_markets_sequential(self, *, total_limit: int, page_size: int) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        offset = 0
        while total_limit <= 0 or len(markets) < total_limit:
            current_limit = page_size if total_limit <= 0 else min(page_size, total_limit - len(markets))
            if current_limit <= 0:
                break
            page = self.fetch_markets_page(limit=current_limit, offset=offset)
            if not page:
                break
            markets.extend(page)
            offset += len(page)
            if len(page) < current_limit:
                break
        return markets

    def _fetch_markets_concurrent(self, *, total_limit: int, page_size: int) -> list[dict[str, Any]]:
        requests: list[tuple[int, int]] = []
        offset = 0
        while offset < total_limit:
            current_limit = min(page_size, total_limit - offset)
            requests.append((offset, current_limit))
            offset += current_limit

        pages_by_offset: dict[int, list[dict[str, Any]]] = {}
        max_workers = max(1, min(int(self.cfg.catalog_fetch_workers), len(requests)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.fetch_markets_page, limit=current_limit, offset=offset): (offset, current_limit)
                for offset, current_limit in requests
            }
            for future in as_completed(futures):
                offset, _current_limit = futures[future]
                pages_by_offset[offset] = future.result()

        markets: list[dict[str, Any]] = []
        for offset, current_limit in requests:
            page = pages_by_offset.get(offset, [])
            if not page:
                break
            markets.extend(page)
            if len(page) < current_limit:
                break
        return markets[:total_limit]

    def fetch_fee_rate_bps(self, token_id: str) -> float | None:
        try:
            data = self._get_json("fee-rate", params={"asset_id": token_id})
        except Exception:
            return None
        for key in ("fee_rate_bps", "feeRateBps", "rate_bps", "rateBps"):
            value = data.get(key) if isinstance(data, dict) else None
            if value is not None:
                return float(value)
        return None


def _parse_json_list(raw_value: Any) -> list[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _token_pair(token_ids: Iterable[Any], outcomes: Iterable[Any]) -> tuple[str, str] | None:
    ordered = list(zip(token_ids, outcomes))
    yes_token = None
    no_token = None
    for token_id, outcome in ordered:
        name = str(outcome).strip().lower()
        if name == "yes":
            yes_token = str(token_id)
        elif name == "no":
            no_token = str(token_id)
    if yes_token and no_token:
        return yes_token, no_token
    return None


def normalize_market(raw_market: dict[str, Any], fee_rate_bps: float | None = None) -> MarketInfo | None:
    if not raw_market.get("active") or raw_market.get("closed"):
        return None
    if raw_market.get("negRisk") or raw_market.get("neg_risk"):
        return None
    enable_order_book = raw_market.get("enableOrderBook")
    if enable_order_book is None:
        enable_order_book = raw_market.get("enable_order_book")
    if enable_order_book is False:
        return None

    token_ids = _parse_json_list(raw_market.get("clobTokenIds"))
    outcomes = _parse_json_list(raw_market.get("outcomes"))
    if len(token_ids) != 2 or len(outcomes) != 2:
        return None

    pair = _token_pair(token_ids, outcomes)
    if pair is None:
        return None

    min_order_size = raw_market.get("minimumOrderSize")
    tick_size = raw_market.get("minimumTickSize")
    taker_fee_bps = fee_rate_bps
    if taker_fee_bps is None:
        taker_fee_bps = raw_market.get("takerBaseFee")
    if taker_fee_bps is None:
        taker_fee_bps = raw_market.get("taker_base_fee")
    events = raw_market.get("events") or []
    slug = raw_market.get("marketSlug") or raw_market.get("market_slug")
    if slug is None and events:
        slug = events[0].get("slug")

    return MarketInfo(
        market_id=str(raw_market.get("conditionId") or raw_market.get("condition_id") or raw_market.get("id")),
        condition_id=str(raw_market.get("conditionId") or raw_market.get("condition_id") or raw_market.get("id")),
        question=str(raw_market.get("question") or ""),
        slug=slug,
        end_date_iso=raw_market.get("endDateIso") or raw_market.get("end_date_iso"),
        min_order_size=float(min_order_size or 0),
        tick_size=float(tick_size or 0.01),
        neg_risk=bool(raw_market.get("negRisk") or raw_market.get("neg_risk")),
        enable_order_book=bool(raw_market.get("enableOrderBook", raw_market.get("enable_order_book", True))),
        taker_base_fee_bps=float(taker_fee_bps or 0),
        yes_token_id=pair[0],
        no_token_id=pair[1],
    )


def load_binary_markets(cfg: PolySettings | None = None) -> list[MarketInfo]:
    cfg = cfg or settings
    client = GammaClient(cfg=cfg)
    normalized: list[MarketInfo] = []
    fee_rate_cache: dict[str, float | None] = {}
    seen_market_ids: set[str] = set()
    for raw_market in client.fetch_markets():
        fee_rate_bps = raw_market.get("takerBaseFee")
        if fee_rate_bps is None:
            fee_rate_bps = raw_market.get("taker_base_fee")
        market = normalize_market(raw_market, fee_rate_bps=float(fee_rate_bps) if fee_rate_bps is not None else None)
        if market is not None:
            if market.market_id in seen_market_ids:
                continue
            seen_market_ids.add(market.market_id)
            if fee_rate_bps is None and cfg.catalog_fetch_fee_rates:
                token_id = market.yes_token_id
                if token_id not in fee_rate_cache:
                    fee_rate_cache[token_id] = client.fetch_fee_rate_bps(token_id)
                fetched_fee = fee_rate_cache[token_id]
                market.taker_base_fee_bps = float(fetched_fee or 0)
            normalized.append(market)
    return normalized
