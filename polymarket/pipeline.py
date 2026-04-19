"""Orchestration pipeline for isolated Polymarket paper trading."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from polymarket.books import ClobClient
from polymarket.catalog import load_binary_markets
from polymarket.config import PolySettings, settings
from polymarket.paper.simulator import PaperSimulator
from polymarket.scanner.full_set import scan_market
from polymarket.storage import PolyStorage


@dataclass(slots=True)
class PipelineResult:
    markets_seen: int
    opportunities_found: int
    trades_simulated: int


class PolymarketPipeline:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.storage = PolyStorage(self.cfg)
        self.clob = ClobClient(self.cfg)
        self.simulator = PaperSimulator(storage=self.storage, cfg=self.cfg)

    def run_once(self) -> PipelineResult:
        markets = load_binary_markets(self.cfg)
        self.storage.save_catalog_snapshot(markets)
        opportunities_found = 0
        trades_simulated = 0

        for market in markets:
            try:
                yes_book = self.clob.fetch_book(market.yes_token_id)
                no_book = self.clob.fetch_book(market.no_token_id)
                self.storage.save_book_snapshot(market, yes_book, no_book)
                opportunities = scan_market(market, yes_book, no_book, cfg=self.cfg)
                opportunities_found += self.storage.save_opportunities(opportunities)
                trades_simulated += self.simulator.consume(market, opportunities)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"polymarket pipeline skipped {market.market_id}: {exc}")

        return PipelineResult(
            markets_seen=len(markets),
            opportunities_found=opportunities_found,
            trades_simulated=trades_simulated,
        )
