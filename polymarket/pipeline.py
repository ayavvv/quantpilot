"""Orchestration pipeline for isolated Polymarket paper trading."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from polymarket.books import ClobClient
from polymarket.catalog import load_binary_markets
from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity
from polymarket.paper.simulator import PaperSimulator
from polymarket.scanner.full_set import scan_market
from polymarket.storage import PolyStorage
from polymarket.traders.discovery import LeaderboardClient, normalize_leaderboard_profiles
from polymarket.traders.history import TraderHistoryClient, normalize_activity_events, summarize_trader_history
from polymarket.traders.mirror import generate_mirror_signals
from polymarket.traders.paper import MirrorBookkeeper
from polymarket.traders.profile import PublicProfileClient
from polymarket.traders.ranking import compute_trader_scores
from polymarket.traders.storage import MirrorStorage


@dataclass(slots=True)
class PipelineResult:
    markets_seen: int
    opportunities_found: int
    trades_simulated: int
    mirror_traders_tracked: int = 0
    mirror_signals_generated: int = 0


class PolymarketPipeline:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.storage = PolyStorage(self.cfg)
        self.mirror_storage = MirrorStorage(self.cfg)
        self.clob = ClobClient(self.cfg)
        self.simulator = PaperSimulator(storage=self.storage, cfg=self.cfg)
        self.leaderboard = LeaderboardClient(self.cfg)
        self.profile_client = PublicProfileClient(self.cfg)
        self.history_client = TraderHistoryClient(self.cfg)
        self.mirror_bookkeeper = MirrorBookkeeper(storage=self.mirror_storage, initial_cash=self.cfg.paper_initial_cash)

    def _run_full_set_strategy(self, markets: list[MarketInfo]) -> tuple[int, int]:
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
        return opportunities_found, trades_simulated

    def _build_market_map(self, markets: list[MarketInfo]) -> dict[str, MarketInfo]:
        mapping: dict[str, MarketInfo] = {}
        for market in markets:
            mapping[market.market_id] = market
        return mapping

    def _run_top_trader_mirror(self, market_map: dict[str, MarketInfo]) -> tuple[int, int]:
        leaderboard_rows = self.leaderboard.fetch_leaderboard(self.cfg.top_trader_candidate_limit)
        base_profiles = normalize_leaderboard_profiles(leaderboard_rows)
        profiles = []
        history_stats: dict[str, dict] = {}
        all_events = []
        for profile in base_profiles:
            try:
                enriched = self.profile_client.fetch_public_profile(profile.wallet)
            except Exception:
                enriched = profile
            profiles.append(enriched)
            try:
                positions = self.history_client.fetch_positions(enriched.wallet, limit=50)
                closed_positions = self.history_client.fetch_closed_positions(enriched.wallet, limit=50)
                trades = self.history_client.fetch_trades(enriched.wallet, limit=50)
                activity = self.history_client.fetch_activity(enriched.wallet, limit=50)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"polymarket mirror skipped trader {enriched.wallet}: {exc}")
                continue
            history_stats[enriched.wallet] = summarize_trader_history(positions, closed_positions, trades)
            all_events.extend(normalize_activity_events(enriched.wallet, activity))

        scores = compute_trader_scores(leaderboard_rows, history_stats, cfg=self.cfg)
        tracked_wallets = {score.wallet for score in scores}
        tracked_profiles = [profile for profile in profiles if profile.wallet in tracked_wallets]
        tracked_events = [event for event in all_events if event.wallet in tracked_wallets]
        mirror_signals = generate_mirror_signals(tracked_events, cfg=self.cfg)

        self.mirror_storage.upsert_trader_profiles(tracked_profiles)
        self.mirror_storage.upsert_trader_scores(scores)
        self.mirror_storage.save_trader_events(tracked_events)
        inserted_signals = self.mirror_storage.save_mirror_signals(mirror_signals)

        simulated = 0
        for signal in mirror_signals:
            market = market_map.get(signal.market_id)
            if market is None:
                continue
            execution_price = signal.source_price or 0.5
            executed = self.mirror_bookkeeper.apply_signal(
                signal=signal,
                market_id=market.market_id,
                title=signal.title,
                outcome=signal.outcome,
                execution_price=execution_price,
                fee_bps=market.taker_base_fee_bps,
            )
            simulated += 1 if executed else 0

        self.mirror_storage.upsert_daily_summary(
            {
                "date": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).date().isoformat(),
                "tracked_traders": len(scores),
                "signals": inserted_signals,
                "accepted_signals": simulated,
                "simulated_trades": simulated,
                "realized_pnl": self.mirror_bookkeeper.state.realized_pnl,
                "max_inventory_used": self.cfg.top_trader_max_signal_notional,
                "updated_at": __import__('datetime').datetime.now(__import__('datetime').timezone.utc),
            }
        )
        return len(scores), len(mirror_signals)

    def run_once(self) -> PipelineResult:
        markets = load_binary_markets(self.cfg)
        self.storage.save_catalog_snapshot(markets)
        opportunities_found, trades_simulated = self._run_full_set_strategy(markets)
        mirror_traders_tracked = 0
        mirror_signals_generated = 0
        if self.cfg.enable_top_trader_mirror:
            mirror_traders_tracked, mirror_signals_generated = self._run_top_trader_mirror(self._build_market_map(markets))

        return PipelineResult(
            markets_seen=len(markets),
            opportunities_found=opportunities_found,
            trades_simulated=trades_simulated,
            mirror_traders_tracked=mirror_traders_tracked,
            mirror_signals_generated=mirror_signals_generated,
        )
