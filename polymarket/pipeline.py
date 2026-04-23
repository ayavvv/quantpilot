"""Orchestration pipeline for isolated Polymarket paper trading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from loguru import logger

from polymarket.books import ClobClient
from polymarket.catalog import load_binary_markets
from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook
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
    stage_timings: dict[str, float] | None = None


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
        self._markets_cache: list[MarketInfo] | None = None
        self._markets_refreshed_at: datetime | None = None

    def _get_markets(self) -> tuple[list[MarketInfo], bool, float]:
        started = perf_counter()
        now = datetime.now(timezone.utc)
        expires_at = None
        if self._markets_refreshed_at is not None:
            expires_at = self._markets_refreshed_at + timedelta(seconds=self.cfg.catalog_refresh_seconds)
        if self._markets_cache is not None and expires_at is not None and now < expires_at:
            return self._markets_cache, False, perf_counter() - started
        if self._markets_cache is None:
            persisted, persisted_updated_at = self.storage.load_markets()
            if persisted and persisted_updated_at is not None:
                persisted_age = (now - persisted_updated_at).total_seconds()
                if persisted_age <= self.cfg.catalog_refresh_seconds:
                    self._markets_cache = persisted
                    self._markets_refreshed_at = persisted_updated_at
                    logger.info("polymarket catalog loaded from persisted snapshot")
                    return persisted, False, perf_counter() - started
        try:
            markets = load_binary_markets(self.cfg)
            self._markets_cache = markets
            self._markets_refreshed_at = now
            return markets, True, perf_counter() - started
        except Exception as exc:
            if self._markets_cache is not None:
                logger.warning(f"polymarket catalog refresh failed, using in-memory cache: {exc}")
                return self._markets_cache, False, perf_counter() - started
            raise

    def _fetch_market_books(self, markets: list[MarketInfo]) -> tuple[dict[str, dict[str, OrderBook]], dict[str, Exception]]:
        books_by_market: dict[str, dict[str, OrderBook]] = {}
        errors: dict[str, Exception] = {}
        total_requests = max(1, len(markets) * 2)
        max_workers = max(1, min(self.cfg.book_fetch_workers, total_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for market in markets:
                futures[executor.submit(self.clob.fetch_book, market.yes_token_id)] = (market.market_id, "yes")
                futures[executor.submit(self.clob.fetch_book, market.no_token_id)] = (market.market_id, "no")
            for future in as_completed(futures):
                market_id, side = futures[future]
                try:
                    books_by_market.setdefault(market_id, {})[side] = future.result()
                except Exception as exc:  # pragma: no cover
                    errors[market_id] = exc
        return books_by_market, errors

    def _run_full_set_strategy(self, markets: list[MarketInfo]) -> tuple[int, int, dict[str, float]]:
        opportunities_found = 0
        trades_simulated = 0
        scanned_markets = 0
        stage_timings = {"book_fetch_seconds": 0.0, "scan_compute_seconds": 0.0, "storage_write_seconds": 0.0}
        fetch_started = perf_counter()
        books_by_market, errors = self._fetch_market_books(markets)
        stage_timings["book_fetch_seconds"] = perf_counter() - fetch_started
        top_rows: list[dict[str, object]] = []
        now = datetime.now(timezone.utc)
        for market in markets:
            if market.market_id in errors:
                logger.warning(f"polymarket pipeline skipped {market.market_id}: {errors[market.market_id]}")
                continue
            market_books = books_by_market.get(market.market_id, {})
            yes_book = market_books.get("yes")
            no_book = market_books.get("no")
            if yes_book is None or no_book is None:
                logger.warning(f"polymarket pipeline skipped {market.market_id}: missing paired books")
                continue
            top_rows.extend(
                [
                    {
                        "ts": now,
                        "market_id": market.market_id,
                        "token_id": book.token_id,
                        "best_bid": book.best_bid.price if book.best_bid else None,
                        "best_bid_size": book.best_bid.size if book.best_bid else None,
                        "best_ask": book.best_ask.price if book.best_ask else None,
                        "best_ask_size": book.best_ask.size if book.best_ask else None,
                        "last_trade": book.last_trade_price,
                        "book_timestamp_ms": book.timestamp_ms,
                    }
                    for book in (yes_book, no_book)
                ]
            )
            scanned_markets += 1
            try:
                scan_started = perf_counter()
                opportunities = scan_market(market, yes_book, no_book, cfg=self.cfg)
                stage_timings["scan_compute_seconds"] += perf_counter() - scan_started
                if opportunities:
                    write_started = perf_counter()
                    self.storage.save_book_snapshot(market, yes_book, no_book, persist_depth=True, persist_top=False)
                    opportunities_found += self.storage.save_opportunities(opportunities)
                    trades_simulated += self.simulator.consume(market, opportunities)
                    stage_timings["storage_write_seconds"] += perf_counter() - write_started
            except Exception as exc:  # pragma: no cover
                logger.warning(f"polymarket pipeline skipped {market.market_id}: {exc}")
        write_started = perf_counter()
        try:
            self.storage.save_book_tops(top_rows)
            if opportunities_found == 0 and scanned_markets > 0:
                self.simulator.record_scan_heartbeat()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"polymarket pipeline storage stage degraded: {exc}")
        stage_timings["storage_write_seconds"] += perf_counter() - write_started
        logger.info(f"polymarket book fetch stage complete: markets={scanned_markets} duration_seconds={stage_timings['book_fetch_seconds']:.2f}")
        return opportunities_found, trades_simulated, stage_timings

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
        markets, refreshed, catalog_load_seconds = self._get_markets()
        catalog_snapshot_seconds = 0.0
        if refreshed:
            started = perf_counter()
            self.storage.save_catalog_snapshot(markets)
            catalog_snapshot_seconds = perf_counter() - started
        opportunities_found, trades_simulated, stage_timings = self._run_full_set_strategy(markets)
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
            stage_timings={
                **stage_timings,
                "catalog_load_seconds": catalog_load_seconds,
                "catalog_snapshot_seconds": catalog_snapshot_seconds,
            },
        )
