"""Orchestration pipeline for isolated Polymarket paper trading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from time import monotonic, perf_counter

from loguru import logger

from polymarket.async_storage import AsyncScanStorageWriter
from polymarket.books import ClobClient
from polymarket.catalog import load_binary_markets
from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook
from polymarket.paper.simulator import PaperSimulator
from polymarket.scanner.full_set import rejection_reason, scan_market
from polymarket.storage import PolyStorage
from polymarket.traders.discovery import LeaderboardClient, normalize_leaderboard_profiles
from polymarket.traders.history import TraderHistoryClient, normalize_activity_events, summarize_trader_history
from polymarket.traders.mirror import generate_mirror_signals
from polymarket.traders.paper import MirrorBookkeeper
from polymarket.traders.profile import PublicProfileClient
from polymarket.traders.ranking import compute_trader_scores
from polymarket.traders.storage import MirrorStorage
from polymarket.ws_books import PolymarketBookCache, PolymarketBookStream


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
        self.book_cache = PolymarketBookCache()
        self.book_stream: PolymarketBookStream | None = None
        self._ws_asset_ids: list[str] = []
        self._scan_lock = Lock()
        self._storage_write_lock = Lock()
        self._reconcile_lock = Lock()
        self._full_scan_requested = Event()
        self._reconcile_cursor = 0
        self._dirty_stop = Event()
        self._dirty_thread: Thread | None = None
        self._reconcile_stop = Event()
        self._reconcile_thread: Thread | None = None
        self._last_book_top_sample_monotonic = 0.0
        self.async_writer: AsyncScanStorageWriter | None = None
        if self.cfg.storage_async_flush_enabled:
            self.async_writer = AsyncScanStorageWriter(
                storage=self.storage,
                simulator=self.simulator,
                storage_write_lock=self._storage_write_lock,
                cfg=self.cfg,
            )
            self.async_writer.start()
        self._markets_cache: list[MarketInfo] | None = None
        self._markets_refreshed_at: datetime | None = None

    def _load_scan_markets(self) -> tuple[list[MarketInfo], float]:
        started = perf_counter()
        if self._markets_cache is not None:
            return self._markets_cache, perf_counter() - started
        persisted, persisted_updated_at = self.storage.load_markets()
        if persisted:
            self._markets_cache = persisted
            self._markets_refreshed_at = persisted_updated_at
            logger.info("polymarket scan loaded persisted snapshot")
            return persisted, perf_counter() - started
        raise RuntimeError("no persisted polymarket catalog snapshot available for fast scan")

    def refresh_catalog(self) -> tuple[int, float, float]:
        load_started = perf_counter()
        markets = load_binary_markets(self.cfg)
        catalog_load_seconds = perf_counter() - load_started
        if not markets:
            raise RuntimeError("live catalog refresh returned no eligible markets")
        snapshot_started = perf_counter()
        self.storage.save_catalog_snapshot(markets, source="live_refresh")
        catalog_snapshot_seconds = perf_counter() - snapshot_started
        self._markets_cache = markets
        self._markets_refreshed_at = datetime.now(timezone.utc)
        self._ensure_ws_stream(markets)
        return len(markets), catalog_load_seconds, catalog_snapshot_seconds

    def _book_source(self) -> str:
        return self.cfg.book_source.strip().lower()

    def _market_token_ids(self, markets: list[MarketInfo]) -> list[str]:
        token_ids: list[str] = []
        for market in markets:
            token_ids.append(market.yes_token_id)
            token_ids.append(market.no_token_id)
        return list(dict.fromkeys(token_ids))

    def _ensure_ws_stream(self, markets: list[MarketInfo], wait_ready: bool = True) -> None:
        if self._book_source() != "ws":
            return
        token_ids = self._market_token_ids(markets)
        if self.book_stream is None:
            self.book_stream = PolymarketBookStream(self.cfg, cache=self.book_cache)
        if token_ids == self._ws_asset_ids and self.book_stream.is_running():
            return
        self._ws_asset_ids = token_ids
        self.book_stream.update_assets(token_ids)
        self.book_stream.start()
        if not wait_ready:
            return
        if self.book_cache.ready_count(token_ids) >= max(1, int(len(token_ids) * self.cfg.ws_min_ready_ratio)):
            return
        ready = self.book_cache.wait_until_ready(
            token_ids,
            timeout_seconds=self.cfg.ws_ready_timeout_seconds,
            min_ready_ratio=self.cfg.ws_min_ready_ratio,
        )
        stats = self.book_cache.stats(token_ids)
        if ready:
            logger.info(
                f"polymarket websocket cache ready: ready_tokens={stats['ready_tokens']} "
                f"total_tokens={stats['total_tokens']}"
            )
        else:
            logger.warning(
                f"polymarket websocket cache warmup incomplete: ready_tokens={stats['ready_tokens']} "
                f"total_tokens={stats['total_tokens']} last_error={stats['last_error']}"
            )

    def _fetch_market_books(
        self,
        markets: list[MarketInfo],
        wait_ready: bool = True,
        ensure_markets: list[MarketInfo] | None = None,
    ) -> tuple[dict[str, dict[str, OrderBook]], dict[str, Exception]]:
        if self._book_source() == "ws":
            self._ensure_ws_stream(ensure_markets or markets, wait_ready=wait_ready)
            connection_stale_seconds = max(
                self.cfg.ws_heartbeat_seconds * 3,
                self.cfg.max_book_staleness_ms / 1000.0,
            )
            return self.book_cache.get_market_books(
                markets,
                connection_stale_seconds=connection_stale_seconds,
                top_only=True,
            )

        books_by_market: dict[str, dict[str, OrderBook]] = {}
        errors: dict[str, Exception] = {}
        token_map: dict[str, tuple[str, str]] = {}
        for market in markets:
            token_map[market.yes_token_id] = (market.market_id, "yes")
            token_map[market.no_token_id] = (market.market_id, "no")

        if self.cfg.book_fetch_use_batch:
            batch_size = max(1, min(self.cfg.book_fetch_batch_size, 500))
            token_ids = list(token_map)
            try:
                for start in range(0, len(token_ids), batch_size):
                    chunk = token_ids[start : start + batch_size]
                    for token_id, book in self.clob.fetch_books(chunk).items():
                        market_id, side = token_map.get(token_id, ("", ""))
                        if market_id:
                            books_by_market.setdefault(market_id, {})[side] = book
            except Exception as exc:
                logger.warning(f"polymarket batch book fetch failed, falling back to single fetch: {exc}")
                books_by_market.clear()

        missing_token_map = {
            token_id: market_side
            for token_id, market_side in token_map.items()
            if market_side[1] not in books_by_market.get(market_side[0], {})
        }
        if not missing_token_map:
            return books_by_market, errors

        total_requests = max(1, len(missing_token_map))
        max_workers = max(1, min(self.cfg.book_fetch_workers, total_requests))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for token_id, (market_id, side) in missing_token_map.items():
                futures[executor.submit(self.clob.fetch_book, token_id)] = (market_id, side)
            for future in as_completed(futures):
                market_id, side = futures[future]
                try:
                    books_by_market.setdefault(market_id, {})[side] = future.result()
                except Exception as exc:  # pragma: no cover
                    errors[market_id] = exc
        return books_by_market, errors

    def _run_full_set_strategy(
        self,
        markets: list[MarketInfo],
        *,
        persist_scan_artifacts: bool = True,
        log_stage: bool = True,
        log_skips: bool = True,
        wait_ready: bool = True,
        ensure_markets: list[MarketInfo] | None = None,
    ) -> tuple[int, int, dict[str, float]]:
        opportunities_found = 0
        trades_simulated = 0
        scanned_markets = 0
        stage_timings = {"book_fetch_seconds": 0.0, "scan_compute_seconds": 0.0, "storage_write_seconds": 0.0}
        fetch_started = perf_counter()
        books_by_market, errors = self._fetch_market_books(
            markets,
            wait_ready=wait_ready,
            ensure_markets=ensure_markets,
        )
        stage_timings["book_fetch_seconds"] = perf_counter() - fetch_started
        top_rows: list[dict[str, object]] = []
        rejection_counts: dict[str, int] = {}
        skip_counts: dict[str, int] = {}
        now = datetime.now(timezone.utc)

        def record_skip(reason: str) -> None:
            if not (log_skips or log_stage):
                return
            skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1

        for market in markets:
            if market.market_id in errors:
                error_text = str(errors[market.market_id])
                if "missing cached book" in error_text:
                    record_skip("missing_cached_book")
                else:
                    record_skip(f"book_error:{type(errors[market.market_id]).__name__}")
                continue
            market_books = books_by_market.get(market.market_id, {})
            yes_book = market_books.get("yes")
            no_book = market_books.get("no")
            if yes_book is None or no_book is None:
                record_skip("missing_paired_books")
                continue
            if persist_scan_artifacts:
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
                reason = rejection_reason(market, yes_book, no_book, cfg=self.cfg)
                opportunities = [] if reason is not None else scan_market(market, yes_book, no_book, cfg=self.cfg)
                stage_timings["scan_compute_seconds"] += perf_counter() - scan_started
                if reason is not None and persist_scan_artifacts:
                    rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
                if opportunities:
                    write_started = perf_counter()
                    with self._storage_write_lock:
                        self.storage.save_book_snapshot(market, yes_book, no_book, persist_depth=True, persist_top=False)
                        opportunities_found += self.storage.save_opportunities(opportunities)
                        trades_simulated += self.simulator.consume(market, opportunities)
                    stage_timings["storage_write_seconds"] += perf_counter() - write_started
            except Exception as exc:  # pragma: no cover
                record_skip(f"scan_error:{type(exc).__name__}")
        write_started = perf_counter()
        try:
            if persist_scan_artifacts:
                if self.async_writer is not None:
                    self.async_writer.enqueue_book_tops(top_rows)
                    if scanned_markets > 0:
                        self.async_writer.enqueue_heartbeat(rejection_counts=rejection_counts)
                else:
                    with self._storage_write_lock:
                        self.storage.save_book_tops(top_rows)
                        if scanned_markets > 0:
                            self.simulator.record_scan_heartbeat(rejection_counts=rejection_counts)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"polymarket pipeline storage stage degraded: {exc}")
        stage_timings["storage_write_seconds"] += perf_counter() - write_started
        stage_timings["scanned_markets"] = float(scanned_markets)
        cache_stats = self.book_cache.stats(self._market_token_ids(markets)) if self._book_source() == "ws" else {}
        if log_stage:
            skipped_markets = sum(skip_counts.values())
            skip_reasons = ",".join(f"{reason}:{count}" for reason, count in sorted(skip_counts.items()))
            logger.info(
                f"polymarket book fetch stage complete: markets={scanned_markets} "
                f"skipped_markets={skipped_markets} skip_reasons={skip_reasons or 'none'} "
                f"source={self._book_source()} duration_seconds={stage_timings['book_fetch_seconds']:.2f} "
                f"cache_ready_tokens={cache_stats.get('ready_tokens', 0)} "
                f"cache_total_tokens={cache_stats.get('total_tokens', 0)}"
            )
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

    def _should_persist_scan_artifacts(self) -> bool:
        sample_seconds = max(float(self.cfg.book_top_sample_seconds), 0.0)
        if sample_seconds <= 0:
            return True
        now = monotonic()
        if self._last_book_top_sample_monotonic <= 0:
            self._last_book_top_sample_monotonic = now
            return True
        if now - self._last_book_top_sample_monotonic >= sample_seconds:
            self._last_book_top_sample_monotonic = now
            return True
        return False

    def run_dirty_once(self) -> PipelineResult:
        if self._book_source() != "ws" or not self.cfg.dirty_scan_enabled:
            return PipelineResult(0, 0, 0, stage_timings={"dirty_scan_disabled": 1.0})
        if self._full_scan_requested.is_set():
            return PipelineResult(0, 0, 0, stage_timings={"full_scan_pending": 1.0})
        if not self._scan_lock.acquire(blocking=False):
            return PipelineResult(0, 0, 0, stage_timings={"scan_lock_busy": 1.0})
        try:
            dirty_market_ids = self.book_cache.pop_dirty_market_ids()
            if not dirty_market_ids:
                return PipelineResult(0, 0, 0, stage_timings={"dirty_markets": 0.0})
            markets, catalog_load_seconds = self._load_scan_markets()
            self._ensure_ws_stream(markets, wait_ready=False)
            market_map = self._build_market_map(markets)
            dirty_markets = [market_map[market_id] for market_id in dirty_market_ids if market_id in market_map]
            if not dirty_markets:
                return PipelineResult(
                    0,
                    0,
                    0,
                    stage_timings={
                        "catalog_load_seconds": catalog_load_seconds,
                        "dirty_markets": float(len(dirty_market_ids)),
                    },
                )
            opportunities_found, trades_simulated, stage_timings = self._run_full_set_strategy(
                dirty_markets,
                persist_scan_artifacts=False,
                log_stage=False,
                log_skips=False,
                wait_ready=False,
                ensure_markets=markets,
            )
            stage_timings["catalog_load_seconds"] = catalog_load_seconds
            stage_timings["dirty_markets"] = float(len(dirty_market_ids))
            return PipelineResult(
                markets_seen=int(stage_timings.get("scanned_markets", len(dirty_markets))),
                opportunities_found=opportunities_found,
                trades_simulated=trades_simulated,
                stage_timings=stage_timings,
            )
        finally:
            self._scan_lock.release()

    def run_reconcile_once(self) -> dict[str, object]:
        if self._book_source() != "ws" or not self.cfg.ws_reconcile_enabled:
            return {"enabled": False}
        if not self._reconcile_lock.acquire(blocking=False):
            return {"enabled": True, "skipped": True, "reason": "already_running"}
        started = perf_counter()
        try:
            markets, catalog_load_seconds = self._load_scan_markets()
            self._ensure_ws_stream(markets, wait_ready=False)
            token_ids = self._market_token_ids(markets)
            total_tokens = len(token_ids)
            token_ids = self._reconcile_token_window(token_ids)
            batch_size = max(1, min(self.cfg.ws_reconcile_batch_size, 500))
            chunks = [token_ids[start : start + batch_size] for start in range(0, len(token_ids), batch_size)]
            max_workers = max(1, min(self.cfg.ws_reconcile_workers, len(chunks) or 1))
            fetched_tokens = 0
            top_drifted_tokens = 0
            failed_batches = 0
            failure_counts: dict[str, int] = {}
            timeout_seconds = max(float(self.cfg.ws_reconcile_timeout_seconds), 0.1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.clob.fetch_books,
                        chunk,
                        attempts=1,
                        timeout_seconds=timeout_seconds,
                    ): chunk
                    for chunk in chunks
                }
                for future in as_completed(futures):
                    try:
                        books = future.result()
                    except Exception as exc:  # pragma: no cover
                        failed_batches += 1
                        failure_key = f"{type(exc).__name__}:{str(exc).splitlines()[0][:120]}"
                        failure_counts[failure_key] = int(failure_counts.get(failure_key, 0)) + 1
                        continue
                    fetched_tokens += len(books)
                    for book in books.values():
                        while self._full_scan_requested.is_set():
                            if self._reconcile_stop.wait(0.005):
                                break
                        if self.book_cache.reconcile_order_book(book):
                            top_drifted_tokens += 1
            duration_seconds = perf_counter() - started
            log = logger.warning if failed_batches else logger.info
            failure_reasons = ",".join(f"{reason}:{count}" for reason, count in sorted(failure_counts.items()))
            log(
                f"polymarket websocket reconcile complete: tokens={fetched_tokens} "
                f"cycle_tokens={len(token_ids)} total_tokens={total_tokens} next_cursor={self._reconcile_cursor} "
                f"top_drifted_tokens={top_drifted_tokens} failed_batches={failed_batches} "
                f"failure_reasons={failure_reasons or 'none'} "
                f"duration_seconds={duration_seconds:.2f} "
                f"catalog_load_seconds={catalog_load_seconds:.2f}"
            )
            return {
                "enabled": True,
                "tokens": fetched_tokens,
                "cycle_tokens": len(token_ids),
                "total_tokens": total_tokens,
                "next_cursor": self._reconcile_cursor,
                "top_drifted_tokens": top_drifted_tokens,
                "failed_batches": failed_batches,
                "failure_reasons": failure_counts,
                "duration_seconds": duration_seconds,
                "catalog_load_seconds": catalog_load_seconds,
            }
        except Exception as exc:  # pragma: no cover
            duration_seconds = perf_counter() - started
            logger.warning(
                f"polymarket websocket reconcile failed: duration_seconds={duration_seconds:.2f} error={exc}"
            )
            return {"enabled": True, "error": str(exc), "duration_seconds": duration_seconds}
        finally:
            self._reconcile_lock.release()

    def _reconcile_token_window(self, token_ids: list[str]) -> list[str]:
        if not token_ids:
            self._reconcile_cursor = 0
            return []
        max_tokens = max(int(self.cfg.ws_reconcile_max_tokens_per_cycle), 0)
        if max_tokens <= 0 or max_tokens >= len(token_ids):
            self._reconcile_cursor = 0
            return token_ids
        start = self._reconcile_cursor % len(token_ids)
        end = start + max_tokens
        if end <= len(token_ids):
            selected = token_ids[start:end]
        else:
            selected = token_ids[start:] + token_ids[: end - len(token_ids)]
        self._reconcile_cursor = end % len(token_ids)
        return selected

    def start_background_workers(self) -> None:
        self.start_reconciler()
        self.start_dirty_scanner()

    def start_dirty_scanner(self) -> None:
        if self._book_source() != "ws" or not self.cfg.dirty_scan_enabled:
            return
        if self._dirty_thread is not None and self._dirty_thread.is_alive():
            return
        self._dirty_stop.clear()
        self._dirty_thread = Thread(target=self._run_dirty_scanner, name="polymarket-dirty-scan", daemon=True)
        self._dirty_thread.start()
        logger.info(
            f"polymarket dirty scanner started: interval_seconds={self.cfg.dirty_scan_interval_seconds}"
        )

    def start_reconciler(self) -> None:
        if self._book_source() != "ws" or not self.cfg.ws_reconcile_enabled:
            return
        if self._reconcile_thread is not None and self._reconcile_thread.is_alive():
            return
        self._reconcile_stop.clear()
        self._reconcile_thread = Thread(target=self._run_reconciler, name="polymarket-ws-reconcile", daemon=True)
        self._reconcile_thread.start()
        logger.info(
            f"polymarket websocket reconciler started: interval_seconds={self.cfg.ws_reconcile_seconds}"
        )

    def stop_background_workers(self) -> None:
        self._dirty_stop.set()
        self._reconcile_stop.set()
        if self._dirty_thread is not None:
            self._dirty_thread.join(timeout=2.0)
        if self._reconcile_thread is not None:
            self._reconcile_thread.join(timeout=2.0)

    def _run_dirty_scanner(self) -> None:  # pragma: no cover - exercised in production smoke checks
        interval = max(float(self.cfg.dirty_scan_interval_seconds), 0.01)
        last_log = monotonic()
        ticks = 0
        scanned = 0
        opportunities = 0
        trades = 0
        lock_skips = 0
        while not self._dirty_stop.wait(interval):
            ticks += 1
            started = perf_counter()
            try:
                result = self.run_dirty_once()
            except Exception as exc:
                logger.warning(f"polymarket dirty scan failed: {exc}")
                continue
            duration_seconds = perf_counter() - started
            stage_timings = result.stage_timings or {}
            if stage_timings.get("scan_lock_busy"):
                lock_skips += 1
            if stage_timings.get("full_scan_pending"):
                lock_skips += 1
            scanned += result.markets_seen
            opportunities += result.opportunities_found
            trades += result.trades_simulated
            if duration_seconds > max(interval * 2, 0.2):
                logger.warning(
                    f"polymarket dirty scan slow: markets={result.markets_seen} "
                    f"duration_seconds={duration_seconds:.3f}"
                )
            now = monotonic()
            if now - last_log >= 60:
                logger.info(
                    f"polymarket dirty scanner heartbeat: ticks={ticks} scanned_markets={scanned} "
                    f"opps={opportunities} trades={trades} lock_skips={lock_skips}"
                )
                last_log = now
                ticks = 0
                scanned = 0
                opportunities = 0
                trades = 0
                lock_skips = 0

    def _run_reconciler(self) -> None:  # pragma: no cover - exercised in production smoke checks
        interval = max(float(self.cfg.ws_reconcile_seconds), 1.0)
        while not self._reconcile_stop.is_set():
            self.run_reconcile_once()
            if self._reconcile_stop.wait(interval):
                break

    def run_once(self) -> PipelineResult:
        self._full_scan_requested.set()
        with self._scan_lock:
            try:
                markets, catalog_load_seconds = self._load_scan_markets()
                catalog_snapshot_seconds = 0.0
                opportunities_found, trades_simulated, stage_timings = self._run_full_set_strategy(
                    markets,
                    persist_scan_artifacts=self._should_persist_scan_artifacts(),
                )
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
            finally:
                self._full_scan_requested.clear()

    def flush_async_writes(self) -> tuple[int, dict[str, int]]:
        if self.async_writer is None:
            return 0, {}
        return self.async_writer.flush_once()

    def prune_book_data(self) -> tuple[int, int]:
        self.flush_async_writes()
        with self._storage_write_lock:
            deleted_rows = self.storage.prune_book_tops(self.cfg.book_top_retention_hours)
            deleted_snapshot_files = self.storage.prune_book_snapshots(self.cfg.book_top_retention_hours)
        return deleted_rows, deleted_snapshot_files

    def close(self) -> None:
        self.stop_background_workers()
        if self.async_writer is not None:
            self.async_writer.stop()
        if self.book_stream is not None:
            self.book_stream.stop()
