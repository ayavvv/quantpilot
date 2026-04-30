from datetime import datetime, timezone

from polymarket.config import PolySettings
from polymarket.execution_guards import LocalBookDepletion
from polymarket.models import BookLevel, MarketInfo, Opportunity, OrderBook
from polymarket.pipeline import PipelineResult, PolymarketPipeline


def test_pipeline_result_supports_mirror_fields():
    result = PipelineResult(markets_seen=1, opportunities_found=2, trades_simulated=3, mirror_traders_tracked=4, mirror_signals_generated=5, stage_timings={"book_fetch_seconds": 1.0})

    assert result.mirror_traders_tracked == 4
    assert result.mirror_signals_generated == 5
    assert result.stage_timings == {"book_fetch_seconds": 1.0}


def test_local_book_depletion_subtracts_simulated_ask_liquidity():
    depletion = LocalBookDepletion(ttl_seconds=60)
    market = _market()
    yes_book = OrderBook(
        token_id="yes",
        market_id="m1",
        timestamp_ms=1,
        bids=[],
        asks=[BookLevel(price=0.45, size=5), BookLevel(price=0.48, size=10)],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
    )
    no_book = OrderBook(
        token_id="no",
        market_id="m1",
        timestamp_ms=1,
        bids=[],
        asks=[BookLevel(price=0.45, size=5), BookLevel(price=0.48, size=10)],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
    )
    opportunity = Opportunity(
        market_id="m1",
        question="Q",
        direction="buy_both_merge",
        gross_cost=9.0,
        fee_cost=0.0,
        yes_fee_cost=0.0,
        no_fee_cost=0.0,
        gas_cost=0.0,
        slippage_buffer=0.0,
        net_cost=9.0,
        net_edge=1.0,
        capacity=10.0,
        mergeable_qty=10.0,
        yes_qty=10.0,
        no_qty=10.0,
        yes_price=0.45,
        no_price=0.45,
    )

    depletion.record(market, yes_book, no_book, opportunity)
    adjusted = depletion.apply(yes_book)

    assert adjusted.asks == [BookLevel(price=0.48, size=5.0)]


def _market(market_id="m1", yes_token_id="yes", no_token_id="no") -> MarketInfo:
    return MarketInfo(
        market_id=market_id,
        condition_id=market_id,
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
    )


def _fresh_book(token_id: str, market_id: str, ask: float = 0.5) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_id=market_id,
        timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        bids=[BookLevel(price=max(ask - 0.02, 0.01), size=10)],
        asks=[BookLevel(price=ask, size=10)],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
    )


def _fresh_bid_only_book(token_id: str, market_id: str, bid: float = 0.53, size: float = 10) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_id=market_id,
        timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        bids=[BookLevel(price=bid, size=size)],
        asks=[],
        tick_size=0.01,
        min_order_size=1,
        neg_risk=False,
    )


def test_pipeline_reuses_catalog_until_ttl_expires(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), catalog_refresh_seconds=3600)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    calls = {"count": 0}

    def fake_load_binary_markets(cfg=None):
        calls["count"] += 1
        return [market]

    monkeypatch.setattr("polymarket.pipeline.load_binary_markets", fake_load_binary_markets)

    refreshed_count, load_first, snapshot_first = pipeline.refresh_catalog()
    first, load_scan = pipeline._load_scan_markets()

    assert refreshed_count == 1
    assert len(first) == 1
    assert calls["count"] == 1
    assert load_first >= 0
    assert snapshot_first >= 0
    assert load_scan >= 0


def test_pipeline_fetch_market_books_returns_per_market_pairs(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_fetch_workers=2, book_fetch_use_batch=False)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )

    def fake_fetch_book(token_id):
        return OrderBook(token_id=token_id, market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False)

    monkeypatch.setattr(pipeline.clob, "fetch_book", fake_fetch_book)

    books_by_market, errors = pipeline._fetch_market_books([market])

    assert errors == {}
    assert set(books_by_market["m1"].keys()) == {"yes", "no"}
    assert books_by_market["m1"]["yes"].token_id == "yes"
    assert books_by_market["m1"]["no"].token_id == "no"


def test_pipeline_fetch_market_books_uses_batch_books(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_fetch_workers=2, book_fetch_use_batch=True)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    calls = {}

    def fake_fetch_books(token_ids):
        calls["token_ids"] = token_ids
        return {
            "yes": OrderBook(token_id="yes", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False),
            "no": OrderBook(token_id="no", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False),
        }

    monkeypatch.setattr(pipeline.clob, "fetch_books", fake_fetch_books)
    monkeypatch.setattr(pipeline.clob, "fetch_book", lambda token_id: (_ for _ in ()).throw(AssertionError("single fetch should not be called")))

    books_by_market, errors = pipeline._fetch_market_books([market])

    assert errors == {}
    assert calls["token_ids"] == ["yes", "no"]
    assert set(books_by_market["m1"].keys()) == {"yes", "no"}


def test_pipeline_fetch_market_books_falls_back_for_missing_batch_tokens(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_fetch_workers=2, book_fetch_use_batch=True)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )

    monkeypatch.setattr(
        pipeline.clob,
        "fetch_books",
        lambda token_ids: {
            "yes": OrderBook(token_id="yes", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False),
        },
    )
    monkeypatch.setattr(
        pipeline.clob,
        "fetch_book",
        lambda token_id: OrderBook(token_id=token_id, market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False),
    )

    books_by_market, errors = pipeline._fetch_market_books([market])

    assert errors == {}
    assert set(books_by_market["m1"].keys()) == {"yes", "no"}


def test_pipeline_fetch_market_books_can_use_ws_cache(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_source="ws")
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    yes_book = OrderBook(token_id="yes", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False)
    no_book = OrderBook(token_id="no", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False)
    calls = {}

    monkeypatch.setattr(pipeline, "_ensure_ws_stream", lambda markets, wait_ready=True: calls.__setitem__("ensured", len(markets)))
    def fake_get_market_books(markets, connection_stale_seconds, top_only=True):
        calls["top_only"] = top_only
        return {"m1": {"yes": yes_book, "no": no_book}}, {}

    monkeypatch.setattr(
        pipeline.book_cache,
        "get_market_books",
        fake_get_market_books,
    )
    monkeypatch.setattr(pipeline.clob, "fetch_books", lambda token_ids: (_ for _ in ()).throw(AssertionError("HTTP should not be called")))
    monkeypatch.setattr(pipeline.clob, "fetch_book", lambda token_id: (_ for _ in ()).throw(AssertionError("HTTP should not be called")))

    books_by_market, errors = pipeline._fetch_market_books([market])

    assert errors == {}
    assert calls["ensured"] == 1
    assert calls["top_only"] is False
    assert books_by_market["m1"]["yes"] == yes_book
    assert books_by_market["m1"]["no"] == no_book


def test_pipeline_get_markets_falls_back_to_persisted_catalog(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), catalog_refresh_seconds=3600)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    pipeline.storage.save_catalog_snapshot([market])

    def fail_load_binary_markets(cfg=None):
        raise RuntimeError('refresh failed')

    monkeypatch.setattr("polymarket.pipeline.load_binary_markets", fail_load_binary_markets)

    markets, load_time = pipeline._load_scan_markets()

    assert len(markets) == 1
    assert markets[0].market_id == 'm1'
    assert pipeline._markets_cache[0].market_id == 'm1'
    assert pipeline._markets_refreshed_at is not None
    assert load_time >= 0


def test_pipeline_does_not_resurrect_stale_persisted_catalog_after_empty_refresh(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), catalog_refresh_seconds=3600)
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    pipeline.storage.save_catalog_snapshot([market])

    def empty_refresh(cfg=None):
        return []

    monkeypatch.setattr("polymarket.pipeline.load_binary_markets", empty_refresh)

    try:
        pipeline.refresh_catalog()
    except RuntimeError:
        pass

    markets, load_time = pipeline._load_scan_markets()

    assert len(markets) == 1
    assert markets[0].market_id == 'm1'
    assert load_time >= 0


def test_pipeline_full_set_strategy_skips_snapshot_and_simulator_when_no_opportunities(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path))
    pipeline = PolymarketPipeline(cfg)
    market = MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Q",
        slug="q",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )
    book = OrderBook(token_id="yes", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False)
    calls = {"tops": 0, "depth": 0, "simulated": 0, "heartbeat": 0}

    monkeypatch.setattr(
        pipeline,
        "_fetch_market_books",
        lambda markets, wait_ready=True, ensure_markets=None: (
            {"m1": {"yes": book, "no": OrderBook(token_id="no", market_id="m1", timestamp_ms=1, bids=[], asks=[], tick_size=0.01, min_order_size=1, neg_risk=False)}},
            {},
        ),
    )
    monkeypatch.setattr("polymarket.pipeline.scan_market", lambda market, yes_book, no_book, cfg=None: [])
    monkeypatch.setattr(pipeline.storage, "save_book_snapshot", lambda *args, **kwargs: calls.__setitem__("depth", calls["depth"] + 1))
    monkeypatch.setattr(pipeline.storage, "save_book_tops", lambda rows: calls.__setitem__("tops", len(rows)))
    monkeypatch.setattr(pipeline.simulator, "consume", lambda *args, **kwargs: calls.__setitem__("simulated", calls["simulated"] + 1))
    monkeypatch.setattr(pipeline.simulator, "record_scan_heartbeat", lambda *args, **kwargs: calls.__setitem__("heartbeat", calls["heartbeat"] + 1))

    opportunities_found, trades_simulated, stage_timings = pipeline._run_full_set_strategy([market])

    assert opportunities_found == 0
    assert trades_simulated == 0
    assert stage_timings["book_fetch_seconds"] >= 0
    assert calls["tops"] == 2
    assert calls["depth"] == 0
    assert calls["simulated"] == 0
    assert calls["heartbeat"] == 1


def test_pipeline_scans_split_sell_after_buy_side_rejection(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None,
        data_dir=str(tmp_path),
        enable_split_sell=True,
        min_net_edge=0.01,
        slippage_buffer=0.0,
        default_gas_cost=0.0,
        market_cooldown_seconds=0,
    )
    pipeline = PolymarketPipeline(cfg)
    market = _market()

    monkeypatch.setattr(
        pipeline,
        "_fetch_market_books",
        lambda markets, wait_ready=True, ensure_markets=None: (
            {
                "m1": {
                    "yes": _fresh_bid_only_book("yes", "m1", bid=0.53, size=10),
                    "no": _fresh_bid_only_book("no", "m1", bid=0.53, size=10),
                }
            },
            {},
        ),
    )

    opportunities_found, trades_simulated, stage_timings = pipeline._run_full_set_strategy([market])

    assert opportunities_found == 1
    assert trades_simulated == 1
    assert stage_timings["scanned_markets"] == 1
    assert pipeline.simulator.last_accepted_opportunities[0].direction == "split_sell_both"


def test_pipeline_book_top_sampling_gates_scan_artifacts(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_top_sample_seconds=60)
    pipeline = PolymarketPipeline(cfg)

    assert pipeline._should_persist_scan_artifacts() is True
    assert pipeline._should_persist_scan_artifacts() is False
    pipeline._last_book_top_sample_monotonic -= 61
    assert pipeline._should_persist_scan_artifacts() is True


def test_pipeline_dirty_scan_scans_only_dirty_markets_without_book_top_writes(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_source="ws", dirty_scan_enabled=True)
    pipeline = PolymarketPipeline(cfg)
    first = _market("m1", "yes1", "no1")
    second = _market("m2", "yes2", "no2")
    pipeline._markets_cache = [first, second]
    pipeline.book_cache.mark_market_dirty("m2")
    calls = {"tops": 0, "heartbeat": 0, "ensured": 0, "fetched_market_ids": []}

    def fake_fetch_market_books(markets, wait_ready=True, ensure_markets=None):
        calls["fetched_market_ids"] = [market.market_id for market in markets]
        assert ensure_markets == [first, second]
        return {
            "m2": {
                "yes": _fresh_book("yes2", "m2", ask=0.5),
                "no": _fresh_book("no2", "m2", ask=0.51),
            }
        }, {}

    monkeypatch.setattr(pipeline, "_ensure_ws_stream", lambda markets, wait_ready=True: calls.__setitem__("ensured", len(markets)))
    monkeypatch.setattr(pipeline, "_fetch_market_books", fake_fetch_market_books)
    monkeypatch.setattr(pipeline.storage, "save_book_tops", lambda rows: calls.__setitem__("tops", len(rows)))
    monkeypatch.setattr(pipeline.simulator, "record_scan_heartbeat", lambda *args, **kwargs: calls.__setitem__("heartbeat", calls["heartbeat"] + 1))

    result = pipeline.run_dirty_once()

    assert result.markets_seen == 1
    assert calls["ensured"] == 2
    assert calls["fetched_market_ids"] == ["m2"]
    assert calls["tops"] == 0
    assert calls["heartbeat"] == 0


def test_pipeline_dirty_scan_yields_when_full_scan_is_waiting(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_source="ws", dirty_scan_enabled=True)
    pipeline = PolymarketPipeline(cfg)
    pipeline.book_cache.mark_market_dirty("m1")
    pipeline._full_scan_requested.set()

    result = pipeline.run_dirty_once()

    assert result.markets_seen == 0
    assert result.stage_timings == {"full_scan_pending": 1.0}
    assert pipeline.book_cache.pop_dirty_market_ids() == {"m1"}


def test_pipeline_reconcile_token_window_rolls_across_assets(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), ws_reconcile_max_tokens_per_cycle=3)
    pipeline = PolymarketPipeline(cfg)

    first = pipeline._reconcile_token_window(["t1", "t2", "t3", "t4", "t5"])
    second = pipeline._reconcile_token_window(["t1", "t2", "t3", "t4", "t5"])
    third = pipeline._reconcile_token_window(["t1", "t2", "t3", "t4", "t5"])

    assert first == ["t1", "t2", "t3"]
    assert second == ["t4", "t5", "t1"]
    assert third == ["t2", "t3", "t4"]


def test_pipeline_reconcile_overwrites_ws_cache_and_marks_drift(tmp_path, monkeypatch):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), book_source="ws", ws_reconcile_enabled=True)
    pipeline = PolymarketPipeline(cfg)
    market = _market()
    pipeline._markets_cache = [market]
    pipeline.book_cache.apply_book(
        {
            "asset_id": "yes",
            "market": "m1",
            "timestamp": "1777442824176",
            "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.50", "size": "10"}],
        }
    )
    pipeline.book_cache.apply_book(
        {
            "asset_id": "no",
            "market": "m1",
            "timestamp": "1777442824176",
            "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.50", "size": "10"}],
        }
    )
    pipeline.book_cache.pop_dirty_market_ids()
    calls = {}

    def fake_fetch_books(token_ids, attempts=3, timeout_seconds=None):
        calls["token_ids"] = token_ids
        calls["attempts"] = attempts
        calls["timeout_seconds"] = timeout_seconds
        return {
            "yes": _fresh_book("yes", "m1", ask=0.49),
            "no": OrderBook(
                token_id="no",
                market_id="m1",
                timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                bids=[BookLevel(price=0.40, size=10)],
                asks=[BookLevel(price=0.50, size=10)],
                tick_size=0.01,
                min_order_size=1,
                neg_risk=False,
            ),
        }

    monkeypatch.setattr(pipeline, "_ensure_ws_stream", lambda markets, wait_ready=True: None)
    monkeypatch.setattr(pipeline.clob, "fetch_books", fake_fetch_books)

    result = pipeline.run_reconcile_once()

    assert result["enabled"] is True
    assert result["tokens"] == 2
    assert result["top_drifted_tokens"] == 1
    assert calls == {"token_ids": ["yes", "no"], "attempts": 1, "timeout_seconds": 3.0}
    assert pipeline.book_cache.pop_dirty_market_ids() == {"m1"}
