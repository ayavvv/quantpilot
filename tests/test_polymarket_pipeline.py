from datetime import datetime, timezone

from polymarket.config import PolySettings
from polymarket.models import MarketInfo, OrderBook
from polymarket.pipeline import PipelineResult, PolymarketPipeline


def test_pipeline_result_supports_mirror_fields():
    result = PipelineResult(markets_seen=1, opportunities_found=2, trades_simulated=3, mirror_traders_tracked=4, mirror_signals_generated=5, stage_timings={"book_fetch_seconds": 1.0})

    assert result.mirror_traders_tracked == 4
    assert result.mirror_signals_generated == 5
    assert result.stage_timings == {"book_fetch_seconds": 1.0}


def test_pipeline_reuses_catalog_until_ttl_expires(tmp_path, monkeypatch):
    cfg = PolySettings(data_dir=str(tmp_path), catalog_refresh_seconds=3600)
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

    first, refreshed_first, load_first = pipeline._get_markets()
    second, refreshed_second, load_second = pipeline._get_markets()

    assert len(first) == 1
    assert len(second) == 1
    assert refreshed_first is True
    assert refreshed_second is False
    assert calls["count"] == 1
    assert load_first >= 0
    assert load_second >= 0


def test_pipeline_fetch_market_books_returns_per_market_pairs(tmp_path, monkeypatch):
    cfg = PolySettings(data_dir=str(tmp_path), book_fetch_workers=2)
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


def test_pipeline_get_markets_falls_back_to_persisted_catalog(tmp_path, monkeypatch):
    cfg = PolySettings(data_dir=str(tmp_path), catalog_refresh_seconds=3600)
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

    markets, refreshed, _ = pipeline._get_markets()

    assert refreshed is False
    assert len(markets) == 1
    assert markets[0].market_id == 'm1'
    assert pipeline._markets_cache[0].market_id == 'm1'


def test_pipeline_does_not_resurrect_stale_persisted_catalog_after_empty_refresh(tmp_path, monkeypatch):
    cfg = PolySettings(data_dir=str(tmp_path), catalog_refresh_seconds=3600)
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
    pipeline._markets_cache = []
    pipeline._markets_refreshed_at = datetime.now(timezone.utc)

    markets, refreshed, _ = pipeline._get_markets()

    assert refreshed is False
    assert markets == []


def test_pipeline_full_set_strategy_skips_snapshot_and_simulator_when_no_opportunities(tmp_path, monkeypatch):
    cfg = PolySettings(data_dir=str(tmp_path))
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
        lambda markets: (
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
