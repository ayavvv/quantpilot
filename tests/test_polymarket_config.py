from pathlib import Path

from polymarket.config import PolySettings


def test_polymarket_scan_interval_default_is_service_safe():
    cfg = PolySettings(_env_file=None)
    assert cfg.scan_interval_seconds == 300
    assert cfg.book_source == "http"
    assert cfg.book_fetch_use_batch is True
    assert cfg.book_fetch_batch_size == 500
    assert cfg.catalog_page_size == 1000
    assert cfg.catalog_fetch_workers == 4
    assert cfg.catalog_fetch_fee_rates is True
    assert cfg.ws_market_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    assert cfg.ws_reconcile_enabled is False
    assert cfg.ws_reconcile_seconds == 10
    assert cfg.ws_reconcile_timeout_seconds == 3
    assert cfg.ws_reconcile_batch_size == 50
    assert cfg.ws_reconcile_workers == 4
    assert cfg.ws_reconcile_max_tokens_per_cycle == 500
    assert cfg.dirty_scan_enabled is False
    assert cfg.dirty_scan_interval_seconds == 0.1
    assert cfg.storage_async_flush_enabled is False
    assert cfg.book_top_sample_seconds == 0
    assert cfg.book_top_retention_hours == 72
    assert cfg.book_top_retention_job_seconds == 3600


def test_env_example_documents_service_safe_scan_interval():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text()
    assert "POLY_SCAN_INTERVAL_SECONDS=300" in content
    assert "POLY_BOOK_SOURCE=http" in content
    assert "POLY_CATALOG_PAGE_SIZE=1000" in content
    assert "POLY_CATALOG_FETCH_WORKERS=4" in content
    assert "POLY_CATALOG_FETCH_FEE_RATES=true" in content
    assert "POLY_WS_MARKET_URL=wss://ws-subscriptions-clob.polymarket.com/ws/market" in content
    assert "POLY_WS_RECONCILE_ENABLED=false" in content
    assert "POLY_WS_RECONCILE_SECONDS=10" in content
    assert "POLY_WS_RECONCILE_TIMEOUT_SECONDS=3" in content
    assert "POLY_WS_RECONCILE_BATCH_SIZE=50" in content
    assert "POLY_WS_RECONCILE_WORKERS=4" in content
    assert "POLY_WS_RECONCILE_MAX_TOKENS_PER_CYCLE=500" in content
    assert "POLY_DIRTY_SCAN_ENABLED=false" in content
    assert "POLY_DIRTY_SCAN_INTERVAL_SECONDS=0.1" in content
    assert "POLY_STORAGE_ASYNC_FLUSH_ENABLED=false" in content
    assert "POLY_BOOK_TOP_SAMPLE_SECONDS=0" in content
    assert "POLY_BOOK_TOP_RETENTION_HOURS=72" in content
    assert "POLY_BOOK_TOP_RETENTION_JOB_SECONDS=3600" in content
