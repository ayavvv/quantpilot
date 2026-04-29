from pathlib import Path

from polymarket.config import PolySettings


def test_polymarket_scan_interval_default_is_service_safe():
    cfg = PolySettings(_env_file=None)
    assert cfg.scan_interval_seconds == 300
    assert cfg.book_source == "http"
    assert cfg.book_fetch_use_batch is True
    assert cfg.book_fetch_batch_size == 500
    assert cfg.ws_market_url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    assert cfg.storage_async_flush_enabled is False
    assert cfg.book_top_retention_hours == 72
    assert cfg.book_top_retention_job_seconds == 3600


def test_env_example_documents_service_safe_scan_interval():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text()
    assert "POLY_SCAN_INTERVAL_SECONDS=300" in content
    assert "POLY_BOOK_SOURCE=http" in content
    assert "POLY_WS_MARKET_URL=wss://ws-subscriptions-clob.polymarket.com/ws/market" in content
    assert "POLY_STORAGE_ASYNC_FLUSH_ENABLED=false" in content
    assert "POLY_BOOK_TOP_RETENTION_HOURS=72" in content
    assert "POLY_BOOK_TOP_RETENTION_JOB_SECONDS=3600" in content
