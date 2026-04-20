from pathlib import Path

from polymarket.config import PolySettings


def test_polymarket_scan_interval_default_is_service_safe():
    cfg = PolySettings()
    assert cfg.scan_interval_seconds == 300


def test_env_example_documents_service_safe_scan_interval():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    content = env_example.read_text()
    assert "POLY_SCAN_INTERVAL_SECONDS=300" in content
