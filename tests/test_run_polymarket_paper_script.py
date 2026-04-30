from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_POLY = REPO_ROOT / "scripts" / "run_polymarket_paper.sh"


def test_run_polymarket_script_stays_poly_scoped():
    content = RUN_POLY.read_text()
    assert 'POLY_DATA_DIR="$POLY_DATA_DIR" \\' in content
    assert 'POLY_ENABLE_SPLIT_SELL="${POLY_ENABLE_SPLIT_SELL:-true}" \\' in content
    assert 'from polymarket.pipeline import PolymarketPipeline' in content
    assert 'from polymarket.reporting.daily import generate_daily_report' in content
    assert 'LATEST_REPORT="$POLY_DATA_DIR/reports/daily_summary_latest.json"' in content


def test_run_polymarket_script_does_not_call_a_share_paths():
    content = RUN_POLY.read_text()
    assert 'trader.trade_daily' not in content
    assert 'run_daily.sh' not in content
    assert 'scripts.daily_healthcheck' not in content
    assert 'a_share_readiness' not in content
