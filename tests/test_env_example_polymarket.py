from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / '.env.example'


def test_env_example_documents_polymarket_keys():
    content = ENV_EXAMPLE.read_text()
    for key in (
        'POLY_DATA_DIR=',
        'POLY_GAMMA_BASE_URL=',
        'POLY_CLOB_BASE_URL=',
        'POLY_CHAIN_ID=',
        'POLY_SCAN_INTERVAL_SECONDS=',
        'POLY_CATALOG_REFRESH_SECONDS=',
        'POLY_MIN_NET_EDGE=',
        'POLY_MAX_BOOK_STALENESS_MS=',
        'POLY_DEFAULT_GAS_COST=',
        'POLY_SLIPPAGE_BUFFER=',
        'POLY_MAX_NOTIONAL_PER_OPP=',
        'POLY_PAPER_INITIAL_CASH=',
        'POLY_PAPER_ONLY=',
        'POLY_ENABLE_SPLIT_SELL=',
        'POLY_MAX_ACTIVE_MARKETS=',
        'POLY_HTTP_TIMEOUT_SECONDS=',
        'POLY_USER_AGENT=',
        'POLY_EMAIL_REPORT_ENABLED=',
        'POLY_EMAIL_REPORT_ATTACH_JSON=',
    ):
        assert key in content


def test_env_example_documents_paper_only_default():
    content = ENV_EXAMPLE.read_text()
    assert 'POLY_PAPER_ONLY=true' in content
