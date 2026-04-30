from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_POLY_SCHEDULER = REPO_ROOT / "scripts" / "run_polymarket_scheduler.sh"


def test_run_polymarket_scheduler_script_stays_poly_scoped():
    content = RUN_POLY_SCHEDULER.read_text()
    assert 'load_env_defaults "$PROJECT_DIR/.env"' in content
    assert ': "${POLY_DATA_DIR:=$DATA_DIR/polymarket}"' in content
    assert ': "${POLY_ENABLE_SPLIT_SELL:=true}"' in content
    assert ': "${POLY_ENABLE_TOP_TRADER_MIRROR:=false}"' in content
    assert ': "${POLY_SCAN_INTERVAL_SECONDS:=300}"' in content
    assert ': "${POLY_CATALOG_REFRESH_JOB_SECONDS:=1800}"' in content
    assert ': "${POLY_BOOK_SOURCE:=http}"' in content
    assert ': "${POLY_BOOK_FETCH_USE_BATCH:=true}"' in content
    assert ': "${POLY_BOOK_FETCH_BATCH_SIZE:=500}"' in content
    assert ': "${POLY_MAX_ACTIVE_MARKETS:=250}"' in content
    assert ': "${POLY_CATALOG_PAGE_SIZE:=1000}"' in content
    assert ': "${POLY_CATALOG_FETCH_WORKERS:=4}"' in content
    assert ': "${POLY_CATALOG_FETCH_FEE_RATES:=true}"' in content
    assert ': "${POLY_WS_MARKET_URL:=wss://ws-subscriptions-clob.polymarket.com/ws/market}"' in content
    assert ': "${POLY_WS_RECONCILE_ENABLED:=false}"' in content
    assert ': "${POLY_WS_RECONCILE_SECONDS:=10}"' in content
    assert ': "${POLY_WS_RECONCILE_TIMEOUT_SECONDS:=3}"' in content
    assert ': "${POLY_WS_RECONCILE_BATCH_SIZE:=50}"' in content
    assert ': "${POLY_WS_RECONCILE_WORKERS:=4}"' in content
    assert ': "${POLY_WS_RECONCILE_MAX_TOKENS_PER_CYCLE:=500}"' in content
    assert ': "${POLY_DIRTY_SCAN_ENABLED:=false}"' in content
    assert ': "${POLY_DIRTY_SCAN_INTERVAL_SECONDS:=0.1}"' in content
    assert ': "${POLY_STORAGE_ASYNC_FLUSH_ENABLED:=false}"' in content
    assert ': "${POLY_BOOK_TOP_SAMPLE_SECONDS:=0}"' in content
    assert ': "${POLY_BOOK_TOP_RETENTION_HOURS:=72}"' in content
    assert ': "${POLY_BOOK_TOP_RETENTION_JOB_SECONDS:=3600}"' in content
    assert ': "${POLY_TARGET_NOTIONAL_PER_OPP:=25}"' in content
    assert ': "${POLY_MARKET_COOLDOWN_SECONDS:=60}"' in content
    assert ': "${POLY_MAX_MARKET_NOTIONAL_PER_DAY:=50}"' in content
    assert ': "${POLY_MAX_DAILY_NOTIONAL:=250}"' in content
    assert ': "${POLY_MAX_DAILY_LOSS:=25}"' in content
    assert 'PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"' in content
    assert 'export DATA_DIR POLY_DATA_DIR POLY_PAPER_ONLY POLY_ENABLE_SPLIT_SELL POLY_ENABLE_TOP_TRADER_MIRROR' in content
    assert 'export POLY_MAX_ACTIVE_MARKETS POLY_CATALOG_PAGE_SIZE POLY_CATALOG_FETCH_WORKERS POLY_CATALOG_FETCH_FEE_RATES' in content
    assert 'export POLY_BOOK_SOURCE POLY_BOOK_FETCH_USE_BATCH POLY_BOOK_FETCH_BATCH_SIZE POLY_BOOK_FETCH_WORKERS' in content
    assert 'export POLY_WS_RECONCILE_ENABLED POLY_WS_RECONCILE_SECONDS POLY_WS_RECONCILE_TIMEOUT_SECONDS' in content
    assert 'export POLY_WS_RECONCILE_BATCH_SIZE POLY_WS_RECONCILE_WORKERS' in content
    assert 'export POLY_WS_RECONCILE_MAX_TOKENS_PER_CYCLE' in content
    assert 'export POLY_DIRTY_SCAN_ENABLED POLY_DIRTY_SCAN_INTERVAL_SECONDS' in content
    assert 'export POLY_STORAGE_ASYNC_FLUSH_ENABLED POLY_STORAGE_ASYNC_FLUSH_SECONDS' in content
    assert 'export POLY_BOOK_TOP_SAMPLE_SECONDS' in content
    assert 'export POLY_BOOK_TOP_RETENTION_HOURS POLY_BOOK_TOP_RETENTION_JOB_SECONDS' in content
    assert 'export POLY_MIN_NET_EDGE POLY_DEFAULT_GAS_COST POLY_SLIPPAGE_BUFFER POLY_TARGET_NOTIONAL_PER_OPP' in content
    assert 'export POLY_MAX_MARKET_NOTIONAL_PER_DAY POLY_MAX_DAILY_NOTIONAL POLY_MAX_DAILY_LOSS' in content
    assert '"$PYTHON_BIN" -m polymarket.scheduler' in content
    assert 'mkdir -p "$PROJECT_DIR/logs"' in content


def test_run_polymarket_scheduler_script_does_not_call_a_share_paths():
    content = RUN_POLY_SCHEDULER.read_text()
    assert 'trader.trade_daily' not in content
    assert 'run_daily.sh' not in content
    assert 'scripts.daily_healthcheck' not in content
    assert 'a_share_readiness' not in content
