from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_MARKET_CAPITAL_FLOW = REPO_ROOT / "scripts" / "run_market_capital_flow.sh"


def test_run_market_capital_flow_wraps_futu_scanner_and_digest():
    content = RUN_MARKET_CAPITAL_FLOW.read_text()

    assert 'FUTU_MARKET_FLOW_MARKETS="${FUTU_MARKET_FLOW_MARKETS:-HK,US}"' in content
    assert 'FUTU_MARKET_FLOW_MIN_OK_RATIO="${FUTU_MARKET_FLOW_MIN_OK_RATIO:-0}"' in content
    assert 'FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES="${FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES:-US_PINK,N/A}"' in content
    assert 'MAJOR_MONEY_EXPECTED_MARKETS="${MAJOR_MONEY_EXPECTED_MARKETS:-A,HK,US,US_OTC}"' in content
    assert 'ENABLE_US_OTC_PROXY_FLOW="${ENABLE_US_OTC_PROXY_FLOW:-false}"' in content
    assert 'mkdir "$LOCK_DIR"' in content
    assert '"$PYTHON_BIN" -m scripts.scan_futu_market_capital_flow "${SCAN_ARGS[@]}"' in content
    assert 'market_list_contains "$FUTU_MARKET_FLOW_MARKETS" "US"' in content
    assert '"$PYTHON_BIN" -m scripts.scan_us_otc_proxy_flow "${US_OTC_PROXY_ARGS[@]}"' in content
    assert 'MAJOR_MONEY_DIGEST_SOURCES="${MAJOR_MONEY_DIGEST_SOURCES:-auto}"' in content
    assert 'MAJOR_MONEY_SOURCE_ARGS+=(--source "$market:$latest_flow:futu")' in content
    assert 'MAJOR_MONEY_SOURCE_ARGS+=(--source "US_OTC:$otc_latest_flow:${US_OTC_PROXY_FLOW_PROVIDER}_otc_proxy")' in content
    assert '"$PYTHON_BIN" -m scripts.build_major_money_digest' in content


def test_run_market_capital_flow_supports_smoke_codes():
    content = RUN_MARKET_CAPITAL_FLOW.read_text()

    assert 'FUTU_MARKET_FLOW_CODES="${FUTU_MARKET_FLOW_CODES:-}"' in content
    assert 'SCAN_ARGS+=(--codes "$FUTU_MARKET_FLOW_CODES")' in content
    assert 'SCAN_ARGS+=(--max-codes "$FUTU_MARKET_FLOW_MAX_CODES")' in content
