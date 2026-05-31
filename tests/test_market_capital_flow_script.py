from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_MARKET_CAPITAL_FLOW = REPO_ROOT / "scripts" / "run_market_capital_flow.sh"


def test_run_market_capital_flow_wraps_futu_scanner_and_digest():
    content = RUN_MARKET_CAPITAL_FLOW.read_text()

    assert 'FUTU_MARKET_FLOW_MARKETS="${FUTU_MARKET_FLOW_MARKETS:-HK,US}"' in content
    assert 'FUTU_MARKET_FLOW_MIN_OK_RATIO="${FUTU_MARKET_FLOW_MIN_OK_RATIO:-0}"' in content
    assert 'FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_ATTEMPTS="${FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_ATTEMPTS:-2}"' in content
    assert 'FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_SECONDS="${FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_SECONDS:-31}"' in content
    assert 'FUTU_MARKET_FLOW_TRANSIENT_RETRY_ATTEMPTS="${FUTU_MARKET_FLOW_TRANSIENT_RETRY_ATTEMPTS:-2}"' in content
    assert 'FUTU_MARKET_FLOW_TRANSIENT_RETRY_SECONDS="${FUTU_MARKET_FLOW_TRANSIENT_RETRY_SECONDS:-5}"' in content
    assert 'FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES="${FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES:-US_PINK,N/A}"' in content
    assert 'FUTU_MARKET_FLOW_EXCLUDE_SECURITY_CLASSES="${FUTU_MARKET_FLOW_EXCLUDE_SECURITY_CLASSES:-preferred,note_debt,unit,warrant_right,delisted_label,future_listing}"' in content
    assert 'FUTU_MARKET_FLOW_OVERWRITE="${FUTU_MARKET_FLOW_OVERWRITE:-false}"' in content
    assert 'MAJOR_MONEY_EXPECTED_MARKETS="${MAJOR_MONEY_EXPECTED_MARKETS:-A,HK,US,US_OTC}"' in content
    assert 'MAJOR_MONEY_DIGEST_ARCHIVE_DIR="${MAJOR_MONEY_DIGEST_ARCHIVE_DIR:-$DATA_DIR/output/major_money_digest}"' in content
    assert 'ENABLE_US_OTC_PROXY_FLOW="${ENABLE_US_OTC_PROXY_FLOW:-false}"' in content
    assert 'US_OTC_PROXY_FLOW_REQUEST_DELAY="${US_OTC_PROXY_FLOW_REQUEST_DELAY:-0.2}"' in content
    assert 'US_OTC_PROXY_FLOW_MAX_RETRIES="${US_OTC_PROXY_FLOW_MAX_RETRIES:-2}"' in content
    assert 'US_OTC_PROXY_FLOW_TIMEOUT="${US_OTC_PROXY_FLOW_TIMEOUT:-15}"' in content
    assert 'mkdir "$LOCK_DIR"' in content
    assert '"$PYTHON_BIN" -m scripts.scan_futu_market_capital_flow "${SCAN_ARGS[@]}"' in content
    assert 'US_OTC_PROXY_FLOW_AVAILABLE=false' in content
    assert 'market_list_contains "$FUTU_MARKET_FLOW_MARKETS" "US"' in content
    assert '[ -z "${POLYGON_API_KEY:-}" ] && [ -z "${POLYGON_API_KEY_FILE:-}" ]' in content
    assert 'POLYGON_API_KEY or POLYGON_API_KEY_FILE is not set' in content
    assert '--rate-limit-retry-attempts "$FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_ATTEMPTS"' in content
    assert '--rate-limit-retry-seconds "$FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_SECONDS"' in content
    assert '--transient-retry-attempts "$FUTU_MARKET_FLOW_TRANSIENT_RETRY_ATTEMPTS"' in content
    assert '--transient-retry-seconds "$FUTU_MARKET_FLOW_TRANSIENT_RETRY_SECONDS"' in content
    assert '--exclude-security-classes "$FUTU_MARKET_FLOW_EXCLUDE_SECURITY_CLASSES"' in content
    assert 'SCAN_ARGS+=(--overwrite)' in content
    assert 'US_OTC_PROXY_FLOW_AVAILABLE=true' in content
    assert '"$PYTHON_BIN" -m scripts.scan_us_otc_proxy_flow "${US_OTC_PROXY_ARGS[@]}"' in content
    assert '--request-delay "$US_OTC_PROXY_FLOW_REQUEST_DELAY"' in content
    assert '--max-retries "$US_OTC_PROXY_FLOW_MAX_RETRIES"' in content
    assert '--timeout "$US_OTC_PROXY_FLOW_TIMEOUT"' in content
    assert 'MAJOR_MONEY_DIGEST_SOURCES="${MAJOR_MONEY_DIGEST_SOURCES:-auto}"' in content
    assert 'MAJOR_MONEY_SOURCE_ARGS+=(--source "$market:$latest_flow:futu")' in content
    assert 'if [ "$US_OTC_PROXY_FLOW_AVAILABLE" = "true" ] && [ -f "$otc_latest_flow" ]; then' in content
    assert '"$PYTHON_BIN" -m scripts.build_major_money_digest' in content
    assert '--archive-dir "$MAJOR_MONEY_DIGEST_ARCHIVE_DIR"' in content


def test_run_market_capital_flow_supports_smoke_codes():
    content = RUN_MARKET_CAPITAL_FLOW.read_text()

    assert 'FUTU_MARKET_FLOW_CODES="${FUTU_MARKET_FLOW_CODES:-}"' in content
    assert 'SCAN_ARGS+=(--codes "$FUTU_MARKET_FLOW_CODES")' in content
    assert 'SCAN_ARGS+=(--max-codes "$FUTU_MARKET_FLOW_MAX_CODES")' in content
