# Major Money Notification System

## Feasibility

The request is useful, but only if it is framed as a vendor/proxy signal rather
than literal identification of institutions.

- Exact "main force entered/exited" needs tick-by-tick trades, order book, or
  exchange/pro data. Daily OHLCV cannot identify the real buyer or seller.
- Futu `get_capital_flow` returns per-stock capital-flow fields. For historical
  Day/Week/Month periods it includes `main_in_flow`, `super_in_flow`,
  `big_in_flow`, `mid_in_flow`, and `sml_in_flow`.
- Futu `get_capital_distribution` returns order-size inflow/outflow buckets
  derived from historical tick-by-tick transactions.
- Futu stock basic info can list broad universes. Current local OpenD probe on
  2026-05-31 returned about HK 3695, US 12880, SH 2373, SZ 2954 stock rows.
- Because Futu capital flow is a per-symbol API, a true daily full US + HK + A
  scan is operationally heavy. At a conservative one request per second, just
  US+HK is roughly 4.6 hours for flow only; including distribution doubles that.
- A-share market-wide daily coverage is practical today through Eastmoney's
  processed fund-flow rank. It is still vendor-derived, not raw Level-2.

Conclusion: the reliable production shape is a daily advisory notification that
reports source, coverage, entry/exit counts, and amounts by currency. Turning it
into automatic trading filters needs forward-return validation and promotion
gates.

## Implemented Shape

Artifacts:

- `scripts/build_major_money_digest.py`
  - Builds `~/quantpilot_data/output/major_money_digest_latest.json`.
  - Builds `~/quantpilot_data/output/major_money_digest_latest.csv`.
  - Archives dated JSON/CSV copies under
    `~/quantpilot_data/output/major_money_digest/`.
  - Defaults to A-share Eastmoney full-market flow if present.
  - Also consumes HK/US Futu full-market scan artifacts when present:
    - `~/quantpilot_data/capital_flow/futu_market/HK_latest_flow.csv`
    - `~/quantpilot_data/capital_flow/futu_market/US_latest_flow.csv`
  - Also consumes an optional `US_OTC` proxy artifact when present:
    - `~/quantpilot_data/capital_flow/us_otc_proxy/US_OTC_latest_flow.csv`
  - The default source discovery only includes `US_OTC` when
    `US_OTC_latest_status.json` is healthy, so stale proxy files do not silently
    satisfy market coverage.
- `scripts/refresh_eastmoney_fund_flow_rank.py`
  - Refreshes the A-share market-wide Eastmoney rank artifact before the daily
    report.
  - Writes `~/quantpilot_data/output/eastmoney_fund_flow_rank_latest.csv`.
  - Archives daily copies under `~/quantpilot_data/fund_flow/eastmoney/`.
  - Refuses to replace the latest artifact if the fetched row count is below
    `EASTMONEY_FUND_FLOW_MIN_ROWS`.
- `scripts/scan_futu_market_capital_flow.py`
  - Resumable Futu scanner for HK/US/SH/SZ stock universes.
  - Writes dated and latest flow CSVs under
    `~/quantpilot_data/capital_flow/futu_market/`.
  - Writes `{MARKET}_latest_status.json` with scan status, attempted count,
    ok/error/empty counts, coverage ratio, and output paths.
  - Status files also include source/selected/excluded exchange-type counts and
    per-exchange OK/error/empty counts.
  - Status files include `scanner_schema_version`; health/readiness checks
    warn when the latest HK/US scan was produced by an older scanner and needs
    a refresh before the digest can claim current stock-universe filtering.
  - The scheduled stock universe now filters obvious non-common instruments
    before per-symbol requests: preferred shares, debt/notes, units, warrants
    or rights, delisted-label rows, and listings whose `listing_date` is after
    the scan date. This keeps the "all stocks" notification focused on common
    stock candidates instead of stale or non-stock listed securities.
  - Writes `US_latest_source_universe.csv`, preserving the full source universe
    before scheduled filters such as `US_PINK,N/A` are applied.
  - Use `--max-codes` for smoke tests; omit it only when ready for long full
    scans.
- `scripts/scan_us_otc_proxy_flow.py`
  - Optional Polygon/Massive daily aggregate scanner for `US_PINK`.
  - Produces a `US_OTC` proxy artifact using directional dollar volume from
    daily OHLCV. This is lower-confidence than vendor capital-flow fields and
    is labeled as a proxy in the artifact source.
  - Defaults to the latest completed US session date, so China-time Monday
    evening does not request a Sunday aggregate.
  - Requires `POLYGON_API_KEY` or `POLYGON_API_KEY_FILE`; without a configured
    key, the daily digest will keep `US_OTC` visible as missing coverage.
  - Auto digest rebuilds only include `US_OTC` after the current proxy scan
    succeeds, avoiding stale proxy artifacts after API/key failures.
- `scripts/run_market_capital_flow.sh`
  - Host-side cron wrapper around the Futu scanner.
  - Uses a lock directory so a long full-market scan cannot overlap itself.
  - Rebuilds the digest after a scan by default.
  - Same-day reruns automatically refresh instead of resuming old rows when the
    dated status file was produced by an older scanner schema. Set
    `FUTU_MARKET_FLOW_OVERWRITE=true` for a manual full refresh regardless of
    schema.
  - Can also build the optional `US_OTC` proxy artifact during the scheduled US
    after-close scan when `ENABLE_US_OTC_PROXY_FLOW=true`.
  - Defaults to excluding `US_PINK,N/A`, so the scheduled US job covers
    exchange-listed NYSE/NASDAQ/AMEX names unless
    `FUTU_MARKET_FLOW_EXCLUDE_EXCHANGE_TYPES` is overridden.
  - A 2026-05-31 `US_PINK` probe returned `Do not support OTC market data` for
    sampled symbols, so OTC/Pink is treated as a vendor unsupported venue rather
    than silently counted as zero flow.
- `reporter/send_report.py`
  - Adds a "Market-Wide Major Money" section to the daily email.
  - Adds major-money counts/source coverage to the email subject when the digest
    is available, including missing and partial-coverage markets.
  - Adds a top summary with total major-entry/major-exit counts and
    entry/exit/net amounts by currency.
  - Shows market coverage, entry count/amount, exit count/amount, net amount,
    and top entries/exits per market so CNY/HKD/USD names are not mixed by raw
    amount.
  - Coverage includes the exchange-type breakdown when the source provides it,
    making excluded or unsupported venues visible in the email.
  - Coverage notes also include excluded security-class counts when the scanner
    filtered non-common instruments from the stock universe.
  - Coverage notes include vendor empty/error row counts so partial scans are
    visible, not just the final entry/exit counts.
  - The summary sentence and per-market notes identify missing expected
    markets, such as `US_OTC`, instead of leaving the table reason blank.
  - Missing HK/US artifacts are shown as missing coverage, not silently treated
    as zero signals.
- `scripts/run_daily.sh`
  - Adds Step 2d to build the digest before sending the daily report.
- `scripts/daily_healthcheck.py`
  - Warns when A-share fund-flow rank, major-money digest, or HK/US full-market
    Futu scan status files are missing, stale, unreadable, or below coverage
    thresholds.
  - Warns when HK/US market-wide scan status files lack the current scanner
    schema metadata, so old artifacts cannot silently satisfy coverage after a
    scanner upgrade.
  - Verifies the dated major-money digest archive JSON/CSV exists for the
    digest `flow_date` and that the archived JSON matches the latest digest
    date.
  - Warns when available major-money sources exceed
    `HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO` for vendor empty/error rows.
  - Checks `MAJOR_MONEY_EXPECTED_MARKETS` explicitly and reports whether
    `US_OTC` is missing because the proxy is disabled, the provider key is
    absent, the universe is missing, or the proxy scan/status is unhealthy.
  - Nightly health snapshots include the end-to-end readiness result, while
    de-duplicating overlapping market-money issues in the alert list.
- `scripts/major_money_readiness.py`
  - End-to-end readiness check for the notification system.
  - Checks daily/HK/US cron entries, reporter email configuration, major-money
    scan freshness/schema, digest coverage, dated digest archive output, and
    `US_OTC` proxy state.
  - Uses `HEALTHCHECK_MAJOR_MONEY_MAX_NON_OK_RATIO` to flag available markets
    whose vendor empty/error rows are too high.
  - Exits non-zero until every expected market, including `US_OTC`, is covered.

## Daily Commands

Build the digest from currently available artifacts:

```bash
.venv/bin/python -m scripts.refresh_eastmoney_fund_flow_rank \
  --output ~/quantpilot_data/output/eastmoney_fund_flow_rank_latest.csv \
  --archive-dir ~/quantpilot_data/fund_flow/eastmoney \
  --limit 6000 \
  --min-rows 1000

.venv/bin/python -m scripts.build_major_money_digest \
  --expected-markets A,HK,US,US_OTC \
  --output-json ~/quantpilot_data/output/major_money_digest_latest.json \
  --output-csv ~/quantpilot_data/output/major_money_digest_latest.csv
```

Optional US OTC/Pink proxy flow, if a Polygon/Massive API key is configured:

```bash
POLYGON_API_KEY=... \
.venv/bin/python -m scripts.scan_us_otc_proxy_flow \
  --universe-csv ~/quantpilot_data/capital_flow/futu_market/US_latest_source_universe.csv \
  --exchange-types US_PINK \
  --output-dir ~/quantpilot_data/capital_flow/us_otc_proxy
```

To enable this in the scheduled US after-close scan, set these in `.env`:

```bash
ENABLE_US_OTC_PROXY_FLOW=true
POLYGON_API_KEY=...
# Or keep the secret out of .env:
# POLYGON_API_KEY_FILE=/Users/theo/.config/quantpilot/polygon_api_key
US_OTC_PROXY_FLOW_EXCHANGE_TYPES=US_PINK
```

Smoke-test Futu HK/US scanning without claiming full-market coverage:

```bash
PYTHONPATH=/Users/theo/quantpilot .venv/bin/python -m scripts.scan_futu_market_capital_flow \
  --markets HK,US \
  --max-codes 5 \
  --host 127.0.0.1 \
  --port 11111 \
  --connect-timeout 5 \
  --output-dir /tmp/quantpilot_futu_market_flow_smoke
```

Production-style wrapper smoke test:

```bash
FUTU_MARKET_FLOW_MARKETS=HK,US \
FUTU_MARKET_FLOW_CODES=HK.00700,US.AAPL,US.NVDA \
FUTU_MARKET_FLOW_MAX_CODES=0 \
FUTU_MARKET_FLOW_PAUSE_SECONDS=0 \
FUTU_MARKET_FLOW_RATE_LIMIT_DELAY=0.1 \
FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_ATTEMPTS=2 \
FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_SECONDS=31 \
FUTU_MARKET_FLOW_OUTPUT_DIR=/tmp/quantpilot_futu_market_flow_smoke \
MAJOR_MONEY_DIGEST_JSON=/tmp/major_money_digest_smoke.json \
MAJOR_MONEY_DIGEST_CSV=/tmp/major_money_digest_smoke.csv \
./scripts/run_market_capital_flow.sh
```

End-to-end readiness check:

```bash
PYTHONPATH=/Users/theo/quantpilot .venv/bin/python -m scripts.major_money_readiness
```

Current expected failure without a provider key:

```text
Major-money digest expected market unavailable: US_OTC
US OTC/Pink proxy disabled: set ENABLE_US_OTC_PROXY_FLOW=true
```

Recommended host cron entries, Asia/Shanghai:

```cron
# HK after close; expected to finish before the 19:00 A-share daily report.
40 16 * * 1-5 FUTU_MARKET_FLOW_MARKETS=HK /Users/theo/quantpilot/scripts/run_market_capital_flow.sh >> /Users/theo/quantpilot/logs/market_capital_flow_hk.log 2>&1

# US after regular close; Tuesday-Saturday China time maps to Monday-Friday US sessions.
10 5 * * 2-6 FUTU_MARKET_FLOW_MARKETS=US /Users/theo/quantpilot/scripts/run_market_capital_flow.sh >> /Users/theo/quantpilot/logs/market_capital_flow_us.log 2>&1
```

Full Futu market scans should run as separate off-hours jobs, not inside the
main A-share pipeline, until runtime and rate limits are measured on this
account. Keep the effective request interval at about one request per second or
slower; when Futu still returns its 30-requests-per-30-seconds limit error, the
scanner sleeps `FUTU_MARKET_FLOW_RATE_LIMIT_RETRY_SECONDS` and retries the same
symbol before counting it as an error.
