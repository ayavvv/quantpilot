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
  - Defaults to A-share Eastmoney full-market flow if present.
  - Also consumes HK/US Futu full-market scan artifacts when present:
    - `~/quantpilot_data/capital_flow/futu_market/HK_latest_flow.csv`
    - `~/quantpilot_data/capital_flow/futu_market/US_latest_flow.csv`
- `scripts/scan_futu_market_capital_flow.py`
  - Resumable Futu scanner for HK/US/SH/SZ stock universes.
  - Writes dated and latest flow CSVs under
    `~/quantpilot_data/capital_flow/futu_market/`.
  - Use `--max-codes` for smoke tests; omit it only when ready for long full
    scans.
- `scripts/run_market_capital_flow.sh`
  - Host-side cron wrapper around the Futu scanner.
  - Uses a lock directory so a long full-market scan cannot overlap itself.
  - Rebuilds the digest after a scan by default.
- `reporter/send_report.py`
  - Adds a "Market-Wide Major Money" section to the daily email.
  - Shows market coverage, entry count/amount, exit count/amount, net amount,
    top entries, and top exits.
  - Missing HK/US artifacts are shown as missing coverage, not silently treated
    as zero signals.
- `scripts/run_daily.sh`
  - Adds Step 2d to build the digest before sending the daily report.

## Daily Commands

Build the digest from currently available artifacts:

```bash
.venv/bin/python -m scripts.build_major_money_digest \
  --expected-markets A,HK,US \
  --output-json ~/quantpilot_data/output/major_money_digest_latest.json \
  --output-csv ~/quantpilot_data/output/major_money_digest_latest.csv
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
FUTU_MARKET_FLOW_OUTPUT_DIR=/tmp/quantpilot_futu_market_flow_smoke \
MAJOR_MONEY_DIGEST_JSON=/tmp/major_money_digest_smoke.json \
MAJOR_MONEY_DIGEST_CSV=/tmp/major_money_digest_smoke.csv \
./scripts/run_market_capital_flow.sh
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
account.
