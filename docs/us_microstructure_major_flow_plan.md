# US Microstructure Major-Flow Plan

## Objective

Build a daily US-stock report for high-confidence "major-flow" candidates using
Futu OpenD real-time tick prints and order-book snapshots. The report should
surface probable stealth accumulation and distribution behavior, not claim
account-level proof of institutional buying or selling.

## Current Data Reality

Verified on the Mac mini with Futu OpenD listening on `127.0.0.1:11111`.

- `SubType.TICKER` works for US symbols after enabling the local RSA key.
- `get_rt_ticker("US.AAPL", num=...)` returns recent prints with:
  `code`, `name`, `time`, `price`, `volume`, `turnover`,
  `ticker_direction`, `sequence`, `type`.
- `get_rt_ticker` signature is `get_rt_ticker(code, num=500)`.
  A request above 1000 rows still returns 1000 rows.
- `SubType.ORDER_BOOK` works for US symbols.
- `get_order_book("US.AAPL", num=10)` returns top bid/ask levels with price
  and size.
- `SubType.QUOTE`, `get_stock_quote`, and `get_market_snapshot` work and give
  bid/ask, day volume, pre/after/overnight fields, and other quote fields.
- Local Mac storage currently has no US tick/order-book history under
  `~/quantpilot_data`; it has only daily US ETF bars, US reports/signals, and
  Futu-derived daily capital-flow artifacts.
- The NAS has ample storage and should be the authoritative long-term archive:
  `/volume1/docker/quantpilot/us_microstructure/`. A quick capacity check on
  2026-06-01 showed about 16T total and 14T available on `/volume1/docker`.

Important limitation:

- Futu OpenD gives current/recent tick and current order-book data. It is not a
  historical full-tape replay API for arbitrary past dates. A proper
  microstructure backtest cannot be done today without first collecting live
  data going forward.

## Report Standard

The daily report should distinguish three states:

- `warmup`: data is being collected; signals are diagnostic only.
- `validated`: enough forward samples exist and the rule passed promotion gates.
- `disabled`: data coverage was insufficient or the model failed validation.

Only `validated` signals should be described as high-confidence. During warmup,
the report can still show candidates, but it must label them as experimental and
must not present probabilities as calibrated.

## Collection Universe

Do not try to collect the whole US market at the start. Even though current
subscription capacity showed roughly 1000 ticker slots, whole-market tick and
book collection would be noisy and storage-heavy.

Recommended tiers:

1. Core watchlist, always on:
   `SPY`, `QQQ`, `IWM`, `DIA`, `LI`, `YINN`, `CQQQ`, `KWEB`, `FXI`,
   `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMD`, `AMZN`, `META`, `GOOGL`,
   `AVGO`, `SMCI`, `COIN`, `MSTR`.
2. Dynamic liquid universe:
   Top 100-300 US names by previous-day dollar volume, current premarket
   turnover, gap, and existing Futu daily capital-flow anomaly.
3. Event universe:
   Earnings/news/high-gap names, capped separately so event names do not crowd
   out the liquid core.

Start with 50-100 names until data loss, CPU, and disk write rate are measured.
Expand only after coverage is stable.

## Storage Architecture

Futu OpenD runs on the Mac mini, so the Mac should collect live data. The NAS
should store the long-lived raw archive.

Recommended layout:

```text
Mac mini hot cache:
~/quantpilot_data/us_microstructure/
  raw_buffer/
  features_1m/
  signals/
  quality/
  validation/
  readiness/
  logs/

NAS authoritative archive:
/volume1/docker/quantpilot/us_microstructure/
  trades/
  order_book/
  quotes/
  features_1m/
  signals/
  quality/
  validation/
  readiness/
  logs/
```

Write path:

1. Collector writes small parquet batches to the Mac hot cache during the
   session.
2. Every few minutes, completed batches are copied to NAS with an `ssh+tar`
   stream over `ssh nas`.
3. After each copy, the collector writes a manifest row and mirrors the
   manifest to NAS:
   file path, row count, min/max time, sha256, NAS path, upload status.
4. Files older than a configurable retention window, for example 7-14 days, can
   be removed from Mac only after the manifest confirms the NAS copy.

This avoids using a network filesystem as the direct real-time write target.
If Wi-Fi/LAN/NAS hiccups during the US session, the collector keeps writing to
local disk and catches up when the NAS is reachable.

NAS retention:

- Keep raw trades/order books indefinitely at first.
- Compact older partitions from symbol-level tiny files into daily market-level
  parquet datasets once disk and query patterns are measured.
- Keep one-minute features and signal tables forever; these are small and
  needed for validation.

## Raw Data Schema

Write partitioned parquet files under:

```text
~/quantpilot_data/us_microstructure/
  trades/date=YYYY-MM-DD/symbol=US.AAPL/part-*.parquet
  order_book/date=YYYY-MM-DD/symbol=US.AAPL/part-*.parquet
  quotes/date=YYYY-MM-DD/symbol=US.AAPL/part-*.parquet
  features_1m/date=YYYY-MM-DD/part-*.parquet
  signals/date=YYYY-MM-DD/us_major_flow_signals.csv
  validation/
```

The same partitioning should exist under
`/volume1/docker/quantpilot/us_microstructure/` on the NAS.

Trades:

- `event_time`, `recv_time`, `symbol`, `price`, `volume`, `turnover`
- `ticker_direction`, `sequence`, `type`, `session`

Order book:

- `event_time`, `recv_time`, `symbol`
- `bid_px_1..10`, `bid_sz_1..10`, `ask_px_1..10`, `ask_sz_1..10`
- derived fields: `mid`, `spread_bps`, `depth_imbalance_1`,
  `depth_imbalance_5`, `bid_replenish_rate`, `ask_replenish_rate`

Quotes:

- `event_time`, `recv_time`, `symbol`, `last_price`, `bid_price`,
  `ask_price`, `bid_vol`, `ask_vol`, `volume`, `turnover`, session fields

Use `sequence` to de-duplicate trades. Record receive time even when Futu
timestamps are stale or regular-session only.

## Feature Design

Compute one-minute and session-to-date features.

Tape pressure:

- `active_buy_dollar`, `active_sell_dollar`, `net_active_dollar`
- `active_buy_ratio`
- `neutral_dollar_ratio`
- `trade_count_z`, `dollar_volume_z`
- `avg_trade_size`, `small_trade_streak`, `odd_lot_ratio`

Order-book absorption:

- `spread_bps`
- `depth_imbalance_1`, `depth_imbalance_5`
- `bid_replenish_after_sell`
- `ask_replenish_after_buy`
- `book_depletion_speed`
- `same_price_volume_cluster`

Price impact:

- `price_impact_per_1m_dollar`
- `vwap_deviation_bps`
- `anchored_vwap_deviation_bps`
- `range_bps`
- `realized_vol_5m`

Context:

- `rel_volume_intraday`
- `premarket_gap`
- `market_beta_proxy` against `SPY` or `QQQ`
- existing daily capital-flow labels when available

## Signal Definitions

### Stealth Accumulation

Candidate behavior:

- Sustained positive or absorbed selling flow over 15-60 minutes.
- High dollar volume relative to normal, but low price impact.
- Price holds above or near intraday VWAP.
- Bid replenishes after sell prints, or buy prints repeatedly fail to push
  price far above VWAP.
- Spread remains controlled and depth does not collapse.

High-confidence gate before validation:

- Data coverage at least 80% of the regular session for the symbol, measured
  only from 09:30-16:00 US Eastern minutes.
- Trade coverage and order-book coverage must each meet the minimum, so a
  candidate cannot become high-confidence from prints alone when book evidence
  is missing.
- Minimum regular-session dollar volume, default 50m USD.
- Minimum collected trade count, default 1000 prints.
- At least two independent evidence blocks agree:
  tape pressure, order-book absorption, price-impact control.
- No signal if the move is already overheated:
  extreme gap plus high impact plus weak book support.

### Distribution / Exit Risk

Candidate behavior:

- Sustained negative active flow or aggressive sell pressure.
- Bid depth repeatedly disappears or fails to replenish.
- Large turnover below VWAP or repeated failed VWAP reclaim.
- Price impact is high on sell bursts, or price refuses to bounce despite
  high volume.

## Scoring

Use two scores:

- `accumulation_score`: 0-100
- `distribution_score`: 0-100

Initial weights should be conservative:

- 30% tape pressure
- 30% order-book absorption/replenishment
- 25% low price impact and VWAP behavior
- 15% liquidity/session/context quality

Confidence labels:

- `high`: score >= 85 and validation gate is active.
- `watch`: score >= 70 but not promoted.
- `diagnostic`: warmup only or insufficient validation.

The daily email should show only `high` in the subject. `watch` can live in the
body during warmup with explicit labeling.

## Validation and Backtest Plan

Historical microstructure backtest is not available now because there is no
local US tick/order-book archive and Futu OpenD is not an arbitrary historical
tick replay source.

This means the first production-quality claim cannot be "backtested over the
past year." The honest substitute is a forward paper-trading validation ledger
fed by the new NAS archive. Until that ledger has enough samples, the report is
warmup/diagnostic.

Therefore validation must be forward-collected:

1. Collect raw trades/book/quotes every US session.
2. Build features and candidate signals after the close.
3. Store labels:
   - next 30m return, if signal occurs intraday and still tradable
   - close-to-next-open return
   - next 1d, 3d, 5d return
   - alpha versus `SPY` and `QQQ`
   - max favorable/adverse excursion
4. Promote rules only with walk-forward validation.

Promotion gates:

- At least 20 signal days per side.
- At least 100 symbol-signal observations per side.
- Out-of-sample 5d alpha >= 0.75% after conservative slippage.
- Out-of-sample hit rate >= 58%.
- Recent 20-trading-day hit rate >= 55%.
- Wilson lower bound of hit rate above 50%, or equivalent bootstrap confidence.
- No single ticker contributes more than 20% of the validation sample.

Before these gates pass, the system may send a warmup report but must not call
signals high-confidence.

Data-quality gates:

- At least 95% collector process uptime during regular session.
- At least 80% expected one-minute bars with both trade and book data per
  reported symbol.
- Trade sequence duplicate rate below 1% after de-duplication.
- Median order-book snapshot interval within the configured sampling interval
  plus 2 seconds.
- NAS upload completeness at 100% for all files used in the report.

Slippage assumptions:

- For end-of-day signals, use next regular-session open plus 5 bps.
- For intraday signal studies, use mid/ask crossing from the next quote snapshot
  after signal time plus at least 3 bps.
- Reject signals whose quoted spread is wider than 20 bps unless the symbol is
  explicitly in an event basket.

Forward validation tables:

```text
validation/signal_events.parquet
validation/forward_returns.parquet
validation/rule_metrics.json
validation/active_gate.json
```

`active_gate.json` is the only file the daily high-confidence report may trust.
If it is missing, stale, or failed, the report stays in warmup mode.

## Daily Workflow

China-time regular US session collection:

- Summer time: roughly 21:30-04:00.
- Winter time: roughly 22:30-05:00.

Jobs:

1. Before US open:
   build the collection universe.
2. During session:
   subscribe to `TICKER`, `ORDER_BOOK`, `QUOTE`; write raw parquet batches.
   Upload completed batches to NAS asynchronously.
3. After US close:
   aggregate one-minute features, compute signals, update validation labels
   when future returns become available. Store final features and signals on
   both Mac and NAS.
4. Morning China time:
   send the report.

Suggested cron shape on Mac mini:

```text
# Build universe before US open
20:45 Asia/Shanghai summer, 21:45 winter

# Collect during regular US session
21:25-04:10 summer, 22:25-05:10 winter

# Aggregate and score after US close
04:15 summer, 05:15 winter

# Send China-morning report
08:30 Asia/Shanghai
```

The exact schedule should be generated with exchange-calendar logic, not hard
coded, because US daylight saving time and US holidays do not align with China
holidays.

## Report Content

Subject examples:

- Warmup: `US Microstructure Flow - warmup, 0 validated`
- Validated: `US Micro Flow - 3 accumulation / 2 distribution`

Body sections:

- Data coverage:
  symbols collected, trade rows, book snapshots, missing symbols.
- High-confidence candidates:
  symbol, side, score, confidence, data coverage, evidence blocks.
- Evidence table:
  net active dollar, passive absorption score, VWAP behavior, price impact,
  depth imbalance, dollar volume z-score.
- Validation status:
  sample size, hit rate, alpha, recent hit rate, current gate status.
- Warmup candidates:
  optional, clearly marked as not validated.

## Implementation Phases

Phase 1: collector smoke

- Build a standalone collector for 20-50 symbols.
- Store trades, order book snapshots, and quotes.
- Upload completed batches to NAS.
- Run for one full US session.
- Verify row counts, timestamp quality, duplicate rate, Mac disk footprint, NAS
  upload completeness, and manifest checksums.

Phase 2: feature and warmup report

- Aggregate one-minute features.
- Generate diagnostic candidates.
- Send a warmup report with no high-confidence language.

Phase 3: forward validation

- Run for at least 20 trading days.
- Evaluate next-day and 5-day outcomes.
- Tune thresholds only on train windows and validate on forward windows.

Phase 4: production gate

- Promote only rules that pass gates.
- Daily report subject includes only promoted high-confidence signals.

## First Implementation Target

Build:

- `scripts/collect_us_microstructure.py`
- `scripts/run_us_microstructure_collect.sh`
- `strategy/us_microstructure_features.py`
- `strategy/us_microstructure_signals.py`
- `strategy/us_microstructure_validation.py`
- `scripts/report_us_microstructure_flow.py`
- `scripts/update_us_microstructure_prices.py`
- `scripts/validate_us_microstructure_flow.py`
- `scripts/us_microstructure_readiness.py`
- `scripts/run_us_microstructure_report.sh`
- `deploy/launchd/com.quantpilot.us_microstructure.collect.plist`
- `deploy/launchd/com.quantpilot.us_microstructure.report.plist`
- `scripts/install_us_microstructure_launchd.sh`

Implemented status as of 2026-06-01:

- `scripts/collect_us_microstructure.py` collects Futu `TICKER`,
  `ORDER_BOOK`, and `QUOTE` data into local parquet batches and mirrors them to
  NAS with `ssh+tar`. Ticker rows whose Futu event date does not match the
  collection date are skipped before de-duplication, because OpenD can return
  the previous trading day's prints before the US open. The collection
  partition date is derived from US Eastern time rather than the China local
  calendar date, so a collector restart after China midnight still writes to
  the correct US session partition.
- `scripts/run_us_microstructure_collect.sh` is the Mac-side collection
  wrapper. It loads `.env`, applies a lock, uses
  `config/us_microstructure_core_symbols.txt` by default, and runs the collector
  for the configured session duration.
- `strategy/us_microstructure_features.py` aggregates raw trades, order book,
  and quotes into one-minute tape/book/impact features. Futu trade timestamps
  are interpreted as US Eastern time and normalized to UTC so they align with
  collector receive-time book snapshots. It also writes separate trade,
  order-book, quote, and combined regular-session coverage ratios. Coverage and
  regular-session scoring only count 09:30-16:00 US Eastern minutes; premarket
  and after-hours rows remain available as context but cannot lift
  high-confidence coverage. As a second guard, the reader filters stale trade
  rows out of date partitions if they were collected before this protection
  existed.
- `strategy/us_microstructure_signals.py` scores accumulation and distribution
  candidates, but only emits `high` confidence when a validation gate is
  promoted for that side and both regular-session trade and order-book coverage
  gates pass. Without `validation/active_gate.json`, candidates stay
  `warmup`/`diagnostic` or `watch`.
- `strategy/us_microstructure_validation.py` maintains the forward validation
  ledger: `signal_events.parquet`, `forward_returns.parquet`,
  `rule_metrics.csv`, and `active_gate.json`. It promotes only after the
  configured sample-size, 5-day alpha, hit-rate, recent hit-rate, Wilson lower
  bound, and symbol-concentration gates pass. The ledger only consumes
  reportable `watch`/`high` signals with `data_quality_pass=true`; diagnostic
  or low-coverage rows are not allowed to train the confidence gate. Signal
  events preserve coverage, liquidity, duplicate-rate, spread, and evidence
  block fields so every validation sample remains auditable.
- `scripts/update_us_microstructure_prices.py` uses Futu OpenD `K_DAY` data to
  maintain `validation/prices/us_daily_prices.csv` and parquet. This is the
  daily close-price source used to turn signal events into forward-return labels.
- `scripts/validate_us_microstructure_flow.py` updates the validation ledger
  from archived signal CSV files and daily close prices. It can read a
  `date,symbol,close` price CSV and/or Qlib daily close data. By default it
  auto-detects `validation/prices/us_daily_prices.csv`.
- `scripts/report_us_microstructure_flow.py` writes feature parquet, signal CSV,
  per-symbol data-quality CSV, status JSON, and Markdown/HTML reports, then
  mirrors report artifacts to NAS. The status JSON and CSV record which symbols
  are eligible for high-confidence reporting based on regular-session coverage,
  liquidity, duplicate sequence rate, and spread. The CSV is also attached to
  emailed reports for daily audit.
- `scripts/us_microstructure_readiness.py` checks the latest manifest, price
  feed, validation gate, report artifacts, data-quality gate, and launchd
  services. When run from the daily wrapper it writes dated/latest readiness
  JSON snapshots locally and mirrors them to the NAS `readiness/` archive. The
  snapshot separates pipeline health (`ok`) from `high_confidence_ready`, which
  requires both a promoted validation gate and a passing report data-quality
  gate. Manifest readiness also audits per-symbol channel coverage across
  trades, order-book, and quotes so a missing data stream is visible before it
  reaches scoring.
- `scripts/run_us_microstructure_report.sh` is the Mac-side entrypoint for cron
  or launchd. It updates daily prices, updates validation, generates the
  report, then writes the readiness snapshot. If `US_MICROSTRUCTURE_DATE` is
  not set, it resolves the latest available collection partition instead of
  using the China morning calendar date, so the 08:30 report reads the previous
  evening's US session. Validation defaults to ending one calendar day before
  the report date, because validation runs before the current day's final report
  is written; this prevents intraday/manual same-date signal files from entering
  the forward ledger. It sends email only when `US_MICROSTRUCTURE_SEND_EMAIL=true`.
- Launchd templates are available for weekday evening collection and
  China-morning report generation. `scripts/install_us_microstructure_launchd.sh`
  renders both templates into `/Library/LaunchDaemons` when passwordless sudo is
  available, or user `LaunchAgents` otherwise.

Start with a fixed universe file:

```text
~/quantpilot_data/us_microstructure/universe/core_watchlist.txt
```

This keeps the first version operationally small and makes validation honest.

Smoke-test command shape:

```bash
PYTHONPATH=/Users/theo/quantpilot \
.venv/bin/python -m scripts.collect_us_microstructure \
  --symbols US.AAPL,US.NVDA,US.SPY \
  --duration-seconds 300 \
  --book-interval-seconds 1 \
  --quote-interval-seconds 5 \
  --local-dir ~/quantpilot_data/us_microstructure \
  --nas-dir /volume1/docker/quantpilot/us_microstructure \
  --nas-host nas
```

The five-minute smoke must prove:

- Futu reconnects with RSA encryption.
- Tick rows are de-duplicated by `symbol + sequence`.
- Book rows are captured at the configured interval.
- Local parquet files are readable.
- Files are copied to NAS and manifest checksums match.
