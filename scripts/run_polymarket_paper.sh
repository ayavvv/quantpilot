#!/bin/bash
# Polymarket paper scanner — isolated from A-share automation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

EXTERNAL_DATA_DIR="${DATA_DIR-}"
EXTERNAL_POLY_DATA_DIR="${POLY_DATA_DIR-}"
EXTERNAL_POLY_SCAN_INTERVAL_SECONDS="${POLY_SCAN_INTERVAL_SECONDS-}"
EXTERNAL_POLY_CATALOG_REFRESH_SECONDS="${POLY_CATALOG_REFRESH_SECONDS-}"
EXTERNAL_POLY_MIN_NET_EDGE="${POLY_MIN_NET_EDGE-}"
EXTERNAL_POLY_DEFAULT_GAS_COST="${POLY_DEFAULT_GAS_COST-}"
EXTERNAL_POLY_SLIPPAGE_BUFFER="${POLY_SLIPPAGE_BUFFER-}"
EXTERNAL_POLY_PAPER_ONLY="${POLY_PAPER_ONLY-}"
EXTERNAL_POLY_ENABLE_SPLIT_SELL="${POLY_ENABLE_SPLIT_SELL-}"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

[ -n "$EXTERNAL_DATA_DIR" ] && DATA_DIR="$EXTERNAL_DATA_DIR"
[ -n "$EXTERNAL_POLY_DATA_DIR" ] && POLY_DATA_DIR="$EXTERNAL_POLY_DATA_DIR"
[ -n "$EXTERNAL_POLY_SCAN_INTERVAL_SECONDS" ] && POLY_SCAN_INTERVAL_SECONDS="$EXTERNAL_POLY_SCAN_INTERVAL_SECONDS"
[ -n "$EXTERNAL_POLY_CATALOG_REFRESH_SECONDS" ] && POLY_CATALOG_REFRESH_SECONDS="$EXTERNAL_POLY_CATALOG_REFRESH_SECONDS"
[ -n "$EXTERNAL_POLY_MIN_NET_EDGE" ] && POLY_MIN_NET_EDGE="$EXTERNAL_POLY_MIN_NET_EDGE"
[ -n "$EXTERNAL_POLY_DEFAULT_GAS_COST" ] && POLY_DEFAULT_GAS_COST="$EXTERNAL_POLY_DEFAULT_GAS_COST"
[ -n "$EXTERNAL_POLY_SLIPPAGE_BUFFER" ] && POLY_SLIPPAGE_BUFFER="$EXTERNAL_POLY_SLIPPAGE_BUFFER"
[ -n "$EXTERNAL_POLY_PAPER_ONLY" ] && POLY_PAPER_ONLY="$EXTERNAL_POLY_PAPER_ONLY"
[ -n "$EXTERNAL_POLY_ENABLE_SPLIT_SELL" ] && POLY_ENABLE_SPLIT_SELL="$EXTERNAL_POLY_ENABLE_SPLIT_SELL"

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
POLY_DATA_DIR="${POLY_DATA_DIR:-$DATA_DIR/polymarket}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
source .venv/bin/activate

POLY_DATA_DIR="$POLY_DATA_DIR" \
POLY_PAPER_ONLY="${POLY_PAPER_ONLY:-true}" \
POLY_ENABLE_SPLIT_SELL="${POLY_ENABLE_SPLIT_SELL:-true}" \
POLY_SCAN_INTERVAL_SECONDS="${POLY_SCAN_INTERVAL_SECONDS:-5}" \
POLY_CATALOG_REFRESH_SECONDS="${POLY_CATALOG_REFRESH_SECONDS:-900}" \
POLY_MIN_NET_EDGE="${POLY_MIN_NET_EDGE:-0.01}" \
POLY_DEFAULT_GAS_COST="${POLY_DEFAULT_GAS_COST:-0}" \
POLY_SLIPPAGE_BUFFER="${POLY_SLIPPAGE_BUFFER:-0.005}" \
PYTHONPATH="$PYTHONPATH" \
"$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone

from polymarket.pipeline import PolymarketPipeline
from polymarket.reporting.daily import generate_daily_report

result = PolymarketPipeline().run_once()
target_date = datetime.now(timezone.utc).date().isoformat()
payload, paths = generate_daily_report(target_date=target_date)
print(
    f"polymarket paper run complete: markets={result.markets_seen} "
    f"opportunities={result.opportunities_found} trades={result.trades_simulated}"
)
print(f"polymarket report status: {payload['status']}")
print(f"polymarket report date: {payload['report_date']}")
print(f"polymarket latest artifact: {paths['latest']}")
print(f"polymarket dated artifact: {paths['dated']}")
PY

LATEST_REPORT="$POLY_DATA_DIR/reports/daily_summary_latest.json"
if [ ! -f "$LATEST_REPORT" ]; then
    echo "Missing Polymarket latest report artifact: $LATEST_REPORT" >&2
    exit 1
fi

if ! grep -q '"status": "ok"' "$LATEST_REPORT"; then
    echo "Polymarket report artifact does not contain status=ok: $LATEST_REPORT" >&2
    exit 1
fi

echo "polymarket data dir: $POLY_DATA_DIR"
echo "polymarket verified report artifact: $LATEST_REPORT"
