"""Generate and print isolated Polymarket daily report artifacts."""
from __future__ import annotations

import argparse
import json

from polymarket.reporting.daily import generate_daily_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Polymarket daily report")
    parser.add_argument("--date", dest="target_date", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    payload, paths = generate_daily_report(target_date=args.target_date)

    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload['status']}")
        print(f"report_date: {payload['report_date']}")
        print(f"generated_at: {payload['generated_at']}")
        print(f"db_path: {payload['db_path']}")
        print(f"latest_artifact: {paths['latest']}")
        print(f"dated_artifact: {paths['dated']}")
        summary = payload.get("summary")
        if summary is None:
            print("summary: no_data")
        else:
            for key in (
                "signals",
                "accepted_signals",
                "simulated_trades",
                "gross_edge_sum",
                "net_edge_sum",
                "realized_pnl",
                "max_inventory_used",
                "fill_count",
                "opportunity_count",
                "market_count",
                "updated_at",
            ):
                print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
