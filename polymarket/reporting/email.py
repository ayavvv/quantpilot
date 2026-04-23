"""Email rendering for isolated Polymarket daily reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from polymarket.config import PolySettings, settings
from reporter.send_report import send_email

REPORT_TEMPLATE = """
<html>
<head>
<style>
body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 0 auto; padding: 20px; }
h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }
h2 { color: #16213e; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #16213e; color: white; }
.metric { display: inline-block; margin: 8px 16px 8px 0; }
.metric-value { font-size: 24px; font-weight: bold; color: #16213e; }
.metric-label { font-size: 12px; color: #666; }
.ok { color: #28a745; }
.warn { color: #ffc107; }
.error { color: #dc3545; }
</style>
</head>
<body>
<h1>Polymarket Daily Report - {{ report_date }}</h1>

<div class="metric">
    <div class="metric-value {{ status_class }}">{{ status }}</div>
    <div class="metric-label">Report Status</div>
</div>
{% if summary %}
<div class="metric">
    <div class="metric-value">{{ realized_pnl }}</div>
    <div class="metric-label">Realized PnL</div>
</div>
<div class="metric">
    <div class="metric-value">{{ simulated_trades }}</div>
    <div class="metric-label">Simulated Trades</div>
</div>
<div class="metric">
    <div class="metric-value">{{ fill_count }}</div>
    <div class="metric-label">Fill Count</div>
</div>
{% endif %}

<h2>Strategy Summary</h2>
{% if summary %}
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Signals</td><td>{{ summary.signals }}</td></tr>
<tr><td>Accepted Signals</td><td>{{ summary.accepted_signals }}</td></tr>
<tr><td>Simulated Trades</td><td>{{ summary.simulated_trades }}</td></tr>
<tr><td>Gross Edge Sum</td><td>{{ summary.gross_edge_sum }}</td></tr>
<tr><td>Net Edge Sum</td><td>{{ summary.net_edge_sum }}</td></tr>
<tr><td>Realized PnL</td><td>{{ summary.realized_pnl }}</td></tr>
<tr><td>Max Inventory Used</td><td>{{ summary.max_inventory_used }}</td></tr>
<tr><td>Fill Count</td><td>{{ summary.fill_count }}</td></tr>
<tr><td>Opportunity Count</td><td>{{ summary.opportunity_count }}</td></tr>
<tr><td>Market Count</td><td>{{ summary.market_count }}</td></tr>
<tr><td>Updated At</td><td>{{ summary.updated_at }}</td></tr>
</table>
{% else %}
<p class="warn">No Polymarket data was available for this report date.</p>
{% endif %}

{% if mirror_summary %}
<h2>Top Trader Mirror</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Tracked Traders</td><td>{{ mirror_summary.tracked_traders }}</td></tr>
<tr><td>Signals</td><td>{{ mirror_summary.signals }}</td></tr>
<tr><td>Accepted Signals</td><td>{{ mirror_summary.accepted_signals }}</td></tr>
<tr><td>Simulated Trades</td><td>{{ mirror_summary.simulated_trades }}</td></tr>
<tr><td>Realized PnL</td><td>{{ mirror_summary.realized_pnl }}</td></tr>
<tr><td>Max Inventory Used</td><td>{{ mirror_summary.max_inventory_used }}</td></tr>
<tr><td>Updated At</td><td>{{ mirror_summary.updated_at }}</td></tr>
</table>
{% endif %}

<hr>
<p style="color: #999; font-size: 12px;">
Polymarket Auto Report | Generated: {{ generated_at }}
</p>
</body>
</html>
"""


def build_email_subject(payload: dict[str, Any]) -> str:
    if payload["status"] != "ok" or payload.get("summary") is None:
        return f"Polymarket Daily Report {payload['report_date']} | no_data"
    realized_pnl = float(payload["summary"]["realized_pnl"])
    return f"Polymarket Daily Report {payload['report_date']} | PnL {realized_pnl:.2f}"


def render_email_html(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    template = Template(REPORT_TEMPLATE)
    status = payload["status"]
    status_class = "ok" if status == "ok" else "warn"
    return template.render(
        report_date=payload["report_date"],
        generated_at=payload["generated_at"],
        status=status,
        status_class=status_class,
        summary=summary,
        mirror_summary=payload.get("mirror_summary"),
        realized_pnl=f"{float(summary['realized_pnl']):.2f}" if summary else "N/A",
        simulated_trades=summary["simulated_trades"] if summary else 0,
        fill_count=summary["fill_count"] if summary else 0,
    )


def send_daily_report_email(
    payload: dict[str, Any],
    paths: dict[str, Path],
    cfg: PolySettings | None = None,
) -> bool:
    cfg = cfg or settings
    report_filename = f"polymarket_email_report_{payload['report_date']}.html"
    attachments: list[Path] = []
    if cfg.email_report_attach_json:
        attachments.append(Path(paths["dated"]))
        mirror_dated = cfg.mirror_reports_path / f"mirror_daily_summary_{payload['report_date']}.json"
        if mirror_dated.exists():
            attachments.append(mirror_dated)
    return send_email(
        render_email_html(payload),
        build_email_subject(payload),
        report_filename=report_filename,
        report_dir=cfg.reports_path,
        attachment_paths=attachments,
    )
