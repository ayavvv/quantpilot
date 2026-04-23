from pathlib import Path

from polymarket.config import PolySettings
from polymarket.reporting.email import build_email_subject, render_email_html, send_daily_report_email


def test_build_email_subject_includes_pnl_for_ok_payload():
    payload = {
        "status": "ok",
        "report_date": "2026-04-19",
        "summary": {"realized_pnl": 12.345},
    }

    subject = build_email_subject(payload)

    assert subject == "Polymarket Daily Report 2026-04-19 | PnL 12.35"


def test_render_email_html_handles_no_data_payload():
    payload = {
        "status": "no_data",
        "report_date": "2026-04-19",
        "generated_at": "2026-04-20T00:05:00+00:00",
        "summary": None,
        "mirror_summary": None,
    }

    html = render_email_html(payload)

    assert "Polymarket Daily Report - 2026-04-19" in html
    assert "No Polymarket data was available for this report date." in html


def test_send_daily_report_email_uses_shared_sender(tmp_path, monkeypatch):
    calls = {}

    def fake_send_email(html_content, subject, report_filename=None, report_dir=None, attachment_paths=None):
        calls["html_content"] = html_content
        calls["subject"] = subject
        calls["report_filename"] = report_filename
        calls["report_dir"] = report_dir
        calls["attachment_paths"] = attachment_paths
        return True

    monkeypatch.setattr("polymarket.reporting.email.send_email", fake_send_email)

    cfg = PolySettings(data_dir=str(tmp_path), email_report_enabled=True, email_report_attach_json=True)
    dated = cfg.reports_path / "daily_summary_2026-04-19.json"
    dated.parent.mkdir(parents=True, exist_ok=True)
    dated.write_text("{}", encoding="utf-8")

    payload = {
        "status": "ok",
        "report_date": "2026-04-19",
        "generated_at": "2026-04-20T00:05:00+00:00",
        "summary": {
            "signals": 1,
            "accepted_signals": 1,
            "simulated_trades": 1,
            "gross_edge_sum": 0.02,
            "net_edge_sum": 0.02,
            "realized_pnl": 5.0,
            "max_inventory_used": 0.98,
            "fill_count": 2,
            "opportunity_count": 1,
            "market_count": 1,
            "updated_at": "2026-04-19T23:59:00+00:00",
        },
        "mirror_summary": None,
    }
    paths = {"dated": Path(dated), "latest": cfg.reports_path / "daily_summary_latest.json"}

    sent = send_daily_report_email(payload, paths, cfg=cfg)

    assert sent is True
    assert calls["subject"] == "Polymarket Daily Report 2026-04-19 | PnL 5.00"
    assert calls["report_filename"] == "polymarket_email_report_2026-04-19.html"
    assert calls["report_dir"] == cfg.reports_path
    assert calls["attachment_paths"] == [dated]
    assert "Realized PnL" in calls["html_content"]
