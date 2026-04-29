import json

from inference import run_us_watchlist_report as watch_report


def test_save_watchlist_normalizes_symbols_and_dedupes(tmp_path):
    path = tmp_path / "us_watchlist.json"
    watch_report.save_watchlist(
        {
            "enabled": True,
            "symbols": [
                {"symbol": "us.aapl", "name": "Apple", "enabled": True, "notes": "core"},
                {"symbol": "AAPL", "name": "Duplicate", "enabled": True, "notes": ""},
                {"symbol": "brk-b", "name": "Berkshire", "enabled": False, "notes": ""},
                {"symbol": "", "name": "Empty", "enabled": True, "notes": ""},
            ],
            "analysis": {"concurrency": 2, "timeout_seconds": 3600, "retry_count": 0},
        },
        path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert [item["symbol"] for item in payload["symbols"]] == ["AAPL", "BRK.B"]
    assert payload["symbols"][0]["enabled"] is True
    assert payload["symbols"][1]["enabled"] is False


def test_enabled_items_returns_us_codes():
    payload = {
        "symbols": [
            {"symbol": "LI", "enabled": True, "name": "Li Auto"},
            {"symbol": "US.SPY", "enabled": True, "name": "SPY"},
            {"symbol": "MSFT", "enabled": False, "name": "Microsoft"},
        ]
    }

    items = watch_report.enabled_items(payload)

    assert items == [
        {"symbol": "LI", "code": "US.LI", "name": "Li Auto", "notes": ""},
        {"symbol": "SPY", "code": "US.SPY", "name": "SPY", "notes": ""},
    ]


def test_analyze_watchlist_skips_remaining_after_budget_error(monkeypatch):
    monkeypatch.setattr(watch_report.us_daily, "US_ANALYSIS_CONCURRENCY", 1)
    calls = []

    def fake_analyze(code, scores, expected_date):
        calls.append(code)
        raise RuntimeError("API Error: Request rejected (429) · daily budget exceeded")

    monkeypatch.setattr(watch_report, "_analyze_watch_code", fake_analyze)
    items = [
        {"symbol": "AAPL", "code": "US.AAPL", "name": "", "notes": ""},
        {"symbol": "MSFT", "code": "US.MSFT", "name": "", "notes": ""},
        {"symbol": "NVDA", "code": "US.NVDA", "name": "", "notes": ""},
    ]

    analyses, failures, skipped = watch_report.analyze_watchlist(items, "2026-04-29")

    assert analyses == []
    assert len(failures) == 1
    assert [item["code"] for item in skipped] == ["US.MSFT", "US.NVDA"]
    assert calls == ["US.AAPL"]


def test_run_watchlist_report_writes_json_and_html_without_email(monkeypatch, tmp_path):
    watchlist_path = tmp_path / "config" / "us_watchlist.json"
    report_dir = tmp_path / "reports"
    watch_report.save_watchlist(
        {
            "enabled": True,
            "symbols": [{"symbol": "AAPL", "enabled": True, "name": "Apple", "notes": ""}],
            "analysis": {"concurrency": 1, "timeout_seconds": 3600, "retry_count": 0},
        },
        watchlist_path,
    )
    monkeypatch.setattr(watch_report, "WATCHLIST_FILE", watchlist_path)
    monkeypatch.setattr(watch_report, "REPORT_DIR", report_dir)
    monkeypatch.setattr(watch_report, "SEND_EMAIL", False)

    def fake_analyze(items, expected_date):
        return (
            [
                {
                    "code": "US.AAPL",
                    "symbol": "AAPL",
                    "name": "Apple",
                    "action": "HOLD",
                    "rating": "HOLD",
                    "state_path": "/tmp/aapl_state.json",
                    "final_decision": "评级: Hold",
                    "investment_plan": "Hold",
                    "trade_proposal": "最终交易提案: HOLD",
                    "analyst_summary": "summary",
                    "reports": {"market": "market"},
                    "debate_history": [],
                    "risk_history": [],
                }
            ],
            [],
            [],
        )

    monkeypatch.setattr(watch_report, "analyze_watchlist", fake_analyze)
    result = watch_report.run_watchlist_report()

    assert result["analysis_count"] == 1
    assert (report_dir / "us_watchlist_report_latest.json").exists()
    assert "us_watchlist_report_" in result["report"]
