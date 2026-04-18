import pandas as pd

from inference import run_us_daily


def test_parse_trade_action_reads_explicit_marker():
    state = {"trade_proposal": "执行细节...\n最终交易提案: BUY"}
    assert run_us_daily.parse_trade_action(state) == "BUY"


def test_parse_rating_reads_markdown_rating():
    state = {"final_decision": "**评级**：Overweight\n\n其余内容"}
    assert run_us_daily.parse_rating(state) == "OVERWEIGHT"


def test_build_candidate_frame_filters_by_price_and_liquidity(monkeypatch):
    monkeypatch.setattr(run_us_daily, "US_MIN_PRICE", 5.0)
    monkeypatch.setattr(run_us_daily, "US_MIN_DOLLAR_VOLUME", 10_000_000.0)

    df = run_us_daily.build_candidate_frame(
        ["US.AAA", "US.BBB", "US.CCC"],
        {
            "US.AAA": {"last_price": 10.0, "turnover": 20_000_000.0, "change_rate": 1.0},
            "US.BBB": {"last_price": 3.0, "turnover": 20_000_000.0, "change_rate": 5.0},
            "US.CCC": {"last_price": 10.0, "turnover": 1_000_000.0, "change_rate": 5.0},
        },
    )

    assert df["code"].tolist() == ["US.AAA"]


def test_build_trade_plan_caps_target_count(monkeypatch):
    monkeypatch.setattr(run_us_daily, "US_MAX_POSITIONS", 2)
    analyses = [
        {"code": "US.A", "action": "BUY", "rating": "BUY", "decision_score": 2, "candidate_score": 10.0, "run_id": "1", "state_path": "/tmp/1", "trade_proposal": "a"},
        {"code": "US.B", "action": "BUY", "rating": "OVERWEIGHT", "decision_score": 1, "candidate_score": 9.0, "run_id": "2", "state_path": "/tmp/2", "trade_proposal": "b"},
        {"code": "US.C", "action": "BUY", "rating": "BUY", "decision_score": 2, "candidate_score": 8.0, "run_id": "3", "state_path": "/tmp/3", "trade_proposal": "c"},
    ]

    plan = run_us_daily.build_trade_plan(analyses, current_positions={})

    assert len(plan["target_codes"]) == 2
    assert set(plan["target_codes"]) == {"US.A", "US.C"}


def test_run_us_daily_requires_analysis_for_existing_positions(monkeypatch):
    monkeypatch.setattr(run_us_daily, "load_universe_codes", lambda: (["US.AAPL"], "test"))
    monkeypatch.setattr(run_us_daily, "get_us_positions", lambda: {"US.MSFT": {"qty": 1}})
    monkeypatch.setattr(run_us_daily, "get_us_snapshots", lambda codes: {"US.AAPL": {"last_price": 10.0, "turnover": 20_000_000.0, "change_rate": 1.0}, "US.MSFT": {"last_price": 20.0, "turnover": 20_000_000.0, "change_rate": 1.0}})
    monkeypatch.setattr(run_us_daily, "analyze_codes", lambda codes, scores: [{"code": "US.AAPL", "action": "BUY", "rating": "BUY", "decision_score": 2, "candidate_score": 10.0, "run_id": "1", "state_path": "/tmp/1", "trade_proposal": "a", "final_decision": "", "investment_plan": ""}])

    try:
        run_us_daily.run_us_daily()
    except RuntimeError as exc:
        assert "missing deep-analysis results" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_analyze_codes_preserves_input_order(monkeypatch):
    monkeypatch.setattr(run_us_daily, "US_ANALYSIS_CONCURRENCY", 10)

    def fake_analyze(code, scores):
        return {
            "code": code,
            "action": "BUY",
            "rating": "BUY",
            "decision_score": 2,
            "candidate_score": scores[code],
            "run_id": code,
            "state_path": f"/tmp/{code}",
            "trade_proposal": code,
            "final_decision": code,
            "investment_plan": code,
        }

    monkeypatch.setattr(run_us_daily, "_analyze_code", fake_analyze)
    results = run_us_daily.analyze_codes(["US.C", "US.A", "US.B"], {"US.A": 1.0, "US.B": 2.0, "US.C": 3.0})

    assert [item["code"] for item in results] == ["US.C", "US.A", "US.B"]
