import json

import pandas as pd
from zoneinfo import ZoneInfo

from trader import trade_us_daily


class _QuoteCtx:
    def __init__(self, snapshots_by_code):
        self.snapshots_by_code = snapshots_by_code

    def get_market_snapshot(self, codes):
        rows = []
        for code in codes:
            row = {"code": code}
            row.update(self.snapshots_by_code.get(code, {}))
            rows.append(row)
        return trade_us_daily.RET_OK, pd.DataFrame(rows)


def test_resolve_dry_run_forces_preview_outside_us_session(monkeypatch):
    monkeypatch.setattr(trade_us_daily, "ALLOW_OFF_HOURS_TRADING", False)
    dry_run, reason = trade_us_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-04-20 08:00:00", tz=ZoneInfo("America/New_York")).to_pydatetime(),
    )

    assert dry_run is True
    assert "不在美股常规交易时段" in reason


def test_resolve_dry_run_allows_live_during_us_session(monkeypatch):
    monkeypatch.setattr(trade_us_daily, "ALLOW_OFF_HOURS_TRADING", False)
    dry_run, reason = trade_us_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-04-20 10:00:00", tz=ZoneInfo("America/New_York")).to_pydatetime(),
    )

    assert dry_run is False
    assert reason is None


def test_build_signals_df_keeps_buy_and_hold_targets():
    plan = {
        "generated_at": "2026-04-20T20:00:00",
        "orders": [
            {"code": "US.AAPL", "action": "BUY", "target_weight": 0.5},
            {"code": "US.MSFT", "action": "HOLD", "target_weight": 0.5},
            {"code": "US.NVDA", "action": "SELL", "target_weight": 0.0},
        ],
    }

    df, signal_date = trade_us_daily.build_signals_df(plan)

    assert signal_date == "2026-04-20"
    assert df["code"].tolist() == ["US.AAPL", "US.MSFT"]


def test_load_trade_plan_requires_orders(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"orders": []}), encoding="utf-8")

    try:
        trade_us_daily.load_trade_plan(path)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_signals_df_allows_full_sell_plan():
    plan = {
        "generated_at": "2026-04-20T20:00:00",
        "orders": [
            {"code": "US.AAPL", "action": "SELL", "target_weight": 0.0},
        ],
    }

    df, signal_date = trade_us_daily.build_signals_df(plan)

    assert signal_date == "2026-04-20"
    assert df.empty
    assert list(df.columns) == ["code", "score", "target_weight", "action"]
