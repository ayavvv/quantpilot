import pandas as pd
from zoneinfo import ZoneInfo

from trader import trade_daily


def test_resolve_dry_run_forces_preview_outside_session(monkeypatch):
    monkeypatch.setattr(trade_daily, "ALLOW_OFF_HOURS_TRADING", False)
    dry_run, reason = trade_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-03-30 19:22:00", tz=ZoneInfo("Asia/Shanghai")).to_pydatetime(),
    )

    assert dry_run is True
    assert "不在 A 股交易时段" in reason


def test_resolve_dry_run_allows_live_during_session(monkeypatch):
    monkeypatch.setattr(trade_daily, "ALLOW_OFF_HOURS_TRADING", False)
    dry_run, reason = trade_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-03-30 14:50:00", tz=ZoneInfo("Asia/Shanghai")).to_pydatetime(),
    )

    assert dry_run is False
    assert reason is None


def test_resolve_dry_run_respects_off_hours_override(monkeypatch):
    monkeypatch.setattr(trade_daily, "ALLOW_OFF_HOURS_TRADING", True)
    dry_run, reason = trade_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-03-30 19:22:00", tz=ZoneInfo("Asia/Shanghai")).to_pydatetime(),
    )

    assert dry_run is False
    assert "不在 A 股交易时段" in reason


def test_resolve_dry_run_uses_market_state_for_holiday_workday(monkeypatch):
    monkeypatch.setattr(trade_daily, "ALLOW_OFF_HOURS_TRADING", False)
    dry_run, reason = trade_daily.resolve_dry_run_mode(
        False,
        now=pd.Timestamp("2026-04-06 14:50:00", tz=ZoneInfo("Asia/Shanghai")).to_pydatetime(),
        global_state={"market_sh": "CLOSED", "market_sz": "CLOSED"},
    )

    assert dry_run is True
    assert reason == "OpenD 市场状态: SH=CLOSED, SZ=CLOSED"


def test_select_sim_acc_id_prefers_explicit_id():
    acc_list = pd.DataFrame(
        [
            {"acc_id": 11, "trd_env": "SIMULATE"},
            {"acc_id": 22, "trd_env": "SIMULATE"},
        ]
    )

    assert trade_daily.select_sim_acc_id(acc_list, preferred_acc_id=22) == 22


def test_get_positions_binds_account_and_refreshes_cache():
    class FakeTradeContext:
        def __init__(self):
            self.calls = []

        def position_list_query(self, **kwargs):
            self.calls.append(kwargs)
            return trade_daily.RET_OK, pd.DataFrame(
                [
                    {
                        "code": "SH.600000",
                        "qty": 1000,
                        "can_sell_qty": 900,
                        "market_val": 10000,
                        "cost_price": 10,
                        "pl_ratio": 5,
                    }
                ]
            )

    trd_ctx = FakeTradeContext()
    positions = trade_daily.get_positions(trd_ctx, acc_id=3523785, refresh_cache=True)

    assert positions["SH.600000"]["qty"] == 1000
    assert positions["SH.600000"]["can_sell_qty"] == 900
    assert trd_ctx.calls == [
        {
            "code": "",
            "trd_env": trade_daily.SAFE_TRD_ENV,
            "acc_id": 3523785,
            "refresh_cache": True,
        }
    ]


def test_run_trade_skips_sell_when_live_position_is_missing(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []
            self.position_calls = []

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [
                    {
                        "total_assets": 100000,
                        "cash": 50000,
                        "market_val": 0,
                    }
                ]
            )

        def position_list_query(self, **kwargs):
            self.position_calls.append(kwargs)
            code = kwargs.get("code", "")
            if code:
                return trade_daily.RET_OK, pd.DataFrame()
            return trade_daily.RET_OK, pd.DataFrame(
                [
                    {
                        "code": "SH.688066",
                        "qty": 1000,
                        "can_sell_qty": 1000,
                        "market_val": 20000,
                        "cost_price": 20,
                        "pl_ratio": 0,
                    }
                ]
            )

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 1)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.0)
    monkeypatch.setattr(
        trade_daily,
        "get_latest_prices",
        lambda quote_ctx, codes: (
            {"SH.688066": 20.0, "SH.600000": 10.0},
            {"SH.688066": 0.0, "SH.600000": 0.0},
        ),
    )

    signals_df = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 1.0},
            {"code": "SH.688066", "score": 0.5},
        ]
    )

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=None,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    assert len(trd_ctx.order_calls) == 1
    assert trd_ctx.order_calls[0]["trd_side"] == trade_daily.TrdSide.BUY
    assert trd_ctx.order_calls[0]["code"] == "SH.600000"
    assert trd_ctx.order_calls[0]["acc_id"] == 3523785


def test_run_trade_removes_stale_position_before_hold_calculation(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [
                    {
                        "total_assets": 100000,
                        "cash": 50000,
                        "market_val": 0,
                    }
                ]
            )

        def position_list_query(self, **kwargs):
            code = kwargs.get("code", "")
            if code:
                return trade_daily.RET_OK, pd.DataFrame()
            return trade_daily.RET_OK, pd.DataFrame(
                [
                    {
                        "code": "SH.688066",
                        "qty": 1000,
                        "can_sell_qty": 1000,
                        "market_val": 20000,
                        "cost_price": 20,
                        "pl_ratio": 0,
                    }
                ]
            )

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 1)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.05)
    monkeypatch.setattr(
        trade_daily,
        "get_latest_prices",
        lambda quote_ctx, codes: (
            {"SH.688066": 20.0},
            {"SH.688066": 0.0},
        ),
    )

    signals_df = pd.DataFrame([{"code": "SH.688066", "score": 1.0}])

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=None,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    assert len(trd_ctx.order_calls) == 1
    assert trd_ctx.order_calls[0]["trd_side"] == trade_daily.TrdSide.BUY
    assert trd_ctx.order_calls[0]["code"] == "SH.688066"
