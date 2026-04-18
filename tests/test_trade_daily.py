import pickle
import pandas as pd
from zoneinfo import ZoneInfo

from trader import trade_daily


class _QuoteCtx:
    def __init__(self, snapshots_by_code):
        self.snapshots_by_code = snapshots_by_code
        self.calls = []

    def get_market_snapshot(self, codes):
        self.calls.append(list(codes))
        rows = []
        for code in codes:
            row = {"code": code}
            row.update(self.snapshots_by_code.get(code, {}))
            rows.append(row)
        return trade_daily.RET_OK, pd.DataFrame(rows)


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


def test_build_order_price_sell_clamps_to_limit_down():
    snapshot = {
        "last_price": 2.92,
        "lower_limit_price": 2.9,
        "upper_limit_price": 3.54,
    }

    price = trade_daily.build_order_price(
        "SH.600381",
        trade_daily.TrdSide.SELL,
        snapshot,
        trade_daily.SELL_PRICE_SLIPPAGE,
    )

    assert price == 2.90


def test_build_order_price_buy_clamps_to_limit_up():
    snapshot = {
        "last_price": 9.95,
        "lower_limit_price": 9.0,
        "upper_limit_price": 10.0,
    }

    price = trade_daily.build_order_price(
        "SH.600000",
        trade_daily.TrdSide.BUY,
        snapshot,
        trade_daily.BUY_PRICE_SLIPPAGE,
    )

    assert price == 10.00


def test_build_order_price_derives_limits_from_prev_close_when_snapshot_omits_them():
    snapshot = {
        "last_price": 2.77,
        "prev_close_price": 2.77,
        "low_price": 2.63,
        "high_price": 2.75,
    }

    price = trade_daily.build_order_price(
        "SH.600381",
        trade_daily.TrdSide.SELL,
        snapshot,
        trade_daily.SELL_PRICE_SLIPPAGE,
    )

    assert price == 2.74


def test_extract_signals_respects_tradeable_prefixes(tmp_path, monkeypatch):
    pred_path = tmp_path / "pred.pkl"
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-08"), "SH.600000"),
            (pd.Timestamp("2026-04-08"), "SZ.000001"),
        ],
        names=["datetime", "instrument"],
    )
    series = pd.Series([0.8, 0.9], index=index)
    pred_path.write_bytes(pickle.dumps(series))
    monkeypatch.setattr(trade_daily, "A_SHARE_TRADEABLE_PREFIXES", ("SH.",))

    df, signal_date = trade_daily.extract_signals(pred_path)

    assert signal_date == "2026-04-08"
    assert df["code"].tolist() == ["SH.600000"]


def test_latest_a_share_date_uses_model_prefixes_for_freshness(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text(
        "SH.600000\t2006-01-03\t2026-04-10\n"
        "SZ.000001\t2006-01-03\t2026-04-11\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(trade_daily, "QLIB_DATA_DIR", qlib_dir)
    monkeypatch.setattr(trade_daily, "A_SHARE_TRADEABLE_PREFIXES", ("SH.",))
    monkeypatch.setattr(trade_daily, "A_SHARE_MODEL_PREFIXES", ("SH.", "SZ."))

    assert trade_daily._latest_a_share_date() == "2026-04-11"


def test_run_trade_continues_buy_after_sell_failure(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []
            self.position_calls = []
            self._position_query_count = 0

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [{"total_assets": 100000, "cash": 50000, "market_val": 50000}]
            )

        def position_list_query(self, **kwargs):
            self.position_calls.append(kwargs)
            self._position_query_count += 1
            if self._position_query_count == 1:
                return trade_daily.RET_OK, pd.DataFrame(
                    [{
                        "code": "SH.600381",
                        "qty": 1000,
                        "can_sell_qty": 1000,
                        "market_val": 3000,
                        "cost_price": 3.0,
                        "pl_ratio": 0,
                    }]
                )
            return trade_daily.RET_OK, pd.DataFrame(
                [{
                    "code": "SH.600381",
                    "qty": 1000,
                    "can_sell_qty": 1000,
                    "market_val": 3000,
                    "cost_price": 3.0,
                    "pl_ratio": 0,
                }]
            )

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            if kwargs["trd_side"] == trade_daily.TrdSide.SELL:
                return -1, "price not in the limit move"
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 2)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.0)

    quote_ctx = _QuoteCtx(
        {
            "SH.600381": {
                "last_price": 2.92,
                "change_rate": 0.0,
                "lower_limit_price": 2.90,
                "upper_limit_price": 3.54,
                "lot_size": 100,
            },
            "SH.600000": {
                "last_price": 10.0,
                "change_rate": 0.0,
                "lower_limit_price": 9.0,
                "upper_limit_price": 11.0,
                "lot_size": 100,
            },
            "SH.600010": {
                "last_price": 5.0,
                "change_rate": 0.0,
                "lower_limit_price": 4.5,
                "upper_limit_price": 5.5,
                "lot_size": 100,
            },
        }
    )

    signals_df = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 1.0},
            {"code": "SH.600010", "score": 0.9},
        ]
    )

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=quote_ctx,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    assert any(call["trd_side"] == trade_daily.TrdSide.SELL for call in trd_ctx.order_calls)
    assert any(call["trd_side"] == trade_daily.TrdSide.BUY for call in trd_ctx.order_calls)
    assert all("adjust_limit" in call for call in trd_ctx.order_calls)
    assert any(call["adjust_limit"] == 0.01 for call in trd_ctx.order_calls if call["trd_side"] == trade_daily.TrdSide.SELL)
    assert any(call["adjust_limit"] == -0.01 for call in trd_ctx.order_calls if call["trd_side"] == trade_daily.TrdSide.BUY)


def test_run_trade_does_not_rebuy_stop_loss_name_same_day(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []
            self._position_query_count = 0

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [{"total_assets": 100000, "cash": 100000, "market_val": 0}]
            )

        def position_list_query(self, **kwargs):
            self._position_query_count += 1
            if self._position_query_count == 1:
                return trade_daily.RET_OK, pd.DataFrame(
                    [{
                        "code": "SH.600000",
                        "qty": 1000,
                        "can_sell_qty": 1000,
                        "market_val": 9000,
                        "cost_price": 10.0,
                        "pl_ratio": -10.0,
                    }]
                )
            return trade_daily.RET_OK, pd.DataFrame([])

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 1)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.05)
    monkeypatch.setattr(trade_daily, "STOP_LOSS_PCT", -0.08)

    quote_ctx = _QuoteCtx(
        {
            "SH.600000": {
                "last_price": 9.0,
                "change_rate": -10.0,
                "lower_limit_price": 9.0,
                "upper_limit_price": 11.0,
                "lot_size": 100,
            },
            "SH.600001": {
                "last_price": 5.0,
                "change_rate": 0.0,
                "lower_limit_price": 4.5,
                "upper_limit_price": 5.5,
                "lot_size": 100,
            },
        }
    )

    signals_df = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 1.0},
            {"code": "SH.600001", "score": 0.9},
        ]
    )

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=quote_ctx,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    buy_codes = [call["code"] for call in trd_ctx.order_calls if call["trd_side"] == trade_daily.TrdSide.BUY]
    assert buy_codes == ["SH.600001"]


def test_run_trade_skips_live_order_when_price_limits_missing(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []
            self.position_calls = []

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [{"total_assets": 100000, "cash": 50000, "market_val": 50000}]
            )

        def position_list_query(self, **kwargs):
            self.position_calls.append(kwargs)
            return trade_daily.RET_OK, pd.DataFrame(
                [{
                    "code": "SH.600381",
                    "qty": 1000,
                    "can_sell_qty": 1000,
                    "market_val": 3000,
                    "cost_price": 3.0,
                    "pl_ratio": 0,
                }]
            )

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 1)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.0)

    quote_ctx = _QuoteCtx(
        {
            "SH.600381": {"last_price": 2.92, "change_rate": 0.0, "lot_size": 100},
            "SH.600000": {"last_price": 10.0, "change_rate": 0.0, "lot_size": 100},
        }
    )

    signals_df = pd.DataFrame([{"code": "SH.600000", "score": 1.0}])

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=quote_ctx,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    assert trd_ctx.order_calls == []


def test_run_trade_uses_bounded_position_refreshes(monkeypatch):
    class FakeTradeContext:
        def __init__(self):
            self.order_calls = []
            self.position_calls = []

        def accinfo_query(self, **kwargs):
            return trade_daily.RET_OK, pd.DataFrame(
                [{"total_assets": 100000, "cash": 50000, "market_val": 50000}]
            )

        def position_list_query(self, **kwargs):
            self.position_calls.append(kwargs)
            if len(self.position_calls) == 1:
                return trade_daily.RET_OK, pd.DataFrame(
                    [{
                        "code": "SH.600381",
                        "qty": 1000,
                        "can_sell_qty": 1000,
                        "market_val": 3000,
                        "cost_price": 3.0,
                        "pl_ratio": 0,
                    }]
                )
            return trade_daily.RET_OK, pd.DataFrame()

        def place_order(self, **kwargs):
            self.order_calls.append(kwargs)
            return trade_daily.RET_OK, "ok"

    monkeypatch.setattr(trade_daily, "TOP_N", 1)
    monkeypatch.setattr(trade_daily, "HOLD_BONUS", 0.0)

    quote_ctx = _QuoteCtx(
        {
            "SH.600381": {
                "last_price": 2.92,
                "change_rate": 0.0,
                "lower_limit_price": 2.90,
                "upper_limit_price": 3.54,
                "lot_size": 100,
            },
            "SH.600000": {
                "last_price": 10.0,
                "change_rate": 0.0,
                "lower_limit_price": 9.0,
                "upper_limit_price": 11.0,
                "lot_size": 100,
            },
        }
    )

    signals_df = pd.DataFrame([{"code": "SH.600000", "score": 1.0}])

    trd_ctx = FakeTradeContext()
    trade_daily.run_trade(
        trd_ctx,
        quote_ctx=quote_ctx,
        acc_id=3523785,
        signals_df=signals_df,
        signal_day_changes={},
        dry_run=False,
    )

    assert len(trd_ctx.position_calls) == 2
    assert all(call["code"] == "" for call in trd_ctx.position_calls)
