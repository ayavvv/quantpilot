import json

import pandas as pd

import scripts.build_us_microstructure_universe as builder


class FakeUniverseCtx:
    def __init__(self):
        self.basic_rows = [
            {"code": "US.AAPL", "name": "Apple", "exchange_type": "US_NASDAQ", "delisting": False},
            {"code": "US.ABC", "name": "ABC Corp", "exchange_type": "US_NASDAQ", "delisting": False},
            {"code": "US.XYZ", "name": "XYZ Corp", "exchange_type": "US_NYSE", "delisting": False},
            {"code": "US.PINKY", "name": "Pink Sheet", "exchange_type": "US_PINK", "delisting": False},
        ]
        self.snapshots = {
            "US.AAPL": {"code": "US.AAPL", "last_price": 190, "open_price": 189, "prev_close_price": 188, "volume": 1000, "turnover": 190000, "change_rate": 1.0},
            "US.ABC": {"code": "US.ABC", "last_price": 50, "open_price": 48, "prev_close_price": 45, "volume": 900000, "turnover": 45000000, "change_rate": 11.1},
            "US.XYZ": {"code": "US.XYZ", "last_price": 25, "open_price": 25, "prev_close_price": 24.5, "volume": 250000, "turnover": 6250000, "change_rate": 2.0},
        }
        self.daily_rows = {
            "US.ABC": [
                {"time_key": "2026-05-28 00:00:00", "close": 45, "volume": 100000, "turnover": 4500000},
                {"time_key": "2026-05-29 00:00:00", "close": 46, "volume": 120000, "turnover": 5520000},
                {"time_key": "2026-06-01 00:00:00", "close": 50, "volume": 900000, "turnover": 45000000},
            ],
            "US.XYZ": [
                {"time_key": "2026-05-28 00:00:00", "close": 24, "volume": 230000, "turnover": 5520000},
                {"time_key": "2026-05-29 00:00:00", "close": 24.5, "volume": 240000, "turnover": 5880000},
                {"time_key": "2026-06-01 00:00:00", "close": 25, "volume": 250000, "turnover": 6250000},
            ],
            "US.AAPL": [
                {"time_key": "2026-05-29 00:00:00", "close": 188, "volume": 1000, "turnover": 188000},
                {"time_key": "2026-06-01 00:00:00", "close": 190, "volume": 1000, "turnover": 190000},
            ],
        }
        self.minute_rows = {
            "US.ABC": [
                {"time_key": "2026-06-01 09:30:00", "close": 48, "volume": 1000, "turnover": 48000},
                {"time_key": "2026-06-01 09:31:00", "close": 49, "volume": 1000, "turnover": 49000},
                {"time_key": "2026-06-01 09:32:00", "close": 50, "volume": 8000, "turnover": 400000},
                {"time_key": "2026-06-01 09:33:00", "close": 50, "volume": 9000, "turnover": 450000},
                {"time_key": "2026-06-01 09:34:00", "close": 50, "volume": 9000, "turnover": 450000},
                {"time_key": "2026-06-01 09:35:00", "close": 50, "volume": 9000, "turnover": 450000},
                {"time_key": "2026-06-01 09:36:00", "close": 50, "volume": 9000, "turnover": 450000},
            ],
            "US.XYZ": [
                {"time_key": "2026-06-01 09:30:00", "close": 25, "volume": 1000, "turnover": 25000},
                {"time_key": "2026-06-01 09:31:00", "close": 25, "volume": 1000, "turnover": 25000},
                {"time_key": "2026-06-01 09:32:00", "close": 25, "volume": 1000, "turnover": 25000},
                {"time_key": "2026-06-01 09:33:00", "close": 25, "volume": 1000, "turnover": 25000},
                {"time_key": "2026-06-01 09:34:00", "close": 25, "volume": 1000, "turnover": 25000},
            ],
            "US.AAPL": [],
        }

    def get_stock_basicinfo(self, market, security_type):
        return 0, pd.DataFrame(self.basic_rows)

    def get_market_snapshot(self, codes):
        rows = [self.snapshots[code] for code in codes if code in self.snapshots]
        return 0, pd.DataFrame(rows)

    def request_history_kline(self, code, start, end, ktype, autype, max_count, page_req_key=None):
        return 0, pd.DataFrame(self.daily_rows.get(code, [])), None

    def get_cur_kline(self, code, num, ktype, autype=None):
        return 0, pd.DataFrame(self.minute_rows.get(code, []))


class FakeZeroSnapshotFlowCtx:
    def __init__(self):
        self.basic_rows = [
            {"code": "US.AAA", "name": "AAA Corp", "exchange_type": "US_NASDAQ", "delisting": False},
            {"code": "US.BBB", "name": "BBB Corp", "exchange_type": "US_NYSE", "delisting": False},
            {"code": "US.XYZ", "name": "XYZ Corp", "exchange_type": "US_NASDAQ", "delisting": False},
            {"code": "US.ZZZ", "name": "ZZZ Corp", "exchange_type": "US_NYSE", "delisting": False},
        ]
        self.daily_rows = {
            "US.XYZ": [
                {"time_key": "2026-05-29 00:00:00", "close": 20, "volume": 100000, "turnover": 2000000},
                {"time_key": "2026-06-01 00:00:00", "close": 30, "volume": 400000, "turnover": 12000000},
            ],
            "US.ZZZ": [
                {"time_key": "2026-05-29 00:00:00", "close": 40, "volume": 100000, "turnover": 4000000},
                {"time_key": "2026-06-01 00:00:00", "close": 50, "volume": 500000, "turnover": 25000000},
            ],
        }

    def get_stock_basicinfo(self, market, security_type):
        return 0, pd.DataFrame(self.basic_rows)

    def get_market_snapshot(self, codes):
        rows = [
            {
                "code": code,
                "last_price": 0,
                "open_price": 0,
                "prev_close_price": 10,
                "volume": 0,
                "turnover": 0,
                "change_rate": 0,
            }
            for code in codes
        ]
        return 0, pd.DataFrame(rows)

    def request_history_kline(self, code, start, end, ktype, autype, max_count, page_req_key=None):
        return 0, pd.DataFrame(self.daily_rows.get(code, [])), None


class FakeWatchlistCtx:
    def __init__(self):
        self.group_type = None
        self.securities = {
            "美股": [
                {"code": "US.LI", "stock_type": "STOCK", "option_type": "N/A"},
                {"code": "US.QQQ", "stock_type": "ETF", "option_type": "N/A"},
                {"code": "HK.00700", "stock_type": "STOCK", "option_type": "N/A"},
                {"code": "US.AAPL260619C00190000", "stock_type": "DRVT", "option_type": "CALL"},
            ],
            "Growth": [
                {"code": "TSLA", "stock_type": "STOCK", "option_type": "N/A"},
                {"code": "US.LI", "stock_type": "STOCK", "option_type": "N/A"},
            ],
        }

    def get_user_security_group(self, group_type="ALL"):
        self.group_type = group_type
        return 0, pd.DataFrame(
            [
                {"group_name": "美股", "group_type": "CUSTOM"},
                {"group_name": "Growth", "group_type": "CUSTOM"},
            ]
        )

    def get_user_security(self, group_name):
        return 0, pd.DataFrame(self.securities.get(group_name, []))


class FakeAggregateWatchlistCtx:
    def __init__(self):
        self.groups_requested = []

    def get_user_security_group(self, group_type="ALL"):
        return 0, pd.DataFrame(
            [
                {"group_name": "US", "group_type": "SYSTEM"},
                {"group_name": "All", "group_type": "SYSTEM"},
                {"group_name": "Growth", "group_type": "CUSTOM"},
            ]
        )

    def get_user_security(self, group_name):
        self.groups_requested.append(group_name)
        if group_name != "All":
            return 1, "rate limited"
        return 0, pd.DataFrame(
            [
                {"code": "US.AAPL", "stock_type": "STOCK", "option_type": "N/A"},
                {"code": "US.LI", "stock_type": "STOCK", "option_type": "N/A"},
            ]
        )


class FailingWatchlistCtx:
    def get_user_security_group(self, group_type="ALL"):
        return 1, "not logged in"


def test_resolve_core_symbols_prefers_futu_watchlist(tmp_path):
    core_file = tmp_path / "core.txt"
    core_file.write_text("US.SPY\n", encoding="utf-8")
    ctx = FakeWatchlistCtx()

    symbols, meta = builder.resolve_core_symbols(
        ctx,
        core_symbols_file=core_file,
        core_source="futu_watchlist",
    )

    assert symbols == ["US.LI", "US.QQQ", "US.TSLA"]
    assert ctx.group_type == "ALL"
    assert meta["core_symbol_source"] == "futu_watchlist"
    assert meta["core_symbol_fallback_used"] is False
    assert meta["core_watchlist_group_count"] == 2
    assert meta["core_watchlist_us_symbol_count"] == 3


def test_resolve_core_symbols_uses_aggregate_watchlist_group_when_available(tmp_path):
    core_file = tmp_path / "core.txt"
    core_file.write_text("US.SPY\n", encoding="utf-8")
    ctx = FakeAggregateWatchlistCtx()

    symbols, meta = builder.resolve_core_symbols(
        ctx,
        core_symbols_file=core_file,
        core_source="futu_watchlist",
    )

    assert symbols == ["US.AAPL", "US.LI"]
    assert ctx.groups_requested == ["All"]
    assert meta["core_watchlist_groups"] == ["All"]
    assert meta["core_watchlist_error_count"] == 0


def test_resolve_core_symbols_falls_back_to_static_file_when_watchlist_unavailable(tmp_path):
    core_file = tmp_path / "core.txt"
    core_file.write_text("US.SPY\nUS.LI\n", encoding="utf-8")

    symbols, meta = builder.resolve_core_symbols(
        FailingWatchlistCtx(),
        core_symbols_file=core_file,
        core_source="futu_watchlist",
    )

    assert symbols == ["US.SPY", "US.LI"]
    assert meta["core_symbol_source"] == "file_fallback"
    assert meta["core_symbol_fallback_used"] is True
    assert "get_user_security_group" in meta["core_symbol_fallback_reason"]


def test_build_universe_scores_broad_market_and_keeps_core_symbol(tmp_path):
    candidates, scored, status = builder.build_universe(
        ctx=FakeUniverseCtx(),
        base_dir=tmp_path,
        date_value="2026-06-01",
        target_size=2,
        core_symbols=["US.AAPL"],
        include_exchange_types=set(),
        exclude_exchange_types={"US_PINK"},
        exclude_security_classes=set(),
        max_universe_codes=0,
        min_price=2,
        min_snapshot_turnover=1_000_000,
        min_snapshot_volume=50_000,
        history_pool_size=3,
        minute_pool_size=3,
        daily_lookback_days=30,
        minute_lookback=30,
        snapshot_batch_size=10,
        snapshot_sleep_seconds=0,
        history_sleep_seconds=0,
        minute_sleep_seconds=0,
        skip_daily_kline=False,
        skip_minute_kline=False,
    )

    assert candidates["symbol"].tolist() == ["US.ABC", "US.AAPL"]
    assert scored.loc[scored["symbol"] == "US.ABC", "screen_reason"].iloc[0] == "turnover,move,gap,abnormal_volume,minute_burst"
    assert status["universe_count"] == 3
    assert status["snapshot_symbol_count"] == 3
    assert status["daily_symbol_count"] == 3
    assert status["minute_symbol_count"] == 3
    assert status["candidate_core_count"] == 1
    assert status["candidate_liquidity_ranked_count"] == 1
    assert status["candidate_fallback_ranked_count"] == 0
    assert status["candidate_target_shortfall"] == 0


def test_build_universe_uses_flow_ranking_when_snapshot_liquidity_is_zero(tmp_path):
    flow_path = tmp_path / "US_latest_flow.csv"
    pd.DataFrame(
        [
            {"code": "US.ZZZ", "capital_flow_status": "ok", "main_3d_sum": 100_000_000},
            {"code": "US.XYZ", "capital_flow_status": "ok", "main_3d_sum": 50_000_000},
            {"code": "US.AAA", "capital_flow_status": "ok", "main_3d_sum": 1},
        ]
    ).to_csv(flow_path, index=False)

    candidates, scored, status = builder.build_universe(
        ctx=FakeZeroSnapshotFlowCtx(),
        base_dir=tmp_path,
        date_value="2026-06-01",
        target_size=2,
        core_symbols=[],
        include_exchange_types=set(),
        exclude_exchange_types=set(),
        exclude_security_classes=set(),
        max_universe_codes=0,
        min_price=2,
        min_snapshot_turnover=1_000_000,
        min_snapshot_volume=50_000,
        history_pool_size=2,
        minute_pool_size=0,
        daily_lookback_days=30,
        minute_lookback=30,
        snapshot_batch_size=10,
        snapshot_sleep_seconds=0,
        history_sleep_seconds=0,
        minute_sleep_seconds=0,
        skip_daily_kline=False,
        skip_minute_kline=True,
        flow_ranking_file=flow_path,
    )

    assert candidates["symbol"].tolist() == ["US.ZZZ", "US.XYZ"]
    assert set(scored[scored["daily_status"] == "ok"]["symbol"]) == {"US.ZZZ", "US.XYZ"}
    assert status["snapshot_positive_liquidity_count"] == 0
    assert status["enrichment_ranking_source"] == "capital_flow_activity"
    assert status["daily_symbol_requested_count"] == 2
    assert status["candidate_liquidity_ranked_count"] == 2


def test_normalize_snapshot_frame_ignores_zero_open_for_gap():
    snapshot = builder._normalize_snapshot_frame(
        pd.DataFrame(
            [
                {
                    "code": "US.AAA",
                    "last_price": 10,
                    "open_price": 0,
                    "prev_close_price": 10,
                    "volume": 0,
                    "turnover": 0,
                }
            ]
        )
    )

    assert snapshot.loc[0, "snapshot_gap_pct"] == 0.0


def test_select_candidates_falls_back_to_ranked_universe_when_liquidity_gate_is_empty():
    scored = pd.DataFrame(
        [
            {"symbol": "US.AAA", "coarse_score": 80.0, "snapshot_turnover": 0.0, "liquidity_pass": False, "core_symbol": False},
            {"symbol": "US.BBB", "coarse_score": 70.0, "snapshot_turnover": 0.0, "liquidity_pass": False, "core_symbol": False},
            {"symbol": "US.CCC", "coarse_score": 60.0, "snapshot_turnover": 0.0, "liquidity_pass": False, "core_symbol": False},
            {"symbol": "US.DDD", "coarse_score": 50.0, "snapshot_turnover": 0.0, "liquidity_pass": False, "core_symbol": False},
            {"symbol": "US.AAPL", "coarse_score": 10.0, "snapshot_turnover": 0.0, "liquidity_pass": False, "core_symbol": True},
        ]
    )

    candidates = builder.select_candidates(scored, target_size=4, core_symbols=["US.AAPL"])

    assert len(candidates) == 4
    assert set(candidates["symbol"]) == {"US.AAA", "US.BBB", "US.CCC", "US.AAPL"}
    assert candidates["selection_source"].value_counts().to_dict() == {
        "fallback_ranked": 3,
        "core": 1,
    }
    assert "US.DDD" not in set(candidates["symbol"])


def test_select_candidates_stratifies_fallback_to_avoid_alphabet_bias():
    scored = pd.DataFrame(
        [
            *[
                {
                    "symbol": f"US.A{i:03d}",
                    "coarse_score": 100.0 - i,
                    "snapshot_turnover": 0.0,
                    "liquidity_pass": False,
                    "core_symbol": False,
                }
                for i in range(20)
            ],
            *[
                {
                    "symbol": f"US.B{i:03d}",
                    "coarse_score": 70.0 - i,
                    "snapshot_turnover": 0.0,
                    "liquidity_pass": False,
                    "core_symbol": False,
                }
                for i in range(20)
            ],
            *[
                {
                    "symbol": f"US.C{i:03d}",
                    "coarse_score": 40.0 - i,
                    "snapshot_turnover": 0.0,
                    "liquidity_pass": False,
                    "core_symbol": False,
                }
                for i in range(20)
            ],
        ]
    )

    candidates = builder.select_candidates(scored, target_size=12, core_symbols=[])

    letters = candidates["symbol"].str.split(".").str[-1].str[0].value_counts().to_dict()
    assert letters == {"A": 4, "B": 4, "C": 4}
    assert candidates["selection_source"].value_counts().to_dict() == {"fallback_ranked": 12}


def test_write_universe_outputs_writes_latest_text_and_status(tmp_path):
    candidates = pd.DataFrame(
        [
            {"rank": 1, "symbol": "US.ABC", "coarse_score": 90.0, "screen_reason": "turnover"},
            {"rank": 2, "symbol": "US.AAPL", "coarse_score": 20.0, "screen_reason": "core"},
        ]
    )
    scored = candidates.copy()
    status = {"status": "ok", "date": "2026-06-01", "candidate_count": 2}

    outputs = builder.write_universe_outputs(
        tmp_path,
        date_value="2026-06-01",
        candidates=candidates,
        scored=scored,
        status=status,
    )

    assert outputs["candidates_latest_txt"].read_text(encoding="utf-8").splitlines() == ["US.ABC", "US.AAPL"]
    payload = json.loads(outputs["status_latest"].read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 2
    assert (tmp_path / "universe" / "date=2026-06-01" / "us_microstructure_screened_universe.csv").exists()


def test_score_candidates_prefers_code_when_symbol_column_has_missing_values():
    universe = pd.DataFrame(
        [
            {"code": "US.ABC", "symbol": pd.NA, "name": "ABC"},
            {"code": "US.XYZ", "symbol": pd.NA, "name": "XYZ"},
            {"code": "US.AAPL", "symbol": "US.AAPL", "name": "Apple"},
        ]
    )
    snapshot = pd.DataFrame(
        [
            {"symbol": "US.ABC", "snapshot_price": 20, "snapshot_volume": 100000, "snapshot_turnover": 2000000},
            {"symbol": "US.XYZ", "snapshot_price": 30, "snapshot_volume": 100000, "snapshot_turnover": 3000000},
            {"symbol": "US.AAPL", "snapshot_price": 190, "snapshot_volume": 1000, "snapshot_turnover": 190000},
        ]
    )

    scored = builder._score_candidates(
        universe,
        snapshot,
        pd.DataFrame(),
        pd.DataFrame(),
        core_symbols=["US.AAPL"],
        min_price=2,
        min_snapshot_turnover=1_000_000,
        min_snapshot_volume=50_000,
    )

    assert set(scored["symbol"]) == {"US.ABC", "US.XYZ", "US.AAPL"}
    assert "US.NAN" not in set(scored["symbol"])
