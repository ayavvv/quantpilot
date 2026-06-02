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
