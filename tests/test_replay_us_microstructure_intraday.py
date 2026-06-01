import json

import pandas as pd

from scripts import replay_us_microstructure_intraday as replay


def _feature_rows(symbol: str, *, price_start: float, minutes: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for idx, minute in enumerate(minutes):
        price = price_start + idx * 0.1
        rows.append(
            {
                "symbol": symbol,
                "minute": minute,
                "is_regular_session": True,
                "trade_count": 20,
                "raw_trade_count": 20,
                "duplicate_sequence_count": 0,
                "dollar_volume": 1_000_000.0,
                "active_buy_dollar": 700_000.0,
                "active_sell_dollar": 300_000.0,
                "has_trade_data": True,
                "has_book_data": True,
                "has_quote_data": True,
                "coverage_ratio_regular": 0.1,
                "trade_coverage_ratio_regular": 0.1,
                "book_coverage_ratio_regular": 0.1,
                "quote_coverage_ratio_regular": 0.1,
                "reference_price": price,
                "vwap_deviation_bps": 20.0,
                "price_impact_bps_per_musd": 5.0,
                "spread_bps": 2.0,
                "depth_imbalance_1": 0.5,
                "depth_imbalance_5": 0.4,
                "bid_replenish_1": 900.0,
                "ask_replenish_1": 0.0,
                "dollar_volume_z": 3.0,
                "odd_lot_ratio": 0.1,
                "duplicate_sequence_rate": 0.0,
            }
        )
    return rows


def test_apply_elapsed_coverage_uses_cutoff_elapsed_minutes():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=60, freq="min")
    features = pd.DataFrame(_feature_rows("US.AAPL", price_start=100.0, minutes=minutes))

    adjusted = replay._apply_elapsed_coverage(features, minutes[-1])

    assert adjusted.iloc[-1]["trade_coverage_minutes"] == 60
    assert adjusted.iloc[-1]["book_coverage_minutes"] == 60
    assert adjusted.iloc[-1]["trade_coverage_ratio_regular"] == 1.0
    assert adjusted.iloc[-1]["book_coverage_ratio_regular"] == 1.0


def test_build_intraday_replay_creates_events_and_forward_returns():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=120, freq="min")
    features = pd.DataFrame(
        _feature_rows("US.AAPL", price_start=100.0, minutes=minutes)
        + _feature_rows("US.SPY", price_start=500.0, minutes=minutes)
    )

    events, returns, metrics = replay.build_intraday_replay(
        features,
        date="2026-06-01",
        horizons_minutes=(30,),
        cutoff_interval_minutes=30,
        min_elapsed_minutes=60,
        min_event_score=50,
        benchmark="US.SPY",
    )

    assert not events.empty
    assert not returns.empty
    assert not metrics.empty
    aapl_events = events[events["symbol"] == "US.AAPL"]
    assert not aapl_events.empty
    assert bool(aapl_events.iloc[0]["data_quality_pass"]) is True
    aapl_returns = returns[returns["symbol"] == "US.AAPL"]
    assert not aapl_returns.empty
    assert set(aapl_returns["horizon_minutes"]) == {30}
    assert aapl_returns["fwd_return"].notna().all()


def test_write_intraday_replay_outputs_writes_latest_status(tmp_path):
    events = pd.DataFrame([{"event_id": "e1", "symbol": "US.AAPL"}])
    returns = pd.DataFrame([{"event_id": "e1", "horizon_minutes": 30, "fwd_return": 0.01}])
    metrics = pd.DataFrame([{"side": "accumulation", "horizon_minutes": 30, "observation_count": 1}])
    status = {"date": "2026-06-01", "event_count": 1}

    outputs = replay.write_intraday_replay_outputs(
        tmp_path,
        date="2026-06-01",
        events=events,
        returns=returns,
        metrics=metrics,
        status=status,
    )

    assert outputs["events"].exists()
    assert outputs["returns"].exists()
    assert outputs["metrics"].exists()
    latest = tmp_path / "validation" / "intraday_replay" / "latest_status.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8"))["event_count"] == 1


def test_write_cumulative_intraday_replay_outputs_combines_date_returns(tmp_path):
    root = tmp_path / "validation" / "intraday_replay"
    date1 = root / "date=2026-06-01"
    date2 = root / "date=2026-06-02"
    date1.mkdir(parents=True)
    date2.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "signal_date": "2026-06-01",
                "cutoff_time": "2026-06-01T14:30:00+00:00",
                "symbol": "US.AAPL",
                "side": "accumulation",
                "horizon_minutes": 30,
                "data_quality_pass": True,
                "fwd_return": 0.01,
                "directional_alpha": 0.004,
                "directional_hit": True,
            },
            {
                "event_id": "e2",
                "signal_date": "2026-06-01",
                "cutoff_time": "2026-06-01T15:00:00+00:00",
                "symbol": "US.TSLA",
                "side": "distribution",
                "horizon_minutes": 30,
                "data_quality_pass": False,
                "fwd_return": 0.02,
                "directional_alpha": -0.01,
                "directional_hit": False,
            },
        ]
    ).to_parquet(date1 / "intraday_replay_returns.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "signal_date": "2026-06-02",
                "cutoff_time": "2026-06-02T14:30:00+00:00",
                "symbol": "US.AAPL",
                "side": "accumulation",
                "horizon_minutes": 30,
                "data_quality_pass": True,
                "fwd_return": 0.03,
                "directional_alpha": 0.02,
                "directional_hit": True,
            }
        ]
    ).to_parquet(date2 / "intraday_replay_returns.parquet", index=False)

    loaded = replay.load_intraday_replay_returns(tmp_path)
    outputs, status = replay.write_cumulative_intraday_replay_outputs(
        tmp_path,
        generated_at="2026-06-03T00:00:00+00:00",
    )

    assert len(loaded) == 2
    assert outputs["cumulative_returns"].exists()
    assert outputs["cumulative_metrics"].exists()
    assert outputs["cumulative_status"].exists()
    assert status["date_count"] == 2
    assert status["first_date"] == "2026-06-01"
    assert status["last_date"] == "2026-06-02"
    assert status["event_count"] == 2
    assert status["quality_event_count"] == 1
    assert status["return_count"] == 2
    assert status["quality_return_count"] == 1
    assert status["horizons_minutes"] == [30]
    metrics = pd.read_csv(outputs["cumulative_metrics"])
    assert set(metrics["side"]) == {"accumulation", "distribution"}
