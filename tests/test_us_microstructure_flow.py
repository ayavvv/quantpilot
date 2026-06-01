import json
from pathlib import Path

import pandas as pd

from scripts import report_us_microstructure_flow as report_script
from scripts import update_us_microstructure_prices as price_script
from scripts import validate_us_microstructure_flow as validate_script
from strategy.us_microstructure_features import compute_microstructure_features
from strategy.us_microstructure_features import read_microstructure_inputs
from strategy.us_microstructure_signals import MicrostructureSignalConfig, score_microstructure_signals
from strategy.us_microstructure_validation import (
    ForwardValidationConfig,
    build_active_gate,
    build_rule_metrics,
    compute_forward_returns,
    load_price_history_from_csv,
    load_signal_events,
)


def test_feature_builder_aligns_futu_eastern_trades_with_utc_book():
    trades = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:05.000",
                "price": 100.0,
                "volume": 100,
                "turnover": 10_000.0,
                "ticker_direction": "BUY",
                "sequence": 1,
                "type": "AUTO_MATCH",
            }
        ]
    )
    order_book = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "recv_time": "2026-06-01T13:30:06.000+00:00",
                "bid_px_1": 99.99,
                "bid_sz_1": 500,
                "ask_px_1": 100.01,
                "ask_sz_1": 200,
                "mid": 100.0,
                "spread_bps": 2.0,
            }
        ]
    )

    features = compute_microstructure_features(trades, order_book, pd.DataFrame())

    assert len(features) == 1
    row = features.iloc[0]
    assert str(row["minute"]) == "2026-06-01 13:30:00+00:00"
    assert row["trade_count"] == 1
    assert row["book_snapshot_count"] == 1
    assert row["trade_coverage_minutes"] == 1
    assert row["book_coverage_minutes"] == 1
    assert row["trade_coverage_ratio_regular"] > 0
    assert row["book_coverage_ratio_regular"] > 0
    assert bool(row["is_regular_session"]) is True
    assert row["active_buy_dollar"] == 10_000.0
    assert row["depth_imbalance_1"] > 0


def test_feature_builder_counts_only_regular_session_for_coverage():
    trades = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 08:00:05.000",
                "price": 99.0,
                "volume": 100,
                "turnover": 9_900.0,
                "ticker_direction": "BUY",
                "sequence": 1,
                "type": "AUTO_MATCH",
            },
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:05.000",
                "price": 100.0,
                "volume": 100,
                "turnover": 10_000.0,
                "ticker_direction": "BUY",
                "sequence": 2,
                "type": "AUTO_MATCH",
            },
        ]
    )

    features = compute_microstructure_features(trades, pd.DataFrame(), pd.DataFrame())

    assert len(features) == 2
    assert features["is_regular_session"].tolist() == [False, True]
    assert features.iloc[-1]["trade_coverage_minutes"] == 1
    assert features.iloc[-1]["coverage_ratio_regular"] == 1 / 390


def test_signal_scoring_keeps_strong_candidate_warmup_without_validation_gate():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=6, freq="min")
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL"] * len(minutes),
            "minute": minutes,
            "trade_count": [100] * len(minutes),
            "dollar_volume": [1_000_000.0] * len(minutes),
            "active_buy_dollar": [700_000.0] * len(minutes),
            "active_sell_dollar": [300_000.0] * len(minutes),
            "has_trade_data": [True] * len(minutes),
            "has_book_data": [True] * len(minutes),
            "coverage_ratio_regular": [1.0] * len(minutes),
            "trade_coverage_ratio_regular": [1.0] * len(minutes),
            "book_coverage_ratio_regular": [1.0] * len(minutes),
            "reference_price": [100, 100.1, 100.2, 100.25, 100.3, 100.35],
            "vwap_deviation_bps": [5, 8, 10, 12, 13, 14],
            "price_impact_bps_per_musd": [5] * len(minutes),
            "spread_bps": [3] * len(minutes),
            "depth_imbalance_1": [0.35] * len(minutes),
            "depth_imbalance_5": [0.25] * len(minutes),
            "bid_replenish_1": [800] * len(minutes),
            "ask_replenish_1": [0] * len(minutes),
            "dollar_volume_z": [2.5] * len(minutes),
            "odd_lot_ratio": [0.1] * len(minutes),
            "duplicate_sequence_rate": [0.0] * len(minutes),
        }
    )

    signals = score_microstructure_signals(
        features,
        config=MicrostructureSignalConfig(min_trade_count=100, min_dollar_volume=1_000, min_data_coverage=0.1),
        validation_gate={"state": "warmup", "validated": False, "reason": "collecting samples"},
    )

    row = signals.iloc[0]
    assert row["symbol"] == "US.AAPL"
    assert row["side"] == "accumulation"
    assert row["side_score"] >= 70
    assert row["confidence"] == "watch"
    assert row["report_state"] == "warmup"


def test_signal_scoring_requires_side_specific_validation_for_high_confidence():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=6, freq="min")
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL"] * len(minutes),
            "minute": minutes,
            "trade_count": [100] * len(minutes),
            "dollar_volume": [1_000_000.0] * len(minutes),
            "active_buy_dollar": [700_000.0] * len(minutes),
            "active_sell_dollar": [300_000.0] * len(minutes),
            "has_trade_data": [True] * len(minutes),
            "has_book_data": [True] * len(minutes),
            "coverage_ratio_regular": [1.0] * len(minutes),
            "trade_coverage_ratio_regular": [1.0] * len(minutes),
            "book_coverage_ratio_regular": [1.0] * len(minutes),
            "reference_price": [100, 100.1, 100.2, 100.25, 100.3, 100.35],
            "vwap_deviation_bps": [20] * len(minutes),
            "price_impact_bps_per_musd": [5] * len(minutes),
            "spread_bps": [2] * len(minutes),
            "depth_imbalance_1": [0.50] * len(minutes),
            "depth_imbalance_5": [0.40] * len(minutes),
            "bid_replenish_1": [900] * len(minutes),
            "ask_replenish_1": [0] * len(minutes),
            "dollar_volume_z": [3] * len(minutes),
            "odd_lot_ratio": [0.1] * len(minutes),
            "duplicate_sequence_rate": [0.0] * len(minutes),
        }
    )

    signals = score_microstructure_signals(
        features,
        config=MicrostructureSignalConfig(
            min_trade_count=100,
            min_dollar_volume=1_000,
            min_data_coverage=0.1,
            high_score=70,
        ),
        validation_gate={
            "state": "validated",
            "validated": True,
            "validated_sides": {"accumulation": True, "distribution": False},
        },
    )

    assert signals.iloc[0]["side"] == "accumulation"
    assert signals.iloc[0]["confidence"] == "high"


def test_high_confidence_requires_order_book_coverage_even_when_validated():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=6, freq="min")
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL"] * len(minutes),
            "minute": minutes,
            "trade_count": [200] * len(minutes),
            "dollar_volume": [10_000_000.0] * len(minutes),
            "active_buy_dollar": [7_000_000.0] * len(minutes),
            "active_sell_dollar": [3_000_000.0] * len(minutes),
            "has_trade_data": [True] * len(minutes),
            "has_book_data": [False] * len(minutes),
            "coverage_ratio_regular": [1.0] * len(minutes),
            "trade_coverage_ratio_regular": [1.0] * len(minutes),
            "book_coverage_ratio_regular": [0.0] * len(minutes),
            "reference_price": [100, 100.1, 100.2, 100.25, 100.3, 100.35],
            "vwap_deviation_bps": [20] * len(minutes),
            "price_impact_bps_per_musd": [5] * len(minutes),
            "spread_bps": [2] * len(minutes),
            "depth_imbalance_1": [0.50] * len(minutes),
            "depth_imbalance_5": [0.40] * len(minutes),
            "bid_replenish_1": [900] * len(minutes),
            "ask_replenish_1": [0] * len(minutes),
            "dollar_volume_z": [3] * len(minutes),
            "odd_lot_ratio": [0.1] * len(minutes),
            "duplicate_sequence_rate": [0.0] * len(minutes),
        }
    )

    signals = score_microstructure_signals(
        features,
        config=MicrostructureSignalConfig(
            min_trade_count=100,
            min_dollar_volume=1_000,
            min_data_coverage=0.8,
            high_score=70,
        ),
        validation_gate={
            "state": "validated",
            "validated": True,
            "validated_sides": {"accumulation": True, "distribution": False},
        },
    )

    assert signals.iloc[0]["side"] == "accumulation"
    assert signals.iloc[0]["confidence"] != "high"


def test_signal_scoring_ignores_premarket_rows_when_session_flag_exists():
    minutes = pd.to_datetime(["2026-06-01 12:00:00+00:00", "2026-06-01 13:30:00+00:00"])
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL", "US.AAPL"],
            "minute": minutes,
            "is_regular_session": [False, True],
            "trade_count": [10_000, 1],
            "dollar_volume": [500_000_000.0, 1_000.0],
            "active_buy_dollar": [400_000_000.0, 500.0],
            "active_sell_dollar": [100_000_000.0, 500.0],
            "has_trade_data": [True, True],
            "has_book_data": [True, True],
            "coverage_ratio_regular": [1.0, 1 / 390],
            "trade_coverage_ratio_regular": [1.0, 1 / 390],
            "book_coverage_ratio_regular": [1.0, 1 / 390],
            "reference_price": [100.0, 100.0],
            "vwap_deviation_bps": [30.0, 0.0],
            "price_impact_bps_per_musd": [1.0, 0.0],
            "spread_bps": [1.0, 1.0],
            "depth_imbalance_1": [0.5, 0.0],
            "depth_imbalance_5": [0.4, 0.0],
            "bid_replenish_1": [900.0, 0.0],
            "ask_replenish_1": [0.0, 0.0],
            "dollar_volume_z": [3.0, 0.0],
            "odd_lot_ratio": [0.1, 0.1],
            "duplicate_sequence_rate": [0.0, 0.0],
        }
    )

    signals = score_microstructure_signals(
        features,
        config=MicrostructureSignalConfig(min_trade_count=1, min_dollar_volume=1, min_data_coverage=0.001),
        validation_gate={"state": "warmup", "validated": False},
    )

    assert len(signals) == 1
    assert signals.iloc[0]["trade_count"] == 1
    assert signals.iloc[0]["dollar_volume"] == 1_000.0


def _write_raw(base: Path, kind: str, symbol: str, rows: list[dict]):
    path = base / kind / "date=2026-06-01" / f"symbol={symbol}" / "part-test.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_report_script_writes_warmup_artifacts(tmp_path):
    trades = []
    for idx in range(5):
        trades.append(
            {
                "symbol": "US.AAPL",
                "event_time": f"2026-06-01 09:3{idx}:05.000",
                "price": 100.0 + idx * 0.1,
                "volume": 100 + idx,
                "turnover": (100.0 + idx * 0.1) * (100 + idx),
                "ticker_direction": "BUY",
                "sequence": idx,
                "type": "AUTO_MATCH",
            }
        )
    _write_raw(tmp_path, "trades", "US.AAPL", trades)
    _write_raw(
        tmp_path,
        "order_book",
        "US.AAPL",
        [
            {
                "symbol": "US.AAPL",
                "recv_time": "2026-06-01T13:30:06.000+00:00",
                "bid_px_1": 100.0,
                "bid_sz_1": 500,
                "ask_px_1": 100.1,
                "ask_sz_1": 250,
                "mid": 100.05,
                "spread_bps": 10.0,
            }
        ],
    )
    _write_raw(
        tmp_path,
        "quotes",
        "US.AAPL",
        [{"symbol": "US.AAPL", "recv_time": "2026-06-01T13:30:06.000+00:00", "last_price": 100.1}],
    )

    report_script.main(
        [
            "--date",
            "2026-06-01",
            "--base-dir",
            str(tmp_path),
            "--symbols",
            "AAPL",
            "--no-nas-sync",
        ]
    )

    assert (tmp_path / "features_1m/date=2026-06-01/part-us-microstructure-features.parquet").exists()
    assert (tmp_path / "signals/date=2026-06-01/us_major_flow_signals.csv").exists()
    assert (tmp_path / "quality/date=2026-06-01/us_microstructure_data_quality.csv").exists()
    assert (tmp_path / "quality/us_microstructure_data_quality_latest.csv").exists()
    assert (tmp_path / "reports/date=2026-06-01/us_microstructure_flow_report.html").exists()
    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    assert status["validation_gate"]["state"] == "warmup"
    assert status["high_count"] == 0
    assert "data_quality" in status
    assert status["data_quality"]["eligible_symbol_count"] == 0
    quality = pd.read_csv(tmp_path / "quality/date=2026-06-01/us_microstructure_data_quality.csv")
    assert quality.iloc[0]["symbol"] == "US.AAPL"
    html_report = (tmp_path / "reports/date=2026-06-01/us_microstructure_flow_report.html").read_text(encoding="utf-8")
    assert "Data Quality By Symbol" in html_report


def test_read_microstructure_inputs_filters_stale_trades_from_date_partition(tmp_path):
    _write_raw(
        tmp_path,
        "trades",
        "US.AAPL",
        [
            {
                "symbol": "US.AAPL",
                "event_time": "2026-05-29 15:59:59.000",
                "price": 99.0,
                "volume": 100,
                "turnover": 9_900.0,
            },
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:00.000",
                "price": 100.0,
                "volume": 100,
                "turnover": 10_000.0,
            },
        ],
    )

    inputs = read_microstructure_inputs(tmp_path, date="2026-06-01", symbols=["AAPL"])

    assert len(inputs["trades"]) == 1
    assert inputs["trades"].iloc[0]["event_time"] == "2026-06-01 09:30:00.000"


def _write_signal(base: Path, date: str, rows: list[dict]):
    path = base / "signals" / f"date={date}" / "us_major_flow_signals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_forward_validation_builds_active_gate_from_price_csv(tmp_path):
    signal_path = _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 88,
                "rank": 1,
                "confidence": "watch",
                "stage": "accumulation_watch",
                "reason": "test",
                "data_quality_pass": True,
            },
            {
                "symbol": "US.MSFT",
                "side": "accumulation",
                "side_score": 86,
                "rank": 2,
                "confidence": "watch",
                "stage": "accumulation_watch",
                "reason": "test",
                "data_quality_pass": True,
            },
        ],
    )
    price_rows = []
    dates = pd.bdate_range("2026-01-02", periods=8)
    for idx, day in enumerate(dates):
        date = day.strftime("%Y-%m-%d")
        price_rows.append({"date": date, "symbol": "US.AAPL", "close": 100 + idx * 2})
        price_rows.append({"date": date, "symbol": "US.MSFT", "close": 200 + idx * 3})
        price_rows.append({"date": date, "symbol": "US.SPY", "close": 500 + idx * 1})
    price_csv = tmp_path / "prices.csv"
    pd.DataFrame(price_rows).to_csv(price_csv, index=False)

    events = load_signal_events([signal_path], min_event_score=70)
    prices = load_price_history_from_csv(price_csv)
    cfg = ForwardValidationConfig(
        horizons=(5,),
        min_signal_days_per_side=1,
        min_observations_per_side=2,
        min_alpha=0.001,
        min_hit_rate=0.5,
        min_recent_hit_rate=0.5,
        min_wilson_lower=0.2,
        max_symbol_sample_share=0.51,
    )
    returns = compute_forward_returns(events, prices, config=cfg)
    metrics = build_rule_metrics(returns, config=cfg)
    gate = build_active_gate(metrics, config=cfg)

    assert len(events) == 2
    assert len(returns) == 2
    assert gate["validated"] is True
    assert gate["validated_sides"]["accumulation"] is True


def test_validation_script_writes_warmup_gate_without_future_prices(tmp_path):
    _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 88,
                "rank": 1,
                "confidence": "watch",
                "stage": "accumulation_watch",
                "reason": "test",
                "data_quality_pass": True,
            }
        ],
    )

    validate_script.main(["--base-dir", str(tmp_path), "--qlib-dir", str(tmp_path / "missing_qlib"), "--no-nas-sync"])

    gate = json.loads((tmp_path / "validation" / "active_gate.json").read_text(encoding="utf-8"))
    assert gate["state"] == "warmup"
    assert gate["validated"] is False
    assert gate["event_count"] == 1
    assert gate["forward_return_count"] == 0


def test_price_symbol_universe_includes_defaults_explicit_signals_and_benchmark(tmp_path):
    _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 88,
                "rank": 1,
                "confidence": "watch",
                "data_quality_pass": True,
            }
        ],
    )

    symbols = price_script.build_price_symbol_universe(
        tmp_path,
        explicit_symbols=["nvda"],
        benchmark="SPY",
        include_default_symbols=False,
    )

    assert symbols == ["US.SPY", "US.NVDA", "US.AAPL"]


def test_load_signal_events_requires_reportable_quality_signals(tmp_path):
    signal_path = _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 90,
                "rank": 1,
                "confidence": "watch",
                "data_quality_pass": True,
                "coverage_ratio_regular": 0.95,
                "trade_coverage_ratio_regular": 0.95,
                "book_coverage_ratio_regular": 0.94,
                "quote_coverage_ratio_regular": 0.93,
                "trade_count": 12_000,
                "dollar_volume": 90_000_000.0,
                "duplicate_sequence_rate": 0.0,
                "spread_bps": 2.5,
                "evidence_blocks": 2,
            },
            {
                "symbol": "US.NVDA",
                "side": "accumulation",
                "side_score": 92,
                "rank": 2,
                "confidence": "watch",
                "data_quality_pass": False,
            },
            {
                "symbol": "US.MSFT",
                "side": "distribution",
                "side_score": 91,
                "rank": 3,
                "confidence": "diagnostic",
                "data_quality_pass": True,
            },
        ],
    )

    events = load_signal_events([signal_path], min_event_score=70)

    assert events["symbol"].tolist() == ["US.AAPL"]
    row = events.iloc[0]
    assert bool(row["data_quality_pass"]) is True
    assert row["coverage_ratio_regular"] == 0.95
    assert row["book_coverage_ratio_regular"] == 0.94
    assert row["trade_count"] == 12_000
    assert row["dollar_volume"] == 90_000_000.0
    assert row["spread_bps"] == 2.5


def test_update_price_history_merges_existing_and_fetcher_rows(tmp_path):
    price_dir = tmp_path / "validation" / "prices"
    price_dir.mkdir(parents=True)
    (price_dir / "us_daily_prices.csv").write_text(
        "date,symbol,open,high,low,close,volume,turnover,amount,source,updated_at\n"
        "2026-01-02,US.AAPL,99,101,98,100,1000,100000,100000,old,2026-01-02T00:00:00\n",
        encoding="utf-8",
    )

    def fake_fetcher(symbols, start, end):
        assert symbols == ["US.AAPL", "US.SPY"]
        assert start == "2026-01-02"
        assert end == "2026-01-03"
        rows = price_script.normalize_kline_rows(
            [
                {
                    "time_key": "2026-01-02 00:00:00",
                    "open": 100,
                    "high": 102,
                    "low": 99,
                    "close": 101,
                    "volume": 2000,
                    "turnover": 202000,
                }
            ],
            symbol="US.AAPL",
            source="fake",
        )
        spy = price_script.normalize_kline_rows(
            [
                {
                    "time_key": "2026-01-03 00:00:00",
                    "open": 500,
                    "high": 505,
                    "low": 499,
                    "close": 504,
                    "volume": 3000,
                    "turnover": 1512000,
                }
            ],
            symbol="SPY",
            source="fake",
        )
        return pd.concat([rows, spy], ignore_index=True), {}

    prices, errors, outputs = price_script.update_price_history(
        tmp_path,
        symbols=["US.AAPL", "US.SPY"],
        start_date="2026-01-02",
        end_date="2026-01-03",
        fetcher=fake_fetcher,
    )

    by_key = {(row["date"], row["symbol"]): row for row in prices.to_dict("records")}
    assert not errors
    assert by_key[("2026-01-02", "US.AAPL")]["close"] == 101
    assert by_key[("2026-01-03", "US.SPY")]["close"] == 504
    assert outputs["csv"].exists()
    assert outputs["parquet"].exists()
    assert outputs["status"].exists()
