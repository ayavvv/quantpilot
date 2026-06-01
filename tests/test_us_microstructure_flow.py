import json
from pathlib import Path

import pandas as pd

from scripts import report_us_microstructure_flow as report_script
from strategy.us_microstructure_features import compute_microstructure_features
from strategy.us_microstructure_signals import MicrostructureSignalConfig, score_microstructure_signals


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
    assert row["active_buy_dollar"] == 10_000.0
    assert row["depth_imbalance_1"] > 0


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
    assert (tmp_path / "reports/date=2026-06-01/us_microstructure_flow_report.html").exists()
    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    assert status["validation_gate"]["state"] == "warmup"
    assert status["high_count"] == 0
