import json
from datetime import datetime
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
    load_exploration_signal_events,
    load_shadow_signal_events,
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


def test_feature_builder_dedupes_trade_sequences_before_summing_flow():
    trades = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:01.000",
                "price": 100.0,
                "volume": 100,
                "turnover": 10_000.0,
                "ticker_direction": "BUY",
                "sequence": 1,
            },
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:01.000",
                "price": 100.0,
                "volume": 100,
                "turnover": 10_000.0,
                "ticker_direction": "BUY",
                "sequence": 1,
            },
            {
                "symbol": "US.AAPL",
                "event_time": "2026-06-01 09:30:10.000",
                "price": 101.0,
                "volume": 100,
                "turnover": 10_100.0,
                "ticker_direction": "SELL",
                "sequence": 2,
            },
        ]
    )

    features = compute_microstructure_features(trades, pd.DataFrame(), pd.DataFrame())

    row = features.iloc[0]
    assert row["trade_count"] == 2
    assert row["raw_trade_count"] == 3
    assert row["duplicate_sequence_count"] == 1
    assert row["duplicate_sequence_rate"] == 1 / 3
    assert row["dollar_volume"] == 20_100.0
    assert row["active_buy_dollar"] == 10_000.0
    assert row["active_sell_dollar"] == 10_100.0


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
    assert row["raw_trade_count"] == row["trade_count"]
    assert row["duplicate_sequence_count"] == 0


def test_coarse_universe_summary_flags_alphabet_biased_collection(tmp_path):
    universe_dir = tmp_path / "universe" / "date=2026-06-03"
    universe_dir.mkdir(parents=True)
    (universe_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "date": "2026-06-03",
                "candidate_count": 5,
                "target_size": 5,
                "universe_count": 100,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"symbol": ["US.AAA", "US.AAB", "US.AAC", "US.AAD", "US.BBB"]}).to_csv(
        universe_dir / "us_microstructure_candidates.csv",
        index=False,
    )
    pd.DataFrame({"symbol": [f"US.A{i:03d}" for i in range(60)] + ["US.BBB", "US.CCC"]}).to_csv(
        universe_dir / "us_microstructure_collection_universe.csv",
        index=False,
    )

    summary = report_script._load_coarse_universe_summary(tmp_path, "2026-06-03")
    markdown = report_script._coarse_universe_markdown(summary)

    assert summary["alphabet_bias_warning"] is True
    assert summary["collection_dominant_letter"]["letter"] == "A"
    assert "不能作为全市场追主力结论" in markdown


def test_chinese_conclusion_suppresses_candidates_when_universe_is_alphabet_biased():
    signals = pd.DataFrame(
        [
            {
                "symbol": "US.AEC",
                "side": "accumulation",
                "confidence": "watch",
                "side_score": 85.0,
                "net_active_dollar": 1_000_000.0,
                "active_buy_ratio": 0.78,
            }
        ]
    )

    markdown = report_script._chinese_conclusion_markdown(
        signals=signals,
        validation_gate={"state": "warmup", "validated": False},
        data_quality={"symbol_count": 300, "eligible_symbol_count": 250},
        validation_progress={},
        confidence_gap={"ready": False},
        coarse_universe={
            "alphabet_bias_warning": True,
            "candidate_dominant_letter": {"letter": "A", "count": 262, "total": 300, "share": 0.873},
            "collection_dominant_letter": {"letter": "A", "count": 262, "total": 304, "share": 0.862},
        },
    )

    assert "本日报作废" in markdown
    assert "今日观察候选：不展示" in markdown
    assert "AEC：吸筹" not in markdown


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


def test_signal_scoring_uses_total_duplicate_sequence_rate_for_quality():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=3, freq="min")
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL"] * len(minutes),
            "minute": minutes,
            "trade_count": [10_000, 0, 10_000],
            "raw_trade_count": [10_000, 1, 10_000],
            "duplicate_sequence_count": [0, 1, 0],
            "dollar_volume": [10_000_000.0] * len(minutes),
            "active_buy_dollar": [7_000_000.0] * len(minutes),
            "active_sell_dollar": [3_000_000.0] * len(minutes),
            "has_trade_data": [True] * len(minutes),
            "has_book_data": [True] * len(minutes),
            "coverage_ratio_regular": [1.0] * len(minutes),
            "trade_coverage_ratio_regular": [1.0] * len(minutes),
            "book_coverage_ratio_regular": [1.0] * len(minutes),
            "reference_price": [100.0, 100.2, 100.4],
            "vwap_deviation_bps": [20.0] * len(minutes),
            "price_impact_bps_per_musd": [5.0] * len(minutes),
            "spread_bps": [2.0] * len(minutes),
            "depth_imbalance_1": [0.50] * len(minutes),
            "depth_imbalance_5": [0.40] * len(minutes),
            "bid_replenish_1": [900.0] * len(minutes),
            "ask_replenish_1": [0.0] * len(minutes),
            "dollar_volume_z": [3.0] * len(minutes),
            "odd_lot_ratio": [0.1] * len(minutes),
            "duplicate_sequence_rate": [0.0, 1.0, 0.0],
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

    row = signals.iloc[0]
    assert row["confidence"] == "high"
    assert row["raw_trade_count"] == 20_001
    assert row["duplicate_sequence_count"] == 1
    assert row["duplicate_sequence_rate"] == 1 / 20_001


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


def test_report_data_quality_uses_total_duplicate_sequence_rate():
    minutes = pd.date_range("2026-06-01 13:30:00+00:00", periods=3, freq="min")
    features = pd.DataFrame(
        {
            "symbol": ["US.AAPL"] * len(minutes),
            "minute": minutes,
            "is_regular_session": [True] * len(minutes),
            "trade_count": [100, 80, 100],
            "raw_trade_count": [100, 100, 100],
            "duplicate_sequence_count": [0, 20, 0],
            "dollar_volume": [30_000_000.0] * len(minutes),
            "coverage_ratio_regular": [1.0] * len(minutes),
            "trade_coverage_ratio_regular": [1.0] * len(minutes),
            "book_coverage_ratio_regular": [1.0] * len(minutes),
            "quote_coverage_ratio_regular": [1.0] * len(minutes),
            "spread_bps": [2.0] * len(minutes),
            "duplicate_sequence_rate": [0.0, 0.2, 0.0],
        }
    )

    quality = report_script._data_quality_summary(
        features,
        MicrostructureSignalConfig(min_trade_count=1, min_dollar_volume=1_000, min_data_coverage=0.1),
    )

    row = quality["symbols"][0]
    assert row["eligible"] is False
    assert row["raw_trade_count"] == 300
    assert row["duplicate_sequence_count"] == 20
    assert row["duplicate_sequence_rate"] == 20 / 300
    assert quality["duplicate_sequence_rate"] == 20 / 300


def test_validation_eligibility_summary_exposes_sample_blockers():
    signals = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 72,
                "confidence": "watch",
                "data_quality_pass": True,
                "is_final_report": True,
            },
            {
                "symbol": "US.NVDA",
                "side": "distribution",
                "side_score": 68,
                "confidence": "diagnostic",
                "data_quality_pass": False,
                "is_final_report": False,
            },
        ]
    )

    summary = report_script._validation_eligibility_summary(signals, min_event_score=70)

    assert summary["validation_eligible_count"] == 1
    assert summary["validation_eligible_if_final_count"] == 1
    assert summary["score_pass_count"] == 1
    assert summary["near_score_count"] == 1
    assert summary["watch_or_high_count"] == 1
    assert summary["data_quality_pass_count"] == 1
    assert summary["final_report_count"] == 1
    assert summary["blocking_counts"]["score_below_min"] == 1
    assert summary["blocking_counts"]["not_watch_or_high"] == 1
    assert summary["blocking_counts"]["data_quality_failed"] == 1
    assert summary["blocking_counts"]["not_final_report"] == 1


def test_manifest_gate_prevents_high_confidence_and_validation_sample():
    signals = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 90,
                "confidence": "high",
                "data_quality_pass": True,
                "validation_reason": "validated",
            }
        ]
    )

    adjusted = report_script._apply_manifest_quality_to_signals(
        signals,
        {"ok": False, "issues": ["manifest contains failed NAS uploads: 1"]},
    )
    summary = report_script._validation_eligibility_summary(adjusted, min_event_score=70)

    assert adjusted.iloc[0]["confidence"] == "watch"
    assert bool(adjusted.iloc[0]["data_quality_pass"]) is False
    assert bool(adjusted.iloc[0]["nas_upload_complete"]) is False
    assert "manifest contains failed NAS uploads" in adjusted.iloc[0]["validation_reason"]
    assert summary["data_quality_pass_count"] == 0
    assert summary["validation_eligible_count"] == 0


def test_final_report_gate_downgrades_intraday_high_confidence():
    signals = pd.DataFrame(
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 90,
                "confidence": "high",
                "data_quality_pass": True,
                "validation_reason": "validated",
            }
        ]
    )

    adjusted = report_script._apply_final_report_gate_to_signals(signals, is_final_report=False)

    assert adjusted.iloc[0]["confidence"] == "watch"
    assert bool(adjusted.iloc[0]["data_quality_pass"]) is True
    assert adjusted.iloc[0]["validation_reason"] == "not final post-close report"


def test_intraday_replay_summary_loads_metrics_for_report(tmp_path):
    replay_dir = tmp_path / "validation" / "intraday_replay" / "date=2026-06-01"
    replay_dir.mkdir(parents=True)
    cumulative_dir = tmp_path / "validation" / "intraday_replay"
    (replay_dir / "status.json").write_text(
        json.dumps(
            {
                "event_count": 2,
                "quality_event_count": 1,
                "return_count": 4,
                "quality_return_count": 2,
                "cutoff_count": 1,
                "metric_count": 1,
                "horizons_minutes": [30],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "side": "distribution",
                "horizon_minutes": 30,
                "observation_count": 2,
                "quality_observation_count": 1,
                "hit_rate": 0.5,
                "avg_alpha": 0.001,
                "max_symbol_sample_share": 1.0,
            }
        ]
    ).to_csv(replay_dir / "intraday_replay_metrics.csv", index=False)
    (cumulative_dir / "cumulative_status.json").write_text(
        json.dumps(
            {
                "date_count": 3,
                "first_date": "2026-05-28",
                "last_date": "2026-06-01",
                "event_count": 12,
                "quality_event_count": 8,
                "return_count": 24,
                "quality_return_count": 16,
                "metric_count": 1,
                "horizons_minutes": [30],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "side": "accumulation",
                "horizon_minutes": 30,
                "observation_count": 10,
                "quality_observation_count": 8,
                "hit_rate": 0.7,
                "avg_alpha": 0.002,
                "max_symbol_sample_share": 0.4,
            }
        ]
    ).to_csv(cumulative_dir / "cumulative_metrics.csv", index=False)

    summary = report_script._load_intraday_replay_summary(tmp_path, "2026-06-01")
    markdown = report_script._intraday_replay_markdown(summary)
    html = report_script._intraday_replay_html(summary)

    assert summary["exists"] is True
    assert summary["quality_event_count"] == 1
    assert summary["quality_return_count"] == 2
    assert summary["metrics"][0]["side"] == "distribution"
    assert summary["cumulative_date_count"] == 3
    assert summary["cumulative_quality_return_count"] == 16
    assert summary["cumulative_metrics"][0]["side"] == "accumulation"
    assert "出货" in markdown
    assert "50.0%" in markdown
    assert "累计指标" in markdown
    assert "70.0%" in markdown
    assert "累计日内回放" in html
    assert "2026-05-28 到 2026-06-01" in html


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


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_script_writes_warmup_artifacts(tmp_path, monkeypatch):
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
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {
            "state": "warmup",
            "validated": False,
            "validated_sides": {"accumulation": False, "distribution": False},
            "reason": "forward validation sample not promoted yet",
            "side_reasons": {
                "accumulation": "missing 5d validation metrics",
                "distribution": "failed: observations",
            },
            "side_metrics": {
                "distribution": {
                    "observation_count": 12,
                    "signal_day_count": 4,
                    "avg_alpha": 0.002,
                    "hit_rate": 0.5,
                    "recent_hit_rate": 0.5,
                    "wilson_lower": 0.25,
                    "max_symbol_sample_share": 0.4,
                }
            },
            "criteria": {
                "benchmark": "US.SPY",
                "promotion_horizon": 5,
                "min_signal_days_per_side": 20,
                "min_observations_per_side": 100,
                "min_alpha": 0.0075,
                "min_hit_rate": 0.58,
                "min_recent_hit_rate": 0.55,
                "min_wilson_lower": 0.5,
                "max_symbol_sample_share": 0.2,
            },
            "signal_file_count": 3,
            "event_count": 12,
            "forward_return_count": 24,
            "price_symbol_count": 15,
        },
    )

    monkeypatch.setattr(report_script, "_is_final_report", lambda date: False)
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
    assert not (tmp_path / "signals/us_major_flow_signals_latest.csv").exists()
    assert (tmp_path / "quality/date=2026-06-01/us_microstructure_data_quality.csv").exists()
    assert not (tmp_path / "quality/us_microstructure_data_quality_latest.csv").exists()
    assert (tmp_path / "reports/date=2026-06-01/us_microstructure_flow_report.html").exists()
    assert not (tmp_path / "reports/us_microstructure_flow_report_latest.html").exists()
    assert not (tmp_path / "reports/us_microstructure_flow_status_latest.json").exists()
    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    assert status["is_final_report"] is False
    assert status["latest_alias_updated"] is False
    assert status["email_delivery"]["requested"] is False
    assert status["validation_gate"]["state"] == "warmup"
    assert status["validation_progress"]["event_count"] == 12
    assert status["validation_progress"]["forward_return_count"] == 24
    assert status["validation_progress"]["sides"][0]["reason"] == "missing 5d validation metrics"
    assert "validation_eligibility" in status
    assert status["validation_eligibility"]["signal_count"] == 1
    assert status["validation_eligibility"]["validation_eligible_count"] == 0
    assert status["confidence_gap"]["ready"] is False
    assert status["confidence_gap"]["requirements"]["final_report_complete"] is False
    assert "report is not a final post-close report" in status["confidence_gap"]["blockers"]
    assert status["high_count"] == 0
    assert "data_quality" in status
    assert status["data_quality"]["eligible_symbol_count"] == 0
    quality = pd.read_csv(tmp_path / "quality/date=2026-06-01/us_microstructure_data_quality.csv")
    assert quality.iloc[0]["symbol"] == "US.AAPL"
    assert "raw_trade_count" in quality.columns
    assert "duplicate_sequence_count" in quality.columns
    html_report = (tmp_path / "reports/date=2026-06-01/us_microstructure_flow_report.html").read_text(encoding="utf-8")
    assert "追主力日报 - 2026-06-01" in html_report
    assert "今日结论" in html_report
    assert "分标的数据质量" in html_report
    assert "重复序列审计" in html_report
    assert "分方向验证进度" in html_report
    assert "验证样本入账资格" in html_report
    assert "高置信准备度" in html_report
    assert "日内回放校准" in html_report
    assert "缺少 5 日验证指标" in html_report


def test_report_script_updates_latest_aliases_for_final_report(tmp_path, monkeypatch):
    _write_raw(
        tmp_path,
        "trades",
        "US.AAPL",
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
        ],
    )
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
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {"state": "warmup", "validated": False, "validated_sides": {"accumulation": False, "distribution": False}},
    )

    monkeypatch.setattr(report_script, "_is_final_report", lambda date: True)
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

    assert (tmp_path / "signals/us_major_flow_signals_latest.csv").exists()
    assert (tmp_path / "quality/us_microstructure_data_quality_latest.csv").exists()
    assert (tmp_path / "reports/us_microstructure_flow_report_latest.html").exists()
    latest_status_path = tmp_path / "reports/us_microstructure_flow_status_latest.json"
    assert latest_status_path.exists()
    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    latest_status = json.loads(latest_status_path.read_text(encoding="utf-8"))
    assert status["is_final_report"] is True
    assert status["latest_alias_updated"] is True
    assert latest_status["latest_alias_updated"] is True


def test_report_script_records_successful_email_delivery(tmp_path, monkeypatch):
    _write_raw(
        tmp_path,
        "trades",
        "US.AAPL",
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
        ],
    )
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
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {"state": "warmup", "validated": False, "validated_sides": {"accumulation": False, "distribution": False}},
    )
    sent = []

    def fake_send_email(html_content, subject, report_filename=None, report_dir=None, attachment_paths=None):
        sent.append((html_content, subject, report_filename, Path(report_dir), [Path(path) for path in attachment_paths]))
        return True

    monkeypatch.setattr(report_script, "_is_final_report", lambda date: True)
    monkeypatch.setattr("reporter.send_report.send_email", fake_send_email)

    code = report_script.main(
        [
            "--date",
            "2026-06-01",
            "--base-dir",
            str(tmp_path),
            "--symbols",
            "AAPL",
            "--no-nas-sync",
            "--send-email",
        ]
    )

    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    assert code == 0
    assert len(sent) == 1
    assert status["email_delivery"]["requested"] is True
    assert status["email_delivery"]["sent"] is True
    assert status["email_delivery"]["subject"] == "追主力日报 - 暖场验证中，暂无高置信信号"
    assert len(status["email_delivery"]["attachment_paths"]) == 3
    assert "今日结论" in sent[0][0]
    assert sent[0][1] == "追主力日报 - 暖场验证中，暂无高置信信号"
    assert sent[0][4][-1].name == "status.json"


def test_report_script_returns_nonzero_when_email_delivery_fails(tmp_path, monkeypatch):
    _write_raw(
        tmp_path,
        "trades",
        "US.AAPL",
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
        ],
    )
    _write_json(
        tmp_path / "validation" / "active_gate.json",
        {"state": "warmup", "validated": False, "validated_sides": {"accumulation": False, "distribution": False}},
    )

    monkeypatch.setattr(report_script, "_is_final_report", lambda date: True)
    monkeypatch.setattr("reporter.send_report.send_email", lambda *args, **kwargs: False)

    code = report_script.main(
        [
            "--date",
            "2026-06-01",
            "--base-dir",
            str(tmp_path),
            "--symbols",
            "AAPL",
            "--no-nas-sync",
            "--send-email",
        ]
    )

    status = json.loads((tmp_path / "reports/date=2026-06-01/status.json").read_text(encoding="utf-8"))
    assert code == 1
    assert status["email_delivery"]["requested"] is True
    assert status["email_delivery"]["sent"] is False
    assert status["email_delivery"]["error"] == "send_email returned false"


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
    frame = pd.DataFrame(rows)
    if "is_final_report" not in frame.columns:
        frame["is_final_report"] = True
    else:
        frame["is_final_report"] = frame["is_final_report"].where(frame["is_final_report"].notna(), True)
    frame.to_csv(path, index=False)
    return path


def test_report_final_flag_uses_us_eastern_close_buffer():
    assert report_script._is_final_report(
        "2026-06-01",
        now=datetime(2026, 6, 1, 15, 59, tzinfo=report_script.US_EASTERN),
    ) is False
    assert report_script._is_final_report(
        "2026-06-01",
        now=datetime(2026, 6, 1, 16, 10, tzinfo=report_script.US_EASTERN),
    ) is True


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


def test_validation_script_writes_shadow_near_threshold_events(tmp_path):
    _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 66,
                "rank": 1,
                "confidence": "diagnostic",
                "stage": "accumulation_diagnostic",
                "reason": "near threshold",
                "data_quality_pass": True,
            }
        ],
    )
    price_dir = tmp_path / "validation" / "prices"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "US.AAPL", "close": 100},
            {"date": "2026-01-05", "symbol": "US.AAPL", "close": 101},
            {"date": "2026-01-06", "symbol": "US.AAPL", "close": 103},
            {"date": "2026-01-02", "symbol": "US.SPY", "close": 500},
            {"date": "2026-01-05", "symbol": "US.SPY", "close": 501},
            {"date": "2026-01-06", "symbol": "US.SPY", "close": 502},
        ]
    ).to_csv(price_dir / "us_daily_prices.csv", index=False)

    validate_script.main(
        [
            "--base-dir",
            str(tmp_path),
            "--qlib-dir",
            str(tmp_path / "missing_qlib"),
            "--horizons",
            "1",
            "--shadow-min-event-score",
            "65",
            "--no-nas-sync",
        ]
    )

    gate = json.loads((tmp_path / "validation" / "active_gate.json").read_text(encoding="utf-8"))
    shadow_events = pd.read_parquet(tmp_path / "validation" / "shadow_signal_events.parquet")
    shadow_returns = pd.read_parquet(tmp_path / "validation" / "shadow_forward_returns.parquet")

    assert gate["event_count"] == 0
    assert gate["forward_return_count"] == 0
    assert gate["shadow_min_event_score"] == 65
    assert gate["shadow_event_count"] == 1
    assert gate["shadow_forward_return_count"] == 1
    assert shadow_events.iloc[0]["validation_scope"] == "shadow"
    assert shadow_returns.iloc[0]["horizon"] == 1


def test_validation_script_writes_exploration_events_below_shadow_threshold(tmp_path):
    _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "distribution",
                "side_score": 55,
                "rank": 1,
                "confidence": "diagnostic",
                "stage": "distribution_diagnostic",
                "reason": "exploration threshold",
                "data_quality_pass": True,
            }
        ],
    )
    price_dir = tmp_path / "validation" / "prices"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "US.AAPL", "close": 100},
            {"date": "2026-01-05", "symbol": "US.AAPL", "close": 99},
            {"date": "2026-01-06", "symbol": "US.AAPL", "close": 97},
            {"date": "2026-01-02", "symbol": "US.SPY", "close": 500},
            {"date": "2026-01-05", "symbol": "US.SPY", "close": 501},
            {"date": "2026-01-06", "symbol": "US.SPY", "close": 502},
        ]
    ).to_csv(price_dir / "us_daily_prices.csv", index=False)

    validate_script.main(
        [
            "--base-dir",
            str(tmp_path),
            "--qlib-dir",
            str(tmp_path / "missing_qlib"),
            "--horizons",
            "1",
            "--shadow-min-event-score",
            "65",
            "--exploration-min-event-score",
            "50",
            "--no-nas-sync",
        ]
    )

    gate = json.loads((tmp_path / "validation" / "active_gate.json").read_text(encoding="utf-8"))
    exploration_events = pd.read_parquet(tmp_path / "validation" / "exploration_signal_events.parquet")
    exploration_returns = pd.read_parquet(tmp_path / "validation" / "exploration_forward_returns.parquet")
    shadow_events = pd.read_parquet(tmp_path / "validation" / "shadow_signal_events.parquet")

    assert gate["event_count"] == 0
    assert gate["shadow_event_count"] == 0
    assert gate["exploration_min_event_score"] == 50
    assert gate["exploration_event_count"] == 1
    assert gate["exploration_forward_return_count"] == 1
    assert shadow_events.empty
    assert exploration_events.iloc[0]["validation_scope"] == "exploration"
    assert exploration_returns.iloc[0]["horizon"] == 1


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


def test_price_symbol_universe_includes_dynamic_candidate_file(tmp_path):
    universe_dir = tmp_path / "universe"
    universe_dir.mkdir()
    (universe_dir / "us_microstructure_candidates_latest.txt").write_text(
        "US.MSFT\nnvda\nUS.SPY\n",
        encoding="utf-8",
    )

    symbols = price_script.build_price_symbol_universe(
        tmp_path,
        explicit_symbols=["aapl"],
        benchmark="SPY",
        include_default_symbols=False,
        include_signal_symbols=False,
    )

    assert symbols == ["US.SPY", "US.AAPL", "US.MSFT", "US.NVDA"]


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
                "raw_trade_count": 12_010,
                "duplicate_sequence_count": 10,
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
            {
                "symbol": "US.TSLA",
                "side": "accumulation",
                "side_score": 93,
                "rank": 4,
                "confidence": "watch",
                "data_quality_pass": True,
                "is_final_report": False,
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
    assert row["raw_trade_count"] == 12_010
    assert row["duplicate_sequence_count"] == 10
    assert row["dollar_volume"] == 90_000_000.0
    assert row["spread_bps"] == 2.5


def test_load_shadow_signal_events_keeps_final_quality_near_threshold_rows(tmp_path):
    signal_path = _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 66,
                "rank": 1,
                "confidence": "diagnostic",
                "data_quality_pass": True,
            },
            {
                "symbol": "US.NVDA",
                "side": "distribution",
                "side_score": 64,
                "rank": 2,
                "confidence": "diagnostic",
                "data_quality_pass": True,
            },
            {
                "symbol": "US.MSFT",
                "side": "accumulation",
                "side_score": 68,
                "rank": 3,
                "confidence": "watch",
                "data_quality_pass": False,
            },
            {
                "symbol": "US.TSLA",
                "side": "accumulation",
                "side_score": 69,
                "rank": 4,
                "confidence": "watch",
                "data_quality_pass": True,
                "is_final_report": False,
            },
        ],
    )

    official = load_signal_events([signal_path], min_event_score=70)
    shadow = load_shadow_signal_events([signal_path], min_event_score=65)

    assert official.empty
    assert shadow["symbol"].tolist() == ["US.AAPL"]
    assert shadow.iloc[0]["validation_scope"] == "shadow"
    assert shadow.iloc[0]["confidence"] == "diagnostic"


def test_load_exploration_signal_events_keeps_broader_final_quality_rows(tmp_path):
    signal_path = _write_signal(
        tmp_path,
        "2026-01-02",
        [
            {
                "symbol": "US.AAPL",
                "side": "accumulation",
                "side_score": 55,
                "rank": 1,
                "confidence": "diagnostic",
                "data_quality_pass": True,
            },
            {
                "symbol": "US.MSFT",
                "side": "accumulation",
                "side_score": 49,
                "rank": 2,
                "confidence": "diagnostic",
                "data_quality_pass": True,
            },
        ],
    )

    exploration = load_exploration_signal_events([signal_path], min_event_score=50)

    assert exploration["symbol"].tolist() == ["US.AAPL"]
    assert exploration.iloc[0]["validation_scope"] == "exploration"


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
