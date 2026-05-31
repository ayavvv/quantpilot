import pandas as pd

from converter.incremental import QlibDirectWriter
from strategy.futu_capital_flow_eval import (
    build_capital_flow_promotion_gate,
    evaluate_archived_capital_flow_overlays,
    load_archived_overlays,
)


def _price_records(dates, closes):
    rows = []
    prev = closes[0]
    for day, close in zip(dates, closes):
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000,
                "turnover": close * 1_000_000,
                "change_rate": (close / prev - 1.0) * 100.0 if prev else 0.0,
            }
        )
        prev = close
    return rows


def test_evaluate_archived_capital_flow_overlays_groups_by_label(tmp_path):
    qlib_dir = tmp_path / "qlib"
    writer = QlibDirectWriter(qlib_dir)
    dates = pd.bdate_range("2026-05-25", periods=8)
    writer.write_stock_records("SH.600000", _price_records(dates, [10, 11, 12, 13, 14, 15, 16, 17]))
    writer.write_stock_records("SZ.000001", _price_records(dates, [10, 9, 8, 7, 6, 5, 4, 3]))
    writer.flush()

    overlay = pd.DataFrame(
        [
            {
                "code": "SH.600000",
                "signal_date": dates[1].strftime("%Y-%m-%d"),
                "model_rank": 1,
                "capital_flow_label": "capital_flow_confirm",
            },
            {
                "code": "SZ.000001",
                "signal_date": dates[1].strftime("%Y-%m-%d"),
                "model_rank": 2,
                "capital_flow_label": "risk_flag_main_outflow",
            },
        ]
    )

    summary, rows = evaluate_archived_capital_flow_overlays(
        qlib_dir,
        overlay,
        horizons=(2,),
        entry_lag_days=1,
    )

    assert not rows.empty
    by_label = {row["capital_flow_label"]: row for _, row in summary.iterrows()}
    assert by_label["capital_flow_confirm"]["avg_return"] > 0
    assert by_label["risk_flag_main_outflow"]["avg_return"] < 0


def test_load_archived_overlays_reads_multiple_files(tmp_path):
    first = tmp_path / "20260529_overlay.csv"
    second = tmp_path / "20260601_overlay.csv"
    pd.DataFrame([{"code": "SH.600000", "signal_date": "2026-05-29", "capital_flow_label": "watch"}]).to_csv(
        first, index=False
    )
    pd.DataFrame([{"code": "SZ.000001", "signal_date": "2026-06-01", "capital_flow_label": "risk"}]).to_csv(
        second, index=False
    )

    df = load_archived_overlays([first, second])

    assert set(df["code"]) == {"SH.600000", "SZ.000001"}
    assert "overlay_file" in df.columns


def test_build_capital_flow_promotion_gate_requires_samples():
    gate = build_capital_flow_promotion_gate(pd.DataFrame(), min_date_count=2)

    assert gate["overall_action"] == "insufficient_samples"
    assert gate["decisions"] == []


def test_build_capital_flow_promotion_gate_flags_risk_review():
    summary = pd.DataFrame(
        [
            {
                "capital_flow_label": "risk_flag_main_outflow",
                "horizon": 1,
                "date_count": 3,
                "avg_return": -0.03,
                "avg_universe_return": 0.0,
                "avg_alpha": -0.03,
                "avg_hit_rate": 0.2,
            },
            {
                "capital_flow_label": "risk_flag_main_outflow",
                "horizon": 3,
                "date_count": 3,
                "avg_return": -0.04,
                "avg_universe_return": -0.01,
                "avg_alpha": -0.03,
                "avg_hit_rate": 0.3,
            },
        ]
    )

    gate = build_capital_flow_promotion_gate(summary, min_date_count=3, min_confirming_horizons=2)

    assert gate["overall_action"] == "review_filter"
    assert gate["decisions"][0]["status"] == "candidate_filter_review"
    assert gate["decisions"][0]["matching_horizon_count"] == 2


def test_build_capital_flow_promotion_gate_keeps_advisory_on_mixed_evidence():
    summary = pd.DataFrame(
        [
            {
                "capital_flow_label": "capital_flow_confirm",
                "horizon": 1,
                "date_count": 5,
                "avg_return": 0.01,
                "avg_universe_return": 0.009,
                "avg_alpha": 0.001,
                "avg_hit_rate": 0.51,
            }
        ]
    )

    gate = build_capital_flow_promotion_gate(summary, min_date_count=3)

    assert gate["overall_action"] == "keep_advisory"
    assert gate["decisions"][0]["status"] == "keep_advisory"
