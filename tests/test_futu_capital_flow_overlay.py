import pandas as pd

from strategy.futu_capital_flow_overlay import (
    archive_capital_flow_outputs,
    build_capital_flow_overlay,
    summarize_capital_flow,
)


def test_summarize_capital_flow_builds_recent_flow_features():
    records = [
        {"code": "SH.600000", "date": "2026-05-26", "main_in_flow": -1, "super_in_flow": 1, "big_in_flow": 2},
        {"code": "SH.600000", "date": "2026-05-27", "main_in_flow": 10, "super_in_flow": 3, "big_in_flow": 7},
        {"code": "SH.600000", "date": "2026-05-28", "main_in_flow": 20, "super_in_flow": 5, "big_in_flow": 15},
        {"code": "SH.600000", "date": "2026-05-29", "main_in_flow": 30, "super_in_flow": 8, "big_in_flow": 22},
    ]
    distribution = {
        "net_main": 30,
        "net_super": 8,
        "net_big": 22,
        "capital_in_main": 130,
        "capital_out_main": 100,
        "update_time": "2026-05-29 15:00:00",
    }

    summary = summarize_capital_flow("SH.600000", records, distribution)

    assert summary["capital_flow_status"] == "ok"
    assert summary["capital_flow_count"] == 4
    assert summary["capital_flow_latest_date"] == "2026-05-29"
    assert summary["latest_main_in_flow"] == 30
    assert summary["main_3d_sum"] == 60
    assert summary["main_positive_5d"] == 3
    assert summary["distribution_main_in_out_ratio"] == 1.3


def test_build_capital_flow_overlay_classifies_confirm_watch_and_risk():
    signals = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 0.9, "rank": 1, "signal_date": "2026-05-29"},
            {"code": "SZ.000001", "score": 0.8, "rank": 2, "signal_date": "2026-05-29"},
            {"code": "SZ.000002", "score": 0.7, "rank": 3, "signal_date": "2026-05-29"},
            {"code": "SH.600001", "score": 0.6, "rank": 4, "signal_date": "2026-05-29"},
        ]
    )
    flow = pd.DataFrame(
        [
            {
                "code": "SH.600000",
                "capital_flow_status": "ok",
                "latest_main_in_flow": 15_000_000,
                "main_5d_sum": 35_000_000,
                "main_positive_5d": 4,
                "distribution_net_main": 10_000_000,
            },
            {
                "code": "SZ.000001",
                "capital_flow_status": "ok",
                "latest_main_in_flow": -6_000_000,
                "main_5d_sum": -1_000_000,
                "main_positive_5d": 2,
            },
            {
                "code": "SZ.000002",
                "capital_flow_status": "ok",
                "latest_main_in_flow": 2_000_000,
                "main_5d_sum": 3_000_000,
                "main_positive_5d": 2,
            },
        ]
    )

    overlay = build_capital_flow_overlay(signals, flow, signal_top_n=10)

    labels = dict(zip(overlay["code"], overlay["capital_flow_label"]))
    assert labels["SH.600000"] == "capital_flow_confirm"
    assert labels["SZ.000001"] == "risk_flag_main_outflow"
    assert labels["SZ.000002"] == "capital_flow_watch"
    assert labels["SH.600001"] == "model_only_no_capital_flow"


def test_archive_capital_flow_outputs_writes_dated_files(tmp_path):
    flow = pd.DataFrame([{"code": "SH.600000", "latest_main_in_flow": 1.0}])
    overlay = pd.DataFrame([{"code": "SH.600000", "signal_date": "2026-05-29", "capital_flow_label": "watch"}])

    paths = archive_capital_flow_outputs(flow, overlay, tmp_path, archive_date="2026-05-29")

    assert paths["flow"].name == "20260529_flow.csv"
    assert paths["overlay"].name == "20260529_overlay.csv"
    assert paths["flow"].exists()
    assert paths["overlay"].exists()
