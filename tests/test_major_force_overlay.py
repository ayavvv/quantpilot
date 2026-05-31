import pandas as pd

from strategy.major_force_overlay import build_major_force_overlay


def test_overlay_flags_model_candidate_with_major_risk():
    signals = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 0.9, "rank": 1, "signal_date": "2026-05-29"},
            {"code": "SZ.000001", "score": 0.8, "rank": 2, "signal_date": "2026-05-29"},
        ]
    )
    major = pd.DataFrame(
        [
            {
                "code": "SH.600000",
                "rank": 5,
                "score": 91.0,
                "stage": "washout_or_risk",
                "reason": "20d_positive_flow,sharp_down_day",
                "cmf_20": 0.2,
                "amount_ratio_5_20": 1.5,
                "today_chg_pct": -7.0,
            },
            {
                "code": "SZ.000001",
                "rank": 10,
                "score": 90.0,
                "stage": "accumulation_candidate",
                "reason": "20d_positive_flow,volume_expansion",
                "cmf_20": 0.3,
                "amount_ratio_5_20": 1.8,
                "today_chg_pct": 2.0,
            },
        ]
    )

    overlay = build_major_force_overlay(signals, major, signal_top_n=10, confirm_rank=20, confirm_score=80)

    labels = dict(zip(overlay["code"], overlay["overlay_label"]))
    assert labels["SH.600000"] == "risk_flag_major_washout"
    assert labels["SZ.000001"] == "secondary_confirm_accumulation"
    assert overlay.iloc[0]["code"] == "SH.600000"


def test_overlay_keeps_model_only_candidate_when_major_missing():
    signals = pd.DataFrame(
        [{"code": "SH.600000", "score": 0.9, "rank": 1, "signal_date": "2026-05-29"}]
    )

    overlay = build_major_force_overlay(signals, pd.DataFrame(), signal_top_n=10)

    assert overlay.iloc[0]["overlay_label"] == "model_only_no_major_signal"
    assert overlay.iloc[0]["model_top5"]
