from strategy.us_microstructure_confidence import build_confidence_gap


def test_confidence_gap_reports_validation_and_final_report_blockers():
    summary = build_confidence_gap(
        {
            "state": "warmup",
            "validated": False,
            "reason": "collecting samples",
            "validated_sides": {"accumulation": False, "distribution": False},
            "side_reasons": {"accumulation": "missing 5d validation metrics"},
            "side_metrics": {
                "accumulation": {
                    "observation_count": 12,
                    "signal_day_count": 4,
                    "avg_alpha": 0.0025,
                    "hit_rate": 0.52,
                    "recent_hit_rate": 0.5,
                    "wilson_lower": 0.25,
                    "max_symbol_sample_share": 0.4,
                }
            },
            "criteria": {
                "min_observations_per_side": 100,
                "min_signal_days_per_side": 20,
                "min_alpha": 0.0075,
                "min_hit_rate": 0.58,
                "min_recent_hit_rate": 0.55,
                "min_wilson_lower": 0.5,
                "max_symbol_sample_share": 0.2,
            },
            "event_count": 12,
            "forward_return_count": 24,
        },
        data_quality={"high_confidence_data_quality_ok": False, "nas_upload_complete": True},
        validation_eligibility={"validation_eligible_if_final_count": 3},
        intraday_replay={
            "cumulative_date_count": 2,
            "cumulative_quality_event_count": 5,
            "cumulative_quality_return_count": 5,
        },
        manifest_quality={"ok": True},
        is_final_report=False,
    )

    acc = summary["side_gaps"][0]
    assert summary["ready"] is False
    assert summary["requirements"]["final_report_complete"] is False
    assert "report is not a final post-close report" in summary["blockers"]
    assert acc["observations_needed"] == 88
    assert acc["signal_days_needed"] == 16
    assert round(acc["alpha_gap"], 4) == 0.005
    assert round(acc["hit_rate_gap"], 2) == 0.06
    assert round(acc["concentration_excess"], 2) == 0.2
    assert summary["validation_eligible_if_final_count"] == 3
    assert summary["cumulative_intraday_replay"]["quality_event_count"] == 5


def test_confidence_gap_ready_when_all_requirements_pass():
    summary = build_confidence_gap(
        {"state": "validated", "validated": True, "validated_sides": {"accumulation": True}},
        data_quality={"high_confidence_data_quality_ok": True},
        manifest_quality={"ok": True},
        is_final_report=True,
    )

    assert summary["ready"] is True
    assert summary["blockers"] == []
