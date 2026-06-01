import pandas as pd

from converter.incremental import QlibDirectWriter
from strategy.major_force_eval import MajorForceEvalConfig, evaluate_major_force_forward_returns
from strategy.major_force import MajorForceConfig, scan_major_force
from strategy.major_force_validate import ValidationCriteria, validate_major_force_eval


def _records(dates, pattern):
    rows = []
    close = 10.0
    prev_close = close
    for idx, day in enumerate(dates):
        if pattern == "accumulation":
            close *= 1.001 if idx < len(dates) - 10 else 1.004
            open_price = close * 0.992
            low = open_price * 0.996
            high = close * 1.002
            amount = 40_000_000 if idx < len(dates) - 10 else 95_000_000
            turnover_rate = 0.8 if idx < len(dates) - 10 else 1.8
        elif pattern == "distribution":
            close *= 1.001 if idx < len(dates) - 20 else 0.992
            open_price = close * 1.014
            low = close * 0.998
            high = open_price * 1.004
            amount = 45_000_000 if idx < len(dates) - 20 else 130_000_000
            turnover_rate = 0.8 if idx < len(dates) - 20 else 2.2
        else:
            close *= 0.999
            open_price = close * 1.008
            low = close * 0.998
            high = open_price * 1.002
            amount = 55_000_000
            turnover_rate = 1.0

        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": amount / close,
                "turnover": amount,
                "turnover_rate": turnover_rate,
                "change_rate": (close / prev_close - 1.0) * 100.0,
                "is_st": 0.0,
            }
        )
        prev_close = close
    return rows


def test_scan_major_force_ranks_accumulation_footprint_first(tmp_path):
    qlib_dir = tmp_path / "qlib"
    writer = QlibDirectWriter(qlib_dir)
    dates = pd.bdate_range("2026-01-01", periods=80)
    writer.write_stock_records("SH.600000", _records(dates, "accumulation"))
    writer.write_stock_records("SZ.000001", _records(dates, "weak"))
    writer.flush()

    result = scan_major_force(
        qlib_dir,
        as_of_date=dates[-1].strftime("%Y-%m-%d"),
        config=MajorForceConfig(min_amount=0, min_history=60, exclude_limit_up=False),
        top_n=None,
    )

    assert result["code"].tolist()[0] == "SH.600000"
    scores = dict(zip(result["code"], result["score"]))
    assert scores["SH.600000"] > scores["SZ.000001"]
    assert result.iloc[0]["stage"] in {"stealth_accumulation", "accumulation_candidate", "watch"}
    assert "volume_expansion" in result.iloc[0]["reason"]
    assert "stealth_score" in result.columns
    assert "market_positive_rate_20" in result.columns


def test_scan_major_force_excludes_st_by_default(tmp_path):
    qlib_dir = tmp_path / "qlib"
    writer = QlibDirectWriter(qlib_dir)
    dates = pd.bdate_range("2026-01-01", periods=70)
    rows = _records(dates, "accumulation")
    for row in rows:
        row["is_st"] = 1.0
    writer.write_stock_records("SH.600001", rows)
    writer.flush()

    result = scan_major_force(
        qlib_dir,
        as_of_date=dates[-1].strftime("%Y-%m-%d"),
        config=MajorForceConfig(min_amount=0, min_history=60),
        top_n=None,
    )

    assert result.empty


def test_evaluate_major_force_forward_returns_outputs_summary(tmp_path):
    qlib_dir = tmp_path / "qlib"
    writer = QlibDirectWriter(qlib_dir)
    dates = pd.bdate_range("2026-01-01", periods=90)
    writer.write_stock_records("SH.600000", _records(dates, "accumulation"))
    writer.write_stock_records("SZ.000001", _records(dates, "weak"))
    writer.flush()

    summary, daily, picks = evaluate_major_force_forward_returns(
        qlib_dir,
        start_date=dates[65].strftime("%Y-%m-%d"),
        end_date=dates[75].strftime("%Y-%m-%d"),
        scan_config=MajorForceConfig(min_amount=0, min_history=40, lookback_days=50, exclude_limit_up=False),
        eval_config=MajorForceEvalConfig(top_ns=(1,), horizons=(5,), entry_lag_days=1, min_active_stocks=1),
    )

    assert not summary.empty
    assert not daily.empty
    assert not picks.empty
    row = summary.iloc[0]
    assert row["top_n"] == 1
    assert row["horizon"] == 5
    assert row["avg_return"] > row["avg_universe_return"]
    assert "market_return_20" in daily.columns
    assert "market_positive_rate_20" in picks.columns


def test_evaluate_major_force_forward_returns_scores_sell_side(tmp_path):
    qlib_dir = tmp_path / "qlib"
    writer = QlibDirectWriter(qlib_dir)
    dates = pd.bdate_range("2026-01-01", periods=100)
    writer.write_stock_records("SH.600000", _records(dates, "distribution"))
    writer.write_stock_records("SZ.000001", _records(dates, "weak"))
    writer.flush()

    summary, daily, picks = evaluate_major_force_forward_returns(
        qlib_dir,
        start_date=dates[72].strftime("%Y-%m-%d"),
        end_date=dates[88].strftime("%Y-%m-%d"),
        scan_config=MajorForceConfig(
            min_amount=0,
            min_history=40,
            lookback_days=60,
            exclude_limit_down=False,
        ),
        eval_config=MajorForceEvalConfig(
            sides=("sell",),
            top_ns=(1,),
            horizons=(5,),
            entry_lag_days=1,
            min_active_stocks=1,
            min_score=50,
            stages=("weak", "watch", "distribution_risk"),
        ),
    )

    assert not summary.empty
    assert set(summary["signal_side"]) == {"sell"}
    row = summary.iloc[0]
    assert row["avg_hit_rate"] > 0.5
    assert row["win_rate_days"] > 0.5
    assert picks["signal_side"].unique().tolist() == ["sell"]


def test_validate_major_force_eval_exports_validated_rule(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    daily_rows = [
        f"{date},buy,10,10,0.0"
        for date in dates
    ]
    (eval_dir / "daily.csv").write_text(
        "date,signal_side,horizon,top_n,universe_return\n" + "\n".join(daily_rows),
        encoding="utf-8",
    )
    pick_rows = [
        f"{date},buy,1,90,1.5,0.2,0.7,0.0,0.02"
        for date in dates
    ]
    (eval_dir / "picks.csv").write_text(
        "eval_date,signal_side,rank,side_score,amount_ratio_5_20,cmf_20,close_location_10,breakout_20,fwd_return_10d\n"
        + "\n".join(pick_rows),
        encoding="utf-8",
    )

    payload = validate_major_force_eval(
        eval_dir,
        criteria=ValidationCriteria(
            min_train_dates=1,
            min_test_dates=1,
            min_train_alpha=0.001,
            min_test_alpha=0.001,
            min_test_hit_rate=0.5,
            min_test_win_rate_days=0.5,
            min_recent_dates=1,
            min_recent_alpha=0.001,
            min_recent_hit_rate=0.5,
            min_recent_win_rate_days=0.5,
            split_ratio=0.5,
        ),
    )

    assert payload["validated"] is True
    assert payload["rules"][0]["side"] == "buy"
    assert payload["rules"][0]["test"]["avg_alpha"] > 0
    assert payload["candidate_rule_count_by_side"]["buy"] > 0
    assert payload["train_passed_count_by_side"]["buy"] > 0
    assert payload["best_rules_by_side"]["buy"][0]["side"] == "buy"


def test_validate_major_force_eval_requires_recent_robustness(tmp_path):
    eval_dir = tmp_path / "eval_recent"
    eval_dir.mkdir()
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    daily_rows = [
        f"{date},buy,10,10,0.0"
        for date in dates
    ]
    (eval_dir / "daily.csv").write_text(
        "date,signal_side,horizon,top_n,universe_return\n" + "\n".join(daily_rows),
        encoding="utf-8",
    )
    pick_rows = [
        "2026-01-02,buy,1,90,1.5,0.2,0.7,0.0,0.02",
        "2026-01-09,buy,1,90,1.5,0.2,0.7,0.0,0.02",
        "2026-01-16,buy,1,90,1.5,0.2,0.7,0.0,0.02",
        "2026-01-23,buy,1,90,1.5,0.2,0.7,0.0,-0.02",
    ]
    (eval_dir / "picks.csv").write_text(
        "eval_date,signal_side,rank,side_score,amount_ratio_5_20,cmf_20,close_location_10,breakout_20,fwd_return_10d\n"
        + "\n".join(pick_rows),
        encoding="utf-8",
    )

    payload = validate_major_force_eval(
        eval_dir,
        criteria=ValidationCriteria(
            min_train_dates=1,
            min_test_dates=1,
            min_train_alpha=0.001,
            min_test_alpha=-0.001,
            min_test_hit_rate=0.5,
            min_test_win_rate_days=0.5,
            min_recent_dates=1,
            min_recent_alpha=0.001,
            min_recent_hit_rate=0.5,
            min_recent_win_rate_days=0.5,
            split_ratio=0.5,
            recent_ratio=0.25,
        ),
    )

    assert payload["validated"] is False
    assert payload["best_rules"][0]["recent"]["avg_alpha"] < 0
    assert payload["best_rules_by_side"]["buy"][0]["recent"]["avg_alpha"] < 0
