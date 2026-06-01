import pandas as pd

from converter.incremental import QlibDirectWriter
from strategy.major_force_eval import MajorForceEvalConfig, evaluate_major_force_forward_returns
from strategy.major_force import MajorForceConfig, scan_major_force


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
    assert result.iloc[0]["stage"] in {"accumulation_candidate", "watch"}
    assert "volume_expansion" in result.iloc[0]["reason"]


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
