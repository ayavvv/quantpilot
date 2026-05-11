import pickle

import pandas as pd

from trainer.backtest.data_loader import load_predictions
from trainer.backtest.backtest import run_backtest


def test_run_backtest_matches_timestamp_predictions_with_string_price_index():
    pred_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-01"), "SH.600000"),
            (pd.Timestamp("2026-04-01"), "SH.600001"),
            (pd.Timestamp("2026-04-02"), "SH.600000"),
            (pd.Timestamp("2026-04-02"), "SH.600001"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([1.0, 0.5, 0.8, 0.3], index=pred_index)

    close_df = pd.DataFrame(
        {
            "SH.600000": [10.0, 11.0, 12.0, 13.0],
            "SH.600001": [20.0, 19.5, 19.0, 18.5],
        },
        index=["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06"],
    )

    results = run_backtest(pred, close_df, top_n=1)

    assert len(results) == 2
    assert list(results["signal_date"].dt.strftime("%Y-%m-%d")) == ["2026-04-01", "2026-04-02"]
    assert list(results["entry_date"].dt.strftime("%Y-%m-%d")) == ["2026-04-02", "2026-04-03"]
    assert list(results["exit_date"].dt.strftime("%Y-%m-%d")) == ["2026-04-03", "2026-04-06"]


def test_load_predictions_filters_to_allowed_prefixes(tmp_path):
    pred_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-01"), "SH.600000"),
            (pd.Timestamp("2026-04-01"), "SZ.000001"),
            (pd.Timestamp("2026-04-01"), "HK.00700"),
            (pd.Timestamp("2026-04-01"), "MACRO.VIX"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([1.0, 0.9, 0.8, 0.7], index=pred_index)
    pred_path = tmp_path / "pred.pkl"
    pred_path.write_bytes(pickle.dumps(pred))

    filtered = load_predictions(pred_path, allowed_prefixes=("SH.",))

    assert list(filtered.index.get_level_values("instrument").unique()) == ["SH.600000"]


def test_run_backtest_skips_point_in_time_st_names():
    pred_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-01"), "SH.600000"),
            (pd.Timestamp("2026-04-01"), "SH.600001"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([1.0, 0.5], index=pred_index)

    close_df = pd.DataFrame(
        {
            "SH.600000": [10.0, 10.0, 20.0],
            "SH.600001": [10.0, 10.0, 11.0],
        },
        index=["2026-04-01", "2026-04-02", "2026-04-03"],
    )
    st_df = pd.DataFrame(
        {
            "SH.600000": [1.0, 1.0, 1.0],
            "SH.600001": [0.0, 0.0, 0.0],
        },
        index=close_df.index,
    )

    results = run_backtest(pred, close_df, top_n=1, st_df=st_df)

    assert len(results) == 1
    assert results.iloc[0]["positions"] == "SH.600001"


def test_run_backtest_applies_stop_loss_without_rebuying_same_day():
    pred_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-01"), "SH.600000"),
            (pd.Timestamp("2026-04-01"), "SH.600001"),
            (pd.Timestamp("2026-04-02"), "SH.600000"),
            (pd.Timestamp("2026-04-02"), "SH.600001"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.Series([1.0, 0.9, 1.0, 0.8], index=pred_index)

    close_df = pd.DataFrame(
        {
            "SH.600000": [10.0, 10.0, 9.0, 9.2],
            "SH.600001": [5.0, 5.0, 5.5, 6.0],
        },
        index=["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06"],
    )
    change_df = pd.DataFrame(
        {
            "SH.600000": [0.0, 0.0, -10.0, 0.0],
            "SH.600001": [0.0, 0.0, 10.0, 0.0],
        },
        index=close_df.index,
    )

    results = run_backtest(
        pred,
        close_df,
        top_n=1,
        hold_bonus=0.05,
        change_df=change_df,
        filter_limit_up=False,
        stop_loss_pct=-0.08,
        position_ratio=1.0,
    )

    assert len(results) == 2
    assert results.iloc[0]["positions"] == "SH.600000"
    assert results.iloc[1]["positions"] == "SH.600001"
    assert results.iloc[1]["n_stop_loss_sells"] == 1
    assert results.iloc[1]["n_buys"] == 1
