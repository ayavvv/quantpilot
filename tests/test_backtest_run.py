import pandas as pd

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
