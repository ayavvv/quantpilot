"""Data loading: read pred.pkl and price data from Qlib bin format."""

import pickle
from pathlib import Path

import pandas as pd

from .config import NON_TRADEABLE_PREFIXES, TRADEABLE_PREFIXES


def load_predictions(
    pred_path: Path,
    allowed_prefixes: tuple[str, ...] = TRADEABLE_PREFIXES,
) -> pd.Series:
    """Load pred.pkl and keep only backtest-tradeable instruments."""
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    instruments = pred.index.get_level_values("instrument")
    mask = ~instruments.str.startswith(NON_TRADEABLE_PREFIXES)
    if allowed_prefixes:
        mask &= instruments.str.startswith(allowed_prefixes)
    pred = pred[mask]
    print(f"Loaded predictions: {len(pred.index.get_level_values('datetime').unique())} days, "
          f"{len(pred.index.get_level_values('instrument').unique())} instruments"
          f" | prefixes={allowed_prefixes or 'ALL'}")
    return pred


def load_close_prices(data_dir: Path, instruments: list[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
    """Build close price matrix (date x code) from Qlib bin data."""
    from converter.incremental import QlibBinReader
    reader = QlibBinReader(data_dir)
    df = reader.read_field_matrix(instruments, "close", start_date, end_date)
    print(f"Price matrix: {df.shape[0]} days x {df.shape[1]} instruments")
    return df


def load_change_rates(data_dir: Path, instruments: list[str],
                      start_date: str, end_date: str) -> pd.DataFrame:
    """Build change rate matrix (date x code) from Qlib bin data."""
    from converter.incremental import QlibBinReader
    reader = QlibBinReader(data_dir)
    return reader.read_field_matrix(instruments, "change_rate", start_date, end_date)


def load_st_flags(data_dir: Path, instruments: list[str],
                  start_date: str, end_date: str) -> pd.DataFrame:
    """Build point-in-time ST flag matrix (date x code) from Qlib bin data."""
    from converter.incremental import QlibBinReader
    reader = QlibBinReader(data_dir)
    return reader.read_field_matrix(instruments, "is_st", start_date, end_date)
