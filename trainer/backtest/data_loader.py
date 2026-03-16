"""Data loading: read pred.pkl and price data from Qlib bin format."""

import pickle
from pathlib import Path

import pandas as pd

from .config import NON_TRADEABLE_PREFIXES


def load_predictions(pred_path: Path) -> pd.Series:
    """Load pred.pkl, filter out non-tradeable codes (MACRO, etc.)."""
    with open(pred_path, "rb") as f:
        pred = pickle.load(f)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    # Filter non-tradeable codes
    instruments = pred.index.get_level_values("instrument")
    mask = ~instruments.str.startswith(NON_TRADEABLE_PREFIXES)
    pred = pred[mask]
    print(f"Loaded predictions: {len(pred.index.get_level_values('datetime').unique())} days, "
          f"{len(pred.index.get_level_values('instrument').unique())} instruments")
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
