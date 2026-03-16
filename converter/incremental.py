"""
Qlib bin direct writer — collector writes Qlib format without parquet intermediate.

Two usage modes:
1. Direct write: collector calls write_stock_records() after each baostock fetch
2. Migration: update() reads existing parquet files and converts to bin

Qlib bin format:
  calendars/day.txt           — one date per line, sorted
  instruments/{market}.txt    — {code}\\t{start_date}\\t{end_date}
  features/{code}/{field}.day.bin — binary: [start_index(f32), val1(f32), val2(f32), ...]
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    from qlib.utils import code_to_fname
except ImportError:
    def code_to_fname(code: str) -> str:
        replace_names = ["CON", "PRN", "AUX", "NUL"] + [
            f"COM{i}" for i in range(10)
        ] + [f"LPT{i}" for i in range(10)]
        if str(code).upper() in replace_names:
            return "_qlib_" + str(code)
        return str(code)

DUMP_FIELDS = [
    "open", "close", "high", "low", "volume", "amount",
    "factor", "vwap", "pe_ratio", "turnover_rate", "change_rate",
]
DAILY_FMT = "%Y-%m-%d"
FREQ = "day"

# Column name aliases (baostock/futu → qlib standard)
COLUMN_ALIAS = {
    "time_key": "date",
    "turnover": "amount",
    "turn_over": "amount",
    "turn": "turnover_rate",
    "pctChg": "change_rate",
}


def _normalize_record(record: dict[str, Any]) -> dict[str, float]:
    """Normalize a single baostock/futu record to Qlib field values."""
    # Apply aliases
    r = {}
    for k, v in record.items():
        key = COLUMN_ALIAS.get(k, k)
        r[key] = v

    values = {}
    for field in DUMP_FIELDS:
        try:
            v = float(r.get(field, np.nan))
            values[field] = v if np.isfinite(v) else np.nan
        except (ValueError, TypeError):
            values[field] = np.nan

    # Compute vwap if missing
    if np.isnan(values.get("vwap", np.nan)):
        vol = values.get("volume", 0)
        amt = values.get("amount", 0)
        if vol and vol > 0 and np.isfinite(amt) and np.isfinite(vol):
            values["vwap"] = amt / vol
        elif all(np.isfinite(values.get(f, np.nan)) for f in ["open", "high", "low", "close"]):
            values["vwap"] = (values["open"] + values["high"] + values["low"] + values["close"]) / 4
        else:
            values["vwap"] = values.get("close", np.nan)

    # Factor default
    if np.isnan(values.get("factor", np.nan)):
        values["factor"] = 1.0

    # Clean PE (negative → NaN)
    pe = values.get("pe_ratio", np.nan)
    if np.isfinite(pe) and pe <= 0:
        values["pe_ratio"] = np.nan

    return values


class QlibDirectWriter:
    """
    Write stock data directly to Qlib bin format.

    Usage:
        writer = QlibDirectWriter("/data/qlib")
        # After each baostock fetch:
        writer.write_stock_records("SH.600000", records_list)
        # After all stocks done:
        writer.flush()
    """

    def __init__(self, qlib_dir: str | Path):
        self.qlib_dir = Path(qlib_dir)
        self.calendars_dir = self.qlib_dir / "calendars"
        self.instruments_dir = self.qlib_dir / "instruments"
        self.features_dir = self.qlib_dir / "features"

        # In-memory state (loaded lazily)
        self._calendar: list[str] | None = None
        self._cal_set: set[str] | None = None
        self._instruments: dict[str, tuple[str, str]] | None = None
        self._dirty = False

    @property
    def calendar(self) -> list[str]:
        if self._calendar is None:
            self._calendar = self._load_calendar()
            self._cal_set = set(self._calendar)
        return self._calendar

    @property
    def cal_set(self) -> set[str]:
        if self._cal_set is None:
            _ = self.calendar
        return self._cal_set

    @property
    def instruments(self) -> dict[str, tuple[str, str]]:
        if self._instruments is None:
            self._instruments = self._load_instruments("all")
        return self._instruments

    # --- Calendar ---

    def _load_calendar(self) -> list[str]:
        cal_path = self.calendars_dir / f"{FREQ}.txt"
        if not cal_path.exists():
            return []
        return [line.strip() for line in cal_path.read_text().splitlines() if line.strip()]

    def _save_calendar(self):
        self.calendars_dir.mkdir(parents=True, exist_ok=True)
        cal_path = self.calendars_dir / f"{FREQ}.txt"
        cal_path.write_text("\n".join(self.calendar) + "\n")

    def _ensure_date_in_calendar(self, date_str: str) -> int:
        """Add date to calendar if not present, return its index."""
        if date_str in self.cal_set:
            return bisect.bisect_left(self.calendar, date_str)

        idx = bisect.insort(self.calendar, date_str)
        self.cal_set.add(date_str)
        self._dirty = True
        return bisect.bisect_left(self.calendar, date_str)

    # --- Instruments ---

    def _load_instruments(self, market: str = "all") -> dict[str, tuple[str, str]]:
        inst_path = self.instruments_dir / f"{market}.txt"
        if not inst_path.exists():
            return {}
        result = {}
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                result[parts[0]] = (parts[1], parts[2])
        return result

    def _save_instruments(self):
        self.instruments_dir.mkdir(parents=True, exist_ok=True)

        # Save all.txt
        inst_path = self.instruments_dir / "all.txt"
        lines = [f"{code}\t{s}\t{e}" for code, (s, e) in sorted(self.instruments.items())]
        inst_path.write_text("\n".join(lines) + "\n")

        # Save per-market
        for market_tag, prefix in [("sh", "SH."), ("sz", "SZ."), ("hk", "HK.")]:
            market_insts = {k: v for k, v in self.instruments.items() if k.startswith(prefix)}
            mp = self.instruments_dir / f"{market_tag}.txt"
            if market_insts:
                lines = [f"{code}\t{s}\t{e}" for code, (s, e) in sorted(market_insts.items())]
                mp.write_text("\n".join(lines) + "\n")

    def _update_instrument(self, code: str, date_str: str):
        if code in self.instruments:
            old_start, old_end = self.instruments[code]
            self.instruments[code] = (min(old_start, date_str), max(old_end, date_str))
        else:
            self.instruments[code] = (date_str, date_str)

    # --- Bin file I/O ---

    def _get_feat_dir(self, code: str) -> Path:
        return self.features_dir / code_to_fname(code).lower()

    def _read_bin(self, bin_path: Path) -> tuple[int, np.ndarray] | None:
        """Read bin file, return (start_index, values_array) or None."""
        if not bin_path.exists():
            return None
        data = np.fromfile(str(bin_path), dtype="<f4")
        if len(data) == 0:
            return None
        return int(data[0]), data[1:]

    def _write_bin(self, bin_path: Path, start_index: int, values: np.ndarray):
        """Write complete bin file."""
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        result = np.concatenate([
            np.array([start_index], dtype="<f4"),
            values.astype("<f4"),
        ])
        result.tofile(str(bin_path))

    # --- Public API ---

    def get_stock_last_date(self, code: str) -> str | None:
        """Get the last collected date for a stock (from bin file)."""
        feat_dir = self._get_feat_dir(code)
        bin_path = feat_dir / f"close.{FREQ}.bin"
        result = self._read_bin(bin_path)
        if result is None:
            return None

        start_idx, values = result
        if len(values) == 0:
            return None

        last_idx = start_idx + len(values) - 1
        cal = self.calendar
        if last_idx < len(cal):
            return cal[last_idx]
        return cal[-1] if cal else None

    def write_stock_records(
        self, code: str, records: list[dict[str, Any]], skip_existing_check: bool = False,
    ) -> int:
        """
        Write stock daily records directly to Qlib bin format.

        Args:
            code: Stock code (e.g., "SH.600000")
            records: List of dicts from baostock/futu with fields:
                     date/time_key, open, close, high, low, volume, amount/turnover, ...
            skip_existing_check: Skip per-record bin file read for dedup (fast migration mode)

        Returns:
            Number of new dates written
        """
        if not records:
            return 0

        # Parse records, find new dates
        new_data = {}  # date_str -> field_values
        _checked_bin = None  # cached bin read for dedup
        for record in records:
            # Get date
            date_str = None
            for dk in ["date", "time_key"]:
                if dk in record and record[dk]:
                    date_str = str(record[dk])[:10]
                    break
            if not date_str:
                continue

            if not skip_existing_check and date_str in self.cal_set:
                # Check if this stock already has data for this date (read bin once)
                if _checked_bin is None:
                    feat_dir = self._get_feat_dir(code)
                    bin_path = feat_dir / f"close.{FREQ}.bin"
                    _checked_bin = self._read_bin(bin_path) or False
                if _checked_bin:
                    start_idx, values = _checked_bin
                    cal_idx = bisect.bisect_left(self.calendar, date_str)
                    data_idx = cal_idx - start_idx
                    if 0 <= data_idx < len(values) and np.isfinite(values[data_idx]):
                        continue  # Already have data

            new_data[date_str] = _normalize_record(record)

        if not new_data:
            return 0

        # Ensure all dates are in calendar
        for date_str in sorted(new_data.keys()):
            self._ensure_date_in_calendar(date_str)

        # Get or create start_index for this stock
        feat_dir = self._get_feat_dir(code)
        existing_start = None
        existing_values = {}

        for field in DUMP_FIELDS:
            bin_path = feat_dir / f"{field}.{FREQ}.bin"
            result = self._read_bin(bin_path)
            if result is not None:
                si, vals = result
                existing_start = si
                existing_values[field] = vals

        all_dates = sorted(new_data.keys())
        first_new_date = all_dates[0]

        if existing_start is None:
            # New stock
            existing_start = bisect.bisect_left(self.calendar, first_new_date)

        # Compute the full range this stock should cover
        last_new_date = all_dates[-1]
        last_cal_idx = bisect.bisect_left(self.calendar, last_new_date)
        total_length = last_cal_idx - existing_start + 1

        # Build/extend value arrays for each field
        for field in DUMP_FIELDS:
            old_vals = existing_values.get(field, np.array([], dtype="<f4"))
            # Extend to total_length, filling with NaN
            new_arr = np.full(total_length, np.nan, dtype="<f4")
            # Copy existing values
            new_arr[:len(old_vals)] = old_vals[:total_length]
            # Fill in new data
            for date_str, values in new_data.items():
                cal_idx = bisect.bisect_left(self.calendar, date_str)
                data_idx = cal_idx - existing_start
                if 0 <= data_idx < total_length:
                    v = values.get(field, np.nan)
                    new_arr[data_idx] = np.float32(v) if np.isfinite(v) else np.nan

            bin_path = feat_dir / f"{field}.{FREQ}.bin"
            self._write_bin(bin_path, existing_start, new_arr)

        # Update instruments
        for date_str in all_dates:
            self._update_instrument(code, date_str)

        self._dirty = True
        return len(new_data)

    def flush(self):
        """Save calendar and instruments to disk. Call after batch updates."""
        if not self._dirty:
            return
        self._save_calendar()
        self._save_instruments()
        self._dirty = False
        logger.info(f"Qlib data flushed: {len(self.calendar)} days, {len(self.instruments)} stocks")

    # --- Migration from parquet ---

    def migrate_from_parquet(self, kline_dir: str | Path) -> dict:
        """
        One-time: convert all parquet data to Qlib bin format.

        Args:
            kline_dir: Directory containing {code}/data.parquet files

        Returns:
            dict with migration stats
        """
        kline_dir = Path(kline_dir)
        stock_dirs = sorted([
            d for d in kline_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not stock_dirs:
            return {"status": "skip", "message": "no stocks found"}

        logger.info(f"Migrating {len(stock_dirs)} stocks from parquet to Qlib bin...")

        migrated = 0
        for i, d in enumerate(stock_dirs):
            code = d.name
            parquet_files = list(d.glob("*.parquet"))
            if not parquet_files:
                continue
            try:
                df = pd.read_parquet(parquet_files[0])

                # Normalize date column
                if "time_key" in df.columns:
                    df["date"] = pd.to_datetime(df["time_key"]).dt.strftime(DAILY_FMT)
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime(DAILY_FMT)
                else:
                    continue

                # Convert to records
                records = []
                for _, row in df.iterrows():
                    r = row.to_dict()
                    r["date"] = row["date"]
                    records.append(r)

                n = self.write_stock_records(code, records, skip_existing_check=True)
                if n > 0:
                    migrated += 1

                if (i + 1) % 200 == 0:
                    logger.info(f"  Progress: {i+1}/{len(stock_dirs)} ({migrated} migrated)")
                    self.flush()  # Periodic flush

            except Exception as e:
                logger.warning(f"  {code}: failed - {e}")

        self.flush()

        logger.info(f"Migration complete: {migrated}/{len(stock_dirs)} stocks")
        return {
            "status": "ok",
            "total_stocks": len(stock_dirs),
            "migrated": migrated,
            "calendar_days": len(self.calendar),
        }


# --- Reader for downstream modules ---

class QlibBinReader:
    """Read Qlib bin data for downstream modules (observer, trader, reporter, backtest)."""

    def __init__(self, qlib_dir: str | Path):
        self.qlib_dir = Path(qlib_dir)
        self._calendar: list[str] | None = None

    @property
    def calendar(self) -> list[str]:
        if self._calendar is None:
            cal_path = self.qlib_dir / "calendars" / f"{FREQ}.txt"
            if cal_path.exists():
                self._calendar = [l.strip() for l in cal_path.read_text().splitlines() if l.strip()]
            else:
                self._calendar = []
        return self._calendar

    @property
    def latest_date(self) -> str | None:
        return self.calendar[-1] if self.calendar else None

    def list_instruments(self, market: str = "all") -> dict[str, tuple[str, str]]:
        """Return {code: (start_date, end_date)} from instruments file."""
        inst_path = self.qlib_dir / "instruments" / f"{market}.txt"
        if not inst_path.exists():
            return {}
        result = {}
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                result[parts[0]] = (parts[1], parts[2])
        return result

    def _read_bin(self, bin_path: Path) -> tuple[int, np.ndarray] | None:
        if not bin_path.exists():
            return None
        data = np.fromfile(str(bin_path), dtype="<f4")
        if len(data) == 0:
            return None
        return int(data[0]), data[1:]

    def read_field(self, code: str, field: str) -> pd.Series:
        """Read a single field for a stock, return Series indexed by date string."""
        feat_dir = self.qlib_dir / "features" / code_to_fname(code).lower()
        bin_path = feat_dir / f"{field}.{FREQ}.bin"
        result = self._read_bin(bin_path)
        if result is None:
            return pd.Series(dtype="float64")
        start_idx, values = result
        cal = self.calendar
        end_idx = start_idx + len(values)
        dates = cal[start_idx:end_idx]
        return pd.Series(values.astype("float64"), index=dates, name=field)

    def read_stock(self, code: str, fields: list[str] | None = None) -> pd.DataFrame:
        """Read multiple fields for a stock, return DataFrame indexed by date."""
        if fields is None:
            fields = ["open", "close", "high", "low", "volume", "amount", "change_rate"]
        data = {}
        for f in fields:
            s = self.read_field(code, f)
            if not s.empty:
                data[f] = s
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.index.name = "date"
        return df

    def read_field_matrix(self, codes: list[str], field: str,
                          start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Read one field across multiple stocks, return DataFrame (date x code)."""
        frames = {}
        for code in codes:
            s = self.read_field(code, field)
            if not s.empty:
                frames[code] = s
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames).sort_index()
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]
        return df


# --- Convenience functions ---

def incremental_update(kline_dir: str, qlib_dir: str) -> dict:
    """Backward-compatible: update from parquet (for legacy pipelines)."""
    writer = QlibDirectWriter(qlib_dir)
    return writer.migrate_from_parquet(kline_dir)
