"""Storage engine - Parquet read/write and DuckDB job logging."""
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import duckdb
from loguru import logger


class DBEngine:
    """Data storage engine."""

    def __init__(self, data_dir: Path):
        """
        Initialize storage engine.

        Args:
            data_dir: Data storage root directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_db_path = self.data_dir / "meta.duckdb"
        self._init_meta_db()

    @property
    def data_path(self):
        """Alias for data_dir (used by scheduler)."""
        return self.data_dir

    def _init_meta_db(self):
        """Initialize metadata database."""
        try:
            conn = duckdb.connect(str(self.meta_db_path))
            try:
                conn.execute("SELECT 1 FROM job_logs LIMIT 1")
                table_exists = True
            except duckdb.CatalogException:
                table_exists = False

            if not table_exists:
                conn.execute("""
                    CREATE TABLE job_logs (
                        id BIGINT PRIMARY KEY,
                        timestamp TIMESTAMP,
                        status TEXT,
                        message TEXT,
                        code TEXT,
                        data_type TEXT
                    )
                """)
            conn.close()
            logger.info(f"Metadata database initialized: {self.meta_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize metadata database: {e}")
            raise

    def get_kline_max_date(
        self,
        code: str,
        freq: str,
        year: Optional[int] = None
    ) -> Optional[str]:
        """
        Get the max date of collected K-line data (for incremental fetching).

        Args:
            code: Stock code
            freq: Frequency (K_DAY, K_1M, etc.)
            year: Year (optional, for 1-min sharding)

        Returns:
            Max date "YYYY-MM-DD", or None if no data
        """
        if year:
            file_path = self.data_dir / "kline" / freq / code / f"{year}.parquet"
        else:
            file_path = self.data_dir / "kline" / freq / code / "data.parquet"

        if not file_path.exists():
            return None
        conn = None
        try:
            conn = duckdb.connect(":memory:")
            row = conn.execute(
                "SELECT max(CAST(time_key AS DATE)) FROM read_parquet(?)",
                [str(file_path)]
            ).fetchone()
            if row and row[0] is not None:
                return row[0].strftime("%Y-%m-%d")
            return None
        except Exception as e:
            logger.warning(f"Failed to read K-line max date: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_kline_count_in_range(
        self,
        code: str,
        freq: str,
        start: str,
        end: str,
        year: Optional[int] = None
    ) -> int:
        """
        Get count of collected K-line records in a date range.

        Args:
            code: Stock code
            freq: Frequency (K_DAY, K_1M, etc.)
            start: Range start date (YYYY-MM-DD)
            end: Range end date (YYYY-MM-DD)
            year: Year (optional, for 1-min sharding)

        Returns:
            Record count in range, 0 if no file or error
        """
        if year:
            file_path = self.data_dir / "kline" / freq / code / f"{year}.parquet"
        else:
            file_path = self.data_dir / "kline" / freq / code / "data.parquet"

        if not file_path.exists():
            return 0
        conn = None
        try:
            conn = duckdb.connect(":memory:")
            row = conn.execute(
                """
                SELECT count(*) FROM read_parquet(?)
                WHERE CAST(time_key AS DATE) >= CAST(? AS DATE) AND CAST(time_key AS DATE) <= CAST(? AS DATE)
                """,
                [str(file_path), start, end],
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.warning(f"Failed to read K-line range count: {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def ticker_file_exists(self, code: str, date: str) -> bool:
        """
        Check if tick data file exists for a given date.

        Args:
            code: Stock code
            date: Date (YYYY-MM-DD)

        Returns:
            Whether the tick file exists
        """
        file_path = self.data_dir / "ticks" / code / f"{date}.parquet"
        return file_path.exists()

    def append_kline(
        self,
        df: pd.DataFrame,
        code: str,
        freq: str,
        year: Optional[int] = None
    ) -> int:
        """
        Append K-line data to Parquet file.

        Args:
            df: K-line data DataFrame
            code: Stock code
            freq: Frequency (K_DAY, K_1M, etc.)
            year: Year (optional, for 1-min sharding)

        Returns:
            Number of records actually appended
        """
        if df is None or len(df) == 0:
            logger.warning(f"{code} {freq} data is empty, skipping save")
            return 0

        # Determine file path
        if year:
            file_path = self.data_dir / "kline" / freq / code / f"{year}.parquet"
        else:
            file_path = self.data_dir / "kline" / freq / code / "data.parquet"

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure time_key column exists
        if 'time_key' not in df.columns:
            logger.error("DataFrame missing time_key column")
            return 0

        # Convert time_key to datetime
        df['time_key'] = pd.to_datetime(df['time_key'])

        # Sort by time_key
        df = df.sort_values('time_key').reset_index(drop=True)

        # Read existing data if present
        existing_df = None
        if file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                existing_df['time_key'] = pd.to_datetime(existing_df['time_key'])
                logger.info(f"Read existing data: {len(existing_df)} records")
            except Exception as e:
                logger.warning(f"Failed to read existing Parquet file: {e}, creating new file")

        # Merge and deduplicate
        if existing_df is not None and len(existing_df) > 0:
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['time_key'], keep='last')
            combined_df = combined_df.sort_values('time_key').reset_index(drop=True)
            new_count = len(combined_df) - len(existing_df)
        else:
            combined_df = df
            new_count = len(df)

        # Save to Parquet
        try:
            combined_df.to_parquet(
                file_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            logger.info(
                f"Saved {code} {freq} data to {file_path} "
                f"(new: {new_count}, total: {len(combined_df)})"
            )
            return new_count
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            raise

    def append_ticker(
        self,
        df: pd.DataFrame,
        code: str,
        date: str
    ) -> int:
        """
        Append tick data to Parquet file.

        Args:
            df: Tick data DataFrame
            code: Stock code
            date: Date (YYYY-MM-DD)

        Returns:
            Number of records actually appended
        """
        if df is None or len(df) == 0:
            logger.warning(f"{code} {date} tick data is empty, skipping save")
            return 0

        # Determine file path
        file_path = self.data_dir / "ticks" / code / f"{date}.parquet"

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure time column exists
        if 'time' not in df.columns:
            logger.error("DataFrame missing time column")
            return 0

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Read existing data if present
        existing_df = None
        if file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                existing_df['time'] = pd.to_datetime(existing_df['time'])
                logger.info(f"Read existing data: {len(existing_df)} records")
            except Exception as e:
                logger.warning(f"Failed to read existing Parquet file: {e}, creating new file")

        # Merge and deduplicate
        if existing_df is not None and len(existing_df) > 0:
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            new_count = len(combined_df) - len(existing_df)
        else:
            combined_df = df
            new_count = len(df)

        # Save to Parquet
        try:
            combined_df.to_parquet(
                file_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            logger.info(
                f"Saved {code} {date} tick data to {file_path} "
                f"(new: {new_count}, total: {len(combined_df)})"
            )
            return new_count
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            raise

    def log_job(
        self,
        status: str,
        message: str,
        code: Optional[str] = None,
        data_type: Optional[str] = None
    ):
        """
        Log job execution to DuckDB.

        Args:
            status: Status (success, error, warning, etc.)
            message: Log message
            code: Stock code (optional)
            data_type: Data type (optional, e.g. K_DAY, K_1M, Ticker)
        """
        conn = None
        try:
            conn = duckdb.connect(str(self.meta_db_path))
            max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) FROM job_logs").fetchone()
            next_id = (max_id_result[0] if max_id_result else 0) + 1

            conn.execute("""
                INSERT INTO job_logs (id, timestamp, status, message, code, data_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [next_id, datetime.now(), status, message, code, data_type])
            logger.debug(f"Logged: {status} - {message}")
        except Exception as e:
            logger.error(f"Failed to log job: {e}")
        finally:
            if conn:
                conn.close()
