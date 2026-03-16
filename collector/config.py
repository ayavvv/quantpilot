"""Configuration module - Pydantic-based environment variable management."""
import os
from pathlib import Path
from typing import List

try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Pydantic v1
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration."""

    # Futu API
    futu_host: str = Field(default="localhost", env="FUTU_HOST")
    futu_port: int = Field(default=11111, env="FUTU_PORT")

    # Target indexes / plates (comma-separated) - primary source, dynamically fetches constituents
    target_indexes: str = Field(
        default="",
        env="TARGET_INDEXES",
        description="Target index or plate codes, e.g. HK.800000,HK.800700",
    )

    # Target stock codes (comma-separated) - fallback when TARGET_INDEXES is empty
    target_codes: str = Field(
        default="",
        env="TARGET_CODES",
        description="Fallback: explicit stock codes, used only when TARGET_INDEXES is empty",
    )

    # Extra fixed codes (cross-asset: US indices, FX, etc.), always appended to the pool
    extra_codes_str: str = Field(
        default="",
        env="EXTRA_CODES",
        description="Extra fixed codes, e.g. US.SPY,US.QQQ",
    )

    # Data storage directory
    data_dir: str = Field(default="/data", env="DATA_DIR")

    # Cron schedule time (optional)
    cron_time: str = Field(default="16:30", env="CRON_TIME")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # ignore undefined extra fields

    @property
    def index_list(self) -> List[str]:
        """Parse target index list."""
        return [idx.strip() for idx in self.target_indexes.split(",") if idx.strip()]

    @property
    def code_list(self) -> List[str]:
        """Parse target code list (fallback)."""
        return [code.strip() for code in self.target_codes.split(",") if code.strip()]

    @property
    def extra_codes(self) -> List[str]:
        """Parse extra fixed code list."""
        return [c.strip() for c in self.extra_codes_str.split(",") if c.strip()]

    @property
    def data_path(self) -> Path:
        """Data directory path."""
        return Path(self.data_dir)

    def get_kline_path(self, code: str, freq: str, year: int = None) -> Path:
        """
        Get K-line data storage path.

        Args:
            code: Stock code
            freq: Frequency (K_DAY, K_1M, etc.)
            year: Year (optional, for 1-min sharding)

        Returns:
            Parquet file path
        """
        if year:
            return self.data_path / "kline" / freq / code / f"{year}.parquet"
        return self.data_path / "kline" / freq / code / "data.parquet"

    def get_ticker_path(self, code: str, date: str) -> Path:
        """
        Get tick data storage path.

        Args:
            code: Stock code
            date: Date (YYYY-MM-DD)

        Returns:
            Parquet file path
        """
        return self.data_path / "ticks" / code / f"{date}.parquet"

    def get_meta_db_path(self) -> Path:
        """Get metadata database path."""
        return self.data_path / "meta.duckdb"


# Global settings instance
settings = Settings()
