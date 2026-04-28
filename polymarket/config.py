"""Configuration for isolated Polymarket paper-trading."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolySettings(BaseSettings):
    """Polymarket-only runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="POLY_",
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: str = Field(default="")
    gamma_base_url: str = Field(default="https://gamma-api.polymarket.com")
    clob_base_url: str = Field(default="https://clob.polymarket.com")
    chain_id: int = Field(default=137)
    scan_interval_seconds: int = Field(default=300)
    catalog_refresh_seconds: int = Field(default=900)
    catalog_refresh_job_seconds: int = Field(default=1800)
    min_net_edge: float = Field(default=0.01)
    max_book_staleness_ms: int = Field(default=15000)
    default_gas_cost: float = Field(default=0.0)
    slippage_buffer: float = Field(default=0.005)
    max_notional_per_opp: float = Field(default=250.0)
    paper_initial_cash: float = Field(default=1000.0)
    paper_only: bool = Field(default=True)
    enable_split_sell: bool = Field(default=False)
    max_active_markets: int = Field(default=250)
    http_timeout_seconds: int = Field(default=20)
    user_agent: str = Field(default="Mozilla/5.0")
    enable_top_trader_mirror: bool = Field(default=False)
    top_trader_candidate_limit: int = Field(default=25)
    top_trader_tracked_count: int = Field(default=3)
    top_trader_min_trades: int = Field(default=10)
    top_trader_min_diversity: int = Field(default=3)
    top_trader_poll_seconds: int = Field(default=300)
    top_trader_mirror_lag_seconds: int = Field(default=300)
    top_trader_min_signal_size: float = Field(default=50.0)
    top_trader_max_signal_notional: float = Field(default=100.0)
    book_fetch_use_batch: bool = Field(default=True)
    book_fetch_batch_size: int = Field(default=500)
    book_fetch_workers: int = Field(default=8)
    book_top_retention_hours: int = Field(default=72)
    book_top_retention_job_seconds: int = Field(default=3600)
    email_report_enabled: bool = Field(default=False)
    email_report_attach_json: bool = Field(default=True)

    @property
    def root_data_path(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir).expanduser()
        base_dir = Path(os.environ.get("DATA_DIR", "~/quantpilot_data")).expanduser()
        return base_dir / "polymarket"

    @property
    def catalog_path(self) -> Path:
        return self.root_data_path / "catalog"

    @property
    def books_path(self) -> Path:
        return self.root_data_path / "books"

    @property
    def paper_path(self) -> Path:
        return self.root_data_path / "paper"

    @property
    def reports_path(self) -> Path:
        return self.root_data_path / "reports"

    @property
    def duckdb_path(self) -> Path:
        return self.root_data_path / "polymarket.duckdb"

    @property
    def mirror_root_path(self) -> Path:
        return self.root_data_path / "top_trader_mirror"

    @property
    def mirror_reports_path(self) -> Path:
        return self.mirror_root_path / "reports"

    @property
    def mirror_duckdb_path(self) -> Path:
        return self.mirror_root_path / "mirror.duckdb"


settings = PolySettings()
