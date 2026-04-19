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
    scan_interval_seconds: int = Field(default=5)
    catalog_refresh_seconds: int = Field(default=900)
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


settings = PolySettings()
