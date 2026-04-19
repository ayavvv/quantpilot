"""Storage helpers for isolated Polymarket paper-trading."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

import duckdb
import pandas as pd

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook, PaperFill


class PolyStorage:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.root = self.cfg.root_data_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg.catalog_path.mkdir(parents=True, exist_ok=True)
        self.cfg.books_path.mkdir(parents=True, exist_ok=True)
        self.cfg.paper_path.mkdir(parents=True, exist_ok=True)
        self.cfg.reports_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cfg.duckdb_path
        self._init_db()

    def _connect(self):
        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markets (
                    market_id TEXT PRIMARY KEY,
                    condition_id TEXT,
                    question TEXT,
                    slug TEXT,
                    end_date_iso TEXT,
                    min_order_size DOUBLE,
                    tick_size DOUBLE,
                    neg_risk BOOLEAN,
                    enable_order_book BOOLEAN,
                    taker_base_fee_bps DOUBLE,
                    yes_token_id TEXT,
                    no_token_id TEXT,
                    collateral_symbol TEXT,
                    fee_source TEXT,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS book_top (
                    ts TIMESTAMP,
                    market_id TEXT,
                    token_id TEXT,
                    best_bid DOUBLE,
                    best_bid_size DOUBLE,
                    best_ask DOUBLE,
                    best_ask_size DOUBLE,
                    last_trade DOUBLE,
                    book_timestamp_ms BIGINT,
                    PRIMARY KEY (market_id, token_id, ts)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opportunities (
                    ts TIMESTAMP,
                    market_id TEXT,
                    question TEXT,
                    direction TEXT,
                    gross_cost DOUBLE,
                    fee_cost DOUBLE,
                    yes_fee_cost DOUBLE,
                    no_fee_cost DOUBLE,
                    gas_cost DOUBLE,
                    slippage_buffer DOUBLE,
                    net_cost DOUBLE,
                    net_edge DOUBLE,
                    capacity DOUBLE,
                    mergeable_qty DOUBLE,
                    yes_qty DOUBLE,
                    no_qty DOUBLE,
                    yes_price DOUBLE,
                    no_price DOUBLE,
                    yes_book_timestamp_ms BIGINT,
                    no_book_timestamp_ms BIGINT,
                    rejection_reason TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_fills (
                    opportunity_id TEXT,
                    market_id TEXT,
                    token_id TEXT,
                    side TEXT,
                    qty DOUBLE,
                    price DOUBLE,
                    fee DOUBLE,
                    filled_at TIMESTAMP,
                    PRIMARY KEY (opportunity_id, token_id, side)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_positions (
                    token_id TEXT PRIMARY KEY,
                    qty DOUBLE,
                    avg_price DOUBLE,
                    realized_pnl DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_opportunities (
                    opportunity_id TEXT PRIMARY KEY,
                    claimed_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_state (
                    state_key TEXT PRIMARY KEY,
                    cash DOUBLE,
                    realized_pnl DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_daily_summary (
                    date TEXT PRIMARY KEY,
                    signals INTEGER,
                    accepted_signals INTEGER,
                    simulated_trades INTEGER,
                    gross_edge_sum DOUBLE,
                    net_edge_sum DOUBLE,
                    realized_pnl DOUBLE,
                    max_inventory_used DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )

    def save_catalog_snapshot(self, markets: Iterable[MarketInfo]) -> Path:
        now = datetime.now(timezone.utc)
        rows = []
        for market in markets:
            payload = market.as_dict()
            payload["updated_at"] = now
            rows.append(payload)
        frame = pd.DataFrame(rows)
        path = self.cfg.catalog_path / f"markets_{now.strftime('%Y%m%dT%H%M%SZ')}.parquet"
        if not frame.empty:
            frame.to_parquet(path, index=False)
            with self._connect() as conn:
                conn.execute("DELETE FROM markets")
                conn.register("markets_frame", frame)
                conn.execute("INSERT INTO markets SELECT * FROM markets_frame")
                conn.unregister("markets_frame")
        return path

    def save_book_snapshot(self, market: MarketInfo, yes_book: OrderBook, no_book: OrderBook) -> Path:
        now = datetime.now(timezone.utc)
        rows = []
        for book in (yes_book, no_book):
            rows.append(
                {
                    "market_id": market.market_id,
                    "token_id": book.token_id,
                    "timestamp_ms": book.timestamp_ms,
                    "last_trade_price": book.last_trade_price,
                    "tick_size": book.tick_size,
                    "min_order_size": book.min_order_size,
                    "neg_risk": book.neg_risk,
                    "bids": json.dumps(book.as_dict()["bids"]),
                    "asks": json.dumps(book.as_dict()["asks"]),
                }
            )
        frame = pd.DataFrame(rows)
        day_dir = self.cfg.books_path / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{market.market_id}_{now.strftime('%H%M%S%f')}.parquet"
        frame.to_parquet(path, index=False)

        top_rows = pd.DataFrame(
            [
                {
                    "ts": now,
                    "market_id": market.market_id,
                    "token_id": book.token_id,
                    "best_bid": book.best_bid.price if book.best_bid else None,
                    "best_bid_size": book.best_bid.size if book.best_bid else None,
                    "best_ask": book.best_ask.price if book.best_ask else None,
                    "best_ask_size": book.best_ask.size if book.best_ask else None,
                    "last_trade": book.last_trade_price,
                    "book_timestamp_ms": book.timestamp_ms,
                }
                for book in (yes_book, no_book)
            ]
        )
        with self._connect() as conn:
            conn.register("top_rows", top_rows)
            conn.execute("INSERT INTO book_top SELECT * FROM top_rows")
            conn.unregister("top_rows")
        return path

    def save_opportunities(self, opportunities: Iterable[Opportunity]) -> int:
        rows = [opportunity.as_dict() for opportunity in opportunities]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["ts"] = pd.to_datetime(frame["ts"])
        with self._connect() as conn:
            conn.register("opp_rows", frame)
            conn.execute(
                """
                INSERT INTO opportunities (
                    ts, market_id, question, direction, gross_cost, fee_cost,
                    yes_fee_cost, no_fee_cost, gas_cost, slippage_buffer,
                    net_cost, net_edge, capacity, mergeable_qty, yes_qty, no_qty,
                    yes_price, no_price, yes_book_timestamp_ms, no_book_timestamp_ms,
                    rejection_reason
                )
                SELECT
                    ts, market_id, question, direction, gross_cost, fee_cost,
                    yes_fee_cost, no_fee_cost, gas_cost, slippage_buffer,
                    net_cost, net_edge, capacity, mergeable_qty, yes_qty, no_qty,
                    yes_price, no_price, yes_book_timestamp_ms, no_book_timestamp_ms,
                    rejection_reason
                FROM opp_rows
                """
            )
            conn.unregister("opp_rows")
        return len(rows)

    def claim_opportunity(self, opportunity_id: str) -> bool:
        frame = pd.DataFrame(
            [{"opportunity_id": opportunity_id, "claimed_at": datetime.now(timezone.utc)}]
        )
        frame["claimed_at"] = pd.to_datetime(frame["claimed_at"])
        with self._connect() as conn:
            conn.register("claim_row", frame)
            row = conn.execute(
                """
                INSERT OR IGNORE INTO processed_opportunities
                SELECT opportunity_id, claimed_at FROM claim_row
                RETURNING opportunity_id
                """
            ).fetchone()
            conn.unregister("claim_row")
        return row is not None

    def save_fills(self, fills: Iterable[PaperFill]) -> int:
        rows = [fill.as_dict() for fill in fills]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["filled_at"] = pd.to_datetime(frame["filled_at"])
        with self._connect() as conn:
            conn.register("fill_rows", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_fills
                SELECT opportunity_id, market_id, token_id, side, qty, price, fee, filled_at
                FROM fill_rows
                """
            )
            conn.unregister("fill_rows")
        return len(rows)

    def upsert_positions(self, positions: pd.DataFrame) -> None:
        if positions.empty:
            return
        positions = positions.copy()
        positions["updated_at"] = pd.to_datetime(positions["updated_at"])
        with self._connect() as conn:
            conn.register("position_rows", positions)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_positions
                SELECT token_id, qty, avg_price, realized_pnl, updated_at
                FROM position_rows
                """
            )
            conn.unregister("position_rows")

    def load_state(self) -> dict[str, float] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cash, realized_pnl FROM paper_state WHERE state_key = ?",
                ["global"],
            ).fetchone()
        if row is None:
            return None
        return {"cash": float(row[0]), "realized_pnl": float(row[1])}

    def save_state(self, cash: float, realized_pnl: float) -> None:
        frame = pd.DataFrame(
            [{
                "state_key": "global",
                "cash": cash,
                "realized_pnl": realized_pnl,
                "updated_at": datetime.now(timezone.utc),
            }]
        )
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("state_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_state
                SELECT state_key, cash, realized_pnl, updated_at
                FROM state_row
                """
            )
            conn.unregister("state_row")

    def upsert_daily_summary(self, summary: dict[str, object]) -> None:
        date_value = str(summary["date"])
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl, max_inventory_used
                FROM paper_daily_summary
                WHERE date = ?
                """,
                [date_value],
            ).fetchone()
        merged = dict(summary)
        if existing is not None:
            merged["signals"] = int(existing[0]) + int(summary["signals"])
            merged["accepted_signals"] = int(existing[1]) + int(summary["accepted_signals"])
            merged["simulated_trades"] = int(existing[2]) + int(summary["simulated_trades"])
            merged["gross_edge_sum"] = float(existing[3]) + float(summary["gross_edge_sum"])
            merged["net_edge_sum"] = float(existing[4]) + float(summary["net_edge_sum"])
            merged["realized_pnl"] = float(summary["realized_pnl"])
            merged["max_inventory_used"] = max(float(existing[6]), float(summary["max_inventory_used"]))
        frame = pd.DataFrame([merged])
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("summary_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_daily_summary
                SELECT date, signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl,
                       max_inventory_used, updated_at
                FROM summary_row
                """
            )
            conn.unregister("summary_row")

    @staticmethod
    def _read_only_connect(db_path: Path):
        return duckdb.connect(str(db_path), read_only=True)

    @classmethod
    def load_daily_summary(cls, db_path: Path, date_value: str) -> dict[str, Any] | None:
        if not Path(db_path).exists():
            return None
        with cls._read_only_connect(Path(db_path)) as conn:
            row = conn.execute(
                """
                SELECT date, signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl,
                       max_inventory_used, updated_at
                FROM paper_daily_summary
                WHERE date = ?
                LIMIT 1
                """,
                [date_value],
            ).fetchone()
        if row is None:
            return None
        return {
            "date": row[0],
            "signals": int(row[1]),
            "accepted_signals": int(row[2]),
            "simulated_trades": int(row[3]),
            "gross_edge_sum": float(row[4]),
            "net_edge_sum": float(row[5]),
            "realized_pnl": float(row[6]),
            "max_inventory_used": float(row[7]),
            "updated_at": str(row[8]),
        }

    @classmethod
    def load_latest_daily_summary(cls, db_path: Path) -> dict[str, Any] | None:
        if not Path(db_path).exists():
            return None
        with cls._read_only_connect(Path(db_path)) as conn:
            row = conn.execute(
                """
                SELECT date
                FROM paper_daily_summary
                ORDER BY date DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return cls.load_daily_summary(db_path, str(row[0]))

    @classmethod
    def load_report_snapshot(cls, db_path: Path, date_value: str | None = None) -> dict[str, Any] | None:
        summary = cls.load_daily_summary(db_path, date_value) if date_value else cls.load_latest_daily_summary(db_path)
        if summary is None:
            return None
        target_date = summary["date"]
        with cls._read_only_connect(Path(db_path)) as conn:
            fill_count = conn.execute(
                """
                SELECT count(*)
                FROM paper_fills
                WHERE CAST(filled_at AS DATE) = CAST(? AS DATE)
                """,
                [target_date],
            ).fetchone()[0]
            opportunity_count = conn.execute(
                """
                SELECT count(*)
                FROM opportunities
                WHERE CAST(ts AS DATE) = CAST(? AS DATE)
                """,
                [target_date],
            ).fetchone()[0]
            market_count = conn.execute(
                """
                SELECT count(DISTINCT market_id)
                FROM opportunities
                WHERE CAST(ts AS DATE) = CAST(? AS DATE)
                """,
                [target_date],
            ).fetchone()[0]
        return {
            **summary,
            "fill_count": int(fill_count),
            "opportunity_count": int(opportunity_count),
            "market_count": int(market_count),
        }
