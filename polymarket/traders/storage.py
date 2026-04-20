"""Storage for the isolated top-trader mirror strategy."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from polymarket.config import PolySettings, settings
from polymarket.models import MirrorSignal, TraderEvent, TraderProfile, TraderScore


class MirrorStorage:
    def __init__(self, cfg: PolySettings | None = None):
        self.cfg = cfg or settings
        self.root = self.cfg.mirror_root_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg.mirror_reports_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cfg.mirror_duckdb_path
        self._init_db()

    def _connect(self):
        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracked_traders (
                    wallet TEXT PRIMARY KEY,
                    user_name TEXT,
                    pseudonym TEXT,
                    verified_badge BOOLEAN,
                    profile_image TEXT,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_scores (
                    wallet TEXT PRIMARY KEY,
                    score DOUBLE,
                    rank INTEGER,
                    pnl DOUBLE,
                    volume DOUBLE,
                    trade_count INTEGER,
                    diversity_count INTEGER,
                    realized_pnl DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_events (
                    fingerprint TEXT PRIMARY KEY,
                    wallet TEXT,
                    event_type TEXT,
                    market_id TEXT,
                    asset TEXT,
                    side TEXT,
                    size DOUBLE,
                    price DOUBLE,
                    timestamp BIGINT,
                    transaction_hash TEXT,
                    title TEXT,
                    outcome TEXT,
                    user_name TEXT,
                    observed_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mirror_signals (
                    fingerprint TEXT PRIMARY KEY,
                    wallet TEXT,
                    market_id TEXT,
                    asset TEXT,
                    title TEXT,
                    outcome TEXT,
                    side TEXT,
                    source_size DOUBLE,
                    source_price DOUBLE,
                    signal_size DOUBLE,
                    lag_seconds INTEGER,
                    timestamp BIGINT,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_signals (
                    fingerprint TEXT PRIMARY KEY,
                    claimed_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mirror_fills (
                    signal_fingerprint TEXT PRIMARY KEY,
                    wallet TEXT,
                    market_id TEXT,
                    asset TEXT,
                    side TEXT,
                    qty DOUBLE,
                    gross_price DOUBLE,
                    fee_cash DOUBLE,
                    net_qty DOUBLE,
                    proceeds DOUBLE,
                    filled_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mirror_positions (
                    asset TEXT PRIMARY KEY,
                    market_id TEXT,
                    title TEXT,
                    outcome TEXT,
                    qty DOUBLE,
                    avg_price DOUBLE,
                    realized_pnl DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mirror_state (
                    state_key TEXT PRIMARY KEY,
                    cash DOUBLE,
                    realized_pnl DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mirror_daily_summary (
                    date TEXT PRIMARY KEY,
                    tracked_traders INTEGER,
                    signals INTEGER,
                    accepted_signals INTEGER,
                    simulated_trades INTEGER,
                    realized_pnl DOUBLE,
                    max_inventory_used DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )

    def upsert_trader_profiles(self, profiles: Iterable[TraderProfile]) -> int:
        rows = [profile.as_dict() | {"updated_at": datetime.now(timezone.utc)} for profile in profiles]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("profile_rows", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO tracked_traders
                SELECT wallet, user_name, pseudonym, verified_badge, profile_image, updated_at
                FROM profile_rows
                """
            )
            conn.unregister("profile_rows")
        return len(rows)

    def upsert_trader_scores(self, scores: Iterable[TraderScore]) -> int:
        rows = [score.as_dict() | {"updated_at": datetime.now(timezone.utc)} for score in scores]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("score_rows", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO trader_scores
                SELECT wallet, score, rank, pnl, volume, trade_count, diversity_count, realized_pnl, updated_at
                FROM score_rows
                """
            )
            conn.unregister("score_rows")
        return len(rows)

    def save_trader_events(self, events: Iterable[TraderEvent]) -> int:
        rows = [event.as_dict() | {"observed_at": datetime.now(timezone.utc)} for event in events]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["observed_at"] = pd.to_datetime(frame["observed_at"])
        with self._connect() as conn:
            conn.register("event_rows", frame)
            conn.execute(
                """
                INSERT OR IGNORE INTO trader_events
                SELECT fingerprint, wallet, event_type, market_id, asset, side, size, price,
                       timestamp, transaction_hash, title, outcome, user_name, observed_at
                FROM event_rows
                """
            )
            conn.unregister("event_rows")
        return len(rows)

    def save_mirror_signals(self, signals: Iterable[MirrorSignal]) -> int:
        rows = [signal.as_dict() | {"created_at": datetime.now(timezone.utc)} for signal in signals]
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        frame["created_at"] = pd.to_datetime(frame["created_at"])
        with self._connect() as conn:
            conn.register("signal_rows", frame)
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO mirror_signals
                SELECT fingerprint, wallet, market_id, asset, title, outcome, side,
                       source_size, source_price, signal_size, lag_seconds, timestamp, created_at
                FROM signal_rows
                RETURNING fingerprint
                """
            ).fetchall()
            conn.unregister("signal_rows")
        return len(inserted)

    def claim_signal(self, fingerprint: str) -> bool:
        frame = pd.DataFrame([{"fingerprint": fingerprint, "claimed_at": datetime.now(timezone.utc)}])
        frame["claimed_at"] = pd.to_datetime(frame["claimed_at"])
        with self._connect() as conn:
            conn.register("claim_row", frame)
            row = conn.execute(
                """
                INSERT OR IGNORE INTO processed_signals
                SELECT fingerprint, claimed_at FROM claim_row
                RETURNING fingerprint
                """
            ).fetchone()
            conn.unregister("claim_row")
        return row is not None

    def save_mirror_fill(self, fill: dict[str, Any]) -> None:
        frame = pd.DataFrame([fill])
        frame["filled_at"] = pd.to_datetime(frame["filled_at"])
        with self._connect() as conn:
            conn.register("fill_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO mirror_fills
                SELECT signal_fingerprint, wallet, market_id, asset, side, qty, gross_price,
                       fee_cash, net_qty, proceeds, filled_at
                FROM fill_row
                """
            )
            conn.unregister("fill_row")

    def load_state(self) -> dict[str, float] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cash, realized_pnl FROM mirror_state WHERE state_key = ?",
                ["global"],
            ).fetchone()
        if row is None:
            return None
        return {"cash": float(row[0]), "realized_pnl": float(row[1])}

    def save_state(self, cash: float, realized_pnl: float) -> None:
        frame = pd.DataFrame([
            {
                "state_key": "global",
                "cash": cash,
                "realized_pnl": realized_pnl,
                "updated_at": datetime.now(timezone.utc),
            }
        ])
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("state_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO mirror_state
                SELECT state_key, cash, realized_pnl, updated_at
                FROM state_row
                """
            )
            conn.unregister("state_row")

    def upsert_positions(self, positions: pd.DataFrame) -> None:
        if positions.empty:
            return
        frame = positions.copy()
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("position_rows", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO mirror_positions
                SELECT asset, market_id, title, outcome, qty, avg_price, realized_pnl, updated_at
                FROM position_rows
                """
            )
            conn.unregister("position_rows")

    def upsert_daily_summary(self, summary: dict[str, Any]) -> None:
        date_value = str(summary["date"])
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT tracked_traders, signals, accepted_signals, simulated_trades, realized_pnl, max_inventory_used FROM mirror_daily_summary WHERE date = ?",
                [date_value],
            ).fetchone()
        merged = dict(summary)
        if existing is not None:
            merged["tracked_traders"] = max(int(existing[0]), int(summary["tracked_traders"]))
            merged["signals"] = int(existing[1]) + int(summary["signals"])
            merged["accepted_signals"] = int(existing[2]) + int(summary["accepted_signals"])
            merged["simulated_trades"] = int(existing[3]) + int(summary["simulated_trades"])
            merged["realized_pnl"] = float(summary["realized_pnl"])
            merged["max_inventory_used"] = max(float(existing[5]), float(summary["max_inventory_used"]))
        frame = pd.DataFrame([merged])
        frame["updated_at"] = pd.to_datetime(frame["updated_at"])
        with self._connect() as conn:
            conn.register("summary_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO mirror_daily_summary
                SELECT date, tracked_traders, signals, accepted_signals, simulated_trades, realized_pnl, max_inventory_used, updated_at
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
                "SELECT date, tracked_traders, signals, accepted_signals, simulated_trades, realized_pnl, max_inventory_used, updated_at FROM mirror_daily_summary WHERE date = ? LIMIT 1",
                [date_value],
            ).fetchone()
        if row is None:
            return None
        return {
            "date": row[0],
            "tracked_traders": int(row[1]),
            "signals": int(row[2]),
            "accepted_signals": int(row[3]),
            "simulated_trades": int(row[4]),
            "realized_pnl": float(row[5]),
            "max_inventory_used": float(row[6]),
            "updated_at": str(row[7]),
        }
