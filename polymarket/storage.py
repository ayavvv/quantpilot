"""Storage helpers for isolated Polymarket paper-trading."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Any

import duckdb
import pandas as pd

from polymarket.config import PolySettings, settings
from polymarket.models import MarketInfo, Opportunity, OrderBook, PaperFill, TraderProfile, TraderScore, TraderEvent, MirrorSignal


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
                CREATE TABLE IF NOT EXISTS market_snapshot_meta (
                    snapshot_key TEXT PRIMARY KEY,
                    updated_at TIMESTAMP,
                    market_count INTEGER,
                    source TEXT
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
                    strategy_type TEXT,
                    source_trader_wallet TEXT,
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
                    date TEXT,
                    strategy_type TEXT,
                    signals INTEGER,
                    accepted_signals INTEGER,
                    simulated_trades INTEGER,
                    gross_edge_sum DOUBLE,
                    net_edge_sum DOUBLE,
                    realized_pnl DOUBLE,
                    max_inventory_used DOUBLE,
                    rejection_counts_json TEXT,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (date, strategy_type)
                )
                """
            )

    def save_catalog_snapshot(self, markets: Iterable[MarketInfo], source: str = "live_refresh") -> Path:
        now = datetime.now(timezone.utc)
        rows = []
        for market in markets:
            payload = market.as_dict()
            payload["updated_at"] = now
            rows.append(payload)
        frame = pd.DataFrame(rows)
        path = self.cfg.catalog_path / f"markets_{now.strftime('%Y%m%dT%H%M%SZ')}.parquet"
        with self._connect() as conn:
            conn.execute("DELETE FROM markets")
            if not frame.empty:
                frame.to_parquet(path, index=False)
                conn.register("markets_frame", frame)
                conn.execute("INSERT INTO markets SELECT * FROM markets_frame")
                conn.unregister("markets_frame")
            meta_frame = pd.DataFrame([
                {
                    "snapshot_key": "active",
                    "updated_at": now,
                    "market_count": len(rows),
                    "source": source,
                }
            ])
            meta_frame["updated_at"] = pd.to_datetime(meta_frame["updated_at"])
            conn.register("meta_row", meta_frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO market_snapshot_meta
                SELECT snapshot_key, updated_at, market_count, source
                FROM meta_row
                """
            )
            conn.unregister("meta_row")
        return path

    def load_markets(self) -> tuple[list[MarketInfo], datetime | None]:
        if not self.db_path.exists():
            return [], None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, condition_id, question, slug, end_date_iso, min_order_size,
                       tick_size, neg_risk, enable_order_book, taker_base_fee_bps,
                       yes_token_id, no_token_id, collateral_symbol, fee_source, updated_at
                FROM markets
                """
            ).fetchall()
        markets = [
            MarketInfo(
                market_id=str(row[0]),
                condition_id=str(row[1]),
                question=str(row[2]),
                slug=row[3],
                end_date_iso=row[4],
                min_order_size=float(row[5] or 0),
                tick_size=float(row[6] or 0.01),
                neg_risk=bool(row[7]),
                enable_order_book=bool(row[8]),
                taker_base_fee_bps=float(row[9] or 0),
                yes_token_id=str(row[10]),
                no_token_id=str(row[11]),
                collateral_symbol=str(row[12] or 'USDC.e'),
                fee_source=str(row[13] or 'taker_base_fee'),
            )
            for row in rows
        ]
        updated_at_values = [row[14] for row in rows if row[14] is not None]
        updated_at = max(updated_at_values) if updated_at_values else None
        if updated_at is not None and updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return markets, updated_at

    def load_catalog_meta(self) -> dict[str, object] | None:
        if not self.db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_key, updated_at, market_count, source FROM market_snapshot_meta WHERE snapshot_key = 'active'"
            ).fetchone()
        if row is None:
            return None
        updated_at = row[1]
        if updated_at is not None and updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return {
            "snapshot_key": row[0],
            "updated_at": updated_at,
            "market_count": int(row[2] or 0),
            "source": str(row[3] or "unknown"),
        }

    def save_book_snapshot(self, market: MarketInfo, yes_book: OrderBook, no_book: OrderBook, persist_depth: bool = True, persist_top: bool = True) -> Path | None:
        now = datetime.now(timezone.utc)
        if persist_depth:
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
        else:
            path = None

        if persist_top:
            self.save_book_tops(
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
        return path

    def save_book_tops(self, rows: list[dict[str, object]]) -> int:
        if not rows:
            return 0
        top_rows = pd.DataFrame(rows)
        with self._connect() as conn:
            conn.register("top_rows", top_rows)
            conn.execute("INSERT INTO book_top SELECT * FROM top_rows")
            conn.unregister("top_rows")
        return len(rows)

    def prune_book_tops(self, retention_hours: int) -> int:
        if retention_hours <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        with self._connect() as conn:
            delete_count = conn.execute(
                "SELECT count(*) FROM book_top WHERE ts < ?",
                [cutoff],
            ).fetchone()[0]
            if delete_count:
                conn.execute("DELETE FROM book_top WHERE ts < ?", [cutoff])
                conn.execute("CHECKPOINT")
        return int(delete_count)

    def prune_book_snapshots(self, retention_hours: int) -> int:
        if retention_hours <= 0:
            return 0
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).timestamp()
        deleted = 0
        for path in self.cfg.books_path.glob("*/*.parquet"):
            try:
                if path.stat().st_mtime >= cutoff_ts:
                    continue
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        for day_dir in self.cfg.books_path.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                day_dir.rmdir()
            except OSError:
                continue
        return deleted

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
                SELECT opportunity_id, market_id, token_id, side, qty, price, fee, filled_at, strategy_type, source_trader_wallet
                FROM fill_rows
                """
            )
            conn.unregister("fill_rows")
        return len(rows)

    def load_latest_fill_times_by_market(self) -> dict[str, datetime]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, max(filled_at)
                FROM paper_fills
                GROUP BY market_id
                """
            ).fetchall()
        latest: dict[str, datetime] = {}
        for market_id, filled_at in rows:
            if filled_at is None:
                continue
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
            latest[str(market_id)] = filled_at
        return latest

    def load_fill_notional_by_market(self, target_date: str, strategy_type: str = "full_set_arb") -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_id, sum(abs(qty * price) + abs(fee))
                FROM paper_fills
                WHERE CAST(filled_at AS DATE) = CAST(? AS DATE)
                  AND strategy_type = ?
                GROUP BY market_id
                """,
                [target_date, strategy_type],
            ).fetchall()
        return {str(market_id): float(notional or 0.0) for market_id, notional in rows}

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
        strategy_type = str(summary.get("strategy_type", "full_set_arb"))
        merged = dict(summary)
        merged["strategy_type"] = strategy_type
        merged.setdefault("rejection_counts_json", json.dumps({}, ensure_ascii=False, sort_keys=True))
        with self._connect() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info('paper_daily_summary')").fetchall()}
            if "rejection_counts_json" not in columns:
                conn.execute("ALTER TABLE paper_daily_summary ADD COLUMN rejection_counts_json TEXT")
            existing = conn.execute(
                """
                SELECT signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl, max_inventory_used, rejection_counts_json
                FROM paper_daily_summary
                WHERE date = ? AND strategy_type = ?
                """,
                [date_value, strategy_type],
            ).fetchone()
            if existing is not None:
                merged["signals"] = int(existing[0]) + int(summary["signals"])
                merged["accepted_signals"] = int(existing[1]) + int(summary["accepted_signals"])
                merged["simulated_trades"] = int(existing[2]) + int(summary["simulated_trades"])
                merged["gross_edge_sum"] = float(existing[3]) + float(summary["gross_edge_sum"])
                merged["net_edge_sum"] = float(existing[4]) + float(summary["net_edge_sum"])
                merged["realized_pnl"] = float(summary["realized_pnl"])
                merged["max_inventory_used"] = max(float(existing[6]), float(summary["max_inventory_used"]))
                existing_rejections = json.loads(existing[7]) if existing[7] else {}
                new_rejections = summary.get("rejection_counts_json")
                new_rejections_dict = json.loads(new_rejections) if isinstance(new_rejections, str) and new_rejections else {}
                merged_rejections = dict(existing_rejections)
                for key, value in new_rejections_dict.items():
                    merged_rejections[key] = int(merged_rejections.get(key, 0)) + int(value)
                merged["rejection_counts_json"] = json.dumps(merged_rejections, ensure_ascii=False, sort_keys=True)
            frame = pd.DataFrame([merged])
            frame["updated_at"] = pd.to_datetime(frame["updated_at"])
            conn.register("summary_row", frame)
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_daily_summary (
                    date, strategy_type, signals, accepted_signals, simulated_trades,
                    gross_edge_sum, net_edge_sum, realized_pnl,
                    max_inventory_used, updated_at, rejection_counts_json
                )
                SELECT date, strategy_type, signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl,
                       max_inventory_used, updated_at, rejection_counts_json
                FROM summary_row
                """
            )
            conn.unregister("summary_row")

    @staticmethod
    def _read_only_connect(db_path: Path):
        return duckdb.connect(str(db_path), read_only=True)

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
            conn.execute(
                """
                INSERT OR IGNORE INTO mirror_signals
                SELECT fingerprint, wallet, market_id, asset, title, outcome, side,
                       source_size, source_price, signal_size, lag_seconds, timestamp, created_at
                FROM signal_rows
                """
            )
            conn.unregister("signal_rows")
        return len(rows)

    @classmethod
    def load_daily_summary(cls, db_path: Path, date_value: str) -> dict[str, Any] | None:
        if not Path(db_path).exists():
            return None
        with cls._read_only_connect(Path(db_path)) as conn:
            row = conn.execute(
                """
                SELECT date, signals, accepted_signals, simulated_trades,
                       gross_edge_sum, net_edge_sum, realized_pnl,
                       max_inventory_used, rejection_counts_json, updated_at
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
            "rejection_counts": json.loads(row[8]) if row[8] else {},
            "updated_at": str(row[9]),
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
                FROM book_top
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
