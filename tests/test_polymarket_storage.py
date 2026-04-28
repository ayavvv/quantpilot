import os
from datetime import datetime, timedelta, timezone

from polymarket.config import PolySettings
from polymarket.storage import PolyStorage


def test_prune_book_tops_keeps_only_retention_window(tmp_path):
    storage = PolyStorage(PolySettings(data_dir=str(tmp_path)))
    now = datetime.now(timezone.utc)
    rows = [
        {
            "ts": now - timedelta(hours=73),
            "market_id": "old-market",
            "token_id": "old-yes",
            "best_bid": 0.4,
            "best_bid_size": 10,
            "best_ask": 0.5,
            "best_ask_size": 12,
            "last_trade": 0.45,
            "book_timestamp_ms": 1000,
        },
        {
            "ts": now - timedelta(hours=1),
            "market_id": "fresh-market",
            "token_id": "fresh-yes",
            "best_bid": 0.4,
            "best_bid_size": 10,
            "best_ask": 0.5,
            "best_ask_size": 12,
            "last_trade": 0.45,
            "book_timestamp_ms": 1000,
        },
    ]
    storage.save_book_tops(rows)

    deleted = storage.prune_book_tops(72)

    assert deleted == 1
    with storage._connect() as conn:
        market_ids = [
            row[0]
            for row in conn.execute("SELECT market_id FROM book_top ORDER BY market_id").fetchall()
        ]
    assert market_ids == ["fresh-market"]


def test_prune_book_tops_disabled_for_nonpositive_retention(tmp_path):
    storage = PolyStorage(PolySettings(data_dir=str(tmp_path)))
    storage.save_book_tops(
        [
            {
                "ts": datetime.now(timezone.utc) - timedelta(days=10),
                "market_id": "old-market",
                "token_id": "old-yes",
                "best_bid": None,
                "best_bid_size": None,
                "best_ask": None,
                "best_ask_size": None,
                "last_trade": None,
                "book_timestamp_ms": None,
            }
        ]
    )

    assert storage.prune_book_tops(0) == 0
    with storage._connect() as conn:
        count = conn.execute("SELECT count(*) FROM book_top").fetchone()[0]
    assert count == 1


def test_prune_book_snapshots_removes_old_files_and_empty_dirs(tmp_path):
    storage = PolyStorage(PolySettings(data_dir=str(tmp_path)))
    old_dir = storage.cfg.books_path / "2026-04-20"
    fresh_dir = storage.cfg.books_path / "2026-04-28"
    old_dir.mkdir(parents=True)
    fresh_dir.mkdir(parents=True)
    old_file = old_dir / "old.parquet"
    fresh_file = fresh_dir / "fresh.parquet"
    old_file.touch()
    fresh_file.touch()
    old_time = (datetime.now(timezone.utc) - timedelta(hours=73)).timestamp()
    fresh_time = (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
    os.utime(old_file, (old_time, old_time))
    os.utime(fresh_file, (fresh_time, fresh_time))

    deleted = storage.prune_book_snapshots(72)

    assert deleted == 1
    assert not old_file.exists()
    assert not old_dir.exists()
    assert fresh_file.exists()
