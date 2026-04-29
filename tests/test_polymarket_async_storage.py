from threading import Lock

from polymarket.async_storage import AsyncScanStorageWriter
from polymarket.config import PolySettings


class _FakeStorage:
    def __init__(self):
        self.rows = []

    def save_book_tops(self, rows):
        self.rows.extend(rows)
        return len(rows)


class _FakeSimulator:
    def __init__(self):
        self.rejections = []

    def record_scan_heartbeat(self, rejection_counts=None, strategy_type="full_set_arb"):
        self.rejections.append(dict(rejection_counts or {}))


def test_async_scan_storage_writer_batches_rows_and_rejections(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), storage_async_flush_seconds=60)
    storage = _FakeStorage()
    simulator = _FakeSimulator()
    writer = AsyncScanStorageWriter(storage=storage, simulator=simulator, storage_write_lock=Lock(), cfg=cfg)

    writer.enqueue_book_tops([{"market_id": "m1"}, {"market_id": "m2"}])
    writer.enqueue_heartbeat({"stale_books": 2})
    writer.enqueue_heartbeat({"stale_books": 3, "edge_below_threshold": 1})

    rows, rejections = writer.flush_once()

    assert rows == 2
    assert rejections == {"stale_books": 5, "edge_below_threshold": 1}
    assert storage.rows == [{"market_id": "m1"}, {"market_id": "m2"}]
    assert simulator.rejections == [{"stale_books": 5, "edge_below_threshold": 1}]
