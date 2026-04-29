"""Asynchronous storage writer for Polymarket scan hot paths."""
from __future__ import annotations

from collections import Counter
from threading import Event, Lock, Thread
from time import monotonic

from loguru import logger

from polymarket.config import PolySettings, settings
from polymarket.paper.simulator import PaperSimulator
from polymarket.storage import PolyStorage


class AsyncScanStorageWriter:
    """Batch book_top and heartbeat writes outside the scan hot path."""

    def __init__(
        self,
        storage: PolyStorage,
        simulator: PaperSimulator,
        storage_write_lock: Lock,
        cfg: PolySettings | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.storage = storage
        self.simulator = simulator
        self.storage_write_lock = storage_write_lock
        self._pending_lock = Lock()
        self._pending_book_tops: list[dict[str, object]] = []
        self._pending_rejections: Counter[str] = Counter()
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_flush_monotonic = 0.0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="polymarket-scan-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.flush_once()

    def enqueue_book_tops(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with self._pending_lock:
            self._pending_book_tops.extend(rows)

    def enqueue_heartbeat(self, rejection_counts: dict[str, int]) -> None:
        with self._pending_lock:
            for key, value in rejection_counts.items():
                self._pending_rejections[key] += int(value)

    def stats(self) -> dict[str, object]:
        with self._pending_lock:
            return {
                "pending_book_top_rows": len(self._pending_book_tops),
                "pending_rejection_keys": len(self._pending_rejections),
                "last_flush_age_seconds": (
                    monotonic() - self._last_flush_monotonic if self._last_flush_monotonic else None
                ),
                "last_error": self._last_error,
            }

    def flush_once(self) -> tuple[int, dict[str, int]]:
        with self._pending_lock:
            rows = self._pending_book_tops
            rejections = dict(self._pending_rejections)
            self._pending_book_tops = []
            self._pending_rejections = Counter()
        if not rows and not rejections:
            return 0, {}
        try:
            with self.storage_write_lock:
                if rows:
                    self.storage.save_book_tops(rows)
                if rejections:
                    self.simulator.record_scan_heartbeat(rejection_counts=rejections)
            self._last_error = None
            self._last_flush_monotonic = monotonic()
            return len(rows), rejections
        except Exception as exc:  # pragma: no cover
            with self._pending_lock:
                self._pending_book_tops = rows + self._pending_book_tops
                self._pending_rejections.update(rejections)
            self._last_error = str(exc)
            logger.warning(f"polymarket async storage flush failed: {exc}")
            return 0, {}

    def _run(self) -> None:  # pragma: no cover - tested via flush_once
        while not self._stop.wait(self.cfg.storage_async_flush_seconds):
            rows, rejections = self.flush_once()
            if rows or rejections:
                logger.debug(
                    f"polymarket async storage flush complete: book_top_rows={rows} "
                    f"rejection_keys={len(rejections)}"
                )
