from pathlib import Path

import pandas as pd

from scripts.collect_us_microstructure import (
    _flatten_order_book,
    _normalise_symbols,
    _partition_path,
    _prepare_trade_rows,
    _sha256_file,
    _write_partition,
)


def test_normalise_symbols_adds_us_prefix_and_dedupes():
    assert _normalise_symbols(["aapl", "US.AAPL", " nvda ", "", "spy"]) == [
        "US.AAPL",
        "US.NVDA",
        "US.SPY",
    ]


def test_flatten_order_book_outputs_levels_and_spread():
    row = _flatten_order_book(
        {
            "code": "US.AAPL",
            "name": "Apple",
            "svr_recv_time_bid": "2026-06-01 08:16:54.756",
            "svr_recv_time_ask": "2026-06-01 08:16:54.756",
            "Bid": [(100.0, 10, 2, {}), (99.9, 20, 1, {})],
            "Ask": [(100.1, 30, 3, {}), (100.2, 40, 1, {})],
        },
        symbol="US.AAPL",
        recv_time="2026-06-01T00:16:55.000+00:00",
        levels=2,
    )

    assert row["symbol"] == "US.AAPL"
    assert row["bid_px_1"] == 100.0
    assert row["bid_sz_2"] == 20
    assert row["ask_px_1"] == 100.1
    assert row["ask_order_count_2"] == 1
    assert row["mid"] == 100.05
    assert 9.9 < row["spread_bps"] < 10.1


def test_write_partition_groups_by_symbol_and_writes_parquet(tmp_path):
    rows = [
        {"symbol": "US.AAPL", "event_time": "2026-06-01 09:30:00", "price": 100.0},
        {"symbol": "US.AAPL", "event_time": "2026-06-01 09:30:01", "price": 100.1},
        {"symbol": "US.NVDA", "event_time": "2026-06-01 09:30:00", "price": 200.0},
    ]

    manifest = _write_partition(
        rows,
        kind="trades",
        base_dir=tmp_path,
        date="2026-06-01",
        run_id="run",
        batch_index=1,
    )

    assert len(manifest) == 2
    paths = [Path(item["local_path"]) for item in manifest]
    assert all(path.exists() for path in paths)
    assert sorted(item["row_count"] for item in manifest) == [1, 2]
    loaded = pd.concat(pd.read_parquet(path) for path in paths)
    assert sorted(loaded["symbol"].unique()) == ["US.AAPL", "US.NVDA"]
    assert all(item["sha256"] == _sha256_file(Path(item["local_path"])) for item in manifest)


def test_prepare_trade_rows_filters_stale_tickers_before_deduping():
    seen = set()
    rows = _prepare_trade_rows(
        pd.DataFrame(
            [
                {"time": "2026-05-29 15:59:59.000", "sequence": 1, "price": 99.0},
                {"time": "2026-06-01 09:30:00.000", "sequence": 1, "price": 100.0},
                {"time": "2026-06-01 09:30:01.000", "sequence": 1, "price": 100.1},
            ]
        ),
        symbol="US.AAPL",
        recv_time="2026-06-01T13:30:01.000+00:00",
        seen_sequences=seen,
        collection_date="2026-06-01",
    )

    assert len(rows) == 1
    assert rows[0]["event_time"] == "2026-06-01 09:30:00.000"
    assert rows[0]["price"] == 100.0
    assert seen == {"1"}


def test_partition_path_is_hive_style(tmp_path):
    path = _partition_path(tmp_path, "quotes", "2026-06-01", "US.AAPL", "run", 3)
    assert path == tmp_path / "quotes" / "date=2026-06-01" / "symbol=US.AAPL" / "part-run-00003.parquet"
