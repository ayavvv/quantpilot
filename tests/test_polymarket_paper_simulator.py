from pathlib import Path
from dataclasses import replace

import duckdb

from polymarket.config import PolySettings
from polymarket.models import Opportunity, MarketInfo
from polymarket.paper.simulator import PaperSimulator
from polymarket.storage import PolyStorage
from datetime import datetime, timezone


def _market() -> MarketInfo:
    return MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Will X happen?",
        slug="will-x-happen",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )


def test_paper_simulator_persists_summary_and_fills(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)

    accepted = simulator.consume(
        _market(),
        [
            Opportunity(
                market_id="m1",
                question="Will X happen?",
                direction="buy_both_merge",
                gross_cost=0.98,
                fee_cost=0.0,
                yes_fee_cost=0.0,
                no_fee_cost=0.0,
                gas_cost=0.0,
                slippage_buffer=0.0,
                net_cost=0.98,
                net_edge=0.02,
                capacity=1.0,
                mergeable_qty=1.0,
                yes_qty=1.0,
                no_qty=1.0,
                yes_price=0.49,
                no_price=0.49,
                ts=datetime.now(timezone.utc),
            )
        ],
    )

    assert accepted == 1
    assert Path(cfg.duckdb_path).exists()

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        fills = conn.execute("SELECT count(*) FROM paper_fills").fetchone()[0]
        strategy_type = conn.execute("SELECT strategy_type FROM paper_fills LIMIT 1").fetchone()[0]
        summaries = conn.execute("SELECT count(*) FROM paper_daily_summary").fetchone()[0]
        processed = conn.execute("SELECT count(*) FROM processed_opportunities").fetchone()[0]
    finally:
        conn.close()

    assert fills == 2
    assert strategy_type == 'full_set_arb'
    assert summaries == 1
    assert processed == 1


def test_paper_simulator_applies_split_sell_opportunity(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)

    accepted = simulator.consume(
        _market(),
        [
            Opportunity(
                market_id="m1",
                question="Will X happen?",
                direction="split_sell_both",
                gross_cost=1.0,
                fee_cost=0.0,
                yes_fee_cost=0.0,
                no_fee_cost=0.0,
                gas_cost=0.0,
                slippage_buffer=0.0,
                net_cost=1.06,
                net_edge=0.06,
                capacity=1.0,
                mergeable_qty=1.0,
                yes_qty=1.0,
                no_qty=1.0,
                yes_price=0.53,
                no_price=0.53,
                yes_book_timestamp_ms=1000,
                no_book_timestamp_ms=1000,
                ts=datetime.now(timezone.utc),
            )
        ],
    )

    assert accepted == 1
    assert simulator.ledger.state.cash == 100.06
    assert simulator.ledger.state.realized_pnl == 0.06

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        sides = conn.execute("SELECT DISTINCT side FROM paper_fills").fetchall()
        fill_count = conn.execute("SELECT count(*) FROM paper_fills").fetchone()[0]
    finally:
        conn.close()

    assert sides == [("sell",)]
    assert fill_count == 2


def test_paper_simulator_restores_state_between_runs(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)

    first = PaperSimulator(storage=storage, cfg=cfg)
    first.consume(
        _market(),
        [
            Opportunity(
                market_id="m1",
                question="Will X happen?",
                direction="buy_both_merge",
                gross_cost=0.98,
                fee_cost=0.0,
                yes_fee_cost=0.0,
                no_fee_cost=0.0,
                gas_cost=0.0,
                slippage_buffer=0.0,
                net_cost=0.98,
                net_edge=0.02,
                capacity=1.0,
                mergeable_qty=1.0,
                yes_qty=1.0,
                no_qty=1.0,
                yes_price=0.49,
                no_price=0.49,
                strategy_type='full_set_arb',
                yes_book_timestamp_ms=1000,
                no_book_timestamp_ms=1000,
                ts=datetime.now(timezone.utc),
            )
        ],
    )

    second = PaperSimulator(storage=storage, cfg=cfg)
    assert second.ledger.state.cash == 100.02
    assert second.ledger.state.realized_pnl == 0.02


def test_paper_simulator_skips_duplicate_opportunity(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)
    opportunity = Opportunity(
        market_id="m1",
        question="Will X happen?",
        direction="buy_both_merge",
        gross_cost=0.98,
        fee_cost=0.0,
        yes_fee_cost=0.0,
        no_fee_cost=0.0,
        gas_cost=0.0,
        slippage_buffer=0.0,
        net_cost=0.98,
        net_edge=0.02,
        capacity=1.0,
        mergeable_qty=1.0,
        yes_qty=1.0,
        no_qty=1.0,
        yes_price=0.49,
        no_price=0.49,
        yes_book_timestamp_ms=1000,
        no_book_timestamp_ms=1000,
        ts=datetime.now(timezone.utc),
    )

    first = simulator.consume(_market(), [opportunity])
    second = simulator.consume(_market(), [opportunity])

    assert first == 1
    assert second == 0
    assert simulator.ledger.state.realized_pnl == 0.02

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        fills = conn.execute("SELECT count(*) FROM paper_fills").fetchone()[0]
        processed = conn.execute("SELECT count(*) FROM processed_opportunities").fetchone()[0]
    finally:
        conn.close()

    assert fills == 2
    assert processed == 1


def test_paper_simulator_applies_market_cooldown(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100, market_cooldown_seconds=60)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)

    first = Opportunity(
        market_id="m1",
        question="Will X happen?",
        direction="buy_both_merge",
        gross_cost=0.98,
        fee_cost=0.0,
        yes_fee_cost=0.0,
        no_fee_cost=0.0,
        gas_cost=0.0,
        slippage_buffer=0.0,
        net_cost=0.98,
        net_edge=0.02,
        capacity=1.0,
        mergeable_qty=1.0,
        yes_qty=1.0,
        no_qty=1.0,
        yes_price=0.49,
        no_price=0.49,
        yes_book_timestamp_ms=1000,
        no_book_timestamp_ms=1000,
        ts=datetime.now(timezone.utc),
    )
    second = Opportunity(
        market_id="m1",
        question="Will X happen?",
        direction="buy_both_merge",
        gross_cost=0.96,
        fee_cost=0.0,
        yes_fee_cost=0.0,
        no_fee_cost=0.0,
        gas_cost=0.0,
        slippage_buffer=0.0,
        net_cost=0.96,
        net_edge=0.04,
        capacity=1.0,
        mergeable_qty=1.0,
        yes_qty=1.0,
        no_qty=1.0,
        yes_price=0.48,
        no_price=0.48,
        yes_book_timestamp_ms=1001,
        no_book_timestamp_ms=1001,
        ts=datetime.now(timezone.utc),
    )

    assert simulator.consume(_market(), [first]) == 1
    assert simulator.consume(_market(), [second]) == 0
    assert simulator.last_rejection_counts == {"market_cooldown": 1}


def test_paper_simulator_applies_market_notional_limit(tmp_path):
    cfg = PolySettings(_env_file=None,
        data_dir=str(tmp_path),
        paper_initial_cash=100,
        market_cooldown_seconds=0,
        max_market_notional_per_day=1.5,
    )
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)
    opportunity = Opportunity(
        market_id="m1",
        question="Will X happen?",
        direction="buy_both_merge",
        gross_cost=0.98,
        fee_cost=0.0,
        yes_fee_cost=0.0,
        no_fee_cost=0.0,
        gas_cost=0.0,
        slippage_buffer=0.0,
        net_cost=0.98,
        net_edge=0.02,
        capacity=1.0,
        mergeable_qty=1.0,
        yes_qty=1.0,
        no_qty=1.0,
        yes_price=0.49,
        no_price=0.49,
        yes_book_timestamp_ms=1000,
        no_book_timestamp_ms=1000,
        ts=datetime.now(timezone.utc),
    )
    second = replace(opportunity, yes_book_timestamp_ms=1001, no_book_timestamp_ms=1001, ts=datetime.now(timezone.utc))

    assert simulator.consume(_market(), [opportunity]) == 1
    assert simulator.consume(_market(), [second]) == 0
    assert simulator.last_rejection_counts == {"market_notional_limit": 1}


def test_paper_simulator_skips_all_writes_when_no_opportunities(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)

    accepted = simulator.consume(_market(), [])

    assert accepted == 0

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        fills = conn.execute("SELECT count(*) FROM paper_fills").fetchone()[0]
        summaries = conn.execute("SELECT count(*) FROM paper_daily_summary").fetchone()[0]
    finally:
        conn.close()

    assert fills == 0
    assert summaries == 0


def test_paper_simulator_records_zero_activity_heartbeat(tmp_path):
    cfg = PolySettings(_env_file=None, data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)

    simulator.record_scan_heartbeat()

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        row = conn.execute("SELECT signals, accepted_signals, simulated_trades FROM paper_daily_summary LIMIT 1").fetchone()
    finally:
        conn.close()

    assert row == (0, 0, 0)
