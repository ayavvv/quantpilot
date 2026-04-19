from pathlib import Path

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
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
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
        summaries = conn.execute("SELECT count(*) FROM paper_daily_summary").fetchone()[0]
        processed = conn.execute("SELECT count(*) FROM processed_opportunities").fetchone()[0]
    finally:
        conn.close()

    assert fills == 2
    assert summaries == 1
    assert processed == 1


def test_paper_simulator_restores_state_between_runs(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
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
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
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

    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        fills = conn.execute("SELECT count(*) FROM paper_fills").fetchone()[0]
        processed = conn.execute("SELECT count(*) FROM processed_opportunities").fetchone()[0]
    finally:
        conn.close()

    assert fills == 2
    assert processed == 1
