import duckdb

from polymarket.config import PolySettings
from polymarket.models import MirrorSignal
from polymarket.traders.paper import MirrorBookkeeper
from polymarket.traders.storage import MirrorStorage


def test_mirror_bookkeeper_applies_buy_signal(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=1000)
    storage = MirrorStorage(cfg)
    bookkeeper = MirrorBookkeeper(storage=storage, initial_cash=cfg.paper_initial_cash)

    signal = MirrorSignal(
        wallet='0xabc',
        market_id='m1',
        asset='asset1',
        title='Title',
        outcome='Yes',
        side='BUY',
        source_size=100,
        source_price=0.5,
        signal_size=100,
        lag_seconds=300,
        timestamp=1,
        fingerprint='fp1',
    )

    executed = bookkeeper.apply_signal(signal, market_id='m1', title='Title', outcome='Yes', execution_price=0.5, fee_bps=0)

    assert executed is True
    assert bookkeeper.state.cash == 950.0
    assert not bookkeeper.positions.empty


def test_mirror_bookkeeper_dedupes_same_signal(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=1000)
    storage = MirrorStorage(cfg)
    bookkeeper = MirrorBookkeeper(storage=storage, initial_cash=cfg.paper_initial_cash)

    signal = MirrorSignal(
        wallet='0xabc',
        market_id='m1',
        asset='asset1',
        title='Title',
        outcome='Yes',
        side='BUY',
        source_size=100,
        source_price=0.5,
        signal_size=100,
        lag_seconds=300,
        timestamp=1,
        fingerprint='fp1',
    )

    assert bookkeeper.apply_signal(signal, market_id='m1', title='Title', outcome='Yes', execution_price=0.5, fee_bps=0) is True
    assert bookkeeper.apply_signal(signal, market_id='m1', title='Title', outcome='Yes', execution_price=0.5, fee_bps=0) is False

    conn = duckdb.connect(str(cfg.mirror_duckdb_path), read_only=True)
    try:
        fills = conn.execute("SELECT count(*) FROM mirror_fills").fetchone()[0]
        claims = conn.execute("SELECT count(*) FROM processed_signals").fetchone()[0]
    finally:
        conn.close()

    assert fills == 1
    assert claims == 1
