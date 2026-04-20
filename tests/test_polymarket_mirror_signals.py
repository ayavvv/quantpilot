from polymarket.config import PolySettings
from polymarket.models import TraderEvent
from polymarket.traders.mirror import generate_mirror_signals


def test_generate_mirror_signals_filters_small_events():
    cfg = PolySettings(data_dir='/tmp/poly-test', top_trader_min_signal_size=50, top_trader_max_signal_notional=100)
    events = [
        TraderEvent(wallet='0xabc', event_type='TRADE', market_id='m1', asset='a1', side='BUY', size=10, price=0.5, timestamp=1, transaction_hash='0x1'),
        TraderEvent(wallet='0xabc', event_type='TRADE', market_id='m1', asset='a1', side='BUY', size=80, price=0.5, timestamp=2, transaction_hash='0x2'),
    ]

    signals = generate_mirror_signals(events, cfg=cfg)

    assert len(signals) == 1
    assert signals[0].signal_size == 80


def test_generate_mirror_signals_caps_signal_size():
    cfg = PolySettings(data_dir='/tmp/poly-test', top_trader_min_signal_size=10, top_trader_max_signal_notional=100)
    events = [
        TraderEvent(wallet='0xabc', event_type='TRADE', market_id='m1', asset='a1', side='BUY', size=180, price=0.5, timestamp=2, transaction_hash='0x2'),
    ]

    signals = generate_mirror_signals(events, cfg=cfg)

    assert len(signals) == 1
    assert signals[0].signal_size == 100
