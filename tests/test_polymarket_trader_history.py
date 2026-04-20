from polymarket.traders.history import normalize_activity_events, summarize_trader_history


def test_normalize_activity_events_builds_trade_events():
    events = normalize_activity_events('0xabc', [
        {
            'type': 'TRADE',
            'conditionId': 'm1',
            'asset': 'asset1',
            'side': 'BUY',
            'size': 100,
            'price': 0.5,
            'timestamp': 123,
            'transactionHash': '0xhash',
            'title': 'Title',
            'outcome': 'Yes',
            'name': 'alice',
        }
    ])

    assert len(events) == 1
    assert events[0].wallet == '0xabc'
    assert events[0].event_type == 'TRADE'
    assert events[0].fingerprint().startswith('0xabc:TRADE:0xhash')


def test_summarize_trader_history_counts_diversity_and_realized_pnl():
    summary = summarize_trader_history(
        [{'conditionId': 'm1', 'initialValue': 100}],
        [{'conditionId': 'm2', 'realizedPnl': 50, 'totalBought': 200}],
        [{'conditionId': 'm3'}],
    )

    assert summary['trade_count'] == 1
    assert summary['diversity_count'] == 3
    assert summary['realized_pnl'] == 50
