from polymarket.config import PolySettings
from polymarket.traders.ranking import compute_trader_scores


def test_compute_trader_scores_filters_low_activity():
    cfg = PolySettings(data_dir='/tmp/poly-test', top_trader_min_trades=5, top_trader_min_diversity=2, top_trader_tracked_count=3)
    scores = compute_trader_scores(
        [{'proxyWallet': '0xabc', 'pnl': 1000, 'vol': 5000}],
        {'0xabc': {'trade_count': 2, 'diversity_count': 1, 'realized_pnl': 500, 'volume': 5000}},
        cfg=cfg,
    )

    assert scores == []


def test_compute_trader_scores_ranks_high_score_first():
    cfg = PolySettings(data_dir='/tmp/poly-test', top_trader_min_trades=1, top_trader_min_diversity=1, top_trader_tracked_count=3)
    scores = compute_trader_scores(
        [
            {'proxyWallet': '0xa', 'pnl': 1000, 'vol': 5000},
            {'proxyWallet': '0xb', 'pnl': 100, 'vol': 5000},
        ],
        {
            '0xa': {'trade_count': 10, 'diversity_count': 5, 'realized_pnl': 700, 'volume': 5000},
            '0xb': {'trade_count': 10, 'diversity_count': 5, 'realized_pnl': 10, 'volume': 5000},
        },
        cfg=cfg,
    )

    assert [item.wallet for item in scores] == ['0xa', '0xb']
    assert scores[0].rank == 1
