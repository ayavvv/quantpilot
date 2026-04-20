from polymarket.traders.discovery import normalize_leaderboard_profiles


def test_normalize_leaderboard_profiles_uses_wallet_as_identity():
    profiles = normalize_leaderboard_profiles([
        {
            'proxyWallet': '0xabc',
            'userName': 'alice',
            'xUsername': 'alice_x',
            'verifiedBadge': True,
            'profileImage': 'img',
        }
    ])

    assert len(profiles) == 1
    assert profiles[0].wallet == '0xabc'
    assert profiles[0].user_name == 'alice'
    assert profiles[0].verified_badge is True
