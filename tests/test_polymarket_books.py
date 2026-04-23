from polymarket.books import _parse_levels


def test_parse_levels_skips_invalid_rows():
    levels = _parse_levels([
        {"price": "0.5", "size": "10"},
        {"price": "", "size": "3"},
        {"size": "2"},
        {"price": "0.4", "size": "bad"},
    ])

    assert len(levels) == 1
    assert levels[0].price == 0.5
    assert levels[0].size == 10.0
