from collector.baostock_client import BaostockClient


class _FakeResult:
    def __init__(self, rows, error_code="0", error_msg="", fields=None):
        self._rows = rows
        self._idx = -1
        self.error_code = error_code
        self.error_msg = error_msg
        self.fields = fields or ["calendar_date", "is_trading_day"]

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class _FakeBS:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def query_trade_dates(self, start_date=None, end_date=None):
        self.calls.append((start_date, end_date))
        return _FakeResult(self._rows)


def test_get_trade_dates_filters_non_trading_days():
    client = BaostockClient()
    client._bs = _FakeBS(
        [
            ["2026-03-28", "0"],
            ["2026-03-29", "0"],
            ["2026-03-30", "1"],
            ["2026-03-31", "1"],
        ]
    )
    client._logged_in = True

    dates = client.get_trade_dates(start="2026-03-28", end="2026-03-31")

    assert dates == ["2026-03-30", "2026-03-31"]
    assert client._bs.calls == [("2026-03-28", "2026-03-31")]


def test_latest_trade_date_returns_last_available_trading_day(monkeypatch):
    client = BaostockClient()
    monkeypatch.setattr(
        client,
        "get_trade_dates",
        lambda start=None, end=None: ["2026-03-27", "2026-03-30"],
    )

    latest = client.latest_trade_date(on_or_before="2026-03-31", lookback_days=7)

    assert latest == "2026-03-30"
