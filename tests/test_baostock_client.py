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
    def __init__(self, rows, trade_results=None, kline_results=None):
        self._rows = rows
        self._trade_results = list(trade_results or [])
        self._kline_results = list(kline_results or [])
        self.calls = []
        self.kline_calls = []

    def query_trade_dates(self, start_date=None, end_date=None):
        self.calls.append((start_date, end_date))
        if self._trade_results:
            return self._trade_results.pop(0)
        return _FakeResult(self._rows)

    def query_history_k_data_plus(self, code, fields, start_date=None, end_date=None, frequency=None, adjustflag=None):
        self.kline_calls.append((code, start_date, end_date, frequency, adjustflag))
        if self._kline_results:
            return self._kline_results.pop(0)
        return _FakeResult(
            self._rows,
            fields=fields.split(","),
        )


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


def test_get_trade_dates_relogs_when_session_expires():
    client = BaostockClient()
    client._bs = _FakeBS(
        [],
        trade_results=[
            _FakeResult([], error_code="-1", error_msg="用户未登录"),
            _FakeResult(
                [["2026-03-30", "1"], ["2026-03-31", "1"]],
                fields=["calendar_date", "is_trading_day"],
            ),
        ],
    )
    client._logged_in = True
    relogins = []

    def fake_ensure_login():
        relogins.append("login")
        client._logged_in = True

    client._ensure_login = fake_ensure_login

    dates = client.get_trade_dates(start="2026-03-30", end="2026-03-31")

    assert dates == ["2026-03-30", "2026-03-31"]
    assert relogins == ["login", "login"]
    assert client._bs.calls == [
        ("2026-03-30", "2026-03-31"),
        ("2026-03-30", "2026-03-31"),
    ]


def test_get_history_kline_relogs_when_session_expires():
    client = BaostockClient(rate_limit=0)
    client._bs = _FakeBS(
        [],
        kline_results=[
            _FakeResult([], error_code="-1", error_msg="用户未登录"),
            _FakeResult(
                [["2026-03-31", "sh.600000", "10", "11", "9", "10.5", "1000", "10500", "1.2", "3.5"]],
                fields=["date", "code", "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"],
            ),
        ],
    )
    client._logged_in = True
    relogins = []

    def fake_ensure_login():
        relogins.append("login")
        client._logged_in = True

    client._ensure_login = fake_ensure_login

    records = client.get_history_kline("SH.600000", start="2026-03-31", end="2026-03-31")

    assert records == [{
        "code": "SH.600000",
        "time_key": "2026-03-31 00:00:00",
        "open": 10.0,
        "close": 10.5,
        "high": 11.0,
        "low": 9.0,
        "volume": 1000,
        "turnover": 10500.0,
        "pe_ratio": 0.0,
        "turnover_rate": 1.2,
        "change_rate": 3.5,
    }]
    assert relogins == ["login", "login"]
    assert client._bs.kline_calls == [
        ("sh.600000", "2026-03-31", "2026-03-31", "d", "2"),
        ("sh.600000", "2026-03-31", "2026-03-31", "d", "2"),
    ]
