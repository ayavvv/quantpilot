from scripts import a_share_readiness


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_latest_nas_a_share_date_uses_instruments_metadata(monkeypatch):
    captured = {}

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd
        return _Result(stdout="2026-03-30\n")

    monkeypatch.setattr(a_share_readiness.subprocess, "run", fake_run)
    latest = a_share_readiness.latest_nas_a_share_date(
        nas_host="192.168.100.131",
        nas_user="theo",
        ssh_key="/tmp/nas_key",
        nas_qlib_path="/volume1/docker/quantpilot/qlib_data",
    )

    assert latest == "2026-03-30"
    assert captured["cmd"][-1].startswith("python3 -c ")
    assert "/volume1/docker/quantpilot/qlib_data/instruments/all.txt" in captured["cmd"][-1]


def test_latest_trade_date_via_collector_runs_in_collector_container(monkeypatch):
    captured = {}

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd
        return _Result(stdout="2026-03-30\n")

    monkeypatch.setattr(a_share_readiness.subprocess, "run", fake_run)
    latest = a_share_readiness.latest_trade_date_via_collector(
        nas_host="192.168.100.131",
        nas_user="theo",
        ssh_key="/tmp/nas_key",
        today="2026-03-31",
    )

    assert latest == "2026-03-30"
    assert "docker exec quantpilot-collector" in captured["cmd"][-1]
    assert "2026-03-31" in captured["cmd"][-1]


def test_latest_trade_date_via_collector_uses_last_non_empty_line(monkeypatch):
    def fake_run(cmd, check, capture_output, text):
        return _Result(stdout="login success!\n2026-03-30\nlogout success!\n")

    monkeypatch.setattr(a_share_readiness.subprocess, "run", fake_run)
    latest = a_share_readiness.latest_trade_date_via_collector(
        nas_host="192.168.100.131",
        nas_user="theo",
        ssh_key="/tmp/nas_key",
        today="2026-03-31",
    )

    assert latest == "2026-03-30"


def test_is_a_share_ready_compares_dates_lexicographically():
    assert a_share_readiness.is_a_share_ready("2026-03-30", "2026-03-30") is True
    assert a_share_readiness.is_a_share_ready("2026-03-31", "2026-03-30") is True
    assert a_share_readiness.is_a_share_ready("2026-03-27", "2026-03-30") is False
    assert a_share_readiness.is_a_share_ready("", "2026-03-30") is False
