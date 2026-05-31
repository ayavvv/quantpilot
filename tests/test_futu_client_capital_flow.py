import pandas as pd

from collector import futu_client as futu_mod
from collector.futu_client import FutuClient


def _client_with_ctx(ctx):
    client = FutuClient.__new__(FutuClient)
    client.ctx = ctx
    client._retry_wrapper = lambda func, *args, **kwargs: func(*args, **kwargs)
    return client


class _CapitalFlowCtx:
    def get_capital_flow(self, code, period_type=None, start=None, end=None):
        assert code == "SH.600000"
        assert start == "2026-05-01"
        assert end == "2026-05-29"
        return 0, pd.DataFrame(
            [
                {
                    "capital_flow_item_time": "2026-05-29 00:00:00",
                    "in_flow": 100.0,
                    "main_in_flow": 80.0,
                    "super_in_flow": 30.0,
                    "big_in_flow": 50.0,
                    "mid_in_flow": -10.0,
                    "sml_in_flow": 30.0,
                }
            ]
        )


class _CapitalDistributionCtx:
    def get_capital_distribution(self, code):
        assert code == "SH.600000"
        return 0, pd.DataFrame(
            [
                {
                    "capital_in_super": 100.0,
                    "capital_in_big": 80.0,
                    "capital_in_mid": 50.0,
                    "capital_in_small": 20.0,
                    "capital_out_super": 60.0,
                    "capital_out_big": 30.0,
                    "capital_out_mid": 70.0,
                    "capital_out_small": 10.0,
                    "update_time": "2026-05-29 15:00:00",
                }
            ]
        )


class _ErrorCtx:
    def get_capital_flow(self, code, period_type=None, start=None, end=None):
        return -1, "permission denied"


def test_futu_client_uses_configured_rsa_key(monkeypatch, tmp_path):
    calls = []

    class FakeSysConfig:
        @staticmethod
        def enable_proto_encrypt(enabled):
            calls.append(("enable", enabled))

        @staticmethod
        def set_init_rsa_file(path):
            calls.append(("rsa", path))

    key_path = tmp_path / "futu_rsa.pem"
    key_path.write_text("test-key")
    monkeypatch.setattr(futu_mod, "SysConfig", FakeSysConfig)
    monkeypatch.setenv("FUTU_RSA_KEY", str(key_path))

    FutuClient("127.0.0.1", 11111)

    assert calls == [("enable", True), ("rsa", str(key_path))]


def test_futu_client_disables_encryption_when_rsa_key_missing(monkeypatch):
    calls = []

    class FakeSysConfig:
        @staticmethod
        def enable_proto_encrypt(enabled):
            calls.append(("enable", enabled))

        @staticmethod
        def set_init_rsa_file(path):
            calls.append(("rsa", path))

    monkeypatch.setattr(futu_mod, "SysConfig", FakeSysConfig)
    monkeypatch.setenv("FUTU_RSA_KEY", "/missing/futu_rsa.pem")

    FutuClient("127.0.0.1", 11111)

    assert calls == [("enable", False)]


def test_futu_client_uses_project_rsa_key_when_env_absent(monkeypatch, tmp_path):
    calls = []

    class FakeSysConfig:
        @staticmethod
        def enable_proto_encrypt(enabled):
            calls.append(("enable", enabled))

        @staticmethod
        def set_init_rsa_file(path):
            calls.append(("rsa", path))

    key_path = tmp_path / "project_rsa.pem"
    key_path.write_text("test-key")
    monkeypatch.setattr(futu_mod, "SysConfig", FakeSysConfig)
    monkeypatch.setattr(futu_mod, "PROJECT_RSA_KEY", str(key_path))
    monkeypatch.delenv("FUTU_RSA_KEY", raising=False)

    FutuClient("127.0.0.1", 11111)

    assert calls == [("enable", True), ("rsa", str(key_path))]


def test_get_capital_flow_normalizes_major_flow_fields():
    client = _client_with_ctx(_CapitalFlowCtx())

    records = client.get_capital_flow("SH.600000", period_type="DAY", start="2026-05-01", end="2026-05-29")

    assert records == [
        {
            "code": "SH.600000",
            "time": "2026-05-29 00:00:00",
            "date": "2026-05-29",
            "in_flow": 100.0,
            "main_in_flow": 80.0,
            "super_in_flow": 30.0,
            "big_in_flow": 50.0,
            "mid_in_flow": -10.0,
            "sml_in_flow": 30.0,
        }
    ]


def test_get_capital_distribution_derives_net_main_flow():
    client = _client_with_ctx(_CapitalDistributionCtx())

    result = client.get_capital_distribution("SH.600000")

    assert result["net_super"] == 40.0
    assert result["net_big"] == 50.0
    assert result["net_main"] == 90.0
    assert result["capital_in_main"] == 180.0
    assert result["capital_out_main"] == 90.0
    assert result["update_time"] == "2026-05-29 15:00:00"


def test_get_capital_flow_raises_on_permission_error():
    client = _client_with_ctx(_ErrorCtx())

    try:
        client.get_capital_flow("SH.600000", period_type="DAY")
    except RuntimeError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
