import pickle
from pathlib import Path

import pandas as pd

from strategy import engine


class DummyModel:
    def predict(self, dataset, segment="infer"):
        idx = pd.MultiIndex.from_arrays(
            [[pd.Timestamp("2026-04-08")], ["SH.600000"]],
            names=["datetime", "instrument"],
        )
        return pd.Series([0.42], index=idx)


class LaggedDummyModel:
    def predict(self, dataset, segment="infer"):
        idx = pd.MultiIndex.from_arrays(
            [[pd.Timestamp("2026-04-07")], ["SH.600000"]],
            names=["datetime", "instrument"],
        )
        return pd.Series([0.42], index=idx)


class DummyDataset:
    def prepare(self, *args, **kwargs):
        return pd.DataFrame({"feature": [1.0]})


def test_resolve_infer_window_limits_history_to_recent_trading_days():
    calendar = [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03",
        "2026-04-06",
        "2026-04-07",
        "2026-04-08",
    ]

    start_date, n_bars = engine._resolve_infer_window(calendar, "2026-04-08", warmup_trading_days=3)

    assert start_date == "2026-04-03"
    assert n_bars == 4


def test_predict_next_day_uses_compact_infer_window(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text("SH.600000\t2006-01-03\t2026-04-08\n", encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "lightgbm_sh_latest.pkl", "wb") as f:
        pickle.dump(DummyModel(), f)

    monkeypatch.setattr(engine.qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(engine, "DEFAULT_INFER_WARMUP_TRADING_DAYS", 3)
    monkeypatch.setattr(engine, "_active_a_share_instruments", lambda provider_uri, last_date: ["SH.600000"])
    monkeypatch.setattr(
        engine,
        "_load_config",
        lambda path=None: {
            "task": {
                "dataset": {
                    "class": "DatasetH",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158Fund",
                            "kwargs": {
                                "start_time": "2017-07-01",
                                "fit_start_time": "2017-07-01",
                                "fit_end_time": "2025-12-31",
                            },
                        },
                        "segments": {},
                    },
                }
            }
        },
    )

    captured = {}

    def fake_init_instance_by_config(cfg):
        captured["dataset_cfg"] = cfg
        return DummyDataset()

    monkeypatch.setattr(engine, "init_instance_by_config", fake_init_instance_by_config)
    monkeypatch.setattr(
        engine.D,
        "calendar",
        lambda start_time, end_time, freq="day": pd.DatetimeIndex(
            [
                "2026-04-01",
                "2026-04-02",
                "2026-04-03",
                "2026-04-06",
                "2026-04-07",
                "2026-04-08",
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        engine.D,
        "features",
        lambda instruments, fields, start_time, end_time: (_ for _ in ()).throw(
            ValueError("force fallback")
        ),
        raising=False,
    )

    strategy_engine = engine.StrategyEngine(provider_uri=qlib_dir, models_dir=models_dir)
    result = strategy_engine._predict_next_day_impl(hk_mode=False)

    handler_kwargs = captured["dataset_cfg"]["kwargs"]["handler"]["kwargs"]
    assert handler_kwargs["instruments"] == ["SH.600000"]
    assert handler_kwargs["start_time"] == "2026-04-03"
    assert handler_kwargs["end_time"] == "2026-04-08"
    assert handler_kwargs["fit_start_time"] == "2017-07-01"
    assert handler_kwargs["fit_end_time"] == "2026-04-08"
    assert result.attrs["infer_date"] == "2026-04-08"
    assert result["code"].tolist() == ["SH.600000"]


def test_predict_next_day_prefers_direct_feature_fetch_for_a_share(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text(
        "US.SPY\t2006-01-03\t2026-04-08\nSH.600000\t2006-01-03\t2026-04-08\n",
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "lightgbm_sh_latest.pkl", "wb") as f:
        pickle.dump(DummyModel(), f)

    monkeypatch.setattr(engine.qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        engine,
        "_load_config",
        lambda path=None: {
            "task": {
                "dataset": {
                    "class": "DatasetH",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158Fund",
                            "module_path": "strategy.alpha_hk",
                            "kwargs": {},
                        },
                        "segments": {},
                    },
                }
            }
        },
    )
    monkeypatch.setattr(
        engine.D,
        "calendar",
        lambda start_time, end_time, freq="day": pd.DatetimeIndex(["2026-04-08"]),
        raising=False,
    )

    captured = {}

    def fake_features(instruments, fields, start_time, end_time):
        captured["instruments"] = list(instruments)
        captured["field_count"] = len(fields)
        idx = pd.MultiIndex.from_arrays(
            [[pd.Timestamp("2026-04-08")], ["SH.600000"]],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame([[1.0] * len(fields)], index=idx, columns=fields)

    monkeypatch.setattr(engine.D, "features", fake_features, raising=False)
    monkeypatch.setattr(engine, "init_instance_by_config", lambda cfg: (_ for _ in ()).throw(AssertionError("fallback should not run")))

    strategy_engine = engine.StrategyEngine(provider_uri=qlib_dir, models_dir=models_dir)
    result = strategy_engine._predict_next_day_impl(hk_mode=False)

    assert captured["instruments"] == ["SH.600000"]
    assert captured["field_count"] > 100
    assert result.attrs["infer_date"] == "2026-04-08"
    assert result["code"].tolist() == ["SH.600000"]


def test_predict_next_day_fallback_keeps_a_share_instruments_only(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text(
        "HK.00267\t2006-01-03\t2026-04-08\nSH.600000\t2006-01-03\t2026-04-08\nSZ.000001\t2006-01-03\t2026-04-08\n",
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "lightgbm_sh_latest.pkl", "wb") as f:
        pickle.dump(DummyModel(), f)

    monkeypatch.setattr(engine.qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        engine,
        "_load_config",
        lambda path=None: {
            "task": {
                "dataset": {
                    "class": "DatasetH",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158Fund",
                            "module_path": "strategy.alpha_hk",
                            "kwargs": {},
                        },
                        "segments": {},
                    },
                }
            }
        },
    )
    monkeypatch.setattr(
        engine.D,
        "calendar",
        lambda start_time, end_time, freq="day": pd.DatetimeIndex(["2026-04-08"]),
        raising=False,
    )
    monkeypatch.setattr(
        engine.D,
        "features",
        lambda instruments, fields, start_time, end_time: (_ for _ in ()).throw(
            ValueError("force fallback")
        ),
        raising=False,
    )

    captured = {}

    def fake_init_instance_by_config(cfg):
        captured["dataset_cfg"] = cfg
        return DummyDataset()

    monkeypatch.setattr(engine, "init_instance_by_config", fake_init_instance_by_config)

    strategy_engine = engine.StrategyEngine(provider_uri=qlib_dir, models_dir=models_dir)
    result = strategy_engine._predict_next_day_impl(hk_mode=False)

    handler_kwargs = captured["dataset_cfg"]["kwargs"]["handler"]["kwargs"]
    assert handler_kwargs["instruments"] == ["SH.600000", "SZ.000001"]
    assert result.attrs["infer_date"] == "2026-04-08"
    assert result["code"].tolist() == ["SH.600000"]


def test_predict_next_day_forces_live_infer_date_to_requested_last_date(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text("SH.600000\t2006-01-03\t2026-04-08\n", encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with open(models_dir / "lightgbm_sh_latest.pkl", "wb") as f:
        pickle.dump(LaggedDummyModel(), f)

    monkeypatch.setattr(engine.qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        engine,
        "_load_config",
        lambda path=None: {
            "task": {
                "dataset": {
                    "class": "DatasetH",
                    "kwargs": {
                        "handler": {
                            "class": "Alpha158Fund",
                            "module_path": "strategy.alpha_hk",
                            "kwargs": {},
                        },
                        "segments": {},
                    },
                }
            }
        },
    )
    monkeypatch.setattr(
        engine.D,
        "calendar",
        lambda start_time, end_time, freq="day": pd.DatetimeIndex(["2026-04-08"]),
        raising=False,
    )

    def fake_features(instruments, fields, start_time, end_time):
        idx = pd.MultiIndex.from_arrays(
            [[pd.Timestamp("2026-04-08")], ["SH.600000"]],
            names=["datetime", "instrument"],
        )
        return pd.DataFrame([[1.0] * len(fields)], index=idx, columns=fields)

    monkeypatch.setattr(engine.D, "features", fake_features, raising=False)

    strategy_engine = engine.StrategyEngine(provider_uri=qlib_dir, models_dir=models_dir)
    result = strategy_engine._predict_next_day_impl(hk_mode=False)

    assert result.attrs["infer_date"] == "2026-04-08"
