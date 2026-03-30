from pathlib import Path

import pandas as pd
import pytest

from inference import run_daily


def test_latest_a_share_date_reads_a_share_instruments(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib"
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True)
    (inst_dir / "all.txt").write_text(
        "\n".join(
            [
                "US.SPY\t2006-01-03\t2026-03-28",
                "SH.600000\t2006-01-03\t2026-03-27",
                "SZ.000001\t2011-03-18\t2026-03-26",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(run_daily, "QLIB_DATA_DIR", qlib_dir)
    assert run_daily.latest_a_share_date() == "2026-03-27"


def test_validate_signal_alignment_uses_latest_a_share_date():
    run_daily.validate_signal_alignment(
        validated_date="2026-03-28",
        signal_date="2026-03-27",
        latest_a_date="2026-03-27",
    )

    with pytest.raises(RuntimeError, match="latest_a_share=2026-03-27"):
        run_daily.validate_signal_alignment(
            validated_date="2026-03-28",
            signal_date="2026-03-25",
            latest_a_date="2026-03-27",
        )


def test_step3_output_updates_latest_links(tmp_path, monkeypatch):
    monkeypatch.setattr(run_daily, "SIGNAL_DIR", tmp_path)
    monkeypatch.setattr(run_daily, "SIGNAL_OUTPUT_TAG", "20260328")

    df = pd.DataFrame(
        [
            {"code": "SH.600000", "score": 0.8, "rank": 1, "top5": True},
            {"code": "SZ.000001", "score": 0.7, "rank": 2, "top5": True},
        ]
    )

    paths = run_daily.step3_output(df, "2026-03-27", promote_latest=True)

    assert paths["csv"].exists()
    assert paths["pkl"].exists()
    assert paths["latest_csv"].is_symlink()
    assert paths["latest_pkl"].is_symlink()
    assert paths["latest_csv"].resolve() == paths["csv"].resolve()
    assert paths["latest_pkl"].resolve() == paths["pkl"].resolve()
