import numpy as np

from converter.incremental import QlibDirectWriter


def test_get_stock_last_date_prefers_instruments_metadata(tmp_path):
    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "instruments").mkdir(parents=True)
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "instruments" / "all.txt").write_text(
        "SH.600000\t2026-04-08\t2026-04-10\n",
        encoding="utf-8",
    )
    (qlib_dir / "calendars" / "day.txt").write_text(
        "2026-04-08\n2026-04-09\n2026-04-10\n",
        encoding="utf-8",
    )

    writer = QlibDirectWriter(qlib_dir)

    assert writer.get_stock_last_date("SH.600000") == "2026-04-10"


def test_get_stock_last_date_falls_back_to_bin_when_metadata_missing(tmp_path):
    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "calendars" / "day.txt").write_text(
        "2026-04-08\n2026-04-09\n2026-04-10\n",
        encoding="utf-8",
    )

    writer = QlibDirectWriter(qlib_dir)
    feat_dir = writer._get_feat_dir("SH.600000")
    feat_dir.mkdir(parents=True)
    np.array([0.0, 10.0, 11.0, 12.0], dtype="<f4").tofile(str(feat_dir / "close.day.bin"))

    assert writer.get_stock_last_date("SH.600000") == "2026-04-10"
