from scripts.us_microstructure_dates import collection_dates, default_report_date


def test_collection_dates_reads_hive_date_dirs(tmp_path):
    (tmp_path / "manifests" / "date=2026-06-01").mkdir(parents=True)
    (tmp_path / "trades" / "date=2026-05-29").mkdir(parents=True)
    (tmp_path / "quotes" / "date=bad").mkdir(parents=True)

    assert collection_dates(tmp_path) == ["2026-05-29", "2026-06-01"]


def test_default_report_date_prefers_latest_collection_date(tmp_path):
    (tmp_path / "manifests" / "date=2026-06-01").mkdir(parents=True)
    (tmp_path / "manifests" / "date=2026-05-29").mkdir(parents=True)

    assert default_report_date(tmp_path, today="2026-06-02") == "2026-06-01"


def test_default_report_date_falls_back_to_previous_local_date(tmp_path):
    assert default_report_date(tmp_path, today="2026-06-02") == "2026-06-01"
