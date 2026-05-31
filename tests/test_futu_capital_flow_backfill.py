import pandas as pd

from strategy.futu_capital_flow_backfill import (
    backfill_capital_flow_archives,
    backfill_one_signal_file,
    discover_signal_files,
)


def test_discover_signal_files_filters_date_stamped_csvs(tmp_path):
    for name in ["signal_20260527.csv", "signal_latest.csv", "signal_20260529.csv", "other.csv"]:
        (tmp_path / name).write_text("code,score\nSH.600000,1\n", encoding="utf-8")

    paths = discover_signal_files(tmp_path, start_date="2026-05-28")

    assert [path.name for path in paths] == ["signal_20260529.csv"]


def test_backfill_one_signal_file_writes_archive_without_distribution(tmp_path):
    signal = tmp_path / "signal_20260529.csv"
    signal.write_text(
        "code,score,rank,signal_date\n"
        "SH.600000,0.9,1,2026-05-29\n"
        "SZ.000001,0.8,2,2026-05-29\n",
        encoding="utf-8",
    )
    calls = []

    def fake_fetcher(codes, **kwargs):
        calls.append((codes, kwargs))
        return pd.DataFrame(
            [
                {
                    "code": "SH.600000",
                    "capital_flow_status": "ok",
                    "latest_main_in_flow": 15_000_000,
                    "main_5d_sum": 35_000_000,
                    "main_positive_5d": 4,
                },
                {
                    "code": "SZ.000001",
                    "capital_flow_status": "ok",
                    "latest_main_in_flow": -8_000_000,
                    "main_5d_sum": -25_000_000,
                    "main_positive_5d": 1,
                },
            ]
        )

    result = backfill_one_signal_file(
        signal,
        archive_dir=tmp_path / "archive",
        signal_top_n=2,
        flow_days=10,
        include_distribution=False,
        fetcher=fake_fetcher,
    )

    assert result["status"] == "written"
    assert calls[0][0] == ["SH.600000", "SZ.000001"]
    assert calls[0][1]["start"] == "2026-05-19"
    assert calls[0][1]["end"] == "2026-05-29"
    assert calls[0][1]["include_distribution"] is False
    overlay = pd.read_csv(tmp_path / "archive" / "20260529_overlay.csv")
    labels = dict(zip(overlay["code"], overlay["capital_flow_label"]))
    assert labels["SH.600000"] == "capital_flow_confirm"
    assert labels["SZ.000001"] == "risk_flag_main_outflow"


def test_backfill_capital_flow_archives_skips_existing_unless_overwrite(tmp_path):
    signal = tmp_path / "signal_20260529.csv"
    signal.write_text("code,score,rank,signal_date\nSH.600000,0.9,1,2026-05-29\n", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "20260529_overlay.csv").write_text("code\nSH.600000\n", encoding="utf-8")
    (archive / "20260529_flow.csv").write_text("code\nSH.600000\n", encoding="utf-8")

    def fake_fetcher(codes, **kwargs):
        raise AssertionError("fetcher should not be called")

    results = backfill_capital_flow_archives(
        tmp_path,
        archive_dir=archive,
        fetcher=fake_fetcher,
    )

    assert results[0]["status"] == "skipped"


def test_backfill_one_signal_file_refuses_low_ok_ratio(tmp_path):
    signal = tmp_path / "signal_20260529.csv"
    signal.write_text(
        "code,score,rank,signal_date\n"
        "SH.600000,0.9,1,2026-05-29\n"
        "SZ.000001,0.8,2,2026-05-29\n",
        encoding="utf-8",
    )

    def fake_fetcher(codes, **kwargs):
        return pd.DataFrame(
            [
                {"code": "SH.600000", "capital_flow_status": "error"},
                {"code": "SZ.000001", "capital_flow_status": "error"},
            ]
        )

    result = backfill_one_signal_file(
        signal,
        archive_dir=tmp_path / "archive",
        signal_top_n=2,
        min_ok_ratio=0.5,
        fetcher=fake_fetcher,
    )

    assert result["status"] == "failed"
    assert result["ok_ratio"] == 0
    assert "below threshold" in result["error"]
    assert not (tmp_path / "archive" / "20260529_overlay.csv").exists()
