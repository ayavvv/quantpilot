import pandas as pd
import pytest

from scripts import refresh_eastmoney_fund_flow_rank as refresher


def test_refresh_rank_writes_latest_and_archive(monkeypatch, tmp_path):
    df = pd.DataFrame(
        [
            {"code": "SH.600000", "main_net_inflow": 10.0},
            {"code": "SZ.000001", "main_net_inflow": -5.0},
        ]
    )
    monkeypatch.setattr(refresher, "fetch_fund_flow_rank", lambda **kwargs: df)

    latest = tmp_path / "output" / "rank_latest.csv"
    archive_dir = tmp_path / "archive"
    result, paths = refresher.refresh_rank(
        output=latest,
        archive_dir=archive_dir,
        limit=6000,
        timeout=10,
        source="auto",
        min_rows=2,
    )

    assert len(result) == 2
    assert paths["latest"] == latest
    assert latest.exists()
    assert "archive" in paths
    assert paths["archive"].exists()
    assert pd.read_csv(latest)["code"].tolist() == ["SH.600000", "SZ.000001"]


def test_refresh_rank_does_not_clobber_latest_when_too_few_rows(monkeypatch, tmp_path):
    latest = tmp_path / "rank_latest.csv"
    latest.write_text("code,main_net_inflow\nSH.600000,1\n", encoding="utf-8")
    monkeypatch.setattr(refresher, "fetch_fund_flow_rank", lambda **kwargs: pd.DataFrame([{"code": "SH.600001"}]))

    with pytest.raises(RuntimeError, match="below minimum"):
        refresher.refresh_rank(
            output=latest,
            archive_dir=None,
            limit=6000,
            timeout=10,
            source="auto",
            min_rows=2,
        )

    assert latest.read_text(encoding="utf-8") == "code,main_net_inflow\nSH.600000,1\n"
