import json

from scripts import build_major_money_digest as builder


def test_default_sources_skip_unhealthy_us_otc_proxy(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "output").mkdir(parents=True)
    otc_dir = data_dir / "capital_flow" / "us_otc_proxy"
    otc_dir.mkdir(parents=True)
    a_path = data_dir / "output" / "eastmoney_fund_flow_rank_latest.csv"
    otc_path = otc_dir / "US_OTC_latest_flow.csv"
    a_path.write_text("code,main_net_inflow\nSH.600000,100\n", encoding="utf-8")
    otc_path.write_text("code,capital_flow_status,latest_main_in_flow\nUS.AABB,ok,100\n", encoding="utf-8")
    (otc_dir / "US_OTC_latest_status.json").write_text(
        json.dumps({"status": "failed", "ok_count": 0}),
        encoding="utf-8",
    )

    sources = builder._default_sources(data_dir)

    assert ("A", a_path, "eastmoney") in sources
    assert not any(market == "US_OTC" for market, _, _ in sources)


def test_default_sources_include_healthy_us_otc_proxy(tmp_path):
    data_dir = tmp_path / "data"
    otc_dir = data_dir / "capital_flow" / "us_otc_proxy"
    otc_dir.mkdir(parents=True)
    otc_path = otc_dir / "US_OTC_latest_flow.csv"
    otc_path.write_text("code,capital_flow_status,latest_main_in_flow\nUS.AABB,ok,100\n", encoding="utf-8")
    (otc_dir / "US_OTC_latest_status.json").write_text(
        json.dumps({"status": "ok", "ok_count": 1}),
        encoding="utf-8",
    )

    assert ("US_OTC", otc_path, "polygon_otc_proxy") in builder._default_sources(data_dir)


def test_main_writes_dated_archive_outputs(tmp_path):
    source = tmp_path / "a_flow.csv"
    output_json = tmp_path / "latest.json"
    output_csv = tmp_path / "latest.csv"
    archive_dir = tmp_path / "archive"
    source.write_text(
        "code,name,main_net_inflow,update_time\nSH.600000,Entry,80000000,2026-05-29\n",
        encoding="utf-8",
    )

    assert (
        builder.main(
            [
                "--source",
                f"A:{source}:eastmoney",
                "--expected-markets",
                "A",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--archive-dir",
                str(archive_dir),
            ]
        )
        == 0
    )

    assert output_json.exists()
    assert output_csv.exists()
    archived_json = archive_dir / "20260529_major_money_digest.json"
    archived_csv = archive_dir / "20260529_major_money_digest.csv"
    assert archived_json.exists()
    assert archived_csv.exists()
    payload = json.loads(archived_json.read_text(encoding="utf-8"))
    assert payload["flow_date"] == "2026-05-29"
