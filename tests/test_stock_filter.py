import json

from strategy.stock_filter import (
    A_SHARE_ST_METADATA,
    build_a_share_stock_basic_metadata,
    filter_st_codes,
    is_st_stock_name,
)


def test_is_st_stock_name_matches_st_prefixes():
    assert is_st_stock_name("ST榕泰")
    assert is_st_stock_name("*ST美谷")
    assert is_st_stock_name(" S*ST佳通 ")
    assert not is_st_stock_name("浦发银行")
    assert not is_st_stock_name("长江存储")


def test_build_a_share_stock_basic_metadata_marks_st_codes():
    payload = build_a_share_stock_basic_metadata(
        [
            {"code": "SH.600000", "name": "浦发银行", "type": "1", "status": "1"},
            {"code": "SZ.000001", "name": "*ST测试", "type": "1", "status": "1"},
            {"code": "US.SPY", "name": "SPY"},
        ],
        source="test",
    )

    assert payload["total"] == 2
    assert payload["st_count"] == 1
    assert payload["st_codes"] == ["SZ.000001"]


def test_filter_st_codes_uses_qlib_metadata(tmp_path):
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / f"{A_SHARE_ST_METADATA}.json").write_text(
        json.dumps(
            {
                "st_codes": ["SH.600000"],
                "stocks": [
                    {"code": "SH.600000", "name": "ST测试", "is_st": True},
                    {"code": "SZ.000001", "name": "平安银行", "is_st": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert filter_st_codes(tmp_path, ["SH.600000", "SZ.000001"], context="test") == ["SZ.000001"]
