import json

import pandas as pd

import scripts.build_us_microstructure_collection_universe as builder


def test_parse_args_defaults_to_focused_collection_cap(monkeypatch):
    monkeypatch.delenv("US_MICROSTRUCTURE_COLLECTION_MAX_SYMBOLS", raising=False)

    args = builder.parse_args([])

    assert args.max_total_symbols == 124


def _write_signals(base_dir, date_value, rows):
    path = base_dir / "signals" / f"date={date_value}"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "us_major_flow_signals.csv", index=False)


def test_build_collection_universe_adds_prior_strong_followups(tmp_path):
    current = tmp_path / "universe" / "us_microstructure_candidates_latest.txt"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("US.AAPL\nUS.NVDA\nUS.SPY\n", encoding="utf-8")
    _write_signals(
        tmp_path,
        "2026-06-02",
        [
            {"symbol": "US.LI", "side_score": 70, "stage": "accumulation_watch", "confidence": "watch", "data_quality_pass": True, "rank": 1},
            {"symbol": "US.AMD", "side_score": 40, "stage": "accumulation_diagnostic", "confidence": "high", "data_quality_pass": True, "rank": 2},
            {"symbol": "US.MSFT", "side_score": 54, "stage": "accumulation_diagnostic", "confidence": "diagnostic", "data_quality_pass": True, "rank": 3},
            {"symbol": "US.AAPL", "side_score": 99, "stage": "accumulation_watch", "confidence": "watch", "data_quality_pass": True, "rank": 4},
            {"symbol": "US.BAD", "side_score": 99, "stage": "accumulation_watch", "confidence": "watch", "data_quality_pass": False, "rank": 5},
        ],
    )
    _write_signals(
        tmp_path,
        "2026-06-01",
        [
            {"symbol": "US.TSLA", "side_score": 80, "stage": "accumulation_diagnostic", "confidence": "diagnostic", "data_quality_pass": True, "rank": 1},
        ],
    )

    collection, status = builder.build_collection_universe(
        base_dir=tmp_path,
        date_value="2026-06-03",
        current_universe_file=current,
        followup_days=2,
        followup_max_symbols=2,
        followup_min_score=55,
        max_total_symbols=5,
    )

    assert collection["symbol"].tolist() == ["US.AAPL", "US.NVDA", "US.SPY", "US.AMD", "US.LI"]
    assert collection["collection_source"].tolist() == ["current", "current", "current", "followup", "followup"]
    assert status["current_symbol_count"] == 3
    assert status["followup_candidate_count"] == 3
    assert status["followup_selected_count"] == 2
    assert status["collection_symbol_count"] == 5
    assert status["followup_origin_date_counts"] == {"2026-06-02": 2}


def test_build_collection_universe_preserves_current_symbols_when_total_cap_is_tight(tmp_path):
    current = tmp_path / "current.txt"
    current.write_text("US.AAPL\nUS.NVDA\nUS.SPY\nUS.LI\n", encoding="utf-8")
    _write_signals(
        tmp_path,
        "2026-06-02",
        [
            {"symbol": "US.TSLA", "side_score": 90, "stage": "accumulation_watch", "confidence": "watch", "data_quality_pass": True, "rank": 1},
        ],
    )

    collection, status = builder.build_collection_universe(
        base_dir=tmp_path,
        date_value="2026-06-03",
        current_universe_file=current,
        followup_days=1,
        followup_max_symbols=10,
        max_total_symbols=3,
    )

    assert collection["symbol"].tolist() == ["US.AAPL", "US.NVDA", "US.SPY", "US.LI"]
    assert status["followup_selected_count"] == 0
    assert status["collection_symbol_count"] == 4


def test_write_collection_outputs_writes_latest_files(tmp_path):
    collection = pd.DataFrame(
        [
            {"collection_rank": 1, "symbol": "US.AAPL", "collection_source": "current"},
            {"collection_rank": 2, "symbol": "US.LI", "collection_source": "followup"},
        ]
    )
    status = {"status": "ok", "date": "2026-06-03", "collection_symbol_count": 2}

    outputs = builder.write_collection_outputs(
        tmp_path,
        date_value="2026-06-03",
        collection=collection,
        status=status,
    )

    assert outputs["collection_latest_txt"].read_text(encoding="utf-8").splitlines() == ["US.AAPL", "US.LI"]
    payload = json.loads(outputs["collection_status_latest"].read_text(encoding="utf-8"))
    assert payload["collection_symbol_count"] == 2
    assert (tmp_path / "universe" / "date=2026-06-03" / "us_microstructure_collection_universe.csv").exists()
