import json
from datetime import datetime, timezone
from pathlib import Path

from polymarket.config import PolySettings
from polymarket.models import Opportunity, MarketInfo
from polymarket.paper.simulator import PaperSimulator
from polymarket.reporting.daily import build_daily_report, generate_daily_report
from polymarket.storage import PolyStorage


def _market() -> MarketInfo:
    return MarketInfo(
        market_id="m1",
        condition_id="m1",
        question="Will X happen?",
        slug="will-x-happen",
        end_date_iso="2026-12-31",
        min_order_size=1,
        tick_size=0.01,
        neg_risk=False,
        enable_order_book=True,
        taker_base_fee_bps=0,
        yes_token_id="yes",
        no_token_id="no",
    )


def test_build_daily_report_returns_no_data_when_db_missing(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path))
    payload = build_daily_report(cfg=cfg, target_date="2026-04-17")

    assert payload["status"] == "no_data"
    assert payload["summary"] is None


def test_generate_daily_report_writes_json_artifacts(tmp_path):
    cfg = PolySettings(data_dir=str(tmp_path), paper_initial_cash=100)
    storage = PolyStorage(cfg)
    simulator = PaperSimulator(storage=storage, cfg=cfg)
    simulator.consume(
        _market(),
        [
            Opportunity(
                market_id="m1",
                question="Will X happen?",
                direction="buy_both_merge",
                gross_cost=0.98,
                fee_cost=0.0,
                yes_fee_cost=0.0,
                no_fee_cost=0.0,
                gas_cost=0.0,
                slippage_buffer=0.0,
                net_cost=0.98,
                net_edge=0.02,
                capacity=1.0,
                mergeable_qty=1.0,
                yes_qty=1.0,
                no_qty=1.0,
                yes_price=0.49,
                no_price=0.49,
                yes_book_timestamp_ms=1000,
                no_book_timestamp_ms=1000,
                ts=datetime.now(timezone.utc),
            )
        ],
    )
    report_date = datetime.now(timezone.utc).date().isoformat()

    payload, paths = generate_daily_report(cfg=cfg, target_date=report_date)

    assert payload["status"] == "ok"
    assert payload["summary"]["signals"] == 1
    assert payload["summary"]["fill_count"] == 2
    assert payload["mirror_enabled"] is False
    assert payload["mirror_summary"] is None
    assert Path(paths["latest"]).exists()
    assert Path(paths["dated"]).exists()

    latest_payload = json.loads(Path(paths["latest"]).read_text())
    assert latest_payload["report_date"] == report_date
    assert latest_payload["summary"]["fill_count"] == 2
    assert latest_payload["summary"]["market_count"] >= 0
    assert 'mirror_reports_path' in latest_payload
