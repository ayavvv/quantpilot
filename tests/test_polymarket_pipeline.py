from polymarket.config import PolySettings
from polymarket.pipeline import PipelineResult


def test_pipeline_result_supports_mirror_fields():
    result = PipelineResult(markets_seen=1, opportunities_found=2, trades_simulated=3, mirror_traders_tracked=4, mirror_signals_generated=5)

    assert result.mirror_traders_tracked == 4
    assert result.mirror_signals_generated == 5
