from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_DOCKERFILE = REPO_ROOT / "collector" / "Dockerfile"


def test_collector_image_includes_strategy_package():
    content = COLLECTOR_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY strategy/ ./strategy/" in content
