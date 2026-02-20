
import pytest

from xrtm.eval.core.epistemics import IntegrityGuardian, SourceTrustRegistry
from xrtm.eval.kit.eval.epistemic_evaluator import EpistemicEvaluator


# Mock ForecastOutput
class ForecastOutput:
    def __init__(self, sources):
        self.metadata = {"sources": sources}

@pytest.fixture
def registry():
    reg = SourceTrustRegistry()
    reg.register_source("trusted.com", 0.9)
    reg.register_source("sketchy.com", 0.2)
    reg.register_source("unknown.com", 0.5) # Default is 0.5 anyway
    return reg

@pytest.fixture
def guardian(registry):
    return IntegrityGuardian(registry)

@pytest.mark.asyncio
async def test_validate_data_sources(guardian):
    sources = ["trusted.com", "sketchy.com", "unknown.com"]
    # This test asserts current behavior. I will need to update it if I change the return type.
    result, scores = await guardian.validate_data_sources(sources)

    assert "passed" in result
    assert "blocked" in result
    assert "flagged" in result

    assert "trusted.com" in result["passed"]
    assert "sketchy.com" in result["blocked"]
    # unknown.com has score 0.5. Current logic:
    # < threshold (0.3) -> blocked
    # < 0.5 -> flagged
    # else -> passed
    # So 0.5 is passed.
    assert "unknown.com" in result["passed"]

    assert len(scores) == 3
    assert scores[0] == 0.9
    assert scores[1] == 0.2
    assert scores[2] == 0.5

@pytest.mark.asyncio
async def test_evaluate_forecast_integrity(registry):
    evaluator = EpistemicEvaluator(registry)
    output = ForecastOutput(["trusted.com", "sketchy.com"])

    result = await evaluator.evaluate_forecast_integrity(output)

    # 0.9 + 0.2 = 1.1 / 2 = 0.55
    assert result["aggregate_trust_score"] == pytest.approx(0.55)
    assert result["integrity_level"] == "MEDIUM"
    assert "source_validation" in result
