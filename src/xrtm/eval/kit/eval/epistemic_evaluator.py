# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging
from typing import Any, Dict, Optional

# From xrtm-forecast
from xrtm.forecast.core.epistemics import IntegrityGuardian, SourceTrustRegistry
# From xrtm-data
from xrtm.data.schemas.forecast import ForecastOutput

logger = logging.getLogger(__name__)

class EpistemicEvaluator:
    def __init__(self, registry: Optional[SourceTrustRegistry] = None):
        self.registry = registry or SourceTrustRegistry()
        self.guardian = IntegrityGuardian(self.registry)

    async def evaluate_forecast_integrity(self, output: ForecastOutput) -> Dict[str, Any]:
        sources = output.metadata.get("sources", [])
        validation = await self.guardian.validate_data_sources(sources)
        scores = [self.registry.get_trust_score(s) for s in sources]
        avg_trust = sum(scores) / len(scores) if scores else 0.5
        return {
            "aggregate_trust_score": avg_trust,
            "source_validation": validation,
            "integrity_level": "HIGH" if avg_trust > 0.8 else "MEDIUM" if avg_trust >= 0.5 else "LOW",
        }

__all__ = ["EpistemicEvaluator"]
