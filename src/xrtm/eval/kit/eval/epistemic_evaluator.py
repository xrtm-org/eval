# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, Optional

# From xrtm-data
from xrtm.data.core.schemas.forecast import ForecastOutput

# From xrtm-eval (local)
from xrtm.eval.core.epistemics import IntegrityGuardian, SourceTrustRegistry

logger = logging.getLogger(__name__)


class EpistemicEvaluator:
    def __init__(self, registry: Optional[SourceTrustRegistry] = None):
        self.registry = registry or SourceTrustRegistry()
        self.guardian = IntegrityGuardian(self.registry)

    async def evaluate_forecast_integrity(self, output: ForecastOutput) -> Dict[str, Any]:
        sources = output.metadata.get("sources", [])
        validation, scores = await self.guardian.validate_data_sources(sources)
        avg_trust = sum(scores) / len(scores) if scores else 0.5
        return {
            "aggregate_trust_score": avg_trust,
            "source_validation": validation,
            "integrity_level": "HIGH" if avg_trust > 0.8 else "MEDIUM" if avg_trust >= 0.5 else "LOW",
        }


__all__ = ["EpistemicEvaluator"]
