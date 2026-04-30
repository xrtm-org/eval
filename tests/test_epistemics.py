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

import pytest
from xrtm.data import ForecastOutput, MetadataBase

from xrtm.eval.core.epistemics import IntegrityGuardian, SourceTrustRegistry
from xrtm.eval.kit.eval.epistemic_evaluator import EpistemicEvaluator


@pytest.mark.asyncio
async def test_validate_data_sources_with_scores_preserves_classification():
    registry = SourceTrustRegistry(default_trust=0.5)
    registry.register_source("trusted.example", 0.9)
    registry.register_source("flagged.example", 0.4)
    registry.register_source("blocked.example", 0.1)
    guardian = IntegrityGuardian(registry)

    validation, scores = await guardian.validate_data_sources_with_scores(
        ["trusted.example", "flagged.example", "blocked.example"]
    )

    assert validation == {
        "passed": ["trusted.example"],
        "flagged": ["flagged.example"],
        "blocked": ["blocked.example"],
    }
    assert scores == [0.9, 0.4, 0.1]


@pytest.mark.asyncio
async def test_epistemic_evaluator_uses_single_pass_scores():
    registry = SourceTrustRegistry(default_trust=0.5)
    registry.register_source("trusted.example", 0.9)
    output = ForecastOutput(
        question_id="q1",
        probability=0.7,
        reasoning="test",
        metadata=MetadataBase(sources=["trusted.example"]),
    )

    report = await EpistemicEvaluator(registry).evaluate_forecast_integrity(output)

    assert report["aggregate_trust_score"] == 0.9
    assert report["integrity_level"] == "HIGH"
