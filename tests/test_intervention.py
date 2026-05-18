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
from xrtm.data import CausalEdge, CausalNode, ForecastOutput

from xrtm.eval.kit.eval.intervention import InterventionEngine


def _forecast_output() -> ForecastOutput:
    return ForecastOutput(
        question_id="q1",
        probability=0.5,
        reasoning_trace={
            "narrative": "test",
            "causal_graph": {
                "nodes": [
                    CausalNode(node_id="a", event="A", probability=0.5),
                    CausalNode(node_id="b", event="B", probability=0.5),
                    CausalNode(node_id="c", event="C", probability=0.5),
                ],
                "edges": [
                    CausalEdge(source="a", target="b", weight=0.5),
                    CausalEdge(source="b", target="c", weight=0.5),
                ],
            },
        },
    )


def test_apply_intervention_updates_downstream_probabilities_without_mutating_original():
    output = _forecast_output()

    new_output = InterventionEngine.apply_intervention(output, node_id="a", new_probability=0.9)

    assert output.logical_trace[0].probability == 0.5
    assert new_output.logical_trace[0].probability == 0.9
    assert new_output.logical_trace[1].probability > 0.5
    assert new_output.logical_trace[2].probability > 0.5
    assert new_output.probability == pytest.approx(new_output.logical_trace[2].probability)


def test_apply_intervention_rejects_missing_node():
    with pytest.raises(ValueError, match="not found"):
        InterventionEngine.apply_intervention(_forecast_output(), node_id="missing", new_probability=0.9)


def test_apply_intervention_handles_none_leaf_probability():
    output = ForecastOutput(
        question_id="q1",
        probability=0.5,
        reasoning_trace={
            "narrative": "test",
            "causal_graph": {
                "nodes": [
                    CausalNode(node_id="a", event="A", probability=0.5),
                    CausalNode(node_id="b", event="B", probability=None),
                ],
                "edges": [],
            },
        },
    )

    new_output = InterventionEngine.apply_intervention(output, node_id="a", new_probability=0.9)

    assert new_output.probability == pytest.approx(0.45)
