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


class TestInterventionEngine:
    """Test suite for InterventionEngine.apply_intervention method."""

    def test_basic_intervention_updates_target_node(self):
        """Verify that intervening on a node updates its probability."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
            CausalNode(node_id="b", event="Event B", probability=0.5),
        ]
        edges = [CausalEdge(source="a", target="b", weight=0.8)]
        
        output = ForecastOutput(
            question_id="test-q1",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="a", new_probability=0.9)
        
        # Find the intervened node
        node_a = next(n for n in new_output.logical_trace if n.node_id == "a")
        assert node_a.probability == 0.9, "Intervened node probability should be updated"
        
        # Original should be unchanged
        original_node_a = next(n for n in output.logical_trace if n.node_id == "a")
        assert original_node_a.probability == 0.5, "Original output should be unchanged"

    def test_downstream_propagation(self):
        """Verify that probability changes propagate through the causal graph."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
            CausalNode(node_id="b", event="Event B", probability=0.5),
            CausalNode(node_id="c", event="Event C", probability=0.5),
        ]
        edges = [
            CausalEdge(source="a", target="b", weight=0.8),
            CausalEdge(source="b", target="c", weight=0.6),
        ]
        
        output = ForecastOutput(
            question_id="test-q2",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        # Increase probability of root node
        new_output = engine.apply_intervention(output, node_id="a", new_probability=0.9)
        
        # Check that downstream nodes were affected
        node_b = next(n for n in new_output.logical_trace if n.node_id == "b")
        node_c = next(n for n in new_output.logical_trace if n.node_id == "c")
        
        # Node B should increase due to A's increase
        assert node_b.probability > 0.5, "Downstream node B should have increased probability"
        # Node C should also increase due to B's increase
        assert node_c.probability > 0.5, "Downstream node C should have increased probability"

    def test_intervention_with_zero_probability(self):
        """Verify that intervening with 0.0 probability works correctly."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.7),
            CausalNode(node_id="b", event="Event B", probability=0.6),
        ]
        edges = [CausalEdge(source="a", target="b", weight=0.8)]
        
        output = ForecastOutput(
            question_id="test-q3",
            confidence=0.6,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="a", new_probability=0.0)
        
        node_a = next(n for n in new_output.logical_trace if n.node_id == "a")
        assert node_a.probability == 0.0, "Node should be set to 0.0"
        
        # Downstream node should decrease
        node_b = next(n for n in new_output.logical_trace if n.node_id == "b")
        assert node_b.probability < 0.6, "Downstream node should decrease"
        assert node_b.probability >= 0.0, "Probability should not go below 0.0"

    def test_intervention_with_one_probability(self):
        """Verify that intervening with 1.0 probability works correctly."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.3),
            CausalNode(node_id="b", event="Event B", probability=0.4),
        ]
        edges = [CausalEdge(source="a", target="b", weight=0.8)]
        
        output = ForecastOutput(
            question_id="test-q4",
            confidence=0.4,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="a", new_probability=1.0)
        
        node_a = next(n for n in new_output.logical_trace if n.node_id == "a")
        assert node_a.probability == 1.0, "Node should be set to 1.0"
        
        # Downstream node should increase
        node_b = next(n for n in new_output.logical_trace if n.node_id == "b")
        assert node_b.probability > 0.4, "Downstream node should increase"
        assert node_b.probability <= 1.0, "Probability should not exceed 1.0"

    def test_intervention_on_nonexistent_node_in_graph(self):
        """Verify that intervening on a node not in the graph raises ValueError."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
        ]
        
        output = ForecastOutput(
            question_id="test-q5",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=[],
        )
        
        engine = InterventionEngine()
        with pytest.raises(ValueError, match="Node ID 'nonexistent' not found in the causal graph"):
            engine.apply_intervention(output, node_id="nonexistent", new_probability=0.7)

    def test_intervention_on_node_not_in_logical_trace(self):
        """Verify that intervening on a node not in logical_trace raises ValueError."""
        # Create a scenario where a node is in the graph but not in logical_trace
        # This is more of an edge case test
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
        ]
        
        output = ForecastOutput(
            question_id="test-q6",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=[],
        )
        
        engine = InterventionEngine()
        
        # This should fail because the node doesn't exist at all
        with pytest.raises(ValueError, match="not found in"):
            engine.apply_intervention(output, node_id="missing", new_probability=0.7)

    def test_leaf_node_confidence_calculation(self):
        """Verify that the output confidence is updated based on leaf nodes."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
            CausalNode(node_id="b", event="Event B", probability=0.5),
            CausalNode(node_id="c", event="Event C", probability=0.5),
        ]
        edges = [
            CausalEdge(source="a", target="b", weight=0.8),
            # b and c are both leaf nodes (no outgoing edges)
        ]
        
        output = ForecastOutput(
            question_id="test-q7",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="a", new_probability=0.9)
        
        # Node b should be affected (downstream of a)
        node_b = next(n for n in new_output.logical_trace if n.node_id == "b")
        # Node c should not be affected (not connected to a)
        node_c = next(n for n in new_output.logical_trace if n.node_id == "c")
        
        # The confidence should be the average of leaf nodes (b and c)
        expected_confidence = (node_b.probability + node_c.probability) / 2
        assert abs(new_output.confidence - expected_confidence) < 0.001, \
            "Confidence should be average of leaf nodes"

    def test_intervention_on_middle_node(self):
        """Verify that intervening on a middle node (not root) works correctly."""
        nodes = [
            CausalNode(node_id="a", event="Event A", probability=0.5),
            CausalNode(node_id="b", event="Event B", probability=0.5),
            CausalNode(node_id="c", event="Event C", probability=0.5),
        ]
        edges = [
            CausalEdge(source="a", target="b", weight=0.8),
            CausalEdge(source="b", target="c", weight=0.6),
        ]
        
        output = ForecastOutput(
            question_id="test-q8",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        # Intervene on middle node b
        new_output = engine.apply_intervention(output, node_id="b", new_probability=0.2)
        
        # Node a should be unchanged (upstream)
        node_a = next(n for n in new_output.logical_trace if n.node_id == "a")
        assert node_a.probability == 0.5, "Upstream node should be unchanged"
        
        # Node b should be set to new value
        node_b = next(n for n in new_output.logical_trace if n.node_id == "b")
        assert node_b.probability == 0.2, "Intervened node should be updated"
        
        # Node c should be affected (downstream)
        node_c = next(n for n in new_output.logical_trace if n.node_id == "c")
        assert node_c.probability < 0.5, "Downstream node should decrease"

    def test_complex_graph_with_multiple_paths(self):
        """Verify intervention works correctly in a more complex DAG structure."""
        nodes = [
            CausalNode(node_id="root", event="Root Event", probability=0.5),
            CausalNode(node_id="left", event="Left Branch", probability=0.5),
            CausalNode(node_id="right", event="Right Branch", probability=0.5),
            CausalNode(node_id="merge", event="Merged Event", probability=0.5),
        ]
        edges = [
            CausalEdge(source="root", target="left", weight=0.8),
            CausalEdge(source="root", target="right", weight=0.7),
            CausalEdge(source="left", target="merge", weight=0.6),
            CausalEdge(source="right", target="merge", weight=0.5),
        ]
        
        output = ForecastOutput(
            question_id="test-q9",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=edges,
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="root", new_probability=0.9)
        
        # Both branches should be affected
        left_node = next(n for n in new_output.logical_trace if n.node_id == "left")
        right_node = next(n for n in new_output.logical_trace if n.node_id == "right")
        merge_node = next(n for n in new_output.logical_trace if n.node_id == "merge")
        
        assert left_node.probability > 0.5, "Left branch should increase"
        assert right_node.probability > 0.5, "Right branch should increase"
        assert merge_node.probability > 0.5, "Merge node should increase"

    def test_no_edges_single_node(self):
        """Verify intervention works with a single node and no edges."""
        nodes = [
            CausalNode(node_id="only", event="Only Event", probability=0.5),
        ]
        
        output = ForecastOutput(
            question_id="test-q10",
            confidence=0.5,
            reasoning="Test reasoning",
            logical_trace=nodes,
            logical_edges=[],
        )
        
        engine = InterventionEngine()
        new_output = engine.apply_intervention(output, node_id="only", new_probability=0.8)
        
        only_node = next(n for n in new_output.logical_trace if n.node_id == "only")
        assert only_node.probability == 0.8, "Node should be updated"
        # Since it's the only node and a leaf, confidence should equal its probability
        assert new_output.confidence == 0.8, "Confidence should equal the only leaf node"
