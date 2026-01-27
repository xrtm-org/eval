# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging

import networkx as nx
from xrtm.data.schemas.forecast import ForecastOutput

logger = logging.getLogger(__name__)

class InterventionEngine:
    @staticmethod
    def apply_intervention(output: ForecastOutput, node_id: str, new_probability: float) -> ForecastOutput:
        new_output = output.model_copy(deep=True)
        dg = new_output.to_networkx()
        if node_id not in dg:
            raise ValueError(f"Node ID '{node_id}' not found in the causal graph.")
        for node in new_output.logical_trace:
            if node.node_id == node_id:
                node.probability = new_probability
                break
        else:
            raise ValueError(f"Node ID '{node_id}' not found in logical_trace.")
        nodes_ordered = list(nx.topological_sort(dg))
        start_index = nodes_ordered.index(node_id)
        for current_id in nodes_ordered[start_index:]:
            current_node = next(n for n in new_output.logical_trace if n.node_id == current_id)
            for _, target_id, data in dg.out_edges(current_id, data=True):
                weight = data.get("weight", 1.0)
                target_node = next(n for n in new_output.logical_trace if n.node_id == target_id)
                old_target_prob = target_node.probability or 0.5
                normalized_delta = (current_node.probability - (dg.nodes[current_id].get("probability") or 0.5)) * weight
                target_node.probability = max(0.0, min(1.0, old_target_prob + normalized_delta))
        leaf_nodes = [n for n in dg.nodes() if dg.out_degree(n) == 0]
        if leaf_nodes:
            avg_leaf_prob = sum(next(n.probability for n in new_output.logical_trace if n.node_id == leaf_id) or 0.0 for leaf_id in leaf_nodes) / len(leaf_nodes)
            new_output.confidence = avg_leaf_prob
        return new_output

__all__ = ["InterventionEngine"]
