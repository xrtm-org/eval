# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

from typing import Any, Dict

from xrtm.eval.core.eval.definitions import EvaluationResult, Evaluator


class BiasInterceptor(Evaluator):
    COGNITIVE_BIASES = [
        "Base-Rate Neglect", "Overconfidence", "Availability Heuristic",
        "Confirmation Bias", "Anchoring Bias", "Sunk Cost Fallacy",
        "Hindsight Bias", "Optimism Bias", "Pessimism Bias",
        "Status Quo Bias", "Framing Effect", "Recency Bias",
    ]

    def __init__(self, model: Any):
        self.model = model

    def score(self, prediction: Any, ground_truth: Any) -> float:
        return 0.0

    async def evaluate_reasoning(self, reasoning: str) -> Dict[str, Any]:
        prompt = f"""
        You are a Cognitive Bias Auditor specialized in probabilistic forecasting.
        Analyze the following reasoning trace for indicators of any of these biases:
        {", ".join(self.COGNITIVE_BIASES)}

        Reasoning:
        "{reasoning}"

        Return a JSON object with:
        - "detected_biases": [list of bias names]
        - "severity": [0-10]
        - "explanation": "Brief rationale"
        """
        response = await self.model.generate(prompt)
        return {"raw_audit": response.text}

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        return EvaluationResult(
            subject_id=subject_id,
            score=0.0,
            ground_truth=ground_truth,
            prediction=prediction,
            metadata={"type": "bias_audit"},
        )

__all__ = ["BiasInterceptor"]
