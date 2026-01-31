# coding=utf-8
from typing import Any

import pytest

from xrtm.eval.kit.eval.bias import BiasInterceptor


class MockResponse:
    def __init__(self, text: str):
        self.text = text

class MockModel:
    def __init__(self):
        self.last_prompt = None

    async def generate(self, prompt: str) -> Any:
        self.last_prompt = prompt
        return MockResponse("{}")

@pytest.mark.asyncio
async def test_evaluate_reasoning():
    model = MockModel()
    interceptor = BiasInterceptor(model)
    reasoning = "I think it will rain because it rained yesterday."

    result = await interceptor.evaluate_reasoning(reasoning)

    assert result == {"raw_audit": "{}"}

    # Check that the prompt contains the reasoning and some biases
    assert reasoning in model.last_prompt
    assert "Base-Rate Neglect" in model.last_prompt
    assert "Cognitive Bias Auditor" in model.last_prompt
    assert "Reasoning:" in model.last_prompt
