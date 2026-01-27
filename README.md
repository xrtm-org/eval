# xrtm-eval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**The Judge for XRTM.**

`xrtm-eval` is the rigorous scoring engine used to grade probabilistic forecasts. It operates independently of the inference engine to ensure objective evaluation.

## Installation

```bash
uv pip install xrtm-eval
```

## Core Primitives

### 1. Brier Score Breakdown
We do not use simple accuracy. We use the **Brier Score**, decomposed into its three component terms:

*   **Reliability**: How well do the predicted probabilities match observed frequencies?
*   **Resolution**: How well does the forecast distinguish between events that happen and those that don't?
*   **Uncertainty**: The inherent difficulty of the problem.

```python
from xrtm.eval import BrierScoreEvaluator

evaluator = BrierScoreEvaluator()
score = evaluator.score(prediction=0.7, ground_truth=1)
# score = (0.7 - 1.0)^2 = 0.09
```

### 2. Expected Calibration Error (ECE)
Use the `ExpectedCalibrationErrorEvaluator` to measure the gap between confidence and accuracy across bin buckets.

## Development

Prerequisites:
- [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```
