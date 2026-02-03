# xrtm-eval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/xrtm-eval.svg)](https://pypi.org/project/xrtm-eval/)

**The Judge for XRTM.**

`xrtm-eval` is the rigorous scoring engine used to grade probabilistic forecasts. It operates independently of the inference engine to ensure objective evaluation.

## Part of the XRTM Ecosystem

```
Layer 4: xrtm-train    → (imports all)
Layer 3: xrtm-forecast → (imports eval, data)
Layer 2: xrtm-eval     → (imports data) ← YOU ARE HERE
Layer 1: xrtm-data     → (zero dependencies)
```

`xrtm-eval` provides scoring metrics AND trust primitives used by the forecast engine.

## Installation

```bash
pip install xrtm-eval
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

### 3. Epistemic Trust Primitives (v0.1.1+)
`xrtm-eval` now includes trust scoring infrastructure:

```python
from xrtm.eval.core.epistemics import IntegrityGuardian, SourceTrustRegistry

registry = SourceTrustRegistry()
guardian = IntegrityGuardian(registry)
```

## Project Structure

```
src/xrtm/eval/
├── core/            # Interfaces & Schemas
│   ├── eval/            # Evaluator protocol, EvaluationResult
│   ├── epistemics.py    # Trust primitives (SourceTrustRegistry)
│   └── schemas/         # ForecastResolution
├── kit/             # Composable evaluator implementations
│   └── eval/metrics.py  # BrierScoreEvaluator, ECE
└── providers/       # External evaluation services (future)
```

## Development

Prerequisites:
- [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```
