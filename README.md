# xrtm-eval v0.3.0

[![PyPI](https://img.shields.io/pypi/v/xrtm-eval?style=flat-square)](https://pypi.org/project/xrtm-eval/)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**The Judge for XRTM.**

`xrtm-eval` is the scoring engine for probabilistic forecasts. It measures calibration, accuracy, and resolution using strict scoring rules.

## Installation

```bash
pip install xrtm-eval
```

## Scoring Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| **Brier Score** | `BrierScoreEvaluator` | Mean squared error `(p - o)^2`. Murphy decomposition into reliability/resolution/uncertainty. |
| **ECE** | `ExpectedCalibrationErrorEvaluator` | Expected Calibration Error via reliability binning. |
| **Log Score** | `LogScoreEvaluator` | Negative log-likelihood with epsilon clamping. |
| **Dashboard** | `summarize_binary_forecasts()` | Aggregate Brier, ECE, LogScore + calibration curve in one call. |

## Usage

```python
from xrtm.eval import BrierScoreEvaluator, summarize_binary_forecasts

# Single forecast


evaluator = BrierScoreEvaluator()
score = evaluator.score(probability=0.7, ground_truth="yes")

# Batch evaluation
predictions = [(0.7, "yes"), (0.3, "no"), (0.9, "yes")]
summary = summarize_binary_forecasts(predictions)
print(f"Brier: {summary['brier_score']:.4f}")
print(f"ECE:   {summary['ece']:.4f}")
```

## License

Apache 2.0
