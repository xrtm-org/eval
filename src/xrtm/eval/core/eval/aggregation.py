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

r"""Aggregation utilities: inverse-variance weighting and outlier filtering.

Reference implementations used by consensus pipelines and robustness checks.
They operate on plain floats so any package can use them without importing
forecast schemas.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

__all__ = [
    "inverse_variance_weighting",
    "inverse_variance_weighting_with_variance",
    "robustness_check_mad",
]

# MAD -> sigma for normally distributed data.
_MAD_SCALE = 1.4826


def _weights(variances: Optional[Sequence[float]], count: int) -> List[float]:
    r"""Unit weights when variances are absent; ``1 / variance`` otherwise."""
    if variances is None:
        return [1.0] * count
    weights: List[float] = []
    for variance in variances:
        try:
            value = float(variance)
        except (TypeError, ValueError):
            value = 0.0
        weights.append(1.0 / value if value > 0 else 1.0)
    if len(weights) < count:
        weights.extend([1.0] * (count - len(weights)))
    return weights[:count]


def inverse_variance_weighting(predictions: Sequence[float], variances: Optional[Sequence[float]] = None) -> float:
    r"""Inverse-variance weighted mean of *predictions*.

    Args:
        predictions: Probability estimates.
        variances: Optional per-prediction variances; missing or non-positive
            values receive unit weight.

    Returns:
        The weighted mean (``0.5`` when *predictions* is empty).

    Example:
        >>> inverse_variance_weighting([0.2, 0.8], [0.01, 0.25])
        0.2230769230769231
    """
    values = [float(value) for value in predictions]
    if not values:
        return 0.5
    weights = _weights(variances, len(values))
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def inverse_variance_weighting_with_variance(
    predictions: Sequence[float], variances: Optional[Sequence[float]] = None
) -> tuple[float, float]:
    r"""Return ``(weighted_mean, effective_variance)``.

    The effective variance is ``1 / sum(weights)`` — the variance of the
    weighted mean, which shrinks as more (or more precise) inputs agree.
    """
    values = [float(value) for value in predictions]
    if not values:
        return 0.5, 1.0
    weights = _weights(variances, len(values))
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values), 1.0
    mean = sum(value * weight for value, weight in zip(values, weights)) / total
    return mean, 1.0 / total


def robustness_check_mad(values: Sequence[float], threshold: float = 2.0) -> List[float]:
    r"""Drop values further than ``threshold`` scaled MADs from the median.

    A zero median absolute deviation (all values identical) leaves the input
    unchanged.
    """
    numbers = [float(value) for value in values]
    if not numbers:
        return []
    median = sorted(numbers)[len(numbers) // 2]
    deviations = sorted(abs(value - median) for value in numbers)
    mad = deviations[len(deviations) // 2]
    if mad <= 0:
        return list(numbers)
    limit = threshold * _MAD_SCALE * mad
    return [value for value in numbers if abs(value - median) <= limit]
