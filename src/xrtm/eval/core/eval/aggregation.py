# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

from typing import List, Tuple

# From xrtm-data
from xrtm.data.schemas.forecast import ForecastOutput


def inverse_variance_weighting(
    predictions: List[ForecastOutput], default_variance: float = 0.05
) -> Tuple[float, float]:
    if not predictions:
        return 0.5, 1.0

    values = []
    weights = []

    for p in predictions:
        val = p.confidence
        if p.uncertainty is not None:
            variance = p.uncertainty
        else:
            dist = abs(val - 0.5) * 2
            variance = 0.25 * (1.0 - dist) + 0.01

        weight = 1.0 / max(variance, 0.01)
        values.append(val)
        weights.append(weight)

    sum_weights = sum(weights)
    if sum_weights == 0:
        return 0.5, 1.0

    weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum_weights
    combined_variance = 1.0 / sum_weights

    return weighted_mean, combined_variance


def robustness_check_mad(values: List[float], threat_level: float = 2.0) -> List[float]:
    if len(values) < 3:
        return values

    median = sorted(values)[len(values) // 2]
    deviations = [abs(x - median) for x in values]
    mad = sorted(deviations)[len(deviations) // 2]

    if mad == 0:
        return values

    return [x for x in values if abs(x - median) <= threat_level * mad]
