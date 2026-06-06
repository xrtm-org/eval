"""Stub: aggregation deferred during simplification."""
from __future__ import annotations
from typing import Any


def inverse_variance_weighting(predictions: list[float], variances: list[float] | None = None) -> float:
    if not predictions:
        return 0.5
    return sum(predictions) / len(predictions)


def robustness_check_mad(values: list[float], threshold: float = 2.0) -> list[float]:
    return values
