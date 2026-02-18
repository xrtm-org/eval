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

r"""
Core evaluation definitions and data structures.

Provides the foundational schemas for the xRTM evaluation pipeline,
including result containers, Brier decomposition, reliability binning,
and the ``Evaluator`` protocol that all scoring backends must implement.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    r"""A single evaluation result pairing a prediction against ground truth."""
    subject_id: str
    score: float
    ground_truth: Any
    prediction: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReliabilityBin(BaseModel):
    r"""One bin in a reliability diagram, used for calibration analysis."""
    bin_center: float
    mean_prediction: float
    mean_ground_truth: float
    count: int


class BrierDecomposition(BaseModel):
    r"""Murphy decomposition of the Brier score into reliability, resolution, and uncertainty."""
    reliability: float
    resolution: float
    uncertainty: float
    score: float


class Evaluator(Protocol):
    r"""Protocol that all scoring backends must implement."""
    def score(self, prediction: Any, ground_truth: Any) -> float: ...

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult: ...


class EvaluationReport(BaseModel):
    r"""Aggregate evaluation report containing results, statistics, and optional reliability bins."""
    metric_name: str
    mean_score: float
    total_evaluations: int
    results: List[EvaluationResult] = Field(default_factory=list)
    summary_statistics: Dict[str, float] = Field(default_factory=dict)
    reliability_bins: Optional[List[ReliabilityBin]] = None
    slices: Optional[Dict[str, "EvaluationReport"]] = Field(
        default=None, description="Sub-reports grouped by metadata tags"
    )

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def to_pandas(self) -> Any:
        try:
            import pandas as pd

            return pd.DataFrame([r.model_dump() for r in self.results])
        except ImportError:
            raise ImportError("Pandas is required for to_pandas(). Install it with `pip install pandas`.")


__all__ = ["EvaluationResult", "Evaluator", "EvaluationReport", "ReliabilityBin", "BrierDecomposition"]
