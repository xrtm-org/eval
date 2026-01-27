# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    subject_id: str
    score: float
    ground_truth: Any
    prediction: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReliabilityBin(BaseModel):
    bin_center: float
    mean_prediction: float
    mean_ground_truth: float
    count: int


class BrierDecomposition(BaseModel):
    reliability: float
    resolution: float
    uncertainty: float
    score: float


class Evaluator(Protocol):
    def score(self, prediction: Any, ground_truth: Any) -> float:
        ...

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        ...


class EvaluationReport(BaseModel):
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
