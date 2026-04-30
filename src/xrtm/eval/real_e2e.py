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

r"""Evaluation helpers for deterministic real-question forecast artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from xrtm.data.core.schemas.forecast import ForecastOutput
from xrtm.data.corpora import REAL_BINARY_CORPUS_ID, load_real_binary_resolved_outcomes

from xrtm.eval.core.eval.definitions import EvaluationReport
from xrtm.eval.kit.eval.metrics import BrierScoreEvaluator, ExpectedCalibrationErrorEvaluator


class ForecastArtifactError(ValueError):
    r"""Raised when a real-question forecast artifact cannot be parsed."""


def load_forecast_output_records(path: str | Path) -> list[ForecastOutput]:
    r"""Load ``ForecastOutput`` records from a forecast real-E2E JSONL artifact."""
    artifact_path = Path(path)
    outputs: list[ForecastOutput] = []
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ForecastArtifactError(f"{artifact_path}:{line_number}: invalid JSON") from exc
            try:
                outputs.append(coerce_forecast_output(raw_record))
            except ForecastArtifactError as exc:
                raise ForecastArtifactError(f"{artifact_path}:{line_number}: {exc}") from exc
    return outputs


def coerce_forecast_output(record: Any) -> ForecastOutput:
    r"""Coerce a harness wrapper or raw mapping into the canonical ``ForecastOutput`` schema."""
    if isinstance(record, ForecastOutput):
        return record

    if hasattr(record, "output"):
        output = getattr(record, "output")
        if isinstance(output, ForecastOutput):
            return output
        record = output
    elif isinstance(record, Mapping) and "output" in record:
        record = record["output"]

    if isinstance(record, Mapping):
        try:
            return ForecastOutput.model_validate(record)
        except Exception as exc:
            raise ForecastArtifactError(f"invalid ForecastOutput payload: {exc}") from exc

    raise ForecastArtifactError(f"expected ForecastOutput or mapping, got {type(record).__name__}")


def coerce_forecast_outputs(records: Iterable[Any]) -> list[ForecastOutput]:
    r"""Coerce an iterable of real-E2E artifacts into canonical forecast outputs."""
    return [coerce_forecast_output(record) for record in records]


def evaluate_resolved_forecasts(
    records: Iterable[Any],
    *,
    outcomes: Mapping[str, bool] | None = None,
    num_bins: int = 10,
) -> EvaluationReport:
    r"""Compute Brier, ECE, and calibration summaries for resolved real questions."""
    resolved_outcomes = load_real_binary_resolved_outcomes() if outcomes is None else outcomes
    brier = BrierScoreEvaluator()
    outputs = coerce_forecast_outputs(records)
    skipped_count = 0
    results = []

    for output in outputs:
        outcome = resolved_outcomes.get(output.question_id)
        if outcome is None:
            skipped_count += 1
            continue

        result = brier.evaluate(output.probability, outcome, output.question_id)
        result.metadata.update(
            {
                "corpus_id": output.metadata.source_version or REAL_BINARY_CORPUS_ID,
                "forecast_probability": output.probability,
                "snapshot_time": output.metadata.snapshot_time.isoformat(),
            }
        )
        if output.metadata.tags:
            result.metadata["tags"] = list(output.metadata.tags)
        results.append(result)

    total = len(results)
    mean_brier = sum(result.score for result in results) / total if total else 0.0
    ece, reliability_bins = ExpectedCalibrationErrorEvaluator(num_bins=num_bins).compute_calibration_data(results)
    decomposition = brier.compute_decomposition(results, num_bins=num_bins)

    return EvaluationReport(
        metric_name="Real Binary Forecast Brier Score",
        mean_score=mean_brier,
        total_evaluations=total,
        results=results,
        reliability_bins=reliability_bins,
        summary_statistics={
            "brier_score": mean_brier,
            "ece": ece,
            "reliability": decomposition.reliability,
            "resolution": decomposition.resolution,
            "uncertainty": decomposition.uncertainty,
            "decomposed_brier_score": decomposition.score,
            "resolved_count": total,
            "skipped_count": skipped_count,
        },
    )


__all__ = [
    "ForecastArtifactError",
    "coerce_forecast_output",
    "coerce_forecast_outputs",
    "evaluate_resolved_forecasts",
    "load_forecast_output_records",
]
