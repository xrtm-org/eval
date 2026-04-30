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

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from xrtm.data.core.schemas.forecast import ForecastOutput, MetadataBase

if importlib.util.find_spec("xrtm.data.corpora") is None:
    pytest.skip("xrtm.data.corpora is not available until the data corpus release lands", allow_module_level=True)

real_corpus = importlib.import_module("xrtm.data.corpora")
REAL_BINARY_CORPUS_ID = real_corpus.REAL_BINARY_CORPUS_ID
load_real_binary_corpus = real_corpus.load_real_binary_corpus
real_e2e = importlib.import_module("xrtm.eval.real_e2e")
coerce_forecast_outputs = real_e2e.coerce_forecast_outputs
evaluate_resolved_forecasts = real_e2e.evaluate_resolved_forecasts
load_forecast_output_records = real_e2e.load_forecast_output_records


def _synthetic_forecast_artifacts(limit: int = 4) -> list[dict]:
    artifacts = []
    for record in load_real_binary_corpus()[:limit]:
        probability = 0.9 if record.resolved_outcome else 0.1
        output = ForecastOutput(
            question_id=record.id,
            probability=probability,
            reasoning=f"Synthetic deterministic forecast for {record.id}",
            metadata=MetadataBase(
                snapshot_time=record.snapshot_time,
                tags=["real-question-e2e", "synthetic-fixture"],
                subject_type="binary",
                source_version=REAL_BINARY_CORPUS_ID,
            ),
        )
        artifacts.append(
            {
                "question_id": record.id,
                "output": output.model_dump(mode="json"),
                "provider_metadata": {"provider": "fixture"},
            }
        )
    return artifacts


def test_evaluate_resolved_forecast_artifacts_computes_brier_ece_and_calibration() -> None:
    report = evaluate_resolved_forecasts(_synthetic_forecast_artifacts(), num_bins=2)

    assert report.total_evaluations == 4
    assert report.mean_score == pytest.approx(0.01)
    assert report.summary_statistics["brier_score"] == pytest.approx(0.01)
    assert report.summary_statistics["ece"] == pytest.approx(0.1)
    assert report.summary_statistics["resolved_count"] == 4
    assert report.summary_statistics["skipped_count"] == 0
    assert report.reliability_bins is not None
    assert sum(bin.count for bin in report.reliability_bins) == 4
    assert {result.metadata["corpus_id"] for result in report.results} == {REAL_BINARY_CORPUS_ID}


def test_real_e2e_jsonl_artifact_loader_uses_canonical_forecast_output_schema() -> None:
    artifact_dir = Path(".cache/real-e2e-eval-tests")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "forecast-artifacts.jsonl"
    try:
        artifact_path.write_text(
            "\n".join(json.dumps(record) for record in _synthetic_forecast_artifacts(limit=2)) + "\n",
            encoding="utf-8",
        )

        outputs = load_forecast_output_records(artifact_path)
    finally:
        if artifact_path.exists():
            artifact_path.unlink()
        if artifact_dir.exists():
            artifact_dir.rmdir()

    assert [output.question_id for output in outputs] == [
        record.id for record in load_real_binary_corpus()[:2]
    ]
    assert all(isinstance(output, ForecastOutput) for output in outputs)
    assert coerce_forecast_outputs(outputs) == outputs
