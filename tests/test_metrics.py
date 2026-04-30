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

import pytest

from xrtm.eval import BrierScoreEvaluator, ExpectedCalibrationErrorEvaluator
from xrtm.eval.core.eval.definitions import EvaluationResult


def test_brier_score_perfect_accurate():
    """Verify Brier score is 0.0 for perfect prediction."""
    evaluator = BrierScoreEvaluator()
    score = evaluator.score(prediction=1.0, ground_truth=1)
    assert score == 0.0

    score = evaluator.score(prediction=0.0, ground_truth=0)
    assert score == 0.0


def test_brier_score_worst_case():
    """Verify Brier score is 1.0 for completely wrong prediction."""
    evaluator = BrierScoreEvaluator()
    score = evaluator.score(prediction=1.0, ground_truth=0)
    assert score == 1.0

    score = evaluator.score(prediction=0.0, ground_truth=1)
    assert score == 1.0


def test_brier_score_uncertainty():
    """Verify Brier score for 0.5 prediction."""
    evaluator = BrierScoreEvaluator()
    score = evaluator.score(prediction=0.5, ground_truth=1)
    assert score == 0.25  # (0.5 - 1.0)^2 = 0.25


def test_string_ground_truth_handling():
    """Verify string handling (Resolution logic)."""
    evaluator = BrierScoreEvaluator()
    score = evaluator.score(prediction=0.9, ground_truth="Yes")
    assert score == (0.9 - 1.0) ** 2

    score = evaluator.score(prediction=0.1, ground_truth="No")
    assert score == (0.1 - 0.0) ** 2


def test_brier_decomposition_skips_invalid_predictions_consistently():
    evaluator = BrierScoreEvaluator()
    results = [
        EvaluationResult(subject_id="a", prediction=0.8, ground_truth=1, score=0.04),
        EvaluationResult(subject_id="b", prediction=0.2, ground_truth=0, score=0.04),
        EvaluationResult(subject_id="c", prediction="invalid", ground_truth=1, score=1.0),
    ]

    decomp = evaluator.compute_decomposition(results, num_bins=2)

    assert decomp.score == pytest.approx(decomp.reliability - decomp.resolution + decomp.uncertainty)
    assert decomp.uncertainty == pytest.approx(0.25)


def test_brier_decomposition_all_invalid_returns_zero():
    evaluator = BrierScoreEvaluator()
    results = [
        EvaluationResult(subject_id="a", prediction="invalid", ground_truth=1, score=1.0),
        EvaluationResult(subject_id="b", prediction=None, ground_truth=0, score=1.0),
    ]

    decomp = evaluator.compute_decomposition(results)
    ece, bins = ExpectedCalibrationErrorEvaluator().compute_calibration_data(results)

    assert decomp.score == 0.0
    assert ece == 0.0
    assert sum(bin.count for bin in bins) == 0
