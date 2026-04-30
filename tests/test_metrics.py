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


@pytest.mark.parametrize("prediction", [i / 10 for i in range(11)])
def test_brier_score_probability_properties(prediction):
    evaluator = BrierScoreEvaluator()

    positive_score = evaluator.score(prediction=prediction, ground_truth=1)
    negative_score = evaluator.score(prediction=prediction, ground_truth=0)

    assert 0.0 <= positive_score <= 1.0
    assert 0.0 <= negative_score <= 1.0
    assert positive_score == pytest.approx((prediction - 1.0) ** 2)
    assert positive_score == pytest.approx(evaluator.score(prediction=1.0 - prediction, ground_truth=0))


def test_string_ground_truth_handling():
    """Verify string handling (Resolution logic)."""
    evaluator = BrierScoreEvaluator()
    score = evaluator.score(prediction=0.9, ground_truth="Yes")
    assert score == (0.9 - 1.0) ** 2

    score = evaluator.score(prediction=0.1, ground_truth="No")
    assert score == (0.1 - 0.0) ** 2


@pytest.mark.parametrize(
    ("ground_truth", "normalized"),
    [
        (True, 1.0),
        (False, 0.0),
        (1, 1.0),
        (0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (" true ", 1.0),
        ("FALSE", 0.0),
        ("won", 1.0),
        ("lost", 0.0),
    ],
)
def test_brier_score_normalizes_binary_outcomes(ground_truth, normalized):
    evaluator = BrierScoreEvaluator()

    assert evaluator.score(prediction=normalized, ground_truth=ground_truth) == 0.0


@pytest.mark.parametrize("prediction", [None, "invalid", float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_brier_score_rejects_invalid_predictions(prediction):
    evaluator = BrierScoreEvaluator()

    with pytest.raises(ValueError):
        evaluator.score(prediction=prediction, ground_truth=1)


def test_brier_evaluate_rejects_invalid_inputs():
    evaluator = BrierScoreEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(prediction=float("nan"), ground_truth=1, subject_id="invalid-prediction")
    with pytest.raises(ValueError):
        evaluator.evaluate(prediction=0.5, ground_truth="unknown", subject_id="invalid-outcome")


@pytest.mark.parametrize("ground_truth", [None, "", "unknown", float("nan"), float("inf"), -1, 2])
def test_brier_score_rejects_missing_or_invalid_outcomes(ground_truth):
    evaluator = BrierScoreEvaluator()

    with pytest.raises(ValueError):
        evaluator.score(prediction=0.5, ground_truth=ground_truth)


def test_brier_decomposition_skips_invalid_predictions_consistently():
    evaluator = BrierScoreEvaluator()
    results = [
        EvaluationResult(subject_id="a", prediction=0.8, ground_truth=1, score=0.04),
        EvaluationResult(subject_id="b", prediction=0.2, ground_truth=0, score=0.04),
        EvaluationResult(subject_id="c", prediction="invalid", ground_truth=1, score=1.0),
        EvaluationResult(subject_id="d", prediction=float("nan"), ground_truth=1, score=1.0),
        EvaluationResult(subject_id="e", prediction=0.4, ground_truth=None, score=1.0),
    ]

    decomp = evaluator.compute_decomposition(results, num_bins=2)

    assert decomp.score == pytest.approx(decomp.reliability - decomp.resolution + decomp.uncertainty)
    assert decomp.uncertainty == pytest.approx(0.25)


def test_brier_decomposition_all_invalid_returns_zero():
    evaluator = BrierScoreEvaluator()
    results = [
        EvaluationResult(subject_id="a", prediction="invalid", ground_truth=1, score=1.0),
        EvaluationResult(subject_id="b", prediction=None, ground_truth=0, score=1.0),
        EvaluationResult(subject_id="c", prediction=0.5, ground_truth=None, score=1.0),
        EvaluationResult(subject_id="d", prediction=0.5, ground_truth=float("nan"), score=1.0),
    ]

    decomp = evaluator.compute_decomposition(results)
    ece, bins = ExpectedCalibrationErrorEvaluator().compute_calibration_data(results)

    assert decomp.score == 0.0
    assert ece == 0.0
    assert sum(bin.count for bin in bins) == 0
