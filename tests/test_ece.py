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

from xrtm.eval.core.eval.definitions import EvaluationResult
from xrtm.eval.kit.eval.metrics import ExpectedCalibrationErrorEvaluator


def test_ece_basic():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=10)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth=1, prediction=0.9, metadata={}),  # Bin 9
        EvaluationResult(subject_id="2", score=0, ground_truth=0, prediction=0.1, metadata={}),  # Bin 1
    ]
    ece, bins = evaluator.compute_calibration_data(results)
    # Bin 9: 1 item, pred 0.9, gt 1. acc 1. mean_conf 0.9. abs(1 - 0.9) = 0.1
    # Bin 1: 1 item, pred 0.1, gt 0. acc 0. mean_conf 0.1. abs(0 - 0.1) = 0.1
    # ECE = (1/2)*0.1 + (1/2)*0.1 = 0.1
    assert abs(ece - 0.1) < 1e-6


def test_ece_mixed_types():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=2)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth="yes", prediction=0.8, metadata={}),
        EvaluationResult(subject_id="2", score=0, ground_truth="no", prediction="0.2", metadata={}),
        EvaluationResult(subject_id="3", score=0, ground_truth=True, prediction=0.9, metadata={}),
        EvaluationResult(subject_id="4", score=0, ground_truth=False, prediction=0.1, metadata={}),
    ]
    # Bin 0 (0-0.5): Items 2 (0.2), 4 (0.1).
    # Item 2: gt "no" -> 0.0. pred 0.2.
    # Item 4: gt False -> 0.0. pred 0.1.
    # Bin 0 mean_conf = (0.2 + 0.1)/2 = 0.15. mean_acc = 0.
    # Bin 1 (0.5-1.0): Items 1 (0.8), 3 (0.9).
    # Item 1: gt "yes" -> 1.0. pred 0.8.
    # Item 3: gt True -> 1.0. pred 0.9.
    # Bin 1 mean_conf = (0.8 + 0.9)/2 = 0.85. mean_acc = 1.0.

    # ECE = (2/4)*abs(0 - 0.15) + (2/4)*abs(1 - 0.85) = 0.5 * 0.15 + 0.5 * 0.15 = 0.075 + 0.075 = 0.15
    ece, bins = evaluator.compute_calibration_data(results)
    assert abs(ece - 0.15) < 1e-6


def test_ece_clamps_out_of_bounds_predictions_for_compatibility():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=10)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth=1, prediction=1.5, metadata={}),
        EvaluationResult(subject_id="2", score=0, ground_truth=0, prediction=-0.5, metadata={}),
        EvaluationResult(subject_id="3", score=0, ground_truth=1, prediction=0.5, metadata={}),
    ]

    ece, bins = evaluator.compute_calibration_data(results)
    assert ece == pytest.approx((0.0 + 0.0 + 0.5) / 3)
    assert sum(bin.count for bin in bins) == 3
    assert bins[0].count == 1
    assert bins[5].count == 1
    assert bins[5].mean_prediction == 0.5
    assert bins[9].count == 1


def test_ece_skips_nan_none_and_missing_outcomes():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=4)
    results = [
        EvaluationResult(subject_id="valid", score=0, ground_truth=0, prediction="0.25", metadata={}),
        EvaluationResult(subject_id="nan-pred", score=0, ground_truth=1, prediction=float("nan"), metadata={}),
        EvaluationResult(subject_id="inf-pred", score=0, ground_truth=1, prediction=float("inf"), metadata={}),
        EvaluationResult(subject_id="none-pred", score=0, ground_truth=1, prediction=None, metadata={}),
        EvaluationResult(subject_id="none-outcome", score=0, ground_truth=None, prediction=0.5, metadata={}),
        EvaluationResult(subject_id="nan-outcome", score=0, ground_truth=float("nan"), prediction=0.5, metadata={}),
        EvaluationResult(subject_id="unknown-outcome", score=0, ground_truth="unknown", prediction=0.5, metadata={}),
    ]

    ece, bins = evaluator.compute_calibration_data(results)

    assert ece == 0.25
    assert sum(bin.count for bin in bins) == 1
    assert bins[1].count == 1
    assert bins[1].mean_prediction == 0.25
    assert bins[1].mean_ground_truth == 0.0


def test_calibration_bin_boundaries_count_each_valid_prediction_once():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=10)
    probabilities = [0.0, 0.099999, 0.1, 0.999999, 1.0]
    results = [
        EvaluationResult(subject_id=str(i), score=0, ground_truth=1, prediction=probability, metadata={})
        for i, probability in enumerate(probabilities)
    ]

    ece, bins = evaluator.compute_calibration_data(results)

    assert 0.0 <= ece <= 1.0
    assert sum(bin.count for bin in bins) == len(probabilities)
    assert bins[0].count == 2
    assert bins[1].count == 1
    assert bins[-1].count == 2


@pytest.mark.parametrize("num_bins", [0, -1, 1.5, True])
def test_ece_rejects_invalid_num_bins(num_bins):
    with pytest.raises(ValueError):
        ExpectedCalibrationErrorEvaluator(num_bins=num_bins)


if __name__ == "__main__":
    test_ece_basic()
    test_ece_mixed_types()
    test_ece_clamps_out_of_bounds_predictions_for_compatibility()
    print("All tests passed!")
