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

r"""Tests for aggregation utilities and slice analytics."""

import pytest

from xrtm.eval.core.eval.aggregation import (
    inverse_variance_weighting,
    inverse_variance_weighting_with_variance,
    robustness_check_mad,
)
from xrtm.eval.core.eval.analytics import SliceAnalytics
from xrtm.eval.kit.eval.aggregation import inverse_variance_weighting as kit_ivw


def test_ivw_without_variances_is_the_mean():
    assert inverse_variance_weighting([0.2, 0.8]) == pytest.approx(0.5)
    assert inverse_variance_weighting([]) == 0.5


def test_ivw_weights_low_variance_more():
    # weights 100 vs 4 → the precise (0.2) estimate dominates.
    mean = inverse_variance_weighting([0.2, 0.8], [0.01, 0.25])
    assert mean == pytest.approx((0.2 * 100 + 0.8 * 4) / 104)
    # missing/invalid variances fall back to unit weight
    assert inverse_variance_weighting([0.2, 0.8], [None, 0]) == pytest.approx(0.5)


def test_ivw_with_variance_shrinks_when_inputs_agree():
    mean, variance = inverse_variance_weighting_with_variance([0.5, 0.5], [0.04, 0.04])
    assert mean == pytest.approx(0.5)
    assert variance == pytest.approx(0.02)  # 1 / (25 + 25)


def test_kit_path_reexports_core_implementation():
    assert kit_ivw is inverse_variance_weighting


def test_mad_filter_drops_outliers():
    kept = robustness_check_mad([0.50, 0.51, 0.49, 0.52, 0.99])
    assert 0.99 not in kept
    assert len(kept) == 4
    assert robustness_check_mad([]) == []


def test_mad_filter_zero_spread_keeps_all():
    assert robustness_check_mad([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5]


def test_compute_slices_buckets_by_prediction():
    class Result:
        def __init__(self, prediction, ground_truth, score):
            self.prediction = prediction
            self.ground_truth = ground_truth
            self.score = score

    results = [
        Result(0.05, 0.0, 0.0025),
        Result(0.15, 1.0, 0.7225),
        Result(0.95, 1.0, 0.0025),
        Result(None, 1.0, 1.0),  # unusable prediction: skipped
    ]

    slices = SliceAnalytics.compute_slices(results)

    assert set(slices) == {"0.0-0.1", "0.1-0.2", "0.9-1.0"}
    assert slices["0.1-0.2"]["count"] == 1
    assert slices["0.1-0.2"]["mean_outcome"] == 1.0
    assert slices["0.9-1.0"]["mean_score"] == pytest.approx(0.0025)
    assert SliceAnalytics.compute_slices([]) == {}


def test_compute_slices_rejects_bad_slice_count():
    with pytest.raises(ValueError):
        SliceAnalytics.compute_slices([], num_slices=0)
