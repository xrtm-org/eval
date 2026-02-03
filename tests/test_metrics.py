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

from xrtm.eval import BrierScoreEvaluator


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
