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

r"""Unit tests for Beta NLL loss and evaluator."""


import pytest

from xrtm.eval.kit.metrics import (
    BetaDistributionEvaluator,
    BetaNLLLoss,
    beta_calibration_error,
    kl_divergence_beta,
)


class TestBetaNLLLoss:
    r"""Tests for BetaNLLLoss."""

    def test_clamp_target_low(self) -> None:
        r"""Values below eps_floor are clamped."""
        loss_fn = BetaNLLLoss()
        assert loss_fn.clamp_target(0.0) == 0.01
        assert loss_fn.clamp_target(-0.5) == 0.01

    def test_clamp_target_high(self) -> None:
        r"""Values above eps_ceil are clamped."""
        loss_fn = BetaNLLLoss()
        assert loss_fn.clamp_target(1.0) == 0.99
        assert loss_fn.clamp_target(1.5) == 0.99

    def test_clamp_target_middle(self) -> None:
        r"""Values in range are unchanged."""
        loss_fn = BetaNLLLoss()
        assert loss_fn.clamp_target(0.5) == 0.5
        assert loss_fn.clamp_target(0.7) == 0.7

    def test_compute_perfect_match(self) -> None:
        r"""Loss should be low when prediction matches target."""
        loss_fn = BetaNLLLoss()
        # Beta(7, 3) has mean 0.7
        loss = loss_fn.compute(pred_alpha=7.0, pred_beta=3.0, target_mean=0.7)
        # Compare against a mismatched prediction
        loss_bad = loss_fn.compute(pred_alpha=3.0, pred_beta=7.0, target_mean=0.7)
        assert loss < loss_bad

    def test_compute_batch_mean(self) -> None:
        r"""Batch compute should return mean loss."""
        loss_fn = BetaNLLLoss()
        alphas = [7.0, 7.0]
        betas = [3.0, 3.0]
        targets = [0.7, 0.7]
        batch_loss = loss_fn.compute_batch(alphas, betas, targets)
        single_loss = loss_fn.compute(7.0, 3.0, 0.7)
        assert batch_loss == pytest.approx(single_loss)

    def test_compute_batch_mismatched_lengths(self) -> None:
        r"""Should raise ValueError for mismatched lengths."""
        loss_fn = BetaNLLLoss()
        with pytest.raises(ValueError):
            loss_fn.compute_batch([7.0], [3.0, 3.0], [0.7])


class TestKLDivergence:
    r"""Tests for KL divergence between Beta distributions."""

    def test_kl_same_distribution_is_zero(self) -> None:
        r"""KL(P || P) = 0."""
        kl = kl_divergence_beta(7.0, 3.0, 7.0, 3.0)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_kl_is_non_negative(self) -> None:
        r"""KL divergence is always non-negative."""
        kl = kl_divergence_beta(7.0, 3.0, 3.0, 7.0)
        assert kl >= 0


class TestCalibrationError:
    r"""Tests for calibration error."""

    def test_perfect_calibration(self) -> None:
        r"""Perfect calibration should have low error."""
        # Predictions at 0.7 with 70% positive outcomes
        predictions = [(7.0, 3.0)] * 10
        outcomes = [1.0] * 7 + [0.0] * 3
        error = beta_calibration_error(predictions, outcomes, n_bins=5)
        assert error < 0.1

    def test_empty_predictions(self) -> None:
        r"""Empty predictions should return 0."""
        error = beta_calibration_error([], [])
        assert error == 0.0


class TestBetaDistributionEvaluator:
    r"""Tests for BetaDistributionEvaluator."""

    def test_score_method(self) -> None:
        r"""Score should return Beta NLL."""
        evaluator = BetaDistributionEvaluator()
        score = evaluator.score((7.0, 3.0), 0.7)
        assert isinstance(score, float)

    def test_evaluate_returns_dict(self) -> None:
        r"""Evaluate should return structured result."""
        evaluator = BetaDistributionEvaluator()
        result = evaluator.evaluate((7.0, 3.0), 1.0, "question_123")
        assert result["subject_id"] == "question_123"
        assert "score" in result
        assert result["prediction"]["mean"] == pytest.approx(0.7)
        assert "calibration_error" in result["metadata"]
