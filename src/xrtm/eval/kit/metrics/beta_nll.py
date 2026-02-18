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
Beta distribution loss functions for training.

This module provides loss functions for training models that output
Beta distribution parameters. Implements Decision 6 (target relaxation)
for numerical stability with BFloat16 precision.

Example:
    >>> from xrtm.eval.kit.metrics import BetaNLLLoss
    >>> loss_fn = BetaNLLLoss()
    >>> loss = loss_fn.compute(pred_alpha=7.0, pred_beta=3.0, target_mean=0.8)
"""

import math
from typing import Tuple

from pydantic import BaseModel, Field


class BetaNLLLoss(BaseModel):
    r"""
    Negative log-likelihood loss for Beta distribution outputs.

    Decision 6 Implementation: Handles numerical stability for extreme
    targets by clamping to [eps_floor, eps_ceil]. This prevents NaN
    gradients when training with BFloat16 precision.

    Attributes:
        eps_floor: Minimum allowed target value. Defaults to 0.01.
        eps_ceil: Maximum allowed target value. Defaults to 0.99.

    Example:
        >>> loss_fn = BetaNLLLoss()
        >>> loss = loss_fn.compute(pred_alpha=7.0, pred_beta=3.0, target_mean=0.8)
        >>> print(f"Loss: {loss:.4f}")
    """

    eps_floor: float = Field(
        default=0.01,
        ge=0.0,
        lt=0.5,
        description="Minimum allowed target value for clamping",
    )
    eps_ceil: float = Field(
        default=0.99,
        gt=0.5,
        le=1.0,
        description="Maximum allowed target value for clamping",
    )

    def clamp_target(self, y: float) -> float:
        r"""
        Clamp target to [eps_floor, eps_ceil] for numerical stability.

        Decision 6: This prevents log(0) and ensures stable gradients
        when training with reduced precision (BFloat16, FP16).

        Args:
            y: Raw target value in [0, 1].

        Returns:
            Clamped value in [eps_floor, eps_ceil].

        Example:
            >>> loss_fn = BetaNLLLoss()
            >>> loss_fn.clamp_target(0.0)
            0.01
            >>> loss_fn.clamp_target(1.0)
            0.99
        """
        return max(self.eps_floor, min(self.eps_ceil, y))

    def compute(
        self,
        pred_alpha: float,
        pred_beta: float,
        target_mean: float,
    ) -> float:
        r"""
        Compute Beta NLL loss: -log(Beta(y | α, β)).

        The Beta PDF is:
            p(y | α, β) = y^(α-1) * (1-y)^(β-1) / B(α, β)

        Taking negative log:
            NLL = -(α-1)log(y) - (β-1)log(1-y) + log(B(α, β))

        Where B(α, β) = Γ(α)Γ(β) / Γ(α+β)

        Args:
            pred_alpha: Predicted α parameter (must be > 0).
            pred_beta: Predicted β parameter (must be > 0).
            target_mean: Target mean value (will be clamped).

        Returns:
            Negative log-likelihood loss value.

        Example:
            >>> loss_fn = BetaNLLLoss()
            >>> # Perfect prediction: α=7, β=3 has mean=0.7
            >>> loss = loss_fn.compute(7.0, 3.0, 0.7)
        """
        y = self.clamp_target(target_mean)
        alpha = max(0.1, pred_alpha)
        beta = max(0.1, pred_beta)

        # Log probability: (α-1)log(y) + (β-1)log(1-y) - log(B(α,β))
        # log(B(α,β)) = lgamma(α) + lgamma(β) - lgamma(α+β)
        log_beta_fn = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

        log_prob = (alpha - 1) * math.log(y) + (beta - 1) * math.log(1 - y) - log_beta_fn

        return -log_prob

    def compute_batch(
        self,
        pred_alphas: list[float],
        pred_betas: list[float],
        target_means: list[float],
    ) -> float:
        r"""
        Compute mean Beta NLL loss over a batch.

        Args:
            pred_alphas: List of predicted α parameters.
            pred_betas: List of predicted β parameters.
            target_means: List of target mean values.

        Returns:
            Mean negative log-likelihood loss.

        Raises:
            ValueError: If input lists have different lengths.
        """
        if not (len(pred_alphas) == len(pred_betas) == len(target_means)):
            msg = "All input lists must have the same length"
            raise ValueError(msg)

        if not pred_alphas:
            return 0.0

        total_loss = sum(self.compute(a, b, t) for a, b, t in zip(pred_alphas, pred_betas, target_means, strict=True))
        return total_loss / len(pred_alphas)


def kl_divergence_beta(
    p_alpha: float,
    p_beta: float,
    q_alpha: float,
    q_beta: float,
) -> float:
    r"""
    Compute KL divergence between two Beta distributions.

    KL(P || Q) where P = Beta(p_α, p_β) and Q = Beta(q_α, q_β).

    Args:
        p_alpha: α parameter of distribution P.
        p_beta: β parameter of distribution P.
        q_alpha: α parameter of distribution Q.
        q_beta: β parameter of distribution Q.

    Returns:
        KL divergence value (non-negative).

    Example:
        >>> kl = kl_divergence_beta(7.0, 3.0, 8.0, 2.0)
    """
    # KL(Beta(a1,b1) || Beta(a2,b2)) =
    #   log(B(a2,b2)/B(a1,b1))
    #   + (a1-a2)*ψ(a1) + (b1-b2)*ψ(b1)
    #   + (a2-a1+b2-b1)*ψ(a1+b1)
    # where ψ = digamma function

    # Ensure positive values
    p_alpha = max(0.1, p_alpha)
    p_beta = max(0.1, p_beta)
    q_alpha = max(0.1, q_alpha)
    q_beta = max(0.1, q_beta)

    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def digamma(x: float) -> float:
        r"""Approximate digamma function using asymptotic expansion."""
        # For small x, use recurrence relation
        result = 0.0
        while x < 6:
            result -= 1 / x
            x += 1
        # Asymptotic expansion
        result += math.log(x) - 1 / (2 * x)
        x2 = x * x
        result -= 1 / (12 * x2)
        result += 1 / (120 * x2 * x2)
        return result

    log_ratio = log_beta(q_alpha, q_beta) - log_beta(p_alpha, p_beta)
    psi_p_alpha = digamma(p_alpha)
    psi_p_beta = digamma(p_beta)
    psi_p_sum = digamma(p_alpha + p_beta)

    kl = (
        log_ratio
        + (p_alpha - q_alpha) * psi_p_alpha
        + (p_beta - q_beta) * psi_p_beta
        + (q_alpha - p_alpha + q_beta - p_beta) * psi_p_sum
    )

    return max(0.0, kl)


def beta_calibration_error(
    predictions: list[Tuple[float, float]],
    outcomes: list[float],
    n_bins: int = 10,
) -> float:
    r"""
    Compute calibration error for Beta distribution predictions.

    Similar to Expected Calibration Error (ECE) but for Beta outputs.
    Groups predictions by their mean and compares to actual outcomes.

    Args:
        predictions: List of (α, β) tuples.
        outcomes: List of binary outcomes (0 or 1).
        n_bins: Number of bins for grouping. Defaults to 10.

    Returns:
        Weighted average calibration error.
    """
    if not predictions:
        return 0.0

    # Compute means from predictions
    means = [p[0] / (p[0] + p[1]) for p in predictions]

    # Create bins
    bins: list[list[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for mean, outcome in zip(means, outcomes, strict=True):
        bin_idx = min(int(mean * n_bins), n_bins - 1)
        bins[bin_idx].append((mean, outcome))

    # Compute calibration error per bin
    total_error = 0.0
    total_count = 0
    for bin_data in bins:
        if not bin_data:
            continue
        avg_pred = sum(m for m, _ in bin_data) / len(bin_data)
        avg_outcome = sum(o for _, o in bin_data) / len(bin_data)
        total_error += len(bin_data) * abs(avg_pred - avg_outcome)
        total_count += len(bin_data)

    return total_error / total_count if total_count > 0 else 0.0


class BetaDistributionEvaluator:
    r"""
    Evaluator for predictions expressed as Beta distributions.

    Implements the Evaluator protocol for Beta distribution outputs,
    computing KL divergence from predicted distribution to point mass
    at the ground truth.

    Attributes:
        name: Human-readable name for this evaluator.

    Example:
        >>> evaluator = BetaDistributionEvaluator()
        >>> result = evaluator.evaluate(
        ...     prediction=(7.0, 3.0),  # Beta(7, 3) with mean 0.7
        ...     ground_truth=1.0,        # Resolved as Yes
        ...     subject_id="question_123",
        ... )
    """

    def __init__(self, name: str = "Beta KL Divergence") -> None:
        r"""
        Initialize the Beta distribution evaluator.

        Args:
            name: Human-readable name for this evaluator.
        """
        self.name = name
        self._loss_fn = BetaNLLLoss()

    def score(
        self,
        prediction: Tuple[float, float],
        ground_truth: float,
    ) -> float:
        r"""
        Compute score for a single prediction.

        Uses negative Beta NLL as the score (lower is better).

        Args:
            prediction: Tuple of (α, β) parameters.
            ground_truth: True outcome in [0, 1].

        Returns:
            Beta NLL score (lower is better).
        """
        alpha, beta = prediction
        return self._loss_fn.compute(alpha, beta, ground_truth)

    def evaluate(
        self,
        prediction: Tuple[float, float],
        ground_truth: float,
        subject_id: str,
    ) -> dict:
        r"""
        Evaluate a single prediction and return detailed result.

        Args:
            prediction: Tuple of (α, β) parameters.
            ground_truth: True outcome in [0, 1].
            subject_id: Identifier for the subject/question.

        Returns:
            Dict with evaluation result including score and metadata.
        """
        alpha, beta = prediction
        score = self.score(prediction, ground_truth)
        pred_mean = alpha / (alpha + beta)

        return {
            "subject_id": subject_id,
            "score": score,
            "ground_truth": ground_truth,
            "prediction": {"alpha": alpha, "beta": beta, "mean": pred_mean},
            "metadata": {
                "pred_mean": pred_mean,
                "concentration": alpha + beta,
                "calibration_error": abs(pred_mean - ground_truth),
            },
        }


__all__ = [
    "BetaNLLLoss",
    "kl_divergence_beta",
    "beta_calibration_error",
    "BetaDistributionEvaluator",
]
