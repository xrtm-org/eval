#!/usr/bin/env python3
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
Example: Mock Evaluation of Beta Distribution Predictions.

This example demonstrates how to:
1. Create mock predictions (Beta distribution parameters)
2. Evaluate predictions against ground truth outcomes
3. Compute calibration metrics

Run:
    python examples/mock_beta_evaluation.py
"""

from xrtm.eval.kit.metrics import (
    BetaDistributionEvaluator,
    BetaNLLLoss,
    beta_calibration_error,
    kl_divergence_beta,
)


def create_mock_predictions() -> list[dict]:
    r"""Create mock prediction-outcome pairs for evaluation."""
    # Simulate predictions for resolved questions
    return [
        # High confidence correct (predicted ~0.8, outcome YES)
        {"question_id": "q1", "alpha": 8.0, "beta": 2.0, "outcome": 1.0},
        # Low confidence correct (predicted ~0.3, outcome NO)
        {"question_id": "q2", "alpha": 3.0, "beta": 7.0, "outcome": 0.0},
        # High confidence wrong (predicted ~0.9, outcome NO) - penalized heavily
        {"question_id": "q3", "alpha": 9.0, "beta": 1.0, "outcome": 0.0},
        # Uncertain correct (predicted ~0.5, outcome YES)
        {"question_id": "q4", "alpha": 5.0, "beta": 5.0, "outcome": 1.0},
        # Good prediction (predicted ~0.7, outcome YES)
        {"question_id": "q5", "alpha": 7.0, "beta": 3.0, "outcome": 1.0},
    ]


def main() -> None:
    r"""Demonstrate mock Beta distribution evaluation."""
    print("=" * 60)
    print("Mock Evaluation of Beta Distribution Predictions")
    print("=" * 60)

    # Initialize evaluators
    loss_fn = BetaNLLLoss()
    evaluator = BetaDistributionEvaluator()
    predictions = create_mock_predictions()

    print(f"\nEvaluating {len(predictions)} mock predictions...\n")

    # Evaluate each prediction
    print("-" * 60)
    print(f"{'ID':<5} {'Pred α':>8} {'Pred β':>8} {'Mean':>8} {'Outcome':>8} {'Loss':>10}")
    print("-" * 60)

    results = []
    for pred in predictions:
        alpha, beta = pred["alpha"], pred["beta"]
        outcome = pred["outcome"]
        mean = alpha / (alpha + beta)

        loss = loss_fn.compute(alpha, beta, outcome)
        result = evaluator.evaluate((alpha, beta), outcome, pred["question_id"])
        results.append(result)

        print(f"{pred['question_id']:<5} {alpha:>8.1f} {beta:>8.1f} {mean:>8.3f} {outcome:>8.0f} {loss:>10.4f}")

    print("-" * 60)

    # Summary statistics
    total_loss = sum(r["score"] for r in results)
    mean_loss = total_loss / len(results)
    print(f"\nMean Loss: {mean_loss:.4f}")

    # Calibration analysis
    print("\n--- Calibration Analysis ---")
    pred_tuples = [(p["alpha"], p["beta"]) for p in predictions]
    outcomes = [p["outcome"] for p in predictions]
    cal_error = beta_calibration_error(pred_tuples, outcomes, n_bins=3)
    print(f"Calibration Error (ECE): {cal_error:.4f}")

    # KL Divergence example
    print("\n--- KL Divergence Demo ---")
    # Compare two Beta distributions
    kl = kl_divergence_beta(8.0, 2.0, 7.0, 3.0)
    print(f"KL(Beta(8,2) || Beta(7,3)) = {kl:.4f}")

    # Show that different predictions have different losses
    print("\n--- Loss Comparison ---")
    print("Confident correct (Beta(9,1), outcome=1):")
    print(f"  Loss = {loss_fn.compute(9.0, 1.0, 1.0):.4f}")
    print("Confident wrong (Beta(9,1), outcome=0):")
    print(f"  Loss = {loss_fn.compute(9.0, 1.0, 0.0):.4f}")
    print("Uncertain (Beta(1,1), outcome=1):")
    print(f"  Loss = {loss_fn.compute(1.0, 1.0, 1.0):.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
