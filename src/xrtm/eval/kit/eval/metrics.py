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
Scoring evaluators for probabilistic forecasting.

Provides concrete ``Evaluator`` implementations for Brier Score and
Expected Calibration Error (ECE), including full Murphy decomposition.
"""

import math
from typing import Any, Iterable, List, Tuple, Union

from xrtm.eval.core.eval.definitions import BrierDecomposition, EvaluationResult, Evaluator, ReliabilityBin

TRUE_VALUES = {"yes", "1", "true", "won", "pass"}
FALSE_VALUES = {"no", "0", "false", "lost", "fail"}


def _normalize_ground_truth(ground_truth: Any) -> float:
    if ground_truth is None:
        raise ValueError("Ground truth outcome is missing.")

    if isinstance(ground_truth, bool):
        return 1.0 if ground_truth else 0.0

    if isinstance(ground_truth, str):
        normalized = ground_truth.strip().lower()
        if normalized in TRUE_VALUES:
            return 1.0
        if normalized in FALSE_VALUES:
            return 0.0
        raise ValueError(f"Ground truth must be a recognized binary outcome. Got {ground_truth!r}")

    try:
        outcome = float(ground_truth)
    except (ValueError, TypeError):
        raise ValueError(f"Ground truth must be a binary outcome. Got {ground_truth!r}")

    if not math.isfinite(outcome) or outcome not in {0.0, 1.0}:
        raise ValueError(f"Ground truth must be 0 or 1. Got {ground_truth!r}")
    return outcome


def _normalize_probability(prediction: Any) -> float:
    try:
        probability = float(prediction)
    except (ValueError, TypeError):
        raise ValueError(f"Prediction must be convertible to float. Got {prediction!r}")

    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"Prediction must be a finite probability in [0, 1]. Got {prediction!r}")

    return probability


def _coerce_calibration_probability(prediction: Any) -> float:
    try:
        probability = float(prediction)
    except (ValueError, TypeError):
        raise ValueError(f"Prediction must be convertible to float. Got {prediction!r}")

    if not math.isfinite(probability):
        raise ValueError(f"Prediction must be finite. Got {prediction!r}")
    return min(max(probability, 0.0), 1.0)


class BrierScoreEvaluator(Evaluator):
    r"""Evaluator that computes the Brier score for binary probabilistic predictions."""
    def score(self, prediction: Union[float, Any], ground_truth: Union[int, bool, str, Any]) -> float:
        f = _normalize_probability(prediction)
        o = _normalize_ground_truth(ground_truth)

        return (f - o) ** 2

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        s = self.score(prediction, ground_truth)
        return EvaluationResult(
            subject_id=subject_id,
            score=s,
            ground_truth=ground_truth,
            prediction=prediction,
            metadata={"metric": "Brier Score"},
        )

    def compute_decomposition(self, results: List[EvaluationResult], num_bins: int = 10) -> BrierDecomposition:
        ece_eval = ExpectedCalibrationErrorEvaluator(num_bins=num_bins)
        _, bins = ece_eval.compute_calibration_data(results)

        valid_results = []
        for r in results:
            try:
                valid_results.append(
                    (_coerce_calibration_probability(r.prediction), _normalize_ground_truth(r.ground_truth))
                )
            except (ValueError, TypeError):
                continue

        valid_count = len(valid_results)
        if valid_count == 0:
            return BrierDecomposition(reliability=0.0, resolution=0.0, uncertainty=0.0, score=0.0)

        all_outcomes = [outcome for _, outcome in valid_results]
        o_bar = sum(all_outcomes) / valid_count
        uncertainty = o_bar * (1.0 - o_bar)

        reliability = 0.0
        resolution = 0.0

        for b in bins:
            w_k = b.count / valid_count
            reliability += w_k * (b.mean_prediction - b.mean_ground_truth) ** 2
            resolution += w_k * (b.mean_ground_truth - o_bar) ** 2

        score = reliability - resolution + uncertainty
        return BrierDecomposition(reliability=reliability, resolution=resolution, uncertainty=uncertainty, score=score)


class ExpectedCalibrationErrorEvaluator(Evaluator):
    r"""Evaluator that computes Expected Calibration Error via reliability diagrams."""
    def __init__(self, num_bins: int = 10):
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
            raise ValueError(f"num_bins must be a positive integer. Got {num_bins!r}")
        self.num_bins = num_bins

    def score(self, prediction: Any, ground_truth: Any) -> float:
        return BrierScoreEvaluator().score(prediction, ground_truth)

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        return BrierScoreEvaluator().evaluate(prediction, ground_truth, subject_id)

    def compute_calibration_data(self, results: List[EvaluationResult]) -> Tuple[float, List[ReliabilityBin]]:
        bin_size = 1.0 / self.num_bins
        bins: List[List[Tuple[float, float]]] = [[] for _ in range(self.num_bins)]

        for res in results:
            try:
                conf = _coerce_calibration_probability(res.prediction)
                idx = int(conf / bin_size)
                if idx == self.num_bins:
                    idx -= 1

                normalized_gt = _normalize_ground_truth(res.ground_truth)

                bins[idx].append((conf, normalized_gt))
            except (ValueError, TypeError):
                continue

        valid_count = sum(len(bin_items) for bin_items in bins)
        ece = 0.0
        reliability_data = []

        for i, bin_items in enumerate(bins):
            n_b = len(bin_items)
            bin_center = (i + 0.5) * bin_size

            if n_b > 0:
                mean_conf = sum(x[0] for x in bin_items) / n_b
                mean_acc = sum(x[1] for x in bin_items) / n_b
                ece += (n_b / valid_count) * abs(mean_acc - mean_conf)
                reliability_data.append(
                    ReliabilityBin(
                        bin_center=bin_center, mean_prediction=mean_conf, mean_ground_truth=mean_acc, count=n_b
                    )
                )
            else:
                reliability_data.append(
                    ReliabilityBin(bin_center=bin_center, mean_prediction=0.0, mean_ground_truth=0.0, count=0)
                )

        return ece, reliability_data


class LogScoreEvaluator(Evaluator):
    r"""Evaluator that computes binary negative log score for probabilistic predictions."""

    def __init__(self, epsilon: float = 1e-15):
        if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
            raise ValueError(f"epsilon must be a finite value in (0, 0.5). Got {epsilon!r}")
        self.epsilon = epsilon

    def score(self, prediction: Any, ground_truth: Any) -> float:
        probability = _normalize_probability(prediction)
        outcome = _normalize_ground_truth(ground_truth)
        clamped_probability = min(max(probability, self.epsilon), 1.0 - self.epsilon)
        likelihood = clamped_probability if outcome == 1.0 else 1.0 - clamped_probability
        return -math.log(likelihood)

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        s = self.score(prediction, ground_truth)
        return EvaluationResult(
            subject_id=subject_id,
            score=s,
            ground_truth=ground_truth,
            prediction=prediction,
            metadata={"metric": "Log Score"},
        )


def summarize_binary_forecasts(
    records: Iterable[Tuple[Any, Any]],
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    r"""Summarize binary probabilistic forecasts with dashboard-ready scoring primitives.

    ``records`` is an iterable of ``(prediction_probability, binary_outcome)``
    pairs. Invalid or unresolved rows are skipped consistently across Brier,
    ECE, reliability, calibration curve, and log score outputs.
    """

    brier = BrierScoreEvaluator()
    log_score = LogScoreEvaluator()
    results: List[EvaluationResult] = []
    log_scores: List[float] = []
    skipped_count = 0

    for index, (prediction, ground_truth) in enumerate(records):
        try:
            probability = _normalize_probability(prediction)
            outcome = _normalize_ground_truth(ground_truth)
        except (ValueError, TypeError):
            skipped_count += 1
            continue
        results.append(
            EvaluationResult(
                subject_id=str(index),
                score=brier.score(probability, outcome),
                ground_truth=outcome,
                prediction=probability,
                metadata={"metric": "Brier Score"},
            )
        )
        log_scores.append(log_score.score(probability, outcome))

    valid_count = len(results)
    mean_brier = sum(result.score for result in results) / valid_count if valid_count else None
    mean_log_score = sum(log_scores) / valid_count if valid_count else None
    ece, reliability_bins = ExpectedCalibrationErrorEvaluator(num_bins=num_bins).compute_calibration_data(results)
    decomposition = brier.compute_decomposition(results, num_bins=num_bins)

    calibration_curve = []
    for index, reliability_bin in enumerate(reliability_bins):
        count = reliability_bin.count
        probability_sum = reliability_bin.mean_prediction * count if count else 0.0
        observed_true = int(round(reliability_bin.mean_ground_truth * count)) if count else 0
        calibration_curve.append(
            {
                "index": index,
                "label": f"{round(index * 100 / num_bins)}-{round((index + 1) * 100 / num_bins)}%",
                "lower": index / num_bins,
                "upper": (index + 1) / num_bins,
                "count": count,
                "observed_true": observed_true,
                "probability_sum": probability_sum,
                "mean_probability": reliability_bin.mean_prediction if count else None,
                "observed_frequency": reliability_bin.mean_ground_truth if count else None,
            }
        )

    return {
        "resolved_count": valid_count,
        "skipped_count": skipped_count,
        "brier_score": mean_brier,
        "ece": ece if valid_count else None,
        "log_score": mean_log_score,
        "reliability": decomposition.reliability if valid_count else None,
        "resolution": decomposition.resolution if valid_count else None,
        "uncertainty": decomposition.uncertainty if valid_count else None,
        "decomposed_brier_score": decomposition.score if valid_count else None,
        "reliability_bins": [bin_item.model_dump() for bin_item in reliability_bins],
        "calibration_curve": calibration_curve,
    }


__all__ = [
    "BrierScoreEvaluator",
    "ExpectedCalibrationErrorEvaluator",
    "LogScoreEvaluator",
    "summarize_binary_forecasts",
]
