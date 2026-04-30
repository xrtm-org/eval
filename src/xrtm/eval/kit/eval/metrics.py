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

from typing import Any, List, Tuple, Union

from xrtm.eval.core.eval.definitions import BrierDecomposition, EvaluationResult, Evaluator, ReliabilityBin

TRUE_VALUES = {"yes", "1", "true", "won", "pass"}


def _normalize_ground_truth(ground_truth: Any) -> float:
    if isinstance(ground_truth, str):
        return 1.0 if ground_truth.lower() in TRUE_VALUES else 0.0
    return 1.0 if ground_truth else 0.0


class BrierScoreEvaluator(Evaluator):
    r"""Evaluator that computes the Brier score for binary probabilistic predictions."""
    def score(self, prediction: Union[float, Any], ground_truth: Union[int, bool, str, Any]) -> float:
        try:
            f = float(prediction)
        except (ValueError, TypeError):
            raise ValueError(f"Prediction must be convertible to float. Got {prediction}")

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
                valid_results.append((float(r.prediction), _normalize_ground_truth(r.ground_truth)))
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
                raw_conf = float(res.prediction)
                conf = min(max(raw_conf, 0.0), 1.0)
                idx = int(conf / bin_size)
                if idx == self.num_bins:
                    idx -= 1

                normalized_gt = _normalize_ground_truth(res.ground_truth)

                bins[idx].append((raw_conf, normalized_gt))
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


__all__ = ["BrierScoreEvaluator", "ExpectedCalibrationErrorEvaluator"]
