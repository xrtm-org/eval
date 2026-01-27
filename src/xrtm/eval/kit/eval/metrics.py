# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

from typing import Any, List, Tuple, Union

from xrtm.eval.core.eval.definitions import BrierDecomposition, EvaluationResult, Evaluator, ReliabilityBin


class BrierScoreEvaluator(Evaluator):
    def score(self, prediction: Union[float, Any], ground_truth: Union[int, bool, str, Any]) -> float:
        try:
            f = float(prediction)
        except (ValueError, TypeError):
            raise ValueError(f"Prediction must be convertible to float. Got {prediction}")

        if isinstance(ground_truth, str):
            o = 1.0 if ground_truth.lower() in ["yes", "1", "true", "won", "pass"] else 0.0
        else:
            o = 1.0 if ground_truth else 0.0

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

        total_count = len(results)
        if total_count == 0:
            return BrierDecomposition(reliability=0.0, resolution=0.0, uncertainty=0.0, score=0.0)

        all_outcomes = []
        for r in results:
            if isinstance(r.ground_truth, str):
                o = 1.0 if r.ground_truth.lower() in ["yes", "1", "true", "won", "pass"] else 0.0
            else:
                o = 1.0 if r.ground_truth else 0.0
            all_outcomes.append(o)

        o_bar = sum(all_outcomes) / total_count
        uncertainty = o_bar * (1.0 - o_bar)

        reliability = 0.0
        resolution = 0.0

        for b in bins:
            w_k = b.count / total_count
            reliability += w_k * (b.mean_prediction - b.mean_ground_truth) ** 2
            resolution += w_k * (b.mean_ground_truth - o_bar) ** 2

        score = reliability - resolution + uncertainty
        return BrierDecomposition(reliability=reliability, resolution=resolution, uncertainty=uncertainty, score=score)


class ExpectedCalibrationErrorEvaluator(Evaluator):
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def score(self, prediction: Any, ground_truth: Any) -> float:
        return BrierScoreEvaluator().score(prediction, ground_truth)

    def evaluate(self, prediction: Any, ground_truth: Any, subject_id: str) -> EvaluationResult:
        return BrierScoreEvaluator().evaluate(prediction, ground_truth, subject_id)

    def compute_calibration_data(self, results: List[EvaluationResult]) -> Tuple[float, List[ReliabilityBin]]:
        bin_size = 1.0 / self.num_bins
        bins: List[List[EvaluationResult]] = [[] for _ in range(self.num_bins)]

        for res in results:
            try:
                conf = min(max(float(res.prediction), 0.0), 1.0)
                idx = int(conf / bin_size)
                if idx == self.num_bins:
                    idx -= 1
                bins[idx].append(res)
            except (ValueError, TypeError):
                continue

        total_count = len(results)
        ece = 0.0
        reliability_data = []

        for i, bin_items in enumerate(bins):
            n_b = len(bin_items)
            bin_center = (i + 0.5) * bin_size

            if n_b > 0:
                mean_conf = sum(float(x.prediction) for x in bin_items) / n_b
                accuracies = []
                for x in bin_items:
                    gt = x.ground_truth
                    normalized_gt = 1.0 if (gt.lower() in ["yes", "1", "true", "won", "pass"] if isinstance(gt, str) else gt) else 0.0
                    accuracies.append(normalized_gt)

                mean_acc = sum(accuracies) / n_b
                ece += (n_b / total_count) * abs(mean_acc - mean_conf)
                reliability_data.append(
                    ReliabilityBin(bin_center=bin_center, mean_prediction=mean_conf, mean_ground_truth=mean_acc, count=n_b)
                )
            else:
                reliability_data.append(
                    ReliabilityBin(bin_center=bin_center, mean_prediction=0.0, mean_ground_truth=0.0, count=0)
                )

        return ece, reliability_data


__all__ = ["BrierScoreEvaluator", "ExpectedCalibrationErrorEvaluator"]
