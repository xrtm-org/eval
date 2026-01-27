# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging
from typing import Dict, List

from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult
from xrtm.eval.kit.eval.metrics import ExpectedCalibrationErrorEvaluator

__all__ = ["SliceAnalytics"]

logger = logging.getLogger(__name__)


class SliceAnalytics:
    @staticmethod
    def compute_slices(results: List[EvaluationResult]) -> Dict[str, EvaluationReport]:
        slices: Dict[str, List[EvaluationResult]] = {}
        for res in results:
            tags = res.metadata.get("tags", [])
            if not tags:
                continue
            for tag in tags:
                key = f"tag:{tag}"
                if key not in slices:
                    slices[key] = []
                slices[key].append(res)

        reports: Dict[str, EvaluationReport] = {}
        for slice_key, slice_results in slices.items():
            count = len(slice_results)
            if count == 0:
                continue
            total_score = sum(r.score for r in slice_results)
            mean_score = total_score / count
            ece_evaluator = ExpectedCalibrationErrorEvaluator()
            try:
                ece_score, bins = ece_evaluator.compute_calibration_data(slice_results)
            except Exception as e:
                logger.warning(f"Failed to compute ECE for slice {slice_key}: {e}")
                ece_score = 0.0
                bins = None
            reports[slice_key] = EvaluationReport(
                metric_name="Slice Brier",
                mean_score=mean_score,
                total_evaluations=count,
                results=slice_results,
                reliability_bins=bins,
                summary_statistics={"ece": ece_score},
            )
        return reports
