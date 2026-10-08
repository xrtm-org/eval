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

r"""Slice analytics: per-slice summaries of evaluation results."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

__all__ = ["SliceAnalytics"]


def _field(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


class SliceAnalytics:
    r"""Buckets evaluation results into equal-width probability slices."""

    @staticmethod
    def compute_slices(results: Iterable[Any], num_slices: int = 10) -> Dict[str, Dict[str, Any]]:
        r"""Summarise *results* by predicted-probability slice.

        Each slice reports ``count``, ``mean_prediction``, ``mean_outcome`` and
        ``mean_score`` (``None`` when scores are unavailable). Results without a
        usable prediction/outcome are skipped; empty input yields ``{}``.

        Example:
            >>> class R:
            ...     def __init__(self, p, g, s):
            ...         self.prediction, self.ground_truth, self.score = p, g, s
            >>> slices = SliceAnalytics.compute_slices([R(0.05, 0.0, 0.0025)])
            >>> sorted(slices)
            ['0.0-0.1']
        """
        if num_slices <= 0:
            raise ValueError("num_slices must be a positive integer")
        buckets: Dict[int, List[tuple]] = {}
        for result in results or []:
            try:
                prediction = float(_field(result, "prediction"))
                outcome = float(_field(result, "ground_truth"))
            except (TypeError, ValueError):
                continue
            raw_score = _field(result, "score")
            try:
                score: Optional[float] = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                score = None
            index = min(max(0, int(prediction * num_slices)), num_slices - 1)
            buckets.setdefault(index, []).append((prediction, outcome, score))

        slices: Dict[str, Dict[str, Any]] = {}
        for index in sorted(buckets):
            rows = buckets[index]
            count = len(rows)
            scores = [row[2] for row in rows if row[2] is not None]
            slices[f"{index / num_slices:.1f}-{(index + 1) / num_slices:.1f}"] = {
                "count": count,
                "mean_prediction": sum(row[0] for row in rows) / count,
                "mean_outcome": sum(row[1] for row in rows) / count,
                "mean_score": (sum(scores) / len(scores)) if scores else None,
            }
        return slices
