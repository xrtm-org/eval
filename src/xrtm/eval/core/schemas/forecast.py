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
Forecast resolution schema for evaluation.

This module defines the ground-truth outcome schema used to evaluate
forecast accuracy.

Example:
    >>> from xrtm.eval.core.schemas import ForecastResolution
    >>> resolution = ForecastResolution(
    ...     question_id="q1",
    ...     outcome="yes",
    ... )
"""

from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field


class ForecastResolution(BaseModel):
    r"""
    The ground-truth outcome used to evaluate forecast accuracy.

    Attributes:
        question_id: Reference to the forecasted question.
        outcome: The final winning outcome or value.
        resolved_at: When the outcome was determined.
        metadata: Source info, verification method, etc.

    Example:
        >>> resolution = ForecastResolution(question_id="q1", outcome="yes")
    """

    question_id: str = Field(..., description="Reference to the forecasted question")
    outcome: str = Field(..., description="The final winning outcome or value")
    resolved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the outcome was determined",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source info, verification method")


__all__ = ["ForecastResolution"]
