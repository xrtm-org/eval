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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ForecastResolution(BaseModel):
    r"""
    The ground-truth outcome used to evaluate forecast accuracy.

    Attributes:
        forecast_request_id: Reference to the forecast request being resolved.
        outcome: The final winning outcome or value.
        resolved_at: When the outcome was determined.
        metadata: Source info, verification method, etc.

    Example:
        >>> resolution = ForecastResolution(forecast_request_id="q1", outcome="yes")
    """

    model_config = ConfigDict(populate_by_name=True)

    forecast_request_id: str = Field(
        ...,
        validation_alias=AliasChoices("forecast_request_id", "question_id"),
        description="Reference to the forecast request being resolved",
    )
    outcome: str = Field(..., description="The final winning outcome or value")
    resolved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the outcome was determined",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source info, verification method")

    @property
    def question_id(self) -> str:
        r"""Backward compatibility alias for ``forecast_request_id``."""
        return self.forecast_request_id

    @question_id.setter
    def question_id(self, value: str) -> None:
        r"""Backward compatibility setter for ``forecast_request_id``."""
        self.forecast_request_id = value


__all__ = ["ForecastResolution"]
