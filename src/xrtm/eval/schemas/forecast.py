# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ForecastResolution(BaseModel):
    r"""
    The ground-truth outcome used to evaluate forecast accuracy.
    """

    question_id: str
    outcome: str = Field(..., description="The final winning outcome or value")
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source info, verification method")


__all__ = ["ForecastResolution"]
